"""Raw request-body ceiling as pure ASGI.

``MAX_REQUEST_BODY_BYTES`` used to be enforced inside the admission middleware
and therefore covered exactly the three ingestion routes admission cares about.
That made the name a lie: an operator could set it, watch ``/documents/text``
reject an 8 MiB body in milliseconds, and still have ``/api/chat`` — which needs
no credentials under the shipped whitelist — accept the same body and stall the
whole process tokenizing it (GHSA-r8jh-295g-vv42).

The limit now lives here, applies to every route, and is layered because routes
legitimately differ by orders of magnitude in how much body they should accept:

* ``/documents/upload`` derives its ceiling from ``MAX_UPLOAD_SIZE``, plus slack
  for multipart framing;
* ``/documents/text`` and ``/documents/texts`` get a generous ingestion ceiling,
  because a pasted document or a batch insert is not a chat turn and a batch has
  no equivalent on the upload route;
* everything else gets the default, which is what actually bounds the
  unauthenticated surface.

Placement matters. This middleware is installed OUTSIDE the admission middleware
so an oversized body is refused before it can consume a capacity slot, and INSIDE
CORS so a 413 carries the headers a browser needs in order to read the status.
"""

from __future__ import annotations

from typing import Any, Optional

from lightrag.utils import logger

from .asgi_helpers import BodyLimitExceeded, header_value, limited_receive, send_json
from .utils_api import get_route_path

# Routes whose legitimate body size differs from the default by orders of
# magnitude. Everything not listed here uses ``default_limit``.
UPLOAD_PATH = "/documents/upload"
INGEST_TEXT_PATHS: tuple[str, ...] = ("/documents/text", "/documents/texts")


class BodyLimitMiddleware:
    """Cut off a request body that exceeds the ceiling for its route.

    Args:
        app: the wrapped ASGI application.
        default_limit: ceiling for ordinary routes; ``0`` disables.
        ingest_limit: ceiling for the text-ingestion routes; ``0`` disables.
        upload_limit: ceiling for ``/documents/upload``; ``0`` disables.
        api_prefix: mount prefix to strip before matching paths.
    """

    def __init__(
        self,
        app,
        *,
        default_limit: int = 0,
        ingest_limit: int = 0,
        upload_limit: int = 0,
        api_prefix: str = "",
    ) -> None:
        self.app = app
        self._default_limit = max(int(default_limit), 0)
        self._ingest_limit = max(int(ingest_limit), 0)
        self._upload_limit = max(int(upload_limit), 0)
        self._api_prefix = (api_prefix or "").rstrip("/")

    def _limit_for(self, scope: dict[str, Any]) -> int:
        route = get_route_path(scope, self._api_prefix)
        if route == UPLOAD_PATH:
            return self._upload_limit
        if route in INGEST_TEXT_PATHS:
            return self._ingest_limit
        return self._default_limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Applied to every method rather than POST only: a body on GET or DELETE
        # is unusual but legal, and "unusual" is exactly where an unbounded read
        # would otherwise survive. Methods that carry no body make this a no-op.
        limit = self._limit_for(scope)
        if limit <= 0:
            await self.app(scope, receive, send)
            return

        # An honest client's Content-Length lets the refusal happen without
        # touching receive() at all, so the body never leaves the client and
        # uvicorn never sends the "100 Continue" that would invite it. The
        # counting wrapper below is the actual enforcement.
        declared = header_value(scope, b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            await send_json(send, 413, self._detail(limit))
            return

        response_started = False

        async def _tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive(receive, limit), _tracking_send)
        except BodyLimitExceeded as exceeded:
            logger.warning(
                f"Rejected {scope.get('path')}: request body exceeded the "
                f"{limit}-byte ceiling for this route after "
                f"{exceeded.args[0] if exceeded.args else '?'} bytes"
            )
            if not response_started:
                await send_json(send, 413, self._detail(limit))
            # Otherwise the app already committed a status line; the connection
            # ends with a short body, which is all the protocol allows at that
            # point. Nothing is swallowed silently — the warning above is the
            # record.

    @staticmethod
    def _detail(limit: int) -> str:
        return f"Request body too large. This endpoint accepts at most {limit} bytes."


def resolve_body_limits(args: Any) -> Optional[dict[str, int]]:
    """Derive the three per-route ceilings from configuration.

    Returns ``None`` when every ceiling is disabled, so the caller can skip
    installing the middleware entirely.

    Semantics:

    * ``MAX_REQUEST_BODY_BYTES = 0`` — an explicit opt-out. Every ceiling is
      off, including the derived upload one.
    * ``MAX_REQUEST_BODY_BYTES`` not configured — ordinary routes get the
      default, the text-ingestion routes get the built-in ingestion ceiling.
    * ``MAX_REQUEST_BODY_BYTES`` configured to any positive value — the
      operator's intent wins for every non-upload route, ingestion included.
    * ``MAX_UPLOAD_SIZE`` of ``0``/``None`` means "unlimited upload", a
      documented setting; there is then no size to derive a ceiling from, so the
      upload route is left uncapped.

    Whether a value was configured is read from
    ``args.max_request_body_bytes_explicit`` and is deliberately NOT inferred by
    comparing the value to the default: an operator who sets the default value
    on purpose would be indistinguishable from one who set nothing, and the
    text-ingestion routes would silently get the 50 MiB tier instead of the
    ceiling that was asked for — widening a configured security boundary.

    A caller that builds ``args`` by hand and omits the flag is treated as having
    configured the value. Guessing wrong in that direction costs a 413 on a large
    ingest, which is visible and recoverable; guessing wrong the other way
    silently relaxes a ceiling, which is the failure this parameter exists to
    prevent.
    """
    from lightrag.constants import (
        DEFAULT_MAX_INGEST_BODY_BYTES,
        MULTIPART_OVERHEAD_BYTES,
    )

    default_limit = max(int(getattr(args, "max_request_body_bytes", 0) or 0), 0)
    if default_limit <= 0:
        return None

    configured = bool(getattr(args, "max_request_body_bytes_explicit", True))
    ingest_limit = default_limit if configured else DEFAULT_MAX_INGEST_BODY_BYTES

    max_upload_size = getattr(args, "max_upload_size", None)
    if isinstance(max_upload_size, int) and max_upload_size > 0:
        upload_limit = max_upload_size + MULTIPART_OVERHEAD_BYTES
    else:
        upload_limit = 0

    return {
        "default_limit": default_limit,
        "ingest_limit": ingest_limit,
        "upload_limit": upload_limit,
    }
