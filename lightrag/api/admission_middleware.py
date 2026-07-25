"""Pre-body admission control as pure ASGI (LR2 §9.3).

A FastAPI handler only runs after its ``UploadFile`` / Pydantic parameters have
been parsed, which means the request body has already been read off the socket.
That is too late for admission: the point of a capacity limit is to refuse the
upload *before* paying for it. So the decision moves into a pure-ASGI middleware
that runs authentication and takes the reservation before it ever calls
``receive()``.

Two properties follow from never calling ``receive()`` on a refused request:

* the body is not transferred — and with ``Expect: 100-continue`` the server
  never sends the ``100 Continue`` that would invite the client to send it,
  because uvicorn emits that only on the first ``receive()``;
* the reservation is atomic, not a stateless guess. With capacity 1 and 500
  concurrent uploads, exactly one request wins the reservation and reads a body;
  the other 499 are refused with 429 having transferred nothing.

The reservation is then handed to the route through ``scope["state"]`` (see
:mod:`lightrag.api.admission`) so ONE reservation covers the whole request
instead of the route taking a second one.

Degradation is deliberate: when this middleware is not installed, or cannot
resolve the LightRAG instance, the route's own reservation still enforces the
same capacity — just after the body has been read. Admission is never silently
disabled, only made less economical.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException

from lightrag.utils import logger

from .admission import AdmissionTicket, publish_admission_ticket
from .utils_api import credentials_accepted, path_is_whitelisted

# Ingestion routes that accept a body and create documents. ``/documents/scan``
# is absent on purpose: it takes no body and is exempt from capacity by design
# (§9.1).
ADMISSION_PATHS: tuple[str, ...] = (
    "/documents/upload",
    "/documents/text",
    "/documents/texts",
)


def _header(scope: dict[str, Any], name: bytes) -> Optional[str]:
    """First value of a raw ASGI header, decoded, or None."""
    for key, value in scope.get("headers") or ():
        if key == name:
            try:
                return value.decode("latin-1")
            except Exception:
                return None
    return None


def _bearer_token(scope: dict[str, Any]) -> Optional[str]:
    """The OAuth2 bearer token, matching how FastAPI's scheme extracts it."""
    authorization = _header(scope, b"authorization")
    if not authorization:
        return None
    scheme, _, param = authorization.partition(" ")
    if scheme.lower() != "bearer" or not param:
        return None
    return param


async def _send_json(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    status_code: int,
    detail: str,
    extra_headers: Optional[dict[str, str]] = None,
) -> None:
    """Answer without touching ``receive()``.

    The body shape matches FastAPI's ``HTTPException`` responses so a client
    cannot tell whether a refusal came from the middleware or from a route.
    """
    body = json.dumps({"detail": detail}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    for name, value in (extra_headers or {}).items():
        # Lowercased: HTTP/2 requires it and the ASGI spec asks apps for it, so a
        # client sees the same header name whichever protocol it arrived on.
        headers.append((name.lower().encode("latin-1"), str(value).encode("latin-1")))
    await send(
        {"type": "http.response.start", "status": status_code, "headers": headers}
    )
    await send({"type": "http.response.body", "body": body})


class _BodyLimitExceeded(Exception):
    """Raised inside the wrapped ``receive`` once the body ceiling is passed.

    Private and caught by the middleware itself: it must reach neither the route
    (whose ``except Exception`` would turn it into a 500) nor Starlette's
    ServerErrorMiddleware. The ASGI app is blocked on ``receive()`` when the
    limit trips, and raising is the only way to unblock it without pretending the
    body ended — which would hand the route a truncated document.
    """


def _limited_receive(
    receive: Callable[[], Awaitable[dict[str, Any]]],
    limit: int,
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Wrap ``receive`` so the body is cut off after ``limit`` bytes.

    Chunks are counted and passed straight through — nothing is buffered, so the
    limit costs O(1) memory and the streaming upload path keeps streaming.
    Content-Length is only an early-rejection hint (a client may omit or
    understate it); this counter is the actual protection.
    """
    total = 0

    async def _receive() -> dict[str, Any]:
        nonlocal total
        message = await receive()
        if message.get("type") == "http.request":
            total += len(message.get("body", b"") or b"")
            if total > limit:
                raise _BodyLimitExceeded(total)
        return message

    return _receive


class AdmissionMiddleware:
    """Reserve capacity for an ingestion request before its body is read.

    Args:
        app: the wrapped ASGI application.
        rag_getter: resolves the LightRAG instance at request time. The instance
            is built after the middleware is installed, so it cannot be captured
            by value.
        api_key: the configured ``LIGHTRAG_API_KEY``, for the pre-auth check.
        api_prefix: mount prefix to strip before matching paths.
    """

    def __init__(
        self,
        app,
        *,
        rag_getter: Callable[[], Any],
        api_key: Optional[str] = None,
        api_prefix: str = "",
        max_body_bytes: int = 0,
    ) -> None:
        self.app = app
        self._rag_getter = rag_getter
        self._api_key = api_key
        self._api_prefix = (api_prefix or "").rstrip("/")
        self._max_body_bytes = max_body_bytes if max_body_bytes > 0 else 0

    def _is_admission_path(self, scope: dict[str, Any]) -> bool:
        if scope.get("method") != "POST":
            return False
        path = scope.get("path") or ""
        if self._api_prefix and path.startswith(self._api_prefix):
            path = path[len(self._api_prefix) :] or "/"
        return path in ADMISSION_PATHS

    def _resolve_rag(self) -> Optional[Any]:
        try:
            return self._rag_getter()
        except Exception as resolve_error:  # pragma: no cover - defensive
            logger.warning(
                "Admission middleware cannot resolve the LightRAG instance "
                f"({resolve_error}); the route's own reservation still applies."
            )
            return None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self._is_admission_path(scope):
            await self.app(scope, receive, send)
            return

        # 1. Body ceiling, evaluated first: a request that is too large to accept
        #    must not consume a capacity slot on its way to being refused.
        #    Content-Length is only a shortcut for an honest client; the real
        #    enforcement is the counting wrapper installed in step 3.
        if self._max_body_bytes > 0:
            declared = _header(scope, b"content-length")
            if declared and declared.isdigit() and int(declared) > self._max_body_bytes:
                await _send_json(send, 413, self._body_limit_detail())
                return

        # 2. Admission: authenticate, then reserve — both before any receive().
        #    Each branch of this middleware honours its own switch (§9.4), so a
        #    deployment may enable the body limit alone.
        rag = self._resolve_rag()
        # Imported here: document_routes must not import this module (that would
        # be a cycle), and the reservation helpers live with the routes that own
        # them.
        from .routers.document_routes import (
            _admission_capacity,
            _release_enqueue_slot,
            _reserve_enqueue_slot,
        )

        ticket: Optional[AdmissionTicket] = None
        if rag is not None and _admission_capacity(rag) > 0:
            path = scope.get("path") or ""
            if not path_is_whitelisted(path) and not credentials_accepted(
                token=_bearer_token(scope),
                api_key_header_value=_header(scope, b"x-api-key"),
                api_key=self._api_key,
            ):
                # Refuse before reserving AND before reading the body: an
                # unauthenticated caller must not be able to consume capacity,
                # nor to upload anything. The route's own dependency remains the
                # authority on the exact 401/403 wording for what gets past here.
                await _send_json(
                    send, 401, "Not authenticated. Please provide valid credentials."
                )
                return

            candidate = AdmissionTicket(token=f"admission-{uuid.uuid4().hex}", weight=1)
            try:
                reserved = await _reserve_enqueue_slot(rag, candidate.token, weight=1)
            except HTTPException as refusal:
                # 429 over capacity, 409 fenced, 503 count failure — all decided
                # without a single byte of body.
                await _send_json(
                    send,
                    refusal.status_code,
                    str(refusal.detail),
                    extra_headers=refusal.headers,
                )
                return
            if reserved:
                ticket = candidate
                publish_admission_ticket(scope, ticket)
            # ``reserved`` False means pipeline_status was never bootstrapped:
            # there is nothing to charge against, so stay out of the way.

        # 3. Stream the body under the ceiling.
        wrapped_receive = receive
        if self._max_body_bytes > 0:
            wrapped_receive = _limited_receive(receive, self._max_body_bytes)

        response_started = False

        async def _tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, wrapped_receive, _tracking_send)
        except _BodyLimitExceeded as exceeded:
            logger.warning(
                f"Rejected {scope.get('path')}: request body exceeded "
                f"MAX_REQUEST_BODY_BYTES ({self._max_body_bytes}) after "
                f"{exceeded.args[0] if exceeded.args else '?'} bytes"
            )
            if not response_started:
                await _send_json(send, 413, self._body_limit_detail())
            # Otherwise the app already committed a status line; the connection
            # ends with a short body, which is all the protocol allows at that
            # point. Nothing is swallowed silently — the warning above is the
            # record.
        finally:
            # Only when nobody downstream took ownership. Release is idempotent
            # and owner-identified, so this cannot decrement a sibling's slot
            # even if the route released the same token first.
            if ticket is not None and not ticket.adopted:
                await _release_enqueue_slot(rag, ticket.token)

    def _body_limit_detail(self) -> str:
        return (
            "Request body too large. This endpoint accepts at most "
            f"{self._max_body_bytes} bytes."
        )
