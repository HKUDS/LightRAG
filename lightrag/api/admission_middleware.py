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

import uuid
from typing import Any, Callable, Optional

from fastapi import HTTPException

from lightrag.utils import logger

from .admission import AdmissionTicket, publish_admission_ticket
from .asgi_helpers import bearer_token, header_value, send_json
from .utils_api import credentials_accepted, get_route_path, path_is_whitelisted

# Ingestion routes that accept a body and create documents. ``/documents/scan``
# is absent on purpose: it takes no body and is exempt from capacity by design
# (§9.1).
ADMISSION_PATHS: tuple[str, ...] = (
    "/documents/upload",
    "/documents/text",
    "/documents/texts",
)


class AdmissionMiddleware:
    """Reserve capacity for an ingestion request before its body is read.

    Args:
        app: the wrapped ASGI application.
        rag_getter: resolves the LightRAG instance at request time. The instance
            is built after the middleware is installed, so it cannot be captured
            by value.
        api_key: the configured ``LIGHTRAG_API_KEY``, for the pre-auth check.
        api_prefix: mount prefix to strip before matching paths.

    The raw-body ceiling is NOT this middleware's business; it lives in
    :class:`lightrag.api.body_limit_middleware.BodyLimitMiddleware`, installed
    outside this one so an oversized body is refused before it can consume a
    capacity slot.
    """

    def __init__(
        self,
        app,
        *,
        rag_getter: Callable[[], Any],
        api_key: Optional[str] = None,
        api_prefix: str = "",
    ) -> None:
        self.app = app
        self._rag_getter = rag_getter
        self._api_key = api_key
        self._api_prefix = (api_prefix or "").rstrip("/")

    def _is_admission_path(self, scope: dict[str, Any]) -> bool:
        if scope.get("method") != "POST":
            return False
        return get_route_path(scope, self._api_prefix) in ADMISSION_PATHS

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

        # Authenticate, then reserve — both before any receive(). An oversized
        # body has already been refused further out by BodyLimitMiddleware, so
        # nothing that gets here can consume a capacity slot on its way to a 413.
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
            if not path_is_whitelisted(
                scope, mount_prefix=self._api_prefix
            ) and not credentials_accepted(
                token=bearer_token(scope),
                api_key_header_value=header_value(scope, b"x-api-key"),
                api_key=self._api_key,
            ):
                # Refuse before reserving AND before reading the body: an
                # unauthenticated caller must not be able to consume capacity,
                # nor to upload anything. The route's own dependency remains the
                # authority on the exact 401/403 wording for what gets past here.
                await send_json(
                    send, 401, "Not authenticated. Please provide valid credentials."
                )
                return

            candidate = AdmissionTicket(token=f"admission-{uuid.uuid4().hex}", weight=1)
            try:
                reserved = await _reserve_enqueue_slot(rag, candidate.token, weight=1)
            except HTTPException as refusal:
                # 429 over capacity, 409 fenced, 503 count failure — all decided
                # without a single byte of body.
                await send_json(
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

        try:
            await self.app(scope, receive, send)
        finally:
            # Only when nobody downstream took ownership. Release is idempotent
            # and owner-identified, so this cannot decrement a sibling's slot
            # even if the route released the same token first.
            #
            # Reached on the body-limit path too: BodyLimitExceeded is raised
            # from a receive() wrapper installed further out and travels up
            # through here, so a 413 mid-body still returns the reservation.
            if ticket is not None and not ticket.adopted:
                await _release_enqueue_slot(rag, ticket.token)
