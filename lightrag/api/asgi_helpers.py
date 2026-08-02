"""Primitives shared by the pure-ASGI middlewares.

Both :mod:`lightrag.api.admission_middleware` and
:mod:`lightrag.api.body_limit_middleware` answer requests before the body has
been read, which means neither can use FastAPI's request/response objects: those
only exist once the body has been parsed. They speak raw ASGI instead, and share
the small amount of machinery that entails.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional


def header_value(scope: dict[str, Any], name: bytes) -> Optional[str]:
    """First value of a raw ASGI header, decoded, or None."""
    for key, value in scope.get("headers") or ():
        if key == name:
            try:
                return value.decode("latin-1")
            except Exception:
                return None
    return None


def bearer_token(scope: dict[str, Any]) -> Optional[str]:
    """The OAuth2 bearer token, matching how FastAPI's scheme extracts it."""
    authorization = header_value(scope, b"authorization")
    if not authorization:
        return None
    scheme, _, param = authorization.partition(" ")
    if scheme.lower() != "bearer" or not param:
        return None
    return param


async def send_json(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    status_code: int,
    detail: str,
    extra_headers: Optional[dict[str, str]] = None,
) -> None:
    """Answer without touching ``receive()``.

    The body shape matches FastAPI's ``HTTPException`` responses so a client
    cannot tell whether a refusal came from a middleware or from a route.
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


class BodyLimitExceeded(Exception):
    """Raised inside the wrapped ``receive`` once the body ceiling is passed.

    Caught by the middleware that installed the wrapper: it must reach neither
    the route (whose ``except Exception`` would turn it into a 500) nor
    Starlette's ServerErrorMiddleware. The ASGI app is blocked on ``receive()``
    when the limit trips, and raising is the only way to unblock it without
    pretending the body ended — which would hand the route a truncated document.
    """


def limited_receive(
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
                raise BodyLimitExceeded(total)
        return message

    return _receive
