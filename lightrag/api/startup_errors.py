"""Concise ASGI startup diagnostics for actionable storage refusals."""

import os
import sys

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lightrag.exceptions import VectorStorageEmptyError


class StartupErrorMiddleware:
    """Replace only expected startup tracebacks, preserving failure and cleanup.

    Starlette sends a traceback in startup.failed before re-raising. Defer that
    message until the exception reaches us so classification uses its type,
    not text matching. Shutdown and unexpected failures retain their traceback.
    """

    def __init__(self, app: ASGIApp, *, workspace: str, vector_storage: str):
        self.app = app
        self.workspace = workspace
        self.vector_storage = vector_storage

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        failure: Message | None = None

        async def capture(message: Message) -> None:
            nonlocal failure
            if message["type"] == "lifespan.startup.failed":
                failure = message
            else:
                await send(message)

        try:
            await self.app(scope, receive, capture)
        except BaseException as exc:
            if failure is not None:
                if isinstance(exc, VectorStorageEmptyError):
                    where = f" ({exc.container})" if exc.container else ""
                    message = (
                        "Startup blocked: vector index is empty.\n"
                        f"  Workspace: {self.workspace or '(default)'}\n"
                        f"  Storage: {self.vector_storage}\n"
                        f"  Missing index: {exc.vdb_name}{where}\n"
                        f"  Source still contains data: {exc.source}.\n"
                        "  Check the storage backend, workspace and data directory.\n"
                        "  To rebuild: stop all writers, run lightrag-rebuild-vdb\n"
                        "  with the current embedding configuration, and select [4]\n"
                        "  to rebuild ALL vector storages. Then restart the server."
                    )
                    if sys.stderr.isatty() and "NO_COLOR" not in os.environ:
                        message = f"\033[1;31m{message}\033[0m"
                    failure = {**failure, "message": message}
                await send(failure)
            # Preserve the exception for ASGI callers. Uvicorn sees the explicit
            # startup.failed and exits without logging a second traceback.
            raise
        else:
            # ASGI applications may signal failure without raising an exception.
            if failure is not None:
                await send(failure)
