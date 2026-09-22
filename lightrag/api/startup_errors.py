"""Concise ASGI startup diagnostics for actionable storage refusals."""

import logging
import os
import sys

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lightrag.exceptions import VectorStorageEmptyError
from lightrag.utils import logger


class StartupDiagnostic(str):
    """Plain text of a startup refusal, marked for console highlighting.

    Logged to every handler on the ``lightrag`` logger, the log file included,
    so the text itself never carries terminal escapes; only
    ``ConsoleFormatter`` adds them, on output it writes to an interactive
    terminal.
    """


class ConsoleFormatter(logging.Formatter):
    """Console formatter that highlights a ``StartupDiagnostic`` in bold red.

    Attach it to a handler writing to ``sys.stderr`` only, never to a file
    handler. Plain text when stderr is not a terminal or ``NO_COLOR`` is set.
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if (
            isinstance(record.msg, StartupDiagnostic)
            and sys.stderr.isatty()
            and "NO_COLOR" not in os.environ
        ):
            return f"\033[1;31m{text}\033[0m"
        return text


class StartupErrorMiddleware:
    """Replace only expected startup tracebacks, preserving failure and cleanup.

    Starlette sends a traceback in startup.failed before re-raising. Defer that
    message until the exception reaches us so classification uses its type,
    not text matching. Shutdown and unexpected failures retain their traceback.

    The diagnostic goes to the ``lightrag`` logger, not in the ASGI message:
    Uvicorn logs that message to ``uvicorn.error``, which ``lightrag-gunicorn``
    silences in every worker (``gunicorn_config.post_fork``), so under Gunicorn
    it would reach no one.
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
                    # The storage's own workspace: a backend override
                    # (QDRANT_WORKSPACE, ...) can move it off the server's.
                    workspace = (
                        self.workspace if exc.workspace is None else exc.workspace
                    )
                    message = (
                        "Startup blocked: vector index is empty.\n"
                        f"  Workspace: {workspace or '(default)'}\n"
                        f"  Storage: {self.vector_storage}\n"
                        f"  Missing index: {exc.vdb_name}{where}\n"
                        f"  Source still contains data: {exc.source}.\n"
                        "  Check the storage backend, workspace and data directory.\n"
                        "  To rebuild: stop all writers, run lightrag-rebuild-vdb\n"
                        "  with the current embedding configuration, and select [4]\n"
                        "  to rebuild ALL vector storages. Then restart the server."
                    )
                    logger.error(StartupDiagnostic(message))
                    # Without a message Uvicorn logs no second copy, and
                    # Starlette's traceback text is dropped with it.
                    failure = {
                        key: value for key, value in failure.items() if key != "message"
                    }
                await send(failure)
            # Preserve the exception for ASGI callers. Uvicorn sees the explicit
            # startup.failed and exits without logging a second traceback.
            raise
        else:
            # ASGI applications may signal failure without raising an exception.
            if failure is not None:
                await send(failure)
