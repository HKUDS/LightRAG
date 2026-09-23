"""Concise ASGI startup diagnostics for actionable storage refusals."""

import logging
import os
import sys

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lightrag.exceptions import (
    CorruptStorageSnapshotError,
    EmbeddingBaselineMismatchError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)
from lightrag.utils import logger

# The two ways out of every refusal below, in the order an operator should
# consider them. A rebuild keeps the data; the clear is for data nobody needs,
# and it is the ONLY way out that does not start by re-embedding everything.
_RECOVERY_LINES = (
    "  To keep the data: stop all writers, run lightrag-rebuild-vdb with the\n"
    "  current embedding configuration, and select [4] to rebuild ALL vector\n"
    "  storages. Then restart the server.\n"
    "  If the data is disposable: run lightrag-clear-storage instead to drop\n"
    "  this workspace without re-embedding. Then restart the server."
)


def describe_startup_refusal(
    exc: BaseException, *, workspace: str, vector_storage: str
) -> str | None:
    """The concise message for an expected vector refusal, or ``None``.

    Classified on the exception's TYPE, never on its text. Every message
    ends with the same two recovery paths, because every one of these
    refusals is cleared by either: a rebuild re-embeds from the sources, a
    clear drops the workspace, and both remove the condition the server
    refused on.

    The ORDER of the checks below is load-bearing:
    ``EmbeddingBaselineMismatchError`` subclasses ``VectorSpaceMismatchError``
    and must be tested first, or it silently degrades to the parent's block --
    which names one container and one pair of values where the baseline
    refusal has one line per mismatched target. Do not sort these branches.
    """
    if isinstance(exc, VectorStorageEmptyError) and exc.workspace is not None:
        # Backend overrides can move storage off the server's workspace.
        workspace = exc.workspace
    header = f"  Workspace: {workspace or '(default)'}\n  Storage: {vector_storage}\n"
    if isinstance(exc, VectorStorageEmptyError):
        where = f" ({exc.container})" if exc.container else ""
        return (
            "Startup blocked: vector index is empty.\n"
            f"{header}"
            f"  Missing index: {exc.vdb_name}{where}\n"
            f"  Source still contains data: {exc.source}.\n"
            "  Check the storage backend, workspace and data directory.\n"
            f"{_RECOVERY_LINES}"
        )
    if isinstance(exc, EmbeddingBaselineMismatchError):
        # One line per target: each was adopted under its own baseline, and
        # the first target's values say nothing about the others'.
        per_target = "".join(
            f"  {m.get('target')}: recorded model {m.get('recorded_model')!r} "
            f"dim {m.get('recorded_dim')}; configured model "
            f"{m.get('expected_model')!r} dim {m.get('expected_dim')}\n"
            for m in exc.mismatches
        )
        return (
            "Startup blocked: the recorded embedding baseline differs from the "
            "configured embedding model.\n"
            f"{header}"
            f"  Mismatched targets ({len(exc.mismatches)}):\n"
            f"{per_target}"
            "  Either restore the previous EMBEDDING_MODEL / EMBEDDING_DIM, or:\n"
            f"{_RECOVERY_LINES}"
        )
    if isinstance(exc, VectorSpaceMismatchError):
        return (
            "Startup blocked: a vector storage holds vectors from a different "
            "embedding space.\n"
            f"{header}"
            f"  Container: {exc.container} ({exc.backend})\n"
            f"  Stored: model {exc.stored_model!r} dim {exc.stored_dim}; "
            f"configured: model {exc.expected_model!r} dim {exc.expected_dim}.\n"
            "  Either restore the previous EMBEDDING_MODEL / EMBEDDING_DIM, or:\n"
            f"{_RECOVERY_LINES}"
        )
    if isinstance(exc, CorruptStorageSnapshotError):
        return (
            "Startup blocked: a vector storage snapshot is corrupt.\n"
            f"{header}"
            f"  File: {exc.container} ({exc.backend})\n"
            f"  Parse error: {exc.detail}\n"
            "  A previous write was likely interrupted. Stop every writer and\n"
            "  verify the graph storage and text_chunks first.\n"
            f"{_RECOVERY_LINES}"
        )
    return None


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
                message = describe_startup_refusal(
                    exc, workspace=self.workspace, vector_storage=self.vector_storage
                )
                if message is not None:
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
