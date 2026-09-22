"""Exercise startup diagnostics through FastAPI and Uvicorn's lifespan driver."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
import pytest
from uvicorn import Config
from uvicorn.lifespan.on import LifespanOn

from lightrag.api.startup_errors import (
    ConsoleFormatter,
    StartupDiagnostic,
    StartupErrorMiddleware,
)
from lightrag.exceptions import VectorStorageEmptyError

pytestmark = pytest.mark.offline


def make_app(error, cleaned, *, shutdown=False):
    @asynccontextmanager
    async def lifespan(app):
        try:
            if not shutdown:
                raise error
            yield
            raise error
        finally:
            cleaned.append(True)

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        StartupErrorMiddleware,
        workspace="space1",
        vector_storage="FaissVectorDBStorage",
    )
    return app


def driver(app):
    return LifespanOn(Config(app, lifespan="on", log_config=None))


@pytest.mark.asyncio
@pytest.mark.parametrize("tty", [True, False])
async def test_empty_index_is_concise_but_still_fails_and_cleans_up(
    monkeypatch, caplog, tty
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    cleaned = []
    error = VectorStorageEmptyError(vdb_name="chunks", source="text chunk storage")
    life = driver(make_app(error, cleaned))
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        await life.startup()
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert "Startup blocked" in messages
    assert "space1" in messages and "FaissVectorDBStorage" in messages
    assert "chunks" in messages and "text chunk storage" in messages
    assert "lightrag-rebuild-vdb" in messages
    assert "Traceback" not in messages
    assert "Application startup complete" not in messages
    # Uvicorn logs this message to every handler, the log file included, so
    # the message itself never carries terminal escapes -- even on a TTY.
    assert "\033[" not in messages
    diagnostic = [r for r in caplog.records if "Startup blocked" in r.getMessage()]
    assert len(diagnostic) == 1
    assert isinstance(diagnostic[0].msg, StartupDiagnostic)


@pytest.mark.parametrize(
    "tty,no_color,colored",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_the_console_formatter_highlights_a_diagnostic_only_on_a_terminal(
    monkeypatch, tty, no_color, colored
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    formatter = ConsoleFormatter("%(levelname)s: %(message)s")

    def record(msg):
        return logging.LogRecord("uvicorn.error", logging.ERROR, "", 0, msg, None, None)

    assert ("\033[1;31m" in formatter.format(record(StartupDiagnostic("x")))) is colored
    # Only the diagnostic is highlighted, never an ordinary error.
    assert "\033[" not in formatter.format(record("Application startup failed."))


@pytest.mark.asyncio
async def test_the_log_file_stays_plain_while_the_terminal_is_highlighted(
    monkeypatch, tmp_path, capsys
):
    """``configure_logging`` sends ``uvicorn.error`` to the console AND the
    log file; only the console may carry the highlight."""
    # Importing the server module parses sys.argv as server arguments.
    monkeypatch.setattr("sys.argv", ["lightrag-server"])
    from lightrag.api import lightrag_server

    names = ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]
    saved = {
        name: (lg.handlers[:], lg.filters[:], lg.level, lg.propagate)
        for name in names
        for lg in [logging.getLogger(name)]
    }
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    try:
        lightrag_server.configure_logging()
        life = driver(
            make_app(VectorStorageEmptyError(vdb_name="chunks"), cleaned := [])
        )
        await life.startup()
        for handler in logging.getLogger("uvicorn.error").handlers:
            handler.flush()
        log_file = (tmp_path / lightrag_server.DEFAULT_LOG_FILENAME).read_text(
            encoding="utf-8"
        )
    finally:
        for name, (handlers, filters, level, propagate) in saved.items():
            lg = logging.getLogger(name)
            for handler in lg.handlers:
                if handler not in handlers:
                    handler.close()
            lg.handlers, lg.filters = handlers, filters
            lg.setLevel(level)
            lg.propagate = propagate

    assert life.startup_failed and cleaned == [True]
    assert "Startup blocked" in log_file
    assert "\033[" not in log_file
    assert "\033[1;31mERROR: Startup blocked" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_the_storage_workspace_wins_over_the_server_workspace(caplog):
    """A backend override (QDRANT_WORKSPACE, ...) moves the vector storage
    off the server's workspace; the diagnostic names the one that is empty."""
    error = VectorStorageEmptyError(vdb_name="chunks", workspace="qdrant_scope")
    life = driver(make_app(error, []))
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.startup()
    assert "Workspace: qdrant_scope" in caplog.text
    assert "space1" not in caplog.text


@pytest.mark.asyncio
async def test_unexpected_startup_error_keeps_traceback(caplog):
    cleaned = []
    life = driver(make_app(RuntimeError("connection broke"), cleaned))
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.startup()
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert "Traceback" in caplog.text
    assert "connection broke" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_error_is_not_reclassified(caplog):
    cleaned = []
    life = driver(
        make_app(VectorStorageEmptyError(vdb_name="chunks"), cleaned, shutdown=True)
    )
    await life.startup()
    assert not life.should_exit
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.shutdown()
    assert life.shutdown_failed and life.should_exit
    assert cleaned == [True]
    assert "Traceback" in caplog.text
    assert "Startup blocked" not in caplog.text


@pytest.mark.asyncio
async def test_failure_message_without_exception_is_forwarded(caplog):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "lifespan.startup.failed", "message": "explicit refusal"})

    life = driver(
        StartupErrorMiddleware(app, workspace="space1", vector_storage="test")
    )
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.startup()
    assert life.startup_failed and life.should_exit
    assert "explicit refusal" in caplog.text
