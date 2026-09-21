"""Exercise startup diagnostics through FastAPI and Uvicorn's lifespan driver."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
import pytest
from uvicorn import Config
from uvicorn.lifespan.on import LifespanOn

from lightrag.api.startup_errors import StartupErrorMiddleware
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
@pytest.mark.parametrize(
    "tty,no_color,colored",
    [(True, False, True), (False, False, False), (True, True, False)],
)
async def test_empty_index_is_concise_but_still_fails_and_cleans_up(
    monkeypatch, caplog, tty, no_color, colored
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
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
    assert "lightrag-clear-storage" in messages
    assert "Traceback" not in messages
    assert ("\033[1;31m" in messages) is colored
    assert "Application startup complete" not in messages


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
