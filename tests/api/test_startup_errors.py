"""Exercise startup diagnostics through FastAPI and Uvicorn's lifespan driver."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
import pytest
from uvicorn import Config
from uvicorn.lifespan.on import LifespanOn

from lightrag.api.startup_errors import StartupErrorMiddleware
from lightrag.exceptions import (
    CorruptStorageSnapshotError,
    EmbeddingBaselineMismatchError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)

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


def _lifespan_messages(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,headline,detail",
    [
        (
            VectorSpaceMismatchError(
                backend="FaissVectorDBStorage",
                container="vdb_chunks.space.json",
                expected_model="new-model",
                expected_dim=1024,
                stored_model="old-model",
                stored_dim=1024,
            ),
            "different embedding space",
            "old-model",
        ),
        (
            EmbeddingBaselineMismatchError(
                workspace="space1",
                mismatches=[
                    {
                        "target": "chunks",
                        "recorded_model": "old-model",
                        "recorded_dim": 1024,
                        "expected_model": "new-model",
                        "expected_dim": 3072,
                    }
                ],
            ),
            "recorded embedding baseline differs",
            "Mismatched targets: chunks",
        ),
        (
            CorruptStorageSnapshotError(
                backend="NanoVectorDBStorage",
                container="vdb_entities.json",
                detail="JSONDecodeError: Expecting value",
                artifacts=("vdb_entities.json",),
            ),
            "snapshot is corrupt",
            "JSONDecodeError: Expecting value",
        ),
    ],
    ids=["space-mismatch", "baseline-mismatch", "corrupt-snapshot"],
)
async def test_every_vector_refusal_names_both_recoveries(
    monkeypatch, caplog, error, headline, detail
):
    """Each startup-time vector refusal is an expected condition with the
    same two ways out: a rebuild, or -- when the data is disposable -- the
    offline clear. The concise message names both and drops the traceback."""
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    cleaned = []
    life = driver(make_app(error, cleaned))
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        await life.startup()
    messages = _lifespan_messages(caplog)
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert "Startup blocked" in messages
    assert headline in messages
    assert detail in messages
    assert "space1" in messages and "FaissVectorDBStorage" in messages
    assert "lightrag-rebuild-vdb" in messages
    assert "lightrag-clear-storage" in messages
    assert "Traceback" not in messages


@pytest.mark.parametrize(
    "error",
    [
        VectorStorageEmptyError(vdb_name="chunks", source="text chunk storage"),
        VectorSpaceMismatchError(
            backend="PGVectorStorage",
            container="lightrag_vdb_chunks_new_model",
            expected_model="new-model",
            stored_model="old-model",
        ),
        EmbeddingBaselineMismatchError(
            workspace="",
            mismatches=[
                {
                    "target": "entities",
                    "recorded_model": "a",
                    "recorded_dim": 8,
                    "expected_model": "b",
                    "expected_dim": 8,
                }
            ],
        ),
        CorruptStorageSnapshotError(
            backend="FaissVectorDBStorage",
            container="vdb_chunks.index",
            detail="truncated",
            artifacts=("vdb_chunks.index",),
        ),
    ],
    ids=["empty", "space-mismatch", "baseline-mismatch", "corrupt-snapshot"],
)
def test_the_exception_text_itself_names_the_clear_tool(error):
    """The library raises these outside the server too (a script calling
    ``initialize_storages``, the Gunicorn master), where no middleware
    rewrites them, so the guidance has to live in the message."""
    assert "lightrag-rebuild-vdb" in str(error)
    assert "lightrag-clear-storage" in str(error)
