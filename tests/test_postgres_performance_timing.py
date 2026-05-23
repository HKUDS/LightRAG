import importlib
from unittest.mock import AsyncMock, patch

import pytest

import lightrag.utils as utils_module
from lightrag.kg.postgres_impl import PGGraphStorage, PostgreSQLDB
from lightrag.namespace import NameSpace


def make_db() -> PostgreSQLDB:
    return PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "postgres",
            "workspace": "test_ws",
            "max_connections": 10,
            "connection_retry_attempts": 3,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 5.0,
        }
    )


@pytest.mark.asyncio
async def test_execute_timing_logs_success():
    db = make_db()

    async def fake_run_with_retry(operation, **kwargs):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        await operation(conn)

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    with patch("lightrag.kg.postgres_impl.performance_timing_log") as timing_log:
        await db.execute("SELECT 1", timing_label="test label")

    assert any(
        "connection.execute completed" in call.args[0]
        for call in timing_log.call_args_list
    )


@pytest.mark.asyncio
async def test_execute_timing_logs_failure():
    db = make_db()

    async def fake_run_with_retry(operation, **kwargs):
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
        await operation(conn)

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    with patch("lightrag.kg.postgres_impl.performance_timing_log") as timing_log:
        with pytest.raises(RuntimeError, match="boom"):
            await db.execute("SELECT 1", timing_label="test label")

    assert any(
        "connection.execute failed" in call.args[0]
        for call in timing_log.call_args_list
    )


@pytest.mark.asyncio
async def test_graph_upsert_node_passes_timing_label():
    storage = PGGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace="test_ws",
        global_config={},
        embedding_func=AsyncMock(),
    )
    storage.graph_name = "test_graph"
    storage._query = AsyncMock(return_value=[])

    await storage.upsert_node(
        "node-1",
        {
            "entity_id": "node-1",
            "description": "desc",
        },
    )

    assert storage._query.await_args.kwargs["timing_label"] == (
        "test_ws PGGraphStorage.upsert_node"
    )


@pytest.mark.asyncio
async def test_graph_upsert_edge_passes_timing_label():
    storage = PGGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace="test_ws",
        global_config={},
        embedding_func=AsyncMock(),
    )
    storage.graph_name = "test_graph"
    storage._query = AsyncMock(return_value=[])

    await storage.upsert_edge(
        "node-1",
        "node-2",
        {
            "weight": 1.0,
            "description": "desc",
        },
    )

    assert storage._query.await_args.kwargs["timing_label"] == (
        "test_ws PGGraphStorage.upsert_edge"
    )


def test_performance_timing_logs_reads_new_env_only(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("LIGHTRAG_DOC_QUERY_TIMING_LOGS", "false")
        m.setenv("LIGHTRAG_PERFORMANCE_TIMING_LOGS", "true")
        reloaded = importlib.reload(utils_module)
        assert reloaded.PERFORMANCE_TIMING_LOGS is True

    importlib.reload(utils_module)


def test_performance_timing_logs_ignores_old_env(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("LIGHTRAG_DOC_QUERY_TIMING_LOGS", "true")
        m.setenv("LIGHTRAG_PERFORMANCE_TIMING_LOGS", "false")
        reloaded = importlib.reload(utils_module)
        assert reloaded.PERFORMANCE_TIMING_LOGS is False

    importlib.reload(utils_module)
