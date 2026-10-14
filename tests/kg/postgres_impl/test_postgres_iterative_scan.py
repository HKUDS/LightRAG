"""HNSW iterative scan must be applied on every pooled connection.

The vector queries filter on ``workspace`` while the HNSW index is global, so
without ``hnsw.iterative_scan`` a workspace that holds a minority of the rows
can get 0 rows and no error.  The setting is session state: it has to be issued
on a new connection and again after ``RESET ALL`` on pool release.  See
``docs/design/PgVectorFilteredSearch.md``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB

SET_SQL = "SET hnsw.iterative_scan = strict_order"


def _db(index_type: str | None = "HNSW", enable_vector: bool = True) -> PostgreSQLDB:
    # Bypass __init__: only the attributes the code under test reads are set.
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.workspace = "gw"
    db.enable_vector = enable_vector
    db.vector_index_type = index_type
    return db


def _statements(connection: AsyncMock) -> list[str]:
    return [call.args[0] for call in connection.execute.await_args_list]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", ["HNSW", "HNSW_HALFVEC"])
async def test_hnsw_sets_iterative_scan_without_load(index_type) -> None:
    connection = AsyncMock()

    await _db(index_type).configure_hnsw_iterative_scan(connection)

    # A bare SET: LOAD 'vector' is refused for non-superusers (managed
    # databases), and a refused LOAD must not take the SET down with it.
    assert _statements(connection) == [SET_SQL]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", ["IVFFLAT", "VCHORDRQ", None])
async def test_non_hnsw_index_issues_nothing(index_type) -> None:
    connection = AsyncMock()

    await _db(index_type).configure_hnsw_iterative_scan(connection)

    assert _statements(connection) == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_missing_guc_is_not_fatal_but_is_logged() -> None:
    """pgvector < 0.8 has no such GUC: connect anyway, leave a trace."""
    connection = AsyncMock()
    connection.execute.side_effect = Exception("unrecognized configuration parameter")

    # LightRAG's logger does not propagate, so caplog sees nothing.
    with patch("lightrag.kg.postgres_impl.logger") as logger:
        await _db().configure_hnsw_iterative_scan(connection)  # must not raise

    assert logger.warning.called
    assert "iterative_scan" in str(logger.warning.call_args)


async def _pool_callbacks(db: PostgreSQLDB):
    """Run ``initdb`` against a stubbed asyncpg and return (init, reset)."""
    for name, value in {
        "user": "u",
        "password": "p",
        "database": "d",
        "host": "h",
        "port": 5432,
        "max": 2,
        "statement_cache_size": None,
        "ssl_mode": None,
        "server_settings": None,
        "connection_retry_attempts": 1,
        "connection_retry_backoff": 0,
        "connection_retry_backoff_max": 0,
    }.items():
        setattr(db, name, value)
    db._transient_exceptions = (ConnectionError,)
    db._create_ssl_context = lambda: None
    db.configure_vector_extension = AsyncMock()

    captured: dict = {}

    async def fake_create_pool(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("lightrag.kg.postgres_impl.asyncpg.connect", AsyncMock()),
        patch(
            "lightrag.kg.postgres_impl.asyncpg.create_pool",
            side_effect=fake_create_pool,
        ),
        patch("pgvector.asyncpg.register_vector", AsyncMock()),
    ):
        await db.initdb()
    return captured["init"], captured["reset"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_new_connection_gets_iterative_scan() -> None:
    init, _ = await _pool_callbacks(_db())
    connection = AsyncMock()

    with patch("pgvector.asyncpg.register_vector", AsyncMock()):
        await init(connection)

    assert _statements(connection) == [SET_SQL]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pool_reset_reapplies_iterative_scan_after_reset_all() -> None:
    """The failure only shows on a recycled connection: RESET ALL clears it."""
    _, reset = await _pool_callbacks(_db())
    connection = MagicMock()
    connection.get_reset_query.return_value = "RESET ALL"
    connection.execute = AsyncMock()

    await reset(connection)

    # Cleanup first, then the setting; the other order would be undone.
    assert [c.args[0] for c in connection.execute.await_args_list] == [
        "RESET ALL",
        SET_SQL,
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_vector_disabled_backend_never_touches_the_guc() -> None:
    _, reset = await _pool_callbacks(_db(enable_vector=False))
    connection = MagicMock()
    connection.get_reset_query.return_value = "RESET ALL"
    connection.execute = AsyncMock()

    await reset(connection)

    assert [c.args[0] for c in connection.execute.await_args_list] == ["RESET ALL"]
