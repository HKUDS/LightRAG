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

from lightrag.kg.postgres_impl import (
    SQL_TEMPLATES,
    PostgreSQLDB,
    _with_distance_resort,
)

SET_SQL = "SET hnsw.iterative_scan = strict_order"
IVFFLAT_SET_SQL = "SET ivfflat.iterative_scan = relaxed_order"


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

    await _db(index_type).configure_iterative_scan(connection)

    # A bare SET: LOAD 'vector' is refused for non-superusers (managed
    # databases), and a refused LOAD must not take the SET down with it.
    assert _statements(connection) == [SET_SQL]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", ["VCHORDRQ", None])
async def test_uncovered_index_issues_nothing(index_type) -> None:
    connection = AsyncMock()

    await _db(index_type).configure_iterative_scan(connection)

    assert _statements(connection) == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ivfflat_uses_relaxed_order_only() -> None:
    """IVFFLAT rejects strict_order (server WARNING, value ignored)."""
    connection = AsyncMock()

    await _db("IVFFLAT").configure_iterative_scan(connection)

    assert _statements(connection) == [IVFFLAT_SET_SQL]


@pytest.mark.offline
@pytest.mark.parametrize("namespace", ["chunks", "entities", "relationships"])
@pytest.mark.parametrize("vector_cast", ["vector", "halfvec"])
def test_resort_wrapper_restores_distance_order(namespace, vector_cast) -> None:
    """relaxed_order rows come out unsorted and the planner adds no sort, so the
    scan is materialised and re-sorted; ``+ 0`` defeats sort elimination."""
    raw = SQL_TEMPLATES[namespace].format(table_name="t", vector_cast=vector_cast)

    wrapped = _with_distance_resort(raw, vector_cast)

    assert wrapped.startswith("WITH relaxed AS MATERIALIZED (")
    assert wrapped.endswith("ORDER BY _distance + 0")
    assert wrapped.count(f"content_vector <=> $4::{vector_cast} AS _distance") == 1
    assert "LIMIT $3" in wrapped  # the bound stays inside the scan
    assert ";" not in wrapped


@pytest.mark.offline
def test_resort_wrapper_refuses_a_template_it_cannot_parse() -> None:
    with pytest.raises(RuntimeError):
        _with_distance_resort("SELECT 1", "vector")


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index_type", "wrapped"),
    [("IVFFLAT", True), ("HNSW", False), ("HNSW_HALFVEC", False)],
)
async def test_query_resorts_only_for_ivfflat(index_type, wrapped) -> None:
    from lightrag.kg.postgres_impl import PGVectorStorage

    storage = PGVectorStorage.__new__(PGVectorStorage)
    storage.namespace = "chunks"
    storage.workspace = "gw"
    storage.table_name = "t"
    storage.cosine_better_than_threshold = 0.2
    storage.db = _db(index_type)
    storage.db.query = AsyncMock(
        return_value=[{"id": "a", "content": "x", "_distance": 0.1}]
    )

    rows = await storage.query("q", top_k=5, query_embedding=[0.1, 0.2])

    sql = storage.db.query.await_args.args[0]
    assert sql.startswith("WITH relaxed AS MATERIALIZED") is wrapped
    if wrapped:
        # the helper column never reaches callers
        assert rows == [{"id": "a", "content": "x"}]
    else:
        assert rows == [{"id": "a", "content": "x", "_distance": 0.1}]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_skips_resort_when_iterative_scan_unsupported() -> None:
    from lightrag.kg.postgres_impl import PGVectorStorage

    storage = PGVectorStorage.__new__(PGVectorStorage)
    storage.namespace = "chunks"
    storage.workspace = "gw"
    storage.table_name = "t"
    storage.cosine_better_than_threshold = 0.2
    storage.db = _db("IVFFLAT")
    storage.db.iterative_scan_supported = False
    storage.db.query = AsyncMock(return_value=[])

    await storage.query("q", top_k=5, query_embedding=[0.1])

    assert not storage.db.query.await_args.args[0].startswith("WITH relaxed")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_set_failure_is_not_fatal_and_logged_once() -> None:
    """A failing SET must not fail the connection, nor flood the log."""
    connection = AsyncMock()
    connection.execute.side_effect = Exception("permission denied")
    db = _db()

    # LightRAG's logger does not propagate, so caplog sees nothing.
    with patch("lightrag.kg.postgres_impl.logger") as logger:
        await db.configure_iterative_scan(connection)  # must not raise
        await db.configure_iterative_scan(connection)  # every pool release

    assert logger.warning.call_count == 1
    assert "iterative scan not applied" in str(logger.warning.call_args)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_old_pgvector_skips_the_set() -> None:
    """On a fresh connection an unknown hnsw.* SET is accepted and dropped
    silently, so the version decides, not the exception."""
    connection = AsyncMock()
    db = _db()
    db.iterative_scan_supported = False

    await db.configure_iterative_scan(connection)

    assert _statements(connection) == []


def _version_connection(version: str | None) -> AsyncMock:
    connection = AsyncMock()
    connection.fetchrow.return_value = (
        None if version is None else {"extversion": version}
    )
    return connection


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "supported"),
    [("0.7.4", False), ("0.8.0", True), ("0.8.6", True), ("1.0.0", True)],
)
async def test_version_probe_decides_support(version, supported) -> None:
    db = _db()
    with patch("lightrag.kg.postgres_impl.logger") as logger:
        await db._probe_iterative_scan_support(_version_connection(version))

    assert db.iterative_scan_supported is supported
    assert logger.warning.called is (not supported)


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, "garbage"])
async def test_unreadable_version_leaves_support_unknown(version) -> None:
    db = _db()
    db.iterative_scan_supported = None

    await db._probe_iterative_scan_support(_version_connection(version))

    assert db.iterative_scan_supported is None  # SET is still attempted


@pytest.mark.offline
@pytest.mark.asyncio
async def test_version_probe_failure_is_not_fatal() -> None:
    connection = AsyncMock()
    connection.fetchrow.side_effect = Exception("boom")
    db = _db()
    db.iterative_scan_supported = None

    await db._probe_iterative_scan_support(connection)  # must not raise

    assert db.iterative_scan_supported is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_version_probe_ignores_uncovered_index() -> None:
    connection = _version_connection("0.7.0")

    await _db("VCHORDRQ")._probe_iterative_scan_support(connection)

    assert connection.fetchrow.await_count == 0


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
