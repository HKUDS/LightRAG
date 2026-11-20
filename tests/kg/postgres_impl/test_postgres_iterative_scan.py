"""Filtered ANN search must run with iterative scan inside its own transaction.

The vector queries filter on ``workspace`` while the vector index is global, so
without an iterative scan a workspace that holds a minority of the rows can get
0 rows and no error.  The setting is applied with ``SET LOCAL`` in the query's
own transaction: behind a transaction-pooling proxy a session ``SET`` may land
on another backend than the query.  See ``docs/design/PgVectorFilteredSearch.md``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import (
    SQL_TEMPLATES,
    PGVectorStorage,
    PostgreSQLDB,
    _with_distance_resort,
)

HNSW_SQL = "SET LOCAL hnsw.iterative_scan = strict_order"
IVFFLAT_SQL = "SET LOCAL ivfflat.iterative_scan = relaxed_order"


def _db(index_type: str | None = "HNSW", enable_vector: bool = True) -> PostgreSQLDB:
    # Bypass __init__: only the attributes the code under test reads are set.
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.workspace = "gw"
    db.enable_vector = enable_vector
    db.vector_index_type = index_type
    return db


class _Connection:
    """Records the order of transaction boundaries and statements."""

    def __init__(self, *, execute_error: Exception | None = None, rows=None):
        self.log: list[str] = []
        self._execute_error = execute_error
        self._rows = rows if rows is not None else [{"id": "a"}]
        self.fetch_error: Exception | None = None

    @asynccontextmanager
    async def transaction(self):
        self.log.append("BEGIN")
        try:
            yield
        except BaseException:
            self.log.append("ROLLBACK")
            raise
        else:
            self.log.append("COMMIT")

    async def execute(self, sql, *args):
        self.log.append(sql)
        if self._execute_error is not None:
            raise self._execute_error

    async def fetch(self, sql, *args):
        self.log.append("FETCH")
        if self.fetch_error is not None:
            raise self.fetch_error
        return self._rows


@pytest.mark.offline
@pytest.mark.parametrize(
    ("index_type", "statement"),
    [
        ("HNSW", HNSW_SQL),
        ("HNSW_HALFVEC", HNSW_SQL),
        ("IVFFLAT", IVFFLAT_SQL),
        ("VCHORDRQ", None),
        (None, None),
    ],
)
def test_statement_per_index_type(index_type, statement) -> None:
    # Never LOAD 'vector': refused for non-superusers (managed databases).
    assert _db(index_type).iterative_scan_statement() == statement


@pytest.mark.offline
def test_old_pgvector_has_no_statement() -> None:
    """On a fresh connection an unknown hnsw.* SET is accepted and dropped
    silently, so the version decides, not the exception."""
    db = _db()
    db.iterative_scan_supported = False

    assert db.iterative_scan_statement() is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_setting_is_local_to_the_query_transaction() -> None:
    """Setting and query share one transaction, in this order.

    This is the property a transaction-pooling proxy needs: nothing is carried
    in session state between the SET and the SELECT.
    """
    connection = _Connection()

    rows = await _db()._fetch_with_iterative_scan(connection, "SELECT 1", ())

    assert connection.log == ["BEGIN", HNSW_SQL, "FETCH", "COMMIT"]
    assert rows == [{"id": "a"}]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_uncovered_index_runs_a_plain_fetch() -> None:
    connection = _Connection()

    await _db("VCHORDRQ")._fetch_with_iterative_scan(connection, "SELECT 1", ())

    assert connection.log == ["FETCH"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_failing_set_falls_back_and_warns_once() -> None:
    """A failed SET LOCAL must neither fail the query nor flood the log."""
    db = _db()

    with patch("lightrag.kg.postgres_impl.logger") as logger:
        for _ in range(2):
            connection = _Connection(execute_error=Exception("permission denied"))
            rows = await db._fetch_with_iterative_scan(connection, "SELECT 1", ())
            # its own transaction is rolled back, then the query runs plain
            assert connection.log == ["BEGIN", HNSW_SQL, "ROLLBACK", "FETCH"]
            assert rows == [{"id": "a"}]

    assert logger.warning.call_count == 1
    assert "iterative scan not applied" in str(logger.warning.call_args)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_failure_is_reraised_without_a_plain_fallback() -> None:
    connection = _Connection()
    connection.fetch_error = RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        await _db()._fetch_with_iterative_scan(connection, "SELECT 1", ())

    # the helper does not rerun it plain; _run_with_retry owns any retry
    assert connection.log.count("FETCH") == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pool_callbacks_carry_no_session_setting() -> None:
    """Nothing is set at connection init or pool reset any more."""
    db = _db()
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
        connection = AsyncMock()
        await captured["init"](connection)
        assert [c.args[0] for c in connection.execute.await_args_list] == []

        connection = MagicMock()
        connection.get_reset_query.return_value = "RESET ALL"
        connection.execute = AsyncMock()
        await captured["reset"](connection)
        assert [c.args[0] for c in connection.execute.await_args_list] == ["RESET ALL"]


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

    assert db.iterative_scan_supported is None  # the SET LOCAL is still attempted


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


def _storage(index_type: str, *, supported=None) -> PGVectorStorage:
    storage = PGVectorStorage.__new__(PGVectorStorage)
    storage.namespace = "chunks"
    storage.workspace = "gw"
    storage.table_name = "t"
    storage.cosine_better_than_threshold = 0.2
    storage.db = _db(index_type)
    storage.db.iterative_scan_supported = supported
    storage.db.query = AsyncMock(
        return_value=[{"id": "a", "content": "x", "_distance": 0.1}]
    )
    return storage


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index_type", "wrapped"),
    [("IVFFLAT", True), ("HNSW", False), ("HNSW_HALFVEC", False)],
)
async def test_query_resorts_only_for_ivfflat(index_type, wrapped) -> None:
    storage = _storage(index_type)

    rows = await storage.query("q", top_k=5, query_embedding=[0.1, 0.2])

    call = storage.db.query.await_args
    assert call.args[0].startswith("WITH relaxed AS MATERIALIZED") is wrapped
    assert call.kwargs["iterative_scan"] is True
    if wrapped:
        # the helper column never reaches callers
        assert rows == [{"id": "a", "content": "x"}]
    else:
        assert rows == [{"id": "a", "content": "x", "_distance": 0.1}]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_skips_resort_when_iterative_scan_unsupported() -> None:
    storage = _storage("IVFFLAT", supported=False)

    await storage.query("q", top_k=5, query_embedding=[0.1])

    assert not storage.db.query.await_args.args[0].startswith("WITH relaxed")
