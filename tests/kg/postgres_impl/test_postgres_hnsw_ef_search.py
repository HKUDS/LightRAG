"""HNSW vector queries must widen ``hnsw.ef_search`` to their ``top_k``.

pgvector's HNSW scan returns at most ``hnsw.ef_search`` candidates (default
40), whatever the ``LIMIT``. Without iterative scan (pgvector < 0.8) a
``top_k`` above that silently returns fewer rows; with it the rows are
complete but the first batch's recall is lower. The setting is raised for the
query's own transaction only, never lowered below the value already in force,
and capped at 1000 (the GUC's maximum). See
``docs/design/PgVectorFilteredSearch.md#hnsw-candidate-list-ef_search``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import PGVectorStorage, PostgreSQLDB

pytestmark = pytest.mark.offline

HNSW_SQL = "SET LOCAL hnsw.iterative_scan = strict_order"
IVFFLAT_SQL = "SET LOCAL ivfflat.iterative_scan = relaxed_order"


def _ef_sql(value: int) -> str:
    return (
        "SELECT set_config('hnsw.ef_search', GREATEST(COALESCE(NULLIF("
        "current_setting('hnsw.ef_search', true), '')::int, 40), "
        f"{value})::text, true)"
    )


def _db(index_type: str | None = "HNSW", *, supported=None) -> PostgreSQLDB:
    # Bypass __init__: only the attributes the code under test reads are set.
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.workspace = "gw"
    db.vector_index_type = index_type
    db.iterative_scan_supported = supported
    return db


class _Connection:
    """Records the order of transaction boundaries and statements."""

    def __init__(self, *, execute_error: Exception | None = None):
        self.log: list[str] = []
        self._execute_error = execute_error

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
        # No bound arguments: a multi-statement string needs the simple protocol.
        assert args == ()
        self.log.append(sql)
        if self._execute_error is not None:
            raise self._execute_error

    async def fetch(self, sql, *args):
        self.log.append("FETCH")
        return [{"id": "a"}]


# Both HNSW index types share one statement: every top_k on HNSW, and the
# cap once on HNSW_HALFVEC.
@pytest.mark.parametrize(
    ("index_type", "top_k", "value"),
    [
        ("HNSW", top_k, value)
        for top_k, value in (
            (10, 10),
            (1000, 1000),
            (1001, 1000),
        )
    ]
    + [("HNSW_HALFVEC", 1001, 1000)],
)
def test_statement_raises_to_top_k_capped_at_the_guc_maximum(
    index_type, top_k, value
) -> None:
    # GREATEST(current, top_k): a value already in force (operator setting,
    # else pgvector's default 40) is never lowered; set_config(..., true) is
    # transaction-local like SET LOCAL.
    assert _db(index_type).ef_search_statement(top_k) == _ef_sql(value)


@pytest.mark.parametrize("index_type", ["IVFFLAT", None])
def test_non_hnsw_index_has_no_statement(index_type) -> None:
    assert _db(index_type).ef_search_statement(100) is None


@pytest.mark.parametrize("top_k", [None, 0])
def test_no_positive_top_k_has_no_statement(top_k) -> None:
    assert _db().ef_search_statement(top_k) is None


@pytest.mark.asyncio
async def test_both_settings_share_one_round_trip_in_the_query_transaction() -> None:
    connection = _Connection()

    rows = await _db()._fetch_with_iterative_scan(connection, "SELECT 1", (), top_k=100)

    assert connection.log == [
        "BEGIN",
        f"{HNSW_SQL}; {_ef_sql(100)}",
        "FETCH",
        "COMMIT",
    ]
    assert rows == [{"id": "a"}]


@pytest.mark.asyncio
async def test_old_pgvector_still_gets_ef_search() -> None:
    """Without iterative scan the candidate list is the only bound left."""
    connection = _Connection()

    await _db(supported=False)._fetch_with_iterative_scan(
        connection, "SELECT 1", (), top_k=100
    )

    assert connection.log == ["BEGIN", _ef_sql(100), "FETCH", "COMMIT"]


@pytest.mark.asyncio
async def test_ivfflat_gets_only_its_iterative_scan() -> None:
    connection = _Connection()

    await _db("IVFFLAT")._fetch_with_iterative_scan(
        connection, "SELECT 1", (), top_k=100
    )

    assert connection.log == ["BEGIN", IVFFLAT_SQL, "FETCH", "COMMIT"]


@pytest.mark.asyncio
async def test_failing_settings_fall_back_to_a_plain_fetch() -> None:
    connection = _Connection(execute_error=Exception("boom"))

    rows = await _db()._fetch_with_iterative_scan(connection, "SELECT 1", (), top_k=100)

    assert connection.log == [
        "BEGIN",
        f"{HNSW_SQL}; {_ef_sql(100)}",
        "ROLLBACK",
        "FETCH",
    ]
    assert rows == [{"id": "a"}]


@pytest.mark.asyncio
async def test_db_query_hands_ann_top_k_to_the_scan_helper() -> None:
    db = _db()
    connection = _Connection()

    async def run_with_retry(operation, **kwargs):
        return await operation(connection)

    db._run_with_retry = run_with_retry

    await db.query("SELECT 1", multirows=True, iterative_scan=True, ann_top_k=100)

    assert connection.log[1] == f"{HNSW_SQL}; {_ef_sql(100)}"


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [10])
async def test_storage_query_passes_its_top_k(top_k) -> None:
    storage = PGVectorStorage.__new__(PGVectorStorage)
    storage.namespace = "chunks"
    storage.workspace = "gw"
    storage.table_name = "t"
    storage.cosine_better_than_threshold = 0.2
    storage.db = _db()
    storage.db.query = AsyncMock(return_value=[])

    await storage.query("q", top_k=top_k, query_embedding=[0.1, 0.2])

    call = storage.db.query.await_args
    assert call.kwargs["iterative_scan"] is True
    assert call.kwargs["ann_top_k"] == top_k
    assert call.kwargs["params"][2] == top_k  # the same value is the LIMIT
