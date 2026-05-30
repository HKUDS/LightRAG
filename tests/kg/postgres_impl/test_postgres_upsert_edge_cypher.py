"""
Unit tests for PGGraphStorage.upsert_edge Cypher query generation.

Verifies the Cypher query sent to AGE uses the OPTIONAL MATCH + DELETE +
CREATE pattern with inline edge properties — the only reliable way to write
edge properties in Apache AGE (SET r += {...}, ON CREATE/ON MATCH SET, and
SET r.key = value all silently fail for DIRECTED edges).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

import asyncpg
from tenacity import wait_none

from lightrag.kg.postgres_impl import (
    PGGraphQueryException,
    PGGraphStorage,
    _is_transient_graph_write_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked db."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.__post_init__()  # resolves chunk-level batch-limit attrs
    storage.db = MagicMock()
    return storage


class _FakeConnection:
    """Captures statements + args passed to a fake asyncpg connection."""

    def __init__(self):
        self.calls: list[dict] = []

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        return ""


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _capture_upsert_edge(storage: PGGraphStorage, src: str, tgt: str, edge_data):
    """Invoke upsert_edge against a fake connection and return the captured calls."""
    conn = _FakeConnection()

    async def fake_run_with_retry(operation, **_kwargs):
        return await operation(conn)

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    await storage.upsert_edge(src, tgt, edge_data)
    return conn.calls


# ---------------------------------------------------------------------------
# upsert_edge: Cypher query correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edge_uses_delete_create_not_set():
    """Cypher must use OPTIONAL MATCH + DELETE + CREATE — not SET-based update.

    Apache AGE silently drops edge properties written via SET r += {...},
    SET r.key = value, and ON CREATE/ON MATCH SET. The only reliable pattern
    is to delete any existing edge and CREATE a new one with inline props.
    """
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(
        storage, "NodeA", "NodeB", {"weight": "1.0", "description": "test edge"}
    )

    # The batch/single edge path takes no advisory lock: a single cypher stmt.
    assert len(calls) == 1
    cypher_sql = calls[0]["sql"]

    # The new query must not contain any SET-based edge update — those silently
    # fail against AGE. All edge props live inline in the CREATE clause.
    assert (
        "SET r" not in cypher_sql
    ), f"Edge SET clauses are silently dropped by AGE: {cypher_sql}"
    assert "ON CREATE SET" not in cypher_sql
    assert "ON MATCH SET" not in cypher_sql


@pytest.mark.asyncio
async def test_upsert_edge_contains_optional_match_delete_create():
    """The Cypher query must use OPTIONAL MATCH + DELETE + CREATE with inline props."""
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(storage, "Alice", "Bob", {"weight": "0.5"})

    cypher_sql = calls[0]["sql"]
    assert "OPTIONAL MATCH (source)-[old:DIRECTED]-(target)" in cypher_sql
    assert "DELETE old" in cypher_sql
    assert "CREATE (source)-[r:DIRECTED" in cypher_sql
    assert "]->(target)" in cypher_sql
    # Edge properties must be inlined into the CREATE clause as a literal map.
    assert '`weight`: "0.5"' in cypher_sql
    assert "RETURN r" in cypher_sql


@pytest.mark.asyncio
async def test_upsert_edge_handles_empty_props():
    """Empty edge_data must inline an empty literal map, not crash."""
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(storage, "Alice", "Bob", {})

    cypher_sql = calls[0]["sql"]
    assert "CREATE (source)-[r:DIRECTED {}]->(target)" in cypher_sql


@pytest.mark.asyncio
async def test_upsert_edge_uses_parameterized_match_ids():
    """Source and target IDs must flow through Cypher parameters as agtype JSON."""
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(storage, "Node A", "Node B", {"weight": "1.0"})

    cypher_call = calls[0]
    cypher_sql = cypher_call["sql"]
    assert "entity_id: $src_id" in cypher_sql
    assert "entity_id: $tgt_id" in cypher_sql
    # Cypher params arrive as a single positional agtype JSON arg.
    params_json = cypher_call["args"][0]
    params = json.loads(params_json)
    assert params["src_id"] == "Node A"
    assert params["tgt_id"] == "Node B"


@pytest.mark.asyncio
async def test_upsert_edge_takes_no_advisory_lock():
    """The edge upsert relies on the single-writer contract, not a DB lock, so
    it must issue exactly one statement (the cypher) and no pg_advisory_xact_lock."""
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(storage, "Alice", "Bob", {"weight": "0.5"})

    assert len(calls) == 1
    assert "pg_advisory_xact_lock" not in calls[0]["sql"]
    assert "CREATE (source)-[r:DIRECTED" in calls[0]["sql"]


@pytest.mark.asyncio
async def test_upsert_edge_wraps_transient_errors_for_retry(monkeypatch):
    """Query-level transient errors must be wrapped in PGGraphQueryException so
    the outer @retry predicate can identify them and retry.

    Regression: upsert_edge runs the cypher via self.db._run_with_retry (not
    self._query), so the _query exception-wrapping path is bypassed. A raw
    asyncpg.DeadlockDetectedError surfacing from connection.execute would
    therefore fail _is_transient_graph_write_error's first guard
    (isinstance(exc, PGGraphQueryException)) and skip the retry loop, silently
    degrading concurrent ingestion under contention. This test pins the
    wrapping back in place and asserts the retry loop actually fires.
    """
    # Make the retry loop fire with zero backoff so the test stays fast.
    monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())

    storage = make_graph_storage()
    deadlock = asyncpg.exceptions.DeadlockDetectedError("simulated deadlock")

    call_count = 0

    async def fake_run_with_retry(_operation, **_kwargs):
        nonlocal call_count
        call_count += 1
        raise deadlock

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    with pytest.raises(PGGraphQueryException) as excinfo:
        await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})

    # The original asyncpg exception is preserved as __cause__ so the predicate
    # can introspect it via exc.__cause__.
    assert excinfo.value.__cause__ is deadlock
    # And the predicate now recognises this exception as retryable.
    assert _is_transient_graph_write_error(excinfo.value) is True
    # Retried up to stop_after_attempt(3) — proves the wrapping actually
    # engages the @retry loop rather than failing fast on the first attempt.
    assert call_count == 3


@pytest.mark.asyncio
async def test_upsert_edge_does_not_retry_non_transient_errors(monkeypatch):
    """Non-transient errors must not be retried by the @retry loop.

    The wrapping in upsert_edge unconditionally re-raises as
    PGGraphQueryException, but _is_transient_graph_write_error only returns
    True for a small set of asyncpg transient causes. A plain ValueError
    bubbling out of _run_with_retry should fail fast, not loop 3 times.
    """
    monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())

    storage = make_graph_storage()
    boom = ValueError("not a transient db error")

    call_count = 0

    async def fake_run_with_retry(_operation, **_kwargs):
        nonlocal call_count
        call_count += 1
        raise boom

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    with pytest.raises(PGGraphQueryException) as excinfo:
        await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})

    assert excinfo.value.__cause__ is boom
    assert _is_transient_graph_write_error(excinfo.value) is False
    assert call_count == 1


async def _capture_upsert_edges_batch(storage: PGGraphStorage, edges):
    """Run upsert_edges_batch against a fake connection; return captured calls."""
    conn = _FakeConnection()

    async def fake_run_with_retry(operation, **_kwargs):
        return await operation(conn)

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    await storage.upsert_edges_batch(edges)
    return conn.calls


@pytest.mark.asyncio
async def test_upsert_edges_batch_iterates_in_sorted_order():
    """A chunk emits edge cypher in canonical (LEAST, GREATEST) order regardless
    of insertion order (deterministic dedup / reproducible replay)."""
    storage = make_graph_storage()

    # Insertion order intentionally non-canonical: B-A, C-A, D-A.
    calls = await _capture_upsert_edges_batch(
        storage,
        [
            ("B", "A", {"weight": "1"}),
            ("C", "A", {"weight": "2"}),
            ("D", "A", {"weight": "3"}),
        ],
    )

    # The batch path takes no advisory lock; order is observed from the cypher
    # calls' bound (src_id, tgt_id) params, canonicalised.
    assert not any("pg_advisory_xact_lock" in c["sql"] for c in calls)
    cypher_calls = [c for c in calls if "CREATE (source)-[r:DIRECTED" in c["sql"]]
    edge_pairs = [
        (json.loads(c["args"][0])["src_id"], json.loads(c["args"][0])["tgt_id"])
        for c in cypher_calls
    ]
    canonical_keys = [tuple(sorted(pair)) for pair in edge_pairs]
    assert canonical_keys == sorted(canonical_keys)
    assert canonical_keys == [("A", "B"), ("A", "C"), ("A", "D")]


@pytest.mark.asyncio
async def test_upsert_edges_batch_dedupes_last_write_wins():
    """Reciprocal duplicates collapse to a single edge upsert carrying the
    LATEST edge_data, regardless of which orientation arrives last."""
    storage = make_graph_storage()

    calls = await _capture_upsert_edges_batch(
        storage,
        [
            ("A", "B", {"weight": "first"}),
            ("B", "A", {"weight": "second"}),  # reciprocal, wins
        ],
    )

    # Exactly one edge cypher (CREATE) ran, with the latest payload inlined and
    # the last write's orientation in the bound params.
    cypher_calls = [c for c in calls if "CREATE (source)-[r:DIRECTED" in c["sql"]]
    assert len(cypher_calls) == 1
    assert '"second"' in cypher_calls[0]["sql"]
    assert '"first"' not in cypher_calls[0]["sql"]
    params_json = cypher_calls[0]["args"][0]
    assert json.loads(params_json) == {"src_id": "B", "tgt_id": "A"}
