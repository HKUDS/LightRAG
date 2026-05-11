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

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked db."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
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

    # The cypher statement is the second one (after the lock acquisition).
    cypher_sql = calls[1]["sql"]

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

    cypher_sql = calls[1]["sql"]
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

    cypher_sql = calls[1]["sql"]
    assert "CREATE (source)-[r:DIRECTED {}]->(target)" in cypher_sql


@pytest.mark.asyncio
async def test_upsert_edge_uses_parameterized_match_ids():
    """Source and target IDs must flow through Cypher parameters as agtype JSON."""
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(storage, "Node A", "Node B", {"weight": "1.0"})

    cypher_call = calls[1]
    cypher_sql = cypher_call["sql"]
    assert "entity_id: $src_id" in cypher_sql
    assert "entity_id: $tgt_id" in cypher_sql
    # Cypher params arrive as a single positional agtype JSON arg.
    params_json = cypher_call["args"][0]
    params = json.loads(params_json)
    assert params["src_id"] == "Node A"
    assert params["tgt_id"] == "Node B"


@pytest.mark.asyncio
async def test_upsert_edge_serialises_with_advisory_lock():
    """Concurrent upserts on the same edge must be serialised via pg_advisory_xact_lock.

    OPTIONAL MATCH + DELETE + CREATE is not atomic on its own: two transactions
    could both pass the OPTIONAL MATCH and both run CREATE, leaving duplicate
    DIRECTED rows. We open a transaction and acquire a transaction-scoped
    advisory lock keyed on the ordered (src_id, tgt_id) pair before running the
    cypher upsert, so concurrent upserts of the same logical edge run serially.

    AGE refuses to plan a join against a cypher() call that contains a CREATE
    clause, so the lock cannot live in a CTE — it must be a separate statement
    on the same connection inside an explicit transaction.
    """
    storage = make_graph_storage()
    calls = await _capture_upsert_edge(storage, "Alice", "Bob", {"weight": "0.5"})

    # Two statements: lock first, then cypher upsert.
    assert len(calls) == 2

    lock_sql = calls[0]["sql"]
    assert "pg_advisory_xact_lock" in lock_sql
    # Key must be order-independent so {A, B} and {B, A} collide on the same lock
    # (the OPTIONAL MATCH is undirected).
    assert "LEAST(" in lock_sql and "GREATEST(" in lock_sql
    # Raw node IDs are passed as positional params, not interpolated into the SQL.
    assert "Alice" not in lock_sql and "Bob" not in lock_sql
    assert calls[0]["args"] == ("Alice", "Bob")

    # The cypher statement must not contain the lock — that would cause AGE to
    # reject the plan with "cypher create clause cannot be rescanned".
    cypher_sql = calls[1]["sql"]
    assert "pg_advisory_xact_lock" not in cypher_sql
