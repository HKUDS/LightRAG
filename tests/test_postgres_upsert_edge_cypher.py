"""
Unit tests for PGGraphStorage.upsert_edge Cypher query generation.

Verifies the Cypher query sent to AGE uses the OPTIONAL MATCH + DELETE +
CREATE pattern with inline edge properties — the only reliable way to write
edge properties in Apache AGE (SET r += {...}, ON CREATE/ON MATCH SET, and
SET r.key = value all silently fail for DIRECTED edges).
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked _query method."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.db = MagicMock()
    return storage


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
    captured_sql: list[str] = []

    async def fake_query(sql, **kwargs):
        captured_sql.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge(
            "NodeA", "NodeB", {"weight": "1.0", "description": "test edge"}
        )

    assert len(captured_sql) == 1
    sql = captured_sql[0]

    # The new query must not contain any SET-based edge update — those silently
    # fail against AGE. All edge props live inline in the CREATE clause.
    assert "SET r" not in sql, f"Edge SET clauses are silently dropped by AGE: {sql}"
    assert "ON CREATE SET" not in sql
    assert "ON MATCH SET" not in sql


@pytest.mark.asyncio
async def test_upsert_edge_contains_optional_match_delete_create():
    """The Cypher query must use OPTIONAL MATCH + DELETE + CREATE with inline props."""
    storage = make_graph_storage()
    captured_sql: list[str] = []

    async def fake_query(sql, **kwargs):
        captured_sql.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge("Alice", "Bob", {"weight": "0.5"})

    sql = captured_sql[0]
    assert "OPTIONAL MATCH (source)-[old:DIRECTED]-(target)" in sql
    assert "DELETE old" in sql
    assert "CREATE (source)-[r:DIRECTED" in sql
    assert "]->(target)" in sql
    # Edge properties must be inlined into the CREATE clause as a literal map.
    assert '`weight`: "0.5"' in sql
    assert "RETURN r" in sql


@pytest.mark.asyncio
async def test_upsert_edge_handles_empty_props():
    """Empty edge_data must inline an empty literal map, not crash."""
    storage = make_graph_storage()
    captured_sql: list[str] = []

    async def fake_query(sql, **kwargs):
        captured_sql.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge("Alice", "Bob", {})

    sql = captured_sql[0]
    assert "CREATE (source)-[r:DIRECTED {}]->(target)" in sql


@pytest.mark.asyncio
async def test_upsert_edge_uses_parameterized_match_ids():
    """Source and target IDs must flow through Cypher parameters."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge("Node A", "Node B", {"weight": "1.0"})

    call = captured_calls[0]
    sql = call["sql"]
    assert "entity_id: $src_id" in sql
    assert "entity_id: $tgt_id" in sql
    params = json.loads(call["params"]["params"])
    assert params["src_id"] == "Node A"
    assert params["tgt_id"] == "Node B"
