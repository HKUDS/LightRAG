"""
Unit tests for PGGraphStorage.upsert_edge Cypher query generation.

Verifies the Cypher query sent to AGE contains exactly one SET clause
(regression test for duplicate SET copy-paste bug).
"""

import json
import re
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
async def test_upsert_edge_single_set_clause():
    """The Cypher query must contain exactly one SET clause, not a duplicate."""
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

    # Count occurrences of SET in the Cypher query
    set_count = len(re.findall(r"\bSET\b", sql))
    assert set_count == 1, f"Expected 1 SET clause, found {set_count} in: {sql}"


@pytest.mark.asyncio
async def test_upsert_edge_contains_merge_and_set():
    """The Cypher query must contain MERGE and SET with edge properties."""
    storage = make_graph_storage()
    captured_sql: list[str] = []

    async def fake_query(sql, **kwargs):
        captured_sql.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge("Alice", "Bob", {"weight": "0.5"})

    sql = captured_sql[0]
    assert "MERGE" in sql
    assert "SET r +=" in sql
    assert "RETURN r" in sql


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
