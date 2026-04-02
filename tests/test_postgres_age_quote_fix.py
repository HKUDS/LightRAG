"""
Unit tests for PGGraphStorage.get_nodes_edges_batch with special characters in entity names.

Verifies the fix for KeyError when entity names contain double quotes (PR #2871).
The root cause: AGE returns the original un-escaped entity_id, but the edges_norm
dict was previously keyed with the normalized (escaped) ID, causing a KeyError on lookup.
"""

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
# _normalize_node_id
# ---------------------------------------------------------------------------


def test_normalize_plain_id():
    assert PGGraphStorage._normalize_node_id("Alice") == "Alice"


def test_normalize_double_quote():
    assert PGGraphStorage._normalize_node_id('John "Smith"') == 'John \\"Smith\\"'


def test_normalize_backslash():
    assert PGGraphStorage._normalize_node_id("C:\\path") == "C:\\\\path"


def test_normalize_both_special_chars():
    assert (
        PGGraphStorage._normalize_node_id('say \\"hello\\"')
        == 'say \\\\\\"hello\\\\\\"'
    )


# ---------------------------------------------------------------------------
# get_nodes_edges_batch — entity names with double quotes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_with_quoted_entity():
    """
    AGE returns the original un-escaped node_id in query results.
    edges_norm must be keyed with the original ID so the result lookup succeeds.
    """
    storage = make_graph_storage()
    entity = 'John "Smith"'

    # Simulate AGE returning the original (un-escaped) node_id
    outgoing_row = {"node_id": entity, "connected_id": "Alice"}
    incoming_row = {"node_id": entity, "connected_id": "Bob"}

    async def fake_query(sql, *args, **kwargs):
        if "OPTIONAL MATCH (n:base)-[]->" in sql:
            return [outgoing_row]
        if "OPTIONAL MATCH (n:base)<-[]-" in sql:
            return [incoming_row]
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        result = await storage.get_nodes_edges_batch([entity])

    assert entity in result
    assert (entity, "Alice") in result[entity]
    assert ("Bob", entity) in result[entity]


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_plain_entity():
    """Entity names without special chars still work correctly."""
    storage = make_graph_storage()
    entity = "Alice"

    async def fake_query(sql, *args, **kwargs):
        if "OPTIONAL MATCH (n:base)-[]->" in sql:
            return [{"node_id": entity, "connected_id": "Bob"}]
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        result = await storage.get_nodes_edges_batch([entity])

    assert entity in result
    assert (entity, "Bob") in result[entity]


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_no_results():
    """Nodes with no edges return an empty list, not a KeyError."""
    storage = make_graph_storage()
    entity = 'Entity "X"'

    async def fake_query(sql, *args, **kwargs):
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        result = await storage.get_nodes_edges_batch([entity])

    assert entity in result
    assert result[entity] == []


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_deduplication():
    """Duplicate input IDs are deduplicated; each maps to the same edge list."""
    storage = make_graph_storage()
    entity = 'Dup "Entity"'

    async def fake_query(sql, *args, **kwargs):
        if "OPTIONAL MATCH (n:base)-[]->" in sql:
            return [{"node_id": entity, "connected_id": "Other"}]
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        result = await storage.get_nodes_edges_batch([entity, entity])

    assert result[entity] == [(entity, "Other")]


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_empty_input():
    """Empty input returns empty dict without calling _query."""
    storage = make_graph_storage()

    with patch.object(storage, "_query") as mock_q:
        result = await storage.get_nodes_edges_batch([])

    assert result == {}
    mock_q.assert_not_called()


@pytest.mark.asyncio
async def test_normalized_id_used_in_cypher_query():
    """The Cypher query string must contain the normalized (escaped) entity ID."""
    storage = make_graph_storage()
    entity = 'John "Smith"'
    captured_queries: list[str] = []

    async def fake_query(sql, *args, **kwargs):
        captured_queries.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.get_nodes_edges_batch([entity])

    normalized = PGGraphStorage._normalize_node_id(entity)
    assert any(normalized in q for q in captured_queries), (
        f"Expected normalized ID '{normalized}' in Cypher query, got: {captured_queries}"
    )
