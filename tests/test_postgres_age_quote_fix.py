"""
Unit tests for PGGraphStorage.get_nodes_edges_batch and get_node_edges
with special characters in entity names.

Verifies the fix for KeyError when entity names contain double quotes (PR #2872)
and the follow-up Option C refactor to parameterized Cypher queries.

The root cause: AGE returns the original un-escaped entity_id, but the edges_norm
dict was previously keyed with the normalized (escaped) ID, causing a KeyError on lookup.

The Option C fix: use $node_ids / $entity_id parameters instead of string interpolation,
eliminating the need for _normalize_node_id in these read paths entirely.
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
# _normalize_node_id (still used by write paths: remove_nodes, upsert_node, etc.)
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
# get_node_edges — parameterized query (Option C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_node_edges_passes_original_id_as_parameter():
    """entity_id must be passed as a JSON parameter, not interpolated into Cypher."""
    storage = make_graph_storage()
    entity = 'John "Smith"'
    captured_params: list[dict] = []

    async def fake_query(sql, **kwargs):
        if kwargs.get("params"):
            captured_params.append(json.loads(list(kwargs["params"].values())[0]))
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.get_node_edges(entity)

    assert len(captured_params) == 1
    assert captured_params[0]["entity_id"] == entity


@pytest.mark.asyncio
async def test_get_node_edges_cypher_uses_parameter_syntax():
    """The SQL sent to _query must use $1::agtype, not a hardcoded escaped string."""
    storage = make_graph_storage()
    entity = 'John "Smith"'
    captured_sql: list[str] = []

    async def fake_query(sql, **kwargs):
        captured_sql.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.get_node_edges(entity)

    assert len(captured_sql) == 1
    assert "$1::agtype" in captured_sql[0]
    # Entity name must NOT appear literally in the SQL string
    assert entity not in captured_sql[0]
    assert '\\"' not in captured_sql[0]


@pytest.mark.asyncio
async def test_get_node_edges_returns_edges():
    storage = make_graph_storage()

    async def fake_query(_sql, **_kwargs):
        return [
            {"source_id": "Alice", "connected_id": "Bob"},
            {"source_id": "Alice", "connected_id": None},
        ]

    with patch.object(storage, "_query", side_effect=fake_query):
        result = await storage.get_node_edges("Alice")

    assert result == [("Alice", "Bob")]


# ---------------------------------------------------------------------------
# get_nodes_edges_batch — parameterized query (Option C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_passes_original_ids_as_parameter():
    """node_ids batch must be passed as a JSON parameter, not interpolated."""
    storage = make_graph_storage()
    entities = ['John "Smith"', "Alice", "O\\Brien"]
    captured_params: list[dict] = []

    async def fake_query(_sql, **kwargs):
        if kwargs.get("params"):
            captured_params.append(json.loads(list(kwargs["params"].values())[0]))
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.get_nodes_edges_batch(entities)

    assert len(captured_params) == 2  # outgoing + incoming
    assert captured_params[0]["node_ids"] == entities
    assert captured_params[1]["node_ids"] == entities


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_cypher_uses_parameter_syntax():
    """The SQL must use $1::agtype, not hardcoded escaped entity names."""
    storage = make_graph_storage()
    entity = 'John "Smith"'
    captured_sql: list[str] = []

    async def fake_query(sql, **_kwargs):
        captured_sql.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.get_nodes_edges_batch([entity])

    assert len(captured_sql) == 2
    for sql in captured_sql:
        assert "$1::agtype" in sql
        assert entity not in sql
        assert '\\"' not in sql


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_with_quoted_entity():
    """
    AGE returns the original un-escaped node_id in query results.
    The result dict must be keyed by the original ID.
    """
    storage = make_graph_storage()
    entity = 'John "Smith"'

    async def fake_query(sql, **_kwargs):
        if "OPTIONAL MATCH (n:base)-[]->" in sql:
            return [{"node_id": entity, "connected_id": "Alice"}]
        if "OPTIONAL MATCH (n:base)<-[]-" in sql:
            return [{"node_id": entity, "connected_id": "Bob"}]
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

    async def fake_query(sql, **_kwargs):
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

    async def fake_query(_sql, **_kwargs):
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

    async def fake_query(sql, **_kwargs):
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
