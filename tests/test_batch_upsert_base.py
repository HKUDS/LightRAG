"""Tests for batch upsert base class default implementations."""
import pytest
from unittest.mock import AsyncMock
from lightrag.base import BaseGraphStorage


class ConcreteGraphStorage(BaseGraphStorage):
    """Minimal concrete subclass for testing."""

    async def delete_node(self, node_id): ...
    async def drop(self): ...
    async def edge_degree(self, src, tgt): ...
    async def get_all_edges(self): ...
    async def get_all_labels(self): ...
    async def get_all_nodes(self): ...
    async def get_edge(self, src, tgt): ...
    async def get_knowledge_graph(self, node_label, max_depth): ...
    async def get_node(self, node_id): ...
    async def get_node_edges(self, node_id): ...
    async def get_popular_labels(self, num): ...
    async def has_edge(self, src, tgt): ...
    async def has_node(self, node_id): ...
    async def index_done_callback(self): ...
    async def node_degree(self, node_id): ...
    async def remove_edges(self, edges): ...
    async def remove_nodes(self, nodes): ...
    async def search_labels(self, query): ...
    async def upsert_edge(self, src, tgt, data): ...
    async def upsert_node(self, node_id, data): ...


@pytest.fixture
def storage():
    s = object.__new__(ConcreteGraphStorage)
    s.upsert_node = AsyncMock()
    s.upsert_edge = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_batch_upsert_nodes_calls_upsert_node(storage):
    nodes = [
        ("entity1", {"entity_id": "entity1", "entity_type": "PERSON", "description": "A person"}),
        ("entity2", {"entity_id": "entity2", "entity_type": "ORG", "description": "An org"}),
    ]
    await storage.batch_upsert_nodes(nodes)
    assert storage.upsert_node.call_count == 2
    storage.upsert_node.assert_any_call("entity1", nodes[0][1])
    storage.upsert_node.assert_any_call("entity2", nodes[1][1])


@pytest.mark.asyncio
async def test_batch_upsert_edges_calls_upsert_edge(storage):
    edges = [
        ("src1", "tgt1", {"weight": "1.0", "description": "related"}),
        ("src2", "tgt2", {"weight": "0.5", "description": "similar"}),
    ]
    await storage.batch_upsert_edges(edges)
    assert storage.upsert_edge.call_count == 2
    storage.upsert_edge.assert_any_call("src1", "tgt1", edges[0][2])
    storage.upsert_edge.assert_any_call("src2", "tgt2", edges[1][2])


@pytest.mark.asyncio
async def test_batch_upsert_nodes_empty_list(storage):
    await storage.batch_upsert_nodes([])
    storage.upsert_node.assert_not_called()


@pytest.mark.asyncio
async def test_batch_upsert_edges_empty_list(storage):
    await storage.batch_upsert_edges([])
    storage.upsert_edge.assert_not_called()


@pytest.mark.asyncio
async def test_has_nodes_batch_default(storage):
    storage.has_node = AsyncMock(side_effect=[True, False, True])
    result = await storage.has_nodes_batch(["a", "b", "c"])
    assert result == {"a", "c"}
    assert storage.has_node.call_count == 3
