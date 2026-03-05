"""
Tests for the optional `label` property on graph nodes.

When a node has a `label` property, it should be used as the display name
instead of the raw node ID (entity_id). This is useful when node IDs are
UUIDs or hashes rather than human-readable strings.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.base import EmbeddingFunc


async def _mock_embed(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 10)


@pytest.fixture
def storage(tmp_path):
    """Create a NetworkXStorage instance with a temp working directory."""
    initialize_share_data()
    embed = EmbeddingFunc(embedding_dim=10, max_token_size=512, func=_mock_embed)
    s = NetworkXStorage(
        namespace="test_ns",
        workspace="test",
        global_config={"working_dir": str(tmp_path), "addon_params": {}},
        embedding_func=embed,
    )
    return s


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_all_labels_without_label_property(storage):
    """Nodes without a label property should return their node ID."""
    await storage.initialize()
    graph = await storage._get_graph()
    graph.add_node("ELON MUSK", entity_type="person")
    graph.add_node("SPACEX", entity_type="organization")

    labels = await storage.get_all_labels()
    assert labels == ["ELON MUSK", "SPACEX"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_all_labels_with_label_property(storage):
    """Nodes with a label property should return the label, not the node ID."""
    await storage.initialize()
    graph = await storage._get_graph()
    graph.add_node("uuid-123", entity_type="person", label="Elon Musk")
    graph.add_node("uuid-456", entity_type="organization", label="SpaceX")

    labels = await storage.get_all_labels()
    assert labels == ["Elon Musk", "SpaceX"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_all_labels_mixed(storage):
    """Mix of nodes with and without label property."""
    await storage.initialize()
    graph = await storage._get_graph()
    graph.add_node("HUMAN_READABLE", entity_type="person")
    graph.add_node("uuid-789", entity_type="organization", label="Some Org")

    labels = await storage.get_all_labels()
    assert "HUMAN_READABLE" in labels
    assert "Some Org" in labels
    assert "uuid-789" not in labels


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_popular_labels_with_label_property(storage):
    """Popular labels should return display labels, not node IDs."""
    await storage.initialize()
    graph = await storage._get_graph()
    graph.add_node("uuid-a", entity_type="person", label="Alice")
    graph.add_node("uuid-b", entity_type="person", label="Bob")
    graph.add_node("uuid-c", entity_type="person")  # no label
    # Add edges to make uuid-a most popular
    graph.add_edge("uuid-a", "uuid-b", weight=1.0)
    graph.add_edge("uuid-a", "uuid-c", weight=1.0)

    labels = await storage.get_popular_labels(limit=10)
    assert labels[0] == "Alice"  # Most connected
    assert "Bob" in labels
    assert "uuid-c" in labels  # No label, falls back to node ID
    assert "uuid-a" not in labels  # Should show "Alice" instead


@pytest.mark.offline
@pytest.mark.asyncio
async def test_search_labels_with_label_property(storage):
    """Search should match against label property, not node IDs."""
    await storage.initialize()
    graph = await storage._get_graph()
    graph.add_node("uuid-100", entity_type="person", label="Elon Musk")
    graph.add_node("uuid-200", entity_type="organization", label="Tesla Inc")
    graph.add_node("VISIBLE_NAME", entity_type="person")

    # Search for "elon" should find the labeled node
    results = await storage.search_labels("elon")
    assert "Elon Musk" in results
    assert "uuid-100" not in results

    # Search for "VISIBLE" should find the unlabeled node by its ID
    results = await storage.search_labels("VISIBLE")
    assert "VISIBLE_NAME" in results

    # Search for "uuid" should NOT match (label takes precedence)
    results = await storage.search_labels("uuid-100")
    assert len(results) == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_knowledge_graph_labels(storage):
    """KnowledgeGraphNode.labels should use label property when available."""
    await storage.initialize()
    graph = await storage._get_graph()
    graph.add_node("uuid-x", entity_type="person", label="Jane Doe")
    graph.add_node("PLAIN_NODE", entity_type="organization")
    graph.add_edge("uuid-x", "PLAIN_NODE", weight=1.0, description="works at")

    kg = await storage.get_knowledge_graph("uuid-x", max_depth=1)

    # Find the nodes in the result
    node_labels = {n.id: n.labels[0] for n in kg.nodes}
    assert node_labels["uuid-x"] == "Jane Doe"
    assert node_labels["PLAIN_NODE"] == "PLAIN_NODE"
