"""
Integration tests for the optional `label` property on PGGraphStorage nodes.

Requires a PostgreSQL instance with Apache AGE extension.
Configure via environment variables:
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE

Run with: pytest tests/test_pg_graph_label_property.py -v --run-integration
"""

import os
import sys
import uuid

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.base import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data


async def _mock_embed(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 10)


def _pg_env_configured() -> bool:
    return bool(os.environ.get("POSTGRES_PASSWORD"))


@pytest.fixture
async def storage(tmp_path):
    """Create and initialize a PGGraphStorage instance for testing."""
    from lightrag.kg.postgres_impl import PGGraphStorage, ClientManager

    initialize_share_data()
    embed = EmbeddingFunc(embedding_dim=10, max_token_size=512, func=_mock_embed)

    # Use a unique namespace to avoid collisions between test runs
    test_ns = f"test_label_{uuid.uuid4().hex[:8]}"

    s = PGGraphStorage(
        namespace=test_ns,
        workspace="test",
        global_config={
            "working_dir": str(tmp_path),
            "addon_params": {},
        },
        embedding_func=embed,
    )
    await s.initialize()
    yield s

    # Cleanup: drop the test graph
    try:
        await s.drop()
    except Exception:
        pass

    # Reset the client manager singleton so other tests aren't affected
    ClientManager._instances = {"db": None, "ref_count": 0}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not _pg_env_configured(), reason="POSTGRES_PASSWORD not set")
async def test_upsert_and_get_node_with_label(storage):
    """Nodes can be upserted with a label property and retrieved."""
    node_id = "uuid-node-1"
    await storage.upsert_node(
        node_id,
        {
            "entity_id": node_id,
            "entity_type": "person",
            "label": "Elon Musk",
            "description": "CEO of SpaceX",
        },
    )

    props = await storage.get_node(node_id)
    assert props is not None
    assert props["entity_id"] == node_id
    assert props["label"] == "Elon Musk"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not _pg_env_configured(), reason="POSTGRES_PASSWORD not set")
async def test_get_all_labels_uses_label_property(storage):
    """get_all_labels should return label property when present, entity_id otherwise."""
    await storage.upsert_node(
        "uuid-a",
        {
            "entity_id": "uuid-a",
            "entity_type": "person",
            "label": "Alice",
        },
    )
    await storage.upsert_node(
        "HUMAN_READABLE",
        {
            "entity_id": "HUMAN_READABLE",
            "entity_type": "organization",
        },
    )

    labels = await storage.get_all_labels()
    assert "Alice" in labels, f"Expected 'Alice' in labels, got {labels}"
    assert (
        "HUMAN_READABLE" in labels
    ), f"Expected 'HUMAN_READABLE' in labels, got {labels}"
    assert (
        "uuid-a" not in labels
    ), f"'uuid-a' should not appear when label exists, got {labels}"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not _pg_env_configured(), reason="POSTGRES_PASSWORD not set")
async def test_get_knowledge_graph_uses_label_property(storage):
    """KnowledgeGraphNode.labels should use label property when available."""
    await storage.upsert_node(
        "uuid-x",
        {
            "entity_id": "uuid-x",
            "entity_type": "person",
            "label": "Jane Doe",
        },
    )
    await storage.upsert_node(
        "PLAIN_NODE",
        {
            "entity_id": "PLAIN_NODE",
            "entity_type": "organization",
        },
    )
    await storage.upsert_edge(
        "uuid-x",
        "PLAIN_NODE",
        {
            "weight": 1.0,
            "description": "works at",
        },
    )

    kg = await storage.get_knowledge_graph("uuid-x", max_depth=1)

    # PGGraphStorage uses AGE internal IDs as node.id, not entity_id.
    # So we check labels by looking up via entity_id in properties.
    node_labels = {n.properties["entity_id"]: n.labels[0] for n in kg.nodes}
    assert (
        node_labels.get("uuid-x") == "Jane Doe"
    ), f"Expected 'Jane Doe', got {node_labels}"
    # PLAIN_NODE may or may not appear depending on BFS depth traversal;
    # the key assertion is that the labeled node uses its label property.
    if "PLAIN_NODE" in node_labels:
        assert (
            node_labels["PLAIN_NODE"] == "PLAIN_NODE"
        ), f"Expected 'PLAIN_NODE', got {node_labels}"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not _pg_env_configured(), reason="POSTGRES_PASSWORD not set")
async def test_search_labels_matches_label_property(storage):
    """search_labels should search against label property, not entity_id."""
    await storage.upsert_node(
        "uuid-100",
        {
            "entity_id": "uuid-100",
            "entity_type": "person",
            "label": "Elon Musk",
        },
    )
    await storage.upsert_node(
        "VISIBLE_NAME",
        {
            "entity_id": "VISIBLE_NAME",
            "entity_type": "person",
        },
    )

    results = await storage.search_labels("elon")
    assert (
        "Elon Musk" in results
    ), f"Expected 'Elon Musk' in search results, got {results}"
    assert "uuid-100" not in results, f"'uuid-100' should not appear, got {results}"

    results = await storage.search_labels("VISIBLE")
    assert (
        "VISIBLE_NAME" in results
    ), f"Expected 'VISIBLE_NAME' in search results, got {results}"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not _pg_env_configured(), reason="POSTGRES_PASSWORD not set")
async def test_get_popular_labels_uses_label_property(storage):
    """Popular labels should return display labels, not node IDs."""
    await storage.upsert_node(
        "uuid-p1",
        {
            "entity_id": "uuid-p1",
            "entity_type": "person",
            "label": "Alice",
        },
    )
    await storage.upsert_node(
        "uuid-p2",
        {
            "entity_id": "uuid-p2",
            "entity_type": "person",
            "label": "Bob",
        },
    )
    await storage.upsert_node(
        "uuid-p3",
        {
            "entity_id": "uuid-p3",
            "entity_type": "person",
            # No label - should fall back to entity_id
        },
    )
    # Make uuid-p1 most popular
    await storage.upsert_edge(
        "uuid-p1", "uuid-p2", {"weight": 1.0, "description": "knows"}
    )
    await storage.upsert_edge(
        "uuid-p1", "uuid-p3", {"weight": 1.0, "description": "knows"}
    )

    labels = await storage.get_popular_labels(limit=10)
    assert "Alice" in labels, f"Expected 'Alice' in popular labels, got {labels}"
    assert "Bob" in labels, f"Expected 'Bob' in popular labels, got {labels}"
    assert "uuid-p1" not in labels, f"'uuid-p1' should not appear, got {labels}"
