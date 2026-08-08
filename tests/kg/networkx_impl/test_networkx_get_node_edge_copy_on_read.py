import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data


def _storage(tmp_path, namespace: str) -> NetworkXStorage:
    initialize_share_data()
    return NetworkXStorage(
        namespace=namespace,
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )


@pytest.mark.asyncio
async def test_get_node_returns_copy_not_live_alias(tmp_path):
    storage = _storage(tmp_path, "copy_on_read_node")
    await storage.initialize()
    await storage.upsert_node(
        "A",
        {"entity_type": "PERSON", "description": "orig", "source_id": "c1"},
    )

    node = await storage.get_node("A")
    assert node is not None
    node["description"] = "MUTATED"

    reread = await storage.get_node("A")
    assert reread is not None
    assert reread["description"] == "orig"


@pytest.mark.asyncio
async def test_get_edge_returns_copy_not_live_alias(tmp_path):
    storage = _storage(tmp_path, "copy_on_read_edge")
    await storage.initialize()
    await storage.upsert_node("A", {"entity_type": "PERSON", "description": "a"})
    await storage.upsert_node("B", {"entity_type": "PERSON", "description": "b"})
    await storage.upsert_edge("A", "B", {"weight": 1.0, "description": "rel-orig"})

    edge = await storage.get_edge("A", "B")
    assert edge is not None
    edge["description"] = "REL-MUTATED"

    reread = await storage.get_edge("A", "B")
    assert reread is not None
    assert reread["description"] == "rel-orig"


@pytest.mark.asyncio
async def test_get_nodes_batch_uses_copied_nodes(tmp_path):
    storage = _storage(tmp_path, "copy_on_read_batch")
    await storage.initialize()
    await storage.upsert_node("A", {"entity_type": "ORG", "description": "keep-me"})

    batch = await storage.get_nodes_batch(["A"])
    batch["A"]["description"] = "batch-mutated"

    reread = await storage.get_node("A")
    assert reread["description"] == "keep-me"
