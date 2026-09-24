"""Batch graph reads must answer from a single ``_get_graph()`` snapshot.

``has_nodes_batch`` / ``upsert_nodes_batch`` / ``upsert_edges_batch`` all take
one snapshot up front and then answer the whole batch synchronously against
it. The read-batch getters (``get_nodes_batch``, ``node_degrees_batch``,
``get_edges_batch``, ``edge_degrees_batch``) instead inherited
``BaseGraphStorage``'s default, which awaits one single-item call per batch
entry. Each of those awaits is a suspension point ``_get_graph`` can reload
across, so a peer commit landing mid-batch can make a single logical batch
call answer with some entries read before that commit and others read after
it -- a combination that was never the graph's actual state at any point in
time.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


def _storage(tmp_path, namespace: str) -> NetworkXStorage:
    return NetworkXStorage(
        namespace=namespace,
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )


@pytest.mark.asyncio
async def test_get_nodes_batch_does_not_tear_across_a_mid_batch_peer_commit(tmp_path):
    storage = _storage(tmp_path, "batch_tear_nodes")
    await storage.initialize()
    await storage.upsert_node("A", {"val": "old"})
    await storage.upsert_node("B", {"val": "old"})
    assert await storage.index_done_callback() is True

    peer = _storage(tmp_path, "batch_tear_nodes")
    await peer.initialize()

    original_get_node = storage.get_node

    async def get_node_then_peer_commits(node_id):
        result = await original_get_node(node_id)
        if node_id == "A":
            # One peer commit updates BOTH nodes together, as a concurrent
            # document-pipeline write landing mid-batch would.
            await peer.upsert_node("A", {"val": "new"})
            await peer.upsert_node("B", {"val": "new"})
            assert await peer.index_done_callback() is True
        return result

    try:
        with patch.object(storage, "get_node", side_effect=get_node_then_peer_commits):
            result = await storage.get_nodes_batch(["A", "B"])

        # A and B were always updated together in one commit; a batch fetch
        # spanning that commit must not report one from before it and the
        # other from after.
        assert result["A"]["val"] == result["B"]["val"], (
            f"torn batch read: A={result['A']['val']!r} B={result['B']['val']!r}"
        )
    finally:
        await peer.finalize()
        await storage.finalize()


@pytest.mark.asyncio
async def test_node_degrees_batch_does_not_tear_across_a_mid_batch_peer_commit(
    tmp_path,
):
    storage = _storage(tmp_path, "batch_tear_degrees")
    await storage.initialize()
    await storage.upsert_node("A", {"entity_id": "A"})
    await storage.upsert_node("B", {"entity_id": "B"})
    assert await storage.index_done_callback() is True

    peer = _storage(tmp_path, "batch_tear_degrees")
    await peer.initialize()

    original_node_degree = storage.node_degree

    async def node_degree_then_peer_commits(node_id):
        result = await original_node_degree(node_id)
        if node_id == "A":
            # One peer commit gives both nodes a new edge together.
            await peer.upsert_node("C", {"entity_id": "C"})
            await peer.upsert_edge("A", "C", {"weight": 1.0})
            await peer.upsert_edge("B", "C", {"weight": 1.0})
            assert await peer.index_done_callback() is True
        return result

    try:
        with patch.object(
            storage, "node_degree", side_effect=node_degree_then_peer_commits
        ):
            result = await storage.node_degrees_batch(["A", "B"])

        assert result["A"] == result["B"], (
            f"torn batch read: A={result['A']} B={result['B']}"
        )
    finally:
        await peer.finalize()
        await storage.finalize()
