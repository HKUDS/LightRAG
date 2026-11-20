"""Batch graph reads must answer from a single ``_get_graph()`` snapshot.

``BaseGraphStorage``'s default batch readers await one single-item call per
entry, and every one of those calls goes through ``_get_graph``, which reloads
when a peer process has committed. A peer commit landing mid-batch then makes
one logical batch answer partly from before that commit and partly from after
it -- a combination that was never the graph's actual state.

The fault is injected right after the batch's FIRST ``_get_graph`` returns, so
it fires whatever the implementation calls next; each case also proves the
commit was real by observing it on a fresh call afterwards.
"""

from __future__ import annotations

import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


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


async def _seed(storage: NetworkXStorage) -> None:
    for node_id in ("A", "B", "X"):
        await storage.upsert_node(node_id, {"entity_id": node_id, "val": "old"})
    await storage.upsert_edge("A", "X", {"weight": 1.0, "val": "old"})
    await storage.upsert_edge("B", "X", {"weight": 1.0, "val": "old"})


async def _peer_commit(peer: NetworkXStorage) -> None:
    """One commit that changes every entry each batch below reads, together."""
    await peer.upsert_node("A", {"entity_id": "A", "val": "new"})
    await peer.upsert_node("B", {"entity_id": "B", "val": "new"})
    await peer.upsert_edge("A", "X", {"weight": 2.0, "val": "new"})
    await peer.upsert_edge("B", "X", {"weight": 2.0, "val": "new"})
    await peer.upsert_node("C", {"entity_id": "C", "val": "new"})
    await peer.upsert_edge("A", "C", {"weight": 1.0})
    await peer.upsert_edge("B", "C", {"weight": 1.0})
    assert await peer.index_done_callback() is True


def _normalized(result):
    """Make ``get_nodes_edges_batch`` lists comparable regardless of order."""
    return {
        key: sorted(tuple(sorted(edge)) for edge in value)
        if isinstance(value, list)
        else value
        for key, value in result.items()
    }


BATCH_READS = [
    pytest.param(lambda s: s.get_nodes_batch(["A", "B"]), id="get_nodes_batch"),
    pytest.param(lambda s: s.node_degrees_batch(["A", "B"]), id="node_degrees_batch"),
    pytest.param(
        lambda s: s.get_edges_batch(
            [{"src": "A", "tgt": "X"}, {"src": "B", "tgt": "X"}]
        ),
        id="get_edges_batch",
    ),
    pytest.param(
        lambda s: s.edge_degrees_batch([("A", "X"), ("B", "X")]),
        id="edge_degrees_batch",
    ),
    pytest.param(
        lambda s: s.get_nodes_edges_batch(["A", "B"]), id="get_nodes_edges_batch"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_read", BATCH_READS)
async def test_batch_read_does_not_tear_across_a_mid_batch_peer_commit(
    tmp_path, batch_read
):
    storage = _storage(tmp_path, "batch_tear")
    await storage.initialize()
    await _seed(storage)
    assert await storage.index_done_callback() is True

    peer = _storage(tmp_path, "batch_tear")
    await peer.initialize()

    try:
        before = _normalized(await batch_read(storage))

        original_get_graph = storage._get_graph
        commits = []

        async def get_graph_then_peer_commits():
            graph = await original_get_graph()
            if not commits:
                commits.append("peer")
                await _peer_commit(peer)
            return graph

        storage._get_graph = get_graph_then_peer_commits
        try:
            during = _normalized(await batch_read(storage))
        finally:
            storage._get_graph = original_get_graph

        assert commits == ["peer"], "the mid-batch peer commit never fired"
        after = _normalized(await batch_read(storage))
        assert after != before, "the peer commit is not visible to this storage"

        # Every entry changed in the one peer commit, so a batch spanning it
        # must answer wholly from the snapshot it started on.
        assert during == before, f"torn batch read: {during!r}"
    finally:
        await peer.finalize()
        await storage.finalize()
