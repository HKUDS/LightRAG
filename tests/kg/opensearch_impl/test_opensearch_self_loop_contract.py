"""Self-loop LISTING contract for ``OpenSearchGraphStorage``.

The document-store sibling of
``tests/kg/mongo_impl/test_mongo_self_loop_contract.py``: one hit satisfies
both endpoint branches of the ``should`` query, so without a guard the scan
lists one edge as two. ``BaseGraphStorage.get_node_edges`` says a self-loop
appears ONCE and is never hidden -- entity deletion, rename, merge and document
purge all resolve a node's relation rows through these methods, so an edge
hidden here becomes an orphan relation vector row.

Degree is NOT asserted here. The contract does not define a self-loop's degree
-- the store is not allowed to hold one -- and the backends answer from their
cheapest natural query, so pinning a number would pin an unreachable case.

Only the low-level client is faked — same convention as
``test_opensearch_bfs_tie_break.py``.
"""

import pytest
from unittest.mock import AsyncMock

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from lightrag.kg.opensearch_impl import OpenSearchGraphStorage

pytestmark = pytest.mark.offline


def _make_search_side_effect(edges: list[dict]):
    """Answer the two search shapes these methods issue against the edge index:
    ``node_degrees_batch``'s filtered terms aggregations, and
    ``get_nodes_edges_batch``'s paged PIT scan."""

    async def _search(*args, **kwargs):
        body = kwargs.get("body") or (args[0] if args else {})
        # PIT scan: one page, every edge touching a requested id. `search_after`
        # means the caller is asking for the page after the last hit -- there is
        # none, so an empty page ends the loop.
        if body.get("search_after"):
            return {"hits": {"hits": []}}
        should = body["query"]["bool"]["should"]
        ids = set(should[0]["terms"]["source_node_id"])
        return {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "source_node_id": e["source_node_id"],
                            "target_node_id": e["target_node_id"],
                        },
                        "sort": [e["source_node_id"], e["target_node_id"]],
                    }
                    for e in edges
                    if e["source_node_id"] in ids or e["target_node_id"] in ids
                ]
            }
        }

    return _search


def _make_storage(edges: list[dict]):
    s = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    s.workspace = "test"
    s.global_config = {"max_graph_nodes": 1000}
    s._nodes_index = "test-nodes"
    s._edges_index = "test-edges"
    s._indices_ready = True
    s._refresh_graph_indices_if_dirty = AsyncMock(return_value=None)
    s.client = AsyncMock()
    s.client.search = AsyncMock(side_effect=_make_search_side_effect(edges))
    s.client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    s.client.delete_pit = AsyncMock(return_value=None)
    return s


_SELF_LOOP = [{"source_node_id": "Loop", "target_node_id": "Loop"}]
_PLAIN = [
    {"source_node_id": "A", "target_node_id": "B"},
    {"source_node_id": "C", "target_node_id": "A"},
]


class TestGetNodesEdgesBatchSelfLoop:
    @pytest.mark.asyncio
    async def test_self_loop_is_listed_once(self):
        """One hit satisfies both endpoint branches; listing it from each
        reports one edge as two."""
        s = _make_storage(_SELF_LOOP)

        result = await s.get_nodes_edges_batch(["Loop"])

        assert result["Loop"] == [("Loop", "Loop")]

    @pytest.mark.asyncio
    async def test_ordinary_edge_still_reaches_both_endpoints(self):
        """The skip must be scoped to ``src == tgt``: an ordinary edge is still
        listed for its source AND its target."""
        s = _make_storage(_PLAIN)

        result = await s.get_nodes_edges_batch(["A", "B"])

        assert ("A", "B") in result["A"]
        assert ("A", "B") in result["B"]
        assert ("C", "A") in result["A"]

    @pytest.mark.asyncio
    async def test_self_loop_alongside_ordinary_edges(self):
        s = _make_storage(_PLAIN + [{"source_node_id": "A", "target_node_id": "A"}])

        result = await s.get_nodes_edges_batch(["A"])

        assert result["A"].count(("A", "A")) == 1
        assert len(result["A"]) == 3
