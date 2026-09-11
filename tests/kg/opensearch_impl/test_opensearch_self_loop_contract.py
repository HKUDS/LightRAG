"""Self-loop read contract for ``OpenSearchGraphStorage``.

The document-store sibling of ``tests/kg/mongo_impl/test_mongo_self_loop_contract.py``
— same two rules, stated on ``BaseGraphStorage`` and pinned for ``pgtable_impl``:

* **degree counts endpoint occurrences**, so a self-loop counts TWICE, matching
  NetworkX ``graph.degree()``;
* **edge listings list edges**, so a self-loop appears ONCE.

Both backends drifted the same way and for the same reason: a ``count`` over
``source == id OR target == id`` sees the self-loop's single DOCUMENT, while the
aggregations behind ``node_degrees_batch`` and ``get_popular_labels`` group per
endpoint field and see it twice; and the separate endpoint branches behind
``get_nodes_edges_batch`` each list that one document.

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
        aggs = body.get("aggs")
        if aggs:
            response = {"aggregations": {}}
            for agg_name, field in (
                ("source_degrees", "source_node_id"),
                ("target_degrees", "target_node_id"),
            ):
                ids = set(aggs[agg_name]["filter"]["terms"][field])
                counts = {}
                for edge in edges:
                    key = edge[field]
                    if key in ids:
                        counts[key] = counts.get(key, 0) + 1
                response["aggregations"][agg_name] = {
                    "ids": {
                        "buckets": [
                            {"key": key, "doc_count": count}
                            for key, count in counts.items()
                        ]
                    }
                }
            return response

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


def _make_count_side_effect(edges: list[dict]):
    """Answer the two count queries ``node_degree`` issues: the ``should`` over
    both endpoint fields (documents touching the node), and the ``filter`` over
    both at once (the node's self-loops). Answering them separately is what
    lets these tests tell the fixed implementation apart from one that counts
    matching documents only."""

    async def _count(*args, **kwargs):
        clauses = kwargs["body"]["query"]["bool"]
        if "should" in clauses:
            node_id = clauses["should"][0]["term"]["source_node_id"]
            matches = [
                e
                for e in edges
                if node_id in (e["source_node_id"], e["target_node_id"])
            ]
        else:
            node_id = clauses["filter"][0]["term"]["source_node_id"]
            matches = [
                e
                for e in edges
                if e["source_node_id"] == node_id and e["target_node_id"] == node_id
            ]
        return {"count": len(matches)}

    return _count


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
    s.client.count = AsyncMock(side_effect=_make_count_side_effect(edges))
    s.client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    s.client.delete_pit = AsyncMock(return_value=None)
    return s


_SELF_LOOP = [{"source_node_id": "Loop", "target_node_id": "Loop"}]
_PLAIN = [
    {"source_node_id": "A", "target_node_id": "B"},
    {"source_node_id": "C", "target_node_id": "A"},
]


class TestNodeDegreeSelfLoop:
    @pytest.mark.asyncio
    async def test_counts_self_loop_twice(self):
        """A self-loop occupies both endpoints of its own edge, so it is degree
        2. Counting documents matching the ``should`` clause answers 1."""
        s = _make_storage(_SELF_LOOP)

        assert await s.node_degree("Loop") == 2

    @pytest.mark.asyncio
    async def test_agrees_with_node_degrees_batch_on_a_self_loop(self):
        s = _make_storage(_SELF_LOOP)

        scalar = await s.node_degree("Loop")
        batch = await s.node_degrees_batch(["Loop"])

        assert scalar == batch["Loop"] == 2

    @pytest.mark.asyncio
    async def test_plain_edges_are_not_inflated(self):
        s = _make_storage(_PLAIN)

        scalar = await s.node_degree("A")
        batch = await s.node_degrees_batch(["A"])

        assert scalar == batch["A"] == 2

    @pytest.mark.asyncio
    async def test_absent_node_has_degree_zero(self):
        s = _make_storage(_PLAIN)

        assert await s.node_degree("Ghost") == 0

    @pytest.mark.asyncio
    async def test_answers_zero_without_querying_when_indices_are_not_ready(self):
        """Unchanged from before: an unready index is answered locally."""
        s = _make_storage(_PLAIN)
        s._indices_ready = False

        assert await s.node_degree("A") == 0
        s.client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edge_degree_sums_both_endpoints(self):
        s = _make_storage(_SELF_LOOP)

        assert await s.edge_degree("Loop", "Loop") == 4


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
