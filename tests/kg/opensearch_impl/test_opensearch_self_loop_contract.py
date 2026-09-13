"""Self-loop read contract for ``OpenSearchGraphStorage``.

The document-store sibling of ``tests/kg/mongo_impl/test_mongo_self_loop_contract.py``
— same two rules, stated on ``BaseGraphStorage`` and pinned for ``pgtable_impl``:

* **degree measures connectivity**, so a self-loop is EXCLUDED -- it connects
  nothing, the same reason ``_reject_self_loop_relation`` refuses to create one;
* **edge listings enumerate what the store holds**, so a self-loop appears ONCE
  and is never hidden -- deletion, rename, merge and document purge all resolve
  a node's relation rows through them.

The scalar and the aggregation paths reach the rule differently, which is what
these tests pin: ``node_degree`` subtracts an index-served count of the node's
own self-loops, while the aggregations carry ``_SELF_LOOP_SCRIPT_CLAUSE`` in
``must_not`` because OpenSearch cannot compare two fields with a term query.

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
            # Apply the query's self-loop exclusion rather than ignoring it: a
            # mock that skipped it would answer identically whether or not the
            # clause is still there, and could not tell the fixed
            # implementation from the broken one.
            must_not = body.get("query", {}).get("bool", {}).get("must_not") or []
            drops_self_loops = any("script" in clause for clause in must_not)
            visible = [
                e
                for e in edges
                if not (drops_self_loops and e["source_node_id"] == e["target_node_id"])
            ]
            response = {"aggregations": {}}
            for agg_name, field in (
                ("source_degrees", "source_node_id"),
                ("target_degrees", "target_node_id"),
            ):
                ids = set(aggs[agg_name]["filter"]["terms"][field])
                counts = {}
                for edge in visible:
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
    async def test_self_loop_only_node_has_degree_zero(self):
        """A self-loop connects nothing, so it earns no degree. Counting
        documents matching the ``should`` clause answers 1; subtracting the
        node's own self-loops takes that one occurrence back off."""
        s = _make_storage(_SELF_LOOP)

        assert await s.node_degree("Loop") == 0

    @pytest.mark.asyncio
    async def test_agrees_with_node_degrees_batch_on_a_self_loop(self):
        """The scalar subtracts a count and the batch filters an aggregation --
        two different mechanisms that must land on the same number."""
        s = _make_storage(_SELF_LOOP)

        scalar = await s.node_degree("Loop")
        batch = await s.node_degrees_batch(["Loop"])

        assert scalar == batch["Loop"] == 0

    @pytest.mark.asyncio
    async def test_self_loop_does_not_inflate_a_connected_node(self):
        s = _make_storage(_PLAIN + [{"source_node_id": "A", "target_node_id": "A"}])

        scalar = await s.node_degree("A")
        batch = await s.node_degrees_batch(["A"])

        assert scalar == batch["A"] == 2

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
        """(Loop, Loop) is 0 + 0."""
        s = _make_storage(_SELF_LOOP)

        assert await s.edge_degree("Loop", "Loop") == 0

    @pytest.mark.asyncio
    async def test_edge_degree_on_an_ordinary_edge(self):
        s = _make_storage(_PLAIN)

        assert await s.edge_degree("A", "B") == 3


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


class TestAggregationPathsCarryTheScriptFilter:
    """The paths that DERIVE degree from an aggregation.

    ``node_degree`` subtracts an index-served count because both endpoints are
    the known node id there. These paths have to compare two FIELDS, which a
    term query cannot do, so they carry ``_SELF_LOOP_SCRIPT_CLAUSE`` in
    ``must_not`` instead. Asserting the clause is present is what keeps a path
    from being left behind: each one is a separate query body.
    """

    @staticmethod
    def _capture_storage():
        bodies = []

        async def _search(*args, **kwargs):
            bodies.append(kwargs.get("body") or (args[0] if args else {}))
            # Aggregations AND empty hits: the star path continues into a
            # hits-based fetch once the degree ranking comes back.
            return {
                "aggregations": {"src": {"buckets": []}, "tgt": {"buckets": []}},
                "hits": {"hits": [], "total": {"value": 0}},
            }

        s = _make_storage([])
        s.client.search = AsyncMock(side_effect=_search)
        return s, bodies

    @staticmethod
    def _excludes_self_loops(body) -> bool:
        must_not = body.get("query", {}).get("bool", {}).get("must_not") or []
        return any("script" in clause for clause in must_not)

    @pytest.mark.asyncio
    async def test_get_popular_labels_excludes_self_loops(self):
        """A self-loop must not lift a node up the entity picker's ranking."""
        s, bodies = self._capture_storage()
        s._collect_isolated_labels = AsyncMock(return_value=["Iso"])

        await s.get_popular_labels(limit=1)

        assert len(bodies) == 1, bodies
        assert self._excludes_self_loops(bodies[0]), bodies[0]

    @pytest.mark.asyncio
    async def test_knowledge_graph_star_ranking_excludes_self_loops(self):
        """The ``*`` path is degree-ranked, so the rule decides which nodes
        survive ``max_nodes`` truncation."""
        s, bodies = self._capture_storage()
        s.client.count = AsyncMock(return_value={"count": 99})

        await s._get_knowledge_graph_all(max_nodes=2)

        assert bodies, "no degree aggregation was issued"
        assert self._excludes_self_loops(bodies[0]), bodies[0]

    @pytest.mark.asyncio
    async def test_node_degrees_batch_keeps_minimum_should_match(self):
        """Adding ``must_not`` beside a bool that carries only ``should`` is
        exactly where the implicit ``minimum_should_match`` flips to 0 --
        which would admit every edge in the index, not just the ones touching
        a requested id. It is set explicitly for that reason."""
        s = _make_storage(_PLAIN)
        captured = []

        async def _search(*args, **kwargs):
            body = kwargs.get("body") or (args[0] if args else {})
            captured.append(body)
            return {
                "aggregations": {
                    "source_degrees": {"ids": {"buckets": []}},
                    "target_degrees": {"ids": {"buckets": []}},
                }
            }

        s.client.search = AsyncMock(side_effect=_search)

        await s.node_degrees_batch(["A"])

        bool_query = captured[0]["query"]["bool"]
        assert bool_query["minimum_should_match"] == 1, bool_query
        assert any("script" in clause for clause in bool_query["must_not"]), bool_query
