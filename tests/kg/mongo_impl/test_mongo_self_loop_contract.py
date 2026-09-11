"""Self-loop read contract for ``MongoGraphStorage``.

Two rules, pointing opposite ways, both stated on ``BaseGraphStorage`` and both
pinned for ``pgtable_impl`` (``test_self_loop_degree_consistency``,
``test_get_nodes_edges_batch_self_loop_counted_once``):

* **degree counts endpoint occurrences**, so a self-loop counts TWICE, matching
  NetworkX ``graph.degree()``;
* **edge listings list edges**, so a self-loop appears ONCE.

A document store gets both wrong by following its own query shape: the ``$or``
that backs ``node_degree`` counts the self-loop's single DOCUMENT once, and the
separate outbound/inbound passes that back ``get_nodes_edges_batch`` each list
that same document.

LightRAG's own writers never create a self-loop (see
``_reject_self_loop_relation``), so this governs imported graphs and direct
storage-API use.

Only the low-level collection is faked, asserting on the actual query issued --
same convention as ``test_mongo_edge_batch_methods.py``.
"""

from collections import Counter

import pytest
from unittest.mock import AsyncMock, Mock

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from lightrag.kg.mongo_impl import MongoGraphStorage

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _make_count_side_effect(edges: list[dict]):
    """Answer ``count_documents`` for the two filters ``node_degree`` issues:
    the bidirectional ``$or`` (documents touching the node at either endpoint)
    and the both-fields equality (the node's self-loops)."""

    def _count(query, **kwargs):
        if "$or" in query:
            node_id = query["$or"][0]["source_node_id"]
            return sum(
                1
                for e in edges
                if node_id in (e["source_node_id"], e["target_node_id"])
            )
        return sum(
            1
            for e in edges
            if e["source_node_id"] == query["source_node_id"]
            and e["target_node_id"] == query["target_node_id"]
        )

    return _count


def _make_aggregate_side_effect(edges: list[dict]):
    """Mock ``aggregate`` for ``node_degrees_batch``'s two pipelines -- outbound
    grouped on ``source_node_id``, inbound on ``target_node_id``."""

    async def _aggregate(pipeline, **kwargs):
        match = pipeline[0]["$match"]
        field = "source_node_id" if "source_node_id" in match else "target_node_id"
        ids = set(match[field]["$in"])
        counts = Counter(e[field] for e in edges if e[field] in ids)
        return _AsyncCursor(
            [{"_id": key, "degree": count} for key, count in counts.items()]
        )

    return _aggregate


def _make_find_side_effect(edges: list[dict]):
    """Mock ``find`` for both read shapes: ``get_nodes_edges_batch``'s separate
    outbound and inbound ``$in`` queries, and ``get_node_edges``'s single
    bidirectional ``$or`` -- each returning only its own matches, so the two
    methods are compared on what MongoDB would really hand them."""

    def _find(query, projection=None, **kwargs):
        if "$or" in query:
            node_id = query["$or"][0]["source_node_id"]
            return _AsyncCursor(
                [
                    e
                    for e in edges
                    if node_id in (e["source_node_id"], e["target_node_id"])
                ]
            )
        field = "source_node_id" if "source_node_id" in query else "target_node_id"
        ids = set(query[field]["$in"])
        return _AsyncCursor([e for e in edges if e[field] in ids])

    return _find


def _make_storage(edges: list[dict]):
    s = MongoGraphStorage.__new__(MongoGraphStorage)
    s.workspace = "test"
    s.namespace = "chunk_entity_relation"
    s.global_config = {}
    s._edge_collection_name = "test_edges"
    s.edge_collection = Mock()
    s.edge_collection.count_documents = AsyncMock(
        side_effect=_make_count_side_effect(edges)
    )
    s.edge_collection.aggregate = AsyncMock(
        side_effect=_make_aggregate_side_effect(edges)
    )
    s.edge_collection.find = Mock(side_effect=_make_find_side_effect(edges))
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
        2 -- what NetworkX ``graph.degree()`` answers and what pgtable_impl's
        node_degree SQL is pinned to. The bidirectional ``$or`` alone matches
        one document and would answer 1."""
        s = _make_storage(_SELF_LOOP)

        assert await s.node_degree("Loop") == 2

    @pytest.mark.asyncio
    async def test_agrees_with_node_degrees_batch_on_a_self_loop(self):
        """The defect this closes: the scalar and the batch ranked the same
        node differently, so which value a caller saw depended on which public
        method it reached for."""
        s = _make_storage(_SELF_LOOP)

        scalar = await s.node_degree("Loop")
        batch = await s.node_degrees_batch(["Loop"])

        assert scalar == batch["Loop"] == 2

    @pytest.mark.asyncio
    async def test_plain_edges_are_not_inflated(self):
        """The self-loop count must add nothing when there is no self-loop --
        A has two ordinary edges (A->B, C->A) and stays at degree 2."""
        s = _make_storage(_PLAIN)

        scalar = await s.node_degree("A")
        batch = await s.node_degrees_batch(["A"])

        assert scalar == batch["A"] == 2

    @pytest.mark.asyncio
    async def test_absent_node_has_degree_zero(self):
        s = _make_storage(_PLAIN)

        assert await s.node_degree("Ghost") == 0

    @pytest.mark.asyncio
    async def test_edge_degree_sums_both_endpoints(self):
        """``edge_degree`` is two ``node_degree`` calls, so the self-loop rule
        reaches it: (Loop, Loop) is 2 + 2."""
        s = _make_storage(_SELF_LOOP)

        assert await s.edge_degree("Loop", "Loop") == 4


class TestGetNodesEdgesBatchSelfLoop:
    @pytest.mark.asyncio
    async def test_self_loop_is_listed_once(self):
        """One edge, one tuple: the self-loop matches both the outbound and the
        inbound query, and listing it from each reports one edge as two."""
        s = _make_storage(_SELF_LOOP)

        result = await s.get_nodes_edges_batch(["Loop"])

        assert result["Loop"] == [("Loop", "Loop")]

    @pytest.mark.asyncio
    async def test_self_loop_matches_get_node_edges(self):
        """``get_node_edges`` reads the same edge through a single ``$or`` and
        has always returned it once; the batch form must not disagree."""
        s = _make_storage(_SELF_LOOP)
        s.has_node = AsyncMock(return_value=True)

        single = await s.get_node_edges("Loop")
        batch = await s.get_nodes_edges_batch(["Loop"])

        assert single == batch["Loop"] == [("Loop", "Loop")]

    @pytest.mark.asyncio
    async def test_ordinary_edge_still_reaches_both_endpoints(self):
        """The skip must be scoped to ``src == tgt``: an ordinary edge is still
        listed for its source AND its target, which is what makes the batch
        represent an undirected graph."""
        s = _make_storage(_PLAIN)

        result = await s.get_nodes_edges_batch(["A", "B"])

        assert ("A", "B") in result["A"]
        assert ("A", "B") in result["B"]
        assert ("C", "A") in result["A"]

    @pytest.mark.asyncio
    async def test_self_loop_alongside_ordinary_edges(self):
        """A node carrying both kinds keeps every ordinary edge and exactly one
        copy of the loop."""
        s = _make_storage(_PLAIN + [{"source_node_id": "A", "target_node_id": "A"}])

        result = await s.get_nodes_edges_batch(["A"])

        assert result["A"].count(("A", "A")) == 1
        assert len(result["A"]) == 3
