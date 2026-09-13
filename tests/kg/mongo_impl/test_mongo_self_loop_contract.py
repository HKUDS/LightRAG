"""Self-loop LISTING contract for ``MongoGraphStorage``.

``BaseGraphStorage.get_node_edges``: a self-loop appears ONCE and is never
hidden. This backend walks its outbound and inbound matches separately, so
without a guard it reports one edge as two -- the same guard
``pgtable_impl.get_nodes_edges_batch`` carries
(``test_get_nodes_edges_batch_self_loop_counted_once``).

Listing it at all is deliberate even though ``BaseGraphStorage.node_degree``
says a graph must not CONTAIN a self-loop: entity deletion, rename, merge and
document purge all resolve a node's relation rows through these methods, so an
edge hidden here becomes an orphan relation vector row. A self-loop the listing
hides also cannot be repaired.

Degree is NOT asserted here. The contract does not define a self-loop's degree
-- the store is not allowed to hold one -- and the backends answer from their
cheapest natural query, so pinning a number would pin an unreachable case.

Only the low-level collection is faked, asserting on the actual query issued --
same convention as ``test_mongo_edge_batch_methods.py``.
"""

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
    s.edge_collection.find = Mock(side_effect=_make_find_side_effect(edges))
    return s


_SELF_LOOP = [{"source_node_id": "Loop", "target_node_id": "Loop"}]
_PLAIN = [
    {"source_node_id": "A", "target_node_id": "B"},
    {"source_node_id": "C", "target_node_id": "A"},
]


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
