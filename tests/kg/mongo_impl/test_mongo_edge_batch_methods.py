"""Read-path contract MongoGraphStorage shares with the other graph backends
for the batch methods BaseGraphStorage declares an override contract for
(``lightrag/base.py``): ``node_degrees_batch``, ``edge_degrees_batch``, and
``get_edges_batch``. Mirrors
``tests/kg/neo4j_impl/test_neo4j_graph_read_contract.py`` — a real storage
instance, only the low-level driver/collection faked, asserting on the
actual query issued and the result shape.
"""

from collections import Counter

import pytest
from unittest.mock import AsyncMock, Mock, patch

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from lightrag.kg.mongo_impl import MongoGraphStorage, _canonical_edge_endpoints

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


def _make_edge_aggregate_side_effect(edges: list[dict]):
    """Mock ``edge_collection.aggregate`` for the two ``node_degrees_batch``
    pipelines that ``edge_degrees_batch`` composes on top of -- outbound
    grouped on ``source_node_id``, inbound on ``target_node_id``. Same
    helper shape as ``test_mongo_storage.py``'s BFS-degree mock, since it is
    the same underlying pipeline."""

    async def _aggregate(pipeline, **kwargs):
        match = pipeline[0]["$match"]
        field = "source_node_id" if "source_node_id" in match else "target_node_id"
        ids = set(match[field]["$in"])
        counts = Counter(e[field] for e in edges if e[field] in ids)
        return _AsyncCursor(
            [{"_id": key, "degree": count} for key, count in counts.items()]
        )

    return _aggregate


def _make_edge_find_side_effect(docs: list[dict]):
    """Mock ``edge_collection.find`` for ``get_edges_batch``'s
    ``$or``-of-``(edge_lo, edge_hi)`` query, returning only the docs whose
    canonical endpoints appear in *that specific call's* ``$or`` clauses --
    unlike a static ``return_value``, this actually simulates a chunked call
    only seeing its own chunk's matches, so a test can tell chunk-scoped
    attribution apart from "every call happens to get everything back"."""

    def _find(query, *args, **kwargs):
        wanted = {(c["edge_lo"], c["edge_hi"]) for c in query["$or"]}
        matches = [d for d in docs if (d["edge_lo"], d["edge_hi"]) in wanted]
        return _AsyncCursor(matches)

    return _find


def _make_storage():
    s = MongoGraphStorage.__new__(MongoGraphStorage)
    s.workspace = "test"
    s.namespace = "chunk_entity_relation"
    s.global_config = {}
    s._edge_collection_name = "test_edges"
    s.edge_collection = Mock()
    # edge_collection.find() is synchronous (returns a cursor directly, like
    # the real pymongo async driver); only iterating the cursor is awaited.
    s.edge_collection.find = Mock()
    return s


class TestNodeDegreesBatch:
    @pytest.mark.asyncio
    async def test_sums_outbound_and_inbound_degree(self):
        s = _make_storage()
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
            {"source_node_id": "D", "target_node_id": "A"},
        ]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        result = await s.node_degrees_batch(["A"])

        # A: 2 outbound (A->B, A->C) + 1 inbound (D->A) = 3.
        assert result == {"A": 3}

    @pytest.mark.asyncio
    async def test_issues_two_aggregate_calls_when_ids_fit_in_one_chunk(self):
        s = _make_storage()
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect([])
        )

        await s.node_degrees_batch(["A", "B", "C"])

        assert s.edge_collection.aggregate.await_count == 2  # outbound + inbound

    @pytest.mark.asyncio
    async def test_chunks_in_when_ids_exceed_the_chunk_size(self):
        """A large node_ids list must be split across bounded $in queries."""
        s = _make_storage()
        node_ids = [f"n{i}" for i in range(5)]
        edges = [{"source_node_id": nid, "target_node_id": "sink"} for nid in node_ids]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        with patch("lightrag.kg.mongo_impl._NODE_DEGREES_BATCH_CHUNK_SIZE", 2):
            result = await s.node_degrees_batch(node_ids)

        # 5 ids / chunk size 2 => 3 chunks => 2 aggregate calls each => 6.
        assert s.edge_collection.aggregate.await_count == 6
        for call in s.edge_collection.aggregate.await_args_list:
            pipeline = call.args[0]
            match = pipeline[0]["$match"]
            field = next(iter(match))
            assert len(match[field]["$in"]) <= 2
        assert result == {nid: 1 for nid in node_ids}

    @pytest.mark.asyncio
    async def test_dedupes_ids_before_chunking(self):
        """Repeated ids in the input must not waste chunks -- the same
        node_id showing up twice (e.g. as both an edge source and target
        elsewhere in the caller's set) is deduped before chunking."""
        s = _make_storage()
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect([])
        )

        with patch("lightrag.kg.mongo_impl._NODE_DEGREES_BATCH_CHUNK_SIZE", 2):
            await s.node_degrees_batch(["A", "A", "A"])

        # 1 unique id fits in one chunk of size 2 => 2 calls, not 4 (3/2 rounded up).
        assert s.edge_collection.aggregate.await_count == 2

    @pytest.mark.asyncio
    async def test_chunks_in_by_payload_bytes_even_under_the_count_cap(self):
        """Long IDs must be split by payload size even below the count cap."""
        s = _make_storage()
        node_ids = [f"very-long-entity-id-{i:040d}" for i in range(5)]
        edges = [{"source_node_id": nid, "target_node_id": "sink"} for nid in node_ids]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        with patch("lightrag.kg.mongo_impl._GRAPH_BATCH_QUERY_MAX_PAYLOAD_BYTES", 80):
            result = await s.node_degrees_batch(node_ids)

        # 5 ids never trip the 8192 count cap; the byte budget forces 1 id
        # per chunk => 5 chunks => 10 aggregate calls (outbound + inbound).
        assert s.edge_collection.aggregate.await_count == 10
        for call in s.edge_collection.aggregate.await_args_list:
            pipeline = call.args[0]
            match = pipeline[0]["$match"]
            field = next(iter(match))
            assert len(match[field]["$in"]) == 1
        assert result == {nid: 1 for nid in node_ids}


class TestEdgeDegreesBatch:
    @pytest.mark.asyncio
    async def test_skips_query_for_empty_input(self):
        s = _make_storage()
        s.edge_collection.aggregate = AsyncMock()

        assert await s.edge_degrees_batch([]) == {}
        s.edge_collection.aggregate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sums_source_and_target_degree_per_pair(self):
        """A-B, A-C, A-D: A has degree 3 (three outbound edges), B has degree
        1. edge_degrees_batch("A", "B") must equal edge_degree("A", "B") --
        3 + 1 -- the same source+target degree sum BaseGraphStorage's default
        computes one pair at a time."""
        s = _make_storage()
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
            {"source_node_id": "A", "target_node_id": "D"},
        ]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        result = await s.edge_degrees_batch([("A", "B"), ("A", "D")])

        assert result == {("A", "B"): 4, ("A", "D"): 4}

    @pytest.mark.asyncio
    async def test_missing_node_contributes_zero_degree(self):
        s = _make_storage()
        edges = [{"source_node_id": "A", "target_node_id": "B"}]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        result = await s.edge_degrees_batch([("A", "Ghost")])

        assert result == {("A", "Ghost"): 1}

    @pytest.mark.asyncio
    async def test_issues_one_aggregate_round_trip_pair_per_direction_not_per_edge(
        self,
    ):
        """Batching contract: node degrees are resolved via node_degrees_batch's
        existing two aggregation pipelines (outbound + inbound), once for the
        whole edge_pairs list -- not one edge_degree() round trip per pair."""
        s = _make_storage()
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "B", "target_node_id": "C"},
            {"source_node_id": "C", "target_node_id": "D"},
        ]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        await s.edge_degrees_batch([("A", "B"), ("B", "C"), ("C", "D")])

        assert s.edge_collection.aggregate.await_count == 2  # outbound + inbound


class TestGetEdgesBatch:
    @pytest.mark.asyncio
    async def test_skips_query_for_empty_input(self):
        s = _make_storage()

        assert await s.get_edges_batch([]) == {}
        s.edge_collection.find.assert_not_called()

    @pytest.mark.asyncio
    async def test_omits_missing_edges(self):
        s = _make_storage()
        s.edge_collection.find.return_value = _AsyncCursor([])

        assert await s.get_edges_batch([{"src": "A", "tgt": "B"}]) == {}

        query = s.edge_collection.find.call_args.args[0]
        lo, hi = _canonical_edge_endpoints("A", "B")
        assert query == {"$or": [{"edge_lo": lo, "edge_hi": hi}]}

    @pytest.mark.asyncio
    async def test_returns_existing_edge_for_each_requested_direction(self):
        s = _make_storage()
        lo, hi = _canonical_edge_endpoints("A", "B")
        doc = {"_id": "x", "edge_lo": lo, "edge_hi": hi, "weight": 1.0}
        s.edge_collection.find.return_value = _AsyncCursor([doc])

        result = await s.get_edges_batch([{"src": "A", "tgt": "B"}])

        assert result == {("A", "B"): {"edge_lo": lo, "edge_hi": hi, "weight": 1.0}}

    @pytest.mark.asyncio
    async def test_undirected_property_forward_and_reverse_requests_both_resolve(self):
        """Two requested pairs that share one canonical edge (opposite
        directions), requested in the SAME call, must both be present in the
        result with equal content -- the collision canonical_to_requested
        exists to handle. Mirrors test_graph_storage.py's
        test_graph_undirected_property check on the same method."""
        s = _make_storage()
        lo, hi = _canonical_edge_endpoints("A", "B")
        doc = {"_id": "x", "edge_lo": lo, "edge_hi": hi, "weight": 2.0}
        s.edge_collection.find.return_value = _AsyncCursor([doc])

        result = await s.get_edges_batch(
            [{"src": "A", "tgt": "B"}, {"src": "B", "tgt": "A"}]
        )

        expected = {"edge_lo": lo, "edge_hi": hi, "weight": 2.0}
        assert result[("A", "B")] == expected
        assert result[("B", "A")] == expected

    @pytest.mark.asyncio
    async def test_undirected_property_results_are_independent_objects(self):
        """Regression test: result[("A","B")] and result[("B","A")] must not
        be the same dict object. A shared object means a caller mutating one
        entry in place (e.g. lightrag.py's purge path setting "source"/
        "target" defaults on the returned edge dict) silently corrupts the
        other requested direction's entry too."""
        s = _make_storage()
        lo, hi = _canonical_edge_endpoints("A", "B")
        doc = {"_id": "x", "edge_lo": lo, "edge_hi": hi}
        s.edge_collection.find.return_value = _AsyncCursor([doc])

        result = await s.get_edges_batch(
            [{"src": "A", "tgt": "B"}, {"src": "B", "tgt": "A"}]
        )

        assert result[("A", "B")] is not result[("B", "A")]

        result[("A", "B")]["source"] = "A"
        assert "source" not in result[("B", "A")]

    @pytest.mark.asyncio
    async def test_strips_mongo_id_from_returned_documents(self):
        s = _make_storage()
        lo, hi = _canonical_edge_endpoints("A", "B")
        doc = {"_id": "x", "edge_lo": lo, "edge_hi": hi}
        s.edge_collection.find.return_value = _AsyncCursor([doc])

        result = await s.get_edges_batch([{"src": "A", "tgt": "B"}])

        assert "_id" not in result[("A", "B")]

    @pytest.mark.asyncio
    async def test_queries_with_or_across_canonical_pairs(self):
        s = _make_storage()
        s.edge_collection.find.return_value = _AsyncCursor([])
        lo1, hi1 = _canonical_edge_endpoints("A", "B")
        lo2, hi2 = _canonical_edge_endpoints("C", "D")

        await s.get_edges_batch([{"src": "A", "tgt": "B"}, {"src": "C", "tgt": "D"}])

        query = s.edge_collection.find.call_args.args[0]
        assert {"edge_lo": lo1, "edge_hi": hi1} in query["$or"]
        assert {"edge_lo": lo2, "edge_hi": hi2} in query["$or"]

    @pytest.mark.asyncio
    async def test_chunks_or_when_pairs_exceed_the_chunk_size(self):
        """A large pairs list must be split across bounded $or queries."""
        s = _make_storage()
        pairs = [{"src": f"n{i}", "tgt": f"n{i + 1}"} for i in range(5)]
        docs = []
        for i, p in enumerate(pairs):
            lo, hi = _canonical_edge_endpoints(p["src"], p["tgt"])
            # A per-pair-unique value (not a shared constant like 1.0) lets
            # the assertions below tell "the right doc landed on the right
            # pair" apart from "some doc landed on every pair".
            docs.append({"edge_lo": lo, "edge_hi": hi, "weight": float(i)})
        # side_effect (not a static return_value) filters to only the docs
        # whose canonical endpoints appear in *that call's* $or -- a real
        # MongoDB find() would only return matches for the filter it was
        # given, so a chunked call only ever sees its own chunk's docs. This
        # is what actually exercises "does chunking attribute results
        # correctly", not just "does the final count come out right".
        s.edge_collection.find = Mock(side_effect=_make_edge_find_side_effect(docs))

        with patch("lightrag.kg.mongo_impl._GET_EDGES_BATCH_CHUNK_SIZE", 2):
            result = await s.get_edges_batch(pairs)

        # 5 pairs / chunk size 2 => 3 chunks => 3 find() calls.
        assert s.edge_collection.find.call_count == 3
        for call in s.edge_collection.find.call_args_list:
            assert len(call.args[0]["$or"]) <= 2
        # Every pair resolved, each to its own doc's weight -- not just any
        # doc's weight, and not missing any (proves no cross-chunk bleed).
        for i, p in enumerate(pairs):
            assert result[(p["src"], p["tgt"])]["weight"] == float(i)

    @pytest.mark.asyncio
    async def test_single_find_call_when_pairs_fit_in_one_chunk(self):
        s = _make_storage()
        pairs = [{"src": f"n{i}", "tgt": f"n{i + 1}"} for i in range(5)]
        s.edge_collection.find.return_value = _AsyncCursor([])

        await s.get_edges_batch(pairs)

        assert s.edge_collection.find.call_count == 1
        assert len(s.edge_collection.find.call_args.args[0]["$or"]) == 5

    @pytest.mark.asyncio
    async def test_chunks_or_by_payload_bytes_even_under_the_count_cap(self):
        """Long endpoint IDs must be split by payload size below the count cap."""
        s = _make_storage()
        pairs = [
            {"src": f"very-long-entity-id-{i:040d}", "tgt": f"other-{i:040d}"}
            for i in range(5)
        ]
        docs = []
        for i, p in enumerate(pairs):
            lo, hi = _canonical_edge_endpoints(p["src"], p["tgt"])
            docs.append({"edge_lo": lo, "edge_hi": hi, "weight": float(i)})
        s.edge_collection.find = Mock(side_effect=_make_edge_find_side_effect(docs))

        with patch("lightrag.kg.mongo_impl._GRAPH_BATCH_QUERY_MAX_PAYLOAD_BYTES", 200):
            result = await s.get_edges_batch(pairs)

        assert s.edge_collection.find.call_count == 5
        for call in s.edge_collection.find.call_args_list:
            assert len(call.args[0]["$or"]) == 1
        for i, p in enumerate(pairs):
            assert result[(p["src"], p["tgt"])]["weight"] == float(i)
