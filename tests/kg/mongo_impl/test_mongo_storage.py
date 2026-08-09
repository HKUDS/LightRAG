import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from pymongo.errors import PyMongoError, BulkWriteError, DuplicateKeyError

from lightrag.kg.mongo_impl import (
    MongoDocStatusStorage,
    MongoGraphStorage,
    _canonical_edge_endpoints,
)
from lightrag.types import KnowledgeGraph

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self.sort_calls = []

    def sort(self, *args, **kwargs):
        self.sort_calls.append((args, kwargs))
        return self

    def limit(self, n: int):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _make_node_find_side_effect(real_ids: set):
    """Mock ``collection.find({"_id": {"$in": ids}})``, dropping ids that have
    no node document -- mirrors a dangling edge endpoint. Writes now materialize
    both endpoints, so new data cannot produce one, but rows written before that
    change still can and the traversal must keep tolerating them."""

    def _find(query, projection=None):
        ids = query["_id"]["$in"]
        return _AsyncCursor(
            [{"_id": i, "entity_type": "person"} for i in ids if i in real_ids]
        )

    return _find


def _make_edge_find_side_effect(edges: list):
    """Mock ``edge_collection.find`` for both the ``$or`` (neighbor lookup)
    and ``$and`` (final edge-set) query shapes used by the bidirectional BFS."""

    def _find(query):
        if "$or" in query:
            ids = set(query["$or"][0]["source_node_id"]["$in"])
            return _AsyncCursor(
                [
                    e
                    for e in edges
                    if e["source_node_id"] in ids or e["target_node_id"] in ids
                ]
            )
        ids = set(query["$and"][0]["source_node_id"]["$in"])
        return _AsyncCursor(
            [
                e
                for e in edges
                if e["source_node_id"] in ids and e["target_node_id"] in ids
            ]
        )

    return _find


class TestMongoGraphStorage:
    def _make_storage(self):
        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.workspace = "test"
        storage.global_config = {"max_graph_nodes": 1000}
        storage._collection_name = "test_nodes"
        storage._edge_collection_name = "test_edges"
        storage.collection = SimpleNamespace()
        storage.edge_collection = SimpleNamespace()
        return storage

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_backfills_isolated_nodes_when_truncated(
        self,
    ):
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=5)
        storage.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor(
                [{"_id": "A", "degree": 1}, {"_id": "B", "degree": 1}]
            )
        )

        # The ranked page is resolved against the node collection BEFORE the
        # isolated top-up runs, so the $in query names only the ranked ids and
        # the top-up excludes what that resolved to. Deliberately answers out of
        # order: _fetch_nodes_by_ids re-imposes the requested order.
        def collection_find_side_effect(query, projection=None):
            if query == {"_id": {"$in": ["A", "B"]}}:
                return _AsyncCursor(
                    [
                        {"_id": "B", "entity_type": "person"},
                        {"_id": "A", "entity_type": "person"},
                    ]
                )
            if query == {"_id": {"$nin": ["A", "B"]}}:
                return _AsyncCursor(
                    [
                        {"_id": "C", "entity_type": "person"},
                        {"_id": "D", "entity_type": "person"},
                        {"_id": "E", "entity_type": "person"},
                    ]
                )
            raise AssertionError(f"Unexpected node query: {query}")

        storage.collection.find = Mock(side_effect=collection_find_side_effect)
        storage.edge_collection.find = Mock(
            return_value=_AsyncCursor(
                [
                    {
                        "source_node_id": "A",
                        "target_node_id": "B",
                        "relationship": "knows",
                    }
                ]
            )
        )

        result = await storage.get_knowledge_graph_all_by_degree(
            max_depth=2, max_nodes=4
        )

        assert result.is_truncated is True
        assert [node.id for node in result.nodes] == ["A", "B", "C", "D"]
        assert len(result.edges) == 1
        assert result.edges[0].source == "A"
        assert result.edges[0].target == "B"

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_breaks_degree_ties_on_the_label(self):
        """``$limit max_nodes`` cuts through a band of equal-degree entities,
        and the BaseGraphStorage contract orders that band by the label.

        Without the second sort key, which entities survived the cutoff was
        whatever order the aggregation happened to emit them in.
        """
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=5)
        storage.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor([{"_id": "A", "degree": 1}])
        )
        storage.collection.find = Mock(return_value=_AsyncCursor([]))
        storage.edge_collection.find = Mock(return_value=_AsyncCursor([]))

        await storage.get_knowledge_graph_all_by_degree(max_depth=2, max_nodes=2)

        pipeline = storage.edge_collection.aggregate.call_args[0][0]
        assert {"$sort": {"degree": -1, "_id": 1}} in pipeline, pipeline
        assert {"$sort": {"degree": -1}} not in pipeline, pipeline

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_tops_up_isolated_nodes_in_label_order(self):
        """The degree-0 top-up needs an order for the same reason the ranking
        does: an unsorted ``find`` handed the remaining slots to whichever
        documents the collection scan reached first."""
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=9)
        storage.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor([{"_id": "Connected", "degree": 2}])
        )
        top_up = _AsyncCursor([{"_id": "Aardvark"}, {"_id": "Orphan"}])

        def _find(query, projection=None):
            if "$in" in query["_id"]:  # resolving the ranked page
                return _AsyncCursor([{"_id": "Connected", "entity_type": "person"}])
            return top_up

        storage.collection.find = Mock(side_effect=_find)
        storage.edge_collection.find = Mock(return_value=_AsyncCursor([]))

        result = await storage.get_knowledge_graph_all_by_degree(
            max_depth=2, max_nodes=3
        )

        assert [node.id for node in result.nodes] == [
            "Connected",
            "Aardvark",
            "Orphan",
        ]
        assert top_up.sort_calls == [(("_id", 1), {})], top_up.sort_calls
        # Asserting the filter AFTER the call is what pins the snapshot: the
        # same list is appended to while this cursor is consumed, so a live
        # reference would read back as ["Connected", "Aardvark", "Orphan"].
        top_up_query = next(
            call[0][0]
            for call in storage.collection.find.call_args_list
            if "$nin" in call[0][0]["_id"]
        )
        assert top_up_query == {"_id": {"$nin": ["Connected"]}}

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_does_not_let_a_dangling_id_take_a_slot(self):
        """The ranked ids are edge ENDPOINTS, and upsert_edge only guarantees
        the source node exists, so one can be a dangling id with no node
        document -- a legacy state this backend tolerates on purpose.

        Treating the server-side ``$limit`` as final let such an id consume a
        real node's slot, and ``_fetch_nodes_by_ids`` then dropped it without
        refilling. The label tie-break is what makes it reachable: here the
        dangling ``A`` sorts ahead of the real ``B`` and, left unresolved, the
        wildcard view comes back empty.
        """
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=3)
        # What the server returns for max_nodes=1 with an edge B -> A: both tie
        # at degree 1 and "A" sorts first.
        storage.edge_collection.aggregate = AsyncMock(
            side_effect=[
                _AsyncCursor([{"_id": "A", "degree": 1}]),
                _AsyncCursor([{"_id": "B", "degree": 1}]),  # the refill page
            ]
        )

        def _find(query, projection=None):
            if "$in" in query["_id"]:  # only B has a node document
                return _AsyncCursor(
                    [
                        {"_id": node_id, "entity_type": "person"}
                        for node_id in query["_id"]["$in"]
                        if node_id == "B"
                    ]
                )
            return _AsyncCursor([])

        storage.collection.find = Mock(side_effect=_find)
        storage.edge_collection.find = Mock(return_value=_AsyncCursor([]))

        result = await storage.get_knowledge_graph_all_by_degree(
            max_depth=2, max_nodes=1
        )

        assert [node.id for node in result.nodes] == ["B"]

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_refills_from_the_next_ranked_page(self):
        """A slot a dangling id vacated goes to the NEXT-RANKED entry.

        The $limit is server-side, so the refill has to re-aggregate with a
        $skip. That is only sound because the sort is a total order -- degree
        descending then the unique _id -- and it must be preferred over the
        isolated top-up, which would hand the slot to an arbitrary entity.
        """
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=9)
        storage.edge_collection.aggregate = AsyncMock(
            side_effect=[
                _AsyncCursor(
                    [{"_id": "Dangling", "degree": 9}, {"_id": "Real", "degree": 5}]
                ),
                _AsyncCursor([{"_id": "Next", "degree": 4}]),
            ]
        )

        def _find(query, projection=None):
            if "$in" in query["_id"]:
                return _AsyncCursor(
                    [
                        {"_id": node_id, "entity_type": "person"}
                        for node_id in query["_id"]["$in"]
                        if node_id != "Dangling"
                    ]
                )
            raise AssertionError("isolated top-up must not run: the band refilled")

        storage.collection.find = Mock(side_effect=_find)
        storage.edge_collection.find = Mock(return_value=_AsyncCursor([]))

        result = await storage.get_knowledge_graph_all_by_degree(
            max_depth=2, max_nodes=2
        )

        assert [node.id for node in result.nodes] == ["Real", "Next"]
        # A whole page, skipping the one already consumed: pages are
        # {_id, degree} rows and cheap next to the aggregation producing them,
        # so each extra pass should make as much progress as it can.
        refill_pipeline = storage.edge_collection.aggregate.call_args_list[1][0][0]
        assert {"$skip": 2} in refill_pipeline, refill_pipeline
        assert {"$limit": 2} in refill_pipeline, refill_pipeline

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_exhausts_the_band_before_topping_up(self):
        """An isolate must never outrank an entity still left in the ranked band.

        Stopping the refill after one page let the degree-0 top-up -- which
        orders on the label alone, over EVERY remaining node document, not just
        isolates -- take slots the band could still fill. Here "Aaa" sorts ahead
        of the degree-1 "Zed", so a single dangling id in each of the first two
        pages was enough to drop Zed from a degree-ranked view.

        The top-up is only correct where it is reachable only on exhaustion:
        that is what makes "the isolated (degree-0) entities" true of whatever
        it can still pick.
        """
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=9)
        # One dangling id in each of the first two pages.
        band = [
            {"_id": "Hi", "degree": 5},
            {"_id": "Dang1", "degree": 4},
            {"_id": "Dang2", "degree": 3},
            {"_id": "Zed", "degree": 1},
        ]

        async def _aggregate(pipeline, **kwargs):
            # Honour $skip/$limit: a mock that ignores them hands a
            # shortfall-sized request a whole page and hides the defect.
            skip = next((s["$skip"] for s in pipeline if "$skip" in s), 0)
            limit = next(s["$limit"] for s in pipeline if "$limit" in s)
            return _AsyncCursor(band[skip : skip + limit])

        storage.edge_collection.aggregate = _aggregate

        real = {"Hi", "Zed", "Aaa"}  # Dang1/Dang2 have no node document

        def _find(query, projection=None):
            if "$in" in query["_id"]:
                return _AsyncCursor(
                    [
                        {"_id": node_id, "entity_type": "person"}
                        for node_id in query["_id"]["$in"]
                        if node_id in real
                    ]
                )
            # The top-up scans every node not already accepted, label-ordered.
            return _AsyncCursor(
                [
                    {"_id": node_id, "entity_type": "person"}
                    for node_id in sorted(real - set(query["_id"]["$nin"]))
                ]
            )

        storage.collection.find = Mock(side_effect=_find)
        storage.edge_collection.find = Mock(return_value=_AsyncCursor([]))

        result = await storage.get_knowledge_graph_all_by_degree(
            max_depth=2, max_nodes=2
        )

        assert [node.id for node in result.nodes] == ["Hi", "Zed"]

    @pytest.mark.asyncio
    async def test_bidirectional_bfs_reports_truncated_and_respects_cap(self):
        """Star graph exceeding max_nodes must be flagged truncated and must
        never return more than max_nodes nodes."""
        storage = self._make_storage()
        real_ids = {"A", "B", "C", "D", "E", "F"}
        edges = [
            {"source_node_id": "A", "target_node_id": leaf}
            for leaf in ["B", "C", "D", "E", "F"]
        ]
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )
        storage.edge_collection.find = Mock(
            side_effect=_make_edge_find_side_effect(edges)
        )

        result = await storage.get_knowledge_subgraph_bidirectional_bfs(
            "A", 0, max_depth=2, max_nodes=3
        )

        assert result.is_truncated is True
        assert len(result.nodes) <= 3

    @pytest.mark.asyncio
    async def test_bidirectional_bfs_not_truncated_when_exactly_at_cap(self):
        """Diamond graph (A-B-D, A-C-D) whose 4 nodes exactly fill max_nodes
        must not be falsely reported as truncated."""
        storage = self._make_storage()
        real_ids = {"A", "B", "C", "D"}
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
            {"source_node_id": "B", "target_node_id": "D"},
            {"source_node_id": "C", "target_node_id": "D"},
        ]
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )
        storage.edge_collection.find = Mock(
            side_effect=_make_edge_find_side_effect(edges)
        )

        result = await storage.get_knowledge_subgraph_bidirectional_bfs(
            "A", 0, max_depth=2, max_nodes=4
        )

        assert result.is_truncated is False
        assert {node.id for node in result.nodes} == real_ids

    @pytest.mark.asyncio
    async def test_bidirectional_bfs_depth_limit_not_reported_as_node_truncation(self):
        """A-B-C chain with max_depth=1, max_nodes=2: A and B exactly fill the
        cap, and C is excluded purely by depth -- must not be misreported as
        node-cap truncation."""
        storage = self._make_storage()
        real_ids = {"A", "B", "C"}
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "B", "target_node_id": "C"},
        ]
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )
        storage.edge_collection.find = Mock(
            side_effect=_make_edge_find_side_effect(edges)
        )

        result = await storage._bidirectional_bfs_nodes(
            ["A"], set(), KnowledgeGraph(), depth=0, max_depth=1, max_nodes=2
        )

        assert [node.id for node in result.nodes] == ["A", "B"]
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_bidirectional_bfs_ignores_dangling_edge_endpoint(self):
        """An edge pointing at "X" with no node document must not be treated
        as a real node, and must not pollute is_truncated."""
        storage = self._make_storage()
        real_ids = {"A", "B"}
        edges = [
            {"source_node_id": "A", "target_node_id": "X"},
            {"source_node_id": "A", "target_node_id": "B"},
        ]
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )
        storage.edge_collection.find = Mock(
            side_effect=_make_edge_find_side_effect(edges)
        )

        result = await storage.get_knowledge_subgraph_bidirectional_bfs(
            "A", 0, max_depth=2, max_nodes=5
        )

        assert {node.id for node in result.nodes} == {"A", "B"}
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_in_out_bound_bfs_reports_truncated_and_respects_cap(self):
        storage = self._make_storage()
        real_ids = {"A", "B", "C", "D", "E", "F"}
        storage.collection.find_one = AsyncMock(
            return_value={"_id": "A", "entity_type": "person"}
        )
        storage.collection.aggregate = AsyncMock(
            return_value=_AsyncCursor(
                [
                    {
                        "connected_edges": [
                            {
                                "source_node_id": "A",
                                "target_node_id": leaf,
                                "depth": 0,
                                "weight": 1.0,
                            }
                            for leaf in ["B", "C", "D", "E", "F"]
                        ]
                    }
                ]
            )
        )
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )

        result = await storage.get_knowledge_subgraph_in_out_bound_bfs(
            "A", max_depth=2, max_nodes=3
        )

        assert result.is_truncated is True
        assert len(result.nodes) <= 3

    @pytest.mark.asyncio
    async def test_in_out_bound_bfs_not_truncated_when_exactly_at_cap(self):
        storage = self._make_storage()
        real_ids = {"A", "B", "C"}
        storage.collection.find_one = AsyncMock(
            return_value={"_id": "A", "entity_type": "person"}
        )
        storage.collection.aggregate = AsyncMock(
            return_value=_AsyncCursor(
                [
                    {
                        "connected_edges": [
                            {
                                "source_node_id": "A",
                                "target_node_id": leaf,
                                "depth": 0,
                                "weight": 1.0,
                            }
                            for leaf in ["B", "C"]
                        ]
                    }
                ]
            )
        )
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )

        result = await storage.get_knowledge_subgraph_in_out_bound_bfs(
            "A", max_depth=2, max_nodes=3
        )

        assert result.is_truncated is False
        assert {node.id for node in result.nodes} == real_ids

    @pytest.mark.asyncio
    async def test_in_out_bound_bfs_dangling_edge_endpoint_does_not_steal_real_node_slot(
        self,
    ):
        """A dangling candidate "X" ordered ahead of the real "C" (higher
        edge weight -> sorted first) must not consume a node slot that would
        otherwise go to a real node."""
        storage = self._make_storage()
        real_ids = {"A", "B", "C"}
        storage.collection.find_one = AsyncMock(
            return_value={"_id": "A", "entity_type": "person"}
        )
        storage.collection.aggregate = AsyncMock(
            return_value=_AsyncCursor(
                [
                    {
                        "connected_edges": [
                            {
                                "source_node_id": "A",
                                "target_node_id": "X",
                                "depth": 0,
                                "weight": 2.0,
                            },
                            {
                                "source_node_id": "A",
                                "target_node_id": "B",
                                "depth": 0,
                                "weight": 1.0,
                            },
                            {
                                "source_node_id": "A",
                                "target_node_id": "C",
                                "depth": 0,
                                "weight": 1.0,
                            },
                        ]
                    }
                ]
            )
        )
        storage.collection.find = Mock(
            side_effect=_make_node_find_side_effect(real_ids)
        )

        result = await storage.get_knowledge_subgraph_in_out_bound_bfs(
            "A", max_depth=2, max_nodes=3
        )

        assert {node.id for node in result.nodes} == {"A", "B", "C"}
        assert result.is_truncated is False


class TestMongoEdgeKey:
    """Canonical edge-endpoint writes, duplicate-key retries, and the migration."""

    def _make_storage(self):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s.global_config = {}
        s._edge_collection_name = "test_edges"
        s._max_upsert_payload_bytes = 16 * 1024 * 1024
        s._max_upsert_records_per_batch = 128
        # upsert_edge materializes both endpoints through one bulk_write.
        s.collection = SimpleNamespace(update_one=AsyncMock(), bulk_write=AsyncMock())
        s.edge_collection = SimpleNamespace()
        return s

    def test_canonical_edge_endpoints_are_direction_independent(self):
        assert _canonical_edge_endpoints("B", "A") == _canonical_edge_endpoints(
            "A", "B"
        )
        assert _canonical_edge_endpoints("B", "A") == ("A", "B")

    def test_canonical_endpoints_never_collide_across_delimiter(self):
        # Distinct pairs that a delimiter-joined key could conflate must stay
        # distinct as separate (lo, hi) fields.
        assert _canonical_edge_endpoints("A\x1fB", "C") != _canonical_edge_endpoints(
            "A", "B\x1fC"
        )

    @pytest.mark.asyncio
    async def test_upsert_edge_filters_and_sets_canonical_endpoints(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock()
        await s.upsert_edge("B", "A", {"weight": 1.0, "source_id": "c1<SEP>c2"})

        args, kwargs = s.edge_collection.update_one.call_args
        filt, update = args[0], args[1]
        lo, hi = _canonical_edge_endpoints("B", "A")
        assert filt == {"edge_lo": lo, "edge_hi": hi}
        assert update["$set"]["edge_lo"] == lo
        assert update["$set"]["edge_hi"] == hi
        assert update["$set"]["source_node_id"] == "B"
        assert update["$set"]["target_node_id"] == "A"
        assert update["$set"]["source_ids"] == ["c1", "c2"]
        assert kwargs.get("upsert") is True

    @pytest.mark.asyncio
    async def test_upsert_edge_materializes_both_endpoints(self):
        """Both endpoints get a placeholder node, not just the source.

        A target-only endpoint would leave an edge whose target has no node
        document — externally visible as has_node() and get_node_edges()
        disagreeing, and as get_popular_labels ranking an id get_node returns
        nothing for. $setOnInsert so an endpoint that already carries real
        properties is never overwritten by the placeholder.
        """
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock()
        await s.upsert_edge("B", "A", {"weight": 1.0})

        (node_ops,), kwargs = s.collection.bulk_write.call_args
        assert [op._filter["_id"] for op in node_ops] == ["B", "A"]
        assert all(
            op._doc == {"$setOnInsert": {"_id": op._filter["_id"]}} for op in node_ops
        )
        assert all(op._upsert for op in node_ops)
        assert kwargs.get("ordered") is False

    @pytest.mark.asyncio
    async def test_upsert_edge_self_loop_materializes_one_endpoint(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock()
        await s.upsert_edge("Loop", "Loop", {"weight": 1.0})

        (node_ops,), _kwargs = s.collection.bulk_write.call_args
        assert [op._filter["_id"] for op in node_ops] == ["Loop"]

    @pytest.mark.asyncio
    async def test_upsert_edge_retries_once_on_duplicate_key(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock(
            side_effect=[DuplicateKeyError("E11000 dup"), None]
        )
        await s.upsert_edge("A", "B", {"weight": 1.0})
        assert s.edge_collection.update_one.await_count == 2

    @pytest.mark.asyncio
    async def test_upsert_edge_reraises_on_persistent_duplicate(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock(
            side_effect=DuplicateKeyError("E11000 dup")
        )
        with pytest.raises(DuplicateKeyError):
            await s.upsert_edge("A", "B", {"weight": 1.0})
        assert s.edge_collection.update_one.await_count == 2

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_dedupes_reciprocal_and_sets_endpoints(self):
        s = self._make_storage()
        calls = []

        async def fake_bulk(collection, ops, **kwargs):
            calls.append((collection, ops))

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            await s.upsert_edges_batch(
                [("A", "B", {"weight": 1.0}), ("B", "A", {"weight": 2.0})]
            )

        # First call is the endpoint-placeholder bulk: BOTH endpoints of every
        # edge, deduped — not just the sources.
        node_collection, node_ops = calls[0]
        assert node_collection is s.collection
        assert [op._filter["_id"] for op, _bytes, _logid in node_ops] == ["A", "B"]

        # Last call is the edge bulk.
        edge_collection, edge_ops = calls[-1]
        assert edge_collection is s.edge_collection
        assert len(edge_ops) == 1  # reciprocal pair collapsed to one op
        op, _bytes, _logid = edge_ops[0]
        lo, hi = _canonical_edge_endpoints("A", "B")
        assert op._filter == {"edge_lo": lo, "edge_hi": hi}
        assert op._doc["$set"]["edge_lo"] == lo
        assert op._doc["$set"]["edge_hi"] == hi
        assert op._doc["$set"]["weight"] == 2.0  # last-write-wins

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_retries_on_duplicate_bulk_error(self):
        s = self._make_storage()
        seq = []

        async def fake_bulk(collection, ops, **kwargs):
            seq.append(collection)
            # Fail the first edge bulk with an all-11000 BulkWriteError.
            if collection is s.edge_collection and seq.count(s.edge_collection) == 1:
                raise BulkWriteError({"writeErrors": [{"code": 11000}]})

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])

        # node bulk + edge bulk (raises) + edge bulk (retry succeeds)
        assert seq.count(s.edge_collection) == 2

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_reraises_non_duplicate_bulk_error(self):
        s = self._make_storage()

        async def fake_bulk(collection, ops, **kwargs):
            if collection is s.edge_collection:
                raise BulkWriteError({"writeErrors": [{"code": 121}]})

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            with pytest.raises(BulkWriteError):
                await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_surfaces_non_dup_error_hidden_behind_dup(self):
        """Under ordered=True the bulk aborts at the first failing op, so a
        non-11000 error sitting *behind* a leading 11000 race is not reported on
        the first pass. It must still surface — on the retry, where the leading
        dup has resolved to an update and the real error now leads."""
        s = self._make_storage()
        edge_calls = []

        async def fake_bulk(collection, ops, **kwargs):
            if collection is s.edge_collection:
                edge_calls.append(collection)
                if len(edge_calls) == 1:
                    # First pass: ordered abort reports only the leading dup race.
                    raise BulkWriteError({"writeErrors": [{"code": 11000}]})
                # Retry: dup resolved to an update, the hidden error now leads.
                raise BulkWriteError({"writeErrors": [{"code": 121}]})

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            with pytest.raises(BulkWriteError) as exc:
                await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])
        # Retried once (dup-only first pass), then the non-dup error re-raised.
        assert len(edge_calls) == 2
        assert exc.value.details["writeErrors"][0]["code"] == 121

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_reraises_write_concern_error(self):
        """A writeConcern-only BulkWriteError (empty writeErrors) is a durability
        failure — it must surface, not be masked by the duplicate-key retry."""
        s = self._make_storage()
        edge_calls = []

        async def fake_bulk(collection, ops, **kwargs):
            if collection is s.edge_collection:
                edge_calls.append(collection)
                raise BulkWriteError(
                    {"writeErrors": [], "writeConcernErrors": [{"code": 64}]}
                )

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            with pytest.raises(BulkWriteError):
                await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])
        assert len(edge_calls) == 1  # not retried

    @pytest.mark.asyncio
    async def test_edge_migration_skips_when_index_exists(self):
        s = self._make_storage()
        s.edge_collection.list_indexes = AsyncMock(
            return_value=SimpleNamespace(
                to_list=AsyncMock(return_value=[{"name": "test_edge_endpoints_unique"}])
            )
        )
        s.edge_collection.aggregate = AsyncMock()
        s.edge_collection.create_index = AsyncMock()

        await s.create_edge_indexes_and_migrate_if_not_exists()

        s.edge_collection.aggregate.assert_not_awaited()
        s.edge_collection.create_index.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edge_migration_dedupes_backfills_and_builds_unique_index(self):
        s = self._make_storage()
        s.edge_collection.list_indexes = AsyncMock(
            return_value=SimpleNamespace(
                to_list=AsyncMock(return_value=[{"name": "_id_"}])
            )
        )
        # One duplicate group for edge {A,B}: two docs with distinct provenance
        # AND distinct relation payload (description/keywords/weight).
        group = {
            "_id": {"lo": "A", "hi": "B"},
            "count": 2,
            "docs": [
                {
                    "_id": 1,
                    "source_id": "c1",
                    "source_ids": ["c1"],
                    "file_path": "f1",
                    "description": "d1",
                    "keywords": "alpha,beta",
                    "weight": 1.0,
                    "created_at": 10,
                },
                {
                    "_id": 2,
                    "source_id": "c2",
                    "source_ids": ["c2"],
                    "file_path": "f2",
                    "description": "d2",
                    "keywords": "beta,gamma",
                    "weight": 2.0,
                    "created_at": 20,
                },
            ],
        }
        s.edge_collection.estimated_document_count = AsyncMock(return_value=4)
        s.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
        s.edge_collection.update_one = AsyncMock()
        s.edge_collection.delete_many = AsyncMock()
        s.edge_collection.update_many = AsyncMock(
            return_value=SimpleNamespace(modified_count=3)
        )
        s.edge_collection.create_index = AsyncMock()

        with patch("lightrag.kg.mongo_impl.logger") as mock_logger:
            await s.create_edge_indexes_and_migrate_if_not_exists()

        # OpenSearch-aligned start/complete log wording.
        info_lines = [c.args[0] for c in mock_logger.info.call_args_list]
        assert any(
            line.startswith("[test] Starting canonical edge migration for ")
            and "(~4 edges to scan)" in line
            for line in info_lines
        )
        assert any(
            "Canonical edge migration complete for" in line
            and "scanned 4, deduped 1, backfilled 3" in line
            for line in info_lines
        )

        # Survivor is the newest (created_at 20 → _id 2); the full relation
        # payload is merged in before the duplicate is deleted (no evidence lost).
        surv_filter, surv_update = s.edge_collection.update_one.call_args[0]
        set_fields = surv_update["$set"]
        assert surv_filter == {"_id": 2}
        assert set_fields["source_ids"] == ["c1", "c2"]
        assert set_fields["file_path"] == "f1<SEP>f2"
        assert set_fields["description"] == "d1<SEP>d2"  # distinct descriptions joined
        assert set_fields["keywords"] == "alpha,beta,gamma"  # comma set-union, sorted
        assert set_fields["weight"] == 3.0  # summed (1.0 + 2.0), idempotent
        # The other duplicate is deleted.
        assert s.edge_collection.delete_many.call_args[0][0] == {"_id": {"$in": [1]}}
        # Compound unique partial index built as the completion flag.
        ci_args = s.edge_collection.create_index.call_args.args
        ci_kwargs = s.edge_collection.create_index.call_args.kwargs
        assert ci_args[0] == [("edge_lo", 1), ("edge_hi", 1)]
        assert ci_kwargs["name"] == "test_edge_endpoints_unique"
        assert ci_kwargs["unique"] is True
        assert set(ci_kwargs["partialFilterExpression"]) == {"edge_lo", "edge_hi"}

    async def _run_dedupe(self, s, group):
        """Run _dedupe_legacy_edges over a single group; return the survivor $set."""
        s.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
        s.edge_collection.update_one = AsyncMock()
        s.edge_collection.delete_many = AsyncMock()
        await s._dedupe_legacy_edges()
        if s.edge_collection.update_one.call_args is None:
            return None
        return s.edge_collection.update_one.call_args[0][1]["$set"]

    @pytest.mark.asyncio
    async def test_dedupe_coerces_string_weights(self):
        """Legacy string weights (graph API is dict[str, str]) must not crash the
        migration; they coerce to float and sum."""
        s = self._make_storage()
        group = {
            "_id": {"lo": "A", "hi": "B"},
            "count": 2,
            "docs": [
                {"_id": 1, "source_ids": ["c1"], "weight": "1.0", "created_at": 10},
                {"_id": 2, "source_ids": ["c2"], "weight": "2.0", "created_at": 20},
            ],
        }
        set_fields = await self._run_dedupe(s, group)
        assert set_fields["weight"] == 3.0  # coerced floats summed, no TypeError

    @pytest.mark.asyncio
    async def test_dedupe_merge_is_idempotent_on_retry(self):
        """A fail-fast retry that re-processes an already-merged survivor must
        produce the same fields (no double-summed weight, no grown description)."""
        s = self._make_storage()
        original = [
            {
                "_id": 1,
                "source_id": "c1",
                "source_ids": ["c1"],
                "file_path": "f1",
                "description": "d1",
                "keywords": "alpha,beta",
                "weight": 1.0,
                "created_at": 10,
            },
            {
                "_id": 2,
                "source_id": "c2",
                "source_ids": ["c2"],
                "file_path": "f2",
                "description": "d2",
                "keywords": "beta,gamma",
                "weight": 2.0,
                "created_at": 20,
            },
        ]
        first = await self._run_dedupe(
            s, {"_id": {"lo": "A", "hi": "B"}, "count": 2, "docs": list(original)}
        )

        # Retry: survivor (_id 2) already carries the merged fields, the other
        # duplicate is still present (the previous delete never committed).
        survivor_merged = {"_id": 2, "created_at": 20, **first}
        second = await self._run_dedupe(
            s,
            {
                "_id": {"lo": "A", "hi": "B"},
                "count": 2,
                "docs": [original[0], survivor_merged],
            },
        )

        # Summed once (1.0 + 2.0); the retry must NOT re-add the duplicate's
        # weight (its source_ids are already folded into the survivor).
        assert second["weight"] == first["weight"] == 3.0
        assert second["description"] == first["description"] == "d1<SEP>d2"
        assert second["source_ids"] == first["source_ids"] == ["c1", "c2"]
        assert second["file_path"] == first["file_path"] == "f1<SEP>f2"
        assert second["keywords"] == first["keywords"] == "alpha,beta,gamma"


class TestMongoEdgeReadCanonicalFilter:
    """Exact single-edge reads/deletes use the canonical (edge_lo, edge_hi)
    filter so they hit the compound unique index instead of the bidirectional
    ``$or``. Per-node scans (node_degree/get_node_edges/delete_node) intentionally
    keep ``$or`` — edge_hi is not a compound-index prefix, so they gain nothing."""

    def _make_storage(self, max_delete_records_per_batch=1000):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s._edge_collection_name = "test_edges"
        s._max_delete_records_per_batch = max_delete_records_per_batch
        s.edge_collection = SimpleNamespace()
        return s

    @pytest.mark.asyncio
    async def test_has_edge_uses_canonical_filter(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value={"_id": "x"})
        assert await s.has_edge("B", "A") is True

        filt, projection = s.edge_collection.find_one.call_args[0]
        lo, hi = _canonical_edge_endpoints("B", "A")
        assert filt == {"edge_lo": lo, "edge_hi": hi}
        assert projection == {"_id": 1}

    @pytest.mark.asyncio
    async def test_has_edge_direction_independent(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value=None)
        await s.has_edge("A", "B")
        forward = s.edge_collection.find_one.call_args[0][0]
        await s.has_edge("B", "A")
        backward = s.edge_collection.find_one.call_args[0][0]
        assert forward == backward

    @pytest.mark.asyncio
    async def test_get_edge_uses_canonical_filter(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value={"weight": 1.0})
        await s.get_edge("B", "A")
        filt = s.edge_collection.find_one.call_args[0][0]
        lo, hi = _canonical_edge_endpoints("B", "A")
        assert filt == {"edge_lo": lo, "edge_hi": hi}

    @pytest.mark.asyncio
    async def test_remove_edges_uses_canonical_pairs_and_collapses_reciprocals(self):
        s = self._make_storage()
        s.edge_collection.delete_many = AsyncMock()
        # (A,B) and its reciprocal (B,A) must collapse to a single clause.
        await s.remove_edges([("A", "B"), ("B", "A"), ("C", "D")])

        query = s.edge_collection.delete_many.call_args[0][0]
        lo_ab, hi_ab = _canonical_edge_endpoints("A", "B")
        lo_cd, hi_cd = _canonical_edge_endpoints("C", "D")
        assert query == {
            "$or": [
                {"edge_lo": lo_ab, "edge_hi": hi_ab},
                {"edge_lo": lo_cd, "edge_hi": hi_cd},
            ]
        }

    @pytest.mark.asyncio
    async def test_remove_edges_empty_is_noop(self):
        s = self._make_storage()
        s.edge_collection.delete_many = AsyncMock()
        await s.remove_edges([])
        s.edge_collection.delete_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_remove_edges_chunks_large_or_by_record_cap(self):
        # 5 distinct edges with a cap of 2 → 3 delete_many calls (2 + 2 + 1),
        # and every input edge is covered exactly once across the chunks.
        s = self._make_storage(max_delete_records_per_batch=2)
        s.edge_collection.delete_many = AsyncMock()
        edges = [(f"n{i}", f"n{i + 1}") for i in range(5)]
        await s.remove_edges(edges)

        calls = s.edge_collection.delete_many.call_args_list
        assert [len(c[0][0]["$or"]) for c in calls] == [2, 2, 1]
        clauses = [clause for c in calls for clause in c[0][0]["$or"]]
        expected = [
            {"edge_lo": lo, "edge_hi": hi}
            for lo, hi in (_canonical_edge_endpoints(src, tgt) for src, tgt in edges)
        ]
        assert clauses == expected

    @pytest.mark.asyncio
    async def test_remove_edges_non_positive_cap_disables_chunking(self):
        # A non-positive cap means "no record-count splitting": one delete_many.
        s = self._make_storage(max_delete_records_per_batch=0)
        s.edge_collection.delete_many = AsyncMock()
        edges = [(f"n{i}", f"n{i + 1}") for i in range(5)]
        await s.remove_edges(edges)

        assert s.edge_collection.delete_many.await_count == 1
        assert len(s.edge_collection.delete_many.call_args[0][0]["$or"]) == 5


class _FakeFindCursor:
    """Records the find(...).sort(...).limit(...).to_list(...) chain."""

    def __init__(self, docs, error=None):
        self._docs = list(docs)
        self._error = error
        self.sort_spec = None
        self.limit_value = None

    def sort(self, spec):
        self.sort_spec = spec
        return self

    def limit(self, n):
        self.limit_value = n
        return self

    async def to_list(self, length=None):
        if self._error is not None:
            raise self._error
        return self._docs


class TestMongoDocStatusLookup:
    """Cover the Mongo-native overrides for basename / content_hash lookups."""

    def _make_storage(self):
        storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
        storage.workspace = "test"
        storage.global_config = {}
        storage._collection_name = "test_doc_status"
        storage._data = SimpleNamespace()
        return storage

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_returns_tuple_on_hit(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(
            return_value={
                "_id": "doc-1",
                "file_path": "report.pdf",
                "status": "processed",
            }
        )

        result = await storage.get_doc_by_file_basename("report.pdf")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["file_path"] == "report.pdf"
        # Primary-only lookup: duplicate markers (metadata.is_duplicate=true)
        # are excluded so filename dedup never matches a dup-* row.
        storage._data.find_one.assert_awaited_once_with(
            {"file_path": "report.pdf", "metadata.is_duplicate": {"$ne": True}}
        )

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_empty_returns_none_without_query(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock()

        assert await storage.get_doc_by_file_basename("") is None
        storage._data.find_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_unknown_source_sentinel(self):
        # Lookup for the sentinel must not match real rows that happen to have
        # file_path == "unknown_source".
        storage = self._make_storage()
        storage._data.find_one = AsyncMock()

        assert await storage.get_doc_by_file_basename("unknown_source") is None
        storage._data.find_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_miss_returns_none(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(return_value=None)

        assert await storage.get_doc_by_file_basename("missing.pdf") is None

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_returns_earliest_on_hit(self):
        # Sorted + limited server-side: the base contract promises the EARLIEST
        # holder, because that id is persisted as a duplicate's original_doc_id.
        storage = self._make_storage()
        cursor = _FakeFindCursor(
            [
                {
                    "_id": "doc-1",
                    "file_path": "report.pdf",
                    "content_hash": "abc123",
                    "status": "processed",
                }
            ]
        )
        storage._data.find = MagicMock(return_value=cursor)

        result = await storage.get_doc_by_content_hash("abc123")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["content_hash"] == "abc123"
        storage._data.find.assert_called_once_with({"content_hash": "abc123"})
        assert cursor.sort_spec == [("created_at", 1), ("_id", 1)]
        assert cursor.limit_value == 1

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_empty_returns_none_without_query(self):
        # Empty hash must short-circuit so it cannot match legacy rows missing
        # the field via accidental coercion.
        storage = self._make_storage()
        storage._data.find = MagicMock()

        assert await storage.get_doc_by_content_hash("") is None
        storage._data.find.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_miss_returns_none(self):
        storage = self._make_storage()
        storage._data.find = MagicMock(return_value=_FakeFindCursor([]))

        assert await storage.get_doc_by_content_hash("zzz999") is None

    @pytest.mark.asyncio
    async def test_basename_lookup_swallows_pymongo_error_and_returns_none(self):
        # The legacy basename lookup keeps its documented best-effort miss;
        # identity-critical callers use resolve_doc_source_strict instead.
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(side_effect=PyMongoError("boom"))

        assert await storage.get_doc_by_file_basename("report.pdf") is None

    @pytest.mark.asyncio
    async def test_content_hash_lookup_raises_instead_of_reporting_no_match(self):
        """Fail-proof: a PyMongoError used to be swallowed into None, which the
        dedup callers read as "no duplicate" — so a transport blip enqueued a
        duplicate row and ingested its content again. It must propagate."""
        storage = self._make_storage()
        storage._data.find = MagicMock(
            return_value=_FakeFindCursor([], error=PyMongoError("boom"))
        )

        with pytest.raises(PyMongoError):
            await storage.get_doc_by_content_hash("abc123")


class TestMongoGraphReadContract:
    """Read-path contracts MongoGraphStorage shares with the other backends.

    Both cases used to answer with a value that means something else:
    ``get_popular_labels`` ranked only nodes that appear in the edge collection,
    so isolated entities were missing entirely, and ``get_node_edges`` returned
    ``[]`` for an entity that does not exist -- indistinguishable from one that
    exists with no relations.
    """

    def _make_storage(self):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s._edge_collection_name = "test_edges"
        s.collection = SimpleNamespace()
        s.edge_collection = SimpleNamespace()
        return s

    # -- get_popular_labels -------------------------------------------------

    @staticmethod
    def _stub_isolated(s, ids):
        """Stub the phase-2 node-collection top-up (find().sort().limit())."""
        captured = {}

        def _find(query, projection=None):
            captured["query"] = query
            cursor = _AsyncCursor([{"_id": i} for i in ids])
            cursor.sort = lambda *a, **k: cursor
            return cursor

        s.collection.find = Mock(side_effect=_find)
        return captured

    @pytest.mark.asyncio
    async def test_popular_labels_ranks_from_edges_without_touching_the_nodes(self):
        """Phase 1 is the cheap edge-side aggregation, and on a graph with
        enough connected entities it is the ONLY thing that runs.

        The node collection holds far more entities than `limit` in any real
        graph, so making it drive the ranking would mean a full pass over it on
        every call to produce a result the edge aggregation already had.
        """
        s = self._make_storage()
        s.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor([{"_id": f"E{i}"} for i in range(5)])
        )
        s.collection.aggregate = AsyncMock()
        s.collection.find = Mock()

        labels = await s.get_popular_labels(limit=5)

        assert labels == [f"E{i}" for i in range(5)]
        s.edge_collection.aggregate.assert_awaited_once()
        # Limit filled by connected entities: the node collection is untouched.
        s.collection.find.assert_not_called()
        s.collection.aggregate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_popular_labels_sums_degrees_and_keeps_ranking_stages(self):
        """Degrees sum across both endpoint groups, ordered and capped."""
        s = self._make_storage()
        s.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([]))
        self._stub_isolated(s, [])

        await s.get_popular_labels(limit=42)

        pipeline = s.edge_collection.aggregate.call_args[0][0]
        assert pipeline[0] == {
            "$group": {"_id": "$source_node_id", "out_degree": {"$sum": 1}}
        }
        union = next(stage for stage in pipeline if "$unionWith" in stage)
        assert union["$unionWith"]["coll"] == "test_edges"
        assert union["$unionWith"]["pipeline"][0]["$group"]["_id"] == "$target_node_id"
        assert {"$sort": {"total_degree": -1, "_id": 1}} in pipeline
        assert {"$limit": 42} in pipeline
        assert s.edge_collection.aggregate.call_args.kwargs == {"allowDiskUse": True}

    @pytest.mark.asyncio
    async def test_popular_labels_tops_up_from_isolated_entities_when_short(self):
        """Regression: a graph whose entities carry no relations reported an
        EMPTY picker, because an entity with no edge document has no group to
        land in. Phase 1 coming up short is exactly that signal -- and since its
        aggregation is exact, every node outside the result is isolated.
        """
        s = self._make_storage()
        s.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor([{"_id": "Connected"}])
        )
        captured = self._stub_isolated(s, ["Aardvark", "Orphan"])

        labels = await s.get_popular_labels(limit=3)

        # Connected first, then isolated in label order.
        assert labels == ["Connected", "Aardvark", "Orphan"]
        # The top-up excludes what phase 1 already returned...
        assert captured["query"] == {"_id": {"$nin": ["Connected"]}}
        # ... and asks only for the shortfall, so it can never scan the whole
        # collection: at most `limit` rows, ordered by the _id index.
        assert s.collection.find.call_args[0][1] == {"_id": 1}

    @pytest.mark.asyncio
    async def test_popular_labels_returns_labels_in_ranked_order(self):
        s = self._make_storage()
        s.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor([{"_id": "Alpha"}, {"_id": "Beta"}])
        )
        self._stub_isolated(s, ["Orphan"])

        assert await s.get_popular_labels() == ["Alpha", "Beta", "Orphan"]

    # -- get_node_edges -----------------------------------------------------

    def _stub_edges(self, s, docs):
        s.edge_collection.find = MagicMock(
            side_effect=lambda query, projection=None: _AsyncCursor(docs)
        )

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_none_for_missing_node(self):
        """Regression: an absent entity answered ``[]``, the same value an
        existing-but-isolated entity yields. The contract (NetworkXStorage) is
        ``None``."""
        s = self._make_storage()
        self._stub_edges(s, [])
        s.collection.find_one = AsyncMock(return_value=None)

        assert await s.get_node_edges("NoSuchEntity") is None

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_empty_list_for_isolated_node(self):
        s = self._make_storage()
        self._stub_edges(s, [])
        s.collection.find_one = AsyncMock(return_value={"_id": "Lonely"})

        assert await s.get_node_edges("Lonely") == []

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_edges_for_an_existing_node(self):
        s = self._make_storage()
        self._stub_edges(
            s,
            [
                {"source_node_id": "Alpha", "target_node_id": "Beta"},
                {"source_node_id": "Gamma", "target_node_id": "Alpha"},
            ],
        )
        s.collection.find_one = AsyncMock(return_value={"_id": "Alpha"})

        assert await s.get_node_edges("Alpha") == [
            ("Alpha", "Beta"),
            ("Gamma", "Alpha"),
        ]

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_none_for_a_dangling_target_endpoint(self):
        """Having edges does NOT prove the node exists in this backend.

        Writes now materialize both endpoints, but rows written before that
        change can still hold an id that appears only as an edge target with no
        node document. Inferring existence from the edge scan would make this
        method answer "exists, here are its edges" for an id that has_node()
        reports absent — and delete_entity 404s on has_node(). The read must not
        depend on a write-path invariant that older data does not satisfy.
        """
        s = self._make_storage()
        self._stub_edges(s, [{"source_node_id": "Alpha", "target_node_id": "Target"}])
        s.collection.find_one = AsyncMock(return_value=None)  # no node document

        assert await s.get_node_edges("Target") is None
        # The absent node also skips the edge scan entirely.
        s.edge_collection.find.assert_not_called()


class TestMongoLabelHelpersFailLoud:
    """A database error in the label helpers must propagate.

    Returning [] reports a dead database as "this graph has no entities" --
    and /graph/label/popular already turns an exception into a 500, so the
    swallow was the only thing standing between the user and an accurate error.
    """

    def _make_storage(self):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s._collection_name = "test_nodes"
        s._edge_collection_name = "test_edges"
        s.collection = SimpleNamespace()
        s.edge_collection = SimpleNamespace()
        return s

    @pytest.mark.asyncio
    async def test_popular_labels_raises_on_aggregation_error(self):
        s = self._make_storage()
        s.edge_collection.aggregate = AsyncMock(side_effect=PyMongoError("boom"))

        with pytest.raises(PyMongoError):
            await s.get_popular_labels(limit=10)

    @pytest.mark.asyncio
    async def test_search_labels_raises_when_node_count_fails(self):
        """The pre-flight count is an optimisation, not a verdict: a failed
        count is not "the graph is empty"."""
        s = self._make_storage()
        s.collection.count_documents = AsyncMock(side_effect=PyMongoError("boom"))

        with pytest.raises(PyMongoError):
            await s.search_labels("alpha")

    @pytest.mark.asyncio
    async def test_search_labels_raises_when_regex_fallback_fails(self):
        """Atlas Search may legitimately be unavailable and fall through to the
        regex scan -- but if that last resort fails too, no answer was produced.
        """
        s = self._make_storage()
        s.collection.count_documents = AsyncMock(return_value=3)
        s._try_atlas_text_search = AsyncMock(side_effect=PyMongoError("no atlas"))
        s._try_atlas_autocomplete_search = AsyncMock(
            side_effect=PyMongoError("no atlas")
        )
        s._try_atlas_compound_search = AsyncMock(side_effect=PyMongoError("no atlas"))
        s.collection.find = MagicMock(side_effect=PyMongoError("scan failed"))

        with pytest.raises(PyMongoError):
            await s.search_labels("alpha")

    @pytest.mark.asyncio
    async def test_search_labels_still_returns_empty_for_an_empty_graph(self):
        """A confirmed-empty node collection is a real "no labels", not an error."""
        s = self._make_storage()
        s.collection.count_documents = AsyncMock(return_value=0)

        assert await s.search_labels("alpha") == []
