import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

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


class TestMongoGraphStorage:
    def _make_storage(self):
        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.workspace = "test"
        storage.global_config = {"max_graph_nodes": 1000}
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

        def collection_find_side_effect(query, projection=None):
            if query == {"_id": {"$nin": ["A", "B"]}}:
                return _AsyncCursor(
                    [
                        {"_id": "C", "entity_type": "person"},
                        {"_id": "D", "entity_type": "person"},
                        {"_id": "E", "entity_type": "person"},
                    ]
                )
            if query == {"_id": {"$in": ["A", "B", "C", "D"]}}:
                return _AsyncCursor(
                    [
                        {"_id": "B", "entity_type": "person"},
                        {"_id": "D", "entity_type": "person"},
                        {"_id": "A", "entity_type": "person"},
                        {"_id": "C", "entity_type": "person"},
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
        s.collection = SimpleNamespace(update_one=AsyncMock())
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

        # Last call is the edge bulk (first is the node-placeholder bulk).
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

    def _make_storage(self):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s._edge_collection_name = "test_edges"
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
        storage._data.find_one.assert_awaited_once_with({"file_path": "report.pdf"})

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
    async def test_get_doc_by_content_hash_returns_tuple_on_hit(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(
            return_value={
                "_id": "doc-1",
                "file_path": "report.pdf",
                "content_hash": "abc123",
                "status": "processed",
            }
        )

        result = await storage.get_doc_by_content_hash("abc123")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["content_hash"] == "abc123"
        storage._data.find_one.assert_awaited_once_with({"content_hash": "abc123"})

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_empty_returns_none_without_query(self):
        # Empty hash must short-circuit so it cannot match legacy rows missing
        # the field via accidental coercion.
        storage = self._make_storage()
        storage._data.find_one = AsyncMock()

        assert await storage.get_doc_by_content_hash("") is None
        storage._data.find_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_miss_returns_none(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(return_value=None)

        assert await storage.get_doc_by_content_hash("zzz999") is None

    @pytest.mark.asyncio
    async def test_lookup_swallows_pymongo_error_and_returns_none(self):
        # PyMongoError must not propagate to the caller; the dedup path treats
        # a storage failure as "no match" and the error is logged instead.
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(side_effect=PyMongoError("boom"))

        assert await storage.get_doc_by_file_basename("report.pdf") is None
        assert await storage.get_doc_by_content_hash("abc123") is None
