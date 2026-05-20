import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from lightrag.kg.mongo_impl import MongoDocStatusStorage, MongoGraphStorage

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
            return_value={"_id": "doc-1", "file_path": "report.pdf", "status": "processed"}
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
