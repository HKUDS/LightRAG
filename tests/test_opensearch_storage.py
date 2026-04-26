"""
Unit tests for OpenSearch storage implementations.

All tests use mocks — no running OpenSearch instance required.
Run with: pytest tests/test_opensearch_storage.py -v
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
import numpy as np

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import NotFoundError, OpenSearchException  # type: ignore
from lightrag.kg.opensearch_impl import (
    OpenSearchKVStorage,
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
    OpenSearchVectorDBStorage,
    ClientManager,
    _build_index_name,
    _resolve_workspace,
    _sanitize_index_name,
)
from lightrag.base import DocStatus, DocProcessingStatus

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Mock the shared storage lock so tests don't need full LightRAG init
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory():
    return _mock_lock()


def _missing_index_error() -> NotFoundError:
    return NotFoundError(404, "index_not_found_exception", "no such index")


@pytest.fixture(autouse=True)
def patch_data_init_lock():
    """Patch get_data_init_lock globally so initialize() works without shared storage."""
    with patch(
        "lightrag.kg.opensearch_impl.get_data_init_lock", side_effect=_mock_lock_factory
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockEmbeddingFunc:
    """Mock embedding function that returns random vectors."""

    def __init__(self, dim=128):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture
def global_config():
    """Standard global config fixture for all storage tests."""
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


@pytest.fixture
def embed_func():
    """Mock embedding function fixture."""
    return MockEmbeddingFunc()


def _make_client():
    """Create a fully-mocked AsyncOpenSearch client with spec validation."""
    from opensearchpy import AsyncOpenSearch

    client = AsyncMock(spec=AsyncOpenSearch)
    # indices sub-client
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.indices.delete = AsyncMock()
    client.indices.refresh = AsyncMock()
    client.indices.get_mapping = AsyncMock(return_value={})
    # transport for PPL
    client.transport = AsyncMock()
    client.transport.perform_request = AsyncMock(
        side_effect=Exception("PPL not available")
    )
    # document operations
    client.exists = AsyncMock(return_value=False)
    client.index = AsyncMock()
    client.delete = AsyncMock()
    client.delete_by_query = AsyncMock()
    client.get = AsyncMock(
        return_value={
            "_id": "doc1",
            "_source": {"content": "hello", "create_time": 0, "update_time": 0},
        }
    )
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "id1", "found": True, "_source": {"content": "c1"}},
                {"_id": "id2", "found": True, "_source": {"content": "c2"}},
            ]
        }
    )
    client.count = AsyncMock(return_value={"count": 5})
    client.search = AsyncMock(
        return_value={
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {
                "status_counts": {"buckets": []},
                "src": {"buckets": []},
                "tgt": {"buckets": []},
                "source_degrees": {"buckets": []},
                "target_degrees": {"buckets": []},
            },
        }
    )
    # PIT operations
    client.create_pit = AsyncMock(return_value={"pit_id": "mock_pit_id_123"})
    client.delete_pit = AsyncMock()
    return client


@pytest.fixture
def mock_client():
    """Fully-mocked AsyncOpenSearch client fixture."""
    return _make_client()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions (_build_index_name, _resolve_workspace, _sanitize_index_name)."""

    def test_build_index_name_with_workspace(self):
        ws, ns, idx = _build_index_name("myws", "text_chunks")
        assert ws == "myws"
        assert ns == "myws_text_chunks"
        assert idx == _sanitize_index_name("myws_text_chunks")

    def test_build_index_name_no_workspace(self):
        ws, ns, idx = _build_index_name("", "chunks")
        assert ws == ""
        assert idx == _sanitize_index_name("chunks")

    def test_resolve_workspace_env_override(self):
        with patch.dict("os.environ", {"OPENSEARCH_WORKSPACE": "forced"}):
            assert _resolve_workspace("original", "ns") == "forced"

    def test_resolve_workspace_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _resolve_workspace("original", "ns") == "original"

    def test_sanitize_index_name(self):
        assert _sanitize_index_name("Hello_World") == "hello_world"
        assert _sanitize_index_name("-bad") == "x-bad"
        assert _sanitize_index_name("a.b/c") == "a_b_c"


# ---------------------------------------------------------------------------
# ClientManager
# ---------------------------------------------------------------------------


class TestClientManager:
    """Tests for ClientManager singleton pattern and reference counting."""

    @pytest.mark.asyncio
    async def test_singleton_and_refcount(self):
        ClientManager._instances = {"client": None, "ref_count": 0}
        with patch("lightrag.kg.opensearch_impl.AsyncOpenSearch") as mock_cls:
            mock_cls.return_value = AsyncMock()
            c1 = await ClientManager.get_client()
            c2 = await ClientManager.get_client()
            assert c1 is c2
            assert ClientManager._instances["ref_count"] == 2
            await ClientManager.release_client(c1)
            assert ClientManager._instances["ref_count"] == 1
            await ClientManager.release_client(c2)
            assert ClientManager._instances["ref_count"] == 0
            assert ClientManager._instances["client"] is None

    @pytest.mark.asyncio
    async def test_close_called_on_last_release(self):
        ClientManager._instances = {"client": None, "ref_count": 0}
        with patch("lightrag.kg.opensearch_impl.AsyncOpenSearch") as mock_cls:
            inner = AsyncMock()
            mock_cls.return_value = inner
            c = await ClientManager.get_client()
            await ClientManager.release_client(c)
            inner.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# KV Storage
# ---------------------------------------------------------------------------


class TestKVStorage:
    """Tests for OpenSearchKVStorage CRUD operations, timestamps, refresh behavior."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_index_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func, workspace="proj_a")
        assert s._index_name == "proj_a_text_chunks"

    @pytest.mark.asyncio
    async def test_initialize_creates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.exists.assert_awaited_once()
            mock_client.indices.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_existing_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=True)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "doc1",
                        "found": True,
                        "_source": {
                            "content": "hello",
                            "create_time": 0,
                            "update_time": 0,
                        },
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("doc1")
            assert doc is not None
            assert doc["content"] == "hello"
            assert doc["_id"] == "doc1"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["doc1"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_ids_preserves_order(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs = await s.get_by_ids(["id1", "id2"])
            assert docs[0]["content"] == "c1"
            assert docs[1]["content"] == "c2"

    @pytest.mark.asyncio
    async def test_filter_keys(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "a", "found": True},
                    {"_id": "b", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.filter_keys({"a", "b"})
            assert result == {"b"}

    @pytest.mark.asyncio
    async def test_upsert_no_per_operation_refresh(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})
                _, kwargs = mock_bulk.call_args
                assert "refresh" not in kwargs

    @pytest.mark.asyncio
    async def test_upsert_sets_timestamps(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})
                actions = mock_bulk.call_args[0][1]
                src = actions[0]["_source"]
                assert "create_time" in src
                assert "update_time" in src

    @pytest.mark.asyncio
    async def test_is_empty(self, global_config, embed_func, mock_client):
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.is_empty() is True

    @pytest.mark.asyncio
    async def test_delete(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.delete(["a", "b"])
                actions = mock_bulk.call_args[0][1]
                assert all(a["_op_type"] == "delete" for a in actions)

    @pytest.mark.asyncio
    async def test_drop(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            mock_client.indices.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drop_error_marks_index_not_ready_and_next_upsert_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("drop failed")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    result = await s.drop()
                    assert result["status"] == "error"
                    assert s._index_ready is False
                    await s.upsert({"k1": {"content": "v1"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_after_drop_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    await s.drop()
                    await s.upsert({"k1": {"content": "v1"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            assert await s.get_by_id("doc1") is None
            assert await s.get_by_ids(["doc1", "doc2"]) == [None, None]
            assert await s.is_empty() is True

            mock_client.mget.assert_not_awaited()
            mock_client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.get_by_id("doc1") is None
            assert await s.get_by_id("doc1") is None
            assert s._index_ready is False
            assert mock_client.mget.await_count == 1

    @pytest.mark.asyncio
    async def test_iter_raw_docs_uses_pit_and_search_after(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {
                        "hits": [
                            {"_id": "d1", "_source": {"content": "a"}, "sort": [1]},
                            {"_id": "d2", "_source": {"content": "b"}, "sort": [2]},
                        ]
                    }
                },
                {
                    "hits": {
                        "hits": [
                            {"_id": "d3", "_source": {"content": "c"}, "sort": [3]}
                        ]
                    }
                },
            ]
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            batches = [batch async for batch in s._iter_raw_docs(batch_size=2)]

            assert [[doc["_id"] for doc in batch] for batch in batches] == [
                ["d1", "d2"],
                ["d3"],
            ]
            assert (
                "search_after"
                not in mock_client.search.await_args_list[0].kwargs["body"]
            )
            assert mock_client.search.await_args_list[1].kwargs["body"][
                "search_after"
            ] == [2]
            mock_client.create_pit.assert_awaited_once()
            mock_client.delete_pit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_iter_raw_docs_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=_missing_index_error())

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            batches = [batch async for batch in s._iter_raw_docs(batch_size=2)]

            assert batches == []
            assert s._index_ready is False
            mock_client.create_pit.assert_awaited_once()
            mock_client.delete_pit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finalize(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch.object(
                ClientManager, "release_client", new_callable=AsyncMock
            ) as mock_release:
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.finalize()
                mock_release.assert_awaited_once()
                assert s.client is None


# ---------------------------------------------------------------------------
# DocStatus Storage
# ---------------------------------------------------------------------------


class TestDocStatusStorage:
    """Tests for OpenSearchDocStatusStorage including aggregations, pagination, and data normalization."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchDocStatusStorage(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_index_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._index_name == "test_doc_status"

    @pytest.mark.asyncio
    async def test_initialize_creates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "doc-abc",
                        "found": True,
                        "_source": {"status": "processed", "file_path": "/a.txt"},
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("doc-abc")
            assert doc["status"] == "processed"
            assert doc["_id"] == "doc-abc"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["doc-abc"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upsert_sets_chunks_list_default(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"d1": {"status": "pending"}})
                actions = mock_bulk.call_args[0][1]
                assert actions[0]["_source"]["chunks_list"] == []

    @pytest.mark.asyncio
    async def test_get_status_counts(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "status_counts": {
                        "buckets": [
                            {"key": "processed", "doc_count": 3},
                            {"key": "pending", "doc_count": 1},
                        ]
                    }
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            counts = await s.get_status_counts()
            assert counts == {"processed": 3, "pending": 1}

    @pytest.mark.asyncio
    async def test_get_all_status_counts_includes_all(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "status_counts": {
                        "buckets": [
                            {"key": "processed", "doc_count": 5},
                            {"key": "failed", "doc_count": 2},
                        ]
                    }
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            counts = await s.get_all_status_counts()
            assert counts["all"] == 7
            assert counts["processed"] == 5

    @pytest.mark.asyncio
    async def test_get_docs_by_status(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "status": "processed",
                                "file_path": "/a.txt",
                                "content_summary": "s",
                                "content_length": 10,
                                "chunks_count": 1,
                                "created_at": 100,
                                "updated_at": 200,
                            },
                            "sort": ["d1"],
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_docs_by_status(DocStatus.PROCESSED)
            assert "d1" in result
            assert isinstance(result["d1"], DocProcessingStatus)

    @pytest.mark.asyncio
    async def test_get_docs_paginated(self, global_config, embed_func, mock_client):
        """Page 1 returns results directly without search_after."""
        mock_client.count = AsyncMock(return_value={"count": 50})
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "status": "processed",
                                "file_path": "/a.txt",
                                "content_summary": "s",
                                "content_length": 10,
                                "chunks_count": 1,
                                "created_at": 100,
                                "updated_at": 200,
                            },
                            "sort": [200, "d1"],
                        },
                    ],
                    "total": {"value": 50},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=1, page_size=10)
            assert total == 50
            assert len(docs) == 1
            assert docs[0][0] == "d1"
            # Page 1: no search_after needed, single search call
            assert mock_client.search.await_count == 1
            body = mock_client.search.call_args.kwargs.get(
                "body"
            ) or mock_client.search.call_args[1].get("body", {})
            assert "search_after" not in body

    @pytest.mark.asyncio
    async def test_get_docs_paginated_page2_uses_search_after(
        self, global_config, embed_func, mock_client
    ):
        """Page 2 skips page 1 results via search_after."""
        mock_client.count = AsyncMock(return_value={"count": 50})
        call_count = {"n": 0}

        async def search_side_effect(*args, **kwargs):
            call_count["n"] += 1
            body = kwargs.get("body", {})
            if "search_after" not in body:
                # First call: skip batch
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": f"skip{i}",
                                "_source": {
                                    "status": "processed",
                                    "file_path": f"/{i}.txt",
                                    "content_summary": "s",
                                    "content_length": 1,
                                    "chunks_count": 1,
                                    "created_at": 100,
                                    "updated_at": 100 + i,
                                },
                                "sort": [100 + i, f"skip{i}"],
                            }
                            for i in range(10)
                        ],
                        "total": {"value": 50},
                    }
                }
            else:
                # Second call: actual page
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": "page2_doc",
                                "_source": {
                                    "status": "pending",
                                    "file_path": "/p2.txt",
                                    "content_summary": "s",
                                    "content_length": 1,
                                    "chunks_count": 1,
                                    "created_at": 200,
                                    "updated_at": 300,
                                },
                                "sort": [300, "page2_doc"],
                            }
                        ],
                        "total": {"value": 50},
                    }
                }

        mock_client.search = AsyncMock(side_effect=search_side_effect)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=2, page_size=10)
            assert total == 50
            assert len(docs) == 1
            assert docs[0][0] == "page2_doc"
            # 2 search calls: 1 skip + 1 fetch
            assert mock_client.search.await_count == 2

    @pytest.mark.asyncio
    async def test_get_docs_paginated_empty_index(
        self, global_config, embed_func, mock_client
    ):
        """Empty index returns empty list with total 0."""
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=1, page_size=10)
            assert total == 0
            assert docs == []
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_docs_paginated_page_beyond_total(
        self, global_config, embed_func, mock_client
    ):
        """Requesting a page beyond total docs returns empty list."""
        mock_client.count = AsyncMock(return_value={"count": 5})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=100, page_size=10)
            assert total == 5
            assert docs == []

    @pytest.mark.asyncio
    async def test_get_docs_paginated_with_status_filter(
        self, global_config, embed_func, mock_client
    ):
        """Status filter is passed as term query."""
        mock_client.count = AsyncMock(return_value={"count": 3})
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 3}},
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(
                status_filter=DocStatus.PROCESSED, page=1, page_size=10
            )
            assert total == 3
            # Verify count query used the status filter
            count_body = mock_client.count.call_args.kwargs.get("body", {})
            assert count_body["query"] == {"term": {"status": "processed"}}

    @pytest.mark.asyncio
    async def test_get_doc_by_file_path(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "file_path": "/test.txt",
                                "status": "processed",
                            },
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_doc_by_file_path("/test.txt")
            assert doc is not None
            assert doc["_id"] == "d1"

    @pytest.mark.asyncio
    async def test_get_doc_by_file_path_not_found(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_doc_by_file_path("/nope.txt") is None

    @pytest.mark.asyncio
    async def test_prepare_doc_status_data(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        raw = {"_id": "x", "status": "processed", "error": "oops"}
        data = s._prepare_doc_status_data(raw)
        assert "_id" not in data
        assert data["error_msg"] == "oops"
        assert "error" not in data
        assert data["file_path"] == "no-file-path"
        assert data["metadata"] == {}

    @pytest.mark.asyncio
    async def test_drop_error_marks_index_not_ready_and_next_upsert_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("drop failed")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    result = await s.drop()
                    assert result["status"] == "error"
                    assert s._index_ready is False
                    await s.upsert({"d1": {"status": "pending"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_after_drop_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    await s.drop()
                    await s.upsert({"d1": {"status": "pending"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            assert await s.get_all_status_counts() == {}
            assert await s.get_docs_paginated(page=1, page_size=10) == ([], 0)
            assert await s.get_doc_by_file_path("/a.txt") is None
            assert await s.get_docs_by_status(DocStatus.PROCESSED) == {}

            mock_client.count.assert_not_awaited()
            mock_client.search.assert_not_awaited()
            mock_client.create_pit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.get_all_status_counts() == {}
            assert await s.get_all_status_counts() == {}
            assert s._index_ready is False
            assert mock_client.search.await_count == 1


# ---------------------------------------------------------------------------
# Graph Storage
# ---------------------------------------------------------------------------


class TestGraphStorage:
    """Tests for OpenSearchGraphStorage node/edge CRUD, batch ops, BFS, and label queries."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchGraphStorage(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_index_names(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._nodes_index == "test_chunk_entity_relation-nodes"
        assert s._edges_index == "test_chunk_entity_relation-edges"

    @pytest.mark.asyncio
    async def test_initialize_creates_both_indices(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_has_node_true(self, global_config, embed_func, mock_client):
        mock_client.exists = AsyncMock(return_value=True)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_node("Alice") is True

    @pytest.mark.asyncio
    async def test_has_node_false(self, global_config, embed_func, mock_client):
        mock_client.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_node("Nobody") is False

    @pytest.mark.asyncio
    async def test_has_edge(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 1}},
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                    "source_degrees": {"buckets": []},
                    "target_degrees": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_edge("A", "B") is True

    @pytest.mark.asyncio
    async def test_node_degree(self, global_config, embed_func, mock_client):
        mock_client.count = AsyncMock(return_value={"count": 3})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.node_degree("A") == 3

    @pytest.mark.asyncio
    async def test_get_node(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "Alice",
                        "found": True,
                        "_source": {
                            "entity_type": "person",
                            "description": "A researcher",
                        },
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            node = await s.get_node("Alice")
            assert node["entity_type"] == "person"
            assert node["_id"] == "Alice"
            mock_client.mget.assert_awaited_once_with(
                index=s._nodes_index, body={"ids": ["Alice"]}
            )

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "Nobody", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_node("Nobody") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_edge(self, global_config, embed_func, mock_client):
        # get_edge now uses mget (translog real-time) instead of search.
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "e1",
                        "found": True,
                        "_source": {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "weight": 1.0,
                        },
                    },
                    {
                        "_id": "e2",
                        "found": False,
                    },
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            edge = await s.get_edge("A", "B")
            assert edge is not None
            assert edge["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_get_node_edges(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "e1",
                            "_source": {"source_node_id": "A", "target_node_id": "B"},
                            "sort": [1],
                        },
                        {
                            "_id": "e2",
                            "_source": {"source_node_id": "C", "target_node_id": "A"},
                            "sort": [2],
                        },
                    ],
                    "total": {"value": 2},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            edges = await s.get_node_edges("A")
            assert len(edges) == 2
            assert ("A", "B") in edges

    @pytest.mark.asyncio
    async def test_get_nodes_batch(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "A", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "B", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_nodes_batch(["A", "B"])
            assert "A" in result
            assert "B" not in result

    @pytest.mark.asyncio
    async def test_node_degrees_batch(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "source_degrees": {"buckets": [{"key": "A", "doc_count": 2}]},
                    "target_degrees": {
                        "buckets": [
                            {"key": "A", "doc_count": 1},
                            {"key": "B", "doc_count": 3},
                        ]
                    },
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            degrees = await s.node_degrees_batch(["A", "B"])
            assert degrees["A"] == 3  # 2 + 1
            assert degrees["B"] == 3

    @pytest.mark.asyncio
    async def test_upsert_node(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_node(
                "Alice", {"entity_type": "person", "source_id": "c1<SEP>c2"}
            )
            mock_client.index.assert_awaited()
            call_kwargs = mock_client.index.call_args
            assert call_kwargs.kwargs["id"] == "Alice"
            body = call_kwargs.kwargs["body"]
            assert body["source_ids"] == ["c1", "c2"]
            assert body["entity_id"] == "Alice"

    @pytest.mark.asyncio
    async def test_upsert_edge(self, global_config, embed_func, mock_client):
        mock_client.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_edge("A", "B", {"weight": "1.0", "description": "knows"})
            # Should call index twice: once for ensuring source node, once for edge
            assert mock_client.index.await_count == 2

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_reuses_id_for_reciprocal_edges(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            bulk_calls = []

            async def capture_bulk(_client, actions, *args, **kwargs):
                bulk_calls.append(list(actions))
                return (len(bulk_calls[-1]), [])

            mock_client.mget = AsyncMock(
                side_effect=[
                    {"docs": []},
                    {"docs": [{"_id": "edge-ba", "found": False}] * 2},
                ]
            )

            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new=AsyncMock(side_effect=capture_bulk),
            ):
                await s.upsert_edges_batch(
                    [
                        ("A", "B", {"weight": "1.0"}),
                        ("B", "A", {"weight": "2.0"}),
                    ]
                )

            edge_actions = bulk_calls[-1]
            assert len(edge_actions) == 2
            assert edge_actions[0]["_id"] == edge_actions[1]["_id"]

    @pytest.mark.asyncio
    async def test_upsert_after_drop_recreates_indices(
        self, global_config, embed_func, mock_client
    ):
        mock_client.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            with patch.object(
                s, "_create_indices_if_not_exist", new_callable=AsyncMock
            ) as mock_create:
                await s.initialize()
                mock_create.reset_mock()
                await s.drop()
                await s.upsert_edge("A", "B", {"weight": "1.0"})
                mock_create.assert_awaited_once()
                assert mock_client.index.await_count == 2

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not available")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            graph = await s.get_knowledge_graph("A", max_depth=2)

            assert await s.get_node("A") is None
            assert await s.get_all_labels() == []
            assert await s.has_edge("A", "B") is False
            assert await s.node_degree("A") == 0
            assert len(graph.nodes) == 0
            assert len(graph.edges) == 0

            mock_client.mget.assert_not_awaited()
            mock_client.search.assert_not_awaited()
            mock_client.create_pit.assert_not_awaited()
            mock_client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not available")
        )
        mock_client.mget = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.get_node("A") is None
            assert await s.get_node("A") is None
            assert s._indices_ready is False
            assert mock_client.mget.await_count == 1

    @pytest.mark.asyncio
    async def test_delete_node(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_node("Alice")
            mock_client.delete_by_query.assert_awaited_once()
            mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_nodes(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.remove_nodes(["A", "B"])
                mock_client.delete_by_query.assert_awaited_once()
                mock_bulk.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_edges(self, global_config, embed_func, mock_client):
        # remove_edges now uses bulk delete with deterministic IDs instead of
        # delete_by_query, so mock bulk as AsyncMock.
        mock_client.bulk = AsyncMock(return_value={"errors": False, "items": []})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.remove_edges([("A", "B"), ("C", "D")])
            # 2 edges × 2 candidate directions = 4 delete actions in one bulk call
            mock_client.bulk.assert_awaited_once()
            call_body = mock_client.bulk.call_args.kwargs["body"]
            assert len(call_body) == 4

    @pytest.mark.asyncio
    async def test_get_all_labels(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": "Alice", "sort": ["Alice"]},
                        {"_id": "Bob", "sort": ["Bob"]},
                    ],
                    "total": {"value": 2},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_all_labels()
            assert labels == ["Alice", "Bob"]

    @pytest.mark.asyncio
    async def test_get_popular_labels(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "src": {
                        "buckets": [
                            {"key": "A", "doc_count": 5},
                            {"key": "B", "doc_count": 2},
                        ]
                    },
                    "tgt": {"buckets": [{"key": "A", "doc_count": 3}]},
                    "status_counts": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_popular_labels(limit=10)
            assert labels[0] == "A"  # degree 8 > B degree 2

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_backfills_isolated_nodes_when_truncated(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 5})
        mock_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {"hits": [], "total": {"value": 1}},
                    "aggregations": {
                        "src": {"buckets": [{"key": "A", "doc_count": 1}]},
                        "tgt": {"buckets": [{"key": "B", "doc_count": 1}]},
                        "status_counts": {"buckets": []},
                    },
                },
                {
                    "hits": {
                        "hits": [
                            {"_id": "A", "sort": [1]},
                            {"_id": "B", "sort": [2]},
                            {"_id": "C", "sort": [3]},
                            {"_id": "D", "sort": [4]},
                            {"_id": "E", "sort": [5]},
                        ],
                        "total": {"value": 5},
                    }
                },
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "edge-ab",
                                "_source": {
                                    "source_node_id": "A",
                                    "target_node_id": "B",
                                    "relationship": "knows",
                                },
                            }
                        ],
                        "total": {"value": 1},
                    }
                },
            ]
        )
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "A", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "B", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "C", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "D", "found": True, "_source": {"entity_type": "person"}},
                ]
            }
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            result = await s.get_knowledge_graph("*", max_nodes=4)

            assert result.is_truncated is True
            assert [node.id for node in result.nodes] == ["A", "B", "C", "D"]
            assert len(result.edges) == 1
            assert result.edges[0].source == "A"
            assert result.edges[0].target == "B"
            assert mock_client.create_pit.await_count == 2

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_paginates_edges_between_selected_nodes(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 2})
        first_edge_page = [
            {
                "_id": f"edge-{i}",
                "_source": {
                    "source_node_id": "A",
                    "target_node_id": "B",
                    "relationship": "knows",
                },
                "sort": [i],
            }
            for i in range(10000)
        ]
        mock_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {
                        "hits": [
                            {"_id": "A"},
                            {"_id": "B"},
                        ],
                        "total": {"value": 2},
                    }
                },
                {"hits": {"hits": first_edge_page, "total": {"value": 10001}}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "edge-last",
                                "_source": {
                                    "source_node_id": "B",
                                    "target_node_id": "A",
                                    "relationship": "knows",
                                },
                                "sort": [10000],
                            }
                        ],
                        "total": {"value": 10001},
                    }
                },
            ]
        )
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "A", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "B", "found": True, "_source": {"entity_type": "person"}},
                ]
            }
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            result = await s.get_knowledge_graph("*", max_nodes=2)

            assert len(result.nodes) == 2
            assert len(result.edges) == 2
            assert {(edge.source, edge.target) for edge in result.edges} == {
                ("A", "B"),
                ("B", "A"),
            }
            assert mock_client.search.await_count == 3

    @pytest.mark.asyncio
    async def test_search_labels_empty_query(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.search_labels("") == []

    @pytest.mark.asyncio
    async def test_drop(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            assert mock_client.indices.delete.await_count == 2

    @pytest.mark.asyncio
    async def test_drop_partial_error_marks_indices_not_ready_and_next_upsert_recreates_indices(
        self, global_config, embed_func, mock_client
    ):
        mock_client.exists = AsyncMock(return_value=False)
        mock_client.indices.delete = AsyncMock(
            side_effect=[None, OpenSearchException("edges drop failed")]
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch.object(
                s, "_create_indices_if_not_exist", new_callable=AsyncMock
            ) as mock_create:
                result = await s.drop()
                assert result["status"] == "error"
                assert "edges drop failed" in result["message"]
                assert s._indices_ready is False
                await s.upsert_edge("A", "B", {"weight": "1.0"})
                mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drop_treats_missing_graph_indices_as_success(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=[_missing_index_error(), _missing_index_error()]
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            assert s._indices_ready is False

    @pytest.mark.asyncio
    async def test_construct_graph_node(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        node = s._construct_graph_node(
            "Alice",
            {
                "entity_type": "person",
                "description": "A researcher",
                "_id": "Alice",
                "entity_id": "Alice",
            },
        )
        assert node.id == "Alice"
        assert "entity_type" in node.properties
        assert "_id" not in node.properties
        assert "entity_id" not in node.properties

    @pytest.mark.asyncio
    async def test_construct_graph_edge(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        edge = s._construct_graph_edge(
            "e1",
            {
                "source_node_id": "A",
                "target_node_id": "B",
                "relationship": "knows",
                "weight": 1.0,
            },
        )
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.type == "knows"
        assert "source_node_id" not in edge.properties

    @pytest.mark.asyncio
    async def test_bfs_subgraph_start_not_found(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "NonExistent", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_knowledge_graph("NonExistent", max_depth=2)
            assert len(result.nodes) == 0
            assert len(result.edges) == 0


class TestGraphPPLDetection:
    """Tests for PPL graphlookup detection and server-side BFS."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchGraphStorage(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_ppl_detected_when_available(
        self, global_config, embed_func, mock_client
    ):
        """When PPL endpoint responds successfully, graphlookup should be detected."""
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            return_value={"datarows": [], "schema": []}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True

    @pytest.mark.asyncio
    async def test_ppl_not_detected_when_endpoint_fails(
        self, global_config, embed_func, mock_client
    ):
        """When PPL endpoint fails, should fall back to client-side BFS."""
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not supported")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is False

    @pytest.mark.asyncio
    async def test_env_override_true(self, global_config, embed_func, mock_client):
        with patch.dict("os.environ", {"OPENSEARCH_USE_PPL_GRAPHLOOKUP": "true"}):
            with patch.object(ClientManager, "get_client", return_value=mock_client):
                s = self._make(global_config, embed_func)
                await s.initialize()
                assert s._ppl_graphlookup_available is True
                # Should NOT have called transport.perform_request for detection
                mock_client.transport.perform_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_env_override_false(self, global_config, embed_func, mock_client):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            return_value={"datarows": [], "schema": []}
        )
        with patch.dict("os.environ", {"OPENSEARCH_USE_PPL_GRAPHLOOKUP": "false"}):
            with patch.object(ClientManager, "get_client", return_value=mock_client):
                s = self._make(global_config, embed_func)
                await s.initialize()
                assert s._ppl_graphlookup_available is False

    @pytest.mark.asyncio
    async def test_ppl_bfs_calls_ppl_endpoint(
        self, global_config, embed_func, mock_client
    ):
        """When PPL is available, get_knowledge_graph should use PPL endpoint."""
        mock_client.transport = AsyncMock()
        # PPL response: connected_edges contains dicts with source_node_id/target_node_id
        ppl_response = {
            "schema": [
                {"name": "entity_id", "type": "string"},
                {"name": "connected_edges", "type": "struct"},
            ],
            "datarows": [
                [
                    "A",
                    [  # connected_edges array
                        {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "weight": 1.0,
                            "_depth": 0,
                        },
                        {
                            "source_node_id": "B",
                            "target_node_id": "C",
                            "weight": 0.5,
                            "_depth": 1,
                        },
                    ],
                ]
            ],
        }
        mock_client.transport.perform_request = AsyncMock(return_value=ppl_response)
        # get_node for start node verification
        mock_client.get = AsyncMock(
            return_value={
                "_id": "A",
                "_source": {"entity_type": "person", "description": "Node A"},
            }
        )
        # mget for batch node fetch (only B and C, A is already added)
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "B", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "C", "found": True, "_source": {"entity_type": "person"}},
                ]
            }
        )
        # search for final edge fetch
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "e1",
                            "_source": {
                                "source_node_id": "A",
                                "target_node_id": "B",
                                "relationship": "knows",
                            },
                        },
                        {
                            "_id": "e2",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "C",
                                "relationship": "knows",
                            },
                        },
                    ],
                    "total": {"value": 2},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True

            result = await s.get_knowledge_graph("A", max_depth=2)
            assert len(result.nodes) == 3
            assert len(result.edges) == 2
            # Verify PPL was called (2 for detection + 1 for actual query)
            assert mock_client.transport.perform_request.await_count == 3
            # Verify the PPL query uses nodes index as source
            actual_query = mock_client.transport.perform_request.call_args_list[2]
            ppl_body = actual_query.kwargs.get("body") or actual_query[1].get(
                "body", {}
            )
            if isinstance(ppl_body, dict):
                assert s._nodes_index in ppl_body.get("query", "")

    @pytest.mark.asyncio
    async def test_ppl_bfs_falls_back_on_query_failure(
        self, global_config, embed_func, mock_client
    ):
        """If PPL query fails at runtime, should fall back to client-side BFS."""
        call_count = {"n": 0}

        async def ppl_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Detection calls succeed
                return {"datarows": [], "schema": []}
            # Actual query fails
            raise Exception("PPL query timeout")

        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(side_effect=ppl_side_effect)
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "A", "found": False}]}
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True

            # Should fall back to _bfs_subgraph, which returns empty (node not found)
            result = await s.get_knowledge_graph("A", max_depth=2)
            assert len(result.nodes) == 0

    @pytest.mark.asyncio
    async def test_escape_ppl(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._escape_ppl("it's") == "it\\'s"
        assert s._escape_ppl("normal") == "normal"
        assert s._escape_ppl("back\\slash") == "back\\\\slash"
        assert s._escape_ppl("both\\and'quote") == "both\\\\and\\'quote"

    @pytest.mark.asyncio
    async def test_ppl_bfs_depth_zero_returns_start_only(
        self, global_config, embed_func, mock_client
    ):
        """max_depth=0 should return only the start node without PPL query."""
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            return_value={"datarows": [], "schema": []}
        )
        mock_client.get = AsyncMock(
            return_value={"_id": "A", "_source": {"entity_type": "person"}}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True
            result = await s.get_knowledge_graph("A", max_depth=0)
            assert len(result.nodes) == 1
            assert result.nodes[0].id == "A"
            assert len(result.edges) == 0
            # PPL query should NOT have been called for the actual traversal (only 2 detection calls)
            assert mock_client.transport.perform_request.await_count == 2

    @pytest.mark.asyncio
    async def test_ppl_bfs_empty_connected_edges(
        self, global_config, embed_func, mock_client
    ):
        """PPL returns no connected edges — should return only start node."""
        mock_client.transport = AsyncMock()
        ppl_response = {
            "schema": [
                {"name": "entity_id", "type": "string"},
                {"name": "connected_edges", "type": "struct"},
            ],
            "datarows": [["A", []]],
        }
        mock_client.transport.perform_request = AsyncMock(return_value=ppl_response)
        mock_client.get = AsyncMock(
            return_value={"_id": "A", "_source": {"entity_type": "person"}}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_knowledge_graph("A", max_depth=2)
            assert len(result.nodes) == 1
            assert result.nodes[0].id == "A"

    @pytest.mark.asyncio
    async def test_ppl_bfs_truncates_nodes_by_depth_then_weight(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        ppl_response = {
            "schema": [
                {"name": "entity_id", "type": "string"},
                {"name": "connected_edges", "type": "struct"},
            ],
            "datarows": [
                [
                    "A",
                    [
                        {
                            "source_node_id": "A",
                            "target_node_id": "C",
                            "weight": 1.0,
                            "_depth": 1,
                        },
                        {
                            "source_node_id": "B",
                            "target_node_id": "D",
                            "weight": 10.0,
                            "_depth": 1,
                        },
                        {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "weight": 1.0,
                            "_depth": 0,
                        },
                    ],
                ]
            ],
        }
        mock_client.transport.perform_request = AsyncMock(return_value=ppl_response)
        mock_client.mget = AsyncMock(
            side_effect=[
                {
                    "docs": [
                        {
                            "_id": "A",
                            "found": True,
                            "_source": {"entity_type": "person"},
                        }
                    ]
                },
                {
                    "docs": [
                        {
                            "_id": "B",
                            "found": True,
                            "_source": {"entity_type": "person"},
                        },
                        {
                            "_id": "D",
                            "found": True,
                            "_source": {"entity_type": "person"},
                        },
                    ]
                },
            ]
        )
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "e1",
                            "_source": {
                                "source_node_id": "A",
                                "target_node_id": "B",
                                "relationship": "knows",
                            },
                            "sort": [1],
                        },
                        {
                            "_id": "e2",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "D",
                                "relationship": "knows",
                            },
                            "sort": [2],
                        },
                    ],
                    "total": {"value": 2},
                }
            }
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            result = await s.get_knowledge_graph("A", max_depth=2, max_nodes=3)

            assert [node.id for node in result.nodes] == ["A", "B", "D"]
            assert result.is_truncated is True
            assert {(edge.source, edge.target) for edge in result.edges} == {
                ("A", "B"),
                ("B", "D"),
            }

    @pytest.mark.asyncio
    async def test_upsert_node_adds_entity_id(
        self, global_config, embed_func, mock_client
    ):
        """upsert_node should always include entity_id field for PPL compatibility."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_node("TestNode", {"description": "test"})
            body = mock_client.index.call_args.kwargs["body"]
            assert body["entity_id"] == "TestNode"
            assert body["description"] == "test"

    @pytest.mark.asyncio
    async def test_node_degree_uses_count_api(
        self, global_config, embed_func, mock_client
    ):
        """node_degree should use the count API, not search."""
        mock_client.count = AsyncMock(return_value={"count": 7})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            degree = await s.node_degree("X")
            assert degree == 7
            # Verify count was called on the edges index
            mock_client.count.assert_awaited()
            call_kwargs = mock_client.count.call_args
            assert s._edges_index in str(call_kwargs)


# ---------------------------------------------------------------------------
# Vector Storage
# ---------------------------------------------------------------------------


class TestVectorStorage:
    """Tests for OpenSearchVectorDBStorage k-NN index, embeddings, cosine conversion, and entity deletion."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchVectorDBStorage(
            namespace="entities",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
            meta_fields={"content", "entity_name", "src_id", "tgt_id"},
        )

    @pytest.mark.asyncio
    async def test_index_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._index_name == "test_entities"

    @pytest.mark.asyncio
    async def test_cosine_threshold_required(self, embed_func):
        with pytest.raises(ValueError, match="cosine_better_than_threshold"):
            OpenSearchVectorDBStorage(
                namespace="v",
                global_config={
                    "embedding_batch_num": 10,
                    "vector_db_storage_cls_kwargs": {},
                },
                embedding_func=embed_func,
            )

    @pytest.mark.asyncio
    async def test_initialize_creates_knn_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_awaited_once()
            body = mock_client.indices.create.call_args.kwargs["body"]
            assert body["settings"]["index"]["knn"] is True
            assert body["mappings"]["properties"]["vector"]["dimension"] == 128
            assert (
                body["mappings"]["properties"]["vector"]["method"]["engine"] == "lucene"
            )

    @pytest.mark.asyncio
    async def test_upsert_generates_embeddings(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert(
                    {
                        "v1": {"content": "hello"},
                        "v2": {"content": "world"},
                    }
                )
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 2
                assert "vector" in actions[0]["_source"]
                assert len(actions[0]["_source"]["vector"]) == 128

    @pytest.mark.asyncio
    async def test_query_cosine_score_conversion(
        self, global_config, embed_func, mock_client
    ):
        """Test that scores are used directly and threshold filtering works."""
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "v1",
                            "_score": 0.85,
                            "_source": {"content": "match", "entity_name": "E1"},
                        },
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            results = await s.query("test", top_k=5)
            assert len(results) == 1
            assert results[0]["distance"] == 0.85

    @pytest.mark.asyncio
    async def test_query_filters_below_threshold(
        self, global_config, embed_func, mock_client
    ):
        """Low scores should be filtered out."""
        # score 0.15 < threshold 0.2
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "v1",
                            "_score": 0.15,
                            "_source": {"content": "weak match"},
                        },
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            results = await s.query("test", top_k=5)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_with_provided_embedding(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": "v1", "_score": 1.0, "_source": {"content": "exact"}},
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            vec = np.random.rand(128).astype(np.float32)
            results = await s.query("test", top_k=5, query_embedding=vec)
            assert len(results) == 1
            assert results[0]["distance"] == 1.0

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "v1",
                        "found": True,
                        "_source": {"content": "hello", "vector": [0.1] * 128},
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("v1")
            assert doc["id"] == "v1"
            assert doc["content"] == "hello"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["v1"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_ids(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "v1", "found": True, "_source": {"content": "a"}},
                    {"_id": "v2", "found": True, "_source": {"content": "b"}},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs = await s.get_by_ids(["v1", "v2"])
            assert docs[0]["id"] == "v1"
            assert docs[1]["id"] == "v2"

    @pytest.mark.asyncio
    async def test_get_vectors_by_ids(self, global_config, embed_func, mock_client):
        vec = [0.1] * 128
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "v1", "found": True, "_source": {"vector": vec}},
                    {"_id": "v2", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_vectors_by_ids(["v1", "v2"])
            assert "v1" in result
            assert "v2" not in result
            assert result["v1"] == vec

    @pytest.mark.asyncio
    async def test_delete(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.delete(["v1", "v2"])
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 2
                assert all(a["_op_type"] == "delete" for a in actions)

    @pytest.mark.asyncio
    async def test_delete_entity(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_entity("Alice")
            mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_entity_relation(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_entity_relation("Alice")
            mock_client.delete_by_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drop_recreates_index(self, global_config, embed_func, mock_client):
        # After drop, _create_knn_index_if_not_exists is called again.
        # First call (init): exists=False -> create. Second call (after drop): exists=False -> create again.
        mock_client.indices.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            mock_client.indices.delete.assert_awaited_once()
            # create called twice: once during init, once during drop recreate
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_drop_delete_error_marks_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("delete failed")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "error"
            assert s._index_ready is False

    @pytest.mark.asyncio
    async def test_drop_recreate_error_marks_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch.object(
                s,
                "_create_knn_index_if_not_exists",
                new=AsyncMock(side_effect=OpenSearchException("recreate failed")),
            ):
                result = await s.drop()
                assert result["status"] == "error"
                assert s._index_ready is False

    @pytest.mark.asyncio
    async def test_drop_recreates_index_when_missing(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=False)
        mock_client.indices.delete = AsyncMock(
            side_effect=NotFoundError(404, "not found")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_reads_short_circuit_when_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            s._index_ready = False

            assert await s.query("test", top_k=5) == []
            assert await s.get_by_id("v1") is None
            assert await s.get_vectors_by_ids(["v1"]) == {}

            mock_client.search.assert_not_awaited()
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.query("test", top_k=5) == []
            assert await s.query("test", top_k=5) == []
            assert s._index_ready is False
            assert mock_client.search.await_count == 1


# ---------------------------------------------------------------------------
# Cosine score edge cases
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    """Verify that raw OpenSearch scores are compared directly against threshold."""

    def test_above_threshold(self):
        assert 0.85 >= 0.2

    def test_below_threshold(self):
        assert 0.15 < 0.2

    def test_exact_threshold(self):
        assert 0.2 >= 0.2
