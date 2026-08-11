"""Offline tests for bounded Milvus ID-query responses."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag.kg.milvus_impl import MilvusVectorDBStorage

pytestmark = pytest.mark.offline


class MockEmbeddingFunc:
    def __init__(self, dim: int):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    cache: dict[tuple[str, str], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        return cache.setdefault(key, asyncio.Lock())

    with patch("lightrag.kg.milvus_impl.get_namespace_lock", side_effect=factory):
        yield


def _make_storage(*, dim: int) -> MilvusVectorDBStorage:
    storage = MilvusVectorDBStorage(
        namespace="chunks",
        workspace="test",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2,
            },
        },
        embedding_func=MockEmbeddingFunc(dim),
        meta_fields={"content"},
    )
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._initialized = True
    return storage


def _ids_from_filter(filter_expr: str) -> list[str]:
    return json.loads(filter_expr.removeprefix("id in "))


@pytest.mark.asyncio
async def test_high_dimension_vector_reads_stay_below_gateway_response_limit():
    storage = _make_storage(dim=3072)
    requested_ids = [f"v{i}" for i in range(600)]

    def query(*, filter, **kwargs):
        page_ids = _ids_from_filter(filter)
        if len(page_ids) > 200:
            raise RuntimeError("RESOURCE_EXHAUSTED: received message larger than max")
        return [
            {"id": doc_id, "vector": [float(i)]} for i, doc_id in enumerate(page_ids)
        ]

    storage._client.query.side_effect = query

    with patch(
        "lightrag.kg.milvus_impl._cooperative_yield", new_callable=AsyncMock
    ) as cooperative_yield:
        result = await storage.get_vectors_by_ids(requested_ids)

    assert set(result) == set(requested_ids)
    assert storage._client.query.call_count > 1
    assert cooperative_yield.await_count == storage._client.query.call_count - 1


def test_query_page_size_tracks_dimension_and_record_cap():
    small_vectors = _make_storage(dim=8)
    large_vectors = _make_storage(dim=3072)

    small_page = small_vectors._resolve_query_page_size(includes_vector=True)
    large_page = large_vectors._resolve_query_page_size(includes_vector=True)

    assert small_page == 256
    assert large_page < small_page

    with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 3):
        assert small_vectors._resolve_query_page_size(includes_vector=True) == 3
        assert small_vectors._resolve_query_page_size(includes_vector=False) == 3


@pytest.mark.asyncio
async def test_get_by_ids_preserves_order_and_skips_buffered_or_deleted_ids():
    storage = _make_storage(dim=8)
    await storage.upsert({"buffered": {"content": "pending"}})
    await storage.delete(["deleted"])

    server_ids: list[str] = []

    def query(*, filter, **kwargs):
        page_ids = _ids_from_filter(filter)
        server_ids.extend(page_ids)
        return [
            {"id": doc_id, "content": f"content:{doc_id}"}
            for doc_id in reversed(page_ids)
        ]

    storage._client.query.side_effect = query
    requested_ids = ["persisted-2", "buffered", "deleted", "persisted-1"]

    with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2):
        result = await storage.get_by_ids(requested_ids)

    assert [row and row["id"] for row in result] == [
        "persisted-2",
        "buffered",
        None,
        "persisted-1",
    ]
    assert result[1]["content"] == "pending"
    assert result[2] is None
    assert server_ids == ["persisted-2", "persisted-1"]
    storage._client.load_collection.assert_called_once_with(storage.final_namespace)


@pytest.mark.asyncio
async def test_query_paging_preserves_id_filter_escaping():
    storage = _make_storage(dim=8)
    requested_ids = ['vec"1', "vec\\2"]
    seen_pages: list[list[str]] = []

    def query(*, filter, **kwargs):
        page_ids = _ids_from_filter(filter)
        seen_pages.append(page_ids)
        return [{"id": doc_id, "vector": [1.0]} for doc_id in page_ids]

    storage._client.query.side_effect = query

    with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 1):
        result = await storage.get_vectors_by_ids(requested_ids)

    assert seen_pages == [['vec"1'], ["vec\\2"]]
    assert set(result) == set(requested_ids)


@pytest.mark.asyncio
async def test_failed_page_does_not_leak_partial_server_results():
    storage = _make_storage(dim=8)
    await storage.upsert({"buffered": {"content": "pending"}})
    storage._pending_vector_docs["buffered"].vector = [9.0]
    call_count = 0

    def query(*, filter, **kwargs):
        nonlocal call_count
        call_count += 1
        page_ids = _ids_from_filter(filter)
        if call_count == 2:
            raise RuntimeError("RESOURCE_EXHAUSTED")
        return [{"id": doc_id, "vector": [1.0]} for doc_id in page_ids]

    storage._client.query.side_effect = query

    with (
        patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2),
        patch("lightrag.kg.milvus_impl.logger.error") as log_error,
    ):
        result = await storage.get_vectors_by_ids(
            ["buffered", "persisted-1", "persisted-2", "persisted-3"]
        )

    assert result == {"buffered": [9.0]}
    assert call_count == 2
    assert "RESOURCE_EXHAUSTED" in log_error.call_args.args[0]
