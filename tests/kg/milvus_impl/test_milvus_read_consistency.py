"""Point reads on MilvusVectorDBStorage must ask for Strong consistency.

Collections are created without a consistency level, so the server default --
Bounded -- applies to any request that does not name one. Under Bounded a read
issued right after a flush can miss the rows that flush just wrote, or still
return the rows it just deleted (reproduced live on Milvus 2.6.11 and 2.6.27,
with MinIO and with local object storage alike). Once the flush has emptied
the write buffer, read-your-writes rests on the level each read passes.

These tests pin the level on every ``client.query`` a point read issues, and
pin vector search staying on the collection default (an accepted residue, see
``MILVUS_POINT_READ_CONSISTENCY_LEVEL``). All mocked; no Milvus server needed.
"""

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lightrag.kg.milvus_impl import (
    MilvusVectorDBStorage,
)

pytestmark = pytest.mark.offline


class MockEmbeddingFunc:
    def __init__(self, dim: int = 8) -> None:
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture(autouse=True)
def patch_namespace_lock() -> Generator[None, None, None]:
    cache: dict[tuple[str, str], asyncio.Lock] = {}

    def factory(
        namespace: str, workspace: str | None = None, enable_logging: bool = False
    ) -> asyncio.Lock:
        return cache.setdefault((namespace, workspace or ""), asyncio.Lock())

    with patch("lightrag.kg.milvus_impl.get_namespace_lock", side_effect=factory):
        yield


def _make_storage(namespace: str = "relationships") -> MilvusVectorDBStorage:
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace="test",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=MockEmbeddingFunc(),
        meta_fields={"content", "src_id", "tgt_id"},
    )
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._client.query = MagicMock(return_value=[])
    storage._client.search = MagicMock(return_value=[[]])
    storage._client.delete = MagicMock(return_value={"delete_count": 0})
    storage._client.load_collection = MagicMock()
    storage._initialized = True
    return storage


def _assert_every_query_strong(storage: MilvusVectorDBStorage) -> None:
    calls = storage._client.query.call_args_list
    assert calls, "the read never reached client.query"
    levels = [call.kwargs.get("consistency_level") for call in calls]
    assert levels == ["Strong"] * len(calls)


@pytest.mark.asyncio
async def test_get_by_id_reads_strong() -> None:
    storage = _make_storage()
    await storage.get_by_id("rel-1")
    _assert_every_query_strong(storage)


@pytest.mark.asyncio
async def test_get_by_ids_reads_strong() -> None:
    storage = _make_storage()
    await storage.get_by_ids(["rel-1", "rel-2"])
    _assert_every_query_strong(storage)


@pytest.mark.asyncio
async def test_is_empty_reads_strong() -> None:
    storage = _make_storage()
    assert await storage.is_empty() is True
    _assert_every_query_strong(storage)


@pytest.mark.asyncio
async def test_get_exact_count_reads_strong() -> None:
    storage = _make_storage()
    storage._client.query.return_value = [{"count(*)": 3}]
    assert await storage.get_exact_count() == 3
    _assert_every_query_strong(storage)
    assert storage._client.query.call_args.kwargs["output_fields"] == ["count(*)"]


@pytest.mark.asyncio
async def test_strict_read_raises_where_get_by_ids_reports_nothing() -> None:
    storage = _make_storage()
    storage._client.query.side_effect = ConnectionError("milvus down")

    assert await storage.get_by_ids(["rel-1"]) == []
    with pytest.raises(ConnectionError):
        await storage.get_by_ids_strict(["rel-1"])


@pytest.mark.asyncio
async def test_delete_entity_relation_victim_lookup_reads_strong() -> None:
    storage = _make_storage()
    storage._client.query.return_value = [{"id": "rel-A-B"}]
    await storage.delete_entity_relation("A")
    _assert_every_query_strong(storage)
    deleted = [
        pk
        for call in storage._client.delete.call_args_list
        for pk in call.kwargs.get("pks", [])
    ]
    assert deleted == ["rel-A-B"]


@pytest.mark.asyncio
async def test_vector_search_stays_on_collection_default() -> None:
    storage = _make_storage()
    await storage.query("anything", top_k=3, query_embedding=[0.0] * 8)
    storage._client.search.assert_called_once()
    assert "consistency_level" not in storage._client.search.call_args.kwargs
