"""Unit tests for MongoVectorDBStorage's deferred-embedding flush pipeline.

All tests use mocks — no running MongoDB instance required.
Mirrors the structure of tests/kg/opensearch_impl/test_opensearch_storage.py's
TestVectorStorageBatching to keep behaviour aligned across backends.
"""

import asyncio
import os

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from pymongo import UpdateOne, DeleteOne  # type: ignore

from lightrag.kg.mongo_impl import MongoVectorDBStorage

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class MockEmbeddingFunc:
    def __init__(self, dim=8):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


class CountingEmbeddingFunc(MockEmbeddingFunc):
    def __init__(self, dim=8, fail_times=0):
        super().__init__(dim=dim)
        self.fail_times = fail_times
        self.call_count = 0
        self.batches: list[list[str]] = []
        self.texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        batch = list(texts)
        self.batches.append(batch)
        self.texts.extend(batch)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("embedding failed")
        return await super().__call__(texts, **kwargs)


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return list(self._docs)


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    """Cache real asyncio.Locks per (namespace, workspace) for shared semantics."""
    cache: dict[tuple[str, str | None], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.mongo_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(
    embed_func,
    *,
    namespace="entities",
    workspace="test",
    meta_fields=None,
):
    if meta_fields is None:
        meta_fields = {"content", "entity_name", "src_id", "tgt_id"}
    storage = MongoVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embed_func,
        meta_fields=meta_fields,
    )
    # Wire a fake AsyncCollection (the only Mongo surface our code touches).
    storage._data = MagicMock()
    storage._data.bulk_write = AsyncMock()
    storage._data.delete_many = AsyncMock(
        return_value=MagicMock(deleted_count=0)
    )
    storage._data.find_one = AsyncMock(return_value=None)
    storage._data.find = MagicMock(return_value=_AsyncCursor([]))
    storage.db = MagicMock()  # non-None so finalize releases it

    from lightrag.kg.mongo_impl import get_namespace_lock

    storage._flush_lock = get_namespace_lock(
        namespace=storage.final_namespace, workspace=""
    )
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_buffers_without_embedding():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})

    assert embed.call_count == 0
    assert set(s._pending_vector_docs.keys()) == {"v1", "v2"}
    assert s._pending_vector_docs["v1"].vector is None
    s._data.bulk_write.assert_not_called()


@pytest.mark.asyncio
async def test_index_done_callback_triggers_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    s._data.bulk_write.assert_called_once()
    ops, kwargs = s._data.bulk_write.call_args.args[0], s._data.bulk_write.call_args.kwargs
    assert kwargs.get("ordered") is False
    assert all(isinstance(op, UpdateOne) for op in ops)
    assert len(ops) == 2
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_repeated_upsert_same_id_embeds_once():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "first"}})
    await s.upsert({"v1": {"content": "second"}})
    await s.upsert({"v1": {"content": "third"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    assert embed.texts == ["third"]
    s._data.bulk_write.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_embeddings_respect_batch_size():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._max_batch_size = 2

    await s.upsert({f"v{i}": {"content": f"doc {i}"} for i in range(5)})
    await s.index_done_callback()

    assert embed.call_count == 3
    assert [len(b) for b in embed.batches] == [2, 2, 1]


@pytest.mark.asyncio
async def test_get_vectors_by_ids_lazy_embed_then_reuse_in_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})

    vectors = await s.get_vectors_by_ids(["v1"])
    assert "v1" in vectors
    assert embed.call_count == 1
    assert s._pending_vector_docs["v1"].vector is not None

    await s.index_done_callback()
    assert embed.call_count == 1
    s._data.bulk_write.assert_called_once()


@pytest.mark.asyncio
async def test_flush_failure_keeps_buffer_no_double_embed_on_retry():
    embed = CountingEmbeddingFunc(fail_times=1)
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="embedding failed"):
        await s.index_done_callback()

    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is None
    s._data.bulk_write.assert_not_called()

    await s.index_done_callback()
    assert embed.call_count == 2
    s._data.bulk_write.assert_called_once()
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_server_write_failure_keeps_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._data.bulk_write.side_effect = RuntimeError("mongo down")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="mongo down"):
        await s.index_done_callback()

    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is not None

    s._data.bulk_write.side_effect = None
    await s.index_done_callback()
    assert embed.call_count == 1
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_finalize_raises_when_buffer_unflushed_and_still_releases_client():
    """finalize() must release the Mongo client even when the flush fails."""
    from lightrag.kg.mongo_impl import ClientManager

    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._data.bulk_write.side_effect = RuntimeError("mongo down")

    await s.upsert({"v1": {"content": "hello"}})

    with patch.object(ClientManager, "release_client", new=AsyncMock()) as rel:
        with pytest.raises(RuntimeError, match="finalize.*flush raised"):
            await s.finalize()
        rel.assert_awaited_once()

    # Client references cleared so a second finalize doesn't release twice.
    assert s.db is None


@pytest.mark.asyncio
async def test_delete_then_upsert_same_id_keeps_upsert():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.delete(["v1"])
    assert "v1" in s._pending_vector_deletes

    await s.upsert({"v1": {"content": "hello"}})
    assert "v1" in s._pending_vector_docs
    assert "v1" not in s._pending_vector_deletes

    await s.index_done_callback()
    ops = s._data.bulk_write.call_args.args[0]
    assert all(isinstance(op, UpdateOne) for op in ops)


@pytest.mark.asyncio
async def test_upsert_then_delete_same_id_keeps_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v1"])

    assert "v1" not in s._pending_vector_docs
    assert "v1" in s._pending_vector_deletes

    await s.index_done_callback()
    ops = s._data.bulk_write.call_args.args[0]
    assert len(ops) == 1
    assert isinstance(ops[0], DeleteOne)


@pytest.mark.asyncio
async def test_bulk_write_uses_update_one_and_delete_one_mix():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"u1": {"content": "u1"}, "u2": {"content": "u2"}})
    await s.delete(["d1", "d2"])

    await s.index_done_callback()
    ops = s._data.bulk_write.call_args.args[0]
    op_types = {type(op).__name__ for op in ops}
    assert op_types == {"UpdateOne", "DeleteOne"}
    assert sum(isinstance(op, UpdateOne) for op in ops) == 2
    assert sum(isinstance(op, DeleteOne) for op in ops) == 2


@pytest.mark.asyncio
async def test_delete_entity_relation_raises_on_server_failure():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._data.find = MagicMock(
        return_value=_AsyncCursor([{"_id": "rel1"}, {"_id": "rel2"}])
    )
    s._data.delete_many = AsyncMock(side_effect=RuntimeError("mongo delete failed"))

    with pytest.raises(RuntimeError, match="mongo delete failed"):
        await s.delete_entity_relation("X")


@pytest.mark.asyncio
async def test_delete_entity_relation_prunes_pending_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert(
        {
            "rel-A-B": {"content": "A → B", "src_id": "A", "tgt_id": "B"},
            "rel-C-D": {"content": "C → D", "src_id": "C", "tgt_id": "D"},
        }
    )
    s._data.find = MagicMock(return_value=_AsyncCursor([]))

    await s.delete_entity_relation("A")
    assert "rel-A-B" not in s._pending_vector_docs
    assert "rel-C-D" in s._pending_vector_docs


@pytest.mark.asyncio
async def test_get_by_id_buffer_excludes_vector():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello", "entity_name": "E1"}})
    doc = await s.get_by_id("v1")
    assert doc is not None
    assert doc["id"] == "v1"
    assert doc.get("entity_name") == "E1"
    assert "vector" not in doc
    s._data.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_get_by_id_fallback_projects_out_vector():
    """Server-side find_one must request projection={'vector': 0}."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._data.find_one = AsyncMock(
        return_value={"_id": "v9", "entity_name": "X", "created_at": 0}
    )

    doc = await s.get_by_id("v9")
    assert doc is not None
    assert "vector" not in doc
    args, kwargs = s._data.find_one.call_args.args, s._data.find_one.call_args.kwargs
    # projection is positional arg #2 in Mongo's API.
    projection = args[1] if len(args) > 1 else kwargs.get("projection")
    assert projection == {"vector": 0}


@pytest.mark.asyncio
async def test_get_by_ids_fallback_projects_out_vector():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    s._data.find = MagicMock(
        return_value=_AsyncCursor(
            [{"_id": "a", "entity_name": "A"}, {"_id": "b", "entity_name": "B"}]
        )
    )
    docs = await s.get_by_ids(["a", "b"])
    assert len(docs) == 2
    assert all("vector" not in d for d in docs if d)
    args, kwargs = s._data.find.call_args.args, s._data.find.call_args.kwargs
    projection = args[1] if len(args) > 1 else kwargs.get("projection")
    assert projection == {"vector": 0}


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_pending_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.delete(["v1"])
    assert await s.get_by_id("v1") is None
    s._data.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_env_workspace_override_shares_flush_lock(patch_namespace_lock):
    cache = patch_namespace_lock
    embed = CountingEmbeddingFunc()

    with patch.dict(os.environ, {"MONGODB_WORKSPACE": "shared_ws"}, clear=False):
        a = _make_storage(embed, workspace="caller_a")
        b = _make_storage(embed, workspace="caller_b")
        assert a.final_namespace == b.final_namespace == "shared_ws_entities"
        assert a._flush_lock is b._flush_lock
        assert len([k for k in cache if k[0] == "shared_ws_entities"]) == 1


@pytest.mark.asyncio
async def test_distinct_namespaces_get_independent_locks():
    embed = CountingEmbeddingFunc()
    a = _make_storage(embed, workspace="a")
    b = _make_storage(embed, workspace="b")
    assert a.final_namespace != b.final_namespace
    assert a._flush_lock is not b._flush_lock


@pytest.mark.asyncio
async def test_drop_clears_pending_buffers():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    with patch.object(s, "create_vector_index_if_not_exists", new=AsyncMock()):
        await s.upsert({"v1": {"content": "hello"}})
        await s.delete(["v2"])
        assert s._pending_vector_docs and s._pending_vector_deletes

        result = await s.drop()
        assert result["status"] == "success"
        assert s._pending_vector_docs == {}
        assert s._pending_vector_deletes == set()
