"""Unit tests for QdrantVectorDBStorage's deferred-embedding flush pipeline.

All tests use mocks — no running Qdrant instance required.
Mirrors the structure of tests/kg/opensearch_impl/test_opensearch_storage.py's
TestVectorStorageBatching to keep behaviour aligned across backends.
"""

import asyncio
import os

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required for Qdrant storage tests",
)

from lightrag.kg.qdrant_impl import (  # noqa: E402
    QdrantVectorDBStorage,
    compute_mdhash_id_for_qdrant,
)

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

    with patch("lightrag.kg.qdrant_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(
    embed_func,
    *,
    namespace="entities",
    workspace="test_ws",
    meta_fields=None,
):
    if meta_fields is None:
        meta_fields = {"content", "entity_name", "src_id", "tgt_id"}

    # Bypass real initialization paths (e.g. model suffix generation),
    # mirroring the existing pattern in test_qdrant_upsert_batching.py.
    storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
    storage.workspace = workspace
    storage.namespace = namespace
    storage.effective_workspace = workspace
    storage.model_suffix = "mock"
    storage.final_namespace = f"lightrag_vdb_{namespace}_mock"
    storage.meta_fields = meta_fields
    storage.embedding_func = embed_func
    storage._max_batch_size = 10
    storage._max_upsert_payload_bytes = 16 * 1024 * 1024
    storage._max_upsert_points_per_batch = 128
    storage._max_delete_points_per_batch = 1000
    storage._pending_vector_docs = {}
    storage._pending_vector_deletes = set()

    storage._client = MagicMock()
    storage._client.upsert = MagicMock()
    storage._client.delete = MagicMock()
    # drop() looks up a legacy collection to clear its workspace points;
    # default to "no legacy collection" so unrelated tests are unaffected.
    storage._client.collection_exists = MagicMock(return_value=False)
    storage._client.retrieve = MagicMock(return_value=[])
    storage._client.scroll = MagicMock(return_value=([], None))

    from lightrag.kg.qdrant_impl import get_namespace_lock

    storage._flush_lock = get_namespace_lock(
        namespace=storage.final_namespace, workspace=storage.effective_workspace
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
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_done_callback_triggers_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    s._client.upsert.assert_called_once()
    kwargs = s._client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == s.final_namespace
    points = kwargs["points"]
    assert len(points) == 2
    expected_ids = {
        compute_mdhash_id_for_qdrant("v1", prefix=s.effective_workspace),
        compute_mdhash_id_for_qdrant("v2", prefix=s.effective_workspace),
    }
    assert {p.id for p in points} == expected_ids
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
    s._client.upsert.assert_called_once()


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
    s._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_flush_failure_keeps_buffer_no_double_embed_on_retry():
    embed = CountingEmbeddingFunc(fail_times=1)
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="embedding failed"):
        await s.index_done_callback()

    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is None
    s._client.upsert.assert_not_called()

    await s.index_done_callback()
    assert embed.call_count == 2
    s._client.upsert.assert_called_once()
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_server_upsert_failure_keeps_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.upsert.side_effect = RuntimeError("qdrant down")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="qdrant down"):
        await s.index_done_callback()

    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is not None

    s._client.upsert.side_effect = None
    await s.index_done_callback()
    assert embed.call_count == 1


@pytest.mark.asyncio
async def test_finalize_raises_when_buffer_unflushed():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.upsert.side_effect = RuntimeError("transient qdrant error")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="finalize.*flush raised"):
        await s.finalize()
    assert "v1" in s._pending_vector_docs


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
    s._client.upsert.assert_called_once()
    s._client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_then_delete_same_id_keeps_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v1"])
    assert "v1" not in s._pending_vector_docs
    assert "v1" in s._pending_vector_deletes

    await s.index_done_callback()
    s._client.upsert.assert_not_called()
    s._client.delete.assert_called_once()
    qdrant_delete_ids = s._client.delete.call_args.kwargs["points_selector"].points
    assert qdrant_delete_ids == [
        compute_mdhash_id_for_qdrant("v1", prefix=s.effective_workspace)
    ]


@pytest.mark.asyncio
async def test_delete_entity_relation_raises_on_server_failure():
    """scroll-then-delete pattern: server-side failure must bubble up."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    fake_point = MagicMock()
    fake_point.id = "qid1"
    s._client.scroll.return_value = ([fake_point], None)
    s._client.delete.side_effect = RuntimeError("qdrant delete failed")

    with pytest.raises(RuntimeError, match="qdrant delete failed"):
        await s.delete_entity_relation("X")


@pytest.mark.asyncio
async def test_delete_entity_relation_prunes_pending_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert(
        {
            "rel-A-B": {"content": "A→B", "src_id": "A", "tgt_id": "B"},
            "rel-C-D": {"content": "C→D", "src_id": "C", "tgt_id": "D"},
        }
    )
    s._client.scroll.return_value = ([], None)

    await s.delete_entity_relation("A")
    assert "rel-A-B" not in s._pending_vector_docs
    assert "rel-C-D" in s._pending_vector_docs


@pytest.mark.asyncio
async def test_get_by_id_reads_pending_buffer_without_vector():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello", "entity_name": "E1"}})
    doc = await s.get_by_id("v1")
    assert doc is not None
    assert doc.get("entity_name") == "E1"
    assert "vector" not in doc
    s._client.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_pending_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.delete(["v1"])
    assert await s.get_by_id("v1") is None
    s._client.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_flush_uses_build_upsert_batches_for_multiple_batches():
    """When the points exceed the per-batch point limit, flush calls
    `_client.upsert` multiple times — and a mid-batch failure keeps the
    entire buffer for retry.
    """
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._max_upsert_points_per_batch = 2  # force batching

    await s.upsert({f"v{i}": {"content": f"c{i}"} for i in range(5)})
    s._client.upsert.side_effect = [None, RuntimeError("batch 2 failed"), None]

    with pytest.raises(RuntimeError, match="batch 2 failed"):
        await s.index_done_callback()

    # Stopped at batch 2, total 2 calls so far.
    assert s._client.upsert.call_count == 2
    # Buffer preserved.
    assert len(s._pending_vector_docs) == 5


@pytest.mark.asyncio
async def test_env_workspace_override_shares_flush_lock(patch_namespace_lock):
    cache = patch_namespace_lock
    embed = CountingEmbeddingFunc()

    with patch.dict(os.environ, {"QDRANT_WORKSPACE": "shared_ws"}, clear=False):
        # Two callers passing different `workspace` would both be redirected
        # by the env override to "shared_ws". Since `_make_storage` skips
        # __post_init__, simulate the override directly:
        a = _make_storage(embed, workspace="shared_ws")
        b = _make_storage(embed, workspace="shared_ws")
        assert a.final_namespace == b.final_namespace
        assert a.effective_workspace == b.effective_workspace == "shared_ws"
        assert a._flush_lock is b._flush_lock
        assert len([k for k in cache if k == (a.final_namespace, "shared_ws")]) == 1


@pytest.mark.asyncio
async def test_distinct_workspaces_in_same_collection_get_independent_locks(
    patch_namespace_lock,
):
    """Same final_namespace but different workspaces → independent locks."""
    embed = CountingEmbeddingFunc()
    a = _make_storage(embed, workspace="ws_a")
    b = _make_storage(embed, workspace="ws_b")
    # final_namespace depends on namespace only (model suffix is mocked),
    # so the two share it, but workspaces differ → different locks.
    assert a.final_namespace == b.final_namespace
    assert a.effective_workspace != b.effective_workspace
    assert a._flush_lock is not b._flush_lock


@pytest.mark.asyncio
async def test_drop_clears_pending_buffers():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v2"])
    assert s._pending_vector_docs and s._pending_vector_deletes

    result = await s.drop()
    assert result["status"] == "success"
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_drop_clears_workspace_points_from_workspace_tagged_legacy():
    """drop() removes only this workspace's points from a workspace-tagged legacy
    collection, so the next startup does not re-migrate the cleared data back."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    legacy_collection = f"lightrag_vdb_{s.namespace}"
    s._client.collection_exists = MagicMock(
        side_effect=lambda name: name == legacy_collection
    )
    # Legacy is workspace-tagged: workspace_id present in the payload schema.
    legacy_info = MagicMock()
    legacy_info.payload_schema = {"workspace_id": MagicMock()}
    s._client.get_collection = MagicMock(return_value=legacy_info)

    result = await s.drop()
    assert result["status"] == "success"

    deleted_collections = [
        call.kwargs.get("collection_name") for call in s._client.delete.call_args_list
    ]
    # Both the active suffixed collection and the legacy collection are cleared
    # via a workspace-filtered delete; the legacy collection itself is kept.
    assert s.final_namespace in deleted_collections
    assert legacy_collection in deleted_collections
    s._client.delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_drop_drops_untagged_legacy_collection():
    """For an untagged (pre-isolation) legacy collection, setup_collection would
    migrate ALL of its points back with no workspace filter, so a filtered delete
    misses them. drop() must drop the whole legacy collection instead."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    legacy_collection = f"lightrag_vdb_{s.namespace}"
    s._client.collection_exists = MagicMock(
        side_effect=lambda name: name == legacy_collection
    )
    # Legacy is untagged: no workspace_id in schema and none in sampled payloads.
    legacy_info = MagicMock()
    legacy_info.payload_schema = {}
    s._client.get_collection = MagicMock(return_value=legacy_info)
    s._client.scroll = MagicMock(return_value=([], None))

    result = await s.drop()
    assert result["status"] == "success"

    # The whole untagged legacy collection is dropped (not a filtered delete).
    s._client.delete_collection.assert_called_once_with(
        collection_name=legacy_collection
    )
    filtered_deletes = [
        call.kwargs.get("collection_name") for call in s._client.delete.call_args_list
    ]
    assert legacy_collection not in filtered_deletes


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_index_ops_clears_buffers():
    """On an internal-error abort the pipeline calls drop_pending_index_ops to
    discard buffered upserts/deletes without flushing them (PR #3187)."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    await s.upsert({"v1": {"content": "x"}, "v2": {"content": "y"}})
    s._pending_vector_deletes.add("old-id")
    assert s._pending_vector_docs
    await s.drop_pending_index_ops()
    assert not s._pending_vector_docs
    assert not s._pending_vector_deletes
