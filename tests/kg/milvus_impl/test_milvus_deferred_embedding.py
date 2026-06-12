"""Unit tests for MilvusVectorDBStorage's deferred-embedding flush pipeline.

All tests use mocks — no running Milvus instance required.
Mirrors the structure of tests/kg/opensearch_impl/test_opensearch_storage.py's
TestVectorStorageBatching to keep behaviour aligned across backends.
"""

import asyncio
import os

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.milvus_impl import MILVUS_MAX_VARCHAR_BYTES, MilvusVectorDBStorage

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockEmbeddingFunc:
    """Mock embedding function that returns random vectors."""

    def __init__(self, dim=8):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


class CountingEmbeddingFunc(MockEmbeddingFunc):
    """Embedding test double that records calls and can fail a fixed number of times."""

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
    """Cache real asyncio.Locks per (namespace, workspace) for shared semantics.

    Two storage instances whose ``final_namespace`` matches must observe the
    same Lock instance — this fixture lets us assert that and also exercises
    real serialization between concurrent flush/upsert coroutines.
    """
    cache: dict[tuple[str, str | None], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.milvus_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(
    embed_func,
    *,
    namespace="entities",
    workspace="test",
    meta_fields=None,
):
    """Build a MilvusVectorDBStorage skipping `initialize()` (no real client)."""
    if meta_fields is None:
        meta_fields = {"content", "entity_name", "src_id", "tgt_id"}
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embed_func,
        meta_fields=meta_fields,
    )
    # Bypass real Milvus client; manually wire the bits initialize() would set.
    # The flush lock is already constructed in __post_init__ via the patched
    # get_namespace_lock factory, so no manual lock wiring is needed here.
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._client.upsert = MagicMock(return_value={"upsert_count": 0})
    storage._client.delete = MagicMock(return_value={"delete_count": 0})
    storage._client.query = MagicMock(return_value=[])
    storage._client.load_collection = MagicMock()
    storage._initialized = True
    return storage


# ---------------------------------------------------------------------------
# Tests: deferred embedding + batched flush
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_buffers_without_embedding():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})

    assert embed.call_count == 0
    assert set(s._pending_vector_docs.keys()) == {"v1", "v2"}
    assert s._pending_vector_docs["v1"].vector is None
    assert s._pending_vector_docs["v2"].vector is None
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_done_callback_triggers_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    s._client.upsert.assert_called_once()
    call_kwargs = s._client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == s.final_namespace
    upserted = call_kwargs["data"]
    assert {row["id"] for row in upserted} == {"v1", "v2"}
    assert all("vector" in row for row in upserted)
    # Buffers cleared after a successful flush.
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_upsert_truncates_oversized_content_for_payload_only():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, meta_fields={"content"})
    content = "x" * (MILVUS_MAX_VARCHAR_BYTES + 10)

    await s.upsert({"v1": {"content": content}})
    await s.index_done_callback()

    upserted = s._client.upsert.call_args.kwargs["data"][0]
    assert len(upserted["content"].encode("utf-8")) == MILVUS_MAX_VARCHAR_BYTES
    assert embed.texts == [content]
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_upsert_truncates_multibyte_content_on_character_boundary():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, meta_fields={"content"})
    content = "源" * (MILVUS_MAX_VARCHAR_BYTES // 3 + 10)

    await s.upsert({"v1": {"content": content}})
    await s.index_done_callback()

    upserted = s._client.upsert.call_args.kwargs["data"][0]
    assert len(upserted["content"].encode("utf-8")) <= MILVUS_MAX_VARCHAR_BYTES
    upserted["content"].encode("utf-8").decode("utf-8")
    assert embed.texts == [content]


@pytest.mark.asyncio
async def test_upsert_rejects_oversized_primary_id():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    long_id = "v" * 65

    with pytest.raises(ValueError, match="primary keys cannot be truncated"):
        await s.upsert({long_id: {"content": "hello"}})

    assert s._pending_vector_docs == {}
    assert embed.call_count == 0
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_rejects_oversized_entity_name():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, meta_fields={"content", "entity_name"})
    long_entity_name = "e" * 513

    with pytest.raises(ValueError, match="identity fields cannot be truncated"):
        await s.upsert({"ent-1": {"content": "hello", "entity_name": long_entity_name}})

    assert s._pending_vector_docs == {}
    assert embed.call_count == 0
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_rejects_oversized_relation_identity_fields():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, namespace="relationships")
    long_entity_id = "e" * 513

    with pytest.raises(ValueError, match="identity fields cannot be truncated"):
        await s.upsert(
            {
                "rel-1": {
                    "content": "hello",
                    "src_id": long_entity_id,
                    "tgt_id": "B",
                }
            }
        )

    assert s._pending_vector_docs == {}
    assert embed.call_count == 0
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_rejects_oversized_full_doc_id():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, namespace="chunks", meta_fields={"content", "full_doc_id"})
    long_doc_id = "d" * 65

    with pytest.raises(ValueError, match="identity fields cannot be truncated"):
        await s.upsert({"chunk-1": {"content": "hello", "full_doc_id": long_doc_id}})

    assert s._pending_vector_docs == {}
    assert embed.call_count == 0
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_truncates_oversized_source_id():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, meta_fields={"content", "source_id"})
    source_id = "s" * (MILVUS_MAX_VARCHAR_BYTES + 10)

    await s.upsert({"v1": {"content": "hello", "source_id": source_id}})
    await s.index_done_callback()

    upserted = s._client.upsert.call_args.kwargs["data"][0]
    assert len(upserted["source_id"].encode("utf-8")) == MILVUS_MAX_VARCHAR_BYTES


@pytest.mark.asyncio
async def test_upsert_truncates_source_id_on_separator_boundary():
    # source_id is a <SEP>-joined list of chunk ids; truncation must drop whole
    # ids at a separator boundary rather than leave a dangling partial id.
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, meta_fields={"content", "source_id"})
    # Each chunk id is comfortably sized; enough of them to overflow the limit.
    chunk_id = "chunk-" + "a" * 50
    chunk_ids = [chunk_id] * ((MILVUS_MAX_VARCHAR_BYTES // len(chunk_id)) + 5)
    source_id = "<SEP>".join(chunk_ids)
    assert len(source_id.encode("utf-8")) > MILVUS_MAX_VARCHAR_BYTES

    await s.upsert({"v1": {"content": "hello", "source_id": source_id}})
    await s.index_done_callback()

    upserted = s._client.upsert.call_args.kwargs["data"][0]
    stored = upserted["source_id"]
    assert len(stored.encode("utf-8")) <= MILVUS_MAX_VARCHAR_BYTES
    # No trailing partial id: every retained id is intact.
    assert all(part == chunk_id for part in stored.split("<SEP>"))


@pytest.mark.asyncio
async def test_upsert_truncates_oversized_single_source_id_without_separator():
    # A single id longer than the limit has no separator to back off to, so it
    # falls back to the raw byte cut instead of dropping the value entirely.
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, meta_fields={"content", "source_id"})
    source_id = "s" * (MILVUS_MAX_VARCHAR_BYTES + 10)

    await s.upsert({"v1": {"content": "hello", "source_id": source_id}})
    await s.index_done_callback()

    upserted = s._client.upsert.call_args.kwargs["data"][0]
    assert len(upserted["source_id"].encode("utf-8")) == MILVUS_MAX_VARCHAR_BYTES


@pytest.mark.asyncio
async def test_repeated_upsert_same_id_embeds_once():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "first"}})
    await s.upsert({"v1": {"content": "second"}})
    await s.upsert({"v1": {"content": "third"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    # Only the latest content survives and was embedded.
    assert embed.texts == ["third"]
    s._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_embeddings_respect_batch_size():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._max_batch_size = 2

    await s.upsert({f"v{i}": {"content": f"doc {i}"} for i in range(5)})
    await s.index_done_callback()

    # 5 docs / batch 2 → 3 batches → 3 embedding calls
    assert embed.call_count == 3
    assert [len(b) for b in embed.batches] == [2, 2, 1]


@pytest.mark.asyncio
async def test_get_vectors_by_ids_lazy_embed_then_reuse_in_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})

    vectors = await s.get_vectors_by_ids(["v1"])
    assert "v1" in vectors
    assert embed.call_count == 1  # lazy embed inside get_vectors_by_ids
    # The lazy-embedded vector is cached on the pending doc.
    assert s._pending_vector_docs["v1"].vector is not None

    await s.index_done_callback()
    # Flush reused the cached vector — no extra embedding call.
    assert embed.call_count == 1
    s._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_flush_failure_keeps_buffer_and_no_double_embed_on_retry():
    embed = CountingEmbeddingFunc(fail_times=1)  # first flush raises
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="embedding failed"):
        await s.index_done_callback()

    # Buffer must remain so the next flush can retry.
    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is None
    s._client.upsert.assert_not_called()

    # Second attempt succeeds; total embed calls is 2 (one failed + one ok),
    # not 3 — the same content was retried exactly once.
    await s.index_done_callback()
    assert embed.call_count == 2
    s._client.upsert.assert_called_once()
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_server_upsert_failure_keeps_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.upsert.side_effect = RuntimeError("milvus down")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="milvus down"):
        await s.index_done_callback()

    # Embedding ran but server write failed; buffer must remain populated.
    assert "v1" in s._pending_vector_docs
    # Vector should be cached so retry doesn't re-embed.
    assert s._pending_vector_docs["v1"].vector is not None

    # On retry, no further embedding call; only the server write is reattempted.
    s._client.upsert.side_effect = None
    s._client.upsert.return_value = {"upsert_count": 1}
    await s.index_done_callback()
    assert embed.call_count == 1
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_finalize_raises_when_buffer_unflushed():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.upsert.side_effect = RuntimeError("transient milvus error")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="finalize.*flush raised"):
        await s.finalize()

    # Buffer still populated — caller knows data was lost.
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
    # No upsert payload, only the delete batch.
    s._client.upsert.assert_not_called()
    s._client.delete.assert_called_once()
    assert s._client.delete.call_args.kwargs["pks"] == ["v1"]


@pytest.mark.asyncio
async def test_delete_entity_relation_raises_on_server_failure():
    """Server-side failure must bubble up — no log-and-swallow."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.query.return_value = [{"id": "rel1"}, {"id": "rel2"}]
    s._client.delete.side_effect = RuntimeError("milvus delete failed")

    with pytest.raises(RuntimeError, match="milvus delete failed"):
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
    s._client.query.return_value = []  # no server-side hits

    await s.delete_entity_relation("A")
    # Pending doc whose src_id == A is pruned, the other survives.
    assert "rel-A-B" not in s._pending_vector_docs
    assert "rel-C-D" in s._pending_vector_docs


@pytest.mark.asyncio
async def test_delete_entity_relation_diverges_when_buffer_overwrites_persisted():
    """Pins the deferred ↔ eager semantic divergence documented on
    ``delete_entity_relation``.

    Scenario: a persisted row ``rel-X-Y`` has ``src_id="X" / tgt_id="Y"``,
    and a pending upsert is about to rewrite that same id so it would
    instead carry ``src_id="A" / tgt_id="B"``. A call to
    ``delete_entity_relation("A")`` arrives before the buffer is flushed.

    Expected (deferred mode, current implementation):
      * server-side filter ``src_id == "A" or tgt_id == "A"`` does NOT
        match the persisted row (its src/tgt are still X/Y), so the
        server-side delete is a no-op;
      * the buffered upsert IS pruned (its buffered src/tgt match);
      * net effect: persisted ``rel-X-Y`` (old values) survives and the
        pending overwrite is lost.

    Under eager ordering (upsert → flush → delete) the persisted row
    would have been rewritten first and then matched by the filter, so
    the final state would have been a deleted ``rel-X-Y``. This test
    locks in the divergence so a future refactor can't silently change
    it without touching the docstring.
    """
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    # Buffered upsert rewriting an (assumed) already-persisted rel-X-Y
    # so that its new src/tgt would match entity "A".
    await s.upsert({"rel-X-Y": {"content": "A → B", "src_id": "A", "tgt_id": "B"}})
    assert "rel-X-Y" in s._pending_vector_docs

    # Server still sees the OLD persisted row (src_id="X" / tgt_id="Y"),
    # so a filter on entity "A" returns nothing.
    s._client.query.return_value = []

    await s.delete_entity_relation("A")

    # Buffered overwrite is pruned (matches buffered src/tgt view) …
    assert "rel-X-Y" not in s._pending_vector_docs
    # … but the server-side delete is not issued, because the filter
    # didn't match the persisted row's actual src/tgt.
    s._client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_entity_relation_eager_ordering_matches_persisted():
    """Counterpart to the divergence test: if the caller flushes before
    invoking ``delete_entity_relation``, the persisted row reflects the
    buffered overwrite and the server-side filter catches it.

    This documents the recommended workaround called out in the
    ``delete_entity_relation`` docstring: ``index_done_callback()`` first
    when eager-equivalent semantics are required.
    """
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"rel-X-Y": {"content": "A → B", "src_id": "A", "tgt_id": "B"}})
    await s.index_done_callback()  # buffered upsert is now persisted
    assert s._pending_vector_docs == {}
    s._client.upsert.assert_called_once()

    # With the row persisted, the server filter on entity "A" now hits.
    s._client.query.return_value = [{"id": "rel-X-Y"}]

    await s.delete_entity_relation("A")

    s._client.delete.assert_called_once()
    assert s._client.delete.call_args.kwargs["pks"] == ["rel-X-Y"]


@pytest.mark.asyncio
async def test_get_by_id_reads_pending_buffer_without_vector():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello", "entity_name": "E1"}})
    doc = await s.get_by_id("v1")
    assert doc is not None
    assert doc["id"] == "v1"
    assert doc.get("entity_name") == "E1"
    assert "vector" not in doc
    # Server was not queried because the buffer answered the read.
    s._client.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_pending_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.delete(["v1"])
    assert await s.get_by_id("v1") is None
    s._client.query.assert_not_called()


@pytest.mark.asyncio
async def test_env_workspace_override_shares_flush_lock(patch_namespace_lock):
    """Two instances whose final_namespace collides must share the flush lock."""
    cache = patch_namespace_lock
    embed = CountingEmbeddingFunc()

    with patch.dict(os.environ, {"MILVUS_WORKSPACE": "shared_ws"}, clear=False):
        a = _make_storage(embed, workspace="caller_a")
        b = _make_storage(embed, workspace="caller_b")
        assert (
            a.final_namespace == b.final_namespace == "shared_ws_entities_mock_embed_8d"
        )
        assert a._flush_lock is b._flush_lock
        # Sanity: only one lock object was cached for that final_namespace.
        assert len([k for k in cache if k[0] == a.final_namespace]) == 1


@pytest.mark.asyncio
async def test_distinct_namespaces_get_independent_locks(patch_namespace_lock):
    """Different final_namespaces must NOT share a lock."""
    embed = CountingEmbeddingFunc()

    # Two instances with no env override and different workspaces produce
    # different final_namespaces ("a_entities" vs "b_entities").
    a = _make_storage(embed, workspace="a")
    b = _make_storage(embed, workspace="b")
    assert a.final_namespace != b.final_namespace
    assert a._flush_lock is not b._flush_lock


@pytest.mark.asyncio
async def test_mixed_upsert_and_delete_in_single_flush():
    """A flush carrying both pending upserts and pending deletes (on disjoint
    ids) must dispatch one server upsert and one server delete in a single
    pass, then clear both buffers."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"a": {"content": "alpha"}})
    await s.delete(["b"])

    assert set(s._pending_vector_docs.keys()) == {"a"}
    assert s._pending_vector_deletes == {"b"}

    await s.index_done_callback()

    s._client.upsert.assert_called_once()
    upsert_kwargs = s._client.upsert.call_args.kwargs
    assert {row["id"] for row in upsert_kwargs["data"]} == {"a"}

    s._client.delete.assert_called_once()
    assert s._client.delete.call_args.kwargs["pks"] == ["b"]

    # Both buffers cleared after a successful flush.
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_finalize_clean_flush_no_raise():
    """Happy-path counterpart to test_finalize_raises_when_buffer_unflushed:
    a successful flush during finalize() must leave both buffers empty and
    must not raise."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v2"])

    await s.finalize()  # must not raise

    s._client.upsert.assert_called_once()
    s._client.delete.assert_called_once()
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_drop_clears_pending_buffers():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.has_collection.return_value = False  # skip drop_collection call
    # Stub out the recreate path to avoid hitting MilvusIndexConfig logic.
    with (
        patch.object(s, "_create_collection_with_schema"),
        patch.object(s, "_ensure_collection_loaded"),
    ):
        await s.upsert({"v1": {"content": "hello"}})
        await s.delete(["v2"])
        assert s._pending_vector_docs and s._pending_vector_deletes

        result = await s.drop()
        assert result["status"] == "success"
        assert s._pending_vector_docs == {}
        assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_drop_recreates_empty_without_legacy_migration():
    # drop() must leave the collection EMPTY. Recreating via
    # _create_collection_if_not_exist would re-run the legacy->suffixed
    # migration (the legacy collection is intentionally kept after migration),
    # pulling the just-dropped rows back in and forcing a needless full
    # migration on every rebuild/clear. Regression for that path.
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.has_collection.return_value = True

    with (
        patch.object(s, "_create_collection_if_not_exist") as recreate_via_migration,
        patch.object(s, "_create_collection_with_schema") as create_empty,
        patch.object(s, "_ensure_collection_loaded") as load,
    ):
        result = await s.drop()

    assert result["status"] == "success"
    s._client.drop_collection.assert_called_once_with(s.final_namespace)
    create_empty.assert_called_once_with(s.final_namespace)
    load.assert_called_once_with()
    # Never the migration-capable path.
    recreate_via_migration.assert_not_called()
    s._client.query_iterator.assert_not_called()


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
