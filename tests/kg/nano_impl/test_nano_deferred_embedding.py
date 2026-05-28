"""Deferred-embedding coverage for ``NanoVectorDBStorage``.

The storage no longer embeds eagerly in ``upsert``: it buffers a pending doc
and embeds once per id at flush time (``index_done_callback`` / ``finalize``).
These tests pin that contract using a counting mock embedding function — no
live model or network. They mirror the protocol proven for
``OpenSearchVectorDBStorage`` (issue #2785).
"""

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")  # noqa: F841

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    initialize_share_data,
    finalize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _CountingEmbed:
    """Async embedding callable that records how many texts it embedded and how
    many times it was invoked (one invocation == one batch)."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.call_count = 0
        self.embedded_texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        self.embedded_texts.extend(texts)
        # Deterministic per-text vector so duplicates are still 1-1.
        return np.array(
            [
                np.full(self.dim, (abs(hash(t)) % 97) + 1, dtype=np.float32)
                for t in texts
            ]
        )


def _make_storage(tmp_path, embed: _CountingEmbed) -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=DIM, max_token_size=512, func=embed),
        meta_fields={"content"},
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_defers_embedding_to_index_done_callback(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert(
        {
            "id1": {"content": "alpha"},
            "id2": {"content": "beta"},
        }
    )
    assert embed.call_count == 0, "upsert must not embed"
    assert len(storage._client) == 0, "nothing should be materialized yet"

    await storage.index_done_callback()
    assert embed.call_count == 1, "flush should embed in a single batch"
    assert sorted(embed.embedded_texts) == ["alpha", "beta"]
    assert len(storage._client) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_repeated_upserts_same_id_embed_once_per_flush(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "v1"}})
    await storage.upsert({"id1": {"content": "v2"}})
    await storage.upsert({"id1": {"content": "v3"}})

    await storage.index_done_callback()

    assert embed.call_count == 1
    assert embed.embedded_texts == ["v3"], "only the latest content is embedded"
    assert len(storage._client) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_vectors_caches_and_flush_reuses(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    vecs = await storage.get_vectors_by_ids(["id1"])
    assert "id1" in vecs and len(vecs["id1"]) == DIM
    assert embed.call_count == 1, "get_vectors_by_ids embeds pending lazily"

    # Flush must reuse the cached vector, not re-embed.
    await storage.index_done_callback()
    assert embed.call_count == 1, "flush should reuse the cached temp vector"
    assert len(storage._client) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reupsert_after_get_vectors_clears_cached_vector(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "old"}})
    await storage.get_vectors_by_ids(["id1"])  # caches a temp vector for "old"
    assert embed.call_count == 1

    # New content version must clear the cached vector and re-embed at flush.
    await storage.upsert({"id1": {"content": "new"}})
    await storage.index_done_callback()

    assert embed.call_count == 2
    assert embed.embedded_texts == ["old", "new"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_cancels_pending_and_removes_materialized(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    # Materialize id1; leave id2 only as a pending (unflushed) upsert.
    await storage.upsert({"id1": {"content": "alpha"}})
    await storage.index_done_callback()
    await storage.upsert({"id2": {"content": "beta"}})

    await storage.delete(["id1", "id2"])

    assert "id2" not in storage._pending_upserts, "delete cancels pending upsert"
    assert len(storage._client) == 0, "delete removes the materialized row immediately"
    assert await storage.get_by_id("id1") is None
    assert await storage.get_by_id("id2") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stale_client_reload_still_flushes_pending_upsert(tmp_path):
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    stale_writer = _make_storage(tmp_path, embed)
    await writer.initialize()
    await stale_writer.initialize()

    await writer.upsert({"id1": {"content": "alpha"}})
    assert await writer.index_done_callback() is True
    assert stale_writer.storage_updated.value is True

    await stale_writer.upsert({"id2": {"content": "beta"}})
    assert await stale_writer.index_done_callback() is True

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    rows = await reader.get_by_ids(["id1", "id2"])
    assert [row["id"] for row in rows] == ["id1", "id2"]
    assert stale_writer._pending_upserts == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_reloads_stale_client_before_mutating(tmp_path):
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    stale_deleter = _make_storage(tmp_path, embed)
    await writer.initialize()
    await stale_deleter.initialize()

    await writer.upsert({"id1": {"content": "alpha"}})
    assert await writer.index_done_callback() is True
    assert stale_deleter.storage_updated.value is True

    await stale_deleter.delete(["id1"])
    assert stale_deleter.storage_updated.value is False
    assert await stale_deleter.index_done_callback() is True

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_reloads_stale_client_before_flushing(tmp_path):
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    stale_finalizer = _make_storage(tmp_path, embed)
    await writer.initialize()
    await stale_finalizer.initialize()

    await writer.upsert({"id1": {"content": "alpha"}})
    assert await writer.index_done_callback() is True
    assert stale_finalizer.storage_updated.value is True

    await stale_finalizer.upsert({"id2": {"content": "beta"}})
    await stale_finalizer.finalize()

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    rows = await reader.get_by_ids(["id1", "id2"])
    assert [row["id"] for row in rows] == ["id1", "id2"]
    assert stale_finalizer._pending_upserts == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_read_your_writes_and_query_after_flush(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    # Before flush: read paths see the pending row, query does not.
    hit = await storage.get_by_id("id1")
    assert hit is not None and hit["id"] == "id1" and hit["content"] == "alpha"
    by_ids = await storage.get_by_ids(["id1", "missing"])
    assert by_ids[0]["id"] == "id1" and by_ids[1] is None
    assert await storage.query("alpha", top_k=5) == [], "query ignores unflushed data"

    # After flush: query returns the row.
    await storage.index_done_callback()
    results = await storage.query("alpha", top_k=5)
    assert any(r["id"] == "id1" for r in results)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_flushes_pending(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})
    await storage.finalize()

    assert embed.call_count == 1
    assert storage._pending_upserts == {}
    assert len(storage._client) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_retries_save_after_flush_failure(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    original_save = storage._save_to_disk_locked
    save_calls = 0

    def fail_once():
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            raise OSError("boom")
        original_save()

    storage._save_to_disk_locked = fail_once

    with pytest.raises(OSError, match="boom"):
        await storage.finalize()

    assert storage._pending_upserts == {}
    assert storage._client_dirty is True

    await storage.finalize()

    assert save_calls == 2
    assert storage._client_dirty is False

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    hit = await reader.get_by_id("id1")
    assert hit is not None and hit["id"] == "id1"
