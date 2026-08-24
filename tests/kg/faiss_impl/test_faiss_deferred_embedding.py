"""Deferred-embedding coverage for ``FaissVectorDBStorage``.

The storage no longer embeds eagerly in ``upsert``: it buffers a pending doc
and embeds once per id at flush time (``index_done_callback`` / ``finalize``).
These tests pin that contract using a counting mock embedding function — no
live model or network. They mirror the protocol proven for
``NanoVectorDBStorage`` (issue #2785) plus three Faiss-specific cases:

- ``test_reupsert_after_flush_replaces_single_fid`` — Faiss has no in-place
  upsert; verify the rebuild keeps a single fid per custom id.
- ``test_index_done_callback_save_failure_raises`` — flush succeeds, save IO
  fails: pending is empty, ``_index_dirty`` stays True, the materialized index
  is preserved for a finalize retry.
- ``test_reload_warns_on_index_meta_skew`` — ``index > meta`` on-disk skew
  (from a crash between the two atomic_writes) is logged on reload but **not**
  auto-repaired.
"""

import contextlib
import json
import os
import time
from unittest.mock import patch

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg import write_seq  # noqa: E402
from lightrag.kg.write_seq import WRITE_SEQ_FIELD  # noqa: E402
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


@contextlib.contextmanager
def _frozen_clock(stamp: int):
    """Pin the ``__created_at__`` the storage stamps inside the block.

    ``upsert`` records ``int(time.time())``, and the replay's ordering guard
    compares those whole seconds, so a test needs an explicit clock to make one
    row strictly newer than another (or to force a same-second tie).
    """
    with patch("lightrag.kg.faiss_impl.time.time", return_value=stamp):
        yield


def _make_storage(tmp_path, embed: _CountingEmbed) -> FaissVectorDBStorage:
    return FaissVectorDBStorage(
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


def _assert_consistent(storage: FaissVectorDBStorage) -> None:
    """Faiss has two structures (index + meta dict); the root failure mode is
    them diverging. Every test that mutates state asserts they match."""
    assert storage._index.ntotal == len(storage._id_to_meta), (
        f"index ntotal ({storage._index.ntotal}) != meta length "
        f"({len(storage._id_to_meta)})"
    )


# ---------------------------------------------------------------------------
# (A) Nano-ported tests
# ---------------------------------------------------------------------------


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
    assert storage._index.ntotal == 0, "nothing should be materialized yet"
    _assert_consistent(storage)

    await storage.index_done_callback()
    assert embed.call_count == 1, "flush should embed in a single batch"
    assert sorted(embed.embedded_texts) == ["alpha", "beta"]
    assert storage._index.ntotal == 2
    _assert_consistent(storage)


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
    assert storage._index.ntotal == 1
    _assert_consistent(storage)


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
    assert storage._index.ntotal == 1
    _assert_consistent(storage)


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
    _assert_consistent(storage)


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
    # Materialized removal is deferred: one batched rebuild at flush time
    # instead of one per delete call (#3681). Read-your-writes hides the row.
    assert storage._index.ntotal == 1, "materialized row survives until the flush"
    assert {"id1", "id2"} <= storage._pending_deletes
    assert await storage.get_by_id("id1") is None
    assert await storage.get_by_id("id2") is None

    assert await storage.index_done_callback() is True
    assert storage._index.ntotal == 0, "flush removes the materialized row"
    _assert_consistent(storage)


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
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_reloads_stale_client_before_mutating(tmp_path):
    """The stale-writer interplay with deferred deletes: the reload happens
    inside ``index_done_callback`` (before the flush applies the queued
    delete), so the delete lands on top of the other writer's committed
    snapshot instead of being silently reverted by it."""
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    stale_deleter = _make_storage(tmp_path, embed)
    await writer.initialize()
    await stale_deleter.initialize()

    await writer.upsert({"id1": {"content": "alpha"}})
    assert await writer.index_done_callback() is True
    assert stale_deleter.storage_updated.value is True

    await stale_deleter.delete(["id1"])
    assert await stale_deleter.index_done_callback() is True
    assert stale_deleter.storage_updated.value is False

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None
    _assert_consistent(reader)


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
    _assert_consistent(reader)


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
    _assert_consistent(storage)


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
    assert storage._index.ntotal == 1
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_entity_relation_cancels_pending(tmp_path):
    embed = _CountingEmbed()
    storage = FaissVectorDBStorage(
        namespace="test_relations",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=DIM, max_token_size=512, func=embed),
        meta_fields={"content", "src_id", "tgt_id"},
    )
    await storage.initialize()

    # Materialize r1 (A->B), leave r2 (A->C) and r3 (X->Y) as pending.
    await storage.upsert({"r1": {"content": "rel1", "src_id": "A", "tgt_id": "B"}})
    await storage.index_done_callback()
    await storage.upsert(
        {
            "r2": {"content": "rel2", "src_id": "A", "tgt_id": "C"},
            "r3": {"content": "rel3", "src_id": "X", "tgt_id": "Y"},
        }
    )

    await storage.delete_entity_relation("A")

    assert "r2" not in storage._pending_upserts, "incident pending entry cancelled"
    assert "r3" in storage._pending_upserts, "unrelated pending entry preserved"
    assert storage._index.ntotal == 0, "materialized A->B removed"
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_flush_embedding_failure_raises_and_keeps_pending(tmp_path):
    class _FailingEmbed:
        def __init__(self):
            self.call_count = 0

        async def __call__(self, texts, **kwargs):
            self.call_count += 1
            raise RuntimeError("embed boom")

    embed = _FailingEmbed()
    storage = FaissVectorDBStorage(
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
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    with pytest.raises(RuntimeError, match="embed boom"):
        await storage.index_done_callback()

    assert "id1" in storage._pending_upserts, "pending preserved for retry"
    assert storage._index.ntotal == 0, "nothing materialized on embed failure"
    # Embed failure happens before self._index.add in _flush_pending_locked,
    # so _index_dirty must NOT be set. (A save-stage failure would leave it True
    # — see test_index_done_callback_save_failure_raises.)
    assert storage._index_dirty is False
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_discards_pending_without_embedding(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})
    assert "id1" in storage._pending_upserts

    result = await storage.drop()

    assert result["status"] == "success"
    assert storage._pending_upserts == {}, "drop discards buffered upserts"
    assert embed.call_count == 0, "drop must not embed"
    assert storage._index_dirty is False
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_retries_save_after_flush_failure(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    original_save = storage._save_faiss_index
    save_calls = 0

    def fail_once():
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            raise OSError("boom")
        original_save()

    storage._save_faiss_index = fail_once

    with pytest.raises(OSError, match="boom"):
        await storage.finalize()

    assert storage._pending_upserts == {}
    assert storage._index_dirty is True

    await storage.finalize()

    assert save_calls == 2
    assert storage._index_dirty is False

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    hit = await reader.get_by_id("id1")
    assert hit is not None and hit["id"] == "id1"
    _assert_consistent(reader)


# ---------------------------------------------------------------------------
# (B) Faiss-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reupsert_after_flush_replaces_single_fid(tmp_path):
    """Faiss has no in-place upsert: re-upserting an already-materialized id
    must rebuild the index without the old fid, so we still end up with
    exactly one row per custom id."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "old"}})
    await storage.index_done_callback()
    assert storage._index.ntotal == 1
    _assert_consistent(storage)

    await storage.upsert({"id1": {"content": "new"}})
    await storage.index_done_callback()

    assert storage._index.ntotal == 1, "rebuild must remove old fid before adding new"
    assert len(storage._id_to_meta) == 1
    _assert_consistent(storage)

    hit = await storage.get_by_id("id1")
    assert hit is not None and hit["content"] == "new"
    assert embed.call_count == 2, "each flush embeds the latest content once"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_index_done_callback_save_failure_raises(tmp_path):
    """Save failure in index_done_callback must propagate, leave pending empty
    (flush already succeeded), and keep _index_dirty=True so finalize retries."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    original_save = storage._save_faiss_index

    def fail_save():
        raise OSError("save boom")

    storage._save_faiss_index = fail_save

    with pytest.raises(OSError, match="save boom"):
        await storage.index_done_callback()

    assert storage._pending_upserts == {}, "flush succeeded so pending is empty"
    assert storage._index_dirty is True, "save failure preserves dirty for retry"
    assert storage._index.ntotal == 1, "materialized state is preserved"
    _assert_consistent(storage)

    # Restore real save; finalize must retry only the save (no re-embed).
    storage._save_faiss_index = original_save
    embed_before = embed.call_count
    await storage.finalize()
    assert embed.call_count == embed_before, "save retry must not re-embed"
    assert storage._index_dirty is False

    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    hit = await reader.get_by_id("id1")
    assert hit is not None and hit["id"] == "id1"
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reload_warns_on_index_meta_skew(tmp_path, caplog):
    """A crash between the .index write and the .meta.json write leaves
    ``ntotal(.index) > rows(.meta)``. ``_load_faiss_index`` must log a warning
    on reload; auto-repair is intentionally not in scope here."""
    import logging

    from lightrag.utils import logger as lightrag_logger

    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    await writer.initialize()

    await writer.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
    await writer.index_done_callback()

    # Corrupt the meta file: drop one entry so disk has index > meta.
    with open(writer._meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert len(meta) == 2
    dropped_key = next(iter(meta))
    del meta[dropped_key]
    with open(writer._meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # The lightrag logger sets propagate=False (lightrag/utils.py), so caplog —
    # which attaches to root by default — never sees its records. Flip propagate
    # for the duration of the reload, then restore.
    caplog.clear()
    old_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            reader = _make_storage(tmp_path, embed)
            await reader.initialize()
    finally:
        lightrag_logger.propagate = old_propagate

    # The reader's index still has 2 vectors but only 1 reachable via meta —
    # this is the "known risk, not auto-repaired" state.
    assert reader._index.ntotal == 2
    assert len(reader._id_to_meta) == 1
    skew_messages = [
        rec.message
        for rec in caplog.records
        if "skew" in rec.message or "index > meta" in rec.message
    ]
    assert skew_messages, (
        f"expected an index>meta skew warning; got: "
        f"{[r.message for r in caplog.records]}"
    )

    # Sanity: state files exist where we left them.
    assert os.path.exists(writer._faiss_index_file)
    assert os.path.exists(writer._meta_file)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_skips_orphan_faiss_hits(tmp_path):
    """After an ``index > meta`` skew the orphan vector is still searchable by
    similarity, but ``query`` must skip it instead of leaking a ghost
    ``{"id": None, ...}`` row to the caller."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    # Materialize two rows.
    await storage.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
    await storage.index_done_callback()
    assert storage._index.ntotal == 2

    # Synthesize the skew: drop one meta row in memory, keeping the faiss
    # index untouched. This mirrors what _load_faiss_index would surface on
    # reload after a crash between the two atomic_writes.
    orphan_fid = next(iter(storage._id_to_meta))
    del storage._id_to_meta[orphan_fid]
    assert storage._index.ntotal == 2
    assert len(storage._id_to_meta) == 1

    # The orphan vector still scores high in similarity search; query must
    # filter it out instead of returning {"id": None, ...}.
    results = await storage.query("anything", top_k=5)
    for row in results:
        assert row["id"] is not None, f"orphan hit leaked: {row}"
    # And the surviving row is still returned.
    surviving_id = next(iter(storage._id_to_meta.values()))["__id__"]
    assert any(r["id"] == surviving_id for r in results)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reupsert_cleans_duplicate_custom_id_rows(tmp_path):
    """Defends against legacy / externally corrupted stores where multiple
    fids in ``_id_to_meta`` share the same ``__id__``. A re-upsert + flush
    must collapse them to a single row; a ``delete`` must remove all of them."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    # Hand-craft a corrupt state: two fids carry the same custom id "dup".
    matrix = np.array([[1.0] * DIM, [2.0] * DIM], dtype=np.float32)
    faiss.normalize_L2(matrix)
    storage._index.add(matrix)
    storage._id_to_meta[0] = {
        "__id__": "dup",
        "__created_at__": 1,
        "content": "v1",
        "__vector__": matrix[0].tolist(),
    }
    storage._id_to_meta[1] = {
        "__id__": "dup",
        "__created_at__": 1,
        "content": "v2",
        "__vector__": matrix[1].tolist(),
    }
    _assert_consistent(storage)
    assert storage._find_faiss_ids_by_custom_id("dup") == [0, 1]

    # Re-upsert + flush: both duplicates must be removed in the rebuild
    # before the new vector is added; final state is a single row.
    await storage.upsert({"dup": {"content": "v3"}})
    await storage.index_done_callback()

    assert storage._index.ntotal == 1, "flush rebuild must drop both duplicates"
    assert len(storage._id_to_meta) == 1
    assert storage._find_faiss_ids_by_custom_id("dup") == list(
        storage._id_to_meta.keys()
    )
    hit = await storage.get_by_id("dup")
    assert hit is not None and hit["content"] == "v3"
    _assert_consistent(storage)

    # Re-seed two more duplicates and verify delete also removes them all.
    matrix2 = np.array([[3.0] * DIM, [4.0] * DIM], dtype=np.float32)
    faiss.normalize_L2(matrix2)
    storage._index.add(matrix2)
    next_fid = max(storage._id_to_meta) + 1
    storage._id_to_meta[next_fid] = {
        "__id__": "dup",
        "__created_at__": 2,
        "content": "dup-a",
        "__vector__": matrix2[0].tolist(),
    }
    storage._id_to_meta[next_fid + 1] = {
        "__id__": "dup",
        "__created_at__": 2,
        "content": "dup-b",
        "__vector__": matrix2[1].tolist(),
    }
    assert len(storage._find_faiss_ids_by_custom_id("dup")) == 3

    await storage.delete(["dup"])
    assert storage._find_faiss_ids_by_custom_id("dup") == [0, 1, 2], (
        "queued delete does not mutate the materialized index"
    )
    assert await storage.index_done_callback() is True
    assert storage._find_faiss_ids_by_custom_id("dup") == []
    assert storage._index.ntotal == 0
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_propagates_errors(tmp_path, monkeypatch):
    """Faiss ``delete`` must NOT swallow errors — the caller (document
    deletion / status update path) needs to abort if vectors weren't
    actually removed. With deferred deletes the destructive work moved to
    the flush, so the abort surfaces from ``index_done_callback``; this
    intentionally diverges from Nano."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})
    await storage.index_done_callback()

    def boom(_self, _fids):
        raise RuntimeError("rebuild boom")

    # _remove_faiss_ids_locked is what the flush calls under the hood to
    # apply the queued deletes.
    monkeypatch.setattr(
        FaissVectorDBStorage, "_remove_faiss_ids_locked", boom, raising=True
    )

    await storage.delete(["id1"])
    assert "id1" in storage._pending_deletes

    with pytest.raises(RuntimeError, match="rebuild boom"):
        await storage.index_done_callback()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_flush_recovers_from_index_add_failure_without_re_embedding(tmp_path):
    """Self-heal contract: if ``index.add`` raises mid-flush (after embedding
    already succeeded), the pending buffer keeps the cached vectors and a
    subsequent ``finalize`` retries the flush **without re-embedding**. Pins
    the "pending is the source of truth on mid-write failure" invariant
    documented on ``_flush_pending_locked``."""

    class _AddFailsOnce:
        """Wraps a real faiss index, raising on the first ``.add`` call. After
        the second add succeeds it swaps the storage's ``_index`` attribute
        back to the real instance, so ``faiss.write_index`` (which requires a
        real SWIG-wrapped object) can run during the retry's save step. This
        is a test-only shim — in production ``self._index`` is always a real
        faiss index throughout the retry.
        """

        def __init__(self, storage, real):
            self._storage = storage
            self._real = real
            self._calls = 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def add(self, arr):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("add boom")
            result = self._real.add(arr)
            self._storage._index = self._real
            return result

    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})

    real_index = storage._index
    storage._index = _AddFailsOnce(storage, real_index)

    with pytest.raises(RuntimeError, match="add boom"):
        await storage.index_done_callback()

    # Embedding completed once (failure happened after embed, in index.add).
    assert embed.call_count == 1
    # Pending preserved with cached vectors — that's the self-healing key.
    assert "id1" in storage._pending_upserts
    assert storage._pending_upserts["id1"].vector is not None
    # _index_dirty stays False: docstring says we deliberately don't flip it
    # on mid-write failure (pending is the source of truth).
    assert storage._index_dirty is False
    assert storage._index.ntotal == 0

    # Retry through the same public entry point. The wrapper's second add
    # succeeds, unwraps itself, and the rest of finalize (save + notify)
    # runs against the real index.
    await storage.finalize()

    assert embed.call_count == 1, "retry must reuse cached vectors, not re-embed"
    assert storage._index is real_index, "wrapper unwrapped itself on the second add"
    assert storage._index.ntotal == 1
    assert storage._pending_upserts == {}
    assert storage._index_dirty is False
    _assert_consistent(storage)

    # And the row was persisted to disk by the retry's save.
    reader = _make_storage(tmp_path, embed)
    await reader.initialize()
    hit = await reader.get_by_id("id1")
    assert hit is not None and hit["content"] == "alpha"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_index_ops_clears_buffer(tmp_path):
    """An internal-error abort calls drop_pending_index_ops to discard the
    not-yet-flushed buffer without materializing anything into the index."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
    assert storage._pending_upserts, "upsert buffers, does not flush"

    await storage.drop_pending_index_ops()

    assert storage._pending_upserts == {}
    assert embed.call_count == 0, "drop must not embed"
    assert storage._index.ntotal == 0, "nothing was materialized"
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_does_not_rollback_materialized(tmp_path):
    """drop_pending_index_ops discards ONLY the pending buffer; rows already
    materialized into the index by a flush whose save then failed
    (``_index_dirty=True``) are intentionally NOT rolled back."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    # Flush id1 into the index, then fail the save so it stays
    # materialized-but-unsaved (dirty) and the pending buffer is emptied.
    await storage.upsert({"id1": {"content": "alpha"}})

    def fail_save():
        raise OSError("save boom")

    storage._save_faiss_index = fail_save
    with pytest.raises(OSError, match="save boom"):
        await storage.index_done_callback()
    assert storage._pending_upserts == {}, "flush succeeded so pending is empty"
    assert storage._index_dirty is True
    assert storage._index.ntotal == 1, "id1 materialized, not saved"

    # A new pending op arrives, then the batch aborts and drops pending.
    await storage.upsert({"id2": {"content": "beta"}})
    assert "id2" in storage._pending_upserts

    await storage.drop_pending_index_ops()

    assert storage._pending_upserts == {}, "pending id2 dropped"
    assert storage._index.ntotal == 1, "materialized id1 NOT rolled back"
    assert storage._index_dirty is True, "still dirty for a later save retry"
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_flush_add_failure_leaves_the_superseded_row_intact(
    tmp_path, monkeypatch
):
    """A failure while preparing the overwrite must leave the row the id
    already had.

    The flush used to drop that row first and add the replacement after, so a
    failure in between left the id with nothing. An aborting batch then
    discards ``_pending_upserts``, which held the only remaining copy of the
    vector, and the next save committed the hole.

    The batch has to carry a delete as well as the overwrite: that is what
    marks the index dirty, and a dirty index is what makes the later save go
    ahead and commit the hole.
    """
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()
    await storage.upsert({"doomed": {"content": "x"}, "keep": {"content": "v1"}})
    assert await storage.index_done_callback() is True
    _assert_consistent(storage)

    # A mixed batch: one delete, one overwrite of an existing row.
    await storage.delete(["doomed"])
    await storage.upsert({"keep": {"content": "v2"}})

    def boom(*_args, **_kwargs):
        raise RuntimeError("faiss add boom")

    monkeypatch.setattr(np, "vstack", boom)
    with pytest.raises(RuntimeError, match="faiss add boom"):
        await storage.index_done_callback()
    monkeypatch.undo()
    _assert_consistent(storage)

    # The pipeline aborts the batch, discarding the buffered replacement —
    # the only remaining copy of that vector — and a later save persists
    # whatever the index holds.
    await storage.drop_pending_index_ops()
    await storage.finalize()

    reloaded = _make_storage(tmp_path, _CountingEmbed())
    await reloaded.initialize()
    survivor = await reloaded.get_by_id("keep")
    assert survivor is not None, "an aborted overwrite must not delete the row"
    assert survivor["content"] == "v1"
    assert await reloaded.get_by_id("doomed") is None, "the delete still applies"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_overwrite_is_one_rebuild_and_all_or_nothing(tmp_path, monkeypatch):
    """An overwrite goes through a single rebuild that both drops the row the
    id had and writes the row it has now.

    Two properties, and they are the same property. It is *one* operation, so
    a failure cannot land one half — no missing row, no two rows of different
    vintages — and it is one *rebuild*, not an append followed by a cleanup
    rebuild, so it does not cost an extra O(rows) pass either.
    """
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()
    await storage.upsert({"keep": {"content": "v1"}, "other": {"content": "o"}})
    assert await storage.index_done_callback() is True

    rebuilds = []
    original = storage._rebuild_index_locked

    def counting(drop_fids, add_records=()):
        rebuilds.append((list(drop_fids), len(add_records)))
        return original(drop_fids, add_records)

    monkeypatch.setattr(storage, "_rebuild_index_locked", counting)
    await storage.upsert({"keep": {"content": "v2"}})
    assert await storage.index_done_callback() is True
    monkeypatch.undo()

    assert len(rebuilds) == 1, "an overwrite is a single rebuild"
    assert rebuilds[0][1] == 1, "the new row is written by that same rebuild"
    assert len(storage._find_faiss_ids_by_custom_id("keep")) == 1
    record = await storage.get_by_id("keep")
    assert record is not None and record["content"] == "v2"
    _assert_consistent(storage)

    # Now fail that rebuild: neither half may land.
    await storage.upsert({"keep": {"content": "v3"}})

    def boom(_drop_fids, _add_records=()):
        raise RuntimeError("faiss rebuild boom")

    monkeypatch.setattr(storage, "_rebuild_index_locked", boom)
    with pytest.raises(RuntimeError, match="faiss rebuild boom"):
        await storage.index_done_callback()
    monkeypatch.undo()

    fids = storage._find_faiss_ids_by_custom_id("keep")
    assert len(fids) == 1, "a failed overwrite must not leave a second row"
    assert storage._id_to_meta[fids[0]]["content"] == "v2", (
        "the row the id had must be intact"
    )
    assert storage._index_dirty is False, "nothing landed, so nothing to save"
    _assert_consistent(storage)

    # The read paths still show the buffered v3 — the retry is pending, not
    # lost — and a successful retry applies it exactly once.
    assert "keep" in storage._pending_upserts
    assert await storage.index_done_callback() is True
    assert len(storage._find_faiss_ids_by_custom_id("keep")) == 1
    record = await storage.get_by_id("keep")
    assert record is not None and record["content"] == "v3"
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_insert_only_flush_does_not_rebuild(tmp_path, monkeypatch):
    """Nothing is superseded when every id is new, so the flush appends
    instead of rebuilding — the O(rows) pass belongs to overwrites only."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()
    await storage.upsert({f"row{i}": {"content": f"c{i}"} for i in range(3)})
    assert await storage.index_done_callback() is True

    rebuilds = []
    original = storage._rebuild_index_locked

    def counting(drop_fids, add_records=()):
        rebuilds.append(list(drop_fids))
        return original(drop_fids, add_records)

    monkeypatch.setattr(storage, "_rebuild_index_locked", counting)
    await storage.upsert({"fresh": {"content": "brand new"}})
    assert await storage.index_done_callback() is True
    monkeypatch.undo()

    assert rebuilds == [], "an insert-only flush must not rebuild the index"
    assert storage._index.ntotal == 4
    record = await storage.get_by_id("fresh")
    assert record is not None and record["content"] == "brand new"
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_failure_leaves_index_and_meta_untouched(tmp_path, monkeypatch):
    """``_rebuild_index_locked`` promises that ``_index`` and ``_id_to_meta``
    flip together; this pins that the promise survives the failure path,
    where the two are actually assigned.

    Building the replacement index in place — assigning ``self._index`` and
    only then adding to it — left an empty index behind a full
    ``_id_to_meta`` when the add raised. Every row then reads as unbacked,
    and once anything persisted that state the whole namespace was dropped
    on the next load.
    """
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()
    await storage.upsert({f"row{i}": {"content": f"c{i}"} for i in range(4)})
    assert await storage.index_done_callback() is True
    _assert_consistent(storage)

    # Overwrite one row, so the flush goes through a rebuild.
    await storage.upsert({"row0": {"content": "new"}})

    # Fail the add of the index the rebuild constructs — not the helper
    # itself, since inside it is where the damage happened. During a flush
    # the only IndexFlatIP built is the rebuild's, so the first one is it.
    real_cls = faiss.IndexFlatIP
    built = {"n": 0}

    def failing_factory(dim):
        index = real_cls(dim)
        built["n"] += 1
        if built["n"] == 1:

            def boom(_arr):
                raise RuntimeError("rebuild add boom")

            index.add = boom
        return index

    monkeypatch.setattr(faiss, "IndexFlatIP", failing_factory)
    with pytest.raises(RuntimeError, match="rebuild add boom"):
        await storage.index_done_callback()
    monkeypatch.undo()

    _assert_consistent(storage)
    assert len(storage._find_faiss_ids_by_custom_id("row0")) == 1, (
        "a failed overwrite leaves the id one row — the one it already had, "
        "not that row plus the replacement"
    )

    # The batch aborts and a later save commits whatever the index holds.
    await storage.drop_pending_index_ops()
    await storage.finalize()

    reloaded = _make_storage(tmp_path, _CountingEmbed())
    await reloaded.initialize()
    for i in range(4):
        assert await reloaded.get_by_id(f"row{i}") is not None, (
            f"row{i} must survive a failed rebuild"
        )
    _assert_consistent(reloaded)


# ---------------------------------------------------------------------------
# Upsert redo log: materialized-but-unsaved rows survive a foreign-commit
# reload (issue #3688). Twin of the Nano coverage in
# tests/kg/nano_impl/test_nano_deferred_embedding.py.
# ---------------------------------------------------------------------------


def _make_save_fail(storage):
    """Make ``_save_faiss_index`` raise; returns a restore callable."""
    original = storage._save_faiss_index

    def boom():
        raise OSError("disk full")

    storage._save_faiss_index = boom

    def restore():
        storage._save_faiss_index = original

    return restore


@pytest.mark.offline
@pytest.mark.asyncio
async def test_index_done_replays_unsaved_upsert_after_foreign_commit(tmp_path):
    """Fix-proof for issue #3688 (the retry-via-callback half).

    Chain: flush materializes id2 and its save fails -> another writer
    commits id3 -> the retry's reload rebuilds the in-memory state from the
    foreign snapshot. Without the redo log nothing replays id2 and the
    following save silently loses it; with the log both rows land — and the
    replay reuses the cached vector, so the retry embeds nothing.
    """
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"id2": {"content": "beta"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError, match="disk full"):
        await writer.index_done_callback()
    restore()
    assert writer._pending_upserts == {}, "flush succeeded so pending is empty"
    assert writer._unsaved_upserts, "the flushed doc moved into the redo log"
    calls_after_failure = embed.call_count

    # Another writer commits, flagging `writer` as stale.
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True
    assert writer.storage_updated.value is True

    assert await writer.index_done_callback() is True
    assert embed.call_count == calls_after_failure, "replay must not re-embed"
    assert not writer._unsaved_upserts, "durable save clears the redo log"
    _assert_consistent(writer)

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("id2"))["content"] == "beta", (
        "the reload dropped the materialized-but-unsaved row and nothing "
        "replayed it (issue #3688)"
    )
    assert (await reader.get_by_id("id3"))["content"] == "gamma"
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_replays_unsaved_upserts_after_a_foreign_commit(tmp_path):
    """Fix-proof for the mirror half: finalize used to skip the reload while
    ``_unsaved_upserts`` was set, and its save wrote our pre-commit snapshot
    over the other writer's durable rows. With the redo log it reloads and
    replays, so both sides survive."""
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"id2": {"content": "beta"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert writer._unsaved_upserts, "the flushed doc moved into the redo log"

    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    await writer.finalize()
    _assert_consistent(writer)

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("id2"))["content"] == "beta", (
        "the redo log must replay our row on top of the reloaded snapshot"
    )
    assert (await reader.get_by_id("id3"))["content"] == "gamma", (
        "the other writer's commit must survive finalize"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_serve_an_unsaved_upsert_after_a_foreign_reload(tmp_path):
    """Read-your-writes across the failed-save window: a foreign commit makes
    the next ``_get_index`` reload drop the unsaved row, so the read paths
    must serve it from the redo log until the replay."""
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"id2": {"content": "beta"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()

    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    # Each read triggers the reload-if-stale path internally.
    got = await writer.get_by_id("id2")
    assert got is not None and got["content"] == "beta"
    by_ids = await writer.get_by_ids(["id2", "id3"])
    assert by_ids[0] is not None and by_ids[0]["content"] == "beta"
    assert by_ids[1] is not None and by_ids[1]["content"] == "gamma"
    vectors = await writer.get_vectors_by_ids(["id2"])
    assert "id2" in vectors and len(vectors["id2"]) == DIM
    assert embed.call_count == 1, "the logged vector is reused, never re-embedded"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_replay_overwrites_an_older_foreign_row_under_the_same_id(tmp_path):
    """Ids are content hashes, so a same-id stored row is an older version of
    the same logical record; our applied-but-unsaved write is the newer intent
    and the replay puts it back on top of the reloaded snapshot (drop-their-fid
    + add ours in one rebuild).

    The foreign row is committed *before* our write on purpose: written after
    it, it would be the newer version and the replay would rightly decline
    (see ``test_same_second_tie_is_broken_by_the_write_sequence``).
    """
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await other.upsert({"idX": {"content": "theirs"}})
    assert await other.index_done_callback() is True

    await writer.upsert({"idX": {"content": "ours"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()

    # A foreign commit forces the reload that drops our materialized row and
    # brings the older idX back; the replay has to put ours on top again.
    await other.upsert({"idZ": {"content": "unrelated"}})
    assert await other.index_done_callback() is True

    assert await writer.index_done_callback() is True
    _assert_consistent(writer)

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("idX"))["content"] == "ours"
    assert (await reader.get_by_id("idZ"))["content"] == "unrelated"
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_after_a_failed_save_evicts_the_redo_entry(tmp_path):
    """A ``delete`` for an id whose row is applied-but-unsaved must win: the
    redo entry is evicted, or the replay would resurrect the row the delete
    just took out."""
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert "id2" in writer._unsaved_upserts

    await writer.delete(["id2"])
    assert "id2" not in writer._unsaved_upserts, "delete evicts the redo entry"

    # Force the reload path before the retry.
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    assert await writer.index_done_callback() is True
    _assert_consistent(writer)

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert await reader.get_by_id("id2") is None, "replay must not resurrect id2"
    assert (await reader.get_by_id("id1"))["content"] == "alpha"
    assert (await reader.get_by_id("id3"))["content"] == "gamma"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_entity_relation_evicts_matching_redo_upserts(tmp_path):
    """Same rule for the relation sweep: the src/tgt predicate prunes the redo
    log unconditionally, because after a foreign reload the unsaved relation
    rows are invisible to the materialized scan."""

    def make(tmp, embed):
        return FaissVectorDBStorage(
            namespace="test_relations",
            workspace="ws",
            global_config={
                "working_dir": str(tmp),
                "embedding_batch_num": 32,
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=DIM, max_token_size=512, func=embed
            ),
            meta_fields={"content", "src_id", "tgt_id"},
        )

    writer = make(tmp_path, _CountingEmbed())
    other = make(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert(
        {
            "r1": {"content": "rel1", "src_id": "A", "tgt_id": "B"},
            "r2": {"content": "rel2", "src_id": "X", "tgt_id": "Y"},
        }
    )
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert "r1" in writer._unsaved_upserts and "r2" in writer._unsaved_upserts

    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    await writer.delete_entity_relation("A")
    assert "r1" not in writer._unsaved_upserts, "incident redo entry evicted"
    assert "r2" in writer._unsaved_upserts, "unrelated redo entry preserved"

    assert await writer.index_done_callback() is True
    _assert_consistent(writer)

    reader = make(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert await reader.get_by_id("r1") is None
    assert (await reader.get_by_id("r2"))["content"] == "rel2"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_aborting_batch_keeps_the_upsert_redo_log(tmp_path):
    """``drop_pending_index_ops`` discards buffered work, not the redo log:
    the logged rows already reached ``self._index`` (the class the abort
    path intentionally does not roll back), so they must still survive a
    foreign-commit reload."""
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"id1": {"content": "alpha"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()

    await writer.drop_pending_index_ops()
    assert writer._unsaved_upserts, "abort keeps the redo log"

    await other.upsert({"id2": {"content": "beta"}})
    assert await other.index_done_callback() is True

    assert await writer.index_done_callback() is True
    _assert_consistent(writer)

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("id1"))["content"] == "alpha"
    assert (await reader.get_by_id("id2"))["content"] == "beta"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_repeated_failed_saves_do_not_rebuild_each_time(tmp_path):
    """The replay is scoped by the redo entry: with no reload in between, the
    stored row still fingerprints equal to the logged record, so a retry has
    nothing to redo. A fingerprint regression here would rebuild the whole
    index on every retry — the O(rows) cost the deferred protocol exists to
    avoid."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()

    rebuilds: list[tuple[list[int], int]] = []
    real_rebuild = storage._rebuild_index_locked

    def spy(drop_fids, add_records=()):
        rebuilds.append((list(drop_fids), len(add_records)))
        return real_rebuild(drop_fids, add_records)

    storage._rebuild_index_locked = spy

    with pytest.raises(OSError):
        await storage.index_done_callback()
    assert rebuilds == [], "no reload happened, so there is nothing to redo"

    restore()
    assert await storage.index_done_callback() is True
    assert rebuilds == [], "the durable retry must not rebuild either"
    assert not storage._unsaved_upserts
    assert embed.call_count == 1
    _assert_consistent(storage)
    assert (await storage.get_by_id("id1"))["content"] == "alpha"


# ---------------------------------------------------------------------------
# finalize must reload even when only the dirty flag is set, and a replay must
# not revert a strictly newer foreign commit (PR #3709 review follow-ups).
# Twin of the Nano coverage.
# ---------------------------------------------------------------------------


async def _strand_dirty_with_empty_buffers(writer):
    """Reach ``_index_dirty=True`` with all four buffers empty.

    A flush materializes two rows and its save fails, so both sit in the redo
    log. ``delete`` then evicts their redo entries and queues pending deletes;
    the batch aborts, and ``drop_pending_index_ops`` discards those queued
    deletes by design. Nothing now names the still-materialized rows for
    replay, which is exactly the state in which skipping the reload would save
    a stale snapshot over another writer's commit.
    """
    await writer.upsert({"idX": {"content": "ours-x"}, "idY": {"content": "ours-y"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()

    await writer.delete(["idX", "idY"])
    await writer.drop_pending_index_ops()

    assert writer._index_dirty is True
    assert not writer._pending_upserts
    assert not writer._pending_deletes
    assert not writer._unsaved_deletes
    assert not writer._unsaved_upserts


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_reloads_when_only_the_dirty_flag_is_set(tmp_path):
    """Fix-proof: ``finalize`` gated its reload on the four buffers, so a
    bare ``_index_dirty`` saved the pre-commit snapshot without reloading —
    dropping the foreign row and persisting the two explicitly deleted ones.
    """
    writer = _make_storage(tmp_path, _CountingEmbed())
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await _strand_dirty_with_empty_buffers(writer)

    await other.upsert({"foreign": {"content": "theirs"}})
    assert await other.index_done_callback() is True

    await writer.finalize()

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    foreign = await reader.get_by_id("foreign")
    assert foreign is not None, (
        "finalize saved its stale pre-commit snapshot over the foreign commit"
    )
    assert foreign["content"] == "theirs"
    assert await reader.get_by_id("idX") is None, "explicitly deleted row resurrected"
    assert await reader.get_by_id("idY") is None, "explicitly deleted row resurrected"
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_replay_declines_a_strictly_newer_foreign_row(tmp_path):
    """Fix-proof: the replay overwrote whatever row the id had, so a writer
    that was superseded (its document reprocessed by a new writer under the
    same content-hash id) reverted that newer commit on its next flush."""
    writer = _make_storage(tmp_path, _CountingEmbed())
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"idX": {"content": "ours-stale"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert "idX" in writer._unsaved_upserts

    # The superseding writer commits a strictly newer row under the same id.
    with _frozen_clock(int(time.time()) + 60):
        await other.upsert({"idX": {"content": "theirs-newer"}})
    assert await other.index_done_callback() is True

    assert await writer.index_done_callback() is True
    assert "idX" not in writer._unsaved_upserts, (
        "a superseded redo entry must be dropped, or it is rescanned by every "
        "later flush and keeps poisoning the read paths"
    )

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("idX"))["content"] == "theirs-newer", (
        "the replay reverted a strictly newer commit"
    )
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_prefer_a_strictly_newer_foreign_row(tmp_path):
    """Fix-proof: the read paths short-circuited on the redo log before
    consulting the index, so they reported a row the replay is about to
    decline to restore."""
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    await writer.upsert({"idX": {"content": "ours-stale"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()

    with _frozen_clock(int(time.time()) + 60):
        await other.upsert({"idX": {"content": "theirs-newer"}})
    assert await other.index_done_callback() is True

    got = await writer.get_by_id("idX")
    assert got is not None and got["content"] == "theirs-newer"
    by_ids = await writer.get_by_ids(["idX"])
    assert by_ids[0] is not None and by_ids[0]["content"] == "theirs-newer"
    vectors = await writer.get_vectors_by_ids(["idX"])
    assert "idX" in vectors and len(vectors["idX"]) == DIM
    assert embed.call_count == 1, "reads must not re-embed to answer this"


def _strip_write_seq_on_disk(storage) -> None:
    """Rewrite the persisted meta rows as a pre-``__write_seq__`` store would.

    The token is stamped by ``upsert``, so a store written by an older version
    carries none. Removing it from the committed snapshot is the only way to
    reach the documented fallback: a same-second pair the token cannot order.
    """
    with open(storage._meta_file, encoding="utf-8") as f:
        snapshot = json.load(f)
    stripped = [record.pop(WRITE_SEQ_FIELD, None) for record in snapshot.values()]
    assert any(token is not None for token in stripped), (
        "the token must reach the committed snapshot, or the replay's ordering "
        "guard loses its tiebreaker across a reload"
    )
    with open(storage._meta_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_same_second_tie_is_broken_by_the_write_sequence(tmp_path):
    """Fix-proof (PR #3709 review): ordering was whole seconds only, and
    ``upsert`` stamps ``int(time.time())`` — so a superseding writer that
    committed inside the same second as the logged row tied, the strict ``>``
    fell through to "replay", and the stale redo record overwrote a durable
    newer row. ``__write_seq__`` orders the two writes inside that second."""
    writer = _make_storage(tmp_path, _CountingEmbed())
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    frozen = int(time.time())
    with _frozen_clock(frozen):
        await writer.upsert({"idX": {"content": "ours-stale"}})
        restore = _make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        # Same whole second, but written after ours: a higher write sequence.
        await other.upsert({"idX": {"content": "theirs-newer"}})
    assert await other.index_done_callback() is True

    logged = writer._unsaved_upserts["idX"].record
    resident = next(
        meta for meta in other._id_to_meta.values() if meta["__id__"] == "idX"
    )
    assert logged["__created_at__"] == resident["__created_at__"], (
        "the scenario under test is a same-second pair"
    )
    assert resident[WRITE_SEQ_FIELD] > logged[WRITE_SEQ_FIELD]

    # Reads must agree with the decision the flush is about to make.
    assert (await writer.get_by_id("idX"))["content"] == "theirs-newer"

    assert await writer.index_done_callback() is True
    assert "idX" not in writer._unsaved_upserts, (
        "a superseded redo entry must be dropped"
    )

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("idX"))["content"] == "theirs-newer", (
        "the replay reverted a row committed later in the same second"
    )
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_compare_every_duplicate_row_before_serving_a_redo_entry(tmp_path):
    """Fix-proof (PR #3709 review): the read paths resolved an id through the
    first matching fid, while the replay checks every fid and declines when
    *any* resident row is newer. In a legacy / corrupt store holding several
    rows under one id, an older first duplicate made the reads serve the redo
    record the next flush is about to refuse to restore."""
    storage = _make_storage(tmp_path, _CountingEmbed())
    other = _make_storage(tmp_path, _CountingEmbed())
    await storage.initialize()
    await other.initialize()

    # Hand-craft the corrupt state: two fids share "dup", the FIRST one older
    # than the row we are about to log, the second newer. Distinct one-hot
    # vectors — constant vectors would normalize to the same unit vector.
    matrix = np.zeros((2, DIM), dtype=np.float32)
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    storage._index.add(matrix)
    storage._id_to_meta[0] = {
        "__id__": "dup",
        "__created_at__": 1,
        "content": "v1-older",
        "__vector__": matrix[0].tolist(),
    }
    storage._id_to_meta[1] = {
        "__id__": "dup",
        "__created_at__": 3,
        "content": "v2-newer",
        "__vector__": matrix[1].tolist(),
    }
    assert await storage.index_done_callback() is True

    # Our write lands between the two, and its save fails -> redo entry.
    with _frozen_clock(2):
        await storage.upsert({"dup": {"content": "ours-stale"}})
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()
    restore()
    assert "dup" in storage._unsaved_upserts

    # A foreign commit forces the reload that brings both duplicates back.
    await other.upsert({"unrelated": {"content": "other"}})
    assert await other.index_done_callback() is True

    got = await storage.get_by_id("dup")
    fids = storage._find_faiss_ids_by_custom_id("dup")
    assert len(fids) == 2, "the scenario under test is a duplicated id"
    assert storage._id_to_meta[fids[0]]["__created_at__"] == 1, (
        "the first matching fid must be the older row, or the old first-match "
        "lookup would have agreed with the replay by accident"
    )
    assert got["content"] == "v2-newer", (
        "a newer duplicate blocks the replay, so the reads must not serve the "
        "redo record"
    )
    assert (await storage.get_by_ids(["dup"]))[0]["content"] == "v2-newer"
    vectors = await storage.get_vectors_by_ids(["dup"])
    assert vectors["dup"][1] == pytest.approx(1.0), (
        "the vector must come from the resident row that blocks the replay"
    )

    # And the flush does exactly what the reads reported.
    assert await storage.index_done_callback() is True
    assert "dup" not in storage._unsaved_upserts
    assert (await storage.get_by_id("dup"))["content"] == "v2-newer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_same_second_tie_without_a_write_sequence_still_replays(tmp_path):
    """Stability: a row written before ``__write_seq__`` existed carries none,
    and a missing token must not read as "older" (that would let the replay
    overwrite any legacy row). Such a same-second pair stays a tie and keeps
    the pre-token behavior — the replay proceeds."""
    writer = _make_storage(tmp_path, _CountingEmbed())
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    frozen = int(time.time())
    with _frozen_clock(frozen):
        await writer.upsert({"idX": {"content": "ours"}})
        restore = _make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        await other.upsert({"idX": {"content": "theirs"}})
        assert await other.index_done_callback() is True
        _strip_write_seq_on_disk(other)

    assert await writer.index_done_callback() is True

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("idX"))["content"] == "ours"
    _assert_consistent(reader)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_backward_clock_step_does_not_revert_a_newer_commit(
    tmp_path, monkeypatch
):
    """Fix-proof (PR #3709 review): ordering compared whole seconds first, so a
    clock stepped backward across a second boundary between our stamp and the
    superseding writer's gave the *newer* row the *smaller* ``__created_at__``
    — declared older, its higher token never read — and the stale redo row
    replayed over a durable commit."""
    writer = _make_storage(tmp_path, _CountingEmbed())
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    monkeypatch.setattr(write_seq, "_last_seq", 0)
    with patch.object(write_seq.time, "time_ns", return_value=1_000):
        with _frozen_clock(100):
            await writer.upsert({"idX": {"content": "ours-stale"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()

    # The clock steps back a full second before the superseding write.
    with patch.object(write_seq.time, "time_ns", return_value=1_001):
        with _frozen_clock(99):
            await other.upsert({"idX": {"content": "theirs-newer"}})
    assert await other.index_done_callback() is True

    logged = writer._unsaved_upserts["idX"].record
    resident = next(
        meta for meta in other._id_to_meta.values() if meta["__id__"] == "idX"
    )
    assert resident["__created_at__"] < logged["__created_at__"], (
        "the scenario under test is a backward step across a second boundary"
    )
    assert resident[WRITE_SEQ_FIELD] > logged[WRITE_SEQ_FIELD]

    assert (await writer.get_by_id("idX"))["content"] == "theirs-newer"
    assert await writer.index_done_callback() is True
    assert "idX" not in writer._unsaved_upserts

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("idX"))["content"] == "theirs-newer", (
        "the replay reverted a commit the clock step made look older"
    )
    _assert_consistent(reader)
