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

import json
import os

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
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
    assert storage._index.ntotal == 0, "delete removes the materialized row"
    assert await storage.get_by_id("id1") is None
    assert await storage.get_by_id("id2") is None
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
    assert storage._find_faiss_ids_by_custom_id("dup") == []
    assert storage._index.ntotal == 0
    _assert_consistent(storage)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_propagates_errors(tmp_path, monkeypatch):
    """Faiss ``delete`` must NOT swallow errors — the caller (document
    deletion / status update path) needs to abort if vectors weren't
    actually removed. This intentionally diverges from Nano."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})
    await storage.index_done_callback()

    def boom(_self, _fids):
        raise RuntimeError("rebuild boom")

    # _remove_faiss_ids_locked is what delete calls under the hood.
    monkeypatch.setattr(
        FaissVectorDBStorage, "_remove_faiss_ids_locked", boom, raising=True
    )

    with pytest.raises(RuntimeError, match="rebuild boom"):
        await storage.delete(["id1"])
