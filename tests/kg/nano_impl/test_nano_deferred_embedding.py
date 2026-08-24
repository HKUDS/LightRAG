"""Deferred-embedding coverage for ``NanoVectorDBStorage``.

The storage no longer embeds eagerly in ``upsert``: it buffers a pending doc
and embeds once per id at flush time (``index_done_callback`` / ``finalize``).
These tests pin that contract using a counting mock embedding function — no
live model or network. They mirror the protocol proven for
``OpenSearchVectorDBStorage`` (issue #2785).
"""

import contextlib
import json
import time
from unittest.mock import patch

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")  # noqa: F841

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
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
    with patch("lightrag.kg.nano_vector_db_impl.time.time", return_value=stamp):
        yield


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
async def test_delete_cancels_pending_and_defers_materialized_removal(tmp_path):
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    # Materialize id1; leave id2 only as a pending (unflushed) upsert.
    await storage.upsert({"id1": {"content": "alpha"}})
    await storage.index_done_callback()
    await storage.upsert({"id2": {"content": "beta"}})

    await storage.delete(["id1", "id2"])

    assert "id2" not in storage._pending_upserts, "delete cancels pending upsert"
    # Materialized removal is deferred: the row stays in the client until the
    # next flush (one batched np.delete per flush instead of one full matrix
    # copy per call), but reads must already report the id as absent.
    assert len(storage._client) == 1, "materialized removal deferred to flush"
    assert await storage.get_by_id("id1") is None
    assert await storage.get_by_id("id2") is None

    await storage.index_done_callback()
    assert len(storage._client) == 0, "flush applies the queued delete"


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
async def test_delete_on_stale_client_applies_after_reload_at_flush(tmp_path):
    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    stale_deleter = _make_storage(tmp_path, embed)
    await writer.initialize()
    await stale_deleter.initialize()

    await writer.upsert({"id1": {"content": "alpha"}})
    assert await writer.index_done_callback() is True
    assert stale_deleter.storage_updated.value is True

    # delete() only queues the id, so the stale flag stays set; the flush
    # reloads the latest on-disk snapshot first and then applies the queued
    # delete, landing it on the row the writer committed.
    await stale_deleter.delete(["id1"])
    assert stale_deleter.storage_updated.value is True
    assert await stale_deleter.index_done_callback() is True
    assert stale_deleter.storage_updated.value is False

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
async def test_delete_entity_relation_cancels_pending(tmp_path):
    embed = _CountingEmbed()
    storage = NanoVectorDBStorage(
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
    assert len(storage._client) == 0, "materialized A->B removed"


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
    storage = NanoVectorDBStorage(
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
    assert len(storage._client) == 0, "nothing materialized on embed failure"
    # Embed failure happens before self._client.upsert in _flush_pending_locked,
    # so _client_dirty must NOT be set. (A save-stage failure would leave it True
    # — see test_finalize_retries_save_after_flush_failure.)
    assert storage._client_dirty is False


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
    assert storage._client_dirty is False


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


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_index_ops_clears_buffer(tmp_path):
    """An internal-error abort calls drop_pending_index_ops to discard the
    not-yet-flushed buffer without materializing anything."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
    assert storage._pending_upserts, "upsert buffers, does not flush"

    await storage.drop_pending_index_ops()

    assert storage._pending_upserts == {}
    assert embed.call_count == 0, "drop must not embed"
    assert len(storage._client) == 0, "nothing was materialized"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_does_not_rollback_materialized(tmp_path):
    """drop_pending_index_ops discards ONLY the pending buffer; records already
    materialized into self._client by a flush whose save then failed
    (``_client_dirty=True``) are intentionally NOT rolled back."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    # Flush id1 into the in-memory client, then fail the save so it stays
    # materialized-but-unsaved (dirty) and the pending buffer is emptied.
    await storage.upsert({"id1": {"content": "alpha"}})

    def fail_save():
        raise OSError("save boom")

    storage._save_to_disk_locked = fail_save
    with pytest.raises(OSError, match="save boom"):
        await storage.index_done_callback()
    assert storage._pending_upserts == {}, "flush succeeded so pending is empty"
    assert storage._client_dirty is True
    assert len(storage._client) == 1, "id1 materialized, not saved"

    # A new pending op arrives, then the batch aborts and drops pending.
    await storage.upsert({"id2": {"content": "beta"}})
    assert "id2" in storage._pending_upserts

    await storage.drop_pending_index_ops()

    assert storage._pending_upserts == {}, "pending id2 dropped"
    assert len(storage._client) == 1, "materialized id1 NOT rolled back"
    assert storage._client_dirty is True, "still dirty for a later save retry"


# ---------------------------------------------------------------------------
# Upsert redo log: materialized-but-unsaved rows survive a foreign-commit
# reload (issue #3688). Mirrors the _unsaved_deletes coverage in
# test_nano_deferred_delete.py.
# ---------------------------------------------------------------------------


def _make_save_fail(storage):
    """Make ``_save_to_disk_locked`` raise; returns a restore callable."""
    original = storage._save_to_disk_locked

    def boom():
        raise OSError("disk full")

    storage._save_to_disk_locked = boom

    def restore():
        storage._save_to_disk_locked = original

    return restore


@pytest.mark.offline
@pytest.mark.asyncio
async def test_index_done_replays_unsaved_upsert_after_foreign_commit(tmp_path):
    """Fix-proof for issue #3688 (the retry-via-callback half).

    Chain: flush materializes id2 and its save fails -> another writer
    commits id3 -> the retry's reload replaces ``self._client`` with the
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

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("id2"))["content"] == "beta", (
        "the reload dropped the materialized-but-unsaved row and nothing "
        "replayed it (issue #3688)"
    )
    assert (await reader.get_by_id("id3"))["content"] == "gamma"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_serve_an_unsaved_upsert_after_a_foreign_reload(tmp_path):
    """Read-your-writes across the failed-save window: a foreign commit makes
    the next ``_get_client`` reload drop the unsaved row from ``self._client``,
    so the read paths must serve it from the redo log until the replay."""
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
    and the replay puts it back on top of the reloaded snapshot.

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

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("idX"))["content"] == "ours"
    assert (await reader.get_by_id("idZ"))["content"] == "unrelated"


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

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert await reader.get_by_id("id2") is None, "replay must not resurrect id2"
    assert (await reader.get_by_id("id1"))["content"] == "alpha"
    assert (await reader.get_by_id("id3"))["content"] == "gamma"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_entity_evicts_the_upsert_redo_entry(tmp_path):
    """The eager ``delete_entity`` path evicts unconditionally: after a
    foreign reload the unsaved row is not in ``self._client`` (nothing for
    the materialized delete to hit), yet a surviving redo entry would replay
    it at the next flush."""
    from lightrag.utils import compute_mdhash_id

    embed = _CountingEmbed()
    writer = _make_storage(tmp_path, embed)
    other = _make_storage(tmp_path, _CountingEmbed())
    await writer.initialize()
    await other.initialize()

    entity_id = compute_mdhash_id("EntityA", prefix="ent-")
    await writer.upsert({entity_id: {"content": "entity row"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert entity_id in writer._unsaved_upserts

    # Foreign commit first, so delete_entity's own reload drops the row
    # before its materialized-side lookup runs (`existing` is empty).
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    await writer.delete_entity("EntityA")
    assert entity_id not in writer._unsaved_upserts

    assert await writer.index_done_callback() is True

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert await reader.get_by_id(entity_id) is None
    assert (await reader.get_by_id("id3"))["content"] == "gamma"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_entity_relation_evicts_matching_redo_upserts(tmp_path):
    """Same rule for the relation sweep: the src/tgt predicate prunes the redo
    log unconditionally, because after a foreign reload the unsaved relation
    rows are invisible to the materialized scan."""

    def make(tmp, embed):
        return NanoVectorDBStorage(
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

    reader = make(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert await reader.get_by_id("r1") is None
    assert (await reader.get_by_id("r2"))["content"] == "rel2"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_aborting_batch_keeps_the_upsert_redo_log(tmp_path):
    """``drop_pending_index_ops`` discards buffered work, not the redo log:
    the logged rows already reached ``self._client`` (the class the abort
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

    reader = _make_storage(tmp_path, _CountingEmbed())
    await reader.initialize()
    assert (await reader.get_by_id("id1"))["content"] == "alpha"
    assert (await reader.get_by_id("id2"))["content"] == "beta"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_repeated_failed_saves_do_not_re_upsert_each_time(tmp_path):
    """The replay is scoped by the redo entry: with no reload in between, the
    stored row still fingerprints equal to the logged record, so a retry has
    nothing to redo. A fingerprint regression here (e.g. digesting the record
    after ``__vector__`` is re-attached) would silently rewrite every logged
    row on every retry."""
    embed = _CountingEmbed()
    storage = _make_storage(tmp_path, embed)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "alpha"}})
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()

    upsert_calls: list[list[str]] = []
    real_upsert = storage._client.upsert

    def spy(datas):
        upsert_calls.append([d["__id__"] for d in datas])
        return real_upsert(datas=datas)

    storage._client.upsert = spy

    with pytest.raises(OSError):
        await storage.index_done_callback()
    assert upsert_calls == [], "no reload happened, so there is nothing to redo"

    restore()
    assert await storage.index_done_callback() is True
    assert upsert_calls == [], "the durable retry must not rewrite either"
    assert not storage._unsaved_upserts
    assert embed.call_count == 1
    assert (await storage.get_by_id("id1"))["content"] == "alpha"


# ---------------------------------------------------------------------------
# finalize must reload even when only the dirty flag is set, and a replay must
# not revert a strictly newer foreign commit (PR #3709 review follow-ups).
# ---------------------------------------------------------------------------


async def _strand_dirty_with_empty_buffers(writer):
    """Reach ``_client_dirty=True`` with all four buffers empty.

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

    assert writer._client_dirty is True
    assert not writer._pending_upserts
    assert not writer._pending_deletes
    assert not writer._unsaved_deletes
    assert not writer._unsaved_upserts


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_reloads_when_only_the_dirty_flag_is_set(tmp_path):
    """Fix-proof: ``finalize`` gated its reload on the four buffers, so a
    bare ``_client_dirty`` saved the pre-commit snapshot without reloading —
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


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_prefer_a_strictly_newer_foreign_row(tmp_path):
    """Fix-proof: the read paths short-circuited on the redo log before
    consulting the client, so they reported a row the replay is about to
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
    """Rewrite the persisted rows as a pre-``__write_seq__`` store would.

    The token is stamped by ``upsert``, so a store written by an older version
    carries none. Removing it from the committed snapshot is the only way to
    reach the documented fallback: a same-second pair the token cannot order.
    """
    with open(storage._client_file_name, encoding="utf-8") as f:
        snapshot = json.load(f)
    stripped = [row.pop(WRITE_SEQ_FIELD, None) for row in snapshot["data"]]
    assert any(token is not None for token in stripped), (
        "the token must reach the committed snapshot, or the replay's ordering "
        "guard loses its tiebreaker across a reload"
    )
    with open(storage._client_file_name, "w", encoding="utf-8") as f:
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
    resident = other._client.get(["idX"])[0]
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
    resident = other._client.get(["idX"])[0]
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
