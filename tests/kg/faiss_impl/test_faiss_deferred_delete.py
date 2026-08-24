"""Deferred-delete coverage for ``FaissVectorDBStorage`` (#3681).

``delete`` queues ids in ``_pending_deletes`` instead of rebuilding the
``IndexFlatIP`` per call; ``_flush_pending_locked`` applies the whole queue in
one batched rebuild. The entity/relation merge stage deletes the stale
forward/reverse rows once per relation, so the eager path rebuilt the whole
index per merged relation — quadratic ingestion, the same defect #3679
profiled on Nano (#3680 fixed it there). Deferring also keeps a delete from
being lost when ``index_done_callback`` reloads a newer on-disk snapshot from
another writer.

The second half of this file covers the ``_unsaved_deletes`` redo log (the
same protocol #3680 ships for Nano): a removal that reached ``self._index``
but not the disk is replayed after any reload, so a failed save followed by
another writer's commit cannot silently resurrect the row — and the replay
matches on a row *fingerprint*, never the bare id, so a newer row another
writer published under the same content-hash id survives the replay.
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
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
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.embedded_texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.embedded_texts.extend(texts)
        return np.array(
            [
                np.full(self.dim, (abs(hash(t)) % 97) + 1, dtype=np.float32)
                for t in texts
            ]
        )


async def _make_storage(
    tmp_path, embed: _CountingEmbed, meta_fields: set[str] | None = None
) -> FaissVectorDBStorage:
    storage = FaissVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=DIM, max_token_size=512, func=embed),
        meta_fields=meta_fields or {"content"},
    )
    await storage.initialize()
    return storage


async def _upsert_and_flush(storage: FaissVectorDBStorage, data: dict) -> None:
    await storage.upsert(data)
    assert await storage.index_done_callback() is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_is_deferred_until_one_batched_flush(tmp_path, monkeypatch):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(
        storage,
        {f"rel-{i}": {"content": f"relation {i}"} for i in range(4)},
    )
    assert len(storage._id_to_meta) == 4

    rebuild_calls = []
    original_rebuild = storage._remove_faiss_ids_locked

    def counting_rebuild(fid_list):
        rebuild_calls.append(list(fid_list))
        return original_rebuild(fid_list)

    monkeypatch.setattr(storage, "_remove_faiss_ids_locked", counting_rebuild)

    # The merge-stage shape: one delete call per relation.
    for i in range(4):
        await storage.delete([f"rel-{i}", f"rel-{i}-reverse"])

    assert rebuild_calls == [], "delete must not rebuild the index eagerly"
    assert len(storage._id_to_meta) == 4, "materialized rows must survive until flush"

    assert await storage.index_done_callback() is True

    assert len(rebuild_calls) == 1, "the whole queued batch is one rebuild"
    assert len(storage._id_to_meta) == 0
    assert await storage.get_by_ids([f"rel-{i}" for i in range(4)]) == [None] * 4


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_cancels_pending_upsert_without_embedding(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)

    await storage.upsert({"doomed": {"content": "never embedded"}})
    await storage.delete(["doomed"])

    assert await storage.index_done_callback() is True
    assert "doomed" not in embed.embedded_texts, (
        "a queued delete must cancel the buffered upsert before embedding"
    )
    assert await storage.get_by_id("doomed") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_fresh_upsert_supersedes_queued_delete(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"row": {"content": "old"}})

    await storage.delete(["row"])
    await storage.upsert({"row": {"content": "new"}})
    assert await storage.index_done_callback() is True

    record = await storage.get_by_id("row")
    assert record is not None
    assert record["content"] == "new"
    # Exactly one materialized row for the custom id.
    matches = [m for m in storage._id_to_meta.values() if m.get("__id__") == "row"]
    assert len(matches) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_read_your_writes_hides_queued_deletes(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(
        storage, {"a": {"content": "alpha"}, "b": {"content": "beta"}}
    )

    await storage.delete(["a"])

    assert await storage.get_by_id("a") is None
    got = await storage.get_by_ids(["a", "b"])
    assert got[0] is None
    assert got[1] is not None and got[1]["content"] == "beta"
    assert "a" not in await storage.get_vectors_by_ids(["a", "b"])
    assert "b" in await storage.get_vectors_by_ids(["a", "b"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_applies_deletes_without_upserts(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"x": {"content": "ex"}})

    await storage.delete(["x"])
    await storage.finalize()

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("x") is None, (
        "finalize alone must persist a queued delete to disk"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_dequeued_delete_survives_reload_from_disk(tmp_path):
    """The lost-delete-across-writers case: a queued id applies after the
    unconditional reload inside index_done_callback, so a concurrent
    writer's commit cannot resurrect the row."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"shared": {"content": "v1"}})

    # Another process commits a newer snapshot...
    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(
        other, {"shared": {"content": "v2"}, "extra": {"content": "e"}}
    )
    # ...and this process's storage_updated flag flips, as set_all_update_flags
    # would have done, so index_done_callback reloads before flushing.
    storage.storage_updated.value = True

    await storage.delete(["shared"])
    assert await storage.index_done_callback() is True

    assert await storage.get_by_id("shared") is None, (
        "the queued delete must win over the reloaded snapshot"
    )
    assert (await storage.get_by_id("extra")) is not None, (
        "the other writer's row must be preserved"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_index_ops_discards_queued_deletes(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"keep": {"content": "kept"}})

    await storage.delete(["keep"])
    await storage.drop_pending_index_ops()
    assert await storage.index_done_callback() is True

    assert (await storage.get_by_id("keep")) is not None, (
        "an aborted batch must discard its queued deletes"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_failed_delete_rebuild_keeps_tombstones_for_retry(tmp_path, monkeypatch):
    """A FAISS rebuild failure during the delete pass must not consume the
    queued tombstones: the next ``index_done_callback`` retry re-applies them
    instead of persisting the stale rows as if they were deleted."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"x": {"content": "ex"}, "y": {"content": "wy"}})
    assert len(storage._id_to_meta) == 2

    await storage.delete(["x", "y"])

    def failing_rebuild(fid_list):
        raise RuntimeError("faiss rebuild boom")

    monkeypatch.setattr(storage, "_remove_faiss_ids_locked", failing_rebuild)

    with pytest.raises(RuntimeError, match="faiss rebuild boom"):
        await storage.index_done_callback()

    assert storage._pending_deletes == {"x", "y"}, (
        "tombstones must survive a failed rebuild so the retry re-applies them"
    )
    assert storage._unsaved_deletes == {}, (
        "a failed rebuild must not record removals that never happened"
    )
    assert len(storage._id_to_meta) == 2, "rows must still be materialized"
    assert storage._index_dirty is False, "nothing landed, so nothing to save"

    monkeypatch.undo()

    # The retry succeeds and the deletes land exactly once.
    assert await storage.index_done_callback() is True
    assert storage._pending_deletes == set()
    assert len(storage._id_to_meta) == 0
    assert await storage.get_by_ids(["x", "y"]) == [None, None]

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_ids(["x", "y"]) == [None, None], (
        "the retried deletes must be persisted to disk"
    )


def _fail_save(storage, monkeypatch):
    """Make the next save(s) raise, simulating a transient IO failure."""

    def boom():
        raise OSError("transient disk error")

    monkeypatch.setattr(storage, "_save_faiss_index", boom)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_survives_a_concurrent_commit_by_another_writer(
    tmp_path, monkeypatch
):
    """The redo-log case: a delete that was APPLIED but whose save failed
    must survive the unconditional reload inside the retry
    ``index_done_callback``. Without ``_unsaved_deletes`` the reload replaces
    the index with the on-disk snapshot, the row silently resurrects, and
    the retry save persists it."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"victim": {"content": "to be deleted"}})

    # Flush applies the delete, the save fails: applied-but-unsaved.
    await storage.delete(["victim"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    assert storage._pending_deletes == set(), "the queue was consumed by the flush"
    assert "victim" in storage._unsaved_deletes, (
        "the applied removal must be recorded in the redo log"
    )
    monkeypatch.undo()

    # Another writer commits, flipping this process's storage_updated flag.
    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(other, {"extra": {"content": "unrelated"}})
    assert storage.storage_updated.value is True

    # The retry reloads (resurrecting the row in memory), replays the log,
    # and persists the removal.
    assert await storage.index_done_callback() is True
    assert await storage.get_by_id("victim") is None
    assert storage._unsaved_deletes == {}, "a landed save clears the redo log"

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("victim") is None, (
        "the delete must not resurrect on disk"
    )
    assert (await reloaded.get_by_id("extra")) is not None, (
        "the other writer's row must be preserved"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_replay_preserves_a_newer_row_under_the_same_id(tmp_path, monkeypatch):
    """A replay matches on the row fingerprint, never the bare id: when
    another writer publishes a NEW row under an id whose removal is in the
    redo log, the replay must keep it — ids are content hashes, so this is a
    legitimate successor row, not the one we removed."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"shared": {"content": "v1"}})

    await storage.delete(["shared"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    monkeypatch.undo()

    # Another writer publishes a successor row under the same id.
    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(other, {"shared": {"content": "v2"}})
    assert storage.storage_updated.value is True

    # Read paths hide only the row the redo entry names — the successor is
    # already readable (the read reload pulls it in) before any flush.
    record = await storage.get_by_id("shared")
    assert record is not None and record["content"] == "v2", (
        "a redo entry must not hide a successor row it would not replay over"
    )

    assert await storage.index_done_callback() is True
    record = await storage.get_by_id("shared")
    assert record is not None and record["content"] == "v2", (
        "the replay must not remove the successor row"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_reloads_before_retrying_a_delete_only_save(
    tmp_path, monkeypatch
):
    """A delete-only dirty state is replayable, so finalize reloads before
    the retry save. Skipping the reload would write this process's
    pre-commit snapshot over rows another writer committed meanwhile."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"victim": {"content": "to be deleted"}})

    await storage.delete(["victim"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    monkeypatch.undo()

    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(other, {"extra": {"content": "committed meanwhile"}})

    await storage.finalize()

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert (await reloaded.get_by_id("extra")) is not None, (
        "the delete-only retry save must not clobber the other writer's row"
    )
    assert await reloaded.get_by_id("victim") is None, (
        "the removal must still be persisted"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_abort_keeps_the_redo_log(tmp_path, monkeypatch):
    """``drop_pending_index_ops`` discards buffered work, not removals that
    already reached the index: the redo log survives an aborting batch so a
    later reload still cannot resurrect the applied delete."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"victim": {"content": "to be deleted"}})

    await storage.delete(["victim"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    monkeypatch.undo()

    await storage.drop_pending_index_ops()
    assert "victim" in storage._unsaved_deletes, (
        "an abort must not discard the redo log"
    )

    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(other, {"extra": {"content": "unrelated"}})

    await storage.finalize()
    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("victim") is None
    assert (await reloaded.get_by_id("extra")) is not None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_eager_entity_relation_removal_survives_reload(tmp_path, monkeypatch):
    """``delete_entity_relation`` stays eager, so its removals are
    applied-but-unsaved the moment it returns — they must be in the redo log
    or the same failed-save + reload sequence resurrects them."""
    embed = _CountingEmbed()
    storage = await _make_storage(
        tmp_path, embed, meta_fields={"content", "src_id", "tgt_id"}
    )
    await _upsert_and_flush(
        storage,
        {"rel-1": {"content": "r1", "src_id": "E", "tgt_id": "F"}},
    )

    await storage.delete_entity_relation("E")
    assert "rel-1" in storage._unsaved_deletes, (
        "eager removals must be recorded in the redo log"
    )
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    monkeypatch.undo()

    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(other, {"extra": {"content": "unrelated"}})

    assert await storage.index_done_callback() is True
    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("rel-1") is None, (
        "the eager removal must not resurrect on disk"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_identical_reupsert_survives_the_delete_replay(tmp_path, monkeypatch):
    """A re-upsert with the same content in the same whole second as the row a
    redo entry removed used to be indistinguishable from it, so materializing
    it had to drop the entry or the next replay would delete the fresh row.
    ``__write_seq__`` separates the two versions: the entry may stay, it names
    only the version it removed, and the re-upsert survives a replay.

    The clock is frozen so ``__created_at__`` cannot be what separates them.
    """
    import lightrag.kg.faiss_impl as faiss_impl_module

    monkeypatch.setattr(faiss_impl_module.time, "time", lambda: 1_700_000_000.0)

    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"row": {"content": "same"}})

    # Flush applies the delete but the save fails -> redo entry for "row".
    await storage.delete(["row"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    assert "row" in storage._unsaved_deletes

    # Re-upsert the same content (same frozen timestamp): a distinct row
    # version, so the entry keeps naming the removed one and hides nothing.
    await storage.upsert({"row": {"content": "same"}})
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    assert "row" in storage._unsaved_deletes, (
        "the entry names the removed version, which the re-upsert is not"
    )
    record = await storage.get_by_id("row")
    assert record is not None and record["content"] == "same"

    # The reload resurrects the removed version: the delete replay must take
    # it out again and the upsert replay must keep the re-upserted row.
    storage.storage_updated.value = True
    monkeypatch.undo()  # also unfreezes the clock; the replay stamps nothing
    assert await storage.index_done_callback() is True
    record = await storage.get_by_id("row")
    assert record is not None and record["content"] == "same"
    assert len(storage._id_to_meta) == 1, "exactly one row per id"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_skips_save_when_flush_changes_nothing(tmp_path):
    """Queued deletes that match no row leave nothing to persist: finalize
    must not rewrite both files and broadcast a reload to every other
    process for a no-op."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"keep": {"content": "kept"}})

    watcher = await _make_storage(tmp_path, _CountingEmbed())
    assert watcher.storage_updated.value is False

    await storage.delete(["ghost"])  # matches nothing
    await storage.finalize()

    assert storage._pending_deletes == set()
    assert watcher.storage_updated.value is False, (
        "a no-op finalize must not broadcast a reload to other processes"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_show_the_duplicate_a_delete_replay_preserves(
    tmp_path, monkeypatch
):
    """Fix-proof (PR #3709 review): the read paths resolved an id through the
    first matching fid, so when a corrupt store holds several rows under one id
    and the redo entry names only some of them, a matching first duplicate hid
    the whole id — including the row the replay deliberately preserves."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"dup": {"content": "removed"}})
    removed_row = dict(storage._id_to_meta[0])

    # The delete lands on the index and the save fails: the entry names this
    # one version.
    await storage.delete(["dup"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    assert storage._unsaved_deletes["dup"] == {storage._row_fingerprint(removed_row)}
    monkeypatch.undo()

    # A reload from a corrupt snapshot: the removed version comes back at the
    # FIRST fid, beside a newer row the entry does not name.
    matrix = np.zeros((2, DIM), dtype=np.float32)
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    storage._index.add(matrix)
    storage._id_to_meta[0] = {**removed_row, "__vector__": matrix[0].tolist()}
    storage._id_to_meta[1] = {
        "__id__": "dup",
        "__created_at__": removed_row["__created_at__"] + 5,
        "content": "survivor",
        "__vector__": matrix[1].tolist(),
    }

    got = await storage.get_by_id("dup")
    assert got is not None and got["content"] == "survivor", (
        "the entry hides only the version it removed; a row it does not name "
        "took the id's place and the replay preserves it"
    )
    assert (await storage.get_by_ids(["dup"]))[0]["content"] == "survivor"
    vectors = await storage.get_vectors_by_ids(["dup"])
    assert vectors["dup"][1] == pytest.approx(1.0), (
        "the vector must come from the surviving row, not the removed one"
    )

    # And the flush does exactly what the reads reported.
    assert await storage.index_done_callback() is True
    assert storage._find_faiss_ids_by_custom_id("dup") == [0], "one row survives"
    assert (await storage.get_by_id("dup"))["content"] == "survivor"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_replay_removes_every_duplicate_row_under_one_id(tmp_path, monkeypatch):
    """``delete`` is find-all: it removes every row carrying the id, which a
    legacy / corrupt store can have several of (``_find_faiss_ids_by_custom_id``
    exists for exactly that). The redo log must therefore name every removed
    version — with one fingerprint per id the unlogged duplicates survive the
    replay and get persisted by the retry, silently undoing part of a delete
    that reported success."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)

    # Hand-craft the corrupt state: two fids share "dup", different metadata.
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
    assert await storage.index_done_callback() is True
    assert len(storage._find_faiss_ids_by_custom_id("dup")) == 2

    # Flush removes both, the save fails: both versions are applied-unsaved.
    await storage.delete(["dup"])
    _fail_save(storage, monkeypatch)
    with pytest.raises(OSError, match="transient disk error"):
        await storage.index_done_callback()
    assert storage._find_faiss_ids_by_custom_id("dup") == []
    assert len(storage._unsaved_deletes["dup"]) == 2, (
        "every removed version must be logged, not just the last one"
    )
    monkeypatch.undo()

    # Another writer commits, so the retry reloads and both rows come back.
    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(other, {"extra": {"content": "unrelated"}})

    assert await storage.index_done_callback() is True
    assert storage._find_faiss_ids_by_custom_id("dup") == [], (
        "the replay must remove every duplicate, not just one"
    )

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert reloaded._find_faiss_ids_by_custom_id("dup") == [], (
        "no duplicate may be persisted by the retry"
    )
    assert (await reloaded.get_by_id("extra")) is not None
