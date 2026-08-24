"""Deferred-delete coverage for ``NanoVectorDBStorage``.

``delete()`` no longer materializes immediately: each ``NanoVectorDB.delete()``
call rebuilds the whole matrix via ``np.delete`` (a full O(N) copy), and the
entity/relation merge stage issues one delete call per relation — on a
multi-GB matrix this dominates ingestion CPU. The storage now queues ids in
``_pending_deletes`` and applies them in ONE batched ``client.delete()`` at
flush time, strictly BEFORE pending upserts materialize.

These tests pin that contract with a counting ``NanoVectorDB.delete`` spy —
no live model or network. They mirror the deferred-embedding protocol tests
in ``test_nano_deferred_embedding.py``.
"""

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

import lightrag.kg.nano_vector_db_impl as nano_impl  # noqa: E402
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    initialize_share_data,
    finalize_share_data,
)
from lightrag.utils import EmbeddingFunc, compute_mdhash_id  # noqa: E402

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _DeterministicEmbed:
    """Async embedding callable with a per-text deterministic vector."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    async def __call__(self, texts, **kwargs):
        return np.array(
            [
                np.full(self.dim, (abs(hash(t)) % 97) + 1, dtype=np.float32)
                for t in texts
            ]
        )


def _make_storage(tmp_path) -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=_DeterministicEmbed()
        ),
        meta_fields={"content"},
    )


def _spy_client_delete(storage, counter):
    """Wrap the materialized client's ``delete`` to count invocations."""
    original = storage._client.delete

    def spy(ids):
        counter.append(list(ids))
        return original(ids)

    storage._client.delete = spy


async def _seed(storage, items: dict[str, str]):
    await storage.upsert(
        {doc_id: {"content": content} for doc_id, content in items.items()}
    )
    await storage.index_done_callback()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_defers_materialization_to_flush(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta", "id3": "gamma"})

    calls: list[list[str]] = []
    _spy_client_delete(storage, calls)

    await storage.delete(["id1"])
    await storage.delete(["id2"])
    assert calls == [], "delete() must not touch the materialized client"
    assert len(storage._client) == 3, "rows stay materialized until flush"

    await storage.index_done_callback()
    assert len(calls) == 1, "flush applies all queued deletes in ONE batch"
    assert sorted(calls[0]) == ["id1", "id2"]
    assert len(storage._client) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_deleted_then_reinserted_id_ends_with_new_row(tmp_path):
    """The merge stage deletes an id then re-upserts it: the flush must apply
    the delete BEFORE the upsert so exactly the new row survives."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "old-content", "id2": "keep"})

    await storage.delete(["id1", "missing-id"])
    await storage.upsert({"id1": {"content": "new-content"}})
    await storage.index_done_callback()

    rows = {d["__id__"]: d for d in (await storage.client_storage)["data"]}
    assert sorted(rows) == ["id1", "id2"]
    assert rows["id1"]["content"] == "new-content"
    matrix = (await storage.client_storage)["matrix"]
    assert matrix.shape == (2, DIM), "no duplicate or leftover rows"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pending_delete_reads_as_absent_on_read_your_writes_paths(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id1"])

    assert await storage.get_by_id("id1") is None
    got = await storage.get_by_ids(["id1", "id2"])
    assert got[0] is None and got[1]["content"] == "beta"
    vectors = await storage.get_vectors_by_ids(["id1", "id2"])
    assert "id1" not in vectors and "id2" in vectors


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_reads_the_materialized_index_only(tmp_path):
    """``query`` ranks the materialized index, so a queued delete is still
    returned until the flush applies it — the same contract the Qdrant and
    PostgreSQL buffers document. Filtering queued ids out of the result
    *after* top-k selection would silently shrink the result set instead.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {f"id{i}": f"doc{i}" for i in range(5)})

    await storage.delete([f"id{i}" for i in range(4)])

    hits = await storage.query("doc0", top_k=3)
    assert len(hits) == 3, "top_k must not be truncated by queued deletes"

    await storage.index_done_callback()
    hits = await storage.query("doc0", top_k=3)
    assert [h["id"] for h in hits] == ["id4"], "flush applies the deletes"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_supersedes_a_queued_delete_for_the_same_id(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "old"})

    await storage.delete(["id1"])
    await storage.upsert({"id1": {"content": "new"}})
    assert storage._pending_deletes == set(), "upsert must cancel the queued delete"

    await storage.index_done_callback()
    assert (await storage.get_by_id("id1"))["content"] == "new"
    assert len(storage._client) == 1, "no duplicate row"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_discards_queued_deletes(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    await storage.delete(["id1"])
    await storage.drop()

    assert storage._pending_deletes == set()
    assert storage._pending_upserts == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_skips_the_save_when_queued_deletes_change_nothing(tmp_path):
    """A queued delete for an id that is not in the index must not trigger a
    full-file rewrite (and the cross-process reload it broadcasts)."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    saves: list[int] = []
    original = storage._save_to_disk_locked
    storage._save_to_disk_locked = lambda: (saves.append(1), original())[1]

    await storage.delete(["never-inserted"])
    await storage.finalize()

    assert saves == [], "no-op delete must not rewrite the file"
    assert storage._pending_deletes == set()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_cancels_pending_upsert_of_same_id(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "never-lands"}})
    await storage.delete(["id1"])
    await storage.index_done_callback()

    assert len(storage._client) == 0
    assert await storage.get_by_id("id1") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_only_flush_persists_and_is_idempotent(tmp_path):
    """A flush with ONLY queued deletes (no pending upserts) must still
    materialize, save, and be idempotent on repeat."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id2"])
    await storage.index_done_callback()
    assert len(storage._client) == 1

    await storage.index_done_callback()  # nothing pending — must be a no-op
    assert len(storage._client) == 1

    # A fresh instance must observe the persisted state (save happened).
    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    assert await reloaded.get_by_id("id2") is None
    assert (await reloaded.get_by_id("id1"))["content"] == "alpha"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_survives_a_concurrent_commit_by_another_writer(tmp_path):
    """Regression: a delete must not be lost when another writer commits
    between the delete and this writer's flush.

    Eager deletion mutated ``self._client`` in place while the removal was
    still unsaved. ``index_done_callback`` then reloads unconditionally when
    another process has committed, and that reload *replaces* ``self._client``
    with the on-disk snapshot — silently resurrecting the deleted row. With
    the delete queued instead, the flush reloads first and then applies the
    queued id, so the removal survives.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()

    await _seed(writer, {"id1": "alpha", "id2": "beta"})
    await writer.delete(["id1"])

    # Another writer commits, flagging `writer` as stale.
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True
    assert writer.storage_updated.value is True

    assert await writer.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None, "delete must not be resurrected"
    assert (await reader.get_by_id("id2"))["content"] == "beta"
    assert (await reader.get_by_id("id3"))["content"] == "gamma"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_unflushed_deletes_roll_back_like_pending_upserts(tmp_path):
    """Queued deletes are in-memory only; a crash before flush drops them,
    matching the pending-upsert recovery semantics."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    await storage.delete(["id1"])  # never flushed

    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    assert (await reloaded.get_by_id("id1"))["content"] == "alpha"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_survives_a_failed_save_then_a_concurrent_commit(tmp_path):
    """Regression: the queue is retained until a save persists the removal.

    Clearing the queue as soon as the delete lands on ``self._client`` leaves
    nothing to replay if the save then fails: ``index_done_callback``'s next
    unconditional reload replaces the client with the on-disk snapshot and the
    row returns. Retaining the ids means the reload picks up the other
    writer's rows and the replay removes ours on top, so neither is lost.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()

    await _seed(writer, {"id1": "alpha", "id2": "beta"})
    await writer.delete(["id1"])

    # The flush applies the delete, then the save fails.
    original_save = writer._save_to_disk_locked
    writer._save_to_disk_locked = lambda: (_ for _ in ()).throw(OSError("disk full"))
    with pytest.raises(OSError):
        await writer.index_done_callback()
    assert set(writer._unsaved_deletes) == {"id1"}, "retained until the save lands"
    assert writer._client_dirty is True

    writer._save_to_disk_locked = original_save

    # Another writer commits, so the next flush reloads and would otherwise
    # resurrect id1.
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True
    assert writer.storage_updated.value is True

    assert await writer.index_done_callback() is True
    assert writer._unsaved_deletes == {}, "a durable save clears the redo log"

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None, "delete must not be resurrected"
    assert (await reader.get_by_id("id2"))["content"] == "beta"
    assert (await reader.get_by_id("id3"))["content"] == "gamma", (
        "the other writer's rows must survive the replay"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_queue_of_absent_ids_is_not_retained(tmp_path):
    """A flush that matched nothing on a clean client has nothing to persist,
    so it must not leave the ids resident."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    await storage.delete(["never-inserted"])
    await storage.index_done_callback()

    assert storage._pending_deletes == set()
    assert storage._unsaved_deletes == {}, "an id that matched nothing is not retained"
    assert storage._client_dirty is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_replay_does_not_remove_a_newer_row_under_the_same_id(tmp_path):
    """The redo log removes the row it removed before, not whatever is there.

    Ids are content hashes, so two writers can legitimately produce the same
    id. Deleting by id alone would destroy a row another writer committed
    after our delete. The guard compares a fingerprint of the whole stored
    row, so it holds however close together the two writes land — there is
    deliberately no sleep here, and a whole-second ``__created_at__`` would
    not survive this test.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()

    await _seed(writer, {"id1": "old"})

    original_save = writer._save_to_disk_locked
    writer._save_to_disk_locked = lambda: (_ for _ in ()).throw(OSError("disk full"))
    await writer.delete(["id1"])
    with pytest.raises(OSError):
        await writer.index_done_callback()
    writer._save_to_disk_locked = original_save

    # Another writer publishes a fresh row under the same id, inside the same
    # second as the row we removed.
    await other.upsert({"id1": {"content": "new"}})
    assert await other.index_done_callback() is True

    await writer.index_done_callback()

    reader = _make_storage(tmp_path)
    await reader.initialize()
    row = await reader.get_by_id("id1")
    assert row is not None and row["content"] == "new", (
        "a replay must not remove a row written after the delete"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_an_id_that_matched_nothing_is_never_replayed(tmp_path):
    """Ids that removed no row carry no removal, so they must not linger and
    fire against a row that appears later."""
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()
    await _seed(writer, {"seed": "s"})

    original_save = writer._save_to_disk_locked
    writer._save_to_disk_locked = lambda: (_ for _ in ()).throw(OSError("disk full"))
    await writer.upsert({"other": {"content": "o"}})
    with pytest.raises(OSError):
        await writer.index_done_callback()  # client is now dirty
    await writer.delete(["ghost"])  # matches nothing
    with pytest.raises(OSError):
        await writer.index_done_callback()
    writer._save_to_disk_locked = original_save

    assert "ghost" not in writer._unsaved_deletes

    await other.upsert({"ghost": {"content": "created later"}})
    assert await other.index_done_callback() is True
    await writer.index_done_callback()

    reader = _make_storage(tmp_path)
    await reader.initialize()
    row = await reader.get_by_id("ghost")
    assert row is not None and row["content"] == "created later"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_repeated_failed_saves_do_not_re_delete_each_time(tmp_path):
    """A retry whose rows are already gone must not rebuild the matrix again."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    calls: list[list[str]] = []
    _spy_client_delete(storage, calls)

    original_save = storage._save_to_disk_locked
    storage._save_to_disk_locked = lambda: (_ for _ in ()).throw(OSError("disk full"))
    await storage.delete(["id1"])
    for _ in range(4):
        with pytest.raises(OSError):
            await storage.index_done_callback()
    storage._save_to_disk_locked = original_save
    await storage.index_done_callback()

    assert len(calls) == 1, f"one matrix rebuild, not one per retry: {calls}"
    assert await storage.get_by_id("id1") is None


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
async def test_redo_log_reads_as_absent_while_the_save_is_pending(tmp_path):
    """A reload can bring the row back between the failed save and the replay;
    the read-your-writes paths must not resurrect it either."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id1"])
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()
    restore()
    assert set(storage._unsaved_deletes) == {"id1"}

    # Simulate another writer's commit: the next client read reloads the
    # on-disk snapshot, which still holds id1.
    storage.storage_updated.value = True

    assert await storage.get_by_id("id1") is None
    assert (await storage.get_by_ids(["id1", "id2"]))[0] is None
    assert "id1" not in await storage.get_vectors_by_ids(["id1", "id2"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_rewrite_after_a_logged_delete_survives_the_replay(tmp_path):
    """Buffering an upsert must not cancel the redo entry, and the replay must
    not remove the row that upsert wrote.

    Both halves matter and pull in opposite directions. Cancelling at
    ``upsert`` time loses the entry while the replacement row does not exist
    yet (see ``test_abort_after_a_rewrite_keeps_the_redo_entry``); keeping it
    without a row-level guard deletes the replacement on the next flush.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "old"})

    await storage.delete(["id1"])
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()
    assert set(storage._unsaved_deletes) == {"id1"}

    await storage.upsert({"id1": {"content": "new"}})
    assert set(storage._unsaved_deletes) == {"id1"}, (
        "a buffered row that may never materialize must not clear the log"
    )

    # Materialize the rewrite on a still-failing save, so the replay runs one
    # flush *after* the new row landed.
    with pytest.raises(OSError):
        await storage.index_done_callback()
    restore()
    assert await storage.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert (await reader.get_by_id("id1"))["content"] == "new"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_abort_after_a_rewrite_keeps_the_redo_entry(tmp_path):
    """An aborting batch drops the buffered rewrite, so the redo entry it
    would have superseded has to still be there.

    Cancelling the entry when the upsert is *buffered* leaves this state with
    nothing to replay: the removal is only in the unsaved client, the
    replacement row was discarded, and the next reload restores the original.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "old", "id2": "beta"})

    await storage.delete(["id1"])
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()

    await storage.upsert({"id1": {"content": "new"}})  # never materializes
    await storage.drop_pending_index_ops()
    assert set(storage._unsaved_deletes) == {"id1"}

    storage.storage_updated.value = True  # foreign commit -> reload
    restore()
    assert await storage.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None, "the removal must not be undone"
    assert (await reader.get_by_id("id2"))["content"] == "beta"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_rewrite_identical_in_content_survives_the_delete_replay(
    tmp_path, monkeypatch
):
    """A rewrite with the same content in the same whole second as the removed
    row used to be indistinguishable from it, so materializing it had to retire
    the redo entry or the next replay would delete the row just written.
    ``__write_seq__`` separates the two versions: the entry may stay, it names
    only the version it removed, and the rewrite survives a replay.

    The clock is frozen so ``__created_at__`` cannot be what separates them.
    """
    monkeypatch.setattr(nano_impl.time, "time", lambda: 1_700_000_000.0)

    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "same", "id2": "beta"})

    await storage.delete(["id1"])
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()
    assert set(storage._unsaved_deletes) == {"id1"}

    await storage.upsert({"id1": {"content": "same"}})
    with pytest.raises(OSError):
        await storage.index_done_callback()
    assert set(storage._unsaved_deletes) == {"id1"}, (
        "the entry names the removed version, which the rewrite is not"
    )
    assert (await storage.get_by_id("id1"))["content"] == "same", (
        "the rewrite must be readable — the entry hides only the removed row"
    )

    # Force the reload that resurrects the removed version before the retry:
    # the delete replay must take it out again and keep the rewrite.
    storage.storage_updated.value = True
    restore()
    assert await storage.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert (await reader.get_by_id("id1"))["content"] == "same"
    assert len(reader._client) == 2, "exactly one row per id"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_aborting_batch_keeps_the_redo_log(tmp_path):
    """``drop_pending_index_ops`` discards buffered work, but a removal that
    already reached ``self._client`` is a materialized change — the class the
    docstring declines to roll back — so its redo log entry must survive."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id1"])
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()

    await storage.drop_pending_index_ops()
    assert set(storage._unsaved_deletes) == {"id1"}, "applied removal stays replayable"

    # A foreign commit resurrects id1 on the next reload; the replay must
    # remove it again before the save.
    storage.storage_updated.value = True
    restore()
    assert await storage.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None
    assert (await reader.get_by_id("id2"))["content"] == "beta"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_eager_entity_delete_is_replayed_across_a_reload(tmp_path):
    """``delete_entity`` / ``delete_entity_relation`` stay eager, but their
    removals are applied-and-unsaved too, so they join the same redo log."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    entity_id = compute_mdhash_id("Alice", prefix="ent-")
    await _seed(storage, {entity_id: "alice", "id2": "beta"})

    await storage.delete_entity("Alice")
    assert set(storage._unsaved_deletes) == {entity_id}
    assert storage._client_dirty is True

    # A foreign commit resurrects the row on the next reload.
    storage.storage_updated.value = True
    assert await storage.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id(entity_id) is None
    assert (await reader.get_by_id("id2"))["content"] == "beta"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_reloads_before_retrying_a_delete_only_save(tmp_path):
    """A delete-only dirty client must not save over a foreign commit.

    ``finalize`` skips the reload when ``self._client`` holds unsaved rows,
    because a reload would drop them. A removal is not such a row: the redo
    log replays it after the reload. Skipping anyway wrote our pre-commit
    snapshot over the other writer's durable rows — here id3 disappeared and
    the file was left holding id2 alone.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()
    await _seed(writer, {"id1": "alpha", "id2": "beta"})

    await writer.delete(["id1"])
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert writer._client_dirty is True
    assert not writer._unsaved_upserts, "the dirty state is removals only"

    # Another writer commits after our failed save.
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    await writer.finalize()

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None, "our removal must still land"
    assert (await reader.get_by_id("id2"))["content"] == "beta"
    assert (await reader.get_by_id("id3"))["content"] == "gamma", (
        "the other writer's commit must survive finalize"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_replays_unsaved_upserts_after_a_foreign_commit(tmp_path):
    """The upsert half of the same trade-off, closed by the redo log (#3688).

    Before the log, finalize skipped the reload while ``_unsaved_upserts``
    was set (the materialized rows existed nowhere else), and its save wrote
    our pre-commit snapshot over the other writer's durable rows — id3
    disappeared. Now the log replays our rows after the reload, so both
    sides survive.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()
    await _seed(writer, {"id1": "alpha"})

    await writer.upsert({"id2": {"content": "beta"}})
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert writer._unsaved_upserts, "the flushed doc moved into the redo log"

    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True

    await writer.finalize()

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert (await reader.get_by_id("id2"))["content"] == "beta", (
        "the redo log must replay our row on top of the reloaded snapshot"
    )
    assert (await reader.get_by_id("id3"))["content"] == "gamma", (
        "the other writer's commit must survive finalize"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_expose_a_row_written_over_a_logged_delete(tmp_path):
    """A redo entry hides the row it removed, not the id.

    The replay deliberately preserves a row another writer put in place after
    our removal, so suppressing reads on id membership alone contradicted it:
    a live row stayed invisible until some later save cleared the log.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()
    await _seed(writer, {"id1": "old", "id2": "beta"})

    await writer.delete(["id1"])
    restore = _make_save_fail(writer)
    with pytest.raises(OSError):
        await writer.index_done_callback()
    restore()
    assert set(writer._unsaved_deletes) == {"id1"}

    # Another writer publishes a different row under the same id.
    await other.upsert({"id1": {"content": "replacement"}})
    assert await other.index_done_callback() is True

    assert (await writer.get_by_id("id1"))["content"] == "replacement"
    assert (await writer.get_by_ids(["id1"]))[0]["content"] == "replacement"
    assert "id1" in await writer.get_vectors_by_ids(["id1"])

    # And the replay still leaves it alone.
    assert await writer.index_done_callback() is True
    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert (await reader.get_by_id("id1"))["content"] == "replacement"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reads_still_hide_the_row_a_redo_entry_names(tmp_path):
    """The other half: while the removed row itself is back (a reload after a
    failed save), every read path must still report it as absent."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id1"])
    restore = _make_save_fail(storage)
    with pytest.raises(OSError):
        await storage.index_done_callback()
    restore()

    storage.storage_updated.value = True  # next read reloads id1 back in

    assert await storage.get_by_id("id1") is None
    assert (await storage.get_by_ids(["id1", "id2"]))[0] is None
    assert "id1" not in await storage.get_vectors_by_ids(["id1", "id2"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_queued_delete_is_scoped_to_the_id_not_the_row(tmp_path):
    """A queued delete is a *request*: the flush removes whatever row carries
    the id, as the eager call and the server-backed backends do.

    Version-scoping it would silently skip a delete the caller asked for —
    purge would leave vectors behind — and pinning the version at ``delete``
    time would put back the per-call ``O(rows)`` lookup this protocol removes.
    The redo log is version-scoped because it records a completed removal
    rather than a request; see the two tests above.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()
    await _seed(writer, {"id1": "old", "id2": "beta"})

    await writer.delete(["id1"])  # queued, nothing touched yet

    await other.upsert({"id1": {"content": "rewritten"}})
    assert await other.index_done_callback() is True

    assert await writer.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None, "the id was asked to go"
    assert (await reader.get_by_id("id2"))["content"] == "beta"
