"""FAISS's reload fence must not depend on the notification alone.

Phase 2 of issue #3854 — the same fence as `tests/kg/nano_impl/`, plus the one
property specific to this backend: its state spans TWO files (`.index` and
`.meta.json`), so the fingerprint samples both and a change to either is a peer
commit. Cross-file atomicity is best-effort here, which is exactly why the pair
must not be readable as "unchanged" when only one of them moved.

Like nano and unlike NetworkX, a stale write is not declined: the write path
reloads the peer snapshot and replays this process's pending buffer and
`_unsaved_*` redo logs on top, so both sides survive.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("faiss")

from lightrag.kg import faiss_impl, file_fingerprint  # noqa: E402
from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture
def multiprocess(monkeypatch):
    """Pretend peer processes exist, so the file channel is armed."""
    monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)


@pytest.fixture
def lost_notification(monkeypatch):
    """Every commit lands on disk and notifies nobody."""

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(faiss_impl, "set_all_update_flags", _noop)


async def _embed(texts, **_kwargs):
    return np.array(
        [np.full(DIM, (abs(hash(t)) % 97) + 1, dtype=np.float32) for t in texts]
    )


async def _worker(tmp_path) -> FaissVectorDBStorage:
    """A storage instance on a fixed file pair — call twice for two 'workers'."""
    storage = FaissVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=_embed
        ),
        meta_fields={"content"},
    )
    await storage.initialize()
    return storage


def _ids_on_disk(worker) -> set[str]:
    """The ids a fresh reader would see — the durable state, not memory."""
    with open(worker._meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = meta["data"] if isinstance(meta, dict) and "data" in meta else meta
    if isinstance(rows, dict):
        rows = rows.values()
    return {row["__id__"] for row in rows}


@pytest.mark.asyncio
async def test_a_stale_writer_replays_instead_of_overwriting_a_peer_commit(
    tmp_path, multiprocess, lost_notification
):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert({"from_b": {"content": "b"}})

        await worker_a.upsert({"from_a": {"content": "a"}})
        assert await worker_a.index_done_callback() is True

        assert worker_b.storage_updated.value is False

        assert await worker_b.index_done_callback() is True
        assert worker_b._missed_notification_reloads == 1

        # No decline, no loss: A's row was reloaded and B's replayed on top.
        assert _ids_on_disk(worker_b) == {"from_a", "from_b"}
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_reader_picks_up_a_peer_commit_it_was_never_told_about(
    tmp_path, multiprocess, lost_notification
):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert({"from_a": {"content": "a"}})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        assert await worker_b.get_by_id("from_a") is not None
        assert worker_b._missed_notification_reloads == 1
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_an_index_newer_than_its_metadata_is_an_interrupted_publication(
    tmp_path, multiprocess
):
    """The completeness test, judged from the files alone.

    ``_save_faiss_index`` writes the index first and renames the metadata
    LAST, so the metadata is this storage's commit marker: a completed
    publication leaves it no older than the index it describes. An index
    newer than its metadata therefore means a publication that did not
    finish, and reloading it is the corruption vector — ``_load_faiss_index``
    binds in-range metadata rows to whatever vectors the index now holds.
    """
    worker = await _worker(tmp_path)
    try:
        await worker.upsert({"n1": {"content": "x"}})
        assert await worker.index_done_callback() is True
        assert worker._peer_commit_detected() is False

        # The index moves past the commit marker: an unfinished publication.
        meta_mtime = os.stat(worker._meta_file).st_mtime_ns
        os.utime(worker._faiss_index_file, ns=(meta_mtime + 1000, meta_mtime + 1000))

        assert worker._peer_commit_detected() is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_peer_generations_behind_still_rejects_a_torn_pair(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """The case a per-file "did it change" comparison cannot catch.

    Codex review on #3867: a peer that recorded pair A, missed the
    notification for a COMPLETE pair B, and then meets a half-landed C sees
    every file changed relative to A — so comparing per-file changes against
    its own recorded fingerprint takes the torn pair for a commit. The
    completeness test does not depend on what the reader recorded: the marker
    being older than the file it commits is a property of the files.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        # Generation A, which B records.
        await worker_a.upsert({"A": {"content": "a"}})
        assert await worker_a.index_done_callback() is True
        assert await worker_b.get_by_id("A") is not None
        recorded_a = worker_b._loaded_fingerprint

        # Generation B lands completely; B is never told.
        await worker_a.upsert({"B": {"content": "b"}})
        assert await worker_a.index_done_callback() is True

        # Generation C tears: its index lands, its metadata does not.
        await worker_a.upsert({"C": {"content": "c"}})
        real_atomic_write = faiss_impl.atomic_write

        def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
            if file_name == worker_a._meta_file:
                raise OSError("meta write boom")
            return real_atomic_write(file_name, write_fn, workspace, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
            with pytest.raises(OSError, match="meta write boom"):
                await worker_a.index_done_callback()

        # Both of B's recorded entries differ from what is on disk now...
        current = worker_b._stat_fingerprint()
        assert all(new != old for new, old in zip(current, recorded_a))
        # ...and it must still refuse, because the pair is torn.
        assert worker_b._peer_commit_detected() is False
        assert await worker_b.get_by_id("C") is None
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_a_peer_does_not_reload_another_workers_half_published_pair(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """The finding the failure-path adoption did not cover.

    Adopting the pair protects the writer that failed. A PEER sees the moved
    ``.index``, has no flag to tell it otherwise, and — before the
    all-files-must-move rule — read that as a peer commit: it reloaded the
    mismatched pair and, as the next writer, persisted the corruption. Codex
    review on #3867 traced it as losing untouched row B.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert({"A": {"content": "a"}, "B": {"content": "b"}})
        assert await worker_a.index_done_callback() is True
        # B adopts the complete pair by reading through it. That read is
        # itself a lost-notification recovery (the fixture publishes nothing),
        # so the counter is already 1 — what matters below is that the
        # half-published pair does not add to it.
        assert await worker_b.get_by_id("B") is not None
        recoveries = worker_b._missed_notification_reloads

        # A deletes A and fails the metadata half of its save — the commit
        # marker never lands, so the publication is visibly unfinished.
        await worker_a.delete(["A"])
        real_atomic_write = faiss_impl.atomic_write

        def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
            if file_name == worker_a._meta_file:
                raise OSError("meta write boom")
            return real_atomic_write(file_name, write_fn, workspace, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
            with pytest.raises(OSError, match="meta write boom"):
                await worker_a.index_done_callback()

        # The peer must NOT take the half-published pair for a commit.
        assert worker_b._peer_commit_detected() is False
        assert worker_b._missed_notification_reloads == recoveries
        # It keeps serving its own consistent snapshot.
        assert await worker_b.get_by_id("B") is not None

        # Once A's retry completes the pair, the peer does converge on it.
        assert await worker_a.index_done_callback() is True
        assert worker_b._peer_commit_detected() is True
        assert await worker_b.get_by_id("A") is None
        assert await worker_b.get_by_id("B") is not None
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_single_process_mode_ignores_a_divergent_file_pair(
    tmp_path, lost_notification
):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert({"from_a": {"content": "a"}})
        assert await worker_a.index_done_callback() is True

        assert worker_b._peer_commit_detected() is False
        assert await worker_b.get_by_id("from_a") is None
        assert worker_b._missed_notification_reloads == 0
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_drop_adopts_the_files_absence(tmp_path, multiprocess):
    worker = await _worker(tmp_path)
    try:
        await worker.upsert({"n1": {"content": "x"}})
        assert await worker.index_done_callback() is True
        assert worker._loaded_fingerprint is not None

        assert (await worker.drop())["status"] == "success"
        # Both files' ABSENCE, adopted — not ``None``, which means "nothing
        # recorded" and would order a reload on the next call.
        assert worker._loaded_fingerprint == (None, None)
        assert worker._peer_commit_detected() is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_partial_two_file_save_is_not_read_as_a_peer_commit(
    tmp_path, multiprocess, monkeypatch
):
    """The one hazard the fence adds to a TWO-file backend.

    ``_write_both`` is two ``atomic_write`` calls, so a failure between them
    publishes a MISMATCHED pair: a new ``.index`` beside the previous
    ``.meta.json``. The pair's fingerprint has moved, so a fence that took that
    at face value would reload this process's own half-finished write as if it
    were a peer commit — replacing the complete in-memory snapshot with a pair
    that binds one row's metadata to another's vector, after which the redo
    replay deletes the wrong vector and the next save makes it permanent.

    Codex review on #3867 traced it as: delete A from ``[A, B]`` and lose B.
    """
    worker = await _worker(tmp_path)
    try:
        await worker.upsert({"A": {"content": "a"}, "B": {"content": "b"}})
        assert await worker.index_done_callback() is True
        assert _ids_on_disk(worker) == {"A", "B"}

        # Delete A, then fail the metadata half of the save.
        await worker.delete(["A"])

        real_atomic_write = faiss_impl.atomic_write

        def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
            if file_name == worker._meta_file:
                raise OSError("meta write boom")
            return real_atomic_write(file_name, write_fn, workspace, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
            with pytest.raises(OSError, match="meta write boom"):
                await worker.index_done_callback()

        # The pair on disk is this process's own partial write, so the fence
        # must NOT offer to reload it.
        assert worker._peer_commit_detected() is False

        # The retry writes both files from the snapshot still held in memory:
        # A is gone, and B — which the operation never touched — survives.
        assert await worker.index_done_callback() is True
        assert _ids_on_disk(worker) == {"B"}
        assert await worker.get_by_id("B") is not None
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_torn_pair_inside_one_timestamp_tick_is_still_refused(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """The completeness test must not depend on the clock's resolution.

    Codex review on #3867: on a coarse-granularity filesystem the next
    publication's index can land in the same timestamp tick as the previous
    one's metadata, making the two mtimes equal. A "no older than" test calls
    that complete and hands back a pair whose files describe different
    generations. The comparison is therefore STRICT.

    The tick is forced with ``os.utime`` rather than hoped for, so this holds
    on the fine-grained filesystems CI runs on too.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert({"A": {"content": "a"}, "B": {"content": "b"}})
        assert await worker_a.index_done_callback() is True
        assert await worker_b.get_by_id("B") is not None

        # Tear the pair: the index lands, the metadata does not.
        await worker_a.delete(["A"])
        real_atomic_write = faiss_impl.atomic_write

        def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
            if file_name == worker_a._meta_file:
                raise OSError("meta write boom")
            return real_atomic_write(file_name, write_fn, workspace, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
            with pytest.raises(OSError, match="meta write boom"):
                await worker_a.index_done_callback()

        # Collapse the two writes into one tick, as a coarse clock would.
        meta_mtime = os.stat(worker_b._meta_file).st_mtime_ns
        os.utime(worker_b._faiss_index_file, ns=(meta_mtime, meta_mtime))
        assert (
            os.stat(worker_b._faiss_index_file).st_mtime_ns
            == os.stat(worker_b._meta_file).st_mtime_ns
        )

        # The pair still differs from what the peer recorded (the index's size
        # moved), so only the completeness test can refuse it.
        assert worker_b._stat_fingerprint() != worker_b._loaded_fingerprint
        assert worker_b._peer_commit_detected() is False
        assert await worker_b.get_by_id("B") is not None
    finally:
        await worker_a.finalize()
        await worker_b.finalize()
