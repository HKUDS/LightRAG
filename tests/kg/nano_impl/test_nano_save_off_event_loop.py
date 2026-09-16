"""``NanoVectorDBStorage`` must not save on the event loop.

Nano's commit is only a PARTIAL win from the offload, and these tests say so
explicitly rather than overstating it. ``NanoVectorDB.save()`` base64-encodes
the whole matrix through ``tobytes()`` + ``b64encode()`` — two single C calls
that hold the GIL wherever they run — before a cooperative ``json.dump``. So the
assertion here is "the loop advances during the write", never "the stall is
gone".

The second test is the sharper one: the save temporarily points
``client.storage_file`` at the tmp sibling, so a cancelled caller that returned
early would release ``_storage_lock`` with that swap still in place, and the
next coroutine through the lock would see a client aimed at a path that is about
to be renamed away. The old synchronous write had no such window because it
could not be cancelled at all.
"""

import asyncio
import json
import logging
import threading
import time

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

import lightrag.kg.nano_vector_db_impl as nano_impl  # noqa: E402
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

DIM = 8
BLOCK_SECONDS = 0.2


class _DeterministicEmbed:
    async def __call__(self, texts, **kwargs):
        return np.array(
            [np.full(DIM, (abs(hash(t)) % 97) + 1, dtype=np.float32) for t in texts]
        )


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _make_storage(tmp_path) -> NanoVectorDBStorage:
    storage = NanoVectorDBStorage(
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
    await storage.initialize()
    return storage


class _Heartbeat:
    def __init__(self):
        self.beats = 0
        self._stop = False
        self._task = None

    async def _run(self):
        while not self._stop:
            self.beats += 1
            await asyncio.sleep(0)

    def start(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task


async def _wait_for(predicate, *, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(0.01)


async def test_commit_keeps_the_event_loop_running(tmp_path, monkeypatch):
    """Fix-proof: called inline, ``time.sleep`` freezes the loop and beats tie."""
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    observed = []
    heartbeat = _Heartbeat().start()
    real_atomic_write = nano_impl.atomic_write

    def blocking_atomic_write(path, write_fn, workspace):
        before = heartbeat.beats
        time.sleep(BLOCK_SECONDS)
        observed.append((before, heartbeat.beats))
        return real_atomic_write(path, write_fn, workspace)

    monkeypatch.setattr(nano_impl, "atomic_write", blocking_atomic_write)

    await storage.index_done_callback()
    await heartbeat.stop()

    assert observed, "the save never ran"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole save"
    assert (await storage.get_by_id("id1"))["content"] == "alpha"


async def test_cancelled_commit_restores_storage_file(tmp_path, monkeypatch):
    """A cancelled commit must not leave the client pointing at the tmp file.

    Fix-proof: with a cancellable offload the caller returns as soon as it is
    cancelled, while the worker is still inside ``_save_atomic`` with
    ``storage_file`` swapped — so the assertion below reads the tmp path.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    entered = threading.Event()
    release = threading.Event()
    observed_during_write = []
    real_save = storage._client.save

    def blocking_save():
        # ``_save_atomic`` has already swapped storage_file to the tmp sibling
        # by the time save() runs -- this IS the state that must not be
        # observable from the loop.
        observed_during_write.append(storage._client.storage_file)
        entered.set()
        assert release.wait(timeout=5), "write was never released"
        return real_save()

    storage._client.save = blocking_save

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(entered.is_set)

    commit.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not commit.done(), "caller returned mid-save with storage_file swapped"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await commit

    assert observed_during_write, "the write body never ran"
    assert observed_during_write[0] != storage._client_file_name
    assert storage._client.storage_file == storage._client_file_name


async def test_cancelled_commit_still_notifies_and_retires_the_redo_logs(
    tmp_path, monkeypatch
):
    """A cancelled commit must not save and then skip its bookkeeping.

    Two things are stranded if it does: the other processes are never told to
    reload (``set_all_update_flags``), and the redo logs keep rows that ARE on
    disk, so a later commit replays them. Both live in the hook that
    ``commit_in_storage_io`` runs inside the write's uncancellable region.

    Fix-proof: inline the bookkeeping after the offload instead, and ``flagged``
    stays empty with ``_client_dirty`` still True — the P2 Codex raised on #3740.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    flagged: list[str] = []
    inside_save = threading.Event()
    may_finish = threading.Event()
    real_save = nano_vectordb.NanoVectorDB.save

    async def spy_set_all_update_flags(namespace, workspace=None):
        flagged.append(namespace)

    def parked_save(self):
        inside_save.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_save(self)

    monkeypatch.setattr(nano_impl, "set_all_update_flags", spy_set_all_update_flags)
    monkeypatch.setattr(nano_vectordb.NanoVectorDB, "save", parked_save)

    commit = asyncio.create_task(storage.index_done_callback())
    while not inside_save.is_set():
        await asyncio.sleep(0.01)

    commit.cancel()
    may_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await commit

    assert flagged == ["test_vectors"], (
        "a cancelled commit saved without telling the other processes to "
        f"reload (flagged={flagged})"
    )
    assert storage._client_dirty is False, "the dirty bit survived a durable save"
    assert storage._unsaved_upserts == {}, "redo log kept rows that are on disk"


async def test_a_failed_notification_is_not_reported_as_a_failed_save(
    tmp_path, monkeypatch, caplog
):
    """A publication failure must not be raised as a save failure.

    ``index_done_callback``'s contract is that a raise means the vectors were
    NOT written, and ``_insert_done`` aborts the document batch on it. But the
    hook runs only after ``atomic_write`` renamed the file into place, so an
    exception out of ``set_all_update_flags`` reports a durable write as a lost
    one.

    What failed is the cross-process reload notification. The residue heals:
    ``_client_dirty`` stays True, so the next commit rewrites this snapshot and
    notifies again.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(nano_impl, "set_all_update_flags", failing_set_all_update_flags)

    # lightrag's logger does not propagate, so caplog cannot see it otherwise.
    logger = logging.getLogger("lightrag")
    monkeypatch.setattr(logger, "propagate", True)

    with caplog.at_level(logging.ERROR, logger="lightrag"):
        committed = await storage.index_done_callback()

    assert committed is True
    with open(storage._client_file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["data"], "the save did not land, so this proves nothing"
    # The dirty bit stays set, which is what retries the publication...
    assert storage._client_dirty is True
    # ...and the redo log is NOT retired, because an unnotified peer can still
    # save its older snapshot over these rows. See the recovery test below.
    assert set(storage._unsaved_upserts) == {"id1"}
    assert any(
        "publishing that write failed" in record.getMessage()
        for record in caplog.records
    ), f"the deferred publication was not logged: {caplog.text}"


async def test_rows_lost_to_an_unnotified_peer_are_replayed_back(tmp_path, monkeypatch):
    """The recovery the retained redo log buys, end to end.

    ``other`` never learns of ``writer``'s commit, so its own commit saves a
    whole-file snapshot that has never seen ``id1`` over this file. That
    overwrite is the fence gap tracked in #3854 and no commit-status choice
    prevents it; what the retained redo log decides is whether the rows come
    back. ``writer``'s next flush reloads the foreign snapshot and replays them
    on top -- the path issue #3688 built for a failed save.

    Fix-proof: retire the redo logs before ``set_all_update_flags`` instead, and
    ``id1`` is gone from disk for good. ``FaissVectorDBStorage`` has the same
    shape; its mirror lives in tests/kg/faiss_impl/.
    """
    writer = await _make_storage(tmp_path)
    other = await _make_storage(tmp_path)

    await writer.upsert({"id1": {"content": "ours"}})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(nano_impl, "set_all_update_flags", failing_set_all_update_flags)
    assert await writer.index_done_callback() is True
    monkeypatch.undo()

    # `other` was never flagged, so it does not reload: its commit publishes a
    # whole-file snapshot in which `id1` has never existed.
    await other.upsert({"id2": {"content": "theirs"}})
    assert await other.index_done_callback() is True
    assert await other.get_by_id("id1") is None, (
        "the peer was supposed to be unaware of id1; the scenario did not set up"
    )

    # `other`'s own commit notifies `writer`, so this flush reloads its snapshot.
    assert await writer.index_done_callback() is True

    with open(writer._client_file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    ids = {row["__id__"] for row in persisted["data"]}
    assert ids == {"id1", "id2"}, (
        "the row the peer overwrote was not replayed back; the loss would be "
        f"permanent and silent (persisted={ids})"
    )


async def test_a_broken_sink_cannot_turn_a_landed_save_into_a_failure(
    tmp_path, monkeypatch
):
    """The publication-failure diagnostic is past the point of no return.

    The rows are on disk by the time that handler runs, so a logging handler,
    formatter or output target that raises must not escape it: `index_done_callback`
    would then report a durable write as one that never happened, and
    `_insert_done` marks the document FAILED and re-runs mutations that already
    landed. Same reasoning, and same remedy (`log_without_raising`), as the
    post-removal `drop` diagnostics.

    Fix-proof: call `logger.error` directly in the handler and this raises.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    def log_boom(msg):
        raise RuntimeError("log sink boom")

    monkeypatch.setattr(nano_impl, "set_all_update_flags", failing_set_all_update_flags)
    monkeypatch.setattr(nano_impl.logger, "error", log_boom)

    assert await storage.index_done_callback() is True

    with open(storage._client_file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["data"], "the save did not land, so this proves nothing"
    assert set(storage._unsaved_upserts) == {"id1"}
