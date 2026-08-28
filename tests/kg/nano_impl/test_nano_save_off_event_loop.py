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
