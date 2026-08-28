"""``FaissVectorDBStorage`` persistence must not block the event loop.

``_save_faiss_index`` rewrites the whole ``.index`` and the whole
``.meta.json``. Run inline on the loop, a large index froze every unrelated
request for the duration — the same failure mode the JSON and GraphML backends
had. Measured on faiss 1.14.2 both halves genuinely yield: ``faiss.write_index``
releases the GIL, and ``json.dump`` is the cooperative pure-Python encoder.

Two properties are pinned here:

* the loop keeps running while the files are written;
* the ``_id_to_meta`` snapshot is taken on the loop BEFORE the offload, so the
  worker never iterates a dict that unlocked readers/writers can still touch;
* a cancelled commit still runs its bookkeeping, so the two published files are
  never left with the other processes unaware of them.
"""

import asyncio
import json
import os
import threading
import time

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg import faiss_impl  # noqa: E402
from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(tmp_path, namespace: str = "vectors") -> FaissVectorDBStorage:
    def _unused(*_args, **_kwargs):  # pragma: no cover — never called here
        raise AssertionError("embedding_func must not be invoked by save path")

    return FaissVectorDBStorage(
        namespace=namespace,
        workspace="",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 1,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=4, func=_unused),
    )


async def _noop() -> None:
    """Post-commit hook stand-in for the tests that call the save directly."""
    return None


def _seed(storage: FaissVectorDBStorage, marker: str) -> None:
    storage._index.add(np.ones((1, 4), dtype=np.float32))
    storage._id_to_meta = {
        storage._index.ntotal - 1: {"__id__": marker, "content": marker}
    }


class _Heartbeat:
    """Counts how often the event loop gets to run us."""

    def __init__(self):
        self.beats = 0
        self._stop = False
        self._task = None

    async def __aenter__(self):
        async def _run():
            while not self._stop:
                self.beats += 1
                await asyncio.sleep(0)

        self._task = asyncio.create_task(_run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_exc):
        self._stop = True
        await self._task


async def test_save_keeps_the_event_loop_running(tmp_path, monkeypatch):
    """The loop must keep being scheduled while the index is written.

    Fix-proof: before the offload, ``_save_faiss_index`` ran the write inline,
    so the 200 ms spent inside the patched ``write_index`` was 200 ms in which
    the heartbeat below could not advance at all — ``after == before``.
    """
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")

    real_write_index = faiss_impl.faiss.write_index
    observed: list[tuple[int, int]] = []

    async with _Heartbeat() as heartbeat:

        def slow_write_index(index, path):
            before = heartbeat.beats
            time.sleep(0.2)
            observed.append((before, heartbeat.beats))
            return real_write_index(index, path)

        monkeypatch.setattr(faiss_impl.faiss, "write_index", slow_write_index)
        await storage._save_faiss_index(_noop)

    assert observed, "patched write_index was never called"
    before, after = observed[0]
    assert after > before, (
        "the event loop did not advance while the index was being written "
        f"(beats {before} -> {after})"
    )

    # The write still has to be correct, not merely non-blocking.
    assert faiss.read_index(storage._faiss_index_file).ntotal == 1
    meta = json.load(open(storage._meta_file))
    assert next(iter(meta.values()))["__id__"] == "v1"


async def test_metadata_snapshot_is_taken_before_the_offload(tmp_path, monkeypatch):
    """A mutation racing the write must not reach the file being written.

    ``client_storage`` reads ``_id_to_meta`` without the lock, so the worker
    must serialize a snapshot rather than the live dict. This test mutates the
    dict while the worker is parked inside ``write_index``; if the snapshot
    were built in the worker the extra row would land on disk (or the
    iteration would raise ``RuntimeError: dictionary changed size``).
    """
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")

    real_write_index = faiss_impl.faiss.write_index
    inside_write = threading.Event()
    may_finish = threading.Event()

    def parked_write_index(index, path):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_write_index(index, path)

    monkeypatch.setattr(faiss_impl.faiss, "write_index", parked_write_index)

    save = asyncio.create_task(storage._save_faiss_index(_noop))
    while not inside_write.is_set():
        await asyncio.sleep(0.01)

    # The loop is free (that is the point of the offload), so an unlocked
    # reader/writer can run right here.
    storage._id_to_meta[999] = {"__id__": "late", "content": "late"}
    may_finish.set()
    await save

    meta = json.load(open(storage._meta_file))
    assert "999" not in meta, "the worker serialized the live dict, not a snapshot"
    assert [entry["__id__"] for entry in meta.values()] == ["v1"]


async def test_cancelled_save_still_completes_both_files(tmp_path, monkeypatch):
    """Cancelling the caller must not abandon a half-written pair of files.

    The caller holds ``_storage_lock`` across the save; returning early would
    release it while the worker is still writing. ``run_in_storage_io`` is
    uncancellable once submitted, which keeps the pre-change guarantee (a
    synchronous write could not be cancelled at all).
    """
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")

    real_write_index = faiss_impl.faiss.write_index
    inside_write = threading.Event()
    may_finish = threading.Event()

    def parked_write_index(index, path):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_write_index(index, path)

    monkeypatch.setattr(faiss_impl.faiss, "write_index", parked_write_index)

    save = asyncio.create_task(storage._save_faiss_index(_noop))
    while not inside_write.is_set():
        await asyncio.sleep(0.01)

    save.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not save.done(), "caller returned while the worker was still writing"

    may_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await save

    assert os.path.exists(storage._faiss_index_file)
    assert os.path.exists(storage._meta_file)
    assert faiss.read_index(storage._faiss_index_file).ntotal == 1
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"unexpected tmp residue: {leftovers}"


async def test_cancelled_commit_still_notifies_and_retires_the_redo_logs(
    tmp_path, monkeypatch
):
    """A cancelled commit must not publish both files and skip its bookkeeping.

    `set_all_update_flags` is what tells the other processes to reload; skipped,
    they keep serving the previous index. The redo logs would likewise keep rows
    that are already durable. Both live in the hook that `commit_in_storage_io`
    runs inside the write's uncancellable region.

    Fix-proof: inline the bookkeeping after the offload instead and `flagged`
    stays empty — the P2 Codex raised on #3740, in this backend.
    """
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    finalize_share_data()
    initialize_share_data()
    try:
        storage = _make_storage(tmp_path)
        await storage.initialize()
        _seed(storage, "v1")
        storage._index_dirty = True

        flagged: list[str] = []
        real_write_index = faiss_impl.faiss.write_index
        inside_write = threading.Event()
        may_finish = threading.Event()

        async def spy_set_all_update_flags(namespace, workspace=None):
            flagged.append(namespace)

        def parked_write_index(index, path):
            inside_write.set()
            assert may_finish.wait(timeout=5), "writer was never released"
            return real_write_index(index, path)

        monkeypatch.setattr(
            faiss_impl, "set_all_update_flags", spy_set_all_update_flags
        )
        monkeypatch.setattr(faiss_impl.faiss, "write_index", parked_write_index)

        commit = asyncio.create_task(storage.index_done_callback())
        while not inside_write.is_set():
            await asyncio.sleep(0.01)

        commit.cancel()
        may_finish.set()

        with pytest.raises(asyncio.CancelledError):
            await commit

        assert flagged == ["vectors"], (
            "a cancelled commit published both files without telling the other "
            f"processes to reload them (flagged={flagged})"
        )
        assert storage._index_dirty is False, "dirty bit survived a durable save"
        assert storage._unsaved_upserts == {}, "redo log kept rows that are on disk"
    finally:
        finalize_share_data()
