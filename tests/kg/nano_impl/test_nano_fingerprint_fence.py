"""The vector backends' reload fence must not depend on the notification alone.

Phase 2 of issue #3854, mirroring `tests/kg/networkx_impl/`. Before it,
`_reload_client_from_disk_locked` opened with
`if not self.storage_updated.value: return False`, and that flag is published
with one Manager RPC per process. A peer whose flag was never flipped kept a
stale matrix and, on the `for_write=True` path, saved it over the durable rows.

The outcome differs from `NetworkXStorage` in a way worth pinning: this backend
does not DECLINE a stale write, it reload-then-replays. The peer's snapshot is
loaded and this process's pending buffer plus its `_unsaved_*` redo logs are
replayed on top (issue #3688), so both sides survive — which is why the
flush-failure propagation NetworkX needs has no counterpart here.

A lost notification is simulated by no-op'ing `set_all_update_flags` in the
storage module: the write lands, no peer flag is flipped. That is exactly the
state a partial publication leaves behind.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lightrag.kg import file_fingerprint, nano_vector_db_impl
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

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

    monkeypatch.setattr(nano_vector_db_impl, "set_all_update_flags", _noop)


async def _embed(texts, **_kwargs):
    return np.array(
        [np.full(DIM, (abs(hash(t)) % 97) + 1, dtype=np.float32) for t in texts]
    )


async def _worker(tmp_path) -> NanoVectorDBStorage:
    """A storage instance on a fixed file — call twice for two 'workers'."""
    storage = NanoVectorDBStorage(
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
    """The ids a fresh reader would see — the durable state, not memory.

    Read through the JSON file rather than a second ``NanoVectorDB``, so the
    assertion does not depend on that class's private storage attribute.
    """
    with open(worker._client_file_name, encoding="utf-8") as fh:
        return {row["__id__"] for row in json.load(fh)["data"]}


@pytest.mark.asyncio
async def test_a_stale_writer_replays_instead_of_overwriting_a_peer_commit(
    tmp_path, multiprocess, lost_notification
):
    """The lost write itself — and here BOTH sides survive it."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert({"from_b": {"content": "b"}})

        await worker_a.upsert({"from_a": {"content": "a"}})
        assert await worker_a.index_done_callback() is True

        # The notification never arrived, so only the file says A committed.
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
async def test_writer_does_not_reload_the_file_it_just_wrote(
    tmp_path, multiprocess, monkeypatch
):
    """Without adopting its own save, every writer would re-parse the whole
    JSON file and rebuild the matrix on its next operation, forever."""
    worker = await _worker(tmp_path)
    try:
        await worker.upsert({"n1": {"content": "x"}})
        assert await worker.index_done_callback() is True

        reloads = [0]
        original = NanoVectorDBStorage._reload_client_from_disk_locked

        def counting(self, **kwargs):
            did = original(self, **kwargs)
            reloads[0] += int(bool(did))
            return did

        monkeypatch.setattr(
            NanoVectorDBStorage, "_reload_client_from_disk_locked", counting
        )
        assert await worker.get_by_id("n1") is not None
        assert reloads[0] == 0
        assert worker._missed_notification_reloads == 0
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_single_process_mode_ignores_a_divergent_file(
    tmp_path, lost_notification
):
    """No ``multiprocess`` fixture: no peer can have committed, so a divergent
    file means an external edit, and reloading for it would discard this
    process's own pending rows."""
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
async def test_drop_adopts_the_files_absence(tmp_path, multiprocess, monkeypatch):
    worker = await _worker(tmp_path)
    try:
        await worker.upsert({"n1": {"content": "x"}})
        assert await worker.index_done_callback() is True
        assert worker._loaded_fingerprint is not None

        assert (await worker.drop())["status"] == "success"
        # The file's ABSENCE, adopted — not ``None``, which means "nothing
        # recorded" and would order a reload on the next call.
        assert worker._loaded_fingerprint == (None,)
        assert worker._peer_commit_detected() is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_failing_reload_does_not_recount_the_same_peer_commit(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """The counter must count peer commits, not attempts to reload out of them.

    A load that raises leaves `_loaded_fingerprint` untouched, so the same peer
    commit is re-detected by every later call. Counted at each detection, one
    commit inflates the counter without bound — and `file_fingerprint`'s module
    docstring designates these counters as the evidence the writer-side
    `os.utime` remedy waits on, so an inflated one is as useless as a
    suppressed one. Issue #3854.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert({"from_a": {"content": "a"}})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        broken = [True]

        def maybe_boom(*args, **kwargs):
            if broken[0]:
                raise OSError("unreadable")
            return real_client(*args, **kwargs)

        real_client = nano_vector_db_impl.NanoVectorDB
        monkeypatch.setattr(nano_vector_db_impl, "NanoVectorDB", maybe_boom)

        for _ in range(5):
            with pytest.raises(OSError, match="unreadable"):
                await worker_b.get_by_id("from_a")
        assert worker_b._missed_notification_reloads == 1

        # A genuinely second commit during the outage IS counted: deduplication
        # must not turn into suppression.
        await worker_a.upsert({"from_a_again": {"content": "a2"}})
        assert await worker_a.index_done_callback() is True
        with pytest.raises(OSError, match="unreadable"):
            await worker_b.get_by_id("from_a")
        assert worker_b._missed_notification_reloads == 2

        broken[0] = False
        assert await worker_b.get_by_id("from_a_again") is not None
        assert worker_b._missed_notification_reloads == 2
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_a_recurring_state_is_counted_again_after_a_notified_reload(
    tmp_path, multiprocess, lost_notification
):
    """Deduplication must not outlive the reload it was protecting.

    The marker exists only to stop ONE detection being re-counted while the
    reload keeps failing. Kept past a successful reload it suppresses a state
    that RECURS — a peer drop, a notified recreation, then a second drop whose
    notification is lost, all sharing the "absent" fingerprint. That is a
    genuine second lost notification, and not the same-tick collision residue.
    """
    worker_a = await _worker(tmp_path)
    await worker_a.upsert({"x": {"content": "x"}})
    assert await worker_a.index_done_callback() is True

    worker_b = await _worker(tmp_path)
    try:
        assert worker_b._missed_notification_reloads == 0

        assert (await worker_a.drop())["status"] == "success"
        assert await worker_b.get_by_id("x") is None
        assert worker_b._missed_notification_reloads == 1

        await worker_a.upsert({"y": {"content": "y"}})
        assert await worker_a.index_done_callback() is True
        worker_b.storage_updated.value = True
        assert await worker_b.get_by_id("y") is not None
        assert worker_b._missed_notification_reloads == 1

        assert (await worker_a.drop())["status"] == "success"
        assert await worker_b.get_by_id("y") is None
        assert worker_b._missed_notification_reloads == 2
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


@pytest.mark.asyncio
async def test_an_unreadable_adoption_keeps_the_dedupe_marker(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """Clearing the marker is only safe once a CONCRETE state is recorded.

    `adopted(UNREADABLE)` is `None`, which means "nothing recorded" — and
    `peer_commit_detected` reports a change against `None` for any state. So a
    clear on that adoption forgets which commit was already counted, and the
    next call counts the same one again. The post-drop state is `(None,)`, a
    real fingerprint, so it still clears. Issue #3854.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert({"from_a": {"content": "a"}})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        # Counted once, and the load fails, so the marker must survive.
        with pytest.MonkeyPatch.context() as broken:
            broken.setattr(
                nano_vector_db_impl,
                "NanoVectorDB",
                lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable")),
            )
            with pytest.raises(OSError, match="unreadable"):
                await worker_b.get_by_id("from_a")
        assert worker_b._missed_notification_reloads == 1

        # The retry's pre-read sample fails while the load itself succeeds, so
        # the adoption records `None` rather than a state.
        original = worker_b._stat_fingerprint

        def unreadable_once():
            worker_b._stat_fingerprint = original
            return file_fingerprint.UNREADABLE

        worker_b._stat_fingerprint = unreadable_once
        assert await worker_b.get_by_id("from_a") is not None
        assert worker_b._loaded_fingerprint is None

        # Same peer commit, still one event.
        assert await worker_b.get_by_id("from_a") is not None
        assert worker_b._missed_notification_reloads == 1
    finally:
        await worker_b.finalize()
        await worker_a.finalize()
