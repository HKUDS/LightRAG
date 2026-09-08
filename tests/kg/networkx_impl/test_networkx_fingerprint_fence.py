"""The cross-process reload fence must not depend on the notification alone.

Issue #3854. ``NetworkXStorage``'s fence used to rest entirely on the
``storage_updated`` flag, which ``set_all_update_flags`` publishes with one
Manager RPC per process — so it can be lost, for one process or for several.
A worker whose flag was never flipped did not reload in ``_get_graph`` and did
not decline in ``index_done_callback``, and its save serializes the WHOLE
GraphML file: it silently overwrote the durable commit it never saw.

*Single writer per workspace* does not prevent that. It gives one writer at a
time, not always the same process — a later request legitimately makes another
worker the next writer.

These tests pin the second channel: the file's ``(st_mtime_ns, st_size)``
against the fingerprint recorded when this process last loaded or wrote it.
They also pin the two rules that keep it honest (sample before the read, adopt
your own commit) and the residues it is documented to keep (a timestamp-tick
collision, an unreadable ``stat``, single-process mode).

A lost notification is simulated by no-op'ing ``set_all_update_flags`` in the
storage module: the write lands, no peer flag is flipped. That is exactly the
state a partial publication leaves behind.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from lightrag.kg import file_fingerprint, networkx_impl
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture
def multiprocess(monkeypatch):
    """Pretend peer processes exist, so the file channel is armed.

    The fence is gated on ``file_fingerprint.fence_enabled()`` — in
    single-process mode a divergent file cannot be a peer commit. Tests that
    want the fence must say so; the gate's other side is asserted by
    ``test_single_process_mode_ignores_a_divergent_file``.
    """
    monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)


@pytest.fixture
def lost_notification(monkeypatch):
    """Every commit lands on disk and notifies nobody."""

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(networkx_impl, "set_all_update_flags", _noop)


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_storage(tmp_path) -> NetworkXStorage:
    """A storage instance on a fixed file — call twice for two 'workers'."""
    return NetworkXStorage(
        namespace="test_graph",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
    )


async def _worker(tmp_path) -> NetworkXStorage:
    storage = _make_storage(tmp_path)
    await storage.initialize()
    return storage


def _count_loads(monkeypatch) -> list[int]:
    """Count full GraphML re-parses; a reload is not otherwise observable."""
    calls = [0]
    original = NetworkXStorage.load_nx_graph

    def counting(file_name):
        calls[0] += 1
        return original(file_name)

    monkeypatch.setattr(NetworkXStorage, "load_nx_graph", staticmethod(counting))
    return calls


@pytest.mark.asyncio
async def test_stale_writer_declines_instead_of_overwriting_a_peer_commit(
    tmp_path, multiprocess, lost_notification
):
    """The lost write itself: B must refuse, and A's commit must survive."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        # B mutates first, so it holds a snapshot that predates A's commit.
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})

        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        # The notification never arrived, so only the file says A committed.
        assert worker_b.storage_updated.value is False

        assert await worker_b.index_done_callback() is False
        assert worker_b._missed_notification_reloads == 1

        # A's node is still on disk, and B's uncommitted one never landed.
        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        assert on_disk.has_node("from_a")
        assert not on_disk.has_node("from_b")
        # B also converged on the file rather than keeping its stale view.
        assert worker_b._graph.has_node("from_a")
        assert not worker_b._graph.has_node("from_b")
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_reader_picks_up_a_peer_commit_it_was_never_told_about(
    tmp_path, multiprocess, lost_notification
):
    """The pre-existing window with no failure needed: reads went stale
    indefinitely, until some later commit anywhere happened to flip the flag."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        assert await worker_b.has_node("from_a") is True
        assert worker_b._missed_notification_reloads == 1
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_writer_does_not_reload_the_file_it_just_wrote(
    tmp_path, multiprocess, monkeypatch
):
    """Rule 3. Without adopting its own commit, every writer would re-parse
    the whole GraphML file on its next operation, forever."""
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("n1", {"entity_id": "n1"})
        assert await worker.index_done_callback() is True

        loads = _count_loads(monkeypatch)
        assert await worker.has_node("n1") is True
        assert loads[0] == 0
        assert worker._missed_notification_reloads == 0
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_fingerprint_is_sampled_before_the_read_not_after(
    tmp_path, multiprocess, monkeypatch
):
    """Rule 2, and the only way this fence could INTRODUCE a lost write.

    A fingerprint sampled after the parse can belong to a file newer than the
    one now in memory. Recording it would suppress the reload that newer file
    needs — permanently, since nothing re-checks.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert_node("first", {"entity_id": "first"})
        assert await worker_a.index_done_callback() is True

        original = NetworkXStorage.load_nx_graph
        replaced_during_read = [False]

        def replace_mid_read(file_name):
            graph = original(file_name)
            if not replaced_during_read[0]:
                replaced_during_read[0] = True
                # A third commit lands while B is parsing the second one.
                later = networkx_impl.nx.Graph()
                later.add_node("second", entity_id="second")
                NetworkXStorage.write_nx_graph(later, file_name, "ws")
            return graph

        monkeypatch.setattr(
            NetworkXStorage, "load_nx_graph", staticmethod(replace_mid_read)
        )

        await worker_b.has_node("first")
        assert replaced_during_read[0] is True

        # The fingerprint B recorded must be the pre-replacement one, so the
        # newer file is still pending — never suppressed.
        assert worker_b._peer_commit_detected() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_a_timestamp_tick_collision_is_the_documented_residue(
    tmp_path, multiprocess, lost_notification
):
    """Pins the residue explicitly rather than leaving it to wall-clock luck.

    Two commits inside one filesystem timestamp tick, with an identical file
    size, are indistinguishable to the file channel. This asserts that
    limitation on purpose — and that the flag channel is what covers it, which
    is why the flag is not removable.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert_node("aa", {"entity_id": "aa"})
        assert await worker_a.index_done_callback() is True
        assert await worker_b.has_node("aa") is True
        recorded = worker_b._loaded_fingerprint
        assert recorded is not None
        # One entry per path; this storage has a single file.
        recorded_mtime, recorded_size = recorded[0]

        # A second commit whose GraphML is the same length as the first.
        await worker_a.delete_node("aa")
        await worker_a.upsert_node("bb", {"entity_id": "bb"})
        assert await worker_a.index_done_callback() is True

        # Force the tick collision. The size must already match, or this test
        # would be pinning nothing.
        os.utime(worker_b._graphml_xml_file, ns=(recorded_mtime, recorded_mtime))
        assert os.stat(worker_b._graphml_xml_file).st_size == recorded_size

        # The residue: the file channel cannot see it.
        assert worker_b._peer_commit_detected() is False
        assert await worker_b.has_node("bb") is False

        # The flag channel can, which is the whole reason both are kept.
        worker_b.storage_updated.value = True
        assert await worker_b.has_node("bb") is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_an_unreadable_stat_degrades_to_the_flag_channel(
    tmp_path, multiprocess, monkeypatch
):
    """Failing towards a reload here would be worse than failing quiet: a
    reload that cannot read the file installs an EMPTY graph, and the next
    commit would serialize that over the real one."""
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("n1", {"entity_id": "n1"})
        assert await worker.index_done_callback() is True

        def denied(*_args, **_kwargs):
            raise PermissionError("stat denied")

        monkeypatch.setattr(networkx_impl.os, "stat", denied)

        assert worker._peer_commit_detected() is False
        assert await worker.has_node("n1") is True
        assert worker._graph.number_of_nodes() == 1
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_single_process_mode_ignores_a_divergent_file(
    tmp_path, lost_notification, monkeypatch
):
    """No ``multiprocess`` fixture: no peer can have committed, so a divergent
    file means an external edit, and reloading for it would discard this
    process's own uncommitted mutations."""
    worker = await _worker(tmp_path)
    try:
        foreign = networkx_impl.nx.Graph()
        foreign.add_node("foreign", entity_id="foreign")
        NetworkXStorage.write_nx_graph(foreign, worker._graphml_xml_file, "ws")

        await worker.upsert_node("pending", {"entity_id": "pending"})

        loads = _count_loads(monkeypatch)
        assert await worker.has_node("pending") is True
        assert loads[0] == 0
        assert worker._peer_commit_detected() is False
        assert worker._missed_notification_reloads == 0
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_drop_adopts_the_files_absence(tmp_path, multiprocess, monkeypatch):
    """After ``drop`` the file is gone and this process's graph is empty to
    match; the next read must not re-read the removal as a peer commit."""
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("n1", {"entity_id": "n1"})
        assert await worker.index_done_callback() is True
        # The commit adopted the file it wrote, so there is a real fingerprint
        # for the drop to have to clear.
        assert worker._loaded_fingerprint is not None

        assert (await worker.drop())["status"] == "success"
        # The file's ABSENCE, adopted — not ``None``, which means "nothing
        # recorded" and would order a reload on the next call.
        assert worker._loaded_fingerprint == (None,)

        loads = _count_loads(monkeypatch)
        assert await worker.has_node("n1") is False
        assert loads[0] == 0
        assert worker._missed_notification_reloads == 0
    finally:
        await worker.finalize()


class _WriteOnlyDeadFlag:
    """A flag proxy whose Manager died after the last successful read.

    Models the reachable sequence: `index_done_callback` reads the flag while
    the manager is alive, the offloaded save then fails, and by the time the
    recovery block runs the manager is gone. Reads keep working so the call
    gets as far as the save.
    """

    def __init__(self):
        self.write_attempts = 0

    @property
    def value(self):
        return False

    @value.setter
    def value(self, _v):
        self.write_attempts += 1
        raise BrokenPipeError("manager gone")


async def _worker_with_a_failed_save(tmp_path) -> NetworkXStorage:
    """A worker left holding a mutation whose save AND recovery reload failed.

    The state `_recovery_reload_pending` exists for: `self._graph` carries
    `never_saved`, the file does not, and nothing in either cross-process
    channel says so -- the file never moved, so the fingerprint still matches,
    and no peer committed, so no notification was sent.

    The breakage is applied through a `MonkeyPatch` instance of this helper's
    OWN, never the caller's fixture. pytest hands the same fixture instance to
    the test and to every fixture it requested, so undoing it here would also
    revert the `multiprocess` fixture's patch -- and each caller that asked for
    the fence would then run its assertions with the fence silently disabled,
    passing for the wrong reason.
    """
    worker = await _worker(tmp_path)
    await worker.upsert_node("durable", {"entity_id": "durable"})
    assert await worker.index_done_callback() is True

    await worker.upsert_node("never_saved", {"entity_id": "never_saved"})

    def save_boom(graph, file_name, workspace):
        raise OSError("save boom")

    def reload_boom(file_name):
        raise OSError("reload boom")

    breakage = pytest.MonkeyPatch()
    breakage.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
    breakage.setattr(NetworkXStorage, "load_nx_graph", staticmethod(reload_boom))
    try:
        with pytest.raises(OSError, match="save boom"):
            await worker.index_done_callback()
    finally:
        # Only the two storage methods: the recovery reload is what the caller
        # does NEXT, so the file has to be readable again by then.
        breakage.undo()

    assert worker._recovery_reload_pending is True
    return worker


@pytest.mark.asyncio
async def test_a_failed_save_forces_a_reload_in_single_process_mode(tmp_path):
    """Recovery must not depend on the multiprocess gate.

    Deliberately without the `multiprocess` fixture. The file channel is gated
    off in single-process mode, so before this change the cross-process flag
    was the only thing arming recovery here -- a fact easy to lose while
    reasoning about a fence whose other two tests are both about peers.
    """
    worker = await _worker_with_a_failed_save(tmp_path)
    try:
        assert file_fingerprint.fence_enabled() is False

        # The unpersisted mutation is discarded, the durable one survives.
        assert await worker.has_node("never_saved") is False
        assert await worker.has_node("durable") is True
        assert worker._recovery_reload_pending is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_failed_save_makes_the_next_commit_decline(tmp_path, multiprocess):
    """Saving would publish mutations from a batch already reported as failed.

    Their documents are marked FAILED and reprocessed from scratch, so the
    graph must not keep them. The decline reaches the caller as `False`, which
    `_commit_graph_or_raise` / `_flush_one` turn into a failure (rule 5).
    """
    worker = await _worker_with_a_failed_save(tmp_path)
    try:
        assert await worker.index_done_callback() is False
        assert worker._recovery_reload_pending is False

        # The declined commit reloaded, so the discard is durable, not just
        # in-memory: a later successful commit cannot resurrect it.
        assert await worker.has_node("never_saved") is False
        assert await worker.index_done_callback() is True
        reader = await _worker(tmp_path)
        try:
            assert await reader.has_node("never_saved") is False
            assert await reader.has_node("durable") is True
        finally:
            await reader.finalize()
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_recovery_reload_is_not_counted_as_a_lost_notification(
    tmp_path, multiprocess
):
    """The evidence counter must only ever count what it names.

    `_missed_notification_reloads` is what #3854 leaves behind to decide
    whether the writer-side `os.utime` monotonicity option is ever needed.
    Arming the FILE channel for recovery -- which invalidating
    `_loaded_fingerprint` did -- made a failed save show up as a notification
    that was lost, in exactly the deployment where someone would be reading
    that counter.
    """
    worker = await _worker_with_a_failed_save(tmp_path)
    try:
        # The helper must not have undone the `multiprocess` fixture: without
        # the fence, _peer_commit_detected() below is trivially False and this
        # test would pass while covering nothing.
        assert file_fingerprint.fence_enabled() is True

        # Armed WITHOUT invalidating the fingerprint: the save failed, so the
        # file is untouched and the recorded value still describes it
        # correctly. Invalidating it here is the false positive -- it makes
        # the file channel claim a peer commit that never happened.
        assert worker._loaded_fingerprint is not None
        assert worker._peer_commit_detected() is False
        assert worker._missed_notification_reloads == 0

        await worker.has_node("durable")

        assert worker._recovery_reload_pending is False
        assert worker._missed_notification_reloads == 0
        assert worker._peer_commit_detected() is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_arming_recovery_does_not_touch_the_manager(
    tmp_path, multiprocess, monkeypatch
):
    """The recovery arm must not be an RPC to a possibly-dead manager.

    It used to write `storage_updated`, which in multiprocess mode is a
    `Manager().Value` proxy -- an RPC to the very process whose outage may be
    why the reload just failed, needing its own failure path so the manager
    error did not replace the save error the caller must see. A plain
    attribute write has no such path to get wrong.
    """
    worker = await _worker(tmp_path)
    real_flag = worker.storage_updated
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("never_saved", {"entity_id": "never_saved"})

        def save_boom(graph, file_name, workspace):
            raise OSError("save boom")

        def reload_boom(file_name):
            raise OSError("reload boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
        monkeypatch.setattr(NetworkXStorage, "load_nx_graph", staticmethod(reload_boom))
        dead_flag = _WriteOnlyDeadFlag()
        worker.storage_updated = dead_flag

        # The SAVE error reaches the caller, unmasked.
        with pytest.raises(OSError, match="save boom"):
            await worker.index_done_callback()

        assert dead_flag.write_attempts == 0
        assert worker._recovery_reload_pending is True
    finally:
        worker.storage_updated = real_flag
        await worker.finalize()


@pytest.mark.asyncio
async def test_drop_clears_a_pending_recovery_reload(tmp_path, multiprocess):
    """A sticky flag survives a `drop` and would decline the next real commit.

    The mutation the recovery protects is destroyed along with everything
    else, so memory matches the file again and there is nothing left to
    discard.
    """
    worker = await _worker_with_a_failed_save(tmp_path)
    try:
        assert (await worker.drop())["status"] == "success"
        assert worker._recovery_reload_pending is False

        # Fresh work after the clear commits normally instead of declining.
        await worker.upsert_node("after_drop", {"entity_id": "after_drop"})
        assert await worker.index_done_callback() is True
        reader = await _worker(tmp_path)
        try:
            assert await reader.has_node("after_drop") is True
            assert await reader.has_node("durable") is False
        finally:
            await reader.finalize()
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_peer_commit_behind_a_recovery_reload_is_still_counted(
    tmp_path, multiprocess, lost_notification
):
    """The recovery flag's precedence must not hide a lost notification.

    Recovery is tested ahead of both channels and one reload discharges all
    of them, so a peer commit that arrives unannounced WHILE recovery is
    pending is handled correctly -- and, without classifying first, never
    counted. `_missed_notification_reloads` is what the `os.utime` decision
    waits on, so undercounting it is as much a defect as overcounting.
    """
    worker_a = await _worker_with_a_failed_save(tmp_path)
    try:
        assert file_fingerprint.fence_enabled() is True

        worker_b = await _worker(tmp_path)
        try:
            await worker_b.upsert_node("from_peer", {"entity_id": "from_peer"})
            assert await worker_b.index_done_callback() is True
        finally:
            await worker_b.finalize()

        assert worker_a._missed_notification_reloads == 0
        assert worker_a.storage_updated.value is False

        # One reload discharges both: the peer's commit is picked up and the
        # unpersisted mutation is discarded.
        assert await worker_a.has_node("from_peer") is True
        assert await worker_a.has_node("never_saved") is False
        assert worker_a._missed_notification_reloads == 1
    finally:
        await worker_a.finalize()


@pytest.mark.asyncio
async def test_a_peer_commit_behind_a_recovery_decline_is_still_counted(
    tmp_path, multiprocess, lost_notification
):
    """Same precedence, same blind spot, at `index_done_callback`."""
    worker_a = await _worker_with_a_failed_save(tmp_path)
    try:
        worker_b = await _worker(tmp_path)
        try:
            await worker_b.upsert_node("from_peer", {"entity_id": "from_peer"})
            assert await worker_b.index_done_callback() is True
        finally:
            await worker_b.finalize()

        assert worker_a._missed_notification_reloads == 0
        assert await worker_a.index_done_callback() is False
        assert worker_a._missed_notification_reloads == 1
        # The decline preserved the peer's commit, which is the point.
        assert await worker_a.has_node("from_peer") is True
    finally:
        await worker_a.finalize()


def _break_reloads(monkeypatch) -> list[bool]:
    """Make `_reload_locked`'s file read fail until the returned flag is cleared.

    Toggleable rather than permanent so a test can let a later reload land.
    Note that a worker only reaches `load_nx_graph` when something tells it to
    reload, so a peer that only ever writes is unaffected by this — but it must
    be CONSTRUCTED before this is installed, since `__post_init__` loads.
    """
    broken = [True]
    original = NetworkXStorage.load_nx_graph

    def maybe_boom(file_name):
        if broken[0]:
            raise OSError("unreadable")
        return original(file_name)

    monkeypatch.setattr(NetworkXStorage, "load_nx_graph", staticmethod(maybe_boom))
    return broken


@pytest.mark.asyncio
async def test_a_failing_reload_does_not_recount_the_same_peer_commit(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """The counter must count peer commits, not attempts to reload out of them.

    A reload that raises leaves `_loaded_fingerprint` untouched, so the same
    peer commit is re-detected by every later call. Counted at detection
    without deduplication, one commit inflates the counter without bound —
    and `_missed_notification_reloads` is what the `os.utime` decision rests
    on, so an inflated value is as useless as a suppressed one.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_peer", {"entity_id": "from_peer"})
        assert await worker_b.index_done_callback() is True

        broken = _break_reloads(monkeypatch)
        for _ in range(5):
            with pytest.raises(OSError, match="unreadable"):
                await worker_a.has_node("from_peer")
        assert worker_a._missed_notification_reloads == 1

        # Once the file is readable the same commit still does not re-count,
        # and the reload finally lands.
        broken[0] = False
        assert await worker_a.has_node("from_peer") is True
        assert worker_a._missed_notification_reloads == 1
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


@pytest.mark.asyncio
async def test_a_second_peer_commit_during_failing_reloads_is_counted(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """Deduplication must not suppress a genuinely different lost notification.

    Two commits sharing one `(st_mtime_ns, st_size)` are still indistinguishable
    — the documented tick-collision residue, inherited here, not introduced.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("peer_one", {"entity_id": "peer_one"})
        assert await worker_b.index_done_callback() is True

        broken = _break_reloads(monkeypatch)
        with pytest.raises(OSError, match="unreadable"):
            await worker_a.has_node("peer_one")
        assert worker_a._missed_notification_reloads == 1

        # A second, distinct commit lands while worker A still cannot reload.
        await worker_b.upsert_node("peer_two_longer_name", {"entity_id": "x"})
        assert await worker_b.index_done_callback() is True

        with pytest.raises(OSError, match="unreadable"):
            await worker_a.has_node("peer_one")
        assert worker_a._missed_notification_reloads == 2

        broken[0] = False
        assert await worker_a.has_node("peer_two_longer_name") is True
        assert worker_a._missed_notification_reloads == 2
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


@pytest.mark.asyncio
async def test_a_failing_recovery_reload_does_not_recount_either(
    tmp_path, multiprocess, lost_notification
):
    """The same, at the recovery branch — where a failing reload is the norm.

    Recovery is armed precisely because a reload already failed, so this is
    the site where a persistently unreadable file is least exotic.
    """
    worker_a = await _worker_with_a_failed_save(tmp_path)
    try:
        worker_b = await _worker(tmp_path)
        try:
            await worker_b.upsert_node("from_peer", {"entity_id": "from_peer"})
            assert await worker_b.index_done_callback() is True
        finally:
            await worker_b.finalize()

        with pytest.MonkeyPatch.context() as broken_reads:
            broken_reads.setattr(
                NetworkXStorage,
                "load_nx_graph",
                staticmethod(_raise_unreadable),
            )
            for _ in range(4):
                with pytest.raises(OSError, match="unreadable"):
                    await worker_a.has_node("from_peer")
                assert worker_a._recovery_reload_pending is True

        assert worker_a._missed_notification_reloads == 1
        assert await worker_a.has_node("from_peer") is True
        assert worker_a._missed_notification_reloads == 1
    finally:
        await worker_a.finalize()


def _raise_unreadable(file_name):
    raise OSError("unreadable")


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
    await worker_a.upsert_node("x", {"entity_id": "x"})
    assert await worker_a.index_done_callback() is True

    worker_b = await _worker(tmp_path)
    try:
        assert worker_b._missed_notification_reloads == 0

        # 1. The peer drops. Notification lost; the file channel catches it.
        assert (await worker_a.drop())["status"] == "success"
        assert await worker_b.has_node("x") is False
        assert worker_b._missed_notification_reloads == 1

        # 2. The peer recreates it, and this time the notification arrives.
        await worker_a.upsert_node("y", {"entity_id": "y"})
        assert await worker_a.index_done_callback() is True
        worker_b.storage_updated.value = True
        assert await worker_b.has_node("y") is True
        assert worker_b._missed_notification_reloads == 1

        # 3. The peer drops again, notification lost again. Same "absent"
        #    fingerprint as step 1 — and a second genuine loss.
        assert (await worker_a.drop())["status"] == "success"
        assert await worker_b.has_node("y") is False
        assert worker_b._missed_notification_reloads == 2
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


@pytest.mark.asyncio
async def test_a_stat_that_fails_only_while_counting_does_not_lose_the_event(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """The count and the adoption must come from ONE observation.

    Sampling separately for each, a transient failure on the counting sample
    alone loses the event for good: the count is skipped, and the reload's own
    successful sample adopts the peer state, so there is no divergence left
    for a later call to count. Sharing the sample ties the two outcomes —
    either it counts, or it adopts nothing that could suppress counting next
    time.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        # B is constructed BEFORE the commit, so it is genuinely stale.
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False
        assert worker_b._missed_notification_reloads == 0

        # Fail the COUNTING sample and nothing else: `_peer_commit_detected`
        # reaches `file_fingerprint.sample` directly, so shadowing
        # `_stat_fingerprint` hits only the hoisted sample. Self-restoring, so
        # the reload that follows within the same call would succeed if it
        # sampled for itself — which is the defect.
        original = worker_b._stat_fingerprint

        def unreadable_once():
            worker_b._stat_fingerprint = original
            return file_fingerprint.UNREADABLE

        worker_b._stat_fingerprint = unreadable_once
        assert await worker_b.has_node("from_a") is True

        # Not counted on that call — an unreadable sample cannot say what it
        # would be counting. What matters is that it is not LOST: no
        # fingerprint was adopted, so the divergence still stands.
        assert worker_b._missed_notification_reloads == 0
        assert worker_b._loaded_fingerprint is None

        assert await worker_b.has_node("from_a") is True
        assert worker_b._missed_notification_reloads == 1
    finally:
        await worker_b.finalize()
        await worker_a.finalize()
