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

import networkx as nx
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


def _save_that_publishes_a_peer_commit_then_fails(peer_node: str):
    """A save that fails, having first let a peer commit land on the file.

    `index_done_callback` releases `_storage_lock` between its fence block and
    its save, so a peer commit can land in that window -- with its
    notification lost, the state the counter exists to record. Injecting it
    from inside the save is the only way to reach the failure handler with a
    file the fence block already found unchanged.

    The peer's node name differs in length from the local one, so
    `(st_mtime_ns, st_size)` diverges on the SIZE half too: an mtime-only
    change would make the test depend on the filesystem's timestamp
    granularity, which is the documented tick-collision residue.
    """
    original = NetworkXStorage.write_nx_graph

    def save_boom(graph, file_name, workspace):
        peer = nx.Graph()
        peer.add_node("durable", entity_id="durable")
        peer.add_node(peer_node, entity_id=peer_node)
        original(peer, file_name, workspace)
        raise OSError("save boom")

    return save_boom


@pytest.mark.asyncio
async def test_a_peer_commit_behind_a_failed_saves_recovery_is_still_counted(
    tmp_path, multiprocess
):
    """The failed-save handler reloads too, so it must classify first.

    Same undercount as the two recovery branches, one site further on: the
    handler's recovery reload adopts the file, and once adopted the question
    "did a peer commit without announcing it?" can no longer be asked. The
    save error still reaches the caller either way -- what is lost is the
    evidence the writer-side `os.utime` decision waits on.
    """
    worker = await _worker(tmp_path)
    try:
        assert file_fingerprint.fence_enabled() is True

        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("never_saved", {"entity_id": "never_saved"})
        # A MonkeyPatch of this test's OWN: undoing the shared `monkeypatch`
        # fixture would also revert `multiprocess`, and the assertions below
        # would run with the fence silently disabled.
        breakage = pytest.MonkeyPatch()
        breakage.setattr(
            NetworkXStorage,
            "write_nx_graph",
            staticmethod(_save_that_publishes_a_peer_commit_then_fails("from_a_peer")),
        )
        try:
            with pytest.raises(OSError, match="save boom"):
                await worker.index_done_callback()
        finally:
            breakage.undo()
        assert file_fingerprint.fence_enabled() is True

        # The lost notification is on the record, once.
        assert worker._missed_notification_reloads == 1
        # And the dedupe marker is already spent: the reload that followed
        # adopted the peer's state, which is the point rule 4 in
        # `kg.file_fingerprint` clears it at -- it guards a reload that did
        # not land, not the counted state forever.
        assert worker._counted_peer_fingerprint is None

        # And the recovery still happened: the reload succeeded, so nothing is
        # armed, the unpersisted mutation is gone and the peer's commit is the
        # view this process serves.
        assert worker._recovery_reload_pending is False
        assert await worker.has_node("from_a_peer") is True
        assert await worker.has_node("never_saved") is False
        assert worker._missed_notification_reloads == 1
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_failed_save_alone_is_not_counted_as_a_lost_notification(
    tmp_path, multiprocess
):
    """The negative half: classifying must not turn every failed save into one.

    `atomic_write` leaves the destination untouched when the write raises, so
    the ordinary failed save diverges from nothing and the counter must stay
    at zero -- the overcount that arming the file channel for recovery used to
    produce.
    """
    worker = await _worker(tmp_path)
    try:
        assert file_fingerprint.fence_enabled() is True

        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("never_saved", {"entity_id": "never_saved"})

        def save_boom(graph, file_name, workspace):
            raise OSError("save boom")

        # This test's own, for the reason above.
        breakage = pytest.MonkeyPatch()
        breakage.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
        try:
            with pytest.raises(OSError, match="save boom"):
                await worker.index_done_callback()
        finally:
            breakage.undo()
        assert file_fingerprint.fence_enabled() is True

        assert worker._missed_notification_reloads == 0
        assert worker._counted_peer_fingerprint is None
        assert worker._recovery_reload_pending is False
        assert await worker.has_node("never_saved") is False
        assert await worker.has_node("durable") is True
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


class _DeadOnReadFlag:
    """A flag proxy whose Manager is gone by the time it is READ again.

    `_WriteOnlyDeadFlag` above models the outage arriving before the flag is
    written. This one arrives earlier, before it is read -- which is what the
    peer classification does, and it does so BEFORE any reload is attempted.
    """

    def __init__(self, reads_before_death: int):
        self.reads = 0
        self._budget = reads_before_death

    @property
    def value(self):
        self.reads += 1
        if self.reads > self._budget:
            raise BrokenPipeError("manager gone")
        return False

    @value.setter
    def value(self, _v):
        raise BrokenPipeError("manager gone")


@pytest.mark.asyncio
async def test_a_classification_that_raises_still_leaves_the_reload_owed(
    tmp_path, multiprocess
):
    """Owed is recorded at the branch entry, not after the reload fails.

    `_count_unannounced_peer_commit_locked` reads `storage_updated.value` over
    the Manager and can raise there -- before `_reload_locked` is reached at
    all. Arming on the reload's failure paths therefore misses it: the flag
    stays false while the failed batch's mutations sit in memory, and the next
    commit finds an unchanged fingerprint and a false notification flag, so it
    publishes them.

    A manager outage is a plausible reason the save failed in the first place,
    which is what makes this the likely case rather than an exotic one.
    """
    worker = await _worker(tmp_path)
    real_flag = worker.storage_updated
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("never_saved", {"entity_id": "never_saved"})

        def save_boom(graph, file_name, workspace):
            raise OSError("save boom")

        # One read survives -- `index_done_callback`'s own fence test -- and
        # the next one, inside the failure handler's classification, does not.
        dead = _DeadOnReadFlag(reads_before_death=1)
        # A MonkeyPatch of this test's own: undoing the shared fixture would
        # revert `multiprocess` and run the assertions with the fence off.
        breakage = pytest.MonkeyPatch()
        breakage.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
        worker.storage_updated = dead
        try:
            # The SAVE error still reaches the caller, unmasked.
            with pytest.raises(OSError, match="save boom"):
                await worker.index_done_callback()
        finally:
            breakage.undo()
            worker.storage_updated = real_flag
        assert dead.reads == 2, "the classification never reached the Manager read"
        assert file_fingerprint.fence_enabled() is True
        owed = worker._recovery_reload_pending

        # The manager is back and the file was never touched by the failed
        # save, so nothing in either channel objects to a commit. What must
        # stop it is the owed reload.
        committed = await worker.index_done_callback()
        on_disk = NetworkXStorage.load_nx_graph(worker._graphml_xml_file)
        assert not on_disk.has_node("never_saved"), (
            "work already reported as FAILED was published"
        )
        assert on_disk.has_node("durable")
        assert committed is False, "the commit should have been declined"
        assert owed is True, "the raised classification left nothing armed"
        assert worker._recovery_reload_pending is False
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
async def test_a_failed_reload_anywhere_arms_the_recovery_reload(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """Every failed reload leaves the same state, so every one must arm.

    A reload that fails leaves this process holding a graph that does not
    match the file, and fails the operation that asked for it -- so what the
    graph holds belongs to work already reported as failed. Only the
    failed-SAVE path used to arm `_recovery_reload_pending` for that; the
    decline path and both `_get_graph` channels armed nothing, and the flag
    and fingerprint they leave behind keep the divergence visible only while
    the file can still be sampled.

    Let the `stat` stay broken and both channels report "no change" -- the
    documented degrade to the flag alone -- and the next commit passes the
    fence and serializes that graph: the peer's DURABLE commit overwritten,
    and work already reported as FAILED published in its place. Two losses
    the fence and the recovery flag each exist to prevent, from one unarmed
    reload.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        # B holds a mutation from a batch that predates A's commit.
        await worker_b.upsert_node("from_b_failed", {"entity_id": "from_b_failed"})
        await worker_a.upsert_node("from_a_durable", {"entity_id": "from_a_durable"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        # B declines, as it must -- but its reload raises, so the decline is
        # reported as a failure and B is left holding the failed batch.
        with pytest.MonkeyPatch.context() as broken_reads:
            broken_reads.setattr(
                NetworkXStorage, "load_nx_graph", staticmethod(_raise_unreadable)
            )
            with pytest.raises(OSError, match="unreadable"):
                await worker_b.index_done_callback()
        # Recorded, not asserted yet: the losses below are what this is for,
        # and they must be what fails if the arming is removed.
        armed = worker_b._recovery_reload_pending

        # The `stat` now stays broken, so neither channel can object to
        # anything. Nothing B does may reach the file.
        monkeypatch.setattr(
            file_fingerprint,
            "sample",
            lambda paths, *, workspace: file_fingerprint.UNREADABLE,
        )
        refused = ""
        try:
            await worker_b.upsert_node("later_batch", {"entity_id": "later_batch"})
            await worker_b.index_done_callback()
        except OSError as exc:
            refused = str(exc)
        monkeypatch.undo()

        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        assert on_disk.has_node("from_a_durable"), (
            "the peer's durable commit was overwritten by a stale snapshot"
        )
        assert not on_disk.has_node("from_b_failed"), (
            "work already reported as FAILED was published"
        )
        assert "could not be sampled" in refused, (
            "the file survived by luck, not by the armed recovery reload"
        )
        assert armed is True, "the failed decline reload armed nothing"
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


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
async def test_an_unreadable_observation_defers_the_event_rather_than_losing_it(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """An unreadable `stat` degrades the channel; it must not erase evidence.

    One observation decides, counts and adopts, so an unreadable one does all
    three of nothing: the divergence is not reported (the documented degrade
    to the flag channel — this call serves the snapshot it already holds),
    nothing is counted, and nothing is adopted. That last part is what keeps
    the event countable, and it is the property the one-observation rule
    exists to guarantee: either the event counts, or no fingerprint is adopted
    that could suppress counting it next time.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        # B is constructed BEFORE the commit, so it is genuinely stale.
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False
        assert worker_b._missed_notification_reloads == 0

        _unreadable_once(worker_b)
        # The stale snapshot is served: B predates A's commit, so it does not
        # have the node. This is the pre-fence behaviour, restored for exactly
        # as long as the file cannot be sampled.
        assert await worker_b.has_node("from_a") is False

        # Not counted on that call — an unreadable sample cannot say what it
        # would be counting. What matters is that it is not LOST: nothing was
        # adopted, so the divergence still stands.
        assert worker_b._missed_notification_reloads == 0
        assert worker_b._loaded_fingerprint is not None

        assert await worker_b.has_node("from_a") is True
        assert worker_b._missed_notification_reloads == 1
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


@pytest.mark.asyncio
async def test_the_file_channel_takes_exactly_one_observation(
    tmp_path, multiprocess, lost_notification, monkeypatch
):
    """Decide, count, adopt — one `stat`, per the rule in `file_fingerprint`.

    A second observation anywhere in the branch reintroduces the split the
    rule forbids: the decision succeeds on one sample while the count is
    skipped on another, and the reload then adopts the good one, erasing the
    divergence a later call would have counted. The event is lost, not merely
    deferred.

    Pinned as a COUNT rather than as an outcome, because the split used to be
    invisible in behavior: `_storage_lock` is cross-process and there is no
    `await` between the branch condition and the body, so no peer could
    actually commit in between. That argument is not written in the contract,
    and a rule stating three-from-one should not rest on it.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        # Let the first observation through, then make every FURTHER one
        # unreadable: a second `stat` anywhere in the branch would land here
        # and change the outcome, which is how this test can see it at all.
        # Targeting `sample` rather than `os.stat` keeps the load's own
        # `os.path.exists` out of it.
        real_sample = file_fingerprint.sample
        taken = [0]

        def budgeted(paths, *, workspace):
            taken[0] += 1
            if taken[0] > 1:
                return file_fingerprint.UNREADABLE
            return real_sample(paths, workspace=workspace)

        monkeypatch.setattr(file_fingerprint, "sample", budgeted)
        assert await worker_b.has_node("from_a") is True
        monkeypatch.undo()

        assert taken[0] == 1, "the file channel observed the file twice"
        assert worker_b._missed_notification_reloads == 1
        assert worker_b._loaded_fingerprint is not None
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


def _unreadable_once(worker) -> None:
    """Make the NEXT `_reload_locked` self-sample fail, and only that one.

    `_peer_commit_detected()` samples through `file_fingerprint` directly, so
    shadowing the instance's `_stat_fingerprint` reaches the reload's own
    sample and nothing else. Self-restoring on its FIRST call rather than on
    scope exit -- that is what lets the retry inside the same test see a
    readable file, and why no `finally` is needed.
    """
    original = worker._stat_fingerprint

    def unreadable_once():
        worker._stat_fingerprint = original
        return file_fingerprint.UNREADABLE

    worker._stat_fingerprint = unreadable_once


@pytest.mark.asyncio
async def test_an_unreadable_sample_does_not_silently_drop_a_mutation(
    tmp_path, multiprocess
):
    """The silent partial loss a `None` fingerprint used to cause.

    Loading on an unreadable sample records `None`, against which every state
    is a divergence -- so the reload inside the NEXT `_get_graph` fired again
    and discarded whatever had been mutated in between, and the commit after
    it SUCCEEDED without those mutations. Their document is then marked
    PROCESSED, which is the one direction the consistency rules forbid: a loss
    that is neither loud nor self-healing.

    Deliberately without `lost_notification`: this is the ordinary writer
    handoff, where the flag DOES arrive and the reload is legitimate. Only its
    unreadable sample is injected.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is True

        _unreadable_once(worker_b)
        refused = ""
        try:
            await worker_b.upsert_node("X", {"entity_id": "X"})
        except OSError as exc:
            # The refusal. Retrying is what the caller does -- at batch
            # granularity that is the FAILED path reprocessing the document.
            refused = str(exc)
            await worker_b.upsert_node("X", {"entity_id": "X"})
        await worker_b.upsert_node("Y", {"entity_id": "Y"})

        assert await worker_b.index_done_callback() is True
        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        # X is the one that used to vanish: applied after the unreadable
        # adoption, discarded by the spurious reload that Y's `_get_graph`
        # triggered, and never mentioned again.
        assert on_disk.has_node("X")
        assert on_disk.has_node("Y")
        assert on_disk.has_node("from_a")
        assert "could not be sampled" in refused, (
            "the mutation survived by luck, not by the reload refusing"
        )
    finally:
        await worker_b.finalize()
        await worker_a.finalize()


def _unreadable_for(budget: int, monkeypatch):
    """Make the next `budget` fence observations fail, then recover.

    Patched at `file_fingerprint.sample` -- the one place the fence observes
    the file -- so it covers the commit's own fence test, the adoption inside
    `_record_fingerprint`, and every later branch condition and reload alike.
    `load_nx_graph` is left real: the file IS readable here, only its `stat`
    is not, which is the case a fingerprint cannot describe.

    Callers depend on the exact count, in this order: (1) `index_done_callback`
    block 1's fence test, (2) `_record_fingerprint` inside `_committed`, (3)
    the next `_get_graph`'s single file-channel observation. A budget of 3
    makes (3) unreadable, which is what reaches the refusal; the file channel
    takes exactly one observation per call, so there is no fourth. Returns the
    remaining counter -- assert it reached 0, or a new `sample()` call anywhere
    on the path shifts the numbers and the test passes for the wrong reason
    (budget spent early, real stat, nothing to detect).
    """
    remaining = [budget]
    real = file_fingerprint.sample

    def flaky(paths, *, workspace):
        if remaining[0] > 0:
            remaining[0] -= 1
            return file_fingerprint.UNREADABLE
        return real(paths, workspace=workspace)

    monkeypatch.setattr(file_fingerprint, "sample", flaky)
    return remaining


@pytest.mark.asyncio
async def test_a_post_commit_stat_failure_does_not_drop_the_next_mutation(
    tmp_path, multiprocess, monkeypatch
):
    """`_record_fingerprint`'s `None` is only harmless if it is resolved FIRST.

    The writer's own adoption is the one remaining producer of a `None`
    fingerprint, and the redundant reload it costs is documented as
    discarding nothing -- true only while that reload happens before any new
    mutation. It does not, by itself: `UNREADABLE` reports "no change", so the
    divergence branch cannot fire while the `stat` keeps failing. The call
    served the graph, a mutation landed on it, and the reload was deferred to
    whichever later call first managed to `stat` -- mid-batch, discarding that
    mutation, after which the commit SUCCEEDED without it and its document
    was marked PROCESSED.
    """
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})

        # All three observations fail: the commit's own fence test, the
        # adoption in `_record_fingerprint`, and the next call's.
        remaining = _unreadable_for(3, monkeypatch)
        assert await worker.index_done_callback() is True
        assert worker._loaded_fingerprint is None
        assert remaining[0] == 1, "the commit consumed an unexpected number"

        # Refused rather than served, so the mutation cannot land on a
        # snapshot whose relationship to the file is unknown. Retrying is what
        # the caller does -- at batch granularity that is the FAILED path
        # reprocessing the document.
        with pytest.raises(OSError, match="could not be sampled"):
            await worker.upsert_node("X", {"entity_id": "X"})
        assert remaining[0] == 0, "the file channel observed the file once"

        monkeypatch.undo()
        monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)
        assert file_fingerprint.fence_enabled() is True

        await worker.upsert_node("X", {"entity_id": "X"})
        await worker.upsert_node("Y", {"entity_id": "Y"})
        assert await worker.index_done_callback() is True

        on_disk = NetworkXStorage.load_nx_graph(worker._graphml_xml_file)
        # X is the one that used to vanish, and its loss was reported as a
        # successful commit.
        assert on_disk.has_node("X")
        assert on_disk.has_node("Y")
        assert on_disk.has_node("durable")
        # The retry's readable observation resolved the `None` through the
        # divergence test, which cannot tell this process's own file from a
        # peer's commit and counts one -- the documented bias towards
        # over-reporting (`file_fingerprint.adopted`), not evidence of a peer.
        assert worker._missed_notification_reloads == 1
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_an_unresolved_fingerprint_refuses_to_serve_the_graph(
    tmp_path, multiprocess, monkeypatch
):
    """While the `stat` keeps failing there is no safe way to serve it.

    Reloading discards the caller's own pending work; adopting the fresh
    `stat` without reloading would take a peer's commit for this process's
    own and overwrite it on the next save. So the resolution refuses, and the
    caller's batch takes the FAILED path until the `stat` works.
    """
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})

        remaining = _unreadable_for(3, monkeypatch)
        assert await worker.index_done_callback() is True
        assert worker._loaded_fingerprint is None
        assert remaining[0] == 1, "the commit consumed an unexpected number"

        # A read, not a mutation: what is unknown is the relationship between
        # the graph and the file, not the caller's intent, so this refuses too.
        with pytest.raises(OSError, match="could not be sampled"):
            await worker.has_node("durable")
        assert remaining[0] == 0, "the file channel observed the file once"

        # Nothing was adopted, so the next readable call still resolves it.
        monkeypatch.undo()
        monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)
        assert await worker.has_node("durable") is True
        assert worker._loaded_fingerprint is not None
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_an_unreadable_sample_never_installs_an_empty_graph(
    tmp_path, multiprocess, lost_notification
):
    """The same failure, at its worst: a whole workspace overwritten.

    `load_nx_graph` gates on `os.path.exists`, which is False for ANY stat
    failure -- the same one that produced `UNREADABLE`. So the load returned
    `None`, `or nx.Graph()` made that an empty graph, and while the stat kept
    failing the fence could not object (`divergence_detected` is False for
    `UNREADABLE`): the next commit serialized the empty graph over the real
    file. The class docstring names that outcome as the one thing this fence
    must never produce.
    """
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        with pytest.MonkeyPatch.context() as unreadable:
            # Both halves of the same stat failure: the fence cannot sample,
            # and `os.path.exists` reports the file gone.
            unreadable.setattr(
                file_fingerprint,
                "sample",
                lambda paths, *, workspace: file_fingerprint.UNREADABLE,
            )
            unreadable.setattr(
                NetworkXStorage, "load_nx_graph", staticmethod(lambda file_name: None)
            )
            worker.storage_updated.value = True
            with pytest.raises(OSError, match="could not be sampled"):
                await worker.has_node("durable")
            # And the commit that used to publish the empty graph cannot
            # proceed either: the refused reload left the flag set, so the
            # writer's own fence still owes a reload and refuses in turn.
            # Nothing is serialized at all -- which is why the file survives.
            with pytest.raises(OSError, match="could not be sampled"):
                await worker.index_done_callback()

        assert NetworkXStorage.load_nx_graph(worker._graphml_xml_file).has_node(
            "durable"
        )
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_an_unreadable_sample_leaves_the_recovery_reload_armed(
    tmp_path, multiprocess
):
    """A refusal must not discharge the reload this process owes itself.

    Same shape as a manager outage during recovery: the divergence stays
    visible, the flag stays armed, and the next readable call discards the
    unpersisted mutation.
    """
    worker = await _worker_with_a_failed_save(tmp_path)
    try:
        _unreadable_once(worker)
        with pytest.raises(OSError, match="could not be sampled"):
            await worker.has_node("durable")
        assert worker._recovery_reload_pending is True

        assert await worker.has_node("durable") is True
        assert await worker.has_node("never_saved") is False
        assert worker._recovery_reload_pending is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_an_unreadable_sample_adopts_nothing_and_keeps_the_marker(
    tmp_path, multiprocess, lost_notification
):
    """A reload cannot adopt what it could not sample — so it must not load.

    `adopted(UNREADABLE)` is `None`, which means "nothing recorded", and
    `peer_commit_detected` reports a change against `None` for any state.
    Recording it would forget which commit was already counted AND leave the
    next `_get_graph` reloading again, discarding whatever was mutated in
    between. `_reload_locked` therefore refuses an unreadable sample outright:
    the marker survives because nothing was adopted at all, which is the same
    guarantee reached one step earlier.
    """
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        # Counted once, and the reload fails, so the marker must survive --
        # and the failure arms the recovery reload, so the retry below goes
        # through that branch rather than the file channel.
        with pytest.MonkeyPatch.context() as broken_reads:
            broken_reads.setattr(
                NetworkXStorage, "load_nx_graph", staticmethod(_raise_unreadable)
            )
            with pytest.raises(OSError, match="unreadable"):
                await worker_b.has_node("from_a")
        assert worker_b._missed_notification_reloads == 1
        assert worker_b._recovery_reload_pending is True

        # The retry cannot sample, so it decides nothing, counts nothing and
        # adopts nothing -- and the marker it would otherwise have cleared has
        # to survive that. It refuses rather than serving: a graph this
        # process owes a reload for is not one it may hand out.
        _unreadable_once(worker_b)
        with pytest.raises(OSError, match="could not be sampled"):
            await worker_b.has_node("from_a")
        assert worker_b._loaded_fingerprint is not None
        assert worker_b._recovery_reload_pending is True

        # Same peer commit, still one event.
        assert await worker_b.has_node("from_a") is True
        assert worker_b._missed_notification_reloads == 1
        assert worker_b._recovery_reload_pending is False
    finally:
        await worker_b.finalize()
        await worker_a.finalize()
