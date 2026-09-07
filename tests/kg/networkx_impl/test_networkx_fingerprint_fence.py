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

from lightrag.kg import networkx_impl
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

    The fence is gated on ``is_multiprocess_mode()`` — in single-process mode a
    divergent file cannot be a peer commit. Tests that want the fence must say
    so; ``test_single_process_mode_ignores_a_divergent_file`` asserts the
    other side of that gate.
    """
    monkeypatch.setattr(networkx_impl, "is_multiprocess_mode", lambda: True)


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

        # A second commit whose GraphML is the same length as the first.
        await worker_a.delete_node("aa")
        await worker_a.upsert_node("bb", {"entity_id": "bb"})
        assert await worker_a.index_done_callback() is True

        # Force the tick collision. The size must already match, or this test
        # would be pinning nothing.
        os.utime(worker_b._graphml_xml_file, ns=(recorded[0], recorded[0]))
        assert os.stat(worker_b._graphml_xml_file).st_size == recorded[1]

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
        assert worker._loaded_fingerprint is None

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
    recovery block tries to arm the flag the manager is gone. Reads keep
    working so the call gets as far as the save.
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


@pytest.mark.asyncio
async def test_failed_save_arms_the_file_channel_even_if_the_flag_write_fails(
    tmp_path, multiprocess, monkeypatch
):
    """The recovery fence must survive the manager being gone.

    When a save fails, the in-memory graph holds a mutation the file does not
    have, and both channels are armed so the next `_get_graph` reloads instead
    of trusting it. The flag half is a Manager RPC, so it can fail for the very
    reason the reload just did. Armed after it, a manager outage would leave
    NEITHER channel armed, and a later flush could persist work already
    reported as failed.
    """
    worker = await _worker(tmp_path)
    real_flag = worker.storage_updated
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True
        assert worker._loaded_fingerprint is not None

        await worker.upsert_node("never_saved", {"entity_id": "never_saved"})

        def save_boom(graph, file_name, workspace):
            raise OSError("save boom")

        def reload_boom(file_name):
            raise OSError("reload boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
        monkeypatch.setattr(NetworkXStorage, "load_nx_graph", staticmethod(reload_boom))
        dead_flag = _WriteOnlyDeadFlag()
        worker.storage_updated = dead_flag

        # The SAVE error is what the caller must see -- not the manager outage
        # raised while arming the flag.
        with pytest.raises(OSError, match="save boom"):
            await worker.index_done_callback()

        # The flag arming was attempted and failed...
        assert dead_flag.write_attempts >= 1
        # ...and the file channel is armed regardless, which is the point.
        assert worker._loaded_fingerprint is None
    finally:
        worker.storage_updated = real_flag
        await worker.finalize()
