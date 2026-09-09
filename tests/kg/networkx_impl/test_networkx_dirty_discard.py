"""A reload that discards uncommitted mutations must fail the WRITER's commit.

Issue #3899 R4. ``NetworkXStorage._get_graph`` reloads the whole graph from
disk whenever a peer commit lands, and a reload discards this process's
uncommitted in-memory mutations: the class has no pending buffer and no redo
log. Between batches that is intended. Reached mid-operation it drops work
silently -- the mutations are simply absent from the commit that follows, that
commit SUCCEEDS, and the document is marked PROCESSED (or the admin caller gets
200) without them.

The admin-write gate and the pipeline ``busy`` reservation make that reload
unreachable by enumeration of its callers, and an enumeration rots. This
backstop makes a bypass loud:

- every mutator marks the graph dirty; a reload that replaces a dirty graph
  logs at ERROR and arms ``_dirty_discard_pending`` -- it does NOT raise, since
  the coroutine that triggered it may be an innocent reader;
- ``index_done_callback`` refuses with ``GraphMutationsDiscardedError`` while
  the flag is set, and clears it in the same step so the following commit is
  not blocked;
- the recovery reload and the decline paths are exempt (what they discard was
  already reported as failed, or is already reported to the caller), but a
  notification whose fingerprint matches the loaded one is NOT: an equal
  fingerprint is the file channel's documented blind spot, not proof of a
  self-notification;
- reachable in SINGLE-process mode with two storage instances on one
  workspace, not only under multiprocess: each instance registers its own
  update flag, and one's commit makes the other reload.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lightrag.exceptions import GraphMutationsDiscardedError
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
    monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)


@pytest.fixture
def lost_notification(monkeypatch):
    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(networkx_impl, "set_all_update_flags", _noop)


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_storage(tmp_path) -> NetworkXStorage:
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


@pytest.mark.asyncio
async def test_writer_commit_raises_and_reader_does_not_single_process(tmp_path):
    """The defect, in single-process mode with two instances on one workspace.

    B holds an uncommitted mutation; A commits; a READER coroutine on B trips
    the reload (as a query would, across B's embedding await). The reader must
    not raise. B's own commit must -- instead of succeeding without ``from_b``.
    Revision 1 of the specification (raise inside the reload) fails this test:
    it would raise at the reader and let B's commit return True.
    """
    assert file_fingerprint.fence_enabled() is False
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        assert worker_b._graph_dirty is True

        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_a._graph_dirty is False
        assert worker_b.storage_updated.value is True

        # An innocent reader triggers the discarding reload: no exception.
        assert await worker_b.has_node("from_a") is True
        assert await worker_b.has_node("from_b") is False  # discarded
        assert worker_b._dirty_discard_pending is True
        assert worker_b._graph_dirty is False

        # The writer's commit is where it becomes loud.
        with pytest.raises(GraphMutationsDiscardedError):
            await worker_b.index_done_callback()

        # Not sticky: the refusal clears the flag, the following commit is fine.
        assert worker_b._dirty_discard_pending is False
        assert await worker_b.index_done_callback() is True

        # And nothing silently landed: A's commit is intact on disk.
        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        assert on_disk.has_node("from_a")
        assert not on_disk.has_node("from_b")
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_concurrent_reader_takes_no_error_while_writer_is_refused(tmp_path):
    """Same shape, with the reader genuinely concurrent: the writer is parked
    at an ``await`` between two mutations (its embedding round-trip) while a
    sibling coroutine reads, exactly as a query during an admin write would."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        parked = asyncio.Event()
        resume = asyncio.Event()

        async def writer():
            await worker_b.upsert_node("step_1", {"entity_id": "step_1"})
            parked.set()
            await resume.wait()  # the embedding round-trip
            await worker_b.upsert_node("step_2", {"entity_id": "step_2"})
            return await worker_b.index_done_callback()

        writer_task = asyncio.create_task(writer())
        await parked.wait()

        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        # The concurrent reader reloads and is fine.
        assert await worker_b.get_node("from_a") == {"entity_id": "from_a"}
        assert worker_b._dirty_discard_pending is True

        resume.set()
        with pytest.raises(GraphMutationsDiscardedError):
            await writer_task

        # step_2 was applied AFTER the discarding reload and belongs to the
        # operation that just failed: the refusal owes a recovery reload so
        # the next graph access discards it rather than a later commit
        # publishing it under someone else's name.
        assert worker_b._recovery_reload_pending is True
        assert await worker_b.has_node("step_2") is False
        assert worker_b._recovery_reload_pending is False
        assert await worker_b.index_done_callback() is True
        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        assert not on_disk.has_node("step_1") and not on_disk.has_node("step_2")
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_reachable_through_the_file_channel_under_multiprocess(
    tmp_path, multiprocess, lost_notification
):
    """A lost notification does not hide the discard: the file channel's
    reload arms the backstop the same way."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False  # notification lost

        assert await worker_b.has_node("from_a") is True  # file channel reload
        assert worker_b._dirty_discard_pending is True
        with pytest.raises(GraphMutationsDiscardedError):
            await worker_b.index_done_callback()
        assert worker_b._dirty_discard_pending is False
        assert await worker_b.index_done_callback() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_decline_path_does_not_arm_the_backstop(
    tmp_path, multiprocess, lost_notification
):
    """The #3854 decline already reports the loss (``False`` -> the caller
    raises); arming the backstop too would make the NEXT commit refuse for a
    loss that was already reported. Pins that ``index_done_callback``'s own
    reload is exempt."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        assert await worker_b.index_done_callback() is False  # declined
        assert worker_b._dirty_discard_pending is False
        assert worker_b._graph_dirty is False
        assert await worker_b.index_done_callback() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_recovery_reload_discards_silently(tmp_path):
    """The recovery branch exists to discard mutations of an operation already
    reported as failed; it must never arm the backstop (pinned alongside
    ``test_a_failed_save_forces_a_reload_in_single_process_mode``)."""
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("failed_op", {"entity_id": "failed_op"})
        assert worker._graph_dirty is True
        worker._recovery_reload_pending = True  # what a failed save leaves

        assert await worker.has_node("failed_op") is False  # discarded
        assert worker._dirty_discard_pending is False
        assert worker._recovery_reload_pending is False
        assert await worker.index_done_callback() is True
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_notification_matching_the_loaded_fingerprint_still_arms(
    tmp_path, multiprocess
):
    """An equal fingerprint does not prove the reload discards nothing.

    Issue #3899 R4 specified disarming the backstop here, calling this case a
    self-notification. But a same-tick, same-size peer commit is the file
    channel's documented blind spot, and a set flag is the evidence that the
    channel is blind right now -- the flag exists to cover that collision. So
    the reload still discards, and the next commit must still refuse.

    Reported by the Codex review of PR #3901 on d9ba12b. Verified red against
    the disarming version, which let the mutation vanish and the commit
    succeed -- the exact silent loss the backstop exists for.
    """
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("unpublished", {"entity_id": "unpublished"})
        worker.storage_updated.value = True  # matches the loaded fingerprint

        assert await worker.has_node("unpublished") is False  # reloaded anyway
        assert worker._dirty_discard_pending is True
        with pytest.raises(GraphMutationsDiscardedError):
            await worker.index_done_callback()
        # Not sticky: the refusal clears it and the next commit is fine.
        assert worker._dirty_discard_pending is False
        assert await worker.index_done_callback() is True
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_fingerprint_colliding_peer_commit_is_not_lost_silently(
    tmp_path, multiprocess, monkeypatch
):
    """The case the disarm actually exposed: a peer commit whose
    ``(st_mtime_ns, st_size)`` collides with the fingerprint this process
    loaded, so only the notification proves the file moved. The peer's content
    must not replace a dirty graph without a word."""
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        # B holds an uncommitted mutation; A commits over the same file.
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        # Force the collision: every sample B takes reports the identity B
        # already recorded, so its file channel sees "no change" and only the
        # notification is left.
        collided = worker_b._loaded_fingerprint
        monkeypatch.setattr(type(worker_b), "_stat_fingerprint", lambda self: collided)
        assert worker_b._peer_commit_detected(collided) is False
        assert worker_b.storage_updated.value is True

        assert await worker_b.has_node("from_a") is True  # A's content loaded
        assert await worker_b.has_node("from_b") is False  # B's was discarded
        assert worker_b._dirty_discard_pending is True
        with pytest.raises(GraphMutationsDiscardedError):
            await worker_b.index_done_callback()
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_every_mutator_marks_the_graph_dirty_and_commit_clears_it(tmp_path):
    worker = await _worker(tmp_path)
    try:
        assert worker._graph_dirty is False
        mutations = [
            lambda: worker.upsert_node("n1", {"entity_id": "n1"}),
            lambda: worker.upsert_edge("n1", "n2", {"weight": "1.0"}),
            lambda: worker.upsert_nodes_batch([("n3", {"entity_id": "n3"})]),
            lambda: worker.upsert_edges_batch([("n3", "n1", {"weight": "1.0"})]),
            lambda: worker.remove_edges([("n3", "n1")]),
            lambda: worker.remove_nodes(["n3"]),
            lambda: worker.delete_node("n2"),
        ]
        for mutate in mutations:
            await mutate()
            assert worker._graph_dirty is True
            assert await worker.index_done_callback() is True
            assert worker._graph_dirty is False
        # A no-op removal of an absent object leaves the graph clean.
        await worker.delete_node("never_there")
        await worker.remove_nodes(["never_there"])
        await worker.remove_edges([("never", "there")])
        assert worker._graph_dirty is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_drop_clears_both_backstop_records(tmp_path):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        await worker_b.has_node("from_a")
        assert worker_b._dirty_discard_pending is True

        assert (await worker_b.drop())["status"] == "success"
        assert worker_b._dirty_discard_pending is False
        assert worker_b._graph_dirty is False
        assert await worker_b.index_done_callback() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()
