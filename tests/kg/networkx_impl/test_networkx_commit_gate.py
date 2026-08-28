"""GraphML serialization off the event loop, and the gate that makes it safe.

The commit hands ``self._graph`` to a worker thread that iterates
``graph._node`` / ``graph._adj``. Invariant (3) in the class docstring — "networkx
operations are fully synchronous, so the event loop gives them implicit mutual
exclusion" — stops covering that, because the loop keeps running. ``_commit_gate``
re-establishes the exclusion.

Holding ``_storage_lock`` across the write is NOT sufficient on its own, and
``test_a_mutator_suspended_in_lock_release_cannot_mutate_during_a_commit`` is
about exactly that gap: ``NamespaceLock.__aexit__`` runs its release on a fresh
task and awaits a shield, so it always suspends at least once. A mutator
suspended there is already past the lock body and would resume mid-write.

Note what each test proves. The heartbeat test is the one that fails against the
pre-change code. The gate tests are vacuous there (no thread, no window); what
they pin is the intermediate mistake of offloading the write WITHOUT the gate,
verified by hand while writing them:

    $ # with `gate.clear()` / `finally: gate.set()` removed
    $ pytest tests/kg/networkx_impl/test_networkx_commit_gate.py
    FAILED ...::test_a_mutator_suspended_in_lock_release_cannot_mutate_during_a_commit
    E   AssertionError: mutator was not held back by the gate
    FAILED ...::test_gate_is_restored_when_the_committer_is_cancelled
    E   AssertionError: gate must stay closed during the write
"""

import asyncio
import threading
import time

import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline

BLOCK_SECONDS = 0.2


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _make_storage(tmp_path) -> NetworkXStorage:
    storage = NetworkXStorage(
        namespace="chunk_entity_relation",
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
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
    """Fix-proof: called inline, ``time.sleep`` freezes the loop and beats tie.

    This also confirms the monkeypatch still bites, i.e. that the commit
    resolves ``write_nx_graph`` at call time instead of hoisting a reference —
    the same property ``test_networkx_index_done.py`` depends on to prove the
    save error is not swallowed.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert_node("n1", {"entity_id": "n1"})

    observed = []
    heartbeat = _Heartbeat().start()
    real_write = NetworkXStorage.write_nx_graph

    def blocking_write(graph, file_name, workspace="_"):
        before = heartbeat.beats
        time.sleep(BLOCK_SECONDS)
        observed.append((before, heartbeat.beats))
        return real_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(blocking_write))

    await storage.index_done_callback()
    await heartbeat.stop()

    assert observed, "write_nx_graph was never called"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole write"
    assert await storage.has_node("n1")


async def test_a_mutator_suspended_in_lock_release_cannot_mutate_during_a_commit(
    tmp_path, monkeypatch
):
    """The window the lock alone leaves open, closed by the gate.

    A mutator is parked inside ``NamespaceLock.__aexit__`` — past the lock body,
    holding a reference to the graph, about to mutate it. The committer then
    starts. The writer thread asserts the graph does not change under it.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert_node("n1", {"entity_id": "n1"})

    # Park the next lock release until we say so.
    from lightrag.kg import shared_storage

    release_reached = asyncio.Event()
    let_release_finish = asyncio.Event()
    arm = {"on": False}
    real_aexit = shared_storage.NamespaceLock.__aexit__

    async def parking_aexit(self, *exc_info):
        result = await real_aexit(self, *exc_info)
        if arm["on"]:
            arm["on"] = False
            release_reached.set()
            await let_release_finish.wait()
        return result

    monkeypatch.setattr(shared_storage.NamespaceLock, "__aexit__", parking_aexit)

    async def suspended_mutator():
        arm["on"] = True
        await storage.upsert_node("n2", {"entity_id": "n2"})

    mutator = asyncio.create_task(suspended_mutator())
    await asyncio.wait_for(release_reached.wait(), timeout=2)

    entered = threading.Event()
    release_write = threading.Event()
    stable = {"ok": None}
    real_write = NetworkXStorage.write_nx_graph

    def blocking_write(graph, file_name, workspace="_"):
        seen = graph.number_of_nodes()
        entered.set()
        assert release_write.wait(timeout=5), "write was never released"
        stable["ok"] = graph.number_of_nodes() == seen
        return real_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(blocking_write))

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(entered.is_set)

    # Release the mutator: it resumes past the lock and tries to mutate while
    # the writer thread is iterating.
    let_release_finish.set()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not mutator.done(), "mutator was not held back by the gate"

    release_write.set()
    await commit
    await mutator

    assert stable["ok"] is True, "graph mutated while the writer was iterating it"
    assert await storage.has_node("n2"), "the mutation must still land afterwards"


async def test_gate_is_restored_when_the_write_raises(tmp_path, monkeypatch):
    """A failed commit must not leave the gate closed.

    ``asyncio.wait_for`` turns "gate leaked" from a hang into a failure.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert_node("n1", {"entity_id": "n1"})

    def boom(graph, file_name, workspace="_"):
        raise OSError("save boom")

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))

    with pytest.raises(OSError, match="save boom"):
        await storage.index_done_callback()

    assert storage._gate().is_set()
    await asyncio.wait_for(storage.upsert_node("n2", {"entity_id": "n2"}), timeout=2)


async def test_gate_is_restored_when_the_committer_is_cancelled(tmp_path, monkeypatch):
    """Cancellation must restore the gate, and only after the worker is done."""
    storage = await _make_storage(tmp_path)
    await storage.upsert_node("n1", {"entity_id": "n1"})

    entered = threading.Event()
    release_write = threading.Event()
    real_write = NetworkXStorage.write_nx_graph

    def blocking_write(graph, file_name, workspace="_"):
        entered.set()
        assert release_write.wait(timeout=5), "write was never released"
        return real_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(blocking_write))

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(entered.is_set)

    commit.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not commit.done(), "committer returned while the writer was running"
    assert not storage._gate().is_set(), "gate must stay closed during the write"

    release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await commit

    assert storage._gate().is_set()
    await asyncio.wait_for(storage.upsert_node("n2", {"entity_id": "n2"}), timeout=2)


async def test_a_woken_waiter_rechecks_the_gate_before_proceeding(tmp_path):
    """``while not is_set(): await wait()``, not a single ``wait()``.

    ``Event.wait()`` returns True for every waiter the ``set()`` woke, even if
    the flag was cleared again before that waiter got to run. A second committer
    taking over immediately after the first one finishes is exactly that
    sequence, so a lone ``await gate.wait()`` would let the mutator through
    while the new commit is already serializing.

    Driven through the gate directly rather than through two real commits: a
    mutator queued behind a running commit waits on ``_storage_lock``, not on
    the gate, and the lock is FIFO — it would be handed the lock before the
    second committer ever got it. The interleaving this pins is a property of
    the gate, and this is the deterministic way to reach it.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert_node("n1", {"entity_id": "n1"})

    gate = storage._gate()
    gate.clear()

    mutator = asyncio.create_task(storage.upsert_node("n2", {"entity_id": "n2"}))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not mutator.done(), "mutator did not stop at the closed gate"
    # Read the graph directly: has_node() goes through _get_graph too, so it
    # would queue on the gate as well and hang this test.
    assert not storage._graph.has_node("n2")

    # First committer finishes and the second one takes over before the woken
    # mutator is scheduled.
    gate.set()
    gate.clear()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not mutator.done(), "waiter slipped through after being woken"
    assert not storage._graph.has_node("n2")

    gate.set()
    await asyncio.wait_for(mutator, timeout=2)
    assert storage._graph.has_node("n2")


async def test_cancelled_commit_still_notifies_the_other_processes(
    tmp_path, monkeypatch
):
    """A cancelled commit must not publish the file and skip the notification.

    ``set_all_update_flags`` is what tells every other worker to reload. Skipped,
    they keep serving the previous graph until some later commit happens to
    notify them. The write is already durable at that point, so the bookkeeping
    runs inside the same uncancellable region (``commit_in_storage_io``).

    Fix-proof: inline ``set_all_update_flags`` after the offload instead, and
    ``flagged`` stays empty — this is the P2 Codex raised on #3740.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert_node("A", {"entity_id": "A"})

    flagged: list[str] = []
    real_write = NetworkXStorage.write_nx_graph
    inside_write = threading.Event()
    may_finish = threading.Event()

    async def spy_set_all_update_flags(namespace, workspace=None):
        flagged.append(namespace)

    def parked_write(graph, path, workspace="_"):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_write(graph, path, workspace)

    monkeypatch.setattr(
        "lightrag.kg.networkx_impl.set_all_update_flags", spy_set_all_update_flags
    )
    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(parked_write))

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(inside_write.is_set)

    commit.cancel()
    may_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await commit

    assert flagged == ["chunk_entity_relation"], (
        "a cancelled commit published the graph without telling the other "
        f"processes to reload it (flagged={flagged})"
    )
    # And the gate is back open, so later graph work is not deadlocked.
    await asyncio.wait_for(storage.upsert_node("B", {"entity_id": "B"}), timeout=2)


def test_gate_is_rebound_after_the_owning_loop_closes(tmp_path):
    """One instance, two loops: the gate must not stay bound to the dead one.

    ``asyncio.Event`` binds to a loop the first time a ``wait()`` suspends on
    it, and afterwards ``wait()`` from another loop raises "is bound to a
    different event loop". A ``NetworkXStorage`` can outlive its loop — the
    synchronous wrappers let calls through on a fresh loop once the original
    ``owning_loop`` has closed (``_run_sync``, i.e. the common
    ``rag = asyncio.run(initialize_rag())`` shape) — so a gate bound during an
    earlier commit would break the next one.

    Deliberately NOT an async test: it needs two real loops, and the failure
    only appears on the second one. Fix-proof: drop the ``_commit_gate_loop``
    check from ``_gate()`` and the second ``asyncio.run`` raises RuntimeError.
    """
    initialize_share_data()
    try:
        storage = None

        async def first_loop():
            nonlocal storage
            storage = await _make_storage(tmp_path)
            await storage.upsert_node("A", {"entity_id": "A"})
            # Force a real suspension on the gate so it binds to THIS loop.
            gate = storage._gate()
            gate.clear()
            waiter = asyncio.create_task(storage._get_graph())
            await _wait_for(lambda: not waiter.done())
            for _ in range(5):
                await asyncio.sleep(0)
            gate.set()
            await waiter
            assert storage._commit_gate_loop is asyncio.get_running_loop()

        asyncio.run(first_loop())

        bound_to = storage._commit_gate

        async def second_loop():
            # A commit that closes and reopens the gate: with the stale Event
            # still in place, the mutator's wait() below raises RuntimeError.
            await storage.index_done_callback()
            await asyncio.wait_for(
                storage.upsert_node("B", {"entity_id": "B"}), timeout=2
            )
            assert storage._commit_gate is not bound_to, "gate was not rebound"
            assert storage._commit_gate_loop is asyncio.get_running_loop()

        asyncio.run(second_loop())
    finally:
        finalize_share_data()
