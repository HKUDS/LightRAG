"""Layer-2 dead-worker recovery for multiprocess keyed locks.

The multiprocess keyed lock is backed by a holder record in the server-side
``KeyedHolderTable`` (``{owner_pid, lease_id}`` from the client plus a
``start_delta`` identity stamped by the Manager server at grant time). A
SIGKILLed holder used to leave the manager mutex locked forever, deadlocking
every other worker; now the record is reclaimed atomically inside the server
on the next acquire IFF the owner process is *confirmed dead* (PID
gone/zombie, or a start-delta mismatch = PID reuse). A live-but-slow owner is
never preempted (dead-only), so no fencing token is needed.

These tests run with real multiprocess shared data (``initialize_share_data(2)``
creates a Manager) and drive the reclaim paths deterministically by seeding
holder records THROUGH the production proxy methods: ``try_acquire`` on an
empty slot accepts any ``{owner_pid, lease_id}`` record (the server stamps
``start_delta`` itself — for a dead PID that stamp is None and deadness is
decided by the PID probe), ``release`` with a known lease clears it. Reads go
through ``holders_snapshot()``.
"""

import asyncio
import os
import subprocess
import sys

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    KeyedHolderTable,
    _get_combined_key,
    _pid_alive,
    finalize_share_data,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline


def _table():
    return shared_storage._keyed_holder_table


def _snapshot():
    return _table().holders_snapshot()


def _seed_holder(key: str, owner_pid: int, lease_id: str) -> None:
    """Seed a holder record through the production grant path."""
    assert _table().try_acquire(key, {"owner_pid": owner_pid, "lease_id": lease_id})


def _dead_pid() -> int:
    """A confirmed-dead PID: spawn a child, SIGKILL it, reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


@pytest.mark.offline
async def test_lease_records_and_releases_holder():
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "k")
        assert key not in _snapshot()

        async with get_storage_keyed_lock("k", namespace="ns"):
            rec = _snapshot().get(key)
            assert rec is not None
            assert rec["owner_pid"] == os.getpid()
            assert rec["lease_id"]  # a lease id was stamped
            assert "start_delta" in rec  # identity stamped by the server

        # Released → holder record popped.
        assert key not in _snapshot()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_lease_reclaims_confirmed_dead_holder():
    """The core guarantee: a holder whose owner was SIGKILLed is reclaimed by the
    next acquirer instead of deadlocking it."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "dead")
        dead = _dead_pid()
        assert _pid_alive(dead) is False

        # Seed a stale holder as if a now-dead worker still held the lock.
        _seed_holder(key, dead, "stale")

        # Must acquire promptly (reclaim), not deadlock.
        async with asyncio.timeout(2):
            async with get_storage_keyed_lock("dead", namespace="ns"):
                rec = _snapshot().get(key)
                assert rec["owner_pid"] == os.getpid()  # reclaimed by us
                assert rec["lease_id"] != "stale"

        assert key not in _snapshot()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_lease_does_not_reclaim_live_owner():
    """A live (merely slow) owner must NEVER be reclaimed — dead-only."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "live")
        # A live owner: our own PID, but a DIFFERENT lease (as if another
        # coroutine/process legitimately holds it).
        _seed_holder(key, os.getpid(), "held-by-someone-else")

        async def _try_acquire():
            async with get_storage_keyed_lock("live", namespace="ns"):
                pass

        # The acquire must keep polling (never reclaim a live owner) → times out.
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(_try_acquire(), timeout=0.5)

        # The live owner's record is untouched.
        assert _snapshot().get(key, {}).get("lease_id") == "held-by-someone-else"
        assert _table().release(key, "held-by-someone-else") is True  # cleanup
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_lease_release_is_owner_checked():
    """Releasing must only pop OUR lease — a record taken over by a new owner
    (after a reclaim) must not be clobbered by a stale releaser."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "ownercheck")
        lock = get_storage_keyed_lock("ownercheck", namespace="ns")
        await lock.__aenter__()
        our_lease = _snapshot()[key]["lease_id"]
        assert _snapshot()[key]["owner_pid"] == os.getpid()

        # Simulate a new owner taking the slot while we still think we hold
        # it: vacate our (still-tracked) lease through the production release,
        # then grant the slot to a different lease.
        assert _table().release(key, our_lease) is True
        _seed_holder(key, os.getpid(), "new-owner")

        # Our release is owner-checked by lease_id → must NOT pop the new owner.
        await lock.__aexit__(None, None, None)
        assert _snapshot().get(key, {}).get("lease_id") == "new-owner"
        assert _table().release(key, "new-owner") is True  # cleanup
    finally:
        finalize_share_data()


def test_process_alive_detects_self_pid_reuse(monkeypatch):
    """_process_alive (the reservation layer's liveness check) must NOT blindly
    treat a record carrying our own PID as alive: a dead predecessor whose PID
    the OS reused for us leaves a record with our PID but a different start id,
    and reporting it alive would wedge the reservation forever. Monkeypatching
    ``_my_start_id`` exercises the branch without needing real /proc reuse."""
    monkeypatch.setattr(shared_storage, "_my_start_id", lambda: "our-start-id")
    mypid = os.getpid()
    # Same PID, DIFFERENT start id = a dead predecessor that reused our PID.
    assert shared_storage._process_alive(mypid, "predecessor-start-id") is False
    # Same PID + matching start id = genuinely us → alive.
    assert shared_storage._process_alive(mypid, "our-start-id") is True
    # Same PID, no recorded start id = cannot confirm reuse → conservatively alive.
    assert shared_storage._process_alive(mypid, None) is True
    # Our start id unknown (non-Linux) = cannot confirm reuse → alive.
    monkeypatch.setattr(shared_storage, "_my_start_id", lambda: None)
    assert shared_storage._process_alive(mypid, "predecessor-start-id") is True


@pytest.mark.offline
async def test_lease_reclaims_on_self_pid_reuse(monkeypatch):
    """The real dead-worker-with-PID-reuse deadlock: the reclaiming worker was
    handed the dead owner's very PID by the OS. The stale holder therefore
    carries OUR pid but the predecessor's start_delta — the lease must still be
    reclaimed (our recomputed delta differs from the recorded one).

    A record with a live PID and a forged start_delta cannot be injected
    through the proxy (the server always stamps the true value), so this uses
    the direct-instance replacement pattern: multiprocess mode is established
    first (``initialize_share_data(1)`` would use asyncio locks and never touch
    the holder table), then the proxy is swapped for a local KeyedHolderTable
    whose internals the test may set directly. start_delta detection is
    platform-independent, so no Linux skip is needed."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        our_delta = shared_storage._start_delta(os.getpid())
        if our_delta is None:
            pytest.skip("no process start identity available on this platform")

        table = KeyedHolderTable()
        key = _get_combined_key("ns", "reuse")
        # Our PID, but a predecessor's start_delta: on Linux any tick
        # difference means a different process; off-Linux the recomputed delta
        # exceeds the forged stamp by more than the 1s tolerance.
        with table._lock:
            table._holders[key] = {
                "owner_pid": os.getpid(),
                "lease_id": "stale",
                "start_delta": our_delta - 2,
            }
        monkeypatch.setattr(shared_storage, "_keyed_holder_table", table)

        async with asyncio.timeout(5):
            async with get_storage_keyed_lock("reuse", namespace="ns"):
                assert table.holders_snapshot()[key]["lease_id"] != "stale"
        assert key not in table.holders_snapshot()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_namespace_lock_reclaims_dead_holder():
    """The pipeline_status lock (NamespaceLock → keyed lock) recovers from a
    SIGKILLed holder through the full stack (NamespaceLock → _KeyedLockContext →
    UnifiedLock → holder lease)."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        from lightrag.kg.shared_storage import (
            get_final_namespace,
            get_namespace_lock,
        )

        final_ns = get_final_namespace("pipeline_status", "ws")
        key = _get_combined_key(final_ns, "default_key")
        dead = _dead_pid()
        _seed_holder(key, dead, "stale")

        lock = get_namespace_lock("pipeline_status", workspace="ws")
        async with asyncio.timeout(2):
            async with lock:
                assert _snapshot()[key]["owner_pid"] == os.getpid()  # reclaimed
        assert key not in _snapshot()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_single_process_keyed_lock_has_no_holder_records():
    """Single-process mode keeps the asyncio.Lock path — no holder table, no
    dead-owner machinery."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        assert shared_storage._keyed_holder_table is None
        async with get_storage_keyed_lock("k", namespace="ns"):
            pass  # works without any holder-record bookkeeping
    finally:
        finalize_share_data()
