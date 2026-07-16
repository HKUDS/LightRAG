"""Layer-2 dead-worker recovery for multiprocess keyed locks.

The multiprocess keyed lock is backed by a holder record in
``_keyed_lock_holders`` (``{owner_pid, process_start_id, lease_id}``) instead of
a ``manager.Lock()``. A SIGKILLed holder used to leave the manager mutex locked
forever, deadlocking every other worker; now the record is reclaimed lazily on
the next acquire IFF the owner process is *confirmed dead* (PID gone, or a
Linux ``/proc`` start-time mismatch = PID reuse). A live-but-slow owner is never
preempted (dead-only), so no fencing token is needed.

These tests run with real multiprocess shared data (``initialize_share_data(2)``
creates a Manager) and drive the reclaim paths deterministically by seeding
holder records.
"""

import asyncio
import os
import subprocess
import sys

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    _get_combined_key,
    _my_start_id,
    _pid_alive,
    finalize_share_data,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline


def _holders():
    return shared_storage._keyed_lock_holders


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
        assert key not in _holders()

        async with get_storage_keyed_lock("k", namespace="ns"):
            rec = _holders().get(key)
            assert rec is not None
            import os

            assert rec["owner_pid"] == os.getpid()
            assert rec["lease_id"]  # a lease id was stamped

        # Released → holder record popped.
        assert key not in _holders()
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
        _holders()[key] = {
            "owner_pid": dead,
            "process_start_id": "gone",
            "lease_id": "stale",
        }

        # Must acquire promptly (reclaim), not deadlock.
        async with asyncio.timeout(2):
            async with get_storage_keyed_lock("dead", namespace="ns"):
                rec = _holders().get(key)
                import os

                assert rec["owner_pid"] == os.getpid()  # reclaimed by us
                assert rec["lease_id"] != "stale"

        assert key not in _holders()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_lease_does_not_reclaim_live_owner():
    """A live (merely slow) owner must NEVER be reclaimed — dead-only."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        import os

        key = _get_combined_key("ns", "live")
        # A live owner: our own PID + matching start id, but a DIFFERENT lease
        # (as if another coroutine/process legitimately holds it).
        _holders()[key] = {
            "owner_pid": os.getpid(),
            "process_start_id": _my_start_id(),
            "lease_id": "held-by-someone-else",
        }

        async def _try_acquire():
            async with get_storage_keyed_lock("live", namespace="ns"):
                pass

        # The acquire must keep polling (never reclaim a live owner) → times out.
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(_try_acquire(), timeout=0.5)

        # The live owner's record is untouched.
        assert _holders().get(key, {}).get("lease_id") == "held-by-someone-else"
        _holders().pop(key, None)  # cleanup
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_lease_release_is_owner_checked():
    """Releasing must only pop OUR lease — a record taken over by a new owner
    (after a reclaim) must not be clobbered by a stale releaser."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        import os

        key = _get_combined_key("ns", "ownercheck")
        lock = get_storage_keyed_lock("ownercheck", namespace="ns")
        await lock.__aenter__()
        assert _holders()[key]["owner_pid"] == os.getpid()

        # Simulate a new owner taking the slot while we still think we hold it.
        _holders()[key] = {
            "owner_pid": os.getpid(),
            "process_start_id": _my_start_id(),
            "lease_id": "new-owner",
        }

        # Our release is owner-checked by lease_id → must NOT pop the new owner.
        await lock.__aexit__(None, None, None)
        assert _holders().get(key, {}).get("lease_id") == "new-owner"
        _holders().pop(key, None)  # cleanup
    finally:
        finalize_share_data()


def test_process_alive_detects_self_pid_reuse(monkeypatch):
    """_process_alive must NOT blindly treat a record carrying our own PID as
    alive: a dead predecessor whose PID the OS reused for us leaves a record with
    our PID but a different start id, and reporting it alive would wedge the lock
    forever (no other worker competes for the key). This is the exact deadlock a
    naive ``pid == os.getpid() -> True`` short-circuit causes. Monkeypatching
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


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PID-reuse detection needs Linux /proc start time",
)
@pytest.mark.offline
async def test_lease_reclaims_on_self_pid_reuse():
    """The real dead-worker-with-PID-reuse deadlock: the reclaiming worker was
    handed the dead owner's very PID by the OS. The stale holder therefore
    carries OUR pid but the predecessor's start id — the lease must still be
    reclaimed (our start id differs from the recorded one)."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "reuse")
        # Our PID, but a start id that is not ours = we reused a dead worker's PID.
        _holders()[key] = {
            "owner_pid": os.getpid(),
            "process_start_id": "predecessor-start-id",
            "lease_id": "stale",
        }
        async with asyncio.timeout(5):
            async with get_storage_keyed_lock("reuse", namespace="ns"):
                assert _holders()[key]["lease_id"] != "stale"  # reclaimed by us
        assert key not in _holders()
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
        import os
        from lightrag.kg.shared_storage import (
            get_final_namespace,
            get_namespace_lock,
        )

        final_ns = get_final_namespace("pipeline_status", "ws")
        key = _get_combined_key(final_ns, "default_key")
        dead = _dead_pid()
        _holders()[key] = {
            "owner_pid": dead,
            "process_start_id": "gone",
            "lease_id": "stale",
        }

        lock = get_namespace_lock("pipeline_status", workspace="ws")
        async with asyncio.timeout(2):
            async with lock:
                assert _holders()[key]["owner_pid"] == os.getpid()  # reclaimed
        assert key not in _holders()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_single_process_keyed_lock_has_no_holder_records():
    """Single-process mode keeps the asyncio.Lock path — no holder records, no
    dead-owner machinery."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        assert shared_storage._keyed_lock_holders is None
        async with get_storage_keyed_lock("k", namespace="ns"):
            pass  # works without any holder-record bookkeeping
    finally:
        finalize_share_data()
