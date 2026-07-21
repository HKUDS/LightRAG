"""Behavior pins for the async keyed-lock registry: delete-on-release.

The per-process async lock table (``KeyedUnifiedLock._async_lock`` /
``_async_lock_count``) is refcount-live only: an entry exists ⟺ some
coroutine holds or awaits that key (count ≥ 1), and the release that takes
the count to 0 drops the entry immediately. The former idle cache (entries
parked at count 0 for up to 300s awaiting a throttled cleanup pass) is gone
— it was designed for the removed multiprocess lock registry, where caching
a ``manager.Lock()`` proxy saved RPCs; for local ``asyncio.Lock`` objects a
cache hit saves only a sub-microsecond allocation.

These tests pin the new invariant on the normal, contended, cancelled and
defensive paths, plus the /health schema compatibility of the status shells.
"""

import asyncio

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    _get_combined_key,
    cleanup_keyed_lock,
    finalize_share_data,
    get_keyed_lock_status,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline

STATUS_KEYS = {
    "total_mp_locks",
    "pending_mp_cleanup",
    "total_async_locks",
    "pending_async_cleanup",
}


def _registry():
    return shared_storage._storage_keyed_lock


async def _settle(ticks: int = 5):
    """Let already-started coroutines run up to their next await point."""
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.offline
async def test_entry_dropped_immediately_on_release():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "k")

        async with get_storage_keyed_lock("k", namespace="ns"):
            assert combined in keyed._async_lock
            assert keyed._async_lock_count[combined] == 1
            assert keyed.get_lock_status()["total_async_locks"] == 1

        # No idle cache: the exit that dropped the count to 0 removed the
        # entry from BOTH tables at once.
        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
        assert keyed.get_lock_status()["total_async_locks"] == 0
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_waiter_keeps_entry_alive_and_mutual_exclusion_holds():
    """Event-gated choreography (no wall-clock windows): the holder cannot
    release until the test has finished asserting, so the count == 2 state is
    observed deterministically even if the test process stalls."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "contended")
        active = 0
        max_active = 0
        holder_entered = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder():
            nonlocal active, max_active
            async with get_storage_keyed_lock("contended", namespace="ns"):
                active += 1
                max_active = max(max_active, active)
                holder_entered.set()
                await release_holder.wait()  # held until the test opens the gate
                active -= 1

        async def waiter():
            nonlocal active, max_active
            async with get_storage_keyed_lock("contended", namespace="ns"):
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0)  # yield once inside the critical section
                active -= 1

        a = asyncio.ensure_future(holder())
        await asyncio.wait_for(holder_entered.wait(), timeout=1.0)
        b = asyncio.ensure_future(waiter())
        # _settle is loop iterations, not wall time: the waiter registers its
        # reference synchronously before its first await, so this is enough.
        await _settle()

        # One demonstrably holds (gated), one waits: entry alive, both counted.
        assert keyed._async_lock_count[combined] == 2
        assert combined in keyed._async_lock

        release_holder.set()
        await asyncio.gather(a, b)
        assert max_active == 1  # never two holders at once
        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_cancelled_waiter_rolls_back_reference_without_deleting_entry():
    """The invariant's key exception branch: a WAITER cancelled mid-acquire
    must roll back exactly its own reference — the holder's entry survives as
    the SAME lock object (deleting it would let a later acquirer mint a fresh
    lock and run concurrently with the holder), and only the holder's release
    finally drops the entry."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "cancelwait")

        holder_ctx = get_storage_keyed_lock("cancelwait", namespace="ns")
        await holder_ctx.__aenter__()
        try:
            lock_obj = keyed._async_lock[combined]

            async def waiter():
                async with get_storage_keyed_lock("cancelwait", namespace="ns"):
                    pass  # pragma: no cover - never acquires in this test

            wtask = asyncio.ensure_future(waiter())
            await _settle()
            assert keyed._async_lock_count[combined] == 2

            wtask.cancel()
            with pytest.raises(asyncio.CancelledError):
                await wtask

            # Rollback decremented the waiter's reference only; the holder's
            # entry is intact and is the very same asyncio.Lock object.
            assert keyed._async_lock_count[combined] == 1
            assert keyed._async_lock[combined] is lock_obj
        finally:
            await holder_ctx.__aexit__(None, None, None)

        # The holder's release was the last reference: entry fully gone.
        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_reacquire_after_full_release_uses_fresh_lock():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "again")

        async with get_storage_keyed_lock("again", namespace="ns"):
            first = keyed._async_lock[combined]

        # A later, non-overlapping acquisition simply mints a fresh lock.
        async with get_storage_keyed_lock("again", namespace="ns"):
            second = keyed._async_lock[combined]
            assert second is not first

        assert combined not in keyed._async_lock
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_unmatched_release_is_ignored_and_leaves_no_phantom():
    """Regression: the old implementation wrote ``count - 1`` back
    unconditionally, so releasing an absent key created a phantom entry with
    count -1. An unmatched release must now be a logged no-op."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "never-acquired")

        keyed._release_async_lock(combined)  # must not raise

        assert combined not in keyed._async_lock_count
        assert combined not in keyed._async_lock
        assert keyed.get_lock_status()["total_async_locks"] == 0
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_distinct_keys_do_not_accumulate():
    """The conservation guarantee that replaces the deleted periodic cleanup:
    N distinct keys leave nothing behind, so unbounded key spaces (entity
    names) can never grow the registry."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        for i in range(500):
            async with get_storage_keyed_lock(f"entity-{i}", namespace="idx"):
                pass
        assert keyed._async_lock == {}
        assert keyed._async_lock_count == {}
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_multiprocess_gate_entry_dropped_immediately_too():
    """In multiprocess mode the registry entry is the per-process RPC-poll
    gate paired with the server-side lease; it follows the same
    delete-on-release lifecycle."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "mp")

        async with get_storage_keyed_lock("mp", namespace="ns"):
            assert combined in keyed._async_lock
            assert keyed._async_lock_count[combined] == 1

        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_status_shells_keep_health_schema():
    finalize_share_data()
    initialize_share_data(1)
    try:
        async with get_storage_keyed_lock("k", namespace="ns"):
            info = cleanup_keyed_lock()
            assert info["cleanup_performed"] == {"mp_cleaned": 0, "async_cleaned": 0}
            assert set(info["current_status"]) == STATUS_KEYS
            assert info["current_status"]["total_async_locks"] == 1
            assert info["current_status"]["pending_async_cleanup"] == 0

            public = get_keyed_lock_status()
            assert set(public) == STATUS_KEYS | {"process_id"}
            assert public["pending_async_cleanup"] == 0

        # Idle: instantaneous count back to zero, still full schema.
        info = cleanup_keyed_lock()
        assert set(info["current_status"]) == STATUS_KEYS
        assert info["current_status"]["total_async_locks"] == 0
        assert info["current_status"]["pending_async_cleanup"] == 0
    finally:
        finalize_share_data()

    # Uninitialized shared data: both shells still answer with the full schema.
    info = cleanup_keyed_lock()
    assert info["cleanup_performed"] == {"mp_cleaned": 0, "async_cleaned": 0}
    assert set(info["current_status"]) == STATUS_KEYS
    assert all(v == 0 for v in info["current_status"].values())
    public = get_keyed_lock_status()
    assert set(public) == STATUS_KEYS | {"process_id"}
