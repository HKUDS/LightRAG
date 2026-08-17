"""PR-0 regression tests: cancellation safety of the keyed-lock release path.

Two distinct bugs are covered:

(0a) ``NamespaceLock.__aexit__`` cleared ``_ctx_var`` only AFTER awaiting
     ``ctx.__aexit__``. A cancellation delivered inside that await (it awaits an
     ``asyncio.shield``) skipped the clear, leaving a stale context; the SAME
     coroutine re-entering the lock in a ``finally`` then hit
     ``RuntimeError("already acquired")`` instead of a clean re-acquire — so any
     cancel-safe release built on top could never run.

(0b) ``_KeyedLockContext.__aexit__`` released the underlying locks on a shielded
     coroutine that read ``self._ul`` at run time, while the ``finally`` set
     ``self._ul = None``. Under (re-)cancellation the release could observe
     ``None`` and skip releasing the real lock, or ``__aexit__`` could return
     before the shielded release actually finished — either way the underlying
     keyed lock stayed held forever and every future acquirer deadlocked. The
     fix snapshots the entries and waits for a FIXED release task to complete,
     resisting repeated cancellation, before returning.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    KeyedUnifiedLock,
    NamespaceLock,
    _KeyedLockContext,
    _get_combined_key,
    finalize_share_data,
    get_storage_keyed_lock,
    initialize_share_data,
)


class _GatedLeafLock:
    """A leaf lock whose ``__aexit__`` blocks until the test opens a gate.

    Lets the test hold the release "in flight" while it injects a cancellation,
    so the resistant-wait behaviour of ``_KeyedLockContext.__aexit__`` can be
    observed deterministically.
    """

    def __init__(self) -> None:
        self.exit_started = asyncio.Event()
        self.release_gate = asyncio.Event()
        self.exit_completed = False

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_started.set()
        await self.release_gate.wait()
        self.exit_completed = True
        return False


def _context_holding(leaf: _GatedLeafLock) -> _KeyedLockContext:
    """Build a real ``_KeyedLockContext`` whose single acquired entry is ``leaf``.

    Reference-count release is stubbed out so the test isolates the __aexit__
    release/wait logic (the class under test is the real one).
    """
    parent = KeyedUnifiedLock(default_enable_logging=False)
    parent._release_lock_for_key = MagicMock()
    ctx = _KeyedLockContext(parent, namespace="ns", keys=["k"], enable_logging=False)
    ctx._ul = [
        {
            "key": "k",
            "lock": leaf,
            "entered": True,
            "debug_inc": False,
            "ref_incremented": False,
        }
    ]
    return ctx


@pytest.mark.offline
async def test_keyed_lock_aexit_waits_for_release_under_cancellation():
    """(0b) __aexit__ snapshots entries and does not return until the release
    actually completes, re-raising the cancellation only afterwards."""
    leaf = _GatedLeafLock()
    ctx = _context_holding(leaf)

    task = asyncio.ensure_future(ctx.__aexit__(None, None, None))

    # Release is now in flight (blocked on the gate).
    await asyncio.wait_for(leaf.exit_started.wait(), timeout=1.0)

    # Snapshot fix: self._ul is cleared up-front, yet the release still proceeds
    # via the captured snapshot (old code cleared it only in the finally).
    assert ctx._ul is None

    # Cancel while the release is still blocked.
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)

    # Resistant-wait fix: __aexit__ must keep waiting for the release task; the
    # leaf release has NOT completed and the coroutine is NOT done yet.
    # (Old code returned immediately on cancel, leaving the lock held.)
    assert not leaf.exit_completed
    assert not task.done()

    # Let the release finish; only then does __aexit__ return, re-raising cancel.
    leaf.release_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert leaf.exit_completed


@pytest.mark.offline
async def test_keyed_lock_aexit_raises_if_release_task_directly_cancelled():
    """(0b) If the internal release task itself is directly cancelled, __aexit__
    must propagate rather than pretend the lock was released."""
    leaf = _GatedLeafLock()
    ctx = _context_holding(leaf)

    task = asyncio.ensure_future(ctx.__aexit__(None, None, None))
    await asyncio.wait_for(leaf.exit_started.wait(), timeout=1.0)

    # Reach into the running release task and cancel IT directly.
    for child in asyncio.all_tasks():
        coro = child.get_coro()
        if getattr(coro, "__name__", "") == "release_all_locks":
            child.cancel()
            break
    else:  # pragma: no cover - defensive
        pytest.fail("release_all_locks task not found")

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not leaf.exit_completed


@pytest.mark.offline
async def test_keyed_lock_real_release_and_reacquire_after_cancel():
    """(0b) End-to-end with the REAL keyed lock: a task cancelled while holding
    the lock still releases it, so a competing task acquires it (no deadlock)
    and the reference count returns to zero."""
    finalize_share_data()
    initialize_share_data(1)  # single-process: real _KeyedLockContext, asyncio leaf
    try:
        namespace = "cancel_ns"
        combined = _get_combined_key(namespace, "k")
        holder_running = asyncio.Event()

        async def holder():
            async with get_storage_keyed_lock(["k"], namespace=namespace):
                holder_running.set()
                await asyncio.sleep(3600)  # hold until cancelled

        htask = asyncio.ensure_future(holder())
        await asyncio.wait_for(holder_running.wait(), timeout=1.0)

        # Cancel while it holds the lock; its async-with __aexit__ must release.
        htask.cancel()
        with pytest.raises(asyncio.CancelledError):
            await htask

        # A competing acquire must succeed promptly (would hang forever if the
        # release had been skipped).
        async with await asyncio.wait_for(_acquire_ctx(namespace, "k"), timeout=1.0):
            pass

        # Immediate-delete invariant: once the last reference is released —
        # including via the cancellation path — the registry entry is GONE,
        # not parked at count 0. (`.get(combined, 0) == 0` would pass for
        # both implementations; absence is the discriminating assertion.)
        keyed = shared_storage._storage_keyed_lock
        assert combined not in keyed._async_lock_count
        assert combined not in keyed._async_lock
    finally:
        finalize_share_data()


async def _acquire_ctx(namespace: str, key: str):
    """Enter a keyed-lock context and return it (helper for wait_for timeout)."""
    ctx = get_storage_keyed_lock([key], namespace=namespace)
    await ctx.__aenter__()
    return _EnteredCtx(ctx)


class _EnteredCtx:
    def __init__(self, ctx):
        self._ctx = ctx

    async def __aenter__(self):
        return self._ctx

    async def __aexit__(self, exc_type, exc, tb):
        return await self._ctx.__aexit__(exc_type, exc, tb)


class _FakeKeyedCtx:
    """Minimal ctx whose __aexit__ raises CancelledError on first call."""

    def __init__(self) -> None:
        self._raised = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if not self._raised:
            self._raised = True
            raise asyncio.CancelledError()
        return False


@pytest.mark.offline
async def test_namespace_lock_reentry_after_cancel_in_exit(monkeypatch):
    """(0a) A cancellation inside ctx.__aexit__ must still clear _ctx_var so the
    same NamespaceLock instance is re-entrable in the same coroutine context."""
    fake = _FakeKeyedCtx()
    monkeypatch.setattr(shared_storage, "get_storage_keyed_lock", lambda *a, **k: fake)

    lock = NamespaceLock("myns", "myws")
    await lock.__aenter__()
    assert lock._ctx_var.get() is fake

    with pytest.raises(asyncio.CancelledError):
        await lock.__aexit__(None, None, None)

    # Old code skipped `_ctx_var.set(None)` when ctx.__aexit__ raised.
    assert lock._ctx_var.get() is None

    # Re-entry in the SAME context must not raise "already acquired".
    await lock.__aenter__()
    assert lock._ctx_var.get() is fake
    await lock.__aexit__(None, None, None)
    assert lock._ctx_var.get() is None
