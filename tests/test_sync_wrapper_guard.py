"""Regression tests for the synchronous-wrapper event-loop guard.

``LightRAG``'s synchronous wrappers (``insert``, ``query``,
``delete_by_entity`` …) all delegate to :func:`lightrag.lightrag._run_sync`,
which drives the matching ``a*`` coroutine via ``loop.run_until_complete()``.

That call is only valid when (a) no event loop is already running on the
current thread, and (b) the loop it drives is the same one the instance's
storages were initialized on (``LightRAG._owning_loop``).  Two misuse modes
break this:

* Called from within a running loop it raises ``RuntimeError: This event loop
  is already running`` — a fail-fast error, *not* a deadlock.
* Driven from a different loop — e.g. ``loop.run_in_executor(None, rag.insert,
  …)`` runs the wrapper on a pool thread that spins up a fresh loop — binds the
  shared ``asyncio.Lock`` objects in ``lightrag.kg.shared_storage`` to the
  wrong loop (``<Lock> is bound to a different event loop`` / a stall).

Both have the same fix: use the ``a*`` coroutine directly.  ``_run_sync``
detects the running loop up front and compares the loop it is about to drive
against ``owning_loop``, raising one clear, actionable error for each.  These
tests pin that behaviour (and guard against regressing back to the misleading
"deadlock" wording / un-awaited-coroutine leak). See HKUDS/LightRAG #1968.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from lightrag.lightrag import _run_sync


@pytest.mark.offline
def test_run_sync_runs_coroutine_when_no_loop_running():
    """With no running loop, the coroutine runs to completion and returns."""

    def factory():
        async def _coro():
            return 42

        return _coro()

    result = _run_sync(factory, sync_name="insert", async_name="ainsert")
    assert result == 42


@pytest.mark.offline
def test_run_sync_raises_clear_error_inside_running_loop():
    """Inside a running loop the guard raises a RuntimeError pointing at the
    async alternative — and the message is honest (no "deadlock" claim)."""

    def factory():  # pragma: no cover - must never be reached
        raise AssertionError("coro_factory must not be called inside a loop")

    async def _inside_loop():
        return _run_sync(factory, sync_name="insert", async_name="ainsert")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(_inside_loop())

    message = str(exc_info.value)
    assert "insert()" in message
    assert "await ainsert(" in message
    # The previous fix (PR #3245) mislabelled this as a deadlock; it is an
    # immediate RuntimeError.  Keep the message free of that wording.
    assert "deadlock" not in message.lower()


@pytest.mark.offline
def test_run_sync_does_not_create_coroutine_when_it_raises():
    """The factory is invoked lazily, so a guard failure never leaves an
    un-awaited coroutine behind (which would emit a RuntimeWarning)."""

    calls: list[bool] = []

    def factory():
        calls.append(True)

        async def _coro():
            return None

        return _coro()

    async def _inside_loop():
        return _run_sync(factory, sync_name="query", async_name="aquery")

    with pytest.raises(RuntimeError):
        asyncio.run(_inside_loop())

    assert calls == [], "coro_factory should not run when the guard rejects"


@pytest.mark.offline
def test_run_sync_runs_when_owning_loop_matches():
    """When the loop being driven is the instance's owning loop, the coroutine
    runs to completion — the cross-loop guard only fires on a mismatch."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def factory():
        async def _coro():
            return 42

        return _coro()

    try:
        # always_get_an_event_loop() returns the loop set above, which is the
        # one we declare as owning -> match -> the coroutine runs.
        result = _run_sync(
            factory, sync_name="insert", async_name="ainsert", owning_loop=loop
        )
        assert result == 42
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.offline
def test_run_sync_raises_when_driven_from_a_different_loop():
    """Driving a sync wrapper from a thread with no event loop of its own
    (the ``loop.run_in_executor(None, rag.insert, …)`` anti-pattern) lands on a
    freshly created loop, not the instance's owning loop. The guard must fail
    fast pointing at the async alternative, without running the coroutine."""

    owning_loop = asyncio.new_event_loop()
    calls: list[bool] = []

    def factory():  # pragma: no cover - must never be reached
        calls.append(True)
        raise AssertionError("coro_factory must not run on the wrong loop")

    def call_off_thread():
        # No loop is set on this worker thread, so always_get_an_event_loop()
        # creates a brand-new one — different from owning_loop.
        return _run_sync(
            factory,
            sync_name="insert",
            async_name="ainsert",
            owning_loop=owning_loop,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(RuntimeError) as exc_info:
                executor.submit(call_off_thread).result()
    finally:
        owning_loop.close()

    message = str(exc_info.value)
    assert "insert()" in message
    assert "await ainsert(" in message
    assert "deadlock" not in message.lower()
    assert calls == [], "coro_factory should not run on a mismatched loop"
