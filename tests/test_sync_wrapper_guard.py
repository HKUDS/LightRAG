"""Regression tests for the synchronous-wrapper event-loop guard.

``LightRAG``'s synchronous wrappers (``insert``, ``query``,
``delete_by_entity`` …) all delegate to :func:`lightrag.lightrag._run_sync`,
which drives the matching ``a*`` coroutine via ``loop.run_until_complete()``.

That call is only valid when no event loop is already running on the current
thread.  Called from within a running loop it raises
``RuntimeError: This event loop is already running`` — a fail-fast error, *not*
a deadlock — and pushing it onto another thread's loop binds the shared
``asyncio.Lock`` objects in ``lightrag.kg.shared_storage`` to the wrong loop.
Both have the same fix: use the ``a*`` coroutine directly.

``_run_sync`` detects a running loop up front and raises one clear, actionable
error.  These tests pin that behaviour (and guard against regressing back to
the misleading "deadlock" wording / un-awaited-coroutine leak). See
HKUDS/LightRAG #1968.
"""

import asyncio

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
