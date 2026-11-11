"""A withheld cancellation must be distinguishable from an ordinary one.

``_wait_deferring_cancellation`` exists so a write that must not be interrupted
finishes even after its caller was cancelled: the cancellation is held and
re-raised once the region is done. That means the operation reporting failure
may have COMMITTED, and a caller that rewrites the cancellation into an error of
its own (``LightRAG._AdminHoldCeiling``) would otherwise report a durable write
as one that did not happen -- what ``AGENTS.md`` *Consistency without
transactions* forbids.

The helper therefore stamps the very instance it returns, and
``cancellation_was_deferred`` reads it back. These tests pin that the stamp is
set exactly when a cancellation was withheld across work that SUCCEEDED, that it
rides the same exception object the caller re-raises, that a stamp from an
earlier step survives a later failure, and that it is absent everywhere else.
"""

from __future__ import annotations

import asyncio

import pytest

from concurrent.futures import ThreadPoolExecutor

from lightrag.utils import (
    CommitBookkeepingError,
    _bounded_submit_impl,
    _wait_deferring_cancellation,
    cancellation_was_deferred,
)

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_a_withheld_cancellation_is_stamped():
    """The caller is cancelled mid-region; the region still finishes, and the
    cancellation handed back says so."""
    landed = []

    async def _uninterruptible():
        await asyncio.sleep(0.05)
        landed.append("committed")

    async def _region():
        future = asyncio.ensure_future(_uninterruptible())
        return await _wait_deferring_cancellation(future, None)

    task = asyncio.ensure_future(_region())
    await asyncio.sleep(0)  # let the region start its work
    task.cancel()
    pending = await task

    assert landed == ["committed"]  # the write was NOT torn apart
    assert isinstance(pending, asyncio.CancelledError)
    assert cancellation_was_deferred(pending) is True


@pytest.mark.asyncio
async def test_work_that_FAILED_is_not_stamped():
    """The stamp claims success, not merely that the region ran.

    ``_bounded_submit_impl`` gives the withheld cancellation precedence over the
    work's own error and only logs the error, so downstream the two look
    identical. Stamping both would tell a caller their write is durable when the
    write raised, which is the mirror of the defect the stamp prevents.

    Reported by the Codex review of PR #3901 on f26cd98.
    """

    async def _fails():
        await asyncio.sleep(0.05)
        raise OSError(28, "No space left on device")

    async def _region():
        future = asyncio.ensure_future(_fails())
        return await _wait_deferring_cancellation(future, None)

    task = asyncio.ensure_future(_region())
    await asyncio.sleep(0)
    task.cancel()
    pending = await task

    # The cancellation is still withheld and handed back ...
    assert isinstance(pending, asyncio.CancelledError)
    # ... but it carries no claim of durability.
    assert cancellation_was_deferred(pending) is False


@pytest.mark.asyncio
async def test_a_stamp_from_an_earlier_step_survives_a_later_failure():
    """``_bounded_submit_impl`` calls this twice with the same instance: once
    for the write, once for the commit hook. A write that succeeded IS durable,
    so a failing hook afterwards must not retract the stamp."""

    async def _ok():
        await asyncio.sleep(0.05)

    async def _fails():
        await asyncio.sleep(0.05)
        raise RuntimeError("publication failed")

    async def _region():
        write = asyncio.ensure_future(_ok())
        pending = await _wait_deferring_cancellation(write, None)
        hook = asyncio.ensure_future(_fails())
        return await _wait_deferring_cancellation(hook, pending)

    task = asyncio.ensure_future(_region())
    await asyncio.sleep(0)
    task.cancel()
    pending = await task

    assert cancellation_was_deferred(pending) is True


@pytest.mark.asyncio
async def test_no_cancellation_means_no_stamp():
    """Nothing was withheld, so there is nothing to warn a caller about."""

    async def _work():
        return "done"

    future = asyncio.ensure_future(_work())
    assert await _wait_deferring_cancellation(future, None) is None


@pytest.mark.asyncio
async def test_the_stamp_survives_the_re_raise_the_caller_does():
    """The stamp rides the exception object itself, which is why it needs no
    contextvar: the instance the helper returns is the instance the caller
    re-raises, and the admin flows in between catch ``Exception``, never
    ``BaseException``."""

    async def _uninterruptible():
        await asyncio.sleep(0.05)

    async def _region():
        future = asyncio.ensure_future(_uninterruptible())
        pending = await _wait_deferring_cancellation(future, None)
        if pending is not None:
            raise pending

    task = asyncio.ensure_future(_region())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert cancellation_was_deferred(excinfo.value) is True


@pytest.mark.asyncio
async def test_a_plain_cancellation_at_a_suspension_point_is_not_stamped():
    """The distinction the ceiling rests on: a cancellation delivered at an
    ordinary await never passed through a region that had to finish."""

    async def _plain():
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_plain())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert cancellation_was_deferred(excinfo.value) is False


def test_cancellation_was_deferred_is_false_for_anything_else():
    assert cancellation_was_deferred(RuntimeError("boom")) is False
    assert cancellation_was_deferred(asyncio.CancelledError()) is False


# ---------------------------------------------------------------------------
# The stamp describes the OPERATION, which only ``_bounded_submit_impl`` knows
# ---------------------------------------------------------------------------


async def _run_bounded(fn, on_committed):
    """Submit ``fn`` through ``_bounded_submit_impl`` and cancel the caller
    once the commit hook has started -- the window the stamp has to cover."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        semaphore = asyncio.Semaphore(1)
        hook_started = asyncio.Event()

        async def _hook():
            hook_started.set()
            await asyncio.sleep(0.05)
            return await on_committed()

        task = asyncio.ensure_future(
            _bounded_submit_impl(
                executor,
                semaphore,
                fn,
                (),
                {},
                wait_for_completion=True,
                on_committed=_hook,
            )
        )
        await hook_started.wait()
        task.cancel()
        return task
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_a_write_that_landed_is_stamped_even_when_its_commit_hook_fails():
    """The write is on disk; only its publication failed.

    ``_wait_deferring_cancellation`` judges the future it was handed, and in
    this ordering neither call ever sees a successful one: the write finished
    before the cancel arrived (so that call withheld nothing), and the hook the
    cancel DID cross then raised. The operation is nonetheless durable, so
    ``_bounded_submit_impl`` stamps on its behalf -- otherwise the ceiling's
    message and ``NetworkXStorage.index_done_callback`` both treat a durable
    write as one that never happened.
    """
    written = []

    def _write():
        written.append("landed")
        return "result"

    async def _publish():
        raise RuntimeError("could not flip the reload flags")

    task = await _run_bounded(_write, _publish)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert written == ["landed"]
    assert cancellation_was_deferred(excinfo.value) is True


@pytest.mark.asyncio
async def test_a_write_that_RAISED_is_still_not_stamped():
    """The mirror case, unchanged: ``fn`` raising means nothing was persisted,
    the hook never runs, and the cancellation must carry no claim of
    durability."""

    def _write():
        raise OSError(28, "No space left on device")

    async def _publish():  # pragma: no cover - must never run
        raise AssertionError("the commit hook must not run after a failed write")

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        semaphore = asyncio.Semaphore(1)

        async def _region():
            future = asyncio.ensure_future(
                _bounded_submit_impl(
                    executor,
                    semaphore,
                    _write,
                    (),
                    {},
                    wait_for_completion=True,
                    on_committed=_publish,
                )
            )
            await asyncio.sleep(0)
            return future

        task = await _region()
        task.cancel()
        with pytest.raises(BaseException) as excinfo:
            await task
    finally:
        executor.shutdown(wait=True)

    # Either the cancellation won the race or the write error did; whichever
    # surfaced, no durability may be claimed.
    assert cancellation_was_deferred(excinfo.value) is False
    assert not isinstance(excinfo.value, CommitBookkeepingError)
