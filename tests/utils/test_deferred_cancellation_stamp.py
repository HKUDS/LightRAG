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
set exactly when a cancellation was withheld, that it rides the same exception
object the caller re-raises, and that it is absent everywhere else.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag.utils import (
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
