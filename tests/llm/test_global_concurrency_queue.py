"""Offline tests for priority_limit_async_func_call's cross-worker global
concurrency mode (concurrency_group + shared-storage gating).

Single-process shared storage exercises the exact code paths used under
gunicorn multi-worker mode (namespaces and keyed locks degrade to in-process
primitives).
"""

import asyncio
import contextvars

import pytest

from lightrag.kg import shared_storage as ss

# Accessed via the module (not ``from``-imports) on purpose: another test in
# the suite reloads lightrag.utils in place, which would leave from-imported
# class references (e.g. QueueFullError) pointing at the pre-reload class and
# break pytest.raises identity checks when run in the same xdist worker.
from lightrag import utils as lr_utils

pytestmark = pytest.mark.offline


GROUP = "llm:test"


@pytest.fixture(autouse=True)
def clean_shared_storage():
    ss.finalize_share_data()
    yield
    ss.finalize_share_data()


def _init(limits=None):
    ss.initialize_share_data(1, global_concurrency_limits=limits)


async def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(interval)


async def _wait_drained(wrapped, timeout=5.0, interval=0.01):
    """Poll until the wrapper's accounting fully quiesces to zero.

    Asserts the live_queued exactly-once invariant at its strongest point:
    once every task has drained, the logical reservation counter
    (``queued`` == live_queued), the physical queue, and task_states
    (``in_flight``) must all be back to zero. A leaked reservation
    (up-drift) leaves ``queued`` stuck above zero forever; a leaked
    task_states entry leaves ``in_flight`` stuck. Returns the final stats.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        stats = await wrapped.get_queue_stats()
        if (
            stats["queued"] == 0
            and stats["in_flight"] == 0
            and stats.get("physical_queued", 0) == 0
        ):
            return stats
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"queue did not drain to zero: {stats}")
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Core: global cap on in-flight executions
# ---------------------------------------------------------------------------


async def test_global_limit_caps_inflight_executions():
    _init({GROUP: 3})
    inflight = 0
    peak = 0

    async def slow_func(value):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        10, queue_name="cap test", concurrency_group=GROUP
    )(slow_func)
    try:
        results = await asyncio.gather(*(wrapped(i) for i in range(10)))
        assert sorted(results) == list(range(10))
        assert peak <= 3
        # Slots all returned after completion.
        assert await ss.global_concurrency_in_use(GROUP) == 0
    finally:
        await wrapped.shutdown()


async def test_limit_shared_across_wrappers_of_same_group():
    """Two wrappers (≈ two processes / instances) share one global gate."""
    _init({GROUP: 1})
    inflight = 0
    peak = 0

    async def slow_func(value):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        return value

    wrapped_a = lr_utils.priority_limit_async_func_call(
        4, queue_name="share a", concurrency_group=GROUP
    )(slow_func)
    wrapped_b = lr_utils.priority_limit_async_func_call(
        4, queue_name="share b", concurrency_group=GROUP
    )(slow_func)
    try:
        results = await asyncio.gather(
            *(wrapped_a(i) for i in range(3)), *(wrapped_b(i) for i in range(3))
        )
        assert len(results) == 6
        assert peak == 1
    finally:
        await wrapped_a.shutdown()
        await wrapped_b.shutdown()


# ---------------------------------------------------------------------------
# Unconfigured / standalone paths stay untouched
# ---------------------------------------------------------------------------


async def test_no_global_limit_keeps_original_path(monkeypatch):
    _init()  # initialized, but no limits configured

    calls = []
    original_plain = ss.try_acquire_global_slot
    original_tracked = ss.try_acquire_global_slot_tracked

    async def spy_plain(group):
        calls.append(group)
        return await original_plain(group)

    async def spy_tracked(group):
        calls.append(group)
        return await original_tracked(group)

    monkeypatch.setattr(ss, "try_acquire_global_slot", spy_plain)
    monkeypatch.setattr(ss, "try_acquire_global_slot_tracked", spy_tracked)

    async def fast_func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="plain test", concurrency_group=GROUP
    )(fast_func)
    try:
        assert await asyncio.gather(*(wrapped(i) for i in range(5))) == list(range(5))
        assert calls == []  # slot acquisition never touched
        stats = await wrapped.get_queue_stats()
        assert "physical_queued" not in stats  # bounded-queue (default) mode
    finally:
        await wrapped.shutdown()


async def test_standalone_group_none_never_touches_shared_storage(monkeypatch):
    # Shared storage NOT initialized at all (library-external usage).
    def boom(*_args, **_kwargs):
        raise AssertionError("shared_storage API must not be called")

    monkeypatch.setattr(ss, "try_acquire_global_slot", boom)
    monkeypatch.setattr(ss, "try_acquire_global_slot_tracked", boom)
    monkeypatch.setattr(ss, "publish_queue_stats", boom)
    monkeypatch.setattr(ss, "aggregate_queue_stats", boom)

    async def fast_func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(2, queue_name="standalone test")(
        fast_func
    )
    try:
        assert await wrapped("ok") == "ok"
        stats = await wrapped.get_aggregated_queue_stats()
        assert stats["completed_total"] == 1
    finally:
        await wrapped.shutdown()


# ---------------------------------------------------------------------------
# Slot waiting: cancellation, stuck detection, priority
# ---------------------------------------------------------------------------


async def test_user_timeout_while_waiting_for_slot_never_calls_func():
    _init({GROUP: 1})
    external = await ss.try_acquire_global_slot(GROUP)  # saturate the gate
    calls = []

    async def func(value):
        calls.append(value)
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="slot wait test",
        concurrency_group=GROUP,
        cleanup_timeout=0.1,
    )(func)
    try:
        with pytest.raises(TimeoutError, match="User timeout"):
            await wrapped("never", _timeout=0.3)
        assert calls == []
        # The lease count is unchanged — the cancelled task never held one.
        assert await ss.global_concurrency_in_use(GROUP) == 1

        stats = await wrapped.get_queue_stats()
        assert stats["running"] == 0  # never misjudged as running/stuck
        assert stats["cancelled_total"] == 1

        # Once the external holder releases, new requests get the slot.
        await ss.release_global_slot(GROUP, external)
        assert await wrapped("now") == "now"
        assert calls == ["now"]
    finally:
        await wrapped.shutdown()


async def test_priority_overtakes_within_process():
    _init({GROUP: 1})
    external = await ss.try_acquire_global_slot(GROUP)
    order = []

    async def func(value):
        order.append(value)
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="priority test", concurrency_group=GROUP
    )(func)
    try:
        low = [asyncio.create_task(wrapped(f"low{i}", _priority=10)) for i in range(3)]
        await asyncio.sleep(0.05)
        high = asyncio.create_task(wrapped("high", _priority=1))
        await asyncio.sleep(0.05)

        await ss.release_global_slot(GROUP, external)
        await asyncio.gather(high, *low)
        assert order[0] == "high"
    finally:
        await wrapped.shutdown()


# ---------------------------------------------------------------------------
# Slot pump: one acquirer per process, no speculative slot grabs
# ---------------------------------------------------------------------------


async def test_pump_acquires_only_for_live_work(monkeypatch):
    """Idle workers must not grab slots speculatively: with one task and
    many local workers, exactly one slot acquisition succeeds (the herd
    used to acquire-and-release, inflating global_in_use and resetting the
    process's waiter seniority)."""
    _init({GROUP: 4})
    successes = 0
    original = ss.try_acquire_global_slot_tracked

    async def counting(group):
        nonlocal successes
        lease, is_priority = await original(group)
        if lease is not None:
            successes += 1
        return lease, is_priority

    monkeypatch.setattr(ss, "try_acquire_global_slot_tracked", counting)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        4, queue_name="pump only-live test", concurrency_group=GROUP
    )(func)
    try:
        assert await wrapped("only") == "only"
        await asyncio.sleep(0.3)  # idle period: nothing else may be acquired
        assert successes == 1
        assert await ss.global_concurrency_in_use(GROUP) == 0
    finally:
        await wrapped.shutdown()


async def test_pump_never_holds_slots_without_free_worker():
    """A slot is requested only when an idle local worker can run the task
    immediately — scarce global capacity is never parked behind a busy
    process."""
    _init({GROUP: 4})
    release = asyncio.Event()
    started = asyncio.Event()

    async def gated(value):
        started.set()
        await release.wait()
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        1, queue_name="pump no-hoard test", concurrency_group=GROUP
    )(gated)
    try:
        first = asyncio.create_task(wrapped("a"))
        await started.wait()
        others = [asyncio.create_task(wrapped(f"b{i}")) for i in range(3)]
        await asyncio.sleep(0.3)
        # Only the running task holds a slot; the queued tasks wait for the
        # single local worker, not for global slots.
        assert await ss.global_concurrency_in_use(GROUP) == 1
        release.set()
        assert await first == "a"
        assert sorted(await asyncio.gather(*others)) == ["b0", "b1", "b2"]
    finally:
        await wrapped.shutdown()


# ---------------------------------------------------------------------------
# Admission semantics (live_queued)
# ---------------------------------------------------------------------------


async def test_admission_counts_only_live_queued_tasks():
    """max_queue_size limits QUEUED tasks; running tasks don't consume it."""
    _init({GROUP: 1})
    release = asyncio.Event()
    started = asyncio.Event()

    async def gated(value):
        started.set()
        await release.wait()
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="admission test",
        concurrency_group=GROUP,
        max_queue_size=2,
    )(gated)
    try:
        running = asyncio.create_task(wrapped("running"))
        await started.wait()  # now running (holds the only slot), not queued

        queued = [asyncio.create_task(wrapped(f"q{i}")) for i in range(2)]

        async def _queued_count():
            return (await wrapped.get_queue_stats())["queued"]

        deadline = asyncio.get_running_loop().time() + 5.0
        while await _queued_count() != 2:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

        stats = await wrapped.get_queue_stats()
        assert stats["queued"] == 2  # both fit: running not counted

        # The third queued request exceeds capacity -> QueueFullError.
        with pytest.raises(lr_utils.QueueFullError):
            await wrapped("overflow", _queue_timeout=0.2)

        release.set()
        assert await running == "running"
        assert sorted(await asyncio.gather(*queued)) == ["q0", "q1"]
    finally:
        await wrapped.shutdown()


async def test_zombies_do_not_consume_admission_capacity():
    _init({GROUP: 1})
    external = await ss.try_acquire_global_slot(GROUP)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="zombie admission test",
        concurrency_group=GROUP,
        max_queue_size=2,
        cleanup_timeout=0.05,
    )(func)
    try:
        # Fill the queue, then let both requests time out -> zombies.
        for i in range(2):
            with pytest.raises(TimeoutError):
                await wrapped(f"dead{i}", _timeout=0.1)

        stats = await wrapped.get_queue_stats()
        assert stats["queued"] == 0  # logical count excludes zombies
        assert stats["physical_queued"] == 2  # tuples still in the queue

        # New requests are admitted despite the zombie tuples...
        new_tasks = [
            asyncio.create_task(wrapped(f"new{i}", _queue_timeout=1.0))
            for i in range(2)
        ]
        await asyncio.sleep(0.05)
        assert all(not t.done() for t in new_tasks)  # admitted, waiting for slot

        # ...and run fine once a slot frees up; zombies are drained without
        # ever calling the provider for them.
        await ss.release_global_slot(GROUP, external)
        assert sorted(await asyncio.gather(*new_tasks)) == ["new0", "new1"]
    finally:
        await wrapped.shutdown(graceful=False)


async def test_max_queue_size_zero_means_unlimited_admission():
    _init({GROUP: 2})

    async def func(value):
        await asyncio.sleep(0.01)
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="unlimited admission test",
        concurrency_group=GROUP,
        max_queue_size=0,
    )(func)
    try:
        results = await asyncio.gather(*(wrapped(i) for i in range(20)))
        assert len(results) == 20
    finally:
        await wrapped.shutdown()


async def test_shutdown_wakes_admission_waiters():
    _init({GROUP: 1})
    await ss.try_acquire_global_slot(GROUP)  # saturate so nothing dequeues

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="shutdown wake test",
        concurrency_group=GROUP,
        max_queue_size=1,
    )(func)
    try:
        filler = asyncio.create_task(wrapped("filler"))
        await asyncio.sleep(0.05)
        # No _queue_timeout: sleeps on the admission condition.
        waiter = asyncio.create_task(wrapped("waiter"))
        await asyncio.sleep(0.05)
        assert not waiter.done()

        shutdown_task = asyncio.create_task(
            wrapped.shutdown(graceful=False, timeout=0.1)
        )
        with pytest.raises(RuntimeError, match="Queue is shutting down"):
            await asyncio.wait_for(waiter, timeout=2.0)
        await shutdown_task
        with pytest.raises(asyncio.CancelledError):
            await filler
    finally:
        pass


# ---------------------------------------------------------------------------
# Failure policy: fail-closed acquire, safe release, best-effort stats
# ---------------------------------------------------------------------------


async def test_fail_closed_acquire_keeps_task_queued_until_recovery(monkeypatch):
    _init({GROUP: 2})

    fail = True
    original = ss.try_acquire_global_slot_tracked

    async def flaky(group):
        if fail:
            raise RuntimeError("manager down")
        return await original(group)

    monkeypatch.setattr(ss, "try_acquire_global_slot_tracked", flaky)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="fail closed test", concurrency_group=GROUP
    )(func)
    try:
        task = asyncio.create_task(wrapped("delayed"))
        await asyncio.sleep(0.3)
        assert not task.done()  # never let through while acquire fails

        fail = False
        assert await asyncio.wait_for(task, timeout=2.0) == "delayed"
    finally:
        await wrapped.shutdown()


async def test_release_failure_preserves_result_and_retries_later(monkeypatch):
    _init({GROUP: 1})

    fail_release = True
    original_release = ss.release_global_slot

    async def flaky_release(group, lease_id):
        if fail_release:
            raise RuntimeError("release boom")
        return await original_release(group, lease_id)

    monkeypatch.setattr(ss, "release_global_slot", flaky_release)

    async def func(value):
        if value == "explode":
            raise ValueError("business error")
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="release fail test", concurrency_group=GROUP
    )(func)
    try:
        # Result passes through even though the lease release failed; the
        # lease is parked for retry (still occupying its slot until then).
        assert await wrapped("ok") == "ok"
        assert await ss.global_concurrency_in_use(GROUP) >= 1
        fail_release = False
        await wrapped.run_maintenance()  # health-check retry frees the slot
        assert await ss.global_concurrency_in_use(GROUP) == 0

        # Same for a failing call: the original business exception is never
        # masked by the release failure.
        fail_release = True
        with pytest.raises(ValueError, match="business error"):
            await wrapped("explode")
        assert await ss.global_concurrency_in_use(GROUP) >= 1
        fail_release = False
        await wrapped.run_maintenance()
        assert await ss.global_concurrency_in_use(GROUP) == 0
    finally:
        await wrapped.shutdown()


async def test_stats_failures_never_break_calls(monkeypatch):
    _init({GROUP: 2})

    def boom(*_args, **_kwargs):
        raise RuntimeError("stats backend down")

    monkeypatch.setattr(ss, "publish_queue_stats", boom)
    monkeypatch.setattr(ss, "aggregate_queue_stats", boom)
    monkeypatch.setattr(ss, "reconcile_global_slots", boom)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="best effort test", concurrency_group=GROUP
    )(func)
    try:
        assert await wrapped("fine") == "fine"
        await wrapped.run_maintenance()  # must not raise
        # Aggregation failure falls back to the local snapshot.
        stats = await wrapped.get_aggregated_queue_stats()
        assert stats["completed_total"] == 1
        assert "reporting_workers" not in stats
    finally:
        await wrapped.shutdown()


# ---------------------------------------------------------------------------
# Zombie drain & physical queue compaction
# ---------------------------------------------------------------------------


async def test_worker_drains_zombie_backlog_then_runs_live_task():
    _init({GROUP: 1})
    external = await ss.try_acquire_global_slot(GROUP)
    calls = []

    async def func(value):
        calls.append(value)
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="drain test",
        concurrency_group=GROUP,
        max_queue_size=0,
        cleanup_timeout=0.05,
    )(func)
    try:
        # Pile up zombies well past the per-slot drain limit (16).
        zombies = [wrapped(f"dead{i}", _priority=1, _timeout=0.05) for i in range(40)]
        results = await asyncio.gather(*zombies, return_exceptions=True)
        assert all(isinstance(r, TimeoutError) for r in results)

        live = asyncio.create_task(wrapped("live", _priority=5))
        await asyncio.sleep(0.05)
        await ss.release_global_slot(GROUP, external)

        # The worker drains zombies in bounded batches (releasing the slot
        # in between) and the live task still completes; no zombie ever
        # reached the provider.
        assert await asyncio.wait_for(live, timeout=5.0) == "live"
        assert calls == ["live"]

        # live_queued invariant: after the 40 zombies drained and the live
        # task completed, the reservation counter and task_states must be
        # fully back to zero (no leaked reservation, no orphaned tuple).
        await _wait_drained(wrapped)
    finally:
        await wrapped.shutdown(graceful=False)


async def test_compaction_bounds_physical_queue_and_keeps_join_working(monkeypatch):
    _init({GROUP: 1})

    async def acquire_always_fails(_group):
        raise RuntimeError("manager down")  # permanent fail-closed

    monkeypatch.setattr(ss, "try_acquire_global_slot_tracked", acquire_always_fails)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="compaction test",
        concurrency_group=GROUP,
        max_queue_size=0,  # threshold = 64
        cleanup_timeout=0.05,
    )(func)
    try:
        zombies = [wrapped(f"dead{i}", _timeout=0.05) for i in range(80)]
        results = await asyncio.gather(*zombies, return_exceptions=True)
        assert all(isinstance(r, TimeoutError) for r in results)

        stats = await wrapped.get_queue_stats()
        assert stats["physical_queued"] == 80
        assert stats["queued"] == 0
        # All 80 timed-out tasks already returned their reservations and
        # popped task_states in their finally blocks; only the physical
        # tuples linger.
        assert stats["in_flight"] == 0

        await wrapped.run_maintenance()  # triggers compaction
        stats = await wrapped.get_queue_stats()
        assert stats["physical_queued"] == 0
        # Compaction must not perturb the logical accounting: live_queued
        # and task_states stay at zero (it only drains dead physical tuples).
        assert stats["queued"] == 0
        assert stats["in_flight"] == 0

        # task_done bookkeeping must be exact: graceful shutdown relies on
        # queue.join() and must complete promptly (not hit the drain timeout).
        await asyncio.wait_for(wrapped.shutdown(graceful=True, timeout=5.0), 10.0)
    finally:
        pass


async def test_compaction_keeps_live_tasks(monkeypatch):
    _init({GROUP: 1})
    external = await ss.try_acquire_global_slot(GROUP)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2,
        queue_name="compaction live test",
        concurrency_group=GROUP,
        max_queue_size=0,
        cleanup_timeout=0.05,
    )(func)
    try:
        zombies = [wrapped(f"dead{i}", _timeout=0.05) for i in range(70)]
        await asyncio.gather(*zombies, return_exceptions=True)
        live = asyncio.create_task(wrapped("live"))
        await asyncio.sleep(0.05)

        await wrapped.run_maintenance()
        stats = await wrapped.get_queue_stats()
        assert stats["queued"] == 1  # live survived compaction
        assert stats["physical_queued"] == 1
        # live_queued invariant at a held steady state: the external slot
        # blocks pickup, so the one live task is queued-not-started
        # (running == 0) and is the only task_states entry (in_flight == 1).
        # live_queued must therefore equal in_flight - running, i.e. exactly
        # the count of not-yet-started tasks, with no phantom reservation.
        assert stats["running"] == 0
        assert stats["in_flight"] == 1
        assert stats["queued"] == stats["in_flight"] - stats["running"]

        await ss.release_global_slot(GROUP, external)
        assert await asyncio.wait_for(live, timeout=5.0) == "live"

        # And once it runs and drains, accounting returns fully to zero.
        await _wait_drained(wrapped)
    finally:
        await wrapped.shutdown(graceful=False)


# ---------------------------------------------------------------------------
# Stats publishing & aggregation through the wrapper
# ---------------------------------------------------------------------------


async def test_aggregated_stats_schema_and_global_fields():
    _init({GROUP: 2})

    async def func(value):
        if value == "fail":
            raise ValueError("x")
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        3, queue_name="agg schema test", concurrency_group=GROUP
    )(func)
    try:
        await wrapped("a")
        await wrapped("b")
        with pytest.raises(ValueError):
            await wrapped("fail")

        # The worker releases its lease in a finally block right after the
        # future resolves; wait for the release before checking in_use.
        async def _in_use_zero():
            return await ss.global_concurrency_in_use(GROUP) == 0

        deadline = asyncio.get_running_loop().time() + 2.0
        while not await _in_use_zero():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

        stats = await wrapped.get_aggregated_queue_stats()
        # Original flat schema intact (webui/health compatibility).
        for field in (
            "queue_name",
            "max_async",
            "max_queue_size",
            "queued",
            "running",
            "in_flight",
            "worker_count",
            "initialized",
            "submitted_total",
            "completed_total",
            "failed_total",
            "cancelled_total",
            "rejected_total",
        ):
            assert field in stats
        assert stats["completed_total"] == 2
        assert stats["failed_total"] == 1
        # Aggregation extras.
        assert stats["reporting_workers"] == 1
        assert list(stats["per_worker"])
        assert stats["global_limit"] == 2
        assert stats["global_in_use"] == 0
    finally:
        await wrapped.shutdown()


async def test_shutdown_clears_waiter_record():
    _init({GROUP: 1})
    await ss.try_acquire_global_slot(GROUP)  # saturate so workers poll

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="waiter cleanup test", concurrency_group=GROUP
    )(func)
    pending = asyncio.create_task(wrapped("stuck"))
    try:
        # Wait until a polling worker registers this process as a waiter.
        deadline = asyncio.get_running_loop().time() + 5.0
        while not await ss.global_slot_waiters(GROUP):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
    finally:
        await wrapped.shutdown(graceful=False, timeout=0.1)
    with pytest.raises(asyncio.CancelledError):
        await pending
    # No-longer-polling process must not linger in the longest-waiter seat.
    assert await ss.global_slot_waiters(GROUP) == []


async def test_aggregated_stats_report_slot_waiters():
    _init({GROUP: 1})
    external = await ss.try_acquire_global_slot(GROUP)

    async def func(value):
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="waiter stats test", concurrency_group=GROUP
    )(func)
    task = asyncio.create_task(wrapped("waiting"))
    try:
        deadline = asyncio.get_running_loop().time() + 5.0
        while not await ss.global_slot_waiters(GROUP):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

        stats = await wrapped.get_aggregated_queue_stats()
        assert stats["global_waiting_workers"] == 1
        assert stats["global_longest_wait"] >= 0.0

        await ss.release_global_slot(GROUP, external)
        assert await asyncio.wait_for(task, timeout=5.0) == "waiting"
    finally:
        await wrapped.shutdown()


async def test_maintenance_renews_held_leases(monkeypatch):
    monkeypatch.setattr(ss, "_heartbeat_ttl", 0.2)
    monkeypatch.setattr(ss, "_suspect_grace", 0.2)
    _init({GROUP: 1})
    release = asyncio.Event()
    started = asyncio.Event()

    async def long_func(value):
        started.set()
        await release.wait()
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="renew test", concurrency_group=GROUP
    )(long_func)
    try:
        task = asyncio.create_task(wrapped("long"))
        await started.wait()

        # Run well past the heartbeat TTL, renewing like the health check.
        for _ in range(5):
            await asyncio.sleep(0.1)
            await wrapped.run_maintenance()

        # The legal long task's lease was never reclaimed.
        assert await ss.global_concurrency_in_use(GROUP) == 1
        release.set()
        assert await task == "long"
    finally:
        await wrapped.shutdown()


# ---------------------------------------------------------------------------
# contextvars propagation (#3382): persistent worker tasks must run each call
# under the enqueuer's context, not the context frozen when the worker was
# first created.
# ---------------------------------------------------------------------------

_ctx_var: contextvars.ContextVar = contextvars.ContextVar("ctx_var", default="unset")


async def test_context_propagates_to_default_path_worker():
    seen = []

    async def record_ctx(value):
        seen.append(_ctx_var.get())
        return value

    wrapped = lr_utils.priority_limit_async_func_call(2, queue_name="ctx default test")(
        record_ctx
    )
    try:
        # Force the persistent worker task to spin up while the contextvar
        # is still at its default — this is the moment the bug snapshotted
        # forever under the old (unpatched) worker.
        assert await wrapped("prime") == "prime"
        assert seen[-1] == "unset"

        _ctx_var.set("request-1")
        assert await wrapped("call") == "call"
        assert seen[-1] == "request-1"
    finally:
        await wrapped.shutdown()


async def test_context_propagates_to_global_limit_path_worker():
    _init({GROUP: 2})
    seen = []

    async def record_ctx(value):
        seen.append(_ctx_var.get())
        return value

    wrapped = lr_utils.priority_limit_async_func_call(
        2, queue_name="ctx global test", concurrency_group=GROUP
    )(record_ctx)
    try:
        # Same priming step, but through the slot-pump / limited_worker path.
        assert await wrapped("prime") == "prime"
        assert seen[-1] == "unset"

        _ctx_var.set("request-2")
        assert await wrapped("call") == "call"
        assert seen[-1] == "request-2"
    finally:
        await wrapped.shutdown()
