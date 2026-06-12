"""Offline tests for the cross-worker global concurrency gate and queue
stats aggregation primitives in lightrag.kg.shared_storage.

All tests run in single-process mode (workers=1): the lease namespace and
keyed locks degrade to in-process primitives, exercising the same code paths
used under gunicorn multi-worker mode.
"""

import os
import subprocess
import sys
import time

import pytest

from lightrag.kg import shared_storage as ss

pytestmark = pytest.mark.offline


GROUP = "llm:test"


@pytest.fixture(autouse=True)
def clean_shared_storage():
    """Each test starts from a non-initialized shared storage."""
    ss.finalize_share_data()
    yield
    ss.finalize_share_data()


def _init(limits=None):
    ss.initialize_share_data(1, global_concurrency_limits=limits)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


async def _lease_ns():
    return await ss._get_lease_namespace()


# ---------------------------------------------------------------------------
# Configuration semantics
# ---------------------------------------------------------------------------


def test_limits_unset_means_not_limited():
    _init()
    assert ss.is_global_concurrency_limited(GROUP) is False
    assert ss.get_global_concurrency_limit(GROUP) is None
    assert ss.is_global_concurrency_limited(None) is False


def test_limits_not_limited_before_initialization():
    assert ss.is_share_data_initialized() is False
    assert ss.is_global_concurrency_limited(GROUP) is False


def test_first_init_sets_limits_and_later_calls_do_not_overwrite():
    _init({GROUP: 3})
    assert ss.is_global_concurrency_limited(GROUP) is True
    assert ss.get_global_concurrency_limit(GROUP) == 3

    # Subsequent no-arg call (e.g. LightRAG.__post_init__) hits the
    # already-initialized guard and must not clear the configuration.
    ss.initialize_share_data()
    assert ss.is_global_concurrency_limited(GROUP) is True
    assert ss.get_global_concurrency_limit(GROUP) == 3


def test_finalize_resets_limits():
    _init({GROUP: 3})
    ss.finalize_share_data()
    assert ss.is_global_concurrency_limited(GROUP) is False


def test_non_positive_limits_are_ignored():
    _init({GROUP: 0, "other": -1})
    assert ss.is_global_concurrency_limited(GROUP) is False
    assert ss.is_global_concurrency_limited("other") is False


# ---------------------------------------------------------------------------
# Slot acquisition / release
# ---------------------------------------------------------------------------


async def test_acquire_up_to_limit_then_release():
    _init({GROUP: 2})

    lease1 = await ss.try_acquire_global_slot(GROUP)
    lease2 = await ss.try_acquire_global_slot(GROUP)
    assert lease1 is not None and lease2 is not None
    assert lease1 != lease2
    assert await ss.global_concurrency_in_use(GROUP) == 2

    # Capacity exhausted
    assert await ss.try_acquire_global_slot(GROUP) is None

    await ss.release_global_slot(GROUP, lease1)
    assert await ss.global_concurrency_in_use(GROUP) == 1
    lease3 = await ss.try_acquire_global_slot(GROUP)
    assert lease3 is not None

    # Idempotent release
    await ss.release_global_slot(GROUP, lease1)
    await ss.release_global_slot(GROUP, lease2)
    await ss.release_global_slot(GROUP, lease3)
    assert await ss.global_concurrency_in_use(GROUP) == 0


async def test_acquire_for_unlimited_group_returns_none():
    _init({GROUP: 1})
    assert await ss.try_acquire_global_slot("unconfigured") is None


async def test_groups_are_independent():
    _init({GROUP: 1, "embedding": 1})
    lease = await ss.try_acquire_global_slot(GROUP)
    assert lease is not None
    assert await ss.try_acquire_global_slot(GROUP) is None
    other = await ss.try_acquire_global_slot("embedding")
    assert other is not None


# ---------------------------------------------------------------------------
# Self-healing: dead PID, heartbeat expiry, suspect grace
# ---------------------------------------------------------------------------


async def test_dead_pid_lease_reclaimed_immediately():
    _init({GROUP: 1})
    ns = await _lease_ns()
    ns[f"{GROUP}{ss.KEY_SEP}deadbeef"] = {"pid": _dead_pid(), "updated_at": time.time()}

    # The dead owner's slot is reclaimed during acquire — capacity is free.
    lease = await ss.try_acquire_global_slot(GROUP)
    assert lease is not None
    assert f"{GROUP}{ss.KEY_SEP}deadbeef" not in list(ns.keys())


async def test_expired_lease_with_live_pid_gets_suspect_grace(monkeypatch):
    monkeypatch.setattr(ss, "_heartbeat_ttl", 1.0)
    monkeypatch.setattr(ss, "_suspect_grace", 1.0)
    _init({GROUP: 1})
    ns = await _lease_ns()
    key = f"{GROUP}{ss.KEY_SEP}stalled"
    # Live PID (our own) with an expired heartbeat.
    ns[key] = {"pid": os.getpid(), "updated_at": time.time() - 5.0}

    # First pass: marked suspect, still counts toward capacity.
    assert await ss.reconcile_global_slots(GROUP) == 1
    assert "suspect_since" in dict(ns[key])
    assert await ss.try_acquire_global_slot(GROUP) is None

    # Within the grace: still not reclaimed.
    assert await ss.reconcile_global_slots(GROUP) == 1

    # After the grace elapses without a renewal: reclaimed.
    lease = dict(ns[key])
    lease["suspect_since"] = time.time() - 2.0
    ns[key] = lease
    assert await ss.reconcile_global_slots(GROUP) == 0
    assert await ss.try_acquire_global_slot(GROUP) is not None


async def test_renewal_clears_suspect_and_protects_long_tasks(monkeypatch):
    monkeypatch.setattr(ss, "_heartbeat_ttl", 1.0)
    monkeypatch.setattr(ss, "_suspect_grace", 1.0)
    _init({GROUP: 1})
    lease = await ss.try_acquire_global_slot(GROUP)
    ns = await _lease_ns()
    key = f"{GROUP}{ss.KEY_SEP}{lease}"

    # Simulate a momentary renewal outage: heartbeat expires, suspect set.
    stale = dict(ns[key])
    stale["updated_at"] = time.time() - 5.0
    ns[key] = stale
    await ss.reconcile_global_slots(GROUP)
    assert "suspect_since" in dict(ns[key])

    # Owner recovers and renews: suspect cleared, lease survives — a legal
    # long-running task is never reclaimed while renewals continue.
    await ss.renew_global_slots(GROUP, [lease])
    refreshed = dict(ns[key])
    assert "suspect_since" not in refreshed
    assert time.time() - refreshed["updated_at"] < 1.0
    assert await ss.reconcile_global_slots(GROUP) == 1


async def test_renew_does_not_resurrect_reclaimed_lease():
    _init({GROUP: 1})
    lease = await ss.try_acquire_global_slot(GROUP)
    await ss.release_global_slot(GROUP, lease)
    await ss.renew_global_slots(GROUP, [lease])
    assert await ss.global_concurrency_in_use(GROUP) == 0


async def test_acquire_fail_closed_on_shared_error(monkeypatch):
    _init({GROUP: 2})

    def boom(*_args, **_kwargs):
        raise RuntimeError("manager down")

    monkeypatch.setattr(ss, "get_storage_keyed_lock", boom)
    assert await ss.try_acquire_global_slot(GROUP) is None


# ---------------------------------------------------------------------------
# Queue stats publish / aggregate
# ---------------------------------------------------------------------------


def _snapshot(pid: int, *, queued=1, running=2, completed_total=3, updated_at=None):
    return {
        "queue_name": "agg test",
        "max_async": 4,
        "pid": pid,
        "updated_at": updated_at if updated_at is not None else time.time(),
        "queued": queued,
        "running": running,
        "in_flight": queued + running,
        "worker_count": 4,
        "submitted_total": 10,
        "completed_total": completed_total,
        "failed_total": 1,
        "cancelled_total": 1,
        "rejected_total": 0,
        "global_slot_waits": 5,
        "physical_queued": queued,
    }


async def test_aggregate_sums_flat_fields_across_workers():
    _init()
    ns = await ss._get_queue_stats_namespace()
    await ss.publish_queue_stats("agg test", _snapshot(os.getpid()))
    # PID 1 (init) exists and os.kill(1, 0) raises PermissionError — treated
    # as alive, standing in for a second worker process.
    ns[f"agg test{ss.KEY_SEP}1"] = _snapshot(1, queued=2, running=1, completed_total=7)

    agg = await ss.aggregate_queue_stats("agg test")
    assert agg["reporting_workers"] == 2
    assert agg["queued"] == 3
    assert agg["running"] == 3
    assert agg["completed_total"] == 10
    assert agg["global_slot_waits"] == 10
    assert set(agg["per_worker"]) == {str(os.getpid()), "1"}
    # Schema: every flat counter field is present.
    for field in ss.QUEUE_STATS_SUM_FIELDS:
        assert field in agg


async def test_aggregate_reaps_dead_pid_and_stale_entries():
    _init()
    ns = await ss._get_queue_stats_namespace()
    await ss.publish_queue_stats("agg test", _snapshot(os.getpid()))
    ns[f"agg test{ss.KEY_SEP}99999999"] = _snapshot(_dead_pid())
    ns[f"agg test{ss.KEY_SEP}1"] = _snapshot(
        1, updated_at=time.time() - ss._queue_stats_stale_ttl - 5
    )

    agg = await ss.aggregate_queue_stats("agg test")
    assert agg["reporting_workers"] == 1
    assert f"agg test{ss.KEY_SEP}99999999" not in list(ns.keys())
    assert f"agg test{ss.KEY_SEP}1" not in list(ns.keys())


async def test_unpublish_removes_own_entry():
    _init()
    await ss.publish_queue_stats("agg test", _snapshot(os.getpid()))
    await ss.unpublish_queue_stats("agg test")
    agg = await ss.aggregate_queue_stats("agg test")
    assert agg["reporting_workers"] == 0


async def test_publish_is_noop_when_uninitialized():
    # Best-effort contract: never raises before initialization.
    await ss.publish_queue_stats("agg test", _snapshot(os.getpid()))
    await ss.unpublish_queue_stats("agg test")
