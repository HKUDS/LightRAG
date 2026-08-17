"""Bounded scheduling metrics (LR2 Phase 6, item 3).

The registry itself has to be boring and unbreakable: a metric must never fail
the operation it measures, and the key set must not grow — a registry whose
cardinality follows traffic is the same unbounded structure the whole plan exists
to remove.

The instrumented paths are pinned where the measurement is load-bearing: the
strict active count records even when it FAILS (a count that times out is exactly
what an operator needs to see when uploads start answering 503), and the manual
freeze reject counter sits at the single conflict-mapping chokepoint so the number
cannot drift between the ingress paths that share it.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag import pipeline_metrics
from lightrag.base import DocStatus
from lightrag.exceptions import StorageControlPlaneError
from lightrag.kg.shared_storage import _conflict_for_status_flag
from lightrag.utils_pipeline import count_active_documents

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _clean_metrics():
    pipeline_metrics.reset()
    yield
    pipeline_metrics.reset()


def test_counters_and_durations_start_at_zero():
    snapshot = pipeline_metrics.snapshot()

    assert snapshot["counters"][pipeline_metrics.FREEZE_REJECTS] == 0
    summary = snapshot["durations"][pipeline_metrics.ACTIVE_COUNT_SECONDS]
    assert summary == {
        "count": 0,
        "total_seconds": 0.0,
        "max_seconds": 0.0,
        "last_seconds": 0.0,
        "mean_seconds": 0.0,
    }


def test_duration_summary_tracks_count_max_and_mean():
    pipeline_metrics.observe(pipeline_metrics.SCHEDULING_PAGE_SECONDS, 0.1)
    pipeline_metrics.observe(pipeline_metrics.SCHEDULING_PAGE_SECONDS, 0.3)

    summary = pipeline_metrics.snapshot()["durations"][
        pipeline_metrics.SCHEDULING_PAGE_SECONDS
    ]
    assert summary["count"] == 2
    assert summary["max_seconds"] == 0.3
    assert summary["last_seconds"] == 0.3
    assert summary["mean_seconds"] == 0.2


def test_unknown_names_are_dropped_not_inserted():
    """No key may enter the registry at runtime."""
    pipeline_metrics.increment("not_a_declared_counter")
    pipeline_metrics.observe("not_a_declared_duration", 1.0)

    snapshot = pipeline_metrics.snapshot()
    assert "not_a_declared_counter" not in snapshot["counters"]
    assert "not_a_declared_duration" not in snapshot["durations"]


def test_negative_and_zero_samples_cannot_corrupt_a_summary():
    pipeline_metrics.increment(pipeline_metrics.FREEZE_REJECTS, 0)
    pipeline_metrics.increment(pipeline_metrics.FREEZE_REJECTS, -5)
    pipeline_metrics.observe(pipeline_metrics.MANUAL_DRAIN_SECONDS, -2.0)

    snapshot = pipeline_metrics.snapshot()
    assert snapshot["counters"][pipeline_metrics.FREEZE_REJECTS] == 0
    drain = snapshot["durations"][pipeline_metrics.MANUAL_DRAIN_SECONDS]
    assert drain["count"] == 1 and drain["max_seconds"] == 0.0


def test_freeze_rejection_is_counted_at_the_shared_chokepoint():
    before = pipeline_metrics.snapshot()["counters"][pipeline_metrics.FREEZE_REJECTS]

    _conflict_for_status_flag("manual_freeze_requested")
    _conflict_for_status_flag("destructive_busy")  # a different fence: not counted

    after = pipeline_metrics.snapshot()["counters"][pipeline_metrics.FREEZE_REJECTS]
    assert after == before + 1


def test_active_count_latency_is_recorded_even_when_the_count_fails():
    class _CountingDocStatus:
        def __init__(self, error=None):
            self.error = error

        async def count_docs_by_statuses(self, statuses, *, strict=True):
            assert set(statuses) == {
                DocStatus.PENDING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
                DocStatus.PROCESSING,
            }
            if self.error:
                raise self.error
            return 3

    async def _run():
        assert await count_active_documents(_CountingDocStatus()) == 3
        with pytest.raises(StorageControlPlaneError):
            await count_active_documents(
                _CountingDocStatus(StorageControlPlaneError("index down"))
            )

    asyncio.run(_run())

    summary = pipeline_metrics.snapshot()["durations"][
        pipeline_metrics.ACTIVE_COUNT_SECONDS
    ]
    assert summary["count"] == 2  # the failure is measured too
