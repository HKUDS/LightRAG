"""Bounded in-process metrics for the scheduling paths (LR2 Phase 6, item 3).

The bounded-memory rework moved several decisions off the happy path — a paged
sweep, a strict active count per admission, a drain-then-reset freeze — and each
one can be slow or noisy without anything visible going wrong. These counters and
duration summaries are the only way to tell "the sweep is paging" from "the sweep
is thrashing", or "a freeze is normal" from "a freeze is stuck".

Design constraints that shaped this module:

* **Fixed key set.** Every metric name is declared below. An unknown name is
  logged once and dropped rather than inserted — a metrics registry that grows
  with cardinality is exactly the unbounded structure this whole plan exists to
  remove.
* **O(1) per metric.** Durations keep count / total / max / last, not samples.
  There are no percentiles: a ring of samples would be a per-metric bound to
  pick, defend and test, and the questions above are answered by max plus mean.
* **In-process, per worker.** Incrementing a shared (Manager-hosted) counter
  would add an RPC to paths taken thousands of times per sweep, which is the cost
  the scheduling rework was built to avoid. Under gunicorn with N workers
  ``/health`` therefore reports the numbers of the worker that answered — the
  same documented boundary as the login rate limiter and the keyed-lock table.
  The pipeline supervisor is single-owner (whoever holds ``busy``), so its
  measurements are not spread across workers in the first place.
* **Never fatal.** A metrics failure must not fail the operation it measures;
  every entry point swallows its own errors.

Deliberately NOT duplicated here: scan classification counts, source-conflict
sightings and abandoned scan jobs. Those already have a bounded, per-job surface
in the scan job record (``GET /documents/scan/status/{track_id}``) — counting
them twice would invite the two numbers to disagree.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from lightrag.utils import logger

# --- counters -------------------------------------------------------------
# Ingress refused because a manual retry froze new work (LR2 §6.1/§7.2). A
# handful during a reset is normal; a steadily climbing number means clients are
# retrying into a freeze that is not ending.
FREEZE_REJECTS = "freeze_rejects"
# Auto-rescan dirty flag re-armed: a processing request that could not name doc
# ids (busy-refused, partial commit, recovery). Each one costs a later strict
# sweep, so a high rate explains sweep load that no upload seems to account for.
AUTO_RESCAN_REARMS = "auto_rescan_rearms"
# Documents moved FAILED→PENDING by an exclusive reset, and the pages it took.
MANUAL_RESET_DOCS = "manual_reset_docs"
MANUAL_RESET_PAGES = "manual_reset_pages"

_COUNTERS: tuple[str, ...] = (
    FREEZE_REJECTS,
    AUTO_RESCAN_REARMS,
    MANUAL_RESET_DOCS,
    MANUAL_RESET_PAGES,
)

# --- duration summaries ---------------------------------------------------
# One keyset page of the doc_status sweep (query + projection).
SCHEDULING_PAGE_SECONDS = "scheduling_page_seconds"
# One strict active-document count, taken per admission decision.
ACTIVE_COUNT_SECONDS = "active_count_seconds"
# DRAIN_TO_IDLE: from freezing ingress to confirmed quiescence. This is the
# window during which uploads are refused, so it is the one an operator feels.
MANUAL_DRAIN_SECONDS = "manual_drain_seconds"
# EXCLUSIVE_RESET: the paged FAILED→PENDING rewrite, with no worker running.
MANUAL_RESET_SECONDS = "manual_reset_seconds"

_DURATIONS: tuple[str, ...] = (
    SCHEDULING_PAGE_SECONDS,
    ACTIVE_COUNT_SECONDS,
    MANUAL_DRAIN_SECONDS,
    MANUAL_RESET_SECONDS,
)

_lock = threading.Lock()
_counters: Dict[str, int] = {name: 0 for name in _COUNTERS}
_durations: Dict[str, Dict[str, float]] = {
    name: {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0, "last_seconds": 0.0}
    for name in _DURATIONS
}


def increment(name: str, amount: int = 1) -> None:
    """Add to a declared counter; unknown names are dropped, never inserted."""
    try:
        if name not in _counters:
            logger.warning(f"Ignoring unknown pipeline metric counter {name!r}")
            return
        if amount <= 0:
            return
        with _lock:
            _counters[name] += amount
    except Exception as metric_error:  # pragma: no cover - defensive
        logger.debug(f"pipeline metric {name!r} not recorded: {metric_error}")


def observe(name: str, seconds: float) -> None:
    """Record one duration sample into a declared summary."""
    try:
        if name not in _durations:
            logger.warning(f"Ignoring unknown pipeline metric duration {name!r}")
            return
        value = max(0.0, float(seconds))
        with _lock:
            summary = _durations[name]
            summary["count"] += 1
            summary["total_seconds"] += value
            summary["last_seconds"] = value
            if value > summary["max_seconds"]:
                summary["max_seconds"] = value
    except Exception as metric_error:  # pragma: no cover - defensive
        logger.debug(f"pipeline metric {name!r} not recorded: {metric_error}")


def snapshot() -> Dict[str, Any]:
    """Current values, rounded for display. Bounded by the declared key set."""
    with _lock:
        counters = dict(_counters)
        durations = {
            name: {
                "count": int(summary["count"]),
                "total_seconds": round(summary["total_seconds"], 4),
                "max_seconds": round(summary["max_seconds"], 4),
                "last_seconds": round(summary["last_seconds"], 4),
                "mean_seconds": (
                    round(summary["total_seconds"] / summary["count"], 4)
                    if summary["count"]
                    else 0.0
                ),
            }
            for name, summary in _durations.items()
        }
    return {"counters": counters, "durations": durations}


def reset() -> None:
    """Zero everything (tests only; there is no runtime reset endpoint)."""
    with _lock:
        for name in _counters:
            _counters[name] = 0
        for summary in _durations.values():
            summary.update(
                {
                    "count": 0,
                    "total_seconds": 0.0,
                    "max_seconds": 0.0,
                    "last_seconds": 0.0,
                }
            )
