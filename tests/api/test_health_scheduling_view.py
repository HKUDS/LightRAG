"""The curated scheduling / capability view on /health (LR2 Phase 6, items 2 & 4).

The raw ``manual_*`` fields stay hidden from ``/documents/pipeline_status`` — they
are coordination internals — but an operator watching a manual retry needs to see
which request holds the freeze, for how long, and what the drain is still waiting
for. And when a doc_status backend lacks a strict capability the whole scheduling
contract fails closed (admission 503s, stale stubs are kept, conflicts cannot be
listed), which from the outside looks like a database problem unless health says
otherwise.

Two invariants get their own tests: the owner TOKEN is never published (it
authorizes releasing a reservation — a status page must not become a control
surface), and a bootstrapped-but-idle pipeline reports plain zeros rather than
nulls that a dashboard would have to special-case.
"""

from __future__ import annotations

import importlib
import sys
import time

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_server = importlib.import_module("lightrag.api.lightrag_server")
_utils_pipeline = importlib.import_module("lightrag.utils_pipeline")
sys.argv = _original_argv

from lightrag.base import DocStatusStorage  # noqa: E402
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage  # noqa: E402

pytestmark = pytest.mark.offline


def _idle_snapshot() -> dict:
    return {
        "busy": False,
        "pending_enqueues": 0,
        "manual_phase": "idle",
        "manual_freeze_requested": False,
        "manual_resetting": False,
        "manual_freeze_started_at": None,
        "manual_owner": None,
    }


def test_idle_pipeline_reports_zeros_not_nulls():
    view = _server._build_scheduling_status(
        _idle_snapshot(), {"manual_retries": 0, "manual_retries_capacity": 64}
    )

    assert view["manual_phase"] == "idle"
    assert view["manual_freeze_requested"] is False
    assert view["manual_freeze_seconds"] is None  # nothing is frozen
    assert view["drain_pending_enqueues"] == 0
    assert view["drain_waiting_on_workers"] is False
    assert view["manual_retries_queued"] == 0
    assert view["manual_retries_capacity"] == 64


def test_freeze_is_reported_with_its_holder_and_age():
    snapshot = _idle_snapshot()
    snapshot.update(
        {
            "manual_phase": "exclusive_reset",
            "manual_freeze_requested": True,
            "manual_resetting": True,
            "manual_freeze_started_at": time.time() - 9,
            "manual_owner": {"request_id": "req-42", "pid": 4242},
            "pending_enqueues": 2,
            "busy": True,
        }
    )

    view = _server._build_scheduling_status(snapshot, {})

    assert view["manual_phase"] == "exclusive_reset"
    assert view["manual_resetting"] is True
    assert 8.0 <= view["manual_freeze_seconds"] <= 15.0
    assert view["manual_owner_request_id"] == "req-42"
    assert view["manual_owner_pid"] == 4242
    # What the drain is still waiting for.
    assert view["drain_pending_enqueues"] == 2
    assert view["drain_waiting_on_workers"] is True


def test_owner_token_is_never_published():
    """The token authorizes releasing another process's reservation."""
    snapshot = _idle_snapshot()
    snapshot["manual_owner"] = {
        "request_id": "req-1",
        "pid": 7,
        "owner_token": "capability-do-not-leak",
        "process_start_id": "abc",
    }

    view = _server._build_scheduling_status(snapshot, {})

    assert "capability-do-not-leak" not in str(view)
    assert not any("token" in key for key in view)


def test_clock_stepping_backwards_never_reports_a_negative_age():
    snapshot = _idle_snapshot()
    snapshot["manual_freeze_started_at"] = time.time() + 60  # future timestamp

    view = _server._build_scheduling_status(snapshot, {})

    assert view["manual_freeze_seconds"] == 0.0


def test_a_first_party_backend_reports_every_capability():
    capabilities = _utils_pipeline.describe_doc_status_capabilities(
        JsonDocStatusStorage.__new__(JsonDocStatusStorage)
    )

    assert all(capabilities.values()), capabilities


def test_a_backend_on_the_fail_closed_defaults_reports_the_gaps():
    """A third-party doc_status that only implements the abstract methods keeps
    the base's fail-closed defaults; health has to say so, otherwise an operator
    sees only 503s with no explanation."""

    class _MinimalDocStatus(DocStatusStorage):
        supports_strict_point_reads = False

        async def get_docs_by_statuses_page(self, *args, **kwargs): ...
        async def get_docs_by_ids(self, *args, **kwargs): ...
        async def resolve_doc_source_strict(self, *args, **kwargs): ...
        async def get_full_docs_by_ids(self, *args, **kwargs): ...

    # The probe only reads the class, so the remaining (unrelated) abstract
    # methods are irrelevant here — clear the ABC guard instead of stubbing 17
    # data-plane methods that this test does not exercise.
    _MinimalDocStatus.__abstractmethods__ = frozenset()

    capabilities = _utils_pipeline.describe_doc_status_capabilities(
        _MinimalDocStatus.__new__(_MinimalDocStatus)
    )

    assert capabilities["scheduling_pages"] is True
    assert capabilities["typed_source_resolution"] is True
    # The three that a deployment can genuinely lack, plus the point-read opt-in.
    assert capabilities["strict_active_count"] is False
    assert capabilities["source_conflict_listing"] is False
    assert capabilities["source_conflict_repair"] is False
    assert capabilities["strict_point_reads"] is False
