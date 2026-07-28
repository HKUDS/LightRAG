"""Endpoint-side admission control (LR2 Phase 5-a, §9.1/§9.2).

The enqueue guard is the last line; the endpoint reservation is where a client
learns the answer. What matters here is the mapping and the ordering:

* over capacity → 429 with the numbers and a Retry-After hint, taken BEFORE the
  uploaded file is written to disk;
* a strict-count failure → 503, never "there is room";
* a mutual-exclusion fence still wins over capacity (409 even when there is
  room), because a freeze is not a capacity problem;
* re-weighting an already-held reservation (``/texts`` after body parse) is NOT
  subject to those fences — a request admitted before a freeze may finish — but
  it IS subject to capacity at its true size.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_shared_storage = importlib.import_module("lightrag.kg.shared_storage")
sys.argv = _original_argv

from lightrag.exceptions import StorageControlPlaneError  # noqa: E402

pytestmark = pytest.mark.offline


class _CountingDocStatus:
    def __init__(self, active: int):
        self.active = active
        self.error: Exception | None = None
        self.calls = 0

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.active


async def _rag(*, capacity: int, active: int = 0):
    workspace = f"admission-{uuid4().hex[:8]}"
    _shared_storage.initialize_share_data()
    await _shared_storage.initialize_pipeline_status(workspace=workspace)
    return SimpleNamespace(
        workspace=workspace,
        doc_status=_CountingDocStatus(active),
        max_pending_documents=capacity,
    )


async def _status(rag):
    return await _shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )


async def test_reservation_refuses_over_capacity_with_429():
    rag = await _rag(capacity=3, active=3)

    with pytest.raises(HTTPException) as excinfo:
        await _document_routes._reserve_enqueue_slot(rag, "tok-1")

    error = excinfo.value
    assert error.status_code == 429
    # The client is told current / requested / capacity, plus when to retry.
    assert "3 document(s) already active" in error.detail
    assert "capacity 3" in error.detail
    assert error.headers["Retry-After"]
    # Nothing reserved: a refused request must not occupy a slot.
    status = await _status(rag)
    assert status["pending_enqueue_tokens"] == {}


async def test_reservation_admits_up_to_capacity_and_records_weight():
    rag = await _rag(capacity=2, active=1)

    assert await _document_routes._reserve_enqueue_slot(rag, "tok-1") is True

    status = await _status(rag)
    assert status["pending_enqueue_tokens"]["tok-1"]["weight"] == 1
    assert status["pending_enqueues"] == 1


async def test_count_failure_is_503_not_capacity():
    rag = await _rag(capacity=5)
    rag.doc_status.error = StorageControlPlaneError("doc_status index unavailable")

    with pytest.raises(HTTPException) as excinfo:
        await _document_routes._reserve_enqueue_slot(rag, "tok-1")

    assert excinfo.value.status_code == 503
    # The storage message is not echoed to the client.
    assert "index unavailable" not in excinfo.value.detail


async def test_fence_wins_over_available_capacity():
    """A manual freeze refuses new ingress regardless of how much room there is
    (LR2 acceptance: "manual freeze 时无论容量是否充足都拒绝新 enqueue")."""
    rag = await _rag(capacity=100)
    status = await _status(rag)
    status["manual_freeze_requested"] = True

    with pytest.raises(HTTPException) as excinfo:
        await _document_routes._reserve_enqueue_slot(rag, "tok-1")

    assert excinfo.value.status_code == 409
    # The strict count is taken before the lock (counting inside it would hold
    # pipeline_status across a storage round-trip), so a fenced request still
    # pays for one count. The fence answer wins regardless of what it returned.
    assert rag.doc_status.calls == 1


async def test_reweight_survives_a_freeze_that_started_after_admission():
    """``/texts`` reserved before the freeze; re-weighting that same token must
    not turn into a 409 mid-request."""
    rag = await _rag(capacity=10)
    await _document_routes._reserve_enqueue_slot(rag, "tok-1")
    status = await _status(rag)
    status["manual_freeze_requested"] = True

    await _document_routes._reweight_enqueue_slot(rag, "tok-1", 4)

    status = await _status(rag)
    assert status["pending_enqueue_tokens"]["tok-1"]["weight"] == 4
    assert status["pending_enqueues"] == 1  # replaced, not added


async def test_reweight_refuses_a_batch_that_does_not_fit():
    rag = await _rag(capacity=3, active=1)
    await _document_routes._reserve_enqueue_slot(rag, "tok-1")

    with pytest.raises(HTTPException) as excinfo:
        await _document_routes._reweight_enqueue_slot(rag, "tok-1", 5)

    assert excinfo.value.status_code == 429
    # The rejected weight is not applied.
    status = await _status(rag)
    assert status["pending_enqueue_tokens"]["tok-1"]["weight"] == 1


async def test_reweight_is_a_noop_when_admission_is_disabled():
    rag = await _rag(capacity=0)
    await _document_routes._reserve_enqueue_slot(rag, "tok-1")

    await _document_routes._reweight_enqueue_slot(rag, "tok-1", 1000)

    assert rag.doc_status.calls == 0
    status = await _status(rag)
    assert status["pending_enqueues"] == 1


async def test_disabled_admission_never_counts():
    rag = await _rag(capacity=0, active=10_000)

    assert await _document_routes._reserve_enqueue_slot(rag, "tok-1") is True

    assert rag.doc_status.calls == 0
