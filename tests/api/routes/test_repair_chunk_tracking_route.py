"""``POST /documents/recovery/repair_chunk_tracking`` — the operator entry point (#3838, R4).

The repair itself is pinned in ``tests/test_chunk_tracking_repair.py``. What the
route owns is the operator contract around it: an explicit confirmation (it
DROPS both tracking namespaces), the destructive reservation (it drops storages
the pipeline and the delete path write concurrently), and releasing that
reservation on every exit — a leaked ``busy`` would wedge the workspace behind a
repair that already finished.
"""

import importlib
import sys
from uuid import uuid4

import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.storage_migrations import ChunkTrackingRepairReport  # noqa: E402

DocumentManager = _document_routes.DocumentManager
RepairChunkTrackingRequest = _document_routes.RepairChunkTrackingRequest
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _RepairRag:
    def __init__(self, workspace, *, report=None, error=None):
        self.workspace = workspace
        self.entity_chunks = object()
        self.relation_chunks = object()
        self._report = report or ChunkTrackingRepairReport(
            scanned_documents=1,
            scanned_chunks=2,
            chunks_with_cache=2,
            entity_rows_written=2,
            relation_rows_written=1,
        )
        self._error = error
        self.calls = 0

    async def arepair_chunk_tracking(self):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._report


async def _endpoint(rag, tmp_path):
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "repair_chunk_tracking"
    ][-1]


async def _workspace():
    workspace = f"repair-tracking-{uuid4().hex[:8]}"
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)
    return workspace, shared_storage


async def test_repair_runs_and_releases_the_destructive_slot(tmp_path):
    workspace, shared_storage = await _workspace()
    rag = _RepairRag(workspace)
    endpoint = await _endpoint(rag, tmp_path)

    response = await endpoint(RepairChunkTrackingRequest(confirm=True))

    assert response.status == "success"
    assert rag.calls == 1
    assert response.entity_rows_written == 2
    assert response.relation_rows_written == 1
    assert response.chunks_with_cache == 2

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False


async def test_repair_requires_confirmation(tmp_path):
    """It drops both namespaces and objects with no cached chunk lose their row,
    so it must never run on an unconfirmed request."""
    workspace, _ = await _workspace()
    rag = _RepairRag(workspace)
    endpoint = await _endpoint(rag, tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await endpoint(RepairChunkTrackingRequest())

    assert excinfo.value.status_code == 400
    assert rag.calls == 0


async def test_repair_refuses_while_another_writer_owns_the_workspace(tmp_path):
    """The pipeline and the delete path write the namespaces this repair drops."""
    workspace, shared_storage = await _workspace()
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    lock = shared_storage.get_namespace_lock("pipeline_status", workspace=workspace)
    async with lock:
        pipeline_status.update({"busy": True})

    rag = _RepairRag(workspace)
    endpoint = await _endpoint(rag, tmp_path)

    response = await endpoint(RepairChunkTrackingRequest(confirm=True))

    assert response.status == "busy"
    assert rag.calls == 0
    # The peer's reservation survives the refusal.
    assert pipeline_status.get("busy") is True


async def test_a_failing_repair_releases_the_slot_and_reports_500(tmp_path):
    workspace, shared_storage = await _workspace()
    rag = _RepairRag(workspace, error=RuntimeError("cache backend down"))
    endpoint = await _endpoint(rag, tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await endpoint(RepairChunkTrackingRequest(confirm=True))

    assert excinfo.value.status_code == 500
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False


async def test_repair_refuses_when_chunk_tracking_is_not_configured(tmp_path):
    workspace, _ = await _workspace()
    rag = _RepairRag(workspace)
    rag.entity_chunks = None
    endpoint = await _endpoint(rag, tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await endpoint(RepairChunkTrackingRequest(confirm=True))

    assert excinfo.value.status_code == 400
    assert rag.calls == 0


async def test_warnings_reach_the_operator(tmp_path):
    """The degradations the repair accepts are only useful if they are reported."""
    workspace, _ = await _workspace()
    report = ChunkTrackingRepairReport(
        scanned_chunks=3,
        chunks_without_cache=3,
        entities_without_evidence=2,
        warnings=["nothing could be recovered from the extraction cache"],
    )
    rag = _RepairRag(workspace, report=report)
    endpoint = await _endpoint(rag, tmp_path)

    response = await endpoint(RepairChunkTrackingRequest(confirm=True))

    assert response.status == "success"
    assert response.warnings == [
        "nothing could be recovered from the extraction cache"
    ]
    assert response.entities_without_evidence == 2
