"""``/documents/clear`` clears the pipeline ingress mailbox (Phase 3).

The clear endpoint owns busy+destructive while it drops every storage: the
documents the mailbox refers to cease to exist, so the mailbox is cleared in
the same window — un-ACKed manual retry requests are retired as
CANCELLED_BY_CLEAR (a delayed replay of the same request id is refused), and
the document/auto channels are emptied.
"""

import importlib
import sys
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.kg.pipeline_ingress import PipelineIngressMessage  # noqa: E402
from lightrag.kg.shared_storage import get_pipeline_ingress  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _NoopStorage:
    """Minimal storage the clear endpoint can ``drop()`` and report on."""

    namespace = "noop"

    async def drop(self):
        return {"status": "success", "message": "data dropped"}


class _ClearRag:
    def __init__(self, workspace: str):
        self.workspace = workspace
        storage = _NoopStorage()
        storage.workspace = workspace
        # The eleven storage attributes the clear endpoint iterates over.
        self.text_chunks = storage
        self.full_docs = storage
        self.full_entities = storage
        self.full_relations = storage
        self.entity_chunks = storage
        self.relation_chunks = storage
        self.entities_vdb = storage
        self.relationships_vdb = storage
        self.chunks_vdb = storage
        self.chunk_entity_relation_graph = storage
        self.doc_status = storage

    async def aclear_cache(self, modes=None):
        return None


async def test_clear_documents_clears_ingress_and_refuses_replay(tmp_path):
    """All three active mailbox channels are emptied under the destructive
    reservation, and the retired manual request id cannot be replayed into the
    fresh (empty) workspace."""
    workspace = f"clear-ingress-{uuid4().hex[:8]}"
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    # Idempotent within a process; a unique workspace keeps this test isolated
    # even when the shared dicts already exist from a sibling test.
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)

    ingress = await get_pipeline_ingress(workspace)
    ingress.put_document(PipelineIngressMessage(kind="document", doc_id="doc-x"))
    ingress.request_auto_rescan()
    manual_msg = PipelineIngressMessage(
        kind="rescan", retry_failed=True, request_id="req-cleared"
    )
    assert ingress.request_manual_retry("req-cleared", manual_msg) is True

    rag = _ClearRag(workspace)
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))
    clear_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]

    response = await clear_endpoint()
    assert response.status in ("success", "partial_success")

    counts = ingress.counts()
    assert counts["documents"] == 0
    assert counts["auto_rescan_pending"] is False
    assert counts["manual_retries"] == 0
    assert ingress.has_work() is False

    # CANCELLED_BY_CLEAR is terminal: a delayed replay of the same id must be
    # refused instead of re-entering the now-empty workspace.
    assert ingress.request_manual_retry("req-cleared", manual_msg) is False
    assert ingress.snapshot_manual_retries() == []

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False


async def test_clear_documents_survives_ingress_clear_failure(tmp_path):
    """An ingress ``clear()`` failure must not fail the clear operation: the
    degradation is safe (residual messages are compacted by consumption
    idempotence; a surviving sticky request ACKs against the emptied
    doc_status) and the destructive reservation is still released."""
    workspace = f"clear-ingress-fail-{uuid4().hex[:8]}"
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)

    ingress = await get_pipeline_ingress(workspace)
    ingress.put_document(PipelineIngressMessage(kind="document", doc_id="doc-x"))

    def dead_clear():
        raise RuntimeError("manager down: clear")

    ingress.clear = dead_clear

    rag = _ClearRag(workspace)
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))
    clear_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]

    response = await clear_endpoint()
    assert response.status in ("success", "partial_success")

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False
