"""``/documents`` DELETE (clear all): the LLM response cache is dropped only
when the caller opts in with clear_llm_cache=true, and the drop runs inside the
destructive reservation this endpoint already holds.

``llm_response_cache.drop()`` states a caller contract it cannot enforce: the
caller must hold the pipeline ``busy`` reservation. The former standalone
``POST /documents/clear_cache`` held nothing, so clearing mid-ingestion wiped
the extraction rows in-flight chunks had already paid for. Folding the
capability in here is what makes the contract hold, so these tests pin both the
opt-in and the fact that the route is gone.
"""

import importlib
import sys
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _NoopStorage:
    namespace = "noop"

    async def drop(self):
        return {"status": "success", "message": "data dropped"}


class _ClearRag:
    def __init__(self, workspace: str, cache_error: Exception | None = None):
        self.workspace = workspace
        storage = _NoopStorage()
        storage.workspace = workspace
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
        self.cache_error = cache_error
        # Recorded at the moment aclear_cache runs, so the test can prove the
        # drop happened while the reservation was still held.
        self.busy_during_cache_clear: bool | None = None
        self.destructive_busy_during_cache_clear: bool | None = None
        self.aclear_cache_calls = 0

    async def aclear_cache(self):
        from lightrag.kg.shared_storage import get_namespace_data

        self.aclear_cache_calls += 1
        status = await get_namespace_data("pipeline_status", workspace=self.workspace)
        self.busy_during_cache_clear = bool(status.get("busy"))
        self.destructive_busy_during_cache_clear = bool(status.get("destructive_busy"))
        if self.cache_error is not None:
            raise self.cache_error


def _clear_endpoint(rag, input_dir):
    router = create_document_routes(rag, DocumentManager(str(input_dir)))
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]


async def _init_workspace(workspace):
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)


async def test_clear_documents_preserves_llm_cache_by_default(tmp_path):
    """The cache survives a clear so re-adding the same documents can reuse
    the extraction results already paid for."""
    workspace = f"clear-cache-default-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint()

    assert response.status == "success"
    assert rag.aclear_cache_calls == 0
    assert "LLM response cache" not in response.message


async def test_clear_documents_drops_llm_cache_when_opted_in(tmp_path):
    workspace = f"clear-cache-optin-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "success"
    assert rag.aclear_cache_calls == 1
    assert "Cleared the LLM response cache." in response.message


async def test_llm_cache_drop_runs_inside_the_destructive_reservation(tmp_path):
    """The whole point of folding the capability in here: ``drop`` requires the
    caller to hold the pipeline ``busy`` reservation, and the standalone
    endpoint held nothing."""
    workspace = f"clear-cache-reserved-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "success"
    assert rag.busy_during_cache_clear is True
    assert rag.destructive_busy_during_cache_clear is True


async def test_clear_documents_refuses_while_pipeline_busy(tmp_path):
    """A busy pipeline refuses the whole clear, cache drop included — the
    behaviour the standalone cache endpoint did not have."""
    workspace = f"clear-cache-busy-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    pipeline_status["busy"] = True
    try:
        rag = _ClearRag(workspace)
        endpoint = _clear_endpoint(rag, tmp_path)

        response = await endpoint(clear_llm_cache=True)

        assert response.status == "busy"
        assert rag.aclear_cache_calls == 0
    finally:
        pipeline_status["busy"] = False


async def test_llm_cache_drop_failure_degrades_to_partial_success(tmp_path):
    """A failed cache drop must not be reported as a clean clear, and must not
    abort the document clear that already succeeded."""
    workspace = f"clear-cache-failure-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, cache_error=RuntimeError("cache backend down"))
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert rag.aclear_cache_calls == 1
    assert "Cleared the LLM response cache." not in response.message


async def test_standalone_clear_cache_route_is_gone(tmp_path):
    """It cleared the whole cache with no concurrency control at all; the
    capability lives on the destructive clear now."""
    workspace = f"clear-cache-route-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))

    paths = {getattr(route, "path", "") for route in router.routes}
    names = {getattr(route, "name", "") for route in router.routes}

    assert "/clear_cache" not in paths
    assert "clear_cache" not in names
