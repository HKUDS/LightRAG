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


class _FailingStorage:
    namespace = "failing"

    async def drop(self):
        return {"status": "error", "message": "backend refused the drop"}


class _ClearRag:
    def __init__(
        self,
        workspace: str,
        cache_error: Exception | None = None,
        failing_chunks: bool = False,
    ):
        self.workspace = workspace
        storage = _NoopStorage()
        storage.workspace = workspace
        if failing_chunks:
            chunks = _FailingStorage()
            chunks.workspace = workspace
            self.text_chunks = chunks
        else:
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


async def test_partial_success_names_the_cache_failure(tmp_path):
    """The WebUI surfaces ``message`` verbatim, so a failure that left the LLM
    cache in place must say so. A bare "some errors" tells the operator to
    retry without saying what to retry -- but the CATEGORY is what carries
    that, not the backend's own words."""
    workspace = f"clear-cache-named-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, cache_error=RuntimeError("cache backend down"))
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "LLM response cache" in response.message
    assert "error_id:" in response.message


async def test_the_cache_failure_does_not_leak_the_backend_text(tmp_path):
    """CWE-209: raw exception text names database hosts, ports and absolute
    paths. The 500 path of this same handler sanitizes via
    ``internal_server_error``; a 200 body must not reopen what that closes.
    The detail reaches the operator through the log, keyed by ``error_id``."""
    workspace = f"clear-cache-leak-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, cache_error=RuntimeError("cache backend down"))
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "cache backend down" not in response.message

    # pipeline_status history is served to clients by
    # GET /documents/pipeline_status, so it is a response channel too.
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    history = list(pipeline_status.get("history_messages", []))
    assert not any("cache backend down" in entry for entry in history)


async def test_a_failed_storage_drop_does_not_leak_the_backend_text(tmp_path):
    """Same rule for the storage-drop branch, including the non-raising
    ``{"status": "error", "message": ...}`` form: that message is
    backend-produced text and is just as free to quote a connection string as
    an exception is."""
    workspace = f"clear-storage-leak-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, failing_chunks=True)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "backend refused the drop" not in response.message
    # ...but the operator is still told which storage to retry.
    assert "_FailingStorage drop failed" in response.message
    assert "error_id:" in response.message


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


async def test_a_partial_storage_drop_preserves_the_llm_cache(tmp_path):
    """A surviving ``text_chunks`` row still names its cache rows through
    ``llm_cache_list``, and its document can still be reprocessed. Dropping
    the cache beside it would break those references en masse and re-bill
    every extraction call the document already paid for -- the exact harm
    folding this capability in here exists to prevent. Guarding only the
    TOTAL-failure case would inflict it on whatever survived."""
    workspace = f"clear-cache-partial-drop-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, failing_chunks=True)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert rag.aclear_cache_calls == 0
    assert "LLM cache preserved" in response.message
    # The operator has to be told it is still there, and what to do about it.
    assert "Re-run the clear" in response.message


async def test_the_preserved_cache_is_dropped_once_every_storage_succeeds(tmp_path):
    """The residue heals: the skip is a deferral, not a refusal."""
    workspace = f"clear-cache-partial-retry-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    failing = _ClearRag(workspace, failing_chunks=True)
    endpoint = _clear_endpoint(failing, tmp_path)
    assert (await endpoint(clear_llm_cache=True)).status == "partial_success"
    assert failing.aclear_cache_calls == 0

    healthy = _ClearRag(workspace)
    endpoint = _clear_endpoint(healthy, tmp_path)
    response = await endpoint(clear_llm_cache=True)

    assert response.status == "success"
    assert healthy.aclear_cache_calls == 1
