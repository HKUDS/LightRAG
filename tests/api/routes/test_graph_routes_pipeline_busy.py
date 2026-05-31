"""Pipeline-busy guard tests for graph mutation endpoints.

These tests verify that all 7 graph-mutation endpoints refuse to operate
with HTTP 409 while the document pipeline is busy:

- POST /graph/entity/edit          (graph_routes)
- POST /graph/relation/edit        (graph_routes)
- POST /graph/entity/create        (graph_routes)
- POST /graph/relation/create      (graph_routes)
- POST /graph/entities/merge       (graph_routes)
- DELETE /graph/entity/delete      (graph_routes)
- DELETE /graph/relation/delete    (graph_routes)

The guard logic itself lives in
``lightrag.api.routers.document_routes.check_pipeline_busy_or_raise`` and is
exercised both at the endpoint integration layer (via monkeypatch, no
shared-storage dependency) and at the unit layer (against a real
``pipeline_status`` namespace).
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# Importing routers loads ``lightrag.api.config`` which parses ``sys.argv`` via
# argparse. Stash argv so pytest's CLI flags don't trip the parser.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

create_graph_routes = _graph_routes.create_graph_routes
create_document_routes = _document_routes.create_document_routes
check_pipeline_busy_or_raise = _document_routes.check_pipeline_busy_or_raise

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _make_mock_rag() -> SimpleNamespace:
    """Build a minimal LightRAG stand-in with the 7 mutation methods stubbed.

    Each ``AsyncMock`` returns a payload shaped enough to satisfy the
    endpoint's response model so the idle pass-through test can verify the
    full request path. Busy tests don't rely on these return values; the
    guard short-circuits before they're reached.
    """
    return SimpleNamespace(
        workspace="",
        aedit_entity=AsyncMock(
            return_value={
                "entity_name": "Alice",
                "description": "updated",
                "operation_summary": {
                    "merged": False,
                    "merge_status": "not_attempted",
                    "merge_error": None,
                    "operation_status": "success",
                    "target_entity": None,
                    "final_entity": "Alice",
                    "renamed": False,
                },
            }
        ),
        aedit_relation=AsyncMock(return_value={"description": "updated"}),
        acreate_entity=AsyncMock(return_value={"entity_name": "Alice"}),
        acreate_relation=AsyncMock(return_value={"src_id": "a", "tgt_id": "b"}),
        amerge_entities=AsyncMock(return_value={"merged_entity": "Alice"}),
        adelete_by_entity=AsyncMock(
            return_value=SimpleNamespace(
                status="success", message="deleted", doc_id="ignored"
            )
        ),
        adelete_by_relation=AsyncMock(
            return_value=SimpleNamespace(
                status="success", message="deleted", doc_id="ignored"
            )
        ),
    )


def _build_client(rag: SimpleNamespace) -> TestClient:
    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    app.include_router(create_document_routes(rag, SimpleNamespace(), api_key=_API_KEY))
    return TestClient(app)


async def _force_busy_guard(_rag) -> None:
    """Stand-in for ``check_pipeline_busy_or_raise`` that always refuses."""
    raise HTTPException(
        status_code=409,
        detail=(
            "Pipeline is busy with another operation. "
            "Wait for the running job to finish before editing "
            "the knowledge graph."
        ),
    )


async def _noop_guard(_rag) -> None:
    """Stand-in for ``check_pipeline_busy_or_raise`` that always permits."""
    return None


def _patch_guard(monkeypatch, replacement) -> None:
    """Replace the guard reference in BOTH consumer modules.

    ``graph_routes`` re-binds the name via ``from .document_routes import ...``
    so patching only ``document_routes`` would miss the graph endpoints.
    """
    monkeypatch.setattr(_graph_routes, "check_pipeline_busy_or_raise", replacement)
    monkeypatch.setattr(_document_routes, "check_pipeline_busy_or_raise", replacement)


# ---------------------------------------------------------------------------
# Part A: endpoint integration -- guard refuses with 409
# ---------------------------------------------------------------------------

_ENDPOINTS = [
    pytest.param(
        "POST",
        "/graph/entity/edit",
        {"entity_name": "Alice", "updated_data": {"description": "x"}},
        id="update_entity",
    ),
    pytest.param(
        "POST",
        "/graph/relation/edit",
        {
            "source_id": "Alice",
            "target_id": "Bob",
            "updated_data": {"description": "x"},
        },
        id="update_relation",
    ),
    pytest.param(
        "POST",
        "/graph/entity/create",
        {"entity_name": "Alice", "entity_data": {"description": "x"}},
        id="create_entity",
    ),
    pytest.param(
        "POST",
        "/graph/relation/create",
        {
            "source_entity": "Alice",
            "target_entity": "Bob",
            "relation_data": {"description": "x"},
        },
        id="create_relation",
    ),
    pytest.param(
        "POST",
        "/graph/entities/merge",
        {"entities_to_change": ["Alic"], "entity_to_change_into": "Alice"},
        id="merge_entities",
    ),
    pytest.param(
        "DELETE",
        "/graph/entity/delete",
        {"entity_name": "Alice"},
        id="delete_entity",
    ),
    pytest.param(
        "DELETE",
        "/graph/relation/delete",
        {"source_entity": "Alice", "target_entity": "Bob"},
        id="delete_relation",
    ),
]


@pytest.mark.parametrize("method, path, body", _ENDPOINTS)
def test_endpoint_refuses_with_409_when_pipeline_busy(method, path, body, monkeypatch):
    rag = _make_mock_rag()
    client = _build_client(rag)
    _patch_guard(monkeypatch, _force_busy_guard)

    response = client.request(method, path, json=body, headers=_HEADERS)

    assert response.status_code == 409, response.text
    payload = response.json()
    assert "Pipeline is busy" in payload["detail"]
    # Guard must short-circuit before the underlying mutation runs.
    for attr in (
        "aedit_entity",
        "aedit_relation",
        "acreate_entity",
        "acreate_relation",
        "amerge_entities",
        "adelete_by_entity",
        "adelete_by_relation",
    ):
        getattr(rag, attr).assert_not_awaited()


def test_endpoint_passes_through_when_pipeline_idle(monkeypatch):
    """Sanity check: with an idle guard, the request reaches ``rag.aedit_entity``."""
    rag = _make_mock_rag()
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entity/edit",
        json={"entity_name": "Alice", "updated_data": {"description": "x"}},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    rag.aedit_entity.assert_awaited_once()


# ---------------------------------------------------------------------------
# Part B: helper unit -- against real pipeline_status namespace
# ---------------------------------------------------------------------------


async def _with_pipeline_status(action):
    """Bootstrap pipeline_status, run ``action(pipeline_status)``, then tear down.

    ``initialize_share_data`` is idempotent within a process but
    ``finalize_share_data`` is required to release the Manager/lock state so
    repeated calls in subsequent tests start clean.
    """
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        get_namespace_data,
        initialize_pipeline_status,
        initialize_share_data,
    )

    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace="")
        pipeline_status = await get_namespace_data("pipeline_status", workspace="")
        await action(pipeline_status)
    finally:
        finalize_share_data()


async def test_helper_raises_409_when_busy_flag_set():
    async def _do(pipeline_status):
        pipeline_status["busy"] = True
        rag = SimpleNamespace(workspace="")
        with pytest.raises(HTTPException) as exc_info:
            await check_pipeline_busy_or_raise(rag)
        assert exc_info.value.status_code == 409
        assert "Pipeline is busy" in exc_info.value.detail

    await _with_pipeline_status(_do)


async def test_helper_returns_silently_when_pipeline_idle():
    async def _do(pipeline_status):
        pipeline_status["busy"] = False
        rag = SimpleNamespace(workspace="")
        # Should not raise.
        await check_pipeline_busy_or_raise(rag)

    await _with_pipeline_status(_do)


async def test_helper_is_noop_when_pipeline_status_uninitialized():
    """When pipeline_status namespace was never bootstrapped the helper must pass.

    ``get_namespace_data`` raises ``PipelineNotInitializedError`` when the
    pipeline_status namespace is missing (share data initialized but the
    pipeline namespace never created); the helper swallows that error so test
    rigs without an end-to-end RAG bootstrap stay green. Mirrors the existing
    contract of ``_acquire_destructive_busy``.
    """
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )

    initialize_share_data()
    try:
        rag = SimpleNamespace(workspace="__never_bootstrapped__")
        # Intentionally skip ``initialize_pipeline_status``: helper should
        # catch ``PipelineNotInitializedError`` and return silently.
        await check_pipeline_busy_or_raise(rag)
    finally:
        finalize_share_data()
