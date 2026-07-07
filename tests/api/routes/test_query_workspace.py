"""
Regression test for #2904: /query endpoints must respect LIGHTRAG-WORKSPACE header.

Verifies that the three query endpoints (/query, /query/stream, /query/data)
route to workspace-scoped RAG instances based on the LIGHTRAG-WORKSPACE header,
and fall back to the default instance when the header is absent.
"""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock


_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")


def _make_mock_rag(workspace: str = "") -> MagicMock:
    """Create a mock RAG instance tagged with its workspace."""
    mock = MagicMock()
    mock.workspace = workspace
    mock.aquery_llm = AsyncMock(
        return_value={
            "llm_response": {"content": f"answer from {workspace or 'default'}"},
            "data": {"references": []},
        }
    )
    mock.aquery_data = AsyncMock(
        return_value={
            "status": "success",
            "message": "ok",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {},
        }
    )
    return mock


def _build_workspace_client():
    """Build a TestClient with workspace-aware query routing."""
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from lightrag.api.routers.query_routes import create_query_routes
        from lightrag.api.utils_api import parse_workspace_header

        default_rag = _make_mock_rag("default")
        tenant_a_rag = _make_mock_rag("tenant_a")
        pool = {"default": default_rag, "tenant_a": tenant_a_rag}

        async def resolve_workspace_rag(request: Request):
            ws = parse_workspace_header(request)
            return pool.get(ws or "default", default_rag)

        app = FastAPI()
        router = create_query_routes(
            default_rag,
            resolve_workspace_rag=resolve_workspace_rag,
        )
        app.include_router(router)
        return TestClient(app), default_rag, tenant_a_rag
    finally:
        sys.argv = original_argv


class TestQueryWorkspaceRouting:
    """The /query endpoints must route to the correct workspace RAG instance."""

    def test_query_without_header_uses_default(self):
        client, default_rag, tenant_a_rag = _build_workspace_client()
        resp = client.post("/query", json={"query": "test question", "mode": "naive"})
        assert resp.status_code == 200
        default_rag.aquery_llm.assert_called_once()
        tenant_a_rag.aquery_llm.assert_not_called()

    def test_query_with_header_routes_to_workspace(self):
        client, default_rag, tenant_a_rag = _build_workspace_client()
        resp = client.post(
            "/query",
            json={"query": "test question", "mode": "naive"},
            headers={"LIGHTRAG-WORKSPACE": "tenant_a"},
        )
        assert resp.status_code == 200
        tenant_a_rag.aquery_llm.assert_called_once()
        default_rag.aquery_llm.assert_not_called()

    def test_query_stream_with_header_routes_to_workspace(self):
        client, default_rag, tenant_a_rag = _build_workspace_client()
        resp = client.post(
            "/query/stream",
            json={"query": "test question", "mode": "naive"},
            headers={"LIGHTRAG-WORKSPACE": "tenant_a"},
        )
        assert resp.status_code == 200
        tenant_a_rag.aquery_llm.assert_called_once()
        default_rag.aquery_llm.assert_not_called()

    def test_query_data_with_header_routes_to_workspace(self):
        client, default_rag, tenant_a_rag = _build_workspace_client()
        resp = client.post(
            "/query/data",
            json={"query": "test question", "mode": "naive"},
            headers={"LIGHTRAG-WORKSPACE": "tenant_a"},
        )
        assert resp.status_code == 200
        tenant_a_rag.aquery_data.assert_called_once()
        default_rag.aquery_data.assert_not_called()

    def test_workspace_header_sanitized(self):
        client, default_rag, _ = _build_workspace_client()
        resp = client.post(
            "/query",
            json={"query": "test question", "mode": "naive"},
            headers={"LIGHTRAG-WORKSPACE": "bad-name!"},
        )
        # Sanitized to "bad_name_" which is not in pool -> falls back to default
        assert resp.status_code == 200
        default_rag.aquery_llm.assert_called_once()

    def test_no_resolver_falls_back_to_closure_rag(self):
        """When resolve_workspace_rag is None, endpoints use the default rag."""
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["lightrag-server"]
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from lightrag.api.routers.query_routes import create_query_routes

            default_rag = _make_mock_rag("default")
            app = FastAPI()
            router = create_query_routes(default_rag)
            app.include_router(router)
            client = TestClient(app)

            resp = client.post(
                "/query",
                json={"query": "test question", "mode": "naive"},
                headers={"LIGHTRAG-WORKSPACE": "anything"},
            )
            assert resp.status_code == 200
            default_rag.aquery_llm.assert_called_once()
        finally:
            sys.argv = original_argv
