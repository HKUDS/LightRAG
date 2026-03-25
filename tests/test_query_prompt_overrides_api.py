import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline


class _DummyRAG:
    def __init__(self, *args, **kwargs):
        self.ollama_server_infos = kwargs.get("ollama_server_infos")

    async def initialize_storages(self):
        return None

    async def check_and_migrate_data(self):
        return None

    async def finalize_storages(self):
        return None

    async def aquery_llm(self, query, param=None):
        return {
            "llm_response": {"content": f"echo:{query}", "is_streaming": False},
            "data": {"references": []},
        }

    async def aquery_data(self, query, param=None):
        return {"status": "success", "message": "ok", "data": {}, "metadata": {}}


class _DummyOllamaAPI:
    def __init__(self, rag, top_k=60, api_key=None):
        self.router = APIRouter()


def _build_test_client(
    monkeypatch,
    *,
    rag_cls=_DummyRAG,
    allow_prompt_overrides_via_api: bool,
):
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    monkeypatch.setenv(
        "ALLOW_PROMPT_OVERRIDES_VIA_API",
        "true" if allow_prompt_overrides_via_api else "false",
    )

    from lightrag.api import config as api_config
    from lightrag.api import lightrag_server
    from lightrag.api.routers import query_routes as query_routes_module

    query_routes_module.router.routes.clear()

    monkeypatch.setattr(lightrag_server, "LightRAG", rag_cls)
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _DummyOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *args, **kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *args, **kwargs: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (False, False))
    monkeypatch.setattr(
        lightrag_server, "get_combined_auth_dependency", lambda *_: (lambda: None)
    )
    monkeypatch.setattr(
        lightrag_server, "global_args", SimpleNamespace(cors_origins="*")
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(lightrag_server, "cleanup_keyed_lock", lambda: {})
    fake_ollama_module = types.ModuleType("lightrag.llm.ollama")

    async def _fake_ollama_model_complete(*args, **kwargs):
        return "ok"

    async def _fake_ollama_embed(*args, **kwargs):
        return []

    fake_ollama_module.ollama_model_complete = _fake_ollama_model_complete
    fake_ollama_module.ollama_embed = _fake_ollama_embed
    monkeypatch.setitem(sys.modules, "lightrag.llm.ollama", fake_ollama_module)

    async def _fake_get_namespace_data(*args, **kwargs):
        return {"busy": False}

    monkeypatch.setattr(lightrag_server, "get_namespace_data", _fake_get_namespace_data)

    args = api_config.parse_args()
    args.allow_prompt_overrides_via_api = allow_prompt_overrides_via_api
    app = lightrag_server.create_app(args)
    return TestClient(app)


@pytest.fixture
def test_client(monkeypatch):
    return _build_test_client(
        monkeypatch,
        allow_prompt_overrides_via_api=False,
    )


@pytest.fixture
def test_client_capability_enabled_with_value_error(monkeypatch):
    class _DummyRAGValueError(_DummyRAG):
        async def aquery_llm(self, query, param=None):
            raise ValueError("Invalid prompt_overrides payload")

    return _build_test_client(
        monkeypatch,
        rag_cls=_DummyRAGValueError,
        allow_prompt_overrides_via_api=True,
    )


@pytest.fixture
def test_client_capability_enabled(monkeypatch):
    return _build_test_client(
        monkeypatch,
        allow_prompt_overrides_via_api=True,
    )


def test_query_request_converts_prompt_overrides_to_query_param():
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.query_routes import QueryRequest
    sys.argv = original_argv

    request = QueryRequest(
        query="hello world",
        mode="mix",
        prompt_overrides={"query": {"rag_response": "{context_data}"}},
    )
    param = request.to_query_params(False)
    assert param.prompt_overrides["query"]["rag_response"] == "{context_data}"


def test_query_request_prompt_overrides_schema_is_structured():
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.query_routes import QueryRequest
    sys.argv = original_argv

    schema = QueryRequest.model_json_schema()
    prompt_schema = schema["properties"]["prompt_overrides"]
    assert prompt_schema["anyOf"][0]["$ref"].endswith("QueryPromptOverridesPayload")


def test_query_endpoint_rejects_prompt_overrides_when_capability_disabled(test_client):
    response = test_client.post(
        "/query",
        json={
            "query": "hello world",
            "mode": "mix",
            "prompt_overrides": {"query": {"rag_response": "{context_data}"}},
        },
    )
    assert response.status_code == 403


def test_query_endpoint_rejects_empty_prompt_overrides_when_capability_disabled(
    test_client,
):
    response = test_client.post(
        "/query",
        json={"query": "hello world", "mode": "mix", "prompt_overrides": {}},
    )
    assert response.status_code == 403


def test_query_stream_endpoint_rejects_empty_prompt_overrides_when_capability_disabled(
    test_client,
):
    response = test_client.post(
        "/query/stream",
        json={"query": "hello world", "mode": "mix", "prompt_overrides": {}},
    )
    assert response.status_code == 403


def test_query_data_endpoint_rejects_empty_prompt_overrides_when_capability_disabled(
    test_client,
):
    response = test_client.post(
        "/query/data",
        json={"query": "hello world", "mode": "mix", "prompt_overrides": {}},
    )
    assert response.status_code == 403


def test_health_exposes_prompt_override_capability(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"]["allow_prompt_overrides_via_api"] is False


def test_query_endpoint_returns_400_for_invalid_prompt_overrides_when_capability_enabled(
    test_client_capability_enabled,
):
    response = test_client_capability_enabled.post(
        "/query",
        json={
            "query": "hello world",
            "mode": "mix",
            "prompt_overrides": {"bad-family": {"x": 1}},
        },
    )
    assert response.status_code == 422


def test_query_endpoint_keeps_backend_value_error_as_500_when_prompt_overrides_are_valid(
    test_client_capability_enabled_with_value_error,
):
    response = test_client_capability_enabled_with_value_error.post(
        "/query",
        json={
            "query": "hello world",
            "mode": "mix",
            "prompt_overrides": {"query": {"rag_response": "{context_data}"}},
        },
    )
    assert response.status_code == 500


def test_query_endpoint_rejects_prompt_overrides_in_bypass_mode(
    test_client_capability_enabled,
):
    response = test_client_capability_enabled.post(
        "/query",
        json={
            "query": "hello world",
            "mode": "bypass",
            "prompt_overrides": {"query": {"rag_response": "{context_data}"}},
        },
    )
    assert response.status_code == 400
