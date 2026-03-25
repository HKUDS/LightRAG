import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.prompt_version_store import PromptVersionStore

pytestmark = pytest.mark.offline


class _DummyRAG:
    def __init__(self, *args, **kwargs):
        self.ollama_server_infos = kwargs.get("ollama_server_infos")
        self.prompt_version_store = PromptVersionStore(
            kwargs["working_dir"], workspace=kwargs.get("workspace", "")
        )

    async def initialize_storages(self):
        return None

    async def check_and_migrate_data(self):
        return None

    async def finalize_storages(self):
        return None


class _DummyOllamaAPI:
    def __init__(self, rag, top_k=60, api_key=None):
        self.router = APIRouter()


def _build_test_client(monkeypatch):
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    from lightrag.api import config as api_config
    from lightrag.api import lightrag_server

    monkeypatch.setattr(lightrag_server, "LightRAG", _DummyRAG)
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _DummyOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *args, **kwargs: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *args, **kwargs: APIRouter()
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
    app = lightrag_server.create_app(args)
    return TestClient(app)


@pytest.fixture
def test_client(monkeypatch):
    return _build_test_client(monkeypatch)


def test_initialize_prompt_config_creates_seed_versions(test_client):
    response = test_client.post("/prompt-config/initialize")

    assert response.status_code == 200
    body = response.json()
    assert body["indexing"]["versions"]
    assert body["retrieval"]["versions"]


def test_activate_indexing_version_returns_warning_metadata(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    version_id = seeded["indexing"]["versions"][0]["version_id"]

    response = test_client.post(f"/prompt-config/indexing/versions/{version_id}/activate")

    assert response.status_code == 200
    assert "warning" in response.json()


def test_delete_active_version_is_rejected(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    active_id = seeded["retrieval"]["versions"][0]["version_id"]
    test_client.post(f"/prompt-config/retrieval/versions/{active_id}/activate")

    response = test_client.delete(f"/prompt-config/retrieval/versions/{active_id}")

    assert response.status_code == 400


def test_health_exposes_active_prompt_version_summary(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    active_id = seeded["retrieval"]["versions"][0]["version_id"]
    active_name = seeded["retrieval"]["versions"][0]["version_name"]
    test_client.post(f"/prompt-config/retrieval/versions/{active_id}/activate")

    response = test_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["configuration"]["active_prompt_versions"]["retrieval"] == {
        "active_version_id": active_id,
        "active_version_name": active_name,
    }
