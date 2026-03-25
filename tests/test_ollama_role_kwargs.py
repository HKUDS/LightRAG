"""Offline regression tests for Ollama API role-specific kwargs."""

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.offline


class _FakeRag:
    def __init__(self):
        self.base_calls = []
        self.query_calls = []
        self.llm_model_kwargs = {"route": "base"}
        self.query_llm_model_kwargs = {"route": "query"}
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_CREATED_AT="2026-03-14T00:00:00Z",
            LIGHTRAG_SIZE=0,
        )

        async def base_func(*args, **kwargs):
            self.base_calls.append(kwargs)
            return "base"

        async def query_func(*args, **kwargs):
            self.query_calls.append(kwargs)
            return "query"

        self.llm_model_func = base_func
        self.query_llm_model_func = query_func

    async def aquery(self, *args, **kwargs):
        return "aquery"


def _make_client(monkeypatch) -> tuple[TestClient, _FakeRag]:
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    for module_name in [
        "lightrag.api.config",
        "lightrag.api.auth",
        "lightrag.api.utils_api",
        "lightrag.api.routers",
        "lightrag.api.routers.ollama_api",
    ]:
        sys.modules.pop(module_name, None)
    ollama_api_module = importlib.import_module("lightrag.api.routers.ollama_api")
    OllamaAPI = ollama_api_module.OllamaAPI
    rag = _FakeRag()
    api = OllamaAPI(rag, top_k=20, api_key=None)
    app = FastAPI()
    app.include_router(api.router, prefix="/api")
    return TestClient(app), rag


def test_generate_non_stream_uses_query_role_kwargs_without_mutating_base(monkeypatch):
    client, rag = _make_client(monkeypatch)

    response = client.post(
        "/api/generate",
        json={
            "model": "lightrag:latest",
            "prompt": "Summarize this",
            "stream": False,
            "system": "custom system",
        },
    )

    assert response.status_code == 200
    assert rag.base_calls == []
    assert rag.query_calls[-1]["route"] == "query"
    assert rag.query_calls[-1]["system_prompt"] == "custom system"
    assert "system_prompt" not in rag.llm_model_kwargs
    assert "system_prompt" not in rag.query_llm_model_kwargs


def test_chat_bypass_stream_uses_query_role_kwargs_without_mutating_base(monkeypatch):
    client, rag = _make_client(monkeypatch)

    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "lightrag:latest",
            "stream": True,
            "system": "chat system",
            "messages": [
                {"role": "assistant", "content": "history"},
                {"role": "user", "content": "/bypass give me a title"},
            ],
        },
    ) as response:
        assert response.status_code == 200
        # Consume the streaming response fully.
        list(response.iter_lines())

    assert rag.base_calls == []
    assert rag.query_calls[-1]["route"] == "query"
    assert rag.query_calls[-1]["system_prompt"] == "chat system"
    assert rag.query_calls[-1]["history_messages"] == [
        {"role": "assistant", "content": "history"}
    ]
    assert "system_prompt" not in rag.llm_model_kwargs
    assert "system_prompt" not in rag.query_llm_model_kwargs
