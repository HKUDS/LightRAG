"""A failed query is reported as a failure, with no raw exception text (#4475).

When the answering LLM call failed, ``/query`` used to answer HTTP 200 with
"No relevant context found for the query.", ``/query/stream`` the same line,
and the Ollama-compatible ``/api/chat`` "No response generated": the client
could not tell a failed query from an empty knowledge base. The failure now
raises out of ``LightRAG.aquery_llm`` and every route reports it through the
client-safe helpers -- an HTTP 500 before a response has started, an inline
``error`` line after -- so neither a status nor a body hides it, and neither
carries the provider's error text.

The routes drive the real ``LightRAG.aquery_llm`` / ``aquery`` over a stand-in
whose retrieval succeeds and whose answering LLM times out.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag import LightRAG
from lightrag.utils import Tokenizer

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_SECRET = "Request to https://internal-llm.corp.example:8443/v1 timed out (deployment=secret-deploy-123)"
# Every needle must contain a character outside [0-9a-f]: the body carries a
# random hex correlation id.
_NEEDLES = ("internal-llm.corp.example", ":8443/v1", "secret-deploy-123", "timed out")
_SAFE_MESSAGE = re.compile(r"^Internal server error \(error_id: [0-9a-f]{12}\)$")
_OLD_PLACEHOLDERS = ("No relevant context found", "No response generated")
_KEYWORDS = json.dumps(
    {"high_level_keywords": ["evidence"], "low_level_keywords": ["evidence"]}
)


class _CharacterTokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _assert_failure_body_is_safe(text: str) -> None:
    for needle in _NEEDLES + _OLD_PLACEHOLDERS:
        assert needle not in text, f"response body carries {needle!r}"


@pytest.fixture(autouse=True)
def _open_auth(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", False)


@pytest.fixture
def failing_rag(monkeypatch):
    """Retrieval finds a chunk; the answering LLM then times out."""
    chunk = {
        "id": "chunk-1",
        "chunk_id": "chunk-1",
        "content": "Retrieved evidence.",
        "file_path": "source.md",
    }
    search = {
        "final_entities": [],
        "final_relations": [],
        "vector_chunks": [chunk],
        "chunk_tracking": {"chunk-1": {"source": "C", "frequency": 1, "order": 1}},
        "query_embedding": None,
    }
    monkeypatch.setattr(
        "lightrag.operate._perform_kg_search", AsyncMock(return_value=search)
    )
    answering_llm = AsyncMock(side_effect=TimeoutError(_SECRET))
    config = {
        "tokenizer": Tokenizer("character", _CharacterTokenizer()),
        "role_llm_funcs": {
            "query": answering_llm,
            "keyword": AsyncMock(return_value=_KEYWORDS),
        },
        "addon_params": {"language": "en"},
    }
    rag = SimpleNamespace(
        _build_global_config=lambda: config,
        _query_done=AsyncMock(),
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks=SimpleNamespace(global_config=config),
        chunks_vdb=SimpleNamespace(
            cosine_better_than_threshold=0.0, query=AsyncMock(return_value=[chunk])
        ),
        llm_response_cache=SimpleNamespace(
            global_config={"enable_llm_cache": False},
            get_by_id=AsyncMock(return_value=None),
            upsert=AsyncMock(),
        ),
        # What the Ollama-compatible router reads.
        ollama_server_infos=SimpleNamespace(
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_SIZE=1,
            LIGHTRAG_CREATED_AT="now",
            LIGHTRAG_DIGEST="sha",
        ),
        role_llm_funcs=config["role_llm_funcs"],
        role_llm_kwargs={"query": None},
        llm_model_kwargs={},
    )
    rag.aquery_llm = partial(LightRAG.aquery_llm, rag)
    rag.aquery = partial(LightRAG.aquery, rag)
    return rag, answering_llm


def _query_client(rag) -> TestClient:
    app = FastAPI()
    app.include_router(_query_routes.create_query_routes(rag))
    return TestClient(app)


@pytest.fixture
def ollama_client(monkeypatch, failing_rag):
    async def _count_tokens(_text):
        return 0

    monkeypatch.setattr(_ollama_api, "aestimate_tokens", _count_tokens)
    app = FastAPI()
    app.include_router(_ollama_api.OllamaAPI(failing_rag[0]).router, prefix="/api")
    return TestClient(app)


# --------------------------------------------------------------------------- #
# /query and /query/stream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["mix", "naive", "bypass"])
def test_query_reports_a_failed_llm_call_as_500(failing_rag, mode):
    rag, answering_llm = failing_rag

    response = _query_client(rag).post(
        "/query", json={"query": "Find evidence", "mode": mode}
    )

    assert response.status_code == 500
    assert _SAFE_MESSAGE.match(response.json()["detail"])
    _assert_failure_body_is_safe(response.text)
    answering_llm.assert_awaited_once()


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("mode", ["mix", "bypass"])
def test_query_stream_reports_a_failure_before_the_stream_as_500(
    failing_rag, mode, stream
):
    rag, answering_llm = failing_rag

    response = _query_client(rag).post(
        "/query/stream",
        json={"query": "Find evidence", "mode": mode, "stream": stream},
    )

    assert response.status_code == 500
    assert _SAFE_MESSAGE.match(response.json()["detail"])
    _assert_failure_body_is_safe(response.text)
    answering_llm.assert_awaited_once()


def test_query_stream_with_progress_reports_the_failure_as_an_error_line(
    failing_rag,
):
    """Progress lines open the stream first, so the failure arrives inline."""
    rag, answering_llm = failing_rag

    response = _query_client(rag).post(
        "/query/stream",
        json={"query": "Find evidence", "mode": "mix", "include_progress": True},
    )

    assert response.status_code == 200
    _assert_failure_body_is_safe(response.text)
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert set(lines[-1]) == {"error"}
    assert _SAFE_MESSAGE.match(lines[-1]["error"])
    assert not any("response" in line for line in lines)
    answering_llm.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Ollama-compatible /api/chat (RAG branch, which goes through LightRAG.aquery)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stream", [True, False])
def test_ollama_chat_reports_a_failed_query_as_500(failing_rag, ollama_client, stream):
    _, answering_llm = failing_rag

    response = ollama_client.post(
        "/api/chat",
        json={
            "model": "lightrag:latest",
            "messages": [{"role": "user", "content": "Find evidence"}],
            "stream": stream,
        },
    )

    assert response.status_code == 500
    assert _SAFE_MESSAGE.match(response.json()["detail"])
    _assert_failure_body_is_safe(response.text)
    answering_llm.assert_awaited_once()
