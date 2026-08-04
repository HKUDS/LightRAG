"""Input ceilings on the Ollama-compatible routes (GHSA-r8jh-295g-vv42).

``/api/chat`` accepted a message of unlimited length and handed it straight to
tiktoken on the event loop, so one 64 MiB body took the whole server offline for
66 seconds. The bound that matters is not the status code but *where* the
request stops: these tests wire the tokenizer and the RAG to spies that raise if
touched, so a passing assertion means the payload was refused before any of the
work it was trying to buy.

There is no in-process test elsewhere that drives the real ``/api/chat``
handler, so the router is mounted here directly.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

from lightrag.constants import (  # noqa: E402
    MAX_IMAGES_PER_MESSAGE,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES_PER_REQUEST,
    MAX_MODEL_NAME_CHARS,
    MAX_QUERY_CHARS,
    MAX_REQUEST_TEXT_CHARS,
    MAX_ROLE_CHARS,
)

pytestmark = pytest.mark.offline


class _ExplodingRAG:
    """Every path a bounded request must not reach."""

    def __init__(self):
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_SIZE=1,
            LIGHTRAG_CREATED_AT="now",
            LIGHTRAG_DIGEST="sha",
        )
        self.role_llm_kwargs = {"query": None}
        self.llm_model_kwargs = {}
        self.role_llm_funcs = {"query": self._llm}

    async def _llm(self, prompt, **kwargs):  # pragma: no cover - overridden
        raise AssertionError("LLM reached with an oversized payload")

    async def aquery(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("aquery reached with an oversized payload")

    async def allm_model_func(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM reached with an oversized payload")


@pytest.fixture
def client(monkeypatch):
    """Real router, spies wired into the two sinks the advisory names."""

    def _explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("tokenizer reached with an oversized payload")

    async def _explode_async(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("tokenizer reached with an oversized payload")

    monkeypatch.setattr(_ollama_api, "estimate_tokens", _explode)
    monkeypatch.setattr(_ollama_api, "aestimate_tokens", _explode_async)
    monkeypatch.setattr(_utils_api, "auth_configured", False)

    app = FastAPI()
    app.include_router(_ollama_api.OllamaAPI(_ExplodingRAG()).router, prefix="/api")
    return TestClient(app)


def _chat(**overrides):
    body = {
        "model": "lightrag:latest",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# /api/chat
# --------------------------------------------------------------------------- #


def test_oversized_message_is_refused_before_tokenization(client):
    """The advisory's payload, scaled down. 413, and the tokenizer never ran."""
    response = client.post(
        "/api/chat",
        json=_chat(
            messages=[{"role": "user", "content": "B" * (MAX_MESSAGE_CHARS + 1)}]
        ),
    )
    assert response.status_code == 413


def test_many_legal_messages_cannot_rebuild_the_payload(client):
    """Without the aggregate check, N messages of the per-message maximum each
    reassemble exactly the body the per-message limit refused."""
    count = (MAX_REQUEST_TEXT_CHARS // MAX_MESSAGE_CHARS) + 2
    assert count <= MAX_MESSAGES_PER_REQUEST  # otherwise the count cap masks it
    response = client.post(
        "/api/chat",
        json=_chat(
            messages=[
                {"role": "user", "content": "B" * MAX_MESSAGE_CHARS}
                for _ in range(count)
            ]
        ),
    )
    assert response.status_code == 413


def test_chat_history_structure_consumes_aggregate_budget(client):
    """Chat history uses the same serialized representation as /query.

    Four maximum-size contents exactly fill the old text-only budget. The
    normalized history for the first three also contains roles and JSON
    structure, so the unified budget must now reject the request before any
    tokenizer or LLM work.
    """
    count = MAX_REQUEST_TEXT_CHARS // MAX_MESSAGE_CHARS
    assert count * MAX_MESSAGE_CHARS == MAX_REQUEST_TEXT_CHARS
    response = client.post(
        "/api/chat",
        json=_chat(
            messages=[
                {"role": "user", "content": "B" * MAX_MESSAGE_CHARS}
                for _ in range(count)
            ]
        ),
    )
    assert response.status_code == 413


def test_chat_forwards_history_shape_counted_by_the_budget(monkeypatch):
    """The handler forwards the same normalized history the validator counts."""
    captured = {}

    async def _count_tokens(_text):
        return 0

    class _RecordingRAG(_ExplodingRAG):
        async def aquery(self, query, *, param):
            captured["query"] = query
            captured["history"] = param.conversation_history
            return "answer"

    monkeypatch.setattr(_ollama_api, "aestimate_tokens", _count_tokens)
    monkeypatch.setattr(_utils_api, "auth_configured", False)
    app = FastAPI()
    app.include_router(_ollama_api.OllamaAPI(_RecordingRAG()).router, prefix="/api")

    response = TestClient(app).post(
        "/api/chat",
        json=_chat(
            messages=[
                {
                    "role": "system",
                    "content": "instructions",
                    "images": ["ignored-image"],
                },
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "current question"},
            ]
        ),
    )

    assert response.status_code == 200
    assert captured["query"] == "current question"
    assert captured["history"] == [
        {"role": "system", "content": "instructions"},
        {"role": "assistant", "content": "prior answer"},
    ]


def test_too_many_messages_are_refused(client):
    response = client.post(
        "/api/chat",
        json=_chat(
            messages=[
                {"role": "user", "content": "hi"}
                for _ in range(MAX_MESSAGES_PER_REQUEST + 1)
            ]
        ),
    )
    assert response.status_code == 413


def test_oversized_role_is_refused(client):
    """``role`` is forwarded to the model like ``content`` is."""
    response = client.post(
        "/api/chat",
        json=_chat(messages=[{"role": "u" * (MAX_ROLE_CHARS + 1), "content": "hello"}]),
    )
    assert response.status_code == 413


def test_oversized_system_prompt_is_refused(client):
    response = client.post(
        "/api/chat", json=_chat(system="S" * (MAX_MESSAGE_CHARS + 1))
    )
    assert response.status_code == 413


def test_too_many_images_are_refused(client):
    """Not tokenized, but buffered — and previously unbounded."""
    response = client.post(
        "/api/chat",
        json=_chat(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                    "images": ["ab"] * (MAX_IMAGES_PER_MESSAGE + 1),
                }
            ]
        ),
    )
    assert response.status_code == 413


def test_oversized_model_name_is_refused(client):
    response = client.post(
        "/api/chat", json=_chat(model="m" * (MAX_MODEL_NAME_CHARS + 1))
    )
    assert response.status_code == 413


# --------------------------------------------------------------------------- #
# /api/generate
# --------------------------------------------------------------------------- #


def test_oversized_prompt_is_refused_before_tokenization(client):
    response = client.post(
        "/api/generate",
        json={
            "model": "lightrag:latest",
            "prompt": "B" * (MAX_QUERY_CHARS + 1),
            "stream": False,
        },
    )
    assert response.status_code == 413


def test_generate_is_bounded_by_its_per_field_limits(monkeypatch):
    """For /api/generate the field limits are the binding constraint.

    ``prompt`` plus ``system`` cannot exceed MAX_REQUEST_TEXT_CHARS while each is
    within its own limit, so the aggregate check on this model is defence in
    depth against a future widening rather than the active bound. Pinning that
    here keeps someone from "fixing" an aggregate test that could never fire.
    """
    assert MAX_QUERY_CHARS + MAX_MESSAGE_CHARS <= MAX_REQUEST_TEXT_CHARS

    monkeypatch.setattr(_utils_api, "auth_configured", False)

    class _RecordingRAG(_ExplodingRAG):
        async def _llm(self, prompt, **kwargs):
            return "answer"

    rag = _RecordingRAG()
    rag.role_llm_funcs = {"query": rag._llm}
    app = FastAPI()
    app.include_router(_ollama_api.OllamaAPI(rag).router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/generate",
        json={
            "model": "lightrag:latest",
            "prompt": "B" * MAX_QUERY_CHARS,
            "system": "S" * MAX_MESSAGE_CHARS,
            "stream": False,
        },
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# The limits must not fire on ordinary traffic
# --------------------------------------------------------------------------- #


def test_a_malformed_body_is_still_a_400_not_a_413(client):
    """Size and shape failures stay distinguishable."""
    response = client.post("/api/chat", json={"model": "lightrag:latest"})
    assert response.status_code == 400


def test_a_normal_conversation_passes_validation(monkeypatch):
    """Guards against the ceilings rejecting a realistic client.

    Open WebUI and friends send the whole conversation; 32 turns of ordinary
    length must sail through. The tokenizer is left real here — reaching it is
    the point.
    """
    monkeypatch.setattr(_utils_api, "auth_configured", False)
    reached = {}

    class _RecordingRAG(_ExplodingRAG):
        async def aquery(self, query, param=None, **kwargs):
            reached["query"] = query
            return "answer"

    app = FastAPI()
    app.include_router(_ollama_api.OllamaAPI(_RecordingRAG()).router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json=_chat(
            # 63 turns, ending on the user message the handler requires.
            messages=[
                {"role": "user" if i % 2 == 0 else "assistant", "content": "hi " * 50}
                for i in range(63)
            ]
        ),
    )

    assert response.status_code == 200
    assert "query" in reached
