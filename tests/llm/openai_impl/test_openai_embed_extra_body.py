"""``openai_embed`` forwards ``OPENAI_LLM_EXTRA_BODY`` on the embedding request.

Chat completions already receive that JSON object through
``OpenAILLMOptions.extra_body``. ``openai_embed`` builds its own parameter dict,
so a gateway that requires metadata in the body accepted LLM calls and rejected
embeddings. A blank value leaves the embedding request unchanged. Invalid JSON
fails the call instead of sending a body the gateway will reject.
"""

import json
from types import SimpleNamespace

import pytest

from lightrag.llm.openai import openai_embed

pytestmark = pytest.mark.offline

MODEL = "text-embedding-3-small"


class _FakeEmbeddingClient:
    def __init__(self, captured):
        self._captured = captured
        self.embeddings = SimpleNamespace(create=self._create)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _create(self, **params):
        self._captured.append(params)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0, 1.0]) for _ in params["input"]]
        )


async def _embed(monkeypatch, extra_body):
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )
    monkeypatch.delenv("OPENAI_LLM_EXTRA_BODY", raising=False)
    if extra_body is not None:
        monkeypatch.setenv("OPENAI_LLM_EXTRA_BODY", extra_body)
    await openai_embed.func(["embed this"], model=MODEL, api_key="test-key")
    assert len(captured) == 1
    return captured[0]


async def test_blank_extra_body_is_omitted(monkeypatch):
    params = await _embed(monkeypatch, None)
    assert "extra_body" not in params

    params = await _embed(monkeypatch, "   ")
    assert "extra_body" not in params


async def test_json_object_is_forwarded(monkeypatch):
    body = {"metadata": {"user_email": "user@example.com"}}
    params = await _embed(monkeypatch, json.dumps(body))
    assert params["extra_body"] == body


async def test_invalid_extra_body_fails(monkeypatch):
    with pytest.raises(ValueError, match="JSON object"):
        await _embed(monkeypatch, "{")

    with pytest.raises(ValueError, match="JSON object"):
        await _embed(monkeypatch, '"a string"')
