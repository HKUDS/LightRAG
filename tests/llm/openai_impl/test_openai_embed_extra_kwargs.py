"""``openai_embed`` must forward unrecognized kwargs to the embeddings API call.

Regression for a gap found wiring up ``QueryParam.extra_embedding_kwargs``:
``openai_complete_if_cache`` (the LLM binding) already accepts and forwards
arbitrary ``**kwargs`` to the underlying API call, but ``openai_embed`` had a
closed parameter list and raised ``TypeError`` on anything it did not name
explicitly -- so a caller-supplied kwarg such as ``extra_headers`` never
reached the embeddings API at all.
"""

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


@pytest.mark.asyncio
async def test_extra_kwargs_reach_the_embeddings_api_call(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )

    await openai_embed.func(
        ["hello"],
        model=MODEL,
        api_key="test-key",
        extra_headers={"X-User-Id": "u-1"},
    )

    assert captured[0]["extra_headers"] == {"X-User-Id": "u-1"}


@pytest.mark.asyncio
async def test_no_extra_kwargs_is_unaffected(monkeypatch):
    """Baseline: the default call shape is unchanged for existing callers."""
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )

    await openai_embed.func(["hello"], model=MODEL, api_key="test-key")

    assert set(captured[0]) == {"model", "input", "encoding_format"}
