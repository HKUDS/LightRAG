"""Unit tests for the task-aware (asymmetric) embedding feature.

Covers:
  * ``wrap_embedding_func_with_attrs`` auto-detects ``supports_asymmetric``
    from the wrapped function's signature so users can't accidentally
    silently disable the feature by forgetting the flag.
  * ``EmbeddingFunc.__call__`` strips the ``context`` kwarg when the wrapped
    function does not declare ``supports_asymmetric=True`` (legacy back-compat).
  * ``jina_embed`` selects the right ``task`` from ``context`` when the caller
    leaves the new ``task=None`` default in place.
  * ``gemini_embed`` selects the right ``task_type`` from ``context``.

All tests are fully mocked; no live API calls.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# wrap_embedding_func_with_attrs auto-detection
# ---------------------------------------------------------------------------


def test_wrap_auto_detects_supports_asymmetric_when_context_present():
    """If the wrapped function takes ``context``, supports_asymmetric should be True."""
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=4, max_token_size=64)
    async def my_embed(texts, context="document"):
        return np.zeros((len(texts), 4), dtype=np.float32)

    assert my_embed.supports_asymmetric is True


def test_wrap_auto_detects_no_supports_asymmetric_for_legacy_func():
    """Legacy embed without ``context`` should default to supports_asymmetric=False."""
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=4, max_token_size=64)
    async def legacy_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    assert legacy_embed.supports_asymmetric is False


def test_wrap_explicit_supports_asymmetric_overrides_auto_detect():
    """Explicit kwarg must win over signature inspection."""
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=4, max_token_size=64, supports_asymmetric=False
    )
    async def my_embed(texts, context="document"):
        return np.zeros((len(texts), 4), dtype=np.float32)

    assert my_embed.supports_asymmetric is False


def test_wrap_auto_detects_per_function_when_decorator_reused():
    """Reusing a decorator must not share auto-detected support between functions."""
    from lightrag.utils import wrap_embedding_func_with_attrs

    decorator = wrap_embedding_func_with_attrs(embedding_dim=4, max_token_size=64)

    @decorator
    async def legacy_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    @decorator
    async def aware_embed(texts, context="document"):
        return np.zeros((len(texts), 4), dtype=np.float32)

    assert legacy_embed.supports_asymmetric is False
    assert aware_embed.supports_asymmetric is True


# ---------------------------------------------------------------------------
# EmbeddingFunc.__call__ strips context for legacy embeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_func_strips_context_for_legacy_func():
    """Legacy func that doesn't accept ``context`` must not see it (no TypeError)."""
    from lightrag.utils import EmbeddingFunc

    received_kwargs: list[dict] = []

    async def legacy_embed(texts):
        # If `context` were still in kwargs we'd never get here -- the call
        # would raise TypeError. So just record what we did receive.
        received_kwargs.append({"texts": texts})
        return np.zeros((len(texts), 4), dtype=np.float32)

    func = EmbeddingFunc(
        embedding_dim=4, max_token_size=64, supports_asymmetric=False, func=legacy_embed
    )
    out = await func(["a", "b"], context="query")
    assert out.shape == (2, 4)
    assert received_kwargs[0] == {"texts": ["a", "b"]}


@pytest.mark.asyncio
async def test_embedding_func_forwards_context_when_supported():
    from lightrag.utils import EmbeddingFunc

    received: list[str] = []

    async def aware_embed(texts, context="document"):
        received.append(context)
        return np.zeros((len(texts), 4), dtype=np.float32)

    func = EmbeddingFunc(
        embedding_dim=4, max_token_size=64, supports_asymmetric=True, func=aware_embed
    )
    await func(["a"], context="query")
    await func(["b"], context="document")
    assert received == ["query", "document"]


# ---------------------------------------------------------------------------
# jina_embed: task auto-selection from context
# ---------------------------------------------------------------------------


def _fake_jina_response(num: int, dim: int = 4) -> list[dict]:
    arr = np.zeros((num, dim), dtype=np.float32)
    return [
        {"embedding": base64.b64encode(arr[i].tobytes()).decode()} for i in range(num)
    ]


@pytest.mark.asyncio
async def test_jina_default_task_is_query_when_context_query(monkeypatch):
    """Default ``task=None`` + ``context='query'`` must produce ``retrieval.query``."""
    monkeypatch.setenv("JINA_API_KEY", "fake")
    from lightrag.llm import jina as jina_mod

    captured: list[dict] = []

    async def fake_fetch(url, headers, data):
        captured.append(data)
        return _fake_jina_response(len(data["input"]))

    with patch.object(jina_mod, "fetch_data", side_effect=fake_fetch):
        await jina_mod.jina_embed.func(texts=["q1"], context="query")
    assert captured[0]["task"] == "retrieval.query"


@pytest.mark.asyncio
async def test_jina_default_task_is_passage_when_context_document(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "fake")
    from lightrag.llm import jina as jina_mod

    captured: list[dict] = []

    async def fake_fetch(url, headers, data):
        captured.append(data)
        return _fake_jina_response(len(data["input"]))

    with patch.object(jina_mod, "fetch_data", side_effect=fake_fetch):
        await jina_mod.jina_embed.func(texts=["d1", "d2"], context="document")
    assert captured[0]["task"] == "retrieval.passage"


@pytest.mark.asyncio
async def test_jina_explicit_task_overrides_context(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "fake")
    from lightrag.llm import jina as jina_mod

    captured: list[dict] = []

    async def fake_fetch(url, headers, data):
        captured.append(data)
        return _fake_jina_response(len(data["input"]))

    with patch.object(jina_mod, "fetch_data", side_effect=fake_fetch):
        await jina_mod.jina_embed.func(
            texts=["x"], context="query", task="text-matching"
        )
    assert captured[0]["task"] == "text-matching"


# ---------------------------------------------------------------------------
# gemini_embed: task_type auto-selection from context
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_client_cache_cleared():
    """gemini.py caches its Client via lru_cache; clear it between tests."""
    pytest.importorskip("google.genai")
    from lightrag.llm import gemini as gemini_mod

    gemini_mod._get_gemini_client.cache_clear()
    yield
    gemini_mod._get_gemini_client.cache_clear()


@pytest.mark.asyncio
async def test_gemini_task_type_query_for_query_context(gemini_client_cache_cleared):
    pytest.importorskip("google.genai")
    from lightrag.llm import gemini as gemini_mod

    captured: list[dict] = []

    async def fake_embed_content(*, model, contents, config):
        captured.append({"task_type": getattr(config, "task_type", None)})
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.1] * 4) for _ in contents]
        return resp

    fake_client = MagicMock()
    fake_client.aio.models.embed_content = fake_embed_content

    with patch.object(gemini_mod.genai, "Client", return_value=fake_client):
        await gemini_mod.gemini_embed.func(
            texts=["q"], api_key="fake", context="query", task_type=None
        )
    assert captured[0]["task_type"] == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_gemini_task_type_document_for_document_context(
    gemini_client_cache_cleared,
):
    pytest.importorskip("google.genai")
    from lightrag.llm import gemini as gemini_mod

    captured: list[dict] = []

    async def fake_embed_content(*, model, contents, config):
        captured.append({"task_type": getattr(config, "task_type", None)})
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.1] * 4) for _ in contents]
        return resp

    fake_client = MagicMock()
    fake_client.aio.models.embed_content = fake_embed_content

    with patch.object(gemini_mod.genai, "Client", return_value=fake_client):
        await gemini_mod.gemini_embed.func(
            texts=["d"], api_key="fake", context="document", task_type=None
        )
    assert captured[0]["task_type"] == "RETRIEVAL_DOCUMENT"


@pytest.mark.asyncio
async def test_gemini_explicit_task_type_overrides_context(gemini_client_cache_cleared):
    pytest.importorskip("google.genai")
    from lightrag.llm import gemini as gemini_mod

    captured: list[dict] = []

    async def fake_embed_content(*, model, contents, config):
        captured.append({"task_type": getattr(config, "task_type", None)})
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.1] * 4) for _ in contents]
        return resp

    fake_client = MagicMock()
    fake_client.aio.models.embed_content = fake_embed_content

    with patch.object(gemini_mod.genai, "Client", return_value=fake_client):
        await gemini_mod.gemini_embed.func(
            texts=["x"],
            api_key="fake",
            context="query",
            task_type="CLASSIFICATION",
        )
    assert captured[0]["task_type"] == "CLASSIFICATION"
