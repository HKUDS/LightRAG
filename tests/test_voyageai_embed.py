"""Unit tests for lightrag.llm.voyageai.

These tests mock voyageai.AsyncClient so they run fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_voyage_response():
    """Build a fake VoyageAI embed response with N rows of fixed-dim vectors."""

    def _make(n: int, dim: int = 8) -> MagicMock:
        rng = np.linspace(0.0, 1.0, num=dim, dtype=np.float32)
        rows = [rng.tolist() for _ in range(n)]
        resp = MagicMock()
        resp.embeddings = rows
        return resp

    return _make


@pytest.fixture
def patched_async_client(fake_voyage_response):
    """Patch voyageai.AsyncClient so each call returns a recorded response."""
    captured: list[dict] = []

    async def fake_embed(**kwargs):
        captured.append(kwargs)
        return fake_voyage_response(len(kwargs["texts"]))

    fake_client = MagicMock()
    fake_client.embed = fake_embed

    with patch(
        "lightrag.llm.voyageai.voyageai.AsyncClient", return_value=fake_client
    ) as m:
        yield captured, m


@pytest.mark.asyncio
async def test_voyageai_embed_passes_model(patched_async_client):
    """The function should forward the model parameter to the SDK."""
    captured, _ = patched_async_client
    from lightrag.llm.voyageai import voyageai_embed

    out = await voyageai_embed.func(
        texts=["hello", "world"], model="voyage-3-lite", api_key="fake"
    )

    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 2
    assert len(captured) == 1
    assert captured[0]["model"] == "voyage-3-lite"


@pytest.mark.asyncio
async def test_voyageai_embed_accepts_legacy_voyage_api_key(
    patched_async_client, monkeypatch
):
    """Setting only VOYAGE_API_KEY (the SDK's name) must work for backward compat."""
    captured, _ = patched_async_client
    monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "key-from-legacy-name")

    from lightrag.llm.voyageai import voyageai_embed

    await voyageai_embed.func(texts=["x"], model="voyage-3")
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_voyageai_embed_accepts_voyageai_api_key(
    patched_async_client, monkeypatch
):
    """The newer VOYAGEAI_API_KEY name must also still work."""
    captured, _ = patched_async_client
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("VOYAGEAI_API_KEY", "key-from-new-name")

    from lightrag.llm.voyageai import voyageai_embed

    await voyageai_embed.func(texts=["x"], model="voyage-3")
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_voyageai_embed_raises_when_no_api_key(monkeypatch):
    """Without any API key configured the call should raise ValueError."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)

    from lightrag.llm.voyageai import voyageai_embed

    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        await voyageai_embed.func(texts=["x"])


@pytest.mark.asyncio
async def test_voyageai_embed_forwards_input_type(patched_async_client):
    """input_type kwarg must reach the SDK so callers can drive query/document selection."""
    captured, _ = patched_async_client
    from lightrag.llm.voyageai import voyageai_embed

    await voyageai_embed.func(texts=["q"], api_key="fake", input_type="query")
    await voyageai_embed.func(texts=["d"], api_key="fake", input_type="document")
    assert captured[0]["input_type"] == "query"
    assert captured[1]["input_type"] == "document"


def test_anthropic_embed_deprecation_shim():
    """``anthropic_embed`` must remain importable and emit DeprecationWarning."""
    import warnings

    from lightrag.llm.anthropic import anthropic_embed  # must not ImportError

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coro = anthropic_embed(texts=["x"], api_key="ignored-mock")
        # Close the coroutine to silence "never awaited" runtime warnings;
        # we only care that the deprecation warning fired at call time.
        if hasattr(coro, "close"):
            coro.close()

    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "anthropic_embed should warn DeprecationWarning"
