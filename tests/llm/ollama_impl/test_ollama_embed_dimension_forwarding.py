"""Offline tests: ollama_embed forwards embedding_dim to Ollama's embed API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.llm.ollama import ollama_embed

pytestmark = pytest.mark.offline


def _make_fake_client(response: dict):
    return SimpleNamespace(
        embed=AsyncMock(return_value=response),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_explicit_embedding_dim_is_forwarded_as_dimensions():
    fake_client = _make_fake_client({"embeddings": [[0.1, 0.2, 0.3]]})

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(
            ["hello"], embed_model="test-model", embedding_dim=256
        )

    fake_client.embed.assert_awaited_once_with(
        model="test-model", input=["hello"], options={}, dimensions=256
    )
    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))


@pytest.mark.asyncio
async def test_no_embedding_dim_means_no_dimensions_kwarg():
    """Unset embedding_dim shouldn't add a dimensions kwarg at all -- not
    even dimensions=None -- so Ollama just returns the model's default."""
    fake_client = _make_fake_client({"embeddings": [[0.1, 0.2, 0.3]]})

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await ollama_embed.func(["hello"], embed_model="test-model")

    fake_client.embed.assert_awaited_once_with(
        model="test-model", input=["hello"], options={}
    )
