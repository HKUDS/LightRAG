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


@pytest.mark.asyncio
async def test_positional_third_arg_still_binds_max_token_size_not_embedding_dim():
    """embedding_dim must stay the last named parameter, after max_token_size,
    context, query_prefix and document_prefix. Those five parameters predate
    embedding_dim; a pre-existing positional caller like
    ollama_embed.func(texts, model, 4096) has always meant max_token_size=4096.
    Inserting embedding_dim earlier in the signature would silently reinterpret
    that same call as embedding_dim=4096 and forward an unwanted
    dimensions=4096 to Ollama."""
    fake_client = _make_fake_client({"embeddings": [[0.1, 0.2, 0.3]]})

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await ollama_embed.func(["hello"], "test-model", 4096)

    fake_client.embed.assert_awaited_once_with(
        model="test-model", input=["hello"], options={}
    )
