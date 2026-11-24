"""Offline tests: ollama_embed forwards embedding_dim to Ollama's embed API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.llm.ollama import ollama_embed

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("args", "kwargs", "expected_extra"),
    [
        pytest.param(
            (),
            {"embed_model": "test-model", "embedding_dim": 256},
            {"dimensions": 256},
            id="explicit-dim-forwarded",
        ),
        # Unset embedding_dim adds no dimensions kwarg at all -- not even
        # dimensions=None -- so Ollama returns the model's default.
        pytest.param((), {"embed_model": "test-model"}, {}, id="unset-dim-omitted"),
        # embedding_dim must stay the last named parameter: a pre-existing
        # positional caller ollama_embed.func(texts, model, 4096) has always
        # meant max_token_size=4096, and inserting embedding_dim earlier would
        # silently forward dimensions=4096 instead.
        pytest.param(
            ("test-model", 4096), {}, {}, id="positional-third-arg-is-max-token-size"
        ),
    ],
)
async def test_embedding_dim_forwarding(args, kwargs, expected_extra):
    fake_client = SimpleNamespace(
        embed=AsyncMock(return_value={"embeddings": [[0.1, 0.2, 0.3]]}),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], *args, **kwargs)

    fake_client.embed.assert_awaited_once_with(
        model="test-model", input=["hello"], options={}, **expected_extra
    )
    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
