"""NVIDIA NIM embedding rows identify their positions in the input batch."""

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

from lightrag.llm import nvidia_openai

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.mark.parametrize("indices", [(1, 0), (0, 0)], ids=["shuffled", "duplicate"])
async def test_nvidia_embedding_input_order(monkeypatch, indices):
    def respond(request):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": [
                    {"object": "embedding", "index": index, "embedding": [float(index)]}
                    for index in indices
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://nvidia.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(nvidia_openai, "AsyncOpenAI", lambda **kwargs: client)
    if indices == (0, 0):
        with pytest.raises(ValueError, match="invalid embedding indices"):
            await nvidia_openai.nvidia_openai_embed.func(["first", "second"])
    else:
        vectors = await nvidia_openai.nvidia_openai_embed.func(["first", "second"])
        np.testing.assert_array_equal(vectors, [[0.0], [1.0]])
    assert client.is_closed()
