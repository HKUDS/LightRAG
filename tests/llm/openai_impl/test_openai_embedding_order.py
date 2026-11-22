"""OpenAI-compatible embedding indices identify input positions."""

import json

import httpx
import numpy as np
import pytest

from lightrag.llm.openai import openai_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.mark.parametrize("indices", [(1, 0), (0, 0)], ids=["shuffled", "duplicate"])
async def test_openai_embedding_input_order(indices):
    def respond(request):
        assert json.loads(request.content)["input"] == ["first", "second"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(i)]}
                    for i in indices
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def embed():
            return await openai_embed.func(
                ["first", "second"],
                api_key="test-key",
                base_url="https://example.invalid/v1",
                client_configs={"http_client": client},
            )

        if indices == (0, 0):
            with pytest.raises(ValueError, match="embedding indices"):
                await embed()
        else:
            np.testing.assert_array_equal(await embed(), [[0.0], [1.0]])
