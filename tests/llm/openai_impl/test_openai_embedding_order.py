"""OpenAI-compatible embedding indices identify input positions."""

import json

import httpx
import numpy as np
import pytest

from lightrag.llm.openai import openai_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _row(index, value):
    row = {"object": "embedding", "embedding": [value]}
    if index is not None:
        row["index"] = index
    return row


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param([_row(1, 1.0), _row(0, 0.0)], [[0.0], [1.0]], id="shuffled"),
        # Endpoints that send no index at all keep their response order.
        pytest.param(
            [_row(None, 1.0), _row(None, 0.0)], [[1.0], [0.0]], id="unindexed"
        ),
        pytest.param([_row(0, 0.0), _row(0, 1.0)], None, id="duplicate"),
        pytest.param([_row(1, 1.0), _row(None, 0.0)], None, id="partial"),
        # True == 1, so only the type check rejects a boolean index.
        pytest.param([_row(True, 1.0), _row(0, 0.0)], None, id="non-int"),
    ],
)
async def test_openai_embedding_input_order(rows, expected):
    def respond(request):
        assert json.loads(request.content)["input"] == ["first", "second"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": rows,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        call = openai_embed.func(
            ["first", "second"],
            api_key="test-key",
            base_url="https://example.invalid/v1",
            client_configs={"http_client": client},
        )
        if expected is None:
            with pytest.raises(ValueError, match="invalid embedding indices"):
                await call
        else:
            np.testing.assert_array_equal(await call, expected)
