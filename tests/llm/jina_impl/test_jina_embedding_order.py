"""Jina embedding rows must remain aligned with the input text batch."""

import base64
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.llm import jina

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture
def serve_embeddings(monkeypatch):
    def serve(rows):
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={"data": rows})
        session = MagicMock()
        session.post.return_value.__aenter__.return_value = response
        factory = MagicMock()
        factory.return_value.__aenter__.return_value = session
        monkeypatch.setattr(jina.aiohttp, "ClientSession", factory)
        return session.post

    return serve


def embedding(value, **metadata):
    encoded = base64.b64encode(np.asarray([value, 0], dtype=np.float32)).decode()
    return {"embedding": encoded, **metadata}


@pytest.mark.parametrize("order", ["indexed", "reversed", "unindexed"])
async def test_embeddings_follow_input_order(serve_embeddings, order):
    rows = [embedding(1, index=0), embedding(2, index=1)]
    if order == "reversed":
        rows.reverse()
    elif order == "unindexed":
        for row in rows:
            del row["index"]
    post = serve_embeddings(rows)

    result = await jina.jina_embed.func(
        ["first", "second"], api_key="test-key", embedding_dim=2
    )

    np.testing.assert_array_equal(result, [[1, 0], [2, 0]])
    assert post.call_args.kwargs["json"]["input"] == ["first", "second"]


@pytest.mark.parametrize(
    "second_index",
    [
        pytest.param(0, id="duplicate"),
        pytest.param(2, id="out-of-range"),
        pytest.param(-1, id="negative"),
        pytest.param("1", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="missing"),
    ],
)
async def test_invalid_indices_do_not_return_misaligned_vectors(
    serve_embeddings, second_index
):
    second = embedding(2) if second_index is None else embedding(2, index=second_index)
    serve_embeddings([embedding(1, index=0), second])

    with pytest.raises(ValueError, match="embedding indices"):
        await jina.jina_embed.func(
            ["first", "second"], api_key="test-key", embedding_dim=2
        )
