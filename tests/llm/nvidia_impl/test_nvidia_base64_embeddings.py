"""Exercise NVIDIA response decoding through the real SDK without network I/O."""

import base64
import json

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

from lightrag.llm import nvidia_openai

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def mock_client(monkeypatch, vectors: np.ndarray, response_format: str):
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "nvidia/llama-3.2-nv-embedqa-1b-v1",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": (
                            base64.b64encode(vector.tobytes()).decode("ascii")
                            if response_format == "base64"
                            else vector.tolist()
                        ),
                    }
                    for index, vector in enumerate(vectors)
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://nvidia.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(nvidia_openai, "AsyncOpenAI", lambda **kwargs: client)
    return client, requests


@pytest.mark.parametrize(
    ("encode", "response_format"),
    [("float", "float"), ("base64", "base64"), ("base64", "float")],
)
@pytest.mark.parametrize("batch_size", [1, 2])
async def test_embedding_encoding_preserves_numeric_vectors(
    monkeypatch, encode, response_format, batch_size
):
    expected = np.arange(batch_size * 2048, dtype=np.float32).reshape(batch_size, 2048)
    expected = (expected - 1024) / 4096
    client, requests = mock_client(monkeypatch, expected, response_format)
    texts = [f"document {index}" for index in range(batch_size)]

    result = await nvidia_openai.nvidia_openai_embed(texts, encode=encode)

    assert result.shape == (batch_size, 2048)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_equal(result, expected)
    assert requests[0]["input"] == texts
    assert requests[0]["encoding_format"] == encode
    assert client.is_closed()


async def test_base64_decoding_keeps_embedding_dimension_validation(monkeypatch):
    client, _ = mock_client(monkeypatch, np.zeros((1, 3), dtype=np.float32), "base64")

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await nvidia_openai.nvidia_openai_embed(["document"], encode="base64")

    assert client.is_closed()


async def test_base64_decoding_keeps_vector_count_validation(monkeypatch):
    client, _ = mock_client(
        monkeypatch, np.zeros((1, 2048), dtype=np.float32), "base64"
    )

    with pytest.raises(ValueError, match="Vector count mismatch"):
        await nvidia_openai.nvidia_openai_embed(["first", "second"], encode="base64")

    assert client.is_closed()
