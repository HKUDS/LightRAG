from __future__ import annotations

from unittest.mock import patch
import httpx
import numpy as np
import pytest

from lightrag.llm.cloudflare import (
    cloudflare_complete_if_cache,
    cloudflare_embed,
    CloudflareAPIError,
    InvalidResponseError,
)
from lightrag.llm.binding_options import (
    CloudflareLLMOptions,
    CloudflareEmbeddingOptions,
)


@pytest.mark.asyncio
async def test_cloudflare_complete_non_stream():
    fake_response = {
        "success": True,
        "result": {
            "response": "Hello from Cloudflare Workers AI!"
        },
        "errors": [],
        "messages": [],
    }

    async def fake_post(url, **kwargs):
        assert "accounts/fake-account-id/ai/run/@cf/meta/llama-3.3-70b-instruct" in url
        assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
        req_json = kwargs["json"]
        assert req_json["messages"][-1]["content"] == "Hello"
        assert req_json["stream"] is False
        assert req_json["temperature"] == 0.7
        resp = httpx.Response(
            status_code=200,
            json=fake_response,
            request=httpx.Request("POST", url),
        )
        return resp

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await cloudflare_complete_if_cache(
            model="@cf/meta/llama-3.3-70b-instruct",
            prompt="Hello",
            system_prompt="You are a helpful assistant",
            api_key="fake-token",
            account_id="fake-account-id",
            temperature=0.7,
        )
        assert result == "Hello from Cloudflare Workers AI!"


@pytest.mark.asyncio
async def test_cloudflare_complete_stream():
    stream_lines = [
        b'data: {"response": "Hello "}\n\n',
        b'data: {"response": "world!"}\n\n',
        b'data: [DONE]\n\n',
    ]

    class FakeStreamResponse:
        status_code = 200
        request = httpx.Request("POST", "https://fake.url")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_lines(self):
            for line in stream_lines:
                yield line.decode("utf-8")

    def fake_stream(method, url, **kwargs):
        return FakeStreamResponse()

    with patch("httpx.AsyncClient.stream", side_effect=fake_stream):
        stream_gen = await cloudflare_complete_if_cache(
            model="@cf/meta/llama-3.3-70b-instruct",
            prompt="Hi",
            api_key="fake-token",
            account_id="fake-account-id",
            stream=True,
        )
        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)
        assert "".join(chunks) == "Hello world!"


@pytest.mark.asyncio
async def test_cloudflare_missing_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)

    with pytest.raises(ValueError, match="Cloudflare API token is required"):
        await cloudflare_complete_if_cache(prompt="Hi")

    with pytest.raises(ValueError, match="Cloudflare Account ID is required"):
        await cloudflare_complete_if_cache(prompt="Hi", api_key="test-token")


@pytest.mark.asyncio
async def test_cloudflare_embed():
    fake_data = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    fake_response = {
        "success": True,
        "result": {
            "shape": [2, 4],
            "data": fake_data,
        },
        "errors": [],
    }

    async def fake_post(url, **kwargs):
        assert kwargs["json"]["text"] == ["doc1", "doc2"]
        return httpx.Response(
            status_code=200,
            json=fake_response,
            request=httpx.Request("POST", url),
        )

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        embeddings = await cloudflare_embed.func(
            texts=["doc1", "doc2"],
            model="@cf/baai/bge-m3",
            api_key="fake-token",
            account_id="fake-account-id",
        )
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 4)
        assert np.allclose(embeddings[0], [0.1, 0.2, 0.3, 0.4])


@pytest.mark.asyncio
async def test_cloudflare_embed_empty():
    res = await cloudflare_embed.func(texts=[])
    assert isinstance(res, np.ndarray)
    assert res.shape == (0, 1024)


def test_cloudflare_binding_options():
    opt = CloudflareLLMOptions(
        temperature=0.5,
        max_tokens=1000,
        top_p=0.9,
    )
    d = opt.asdict()
    assert d["temperature"] == 0.5
    assert d["max_tokens"] == 1000
    assert d["top_p"] == 0.9

    emb_opt = CloudflareEmbeddingOptions()
    assert emb_opt._binding_name == "cloudflare_embedding"


@pytest.mark.asyncio
async def test_cloudflare_api_error_response():
    fake_error_response = {
        "success": False,
        "errors": [{"code": 1000, "message": "Model not found"}],
    }

    async def fake_post(url, **kwargs):
        return httpx.Response(
            status_code=200,
            json=fake_error_response,
            request=httpx.Request("POST", url),
        )

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        with pytest.raises(CloudflareAPIError, match="Model not found"):
            await cloudflare_complete_if_cache(
                prompt="Hi",
                api_key="fake-token",
                account_id="fake-account-id",
            )


@pytest.mark.asyncio
async def test_cloudflare_invalid_response():
    fake_empty_response = {
        "success": True,
        "result": {},
        "errors": [],
    }

    async def fake_post(url, **kwargs):
        return httpx.Response(
            status_code=200,
            json=fake_empty_response,
            request=httpx.Request("POST", url),
        )

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        with pytest.raises(InvalidResponseError, match="Empty response"):
            await cloudflare_complete_if_cache(
                prompt="Hi",
                api_key="fake-token",
                account_id="fake-account-id",
            )

