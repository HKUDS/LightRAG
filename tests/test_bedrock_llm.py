from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.bedrock import (
    bedrock_complete,
    bedrock_complete_if_cache,
    bedrock_embed,
)


class _FakeBedrockClient:
    def __init__(self, captured_calls: list[dict]):
        self._captured_calls = captured_calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        self._captured_calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
                        }
                    ]
                }
            }
        }


class _FakeSession:
    def __init__(self, captured_calls: list[dict], client_kwargs_calls: list[dict]):
        self._captured_calls = captured_calls
        self._client_kwargs_calls = client_kwargs_calls

    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeBedrockClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_forwards_keyword_extraction_to_if_cache():
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "bedrock-model"})

    with patch(
        "lightrag.llm.bedrock.bedrock_complete_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await bedrock_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            keyword_extraction=True,
        )

    assert mocked_complete.await_args.kwargs["keyword_extraction"] is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_keyword_extraction_does_not_inject_system_prompt():
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        result = await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            response_format={"type": "json_object"},
        )

    assert result == '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    assert len(captured_calls) == 1
    assert "system" not in captured_calls[0]
    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_default_endpoint_sentinel_uses_sdk_default():
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            endpoint_url="DEFAULT_BEDROCK_ENDPOINT",
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_empty_endpoint_url_uses_sdk_default():
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            endpoint_url="",
        )

    assert client_kwargs_calls[-1] == {"region_name": None}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_custom_endpoint_url_is_forwarded():
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_complete_if_cache(
            model="bedrock-model",
            prompt="hello",
            endpoint_url="https://proxy.example.com",
        )

    assert client_kwargs_calls[-1] == {
        "region_name": None,
        "endpoint_url": "https://proxy.example.com",
    }


class _FakeEmbeddingBody:
    async def json(self):
        return {"embedding": [0.1] * 1024}


class _FakeEmbeddingResponse:
    def get(self, key):
        assert key == "body"
        return _FakeEmbeddingBody()


class _FakeEmbeddingClient(_FakeBedrockClient):
    async def invoke_model(self, **_kwargs):
        return _FakeEmbeddingResponse()


class _FakeEmbeddingSession(_FakeSession):
    def client(self, *_args, **kwargs):
        self._client_kwargs_calls.append(dict(kwargs))
        return _FakeEmbeddingClient(self._captured_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_embed_custom_endpoint_url_is_forwarded(monkeypatch):
    captured_calls: list[dict] = []
    client_kwargs_calls: list[dict] = []
    monkeypatch.delenv("AWS_REGION", raising=False)

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_FakeEmbeddingSession(captured_calls, client_kwargs_calls),
    ):
        await bedrock_embed(
            texts=["hello"],
            endpoint_url="https://proxy.example.com",
        )

    assert client_kwargs_calls[-1] == {
        "region_name": None,
        "endpoint_url": "https://proxy.example.com",
    }
