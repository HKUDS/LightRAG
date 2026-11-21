"""Regression tests for surfacing Bedrock errors after retry exhaustion."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from botocore.exceptions import ConnectionClosedError
from tenacity import RetryError, wait_none

from lightrag.llm.bedrock import (
    BedrockConnectionError,
    bedrock_complete_if_cache,
    bedrock_embed,
)

pytestmark = pytest.mark.offline

_ENDPOINT = "https://bedrock-runtime.example.com"


class _AlwaysDisconnectedCompletionClient:
    def __init__(self, calls: list[int]):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **_kwargs):
        self._calls.append(1)
        raise ConnectionClosedError(endpoint_url=_ENDPOINT)


class _EmbeddingBody:
    async def json(self):
        return {"embedding": [0.1] * 1024}


class _EmbeddingResponse:
    def get(self, key):
        assert key == "body"
        return _EmbeddingBody()


class _EmbeddingClient:
    def __init__(self, calls: list[int], failures: int):
        self._calls = calls
        self._failures = failures

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def invoke_model(self, **_kwargs):
        self._calls.append(1)
        if len(self._calls) <= self._failures:
            raise ConnectionClosedError(endpoint_url=_ENDPOINT)
        return _EmbeddingResponse()


@pytest.mark.asyncio
async def test_completion_reraises_bedrock_error_after_retry_exhaustion(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(bedrock_complete_if_cache.retry, "wait", wait_none())

    calls: list[int] = []
    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _AlwaysDisconnectedCompletionClient(calls)
    )

    with (
        patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session),
        pytest.raises(BedrockConnectionError) as exc_info,
    ):
        await bedrock_complete_if_cache(model="bedrock-model", prompt="hello")

    assert not isinstance(exc_info.value, RetryError)
    assert len(calls) == 5


@pytest.mark.asyncio
async def test_embedding_reraises_bedrock_error_after_retry_exhaustion(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(bedrock_embed.func.retry, "wait", wait_none())

    calls: list[int] = []
    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _EmbeddingClient(calls, failures=5)
    )

    with (
        patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session),
        pytest.raises(BedrockConnectionError) as exc_info,
    ):
        await bedrock_embed(texts=["hello"])

    assert not isinstance(exc_info.value, RetryError)
    assert len(calls) == 5


@pytest.mark.asyncio
async def test_embedding_returns_result_after_transient_retryable_error(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(bedrock_embed.func.retry, "wait", wait_none())

    calls: list[int] = []
    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _EmbeddingClient(calls, failures=1)
    )

    with patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session):
        result = await bedrock_embed(texts=["hello"])

    assert result.tolist() == [[0.1] * 1024]
    assert len(calls) == 2
