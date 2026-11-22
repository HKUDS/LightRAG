"""Regression test for Bedrock connection-closed exception classification.

aiobotocore's HTTP session translates a dropped-mid-request server
connection (``ServerDisconnectedError`` / ``aiohttp.ClientPayloadError`` /
a malformed status line) into ``botocore.exceptions.ConnectionClosedError``.
That exception does not subclass ``botocore.exceptions.ConnectionError``, so
``_handle_bedrock_exception`` must check for it explicitly or it falls
through to the generic, non-retryable ``BedrockError`` branch and the
``@retry`` decorator on ``bedrock_complete_if_cache`` never retries it.

Once the retries are exhausted, both retrying entry points must raise the
classified ``Bedrock*`` error itself, never a ``tenacity.RetryError``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from botocore.exceptions import ConnectionClosedError
from tenacity import wait_none

from lightrag.llm.bedrock import (
    BedrockConnectionError,
    bedrock_complete_if_cache,
    bedrock_embed,
)

pytestmark = pytest.mark.offline

_ENDPOINT = "https://bedrock-runtime.example.com"


class _FlakyConnectionClient:
    """Fake Bedrock client whose converse() drops the connection N times."""

    def __init__(self, calls: list[int], failures: int):
        self._calls = calls
        self._failures = failures

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        self._calls.append(1)
        if len(self._calls) <= self._failures:
            raise ConnectionClosedError(endpoint_url=_ENDPOINT)
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {},
        }


@pytest.mark.asyncio
async def test_bedrock_complete_if_cache_retries_dropped_connection(monkeypatch):
    """End to end through @retry: a dropped connection is retried, not fatal."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(bedrock_complete_if_cache.retry, "wait", wait_none())

    calls: list[int] = []
    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _FlakyConnectionClient(calls, failures=2)
    )
    with patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session):
        result = await bedrock_complete_if_cache(model="bedrock-model", prompt="hello")

    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_bedrock_complete_if_cache_reraises_after_retry_exhaustion(monkeypatch):
    """Exhausted retries surface the BedrockConnectionError, not RetryError."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(bedrock_complete_if_cache.retry, "wait", wait_none())

    calls: list[int] = []
    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _FlakyConnectionClient(calls, failures=5)
    )
    with (
        patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session),
        pytest.raises(BedrockConnectionError),
    ):
        await bedrock_complete_if_cache(model="bedrock-model", prompt="hello")

    assert len(calls) == 5


class _DisconnectedEmbeddingClient:
    """Fake Bedrock client whose invoke_model() always drops the connection."""

    def __init__(self, calls: list[int]):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def invoke_model(self, **kwargs):
        self._calls.append(1)
        raise ConnectionClosedError(endpoint_url=_ENDPOINT)


@pytest.mark.asyncio
async def test_bedrock_embed_retries_then_reraises_dropped_connection(monkeypatch):
    """bedrock_embed retries a dropped connection, then raises it unwrapped."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(bedrock_embed.func.retry, "wait", wait_none())

    calls: list[int] = []
    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _DisconnectedEmbeddingClient(calls)
    )
    with (
        patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session),
        pytest.raises(BedrockConnectionError),
    ):
        await bedrock_embed(texts=["hello"])

    assert len(calls) == 5
