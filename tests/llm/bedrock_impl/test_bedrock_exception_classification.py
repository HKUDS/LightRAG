"""Regression test for Bedrock connection-closed exception classification.

aiobotocore's HTTP session translates a dropped-mid-request server
connection (``ServerDisconnectedError`` / ``aiohttp.ClientPayloadError`` /
a malformed status line) into ``botocore.exceptions.ConnectionClosedError``.
That exception does not subclass ``botocore.exceptions.ConnectionError``, so
``_handle_bedrock_exception`` must check for it explicitly or it falls
through to the generic, non-retryable ``BedrockError`` branch and the
``@retry`` decorator on ``bedrock_complete_if_cache`` never retries it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from botocore.exceptions import ConnectionClosedError

from lightrag.llm.bedrock import (
    BedrockConnectionError,
    BedrockError,
    _handle_bedrock_exception,
    bedrock_complete_if_cache,
)


@pytest.mark.offline
def test_connection_closed_error_is_classified_as_retryable():
    """A mid-stream server disconnect must map to BedrockConnectionError.

    Not to the generic, non-retryable BedrockError.
    """
    error = ConnectionClosedError(endpoint_url="https://bedrock-runtime.example.com")

    with pytest.raises(BedrockConnectionError):
        _handle_bedrock_exception(error, "Bedrock converse")


@pytest.mark.offline
def test_connection_closed_error_does_not_raise_generic_bedrock_error():
    error = ConnectionClosedError(endpoint_url="https://bedrock-runtime.example.com")

    try:
        _handle_bedrock_exception(error, "Bedrock converse")
    except BedrockConnectionError:
        pass
    except BedrockError as exc:
        pytest.fail(
            f"ConnectionClosedError must not surface as plain BedrockError: {exc}"
        )


class _FakeConnectionClosedClient:
    """Fake Bedrock client whose converse() mirrors a dropped connection."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        raise ConnectionClosedError(endpoint_url="https://bedrock-runtime.example.com")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bedrock_complete_if_cache_surfaces_connection_error(monkeypatch):
    """End to end: a dropped connection must reach callers as BedrockConnectionError."""
    monkeypatch.delenv("AWS_REGION", raising=False)

    fake_session = SimpleNamespace(
        client=lambda *_args, **_kwargs: _FakeConnectionClosedClient()
    )
    with patch("lightrag.llm.bedrock.aioboto3.Session", return_value=fake_session):
        with pytest.raises(BedrockConnectionError):
            # __wrapped__ bypasses the retry decorator so the test exercises
            # a single attempt instead of waiting out five retries.
            await bedrock_complete_if_cache.__wrapped__(
                model="bedrock-model", prompt="hello"
            )
