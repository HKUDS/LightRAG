"""Offline tests for Bedrock token usage tracking."""

from unittest.mock import patch

import pytest

from lightrag.llm.bedrock import bedrock_complete_if_cache
from lightrag.utils import TokenTracker

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _EventStream:
    def __init__(self, events):
        self._events = iter(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def close(self):
        self.closed = True


class _Client:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def converse(self, **kwargs):
        self.request = kwargs
        return self.response

    async def converse_stream(self, **kwargs):
        self.request = kwargs
        return self.response


class _Session:
    def __init__(self, client):
        self.client_instance = client

    def client(self, *_args, **_kwargs):
        return self.client_instance


async def test_non_streaming_usage_is_tracked_without_reaching_sdk():
    tracker = TokenTracker()
    client = _Client(
        {
            "output": {"message": {"content": [{"text": "answer"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
        }
    )

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_Session(client),
    ):
        result = await bedrock_complete_if_cache.__wrapped__(
            model="bedrock-model",
            prompt="hello",
            token_tracker=tracker,
        )

    assert result == "answer"
    assert tracker.get_usage() == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "call_count": 1,
    }
    assert "token_tracker" not in client.request


async def test_streaming_usage_is_tracked_after_message_stop():
    tracker = TokenTracker()
    event_stream = _EventStream(
        [
            {"contentBlockDelta": {"delta": {"text": "answer"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 8,
                        "outputTokens": 3,
                        "totalTokens": 11,
                    }
                }
            },
        ]
    )
    client = _Client({"stream": event_stream})

    with patch(
        "lightrag.llm.bedrock.aioboto3.Session",
        return_value=_Session(client),
    ):
        result = await bedrock_complete_if_cache.__wrapped__(
            model="bedrock-model",
            prompt="hello",
            stream=True,
            token_tracker=tracker,
        )
        chunks = [chunk async for chunk in result]

    assert chunks == ["answer"]
    assert tracker.get_usage() == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
        "call_count": 1,
    }
    assert "token_tracker" not in client.request
    assert event_stream.closed is True
