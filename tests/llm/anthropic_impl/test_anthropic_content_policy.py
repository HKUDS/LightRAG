"""Offline tests for Anthropic refusal verdicts.

``stop_reason="refusal"`` is the model declining on safety grounds, not an
empty-response sampling artifact. It raises the non-retryable
ContentPolicyViolationError on both the non-streaming and streaming paths;
the SDK never retries a successful response, so each path fails after one
request, and any partial text is never returned as a complete answer.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.exceptions import ContentPolicyViolationError
from lightrag.llm.anthropic import anthropic_complete_if_cache

pytestmark = pytest.mark.offline


def _make_client(payload) -> SimpleNamespace:
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=payload)),
        close=AsyncMock(),
    )


class _Stream:
    def __init__(self, events):
        self._events = iter(events)
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        pytest.param([], id="no-text-block"),
        pytest.param(
            [SimpleNamespace(type="text", text="flagged partial answer")],
            id="partial-text",
        ),
    ],
)
async def test_refusal_stop_raises_content_policy_error(content):
    response = SimpleNamespace(content=content, stop_reason="refusal")
    client = _make_client(response)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            await anthropic_complete_if_cache(
                model="claude-test", prompt="hello", api_key="test-key"
            )

    message = str(excinfo.value)
    assert "stop_reason=refusal" in message
    assert "flagged partial answer" not in message
    assert client.messages.create.await_count == 1
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_refusal_raises_after_partial_text():
    """``message_delta`` carries the terminal stop_reason; a refusal must
    reach the consumer as an error, not as a completed stream."""
    stream = _Stream(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="partial"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="refusal"),
                usage=SimpleNamespace(output_tokens=1),
            ),
            SimpleNamespace(type="message_stop"),
        ]
    )
    client = _make_client(stream)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
            model="claude-test", prompt="hello", api_key="test-key", stream=True
        )
        chunks = []
        with pytest.raises(ContentPolicyViolationError) as excinfo:
            async for chunk in result:
                chunks.append(chunk)

    assert chunks == ["partial"]
    assert "stop_reason=refusal" in str(excinfo.value)
    assert client.messages.create.await_count == 1
    stream.close.assert_awaited_once()
    client.close.assert_awaited_once()
