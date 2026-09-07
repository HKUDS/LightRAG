"""Offline tests for Anthropic token usage tracking."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.anthropic import anthropic_complete_if_cache
from lightrag.utils import TokenTracker

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _usage(*, input_tokens, output_tokens, cache_creation=None, cache_read=None):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
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


async def test_non_streaming_usage_is_tracked_without_reaching_sdk():
    tracker = TokenTracker()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="answer")],
        stop_reason="end_turn",
        usage=_usage(
            input_tokens=10,
            output_tokens=4,
            cache_creation=3,
            cache_read=2,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-test",
            prompt="hello",
            api_key="test-key",
            token_tracker=tracker,
        )

    assert result == "answer"
    assert tracker.get_usage() == {
        "prompt_tokens": 15,
        "completion_tokens": 4,
        "total_tokens": 19,
        "call_count": 1,
    }
    assert "token_tracker" not in create.await_args.kwargs


async def test_image_inputs_keeps_its_position_ahead_of_token_tracker():
    """Codex finding: token_tracker must not shift image_inputs out of the
    position an existing positional caller relies on. Confirms the current
    parameter order rather than a specific one, so a future reordering that
    reintroduces the hazard fails this test."""
    import inspect

    params = list(inspect.signature(anthropic_complete_if_cache.__wrapped__).parameters)
    assert params.index("image_inputs") < params.index("token_tracker")


async def test_usage_recorded_even_when_content_extraction_fails():
    """Codex finding: response.usage is already populated once the API call
    returns, so a real, billed call's usage must be recorded even if
    response.content[0].text later raises (e.g. an empty or non-text content
    list)."""
    tracker = TokenTracker()
    response = SimpleNamespace(
        content=[],  # empty -- content[0] raises IndexError
        stop_reason="end_turn",
        usage=_usage(input_tokens=10, output_tokens=4),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(IndexError):
            await anthropic_complete_if_cache.__wrapped__(
                model="claude-test",
                prompt="hello",
                api_key="test-key",
                token_tracker=tracker,
            )

    assert tracker.get_usage() == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "call_count": 1,
    }


async def test_streaming_usage_is_tracked_after_full_consumption():
    tracker = TokenTracker()
    stream = _Stream(
        [
            SimpleNamespace(
                message=SimpleNamespace(
                    usage=_usage(
                        input_tokens=8,
                        output_tokens=1,
                        cache_creation=2,
                        cache_read=3,
                    )
                )
            ),
            SimpleNamespace(delta=SimpleNamespace(text="answer")),
            SimpleNamespace(
                delta=SimpleNamespace(),
                usage=_usage(input_tokens=None, output_tokens=5),
            ),
        ]
    )
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-test",
            prompt="hello",
            api_key="test-key",
            stream=True,
            token_tracker=tracker,
        )
        chunks = [chunk async for chunk in result]

    assert chunks == ["answer"]
    assert tracker.get_usage() == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
        "call_count": 1,
    }
    assert "token_tracker" not in create.await_args.kwargs
    stream.close.assert_awaited_once()
