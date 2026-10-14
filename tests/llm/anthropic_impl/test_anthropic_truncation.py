"""Offline tests for Anthropic token-limit truncation handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm.anthropic import InvalidResponseError, anthropic_complete_if_cache
from lightrag.utils import is_truncated_response

pytestmark = pytest.mark.offline


def _make_client(*, text: str, stop_reason: str) -> SimpleNamespace:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
    )
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_anthropic_max_tokens_stop_reason_marks_result_truncated():
    partial = '{"entities":[{"name":"Ali'
    client = _make_client(text=partial, stop_reason="max_tokens")

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-test", prompt="Extract", api_key="test-key"
        )

    assert result == partial
    assert is_truncated_response(result) is True
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_end_turn_stop_reason_remains_plain_string():
    complete = '{"entities":[]}'
    client = _make_client(text=complete, stop_reason="end_turn")

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-test", prompt="Extract", api_key="test-key"
        )

    assert result == complete
    assert type(result) is str
    assert is_truncated_response(result) is False
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_extracts_text_after_a_leading_thinking_block():
    """Extended thinking puts a ThinkingBlock (no .text attribute) before the
    TextBlock. content[0].text must not be assumed to be the answer."""
    thinking_block = SimpleNamespace(type="thinking", thinking="reasoning...")
    text_block = SimpleNamespace(type="text", text="final answer")
    response = SimpleNamespace(
        content=[thinking_block, text_block], stop_reason="end_turn"
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-test", prompt="hi", api_key="test-key"
        )

    assert result == "final answer"
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_joins_interleaved_text_blocks():
    """Interleaved thinking can produce [thinking, text, thinking, text]:
    every text block must be joined, not just the first one found."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="step one"),
            SimpleNamespace(type="text", text="Part one. "),
            SimpleNamespace(type="thinking", thinking="step two"),
            SimpleNamespace(type="text", text="Part two."),
        ],
        stop_reason="end_turn",
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-test", prompt="hi", api_key="test-key"
        )

    assert result == "Part one. Part two."


@pytest.mark.asyncio
async def test_anthropic_thinking_only_max_tokens_raises_empty_truncated():
    """A thinking model can spend the whole output budget on the reasoning
    trace and never reach a text block. This is deterministic for the given
    prompt/budget, so it must raise the non-retryable
    EmptyTruncatedResponseError, matching the OpenAI/Gemini bindings --
    not the retryable InvalidResponseError, which would buy two more
    full-budget generations plus backoff for an identically empty result."""
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="x" * 50)],
        stop_reason="max_tokens",
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(EmptyTruncatedResponseError):
            await anthropic_complete_if_cache.__wrapped__(
                model="claude-test", prompt="hi", api_key="test-key"
            )


@pytest.mark.asyncio
async def test_anthropic_no_text_block_non_length_reason_stays_retryable():
    """An empty response NOT caused by hitting the token limit is a sampling
    artifact a fresh attempt can genuinely fix, so it keeps the retryable
    InvalidResponseError."""
    response = SimpleNamespace(content=[], stop_reason="end_turn")
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(InvalidResponseError):
            await anthropic_complete_if_cache.__wrapped__(
                model="claude-test", prompt="hi", api_key="test-key"
            )
