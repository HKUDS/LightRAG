"""Offline tests for Anthropic token-limit truncation handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.anthropic import anthropic_complete_if_cache
from lightrag.utils import is_truncated_response

pytestmark = pytest.mark.offline


def _make_client(*, text: str, stop_reason: str) -> SimpleNamespace:
    response = SimpleNamespace(
        content=[SimpleNamespace(text=text)], stop_reason=stop_reason
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
