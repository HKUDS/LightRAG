"""Offline tests for Anthropic token-limit truncation handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm.anthropic import anthropic_complete_if_cache
from lightrag.utils import is_truncated_response

pytestmark = pytest.mark.offline


def _make_client(
    *, stop_reason: str, text: str | None = None, content: list | None = None
) -> SimpleNamespace:
    if content is None:
        content = [SimpleNamespace(type="text", text=text)]
    response = SimpleNamespace(content=content, stop_reason=stop_reason)
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
async def test_anthropic_truncation_stop_reason_marks_result_truncated(stop_reason):
    partial = '{"entities":[{"name":"Ali'
    client = _make_client(text=partial, stop_reason=stop_reason)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
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
        result = await anthropic_complete_if_cache(
            model="claude-test", prompt="Extract", api_key="test-key"
        )

    assert result == complete
    assert type(result) is str
    assert is_truncated_response(result) is False
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_joins_text_blocks_and_skips_thinking_blocks():
    """Extended thinking puts thinking blocks (no ``.text``) before and
    between the text blocks; every text block is joined, in order."""
    client = _make_client(
        content=[
            SimpleNamespace(type="thinking", thinking="step one"),
            SimpleNamespace(type="text", text="Part one. "),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="text", text="Part two."),
        ],
        stop_reason="end_turn",
    )

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
            model="claude-test", prompt="hi", api_key="test-key"
        )

    assert result == "Part one. Part two."
    assert type(result) is str
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking_block",
    [
        SimpleNamespace(type="thinking", thinking="x" * 50),
        SimpleNamespace(type="redacted_thinking", data="opaque"),
    ],
    ids=["thinking", "redacted_thinking"],
)
async def test_anthropic_thinking_only_max_tokens_raises_empty_truncated(
    thinking_block,
):
    """The whole output budget went into reasoning: the shared
    EmptyTruncatedResponseError names the budget knobs, unlike the
    InvalidResponseError raised for a non-length empty response."""
    client = _make_client(content=[thinking_block], stop_reason="max_tokens")

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(EmptyTruncatedResponseError) as excinfo:
            await anthropic_complete_if_cache(
                model="claude-test", prompt="hi", api_key="test-key"
            )

    message = str(excinfo.value)
    assert "budget consumed by reasoning" in message
    assert "max_tokens" in message and "thinking.budget_tokens" in message
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "content", "expected_hint", "absent_hint"),
    [
        (
            "model_context_window_exceeded",
            [],
            "shorten the input",
            "raising max_tokens",
        ),
        (
            "model_context_window_exceeded",
            [SimpleNamespace(type="text", text=" ")],
            "shorten the input",
            "raising max_tokens",
        ),
        (
            "max_tokens",
            [SimpleNamespace(type="text", text=" \n")],
            "raising max_tokens",
            "shorten the input",
        ),
    ],
    ids=["context-no-text-block", "context-blank-text", "max-tokens-blank-text"],
)
async def test_anthropic_blank_truncated_response_raises_with_remedy(
    stop_reason, content, expected_hint, absent_hint
):
    """A truncated response with no visible text is never returned as a
    blank ``TruncatedResponse``; the hint names the remedy for its limit."""
    client = _make_client(content=content, stop_reason=stop_reason)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(EmptyTruncatedResponseError) as excinfo:
            await anthropic_complete_if_cache(
                model="claude-test", prompt="Extract", api_key="test-key"
            )

    message = str(excinfo.value)
    assert f"stop_reason={stop_reason}" in message
    assert expected_hint in message
    assert absent_hint not in message
    client.close.assert_awaited_once()
