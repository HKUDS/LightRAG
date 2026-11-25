"""Offline tests: token-limit-truncated Ollama responses carry the marker.

Partial content is returned for salvage (flagged uncacheable); content that is
EMPTY because generation never got past the reasoning trace is a different
animal and raises, so the document ends FAILED and retryable instead of being
indexed as an empty knowledge graph (issue #3601 gap 4).

The raw-dict response shape mirrors ollama<0.4; ollama>=0.4 returns a
ChatResponse whose .get behaves identically (SubscriptableBaseModel).
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.ollama import InvalidResponseError, _ollama_model_if_cache
from lightrag.utils import TokenTracker, is_truncated_response

pytestmark = pytest.mark.offline


def _make_fake_client(response: dict):
    return SimpleNamespace(
        chat=AsyncMock(return_value=response),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


async def _complete(response: dict, **kwargs):
    fake_client = _make_fake_client(response)
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(
            model="test-model", prompt="Extract", **kwargs
        )
    return result, fake_client


async def test_ollama_length_done_reason_marks_partial_content_truncated():
    """done_reason == "length" (num_predict exhausted) flags the response,
    and non-empty content is still returned for salvage, never raised --
    unstripped, so the padding must survive."""
    raw_json = ' {"entities":[{"name":"Ali '

    result, _ = await _complete(
        {"message": {"content": raw_json}, "done_reason": "length"}
    )

    assert result == raw_json
    assert is_truncated_response(result) is True


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            {"message": {"content": '{"entities":[]}'}, "done_reason": "stop"},
            id="stop",
        ),
        # Older servers may omit done_reason; degrade to cache-everything.
        pytest.param({"message": {"content": "answer"}}, id="missing-done-reason"),
        # Only the token-limit case escalates: an empty response that ended
        # normally is the model's answer, and its callers keep tolerating it.
        pytest.param(
            {"message": {"content": ""}, "done_reason": "stop"}, id="empty-stop"
        ),
    ],
)
async def test_ollama_non_length_response_is_returned_unmarked(response):
    result, _ = await _complete(response)

    assert result == response["message"]["content"]
    assert is_truncated_response(result) is False


@pytest.mark.parametrize("content", ["", "   \n  "])
async def test_ollama_empty_length_truncated_response_raises(
    caplog, lightrag_caplog, content
):
    """Empty AND cut off is structurally broken, not merely short.

    This is the #3597 shape: a thinking model spends the whole num_predict
    budget on its reasoning trace and emits no content. Returning "" here left
    the document PROCESSED with an empty graph; it must fail instead.
    """
    fake_client = _make_fake_client(
        {
            "message": {"content": content},
            "done_reason": "length",
            "eval_count": 64,
            "prompt_eval_count": 900,
        }
    )

    with caplog.at_level(logging.ERROR, logger="lightrag"):
        with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
            with pytest.raises(InvalidResponseError) as excinfo:
                await _ollama_model_if_cache(model="test-model", prompt="Extract")

    message = str(excinfo.value)
    assert "Received empty content from Ollama API" in message
    assert "done_reason=length" in message
    assert "eval_count=64" in message
    # The actionable part must ride along into doc_status.error_msg, not stay
    # behind in the server log.
    assert "OLLAMA_LLM_NUM_PREDICT" in message
    assert "hit the token limit" in caplog.text
    fake_client._client.aclose.assert_awaited()


async def test_ollama_empty_length_response_names_reasoning_and_is_billed():
    """``message.thinking`` present ⇒ say the budget went to reasoning.

    Usage is counted before the raise: the budget was spent whether or not
    the output was usable, and this is exactly the case that spent it."""
    tracker = TokenTracker()

    with pytest.raises(InvalidResponseError) as excinfo:
        await _complete(
            {
                "message": {"content": "", "thinking": "let me carefully consider..."},
                "done_reason": "length",
                "prompt_eval_count": 12,
                "eval_count": 400,
            },
            token_tracker=tracker,
        )

    message = str(excinfo.value)
    assert "budget consumed by reasoning" in message
    assert "thinking_len=28" in message
    assert tracker.get_usage()["total_tokens"] == 412
