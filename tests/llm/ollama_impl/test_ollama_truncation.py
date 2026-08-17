"""Offline tests: token-limit-truncated Ollama responses carry the marker.

Partial content is returned for salvage (flagged uncacheable); content that is
EMPTY because generation never got past the reasoning trace is a different
animal and raises, so the document ends FAILED and retryable instead of being
indexed as an empty knowledge graph (issue #3601 gap 4).
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.ollama import InvalidResponseError, _ollama_model_if_cache
from lightrag.utils import is_truncated_response

pytestmark = pytest.mark.offline


def _make_fake_client(response: dict):
    return SimpleNamespace(
        chat=AsyncMock(return_value=response),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_ollama_length_done_reason_marks_result_truncated():
    """done_reason == "length" (num_predict exhausted) flags the response.

    The raw-dict response shape mirrors ollama<0.4; ollama>=0.4 returns a
    ChatResponse whose .get behaves identically (SubscriptableBaseModel).
    """
    raw_json = '{"entities":[{"name":"Ali'
    fake_client = _make_fake_client(
        {"message": {"content": raw_json}, "done_reason": "length"}
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="Extract")

    assert result == raw_json
    assert is_truncated_response(result) is True


@pytest.mark.asyncio
async def test_ollama_stop_done_reason_is_not_marked_truncated():
    raw_json = '{"entities":[]}'
    fake_client = _make_fake_client(
        {"message": {"content": raw_json}, "done_reason": "stop"}
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="Extract")

    assert result == raw_json
    assert is_truncated_response(result) is False


@pytest.mark.asyncio
async def test_ollama_missing_done_reason_is_not_marked_truncated():
    """Older servers may omit done_reason; degrade to cache-everything."""
    fake_client = _make_fake_client({"message": {"content": "answer"}})

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="Q")

    assert result == "answer"
    assert is_truncated_response(result) is False


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


@pytest.mark.parametrize("content", ["", "   \n  "])
async def test_ollama_empty_length_truncated_response_raises(
    caplog, _propagate_lightrag_logger, content
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


async def test_ollama_empty_length_response_names_reasoning_as_the_consumer():
    """``message.thinking`` present ⇒ say the budget went to reasoning."""
    fake_client = _make_fake_client(
        {
            "message": {"content": "", "thinking": "let me carefully consider..."},
            "done_reason": "length",
            "eval_count": 64,
        }
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(InvalidResponseError) as excinfo:
            await _ollama_model_if_cache(model="test-model", prompt="Extract")

    message = str(excinfo.value)
    assert "budget consumed by reasoning" in message
    assert "thinking_len=28" in message


async def test_ollama_empty_response_without_length_reason_is_unchanged():
    """Scope: only the token-limit case escalates.

    An empty response that ended normally is the model's answer, not a broken
    generation, and the callers that tolerate it today keep doing so.
    """
    fake_client = _make_fake_client({"message": {"content": ""}, "done_reason": "stop"})

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="Q")

    assert result == ""
    assert is_truncated_response(result) is False


async def test_ollama_partial_length_response_is_still_salvaged():
    """The escalation must not swallow the salvage path: non-empty truncated
    content is still returned (flagged), never raised."""
    fake_client = _make_fake_client(
        {"message": {"content": " x "}, "done_reason": "length"}
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="Extract")

    assert result == " x "
    assert is_truncated_response(result) is True
