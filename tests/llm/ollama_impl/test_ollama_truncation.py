"""Offline tests: token-limit-truncated Ollama responses carry the marker."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.ollama import _ollama_model_if_cache
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
