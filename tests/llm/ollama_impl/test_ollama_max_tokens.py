"""Offline tests for Ollama's generic output-token alias."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import lightrag.llm.ollama as ollama_binding
from lightrag.llm.ollama import _ollama_model_if_cache

pytestmark = pytest.mark.offline


def _make_fake_client():
    return SimpleNamespace(
        chat=AsyncMock(
            return_value={"message": {"content": "ok"}, "done_reason": "stop"}
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


async def _sent_options(**generation_kwargs):
    fake_client = _make_fake_client()
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(
            model="test-model",
            prompt="hello",
            **generation_kwargs,
        )

    assert result == "ok"
    return fake_client.chat.call_args.kwargs.get("options")


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_options"),
    [
        pytest.param(
            {"max_tokens": 37},
            {"num_predict": 37},
            id="generic-limit-maps-to-num-predict",
        ),
        pytest.param(
            {"max_tokens": 37, "options": {"temperature": 0.2}},
            {"temperature": 0.2, "num_predict": 37},
            id="generic-limit-preserves-other-options",
        ),
        pytest.param(
            {"options": {"num_predict": 41}},
            {"num_predict": 41},
            id="native-limit-is-preserved",
        ),
        pytest.param(
            {"max_tokens": 37, "options": {"num_predict": 41}},
            {"num_predict": 41},
            id="native-limit-takes-precedence",
        ),
    ],
)
async def test_max_tokens_alias_precedence(generation_kwargs, expected_options):
    assert await _sent_options(**generation_kwargs) == expected_options


async def test_mapping_options_are_not_mutated():
    shared_options = {"temperature": 0.2}

    assert await _sent_options(max_tokens=37, options=shared_options) == {
        "temperature": 0.2,
        "num_predict": 37,
    }
    assert shared_options == {"temperature": 0.2}


async def test_ollama_options_object_is_copied_when_alias_is_applied():
    shared_options = ollama_binding.ollama.Options(temperature=0.2)

    sent_options = await _sent_options(max_tokens=37, options=shared_options)

    assert sent_options.num_predict == 37
    assert sent_options.temperature == 0.2
    assert shared_options.num_predict is None


async def test_native_value_in_ollama_options_object_takes_precedence():
    shared_options = ollama_binding.ollama.Options(num_predict=41)

    sent_options = await _sent_options(max_tokens=37, options=shared_options)

    assert sent_options is shared_options
    assert sent_options.num_predict == 41
