"""Offline tests: `think` is lifted out of `options` into its own top-level
chat() argument, matching Ollama's real API shape (options={...}, think=...),
not nested inside options like the other OllamaLLMOptions fields.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.ollama import _ollama_model_if_cache

pytestmark = pytest.mark.offline


def _make_fake_client():
    return SimpleNamespace(
        chat=AsyncMock(
            return_value={"message": {"content": "ok"}, "done_reason": "stop"}
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_think_is_moved_from_options_to_a_top_level_kwarg():
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await _ollama_model_if_cache(
            model="test-model",
            prompt="Extract",
            options={"think": False, "num_predict": 4096},
        )

    call_kwargs = fake_client.chat.call_args.kwargs
    assert call_kwargs["think"] is False
    # num_predict (and everything else generation-related) stays inside
    # options -- only think has its own place in Ollama's real API.
    assert call_kwargs["options"] == {"num_predict": 4096}


@pytest.mark.asyncio
async def test_no_think_key_means_no_think_kwarg_is_added():
    """Absent from options (e.g. the embedding options dict, which has no
    think field at all) must leave the model at its own default -- not get
    coerced to some hardcoded True/False that wasn't actually configured."""
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await _ollama_model_if_cache(
            model="test-model", prompt="Extract", options={"num_predict": 4096}
        )

    call_kwargs = fake_client.chat.call_args.kwargs
    assert "think" not in call_kwargs
    assert call_kwargs["options"] == {"num_predict": 4096}


@pytest.mark.asyncio
async def test_options_dict_is_not_mutated_across_repeated_calls():
    """options can be the same dict object reused across every call for a
    role's lifetime (library callers bind it once via llm_model_kwargs) --
    popping `think` out of it would only work on the first call and
    silently lose it on every call after that."""
    fake_client = _make_fake_client()
    shared_options = {"think": False, "num_predict": 4096}

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await _ollama_model_if_cache(
            model="test-model", prompt="first", options=shared_options
        )
        await _ollama_model_if_cache(
            model="test-model", prompt="second", options=shared_options
        )

    first_call, second_call = fake_client.chat.call_args_list
    assert first_call.kwargs["think"] is False
    assert second_call.kwargs["think"] is False
    assert shared_options == {"think": False, "num_predict": 4096}


@pytest.mark.asyncio
async def test_a_reasoning_level_is_lifted_out_the_same_way_a_bool_is():
    """think is on/off *or* a named reasoning level; the lift must not assume
    a bool or the level would end up nested back inside options."""
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await _ollama_model_if_cache(
            model="test-model",
            prompt="Extract",
            options={"think": "high", "num_predict": 4096},
        )

    call_kwargs = fake_client.chat.call_args.kwargs
    assert call_kwargs["think"] == "high"
    assert call_kwargs["options"] == {"num_predict": 4096}


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [False, "high"])
async def test_think_on_an_old_ollama_is_refused_before_the_call(monkeypatch, think):
    """The API server validates think= at startup, but library callers never
    go through create_app -- so the call path keeps its own refusal, and it
    must fire before chat() is reached rather than letting the request go out
    and fail inside the client's own validation.

    Both value forms sit behind the same floor: ChatRequest.think only widened
    from Optional[bool] to a Union with a Literal in 0.5.3, and the gate is
    pinned to the binding's declared floor (>=0.5.4) rather than to think's own
    minimum, so there is one ollama version to reason about."""
    import lightrag.llm.ollama as ollama_binding

    monkeypatch.setattr(ollama_binding, "_OLLAMA_SUPPORTS_THINK", False)
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(RuntimeError, match="ollama>=0.5.4"):
            await _ollama_model_if_cache(
                model="test-model", prompt="Extract", options={"think": think}
            )

    fake_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_missing_options_dict_entirely_does_not_crash():
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await _ollama_model_if_cache(model="test-model", prompt="Extract")

    call_kwargs = fake_client.chat.call_args.kwargs
    assert "think" not in call_kwargs
