"""Offline tests: `think` is lifted out of `options` into its own top-level
chat() argument, matching Ollama's real API shape (options={...}, think=...),
not nested inside options like the other OllamaLLMOptions fields.
"""

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


# think is on/off *or* a named reasoning level; the lift must not assume a bool.
@pytest.mark.parametrize("think", [False, "high"])
async def test_think_is_lifted_out_of_a_shared_options_dict(think):
    """options can be the same dict object reused across every call for a
    role's lifetime (library callers bind it once via llm_model_kwargs), so
    popping `think` out of it would only work on the first call."""
    fake_client = _make_fake_client()
    shared_options = {"think": think, "num_predict": 4096}

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        for prompt in ("first", "second"):
            await _ollama_model_if_cache(
                model="test-model", prompt=prompt, options=shared_options
            )

    for call in fake_client.chat.call_args_list:
        assert call.kwargs["think"] == think
        assert type(call.kwargs["think"]) is type(think)
        # num_predict (and everything else generation-related) stays inside
        # options -- only think has its own place in Ollama's real API.
        assert call.kwargs["options"] == {"num_predict": 4096}
    assert len(fake_client.chat.call_args_list) == 2
    assert shared_options == {"think": think, "num_predict": 4096}


@pytest.mark.parametrize(
    "extra",
    [
        # e.g. the embedding options dict, which has no think field at all
        pytest.param({"options": {"num_predict": 4096}}, id="options-without-think"),
        pytest.param({}, id="no-options"),
    ],
)
async def test_absent_think_adds_no_think_kwarg(extra):
    """The model keeps its own default rather than a hardcoded True/False."""
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        await _ollama_model_if_cache(model="test-model", prompt="Extract", **extra)

    call_kwargs = fake_client.chat.call_args.kwargs
    assert "think" not in call_kwargs
    assert call_kwargs.get("options") == extra.get("options")


async def test_think_on_an_old_ollama_is_refused_before_the_call(monkeypatch):
    """The API server validates think= at startup, but library callers never
    go through create_app -- so the call path keeps its own refusal, and it
    must fire before chat() is reached rather than letting the request go out
    and fail inside the client's own validation. The gate does not look at
    the value: a bool and a reasoning level sit behind the same floor."""
    monkeypatch.setattr(ollama_binding, "_OLLAMA_SUPPORTS_THINK", False)
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(RuntimeError, match="ollama>=0.5.4"):
            await _ollama_model_if_cache(
                model="test-model", prompt="Extract", options={"think": False}
            )

    fake_client.chat.assert_not_called()
