"""Offline tests for Gemini's generic output-token alias (#3695).

LightRAG's provider-agnostic ``max_tokens`` knob used to vanish into the
``**_`` catch-all of ``gemini_complete_if_cache`` — the request was sent with
no output-length limit at all. The binding now maps it to Gemini's native
``max_output_tokens`` generation-config field, with an explicitly configured
native value taking precedence (same contract as the ollama/lmdeploy/hf
bindings from #3687/#3689/#3690).
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_gemini_module(monkeypatch, request):
    fake_pm = SimpleNamespace(
        is_installed=lambda name: True,
        install=lambda name: None,
    )

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        HttpOptions=FakeHttpOptions,
        FinishReason=SimpleNamespace(MAX_TOKENS="MAX_TOKENS", STOP="STOP"),
    )
    fake_genai = SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(kwargs=kwargs))
    fake_google_module = ModuleType("google")
    fake_google_module.genai = fake_genai
    fake_api_exceptions = SimpleNamespace(
        InternalServerError=type("InternalServerError", (Exception,), {}),
        ServiceUnavailable=type("ServiceUnavailable", (Exception,), {}),
        ResourceExhausted=type("ResourceExhausted", (Exception,), {}),
        GatewayTimeout=type("GatewayTimeout", (Exception,), {}),
        BadGateway=type("BadGateway", (Exception,), {}),
        DeadlineExceeded=type("DeadlineExceeded", (Exception,), {}),
        Aborted=type("Aborted", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
    )
    fake_google_api_core = ModuleType("google.api_core")
    fake_google_api_core.exceptions = fake_api_exceptions

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(types=fake_types))
    monkeypatch.setitem(sys.modules, "google.api_core", fake_google_api_core)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", fake_api_exceptions)

    parent = sys.modules.get("lightrag.llm")
    original_gemini = sys.modules.get("lightrag.llm.gemini")
    original_parent_attr = getattr(parent, "gemini", None) if parent else None
    sys.modules.pop("lightrag.llm.gemini", None)
    if parent is not None and hasattr(parent, "gemini"):
        delattr(parent, "gemini")

    def _restore_gemini():
        if original_gemini is not None:
            sys.modules["lightrag.llm.gemini"] = original_gemini
        else:
            sys.modules.pop("lightrag.llm.gemini", None)
        if parent is not None:
            if original_parent_attr is not None:
                parent.gemini = original_parent_attr
            elif hasattr(parent, "gemini"):
                delattr(parent, "gemini")

    request.addfinalizer(_restore_gemini)

    return importlib.import_module("lightrag.llm.gemini")


def _make_fake_response():
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="ok", thought=False)]
                ),
                finish_reason="STOP",
            ),
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=2,
        ),
    )


def _make_capturing_client(captured):
    async def _fake_generate_content(**kwargs):
        captured.update(kwargs)
        return _make_fake_response()

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_fake_generate_content)
        )
    )


async def _sent_config(gemini_module, monkeypatch, request, **generation_kwargs):
    captured: dict = {}
    fake_client = _make_capturing_client(captured)
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    result = await gemini_module.gemini_complete_if_cache(
        model="gemini-model",
        prompt="hello",
        api_key="test-key",
        **generation_kwargs,
    )
    assert result == "ok"
    return captured.get("config")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_generic_limit_maps_to_max_output_tokens(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = await _sent_config(gemini_module, monkeypatch, request, max_tokens=37)

    assert config.kwargs["max_output_tokens"] == 37


@pytest.mark.offline
@pytest.mark.asyncio
async def test_generic_limit_preserves_other_generation_config(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = await _sent_config(
        gemini_module,
        monkeypatch,
        request,
        max_tokens=37,
        generation_config={"temperature": 0.2},
    )

    assert config.kwargs["max_output_tokens"] == 37
    assert config.kwargs["temperature"] == 0.2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_native_limit_is_preserved_without_alias(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = await _sent_config(
        gemini_module,
        monkeypatch,
        request,
        generation_config={"max_output_tokens": 41},
    )

    assert config.kwargs["max_output_tokens"] == 41


@pytest.mark.offline
@pytest.mark.asyncio
async def test_native_limit_takes_precedence_over_alias(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = await _sent_config(
        gemini_module,
        monkeypatch,
        request,
        max_tokens=37,
        generation_config={"max_output_tokens": 41},
    )

    assert config.kwargs["max_output_tokens"] == 41


@pytest.mark.offline
@pytest.mark.asyncio
async def test_caller_generation_config_is_not_mutated(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)
    shared_config = {"temperature": 0.2}

    config = await _sent_config(
        gemini_module,
        monkeypatch,
        request,
        max_tokens=37,
        generation_config=shared_config,
    )

    assert config.kwargs["max_output_tokens"] == 37
    assert shared_config == {"temperature": 0.2}, (
        "the caller's generation_config dict must not be mutated"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_limit_sent_when_nothing_configured(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = await _sent_config(gemini_module, monkeypatch, request)

    # With nothing to configure the binding sends no config object at all.
    assert config is None or "max_output_tokens" not in config.kwargs
