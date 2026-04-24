from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeAnthropicError(Exception):
    pass


class _FakeMessages:
    def __init__(self, calls: list[dict], response):
        self._calls = calls
        self._response = response

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        return self._response


class _FakeStream:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield SimpleNamespace(delta=SimpleNamespace(text=chunk))


def _load_anthropic_module(monkeypatch, response):
    calls: list[dict] = []
    clients: list[dict] = []

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.messages = _FakeMessages(calls, response)

    fake_pm = SimpleNamespace(is_installed=lambda name: True, install=lambda name: None)
    fake_tenacity = SimpleNamespace(
        retry=lambda **_kwargs: lambda func: func,
        stop_after_attempt=lambda *_args, **_kwargs: None,
        wait_exponential=lambda *_args, **_kwargs: None,
        retry_if_exception_type=lambda *_args, **_kwargs: None,
    )
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = FakeAsyncAnthropic
    fake_anthropic.APIConnectionError = _FakeAnthropicError
    fake_anthropic.RateLimitError = _FakeAnthropicError
    fake_anthropic.APITimeoutError = _FakeAnthropicError

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "tenacity", fake_tenacity)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    sys.modules.pop("lightrag.llm.anthropic", None)

    return importlib.import_module("lightrag.llm.anthropic"), calls, clients


@pytest.mark.offline
def test_anthropic_non_streaming_returns_text_and_default_max_tokens(
    monkeypatch,
):
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", text="ignored"),
            SimpleNamespace(type="text", text=" world"),
        ]
    )
    anthropic_module, calls, clients = _load_anthropic_module(monkeypatch, response)

    result = asyncio.run(
        anthropic_module.anthropic_complete_if_cache(
            model="claude-test",
            prompt="hi",
            api_key="test-key",
            response_format={"type": "json_object"},
        )
    )

    assert result == "hello world"
    assert clients[-1]["api_key"] == "test-key"
    assert calls[-1]["stream"] is False
    assert calls[-1]["max_tokens"] == 8192
    assert "response_format" not in calls[-1]


@pytest.mark.offline
def test_anthropic_streaming_path_stays_opt_in(monkeypatch):
    anthropic_module, calls, _ = _load_anthropic_module(
        monkeypatch,
        _FakeStream(["hello", " world"]),
    )

    async def run_case():
        stream = await anthropic_module.anthropic_complete_if_cache(
            model="claude-test",
            prompt="hi",
            api_key="test-key",
            stream=True,
            max_tokens=128,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_case())

    assert "".join(chunks) == "hello world"
    assert calls[-1]["stream"] is True
    assert calls[-1]["max_tokens"] == 128
