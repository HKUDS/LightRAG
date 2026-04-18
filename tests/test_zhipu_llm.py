import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _fake_embedding_vector(dim=1024):
    return [0.1] * dim


def _fake_chat_response(content="", reasoning_content=""):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _load_zhipu_module(monkeypatch, client_factory):
    fake_pm = SimpleNamespace(
        is_installed=lambda name: True,
        install=lambda name: None,
    )
    fake_openai = SimpleNamespace(
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
    )
    fake_zhipuai = SimpleNamespace(ZhipuAI=client_factory)

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)
    sys.modules.pop("lightrag.llm.zhipu", None)

    return importlib.import_module("lightrag.llm.zhipu")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_embedding_sends_dimensions_when_embedding_dim_provided(
    monkeypatch,
):
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=_fake_embedding_vector())]
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_embedding.func(
        ["hello"],
        api_key="test-key",
        embedding_dim=2048,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)
    assert captured_calls == [
        {"model": "embedding-3", "input": ["hello"], "dimensions": 2048}
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_embedding_omits_dimensions_when_embedding_dim_not_provided(
    monkeypatch,
):
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=_fake_embedding_vector())]
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    await zhipu_module.zhipu_embedding.func(["hello"], api_key="test-key")

    assert captured_calls == [{"model": "embedding-3", "input": ["hello"]}]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_forwards_official_thinking(monkeypatch):
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return _fake_chat_response(content="final answer")

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello",
        api_key="test-key",
        thinking={"type": "enabled"},
    )

    assert result == "final answer"
    assert captured_calls[0]["thinking"] == {"type": "enabled"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_filters_reasoning_when_cot_disabled(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(
                content="visible answer",
                reasoning_content="hidden chain of thought",
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello",
        api_key="test-key",
        enable_cot=False,
    )

    assert result == "visible answer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_includes_reasoning_when_cot_enabled(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(
                content="visible answer",
                reasoning_content="hidden chain of thought",
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello",
        api_key="test-key",
        enable_cot=True,
    )

    assert result == "<think>hidden chain of thought</think>visible answer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_keyword_extraction_ignores_reasoning_content(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(
                content='{"high_level_keywords": ["AI"], "low_level_keywords": ["RAG"]}',
                reasoning_content="this should not be parsed",
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    with pytest.warns(DeprecationWarning):
        result = await zhipu_module.zhipu_complete(
            prompt="hello",
            api_key="test-key",
            keyword_extraction=True,
            enable_cot=True,
        )

    assert result == '{"high_level_keywords": ["AI"], "low_level_keywords": ["RAG"]}'
