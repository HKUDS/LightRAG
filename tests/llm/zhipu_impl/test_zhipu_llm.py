import asyncio
import importlib
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


def _fake_embedding_vector(dim=1024):
    return [0.1] * dim


def _fake_chat_response(content="", reasoning_content="", usage=None):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


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
async def test_zhipu_complete_records_token_usage(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(content="answer", usage=usage)

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    class FakeTracker:
        def __init__(self):
            self.calls = []

        def add_usage(self, token_counts):
            self.calls.append(token_counts)

    tracker = FakeTracker()
    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key", token_tracker=tracker
    )

    assert result == "answer"
    assert tracker.calls == [
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_token_tracker_never_reaches_the_raw_client_call(
    monkeypatch,
):
    """token_tracker is a LightRAG-only concept, not a real Zhipu API field --
    it must be consumed as a named parameter, never forwarded through
    **kwargs into the raw client call."""
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return _fake_chat_response(content="answer")

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    class FakeTracker:
        def add_usage(self, token_counts):
            pass

    await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key", token_tracker=FakeTracker()
    )

    assert "token_tracker" not in captured_calls[0]


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


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_if_cache_entity_extraction_maps_to_json_object(monkeypatch):
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return _fake_chat_response(
                content='{"entities":[],"relationships":[]}',
                reasoning_content="this should not be parsed",
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    with pytest.warns(DeprecationWarning):
        result = await zhipu_module.zhipu_complete_if_cache(
            prompt="hello",
            api_key="test-key",
            entity_extraction=True,
            enable_cot=True,
        )

    assert result == '{"entities":[],"relationships":[]}'
    assert captured_calls[0]["response_format"] == {"type": "json_object"}
    assert "entity_extraction" not in captured_calls[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_runs_client_call_off_the_event_loop_thread(monkeypatch):
    """ZhipuAI wraps a synchronous httpx.Client, so calling it directly from
    this async function would block the event loop for the whole HTTP
    request. The call must run on a worker thread instead."""
    call_thread_id = {}
    main_thread_id = threading.get_ident()

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            call_thread_id["id"] = threading.get_ident()
            return _fake_chat_response(content="answer")

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert result == "answer"
    assert call_thread_id["id"] != main_thread_id


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_embedding_runs_client_call_off_the_event_loop_thread(monkeypatch):
    call_thread_id = {}
    main_thread_id = threading.get_ident()

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            call_thread_id["id"] = threading.get_ident()
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=_fake_embedding_vector())]
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    await zhipu_module.zhipu_embedding.func(["hello"], api_key="test-key")

    assert call_thread_id["id"] != main_thread_id


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_logs_and_repropagates_cancellation(monkeypatch):
    """Cancelling the outer await (e.g. an execution timeout) still has to
    propagate CancelledError, with a warning noting the SDK call keeps
    running in the background thread until it finishes on its own."""
    call_started = threading.Event()
    release_call = threading.Event()
    warnings_logged = []

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            call_started.set()
            release_call.wait(timeout=5)
            return _fake_chat_response(content="answer")

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)
    monkeypatch.setattr(
        zhipu_module.logger, "warning", lambda msg: warnings_logged.append(msg)
    )

    task = asyncio.ensure_future(
        zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")
    )
    for _ in range(500):
        if call_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert call_started.is_set()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    release_call.set()
    assert len(warnings_logged) == 1
    assert "cancelled while awaiting the SDK call" in warnings_logged[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_if_cache_structured_output_disables_cot(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(
                content='{"answer":"ok"}',
                reasoning_content="this should not be included",
            )

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello",
        api_key="test-key",
        response_format={"type": "json_object"},
        enable_cot=True,
    )

    assert result == '{"answer":"ok"}'
    assert "<think>" not in result
