import asyncio
import importlib
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from tenacity import wait_none

from lightrag.utils import is_truncated_response


def _fake_embedding_vector(dim=1024):
    return [0.1] * dim


def _fake_chat_response(
    content="", reasoning_content="", usage=None, finish_reason="stop"
):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


class FakeZaiError(Exception):
    pass


class FakeAPIStatusError(FakeZaiError):
    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class FakeAPIConnectionError(FakeZaiError):
    pass


class FakeAPITimeoutError(FakeAPIConnectionError):
    pass


# Mirrors the zai-sdk hierarchy the retry predicate relies on; the real
# classes are pinned by test_retry_predicate_matches_real_zai_error_classes.
_fake_zai_errors = SimpleNamespace(
    ZaiError=FakeZaiError,
    APIStatusError=FakeAPIStatusError,
    APIConnectionError=FakeAPIConnectionError,
    APITimeoutError=FakeAPITimeoutError,
)


def _load_zhipu_module(monkeypatch, client_factory):
    fake_pm = SimpleNamespace(
        is_installed=lambda name: True,
        install=lambda name: None,
    )
    fake_zai = SimpleNamespace(
        ZhipuAiClient=client_factory, core=SimpleNamespace(_errors=_fake_zai_errors)
    )

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "zai", fake_zai)
    monkeypatch.setitem(sys.modules, "zai.core", fake_zai.core)
    monkeypatch.setitem(sys.modules, "zai.core._errors", _fake_zai_errors)
    sys.modules.pop("lightrag.llm.zhipu", None)

    return importlib.import_module("lightrag.llm.zhipu")


def _no_retry_wait(monkeypatch, zhipu_module):
    monkeypatch.setattr(zhipu_module.zhipu_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(zhipu_module.zhipu_embedding.func.retry, "wait", wait_none())


def _failing_client(errors, calls, client_kwargs=None):
    """A client whose chat and embeddings calls raise ``errors`` in order, then succeed."""
    pending = list(errors)

    class FakeClient:
        def __init__(self, **kwargs):
            if client_kwargs is not None:
                client_kwargs.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self.embeddings = SimpleNamespace(create=self.embed)

        def _next(self):
            calls.append(1)
            if pending:
                raise pending.pop(0)

        def create(self, **kwargs):
            self._next()
            return _fake_chat_response(content="answer")

        def embed(self, **kwargs):
            self._next()
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=_fake_embedding_vector())]
            )

    return FakeClient


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_embedding_sends_dimensions_when_embedding_dim_provided(
    monkeypatch,
):
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
async def test_zhipu_embedding_reads_legacy_zhipuai_env_when_no_api_key(
    monkeypatch,
):
    received_keys = []

    class FakeClient:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            received_keys.append((api_key, base_url))
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=_fake_embedding_vector())]
            )

    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.setenv("ZHIPUAI_API_KEY", "env-key")
    monkeypatch.setenv("ZHIPUAI_BASE_URL", "https://proxy.example/v4")
    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    await zhipu_module.zhipu_embedding.func(["hello"])

    assert received_keys == [("env-key", "https://proxy.example/v4")]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_forwards_official_thinking(monkeypatch):
    captured_calls = []

    class FakeClient:
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
async def test_zhipu_length_finish_reason_marks_result_truncated(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(content="partial answer", finish_reason="length")

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert is_truncated_response(result)
    assert result == "partial answer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_stop_finish_reason_keeps_plain_response(monkeypatch):
    class FakeClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return _fake_chat_response(content="complete answer", finish_reason="stop")

    zhipu_module = _load_zhipu_module(monkeypatch, FakeClient)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert not is_truncated_response(result)
    assert result == "complete answer"


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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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
        def __init__(self, api_key=None, **kwargs):
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


_TRANSIENT_ERRORS = [
    pytest.param(lambda: FakeAPIStatusError(429), id="429-rate-limit"),
    pytest.param(lambda: FakeAPIStatusError(408), id="408-request-timeout"),
    pytest.param(lambda: FakeAPIStatusError(409), id="409-lock-timeout"),
    pytest.param(lambda: FakeAPIStatusError(500), id="500-internal"),
    pytest.param(lambda: FakeAPIStatusError(503), id="503-flow-exceeded"),
    pytest.param(lambda: FakeAPIConnectionError("Connection error."), id="conn"),
    pytest.param(lambda: FakeAPITimeoutError("Request timed out."), id="timeout"),
]

_PERMANENT_ERRORS = [
    pytest.param(lambda: FakeAPIStatusError(400), id="400-bad-request"),
    pytest.param(lambda: FakeAPIStatusError(401), id="401-auth"),
    pytest.param(lambda: FakeZaiError("api_key not provided"), id="zai-error"),
    pytest.param(lambda: ValueError("not an SDK error"), id="foreign"),
]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", _TRANSIENT_ERRORS)
async def test_zhipu_complete_retries_transient_zai_errors(monkeypatch, make_error):
    calls = []
    zhipu_module = _load_zhipu_module(
        monkeypatch, _failing_client([make_error()], calls)
    )
    _no_retry_wait(monkeypatch, zhipu_module)

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert result == "answer"
    assert len(calls) == 2


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", _PERMANENT_ERRORS)
async def test_zhipu_complete_fails_fast_on_permanent_errors(monkeypatch, make_error):
    calls = []
    error = make_error()
    zhipu_module = _load_zhipu_module(monkeypatch, _failing_client([error], calls))
    _no_retry_wait(monkeypatch, zhipu_module)

    with pytest.raises(type(error)):
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert len(calls) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_complete_reraises_the_sdk_error_after_the_last_attempt(
    monkeypatch,
):
    calls = []
    errors = [FakeAPIStatusError(429) for _ in range(3)]
    zhipu_module = _load_zhipu_module(monkeypatch, _failing_client(errors, calls))
    _no_retry_wait(monkeypatch, zhipu_module)

    with pytest.raises(FakeAPIStatusError) as exc_info:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert exc_info.value.status_code == 429
    assert len(calls) == 3


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", _TRANSIENT_ERRORS)
async def test_zhipu_embedding_retries_transient_zai_errors(monkeypatch, make_error):
    calls = []
    zhipu_module = _load_zhipu_module(
        monkeypatch, _failing_client([make_error()], calls)
    )
    _no_retry_wait(monkeypatch, zhipu_module)

    result = await zhipu_module.zhipu_embedding.func(["hello"], api_key="test-key")

    assert result.shape == (1, 1024)
    assert len(calls) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_embedding_wraps_permanent_errors_without_retrying(monkeypatch):
    calls = []
    error = FakeAPIStatusError(400)
    zhipu_module = _load_zhipu_module(monkeypatch, _failing_client([error], calls))
    _no_retry_wait(monkeypatch, zhipu_module)

    with pytest.raises(Exception, match="Error calling ChatGLM Embedding API") as exc:
        await zhipu_module.zhipu_embedding.func(["hello"], api_key="test-key")

    assert exc.value.__cause__ is error
    assert len(calls) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_zhipu_client_disables_the_sdk_retry_loop(monkeypatch):
    calls, client_kwargs = [], []
    zhipu_module = _load_zhipu_module(
        monkeypatch, _failing_client([], calls, client_kwargs)
    )

    await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")
    await zhipu_module.zhipu_embedding.func(["hello"], api_key="test-key")

    assert [kwargs.get("max_retries") for kwargs in client_kwargs] == [0, 0]


@pytest.mark.offline
def test_retry_predicate_matches_real_zai_error_classes(monkeypatch):
    """Pin the predicate to the real zai-sdk classes and their private import path."""
    errors = pytest.importorskip("zai.core._errors")
    httpx = pytest.importorskip("httpx")
    monkeypatch.delitem(sys.modules, "lightrag.llm.zhipu", raising=False)
    zhipu_module = importlib.import_module("lightrag.llm.zhipu")
    request = httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/x")

    def status_error(cls, code):
        response = httpx.Response(code, request=request)
        return cls(message=f"Error code: {code}", response=response)

    retryable = [
        errors.APITimeoutError(request=request),
        errors.APIConnectionError(request=request),
        status_error(errors.APIReachLimitError, 429),
        status_error(errors.APIInternalError, 500),
        status_error(errors.APIServerFlowExceedError, 503),
        status_error(errors.APIStatusError, 502),
        status_error(errors.APIStatusError, 408),
        status_error(errors.APIStatusError, 409),
    ]
    permanent = [
        status_error(errors.APIRequestFailedError, 400),
        status_error(errors.APIAuthenticationError, 401),
        status_error(errors.APIStatusError, 404),
        errors.ZaiError("api_key not provided"),
    ]

    assert [zhipu_module._is_retryable_zhipu_error(e) for e in retryable] == [
        True
    ] * len(retryable)
    assert [zhipu_module._is_retryable_zhipu_error(e) for e in permanent] == [
        False
    ] * len(permanent)
