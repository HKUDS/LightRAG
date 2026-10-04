"""Offline tests for retrying transient Zhipu API errors and preserving exception types.

The ZhipuAI SDK raises its own exception types (such as RateLimitError,
APIConnectionError, APITimeoutError, and InternalServerError). The retry
predicates and handlers must match those native SDK exceptions, re-raise the
underlying causes instead of wrapping them in tenacity.RetryError, and allow
embedding exceptions to propagate with their native types intact.
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tenacity import wait_none

pytestmark = pytest.mark.offline


class FakeZhipuAIError(Exception):
    """Base error for ZhipuAI SDK."""


class FakeAPIError(FakeZhipuAIError):
    """API error for ZhipuAI SDK."""


class FakeAPIStatusError(FakeAPIError):
    """Status error with HTTP status code."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FakeRateLimitError(FakeAPIStatusError):
    """429 Rate limit error."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class FakeAPIConnectionError(FakeAPIError):
    """Network connection error."""

    def __init__(self, message: str = "Connection error"):
        super().__init__(message)


class FakeAPITimeoutError(FakeAPIConnectionError):
    """Request timeout error."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)


class FakeInternalServerError(FakeAPIStatusError):
    """5xx Server error."""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, status_code=500)


class FakeAuthenticationError(FakeAPIStatusError):
    """401 Authentication error."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401)


def _fake_chat_response(content="ok"):
    message = SimpleNamespace(content=content, reasoning_content="")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


def _fake_embedding_vector(dim=1024):
    return [0.1] * dim


def _load_zhipu_module_with_exceptions(monkeypatch, client_factory):
    fake_pm = SimpleNamespace(
        is_installed=lambda name: True,
        install=lambda name: None,
    )
    fake_zhipuai = SimpleNamespace(
        ZhipuAI=client_factory,
        ZhipuAIError=FakeZhipuAIError,
        APIError=FakeAPIError,
        APIStatusError=FakeAPIStatusError,
        RateLimitError=FakeRateLimitError,
        APIConnectionError=FakeAPIConnectionError,
        APITimeoutError=FakeAPITimeoutError,
        InternalServerError=FakeInternalServerError,
        AuthenticationError=FakeAuthenticationError,
    )

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)
    sys.modules.pop("lightrag.llm.zhipu", None)

    return importlib.import_module("lightrag.llm.zhipu")


def _make_flaky_chat_client(*, failures: int, exc_factory):
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise exc_factory()
        return _fake_chat_response(content="chat-ok")

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    return FakeClient, calls


def _make_flaky_embedding_client(*, failures: int, exc_factory):
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise exc_factory()
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=_fake_embedding_vector())]
        )

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.embeddings = SimpleNamespace(create=create)

    return FakeClient, calls


@pytest.mark.asyncio
async def test_zhipu_complete_retries_transient_rate_limit_error(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=2, exc_factory=lambda: FakeRateLimitError("429 Too Many Requests")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert result == "chat-ok"
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_zhipu_complete_retries_transient_connection_error(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=1, exc_factory=lambda: FakeAPIConnectionError("connect failed")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert result == "chat-ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_zhipu_complete_retries_transient_timeout_error(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=1, exc_factory=lambda: FakeAPITimeoutError("request timed out")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )

    assert result == "chat-ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_zhipu_complete_exhausted_retries_reraises_original_error(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=10, exc_factory=lambda: FakeRateLimitError("rate limited")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError) as exc_info:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert calls["count"] == 3
    assert "rate limited" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zhipu_complete_fails_fast_on_non_transient_error(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=1, exc_factory=lambda: FakeAuthenticationError("invalid api key")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    with pytest.raises(FakeAuthenticationError):
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_zhipu_embedding_retries_transient_connection_error(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(
        failures=1, exc_factory=lambda: FakeAPIConnectionError("connect dropped")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    result = await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_zhipu_embedding_retries_transient_rate_limit_error(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(
        failures=2, exc_factory=lambda: FakeRateLimitError("429 Too Many Requests")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    result = await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_zhipu_embedding_preserves_exception_type_on_failure(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(
        failures=1, exc_factory=lambda: FakeAuthenticationError("invalid key")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    with pytest.raises(FakeAuthenticationError) as exc_info:
        await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert calls["count"] == 1
    assert "invalid key" in str(exc_info.value)
    assert "Error calling ChatGLM" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_zhipu_embedding_exhausted_retries_reraises_original_error(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(
        failures=10, exc_factory=lambda: FakeRateLimitError("rate limited")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError) as exc_info:
        await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert calls["count"] == 3
    assert "rate limited" in str(exc_info.value)
