"""Offline tests for retrying transient Zhipu API errors and preserving exception types.

The ZhipuAI SDK raises its own exception types (such as RateLimitError,
APIReachLimitError, APIConnectionError, APITimeoutError, InternalServerError,
and APIInternalError). The retry predicates and handlers must match those native
SDK exceptions, fail fast on permanent spend stops (account in arrears, quota
exhaustion, plan expiration), re-raise the underlying causes instead of wrapping
them in tenacity.RetryError, and allow embedding exceptions to propagate with
their native types intact.
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

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: dict | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.code = code


class FakeRateLimitError(FakeAPIStatusError):
    """429 Rate limit error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        body: dict | None = None,
        code: str | None = None,
    ):
        super().__init__(message, status_code=429, body=body, code=code)


class FakeAPIReachLimitError(FakeAPIStatusError):
    """Alternative rate limit error class in some SDK versions."""

    def __init__(
        self,
        message: str = "Rate limit reached",
        *,
        body: dict | None = None,
        code: str | None = None,
    ):
        super().__init__(message, status_code=429, body=body, code=code)


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


class FakeAPIInternalError(FakeAPIStatusError):
    """Alternative 5xx server error class in some SDK versions."""

    def __init__(self, message: str = "Internal error"):
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
        APIReachLimitError=FakeAPIReachLimitError,
        APIConnectionError=FakeAPIConnectionError,
        APITimeoutError=FakeAPITimeoutError,
        InternalServerError=FakeInternalServerError,
        APIInternalError=FakeAPIInternalError,
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
        failures=2,
        exc_factory=lambda: FakeRateLimitError(
            "429 Too Many Requests",
            body={"error": {"code": "1302", "message": "Rate limit exceeded"}},
        ),
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
async def test_zhipu_complete_retries_api_reach_limit_and_api_internal_error(
    monkeypatch,
):
    client_cls, calls = _make_flaky_chat_client(
        failures=1, exc_factory=lambda: FakeAPIReachLimitError("reach limit")
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    result = await zhipu_module.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )
    assert result == "chat-ok"
    assert calls["count"] == 2

    client_cls2, calls2 = _make_flaky_chat_client(
        failures=1, exc_factory=lambda: FakeAPIInternalError("internal error")
    )
    zhipu_module2 = _load_zhipu_module_with_exceptions(monkeypatch, client_cls2)
    zhipu_module2.zhipu_complete_if_cache.retry.wait = wait_none()

    result2 = await zhipu_module2.zhipu_complete_if_cache(
        prompt="hello", api_key="test-key"
    )
    assert result2 == "chat-ok"
    assert calls2["count"] == 2


@pytest.mark.asyncio
async def test_zhipu_complete_fails_fast_on_arrears_code_1113(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=1,
        exc_factory=lambda: FakeRateLimitError(
            'Error code: 429, with error text {"error":{"code":"1113","message":"您的账户已欠费，请充值后重试。"}}',
            body={
                "error": {"code": "1113", "message": "您的账户已欠费，请充值后重试。"}
            },
            code="1113",
        ),
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError) as exc_info:
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert calls["count"] == 1
    assert "您的账户已欠费" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zhipu_complete_fails_fast_on_quota_limits_code_1308_and_1310(
    monkeypatch,
):
    for code, desc in [("1308", "达到时间段使用上限"), ("1310", "达到周期使用上限")]:
        client_cls, calls = _make_flaky_chat_client(
            failures=1,
            exc_factory=lambda c=code, d=desc: FakeRateLimitError(
                f"Rate limit: {d}",
                body={"error": {"code": c, "message": d}},
                code=c,
            ),
        )
        zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
        zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

        with pytest.raises(FakeRateLimitError):
            await zhipu_module.zhipu_complete_if_cache(
                prompt="hello", api_key="test-key"
            )

        assert calls["count"] == 1


@pytest.mark.asyncio
async def test_zhipu_complete_fails_fast_on_plan_expired_code_1309(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=1,
        exc_factory=lambda: FakeRateLimitError(
            "Rate limit: 套餐已到期",
            body={"error": {"code": "1309", "message": "套餐已到期"}},
            code="1309",
        ),
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError):
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_zhipu_complete_fails_fast_on_proxy_insufficient_quota(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=1,
        exc_factory=lambda: FakeRateLimitError(
            "insufficient quota",
            body={"error": {"type": "insufficient_quota", "message": "quota exceeded"}},
        ),
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_complete_if_cache.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError):
        await zhipu_module.zhipu_complete_if_cache(prompt="hello", api_key="test-key")

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_zhipu_complete_exhausted_retries_reraises_original_error(monkeypatch):
    client_cls, calls = _make_flaky_chat_client(
        failures=10,
        exc_factory=lambda: FakeRateLimitError(
            "rate limited",
            body={"error": {"code": "1302", "message": "rate limited"}},
        ),
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
        failures=2,
        exc_factory=lambda: FakeRateLimitError(
            "429 Too Many Requests",
            body={"error": {"code": "1302", "message": "Too many requests"}},
        ),
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    result = await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_zhipu_embedding_fails_fast_on_arrears_code_1113(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(
        failures=1,
        exc_factory=lambda: FakeRateLimitError(
            'Error code: 429, with error text {"error":{"code":"1113","message":"您的账户已欠费，请充值后重试。"}}',
            body={
                "error": {"code": "1113", "message": "您的账户已欠费，请充值后重试。"}
            },
            code="1113",
        ),
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError) as exc_info:
        await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert calls["count"] == 1
    assert "您的账户已欠费" in str(exc_info.value)


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
        failures=10,
        exc_factory=lambda: FakeRateLimitError(
            "rate limited",
            body={"error": {"code": "1302", "message": "rate limited"}},
        ),
    )
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)
    zhipu_module.zhipu_embedding.func.retry.wait = wait_none()

    with pytest.raises(FakeRateLimitError) as exc_info:
        await zhipu_module.zhipu_embedding(["text"], api_key="test-key")

    assert calls["count"] == 3
    assert "rate limited" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zhipu_embedding_empty_texts_returns_2d_array(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(failures=0, exc_factory=None)
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)

    result = await zhipu_module.zhipu_embedding([], api_key="test-key")
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 1024)
    assert calls["count"] == 0

    result_dim = await zhipu_module.zhipu_embedding(
        [], api_key="test-key", embedding_dim=512
    )
    assert isinstance(result_dim, np.ndarray)
    assert result_dim.shape == (0, 512)
    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_zhipu_embedding_single_string_converts_to_batch(monkeypatch):
    client_cls, calls = _make_flaky_embedding_client(failures=0, exc_factory=None)
    zhipu_module = _load_zhipu_module_with_exceptions(monkeypatch, client_cls)

    result = await zhipu_module.zhipu_embedding(
        "single string text", api_key="test-key"
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)
    assert calls["count"] == 1
