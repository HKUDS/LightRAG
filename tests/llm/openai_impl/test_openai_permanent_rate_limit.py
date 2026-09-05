"""A permanent spend stop must fail fast instead of being retried.

LiteLLM reports an exhausted spend budget as HTTP 429 with
``type == "budget_exceeded"``, which the OpenAI SDK surfaces as
``RateLimitError`` -- indistinguishable, by type alone, from an ordinary
throughput throttle. The bindings retried it three times with exponential
backoff and then raised ``tenacity.RetryError``, so every remaining chunk in
the run paid the full backoff window and the operator-actionable message
("Budget has been exceeded! Team=... Max budget: 180.0") was buried behind an
opaque wrapper.

The budget only clears when an operator raises or resets it, so the retry
predicate now excludes it while ordinary 429s keep their backoff. OpenAI's
native ``insufficient_quota`` is the same shape of permanent 429 and is
classified alongside it. The original ``RateLimitError`` still propagates
unchanged -- callers doing ``except RateLimitError`` / ``isinstance`` keep
working, and the failure summary carries the provider's own message.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError, RateLimitError
from tenacity import RetryError, wait_none

from lightrag.llm._error_utils import is_permanent_rate_limit_error
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

pytestmark = pytest.mark.offline

# Verbatim from the OpenAI SDK's own unwrapping: ``_make_status_error`` does
# ``data = body.get("error", body)``, so ``error.body`` is the inner object.
BUDGET_BODY = {
    "message": (
        "Budget has been exceeded! Team=6ef183de-04ba-4114-8ef6-4b628370bc8c "
        "Current cost: 180.1173603404399, Max budget: 180.0"
    ),
    "type": "budget_exceeded",
    "param": None,
    "code": "429",
}

# OpenAI's own permanent 429: the account's billing quota is exhausted.
QUOTA_BODY = {
    "message": (
        "You exceeded your current quota, please check your plan and billing details."
    ),
    "type": "insufficient_quota",
    "param": None,
    "code": "insufficient_quota",
}

THROTTLE_BODY = {
    "message": "Rate limit reached for gpt-4o-mini. Please try again in 1s.",
    "type": "requests",
    "param": None,
    "code": "rate_limit_exceeded",
}


def _make_rate_limit_error(body: object, message: str | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://proxy.example/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError(
        message or f"Error code: 429 - {body}", response=response, body=body
    )


def _make_chat_client(error: Exception) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=error))
        ),
        close=AsyncMock(),
    )


class _FailingEmbeddingClient:
    """Minimal async-context-manager client whose embeddings.create always fails."""

    def __init__(self, error: Exception):
        self.calls = 0
        self._error = error
        self.embeddings = SimpleNamespace(create=self._create)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _create(self, **params):
        self.calls += 1
        raise self._error


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Strip the exponential backoff so the retrying cases stay fast.

    ``wraps`` stores the ``AsyncRetrying`` instance on ``.retry`` and each call
    copies it, so patching the attribute here reaches the per-call copy.
    """
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(openai_embed.func.retry, "wait", wait_none())


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, message",
    [
        pytest.param(BUDGET_BODY, None, id="budget-unwrapped-body"),
        pytest.param({"error": BUDGET_BODY}, None, id="budget-nested-envelope"),
        pytest.param(QUOTA_BODY, None, id="quota-unwrapped-body"),
        pytest.param({"error": QUOTA_BODY}, None, id="quota-nested-envelope"),
        pytest.param(
            None,
            "Error code: 429 - {'error': {'message': 'Budget has been exceeded! "
            "Team=x', 'type': 'budget_exceeded', 'param': None, 'code': '429'}}",
            id="message-only",
        ),
    ],
)
def test_permanent_spend_stop_is_detected(body, message):
    assert is_permanent_rate_limit_error(_make_rate_limit_error(body, message)) is True


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(THROTTLE_BODY, id="throughput-throttle"),
        pytest.param({"error": THROTTLE_BODY}, id="nested-throughput-throttle"),
        pytest.param(None, id="no-body"),
        pytest.param("plain text error page", id="non-mapping-body"),
    ],
)
def test_ordinary_rate_limit_is_not_a_spend_stop(body):
    assert is_permanent_rate_limit_error(_make_rate_limit_error(body)) is False


# ---------------------------------------------------------------------------
# Retry behaviour -- the fix proof
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_does_not_retry_budget_exceeded(monkeypatch, no_retry_wait):
    """One attempt, and the provider's own RateLimitError reaches the caller."""
    err = _make_rate_limit_error(BUDGET_BODY)
    fake_client = _make_chat_client(err)
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client", lambda **kw: fake_client
    )

    with pytest.raises(RateLimitError) as excinfo:
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert fake_client.chat.completions.create.await_count == 1
    # The actionable operator message survives instead of being wrapped away.
    assert "Budget has been exceeded" in str(excinfo.value)


@pytest.mark.asyncio
async def test_complete_does_not_retry_insufficient_quota(monkeypatch, no_retry_wait):
    """OpenAI's native exhausted-billing 429 is the same permanent stop."""
    fake_client = _make_chat_client(_make_rate_limit_error(QUOTA_BODY))
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client", lambda **kw: fake_client
    )

    with pytest.raises(RateLimitError):
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert fake_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_complete_still_retries_ordinary_rate_limit(monkeypatch, no_retry_wait):
    """Stability: a throughput 429 keeps its three attempts and RetryError."""
    err = _make_rate_limit_error(THROTTLE_BODY)
    fake_client = _make_chat_client(err)
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client", lambda **kw: fake_client
    )

    with pytest.raises(RetryError):
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert fake_client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_embed_does_not_retry_budget_exceeded(monkeypatch, no_retry_wait):
    """LiteLLM meters embedding spend too, so the embed binding needs the same gate."""
    fake_client = _FailingEmbeddingClient(_make_rate_limit_error(BUDGET_BODY))
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client", lambda **kw: fake_client
    )

    with pytest.raises(RateLimitError):
        await openai_embed(["hello"])

    assert fake_client.calls == 1


@pytest.mark.asyncio
async def test_embed_still_retries_ordinary_rate_limit(monkeypatch, no_retry_wait):
    fake_client = _FailingEmbeddingClient(_make_rate_limit_error(THROTTLE_BODY))
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client", lambda **kw: fake_client
    )

    with pytest.raises(RetryError):
        await openai_embed(["hello"])

    assert fake_client.calls == 3


# ---------------------------------------------------------------------------
# HTTP level -- the SDK has a retry loop of its own
# ---------------------------------------------------------------------------
#
# The tests above stub ``chat.completions.create``, so they never exercise the
# transport and cannot see the OpenAI SDK's own retries. The SDK defaults to
# ``max_retries=2`` and its ``_should_retry`` is body-blind (it retries every
# 429), so a permanent spend stop still cost three HTTP requests plus SDK
# backoff even with the tenacity predicate in place. ``create_openai_async_client``
# now builds the client with ``max_retries=0``; these tests count the requests
# that actually reach the wire.


def _counting_transport(
    counter: list[int], body: dict | None, status_code: int = 429
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        counter[0] += 1
        return httpx.Response(status_code, json={"error": body} if body else {})

    return httpx.MockTransport(handler)


def _client_factory(transport: httpx.MockTransport):
    """Build a fresh client per call, all sharing one counting transport.

    Every tenacity attempt calls ``create_openai_async_client`` anew, and the
    error handlers close the client they were given, so a single shared
    instance would fail the second attempt with a connection error instead of
    reaching the transport.
    """
    from lightrag.llm.openai import create_openai_async_client

    def factory(**_kwargs):
        return create_openai_async_client(
            api_key="test-key",
            base_url="https://proxy.example/v1",
            client_configs={"http_client": httpx.AsyncClient(transport=transport)},
        )

    return factory


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(BUDGET_BODY, id="budget-exceeded"),
        pytest.param(QUOTA_BODY, id="insufficient-quota"),
    ],
)
@pytest.mark.asyncio
async def test_spend_stop_costs_exactly_one_http_request(
    monkeypatch, no_retry_wait, body
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, body)),
    )

    with pytest.raises(RateLimitError):
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert calls[0] == 1


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(BUDGET_BODY, id="budget-exceeded"),
        pytest.param(QUOTA_BODY, id="insufficient-quota"),
    ],
)
@pytest.mark.asyncio
async def test_embed_spend_stop_costs_exactly_one_http_request(
    monkeypatch, no_retry_wait, body
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, body)),
    )

    with pytest.raises(RateLimitError):
        await openai_embed(["hello"])

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_ordinary_429_costs_exactly_the_tenacity_attempts(
    monkeypatch, no_retry_wait
):
    """With the SDK loop off, the request count is the tenacity attempt count.

    Previously this was 3 x 3: the outer loop multiplied with the SDK's own.
    """
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, THROTTLE_BODY)),
    )

    with pytest.raises(RetryError):
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert calls[0] == 3


def test_sdk_retries_are_off_by_default():
    from lightrag.llm.openai import create_openai_async_client

    assert create_openai_async_client(api_key="k").max_retries == 0


def test_client_configs_can_restore_sdk_retries():
    """Escape hatch: the SDK's Retry-After handling stays available on request."""
    from lightrag.llm.openai import create_openai_async_client

    client = create_openai_async_client(api_key="k", client_configs={"max_retries": 2})
    assert client.max_retries == 2


# ---------------------------------------------------------------------------
# Retry ownership -- what the SDK loop used to cover
# ---------------------------------------------------------------------------
#
# ``max_retries=0`` moves the whole retry decision to tenacity, so the outer
# loop must cover everything the SDK's ``_should_retry`` did. Connection
# errors, timeouts, 429 and >=500 already had predicates; 408 and 409 did not.
# ``_make_status_error`` maps 409 to ConflictError and has no branch for 408 at
# all (it arrives as a bare APIStatusError), so neither matched anything.


@pytest.mark.parametrize("status_code", [408, 409])
@pytest.mark.asyncio
async def test_complete_retries_transient_statuses(
    monkeypatch, no_retry_wait, status_code
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, None, status_code)),
    )

    with pytest.raises(RetryError):
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert calls[0] == 3


@pytest.mark.parametrize("status_code", [408, 409])
@pytest.mark.asyncio
async def test_embed_retries_transient_statuses(
    monkeypatch, no_retry_wait, status_code
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, None, status_code)),
    )

    with pytest.raises(RetryError):
        await openai_embed(["hello"])

    assert calls[0] == 3


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
@pytest.mark.asyncio
async def test_client_errors_still_fail_fast(monkeypatch, no_retry_wait, status_code):
    """Ownership transfer must not widen: the SDK never retried these either."""
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, None, status_code)),
    )

    with pytest.raises(APIStatusError):
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert calls[0] == 1
