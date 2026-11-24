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

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    RateLimitError,
)
from tenacity import wait_none

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
        # One case per allow-list member and one per body shape; the two axes
        # are independent, so their cross product is not needed. The body
        # cases carry a message without the slug, so the rendered-message
        # fallback cannot answer for the body probe.
        pytest.param(BUDGET_BODY, "Error code: 429", id="budget-unwrapped-body"),
        pytest.param(
            {"error": QUOTA_BODY}, "Error code: 429", id="quota-nested-envelope"
        ),
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
        pytest.param({"error": THROTTLE_BODY}, id="nested-throughput-throttle"),
        pytest.param(None, id="no-body"),
        pytest.param("plain text error page", id="non-mapping-body"),
    ],
)
def test_ordinary_rate_limit_is_not_a_spend_stop(body):
    assert is_permanent_rate_limit_error(_make_rate_limit_error(body)) is False


# ---------------------------------------------------------------------------
# Retry behaviour, counted at the HTTP level
# ---------------------------------------------------------------------------
#
# These tests count the requests that actually reach the wire, so they see the
# OpenAI SDK's own retry loop as well as tenacity's. The SDK defaults to
# ``max_retries=2`` and its ``_should_retry`` is body-blind (it retries every
# 429), so a permanent spend stop would cost three requests even with the
# tenacity predicate in place; ``create_openai_async_client`` builds the client
# with ``max_retries=0``, and a count of exactly one pins both.


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


async def _call_complete():
    return await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")


async def _call_embed():
    return await openai_embed(["hello"])


# Which permanent type each entry point gets does not matter: the detection
# table above covers every member, and both decorators share one predicate.
@pytest.mark.parametrize(
    "call, body",
    [
        pytest.param(_call_complete, BUDGET_BODY, id="complete-budget-exceeded"),
        pytest.param(_call_embed, QUOTA_BODY, id="embed-insufficient-quota"),
    ],
)
@pytest.mark.asyncio
async def test_spend_stop_costs_exactly_one_http_request(
    monkeypatch, no_retry_wait, call, body
):
    """One request, and the provider's own RateLimitError reaches the caller."""
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, body)),
    )

    with pytest.raises(RateLimitError) as excinfo:
        await call()

    assert calls[0] == 1
    # The actionable operator message survives instead of being wrapped away.
    assert body["message"] in str(excinfo.value)


@pytest.mark.parametrize(
    "call", [_call_complete, _call_embed], ids=["complete", "embed"]
)
@pytest.mark.asyncio
async def test_ordinary_429_costs_exactly_the_tenacity_attempts(
    monkeypatch, no_retry_wait, call
):
    """A throughput 429 keeps its three attempts, then the original
    RateLimitError -- not a tenacity.RetryError -- reaches the caller.

    With the SDK loop off, the request count is the tenacity attempt count;
    it used to be 3 x 3, the outer loop multiplied with the SDK's own.
    """
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, THROTTLE_BODY)),
    )

    with pytest.raises(RateLimitError) as excinfo:
        await call()

    assert calls[0] == 3
    assert "Rate limit reached" in str(excinfo.value)


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

    with pytest.raises(APIStatusError) as excinfo:
        await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")

    assert calls[0] == 3
    assert excinfo.value.status_code == status_code


# Both members are pinned on completion above; embed only has to prove that its
# decorator carries the same predicate.
@pytest.mark.parametrize("status_code", [408])
@pytest.mark.asyncio
async def test_embed_retries_transient_statuses(
    monkeypatch, no_retry_wait, status_code
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, None, status_code)),
    )

    with pytest.raises(APIStatusError) as excinfo:
        await openai_embed(["hello"])

    assert calls[0] == 3
    assert excinfo.value.status_code == status_code


@pytest.mark.parametrize("status_code", [400, 422])
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


# ---------------------------------------------------------------------------
# Exhausted retries surface the provider's own error, not tenacity.RetryError
# ---------------------------------------------------------------------------


def _disconnecting_transport(counter: list[int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        counter[0] += 1
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "call", [_call_complete, _call_embed], ids=["complete", "embed"]
)
@pytest.mark.asyncio
async def test_exhausted_5xx_reraises_internal_server_error(
    monkeypatch, no_retry_wait, call
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(
            _counting_transport(calls, {"message": "upstream connect error"}, 503)
        ),
    )

    with pytest.raises(InternalServerError) as excinfo:
        await call()

    assert calls[0] == 3
    assert excinfo.value.status_code == 503
    assert "upstream connect error" in str(excinfo.value)


@pytest.mark.parametrize(
    "call", [_call_complete, _call_embed], ids=["complete", "embed"]
)
@pytest.mark.asyncio
async def test_exhausted_connection_error_reraises_api_connection_error(
    monkeypatch, no_retry_wait, call
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_disconnecting_transport(calls)),
    )

    with pytest.raises(APIConnectionError):
        await call()

    assert calls[0] == 3
