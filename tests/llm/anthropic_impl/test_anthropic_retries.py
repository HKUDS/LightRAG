"""Offline retry tests using the real Anthropic SDK and a fake HTTP service.

The SDK's own retry loop is the binding's only retry layer, so every request
counted here went through the SDK's real retry decision, backoff calculation,
response parsing and client cleanup.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic._base_client as anthropic_base_client
import pytest
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from tenacity import RetryError

from lightrag.llm import anthropic as anthropic_binding

pytestmark = pytest.mark.offline

# The SDK only accepts an HTTP client from the package it is built on: httpx2
# in current releases (the locked 2.6.0), httpx in older ones such as 0.50.0.
# pyproject.toml still allows those, so take the package from the SDK instead
# of importing one.
http = getattr(anthropic_base_client, "httpx2", None) or anthropic_base_client.httpx

# One initial request plus the binding's default SDK retry budget.
DEFAULT_ATTEMPTS = anthropic_binding._DEFAULT_MAX_RETRIES + 1


@pytest.fixture
def mock_anthropic_http(monkeypatch):
    clients = []
    # Responses without a retry-after header fall back to the SDK's
    # exponential backoff (0.5 s, 1 s, 2 s, 4 s); shrink it so the suite stays
    # fast without replacing the SDK's retry decision.
    monkeypatch.setattr(anthropic_base_client, "INITIAL_RETRY_DELAY", 0.001)

    def install(handler):
        def create_client(**kwargs):
            client = AsyncAnthropic(
                http_client=http.AsyncClient(transport=http.MockTransport(handler)),
                **kwargs,
            )
            clients.append(client)
            return client

        monkeypatch.setattr(anthropic_binding, "AsyncAnthropic", create_client)
        return clients

    return install


def _error_response(status, headers=None):
    return http.Response(
        status,
        headers=headers or {},
        json={
            "type": "error",
            "error": {"type": "api_error", "message": "synthetic failure"},
        },
    )


def test_outer_tenacity_layer_is_gone():
    """The SDK owns retries; an outer tenacity loop would multiply them."""
    assert not hasattr(anthropic_binding.anthropic_complete_if_cache, "retry")
    assert not hasattr(anthropic_binding.anthropic_complete_if_cache, "__wrapped__")


@pytest.mark.parametrize(
    "status,headers,expected_requests",
    [
        (429, {}, DEFAULT_ATTEMPTS),
        (429, {"retry-after-ms": "1"}, DEFAULT_ATTEMPTS),
        (408, {}, DEFAULT_ATTEMPTS),
        (409, {}, DEFAULT_ATTEMPTS),
        (500, {}, DEFAULT_ATTEMPTS),
        (529, {}, DEFAULT_ATTEMPTS),
        (400, {}, 1),
        (429, {"x-should-retry": "false"}, 1),
        (400, {"x-should-retry": "true"}, DEFAULT_ATTEMPTS),
    ],
)
async def test_http_error_has_one_sdk_retry_budget(
    mock_anthropic_http, status, headers, expected_requests
):
    requests = []

    def respond(request):
        requests.append(request)
        return _error_response(status, headers)

    clients = mock_anthropic_http(respond)
    with pytest.raises((APIStatusError, RetryError)) as caught:
        await anthropic_binding.anthropic_complete_if_cache(
            model="claude-test", prompt="hello", api_key="offline-key"
        )

    assert len(requests) == expected_requests
    assert isinstance(caught.value, APIStatusError)
    assert caught.value.status_code == status
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("error_type", ["ConnectError", "ReadTimeout"])
@pytest.mark.parametrize("stream", [False, True])
async def test_transport_failure_does_not_restart_sdk_retries(
    mock_anthropic_http, error_type, stream
):
    requests = []

    def respond(request):
        requests.append(request)
        raise getattr(http, error_type)("synthetic connection failure", request=request)

    clients = mock_anthropic_http(respond)
    with pytest.raises((APIConnectionError, RetryError)) as caught:
        await anthropic_binding.anthropic_complete_if_cache(
            model="claude-test",
            prompt="hello",
            api_key="offline-key",
            base_url="https://anthropic.invalid",
            stream=stream,
        )

    assert len(requests) == DEFAULT_ATTEMPTS
    assert isinstance(caught.value, APIConnectionError)
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("max_retries,expected_requests", [(0, 1), (1, 2)])
async def test_max_retries_overrides_the_sdk_budget(
    mock_anthropic_http, max_retries, expected_requests
):
    requests = []

    def respond(request):
        requests.append(request)
        return _error_response(529)

    clients = mock_anthropic_http(respond)
    with pytest.raises(APIStatusError):
        await anthropic_binding.anthropic_complete_if_cache(
            model="claude-test",
            prompt="hello",
            api_key="offline-key",
            max_retries=max_retries,
        )

    assert len(requests) == expected_requests
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize(
    "call_kwargs,expected_max_retries",
    [
        ({}, anthropic_binding._DEFAULT_MAX_RETRIES),
        ({"max_retries": None}, anthropic_binding._DEFAULT_MAX_RETRIES),
        ({"max_retries": 0}, 0),
        ({"base_url": "https://proxy.invalid", "max_retries": 7}, 7),
    ],
)
async def test_max_retries_is_a_client_option_not_a_request_parameter(
    call_kwargs, expected_max_retries
):
    message = MagicMock()
    message.content = [MagicMock(text="hello")]
    message.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    client.close = AsyncMock()

    with patch(
        "lightrag.llm.anthropic.AsyncAnthropic", return_value=client
    ) as client_cls:
        result = await anthropic_binding.anthropic_complete_if_cache(
            model="claude-test", prompt="hello", api_key="offline-key", **call_kwargs
        )

    assert result == "hello"
    assert client_cls.call_args.kwargs["max_retries"] == expected_max_retries
    assert "max_retries" not in client.messages.create.await_args.kwargs


@pytest.mark.parametrize("stream", [False, True])
async def test_transient_rate_limit_recovers_within_sdk_budget(
    mock_anthropic_http, stream
):
    requests = []
    message = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    def respond(request):
        requests.append(request)
        if len(requests) < DEFAULT_ATTEMPTS:
            return http.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "retry"},
                },
            )
        if not stream:
            return http.Response(200, json=message)
        events = [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "message_stop"},
        ]
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return http.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    clients = mock_anthropic_http(respond)
    result = await anthropic_binding.anthropic_complete_if_cache(
        model="claude-test", prompt="hello", api_key="offline-key", stream=stream
    )
    if stream:
        result = "".join([chunk async for chunk in result])

    assert result == "hello"
    assert len(requests) == DEFAULT_ATTEMPTS
    assert all(client.is_closed() for client in clients)
