"""Offline retry tests using the real Anthropic SDK and a fake HTTP service."""

import json

import httpx2
import pytest
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from tenacity import RetryError, wait_none

from lightrag.llm import anthropic as anthropic_binding

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture
def mock_anthropic_http(monkeypatch):
    clients = []

    def install(handler):
        def create_client(**kwargs):
            client = AsyncAnthropic(
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
                **kwargs,
            )
            clients.append(client)
            return client

        monkeypatch.setattr(anthropic_binding, "AsyncAnthropic", create_client)
        return clients

    return install


@pytest.mark.parametrize(
    "status,headers,expected_requests",
    [
        (429, {}, 3),
        (408, {}, 3),
        (409, {}, 3),
        (500, {}, 3),
        (529, {}, 3),
        (400, {}, 1),
        (429, {"x-should-retry": "false"}, 1),
        (400, {"x-should-retry": "true"}, 3),
    ],
)
async def test_http_error_has_one_sdk_retry_budget(
    mock_anthropic_http, status, headers, expected_requests
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            status,
            headers={"retry-after-ms": "1", **headers},
            json={
                "type": "error",
                "error": {"type": "api_error", "message": "synthetic failure"},
            },
        )

    clients = mock_anthropic_http(respond)
    complete = anthropic_binding.anthropic_complete_if_cache.retry_with(
        wait=wait_none()
    )
    with pytest.raises((APIStatusError, RetryError)) as caught:
        await complete(model="claude-test", prompt="hello", api_key="offline-key")

    assert len(requests) == expected_requests
    assert isinstance(caught.value, APIStatusError)
    assert caught.value.status_code == status
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("error_type", [httpx2.ConnectError, httpx2.ReadTimeout])
@pytest.mark.parametrize("stream", [False, True])
async def test_transport_failure_does_not_restart_sdk_retries(
    mock_anthropic_http, error_type, stream
):
    requests = []

    def respond(request):
        requests.append(request)
        raise error_type("synthetic connection failure", request=request)

    clients = mock_anthropic_http(respond)
    complete = anthropic_binding.anthropic_complete_if_cache.retry_with(
        wait=wait_none()
    )
    with pytest.raises((APIConnectionError, RetryError)) as caught:
        await complete(
            model="claude-test",
            prompt="hello",
            api_key="offline-key",
            base_url="https://anthropic.invalid",
            stream=stream,
        )

    assert len(requests) == 3
    assert isinstance(caught.value, APIConnectionError)
    assert all(client.is_closed() for client in clients)


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
        if len(requests) < 3:
            return httpx2.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "retry"},
                },
            )
        if not stream:
            return httpx2.Response(200, json=message)
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
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    clients = mock_anthropic_http(respond)
    result = await anthropic_binding.anthropic_complete_if_cache(
        model="claude-test", prompt="hello", api_key="offline-key", stream=stream
    )
    if stream:
        result = "".join([chunk async for chunk in result])

    assert result == "hello"
    assert len(requests) == 3
    assert all(client.is_closed() for client in clients)
