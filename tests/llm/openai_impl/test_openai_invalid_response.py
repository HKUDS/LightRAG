"""Real-SDK regressions for non-completion responses from compatible gateways."""

from collections.abc import Callable

import httpx
import pytest
from openai import AsyncOpenAI
from tenacity import RetryError, wait_none

import lightrag.llm.openai as openai_module

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _completion() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "synthetic-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Recovered"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _mock_gateway(
    monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]
) -> tuple[list[httpx.Request], list[AsyncOpenAI]]:
    requests: list[httpx.Request] = []
    clients: list[AsyncOpenAI] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert responses, "Unexpected extra gateway request"
        return responses.pop(0)

    def create_client(**_kwargs: object) -> AsyncOpenAI:
        client = AsyncOpenAI(
            api_key="synthetic-not-a-credential",
            base_url="https://synthetic.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai_module, "create_openai_async_client", create_client)
    # Keep the production retry predicate and attempt limit, but no real waits.
    monkeypatch.setattr(
        openai_module.openai_complete_if_cache.retry, "wait", wait_none()
    )
    return requests, clients


@pytest.mark.parametrize(
    "invalid_response",
    [
        pytest.param(
            lambda: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>Synthetic gateway error</body></html>",
            ),
            id="html",
        ),
        pytest.param(
            lambda: httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="Synthetic upstream unavailable",
            ),
            id="plain-text",
        ),
        pytest.param(
            lambda: httpx.Response(200, json=["unexpected", "array"]),
            id="json-array",
        ),
        pytest.param(
            lambda: httpx.Response(200, json={"choices": []}),
            id="empty-choices-control",
        ),
    ],
)
async def test_invalid_gateway_response_retries_then_recovers(
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: Callable[[], httpx.Response],
) -> None:
    requests, clients = _mock_gateway(monkeypatch, [invalid_response(), _completion()])

    result = await openai_module.openai_complete_if_cache(
        model="synthetic-model", prompt="Synthetic chunk"
    )

    assert result == "Recovered"
    assert len(requests) == 2
    assert all(client.is_closed() for client in clients)


async def test_persistent_html_response_exhausts_existing_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, clients = _mock_gateway(
        monkeypatch,
        [
            httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html>Unavailable</html>",
            )
            for _ in range(3)
        ],
    )

    with pytest.raises(RetryError) as caught:
        await openai_module.openai_complete_if_cache(
            model="synthetic-model", prompt="Synthetic chunk"
        )

    assert isinstance(
        caught.value.last_attempt.exception(), openai_module.InvalidResponseError
    )
    assert len(requests) == 3
    assert all(client.is_closed() for client in clients)


async def test_valid_completion_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    requests, clients = _mock_gateway(monkeypatch, [_completion()])

    assert (
        await openai_module.openai_complete_if_cache(
            model="synthetic-model", prompt="Synthetic chunk"
        )
        == "Recovered"
    )
    assert len(requests) == 1
    assert clients[0].is_closed()
