"""Exercise retry ownership through the real SDK and an offline HTTP transport."""

import importlib
import json
import sys

import httpx
import pytest
from tenacity import wait_none

import lightrag.llm

sdk = pytest.importorskip("zhipuai")
pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture(params=["chat", "embedding"])
def call_sdk(request, monkeypatch):
    # Import afresh: other provider tests replace modules with test doubles.
    monkeypatch.setattr(lightrag.llm, "zhipu", None, raising=False)
    monkeypatch.setitem(sys.modules, "lightrag.llm.zhipu", None)
    monkeypatch.delitem(sys.modules, "lightrag.llm.zhipu")
    module = importlib.import_module("lightrag.llm.zhipu")
    monkeypatch.setattr(module.zhipu_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(module.zhipu_embedding.func.retry, "wait", wait_none())
    # Keep SDK retries active so accidentally re-enabling them changes counts.
    from zhipuai.core import _http_client

    if hasattr(_http_client.HttpClient, "_calculate_retry_timeout"):
        monkeypatch.setattr(
            _http_client.HttpClient, "_calculate_retry_timeout", lambda *a: 0
        )
    client_type = sdk.ZhipuAI
    clients = []
    requests = []
    responses = []

    async def invoke(outcomes, texts=None):
        def respond(http_request):
            requests.append(json.loads(http_request.content))
            outcome = outcomes[min(len(requests) - 1, len(outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == 200:
                body = {
                    "id": "test-response",
                    "created": 1,
                    "model": "test-model",
                    "object": "chat.completion" if request.param == "chat" else "list",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
                if request.param == "chat":
                    body["choices"] = [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                else:
                    body["data"] = [
                        {"object": "embedding", "embedding": [0.1] * 1024, "index": 0}
                    ]
                response = httpx.Response(200, json=body)
            else:
                status, body = outcome
                response = httpx.Response(status, json=body)
            responses.append(response)
            return response

        def factory(**kwargs):
            transport = httpx.Client(transport=httpx.MockTransport(respond))
            clients.append(transport)
            return client_type(http_client=transport, **kwargs)

        monkeypatch.setattr(sdk, "ZhipuAI", factory)
        if request.param == "chat":
            return await module.zhipu_complete_if_cache("hello", api_key="test.secret")
        return await module.zhipu_embedding(texts or ["hello"], api_key="test.secret")

    yield invoke, requests, responses
    for client in clients:
        client.close()


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_transient_status_retries_then_succeeds(call_sdk, status):
    invoke, requests, _ = call_sdk
    # Digits in a message are not an account error code.
    result = await invoke(
        [(status, {"error": {"code": "1302", "message": "id 1113"}}), 200]
    )
    assert len(requests) == 2
    assert result == "ok" if isinstance(result, str) else result.shape == (1, 1024)


@pytest.mark.parametrize(
    "status,code,error_name,attempts",
    [
        (400, "1210", "APIRequestFailedError", 1),
        (401, "1000", "APIAuthenticationError", 1),
        (429, "1113", "APIReachLimitError", 1),
        (429, "1309", "APIReachLimitError", 1),
        (429, "1308", "APIReachLimitError", 3),
        (429, "1310", "APIReachLimitError", 3),
        (429, "1302", "APIReachLimitError", 3),
        (500, "1200", "APIInternalError", 3),
        (503, "", "APIServerFlowExceedError", 3),
    ],
)
async def test_failure_preserves_native_error_and_exact_attempts(
    call_sdk, status, code, error_name, attempts
):
    invoke, requests, responses = call_sdk
    with pytest.raises(getattr(sdk, error_name)) as caught:
        await invoke([(status, {"error": {"code": code}})])
    assert len(requests) == attempts
    assert caught.value.response is responses[-1]
    assert caught.value.response.json()["error"]["code"] == code


@pytest.mark.parametrize("transport_error", [httpx.ReadTimeout, httpx.ConnectError])
async def test_transport_failure_has_three_attempts(call_sdk, transport_error):
    invoke, requests, _ = call_sdk
    original = transport_error("offline transport failure")
    expected = (
        sdk.APITimeoutError
        if transport_error is httpx.ReadTimeout
        else getattr(sdk, "APIConnectionError", httpx.ConnectError)
    )
    with pytest.raises(expected) as caught:
        await invoke([original])
    assert len(requests) == 3
    assert caught.value is original or caught.value.__cause__ is original


@pytest.mark.parametrize("call_sdk", ["embedding"], indirect=True)
async def test_embedding_batch_retry_replays_successful_prefix(call_sdk):
    invoke, requests, _ = call_sdk
    fail = (500, {"error": {"code": "1200"}})
    with pytest.raises(sdk.APIInternalError):
        await invoke([200, fail, 200, fail, 200, fail], texts=["first", "second"])
    assert [r["input"] for r in requests] == [["first"], ["second"]] * 3
