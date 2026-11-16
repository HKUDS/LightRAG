"""Client lifetime checks using the real SDK and an offline HTTP transport."""

import asyncio
import importlib
import threading

import httpx
import numpy as np
import pytest
import zai
from tenacity import wait_none

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture
def offline_sdk(monkeypatch):
    zhipu = importlib.import_module("lightrag.llm.zhipu")
    sdk_client = zai.ZhipuAiClient
    clients = []

    def install(respond):
        def make_client(**kwargs):
            # A retry must release the previous attempt's pool first.
            assert all(client.is_closed() for client in clients)
            transport = httpx.MockTransport(respond)
            client = sdk_client(**kwargs, http_client=httpx.Client(transport=transport))
            clients.append(client)
            return client

        monkeypatch.setattr(zai, "ZhipuAiClient", make_client)
        return clients

    monkeypatch.setattr(zhipu.zhipu_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(zhipu.zhipu_embedding.func.retry, "wait", wait_none())
    yield zhipu, install
    for client in clients:
        client.close()


def _success_response(request):
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [0.25, 0.75]}]}
        )
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        },
    )


async def _call(zhipu, operation):
    if operation == "chat":
        return await zhipu.zhipu_complete_if_cache("hello", api_key="test-key")
    return await zhipu.zhipu_embedding.func(["hello", "world"], api_key="test-key")


@pytest.mark.parametrize("operation", ["chat", "embedding"])
@pytest.mark.parametrize(
    "statuses",
    [(200,), (400,), (503, 200), (503, 503, 503)],
    ids=["success", "permanent-error", "retry-success", "retry-exhausted"],
)
async def test_zhipu_closes_every_attempt(offline_sdk, operation, statuses):
    zhipu, install = offline_sdk
    expected_attempts = len(statuses)
    statuses = list(statuses)
    requests = []

    def respond(request):
        requests.append(request)
        # Leave the successful status available for the second embedding text.
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        if status != 200:
            return httpx.Response(
                status, json={"error": {"code": str(status), "message": "unavailable"}}
            )
        return _success_response(request)

    clients = install(respond)
    final_status = statuses[-1]
    if final_status != 200:
        with pytest.raises(Exception) as exc:
            await _call(zhipu, operation)
        error = (
            exc.value.__cause__
            if operation == "embedding" and final_status == 400
            else exc.value
        )
        assert error.status_code == final_status
    else:
        result = await _call(zhipu, operation)
        if operation == "chat":
            assert result == "answer"
        else:
            np.testing.assert_array_equal(result, [[0.25, 0.75], [0.25, 0.75]])

    assert all(client.is_closed() for client in clients)
    assert len(clients) == expected_attempts
    assert len(requests) == expected_attempts + (
        operation == "embedding" and final_status == 200
    )


@pytest.mark.parametrize("operation", ["chat", "embedding"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
async def test_zhipu_cancel_closes_after_inflight_request(offline_sdk, operation, fail):
    zhipu, install = offline_sdk
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    requests = []
    close_threads = []

    def respond(request):
        requests.append(request)
        started.set()
        assert release.wait(timeout=5)
        if fail:
            raise httpx.ReadError("interrupted", request=request)
        return _success_response(request)

    clients = install(respond)
    task = asyncio.create_task(_call(zhipu, operation))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        client = clients[0]
        original_close = client.close

        def close():
            original_close()
            close_threads.append(threading.get_ident())
            closed.set()

        client.close = close
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.is_closed()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert await asyncio.to_thread(closed.wait, 5)
    assert clients[0].is_closed()
    assert len(requests) == 1
    assert len(close_threads) == 1
    assert close_threads[0] != threading.get_ident()
