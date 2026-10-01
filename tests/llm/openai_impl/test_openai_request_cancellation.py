"""验证请求尚未返回时取消任务仍会释放真实 SDK 客户端。"""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest

from lightrag.llm import openai as provider


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_azure", [False, True])
async def test_cancel_pending_request_closes_client(monkeypatch, stream, use_azure):
    received = asyncio.Event()
    clients = []
    original_factory = provider.create_openai_async_client

    def capture_client(**kwargs):
        client = original_factory(**kwargs)
        clients.append(client)
        return client

    async def handle(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            received.set()
            await reader.read()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(provider, "create_openai_async_client", capture_client)
    task = asyncio.create_task(
        provider.openai_complete_if_cache(
            model="test",
            prompt="hello",
            api_key="test",
            base_url=f"http://127.0.0.1:{port}/v1",
            stream=stream,
            use_azure=use_azure,
            api_version="2024-02-01",
        )
    )
    try:
        await asyncio.wait_for(received.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(clients) == 1
        assert clients[0].is_closed()
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        for client in clients:
            await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cleanup_error_does_not_replace_cancellation(monkeypatch):
    from types import SimpleNamespace

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=asyncio.CancelledError)
            )
        ),
        close=AsyncMock(side_effect=RuntimeError("cleanup failed")),
    )
    monkeypatch.setattr(provider, "create_openai_async_client", lambda **kwargs: client)
    with pytest.raises(asyncio.CancelledError):
        await provider.openai_complete_if_cache(model="test", prompt="hello")
    client.close.assert_awaited_once()
