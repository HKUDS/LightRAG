"""Client-config headers must reach completion and embedding HTTP requests."""

import httpx
import pytest

from lightrag.llm.openai import (
    create_openai_async_client,
    openai_complete_if_cache,
    openai_embed,
)

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["completion", "embedding"])
async def test_client_config_headers_reach_requests(monkeypatch, operation):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "environment-workspace")
    captured = []

    def respond(request):
        captured.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.0] * 1536}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    headers = {
        "X-Custom-Route": "caller-route",
        "User-Agent": "caller-agent",
        "X-DashScope-Workspace": "caller-workspace",
    }
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    configs = {"default_headers": headers, "http_client": http_client}
    options = {"api_key": "test-key", "base_url": "https://example.invalid/v1"}
    try:
        if operation == "completion":
            assert (
                await openai_complete_if_cache(
                    "test-model", "hello", openai_client_configs=configs, **options
                )
                == "done"
            )
        else:
            result = await openai_embed(["hello"], client_configs=configs, **options)
            assert result.shape == (1, 1536)
    finally:
        await http_client.aclose()

    assert len(captured) == 1
    sent = captured[0].headers
    assert sent["X-Custom-Route"] == "caller-route"
    assert sent["User-Agent"] == "caller-agent"
    assert sent["X-DashScope-Workspace"] == "caller-workspace"
    assert sent["Content-Type"] == "application/json"
    assert configs["default_headers"] is headers
    assert headers == {
        "X-Custom-Route": "caller-route",
        "User-Agent": "caller-agent",
        "X-DashScope-Workspace": "caller-workspace",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configs", [None, {}, {"default_headers": None}, {"default_headers": {}}]
)
async def test_missing_custom_headers_keep_lightrag_defaults(monkeypatch, configs):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "  environment-workspace  ")
    async with create_openai_async_client(
        api_key="test-key", client_configs=configs
    ) as client:
        assert "LightRAG/" in client.default_headers["User-Agent"]
        assert client.default_headers["Content-Type"] == "application/json"
        assert (
            client.default_headers["X-DashScope-Workspace"] == "environment-workspace"
        )


@pytest.mark.asyncio
async def test_azure_custom_headers_are_preserved(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "environment-workspace")
    headers = {"X-Custom-Route": "azure-route"}
    async with create_openai_async_client(
        api_key="test-key",
        base_url="https://example.invalid",
        use_azure=True,
        api_version="2024-02-01",
        client_configs={"default_headers": headers},
    ) as client:
        assert client.default_headers["X-Custom-Route"] == "azure-route"
        assert "X-DashScope-Workspace" not in client.default_headers
    assert headers == {"X-Custom-Route": "azure-route"}
