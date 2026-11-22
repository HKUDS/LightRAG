"""The Ollama binding closes each owned HTTP client once on request failure."""

import httpx
import ollama
import pytest

from lightrag.llm.ollama import _ollama_model_if_cache, ollama_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _RejectedRequestTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.closed = 0
        self.paths = []

    async def handle_async_request(self, request):
        self.paths.append(request.url.path)
        return httpx.Response(400, json={"error": "bad request"})

    async def aclose(self):
        self.closed += 1


@pytest.mark.parametrize("operation", ["chat", "embed"])
async def test_request_failure_closes_owned_client_once(monkeypatch, operation):
    transport = _RejectedRequestTransport()
    create = ollama.AsyncClient
    close_calls = []

    def client_factory(**kwargs):
        client = create(transport=transport, **kwargs)
        close = client._client.aclose

        async def close_client():
            close_calls.append(client)
            await close()

        client._client.aclose = close_client
        return client

    monkeypatch.setattr(ollama, "AsyncClient", client_factory)

    with pytest.raises(ollama.ResponseError, match="bad request"):
        if operation == "chat":
            await _ollama_model_if_cache(model="test-model", prompt="hello")
        else:
            await ollama_embed.func(["hello"], embed_model="test-model")

    assert transport.paths == [f"/api/{operation}"]
    assert len(close_calls) == 1
    assert transport.closed == 1
