"""The Ollama binding closes each client it creates exactly once on failure.

A failed request has one owner for its client: the binding call itself until a
stream is handed to the caller, the returned stream generator after that. These
cases use the real SDK over a local transport and count the binding's own
``aclose()`` calls; httpx makes a second close a no-op, so the transport alone
cannot tell a double close from a single one.
"""

import httpx
import ollama
import pydantic
import pytest

from lightrag.llm.ollama import _ollama_model_if_cache, ollama_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


async def _chat(**kwargs):
    return await _ollama_model_if_cache(model="test-model", prompt="hello", **kwargs)


async def _chat_stream_consumed():
    stream = await _chat(stream=True)
    return [part async for part in stream]


async def _chat_stream_setup_fails():
    # An empty model fails the SDK's request validation inside the binding's
    # try block, after the client exists and before it is handed to a stream.
    return await _ollama_model_if_cache(model="", prompt="hello", stream=True)


async def _embed():
    return await ollama_embed.func(["hello"], embed_model="test-model")


@pytest.mark.parametrize(
    ("operation", "error", "paths"),
    [
        pytest.param(_chat, ollama.ResponseError, ["/api/chat"], id="chat"),
        pytest.param(
            _chat_stream_consumed, ollama.ResponseError, ["/api/chat"], id="stream"
        ),
        pytest.param(
            _chat_stream_setup_fails,
            pydantic.ValidationError,
            [],
            id="stream-setup",
        ),
        pytest.param(_embed, ollama.ResponseError, ["/api/embed"], id="embed"),
    ],
)
async def test_request_failure_closes_owned_client_once(
    monkeypatch, operation, error, paths
):
    requested = []
    close_calls = []
    create = ollama.AsyncClient

    def handle(request):
        requested.append(request.url.path)
        return httpx.Response(400, json={"error": "bad request"})

    def client_factory(**kwargs):
        client = create(transport=httpx.MockTransport(handle), **kwargs)
        close = client._client.aclose

        async def close_client():
            close_calls.append(client)
            await close()

        client._client.aclose = close_client
        return client

    monkeypatch.setattr(ollama, "AsyncClient", client_factory)

    with pytest.raises(error):
        await operation()

    assert requested == paths
    assert len(close_calls) == 1
