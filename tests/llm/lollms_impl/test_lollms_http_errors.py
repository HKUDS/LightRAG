"""LoLLMs HTTP error statuses must raise, never become model output.

Driven through a real local server so the status check is the session's own.
Success paths are covered elsewhere: non-streaming generation by
``test_lollms_retry_predicate.py``, streaming by ``test_lollms_stream_text.py``
and embeddings by ``tests/llm/test_embedding_rerank_client_timeout.py``.
"""

from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientResponseError, web
from aiohttp.test_utils import TestServer
from tenacity import wait_none

from lightrag.llm.lollms import lollms_embed, lollms_model_if_cache

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Drop the 4s/8s sleeps; the request count is what these tests assert."""
    monkeypatch.setattr(lollms_model_if_cache.retry, "wait", wait_none())


@asynccontextmanager
async def _error_server(status, body):
    """Answer every request with ``status`` and a body that would otherwise
    pass for a valid result (plain text for generation, a vector for embed)."""
    requests = []

    async def respond(request):
        requests.append(request.path)
        await request.read()
        return web.Response(status=status, text=body, content_type="application/json")

    app = web.Application()
    app.router.add_post("/lollms_generate", respond)
    app.router.add_post("/lollms_embed", respond)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/"), requests


@pytest.mark.parametrize(
    ("status", "attempts"),
    [
        pytest.param(503, 3, id="transient-retried"),
        pytest.param(401, 1, id="permanent-fails-fast"),
    ],
)
async def test_generation_raises_on_error_status(status, attempts):
    async with _error_server(status, "provider error") as (base_url, requests):
        with pytest.raises(ClientResponseError) as exc_info:
            await lollms_model_if_cache("test-model", "hello", base_url=base_url)

    assert exc_info.value.status == status
    assert len(requests) == attempts


async def test_stream_raises_before_yielding_the_error_body():
    chunks = []
    async with _error_server(503, "provider error") as (base_url, requests):
        stream = await lollms_model_if_cache(
            "test-model", "hello", base_url=base_url, stream=True
        )
        with pytest.raises(ClientResponseError) as exc_info:
            async for chunk in stream:
                chunks.append(chunk)

    assert exc_info.value.status == 503
    assert chunks == []
    assert len(requests) == 1


async def test_embedding_raises_before_parsing_the_body():
    """The error body is a well-formed vector: only the status check stops it
    from being returned as an embedding."""
    async with _error_server(503, '{"vector": [1.0, 2.0]}') as (base_url, _):
        with pytest.raises(ClientResponseError) as exc_info:
            await lollms_embed.func(["hello"], base_url=base_url)

    assert exc_info.value.status == 503
