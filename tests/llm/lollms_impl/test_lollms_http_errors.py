"""LoLLMs HTTP failures must not become successful model output."""

from contextlib import asynccontextmanager

import numpy as np
import pytest
from aiohttp import ClientResponseError, web
from aiohttp.test_utils import TestServer

from lightrag.llm.lollms import lollms_embed, lollms_model_if_cache

pytestmark = pytest.mark.offline


@asynccontextmanager
async def _server_response(*, status, body, content_type="text/plain"):
    async def respond(request):
        await request.read()
        return web.Response(status=status, text=body, content_type=content_type)

    app = web.Application()
    app.router.add_post("/lollms_generate", respond)
    app.router.add_post("/lollms_embed", respond)
    async with TestServer(app) as server:
        yield str(server.make_url("/")).rstrip("/")


@pytest.mark.parametrize("status", [401, 429, 503])
@pytest.mark.parametrize("stream", [False, True])
async def test_generation_rejects_http_errors(status, stream):
    chunks = []
    async with _server_response(status=status, body="provider error") as base_url:
        with pytest.raises(ClientResponseError) as exc_info:
            result = await lollms_model_if_cache(
                "test-model", "hello", base_url=base_url, stream=stream
            )
            if stream:
                async for chunk in result:
                    chunks.append(chunk)

    assert status == exc_info.value.status
    assert [] == chunks


@pytest.mark.parametrize("stream", [False, True])
async def test_generation_preserves_successful_responses(stream):
    async with _server_response(status=200, body="model answer") as base_url:
        result = await lollms_model_if_cache(
            "test-model", "hello", base_url=base_url, stream=stream
        )
        if stream:
            result = "".join([chunk async for chunk in result])

    assert "model answer" == result


@pytest.mark.parametrize(
    ("status", "body", "content_type"),
    [
        (401, '{"error": "unauthorized"}', "application/json"),
        (429, '{"vector": [1.0, 2.0]}', "application/json"),
        (503, "<html>Service Unavailable</html>", "text/html"),
    ],
)
async def test_embedding_rejects_http_errors_before_parsing(status, body, content_type):
    async with _server_response(
        status=status, body=body, content_type=content_type
    ) as base_url:
        with pytest.raises(ClientResponseError) as exc_info:
            await lollms_embed.func(["hello"], base_url=base_url)

    assert status == exc_info.value.status
    assert ClientResponseError is type(exc_info.value)


async def test_embedding_preserves_successful_responses():
    async with _server_response(
        status=200, body='{"vector": [1.0, 2.0]}', content_type="application/json"
    ) as base_url:
        result = await lollms_embed.func(["first", "second"], base_url=base_url)

    np.testing.assert_array_equal([[1.0, 2.0], [1.0, 2.0]], result)
