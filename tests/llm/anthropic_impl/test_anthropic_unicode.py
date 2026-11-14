"""Exercise escaped Unicode through the real SDK and ASGI stream encoder."""

import json
from functools import partial

import httpx
import pytest
from anthropic import AsyncAnthropic
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

from lightrag.llm import anthropic

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_surrogate_pair_in_anthropic_stream_is_utf8_encodable(monkeypatch):
    event = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": r"Answer: \ud83d\ude00"},
    }
    sse = f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"

    def provider_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode()
        )

    provider_http = httpx.AsyncClient(transport=httpx.MockTransport(provider_response))
    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", partial(AsyncAnthropic, http_client=provider_http)
    )

    async def answer(request):
        stream = await anthropic.anthropic_complete_if_cache(
            model="claude-test", prompt="Hello", api_key="test-key", stream=True
        )
        return StreamingResponse(stream, media_type="text/plain")

    app = Starlette(routes=[Route("/", answer)])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as consumer:
        response = await consumer.get("/")

    assert response.status_code == 200
    assert response.content == "Answer: 😀".encode("utf-8")
    assert provider_http.is_closed
