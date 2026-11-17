"""LoLLMs raw-text streaming behavior through the HTTP adapter."""

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lightrag.llm.lollms import lollms_model_if_cache

pytestmark = pytest.mark.offline


@pytest.fixture
def http_resources(monkeypatch):
    sessions = []
    responses = []
    headers_received = asyncio.Event()
    client_session = aiohttp.ClientSession

    async def record_response(session, context, params):
        responses.append(params.response)
        headers_received.set()

    def create_session(*args, **kwargs):
        trace = aiohttp.TraceConfig()
        trace.on_request_end.append(record_response)
        session = client_session(*args, trace_configs=[trace], **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr("lightrag.llm.lollms.aiohttp.ClientSession", create_session)
    return sessions, responses, headers_received


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(
            '  # Answer\r\n\n```python\n    print("世界 🌍")  \n```\n\n  ',
            id="markdown-and-code",
        ),
        pytest.param("x" * 150_000, id="long-line"),
        pytest.param("", id="empty"),
        pytest.param(" \t\r\n  ", id="whitespace-only"),
    ],
)
async def test_stream_preserves_response_text(expected, http_resources):
    async def generate(request):
        assert (await request.json())["stream"] is True
        return web.Response(text=expected)

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        actual = "".join([part async for part in stream])

    assert actual == expected
    sessions, responses, _ = http_resources
    assert len(sessions) == len(responses) == 1
    assert sessions[0].closed
    assert responses[0].closed


async def test_stream_delivers_text_before_newline_or_eof():
    prefix_sent = asyncio.Event()
    allow_tail = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await response.write(b"first ")
        prefix_sent.set()
        await allow_tail.wait()
        await response.write(b"second\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        reader = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(prefix_sent.wait(), timeout=2)
            first = await asyncio.wait_for(asyncio.shield(reader), timeout=2)
            assert first == "first "
            allow_tail.set()
            assert first + "".join([part async for part in stream]) == "first second\n"
        finally:
            allow_tail.set()
            await asyncio.gather(reader, return_exceptions=True)
            await stream.aclose()


@pytest.mark.parametrize("character", ["世", "🌍"])
async def test_stream_decodes_utf8_split_across_http_chunks(character):
    prefix_sent = asyncio.Event()
    allow_tail = asyncio.Event()
    encoded = character.encode("utf-8")

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await response.write(b"head " + encoded[:1])
        prefix_sent.set()
        await allow_tail.wait()
        await response.write(encoded[1:] + b" tail")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        reader = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(prefix_sent.wait(), timeout=2)
            first = await asyncio.wait_for(asyncio.shield(reader), timeout=2)
            assert first == "head "
            allow_tail.set()
            parts = [part async for part in stream]
            assert all(parts)
            assert first + "".join(parts) == f"head {character} tail"
        finally:
            allow_tail.set()
            await asyncio.gather(reader, return_exceptions=True)
            await stream.aclose()


@pytest.mark.parametrize("payload", [b"\xff", b"\xf0\x9f"])
async def test_stream_rejects_invalid_or_truncated_utf8_and_closes(
    payload, http_resources
):
    async def generate(request):
        return web.Response(body=payload, content_type="text/plain")

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        with pytest.raises(UnicodeDecodeError):
            _ = [part async for part in stream]

    sessions, responses, _ = http_resources
    assert len(sessions) == len(responses) == 1
    assert sessions[0].closed
    assert responses[0].closed


async def test_stream_early_close_releases_response_and_session(http_resources):
    allow_finish = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await response.write(b"first")
        await allow_finish.wait()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        try:
            assert await asyncio.wait_for(anext(stream), timeout=2) == "first"
            await stream.aclose()
            sessions, responses, _ = http_resources
            assert len(sessions) == len(responses) == 1
            assert sessions[0].closed
            assert responses[0].closed
        finally:
            allow_finish.set()
            await stream.aclose()


async def test_stream_cancellation_releases_response_and_session(http_resources):
    allow_finish = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await allow_finish.wait()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        reader = asyncio.create_task(anext(stream))
        try:
            sessions, responses, headers_received = http_resources
            await asyncio.wait_for(headers_received.wait(), timeout=2)
            reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader
            assert len(sessions) == len(responses) == 1
            assert sessions[0].closed
            assert responses[0].closed
        finally:
            allow_finish.set()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await stream.aclose()
