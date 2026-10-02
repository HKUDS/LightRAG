"""Offline regressions for LoLLMs' raw-text streaming responses."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from lightrag.llm.lollms import lollms_model_if_cache

pytestmark = pytest.mark.offline


@pytest.fixture
def lollms_server(monkeypatch):
    """Run a local HTTP fixture and check every real client session is closed."""
    sessions = []
    client_session = aiohttp.ClientSession

    def track_session(*args, **kwargs):
        session = client_session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr("lightrag.llm.lollms.aiohttp.ClientSession", track_session)

    @asynccontextmanager
    async def serve(handler):
        app = web.Application()
        app.router.add_post("/lollms_generate", handler)
        async with TestServer(app, host="127.0.0.1") as server:
            try:
                yield str(server.make_url("/")).rstrip("/"), sessions
            finally:
                assert sessions
                assert all(session.closed for session in sessions)

    return serve


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("# Heading\n\n- first item  \n- second item\n", id="markdown"),
        pytest.param(
            "```python\nif ready:\n    print('hello')\n\treturn True\n```\n",
            id="indented-code",
        ),
        pytest.param("  leading\r\n\r\ntrailing  ", id="crlf-and-spaces"),
        pytest.param(" \t\n\n  ", id="whitespace-only"),
        pytest.param("café 世界 🌍\n", id="unicode"),
        pytest.param("x" * 150_000, id="long-line"),
        pytest.param("", id="empty"),
    ],
)
async def test_stream_matches_non_stream_text(lollms_server, text):
    async def generate(request):
        await request.json()
        return web.Response(text=text)

    async with lollms_server(generate) as (base_url, sessions):
        complete = await lollms_model_if_cache("test-model", "hello", base_url=base_url)
        stream = await lollms_model_if_cache(
            "test-model", "hello", base_url=base_url, stream=True
        )
        chunks = [chunk async for chunk in stream]

        assert "".join(chunks) == complete == text
        assert all(chunks)
        assert len(sessions) == 2


@pytest.mark.parametrize("close_early", [False, True])
async def test_stream_yields_before_newline_or_eof(lollms_server, close_early):
    finish = asyncio.Event()

    async def generate(request):
        await request.json()
        response = web.StreamResponse(
            headers={"Content-Type": "text/plain; charset=utf-8"}
        )
        await response.prepare(request)
        await response.write(b"first chunk ")
        await finish.wait()
        return response

    async with lollms_server(generate) as (base_url, sessions):
        stream = await lollms_model_if_cache(
            "test-model", "hello", base_url=base_url, stream=True
        )
        try:
            assert await asyncio.wait_for(anext(stream), timeout=2) == "first chunk "
            assert len(sessions) == 1
            assert not sessions[0].closed
            if close_early:
                await stream.aclose()
                assert sessions[0].closed
            else:
                finish.set()
                assert [chunk async for chunk in stream] == []
        finally:
            finish.set()
            await stream.aclose()


@pytest.fixture
def chunked_response(monkeypatch):
    """Control byte boundaries without relying on TCP packet boundaries."""

    def make(chunks):
        async def content():
            for chunk in chunks:
                yield chunk

        response = MagicMock()
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        response.content.__aiter__.side_effect = content
        response.content.iter_any.side_effect = content
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post.return_value = response
        monkeypatch.setattr(
            "lightrag.llm.lollms.aiohttp.ClientSession", lambda **kwargs: session
        )
        return session, response

    return make


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(
            [bytes([byte]) for byte in "café 世界 🌍".encode()], id="one-byte"
        ),
        pytest.param([b"\xe2", b"\x82", b"\xac"], id="split-final-character"),
        pytest.param([b"left", b" ", b"right\n"], id="whitespace-chunk"),
    ],
)
async def test_stream_decodes_across_byte_boundaries(chunked_response, chunks):
    session, response = chunked_response(chunks)
    stream = await lollms_model_if_cache("test-model", "hello", stream=True)

    result = [chunk async for chunk in stream]

    assert "".join(result) == b"".join(chunks).decode("utf-8")
    assert all(result)
    response.__aexit__.assert_awaited_once()
    session.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("chunks", [[b"\xff"], [b"valid", b"\xe2\x82"]])
async def test_invalid_utf8_raises_and_closes_stream(chunked_response, chunks):
    session, response = chunked_response(chunks)
    stream = await lollms_model_if_cache("test-model", "hello", stream=True)

    with pytest.raises(UnicodeDecodeError):
        _ = [chunk async for chunk in stream]

    response.__aexit__.assert_awaited_once()
    session.__aexit__.assert_awaited_once()
