"""A disabled LLM timeout (<= 0) must reach each provider client as "no timeout".

``LLM_TIMEOUT=0`` disables the queue's execution cap, so the provider client
must not keep a 0 or negative timeout either: httpx-based SDKs (OpenAI,
Anthropic, Ollama, zai) time out at once on it. The SDK-backed cases call a
real client against a loopback stub that answers after a short delay.
"""

import asyncio
import contextlib

import httpx
import pytest
from aiohttp import web

from lightrag.utils import timeout_disabled

pytestmark = pytest.mark.offline

_DELAY = 0.3
_CHAT = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_ANTHROPIC = {
    "id": "x",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_OLLAMA = {
    "model": "m",
    "created_at": "2026-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
}

_DISABLED = [pytest.param(0, id="zero"), pytest.param(-1, id="negative")]


@contextlib.asynccontextmanager
async def _slow_stub():
    async def answer(request: web.Request) -> web.Response:
        await asyncio.sleep(_DELAY)
        if request.path.endswith("/messages"):
            return web.json_response(_ANTHROPIC)
        if request.path.endswith("/api/chat"):
            return web.json_response(_OLLAMA)
        return web.json_response(_CHAT)

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, True, id="zero"),
        pytest.param(0.0, True, id="zero-float"),
        pytest.param(-1, True, id="negative"),
        pytest.param(240, False, id="positive"),
        pytest.param(None, False, id="unset"),
        pytest.param(False, False, id="bool"),
        pytest.param(httpx.Timeout(None), False, id="httpx-timeout"),
    ],
)
def test_timeout_disabled(value, expected):
    assert timeout_disabled(value) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", _DISABLED)
async def test_openai_disabled_timeout_waits_for_the_answer(timeout):
    from lightrag.llm.openai import openai_complete_if_cache

    async with _slow_stub() as url:
        result = await openai_complete_if_cache(
            "m",
            "hi",
            base_url=f"{url}/v1",
            api_key="k",
            timeout=timeout,
            openai_client_configs={"max_retries": 0},
        )

    assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", _DISABLED)
async def test_anthropic_disabled_timeout_waits_for_the_answer(timeout):
    from lightrag.llm.anthropic import anthropic_complete_if_cache

    async with _slow_stub() as url:
        result = await anthropic_complete_if_cache(
            "m", "hi", base_url=url, api_key="k", timeout=timeout, max_retries=0
        )

    assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", _DISABLED)
async def test_ollama_disabled_timeout_waits_for_the_answer(timeout):
    from lightrag.llm.ollama import _ollama_model_if_cache

    async with _slow_stub() as url:
        result = await _ollama_model_if_cache("m", "hi", host=url, timeout=timeout)

    assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", _DISABLED)
async def test_zhipu_disabled_timeout_waits_for_the_answer(monkeypatch, timeout):
    from lightrag.llm.zhipu import zhipu_complete_if_cache

    async with _slow_stub() as url:
        monkeypatch.setenv("ZAI_BASE_URL", f"{url}/api/")
        result = await zhipu_complete_if_cache(
            "hi", model="m", api_key="id.secret", timeout=timeout
        )

    assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", _DISABLED)
async def test_openai_responses_disabled_timeout_reaches_client_as_none(
    monkeypatch, timeout
):
    from lightrag.llm import openai_responses

    seen: dict = {}

    class _Stop(Exception):
        pass

    def _client(**kwargs):
        seen["client_timeout"] = kwargs.get("timeout")
        raise _Stop

    monkeypatch.setattr(openai_responses, "create_openai_async_client", _client)

    with pytest.raises(_Stop):
        await openai_responses.openai_responses_complete_if_cache.__wrapped__(
            "m", "hi", api_key="k", timeout=timeout
        )

    client_timeout = seen["client_timeout"]
    assert isinstance(client_timeout, httpx.Timeout)
    assert (client_timeout.connect, client_timeout.read) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", _DISABLED)
async def test_gemini_disabled_timeout_sets_no_http_timeout(monkeypatch, timeout):
    from lightrag.llm import gemini

    seen: dict = {}

    class _Stop(Exception):
        pass

    def _client(api_key, base_url, timeout_ms):
        seen["timeout_ms"] = timeout_ms
        raise _Stop

    monkeypatch.setattr(gemini, "_get_gemini_client", _client)

    with pytest.raises(_Stop):
        await gemini.gemini_complete_if_cache.__wrapped__(
            "m", "hi", api_key="k", timeout=timeout
        )

    assert seen["timeout_ms"] is None


@pytest.mark.parametrize("timeout", _DISABLED)
def test_bedrock_disabled_timeout_means_no_read_timeout(timeout):
    from lightrag.llm.bedrock import _bedrock_client_kwargs

    config = _bedrock_client_kwargs(
        region="us-east-1", endpoint_url=None, timeout=timeout
    )["config"]

    # aiohttp, under aiobotocore, reads 0 as "no timeout".
    assert (config.connect_timeout, config.read_timeout) == (0, 0)
