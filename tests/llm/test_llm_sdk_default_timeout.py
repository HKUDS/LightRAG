"""An LLM timeout <= 0 must leave each provider client at its SDK default.

``LLM_TIMEOUT=0`` means "LightRAG sets no timeout": the queue applies no cap
and the client keeps the provider SDK's own default -- never "wait forever"
(an explicit ``timeout=None`` reads as that in OpenAI-style SDKs) and never a
literal 0 (httpx-based SDKs time out at once on it). A positive value still
reaches the client. The SDK-backed cases build the real client, record what
it was given, and call a loopback stub with it.
"""

import contextlib

import openai
import pytest
from aiohttp import web

from lightrag.utils import provider_timeout

pytestmark = pytest.mark.offline

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

# (configured timeout, what the client must be given; None = SDK default)
_CASES = [
    # A negative value takes provider_timeout's ``<= 0`` branch, like zero
    # (test_provider_timeout[negative]); each provider keeps zero for its wiring.
    pytest.param(0, None, id="zero-sdk-default"),
    # Unset used to reach Anthropic, zai and LoLLMs as an explicit "no timeout".
    pytest.param(None, None, id="unset-sdk-default"),
    pytest.param(30, 30, id="positive-passed-on"),
]


@contextlib.asynccontextmanager
async def _slow_stub():
    async def answer(request: web.Request) -> web.StreamResponse:
        if request.path.endswith("/messages"):
            return web.json_response(_ANTHROPIC)
        if request.path.endswith("/api/chat"):
            return web.json_response(_OLLAMA)
        if request.path.endswith("/lollms_generate"):
            return web.Response(text="ok")
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


def _recording(monkeypatch, owner, name, calls: list[dict]):
    """Replace ``owner.name`` with a wrapper recording its kwargs."""
    real = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, None, id="zero"),
        pytest.param(-1, None, id="negative"),
        pytest.param(None, None, id="unset"),
        pytest.param(240, 240, id="positive"),
        pytest.param(False, False, id="bool-untouched"),
    ],
)
def test_provider_timeout(value, expected):
    assert provider_timeout(value) is expected or provider_timeout(value) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_openai_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import openai as openai_module

    calls: list[dict] = []
    _recording(monkeypatch, openai_module, "AsyncOpenAI", calls)

    async with _slow_stub() as url:
        result = await openai_module.openai_complete_if_cache(
            "m",
            "hi",
            base_url=f"{url}/v1",
            api_key="k",
            timeout=timeout,
            openai_client_configs={"max_retries": 0},
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        assert "timeout" not in calls[0]
        assert openai.AsyncOpenAI(api_key="k").timeout == openai.DEFAULT_TIMEOUT
    else:
        assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_anthropic_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import anthropic as anthropic_module

    calls: list[dict] = []
    _recording(monkeypatch, anthropic_module, "AsyncAnthropic", calls)

    async with _slow_stub() as url:
        result = await anthropic_module.anthropic_complete_if_cache(
            "m", "hi", base_url=url, api_key="k", timeout=timeout, max_retries=0
        )

    assert result == "ok"
    assert len(calls) == 1
    # An explicit timeout=None would mean "no timeout at all" to the SDK.
    if expected is None:
        assert "timeout" not in calls[0]
    else:
        assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_ollama_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import ollama as ollama_module

    calls: list[dict] = []
    _recording(monkeypatch, ollama_module.ollama, "AsyncClient", calls)

    async with _slow_stub() as url:
        result = await ollama_module._ollama_model_if_cache(
            "m", "hi", host=url, timeout=timeout
        )

    assert result == "ok"
    assert len(calls) == 1
    # None is the Ollama client's own default.
    assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_zhipu_request_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import zhipu as zhipu_module

    calls: list[dict] = []
    real_build = zhipu_module._build_client

    def build(client_cls, api_key):
        client = real_build(client_cls, api_key)
        _recording(monkeypatch, client.chat.completions, "create", calls)
        return client

    monkeypatch.setattr(zhipu_module, "_build_client", build)

    async with _slow_stub() as url:
        monkeypatch.setenv("ZAI_BASE_URL", f"{url}/api/")
        result = await zhipu_module.zhipu_complete_if_cache(
            "hi", model="m", api_key="id.secret", timeout=timeout
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        assert "timeout" not in calls[0]
    else:
        assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_lollms_session_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import lollms as lollms_module

    calls: list[dict] = []

    async with _slow_stub() as url:
        # Patched after the stub started, so only the client session counts.
        _recording(monkeypatch, lollms_module.aiohttp, "ClientSession", calls)
        result = await lollms_module.lollms_model_if_cache(
            "m", "hi", base_url=url, timeout=timeout
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        # aiohttp's session default; ClientTimeout(total=None) would be none.
        assert "timeout" not in calls[0]
    else:
        assert calls[0]["timeout"].total == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_openai_responses_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import openai_responses

    seen: dict = {}

    class _Stop(Exception):
        pass

    def _client(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(openai_responses, "create_openai_async_client", _client)

    with pytest.raises(_Stop):
        await openai_responses.openai_responses_complete_if_cache.__wrapped__(
            "m", "hi", api_key="k", timeout=timeout
        )

    # None is not passed on to AsyncOpenAI (see test_openai_client_timeout).
    assert seen["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _CASES)
async def test_gemini_http_timeout(monkeypatch, timeout, expected):
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

    assert seen["timeout_ms"] == (None if expected is None else expected * 1000)


@pytest.mark.parametrize(("timeout", "expected"), _CASES)
def test_bedrock_client_timeout(timeout, expected):
    from lightrag.llm.bedrock import _bedrock_client_kwargs

    client_kwargs = _bedrock_client_kwargs(
        region="us-east-1", endpoint_url=None, timeout=timeout
    )

    if expected is None:
        # botocore's defaults (60s connect / read) stay in effect.
        assert "config" not in client_kwargs
    else:
        config = client_kwargs["config"]
        assert (config.connect_timeout, config.read_timeout) == (expected, expected)
