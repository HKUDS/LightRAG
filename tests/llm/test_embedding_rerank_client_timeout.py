"""EMBEDDING_TIMEOUT / RERANK_TIMEOUT must reach the embedding and rerank clients.

``default_embedding_timeout`` / ``default_rerank_timeout`` used to size only
LightRAG's own queues, so every client ran with its SDK default whatever was
configured. Now LightRAG binds the effective value around each call it
executes, and every bundled binding hands it to its client with the rules the
LLM path uses: a positive value is the client's request timeout, 0 or a
negative value (or no LightRAG call at all) keeps the SDK default, and the
binding's own explicit ``timeout`` wins.

The HTTP bindings build their real client against a loopback stub and record
what it was given; the others record the client constructor's arguments.
"""

import asyncio
import base64
import contextlib
import functools
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from tenacity import wait_none

from lightrag import LightRAG
from lightrag.llm._aiohttp_retry import is_transient_aiohttp_error
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    bind_embedding_timeout,
    bind_rerank_timeout,
    embedding_timeout,
    rerank_timeout,
)

pytestmark = pytest.mark.offline

_VECTOR = [0.25, 0.5, 0.75, 1.0]
_OMITTED = "omitted"


class _Recorded(Exception):
    """Raised by a recording fake once it has seen the client's arguments."""


# --------------------------------------------------------------------------
# Loopback stub
# --------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _stub(*, first_delay: float = 0.0):
    """Answer every bundled HTTP binding; the first request waits ``first_delay``."""
    hits: list[str] = []

    async def answer(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        if len(hits) == 1 and first_delay:
            await asyncio.sleep(first_delay)
        path = request.path
        if path.endswith("/api/embed"):
            return web.json_response({"model": "m", "embeddings": [_VECTOR]})
        if path.endswith("/lollms_embed"):
            return web.json_response({"vector": _VECTOR})
        if path.endswith("/rerank"):
            return web.json_response({"results": [{"index": 0, "relevance_score": 1}]})
        if path.endswith("/jina/embeddings"):
            encoded = base64.b64encode(np.array(_VECTOR, np.float32).tobytes())
            return web.json_response({"data": [{"embedding": encoded.decode()}]})
        return web.json_response(
            {
                "object": "list",
                "model": "m",
                "data": [{"object": "embedding", "index": 0, "embedding": _VECTOR}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        )

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", hits
    finally:
        await runner.cleanup()


def _record_calls(monkeypatch, owner, name, calls: list[dict]):
    """Replace ``owner.name`` with a pass-through recording its kwargs."""
    real = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)


def _session_timeout(kwargs: dict):
    timeout = kwargs.get("timeout")
    return _OMITTED if timeout is None else timeout.total


# --------------------------------------------------------------------------
# One probe per binding: call it, return what its client was given
# (``_OMITTED`` when the timeout was left to the SDK default).
# --------------------------------------------------------------------------


async def _probe_openai(monkeypatch, url, extra):
    from lightrag.llm import openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module, "AsyncOpenAI", calls)
    result = await module.openai_embed.func(["a"], base_url=url, api_key="k", **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_azure(monkeypatch, url, extra):
    import openai

    from lightrag.llm import openai as module

    calls: list[dict] = []
    # Imported inside create_openai_async_client, so patch it at the source.
    _record_calls(monkeypatch, openai, "AsyncAzureOpenAI", calls)
    result = await module.azure_openai_embed.func(
        ["a"], model="d", base_url=url, api_key="k", api_version="v", **extra
    )
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_nvidia(monkeypatch, url, extra):
    from lightrag.llm import nvidia_openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module, "AsyncOpenAI", calls)
    result = await module.nvidia_openai_embed.func(
        ["a"], base_url=url, api_key="k", **extra
    )
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_ollama(monkeypatch, url, extra):
    from lightrag.llm import ollama as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.ollama, "AsyncClient", calls)
    result = await module.ollama_embed.func(["a"], host=url, **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    # The Ollama client's own default is None, so None is "SDK default".
    timeout = calls[0]["timeout"]
    return _OMITTED if timeout is None else timeout


async def _probe_jina(monkeypatch, url, extra):
    from lightrag.llm import jina as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.aiohttp, "ClientSession", calls)
    result = await module.jina_embed.func(
        ["a"],
        embedding_dim=len(_VECTOR),
        base_url=f"{url}/jina/embeddings",
        api_key="k",
        **extra,
    )
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return _session_timeout(calls[0])


async def _probe_lollms(monkeypatch, url, extra):
    from lightrag.llm import lollms as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.aiohttp, "ClientSession", calls)
    result = await module.lollms_embed.func(["a"], base_url=url, **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return _session_timeout(calls[0])


async def _probe_gemini(monkeypatch, url, extra):
    from lightrag.llm import gemini as module

    calls: list[tuple] = []

    def fake_client(api_key, base_url, timeout=None):
        calls.append(timeout)
        raise _Recorded

    monkeypatch.setattr(module, "_get_gemini_client", fake_client)
    with pytest.raises(_Recorded):
        await module.gemini_embed.func(["a"], api_key="k", **extra)
    assert len(calls) == 1
    # Gemini takes milliseconds.
    return _OMITTED if calls[0] is None else calls[0] / 1000


async def _probe_bedrock(monkeypatch, url, extra):
    from lightrag.llm import bedrock as module

    calls: list[dict] = []
    real = module._bedrock_client_kwargs

    def recording(*args, **kwargs):
        calls.append(real(*args, **kwargs))
        raise _Recorded

    monkeypatch.setattr(module, "_bedrock_client_kwargs", recording)
    with pytest.raises(_Recorded):
        await module.bedrock_embed.func(["a"], aws_region="us-east-1", **extra)
    assert len(calls) == 1
    config = calls[0].get("config")
    if config is None:
        return _OMITTED
    assert config.connect_timeout == config.read_timeout
    return config.read_timeout


async def _probe_voyageai(monkeypatch, url, extra):
    from lightrag.llm import voyageai as module

    calls: list[dict] = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        raise _Recorded

    monkeypatch.setattr(module.voyageai, "AsyncClient", fake_client)
    with pytest.raises(_Recorded):
        await module.voyageai_embed.func(["a"], api_key="k", **extra)
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_zhipu(monkeypatch, url, extra):
    from lightrag.llm import zhipu as module

    pytest.importorskip("zai")
    calls: list[dict] = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=_VECTOR)])

    @contextlib.contextmanager
    def fake_build_client(client_cls, api_key):
        yield SimpleNamespace(embeddings=_Embeddings())

    monkeypatch.setattr(module, "_build_client", fake_build_client)
    result = await module.zhipu_embedding.func(["a"], api_key="k", **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_rerank(monkeypatch, url, extra):
    from lightrag import rerank as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.aiohttp, "ClientSession", calls)
    result = await module.cohere_rerank(
        query="q", documents=["a"], api_key="k", base_url=f"{url}/rerank", **extra
    )
    assert [r["index"] for r in result] == [0]
    assert len(calls) == 1
    return _session_timeout(calls[0])


_EMBEDDING_PROBES = [
    pytest.param(_probe_openai, id="openai_embed"),
    pytest.param(_probe_azure, id="azure_openai_embed"),
    pytest.param(_probe_nvidia, id="nvidia_openai_embed"),
    pytest.param(_probe_ollama, id="ollama_embed"),
    pytest.param(_probe_jina, id="jina_embed"),
    pytest.param(_probe_lollms, id="lollms_embed"),
    pytest.param(_probe_gemini, id="gemini_embed"),
    pytest.param(_probe_bedrock, id="bedrock_embed"),
    pytest.param(_probe_voyageai, id="voyageai_embed"),
    pytest.param(_probe_zhipu, id="zhipu_embedding"),
]

# (LightRAG's bound timeout or "unbound", explicit kwargs, what the client gets)
_CASES = [
    pytest.param(7, {}, 7, id="configured-positive"),
    pytest.param(0, {}, _OMITTED, id="configured-zero-sdk-default"),
    pytest.param(-1, {}, _OMITTED, id="configured-negative-sdk-default"),
    pytest.param("unbound", {}, _OMITTED, id="outside-lightrag-sdk-default"),
    pytest.param(7, {"timeout": 3}, 3, id="explicit-argument-wins"),
    pytest.param(7, {"timeout": 0}, _OMITTED, id="explicit-zero-sdk-default"),
]


async def _run(probe, bind, monkeypatch, configured, extra):
    async with _stub() as (url, _hits):

        async def call():
            return await probe(monkeypatch, url, extra)

        if configured != "unbound":
            call = bind(call, configured)
        return await call()


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", _EMBEDDING_PROBES)
@pytest.mark.parametrize(("configured", "extra", "expected"), _CASES)
async def test_embedding_binding_hands_timeout_to_client(
    monkeypatch, probe, configured, extra, expected
):
    observed = await _run(probe, bind_embedding_timeout, monkeypatch, configured, extra)

    assert observed == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured", "extra", "expected"), _CASES)
async def test_rerank_binding_hands_timeout_to_client(
    monkeypatch, configured, extra, expected
):
    observed = await _run(
        _probe_rerank, bind_rerank_timeout, monkeypatch, configured, extra
    )

    assert observed == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe", [_probe_openai, _probe_azure], ids=["openai", "azure"]
)
@pytest.mark.parametrize(
    ("configured", "extra", "expected"),
    [
        pytest.param(7, {}, 5, id="beats-lightrag-timeout"),
        pytest.param(0, {}, 5, id="applies-when-lightrag-sets-none"),
        pytest.param(7, {"timeout": 3}, 3, id="positive-argument-beats-it"),
        # Same as openai_complete_if_cache: a non-positive argument means
        # "LightRAG sets no timeout", which leaves the caller's entry in place.
        pytest.param(7, {"timeout": 0}, 5, id="non-positive-argument-keeps-it"),
    ],
)
async def test_openai_client_configs_timeout_precedence(
    monkeypatch, probe, configured, extra, expected
):
    observed = await _run(
        probe,
        bind_embedding_timeout,
        monkeypatch,
        configured,
        {"client_configs": {"timeout": 5}, **extra},
    )

    assert observed == expected


@pytest.mark.asyncio
async def test_embedding_and_rerank_timeouts_do_not_leak_into_each_other():
    async def read_both():
        return embedding_timeout(), rerank_timeout()

    assert await bind_embedding_timeout(read_both, 7)() == (7, None)
    assert await bind_rerank_timeout(read_both, 9)() == (None, 9)
    # Reset after the call: nothing outside a bound call sees a value.
    assert await read_both() == (None, None)


# --------------------------------------------------------------------------
# Through LightRAG: the queue wrappers bind the instance's timeouts
# --------------------------------------------------------------------------


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _llm(*args, **kwargs) -> str:
    return "ok"


def _make_rag(tmp_path, embedding_func, rerank_func, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace="timeout-forwarding",
        llm_model_func=_llm,
        embedding_func=embedding_func,
        rerank_model_func=rerank_func,
        tokenizer=Tokenizer("char", _CharTokenizer()),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding", "rerank", "expected"),
    [
        pytest.param(11, 13, (11, 13), id="positive"),
        pytest.param(0, 0, (None, None), id="zero-sdk-default"),
    ],
)
async def test_lightrag_binds_its_timeouts_around_each_call(
    tmp_path, embedding, rerank, expected
):
    seen: dict[str, object] = {}

    async def embed(texts, max_token_size=None):
        # The bound wrapper keeps the signature, so EmbeddingFunc still injects.
        seen["max_token_size"] = max_token_size
        seen["embedding"] = embedding_timeout()
        return np.ones((len(texts), 4))

    async def rerank_func(query, documents, top_n=None):
        seen["rerank"] = rerank_timeout()
        return [{"index": 0, "relevance_score": 1.0}]

    rag = _make_rag(
        tmp_path,
        EmbeddingFunc(embedding_dim=4, max_token_size=123, func=embed),
        rerank_func,
        default_embedding_timeout=embedding,
        default_rerank_timeout=rerank,
    )

    await rag.embedding_func(["a"])
    await rag.rerank_model_func(query="q", documents=["a"])

    assert (seen["embedding"], seen["rerank"]) == expected
    assert seen["max_token_size"] == 123
    assert (embedding_timeout(), rerank_timeout()) == (None, None)


@pytest.mark.asyncio
async def test_lightrag_timeout_reaches_a_partial_of_a_bundled_binding(
    tmp_path, monkeypatch
):
    """The common ``partial(openai_embed.func, ...)`` setup sizes its client."""
    from lightrag.llm import openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module, "AsyncOpenAI", calls)

    async with _stub() as (url, _hits):
        rag = _make_rag(
            tmp_path,
            EmbeddingFunc(
                embedding_dim=len(_VECTOR),
                max_token_size=8192,
                func=functools.partial(
                    module.openai_embed.func, base_url=url, api_key="k"
                ),
            ),
            None,
            default_embedding_timeout=17,
        )
        result = await rag.embedding_func(["a"])

    assert result.shape == (1, len(_VECTOR))
    assert [call.get("timeout") for call in calls] == [17]


# --------------------------------------------------------------------------
# A client timeout is retried (it used to be the queue's unretried cancel)
# --------------------------------------------------------------------------


def test_aiohttp_timeout_is_transient():
    assert is_transient_aiohttp_error(asyncio.TimeoutError())
    assert not is_transient_aiohttp_error(asyncio.CancelledError())
    assert not is_transient_aiohttp_error(aiohttp.InvalidURL("x"))


@pytest.mark.asyncio
async def test_rerank_attempt_that_exceeds_the_timeout_is_retried(monkeypatch):
    from lightrag import rerank as module

    monkeypatch.setattr(module.generic_rerank_api.retry, "wait", wait_none())

    async with _stub(first_delay=2.0) as (url, hits):
        result = await bind_rerank_timeout(module.cohere_rerank, 0.3)(
            query="q", documents=["a"], api_key="k", base_url=f"{url}/rerank"
        )

    assert [r["index"] for r in result] == [0]
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_jina_attempt_that_exceeds_the_timeout_is_retried(monkeypatch):
    from lightrag.llm import jina as module

    monkeypatch.setattr(module.jina_embed.func.retry, "wait", wait_none())

    async with _stub(first_delay=2.0) as (url, hits):
        result = await bind_embedding_timeout(module.jina_embed.func, 0.3)(
            ["a"],
            embedding_dim=len(_VECTOR),
            base_url=f"{url}/jina/embeddings",
            api_key="k",
        )

    assert result.shape == (1, len(_VECTOR))
    assert len(hits) == 2
