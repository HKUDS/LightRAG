"""``QueryParam.extra_llm_kwargs`` must reach the LLM call through the same
role-wrapped, queue-limited func every other query-time kwarg goes through --
never by swapping in a caller-supplied replacement func.

Regression coverage for the gap described in the discussion around per-call
LLM context (forwarding caller identity through LightRAG's shared LLM dispatch
queue, akin to the unresolved workspace-header case in #2904): a
``contextvars.ContextVar`` set by the caller does not survive the queue's
persistent worker-task hand-off, so ``extra_llm_kwargs`` is the supported way
to carry per-call data (e.g. an ``extra_headers`` dict for a binding) through
``kg_query``, ``naive_query``, ``extract_keywords_only``, and bypass mode.
"""

from __future__ import annotations

import pytest

from lightrag import LightRAG
from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.utils import Tokenizer


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _fake_tokenizer() -> Tokenizer:
    return Tokenizer("fake", _FakeTokenizerImpl())


class _RecordingModel:
    """Records the kwargs of every call it receives."""

    def __init__(self, response: str = "answer"):
        self._response = response
        self.calls: list[dict] = []

    async def __call__(self, *_args, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "extra_llm_kwargs regression test chunk.",
                "file_path": "test.md",
            }
        ]


def _global_config(llm_func, keyword_func=None) -> dict:
    role_llm_funcs = {"query": llm_func}
    if keyword_func is not None:
        role_llm_funcs["keyword"] = keyword_func
    return {
        "tokenizer": _fake_tokenizer(),
        "role_llm_funcs": role_llm_funcs,
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


def _naive_param(**overrides) -> QueryParam:
    return QueryParam(mode="naive", enable_rerank=False, **overrides)


def _kg_param(**overrides) -> QueryParam:
    # Preset ll_keywords short-circuits keyword extraction for the plain
    # forwarding tests; the dedicated keyword-extraction test below clears it.
    return QueryParam(
        mode="local", enable_rerank=False, ll_keywords=["Tesla"], **overrides
    )


@pytest.fixture
def stub_query_context(monkeypatch):
    """Skip retrieval: kg_query only forwards its storage args into this call."""

    async def _fake_build_query_context(*_args, **_kwargs):
        return QueryContextResult(context="KG CONTEXT", raw_data={})

    monkeypatch.setattr(
        "lightrag.operate._build_query_context", _fake_build_query_context
    )


async def _run_kg_query(param, cfg):
    return await kg_query("who is Tesla?", None, None, None, None, param, cfg)


# ---------------------------------------------------------------------------
# naive_query / kg_query main synthesis call
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_forwards_extra_llm_kwargs():
    model = _RecordingModel()
    cfg = _global_config(model)

    await naive_query(
        "query",
        _FakeChunksVDB(),
        _naive_param(extra_llm_kwargs={"extra_headers": {"X-User-Id": "u-1"}}),
        cfg,
    )

    assert model.calls[0]["extra_headers"] == {"X-User-Id": "u-1"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_forwards_extra_llm_kwargs(stub_query_context):
    model = _RecordingModel()
    cfg = _global_config(model)

    await _run_kg_query(
        _kg_param(extra_llm_kwargs={"extra_headers": {"X-User-Id": "u-1"}}), cfg
    )

    assert model.calls[0]["extra_headers"] == {"X-User-Id": "u-1"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extra_llm_kwargs_default_adds_nothing():
    """The field's default (``None``) must be a strict no-op for existing callers."""
    model = _RecordingModel()
    cfg = _global_config(model)

    await naive_query("query", _FakeChunksVDB(), _naive_param(), cfg)

    # `_priority` is real here too: the fake model stands in directly for the
    # queue-wrapped func, which is what actually strips it in production.
    assert set(model.calls[0]) == {
        "system_prompt",
        "history_messages",
        "enable_cot",
        "stream",
        "_priority",
    }


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("query_func_name", ["naive", "kg"])
async def test_extra_llm_kwargs_colliding_with_pipeline_kwarg_raises(
    query_func_name, stub_query_context
):
    """A key the pipeline itself sets for this call must fail loud, not overwrite."""
    model = _RecordingModel()
    cfg = _global_config(model)
    param_kwargs = {"extra_llm_kwargs": {"stream": True}}

    with pytest.raises(TypeError, match="stream"):
        if query_func_name == "naive":
            await naive_query(
                "query", _FakeChunksVDB(), _naive_param(**param_kwargs), cfg
            )
        else:
            await _run_kg_query(_kg_param(**param_kwargs), cfg)


# ---------------------------------------------------------------------------
# extract_keywords_only (keyword-extraction call, via kg_query)
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_keyword_extraction_forwards_extra_llm_kwargs(stub_query_context):
    model = _RecordingModel()
    keyword_model = _RecordingModel(
        response='{"high_level_keywords": ["physics"], "low_level_keywords": ["Tesla"]}'
    )
    cfg = _global_config(model, keyword_func=keyword_model)

    # No preset keywords: forces the real extraction call.
    param = QueryParam(
        mode="local",
        enable_rerank=False,
        extra_llm_kwargs={"extra_headers": {"X-User-Id": "u-1"}},
    )
    await _run_kg_query(param, cfg)

    assert keyword_model.calls[0]["extra_headers"] == {"X-User-Id": "u-1"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_keyword_extraction_collision_raises(stub_query_context):
    model = _RecordingModel()
    keyword_model = _RecordingModel(
        response='{"high_level_keywords": ["physics"], "low_level_keywords": ["Tesla"]}'
    )
    cfg = _global_config(model, keyword_func=keyword_model)

    param = QueryParam(
        mode="local",
        enable_rerank=False,
        extra_llm_kwargs={"response_format": {"type": "text"}},
    )

    with pytest.raises(TypeError, match="response_format"):
        await _run_kg_query(param, cfg)


# ---------------------------------------------------------------------------
# bypass mode (LightRAG.aquery_llm, does not go through operate.py)
# ---------------------------------------------------------------------------


class _FakeRAG:
    """Minimal stand-in exposing only what the bypass branch touches."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bypass_mode_forwards_extra_llm_kwargs():
    model = _RecordingModel("answer")
    rag = _FakeRAG(model)

    await LightRAG.aquery_llm(
        rag,
        "question",
        param=QueryParam(
            mode="bypass",
            stream=False,
            extra_llm_kwargs={"extra_headers": {"X-User-Id": "u-1"}},
        ),
    )

    assert model.calls[0]["extra_headers"] == {"X-User-Id": "u-1"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bypass_mode_collision_fails_the_query():
    """aquery_llm wraps the whole body in try/except, so the TypeError surfaces
    as a failure response rather than propagating -- pin that shape instead of
    the raise, so this doesn't silently start "working" by returning a
    stream-collision-free answer if the collision detection ever regresses."""
    model = _RecordingModel("answer")
    rag = _FakeRAG(model)

    result = await LightRAG.aquery_llm(
        rag,
        "question",
        param=QueryParam(
            mode="bypass", stream=False, extra_llm_kwargs={"stream": True}
        ),
    )

    assert result["status"] == "failure"
    assert "stream" in result["message"]
    assert model.calls == []
