"""``QueryParam.extra_llm_kwargs`` / ``extra_embedding_kwargs`` /
``extra_rerank_kwargs`` must reach their respective call through the same
queue-limited func every other query-time kwarg goes through -- never by
swapping in a caller-supplied replacement func.

Regression coverage for the gap described in the discussion around per-call
LLM context (forwarding caller identity through LightRAG's shared dispatch
queues, akin to the unresolved workspace-header case in #2904): a
``contextvars.ContextVar`` set by the caller does not survive the queue's
persistent worker-task hand-off, so these fields are the supported way to
carry per-call data (e.g. an ``extra_headers`` dict for a binding) through
every provider call a query can make: ``kg_query``, ``naive_query``,
``extract_keywords_only``, bypass mode, the embedding pre-compute in
``_perform_kg_search``/``naive_query``, and the rerank call.
"""

from __future__ import annotations

import pytest

from lightrag import LightRAG
from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import _perform_kg_search, kg_query, naive_query
from lightrag.utils import Tokenizer, apply_rerank_if_enabled, process_chunks_unified


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


# ---------------------------------------------------------------------------
# extra_embedding_kwargs: naive_query's chunk-embedding pre-compute
# ---------------------------------------------------------------------------


class _RecordingEmbeddingFunc:
    def __init__(self, vector=None, error=None):
        self.calls: list[dict] = []
        self._vector = vector if vector is not None else [0.1, 0.2]
        self._error = error

    async def __call__(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        if self._error:
            raise self._error
        return [self._vector for _ in texts]


class _EmbeddingAwareChunksVDB(_FakeChunksVDB):
    def __init__(self, embedding_func):
        self.embedding_func = embedding_func
        self.query_calls: list[dict] = []

    async def query(self, *args, **kwargs):
        self.query_calls.append(kwargs)
        return await super().query(*args, **kwargs)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_forwards_extra_embedding_kwargs():
    embedding_func = _RecordingEmbeddingFunc()
    chunks_vdb = _EmbeddingAwareChunksVDB(embedding_func)
    cfg = _global_config(_RecordingModel())

    await naive_query(
        "query",
        chunks_vdb,
        _naive_param(extra_embedding_kwargs={"extra_headers": {"X-User-Id": "u-1"}}),
        cfg,
    )

    assert embedding_func.calls[0]["extra_headers"] == {"X-User-Id": "u-1"}
    # The pre-computed embedding must actually reach chunks_vdb.query(), not
    # just get computed and discarded.
    assert chunks_vdb.query_calls[0]["query_embedding"] == [0.1, 0.2]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_default_skips_embedding_precompute():
    """No extra_embedding_kwargs: zero added round-trips for existing callers."""
    embedding_func = _RecordingEmbeddingFunc()
    chunks_vdb = _EmbeddingAwareChunksVDB(embedding_func)
    cfg = _global_config(_RecordingModel())

    await naive_query("query", chunks_vdb, _naive_param(), cfg)

    assert embedding_func.calls == []
    assert chunks_vdb.query_calls[0]["query_embedding"] is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_embedding_failure_propagates_when_extra_kwargs_set():
    embedding_func = _RecordingEmbeddingFunc(error=RuntimeError("embedding down"))
    chunks_vdb = _EmbeddingAwareChunksVDB(embedding_func)
    model = _RecordingModel()
    cfg = _global_config(model)

    with pytest.raises(RuntimeError, match="embedding down"):
        await naive_query(
            "query",
            chunks_vdb,
            _naive_param(
                extra_embedding_kwargs={"extra_headers": {"X-User-Id": "u-1"}}
            ),
            cfg,
        )

    assert model.calls == []


# ---------------------------------------------------------------------------
# extra_embedding_kwargs: _perform_kg_search's batched pre-compute
# ---------------------------------------------------------------------------


class _FakeTextChunksDB:
    def __init__(self, embedding_func):
        self.embedding_func = embedding_func
        self.global_config: dict = {}


async def _run_kg_search(query_param, embedding_func):
    # Empty keyword strings + mode="local" short-circuit every retrieval
    # branch below the pre-compute step (see operate.py's mode dispatch),
    # isolating the pre-compute call this test targets. chunks_vdb only
    # needs to be truthy to include "query" in the batch -- mode="local"
    # never reaches the branch that would actually call it.
    return await _perform_kg_search(
        "who is Tesla?",
        "",
        "",
        None,
        None,
        None,
        _FakeTextChunksDB(embedding_func),
        query_param,
        chunks_vdb=object(),
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_search_forwards_extra_embedding_kwargs():
    embedding_func = _RecordingEmbeddingFunc()

    await _run_kg_search(
        QueryParam(
            mode="local", extra_embedding_kwargs={"extra_headers": {"X-User-Id": "u-1"}}
        ),
        embedding_func,
    )

    assert embedding_func.calls[0]["extra_headers"] == {"X-User-Id": "u-1"}
    assert embedding_func.calls[0]["texts"] == ["who is Tesla?"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_search_embedding_failure_swallowed_by_default():
    """Baseline: without extra_embedding_kwargs, a failed pre-compute must
    keep degrading gracefully, exactly as before this change."""
    embedding_func = _RecordingEmbeddingFunc(error=RuntimeError("embedding down"))

    result = await _run_kg_search(QueryParam(mode="local"), embedding_func)

    assert result is not None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_search_embedding_failure_propagates_when_extra_kwargs_set():
    embedding_func = _RecordingEmbeddingFunc(error=RuntimeError("embedding down"))

    with pytest.raises(RuntimeError, match="embedding down"):
        await _run_kg_search(
            QueryParam(
                mode="local",
                extra_embedding_kwargs={"extra_headers": {"X-User-Id": "u-1"}},
            ),
            embedding_func,
        )


# ---------------------------------------------------------------------------
# extra_rerank_kwargs
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_apply_rerank_forwards_extra_rerank_kwargs():
    captured = {}

    async def rerank_func(**kwargs):
        captured.update(kwargs)
        return [{"index": 0, "relevance_score": 1.0}]

    await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=[{"content": "doc"}],
        global_config={"rerank_model_func": rerank_func},
        extra_rerank_kwargs={"extra_headers": {"X-User-Id": "u-1"}},
    )

    assert captured["extra_headers"] == {"X-User-Id": "u-1"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_apply_rerank_failure_swallowed_by_default():
    """Baseline: without extra_rerank_kwargs, a rerank failure must keep
    falling back to the original chunk order, exactly as before this change."""

    async def rerank_func(**_kwargs):
        raise RuntimeError("rerank down")

    docs = [{"content": "doc"}]
    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=docs,
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == docs


@pytest.mark.offline
@pytest.mark.asyncio
async def test_apply_rerank_failure_propagates_when_extra_kwargs_set():
    async def rerank_func(**_kwargs):
        raise RuntimeError("rerank down")

    with pytest.raises(RuntimeError, match="rerank down"):
        await apply_rerank_if_enabled(
            query="query",
            retrieved_docs=[{"content": "doc"}],
            global_config={"rerank_model_func": rerank_func},
            extra_rerank_kwargs={"extra_headers": {"X-User-Id": "u-1"}},
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_process_chunks_unified_forwards_extra_rerank_kwargs():
    """Confirms QueryParam.extra_rerank_kwargs actually reaches the rerank
    call through process_chunks_unified, not just apply_rerank_if_enabled's
    own parameter."""
    captured = {}

    async def rerank_func(**kwargs):
        captured.update(kwargs)
        return [{"index": 0, "relevance_score": 1.0}]

    query_param = QueryParam(
        mode="naive",
        enable_rerank=True,
        extra_rerank_kwargs={"extra_headers": {"X-User-Id": "u-1"}},
    )
    global_config = {"rerank_model_func": rerank_func, "min_rerank_score": 0.0}

    await process_chunks_unified(
        "query", [{"content": "chunk one"}], query_param, global_config
    )

    assert captured["extra_headers"] == {"X-User-Id": "u-1"}
