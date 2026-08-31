"""Tests for REST-exposed LLM token usage on the query path (issue #2572).

Three layers, each load-bearing on its own:

1. **Capability detection** (``llm_roles.py::_wrap_llm_role_func`` /
   ``role_supports_token_tracker``): only openai/gemini/ollama declare a
   ``token_tracker`` parameter today. Every other binding forwards
   unrecognized kwargs verbatim into its SDK client call, so passing
   ``token_tracker=`` to one of those raises a wire-level ``TypeError``
   rather than being silently ignored -- this is computed once per role
   rebuild from the *real* bound function's signature, never assumed.

2. **Attachment** (``operate.py::_attach_token_usage`` /
   ``_role_supports_token_tracker``): ``kg_query``/``naive_query`` must
   record usage into ``raw_data['metadata']['token_usage']`` only when an
   LLM call actually happened (never on a cache hit, an
   ``only_need_context``/``only_need_prompt`` short-circuit, or an
   unsupported binding) -- "absent" must mean "we don't know", never "zero".

3. **REST surface** (``query_routes.py``): ``QueryResponse.token_usage`` is
   additive and optional, sourced from ``result['metadata']['token_usage']``.
"""

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import (
    _attach_token_usage,
    _role_supports_token_tracker,
    extract_keywords_only,
    get_keywords_from_query,
    kg_query,
    naive_query,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenTracker

pytestmark = pytest.mark.offline


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _fake_tokenizer() -> Tokenizer:
    return Tokenizer("fake", _FakeTokenizerImpl())


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


class _FakeChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "Token usage test chunk about Alice the researcher.",
                "file_path": "test.md",
            }
        ]


def _naive_global_config(llm_func, *, role_supports=True) -> dict:
    return {
        "tokenizer": _fake_tokenizer(),
        "role_llm_funcs": {"query": llm_func},
        "role_supports_token_tracker": {"query": role_supports, "keyword": False},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


def _naive_param(**overrides) -> QueryParam:
    return QueryParam(mode="naive", enable_rerank=False, **overrides)


# ---------------------------------------------------------------------------
# 1. Capability detection (llm_roles.py)
# ---------------------------------------------------------------------------


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 16)


async def _aware_base_llm(*args, token_tracker=None, **kwargs) -> str:
    return "aware"


async def _unaware_base_llm(*args, **kwargs) -> str:
    return "unaware"


def _make_rag(tmp_path, llm_func) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path / "token-usage-role"),
        workspace="token-usage-role",
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=16, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )


def test_role_supports_token_tracker_detects_the_real_signature(tmp_path):
    aware_rag = _make_rag(tmp_path, _aware_base_llm)
    unaware_rag = _make_rag(tmp_path, _unaware_base_llm)

    # Every role inherits the base func here (no per-role override), so every
    # role reflects the same base function's real signature.
    assert all(aware_rag.role_supports_token_tracker.values())
    assert not any(unaware_rag.role_supports_token_tracker.values())

    # And it is threaded into global_config under the exact key operate.py reads.
    aware_cfg = aware_rag._build_global_config()
    unaware_cfg = unaware_rag._build_global_config()
    assert aware_cfg["role_supports_token_tracker"]["query"] is True
    assert unaware_cfg["role_supports_token_tracker"]["query"] is False


def test_role_supports_token_tracker_helper_reads_the_global_config_key():
    assert _role_supports_token_tracker(
        {"role_supports_token_tracker": {"query": True}}, "query"
    )
    assert not _role_supports_token_tracker(
        {"role_supports_token_tracker": {"query": False}}, "query"
    )
    # Missing role or missing key entirely defaults closed (unsupported),
    # never open -- the safe default when nothing has been computed yet.
    assert not _role_supports_token_tracker({}, "query")
    assert not _role_supports_token_tracker(
        {"role_supports_token_tracker": {}}, "query"
    )


# ---------------------------------------------------------------------------
# 2. _attach_token_usage
# ---------------------------------------------------------------------------


def test_attach_token_usage_is_a_no_op_for_an_untouched_tracker():
    raw_data: dict = {}
    _attach_token_usage(raw_data, TokenTracker())
    assert raw_data == {}, "absent must mean unknown, not a zeroed-out usage dict"


def test_attach_token_usage_writes_the_tracker_snapshot():
    tracker = TokenTracker()
    tracker.add_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    raw_data: dict = {}
    _attach_token_usage(raw_data, tracker)
    assert raw_data["metadata"]["token_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "call_count": 1,
    }


def test_attach_token_usage_preserves_existing_metadata_keys():
    raw_data: dict = {"metadata": {"keywords": {"high_level": ["x"]}}}
    tracker = TokenTracker()
    tracker.add_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    _attach_token_usage(raw_data, tracker)
    assert raw_data["metadata"]["keywords"] == {"high_level": ["x"]}
    assert raw_data["metadata"]["token_usage"]["total_tokens"] == 2


# ---------------------------------------------------------------------------
# 3. naive_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_query_reports_token_usage_when_binding_supports_it():
    async def aware_model(*_args, token_tracker=None, **_kwargs):
        assert token_tracker is not None
        token_tracker.add_usage(
            {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}
        )
        return "an answer"

    cfg = _naive_global_config(aware_model, role_supports=True)
    result = await naive_query(
        "who is Alice?",
        _FakeChunksVDB(),
        _naive_param(),
        cfg,
        hashing_kv=_FakeKVStorage(),
    )

    usage = result.raw_data["metadata"]["token_usage"]
    assert usage["total_tokens"] == 50
    assert usage["call_count"] == 1


@pytest.mark.asyncio
async def test_naive_query_omits_token_usage_when_binding_unsupported():
    call_kwargs_seen = {}

    async def strict_model(*_args, **kwargs):
        call_kwargs_seen.update(kwargs)
        return "an answer"

    cfg = _naive_global_config(strict_model, role_supports=False)
    result = await naive_query(
        "who is Alice?",
        _FakeChunksVDB(),
        _naive_param(),
        cfg,
        hashing_kv=_FakeKVStorage(),
    )

    assert "token_tracker" not in call_kwargs_seen, (
        "the gate must keep token_tracker out of the call entirely, not pass "
        "it as None -- an unsupported binding keys on the kwarg's presence"
    )
    assert "token_usage" not in result.raw_data.get("metadata", {})


@pytest.mark.asyncio
async def test_naive_query_omits_token_usage_on_cache_hit():
    calls = 0

    async def aware_model(*_args, token_tracker=None, **_kwargs):
        nonlocal calls
        calls += 1
        if token_tracker is not None:
            token_tracker.add_usage(
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )
        return "an answer"

    cfg = _naive_global_config(aware_model, role_supports=True)
    cache = _FakeKVStorage()
    param = _naive_param()

    first = await naive_query(
        "who is Alice?", _FakeChunksVDB(), param, cfg, hashing_kv=cache
    )
    assert first.raw_data["metadata"]["token_usage"]["total_tokens"] == 2
    assert calls == 1

    second = await naive_query(
        "who is Alice?", _FakeChunksVDB(), param, cfg, hashing_kv=cache
    )
    assert calls == 1, "second call must be served from cache"
    assert "token_usage" not in second.raw_data.get("metadata", {}), (
        "a cached answer consumed no tokens on this call and must not report "
        "the previous call's figure as if it were fresh"
    )


# ---------------------------------------------------------------------------
# 4. kg_query
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_query_context(monkeypatch):
    """Skip retrieval: kg_query only forwards its storage args into this call."""

    async def _fake_build_query_context(*_args, **_kwargs):
        return QueryContextResult(context="KG CONTEXT", raw_data={})

    monkeypatch.setattr(
        "lightrag.operate._build_query_context", _fake_build_query_context
    )


def _kg_param(**overrides) -> QueryParam:
    # Preset ll_keywords short-circuits keyword extraction so tests that
    # don't care about it can ignore the keyword-role call entirely.
    return QueryParam(
        mode="local", enable_rerank=False, ll_keywords=["Alice"], **overrides
    )


def _kg_global_config(
    query_func, *, query_supports=True, keyword_func=None, keyword_supports=False
) -> dict:
    role_llm_funcs = {"query": query_func}
    if keyword_func is not None:
        role_llm_funcs["keyword"] = keyword_func
    return {
        "tokenizer": _fake_tokenizer(),
        "role_llm_funcs": role_llm_funcs,
        "role_supports_token_tracker": {
            "query": query_supports,
            "keyword": keyword_supports,
        },
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


async def _run_kg_query(param, cfg, cache):
    return await kg_query(
        "who is Alice?", None, None, None, None, param, cfg, hashing_kv=cache
    )


@pytest.mark.asyncio
async def test_kg_query_reports_synthesis_usage_with_preset_keywords(
    stub_query_context,
):
    """No keyword-extraction call happens (ll_keywords preset), so the
    reported usage is exactly the synthesis call's."""

    async def aware_model(*_args, token_tracker=None, **_kwargs):
        token_tracker.add_usage(
            {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
        )
        return "an answer"

    cfg = _kg_global_config(aware_model, query_supports=True)
    result = await _run_kg_query(_kg_param(), cfg, _FakeKVStorage())

    usage = result.raw_data["metadata"]["token_usage"]
    assert usage["total_tokens"] == 100
    assert usage["call_count"] == 1


@pytest.mark.asyncio
async def test_kg_query_aggregates_keyword_and_synthesis_usage(stub_query_context):
    async def aware_query_model(*_args, token_tracker=None, **_kwargs):
        token_tracker.add_usage(
            {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
        )
        return "an answer"

    async def aware_keyword_model(*_args, token_tracker=None, **_kwargs):
        token_tracker.add_usage(
            {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
        )
        return '{"high_level_keywords": ["physics"], "low_level_keywords": ["Alice"]}'

    cfg = _kg_global_config(
        aware_query_model,
        query_supports=True,
        keyword_func=aware_keyword_model,
        keyword_supports=True,
    )
    # No preset keywords this time: force real extraction.
    param = QueryParam(mode="local", enable_rerank=False)
    result = await _run_kg_query(param, cfg, _FakeKVStorage())

    usage = result.raw_data["metadata"]["token_usage"]
    assert usage["total_tokens"] == 135
    assert usage["call_count"] == 2


@pytest.mark.asyncio
async def test_kg_query_reports_keyword_only_usage_for_only_need_context(
    stub_query_context,
):
    """only_need_context short-circuits before the synthesis call, so a
    caller that stops there still sees exactly what it cost: the keyword
    extraction, and nothing claimed for a synthesis call that never ran."""

    async def keyword_model(*_args, token_tracker=None, **_kwargs):
        token_tracker.add_usage(
            {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
        )
        return '{"high_level_keywords": ["physics"], "low_level_keywords": ["Alice"]}'

    async def synthesis_model(*_args, **_kwargs):
        raise AssertionError("synthesis must not be called for only_need_context")

    cfg = _kg_global_config(
        synthesis_model,
        query_supports=True,
        keyword_func=keyword_model,
        keyword_supports=True,
    )
    param = QueryParam(mode="local", enable_rerank=False, only_need_context=True)
    result = await _run_kg_query(param, cfg, _FakeKVStorage())

    usage = result.raw_data["metadata"]["token_usage"]
    assert usage["total_tokens"] == 35
    assert usage["call_count"] == 1


@pytest.mark.asyncio
async def test_kg_query_omits_token_usage_when_neither_role_supports_it(
    stub_query_context,
):
    async def strict_query_model(*_args, **_kwargs):
        return "an answer"

    cfg = _kg_global_config(strict_query_model, query_supports=False)
    result = await _run_kg_query(_kg_param(), cfg, _FakeKVStorage())

    assert "token_usage" not in result.raw_data.get("metadata", {})


@pytest.mark.asyncio
async def test_kg_query_reports_keyword_usage_on_the_no_keywords_fail_path():
    """Both keyword lists come back empty and the query is >=50 chars, so
    kg_query bails out with the canned fail_response before ever reaching
    _build_query_context. The keyword-extraction call still ran and spent
    real tokens, so that cost must not be silently dropped."""

    async def keyword_model(*_args, token_tracker=None, **_kwargs):
        token_tracker.add_usage(
            {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        )
        return '{"high_level_keywords": [], "low_level_keywords": []}'

    async def synthesis_model(*_args, **_kwargs):
        raise AssertionError("synthesis must not be called on the fail path")

    cfg = _kg_global_config(
        synthesis_model,
        query_supports=True,
        keyword_func=keyword_model,
        keyword_supports=True,
    )
    long_query = "x" * 60
    param = QueryParam(mode="local", enable_rerank=False)
    result = await kg_query(
        long_query, None, None, None, None, param, cfg, hashing_kv=_FakeKVStorage()
    )

    assert result.llm_generated is False
    usage = result.raw_data["metadata"]["token_usage"]
    assert usage["total_tokens"] == 15
    assert usage["call_count"] == 1


# ---------------------------------------------------------------------------
# 5. extract_keywords_only / get_keywords_from_query
# ---------------------------------------------------------------------------


def _keyword_global_config(keyword_func, *, supports=True) -> dict:
    return {
        "tokenizer": _fake_tokenizer(),
        "role_llm_funcs": {"keyword": keyword_func},
        "role_supports_token_tracker": {"keyword": supports},
        "addon_params": {"language": "en"},
    }


@pytest.mark.asyncio
async def test_extract_keywords_only_receives_token_tracker_when_supported():
    async def aware_model(*_args, token_tracker=None, **_kwargs):
        token_tracker.add_usage(
            {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        )
        return '{"high_level_keywords": ["a"], "low_level_keywords": ["b"]}'

    tracker = TokenTracker()
    cfg = _keyword_global_config(aware_model, supports=True)
    hl, ll = await extract_keywords_only(
        "some text",
        QueryParam(mode="local"),
        cfg,
        hashing_kv=None,
        token_tracker=tracker,
    )
    assert (hl, ll) == (["a"], ["b"])
    assert tracker.call_count == 1


@pytest.mark.asyncio
async def test_extract_keywords_only_never_passes_token_tracker_when_unsupported():
    seen_kwargs = {}

    async def strict_model(*_args, **kwargs):
        seen_kwargs.update(kwargs)
        return '{"high_level_keywords": ["a"], "low_level_keywords": ["b"]}'

    tracker = TokenTracker()
    cfg = _keyword_global_config(strict_model, supports=False)
    await extract_keywords_only(
        "some text",
        QueryParam(mode="local"),
        cfg,
        hashing_kv=None,
        token_tracker=tracker,
    )
    assert "token_tracker" not in seen_kwargs
    assert tracker.call_count == 0


@pytest.mark.asyncio
async def test_get_keywords_from_query_skips_llm_and_tracker_when_preset():
    tracker = TokenTracker()

    async def never_called(*_args, **_kwargs):
        raise AssertionError("must not call the LLM when keywords are preset")

    cfg = _keyword_global_config(never_called, supports=True)
    param = QueryParam(
        mode="local", hl_keywords=["preset-hl"], ll_keywords=["preset-ll"]
    )
    hl, ll = await get_keywords_from_query(
        "some text", param, cfg, hashing_kv=None, token_tracker=tracker
    )
    assert (hl, ll) == (["preset-hl"], ["preset-ll"])
    assert tracker.call_count == 0
