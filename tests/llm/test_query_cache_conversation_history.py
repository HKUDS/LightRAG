"""conversation_history must never share the ``cache_type="query"`` answer cache.

The answer cache key deliberately omits ``QueryParam.conversation_history`` so the
cache stays shared across callers instead of degenerating into a per-session one.
But the history *is* handed to the model as ``history_messages``, so it changes the
generated text: a history-conditioned answer is not interchangeable with the
history-blind key it would be filed under. ``_answer_cache_kv`` therefore takes
history-bearing requests off both sides of the answer cache, while the keywords
cache keeps using ``hashing_kv`` directly.

The same commit bumps the answer-cache policy version, so entries written before
the bypass existed -- which may hold history-conditioned text and record no
history to detect it by -- can no longer be read.
"""

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.utils import (
    compute_args_hash,
    get_llm_cache_identity,
    serialize_llm_cache_identity,
)


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


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
                "content": "Conversation history cache partitioning test chunk.",
                "file_path": "test.md",
            }
        ]


class _RecordingModel:
    """Counts calls and records the history each call received.

    Returns ``f"answer-{calls}"`` rather than indexing a fixed list on purpose: a
    cache hit that should not have happened then surfaces as a failed assert on
    the content or the call count, never as an IndexError.
    """

    def __init__(self):
        self.calls = 0
        self.histories = []

    async def __call__(self, *_args, **kwargs):
        self.calls += 1
        self.histories.append(kwargs.get("history_messages"))
        return f"answer-{self.calls}"


def _query_global_config(llm_func, keyword_func=None) -> dict:
    role_llm_funcs = {"query": llm_func}
    if keyword_func is not None:
        role_llm_funcs["keyword"] = keyword_func
    return {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": role_llm_funcs,
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


QUERY = "who is Tesla?"
STALE = "STALE-HISTORY-CONDITIONED-ANSWER"

HISTORY_A = [
    {"role": "user", "content": "we were discussing Tesla the carmaker"},
    {"role": "assistant", "content": "Yes, the automotive company."},
]
HISTORY_B = [
    {"role": "user", "content": "we were discussing Nikola Tesla the physicist"},
    {"role": "assistant", "content": "Yes, the inventor."},
]


def _answer_cache_keys(cache: _FakeKVStorage) -> list[str]:
    return [key for key in cache._store if ":query:" in key]


def _legacy_answer_cache_key(
    param: QueryParam,
    cfg: dict,
    *,
    keywords: tuple[str, str] | None = None,
) -> str:
    """Frozen snapshot of the pre-v2 answer-cache key composition.

    Deliberately duplicates the historical argument list instead of reusing
    production code: these tests exist to prove that entries written under the
    older scheme are never served again. Do NOT refresh this when new key fields
    are added -- a miss must stay a miss.

    ``keywords`` is ``(hl_keywords_str, ll_keywords_str)`` for the ``kg_query``
    variant, which carried them in the key; ``naive_query`` did not.
    """
    args = [
        param.mode,
        QUERY,
        param.response_type,
        param.top_k,
        param.chunk_top_k,
        param.max_entity_tokens,
        param.max_relation_tokens,
        param.max_total_tokens,
    ]
    if keywords is not None:
        args.extend(keywords)
    args.extend(
        [
            param.user_prompt or "",
            param.enable_rerank,
            cfg.get("enable_content_headings", False),
            "\n<llm_identity>\n",
            serialize_llm_cache_identity(get_llm_cache_identity(cfg, "query")),
        ]
    )
    return f"{param.mode}:query:{compute_args_hash(*args)}"


# ---------------------------------------------------------------------------
# naive_query
# ---------------------------------------------------------------------------


def _naive_param(**overrides) -> QueryParam:
    return QueryParam(mode="naive", enable_rerank=False, **overrides)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_history_turn_does_not_seed_cache_for_plain_turn():
    """A history-conditioned answer must never become the shared cache entry."""
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await naive_query(
        QUERY,
        chunks_vdb,
        _naive_param(conversation_history=HISTORY_A),
        cfg,
        hashing_kv=cache,
    )
    assert first.content == "answer-1"
    # Pins the premise: the history really does reach the model.
    assert model.histories[0] == HISTORY_A
    assert cache._store == {}

    second = await naive_query(QUERY, chunks_vdb, _naive_param(), cfg, hashing_kv=cache)
    assert second.content == "answer-2"
    assert model.calls == 2
    assert model.histories[1] == []
    # The plain answer is still cached: the bypass is not a blanket disable.
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_history_turn_ignores_existing_plain_cache_entry():
    """A multi-turn caller must not be served an answer that ignored its history."""
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await naive_query(QUERY, chunks_vdb, _naive_param(), cfg, hashing_kv=cache)
    assert first.content == "answer-1"
    assert len(_answer_cache_keys(cache)) == 1

    second = await naive_query(
        QUERY,
        chunks_vdb,
        _naive_param(conversation_history=HISTORY_A),
        cfg,
        hashing_kv=cache,
    )
    assert second.content == "answer-2"
    assert model.calls == 2
    assert model.histories[1] == HISTORY_A
    # Read bypassed (calls == 2) and write bypassed (no new entry).
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_distinct_histories_neither_share_nor_grow_the_cache():
    """Distinct histories get distinct answers and add zero cache entries.

    The empty-store assertion is load-bearing in two directions: without the
    bypass the two turns share one entry and the second caller gets the first
    caller's answer, while keying the cache on conversation_history instead
    would leave two per-session entries here. Do not relax it.
    """
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await naive_query(
        QUERY,
        chunks_vdb,
        _naive_param(conversation_history=HISTORY_A),
        cfg,
        hashing_kv=cache,
    )
    second = await naive_query(
        QUERY,
        chunks_vdb,
        _naive_param(conversation_history=HISTORY_B),
        cfg,
        hashing_kv=cache,
    )

    assert (first.content, second.content) == ("answer-1", "answer-2")
    assert model.calls == 2
    assert cache._store == {}


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history_overrides",
    [
        pytest.param({}, id="unset"),
        pytest.param({"conversation_history": []}, id="empty-list"),
        pytest.param({"conversation_history": None}, id="none"),
    ],
)
async def test_naive_single_turn_query_still_hits_cache(history_overrides):
    """Over-fix guard: history-free turns keep sharing one cache entry.

    This passes before the bypass exists too -- its job is to fail if the bypass
    is ever widened. The ``none`` case is what distinguishes the truthiness test
    from an ``is not None`` test, which would disable caching for SDK callers
    that pass ``conversation_history=None`` explicitly.
    """
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await naive_query(
        QUERY, chunks_vdb, _naive_param(**history_overrides), cfg, hashing_kv=cache
    )
    second = await naive_query(
        QUERY, chunks_vdb, _naive_param(**history_overrides), cfg, hashing_kv=cache
    )

    assert first.content == "answer-1"
    assert second.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_pre_version_cache_entry_is_not_served():
    """An entry written under the pre-v2 key scheme must be unreachable.

    Upgrading only bypasses the cache for *new* history-bearing requests; a
    history-conditioned answer already persisted by an older version would still
    be served to plain callers under an unchanged key. The policy version in the
    hash retires those entries instead.
    """
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    param = _naive_param()

    legacy_key = _legacy_answer_cache_key(param, cfg)
    cache._store[legacy_key] = {"return": STALE, "create_time": 1}

    first = await naive_query(QUERY, chunks_vdb, param, cfg, hashing_kv=cache)
    assert first.content == "answer-1"
    assert model.calls == 1

    fresh_keys = [key for key in _answer_cache_keys(cache) if key != legacy_key]
    assert len(fresh_keys) == 1

    # The freshly written entry is itself cacheable as usual.
    second = await naive_query(QUERY, chunks_vdb, param, cfg, hashing_kv=cache)
    assert second.content == "answer-1"
    assert model.calls == 1


# ---------------------------------------------------------------------------
# kg_query
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
    # mode="local" plus preset ll_keywords short-circuits keyword extraction
    # without tripping the empty-hl_keywords warning that hybrid/mix would.
    return QueryParam(
        mode="local", enable_rerank=False, ll_keywords=["Tesla"], **overrides
    )


async def _run_kg_query(param, cfg, cache):
    return await kg_query(QUERY, None, None, None, None, param, cfg, hashing_kv=cache)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_history_turn_does_not_seed_cache_for_plain_turn(stub_query_context):
    """Same guarantee as the naive case, on kg_query's own cache sites."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await _run_kg_query(_kg_param(conversation_history=HISTORY_A), cfg, cache)
    assert first.content == "answer-1"
    assert model.histories[0] == HISTORY_A
    assert cache._store == {}

    second = await _run_kg_query(_kg_param(), cfg, cache)
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_history_turn_ignores_existing_plain_cache_entry(stub_query_context):
    """A multi-turn kg_query caller is not served a history-ignoring answer."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await _run_kg_query(_kg_param(), cfg, cache)
    assert first.content == "answer-1"
    assert len(_answer_cache_keys(cache)) == 1

    second = await _run_kg_query(_kg_param(conversation_history=HISTORY_A), cfg, cache)
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_pre_version_cache_entry_is_not_served(stub_query_context):
    """kg_query must carry the policy version too, not just naive_query."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    param = _kg_param()

    # kg_query hashes hl_keywords_str before ll_keywords_str; the preset
    # ll_keywords=["Tesla"] leaves hl empty.
    legacy_key = _legacy_answer_cache_key(param, cfg, keywords=("", "Tesla"))
    cache._store[legacy_key] = {"return": STALE, "create_time": 1}

    first = await _run_kg_query(param, cfg, cache)
    assert first.content == "answer-1"
    assert model.calls == 1

    fresh_keys = [key for key in _answer_cache_keys(cache) if key != legacy_key]
    assert len(fresh_keys) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_keywords_cache_still_shared_across_histories(stub_query_context):
    """The keywords cache must keep the raw hashing_kv.

    Keywords are derived from the query text alone and never see the history, so
    replacing the keyword call site's storage along with the answer cache's would
    cost a re-extraction per turn for no benefit.
    """
    cache = _FakeKVStorage()
    model = _RecordingModel()
    keyword_calls = 0

    async def keyword_model(*_args, **_kwargs):
        nonlocal keyword_calls
        keyword_calls += 1
        return '{"high_level_keywords": ["physics"], "low_level_keywords": ["Tesla"]}'

    cfg = _query_global_config(model, keyword_func=keyword_model)

    def _param(history):
        # No preset keywords: force real extraction so its cache is exercised.
        return QueryParam(
            mode="local", enable_rerank=False, conversation_history=history
        )

    await kg_query(
        QUERY, None, None, None, None, _param(HISTORY_A), cfg, hashing_kv=cache
    )
    await kg_query(
        QUERY, None, None, None, None, _param(HISTORY_B), cfg, hashing_kv=cache
    )

    keyword_keys = [key for key in cache._store if ":keywords:" in key]
    assert keyword_calls == 1
    assert len(keyword_keys) == 1
    assert keyword_keys[0].startswith("local:keywords:")
    # ...while neither history-conditioned answer was cached.
    assert _answer_cache_keys(cache) == []
    assert model.calls == 2
