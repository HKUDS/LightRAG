"""The server-side ``user_prompt_prefix`` must partition the answer cache.

``USER_PROMPT_PREFIX`` lets an operator prepend global instructions to every
request's ``QueryParam.user_prompt``. It changes the generated text, so it has
to reach the answer-cache key: otherwise an operator who edits the prefix keeps
being served answers written under the old one.

The key component was changed in place -- ``query_param.user_prompt or ""``
became the COMPOSED text -- rather than appended as a new component. That is
what keeps ``_ANSWER_CACHE_POLICY_VERSION`` at v2: with no prefix configured
the composed text is byte-identical to the old value, so every entry written
before this feature existed still hits. These tests pin both halves.

``disable_user_prompt_prefix`` is deliberately NOT a key component of its own:
it acts only through the composed text, so a disabled request with a prefix
configured must share an entry with a request that never had one.
"""

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.utils import (
    Tokenizer,
    compute_args_hash,
    get_llm_cache_identity,
    serialize_llm_cache_identity,
)


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _FakeTokenizer() -> Tokenizer:
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
                "content": "User prompt prefix cache partitioning test chunk.",
                "file_path": "test.md",
            }
        ]


class _RecordingModel:
    """Counts calls and records the system prompt each call received."""

    def __init__(self):
        self.calls = 0
        self.system_prompts = []

    async def __call__(self, *_args, **kwargs):
        self.calls += 1
        self.system_prompts.append(kwargs.get("system_prompt"))
        return f"answer-{self.calls}"


QUERY = "who is Tesla?"
PREFIX_A = "Always answer in Chinese.\n\n"
PREFIX_B = "Always answer in French.\n\n"
USER_PROMPT = "Reply in one sentence."


def _query_global_config(llm_func, prefix=None) -> dict:
    cfg = {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"query": llm_func},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }
    if prefix is not None:
        cfg["user_prompt_prefix"] = prefix
    return cfg


def _answer_cache_keys(cache: _FakeKVStorage) -> list[str]:
    return [key for key in cache._store if ":query:" in key]


def _preprefix_answer_cache_key(
    param: QueryParam,
    cfg: dict,
    *,
    keywords: tuple[str, str] | None = None,
) -> str:
    """Frozen snapshot of the answer-cache key as composed BEFORE this feature.

    Deliberately duplicates the historical argument list rather than reusing
    production code: the point is to prove an entry written by the old code is
    still served when no prefix is configured. Do NOT refresh this when new key
    fields are added -- if a later change makes this miss, that change costs
    every deployment its warm answer cache and must be a deliberate decision.
    """
    args = [
        "query-answer-cache-v2",
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


@pytest.fixture
def stub_query_context(monkeypatch):
    """Skip retrieval: kg_query only forwards its storage args into this call."""

    async def _fake_build_query_context(*_args, **_kwargs):
        return QueryContextResult(context="KG CONTEXT", raw_data={})

    monkeypatch.setattr(
        "lightrag.operate._build_query_context", _fake_build_query_context
    )


def _naive_param(**overrides) -> QueryParam:
    return QueryParam(mode="naive", enable_rerank=False, **overrides)


def _kg_param(**overrides) -> QueryParam:
    return QueryParam(
        mode="local", enable_rerank=False, ll_keywords=["Tesla"], **overrides
    )


async def _run_naive(param, cfg, cache, system_prompt=None):
    return await naive_query(
        QUERY,
        _FakeChunksVDB(),
        param,
        cfg,
        hashing_kv=cache,
        system_prompt=system_prompt,
    )


async def _run_kg(param, cfg, cache, system_prompt=None):
    return await kg_query(
        QUERY,
        None,
        None,
        None,
        None,
        param,
        cfg,
        hashing_kv=cache,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# The prefix reaches the model.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_prefix_is_prepended_into_the_system_prompt(
    runner, param, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    await runner(param(user_prompt=USER_PROMPT), cfg, cache)

    sys_prompt = model.system_prompts[0]
    assert PREFIX_A.strip() in sys_prompt
    assert f"{PREFIX_A}{USER_PROMPT}" in sys_prompt
    # The "no instructions" placeholder must be gone once anything is composed.
    assert "n/a" not in sys_prompt


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
@pytest.mark.parametrize("empty", [None, ""])
async def test_prefix_alone_reaches_the_model_without_a_user_prompt(
    runner, param, empty, stub_query_context
):
    """The common deployment: operator sets a policy, callers send nothing.

    Both spellings of "nothing" must behave identically. ``""`` is not
    hypothetical -- it is the literal default the WebUI ships in its query
    settings, so treating it as "suppress the prefix" would silently disable
    the operator's policy for every WebUI user.
    """
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    await runner(param(user_prompt=empty), cfg, cache)

    assert PREFIX_A.strip() in model.system_prompts[0]
    assert "n/a" not in model.system_prompts[0]


# ---------------------------------------------------------------------------
# Cache partitioning.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param,prompt_a,prompt_b",
    [
        (
            _run_naive,
            _naive_param,
            "Policy A\n{content_data}\n{response_type}\n{user_prompt}",
            "Policy B\n{content_data}\n{response_type}\n{user_prompt}",
        ),
        (
            _run_kg,
            _kg_param,
            "Policy A\n{context_data}\n{response_type}\n{user_prompt}",
            "Policy B\n{context_data}\n{response_type}\n{user_prompt}",
        ),
    ],
)
async def test_changing_system_prompt_does_not_serve_old_answer(
    runner, param, prompt_a, prompt_b, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await runner(param(), cfg, cache, system_prompt=prompt_a)
    second = await runner(param(), cfg, cache, system_prompt=prompt_b)

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 2


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param,system_prompt",
    [
        (
            _run_naive,
            _naive_param,
            "Policy A\n{content_data}\n{response_type}\n{user_prompt}",
        ),
        (
            _run_kg,
            _kg_param,
            "Policy A\n{context_data}\n{response_type}\n{user_prompt}",
        ),
    ],
)
async def test_same_system_prompt_still_hits_cache(
    runner, param, system_prompt, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await runner(param(), cfg, cache, system_prompt=system_prompt)
    second = await runner(param(), cfg, cache, system_prompt=system_prompt)

    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_changing_the_prefix_does_not_serve_the_old_answer(
    runner, param, stub_query_context
):
    """The whole reason the prefix has to be in the key."""
    cache = _FakeKVStorage()
    model = _RecordingModel()

    first = await runner(
        param(user_prompt=USER_PROMPT), _query_global_config(model, PREFIX_A), cache
    )
    assert first.content == "answer-1"

    second = await runner(
        param(user_prompt=USER_PROMPT), _query_global_config(model, PREFIX_B), cache
    )
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 2


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_same_prefix_still_hits_the_cache(runner, param, stub_query_context):
    """Partitioning must not degenerate into never caching."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    first = await runner(param(user_prompt=USER_PROMPT), cfg, cache)
    second = await runner(param(user_prompt=USER_PROMPT), cfg, cache)

    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_prefix_and_user_prompt_do_not_collide_across_the_boundary(
    runner, param, stub_query_context
):
    """Byte concatenation must not make two different splits share a key.

    ``"ab" + "c"`` and ``"a" + "bc"`` compose the same text but are different
    configurations; they legitimately produce the SAME prompt, so they SHOULD
    share an entry. This pins that the composed text -- not the raw fields --
    is what the key reflects.
    """
    cache = _FakeKVStorage()
    model = _RecordingModel()

    first = await runner(
        param(user_prompt="c"), _query_global_config(model, "ab"), cache
    )
    second = await runner(
        param(user_prompt="bc"), _query_global_config(model, "a"), cache
    )

    assert first.content == second.content == "answer-1"
    assert model.calls == 1


# ---------------------------------------------------------------------------
# The opt-out switch is not itself a key component.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_disabled_prefix_shares_the_entry_with_no_prefix_configured(
    runner, param, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()

    first = await runner(
        param(user_prompt=USER_PROMPT), _query_global_config(model), cache
    )
    assert first.content == "answer-1"

    second = await runner(
        param(user_prompt=USER_PROMPT, disable_user_prompt_prefix=True),
        _query_global_config(model, PREFIX_A),
        cache,
    )
    assert second.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_disabled_prefix_keeps_the_prefix_out_of_the_prompt(
    runner, param, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    await runner(
        param(user_prompt=USER_PROMPT, disable_user_prompt_prefix=True), cfg, cache
    )

    sys_prompt = model.system_prompts[0]
    assert PREFIX_A.strip() not in sys_prompt
    assert USER_PROMPT in sys_prompt


# ---------------------------------------------------------------------------
# The invariant: an unconfigured prefix invalidates nothing.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_entry_written_before_the_prefix_feature_still_hits():
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    param = _naive_param(user_prompt=USER_PROMPT)

    cache._store[_preprefix_answer_cache_key(param, cfg)] = {
        "return": "PRE-PREFIX-ANSWER",
        "create_time": 1,
    }

    result = await _run_naive(param, cfg, cache)
    assert result.content == "PRE-PREFIX-ANSWER"
    assert model.calls == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_entry_written_before_the_prefix_feature_still_hits(
    stub_query_context,
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    param = _kg_param(user_prompt=USER_PROMPT)

    cache._store[_preprefix_answer_cache_key(param, cfg, keywords=("", "Tesla"))] = {
        "return": "PRE-PREFIX-ANSWER",
        "create_time": 1,
    }

    result = await _run_kg(param, cfg, cache)
    assert result.content == "PRE-PREFIX-ANSWER"
    assert model.calls == 0


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_configured_prefix_does_not_serve_a_pre_prefix_entry(
    runner, param, stub_query_context
):
    """The other side of the invariant: once a prefix exists, old entries miss."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)
    p = param(user_prompt=USER_PROMPT)

    keywords = ("", "Tesla") if p.mode == "local" else None
    cache._store[_preprefix_answer_cache_key(p, cfg, keywords=keywords)] = {
        "return": "PRE-PREFIX-ANSWER",
        "create_time": 1,
    }

    result = await runner(p, cfg, cache)
    assert result.content == "answer-1"
    assert model.calls == 1


# ---------------------------------------------------------------------------
# Transparency: the prefix must not leak into cache metadata.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
async def test_cache_metadata_records_the_raw_request_user_prompt(
    runner, param, stub_query_context
):
    """``queryparam`` mirrors the REQUEST; the operator prefix is not part of it."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    await runner(param(user_prompt=USER_PROMPT), cfg, cache)

    entry = cache._store[_answer_cache_keys(cache)[0]]
    queryparam = entry["queryparam"]
    assert queryparam["user_prompt"] == USER_PROMPT
    assert PREFIX_A not in str(queryparam)


# ---------------------------------------------------------------------------
# The prefix is counted against the context token budget.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_prefix_shrinks_the_available_chunk_token_budget(monkeypatch):
    """A configured prefix must be counted, or a long one overfills the context.

    ``_build_context_str`` estimates the system-prompt overhead to size
    ``available_chunk_tokens``. Before this feature the estimate used the raw
    request ``user_prompt``, so a server-side prefix would have consumed
    context the estimate knew nothing about. The three cases below pin that the
    budget moves by exactly the prefix length, and not at all when unset or
    opted out.
    """
    import lightrag.operate as operate

    limits = []
    original = operate.process_chunks_unified

    async def _spy(**kwargs):
        limits.append(kwargs["chunk_token_limit"])
        return await original(**kwargs)

    monkeypatch.setattr(operate, "process_chunks_unified", _spy)

    entities = [{"entity": "Scrooge", "description": "A miser."}]
    relations = [{"src": "Scrooge", "tgt": "Marley", "description": "Partners."}]
    chunks = [
        {"chunk_id": f"c{i}", "content": f"Chunk {i}.", "file_path": "a.md"}
        for i in range(5)
    ]
    # The fake tokenizer is 1 char == 1 token, so the delta is exact.
    prefix = "P" * 3000

    async def _budget_for(prefix_value, disable):
        cfg = {
            "tokenizer": _FakeTokenizer(),
            "max_total_tokens": 30000,
            "user_prompt_prefix": prefix_value,
        }
        param = QueryParam(
            mode="hybrid",
            enable_rerank=False,
            user_prompt=USER_PROMPT,
            disable_user_prompt_prefix=disable,
        )
        await operate._build_context_str(entities, relations, chunks, QUERY, param, cfg)
        return limits[-1]

    baseline = await _budget_for("", False)
    with_prefix = await _budget_for(prefix, False)
    disabled = await _budget_for(prefix, True)

    assert with_prefix == baseline - len(prefix)
    assert disabled == baseline


@pytest.mark.offline
@pytest.mark.asyncio
async def test_budget_charges_the_prefix_only_when_the_template_renders_it(
    monkeypatch,
):
    """kg_query picks its template AFTER the context is built and budgeted.

    Without the forwarded ``system_prompt`` the estimate used the default
    template unconditionally, so a caller passing a custom one got a budget
    computed against the wrong text -- and a configured prefix was charged
    against the chunk allowance even when that template had no
    ``{user_prompt}`` placeholder to render it into, discarding chunks to make
    room for text that was never sent.
    """
    import lightrag.operate as operate

    limits = []
    original = operate.process_chunks_unified

    async def _spy(**kwargs):
        limits.append(kwargs["chunk_token_limit"])
        return await original(**kwargs)

    monkeypatch.setattr(operate, "process_chunks_unified", _spy)

    entities = [{"entity": "Scrooge", "description": "A miser."}]
    relations = [{"src": "Scrooge", "tgt": "Marley", "description": "Partners."}]
    chunks = [
        {"chunk_id": f"c{i}", "content": f"Chunk {i}.", "file_path": "a.md"}
        for i in range(5)
    ]
    # No {user_prompt} placeholder: nothing this template renders can carry the
    # prefix, so none of its tokens reach the model.
    template_without_slot = (
        "Custom template. Context: {context_data} Type: {response_type}"
    )
    prefix = "P" * 3000  # the fake tokenizer is 1 char == 1 token

    async def _budget(prefix_value, system_prompt):
        cfg = {
            "tokenizer": _FakeTokenizer(),
            "max_total_tokens": 30000,
            "user_prompt_prefix": prefix_value,
        }
        param = QueryParam(mode="hybrid", enable_rerank=False, user_prompt=USER_PROMPT)
        await operate._build_context_str(
            entities,
            relations,
            chunks,
            QUERY,
            param,
            cfg,
            system_prompt=system_prompt,
        )
        return limits[-1]

    # A template that DOES render the prefix still pays for it.
    default_baseline = await _budget("", None)
    default_with_prefix = await _budget(prefix, None)
    assert default_with_prefix == default_baseline - len(prefix)

    # A template that does NOT render it is charged nothing.
    custom_baseline = await _budget("", template_without_slot)
    custom_with_prefix = await _budget(prefix, template_without_slot)
    assert custom_with_prefix == custom_baseline

    # And the custom template is budgeted at its own size, not the default's --
    # the pre-existing defect this forwarding also repairs.
    assert custom_baseline != default_baseline


# ---------------------------------------------------------------------------
# Debug switches preview the REAL request.
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_only_need_context_budgets_exactly_like_a_real_query(monkeypatch):
    """The chunk budget must not depend on whether an answer will be generated.

    ``only_need_context`` returns before the system prompt is rendered, so it is
    tempting to read the prefix charge as waste and skip it. That would be
    wrong: these switches exist to PREVIEW what a real query sends. A
    retrieval-only call that skipped the charge would report more chunks than
    the live path delivers, and an operator sizing their context against that
    number would be silently truncated at answer time -- the debug switch
    becomes the source of the bug it is meant to expose.
    """
    import lightrag.operate as operate

    limits = []
    original = operate.process_chunks_unified

    async def _spy(**kwargs):
        limits.append(kwargs["chunk_token_limit"])
        return await original(**kwargs)

    monkeypatch.setattr(operate, "process_chunks_unified", _spy)

    entities = [{"entity": "Scrooge", "description": "A miser."}]
    relations = [{"src": "Scrooge", "tgt": "Marley", "description": "Partners."}]
    chunks = [
        {"chunk_id": f"c{i}", "content": f"Chunk {i}.", "file_path": "a.md"}
        for i in range(5)
    ]

    async def _budget(only_need_context):
        cfg = {
            "tokenizer": _FakeTokenizer(),
            "max_total_tokens": 30000,
            "user_prompt_prefix": "P" * 3000,
        }
        param = QueryParam(
            mode="hybrid",
            enable_rerank=False,
            user_prompt=USER_PROMPT,
            only_need_context=only_need_context,
        )
        await operate._build_context_str(entities, relations, chunks, QUERY, param, cfg)
        return limits[-1]

    assert await _budget(True) == await _budget(False)


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param", [(_run_naive, _naive_param), (_run_kg, _kg_param)]
)
@pytest.mark.parametrize("user_prompt", ["Reply in one sentence.", ""])
async def test_only_need_prompt_returns_the_composed_prefix(
    runner, param, user_prompt, stub_query_context
):
    """The debug prompt must show what the model would actually receive.

    Covers the empty-``user_prompt`` case on this exit too: the prefix alone
    becomes the instructions, exactly as it does on the path that reaches the
    model. Previously only the LLM-bound ``system_prompt`` was asserted, not
    what this switch hands back to the caller.
    """
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    result = await runner(
        param(user_prompt=user_prompt, only_need_prompt=True), cfg, cache
    )

    assert PREFIX_A.strip() in result.content
    assert "n/a" not in result.content
    if user_prompt:
        assert user_prompt in result.content
    # A debug preview must not have consulted the model.
    assert model.calls == 0
