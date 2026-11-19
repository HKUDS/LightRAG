"""``enable_hybrid`` must partition the answer cache (plan A6).

On a hybrid-capable vector backend the flag switches retrieval to
lexical+vector fusion, so the same request builds a different context with
the flag on than with it off; an answer cached under one arm must never serve
the other. The key component is conditional -- present only when the flag is
on -- so entries written before the flag existed keep hitting with the flag
off. The frozen legacy-key snapshot imported below pins that half; it is
deliberately NOT refreshed here.

Keys are observed as the REAL ``compute_args_hash`` / ``generate_cache_key``
helpers write them through ``handle_cache`` / ``save_to_cache``: only the KV
backend is a dict fake, the key construction and cache path are production
code. Keyword extraction is retrieval-independent and keeps its
retrieval-free key, mirroring how ``enable_rerank`` stays out of that hash.
"""

import json

import pytest

from lightrag.base import QueryParam
from lightrag.operate import extract_keywords_only

from tests.llm.test_query_cache_user_prompt_prefix import (
    _answer_cache_keys,
    _FakeKVStorage,
    _preprefix_answer_cache_key,
    _query_global_config,
    _RecordingModel,
    _run_kg,
    _run_naive,
    stub_query_context,  # noqa: F401  (pytest fixture)
)

pytestmark = pytest.mark.offline

HL = ["launch costs"]
LL = ["Tesla"]

_RUNNERS = [
    pytest.param(_run_kg, mode, id=f"kg-{mode}")
    for mode in ("local", "global", "hybrid", "mix")
] + [pytest.param(_run_naive, "naive", id="naive")]


def _param(mode: str, **overrides) -> QueryParam:
    overrides.setdefault("enable_rerank", False)
    return QueryParam(mode=mode, hl_keywords=HL, ll_keywords=LL, **overrides)


def _legacy_keywords(mode: str) -> tuple[str, str] | None:
    """The snapshot's keyword components: KG carries them, naive does not."""
    return None if mode == "naive" else (", ".join(HL), ", ".join(LL))


class _KeywordModel:
    """Records calls and returns a fixed, parseable keyword payload."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(
            {"high_level_keywords": HL, "low_level_keywords": LL},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Key construction: the flag partitions, and only the flag's arm moves.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,mode", _RUNNERS)
async def test_hybrid_on_and_off_write_different_keys(
    runner, mode, stub_query_context  # noqa: F811
):
    written = []
    for flag in (False, True):
        cache = _FakeKVStorage()
        cfg = _query_global_config(_RecordingModel())
        await runner(_param(mode, enable_hybrid=flag), cfg, cache)
        (key,) = _answer_cache_keys(cache)
        written.append(key)

    assert written[0] != written[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,mode", _RUNNERS)
async def test_hybrid_off_keeps_the_legacy_key(
    runner, mode, stub_query_context  # noqa: F811
):
    """Default-off costs no deployment its warm answer cache."""
    cache = _FakeKVStorage()
    cfg = _query_global_config(_RecordingModel())
    param = _param(mode, enable_hybrid=False)

    await runner(param, cfg, cache)

    (written,) = _answer_cache_keys(cache)
    assert written == _preprefix_answer_cache_key(
        param, cfg, keywords=_legacy_keywords(mode)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,mode", _RUNNERS)
async def test_legacy_entry_serves_off_but_never_on(
    runner, mode, stub_query_context  # noqa: F811
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    param = _param(mode, enable_hybrid=False)
    cache._store[
        _preprefix_answer_cache_key(param, cfg, keywords=_legacy_keywords(mode))
    ] = {"return": "PRE-HYBRID-ANSWER", "create_time": 1}

    off = await runner(param, cfg, cache)
    assert off.content == "PRE-HYBRID-ANSWER"
    assert model.calls == 0

    on = await runner(_param(mode, enable_hybrid=True), cfg, cache)
    assert on.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,mode", _RUNNERS)
@pytest.mark.parametrize(
    "dimension,arm_a,arm_b",
    [
        pytest.param("enable_rerank", False, True, id="rerank"),
        pytest.param("top_k", 60, 30, id="top-k"),
        pytest.param("chunk_top_k", 20, 10, id="chunk-top-k"),
    ],
)
async def test_other_dimensions_still_partition_with_hybrid_on(
    runner, mode, dimension, arm_a, arm_b, stub_query_context  # noqa: F811
):
    """The new component must not collapse the pre-existing partitioning."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await runner(
        _param(mode, enable_hybrid=True, **{dimension: arm_a}), cfg, cache
    )
    second = await runner(
        _param(mode, enable_hybrid=True, **{dimension: arm_b}), cfg, cache
    )

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 2


# ---------------------------------------------------------------------------
# Functional: on -> off -> on through the real cache path, no contamination.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,mode", _RUNNERS)
@pytest.mark.parametrize(
    "sequence",
    [
        pytest.param(("on", "off", "on"), id="on-off-on"),
        pytest.param(("off", "on", "off"), id="off-on-off"),
    ],
)
async def test_arm_answers_never_cross_contaminate(
    runner, mode, sequence, stub_query_context  # noqa: F811
):
    """A cache HIT means no second LLM call for the same arm; a shared key
    would hand one arm the other's answer instead."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    expected = ["answer-1", "answer-2", "answer-1"]

    for arm, want in zip(sequence, expected):
        result = await runner(_param(mode, enable_hybrid=(arm == "on")), cfg, cache)
        assert result.content == want, f"arm {arm} was served a contaminated answer"

    assert model.calls == 2
    keys = _answer_cache_keys(cache)
    assert len(keys) == 2
    assert {cache._store[key]["return"] for key in keys} == {"answer-1", "answer-2"}


# ---------------------------------------------------------------------------
# Keyword extraction stays retrieval-free: no unnecessary misses.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["mix", "local"])
async def test_keyword_cache_key_ignores_enable_hybrid(mode):
    cache = _FakeKVStorage()
    model = _KeywordModel()
    cfg = _query_global_config(model)
    cfg["role_llm_funcs"] = {"keyword": model}

    on = await extract_keywords_only(
        "who is Tesla?", _param(mode, enable_hybrid=True), cfg, hashing_kv=cache
    )
    off = await extract_keywords_only(
        "who is Tesla?", _param(mode, enable_hybrid=False), cfg, hashing_kv=cache
    )

    assert on == off == (HL, LL)
    assert model.calls == 1
    assert len([key for key in cache._store if ":keywords:" in key]) == 1
