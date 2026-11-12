"""A keyword fallback must partition the answer cache.

When keyword extraction leaves a mode's own list empty, the KG search swaps
the other list in on the same store. Before that routing existed the same
request read a different store (``local`` with no low-level keywords ran a
relationship search), so an answer cached back then was written from another
context and must not be served. The fallback name therefore joins the
answer-cache key -- only when a fallback applies, so every other entry written
before the change still hits.
"""

import pytest

from lightrag.base import QueryParam

from .test_query_cache_user_prompt_prefix import (
    _answer_cache_keys,
    _FakeKVStorage,
    _preprefix_answer_cache_key,
    _query_global_config,
    _RecordingModel,
    _run_kg,
    stub_query_context,  # noqa: F401  (pytest fixture)
)

pytestmark = pytest.mark.offline

HL = ["launch costs"]
LL = ["Tesla"]


def _param(mode: str, *, hl: list[str], ll: list[str]) -> QueryParam:
    return QueryParam(mode=mode, enable_rerank=False, hl_keywords=hl, ll_keywords=ll)


async def _cache_key(param: QueryParam) -> tuple[str, str]:
    """Return (key written by the current code, key the old code would use)."""
    cache = _FakeKVStorage()
    cfg = _query_global_config(_RecordingModel())

    await _run_kg(param, cfg, cache)

    (written,) = _answer_cache_keys(cache)
    old = _preprefix_answer_cache_key(
        param,
        cfg,
        keywords=(", ".join(param.hl_keywords), ", ".join(param.ll_keywords)),
    )
    return written, old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, hl, ll",
    [
        pytest.param("local", HL, [], id="local-ll-empty"),
        pytest.param("hybrid", HL, [], id="hybrid-ll-empty"),
        pytest.param("mix", HL, [], id="mix-ll-empty"),
        pytest.param("global", [], LL, id="global-hl-empty"),
    ],
)
async def test_fallback_request_does_not_reuse_a_pre_fallback_answer(
    mode,
    hl,
    ll,
    stub_query_context,  # noqa: F811
):
    written, old = await _cache_key(_param(mode, hl=hl, ll=ll))

    assert written != old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, hl, ll",
    [
        pytest.param("local", HL, LL, id="local"),
        pytest.param("global", HL, LL, id="global"),
        pytest.param("hybrid", HL, LL, id="hybrid"),
        pytest.param("mix", HL, LL, id="mix"),
        # An empty list that triggers no fallback leaves the key alone too.
        pytest.param("local", [], LL, id="local-hl-empty"),
        pytest.param("hybrid", [], LL, id="hybrid-hl-empty"),
        pytest.param("global", HL, [], id="global-ll-empty"),
    ],
)
async def test_request_without_fallback_keeps_its_existing_cache_key(
    mode,
    hl,
    ll,
    stub_query_context,  # noqa: F811
):
    written, old = await _cache_key(_param(mode, hl=hl, ll=ll))

    assert written == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, hl, ll, expected",
    [
        # A fallback reports the empty list itself: no second, bare warning.
        pytest.param("local", HL, [], [], id="local-ll-empty-fallback"),
        pytest.param("hybrid", HL, [], [], id="hybrid-ll-empty-fallback"),
        pytest.param("global", [], LL, [], id="global-hl-empty-fallback"),
        # Nothing stands in for the empty list: the bare warning stays.
        pytest.param(
            "hybrid", [], LL, ["high_level_keywords is empty"], id="hybrid-hl-empty"
        ),
        pytest.param("local", HL, LL, [], id="both-present"),
    ],
)
async def test_bare_empty_keyword_warning_only_without_a_fallback(
    mode,
    hl,
    ll,
    expected,
    stub_query_context,  # noqa: F811
    lightrag_log_records,
):
    cache = _FakeKVStorage()
    cfg = _query_global_config(_RecordingModel())

    await _run_kg(_param(mode, hl=hl, ll=ll), cfg, cache)

    bare = [
        record.getMessage()
        for record in lightrag_log_records
        if record.getMessage().endswith("_level_keywords is empty")
    ]
    assert bare == expected
