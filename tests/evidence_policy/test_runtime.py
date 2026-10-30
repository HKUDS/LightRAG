"""The evidence policy end to end: dates, expansion, binding, budget.

These are mechanism tests against a synthetic catalog. They prove which version
the policy selects and which spans it hands the pipeline; they say nothing about
answer quality, which needs a model and a scored corpus.
"""

from __future__ import annotations

import datetime as dt

import pytest

from lightrag.evidence_policy import (
    CatalogError,
    RequestError,
    ScopeKey,
    build_policy_config,
    catalog_from_mapping,
    run_evidence_policy,
)
from lightrag.base import QueryParam
from tests.evidence_policy import corpus

pytestmark = pytest.mark.offline

SCOPE = ScopeKey.from_mapping(corpus.SCOPE)

# V-conflict is a same-rank contradiction and is opt-in: a catalog holding it
# makes the question genuinely undecidable, which is its own test.
DEFAULT_INCLUDE = ["V-old", "V-new", "V-third", "V-comment", "V-undated"]


def catalog(include=None, conflict_rank="official"):
    return catalog_from_mapping(
        corpus.catalog_mapping(
            include=DEFAULT_INCLUDE if include is None else include,
            conflict_rank=conflict_rank,
        )
    )


def param(**overrides):
    defaults = {
        "mode": "naive",
        "evidence_policy": "authority-v1",
        "as_of": "2026-08-01",
    }
    defaults.update(overrides)
    return QueryParam(**defaults)


def config(*, catalog_obj, **overrides):
    built = build_policy_config(
        param(**overrides), catalog=catalog_obj, timezone="Asia/Shanghai"
    )
    assert built is not None
    return built


async def run(
    *,
    as_of="2026-08-01",
    seeds=(),
    scopes=(),
    include=None,
    conflict_rank="official",
    reader=None,
    available_tokens=None,
    mode="naive",
):
    cat = catalog(include=include, conflict_rank=conflict_rank)
    built = build_policy_config(
        param(as_of=as_of, scope_keys=list(scopes), mode=mode),
        catalog=cat,
        timezone="Asia/Shanghai",
        available_tokens=available_tokens,
    )
    assert built is not None
    return await run_evidence_policy(
        built,
        cat,
        seed_chunk_ids=list(seeds),
        reader=reader or corpus.FakeChunkStore(),
        instructions="test instructions",
    )


# ---------------------------------------------------------------------------
# The issue's own scenario: 180 -> 270
# ---------------------------------------------------------------------------


async def test_current_request_uses_the_amendment():
    outcome = await run(as_of="2026-08-01", seeds=["c-old"])

    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-new"]
    assert outcome.resolution.required_evidence() == ["V-new", "V-old"]
    # The old wording travels with the amendment, labelled as replaced.
    assert "270" in outcome.context_text
    assert "180" in outcome.context_text
    assert "superseded_for_current_answer" in outcome.context_text


async def test_historical_request_uses_the_rule_then_in_force():
    outcome = await run(as_of="2026-06-01", seeds=["c-old", "c-new"])

    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-old"]
    assert "180" in outcome.context_text
    assert "270" not in outcome.context_text
    assert "requested_as_of: 2026-06-01" in outcome.context_text


async def test_amendment_published_but_not_effective_does_not_displace():
    outcome = await run(as_of="2026-06-30", seeds=["c-new", "c-old"])

    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-old"]


async def test_multi_hop_chain_reaches_the_last_edition():
    outcome = await run(as_of="2026-10-22", seeds=["c-old"])

    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-third"]
    assert outcome.resolution.required_evidence() == ["V-third", "V-new", "V-old"]


async def test_expired_successor_does_not_restore_the_previous_rule():
    outcome = await run(as_of="2026-11-01", seeds=["c-old"])

    assert outcome.status == "no_applicable_rule"
    assert outcome.resolution.selected_versions() == []


# ---------------------------------------------------------------------------
# Expansion and scope identification
# ---------------------------------------------------------------------------


async def test_explicit_scope_resolves_with_no_seed_at_all():
    outcome = await run(as_of="2026-08-01", seeds=[], scopes=[corpus.SCOPE])

    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-new"]
    # The amendment was never retrieved, so expansion is what reached it.
    assert "c-new" in outcome.diagnostics["expanded_chunk_ids"]


async def test_no_seed_and_no_scope_is_insufficient_evidence():
    outcome = await run(as_of="2026-08-01", seeds=[], scopes=[])

    assert outcome.status == "insufficient_evidence"
    assert outcome.context_text == ""


async def test_seed_from_another_scope_is_not_used():
    outcome = await run(as_of="2026-08-01", seeds=["c-comment"])

    # c-comment belongs to this scope but is commentary: it identifies the
    # scope and never determines the rule.
    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-new"]


# ---------------------------------------------------------------------------
# Failure semantics: distinct statuses, never an empty "no results"
# ---------------------------------------------------------------------------


async def test_commentary_alone_cannot_decide():
    cat = catalog(include=["V-comment"])
    built = build_policy_config(
        param(as_of="2026-08-01", scope_keys=[corpus.SCOPE]),
        catalog=cat,
        timezone="Asia/Shanghai",
    )
    outcome = await run_evidence_policy(
        built, cat, seed_chunk_ids=[], reader=corpus.FakeChunkStore()
    )

    assert outcome.status == "insufficient_evidence"
    assert outcome.resolution.selected_versions() == []


async def test_same_rank_contradiction_is_unresolved_not_date_broken():
    outcome = await run(
        as_of="2026-08-02",
        seeds=["c-conflict", "c-new"],
        include=[*DEFAULT_INCLUDE, "V-conflict"],
    )

    assert outcome.status == "unresolved_conflict"
    assert outcome.resolution.selected_versions() == []
    assert set(outcome.resolution.scopes[0].conflict_versions) == {
        "V-conflict",
        "V-new",
    }


async def test_higher_authority_class_settles_the_same_contradiction():
    outcome = await run(
        as_of="2026-08-02",
        seeds=["c-conflict", "c-new"],
        include=[*DEFAULT_INCLUDE, "V-conflict"],
        conflict_rank="statutory",
    )

    assert outcome.status == "resolved"
    assert outcome.resolution.selected_versions() == ["V-conflict"]


async def test_undated_version_blocks_only_when_the_request_surfaced_it():
    blocked = await run(as_of="2026-08-01", seeds=["c-old", "c-undated"])
    assert blocked.status == "unknown_time"
    assert blocked.resolution.selected_versions() == []

    unblocked = await run(as_of="2026-08-01", seeds=["c-old"])
    assert unblocked.status == "resolved"
    assert unblocked.resolution.selected_versions() == ["V-new"]


async def test_deleted_chunk_is_missing_evidence_not_an_empty_answer():
    rows = [row for row in corpus.chunk_rows() if row["_id"] != "c-old"]
    outcome = await run(
        as_of="2026-08-01", seeds=["c-new"], reader=corpus.FakeChunkStore(rows)
    )

    assert outcome.status == "missing_evidence"
    assert "c-old" in outcome.detail
    assert outcome.context_text == ""


async def test_storage_failure_is_backend_error_not_missing_evidence():
    outcome = await run(
        as_of="2026-08-01", seeds=["c-old"], reader=corpus.FailingChunkStore()
    )

    assert outcome.status == "backend_error"
    assert "unavailable" in outcome.detail


async def test_changed_chunk_content_is_rejected_by_the_span_hash():
    rows = corpus.chunk_rows()
    rows[0]["content"] = rows[0]["content"].replace("180", "999")
    outcome = await run(
        as_of="2026-08-01", seeds=["c-old"], reader=corpus.FakeChunkStore(rows)
    )

    assert outcome.status == "missing_evidence"
    assert "sha256" in outcome.detail


# ---------------------------------------------------------------------------
# Budget: whole groups or nothing
# ---------------------------------------------------------------------------


async def test_budget_that_cannot_hold_the_group_refuses_to_answer():
    outcome = await run(as_of="2026-08-01", seeds=["c-old"], available_tokens=1)

    assert outcome.status == "insufficient_context_budget"
    assert outcome.context_text == ""
    assert outcome.budget is not None and outcome.budget.excluded


async def test_budget_that_fits_keeps_the_whole_group():
    outcome = await run(as_of="2026-08-01", seeds=["c-old"], available_tokens=10_000)

    assert outcome.status == "resolved"
    assert "270" in outcome.context_text


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_policy_off_is_the_default_and_changes_nothing():
    assert build_policy_config(QueryParam(), catalog=catalog()) is None


def test_as_of_without_the_policy_is_refused():
    with pytest.raises(RequestError) as info:
        build_policy_config(QueryParam(as_of="2026-08-01"), catalog=catalog())
    assert info.value.code == "invalid_request"


def test_unknown_policy_version_is_refused():
    with pytest.raises(RequestError) as info:
        build_policy_config(param(evidence_policy="authority-v9"), catalog=catalog())
    assert info.value.code == "invalid_request"


def test_missing_catalog_is_refused_not_silently_ignored():
    with pytest.raises(RequestError) as info:
        build_policy_config(param(), catalog=None)
    assert "evidence_catalog" in str(info.value)


@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
def test_only_the_supported_modes_are_accepted(mode):
    if mode == "mix":
        assert build_policy_config(param(mode=mode), catalog=catalog()) is not None
        return
    with pytest.raises(RequestError) as info:
        build_policy_config(param(mode=mode), catalog=catalog())
    assert info.value.code == "unsupported"


def test_streaming_is_refused_because_answers_are_published_after_checks():
    with pytest.raises(RequestError) as info:
        build_policy_config(param(stream=True), catalog=catalog())
    assert "stream" in str(info.value)


@pytest.mark.parametrize("coarse", ["2026", "2026-10", "26-10-01", "2026-13-01"])
def test_coarse_or_invalid_dates_are_never_padded(coarse):
    with pytest.raises(CatalogError) as info:
        build_policy_config(param(as_of=coarse), catalog=catalog())
    assert info.value.code == "unknown_time"


def test_missing_as_of_is_resolved_once_in_the_configured_timezone():
    # 2026-10-26T23:30Z is already 2026-10-27 in Shanghai.
    built = build_policy_config(
        param(as_of=None),
        catalog=catalog(),
        timezone="Asia/Shanghai",
        now=dt.datetime(2026, 10, 26, 23, 30, tzinfo=dt.timezone.utc),
    )
    assert built.as_of == dt.date(2026, 10, 27)
    assert built.as_of_source == "resolved_once_at_request_start"


def test_unknown_timezone_is_refused():
    with pytest.raises(RequestError):
        build_policy_config(
            param(as_of=None), catalog=catalog(), timezone="Mars/Olympus"
        )


def test_scope_keys_need_all_four_components():
    with pytest.raises(CatalogError):
        build_policy_config(
            param(scope_keys=[{"jurisdiction": "SYNTH-A"}]),
            catalog=catalog(),
        )
