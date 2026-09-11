"""Review regressions: score identity, strict reranking, accounting and pairing."""

import asyncio
import copy
import importlib.util
import itertools
import json
import subprocess
import sys

import networkx as nx
import pytest
import tiktoken

from lightrag import QueryParam, operate
from lightrag.constants import DEFAULT_MIN_RERANK_SCORE
from lightrag.evaluation.steiner_ablation import (
    FACTORIAL_ARMS,
    argument_parser,
    demo_cases,
    demo_rerank,
    demo_snapshot,
    run_context,
    validate_live,
)
from lightrag.evaluation.steiner_metrics import RerankMeter, paired_quality, summarize
from lightrag.steiner_context import (
    SelectionOptions,
    _select_sync,
    pcst_proposal,
    select_context,
    utility,
)
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline
requires_steiner = pytest.mark.skipif(
    importlib.util.find_spec("steinerpy") is None,
    reason="Optional SteinerPy dependency",
)


@pytest.fixture
def config():
    return {
        "tokenizer": Tokenizer("cl100k_base", tiktoken.get_encoding("cl100k_base")),
        "rerank_model_func": demo_rerank,
        "min_rerank_score": DEFAULT_MIN_RERANK_SCORE,
        "kg_chunk_pick_method": "WEIGHT",
        "related_chunk_number": 5,
        "enable_content_headings": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", FACTORIAL_ARMS)
async def test_six_arms_use_same_reranker_and_keep_source_attribution(config, arm):
    if arm.endswith("2"):
        pytest.importorskip("steinerpy")
    case = demo_cases()[0]
    snapshot, rag = demo_snapshot(case, config)
    original = copy.deepcopy(snapshot)
    result = await run_context(
        rag,
        snapshot,
        case["question"],
        QueryParam(
            max_entity_tokens=100,
            max_relation_tokens=90,
            max_total_tokens=1600,
            enable_rerank=True,
            chunk_top_k=1,
        ),
        config,
        arm,
        0.15,
        120,
    )
    assert snapshot == original
    assert result["rerank"]["kg"]["calls"] == int(arm.startswith("B"))
    assert result["rerank"]["chunks"]["calls"] == 1
    # top_n=1 caps returned chunks, not the documents that were scored.
    assert result["rerank"]["chunks"]["documents"] > result["chunk_count"]
    assert "rerank_score" not in result["context"]
    if arm.startswith("B"):
        assert result["selection"]["score_source"] == "rerank_score"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["rank", "relevance", "steiner_soft"])
async def test_numeric_scores_reach_every_b_arm_and_preserve_isolate(
    config, monkeypatch, strategy
):
    entities = [
        {"entity": "Wrong", "description": "irrelevant"},
        {"entity": "Isolate", "description": "answer"},
    ]
    seen = []

    async def scores(**kwargs):
        assert kwargs["top_n"] == 2
        return [
            {"index": 0, "relevance_score": 0.01},
            {"index": 1, "relevance_score": 0.93},
        ]

    def proposal(graph, prizes, *args, **kwargs):
        seen.append(prizes)
        return {0}

    monkeypatch.setattr("lightrag.steiner_context.pcst_proposal", proposal)
    cap = len(config["tokenizer"].encode(json.dumps(entities[1], ensure_ascii=False)))
    selected, _, metadata = await select_context(
        entities,
        [],
        config["tokenizer"],
        cap,
        0,
        {"strategy": strategy, "prize_source": "rerank_score"},
        query="question",
        rerank_func=scores,
    )
    assert selected == [entities[1]]
    if strategy == "steiner_soft":
        assert seen and all(x == [0.93, 0.01] for x in seen)
        assert metadata["effective_connectivity_bonus"] == pytest.approx(0.15 * 0.93)


@pytest.mark.asyncio
async def test_b_fallback_still_uses_scored_prefix(config):
    async def scores(**kwargs):
        return [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]

    entities = [{"entity": "A"}, {"entity": "B"}]
    cap = len(config["tokenizer"].encode(json.dumps(entities[0])))
    selected, _, metadata = await select_context(
        entities,
        [],
        config["tokenizer"],
        cap,
        0,
        {
            "strategy": "steiner_soft",
            "prize_source": "rerank_score",
            "max_candidates": 1,
        },
        query="q",
        rerank_func=scores,
    )
    assert selected == [{"entity": "B"}]
    assert metadata["fallback_reason"] == "candidate_limit"
    assert metadata["score_source"] == "rerank_score"
    assert metadata["solver_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 0, "relevance_score": 0.2}],
        [{"index": 0, "relevance_score": 0.2}] * 2,
        [
            {"index": 0, "relevance_score": float("nan")},
            {"index": 1, "relevance_score": 0.2},
        ],
        [{"index": 0, "relevance_score": -0.1}, {"index": 1, "relevance_score": 0.2}],
    ],
)
async def test_missing_duplicate_nonfinite_negative_scores_fail(config, results):
    async def scores(**kwargs):
        return results

    with pytest.raises(ValueError):
        await select_context(
            [{"entity": "A"}, {"entity": "B"}],
            [],
            config["tokenizer"],
            100,
            0,
            {"strategy": "rank", "prize_source": "rerank_score"},
            query="q",
            rerank_func=scores,
        )


@pytest.mark.asyncio
async def test_positive_chunk_threshold_filters_low_score_bridge_source(config):
    case = demo_cases()[0]
    snapshot, rag = demo_snapshot(case, config)

    async def scores(*, documents, **kwargs):
        return [
            {"index": i, "relevance_score": 0.1 if "leads" in doc else 0.9}
            for i, doc in enumerate(documents)
        ]

    config["rerank_model_func"] = scores
    param = QueryParam(
        max_entity_tokens=100,
        max_relation_tokens=90,
        max_total_tokens=1600,
        enable_rerank=True,
    )
    unfiltered = await run_context(
        rag, snapshot, case["question"], param, config, "A0", 0.15, 120
    )
    filtered = await run_context(
        rag,
        snapshot,
        case["question"],
        param,
        {**config, "min_rerank_score": 0.5},
        "A0",
        0.15,
        120,
    )
    assert filtered["chunk_count"] == unfiltered["chunk_count"] - 1
    assert (
        filtered["rerank"]["chunks"]["documents"]
        == unfiltered["rerank"]["chunks"]["documents"]
    )


@pytest.mark.asyncio
async def test_chunk_rerank_failure_cannot_be_reported_as_enabled(config):
    async def broken(**kwargs):
        raise RuntimeError("provider unavailable")

    config["rerank_model_func"] = broken
    snapshot, rag = demo_snapshot(demo_cases()[0], config)
    with pytest.raises(ValueError, match="Invalid reranking comparison"):
        await run_context(
            rag,
            snapshot,
            "question",
            QueryParam(max_total_tokens=1600),
            config,
            "A0",
            0.15,
            120,
        )


@pytest.mark.asyncio
async def test_request_local_meters_do_not_mix_concurrent_queries(config):
    async def scores(*, documents, **kwargs):
        await asyncio.sleep(0)
        return [{"index": i, "relevance_score": 0.8} for i in range(len(documents))]

    meters = [RerankMeter(scores, config["tokenizer"]) for _ in range(4)]
    await asyncio.gather(
        *(
            m.wrap("chunks")(query="q", documents=["x"] * (i + 1), top_n=1)
            for i, m in enumerate(meters)
        )
    )
    assert [m.totals()["total"]["documents"] for m in meters] == [1, 2, 3, 4]
    assert all(m.totals()["total"]["calls"] == 1 for m in meters)


def test_budget_check_survives_python_optimization():
    code = "from lightrag.steiner_context import check_budget; check_budget((2, 0), (1, 0))"
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False
    )
    assert (
        result.returncode != 0
        and "ValueError: Context selection exceeded" in result.stderr
    )


def test_rank_return_path_enforces_budget(config, monkeypatch):
    monkeypatch.setattr(
        "lightrag.steiner_context.truncate_list_by_token_size",
        lambda rows, **kwargs: rows,
    )
    with pytest.raises(ValueError, match="exceeded token budgets"):
        _select_sync(
            [{"entity": "A"}],
            [],
            config["tokenizer"],
            1,
            0,
            SelectionOptions(strategy="rank"),
        )


@requires_steiner
def test_heuristic_certificate_bounds_exhaustive_lagrangian_optimum():
    graph = nx.path_graph(4)
    prizes, costs, beta = [0.9, 0.1, 0.8, 0.6], [0.2, 0.5, 0.4, 0.7], 0.15
    records = []
    selected = pcst_proposal(graph, prizes, costs, beta, 1, diagnostics=records)
    optimum = max(
        utility(graph, set(s), prizes, beta) - sum(costs[i] for i in s)
        for k in range(5)
        for s in itertools.combinations(graph, k)
    )
    objective = utility(graph, selected, prizes, beta) - sum(costs[i] for i in selected)
    assert records[0]["lagrangian_upper_bound"] >= optimum - 1e-6
    assert objective <= optimum + 1e-6
    if records[0]["pcst_relative_gap"] == 0:
        assert objective == pytest.approx(optimum)


def test_paired_uncertainty_uses_question_means_and_ignores_load():
    rows = []
    for q, effect in (("q1", 1.0), ("q2", -1.0)):
        for _ in range(10 if q == "q1" else 1):
            for arm, score in (("B0", float(effect < 0)), ("B2", float(effect > 0))):
                rows.append(
                    {
                        "phase": "quality",
                        "case_id": q,
                        "arm": arm,
                        "category": "isolated evidence",
                        "answer_exact_match": score,
                        "max_total_tokens": 1600,
                        "top_k": 40,
                    }
                )
    rows += [{**rows[0], "phase": "load", "answer_exact_match": 999}] * 20
    report = paired_quality(rows, bootstrap_samples=100)
    aggregate = next(x for x in report if x["category"] == "all")
    assert aggregate["n_questions"] == 2
    assert aggregate["mean_delta"] == 0
    assert aggregate["ci95"][0] < 0 < aggregate["ci95"][1]


def test_net_rerank_load_counts_downstream_savings():
    def row(arm, kg, chunks):
        totals = {
            s: {
                "calls": int(n > 0),
                "documents": n,
                "document_tokens": 10 * n,
                "pair_tokens": 12 * n,
                "wall_ms": n,
            }
            for s, n in (("kg", kg), ("chunks", chunks))
        }
        totals["total"] = {
            key: totals["kg"][key] + totals["chunks"][key] for key in totals["kg"]
        }
        return {
            "arm": arm,
            "case_id": "q",
            "max_total_tokens": 1600,
            "top_k": 40,
            "selector_ms": 1,
            "retrieval_ms": 2,
            "selection": {},
            "rerank": totals,
        }

    report = summarize([row("A0", 0, 100), row("B0", 60, 20)])
    assert report[1]["net_rerank_delta_vs_A0"]["documents"] == -20
    assert report[1]["net_rerank_delta_vs_A0"]["pair_tokens"] == -240


def test_acceptance_preflight_requires_real_models_dataset_and_defaults(config):
    args = argument_parser().parse_args([])
    args.index_id, args.answer_model_id, args.rerank_model_id = (
        "fixed",
        "answer-v1",
        "rerank-v1",
    )
    config["role_llm_funcs"] = {"query": demo_rerank}
    with pytest.raises(ValueError, match="30–60"):
        validate_live(demo_cases(), config, args)
    with pytest.raises(ValueError, match="project-default"):
        validate_live(
            [], {**config, "min_rerank_score": DEFAULT_MIN_RERANK_SCORE + 0.1}, args
        )


@pytest.mark.asyncio
async def test_production_stage_receives_query_for_b0(config, monkeypatch):
    snapshot, rag = demo_snapshot(demo_cases()[0], config)
    config["addon_params"] = {
        "context_selection": {"strategy": "rank", "prize_source": "rerank_score"}
    }

    async def search(*args, **kwargs):
        return copy.deepcopy(snapshot)

    monkeypatch.setattr(operate, "_perform_kg_search", search)
    result = await operate._build_query_context(
        "Who leads Atlas?",
        "Atlas",
        "leadership",
        None,
        None,
        None,
        rag.text_chunks,
        QueryParam(
            max_total_tokens=1600, max_entity_tokens=100, max_relation_tokens=90
        ),
    )
    assert (
        result.raw_data["metadata"]["context_selection"]["score_source"]
        == "rerank_score"
    )
