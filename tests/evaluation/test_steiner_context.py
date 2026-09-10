"""Budget, mapping, downstream propagation and transformation oracles."""

import copy
import importlib.util
import itertools
import json
import random
import sys
from argparse import Namespace
from types import SimpleNamespace

import networkx as nx
import pytest
import tiktoken

from lightrag import QueryParam, operate
from lightrag.evaluation import steiner_ablation
from lightrag.evaluation.steiner_ablation import (
    ARMS,
    answer_scores,
    demo_cases,
    demo_snapshot,
    run_context,
)
from lightrag.steiner_context import (
    SelectionOptions,
    _select_sync,
    pcst_proposal,
    record_graph,
    utility,
)
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline
requires_steiner = pytest.mark.skipif(
    importlib.util.find_spec("steinerpy") is None,
    reason="Install the optional steiner dependency to run this test",
)


@pytest.fixture
def config():
    return {
        "tokenizer": Tokenizer("cl100k_base", tiktoken.get_encoding("cl100k_base")),
        "kg_chunk_pick_method": "WEIGHT",
        "related_chunk_number": 5,
        "enable_content_headings": False,
        "min_rerank_score": 0.0,
    }


@pytest.mark.parametrize("seed", range(8))
@requires_steiner
def test_transformation_against_exhaustive_subset_oracle(seed):
    # This checks the mathematics, not hard-coded expected heuristic output.
    rng = random.Random(seed)
    n = 5
    graph = nx.gnp_random_graph(n, 0.3, seed=seed)
    prizes = [rng.uniform(0.0, 2.0) for _ in graph]
    costs = [rng.uniform(0.05, 2.0) for _ in graph]
    beta = [0.0, 0.15, 0.8][seed % 3]

    def objective(nodes):
        return utility(graph, nodes, prizes, beta) - sum(costs[i] for i in nodes)

    optimal = max(
        objective(set(s))
        for k in range(n + 1)
        for s in itertools.combinations(graph, k)
    )
    exact = pcst_proposal(graph, prizes, costs, beta, 1.0, exact=True)
    heuristic = pcst_proposal(graph, prizes, costs, beta, 1.0)
    assert objective(exact) == pytest.approx(optimal, abs=1e-6)
    assert heuristic <= set(graph)
    assert objective(heuristic) <= optimal + 1e-6


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("budgets", [(0, 0), (1, 1), (70, 45), (100, 90)])
async def test_exact_serialized_budgets_and_immutable_input(config, arm, budgets):
    if arm in {"steiner_soft", "steiner_hard"}:
        pytest.importorskip("steinerpy")
    snapshot, rag = demo_snapshot(demo_cases()[1], config)
    snapshot["final_entities"][0]["description"] += (
        ' café 日本語 "quoted" <|endoftext|>'
    )
    original = copy.deepcopy(snapshot)
    result = await run_context(
        rag,
        snapshot,
        "Who founded Lumen?",
        QueryParam(
            max_entity_tokens=budgets[0],
            max_relation_tokens=budgets[1],
            max_total_tokens=1600,
            enable_rerank=False,
        ),
        config,
        arm,
        0.15,
        120,
    )
    assert result["entity_tokens"] <= budgets[0]
    assert result["relation_tokens"] <= budgets[1]
    assert snapshot == original
    if arm == "steiner_soft":
        assert (
            result["selection"]["utility"]
            >= result["selection"]["baseline_utility"] - 1e-9
        )


@pytest.mark.asyncio
@requires_steiner
async def test_isolate_survives_and_its_source_reaches_final_prompt(config):
    case = demo_cases()[0]
    snapshot, rag = demo_snapshot(case, config)
    param = QueryParam(
        max_entity_tokens=100,
        max_relation_tokens=90,
        max_total_tokens=1600,
        enable_rerank=False,
    )
    soft = await run_context(
        rag, snapshot, case["question"], param, config, "steiner_soft", 0.15, 120
    )
    hard = await run_context(
        rag, snapshot, case["question"], param, config, "steiner_hard", 0.15, 120
    )
    assert "Access" in soft["selected_entities"]
    assert "remote access code is amber" in soft["system_prompt"]
    assert soft["chunk_count"] > hard["chunk_count"]
    assert "remote access code is amber" not in hard["system_prompt"]


@pytest.mark.asyncio
@requires_steiner
async def test_pipeline_uses_filtered_originals_and_reports_selection(
    config, monkeypatch
):
    case = demo_cases()[0]
    snapshot, rag = demo_snapshot(case, config)
    config["addon_params"] = {"context_selection": {"strategy": "steiner_soft"}}

    async def search(*args, **kwargs):
        return copy.deepcopy(snapshot)

    monkeypatch.setattr(operate, "_perform_kg_search", search)
    result = await operate._build_query_context(
        case["question"],
        "Access",
        "Atlas",
        None,
        None,
        None,
        rag.text_chunks,
        QueryParam(
            max_entity_tokens=100,
            max_relation_tokens=90,
            max_total_tokens=1600,
            enable_rerank=False,
        ),
    )
    assert "remote access code is amber" in result.context
    assert result.raw_data["metadata"]["context_selection"]["solver_calls"] == 3


def test_missing_endpoints_and_antiparallel_rows_are_distinct():
    graph = record_graph(
        [{"entity": "A"}],
        [
            {"entity1": "A", "entity2": "B"},
            {"entity1": "B", "entity2": "A"},
            {"entity1": "B", "entity2": "C"},
        ],
    )
    assert set(graph) == {0, 1, 2, 3}
    assert set(graph.neighbors(0)) == {1, 2}
    assert graph.degree(3) == 0
    # No bonus through unselected A; relation directions are never merged here.
    assert utility(graph, {1, 2}, [1, 1, 1, 1], 0.5) == 2


def test_candidate_limit_uses_reported_rank_fallback(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SteinerPy must not run above the candidate limit")

    monkeypatch.setattr("lightrag.steiner_context.pcst_proposal", forbidden)
    e, r, metrics = _select_sync(
        [{"entity": "A"}, {"entity": "B"}],
        [],
        config["tokenizer"],
        100,
        0,
        SelectionOptions(max_candidates=1),
    )
    assert len(e) == 2 and not r
    assert metrics["fallback_reason"] == "candidate_limit"
    assert metrics["executed_strategy"] == "rank"


@pytest.mark.asyncio
async def test_default_path_never_calls_optional_selector(config, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Default retrieval must not call the selector")

    monkeypatch.setattr("lightrag.steiner_context.select_context", forbidden)
    snapshot, _ = demo_snapshot(demo_cases()[0], config)
    result = await operate._apply_token_truncation(snapshot, QueryParam(), config)
    assert "selection_metadata" not in result
    assert len(result["filtered_entities"]) == 3


@requires_steiner
def test_zero_bonus_keeps_disconnected_prizes():
    graph = nx.empty_graph(3)
    result = pcst_proposal(graph, [1, 2, 3], [0.1, 0.1, 0.1], 0, 1)
    assert result == {0, 1, 2}


def test_answer_scores():
    assert answer_scores("The Northstar", ["Northstar"])["answer_exact_match"] == 1
    assert answer_scores("Delft", ["Delft Bergen"])["answer_token_f1"] == pytest.approx(
        2 / 3
    )


@pytest.mark.asyncio
@requires_steiner
async def test_live_driver_writes_paired_answers_and_finalizes(
    config, monkeypatch, tmp_path
):
    # Contract test of the live route; this scripted answerer is NOT evidence
    # of answer quality, and its results are never included in the report.
    case = demo_cases()[0]
    case.update({"ll_keywords": "Access", "hl_keywords": "Atlas"})
    snapshot, rag = demo_snapshot(case, config)
    calls = []

    async def answer(question, **kwargs):
        calls.append(kwargs)
        return "Mira; amber"

    config["role_llm_funcs"] = {"query": answer}
    rag._build_global_config = lambda: config
    rag.entities_vdb = rag.relationships_vdb = None
    finished = []

    async def finalize():
        finished.append(True)

    rag.finalize_storages = finalize

    async def search(*args, **kwargs):
        return copy.deepcopy(snapshot)

    monkeypatch.setattr(steiner_ablation, "_perform_kg_search", search)
    monkeypatch.setitem(
        sys.modules, "steiner_test_factory", SimpleNamespace(make_rag=lambda: rag)
    )
    dataset = tmp_path / "qa.json"
    dataset.write_text(json.dumps([case]))
    args = Namespace(
        seed=3866,
        encoding="cl100k_base",
        factory="steiner_test_factory:make_rag",
        dataset=str(dataset),
        output=tmp_path / "results",
        top_k=[10],
        budgets=[1600],
        mode="hybrid",
        chunk_top_k=10,
        entity_budget=100,
        relation_budget=90,
        rerank=False,
        repeats=1,
        beta=0.15,
        max_candidates=120,
    )
    await steiner_ablation.main(args)
    rows = [
        json.loads(line)
        for line in (args.output / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(calls) == 5
    assert finished == [True]
    assert all(row["answer_token_f1"] == 1 for row in rows)
    assert {row["max_total_tokens"] for row in rows} == {1600}
    assert all(row["input_tokens"] <= 1600 for row in rows)


@pytest.mark.parametrize(
    "options",
    [
        {"connectivity_bonus": -1},
        {"connectivity_bonus": float("nan")},
        {"max_candidates": 0},
        {"multipliers": ()},
        {"strategy": "typo"},
    ],
)
def test_invalid_options_fail_loudly(options):
    with pytest.raises(ValueError):
        SelectionOptions(**options)
