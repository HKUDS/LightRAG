"""The latency gates and CI use the SAME flat all-samples p95 the report
displays, never the per-query round means. Hand-computed fixtures where the
two definitions give opposite verdicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lightrag.evaluation.hybrid_bench import cli, latency as latency_mod
from lightrag.evaluation.hybrid_bench.gates import GateThresholds, evaluate_gates
from lightrag.evaluation.hybrid_bench.latency import LatencyResult
from lightrag.evaluation.hybrid_bench.rag_setup import BenchConfig

pytestmark = pytest.mark.offline


def _args() -> object:
    return cli.parse_args(["--dataset-dir", "d", "--output-dir", "o"])


def _dataset_result(latency: dict) -> dict:
    return {
        "meta": {"domain": "general"},
        "_scored": {"dense": {}, "hybrid": {}},
        "_entity_scored": {},
        "latency": latency,
    }


def _phase(
    dense_per_query: list[float],
    dense_flat: list[float],
    hybrid_per_query: list[float],
    hybrid_flat: list[float],
) -> dict:
    return {
        "dense": {"p50": 0.0, "p95": 0.0, "rounds": 3, "query_slots": 20},
        "hybrid": {"p50": 0.0, "p95": 0.0, "rounds": 3, "query_slots": 20},
        "per_query_dense": dense_per_query,
        "per_query_hybrid": hybrid_per_query,
        "all_samples_dense": dense_flat,
        "all_samples_hybrid": hybrid_flat,
    }


def test_gate_uses_flat_p95_where_round_means_would_fail():
    # Given 20 slots x 3 rounds: dense is 1.0 everywhere; hybrid is 1.0
    # except ONE 100.0 outlier round in each of the last two slots.
    # Flat p95 (60 samples, nearest rank ceil(0.95*60)=57th smallest):
    #   both arms 1.0 -> overhead 0.0 -> PASS (<= 0.25).
    # Per-query round means: the two outlier slots average (1+1+100)/3=34;
    #   p95 of 20 means = 19th smallest = 34.0 -> overhead 33.0 -> FAIL.
    dense_flat = [1.0] * 60
    hybrid_flat = [1.0] * 60
    hybrid_flat[18 * 3 + 2] = 100.0
    hybrid_flat[19 * 3 + 2] = 100.0
    hybrid_means = [1.0] * 18 + [34.0, 34.0]
    phase = _phase([1.0] * 20, dense_flat, hybrid_means, hybrid_flat)
    latency = {"retrieval": phase, "answer": phase}
    inputs = cli._gate_inputs([_dataset_result(latency)], _args())
    assert inputs.retrieval_p95_overhead == 0.0
    verdicts = {
        v.gate_id: v.status for v in evaluate_gates(inputs, GateThresholds(), top_k=10)
    }
    assert verdicts["retrieval-latency/retrieval-p95"] == "pass"


def test_gate_pools_flat_samples_across_datasets():
    # ds1: dense 1.0 x4, hybrid 2.0 x4; ds2: dense 3.0 x4, hybrid 3.0 x4.
    # Pooled nearest-rank p95 (8 samples each, ceil(0.95*8)=8th smallest):
    #   dense 3.0, hybrid 3.0 -> overhead 0.0. Per-dataset overheads would
    #   be 1.0 and 0.0; per-query means below are decoys the gate must ignore.
    phase1 = _phase([100.0], [1.0] * 4, [100.0], [2.0] * 4)
    phase2 = _phase([100.0], [3.0] * 4, [100.0], [3.0] * 4)
    ds1 = _dataset_result({"retrieval": phase1, "answer": phase1})
    ds2 = _dataset_result({"retrieval": phase2, "answer": phase2})
    inputs = cli._gate_inputs([ds1, ds2], _args())
    assert inputs.retrieval_p95_overhead == 0.0


class _DummyClient:
    """Async context manager stand-in for the shared latency HTTP client."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_run_latency_stores_flat_samples_and_cis_them(monkeypatch):
    # Flat samples are constant (dense 2.0, hybrid 2.5) -> overhead CI
    # collapses to exactly [0.25, 0.25]. The per-query means are NOT
    # constant-ratio: paired p95-of-two = max gives resample overheads
    # 1.0 (both slot-0) and 0.0 (any slot-1) -> a per-query CI would be
    # [0.0, 1.0], so this fixture distinguishes the two definitions.
    dense = LatencyResult(
        per_query=[1.0, 3.0],
        all_samples=[2.0] * 4,
        p50=2.0,
        p95=2.0,
        rounds=2,
        query_slots=2,
    )
    hybrid = LatencyResult(
        per_query=[2.0, 3.0],
        all_samples=[2.5] * 4,
        p50=2.5,
        p95=2.5,
        rounds=2,
        query_slots=2,
    )
    measured: list[bool] = []

    async def fake_measure(client, slots, rounds, enable_hybrid, *rest):
        measured.append(enable_hybrid)
        return hybrid if enable_hybrid else dense

    monkeypatch.setattr(latency_mod, "build_latency_client", lambda rag: _DummyClient())
    monkeypatch.setattr(latency_mod, "measure_latency", fake_measure)
    dataset = SimpleNamespace(queries={"q1": "query one", "q2": "query two"})
    config = BenchConfig(latency_queries=2, latency_rounds=2, seed=5, resamples=1000)
    results = await cli._run_latency(None, dataset, config)
    assert measured == [False, True, False, True]
    for phase in ("retrieval", "answer"):
        entry = results[phase]
        assert entry["all_samples_dense"] == [2.0] * 4
        assert entry["all_samples_hybrid"] == [2.5] * 4
        assert entry["overhead"] == pytest.approx(0.25)
        assert entry["overhead_ci95"] == [pytest.approx(0.25), pytest.approx(0.25)]
        assert entry["per_query_dense"] == [1.0, 3.0]
