"""Gate verdicts against the proposal thresholds.

Rules:
- Thresholds are proposals frozen in ``GateThresholds``; the harness prints
  verdicts and NEVER adjusts a threshold. Calibration is a separate,
  user-confirmed decision outside this module.
- Verdict statuses: ``pass``, ``fail``, ``residue`` (rerank survival below
  its floor is a documented residue, not a run failure), ``report``
  (backfill throughput is report-only), ``not-run`` (required measurement
  absent).
- Point-estimate comparisons are inclusive at the threshold; CI conditions
  are strict (lower > 0, upper < band).
- Relative gains need a positive baseline; a zero baseline yields
  ``not-run`` rather than a fabricated verdict.

Interpretations fixed here, quoted in the report. Each gate applies to its
own domain only.
- terminology entity hit@k: relative gain >= +15% AND paired CI lower of
  hybrid-minus-dense > 0.
- general entity hit@k: relative gain >= -1% AND paired CI upper of
  dense-minus-hybrid below the noise band, taken as 1% of the dense mean.
- terminology chunk recall@k and nDCG@k: relative gain >= +10% AND paired
  CI lower > 0, per metric.
- general chunk nDCG@k: absolute delta <= 0.005 AND paired CI upper of the
  ABSOLUTE delta < 0.005.
- retrieval latency and answer latency: p95 overhead of hybrid versus dense
  <= the gate's fraction.
- write amplification: upsert p95 increase <= 10% AND GIN index size
  <= 1.5x vector index size on the CJK corpus.
- ablation: the both-hybrid arm >= each single-hybrid arm on the same metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from lightrag.evaluation.hybrid_bench.metrics import relative_gain


@dataclass(frozen=True, slots=True)
class GateThresholds:
    b2a_j_relative_gain: float = 0.15
    b2a_g_relative_floor: float = -0.01
    b2a_g_noise_band_fraction: float = 0.01
    b2b_j_relative_gain: float = 0.10
    b2b_g_absolute_delta: float = 0.005
    b3a_p95_overhead: float = 0.25
    b3b_p95_overhead: float = 0.15
    b3c_upsert_p95_increase: float = 0.10
    b3c_gin_vector_ratio: float = 1.5
    b3d_chunks_per_min: float = 5000.0
    b4_survival: float = 0.50


@dataclass(frozen=True, slots=True)
class GateVerdict:
    gate_id: str
    status: str
    detail: str


def not_run(gate_id: str, reason: str) -> GateVerdict:
    return GateVerdict(gate_id, "not-run", reason)


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def gate_b2a_j(
    domain: str,
    dense_mean: float,
    hybrid_mean: float,
    diff_ci_lower: float,
    thresholds: GateThresholds,
    metric: str,
) -> GateVerdict:
    gate_id = f"entity-hit/{domain}/{metric}"
    gain = relative_gain(hybrid_mean, dense_mean)
    if gain is None:
        return not_run(gate_id, "dense mean is zero; relative gain undefined")
    passed = gain >= thresholds.b2a_j_relative_gain and diff_ci_lower > 0.0
    return GateVerdict(
        gate_id,
        "pass" if passed else "fail",
        f"relative gain {_fmt(gain)} >= "
        f"{thresholds.b2a_j_relative_gain:.2f} and CI lower "
        f"{_fmt(diff_ci_lower)} > 0",
    )


def gate_b2a_g(
    domain: str,
    dense_mean: float,
    hybrid_mean: float,
    dense_minus_hybrid_ci_upper: float,
    thresholds: GateThresholds,
    metric: str,
) -> GateVerdict:
    gate_id = f"entity-hit/{domain}/{metric}"
    gain = relative_gain(hybrid_mean, dense_mean)
    if gain is None:
        return not_run(gate_id, "dense mean is zero; relative gain undefined")
    band = thresholds.b2a_g_noise_band_fraction * dense_mean
    passed = (
        gain >= thresholds.b2a_g_relative_floor and dense_minus_hybrid_ci_upper < band
    )
    return GateVerdict(
        gate_id,
        "pass" if passed else "fail",
        f"relative gain {_fmt(gain)} >= "
        f"{thresholds.b2a_g_relative_floor:.2f} and CI upper "
        f"{_fmt(dense_minus_hybrid_ci_upper)} < noise band {_fmt(band)}",
    )


def gate_b2b_j(
    domain: str,
    metric: str,
    dense_mean: float,
    hybrid_mean: float,
    diff_ci_lower: float,
    thresholds: GateThresholds,
) -> GateVerdict:
    gate_id = f"chunk-quality/{domain}/{metric}"
    gain = relative_gain(hybrid_mean, dense_mean)
    if gain is None:
        return not_run(gate_id, "dense mean is zero; relative gain undefined")
    passed = gain >= thresholds.b2b_j_relative_gain and diff_ci_lower > 0.0
    return GateVerdict(
        gate_id,
        "pass" if passed else "fail",
        f"relative gain {_fmt(gain)} >= "
        f"{thresholds.b2b_j_relative_gain:.2f} and CI lower "
        f"{_fmt(diff_ci_lower)} > 0",
    )


def gate_b2b_g(
    domain: str,
    dense_ndcg: float,
    hybrid_ndcg: float,
    abs_delta_ci_upper: float,
    thresholds: GateThresholds,
    top_k: int = 10,
) -> GateVerdict:
    gate_id = f"chunk-quality/{domain}/ndcg@{top_k}"
    delta = abs(hybrid_ndcg - dense_ndcg)
    limit = thresholds.b2b_g_absolute_delta
    passed = delta <= limit and abs_delta_ci_upper < limit
    return GateVerdict(
        gate_id,
        "pass" if passed else "fail",
        f"|delta| {_fmt(delta)} <= {limit} and CI upper "
        f"{_fmt(abs_delta_ci_upper)} < {limit}",
    )


def gate_p95_overhead(
    gate_id: str,
    overhead: float | None,
    limit: float,
) -> GateVerdict:
    if overhead is None:
        return not_run(gate_id, "dense p95 is zero; overhead undefined")
    passed = overhead <= limit
    return GateVerdict(
        gate_id,
        "pass" if passed else "fail",
        f"p95 overhead {_fmt(overhead)} <= {limit:.2f}",
    )


def gate_b3c(
    upsert_increase: float | None,
    gin_vector_ratio: float | None,
    thresholds: GateThresholds,
) -> list[GateVerdict]:
    verdicts = []
    if upsert_increase is None:
        verdicts.append(not_run("write-amp/upsert-p95", "baseline upsert p95 absent"))
    else:
        limit = thresholds.b3c_upsert_p95_increase
        verdicts.append(
            GateVerdict(
                "write-amp/upsert-p95",
                "pass" if upsert_increase <= limit else "fail",
                f"upsert p95 increase {_fmt(upsert_increase)} <= {limit:.2f}",
            )
        )
    if gin_vector_ratio is None:
        verdicts.append(not_run("write-amp/gin-size", "index sizes absent"))
    else:
        limit = thresholds.b3c_gin_vector_ratio
        verdicts.append(
            GateVerdict(
                "write-amp/gin-size",
                "pass" if gin_vector_ratio <= limit else "fail",
                f"GIN/vector index size ratio {_fmt(gin_vector_ratio)} <= {limit:.2f}",
            )
        )
    return verdicts


def gate_b3d(chunks_per_min: float | None, thresholds: GateThresholds) -> GateVerdict:
    if chunks_per_min is None:
        return not_run("backfill/throughput", "backfill throughput absent")
    return GateVerdict(
        "backfill/throughput",
        "report",
        f"{chunks_per_min:.0f} chunks/min; proposal floor "
        f"{thresholds.b3d_chunks_per_min:.0f}; report-only, no pass/fail",
    )


def gate_b4(survival: float | None, thresholds: GateThresholds) -> GateVerdict:
    if survival is None:
        return not_run("rerank-survival", "reranker survival absent")
    floor = thresholds.b4_survival
    if survival >= floor:
        return GateVerdict(
            "rerank-survival",
            "pass",
            f"survival {_fmt(survival)} >= {floor:.2f}",
        )
    return GateVerdict(
        "rerank-survival",
        "residue",
        f"survival {_fmt(survival)} < {floor:.2f}; documented residue, "
        "does not fail the run",
    )


def gate_b5(
    metric: str,
    both_value: float,
    entity_only_value: float,
    chunk_only_value: float,
) -> GateVerdict:
    gate_id = f"ablation/{metric}"
    passed = both_value >= entity_only_value and both_value >= chunk_only_value
    return GateVerdict(
        gate_id,
        "pass" if passed else "fail",
        f"both {_fmt(both_value)} >= entity-only {_fmt(entity_only_value)} "
        f"and chunk-only {_fmt(chunk_only_value)}",
    )


@dataclass(frozen=True, slots=True)
class MetricComparison:
    dense_mean: float
    hybrid_mean: float
    diff_ci: tuple[float, float]
    reverse_diff_ci: tuple[float, float]
    abs_diff_ci: tuple[float, float]


@dataclass(frozen=True, slots=True)
class GateInputs:
    """Everything the verdict pass needs; None means the measurement did not
    run and the corresponding gates report ``not-run``."""

    entity_hit: dict[str, MetricComparison]
    chunk_recall: dict[str, MetricComparison]
    chunk_ndcg: dict[str, MetricComparison]
    b3a_overhead: float | None = None
    b3b_overhead: float | None = None
    upsert_increase: float | None = None
    gin_vector_ratio: float | None = None
    backfill_chunks_per_min: float | None = None
    rerank_survival: float | None = None
    b5: dict[str, tuple[float, float, float]] | None = None


def evaluate_gates(
    inputs: GateInputs,
    thresholds: GateThresholds | None = None,
    top_k: int = 10,
) -> list[GateVerdict]:
    """Verdicts for every gate; absent measurements become ``not-run``."""
    thresholds = thresholds or GateThresholds()
    verdicts: list[GateVerdict] = []
    entity_metric = f"entity_hit@{top_k}"
    recall_metric = f"recall@{top_k}"
    ndcg_metric = f"ndcg@{top_k}"
    for domain in ("terminology", "general"):
        entity = inputs.entity_hit.get(domain)
        if entity is None:
            verdicts.append(
                not_run(f"entity-hit/{domain}/{entity_metric}", "domain absent")
            )
        elif domain == "terminology":
            verdicts.append(
                gate_b2a_j(
                    domain,
                    entity.dense_mean,
                    entity.hybrid_mean,
                    entity.diff_ci[0],
                    thresholds,
                    entity_metric,
                )
            )
        else:
            verdicts.append(
                gate_b2a_g(
                    domain,
                    entity.dense_mean,
                    entity.hybrid_mean,
                    entity.reverse_diff_ci[1],
                    thresholds,
                    entity_metric,
                )
            )
        if domain == "terminology":
            for metric, comparisons in (
                (recall_metric, inputs.chunk_recall),
                (ndcg_metric, inputs.chunk_ndcg),
            ):
                comp = comparisons.get(domain)
                if comp is None:
                    verdicts.append(
                        not_run(f"chunk-quality/{domain}/{metric}", "domain absent")
                    )
                    continue
                verdicts.append(
                    gate_b2b_j(
                        domain,
                        metric,
                        comp.dense_mean,
                        comp.hybrid_mean,
                        comp.diff_ci[0],
                        thresholds,
                    )
                )
        else:
            comp = inputs.chunk_ndcg.get(domain)
            if comp is None:
                verdicts.append(
                    not_run(f"chunk-quality/{domain}/{ndcg_metric}", "domain absent")
                )
            else:
                verdicts.append(
                    gate_b2b_g(
                        domain,
                        comp.dense_mean,
                        comp.hybrid_mean,
                        comp.abs_diff_ci[1],
                        thresholds,
                        top_k,
                    )
                )
    verdicts.append(
        gate_p95_overhead(
            "retrieval-latency/retrieval-p95",
            inputs.b3a_overhead,
            thresholds.b3a_p95_overhead,
        )
        if inputs.b3a_overhead is not None
        else not_run("retrieval-latency/retrieval-p95", "latency not measured")
    )
    verdicts.append(
        gate_p95_overhead(
            "answer-latency/answer-p95",
            inputs.b3b_overhead,
            thresholds.b3b_p95_overhead,
        )
        if inputs.b3b_overhead is not None
        else not_run("answer-latency/answer-p95", "answer latency not measured")
    )
    verdicts.extend(
        gate_b3c(inputs.upsert_increase, inputs.gin_vector_ratio, thresholds)
    )
    verdicts.append(gate_b3d(inputs.backfill_chunks_per_min, thresholds))
    verdicts.append(gate_b4(inputs.rerank_survival, thresholds))
    if inputs.b5 is None:
        verdicts.append(not_run("ablation/all", "ablation not measured"))
    else:
        for metric, (both, entity_only, chunk_only) in inputs.b5.items():
            verdicts.append(gate_b5(metric, both, entity_only, chunk_only))
    return verdicts
