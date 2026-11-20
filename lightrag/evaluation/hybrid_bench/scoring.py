"""Per-query scoring and per-domain pooling for the gate inputs.

Rules:
- Chunk metrics score only queries with at least one qrels-positive row;
  zero-positive queries are listed separately, never scored as zero.
- Entity hit@k scores only queries with a non-empty derived truth set.
- Empty retrieval is a real outcome: recall/nDCG/hit of an empty list is 0.0
  against non-empty ground truth, and the query ids are reported per arm.
- Domain pooling concatenates per-query values across the domain's datasets
  in sorted dataset order, keeping query alignment for paired bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

from lightrag.evaluation.hybrid_bench.dataset import BenchDataset
from lightrag.evaluation.hybrid_bench.gates import MetricComparison
from lightrag.evaluation.hybrid_bench.metrics import (
    bootstrap_mean_ci,
    hit_at_k,
    mean,
    ndcg_at_k,
    paired_diff_ci,
    recall_at_k,
)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    mean: float
    ci_lower: float
    ci_upper: float
    n_queries: int

    def as_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci95": [self.ci_lower, self.ci_upper],
            "n_queries": self.n_queries,
        }


def summarize(values: list[float], seed: int, resamples: int) -> MetricSummary:
    lo, hi = bootstrap_mean_ci(values, seed, resamples)
    return MetricSummary(mean(values), lo, hi, len(values))


def score_chunk_arms(
    dataset: BenchDataset,
    arm_docs: dict[str, dict[str, list[str]]],
    top_k: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str], dict[str, list[str]]]:
    """Per-query recall/nDCG per arm, the scored query order, and the query
    ids whose retrieval came back empty per arm."""
    scored_queries = [
        qid for qid in dataset.queries if dataset.positives(qid)
    ]
    zero_positive = [
        qid for qid in dataset.queries if not dataset.positives(qid)
    ]
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    empty: dict[str, list[str]] = {}
    for arm, docs_by_query in arm_docs.items():
        per_query[arm] = {}
        empty[arm] = []
        for qid in scored_queries:
            docs = docs_by_query.get(qid, [])
            if not docs:
                empty[arm].append(qid)
            positives = dataset.positives(qid)
            per_query[arm][qid] = {
                "recall": recall_at_k(docs, positives, top_k),
                "ndcg": ndcg_at_k(docs, positives, top_k),
            }
    return per_query, zero_positive, empty


def score_entity_arms(
    truth_by_query: dict[str, set[str]],
    arm_entities: dict[str, dict[str, list[str]]],
    top_k: int,
) -> dict[str, dict[str, float]]:
    per_query: dict[str, dict[str, float]] = {}
    for arm, entities_by_query in arm_entities.items():
        per_query[arm] = {
            qid: hit_at_k(entities_by_query.get(qid, []), truth, top_k)
            for qid, truth in truth_by_query.items()
        }
    return per_query


def pool_values(
    per_dataset: list[dict[str, dict[str, float]]],
    arm: str,
    metric: str,
) -> list[float]:
    """Concatenate one arm/metric's per-query values in dataset order."""
    values: list[float] = []
    for scored in per_dataset:
        values.extend(scored[arm][qid][metric] for qid in scored[arm])
    return values


def pool_flat(per_dataset: list[dict[str, dict[str, float]]], arm: str) -> list[float]:
    values: list[float] = []
    for scored in per_dataset:
        values.extend(scored[arm].values())
    return values


def comparison(
    dense_values: list[float],
    hybrid_values: list[float],
    seed: int,
    resamples: int,
) -> MetricComparison:
    return MetricComparison(
        dense_mean=mean(dense_values),
        hybrid_mean=mean(hybrid_values),
        diff_ci=paired_diff_ci(hybrid_values, dense_values, seed, resamples),
        reverse_diff_ci=paired_diff_ci(
            dense_values, hybrid_values, seed, resamples
        ),
        abs_diff_ci=paired_diff_ci(
            hybrid_values, dense_values, seed, resamples, absolute=True
        ),
    )