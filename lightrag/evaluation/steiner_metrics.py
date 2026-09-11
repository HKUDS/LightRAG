"""Request-local rerank accounting and paired, question-level summaries."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from time import perf_counter

from lightrag.utils import normalize_rerank_result


class RerankMeter:
    """Measure inputs to the configured adapter, including all chunk candidates.

    These are logical adapter calls and tokenizer estimates, not GPU seconds,
    provider billing, or HTTP calls hidden inside a chunking/retrying adapter.
    Each query owns its meter; no global counters or shared config mutations.
    """

    def __init__(self, func, tokenizer):
        self.func = func
        self.tokenizer = tokenizer
        self.calls = []
        self.errors = []

    def wrap(self, stage):
        async def measured(*, query, documents, top_n=None, **kwargs):
            doc_tokens = sum(len(self.tokenizer.encode(doc)) for doc in documents)
            entry = {
                "stage": stage,
                "documents": len(documents),
                "document_tokens": doc_tokens,
                "pair_tokens": doc_tokens
                + len(documents) * len(self.tokenizer.encode(query)),
                "top_n": top_n,
            }
            started = perf_counter()
            try:
                result = await self.func(
                    query=query, documents=documents, top_n=top_n, **kwargs
                )
                # LightRAG's chunk path catches provider errors and falls back.
                # Such a run cannot count as an enabled-reranker comparison.
                if not isinstance(result, list):
                    raise TypeError("Expected index/relevance_score rerank results")
                indices = set()
                for row in result:
                    normalized, error = normalize_rerank_result(row, len(documents))
                    if error or normalized["index"] in indices:
                        raise ValueError(
                            f"Invalid or duplicate rerank result: {error or row}"
                        )
                    indices.add(normalized["index"])
                expected = min(len(documents), top_n) if top_n else len(documents)
                if len(indices) < expected:
                    raise ValueError(
                        "Reranker returned fewer scored records than requested"
                    )
                entry["returned"] = len(result)
                return result
            except Exception as exc:
                self.errors.append(str(exc))
                raise
            finally:
                entry["wall_ms"] = (perf_counter() - started) * 1000
                self.calls.append(entry)

        return measured

    def totals(self):
        result = {}
        for stage in ("kg", "chunks", "total"):
            calls = [c for c in self.calls if stage == "total" or c["stage"] == stage]
            result[stage] = {"calls": len(calls)}
            for key in ("documents", "document_tokens", "pair_tokens", "wall_ms"):
                result[stage][key] = sum(c[key] for c in calls)
        return result


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("phase", "quality"),
                row.get("concurrency", 1),
                row["arm"],
                row["max_total_tokens"],
                row["top_k"],
            )
        ].append(row)
    summaries = []
    for (phase, concurrency, arm, budget, top_k), group in groups.items():
        item = {
            "phase": phase,
            "concurrency": concurrency,
            "arm": arm,
            "max_total_tokens": budget,
            "top_k": top_k,
            "n": len(group),
            "n_questions": len({x["case_id"] for x in group}),
        }
        for key in ("selector_ms", "retrieval_ms"):
            values = [x[key] for x in group]
            item[key + "_p50"] = statistics.median(values)
            item[key + "_p95"] = percentile(values, 0.95)
        for key in (
            "evidence_recall",
            "answer_exact_match",
            "answer_token_f1",
            "input_tokens",
            "answer_tokens",
            "candidate_count",
        ):
            values = [x[key] for x in group if x.get(key) is not None]
            item[key + "_mean"] = statistics.mean(values) if values else None
        item["fallback_rate"] = statistics.mean(
            bool(x["selection"].get("fallback_reason")) for x in group
        )
        item["rerank"] = {
            stage: {
                key: statistics.mean(x["rerank"][stage][key] for x in group)
                for key in (
                    "calls",
                    "documents",
                    "document_tokens",
                    "pair_tokens",
                    "wall_ms",
                )
            }
            for stage in ("kg", "chunks", "total")
        }
        summaries.append(item)
    # Net load is measured KG + downstream chunk work, relative to A0 under
    # the same budget/top_k/concurrency. top_n is never used as the input count.
    for item in summaries:
        baseline = next(
            (
                x
                for x in summaries
                if x["arm"] == "A0"
                and all(
                    x[k] == item[k]
                    for k in ("phase", "concurrency", "max_total_tokens", "top_k")
                )
            ),
            None,
        )
        if baseline:
            item["net_rerank_delta_vs_A0"] = {
                key: item["rerank"]["total"][key] - baseline["rerank"]["total"][key]
                for key in ("calls", "documents", "pair_tokens", "wall_ms")
            }
    return summaries


def paired_quality(rows, seed=3866, bootstrap_samples=2000):
    """Bootstrap paired question means, never individual timing repetitions.

    Percentile intervals are descriptive, not multiplicity-adjusted adoption
    tests. All four same-prize contrasts are retained, including negative ones.
    """
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("phase", "quality") != "quality":
            continue
        for metric in ("answer_exact_match", "answer_token_f1"):
            if row.get(metric) is None:
                continue
            for category in ("all", row.get("category", "unspecified")):
                key = (row["max_total_tokens"], row["top_k"], category, metric)
                cells[key][(row["case_id"], row["arm"])].append(row[metric])
    report = []
    rng = random.Random(seed)
    for (budget, top_k, category, metric), cell in cells.items():
        for treatment, baseline in (
            ("A2", "A0"),
            ("A2", "A1"),
            ("B2", "B0"),
            ("B2", "B1"),
        ):
            ids = sorted(q for q, a in cell if a == treatment and (q, baseline) in cell)
            if not ids:
                continue
            deltas = [
                statistics.mean(cell[q, treatment]) - statistics.mean(cell[q, baseline])
                for q in ids
            ]
            bootstrap = [
                statistics.mean(rng.choices(deltas, k=len(deltas)))
                for _ in range(bootstrap_samples)
            ]
            report.append(
                {
                    "max_total_tokens": budget,
                    "top_k": top_k,
                    "category": category,
                    "metric": metric,
                    "treatment": treatment,
                    "baseline": baseline,
                    "n_questions": len(ids),
                    "mean_delta": statistics.mean(deltas),
                    "ci95": [
                        percentile(bootstrap, 0.025),
                        percentile(bootstrap, 0.975),
                    ],
                }
            )
    return report
