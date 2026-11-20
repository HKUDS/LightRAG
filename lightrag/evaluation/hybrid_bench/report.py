"""Markdown + JSON report rendering for the hybrid benchmark.

Rules:
- The JSON carries every number, including per-query values and CIs; the
  markdown is the human view of the same data, never a separate computation.
- A synthetic run is labeled SYNTHETIC in both outputs: fake embeddings and
  the fake LLM prove mechanics, not retrieval quality.
- Gate verdicts are printed as-is; the report never adjusts a threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

ENTITY_TRUTH_DERIVATION = (
    "An entity is relevant to a query iff the full_doc_id set of its source "
    "chunks intersects the query's qrels-positive corpus ids. Source chunks "
    "come from the graph node source_id split on GRAPH_FIELD_SEP; chunk ids "
    "resolve to documents through the text_chunks KV store. Queries with "
    "empty derived truth are reported separately and excluded from hit@k."
)

LEXICAL_ARM_NOTE = (
    "The lexical-only arm executes the same lexical_chunks SQL template "
    "hybrid_query uses, read from postgres_impl.SQL_TEMPLATES, with the same "
    "tokenizer and workspace. Diagnostic only; no product change."
)

LATENCY_SURFACE_NOTE = (
    "Latency runs through the production query router mounted on a minimal "
    "FastAPI app via httpx ASGITransport: QueryRequest validation and "
    "to_query_params are the live /query/data and /query code path, minus "
    "the network hop. Keywords are injected to exclude the keyword LLM."
)


def _fmt_ci(entry: dict) -> str:
    lo, hi = entry["ci95"]
    return f"{entry['mean']:.4f} [{lo:.4f}, {hi:.4f}]"


def _metric_table(dataset: dict) -> list[str]:
    lines = [
        "| arm | recall@10 [95% CI] | nDCG@10 [95% CI] |",
        "| --- | --- | --- |",
    ]
    for arm, metrics in dataset["chunk_metrics"].items():
        lines.append(
            f"| {arm} | {_fmt_ci(metrics['recall@10'])} "
            f"| {_fmt_ci(metrics['ndcg@10'])} |"
        )
    entity = dataset.get("entity_metrics") or {}
    for arm, metrics in entity.items():
        lines.append(
            f"| entity-{arm} | hit@10 {_fmt_ci(metrics['hit@10'])} | - |"
        )
    return lines


def _latency_lines(latency: dict | None) -> list[str]:
    if latency is None:
        return ["Latency: not measured."]
    lines = [
        "| phase | arm | p50 s | p95 s | slots x rounds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for phase, arms in latency.items():
        for arm in ("dense", "hybrid"):
            entry = arms[arm]
            lines.append(
                f"| {phase} | {arm} | {entry['p50']:.4f} "
                f"| {entry['p95']:.4f} "
                f"| {entry['query_slots']} x {entry['rounds']} |"
            )
        overhead = arms.get("overhead")
        ci = arms.get("overhead_ci95")
        if overhead is not None and ci is not None:
            lines.append(
                f"| {phase} | overhead | - "
                f"| {overhead:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | - |"
            )
    return lines


def render_markdown(results: dict) -> str:
    lines: list[str] = ["# Hybrid retrieval benchmark report", ""]
    if results["synthetic"]:
        lines += [
            "**SYNTHETIC RUN: fake hash embeddings and/or fake LLM. "
            "Mechanics proof only, NOT retrieval-quality evidence.**",
            "",
        ]
    lines += [
        f"Seed: {results['config']['seed']} | "
        f"bootstrap resamples: {results['config']['resamples']} | "
        f"top_k: {results['config']['top_k']}",
        "",
        "## Protocols",
        "",
        f"- Entity truth: {ENTITY_TRUTH_DERIVATION}",
        f"- Lexical arm: {LEXICAL_ARM_NOTE}",
        f"- Latency surface: {LATENCY_SURFACE_NOTE}",
        "- Extraction subset: all qrels-positive passages plus sorted fill "
        "up to the CLI cap; the remainder ingests chunks-only without LLM "
        "extraction.",
        "",
        "## Gate verdicts (proposal thresholds, never tuned)",
        "",
        "| gate | status | detail |",
        "| --- | --- | --- |",
    ]
    for verdict in results["gates"]:
        lines.append(
            f"| {verdict['gate_id']} | {verdict['status']} "
            f"| {verdict['detail']} |"
        )
    for dataset in results["datasets"]:
        meta = dataset["meta"]
        lines += [
            "",
            f"## Dataset {dataset['name']} "
            f"(domain={meta['domain']}, language={meta['language']})",
            "",
            f"Source: {meta['source']} | license: {meta['license']}",
            f"Subset protocol: {meta['subset_protocol']}",
            "",
            f"Ingest: {dataset['ingest']['chunk_count']} chunks, "
            f"{len(dataset['ingest']['extracted_doc_ids'])} extracted docs, "
            f"{len(dataset['ingest']['chunks_only_doc_ids'])} chunks-only "
            f"docs (cap {dataset['ingest']['extraction_cap']}).",
            f"Recovery: {dataset['ingest']['recovery_rounds_used']} round(s) "
            f"via the sanctioned manual retry path; recovered "
            f"{dataset['ingest']['recovered_doc_ids']}; persistently failed "
            f"{dataset['ingest']['persistently_failed_doc_ids']}.",
            "",
            "### Chunk and entity metrics",
            "",
            *_metric_table(dataset),
            "",
            f"Zero-positive queries (excluded from chunk metrics): "
            f"{dataset['zero_positive_query_ids']}",
            f"Zero-truth queries (excluded from hit@k): "
            f"{dataset['entity_truth']['zero_truth_query_ids']}",
            f"Empty retrieval query ids per arm: "
            f"{dataset['empty_retrieval_query_ids']}",
            "",
            "### Latency",
            "",
            *_latency_lines(dataset.get("latency")),
        ]
        if dataset.get("b5"):
            lines += [
                "",
                "### Ablation means",
                "",
                "| combination | recall@10 | nDCG@10 | entity hit@10 |",
                "| --- | --- | --- | --- |",
            ]
            for combo, metrics in dataset["b5"].items():
                lines.append(
                    f"| {combo} | {metrics['recall@10']:.4f} "
                    f"| {metrics['ndcg@10']:.4f} "
                    f"| {metrics['entity_hit@10']:.4f} |"
                )
        write_amp = dataset.get("write_amp")
        if write_amp:
            lines += [
                "",
                "### Write amplification",
                "",
                f"Branch upsert p95: {write_amp['branch_p95']:.4f}s; "
                f"baseline ({write_amp['baseline_ref']}): "
                f"{write_amp['baseline_p95']}; "
                f"increase: {write_amp['upsert_increase']} "
                "(driver uses synthetic hash embeddings on both sides: a "
                "write-path measurement, not a quality claim)",
            ]
        sizes = dataset.get("index_sizes")
        if sizes:
            lines.append(
                f"GIN index bytes: {sizes['gin_bytes']}; vector index bytes: "
                f"{sizes['vector_bytes']}; ratio: {sizes['ratio']}"
            )
        if dataset.get("backfill_chunks_per_min") is not None:
            lines.append(
                f"Backfill throughput: {dataset['backfill_chunks_per_min']:.0f} "
                "chunks/min (report-only)"
            )
        rerank = dataset.get("rerank")
        if rerank:
            lines.append(
                f"Reranker: {rerank['status']}; survival "
                f"{rerank['survival']}; {rerank['detail']}"
            )
        for phase, reason in (dataset.get("skipped") or {}).items():
            lines.append(f"Skipped {phase}: {reason}")
    lines += [
        "",
        "## Appendix: hardware and configuration",
        "",
        "```json",
        json.dumps(
            {"hardware": results["hardware"], "config": results["config"]},
            indent=2,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(results: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(render_markdown(results), encoding="utf-8")
    return md_path, json_path