"""CLI orchestration: ingest, arms, metrics, optional phases, gates, report.

Run: ``python -m lightrag.evaluation.hybrid_bench --dataset-dir DIR
--output-dir DIR [knobs]``. Every knob is a CLI argument; PG connection uses
the standard POSTGRES_* environment recipe. Gate thresholds are proposals
printed as verdicts, never tuned here.

Phase-gating decisions, documented for a full run:
- Write-amplification and backfill gate inputs come from the first dataset
  in sorted order that measured them; every dataset's numbers are in the
  report.
- Reranker survival pools scored hits across datasets.
- Latency gate inputs pool the flat all-samples latencies across datasets,
  the same p95 definition the report displays.
- The baseline-worktree half of the write-amplification comparison runs only
  with --write-amp-baseline.
- --workspace mints a stable per-dataset workspace (no random suffix) and
  implies --keep-workspace, so re-invoking the same command resumes ingest
  through the product doc_status instead of starting over.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from lightrag.evaluation.hybrid_bench import arms as arms_mod
from lightrag.evaluation.hybrid_bench import latency as latency_mod
from lightrag.evaluation.hybrid_bench import scoring
from lightrag.evaluation.hybrid_bench.dataset import (
    BenchDataset,
    discover_datasets,
    load_dataset,
)
from lightrag.evaluation.hybrid_bench.entity_truth import derive_entity_truth
from lightrag.evaluation.hybrid_bench.fakes import FakeLLM
from lightrag.evaluation.hybrid_bench.gates import (
    GateInputs,
    GateThresholds,
    evaluate_gates,
)
from lightrag.evaluation.hybrid_bench.metrics import (
    bootstrap_p95_overhead_ci,
    hit_at_k,
    mean,
    ndcg_at_k,
    p95_overhead,
    percentile,
    recall_at_k,
)
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_llm_func,
    build_rag,
    cleanup_pg_workspace,
    ingest_dataset,
)
from lightrag.evaluation.hybrid_bench.report import write_report
from lightrag.evaluation.hybrid_bench.rerank import rerank_survival
from lightrag.evaluation.hybrid_bench.write_amp import (
    measure_backfill,
    measure_index_sizes,
    measure_write_amp,
)

# Ingest, the three retrieval arms and metric scoring are the always-on core;
# the optional phases below surround them.
ALL_PHASES = (
    "ablation",
    "latency",
    "write-amp",
    "backfill",
    "rerank",
    "report",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m lightrag.evaluation.hybrid_bench",
        description="Reproducible hybrid-retrieval benchmark harness.",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--storage", choices=("pg", "file"), default="pg")
    parser.add_argument("--workspace-prefix", default="hbench")
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "stable workspace base for resumable runs: the per-dataset "
            "workspace is <workspace>_<dataset> with no random suffix, and "
            "--keep-workspace is implied; re-invoking with the same value "
            "resumes ingest through the product doc_status"
        ),
    )
    parser.add_argument("--working-dir", default=None)
    parser.add_argument(
        "--embedding-backend",
        choices=("ollama", "openai", "fake-hash"),
        default="ollama",
    )
    parser.add_argument(
        "--embedding-host",
        default=None,
        help=(
            "embedding endpoint; unset = local ollama URL for the ollama "
            "backend, EMBEDDING_BINDING_HOST env for the openai backend"
        ),
    )
    parser.add_argument("--embedding-model", default="bge-m3:latest")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument(
        "--llm-backend", choices=("ollama", "openai", "fake"), default="ollama"
    )
    parser.add_argument(
        "--llm-host",
        default=None,
        help=(
            "LLM endpoint; unset = local ollama URL for the ollama backend, "
            "LLM_BINDING_HOST env for the openai backend"
        ),
    )
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument(
        "--llm-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "openai backend only: forward enable_thinking via extra_body; "
            "--no-llm-enable-thinking disables a thinking model's reasoning "
            "for extraction speed; unset sends nothing"
        ),
    )
    parser.add_argument(
        "--max-parallel-insert",
        type=int,
        default=8,
        help="documents processed in parallel (LightRAG max_parallel_insert)",
    )
    parser.add_argument(
        "--max-async-llm",
        type=int,
        default=16,
        help="concurrent LLM requests (LightRAG llm_model_max_async)",
    )
    parser.add_argument(
        "--embedding-func-max-async",
        type=int,
        default=16,
        help="concurrent embedding requests (LightRAG embedding_func_max_async)",
    )
    parser.add_argument(
        "--embedding-batch-num",
        type=int,
        default=32,
        help="texts per embedding request (LightRAG embedding_batch_num)",
    )
    parser.add_argument("--chunk-token-size", type=int, default=8192)
    parser.add_argument("--cosine-threshold", type=float, default=0.2)
    parser.add_argument("--extraction-cap", type=int, default=0)
    parser.add_argument(
        "--max-gleaning",
        type=int,
        default=None,
        help=(
            "override LightRAG entity_extract_max_gleaning "
            "(0 = single extraction pass); default keeps the product default"
        ),
    )
    parser.add_argument(
        "--llm-num-predict",
        type=int,
        default=None,
        help=(
            "cap output tokens per LLM call (ollama options.num_predict; "
            "openai max_tokens); default leaves the output uncapped"
        ),
    )
    parser.add_argument(
        "--ingest-timeout",
        type=int,
        default=86400,
        help=(
            "bounded wait for the whole dataset ingest in seconds, "
            "including recovery rounds (default 86400 = 24 h)"
        ),
    )
    parser.add_argument(
        "--ingest-recovery-rounds",
        type=int,
        default=2,
        help=(
            "sanctioned manual-retry rounds for FAILED extraction documents "
            "(one attempt per document per round, same parameters); "
            "0 disables recovery"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-min-score", type=float, default=0.5)
    parser.add_argument(
        "--phases",
        default=",".join(ALL_PHASES),
        help=f"comma-separated subset of {ALL_PHASES}",
    )
    parser.add_argument("--write-amp-baseline", action="store_true")
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.bootstrap_resamples < 1000:
        parser.error("--bootstrap-resamples must be >= 1000")
    if args.latency_queries < 100:
        parser.error("--latency-queries must be >= 100")
    if args.latency_rounds < 3:
        parser.error("--latency-rounds must be >= 3")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.max_gleaning is not None and args.max_gleaning < 0:
        parser.error("--max-gleaning must be >= 0")
    if args.llm_num_predict is not None and args.llm_num_predict < 1:
        parser.error("--llm-num-predict must be >= 1")
    if args.ingest_timeout < 1:
        parser.error("--ingest-timeout must be >= 1")
    if args.ingest_recovery_rounds < 0:
        parser.error("--ingest-recovery-rounds must be >= 0")
    for flag in (
        "max_parallel_insert",
        "max_async_llm",
        "embedding_func_max_async",
        "embedding_batch_num",
    ):
        if getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1")
    if args.workspace:
        # A resumable workspace must survive the run: cleanup at exit would
        # delete exactly the progress a re-invocation resumes from.
        args.keep_workspace = True
    phases = [part.strip() for part in args.phases.split(",") if part.strip()]
    unknown = set(phases) - set(ALL_PHASES)
    if unknown:
        parser.error(f"unknown phases: {sorted(unknown)}")
    args.phase_set = set(phases)
    return args


def _fake_entity_specs(dataset: BenchDataset) -> dict[str, list[dict[str, str]]]:
    specs: dict[str, list[dict[str, str]]] = {}
    for cid, text in dataset.corpus.items():
        match = re.search(r"[a-z0-9]+|[\u4e00-\u9fff]{1,2}", text.lower())
        name = match.group(0) if match else cid
        specs[text] = [
            {
                "entity_name": name,
                "entity_type": "concept",
                "description": f"{name} appears in passage {cid}.",
            }
        ]
    return specs


def _resolve_workspace(args: argparse.Namespace, dataset_name: str) -> str:
    """Stable base + dataset name for resumable runs; random suffix else."""
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_name)
    if args.workspace:
        return f"{args.workspace}_{safe_name}"
    return f"{args.workspace_prefix}_{safe_name}_{uuid4().hex[:8]}"


def _base_config(args: argparse.Namespace, workspace: str) -> BenchConfig:
    return BenchConfig(
        storage=args.storage,
        working_dir=args.working_dir or str(Path(args.output_dir) / "working"),
        workspace=workspace,
        embedding_backend=args.embedding_backend,
        embedding_host=args.embedding_host or "",
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        llm_backend=args.llm_backend,
        llm_host=args.llm_host or "",
        llm_model=args.llm_model,
        llm_enable_thinking=args.llm_enable_thinking,
        max_parallel_insert=args.max_parallel_insert,
        max_async_llm=args.max_async_llm,
        embedding_func_max_async=args.embedding_func_max_async,
        embedding_batch_num=args.embedding_batch_num,
        chunk_token_size=args.chunk_token_size,
        cosine_threshold=args.cosine_threshold,
        extraction_cap=args.extraction_cap,
        max_gleaning=args.max_gleaning,
        llm_num_predict=args.llm_num_predict,
        ingest_timeout=args.ingest_timeout,
        ingest_recovery_rounds=args.ingest_recovery_rounds,
        top_k=args.top_k,
        seed=args.seed,
        resamples=args.bootstrap_resamples,
        latency_queries=args.latency_queries,
        latency_rounds=args.latency_rounds,
        rerank_model=args.rerank_model,
        rerank_min_score=args.rerank_min_score,
        keep_workspace=args.keep_workspace,
        request_timeout=args.request_timeout,
    )


async def _run_arms(rag, dataset, config, ingest, truth, phase_set):
    lexical_available = config.storage == "pg"
    active_arms = tuple(
        arm for arm in arms_mod.ARMS if arm != "lexical" or lexical_available
    )
    arm_docs: dict[str, dict[str, list[str]]] = {arm: {} for arm in active_arms}
    arm_entities: dict[str, dict[str, list[str]]] = {"dense": {}, "hybrid": {}}
    lexical_positive_hits: list[tuple[str, str]] = []
    skipped: dict[str, str] = {}
    if not lexical_available:
        skipped["lexical-arm"] = "lexical SQL requires PGVectorStorage"
    if config.storage != "pg":
        skipped["hybrid-dispatch"] = (
            "file vector backend lacks supports_hybrid_query; hybrid arm "
            "falls back to dense"
        )
    seeding_cache: dict[str, arms_mod.EntitySeeding] = {}
    seeding_fallbacks: dict[str, str] = {}
    for qid, query in dataset.queries.items():
        embedding = await arms_mod.embed_query(rag, query)
        seeding = await asyncio.wait_for(
            arms_mod.prepare_entity_seeding(rag, query, seeding_cache),
            timeout=config.request_timeout,
        )
        if seeding.fallback:
            seeding_fallbacks[qid] = seeding.fallback
        for arm in arms_mod.ARMS:
            if arm == "lexical" and not lexical_available:
                continue
            hits = await asyncio.wait_for(
                arms_mod.retrieve_chunks(
                    rag,
                    query,
                    arm,
                    config.top_k,
                    embedding,
                    ingest.chunk_doc_map,
                ),
                timeout=config.request_timeout,
            )
            arm_docs[arm][qid] = arms_mod.doc_id_sequence(hits)
            if arm == "lexical":
                positives = dataset.positives(qid)
                lexical_positive_hits.extend(
                    (query, hit.content)
                    for hit in hits
                    if any(doc_id in positives for doc_id in hit.doc_ids)
                )
        for arm, hybrid in (("dense", False), ("hybrid", True)):
            entities = await asyncio.wait_for(
                arms_mod.retrieve_entities(rag, seeding, hybrid, config.top_k),
                timeout=config.request_timeout,
            )
            arm_entities[arm][qid] = entities
    b5_per_query: dict[str, dict[str, dict[str, float]]] = {}
    if "ablation" in phase_set:
        for combo, (entity_hybrid, chunk_hybrid) in arms_mod.B5_COMBINATIONS.items():
            b5_per_query[combo] = {}
            for qid, query in dataset.queries.items():
                hits, entities = await asyncio.wait_for(
                    arms_mod.retrieve_combined(
                        rag,
                        query,
                        entity_hybrid,
                        chunk_hybrid,
                        config.top_k,
                        ingest.chunk_doc_map,
                    ),
                    timeout=config.request_timeout,
                )
                docs = arms_mod.doc_id_sequence(hits)
                positives = dataset.positives(qid)
                entry: dict[str, float] = {}
                if positives:
                    entry["recall"] = recall_at_k(docs, positives, config.top_k)
                    entry["ndcg"] = ndcg_at_k(docs, positives, config.top_k)
                if truth.per_query.get(qid):
                    entry["entity_hit"] = hit_at_k(
                        entities, truth.per_query[qid], config.top_k
                    )
                b5_per_query[combo][qid] = entry
    return (
        arm_docs,
        arm_entities,
        lexical_positive_hits,
        skipped,
        b5_per_query,
        seeding_fallbacks,
    )


async def _run_latency(rag, dataset, config):
    queries = list(dataset.queries.values())
    slots = latency_mod.select_query_slots(queries, config.latency_queries)
    results = {}
    async with latency_mod.build_latency_client(rag) as client:
        for phase, need_answer in (("retrieval", False), ("answer", True)):
            dense = await latency_mod.measure_latency(
                client,
                slots,
                config.latency_rounds,
                False,
                need_answer,
                config.top_k,
                config.request_timeout,
            )
            hybrid = await latency_mod.measure_latency(
                client,
                slots,
                config.latency_rounds,
                True,
                need_answer,
                config.top_k,
                config.request_timeout,
            )
            results[phase] = {
                "dense": {
                    "p50": dense.p50,
                    "p95": dense.p95,
                    "rounds": dense.rounds,
                    "query_slots": dense.query_slots,
                },
                "hybrid": {
                    "p50": hybrid.p50,
                    "p95": hybrid.p95,
                    "rounds": hybrid.rounds,
                    "query_slots": hybrid.query_slots,
                },
                "overhead": p95_overhead(dense.p95, hybrid.p95),
                "overhead_ci95": list(
                    bootstrap_p95_overhead_ci(
                        dense.all_samples,
                        hybrid.all_samples,
                        config.seed,
                        config.resamples,
                    )
                ),
                "per_query_dense": dense.per_query,
                "per_query_hybrid": hybrid.per_query,
                "all_samples_dense": dense.all_samples,
                "all_samples_hybrid": hybrid.all_samples,
            }
    return results


async def run_dataset(
    args: argparse.Namespace, dataset_dir: Path, dataset: BenchDataset
) -> dict:
    workspace = _resolve_workspace(args, dataset.name)
    config = _base_config(args, workspace)
    fake_llm = (
        FakeLLM(_fake_entity_specs(dataset)) if config.llm_backend == "fake" else None
    )
    if not dataset.positive_corpus_ids():
        raise ValueError(
            f"dataset {dataset.name!r} has no qrels-positive rows; chunk "
            "metrics are undefined for every query"
        )
    rag = await build_rag(config, build_llm_func(config, fake_llm))
    result: dict = {
        "name": dataset.name,
        "meta": asdict(dataset.meta),
        "skipped": {},
        "latency": None,
        "write_amp": None,
        "index_sizes": None,
        "backfill_chunks_per_min": None,
        "rerank": None,
        "ablation": None,
    }
    try:
        ingest = await asyncio.wait_for(
            ingest_dataset(rag, dataset, config), timeout=config.ingest_timeout
        )
        result["ingest"] = {
            "chunk_count": ingest.chunk_count,
            "extracted_doc_ids": ingest.extracted_doc_ids,
            "chunks_only_doc_ids": ingest.chunks_only_doc_ids,
            "extraction_cap": config.extraction_cap,
            "recovery_rounds_used": ingest.recovery_rounds_used,
            "recovered_doc_ids": ingest.recovered_doc_ids,
            "persistently_failed_doc_ids": ingest.persistently_failed_doc_ids,
            "duplicate_aliased_doc_ids": ingest.duplicate_aliased_doc_ids,
            "moderation_fallback_doc_ids": ingest.moderation_fallback_doc_ids,
            "resumed": ingest.resumed,
        }
        truth = await derive_entity_truth(
            rag.chunk_entity_relation_graph,
            rag.text_chunks,
            {qid: dataset.positives(qid) for qid in dataset.queries},
            entity_chunks=rag.entity_chunks,
        )
        result["entity_truth"] = {
            "queries_with_truth": len(truth.per_query),
            "zero_truth_query_ids": truth.zero_truth_query_ids,
        }
        (
            arm_docs,
            arm_entities,
            lexical_hits,
            skipped,
            b5_per_query,
            seeding_fallbacks,
        ) = await _run_arms(rag, dataset, config, ingest, truth, args.phase_set)
        result["skipped"].update(skipped)
        result["entity_seeding_protocol"] = arms_mod.ENTITY_SEEDING_PROTOCOL
        result["entity_seeding_fallbacks"] = seeding_fallbacks
        scored, zero_positive, empty = scoring.score_chunk_arms(
            dataset, arm_docs, config.top_k
        )
        entity_scored = scoring.score_entity_arms(
            truth.per_query, arm_entities, config.top_k
        )
        k = config.top_k
        result["zero_positive_query_ids"] = zero_positive
        result["empty_retrieval_query_ids"] = empty
        result["chunk_metrics"] = {
            arm: {
                f"recall@{k}": scoring.summarize(
                    [values["recall"] for values in scored[arm].values()],
                    config.seed,
                    config.resamples,
                ).as_dict(),
                f"ndcg@{k}": scoring.summarize(
                    [values["ndcg"] for values in scored[arm].values()],
                    config.seed,
                    config.resamples,
                ).as_dict(),
                "per_query": scored[arm],
            }
            for arm in scored
        }
        result["entity_metrics"] = {
            arm: {
                f"hit@{k}": scoring.summarize(
                    list(values.values()), config.seed, config.resamples
                ).as_dict(),
                "per_query": values,
            }
            for arm, values in entity_scored.items()
            if values
        }
        result["_scored"] = scored
        result["_entity_scored"] = entity_scored
        if b5_per_query:

            def _b5_mean(entries: dict, key: str) -> float:
                values = [e[key] for e in entries.values() if key in e]
                return mean(values) if values else 0.0

            result["ablation"] = {
                combo: {
                    f"recall@{k}": _b5_mean(entries, "recall"),
                    f"ndcg@{k}": _b5_mean(entries, "ndcg"),
                    f"entity_hit@{k}": _b5_mean(entries, "entity_hit"),
                }
                for combo, entries in b5_per_query.items()
            }
            result["_b5_per_query"] = b5_per_query
        if "latency" in args.phase_set:
            result["latency"] = await _run_latency(rag, dataset, config)
        if "write-amp" in args.phase_set and config.storage == "pg":
            write_amp = await asyncio.to_thread(
                measure_write_amp,
                config,
                dataset_dir / "corpus.jsonl",
                _REPO_ROOT,
                args.baseline_ref,
                32,
                args.write_amp_baseline,
            )
            gin_bytes, vector_bytes = await measure_index_sizes(rag)
            ratio = (
                gin_bytes / vector_bytes
                if gin_bytes is not None and vector_bytes
                else None
            )
            result["write_amp"] = asdict(write_amp)
            result["index_sizes"] = {
                "gin_bytes": gin_bytes,
                "vector_bytes": vector_bytes,
                "ratio": ratio,
            }
        elif "write-amp" in args.phase_set:
            result["skipped"]["write-amp"] = "requires PG storage"
        if "backfill" in args.phase_set and config.storage == "pg":
            result["backfill_chunks_per_min"] = await measure_backfill(rag)
        elif "backfill" in args.phase_set:
            result["skipped"]["backfill"] = "requires PG storage"
        if "rerank" in args.phase_set:
            rerank = await asyncio.to_thread(
                rerank_survival,
                lexical_hits,
                config.rerank_min_score,
                config.rerank_model,
            )
            result["rerank"] = asdict(rerank)
        if fake_llm is not None:
            result["fake_llm_calls"] = {
                family: fake_llm.calls.count(family)
                for family in ("extraction", "keywords", "answer")
            }
        return result
    finally:
        await rag.finalize_storages()
        if config.storage == "pg" and not config.keep_workspace:
            await cleanup_pg_workspace(config)


def _domain_comparisons(
    datasets: list[dict], metric: str, seed: int, resamples: int
) -> dict:
    comparisons = {}
    for domain in ("terminology", "general"):
        members = [d for d in datasets if d["meta"]["domain"] == domain]
        if not members:
            continue
        dense = scoring.pool_values([d["_scored"] for d in members], "dense", metric)
        hybrid = scoring.pool_values([d["_scored"] for d in members], "hybrid", metric)
        if dense and hybrid:
            comparisons[domain] = scoring.comparison(dense, hybrid, seed, resamples)
    return comparisons


def _entity_comparisons(datasets: list[dict], seed: int, resamples: int) -> dict:
    comparisons = {}
    for domain in ("terminology", "general"):
        members = [
            d for d in datasets if d["meta"]["domain"] == domain and d["_entity_scored"]
        ]
        if not members:
            continue
        dense = scoring.pool_flat([d["_entity_scored"] for d in members], "dense")
        hybrid = scoring.pool_flat([d["_entity_scored"] for d in members], "hybrid")
        if dense and hybrid:
            comparisons[domain] = scoring.comparison(dense, hybrid, seed, resamples)
    return comparisons


def _gate_inputs(datasets: list[dict], args: argparse.Namespace) -> GateInputs:
    seed, resamples = args.seed, args.bootstrap_resamples
    overheads: dict[str, float | None] = {"retrieval": None, "answer": None}
    for phase in overheads:
        dense_pool: list[float] = []
        hybrid_pool: list[float] = []
        for dataset in datasets:
            latency = dataset.get("latency")
            if latency:
                # Gate on the SAME statistic the report displays: the pooled
                # all-samples (flat) p95, never the per-query round means.
                dense_pool.extend(latency[phase]["all_samples_dense"])
                hybrid_pool.extend(latency[phase]["all_samples_hybrid"])
        if dense_pool:
            overheads[phase] = p95_overhead(
                percentile(dense_pool, 95), percentile(hybrid_pool, 95)
            )
    first_write_amp = next(
        (d["write_amp"] for d in datasets if d.get("write_amp")), None
    )
    first_sizes = next(
        (d["index_sizes"] for d in datasets if d.get("index_sizes")), None
    )
    first_backfill = next(
        (
            d["backfill_chunks_per_min"]
            for d in datasets
            if d.get("backfill_chunks_per_min") is not None
        ),
        None,
    )
    scored_hits = sum(d["rerank"]["scored_hits"] for d in datasets if d.get("rerank"))
    surviving_hits = sum(
        d["rerank"]["surviving_hits"] for d in datasets if d.get("rerank")
    )
    survival = surviving_hits / scored_hits if scored_hits else None
    ablation = None
    b5_members = [d for d in datasets if d.get("_b5_per_query")]
    if b5_members:
        ablation = {}
        for metric, key in (
            (f"recall@{args.top_k}", "recall"),
            (f"ndcg@{args.top_k}", "ndcg"),
            (f"entity_hit@{args.top_k}", "entity_hit"),
        ):
            pooled = {}
            for combo in arms_mod.B5_COMBINATIONS:
                values = [
                    entry[key]
                    for d in b5_members
                    for entry in d["_b5_per_query"][combo].values()
                    if key in entry
                ]
                pooled[combo] = mean(values) if values else 0.0
            ablation[metric] = (
                pooled["both"],
                pooled["entity_only"],
                pooled["chunk_only"],
            )
    return GateInputs(
        entity_hit=_entity_comparisons(datasets, seed, resamples),
        chunk_recall=_domain_comparisons(datasets, "recall", seed, resamples),
        chunk_ndcg=_domain_comparisons(datasets, "ndcg", seed, resamples),
        retrieval_p95_overhead=overheads["retrieval"],
        answer_p95_overhead=overheads["answer"],
        upsert_increase=(
            first_write_amp["upsert_increase"] if first_write_amp else None
        ),
        gin_vector_ratio=first_sizes["ratio"] if first_sizes else None,
        backfill_chunks_per_min=first_backfill,
        rerank_survival=survival,
        ablation=ablation,
    )


async def run_bench(args: argparse.Namespace) -> dict:
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)
    dataset_dirs = discover_datasets(Path(args.dataset_dir))
    datasets = []
    for dataset_dir in dataset_dirs:
        dataset = load_dataset(dataset_dir)
        print(
            f"[hybrid-bench] dataset {dataset.name}: "
            f"{len(dataset.corpus)} passages, {len(dataset.queries)} queries",
            flush=True,
        )
        datasets.append(await run_dataset(args, dataset_dir, dataset))
    inputs = _gate_inputs(datasets, args)
    verdicts = evaluate_gates(inputs, GateThresholds(), args.top_k)
    for verdict in verdicts:
        print(
            f"[gate] {verdict.gate_id}: {verdict.status.upper()} ({verdict.detail})",
            flush=True,
        )
    for dataset in datasets:
        dataset.pop("_scored", None)
        dataset.pop("_entity_scored", None)
        dataset.pop("_b5_per_query", None)
    config = _base_config(args, "appendix")
    results = {
        "synthetic": config.synthetic,
        "config": asdict(config),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "thresholds": asdict(GateThresholds()),
        "datasets": datasets,
        "gates": [asdict(verdict) for verdict in verdicts],
    }
    if "report" in args.phase_set:
        md_path, json_path = write_report(results, Path(args.output_dir))
        print(f"[hybrid-bench] report: {md_path} and {json_path}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    asyncio.run(run_bench(args))
    return 0
