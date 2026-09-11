"""Paired context-selection experiment; see docs/SteinerContextPrototype.md.

Default: synthetic retrieval diagnostics, real LightRAG stages 2--4, no LLM.
--factory module:make_rag --dataset qa.json: real retrieval and generated QA.
The factory must return an initialized LightRAG instance over an existing KG.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import tiktoken

from lightrag import QueryParam
from lightrag.constants import (
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MIN_RERANK_SCORE,
)
from lightrag.evaluation.steiner_metrics import RerankMeter, paired_quality, summarize
from lightrag.operate import (
    _apply_token_truncation,
    _build_context_str,
    _merge_all_chunks,
    _perform_kg_search,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer, logger

ARMS = ("rank", "relevance", "soft_greedy", "steiner_soft", "steiner_hard")
FACTORIAL_ARMS = {
    "A0": ("ordering", "rank"),
    "A1": ("ordering", "relevance"),
    "A2": ("ordering", "steiner_soft"),
    "B0": ("rerank_score", "rank"),
    "B1": ("rerank_score", "relevance"),
    "B2": ("rerank_score", "steiner_soft"),
}


async def demo_rerank(*, query, documents, top_n=None):
    """Deterministic lexical test double; never represents a real rerank model."""
    query_words = set(normalize(query))
    rows = [
        {
            "index": i,
            "relevance_score": len(query_words & set(normalize(doc)))
            / max(1, len(query_words)),
        }
        for i, doc in enumerate(documents)
    ]
    return sorted(rows, key=lambda row: -row["relevance_score"])[:top_n]


class DemoChunks:
    """In-memory source chunks; production chunk selection still runs."""

    def __init__(self, rows: dict, config: dict):
        self.rows = rows
        self.global_config = config

    async def get_by_ids(self, ids: list) -> list:
        return [self.rows.get(i) for i in ids]


def demo_cases() -> list[dict]:
    # Hand-authored diagnostics, not an accuracy benchmark or a tuned test set.
    return [
        {
            "id": "isolate_and_connection",
            "category": "isolated evidence",
            "question": "Who leads Atlas and what is the remote access code?",
            "answer": "Mira; amber",
            "entities": [
                ("Access", "The remote access code is amber."),
                ("Atlas", "Atlas is a research project."),
                ("Mira", "Mira is a scientist."),
            ],
            "relations": [("Mira", "Atlas", "Mira leads Atlas.")],
            "evidence": ["remote access code is amber", "Mira leads Atlas"],
        },
        {
            "id": "bridge",
            "category": "multi-hop evidence",
            "question": "Which organization employs the person who founded Lumen?",
            "answer": "Northstar",
            "entities": [
                ("Lumen", "Lumen develops optics."),
                ("Northstar", "Northstar employs researchers."),
                ("Optics", "Optics studies light. " * 12),
                ("Iris", "Iris is an engineer."),
            ],
            "relations": [
                ("Lumen", "Optics", "Lumen works in optics. " * 12),
                ("Iris", "Lumen", "Iris founded Lumen."),
                ("Iris", "Northstar", "Northstar employs Iris."),
            ],
            "evidence": ["Iris founded Lumen", "Northstar employs Iris"],
        },
        {
            "id": "two_islands",
            "category": "disconnected synthesis",
            "question": "Which cities host the Aurora and Boreal offices?",
            "answer": "Delft; Bergen",
            "entities": [
                ("Aurora", "Aurora has its office in Delft."),
                ("Boreal", "Boreal has its office in Bergen."),
                ("Archive", "Archive stores old office records. " * 12),
            ],
            "relations": [],
            "evidence": ["office in Delft", "office in Bergen"],
        },
        {
            "id": "verbose_first_hit",
            "category": "packing diagnostic",
            "question": "What is the approved launch month for Cedar?",
            "answer": "October",
            "entities": [
                ("CedarHistory", "Cedar planning has a long history. " * 50),
                ("Launch", "Cedar launches in October."),
                ("Cedar", "Cedar is an engineering program."),
            ],
            "relations": [],
            "evidence": ["Cedar launches in October"],
        },
    ]


def demo_snapshot(case: dict, config: dict) -> tuple[dict, SimpleNamespace]:
    chunks = {}
    entities, relations = [], []
    for i, (name, description) in enumerate(case["entities"]):
        source = f"entity-{i}"
        chunks[source] = {"content": description, "file_path": f"{source}.txt"}
        entities.append(
            {
                "entity_name": name,
                "entity_type": "DEMO",
                "description": description,
                "source_id": source,
            }
        )
    for i, (src, tgt, description) in enumerate(case["relations"]):
        source = f"relation-{i}"
        chunks[source] = {"content": description, "file_path": f"{source}.txt"}
        relations.append(
            {"src_tgt": (src, tgt), "description": description, "source_id": source}
        )
    snapshot = {
        "final_entities": entities,
        "final_relations": relations,
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": None,
    }
    rag = SimpleNamespace(
        text_chunks=DemoChunks(chunks, config),
        chunk_entity_relation_graph=None,
        chunks_vdb=None,
    )
    return snapshot, rag


async def run_context(
    rag,
    snapshot: dict,
    question: str,
    param: QueryParam,
    config: dict,
    arm: str,
    beta: float,
    max_candidates: int,
) -> dict:
    snapshot = copy.deepcopy(snapshot)  # tracking is mutable in stage 3
    if arm in FACTORIAL_ARMS and not param.enable_rerank:
        raise ValueError("The six-arm comparison requires chunk reranking enabled")
    prize_source, strategy = FACTORIAL_ARMS.get(arm, ("ordering", arm))
    meter = RerankMeter(config.get("rerank_model_func"), config["tokenizer"])
    config = {
        **config,
        "addon_params": {
            **config.get("addon_params", {}),
            "context_selection": {
                "strategy": strategy,
                "prize_source": prize_source,
                "connectivity_bonus": beta,
                "max_candidates": max_candidates,
            },
        },
    }
    if param.enable_rerank:
        if not callable(meter.func):
            raise ValueError(
                "Enabled-reranker comparisons require a configured rerank_model_func"
            )
        config["rerank_model_func"] = meter.wrap("kg")
    started = perf_counter()
    selected = await _apply_token_truncation(snapshot, param, config, query=question)
    selector_ms = (perf_counter() - started) * 1000
    merged = await _merge_all_chunks(
        filtered_entities=selected["filtered_entities"],
        filtered_relations=selected["filtered_relations"],
        vector_chunks=snapshot["vector_chunks"],
        query=question,
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        text_chunks_db=rag.text_chunks,
        query_param=param,
        chunks_vdb=rag.chunks_vdb,
        chunk_tracking=snapshot["chunk_tracking"],
        query_embedding=snapshot["query_embedding"],
    )
    if param.enable_rerank:
        config["rerank_model_func"] = meter.wrap("chunks")
    context, raw = await _build_context_str(
        entities_context=selected["entities_context"],
        relations_context=selected["relations_context"],
        merged_chunks=merged,
        query=question,
        query_param=param,
        global_config=config,
        chunk_tracking=snapshot["chunk_tracking"],
        entity_id_to_original=selected["entity_id_to_original"],
        relation_id_to_original=selected["relation_id_to_original"],
    )
    stages_2_4_ms = (perf_counter() - started) * 1000
    if meter.errors:
        raise ValueError(f"Invalid reranking comparison: {meter.errors}")
    # No custom user prefix/history in this experiment: the exact same prompt
    # and question go to every answerer, with only retrieved context changing.
    prompt = PROMPTS["rag_response"].format(
        context_data=context, response_type=param.response_type, user_prompt=""
    )
    tokenizer = config["tokenizer"]
    input_tokens = len(tokenizer.encode(prompt)) + len(tokenizer.encode(question))
    counts = [
        len(
            tokenizer.encode(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in selected[key])
            )
        )
        for key in ("entities_context", "relations_context")
    ]
    # Stage 4 reserves overhead but is not a universal final prompt guarantee.
    # Refuse over-budget comparisons instead of silently using extra tokens.
    if (
        input_tokens > param.max_total_tokens
        or counts[0] > param.max_entity_tokens
        or counts[1] > param.max_relation_tokens
    ):
        raise ValueError(
            f"Budget violation for {arm}: input={input_tokens}, KG={counts}"
        )
    return {
        "context": context,
        "system_prompt": prompt,
        "entity_tokens": counts[0],
        "relation_tokens": counts[1],
        "input_tokens": input_tokens,
        "selector_ms": selector_ms,
        "stages_2_4_ms": stages_2_4_ms,
        "prize_source": prize_source,
        "strategy": strategy,
        "candidate_count": len(snapshot["final_entities"])
        + len(snapshot["final_relations"]),
        "rerank": meter.totals(),
        "rerank_calls": meter.calls,
        "selected_entities": [x["entity"] for x in selected["entities_context"]],
        "selected_relations": [
            [x["entity1"], x["entity2"]] for x in selected["relations_context"]
        ],
        "chunk_count": len(raw.get("data", {}).get("chunks", [])),
        "selection": selected.get(
            "selection_metadata", {"executed_strategy": "rank", "solver_calls": 0}
        ),
    }


def normalize(text: str) -> list[str]:
    return [x for x in re.findall(r"\w+", text.lower()) if x not in {"a", "an", "the"}]


def answer_scores(answer: str, aliases: list[str]) -> dict:
    predicted = Counter(normalize(answer))
    exact, f1 = 0.0, 0.0
    for alias in aliases:
        reference = Counter(normalize(alias))
        common = sum((predicted & reference).values())
        denominator = sum(predicted.values()) + sum(reference.values())
        f1 = max(f1, 2 * common / denominator if denominator else 1.0)
        exact = max(exact, float(normalize(answer) == normalize(alias)))
    return {"answer_exact_match": exact, "answer_token_f1": f1}


def snapshot_fingerprint(snapshot):
    # Ignore mutable chunk tracking and the numerical query embedding.
    payload = {
        key: snapshot[key]
        for key in ("final_entities", "final_relations", "vector_chunks")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def validate_live(cases, config, args):
    if not callable(config.get("rerank_model_func")):
        raise TypeError("Live evaluation requires a configured rerank_model_func")
    if not callable(config.get("role_llm_funcs", {}).get("query")):
        raise TypeError("Live evaluation requires a configured answer model")
    if config.get("user_prompt_prefix") or config.get("enable_llm_cache", False):
        raise ValueError(
            "Disable response caching and user_prompt_prefix in the benchmark factory"
        )
    if (
        config.get("min_rerank_score", DEFAULT_MIN_RERANK_SCORE)
        != DEFAULT_MIN_RERANK_SCORE
    ):
        raise ValueError(
            "Acceptance comparison requires project-default min_rerank_score"
        )
    if not all((args.index_id, args.answer_model_id, args.rerank_model_id)):
        raise ValueError("Record --index-id, --answer-model-id and --rerank-model-id")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a nonempty JSON array")
    if not args.smoke_test and not 30 <= len(cases) <= 60:
        raise ValueError(
            "Use 30–60 held-out questions; --smoke-test labels smaller contract runs"
        )
    ids = set()
    for case in cases:
        for key in (
            "id",
            "category",
            "question",
            "ll_keywords",
            "hl_keywords",
            "answer",
        ):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Every case requires a nonempty string {key}")
        if case["id"] in ids:
            raise ValueError("Question IDs must be unique")
        ids.add(case["id"])
        aliases = case.get("answer_aliases", [case["answer"]])
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(x, str) or not x.strip() for x in aliases)
        ):
            raise ValueError("answer_aliases must contain nonempty answer strings")
    required = {"isolated evidence", "multi-hop evidence", "disconnected evidence"}
    if not args.smoke_test and not required <= {x["category"] for x in cases}:
        raise ValueError(
            f"Include all held-out evidence categories: {sorted(required)}"
        )


async def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tokenizer = Tokenizer(args.encoding, tiktoken.get_encoding(args.encoding))
    config = {
        "tokenizer": tokenizer,
        "kg_chunk_pick_method": "WEIGHT",
        "related_chunk_number": 5,
        "min_rerank_score": DEFAULT_MIN_RERANK_SCORE,
        "rerank_model_func": demo_rerank,
        "enable_content_headings": False,
    }
    live_rag = None
    if args.factory:
        if not args.dataset:
            raise ValueError("--factory requires --dataset")
        module, name = args.factory.split(":", 1)
        live_rag = getattr(importlib.import_module(module), name)()
        if inspect.isawaitable(live_rag):
            live_rag = await live_rag
    try:
        if live_rag:
            config = live_rag._build_global_config()
            cases = json.loads(Path(args.dataset).read_text())
            validate_live(cases, config, args)
        else:
            cases = demo_cases()
        if (
            args.repeats < 1
            or args.load_repeats < 1
            or any(x < 1 for x in args.concurrency)
        ):
            raise ValueError("Repetitions and concurrency must be positive")
        # Defaults mirror production for a live study; tiny limits belong only
        # to the explicitly labelled synthetic diagnostics.
        entity_budget = (
            args.entity_budget
            if args.entity_budget is not None
            else (DEFAULT_MAX_ENTITY_TOKENS if live_rag else 100)
        )
        relation_budget = (
            args.relation_budget
            if args.relation_budget is not None
            else (DEFAULT_MAX_RELATION_TOKENS if live_rag else 90)
        )
        budgets = args.budgets or ([DEFAULT_MAX_TOTAL_TOKENS] if live_rag else [1600])
        args.output.mkdir(parents=True, exist_ok=True)
        rows = []
        load_batches = []

        async def search(case, param):
            started = perf_counter()
            if live_rag:
                snapshot = await _perform_kg_search(
                    case["question"],
                    case["ll_keywords"],
                    case["hl_keywords"],
                    live_rag.chunk_entity_relation_graph,
                    live_rag.entities_vdb,
                    live_rag.relationships_vdb,
                    live_rag.text_chunks,
                    param,
                    live_rag.chunks_vdb,
                )
                return snapshot, live_rag, (perf_counter() - started) * 1000
            snapshot, rag = demo_snapshot(case, config)
            return snapshot, rag, 0.0

        with (
            (args.output / "results.jsonl").open("w") as output,
            (args.output / "snapshots.jsonl").open("w") as snapshots_file,
        ):

            def write(row):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                rows.append(row)

            for top_k in args.top_k:
                for budget in budgets:
                    param = QueryParam(
                        mode=args.mode,
                        top_k=top_k,
                        chunk_top_k=args.chunk_top_k,
                        max_entity_tokens=entity_budget,
                        max_relation_tokens=relation_budget,
                        max_total_tokens=budget,
                        enable_rerank=True,
                        response_type="One short factual answer; no explanation",
                    )
                    canonical = {}
                    for case in cases:
                        snapshot, rag, search_ms = await search(case, param)
                        fingerprint = snapshot_fingerprint(snapshot)
                        canonical[case["id"]] = fingerprint
                        snapshots_file.write(
                            json.dumps(
                                {
                                    "case_id": case["id"],
                                    "top_k": top_k,
                                    "max_total_tokens": budget,
                                    "sha256": fingerprint,
                                    **{
                                        k: snapshot[k]
                                        for k in (
                                            "final_entities",
                                            "final_relations",
                                            "vector_chunks",
                                        )
                                    },
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n"
                        )
                        # All six arms see the identical raw candidates. KG
                        # scores are re-requested, never cached across B arms.
                        for arm in FACTORIAL_ARMS:
                            await run_context(
                                rag,
                                snapshot,
                                case["question"],
                                param,
                                config,
                                arm,
                                args.beta,
                                args.max_candidates,
                            )
                        for repeat in range(args.repeats):
                            arms = list(FACTORIAL_ARMS)
                            rng.shuffle(arms)
                            for arm in arms:
                                result = await run_context(
                                    rag,
                                    snapshot,
                                    case["question"],
                                    param,
                                    config,
                                    arm,
                                    args.beta,
                                    args.max_candidates,
                                )
                                answer = None
                                generation_ms = None
                                scores = {
                                    "answer_exact_match": None,
                                    "answer_token_f1": None,
                                }
                                if live_rag:
                                    started = perf_counter()
                                    answer = await config["role_llm_funcs"]["query"](
                                        case["question"],
                                        system_prompt=result["system_prompt"],
                                        history_messages=[],
                                        stream=False,
                                        temperature=0,
                                    )
                                    generation_ms = (perf_counter() - started) * 1000
                                    scores = answer_scores(
                                        answer,
                                        case.get("answer_aliases", [case["answer"]]),
                                    )
                                evidence = case.get("evidence", [])
                                write(
                                    {
                                        **result,
                                        "phase": "quality",
                                        "concurrency": 1,
                                        "case_id": case["id"],
                                        "category": case["category"],
                                        "question": case["question"],
                                        "arm": arm,
                                        "repeat": repeat,
                                        "top_k": top_k,
                                        "max_total_tokens": budget,
                                        "max_entity_tokens": entity_budget,
                                        "max_relation_tokens": relation_budget,
                                        "snapshot_sha256": fingerprint,
                                        "search_ms": search_ms,
                                        "retrieval_ms": search_ms
                                        + result["stages_2_4_ms"],
                                        "latency_scope": "shared_stage1_plus_stages2_4_excludes_keywords",
                                        "evidence_recall": sum(
                                            x.lower() in result["context"].lower()
                                            for x in evidence
                                        )
                                        / len(evidence)
                                        if evidence
                                        else None,
                                        "answer": answer,
                                        "generation_ms": generation_ms,
                                        "answer_tokens": len(
                                            config["tokenizer"].encode(answer)
                                        )
                                        if answer is not None
                                        else None,
                                        **scores,
                                    }
                                )

                    # Separate homogeneous closed-loop load trials. Every
                    # request performs actual stage 1; no answer generation or
                    # shared search times enter the concurrency measurements.
                    for concurrency in args.concurrency:
                        arms = list(FACTORIAL_ARMS)
                        rng.shuffle(arms)
                        jobs = cases * max(
                            args.load_repeats,
                            math_ceil_div(2 * concurrency, len(cases)),
                        )
                        rng.shuffle(jobs)
                        for arm in arms:
                            semaphore = asyncio.Semaphore(concurrency)

                            async def request(
                                number,
                                case,
                                *,
                                semaphore=semaphore,
                                param=param,
                                canonical=canonical,
                                arm=arm,
                                concurrency=concurrency,
                                top_k=top_k,
                                budget=budget,
                            ):
                                async with semaphore:
                                    started = perf_counter()
                                    snapshot, rag, search_ms = await search(case, param)
                                    if (
                                        snapshot_fingerprint(snapshot)
                                        != canonical[case["id"]]
                                    ):
                                        raise ValueError(
                                            "Index/candidate ordering changed during paired benchmark"
                                        )
                                    result = await run_context(
                                        rag,
                                        snapshot,
                                        case["question"],
                                        param,
                                        config,
                                        arm,
                                        args.beta,
                                        args.max_candidates,
                                    )
                                    retrieval_ms = (perf_counter() - started) * 1000
                                    return {
                                        **result,
                                        "phase": "load",
                                        "concurrency": concurrency,
                                        "case_id": case["id"],
                                        "category": case["category"],
                                        "arm": arm,
                                        "repeat": number,
                                        "top_k": top_k,
                                        "max_total_tokens": budget,
                                        "max_entity_tokens": entity_budget,
                                        "max_relation_tokens": relation_budget,
                                        "search_ms": search_ms,
                                        "retrieval_ms": retrieval_ms,
                                        "latency_scope": "stages1_4_closed_loop_excludes_keywords_and_answers",
                                    }

                            started = perf_counter()
                            tasks = [
                                asyncio.create_task(request(i, case))
                                for i, case in enumerate(jobs)
                            ]
                            try:
                                batch = await asyncio.gather(*tasks)
                            finally:
                                for task in tasks:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.gather(*tasks, return_exceptions=True)
                            elapsed = perf_counter() - started
                            for row in batch:
                                write(row)
                            load_batches.append(
                                {
                                    "arm": arm,
                                    "concurrency": concurrency,
                                    "top_k": top_k,
                                    "max_total_tokens": budget,
                                    "queries": len(batch),
                                    "wall_seconds": elapsed,
                                    "queries_per_second": len(batch) / elapsed,
                                }
                            )

        report = {
            "kind": ("live_smoke_test" if args.smoke_test else "live_paired_qa")
            if live_rag
            else "synthetic_diagnostics_lexical_reranker_no_answers",
            "settings": {
                **vars(args),
                "output": str(args.output),
                "budgets": budgets,
                "entity_budget": entity_budget,
                "relation_budget": relation_budget,
                "enable_rerank": True,
                "min_rerank_score": config.get(
                    "min_rerank_score", DEFAULT_MIN_RERANK_SCORE
                ),
            },
            "dataset_sha256": hashlib.sha256(
                Path(args.dataset).read_bytes()
            ).hexdigest()
            if live_rag
            else None,
            "summary": summarize(rows),
            "paired_quality": paired_quality(rows, args.seed),
            "load_batches": load_batches,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("networkx", "steinerpy", "tiktoken")
                },
            },
            "implementation_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).parent.parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "steiner_context.py",
                    "evaluation/steiner_ablation.py",
                    "evaluation/steiner_metrics.py",
                    "operate.py",
                )
            },
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
        logger.info("Saved %d paired and load results to %s", len(rows), args.output)
    finally:
        if live_rag:
            await live_rag.finalize_storages()


def math_ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--index-id",
        help="Immutable corpus/index version, including embedding configuration",
    )
    parser.add_argument(
        "--answer-model-id",
        help="Exact answer model/version and provider configuration",
    )
    parser.add_argument(
        "--rerank-model-id",
        help="Exact shared KG/chunk reranker and hardware/service configuration",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Label a small live contract run; not acceptance evidence",
    )
    parser.add_argument("--output", type=Path, default=Path("temp/steiner-ablation"))
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--entity-budget", type=int)
    parser.add_argument("--relation-budget", type=int)
    parser.add_argument("--top-k", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--chunk-top-k", type=int, default=DEFAULT_CHUNK_TOP_K)
    parser.add_argument(
        "--mode", choices=["local", "global", "hybrid", "mix"], default="hybrid"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="*",
        default=[1, 8, 16, 32],
        help="Closed-loop retrieval-only load levels; empty list skips load trials",
    )
    parser.add_argument("--load-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3866)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--encoding", default="cl100k_base")
    return parser


if __name__ == "__main__":
    asyncio.run(main(argument_parser().parse_args()))
