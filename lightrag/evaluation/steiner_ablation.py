"""Paired context-selection experiment; see docs/SteinerContextPrototype.md.

Default: synthetic retrieval diagnostics, real LightRAG stages 2--4, no LLM.
--factory module:make_rag --dataset qa.json: real retrieval and generated QA.
The factory must return an initialized LightRAG instance over an existing KG.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib
import inspect
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import tiktoken

from lightrag import QueryParam
from lightrag.operate import (
    _apply_token_truncation,
    _build_context_str,
    _merge_all_chunks,
    _perform_kg_search,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer, logger

ARMS = ("rank", "relevance", "soft_greedy", "steiner_soft", "steiner_hard")


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
    config = {
        **config,
        "addon_params": {
            **config.get("addon_params", {}),
            "context_selection": {
                "strategy": arm,
                "connectivity_bonus": beta,
                "max_candidates": max_candidates,
            },
        },
    }
    started = perf_counter()
    selected = await _apply_token_truncation(snapshot, param, config)
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


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["max_total_tokens"], row["top_k"])].append(row)
    summaries = []
    for (arm, budget, top_k), group in groups.items():
        item = {"arm": arm, "max_total_tokens": budget, "top_k": top_k, "n": len(group)}
        for key in ("selector_ms", "retrieval_ms"):
            values = sorted(x[key] for x in group)
            item[key + "_p50"] = statistics.median(values)
            item[key + "_p95"] = values[max(0, math.ceil(0.95 * len(values)) - 1)]
        for key in (
            "evidence_recall",
            "answer_exact_match",
            "answer_token_f1",
            "input_tokens",
        ):
            values = [x[key] for x in group if x.get(key) is not None]
            item[key + "_mean"] = statistics.mean(values) if values else None
        item["fallback_rate"] = statistics.mean(
            bool(x["selection"].get("fallback_reason")) for x in group
        )
        summaries.append(item)
    return summaries


async def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tokenizer = Tokenizer(args.encoding, tiktoken.get_encoding(args.encoding))
    config = {
        "tokenizer": tokenizer,
        "kg_chunk_pick_method": "WEIGHT",
        "related_chunk_number": 5,
        "min_rerank_score": 0.0,
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
        config = live_rag._build_global_config()
        if config.get("user_prompt_prefix"):
            raise ValueError("Use a benchmark factory without user_prompt_prefix")
        cases = json.loads(Path(args.dataset).read_text())
    else:
        cases = demo_cases()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with (args.output / "results.jsonl").open("w") as output:
            for top_k in args.top_k:
                for budget in args.budgets:
                    param = QueryParam(
                        mode=args.mode,
                        top_k=top_k,
                        chunk_top_k=args.chunk_top_k,
                        max_entity_tokens=args.entity_budget,
                        max_relation_tokens=args.relation_budget,
                        max_total_tokens=budget,
                        enable_rerank=args.rerank,
                        response_type="One short factual answer; no explanation",
                    )
                    for case in cases:
                        for repeat in range(args.repeats):
                            if live_rag:
                                t0 = perf_counter()
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
                                search_ms = (perf_counter() - t0) * 1000
                                rag = live_rag
                            else:
                                snapshot, rag = demo_snapshot(case, config)
                                search_ms = 0.0
                            # Warm each arm once per case/budget, discard timings;
                            # raw candidate snapshot is identical across arms.
                            if repeat == 0:
                                for arm in ARMS:
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
                            arms = list(ARMS)
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
                                scores = {
                                    "answer_exact_match": None,
                                    "answer_token_f1": None,
                                }
                                generation_ms = None
                                if live_rag:
                                    t0 = perf_counter()
                                    answer = await config["role_llm_funcs"]["query"](
                                        case["question"],
                                        system_prompt=result["system_prompt"],
                                        history_messages=[],
                                        stream=False,
                                        temperature=0,
                                    )
                                    generation_ms = (perf_counter() - t0) * 1000
                                    scores = answer_scores(
                                        answer,
                                        case.get("answer_aliases", [case["answer"]]),
                                    )
                                evidence = case.get("evidence", [])
                                result.update(
                                    {
                                        "case_id": case["id"],
                                        "category": case.get("category", "unspecified"),
                                        "question": case["question"],
                                        "arm": arm,
                                        "repeat": repeat,
                                        "top_k": top_k,
                                        "max_total_tokens": budget,
                                        "max_entity_tokens": args.entity_budget,
                                        "max_relation_tokens": args.relation_budget,
                                        "search_ms": search_ms,
                                        "retrieval_ms": search_ms
                                        + result["stages_2_4_ms"],
                                        "latency_scope": "stages_1_4_shared_search_excludes_keywords"
                                        if live_rag
                                        else "stages_2_4_in_memory_sources",
                                        "evidence_recall": sum(
                                            x.lower() in result["context"].lower()
                                            for x in evidence
                                        )
                                        / len(evidence)
                                        if evidence
                                        else None,
                                        "answer": answer,
                                        "generation_ms": generation_ms,
                                        **scores,
                                    }
                                )
                                output.write(
                                    json.dumps(result, ensure_ascii=False) + "\n"
                                )
                                output.flush()
                                rows.append(result)
        report = {
            "kind": "live_paired_qa"
            if live_rag
            else "synthetic_diagnostics_no_answer_generation",
            "settings": {**vars(args), "output": str(args.output)},
            "summary": summarize(rows),
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
        logger.info("Saved %d paired results to %s", len(rows), args.output)
    finally:
        if live_rag:
            await live_rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory")
    parser.add_argument("--dataset")
    parser.add_argument("--output", type=Path, default=Path("temp/steiner-ablation"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1600])
    parser.add_argument("--entity-budget", type=int, default=100)
    parser.add_argument("--relation-budget", type=int, default=90)
    parser.add_argument("--top-k", type=int, nargs="+", default=[60])
    parser.add_argument("--chunk-top-k", type=int, default=10)
    parser.add_argument(
        "--mode", choices=["local", "global", "hybrid", "mix"], default="hybrid"
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3866)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--encoding", default="cl100k_base")
    asyncio.run(main(parser.parse_args()))
