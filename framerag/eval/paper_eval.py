"""Eval 2: LLM-judge evaluation matching the E²RAG paper (arXiv 2506.05939).

The paper evaluates RAG systems using LLM-as-judge on a 1–10 Likert scale
with 3 independent LLM judges (Claude-3.7-Sonnet, GPT-4o, GPT-4.1-mini).

Score = (1/J) * Σ_j (1/N) * Σ_i s_ij

where N = number of QA pairs, J = number of judges, s_ij ∈ {1..10}.

Reference overall average scores (Table 3, E²RAG paper):
  Rank 1: E²RAG (comb. extraction)  7.1257
  Rank 5: LightRAG hybrid           6.8804
  Rank 8: E²RAG (vanilla)           6.7083
  Rank 9: vanilla RAG               6.6022

Datasets:
  - ChronoQA  : https://huggingface.co/datasets/zy113/ChronoQA (narrative QA)
  - HotpotQA  : hotpot_qa/distractor
  - 2WikiMHQA : voidful/2WikiMultihopQA
  - MuSiQue   : drt/musique

Usage:
    python -m framerag.eval.paper_eval \\
        --dataset chronoqa \\
        --max_samples 50 \\
        --working_dir ./eval_storage \\
        --llm_model gpt-4o-mini \\
        --judge_models gpt-4o,gpt-4.1-mini \\
        --output_file results_llm_judge.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Scoring rubric (matches E²RAG paper Appendix D.2 spirit)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_RUBRIC = """You are an expert evaluator for question-answering systems.

Given:
- QUESTION: the question asked
- GROUND TRUTH: the reference correct answer
- SYSTEM RESPONSE: the answer produced by the RAG system under evaluation

Score the SYSTEM RESPONSE on a 1–10 Likert scale:
  10 : Perfectly accurate and comprehensive; matches the ground truth fully.
  8–9: Very accurate; captures all key information with minor gaps.
  6–7: Mostly correct; addresses the main points but has noticeable gaps.
  4–5: Partially correct; some relevant content but significant inaccuracies.
  2–3: Mostly incorrect or irrelevant; only superficial overlap with ground truth.
  1  : Completely wrong or no meaningful content.

Scoring rules:
- Judge factual correctness, not writing style.
- Penalise hallucinations (confident false claims) heavily.
- Reward completeness relative to the ground truth.
- A response that acknowledges uncertainty is better than a confident wrong answer.

Respond with ONLY a JSON object:
{{"score": <integer 1-10>, "reason": "<one sentence explanation>"}}

QUESTION: {question}

GROUND TRUTH: {ground_truth}

SYSTEM RESPONSE: {response}
"""

# Reference numbers from Table 3 of E²RAG paper (LLM-judge, 1–10 scale)
E2RAG_REFERENCE = {
    "overall_avg": {
        "E2RAG (comb. extraction)":  7.1257,
        "E2RAG (comb. embedding)":   7.0719,
        "E2RAG (hyp. extraction)":   6.9832,
        "E2RAG (hyp. embedding)":    6.9395,
        "LightRAG hybrid":           6.8804,
        "GraphRAG drift":            6.8206,
        "GraphRAG local":            6.7997,
        "E2RAG (vanilla)":           6.7083,
        "vanilla RAG":               6.6022,
        "LightRAG local":            6.5497,
        "GraphRAG global":           6.5087,
        "LightRAG global":           6.4583,
        "vanilla HyDE":              6.3555,
    }
}

DEFAULT_JUDGE_MODELS = ["gpt-4o-mini"]


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm(model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Return an async callable (prompt: str) -> str for the given model."""
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_LLM_BINDING_API_KEY")
    url = base_url or os.getenv("EVAL_LLM_BINDING_HOST")
    try:
        from openai import AsyncOpenAI
        kwargs: dict = {"api_key": key} if key else {}
        if url:
            kwargs["base_url"] = url
        client = AsyncOpenAI(**kwargs)

        async def _call(prompt: str) -> str:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""

        return _call
    except ImportError:
        raise ImportError("pip install openai")


def _make_embed(model: str = "text-embedding-3-small",
                api_key: Optional[str] = None, base_url: Optional[str] = None):
    import numpy as np
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_LLM_BINDING_API_KEY")
    url = base_url or os.getenv("EVAL_LLM_BINDING_HOST")
    try:
        from openai import AsyncOpenAI
        kwargs: dict = {"api_key": key} if key else {}
        if url:
            kwargs["base_url"] = url
        client = AsyncOpenAI(**kwargs)

        async def _embed(texts: list[str], **kwargs) -> np.ndarray:
            resp = await client.embeddings.create(model=model, input=texts)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)

        return _embed
    except ImportError:
        raise ImportError("pip install openai")


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge
# ─────────────────────────────────────────────────────────────────────────────

async def _judge_one(
    question: str,
    ground_truth: str,
    response: str,
    judge_llm,
) -> tuple[int, str]:
    """Score one (question, ground_truth, response) triple. Returns (score, reason)."""
    prompt = JUDGE_RUBRIC.format(
        question=question,
        ground_truth=ground_truth,
        response=response,
    )
    try:
        raw = await judge_llm(prompt)
        # Extract JSON from response
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(raw[start:end])
            score = int(data.get("score", 5))
            reason = str(data.get("reason", ""))
            return max(1, min(10, score)), reason
    except Exception as e:
        logger.warning(f"Judge parse error: {e} | raw: {repr(raw[:200])}")
    return 5, "parse error"


async def judge_response(
    question: str,
    ground_truth: str,
    response: str,
    judge_llms: list,
    sem: asyncio.Semaphore,
) -> dict:
    """Score with all judges; return averaged score and per-judge details."""
    results = []
    for judge_llm in judge_llms:
        async with sem:
            score, reason = await _judge_one(question, ground_truth, response, judge_llm)
            results.append({"score": score, "reason": reason})
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"avg_score": avg, "judges": results}


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample RAG evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def _run_framerag(sample: dict, llm_func, embed_func,
                        working_dir: str, embedding_dim: int) -> str:
    from framerag import FrameRAG
    sample_dir = os.path.join(working_dir, "framerag", f"s_{sample['id']}")
    rag = FrameRAG(
        working_dir=sample_dir,
        llm_func=llm_func,
        embed_func=embed_func,
        embedding_dim=embedding_dim,
        enable_event_coref=False,
    )
    await rag.initialize()
    for i, doc in enumerate(sample.get("supporting_docs", [])):
        if doc.strip():
            await rag.ainsert(doc, source_doc=f"doc_{i}")
    answer = await rag.aquery(sample["question"])
    await rag.finalize()
    shutil.rmtree(sample_dir, ignore_errors=True)
    return answer


async def _run_lightrag(sample: dict, llm_func, embed_func,
                        working_dir: str, embedding_dim: int) -> str:
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        logger.warning("LightRAG not importable; skipping")
        return ""
    sample_dir = os.path.join(working_dir, "lightrag", f"s_{sample['id']}")
    os.makedirs(sample_dir, exist_ok=True)
    raw_embed = embed_func.func if hasattr(embed_func, "func") else embed_func
    ef = EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=raw_embed)
    rag = LightRAG(working_dir=sample_dir, llm_model_func=llm_func, embedding_func=ef)
    await rag.initialize_storages()
    for doc in sample.get("supporting_docs", []):
        if doc.strip():
            await rag.ainsert(doc)
    answer = await rag.aquery(
        sample["question"], param=QueryParam(mode="hybrid", top_k=20)
    )
    await rag.finalize_storages()
    shutil.rmtree(sample_dir, ignore_errors=True)
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_paper_evaluation(
    dataset_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    working_dir: str = "./eval_storage",
    llm_model: str = "gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
    embedding_dim: int = 1536,
    judge_models: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    output_file: Optional[str] = None,
    concurrency: int = 4,
    judge_concurrency: int = 8,
    run_lightrag: bool = True,
) -> dict:
    """Evaluate FrameRAG (and optionally LightRAG) with LLM-judge scoring (E²RAG protocol).

    Args:
        dataset_name:    'chronoqa' | 'hotpotqa' | '2wikimultihopqa' | 'musique'
        split:           Dataset split.
        max_samples:     Limit samples (None = all).
        working_dir:     Temp directory for per-sample RAG storages.
        llm_model:       LLM used by FrameRAG/LightRAG for answering.
        embed_model:     Embedding model for retrieval.
        embedding_dim:   Dimensionality of embeddings.
        judge_models:    List of judge LLM model IDs (default: ['gpt-4o-mini']).
                         Paper used: ['claude-3-7-sonnet', 'gpt-4o', 'gpt-4.1-mini']
        api_key:         OpenAI-compatible API key.
        base_url:        Custom endpoint (optional).
        output_file:     Path to save JSON results.
        concurrency:     Max parallel RAG evaluations.
        judge_concurrency: Max parallel judge LLM calls.
        run_lightrag:    Whether to also evaluate LightRAG as baseline.

    Returns:
        dict with FrameRAG and LightRAG results, per-sample scores, and summary.
    """
    from .datasets import (
        load_hotpotqa, load_2wikimultihopqa, load_musique, load_chronoqa
    )

    judge_models = judge_models or DEFAULT_JUDGE_MODELS

    loaders = {
        "chronoqa":        lambda: load_chronoqa(split=split, max_samples=max_samples),
        "hotpotqa":        lambda: load_hotpotqa(split=split, max_samples=max_samples),
        "2wikimultihopqa": lambda: load_2wikimultihopqa(split=split, max_samples=max_samples),
        "musique":         lambda: load_musique(split=split, max_samples=max_samples),
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(loaders)}")

    samples = loaders[dataset_name]()
    if not samples:
        logger.error(f"No samples loaded for {dataset_name}/{split}")
        return {}

    logger.info(f"E²RAG paper eval: {len(samples)} samples from {dataset_name}/{split}")
    logger.info(f"Judge models: {judge_models}")

    llm_func   = _make_llm(llm_model, api_key, base_url)
    embed_func = _make_embed(embed_model, api_key, base_url)
    judge_llms = [_make_llm(m, api_key, base_url) for m in judge_models]
    os.makedirs(working_dir, exist_ok=True)

    rag_sem   = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(judge_concurrency)

    async def _safe_run(coro, label: str):
        async with rag_sem:
            try:
                return await coro
            except Exception as e:
                logger.error(f"[{label}] failed: {e}")
                return None

    # ── Generate answers ──────────────────────────────────────────────────────
    logger.info("=== Generating FrameRAG answers ===")
    framerag_answers: list[Optional[str]] = []
    tasks = [
        _safe_run(_run_framerag(s, llm_func, embed_func, working_dir, embedding_dim),
                  f"FrameRAG/{s['id']}")
        for s in samples
    ]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        ans = await coro
        framerag_answers.append(ans)
        if (i + 1) % 10 == 0:
            logger.info(f"[FrameRAG] {i+1}/{len(samples)} answered")

    lightrag_answers: list[Optional[str]] = []
    if run_lightrag:
        logger.info("=== Generating LightRAG answers ===")
        tasks = [
            _safe_run(_run_lightrag(s, llm_func, embed_func, working_dir, embedding_dim),
                      f"LightRAG/{s['id']}")
            for s in samples
        ]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            ans = await coro
            lightrag_answers.append(ans)
            if (i + 1) % 10 == 0:
                logger.info(f"[LightRAG] {i+1}/{len(samples)} answered")

    # ── Judge all answers ─────────────────────────────────────────────────────
    logger.info("=== LLM judging ===")

    async def _judge_sample(sample: dict, response: Optional[str], system: str) -> dict:
        if not response:
            return {
                "id": sample["id"], "question": sample["question"],
                "gold_answers": sample.get("gold_answers", []),
                "response": "", "avg_score": 0.0, "judges": [], "system": system,
                "type": sample.get("type", ""),
            }
        ground_truth = "; ".join(sample.get("gold_answers", []))
        verdict = await judge_response(
            sample["question"], ground_truth, response, judge_llms, judge_sem
        )
        return {
            "id":           sample["id"],
            "question":     sample["question"],
            "gold_answers": sample.get("gold_answers", []),
            "response":     response,
            "avg_score":    verdict["avg_score"],
            "judges":       verdict["judges"],
            "system":       system,
            "type":         sample.get("type", ""),
        }

    judge_tasks = []
    for sample, ans in zip(samples, framerag_answers):
        judge_tasks.append(_judge_sample(sample, ans, "FrameRAG"))
    if lightrag_answers:
        for sample, ans in zip(samples, lightrag_answers):
            judge_tasks.append(_judge_sample(sample, ans, "LightRAG"))

    all_judgements = await asyncio.gather(*judge_tasks)

    framerag_results = [r for r in all_judgements if r["system"] == "FrameRAG"]
    lightrag_results = [r for r in all_judgements if r["system"] == "LightRAG"]

    def _summary(results: list[dict]) -> dict:
        if not results:
            return {"avg_score": 0.0, "n": 0, "by_type": {}}
        from collections import defaultdict
        avg = sum(r["avg_score"] for r in results) / len(results)
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in results:
            t = r.get("type") or "general"
            by_type[t].append(r["avg_score"])
        return {
            "avg_score": avg,
            "n": len(results),
            "by_type": {t: sum(s) / len(s) for t, s in by_type.items()},
        }

    fr_summary = _summary(framerag_results)
    lr_summary = _summary(lightrag_results)

    _print_results(dataset_name, fr_summary, lr_summary)

    output = {
        "dataset":     dataset_name,
        "split":       split,
        "metric":      "LLM-judge 1–10 Likert (E²RAG paper protocol)",
        "judge_models": judge_models,
        "FrameRAG":    {"summary": fr_summary, "samples": framerag_results},
        "LightRAG":    {"summary": lr_summary, "samples": lightrag_results},
        "reference":   E2RAG_REFERENCE,
    }
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved → {output_file}")

    return output


def _print_results(dataset_name: str, fr_summary: dict, lr_summary: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Dataset: {dataset_name}  |  Metric: LLM-judge 1–10 (E²RAG paper)")
    print(f"{'System':<28}{'Avg Score':>12}{'N':>8}")
    print(sep)

    # Paper reference
    for sys_name, score in E2RAG_REFERENCE["overall_avg"].items():
        print(f"  {sys_name:<26}{score:>12.4f}  (paper)")

    print(sep)
    if lr_summary.get("n", 0) > 0:
        print(
            f"  {'LightRAG (ours)':<26}"
            f"{lr_summary['avg_score']:>12.4f}"
            f"{lr_summary['n']:>8}"
        )
    print(
        f"  {'FrameRAG (ours)':<26}"
        f"{fr_summary['avg_score']:>12.4f}"
        f"{fr_summary['n']:>8}"
    )

    if fr_summary.get("by_type"):
        print(f"\n  FrameRAG by question type:")
        for qtype, score in sorted(fr_summary["by_type"].items()):
            print(f"    {qtype:<22} {score:.4f}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eval 2: LLM-judge evaluation (1–10 Likert) matching E²RAG paper"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["chronoqa", "hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--split",            default="test")
    parser.add_argument("--max_samples",      type=int, default=None)
    parser.add_argument("--working_dir",      default="./eval_storage")
    parser.add_argument("--llm_model",        default="gpt-4o-mini",
                        help="LLM for RAG answering")
    parser.add_argument("--embed_model",      default="text-embedding-3-small")
    parser.add_argument("--embedding_dim",    type=int, default=1536)
    parser.add_argument("--judge_models",     default="gpt-4o-mini",
                        help="Comma-separated judge model IDs. "
                             "Paper uses: claude-3-7-sonnet,gpt-4o,gpt-4.1-mini")
    parser.add_argument("--api_key",          default=None)
    parser.add_argument("--base_url",         default=None)
    parser.add_argument("--output_file",      default=None)
    parser.add_argument("--concurrency",      type=int, default=4)
    parser.add_argument("--judge_concurrency", type=int, default=8)
    parser.add_argument("--no_lightrag",      action="store_true")
    args = parser.parse_args()

    asyncio.run(run_paper_evaluation(
        dataset_name      = args.dataset,
        split             = args.split,
        max_samples       = args.max_samples,
        working_dir       = args.working_dir,
        llm_model         = args.llm_model,
        embed_model       = args.embed_model,
        embedding_dim     = args.embedding_dim,
        judge_models      = [m.strip() for m in args.judge_models.split(",")],
        api_key           = args.api_key,
        base_url          = args.base_url,
        output_file       = args.output_file,
        concurrency       = args.concurrency,
        judge_concurrency = args.judge_concurrency,
        run_lightrag      = not args.no_lightrag,
    ))


if __name__ == "__main__":
    main()
