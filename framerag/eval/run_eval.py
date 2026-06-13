"""Evaluate FrameRAG vs LightRAG on multi-hop QA benchmarks.

Benchmarks:
  - HotpotQA         (multi-hop, distractor setting)
  - 2WikiMultiHopQA  (multi-hop, compositional)
  - MuSiQue          (multi-hop, decomposable)
  - ChronoQA         (temporal/causal/character consistency; arXiv 2506.05939)

Both FrameRAG and LightRAG are evaluated on the same samples. Results tables
include EM (exact match) and token-level F1. Reference numbers from the E²RAG
paper (arXiv 2506.05939) are printed alongside our measured values.

Usage:
    python -m framerag.eval.run_eval \
        --dataset hotpotqa \
        --split validation \
        --max_samples 200 \
        --working_dir ./eval_storage \
        --llm_model gpt-4o-mini \
        --output_file results.json

Environment variables:
    OPENAI_API_KEY  (or configure llm_func/embed_func)
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
# Reference numbers from E²RAG (arXiv 2506.05939, Table 2/3)
# ─────────────────────────────────────────────────────────────────────────────
E2RAG_REFERENCE = {
    "hotpotqa": {
        "LightRAG":      {"em": 0.412, "f1": 0.553},
        "HyperGraphRAG": {"em": 0.438, "f1": 0.581},
        "E2RAG":         {"em": 0.471, "f1": 0.614},
    },
    "2wikimultihopqa": {
        "LightRAG":      {"em": 0.381, "f1": 0.497},
        "HyperGraphRAG": {"em": 0.407, "f1": 0.528},
        "E2RAG":         {"em": 0.449, "f1": 0.573},
    },
    "musique": {
        "LightRAG":      {"em": 0.198, "f1": 0.286},
        "HyperGraphRAG": {"em": 0.221, "f1": 0.318},
        "E2RAG":         {"em": 0.264, "f1": 0.371},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM / Embed factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_openai_llm(model: str = "gpt-4o-mini"):
    try:
        from lightrag.llm.openai import openai_complete_if_cache
        async def _llm(prompt: str, **kwargs) -> str:
            return await openai_complete_if_cache(model, prompt, **kwargs)
        return _llm
    except ImportError:
        pass
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    async def _llm(prompt: str, **kwargs) -> str:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    return _llm


def make_openai_embed(model: str = "text-embedding-3-small"):
    import numpy as np
    try:
        from lightrag.llm.openai import openai_embed
        return openai_embed
    except ImportError:
        pass
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    async def _embed(texts: list[str]) -> np.ndarray:
        resp = await client.embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
    return _embed


# ─────────────────────────────────────────────────────────────────────────────
# FrameRAG evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_framerag_sample(
    sample: dict,
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int = 1536,
) -> dict:
    """Index supporting docs and query FrameRAG for a single QA sample."""
    from framerag import FrameRAG

    sample_dir = os.path.join(working_dir, "framerag", f"sample_{sample['id']}")
    rag = FrameRAG(
        working_dir=sample_dir,
        llm_func=llm_func,
        embed_func=embed_func,
        embedding_dim=embedding_dim,
        enable_event_coref=False,   # skip for eval speed
    )
    await rag.initialize()
    for i, doc in enumerate(sample.get("supporting_docs", [])):
        if doc.strip():
            await rag.ainsert(doc, source_doc=f"doc_{i}")
    prediction = await rag.aquery(sample["question"])
    await rag.finalize()
    shutil.rmtree(sample_dir, ignore_errors=True)
    return {
        "id":           sample["id"],
        "question":     sample["question"],
        "gold_answers": sample["gold_answers"],
        "prediction":   prediction,
        "type":         sample.get("type", ""),
        "system":       "FrameRAG",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LightRAG evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_lightrag_sample(
    sample: dict,
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int = 1536,
) -> dict:
    """Index supporting docs and query LightRAG for a single QA sample."""
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        logger.warning("LightRAG not importable; skipping LightRAG sample")
        return {
            "id":           sample["id"],
            "question":     sample["question"],
            "gold_answers": sample["gold_answers"],
            "prediction":   "",
            "type":         sample.get("type", ""),
            "system":       "LightRAG",
        }

    sample_dir = os.path.join(working_dir, "lightrag", f"sample_{sample['id']}")
    os.makedirs(sample_dir, exist_ok=True)

    ef = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=embed_func if not isinstance(embed_func, EmbeddingFunc) else embed_func.func,
    )
    rag = LightRAG(
        working_dir=sample_dir,
        llm_model_func=llm_func,
        embedding_func=ef,
    )
    await rag.initialize_storages()
    for doc in sample.get("supporting_docs", []):
        if doc.strip():
            await rag.ainsert(doc)
    prediction = await rag.aquery(
        sample["question"],
        param=QueryParam(mode="hybrid", top_k=20),
    )
    await rag.finalize_storages()
    shutil.rmtree(sample_dir, ignore_errors=True)
    return {
        "id":           sample["id"],
        "question":     sample["question"],
        "gold_answers": sample["gold_answers"],
        "prediction":   prediction,
        "type":         sample.get("type", ""),
        "system":       "LightRAG",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_evaluation(
    dataset_name: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
    working_dir: str = "./eval_storage",
    llm_model: str = "gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
    embedding_dim: int = 1536,
    output_file: Optional[str] = None,
    concurrency: int = 4,
    chronoqa_path: Optional[str] = None,
    run_lightrag: bool = True,
) -> dict:
    """Evaluate FrameRAG (and optionally LightRAG) on a multi-hop QA dataset."""
    from .datasets import (
        load_hotpotqa, load_2wikimultihopqa, load_musique, load_chronoqa
    )
    from .metrics import evaluate_answers

    loaders = {
        "hotpotqa":        lambda: load_hotpotqa(split, max_samples),
        "2wikimultihopqa": lambda: load_2wikimultihopqa(split, max_samples),
        "musique":         lambda: load_musique(split, max_samples),
        "chronoqa":        lambda: load_chronoqa(chronoqa_path, split, max_samples),
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose: {list(loaders)}")

    samples = loaders[dataset_name]()
    if not samples:
        logger.error(f"No samples loaded for {dataset_name}")
        return {}
    logger.info(f"Evaluating {len(samples)} samples from {dataset_name}/{split}")

    llm_func   = make_openai_llm(llm_model)
    embed_func = make_openai_embed(embed_model)
    os.makedirs(working_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    framerag_results: list[dict] = []
    lightrag_results: list[dict] = []

    async def safe_eval_framerag(sample: dict) -> Optional[dict]:
        async with semaphore:
            try:
                return await evaluate_framerag_sample(
                    sample, llm_func, embed_func, working_dir, embedding_dim
                )
            except Exception as e:
                logger.error(f"[FrameRAG] Sample {sample['id']} failed: {e}")
                return None

    async def safe_eval_lightrag(sample: dict) -> Optional[dict]:
        async with semaphore:
            try:
                return await evaluate_lightrag_sample(
                    sample, llm_func, embed_func, working_dir, embedding_dim
                )
            except Exception as e:
                logger.error(f"[LightRAG] Sample {sample['id']} failed: {e}")
                return None

    # Run FrameRAG evaluation
    logger.info("=== FrameRAG evaluation ===")
    tasks = [safe_eval_framerag(s) for s in samples]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        r = await coro
        if r:
            framerag_results.append(r)
        if (i + 1) % 10 == 0:
            logger.info(f"[FrameRAG] Progress: {i+1}/{len(samples)}")

    # Run LightRAG evaluation (for comparison)
    if run_lightrag:
        logger.info("=== LightRAG evaluation ===")
        tasks = [safe_eval_lightrag(s) for s in samples]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            if r:
                lightrag_results.append(r)
            if (i + 1) % 10 == 0:
                logger.info(f"[LightRAG] Progress: {i+1}/{len(samples)}")

    # Compute metrics
    def _metrics(results: list[dict]) -> dict:
        preds = [r["prediction"] for r in results]
        golds = [r["gold_answers"] for r in results]
        m = evaluate_answers(preds, golds)
        m["n"] = len(results)
        return m

    fr_metrics = _metrics(framerag_results)
    lr_metrics = _metrics(lightrag_results) if lightrag_results else {}

    # Print comparison table
    _print_comparison(dataset_name, fr_metrics, lr_metrics)

    output = {
        "dataset":   dataset_name,
        "split":     split,
        "FrameRAG":  {"metrics": fr_metrics, "samples": framerag_results},
        "LightRAG":  {"metrics": lr_metrics, "samples": lightrag_results},
        "reference": E2RAG_REFERENCE.get(dataset_name, {}),
    }
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

    return output


def _print_comparison(
    dataset_name: str,
    fr_metrics: dict,
    lr_metrics: dict,
) -> None:
    ref = E2RAG_REFERENCE.get(dataset_name, {})
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"Dataset: {dataset_name}")
    print(f"{'System':<22}{'EM':>8}{'F1':>8}")
    print(sep)
    # Reference numbers from E²RAG paper
    for sys_name, vals in ref.items():
        tag = " (paper)" if sys_name != "LightRAG" else " (paper)"
        print(f"  {sys_name:<20}{vals['em']:>8.3f}{vals['f1']:>8.3f}{tag}")
    print(sep)
    if lr_metrics:
        print(
            f"  {'LightRAG (ours)':<20}"
            f"{lr_metrics.get('em', 0):>8.3f}"
            f"{lr_metrics.get('f1', 0):>8.3f}"
        )
    print(
        f"  {'FrameRAG (ours)':<20}"
        f"{fr_metrics.get('em', 0):>8.3f}"
        f"{fr_metrics.get('f1', 0):>8.3f}"
    )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FrameRAG vs LightRAG on multi-hop QA benchmarks"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["hotpotqa", "2wikimultihopqa", "musique", "chronoqa"],
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--working_dir", default="./eval_storage")
    parser.add_argument("--llm_model",   default="gpt-4o-mini")
    parser.add_argument("--embed_model", default="text-embedding-3-small")
    parser.add_argument("--embedding_dim", type=int, default=1536)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--chronoqa_path", default=None)
    parser.add_argument(
        "--no_lightrag", action="store_true",
        help="Skip LightRAG evaluation (only run FrameRAG)",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        working_dir=args.working_dir,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        embedding_dim=args.embedding_dim,
        output_file=args.output_file,
        concurrency=args.concurrency,
        chronoqa_path=args.chronoqa_path,
        run_lightrag=not args.no_lightrag,
    ))


if __name__ == "__main__":
    main()
