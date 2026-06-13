"""Run FrameRAG evaluation on multi-hop QA benchmarks.

Usage:
    python -m framerag.eval.run_eval \
        --dataset hotpotqa \
        --split validation \
        --max_samples 200 \
        --working_dir ./eval_storage \
        --llm_model gpt-4o-mini \
        --output_file results.json

Environment variables required:
    OPENAI_API_KEY  (or configure llm_func / embed_func below)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import uuid
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM / Embed function factories (OpenAI by default)
# ─────────────────────────────────────────────────────────────────────────────

def make_openai_llm(model: str = "gpt-4o-mini") -> object:
    """Create async OpenAI LLM function."""
    try:
        from lightrag.llm.openai import openai_complete_if_cache
        async def llm_func(prompt: str, **kwargs) -> str:
            return await openai_complete_if_cache(model, prompt, **kwargs)
        return llm_func
    except ImportError:
        pass
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        async def llm_func(prompt: str, **kwargs) -> str:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        return llm_func
    except ImportError:
        raise ImportError("Install openai: pip install openai")


def make_openai_embed(model: str = "text-embedding-3-small") -> object:
    """Create async OpenAI embedding function."""
    try:
        from lightrag.llm.openai import openai_embed
        return openai_embed
    except ImportError:
        pass
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        async def embed_func(texts: list[str]) -> np.ndarray:
            resp = await client.embeddings.create(model=model, input=texts)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        return embed_func
    except ImportError:
        raise ImportError("Install openai: pip install openai")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation logic
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_sample(
    sample: dict,
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int = 1536,
) -> dict:
    """Index supporting docs and query FrameRAG for a single QA sample."""
    from framerag import FrameRAG

    sample_dir = os.path.join(working_dir, f"sample_{sample['id']}")

    rag = FrameRAG(
        working_dir=sample_dir,
        llm_func=llm_func,
        embed_func=embed_func,
        embedding_dim=embedding_dim,
        enable_causal=True,
        enable_event_coref=False,  # skip for speed in eval
    )
    await rag.initialize()

    for i, doc in enumerate(sample["supporting_docs"]):
        if doc.strip():
            await rag.ainsert(doc, source_doc=f"doc_{i}")

    prediction = await rag.aquery(sample["question"])
    await rag.finalize()

    # Clean up to save disk
    shutil.rmtree(sample_dir, ignore_errors=True)

    return {
        "id": sample["id"],
        "question": sample["question"],
        "gold_answers": sample["gold_answers"],
        "prediction": prediction,
        "type": sample.get("type", ""),
    }


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
) -> dict:
    """Run full evaluation on a dataset."""
    from .datasets import (
        load_hotpotqa, load_2wikimultihopqa, load_musique, load_chronoqa
    )
    from .metrics import evaluate_answers

    # Load dataset
    loaders = {
        "hotpotqa": lambda: load_hotpotqa(split, max_samples),
        "2wikimultihopqa": lambda: load_2wikimultihopqa(split, max_samples),
        "musique": lambda: load_musique(split, max_samples),
        "chronoqa": lambda: load_chronoqa(chronoqa_path, split, max_samples),
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose: {list(loaders)}")

    samples = loaders[dataset_name]()
    if not samples:
        logger.error(f"No samples loaded for {dataset_name}")
        return {}

    logger.info(f"Evaluating {len(samples)} samples from {dataset_name}/{split}")

    llm_func = make_openai_llm(llm_model)
    embed_func = make_openai_embed(embed_model)

    os.makedirs(working_dir, exist_ok=True)

    # Run with limited concurrency
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    errors = 0

    async def safe_eval(sample: dict) -> Optional[dict]:
        nonlocal errors
        async with semaphore:
            try:
                return await evaluate_sample(
                    sample, llm_func, embed_func, working_dir, embedding_dim
                )
            except Exception as e:
                logger.error(f"Sample {sample['id']} failed: {e}")
                errors += 1
                return None

    tasks = [safe_eval(s) for s in samples]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        if result:
            results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(samples)} done, {errors} errors")

    # Compute aggregate metrics
    predictions = [r["prediction"] for r in results]
    gold_answers = [r["gold_answers"] for r in results]
    metrics = evaluate_answers(predictions, gold_answers)
    metrics["errors"] = errors
    metrics["dataset"] = dataset_name
    metrics["split"] = split

    # Per-type breakdown if type field present
    types = set(r.get("type", "") for r in results if r.get("type"))
    if types:
        type_metrics = {}
        for t in types:
            t_results = [r for r in results if r.get("type") == t]
            t_preds = [r["prediction"] for r in t_results]
            t_gold = [r["gold_answers"] for r in t_results]
            type_metrics[t] = evaluate_answers(t_preds, t_gold)
        metrics["by_type"] = type_metrics

    logger.info(f"Results: EM={metrics['em']:.3f}, F1={metrics['f1']:.3f}, N={metrics['n']}")

    # Save results
    output = {"metrics": metrics, "samples": results}
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate FrameRAG on multi-hop QA benchmarks")
    parser.add_argument("--dataset", required=True,
                        choices=["hotpotqa", "2wikimultihopqa", "musique", "chronoqa"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--working_dir", default="./eval_storage")
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--embed_model", default="text-embedding-3-small")
    parser.add_argument("--embedding_dim", type=int, default=1536)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--chronoqa_path", default=None,
                        help="Path to ChronoQA JSON file (required for chronoqa dataset)")
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
    ))


if __name__ == "__main__":
    main()
