"""Evaluate FrameRAG vs LightRAG on multi-hop QA benchmarks using RAGAS metrics.

Benchmarks:
  - HotpotQA         (multi-hop, distractor setting)
  - 2WikiMultiHopQA  (multi-hop, compositional)
  - MuSiQue          (multi-hop, decomposable)
  - ChronoQA         (temporal/causal/character consistency; arXiv 2506.05939)

Both FrameRAG and LightRAG are evaluated on the same samples with the same
4 RAGAS metrics used in lightrag/evaluation/eval_rag_quality.py:
  - Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision, ragas_score

Reference EM/F1 numbers from the E²RAG paper (arXiv 2506.05939) are included
for context. RAGAS scores are LLM-judged, so results will differ from paper
numbers which used lexical metrics.

Usage:
    python -m framerag.eval.run_eval \\
        --dataset hotpotqa \\
        --split validation \\
        --max_samples 50 \\
        --working_dir ./eval_storage \\
        --llm_model gpt-4o-mini \\
        --output_file results.json

Environment variables:
    OPENAI_API_KEY                 (required for LLM + RAGAS scoring)
    EVAL_LLM_BINDING_API_KEY       (overrides OPENAI_API_KEY for RAGAS)
    EVAL_LLM_BINDING_HOST          (custom endpoint for local models)
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
# Reference numbers from E²RAG paper (arXiv 2506.05939, Table 2/3)
# Listed for context — paper used lexical EM/F1, not RAGAS.
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
# LLM / Embed factories
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
# FrameRAG — index + query + return contexts for RAGAS
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_framerag_sample(
    sample: dict,
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int = 1536,
) -> dict:
    """Index supporting docs, query FrameRAG, return answer + retrieved contexts."""
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

    # aquery_with_context returns (answer, passages_list) in a single retrieve pass
    prediction, contexts = await rag.aquery_with_context(sample["question"])
    await rag.finalize()
    shutil.rmtree(sample_dir, ignore_errors=True)

    gold_list = sample["gold_answers"]
    gold_str  = gold_list[0] if gold_list else ""

    return {
        "id":           sample["id"],
        "question":     sample["question"],
        "gold_answers": gold_list,
        "ground_truth": gold_str,
        "prediction":   prediction,
        "contexts":     contexts,
        "type":         sample.get("type", ""),
        "system":       "FrameRAG",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LightRAG — index + query + extract contexts for RAGAS
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_lightrag_sample(
    sample: dict,
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int = 1536,
) -> dict:
    """Index supporting docs, query LightRAG, return answer + retrieved contexts."""
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        logger.warning("LightRAG not importable; skipping LightRAG sample")
        return {
            "id": sample["id"], "question": sample["question"],
            "gold_answers": sample["gold_answers"], "ground_truth": "",
            "prediction": "", "contexts": [], "type": sample.get("type", ""),
            "system": "LightRAG",
        }

    sample_dir = os.path.join(working_dir, "lightrag", f"sample_{sample['id']}")
    os.makedirs(sample_dir, exist_ok=True)
    ef = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=embed_func if not isinstance(embed_func, EmbeddingFunc) else embed_func.func,
    )
    rag = LightRAG(working_dir=sample_dir, llm_model_func=llm_func, embedding_func=ef)
    await rag.initialize_storages()
    for doc in sample.get("supporting_docs", []):
        if doc.strip():
            await rag.ainsert(doc)

    prediction = await rag.aquery(
        sample["question"],
        param=QueryParam(mode="hybrid", top_k=20),
    )

    # Retrieve contexts separately for RAGAS (no built-in context return in LightRAG API)
    ctx_result = await rag.aquery(
        sample["question"],
        param=QueryParam(mode="naive", top_k=10),  # naive = pure chunk retrieval
    )
    # LightRAG doesn't expose raw passages easily — use the naive answer as proxy context
    contexts = [ctx_result] if ctx_result else []

    await rag.finalize_storages()
    shutil.rmtree(sample_dir, ignore_errors=True)

    gold_list = sample["gold_answers"]
    return {
        "id":           sample["id"],
        "question":     sample["question"],
        "gold_answers": gold_list,
        "ground_truth": gold_list[0] if gold_list else "",
        "prediction":   prediction,
        "contexts":     contexts,
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
    ragas_llm_model: Optional[str] = None,
    ragas_api_key: Optional[str] = None,
    ragas_base_url: Optional[str] = None,
) -> dict:
    """Evaluate FrameRAG (and optionally LightRAG) on a multi-hop QA dataset.

    Metrics computed (same as lightrag/evaluation/eval_rag_quality.py):
      faithfulness, answer_relevance, context_recall, context_precision, ragas_score
    """
    from .datasets import (
        load_hotpotqa, load_2wikimultihopqa, load_musique, load_chronoqa
    )
    from .metrics import compute_ragas_metrics

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

    async def _safe(coro, label: str) -> Optional[dict]:
        async with semaphore:
            try:
                return await coro
            except Exception as e:
                logger.error(f"[{label}] failed: {e}")
                return None

    # ── FrameRAG ──────────────────────────────────────────────────────────────
    logger.info("=== FrameRAG generation ===")
    tasks = [
        _safe(evaluate_framerag_sample(s, llm_func, embed_func, working_dir, embedding_dim),
              f"FrameRAG/{s['id']}")
        for s in samples
    ]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        r = await coro
        if r:
            framerag_results.append(r)
        if (i + 1) % 10 == 0:
            logger.info(f"[FrameRAG] {i+1}/{len(samples)} done")

    # ── LightRAG ──────────────────────────────────────────────────────────────
    if run_lightrag:
        logger.info("=== LightRAG generation ===")
        tasks = [
            _safe(evaluate_lightrag_sample(s, llm_func, embed_func, working_dir, embedding_dim),
                  f"LightRAG/{s['id']}")
            for s in samples
        ]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            if r:
                lightrag_results.append(r)
            if (i + 1) % 10 == 0:
                logger.info(f"[LightRAG] {i+1}/{len(samples)} done")

    # ── RAGAS scoring (synchronous — RAGAS is not async) ─────────────────────
    scoring_model = ragas_llm_model or llm_model

    def _ragas(results: list[dict]) -> dict:
        if not results:
            return {}
        return compute_ragas_metrics(
            questions     = [r["question"]     for r in results],
            answers       = [r["prediction"]   for r in results],
            contexts      = [r["contexts"]     for r in results],
            ground_truths = [r["ground_truth"] for r in results],
            llm_model     = scoring_model,
            embedding_model = embed_model,
            api_key       = ragas_api_key,
            base_url      = ragas_base_url,
        )

    logger.info("=== RAGAS scoring: FrameRAG ===")
    fr_metrics = _ragas(framerag_results)

    lr_metrics: dict = {}
    if lightrag_results:
        logger.info("=== RAGAS scoring: LightRAG ===")
        lr_metrics = _ragas(lightrag_results)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_comparison(dataset_name, fr_metrics, lr_metrics)

    output = {
        "dataset":   dataset_name,
        "split":     split,
        "FrameRAG":  {"metrics": fr_metrics, "samples": framerag_results},
        "LightRAG":  {"metrics": lr_metrics, "samples": lightrag_results},
        "reference_em_f1": E2RAG_REFERENCE.get(dataset_name, {}),
    }
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

    return output


def _print_comparison(dataset_name: str, fr_metrics: dict, lr_metrics: dict) -> None:
    sep = "─" * 75
    header = f"{'System':<22}{'Faithful':>10}{'AnsRel':>10}{'CtxRec':>10}{'CtxPrec':>10}{'RAGAS':>10}"
    print(f"\n{sep}")
    print(f"Dataset: {dataset_name}  |  Metrics: RAGAS (LLM-judged)")
    print(header)
    print(sep)

    def _row(name: str, m: dict) -> str:
        if not m:
            return f"  {name:<20}" + "         -" * 5
        return (
            f"  {name:<20}"
            f"{m.get('faithfulness', 0):>10.3f}"
            f"{m.get('answer_relevance', 0):>10.3f}"
            f"{m.get('context_recall', 0):>10.3f}"
            f"{m.get('context_precision', 0):>10.3f}"
            f"{m.get('ragas_score', 0):>10.3f}"
        )

    if lr_metrics:
        print(_row("LightRAG (ours)", lr_metrics))
    print(_row("FrameRAG (ours)", fr_metrics))
    print(sep)

    # Print reference EM/F1 from paper for context
    ref = E2RAG_REFERENCE.get(dataset_name, {})
    if ref:
        print(f"\n  Reference EM/F1 from E²RAG paper (lexical, for reference only):")
        for sys_name, vals in ref.items():
            print(f"    {sys_name:<22} EM={vals['em']:.3f}  F1={vals['f1']:.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FrameRAG vs LightRAG using RAGAS metrics"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["hotpotqa", "2wikimultihopqa", "musique", "chronoqa"],
    )
    parser.add_argument("--split",         default="validation")
    parser.add_argument("--max_samples",   type=int, default=None)
    parser.add_argument("--working_dir",   default="./eval_storage")
    parser.add_argument("--llm_model",     default="gpt-4o-mini")
    parser.add_argument("--embed_model",   default="text-embedding-3-small")
    parser.add_argument("--embedding_dim", type=int, default=1536)
    parser.add_argument("--output_file",   default=None)
    parser.add_argument("--concurrency",   type=int, default=4)
    parser.add_argument("--chronoqa_path", default=None)
    parser.add_argument("--ragas_llm_model", default=None,
                        help="LLM for RAGAS scoring (defaults to --llm_model)")
    parser.add_argument("--ragas_api_key", default=None,
                        help="API key for RAGAS scoring (defaults to OPENAI_API_KEY)")
    parser.add_argument("--ragas_base_url", default=None,
                        help="Custom endpoint for RAGAS scoring LLM")
    parser.add_argument(
        "--no_lightrag", action="store_true",
        help="Skip LightRAG evaluation (only run FrameRAG)",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        dataset_name    = args.dataset,
        split           = args.split,
        max_samples     = args.max_samples,
        working_dir     = args.working_dir,
        llm_model       = args.llm_model,
        embed_model     = args.embed_model,
        embedding_dim   = args.embedding_dim,
        output_file     = args.output_file,
        concurrency     = args.concurrency,
        chronoqa_path   = args.chronoqa_path,
        run_lightrag    = not args.no_lightrag,
        ragas_llm_model = args.ragas_llm_model,
        ragas_api_key   = args.ragas_api_key,
        ragas_base_url  = args.ragas_base_url,
    ))


if __name__ == "__main__":
    main()
