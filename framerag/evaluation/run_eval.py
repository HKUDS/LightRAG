"""Evaluate FrameRAG on multi-hop QA benchmarks using RAGAS metrics.

Benchmarks (multi-hop only):
  - HotpotQA         (multi-hop, distractor setting)
  - 2WikiMultiHopQA  (multi-hop, compositional)
  - MuSiQue          (multi-hop, decomposable)

Metrics (4 RAGAS metrics, same as lightrag/evaluation/eval_rag_quality.py):
  - Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision, ragas_score

For ChronoQA evaluation use paper_eval.py (LLM-judge Likert, E2RAG protocol).

Usage:
    python -m framerag.evaluation.run_eval \\
        --dataset hotpotqa \\
        --split validation \\
        --max_samples 50 \\
        --no_lightrag \\
        --llm_model gpt-4o-mini \\
        --output_file results_ragas.json

Environment variables:
    OPENAI_API_KEY              (required for LLM + RAGAS scoring)
    EVAL_LLM_BINDING_API_KEY    (overrides OPENAI_API_KEY for RAGAS scoring)
    EVAL_LLM_BINDING_HOST       (custom endpoint for local models)
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
        _func = openai_embed.func
        async def _embed(texts: list[str], **_kwargs) -> np.ndarray:
            return await _func(texts, model=model)
        return _embed
    except (ImportError, AttributeError):
        pass
    from openai import AsyncOpenAI
    _client = AsyncOpenAI()
    async def _embed(texts: list[str], **_kwargs) -> np.ndarray:
        resp = await _client.embeddings.create(model=model, input=texts)
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
        enable_event_coref=False,
        enable_llm_coref_verify=False,
    )
    await rag.initialize()

    for i, doc in enumerate(sample.get("supporting_docs", [])):
        if doc.strip():
            await rag.ainsert(doc, source_doc=f"doc_{i}")

    prediction, contexts = await rag.aquery_with_context(sample["question"])
    await rag.finalize()
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
    ctx_result = await rag.aquery(
        sample["question"],
        param=QueryParam(mode="naive", top_k=10),
    )
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
# Full evaluation loop — RAGAS only, multi-hop datasets only
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
    run_lightrag: bool = True,
    ragas_llm_model: Optional[str] = None,
    ragas_api_key: Optional[str] = None,
    ragas_base_url: Optional[str] = None,
) -> dict:
    """Evaluate FrameRAG (and optionally LightRAG) on a multi-hop QA dataset.

    Metric: RAGAS (faithfulness, answer_relevance, context_recall,
            context_precision) — same as lightrag/evaluation/eval_rag_quality.py.

    For ChronoQA use paper_eval.py instead (LLM-judge Likert, E2RAG protocol).
    """
    from .datasets import load_hotpotqa, load_2wikimultihopqa, load_musique
    from .metrics import compute_ragas_metrics

    SUPPORTED = ["hotpotqa", "2wikimultihopqa", "musique"]
    if dataset_name not in SUPPORTED:
        raise ValueError(
            f"run_eval.py only supports multi-hop datasets: {SUPPORTED}\n"
            f"For ChronoQA use: python -m framerag.evaluation.paper_eval"
        )

    # Bootstrap LightRAG shared in-process storage (required by JsonKVStorage)
    try:
        from lightrag.kg.shared_storage import initialize_share_data
        initialize_share_data(workers=1)
    except (ImportError, AssertionError):
        pass

    loaders = {
        "hotpotqa":        lambda: load_hotpotqa(split, max_samples),
        "2wikimultihopqa": lambda: load_2wikimultihopqa(split, max_samples),
        "musique":         lambda: load_musique(split, max_samples),
    }

    samples = loaders[dataset_name]()
    if not samples:
        logger.error(f"No samples loaded for {dataset_name}/{split}")
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

    # ── RAGAS scoring ─────────────────────────────────────────────────────────
    scoring_model = ragas_llm_model or llm_model

    def _ragas(results: list[dict]) -> dict:
        if not results:
            return {}
        return compute_ragas_metrics(
            questions       = [r["question"]     for r in results],
            answers         = [r["prediction"]   for r in results],
            contexts        = [r["contexts"]     for r in results],
            ground_truths   = [r["ground_truth"] for r in results],
            llm_model       = scoring_model,
            embedding_model = embed_model,
            api_key         = ragas_api_key,
            base_url        = ragas_base_url,
        )

    logger.info("=== RAGAS scoring: FrameRAG ===")
    fr_metrics = _ragas(framerag_results)

    lr_metrics: dict = {}
    if lightrag_results:
        logger.info("=== RAGAS scoring: LightRAG ===")
        lr_metrics = _ragas(lightrag_results)

    _print_results(dataset_name, fr_metrics, lr_metrics)

    output = {
        "dataset":        dataset_name,
        "split":          split,
        "metric_family":  "ragas",
        "FrameRAG":       {"metrics": fr_metrics, "samples": framerag_results},
        "LightRAG":       {"metrics": lr_metrics, "samples": lightrag_results},
    }
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

    return output


def _print_results(dataset_name: str, fr_metrics: dict, lr_metrics: dict) -> None:
    sep = "-" * 75
    print(f"\n{sep}")
    print(f"Dataset: {dataset_name}  |  Metric: RAGAS (matches LightRAG paper)")
    print(f"  {'System':<22}{'Faithful':>10}{'AnsRel':>10}{'CtxRec':>10}{'CtxPrec':>10}{'RAGAS':>10}")
    print(sep)

    def _row(name: str, m: dict) -> str:
        if not m:
            return f"  {name:<22}" + "         -" * 5
        return (
            f"  {name:<22}"
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
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FrameRAG on multi-hop QA with RAGAS metrics. "
                    "For ChronoQA use paper_eval.py instead."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["hotpotqa", "2wikimultihopqa", "musique"],
        help="Multi-hop QA dataset. For ChronoQA use paper_eval.py.",
    )
    parser.add_argument("--split",          default="validation")
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--working_dir",    default="./eval_storage")
    parser.add_argument("--llm_model",      default="gpt-4o-mini")
    parser.add_argument("--embed_model",    default="text-embedding-3-small")
    parser.add_argument("--embedding_dim",  type=int, default=1536)
    parser.add_argument("--output_file",    default=None)
    parser.add_argument("--concurrency",    type=int, default=4)
    parser.add_argument("--ragas_llm_model", default=None,
                        help="LLM for RAGAS scoring (defaults to --llm_model)")
    parser.add_argument("--ragas_api_key",  default=None)
    parser.add_argument("--ragas_base_url", default=None)
    parser.add_argument("--no_lightrag",    action="store_true",
                        help="Skip LightRAG baseline, evaluate FrameRAG only")
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
        run_lightrag    = not args.no_lightrag,
        ragas_llm_model = args.ragas_llm_model,
        ragas_api_key   = args.ragas_api_key,
        ragas_base_url  = args.ragas_base_url,
    ))


if __name__ == "__main__":
    main()
