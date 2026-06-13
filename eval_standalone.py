#!/usr/bin/env python3
"""
Standalone eval: index sample docs + query, so sánh none vs llm_frames.
Chạy trực tiếp qua Python API — không cần HTTP server.

Each mode runs in its own subprocess so LightRAG singletons/in-memory doc
tracking cannot leak between modes.

Usage:
    python eval_standalone.py                  # so sánh cả 2 mode
    python eval_standalone.py --mode none      # chỉ none
    python eval_standalone.py --mode llm_frames
    python eval_standalone.py --reindex        # xóa storage cũ trước khi index
    python eval_standalone.py --run-single <mode>  # internal: run one mode (subprocess)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env", override=False)

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR   = Path("lightrag/evaluation/sample_documents")
DATASET    = Path("lightrag/evaluation/sample_dataset.json")
RESULTS_DIR = Path("lightrag/evaluation/results")
RESULTS_DIR.mkdir(exist_ok=True)

OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL   = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
QUERY_MODE  = "hybrid"

STORAGE = {
    "none":       "./rag_storage_eval_none",
    "llm_frames": "./rag_storage_eval_llm_frames",
}


# ── LLM / Embedding functions ─────────────────────────────────────────────────

def _make_llm_embed(mode: str):
    """Build llm_func and embedding_func for the given mode."""
    import numpy as np
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, **kwargs):
        # gpt_4o_mini_complete has model hard-coded; pass api_key via kwargs
        return await gpt_4o_mini_complete(
            prompt,
            api_key=OPENAI_KEY,
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8191)
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await openai_embed(
            texts,
            model=EMBED_MODEL,
            api_key=OPENAI_KEY,
        )

    return llm_func, embed_func


# ── LightRAG instance ─────────────────────────────────────────────────────────

def _make_rag(mode: str, llm_func, embed_func):
    # env var is already set by the caller (--run-single path sets it before imports)
    from lightrag import LightRAG
    rag = LightRAG(
        working_dir=STORAGE[mode],
        llm_model_func=llm_func,
        embedding_func=embed_func,
        llm_model_max_async=2,
        embedding_func_max_async=2,
        max_parallel_insert=2,
    )
    return rag


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

async def _ragas_score(questions, answers, ground_truths, contexts_list) -> list[dict]:
    """Run RAGAS evaluation, return list of metric dicts."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_recall, context_precision,
        )
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": [[c] if isinstance(c, str) else c for c in contexts_list],
        }
        dataset = Dataset.from_dict(data)
        llm_eval = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
        emb_eval = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_KEY)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm_eval,
            embeddings=emb_eval,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        scores = []
        for _, row in df.iterrows():
            scores.append({
                "faithfulness":       round(float(row.get("faithfulness",       0) or 0), 4),
                "answer_relevance":   round(float(row.get("answer_relevancy",   0) or 0), 4),
                "context_recall":     round(float(row.get("context_recall",     0) or 0), 4),
                "context_precision":  round(float(row.get("context_precision",  0) or 0), 4),
            })
        return scores
    except Exception as exc:
        print(f"  [RAGAS ERROR] {exc}")
        return [{"faithfulness": 0, "answer_relevance": 0,
                 "context_recall": 0, "context_precision": 0}
                for _ in questions]


# ── Single-mode pipeline ──────────────────────────────────────────────────────

async def run_mode(mode: str, reindex: bool) -> dict:
    print(f"\n{'='*65}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*65}")

    storage_path = Path(STORAGE[mode])

    if reindex and storage_path.exists():
        print(f"  [reindex] Removing {storage_path} ...")
        shutil.rmtree(storage_path)

    storage_path.mkdir(parents=True, exist_ok=True)

    llm_func, embed_func = _make_llm_embed(mode)
    rag = _make_rag(mode, llm_func, embed_func)
    await rag.initialize_storages()

    # ── Indexing ──────────────────────────────────────────────────────────────
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        print(f"  [WARN] No .md files in {DOCS_DIR}")
    else:
        # Copy LLM cache from 'none' mode so llm_frames reuses entity extraction calls
        if mode != "none":
            none_cache = Path(STORAGE["none"]) / "kv_store_llm_response_cache.json"
            mode_cache = storage_path / "kv_store_llm_response_cache.json"
            if none_cache.exists() and not mode_cache.exists():
                import shutil as _shutil
                _shutil.copy2(none_cache, mode_cache)
                print(f"  [cache] Copied LLM cache from 'none' — entity extraction will be reused")

        print(f"\n  Indexing {len(docs)} documents (batch insert) ...")
        t0 = time.time()
        texts = [doc.read_text(encoding="utf-8") for doc in docs]
        file_paths = [str(doc) for doc in docs]
        await rag.ainsert(texts, file_paths=file_paths)
        elapsed_index = time.time() - t0
        print(f"  Indexing done in {elapsed_index:.1f}s")

    # Graph stats
    graph_file = storage_path / "graph_chunk_entity_relation.graphml"
    n_ent = n_rel = 0
    if graph_file.exists():
        import re
        txt = graph_file.read_text(encoding="utf-8")
        n_ent = len(re.findall(r"<node ", txt))
        n_rel = len(re.findall(r"<edge ", txt))
    print(f"\n  Graph: {n_ent} nodes, {n_rel} edges")

    frame_db_file = storage_path / "frame_db.json"
    if frame_db_file.exists():
        fdb = json.loads(frame_db_file.read_text(encoding="utf-8"))
        frame_names = list(fdb.get("frames", {}).keys())
        print(f"  Frame DB: {len(frame_names)} frames: {frame_names}")

    # ── Querying ──────────────────────────────────────────────────────────────
    from lightrag import QueryParam
    test_cases = json.loads(DATASET.read_text(encoding="utf-8"))["test_cases"]

    print(f"\n  Querying {len(test_cases)} questions ...")
    results = []
    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        gt = tc.get("ground_truth", "")
        t0 = time.time()
        try:
            answer = await rag.aquery(q, param=QueryParam(mode=QUERY_MODE))
        except Exception as exc:
            answer = f"[ERROR] {exc}"
        elapsed_q = time.time() - t0
        print(f"    [{i}/{len(test_cases)}] {q[:60]}... ({elapsed_q:.1f}s)")
        results.append({
            "test_number": i,
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "elapsed_s": round(elapsed_q, 2),
        })

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    print("\n  Running RAGAS evaluation ...")
    questions    = [r["question"] for r in results]
    answers      = [r["answer"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]
    contexts     = [[] for _ in results]  # no explicit context extraction here

    scores = await _ragas_score(questions, answers, ground_truths, contexts)
    for r, s in zip(results, scores):
        r["metrics"] = s
        r["ragas_score"] = round(
            (s["faithfulness"] + s["answer_relevance"] +
             s["context_recall"] + s["context_precision"]) / 4, 4
        )

    avg = {
        k: round(sum(r["metrics"][k] for r in results) / len(results), 4)
        for k in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    }
    avg["ragas_score"] = round(sum(r["ragas_score"] for r in results) / len(results), 4)

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"eval_results_{mode}_{ts}.json"
    output = {
        "extraction_mode": mode,
        "query_mode": QUERY_MODE,
        "timestamp": datetime.now().isoformat(),
        "n_entities": n_ent,
        "n_relations": n_rel,
        "average_metrics": avg,
        "results": results,
    }
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_file}")

    await rag.finalize_storages()
    return output


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(results: dict[str, dict]):
    print(f"\n{'='*65}")
    print("  COMPARISON")
    print(f"{'='*65}")

    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "ragas_score"]
    header = f"{'Metric':<22}" + "".join(f"{m:>14}" for m in results)
    print(f"\n  {header}")
    print(f"  {'-'*60}")

    for metric in metrics:
        row = f"  {metric:<22}"
        vals = [r["average_metrics"][metric] for r in results.values()]
        best = max(vals)
        for mode, r in results.items():
            v = r["average_metrics"][metric]
            mark = " *" if v == best and len(results) > 1 else "  "
            row += f"{v:>12.4f}{mark}"
        print(row)

    print(f"\n  (* = best for metric)")

    print(f"\n  Graph size:")
    for mode, r in results.items():
        print(f"    {mode:<14}: {r['n_entities']} nodes / {r['n_relations']} edges")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_mode_subprocess(mode: str, reindex: bool) -> dict | None:
    """Run a single mode in a fresh subprocess to avoid singleton leakage."""
    cmd = [sys.executable, __file__, "--run-single", mode]
    if reindex:
        cmd.append("--reindex")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print(f"\n  [subprocess] Starting mode={mode} ...")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"  [subprocess] mode={mode} FAILED (exit {result.returncode})")
        return None
    # Find the most recent result file for this mode
    ts_files = sorted(RESULTS_DIR.glob(f"eval_results_{mode}_*.json"),
                      key=lambda p: p.stat().st_mtime)
    if not ts_files:
        print(f"  [subprocess] No result file found for mode={mode}")
        return None
    return json.loads(ts_files[-1].read_text(encoding="utf-8"))


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "llm_frames", "both"], default="both")
    parser.add_argument("--reindex", action="store_true", help="Delete storage and re-index")
    parser.add_argument("--run-single", metavar="MODE",
                        help="Internal: run exactly one mode (called by subprocess)")
    args = parser.parse_args()

    # ── Internal single-mode path (called from subprocess) ────────────────────
    if args.run_single:
        mode = args.run_single
        os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = mode
        await run_mode(mode, args.reindex)
        return

    # ── Orchestrator path: run each mode in its own subprocess ────────────────
    modes = ["none", "llm_frames"] if args.mode == "both" else [args.mode]

    all_results = {}
    for mode in modes:
        r = run_mode_subprocess(mode, args.reindex)
        if r:
            all_results[mode] = r

    if len(all_results) > 1:
        print_comparison(all_results)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
