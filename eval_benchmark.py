#!/usr/bin/env python3
"""
Benchmark evaluation: none vs llm_frames on HotpotQA (multi-hop) + SQuAD (single-hop).

Metrics (all LLM-based or string-based — no RAGAS):
  - correctness_score  : GPT-4o-mini judge, 0-5 scale  (main metric, from FRAMES/G-Eval style)
  - correct_binary     : judge score >= 3 → 1 else 0    (used in FRAMES paper)
  - token_f1           : standard QA metric (EM-like, token overlap)
  - exact_match        : EM after lowercasing + punct strip

Usage:
    python eval_benchmark.py                     # both modes, skip re-index if storage exists
    python eval_benchmark.py --reindex           # delete storage and rebuild
    python eval_benchmark.py --mode none
    python eval_benchmark.py --mode llm_frames
    python eval_benchmark.py --run-single <mode> # internal subprocess call
    python eval_benchmark.py --create-dataset    # just create the dataset, then exit
    python eval_benchmark.py --n-questions 20    # use first N questions (faster test run)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env", override=False)

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR    = Path("lightrag/evaluation/benchmark_documents")
DATASET     = Path("lightrag/evaluation/benchmark_dataset.json")
RESULTS_DIR = Path("lightrag/evaluation/bench_results")
RESULTS_DIR.mkdir(exist_ok=True)

OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL   = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
QUERY_MODE  = "hybrid"

STORAGE = {
    "none":       "./rag_storage_bench_none",
    "llm_frames": "./rag_storage_bench_llm_frames",
}


# ── String metrics ────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> int:
    return int(_normalize(pred) == _normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    pt = _normalize(pred).split()
    rt = _normalize(ref).split()
    if not pt or not rt:
        return 1.0 if not pt and not rt else 0.0
    common = set(pt) & set(rt)
    p = len(common) / len(pt)
    r = len(common) / len(rt)
    return round((2 * p * r / (p + r)) if (p + r) else 0.0, 4)


# ── LLM judge ────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert QA evaluator. Given a question, a reference answer, and a predicted answer, score how correct the predicted answer is.

Question: {question}

Reference Answer: {ground_truth}

Predicted Answer: {predicted}

Scoring guide (0–5):
5 – Fully correct: predicted answer matches or correctly paraphrases the reference
4 – Mostly correct: right answer but adds minor irrelevant details or small errors
3 – Partially correct: contains the key fact but also contains significant errors or omissions
2 – Mostly wrong: one correct aspect among mostly incorrect content
1 – Incorrect but topic-relevant: wrong answer, but shows some understanding of the topic
0 – Completely wrong, off-topic, or refused to answer

For factual questions (who/what/when/where), focus on whether the key fact (name, date, number, etc.) is present and correct.
For yes/no questions, score 5 only if the polarity matches.

Return ONLY valid JSON with no markdown:
{{"score": <0-5>, "correct": <true if score>=3 else false>, "reasoning": "<one sentence>"}}"""


async def llm_judge_batch(
    cases: list[dict],
    llm_func,
    batch_size: int = 8,
) -> list[dict]:
    """Judge all cases with LLM, return list of {score, correct, reasoning}."""
    results: list[dict | None] = [None] * len(cases)

    async def judge_one(idx: int, case: dict):
        prompt = JUDGE_PROMPT.format(
            question=case["question"],
            ground_truth=case["ground_truth"],
            predicted=case["answer"],
        )
        try:
            raw = await llm_func(prompt)
            # strip markdown fences if present
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            obj = json.loads(raw)
            results[idx] = {
                "score":     int(obj.get("score", 0)),
                "correct":   bool(obj.get("correct", False)),
                "reasoning": obj.get("reasoning", ""),
            }
        except Exception as exc:
            print(f"    [judge error #{idx}] {exc}")
            results[idx] = {"score": 0, "correct": False, "reasoning": str(exc)}

    # run in batches to avoid rate limits
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        await asyncio.gather(*[judge_one(start + i, c) for i, c in enumerate(batch)])
        print(f"    judged {min(start + batch_size, len(cases))}/{len(cases)}")
        await asyncio.sleep(0.5)  # small pause between batches

    return results  # type: ignore[return-value]


# ── LLM / Embedding factory ───────────────────────────────────────────────────

def _make_llm_embed():
    import numpy as np
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, **kwargs):
        return await gpt_4o_mini_complete(prompt, api_key=OPENAI_KEY, **kwargs)

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8191)
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await openai_embed(texts, model=EMBED_MODEL, api_key=OPENAI_KEY)

    return llm_func, embed_func


def _make_rag(mode: str, llm_func, embed_func):
    from lightrag import LightRAG
    return LightRAG(
        working_dir=STORAGE[mode],
        llm_model_func=llm_func,
        embedding_func=embed_func,
        llm_model_max_async=4,
        embedding_func_max_async=4,
        max_parallel_insert=4,
    )


# ── Single-mode pipeline ──────────────────────────────────────────────────────

async def run_mode(mode: str, reindex: bool, n_questions: int | None) -> dict:
    print(f"\n{'='*70}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*70}")

    storage_path = Path(STORAGE[mode])
    if reindex and storage_path.exists():
        print(f"  [reindex] Removing {storage_path} ...")
        shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    llm_func, embed_func = _make_llm_embed()
    rag = _make_rag(mode, llm_func, embed_func)
    await rag.initialize_storages()

    # ── Indexing ──────────────────────────────────────────────────────────────
    docs = sorted(DOCS_DIR.glob("*.txt"))
    if not docs:
        print(f"  [WARN] No .txt files in {DOCS_DIR} — run create_benchmark_dataset.py first")
    else:
        # Copy LLM cache from 'none' mode to reuse entity extraction calls (same prompts)
        if mode != "none":
            none_cache = Path(STORAGE["none"]) / "kv_store_llm_response_cache.json"
            mode_cache = storage_path / "kv_store_llm_response_cache.json"
            if none_cache.exists() and not mode_cache.exists():
                import shutil as _shutil
                _shutil.copy2(none_cache, mode_cache)
                print(f"  [cache] Copied LLM cache from 'none' storage — entity extraction will be reused")

        print(f"\n  Indexing {len(docs)} documents (batch, max_parallel={4}) ...")
        t0 = time.time()
        # Batch insert all at once — LightRAG handles parallelism via max_parallel_insert
        texts = [doc.read_text(encoding="utf-8") for doc in docs]
        file_paths = [str(doc) for doc in docs]
        await rag.ainsert(texts, file_paths=file_paths)
        elapsed = time.time() - t0
        print(f"  Indexing done in {elapsed:.1f}s")

    # Graph stats
    graph_file = storage_path / "graph_chunk_entity_relation.graphml"
    n_ent = n_rel = 0
    if graph_file.exists():
        txt = graph_file.read_text(encoding="utf-8")
        n_ent = len(re.findall(r"<node ", txt))
        n_rel = len(re.findall(r"<edge ", txt))
    print(f"\n  Graph: {n_ent} nodes, {n_rel} edges")

    frame_db_file = storage_path / "frame_db.json"
    if frame_db_file.exists():
        fdb = json.loads(frame_db_file.read_text(encoding="utf-8"))
        print(f"  Frame DB: {len(fdb.get('frames', {}))} frames")

    # ── Querying ──────────────────────────────────────────────────────────────
    from lightrag import QueryParam
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    test_cases = data["test_cases"]
    if n_questions:
        test_cases = test_cases[:n_questions]

    print(f"\n  Querying {len(test_cases)} questions ...")
    query_results: list[dict] = []
    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        t0 = time.time()
        try:
            answer = await rag.aquery(q, param=QueryParam(mode=QUERY_MODE))
        except Exception as exc:
            answer = f"[ERROR] {exc}"
        elapsed_q = time.time() - t0
        print(f"    [{i}/{len(test_cases)}] ({tc['type']}) {q[:55]}... ({elapsed_q:.1f}s)")
        query_results.append({
            "id":           tc.get("id", i),
            "question":     q,
            "ground_truth": tc["ground_truth"],
            "type":         tc["type"],
            "source":       tc.get("source", ""),
            "answer":       answer,
            "elapsed_s":    round(elapsed_q, 2),
        })

    # ── String metrics ────────────────────────────────────────────────────────
    for r in query_results:
        r["em"]       = exact_match(r["answer"], r["ground_truth"])
        r["token_f1"] = token_f1(r["answer"], r["ground_truth"])

    # ── LLM judge ─────────────────────────────────────────────────────────────
    print("\n  Running LLM judge ...")
    judge_llm = llm_func  # reuse same function
    judgments = await llm_judge_batch(query_results, judge_llm)
    for r, j in zip(query_results, judgments):
        r["judge_score"]  = j["score"]
        r["judge_correct"] = j["correct"]
        r["judge_reason"] = j["reasoning"]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def avg_metrics(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {}
        return {
            "correctness_5":   round(sum(r["judge_score"]   for r in rows) / n / 5, 4),  # normalise to 0-1
            "correct_binary":  round(sum(r["judge_correct"]  for r in rows) / n, 4),
            "token_f1":        round(sum(r["token_f1"]       for r in rows) / n, 4),
            "exact_match":     round(sum(r["em"]             for r in rows) / n, 4),
            "n":               n,
        }

    multi  = [r for r in query_results if r["type"] == "multi_hop"]
    single = [r for r in query_results if r["type"] == "single_hop"]

    avg_all    = avg_metrics(query_results)
    avg_multi  = avg_metrics(multi)
    avg_single = avg_metrics(single)

    print(f"\n  Overall  ({avg_all['n']} q):  correct_binary={avg_all['correct_binary']:.3f}  "
          f"F1={avg_all['token_f1']:.3f}  judge={avg_all['correctness_5']:.3f}")
    print(f"  Multi-hop ({avg_multi.get('n',0)} q): correct_binary={avg_multi.get('correct_binary',0):.3f}  "
          f"F1={avg_multi.get('token_f1',0):.3f}")
    print(f"  Single-hop({avg_single.get('n',0)} q): correct_binary={avg_single.get('correct_binary',0):.3f}  "
          f"F1={avg_single.get('token_f1',0):.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"bench_{mode}_{ts}.json"
    output = {
        "extraction_mode":   mode,
        "query_mode":        QUERY_MODE,
        "timestamp":         datetime.now().isoformat(),
        "n_entities":        n_ent,
        "n_relations":       n_rel,
        "average_all":       avg_all,
        "average_multi_hop": avg_multi,
        "average_single_hop": avg_single,
        "results":           query_results,
    }
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_file}")

    await rag.finalize_storages()
    return output


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(all_results: dict[str, dict]):
    metrics = ["correctness_5", "correct_binary", "token_f1", "exact_match"]
    labels  = ["Correctness/5 (norm)", "Correct (binary)", "Token F1", "Exact Match"]
    modes   = list(all_results.keys())

    def row(label, key, subset):
        vals = [all_results[m][subset].get(key, 0) for m in modes]
        best = max(vals)
        parts = [f"{v:>10.4f}{'*' if v==best and len(modes)>1 else ' '}" for v in vals]
        return f"  {label:<28}" + "".join(parts)

    print(f"\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")

    header = f"  {'Metric':<28}" + "".join(f"{m:>11}" for m in modes)
    print(f"\n{header}")
    print(f"  {'-'*60}")

    for subset_label, subset_key in [("── Overall ──", "average_all"),
                                      ("── Multi-hop ──", "average_multi_hop"),
                                      ("── Single-hop ──", "average_single_hop")]:
        print(f"\n  {subset_label}")
        for label, key in zip(labels, metrics):
            print(row(label, key, subset_key))

    print(f"\n  Graph size:")
    for m, r in all_results.items():
        print(f"    {m:<14}: {r['n_entities']} nodes / {r['n_relations']} edges")

    print("\n  (* = best)")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_mode_subprocess(mode: str, reindex: bool, n_questions: int | None) -> dict | None:
    cmd = [sys.executable, __file__, "--run-single", mode]
    if reindex:
        cmd.append("--reindex")
    if n_questions:
        cmd += ["--n-questions", str(n_questions)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print(f"\n  [subprocess] Starting mode={mode} ...")
    subprocess.run(cmd, env=env)
    ts_files = sorted(
        RESULTS_DIR.glob(f"bench_{mode}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ts_files:
        print(f"  [subprocess] No result file found for mode={mode}")
        return None
    return json.loads(ts_files[-1].read_text(encoding="utf-8"))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "llm_frames", "both"], default="both")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--create-dataset", action="store_true",
                        help="Run dataset creation then exit")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Use only first N questions (quick test)")
    parser.add_argument("--run-single", metavar="MODE",
                        help="Internal: run one mode (subprocess)")
    args = parser.parse_args()

    if args.create_dataset:
        import create_benchmark_dataset
        create_benchmark_dataset.main()
        return

    if not DATASET.exists():
        print(f"[ERROR] Dataset not found: {DATASET}")
        print("  Run first: python eval_benchmark.py --create-dataset")
        sys.exit(1)

    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.txt")):
        print(f"[ERROR] No documents in {DOCS_DIR}")
        print("  Run first: python eval_benchmark.py --create-dataset")
        sys.exit(1)

    # ── Internal single-mode subprocess path ──────────────────────────────────
    if args.run_single:
        mode = args.run_single
        os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = mode
        await run_mode(mode, args.reindex, args.n_questions)
        return

    # ── Orchestrator: run modes in subprocesses ───────────────────────────────
    modes = ["none", "llm_frames"] if args.mode == "both" else [args.mode]
    all_results: dict[str, dict] = {}
    for mode in modes:
        r = run_mode_subprocess(mode, args.reindex, args.n_questions)
        if r:
            all_results[mode] = r

    if len(all_results) > 1:
        print_comparison(all_results)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
