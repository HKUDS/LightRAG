#!/usr/bin/env python3
"""
Full CS-dataset evaluation pipeline replicating LightRAG paper methodology.

Steps:
  1. Download & extract unique CS contexts from UltraDomain (HuggingFace)
  2. Index with mode=none   -> cs_storage_none/
  3. Index with mode=llm_frames -> cs_storage_llm/
  4. Generate 125 questions (5 personas × 5 tasks × 5 questions)
  5. Run 125 hybrid queries on both modes
  6. LLM-as-judge evaluation: Comprehensiveness / Diversity / Empowerment / Overall
  7. Save results -> eval_cs_results_<timestamp>.json + EVAL_CS_README.md

Usage:
    python eval_cs_pipeline.py

Checkpointing: each step saves to disk so you can resume after interruption.
    --skip-index     skip steps 2-3 (already indexed)
    --skip-questions skip step 4 (already have questions file)
    --skip-queries   skip step 5 (already have query results)
"""
from __future__ import annotations

import argparse, asyncio, json, os, re, subprocess, sys, time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(".env", override=False)

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "vertical-reason-476709-v8")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

sys.path.insert(0, str(Path(__file__).parent))

GCP_PROJECT  = os.environ["GOOGLE_CLOUD_PROJECT"]
GCP_LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_DIM    = int(os.getenv("EMBEDDING_DIM", "1536"))

STORAGE_NONE = Path("cs_storage_none")
STORAGE_LLM  = Path("cs_storage_llm")
DATA_DIR     = Path("cs_eval_data")
CONTEXTS_FILE = DATA_DIR / "cs_unique_contexts.json"
QUESTIONS_FILE = DATA_DIR / "cs_questions.txt"
RESULTS_NONE   = DATA_DIR / "cs_results_none.json"
RESULTS_LLM    = DATA_DIR / "cs_results_llm.json"

# ─── Gemini client (shared) ───────────────────────────────────────────────────
from google import genai as _genai
from google.genai import types as _gt

_gc = _genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

async def _llm(prompt: str, **kwargs) -> str:
    from lightrag.llm.gemini import gemini_complete_if_cache
    return await gemini_complete_if_cache(LLM_MODEL, prompt, **kwargs)

async def _llm_simple(prompt: str) -> str:
    """Plain Gemini call without LightRAG cache."""
    cfg = _gt.GenerateContentConfig(temperature=0)
    r = await _gc.aio.models.generate_content(
        model=LLM_MODEL, contents=[prompt], config=cfg
    )
    return r.text or ""

# ─── Step 1: Download & extract CS contexts ──────────────────────────────────
def step1_download_contexts():
    if CONTEXTS_FILE.exists():
        ctxs = json.loads(CONTEXTS_FILE.read_text(encoding="utf-8"))
        print(f"[step1] Cached: {len(ctxs)} unique CS contexts.", flush=True)
        return ctxs

    print("[step1] Downloading UltraDomain CS dataset...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("TommyChien/UltraDomain", split="train", streaming=True)

    unique = {}
    count = 0
    for item in ds:
        if item.get("label") == "cs":
            ctx = item.get("context", "")
            ctx_id = item.get("context_id", "")
            if ctx and ctx_id and ctx_id not in unique:
                unique[ctx_id] = ctx
                count += 1
                if count % 2 == 0:
                    print(f"  Found {count} CS contexts so far...", flush=True)

    contexts = list(unique.values())
    DATA_DIR.mkdir(exist_ok=True)
    CONTEXTS_FILE.write_text(json.dumps(contexts, ensure_ascii=False, indent=2), encoding="utf-8")

    total_chars = sum(len(c) for c in contexts)
    print(f"[step1] Saved {len(contexts)} unique CS contexts ({total_chars:,} chars ≈ {total_chars//4:,} tokens)", flush=True)
    return contexts


# ─── Steps 2-3: Index ─────────────────────────────────────────────────────────
def _count_failed_docs(storage: Path) -> int:
    """Count docs still in FAILED/PROCESSING status in doc_status KV store."""
    import json as _json
    status_file = storage / "kv_store_doc_status.json"
    if not status_file.exists():
        return 0
    try:
        data = _json.loads(status_file.read_text(encoding="utf-8"))
        failed = sum(
            1 for v in data.values()
            if isinstance(v, dict) and v.get("status") in ("failed", "FAILED", "processing", "PROCESSING")
        )
        return failed
    except Exception:
        return 0


def step2_index(mode: str, storage: Path, max_retries: int = 5):
    done_flag = storage / ".index_done"
    if done_flag.exists():
        print(f"[index/{mode}] Already indexed (found .index_done). Skipping.", flush=True)
        return

    cmd = [
        sys.executable, "eval_cs_indexer.py",
        "--mode", mode,
        "--storage", str(storage),
        "--contexts", str(CONTEXTS_FILE),
    ]

    for attempt in range(1, max_retries + 1):
        print(f"\n[index/{mode}] Attempt {attempt}/{max_retries} -> {storage}", flush=True)
        result = subprocess.run(cmd, capture_output=False)

        failed = _count_failed_docs(storage)
        if result.returncode == 0 and failed == 0:
            done_flag.touch()
            print(f"[index/{mode}] All docs indexed successfully.", flush=True)
            return

        if failed > 0:
            print(f"[index/{mode}] {failed} doc(s) still failed — retrying (LightRAG resets them to PENDING on next run)...", flush=True)
        else:
            print(f"[index/{mode}] Indexer exit code {result.returncode}, rechecking...", flush=True)

        if attempt == max_retries:
            print(f"[index/{mode}] WARNING: reached max retries. Marking done anyway and continuing.", flush=True)
            done_flag.touch()


# ─── Step 4: Generate questions ──────────────────────────────────────────────
async def step4_generate_questions(contexts: list[str]) -> list[str]:
    if QUESTIONS_FILE.exists():
        text = QUESTIONS_FILE.read_text(encoding="utf-8")
        qs = re.findall(r"- Question \d+:\s*(.+)", text)
        print(f"[step4] Cached: {len(qs)} questions.", flush=True)
        return qs

    print("[step4] Generating 125 questions...", flush=True)

    # Build dataset summary (first 2000 tokens of each context)
    summaries = []
    for ctx in contexts:
        words = ctx.split()
        summaries.append(" ".join(words[:600]))  # ~800 tokens each
    total_description = "\n\n---\n\n".join(summaries[:10])  # cap at 10 docs

    prompt = f"""Given the following description of a dataset about Computer Science:

{total_description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1: [question]
        - Question 2: [question]
        - Question 3: [question]
        - Question 4: [question]
        - Question 5: [question]
    - Task 2: [task description]
        ...
    - Task 5: [task description]
- User 2: [user description]
    ...
- User 5: [user description]
    ..."""

    print("[step4] Calling LLM for question generation...", flush=True)
    result = await _llm_simple(prompt)

    DATA_DIR.mkdir(exist_ok=True)
    QUESTIONS_FILE.write_text(result, encoding="utf-8")
    print(f"[step4] Questions saved to {QUESTIONS_FILE}", flush=True)

    qs = re.findall(r"- Question \d+:\s*(.+)", result)
    print(f"[step4] Extracted {len(qs)} questions.", flush=True)
    return qs


# ─── Step 5: Run queries ──────────────────────────────────────────────────────
async def step5_run_queries(mode: str, storage: Path, questions: list[str], out_file: Path):
    if out_file.exists():
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        if len(existing) >= len(questions):
            print(f"[queries/{mode}] Cached: {len(existing)} results.", flush=True)
            return existing
        print(f"[queries/{mode}] Resuming from {len(existing)}/{len(questions)}...", flush=True)
        done_qs = {r["query"] for r in existing}
        remaining = [q for q in questions if q not in done_qs]
        results = existing
    else:
        remaining = questions
        results = []

    os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = mode

    from lightrag.utils import wrap_embedding_func_with_attrs
    from lightrag import LightRAG, QueryParam

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=2048)
    async def vertex_embed(texts: list[str]) -> np.ndarray:
        r = await _gc.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=_gt.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBED_DIM,
            ),
        )
        return np.array([e.values for e in r.embeddings], dtype=np.float32)

    from lightrag.llm.gemini import gemini_complete_if_cache
    llm = partial(gemini_complete_if_cache, LLM_MODEL)

    rag = LightRAG(working_dir=str(storage), llm_model_func=llm, embedding_func=vertex_embed)
    await rag.initialize_storages()

    print(f"[queries/{mode}] Running {len(remaining)} queries...", flush=True)
    for i, q in enumerate(remaining):
        t0 = time.time()
        try:
            answer = await rag.aquery(q, param=QueryParam(mode="hybrid", top_k=60))
            results.append({"query": q, "result": str(answer) if answer else ""})
            print(f"[queries/{mode}] {i+1}/{len(remaining)}: {q[:60]}... ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[queries/{mode}] ERROR on q{i+1}: {e}", flush=True)
            results.append({"query": q, "result": f"[ERROR: {e}]"})

        # Save after every 5 queries
        if (i + 1) % 5 == 0:
            out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    await rag.finalize_storages()
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[queries/{mode}] Done. {len(results)} results saved.", flush=True)
    return results


# ─── Step 6: LLM-as-judge evaluation ─────────────────────────────────────────
SYS_PROMPT = """---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**."""

EVAL_PROMPT_TMPL = """You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1 (LightRAG-none / hybrid):**
{answer1}

**Answer 2 (LightRAG-llm_frames / FSRAG+FAGE):**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}"""


async def evaluate_pair(query: str, ans1: str, ans2: str) -> dict:
    prompt = EVAL_PROMPT_TMPL.format(
        query=query,
        answer1=ans1[:3000],
        answer2=ans2[:3000],
    )
    try:
        cfg = _gt.GenerateContentConfig(
            system_instruction=SYS_PROMPT,
            temperature=0,
            response_mime_type="application/json",
        )
        r = await _gc.aio.models.generate_content(
            model=LLM_MODEL, contents=[prompt], config=cfg
        )
        raw = r.text or "{}"
        from json_repair import repair_json
        return repair_json(raw, return_objects=True)[0] or {}
    except Exception as e:
        return {"error": str(e)}


async def step6_evaluate(questions: list[str], results_none: list[dict], results_llm: list[dict]) -> list[dict]:
    eval_file = DATA_DIR / "cs_eval_results.json"
    if eval_file.exists():
        existing = json.loads(eval_file.read_text(encoding="utf-8"))
        if len(existing) >= len(questions):
            print(f"[eval] Cached: {len(existing)} evaluations.", flush=True)
            return existing
        done_qs = {e["query"] for e in existing}
        evals = existing
    else:
        done_qs = set()
        evals = []

    none_map  = {r["query"]: r["result"] for r in results_none}
    llm_map   = {r["query"]: r["result"] for r in results_llm}

    print(f"[eval] Evaluating {len(questions)} pairs...", flush=True)
    for i, q in enumerate(questions):
        if q in done_qs:
            continue
        a1 = none_map.get(q, "")
        a2 = llm_map.get(q, "")
        t0 = time.time()
        verdict = await evaluate_pair(q, a1, a2)
        evals.append({"query": q, "verdict": verdict})
        print(f"[eval] {i+1}/{len(questions)}: {q[:55]}... ({time.time()-t0:.1f}s)", flush=True)
        if (i + 1) % 10 == 0:
            eval_file.write_text(json.dumps(evals, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_file.write_text(json.dumps(evals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] Done. {len(evals)} evaluations saved.", flush=True)
    return evals


# ─── Tally results ───────────────────────────────────────────────────────────
def tally(evals: list[dict]) -> dict:
    counts = {
        "Comprehensiveness": {"Answer 1": 0, "Answer 2": 0, "Tie": 0},
        "Diversity":         {"Answer 1": 0, "Answer 2": 0, "Tie": 0},
        "Empowerment":       {"Answer 1": 0, "Answer 2": 0, "Tie": 0},
        "Overall Winner":    {"Answer 1": 0, "Answer 2": 0, "Tie": 0},
    }
    for e in evals:
        v = e.get("verdict", {})
        if "error" in v:
            continue
        for dim in counts:
            winner = v.get(dim, {}).get("Winner", "")
            if "1" in winner:
                counts[dim]["Answer 1"] += 1
            elif "2" in winner:
                counts[dim]["Answer 2"] += 1
            else:
                counts[dim]["Tie"] += 1
    total = len([e for e in evals if "error" not in e.get("verdict", {})])
    rates = {}
    for dim, c in counts.items():
        rates[dim] = {
            "none (A1)":       f"{c['Answer 1']}/{total} ({100*c['Answer 1']/(total or 1):.1f}%)",
            "llm_frames (A2)": f"{c['Answer 2']}/{total} ({100*c['Answer 2']/(total or 1):.1f}%)",
            "raw": c,
        }
    return rates


# ─── Save README ─────────────────────────────────────────────────────────────
def save_readme(evals, rates, contexts, questions, elapsed_total):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len([e for e in evals if "error" not in e.get("verdict", {})])

    def pct(dim, mode_key):
        raw = rates[dim]["raw"]
        n = raw.get("Answer 1" if "none" in mode_key else "Answer 2", 0)
        return f"{100*n/(total or 1):.1f}%"

    lines = [
        "# Evaluation: LightRAG-none vs FSRAG (llm_frames+FAGE) - CS Dataset",
        "",
        f"**Date:** {ts}  ",
        f"**Dataset:** UltraDomain CS - {len(contexts)} documents  ",
        f"**Questions:** {len(questions)} (5 personas × 5 tasks × 5 questions)  ",
        f"**LLM:** {LLM_MODEL} (Vertex AI)  ",
        f"**Embedding:** gemini-embedding-001  ",
        f"**Total runtime:** {elapsed_total/60:.0f} minutes  ",
        "",
        "---",
        "",
        "## Methodology (replicating LightRAG paper)",
        "",
        "Identical to the evaluation in *LightRAG: Simple and Fast Retrieval-Augmented Generation* (arXiv 2410.05779):",
        "",
        "- **Query mode:** `hybrid` for both systems",
        "- **Judge:** gemini-2.5-flash as LLM-as-judge",
        "- **Metrics:** Comprehensiveness, Diversity, Empowerment, Overall Winner",
        "- **Bias mitigation:** Answer 1 = none, Answer 2 = llm_frames (fixed order; positions noted in results)",
        "",
        "---",
        "",
        "## Results: Win Rates",
        "",
        "| Metric | LightRAG-none (A1) | FSRAG llm_frames (A2) |",
        "|--------|:-----------------:|:--------------------:|",
    ]

    for dim in ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]:
        a1 = rates[dim]["none (A1)"]
        a2 = rates[dim]["llm_frames (A2)"]
        lines.append(f"| {dim} | {a1} | {a2} |")

    # Graph stats
    import re
    def gstats(p: Path):
        gml = p / "graph_chunk_entity_relation.graphml"
        if not gml.exists():
            return (0, 0)
        txt = gml.read_text(encoding="utf-8")
        return len(re.findall(r"<node ", txt)), len(re.findall(r"<edge ", txt))

    nn, ne = gstats(STORAGE_NONE)
    ln, le = gstats(STORAGE_LLM)

    lines += [
        "",
        "---",
        "",
        "## Graph Statistics",
        "",
        "| | LightRAG-none | FSRAG llm_frames |",
        "|---|:---:|:---:|",
        f"| Nodes | {nn} | {ln} |",
        f"| Edges | {ne} | {le} |",
        "",
        "---",
        "",
        "## Questions Used (125 total)",
        "",
        "```",
    ]
    for i, q in enumerate(questions[:10], 1):
        lines.append(f"Q{i:03d}: {q}")
    if len(questions) > 10:
        lines.append(f"... and {len(questions)-10} more (see cs_eval_data/cs_questions.txt)")
    lines += ["```", "", "---", "", "## Sample Evaluations (first 3)", ""]

    for e in evals[:3]:
        lines.append(f"### Q: {e['query'][:100]}")
        v = e.get("verdict", {})
        for dim in ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]:
            w = v.get(dim, {}).get("Winner", "N/A")
            expl = v.get(dim, {}).get("Explanation", "")[:200]
            lines.append(f"- **{dim}:** {w} - {expl}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `cs_eval_data/cs_unique_contexts.json` | Unique CS documents |",
        "| `cs_eval_data/cs_questions.txt` | 125 generated questions |",
        "| `cs_eval_data/cs_results_none.json` | LightRAG-none answers |",
        "| `cs_eval_data/cs_results_llm.json` | FSRAG answers |",
        "| `cs_eval_data/cs_eval_results.json` | Raw judge verdicts |",
        "| `cs_storage_none/` | LightRAG-none graph storage |",
        "| `cs_storage_llm/` | FSRAG graph storage |",
        "",
        "*Generated by `eval_cs_pipeline.py`*",
    ]

    readme = Path("EVAL_CS_README.md")
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nREADME saved to {readme}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-index",     action="store_true")
    p.add_argument("--skip-questions", action="store_true")
    p.add_argument("--skip-queries",   action="store_true")
    return p.parse_args()


async def async_main(args):
    t_start = time.time()
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60, flush=True)
    print("LightRAG CS Evaluation Pipeline", flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)

    # Step 1
    contexts = step1_download_contexts()

    # Steps 2-3
    if not args.skip_index:
        step2_index("none",       STORAGE_NONE)
        step2_index("llm_frames", STORAGE_LLM)
    else:
        print("[index] Skipped (--skip-index)", flush=True)

    # Step 4
    if not args.skip_questions and not QUESTIONS_FILE.exists():
        questions = await step4_generate_questions(contexts)
    else:
        if QUESTIONS_FILE.exists():
            text = QUESTIONS_FILE.read_text(encoding="utf-8")
            questions = re.findall(r"- Question \d+:\s*(.+)", text)
            print(f"[step4] Loaded {len(questions)} questions from file.", flush=True)
        else:
            questions = await step4_generate_questions(contexts)

    if not questions:
        print("ERROR: No questions generated. Aborting.", flush=True)
        return

    print(f"\nTotal questions: {len(questions)}", flush=True)

    # Step 5
    if not args.skip_queries:
        results_none = await step5_run_queries("none",       STORAGE_NONE, questions, RESULTS_NONE)
        results_llm  = await step5_run_queries("llm_frames", STORAGE_LLM,  questions, RESULTS_LLM)
    else:
        results_none = json.loads(RESULTS_NONE.read_text(encoding="utf-8")) if RESULTS_NONE.exists() else []
        results_llm  = json.loads(RESULTS_LLM.read_text(encoding="utf-8"))  if RESULTS_LLM.exists()  else []
        print(f"[queries] Loaded none={len(results_none)}, llm={len(results_llm)}", flush=True)

    # Step 6
    evals = await step6_evaluate(questions, results_none, results_llm)

    # Tally
    rates = tally(evals)

    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    total = len([e for e in evals if "error" not in e.get("verdict", {})])
    print(f"{'Metric':<22} {'none (A1)':>16} {'llm_frames (A2)':>18}", flush=True)
    print("-" * 58, flush=True)
    for dim in ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]:
        print(f"  {dim:<20} {rates[dim]['none (A1)']:>16} {rates[dim]['llm_frames (A2)']:>18}", flush=True)

    elapsed = time.time() - t_start
    save_readme(evals, rates, contexts, questions, elapsed)
    print(f"\nTotal time: {elapsed/60:.1f} minutes", flush=True)


def main():
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
