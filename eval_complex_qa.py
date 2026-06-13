#!/usr/bin/env python3
"""
Complex QA evaluation: 10 câu hỏi yêu cầu reasoning/synthesis sâu.
So sánh none vs llm_frames trên rag_storage_bench_none / rag_storage_bench_llm_frames.

Chạy: python eval_complex_qa.py
"""
from __future__ import annotations
import asyncio, json, os, sys, time, re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env", override=False)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL  = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM  = 1536
QUERY_MODE = "hybrid"

STORAGE = {
    "none":       "./rag_storage_bench_none",
    "llm_frames": "./rag_storage_bench_llm_frames",
}

RESULTS_DIR = Path("lightrag/evaluation/bench_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── 10 Complex QA questions ────────────────────────────────────────────────────
# Yêu cầu synthesis, causal reasoning, cross-document comparison
COMPLEX_QUESTIONS = [
    {
        "id": 1,
        "question": (
            "Compare the defensive strategies of the Denver Broncos and the Carolina Panthers "
            "in Super Bowl 50. What were the key tactical differences, and how did each team's "
            "approach affect the outcome of the game?"
        ),
        "type": "comparative_analysis",
        "requires": ["multi-doc synthesis", "causal reasoning"],
    },
    {
        "id": 2,
        "question": (
            "Analyze Peyton Manning's performance trajectory leading up to and during Super Bowl 50. "
            "How did his regular season statistics reflect his physical decline, and what does his "
            "decision to retire afterward suggest about the relationship between team success and "
            "individual performance?"
        ),
        "type": "analytical_synthesis",
        "requires": ["temporal reasoning", "inference"],
    },
    {
        "id": 3,
        "question": (
            "What role did turnovers and special teams play in determining the outcome of Super Bowl 50? "
            "Explain the chain of events involving fumbles, interceptions, and field position that "
            "shifted momentum between the two teams."
        ),
        "type": "causal_chain",
        "requires": ["event sequencing", "multi-fact synthesis"],
    },
    {
        "id": 4,
        "question": (
            "Von Miller was named Super Bowl 50 MVP. Synthesize the evidence from the game — "
            "his individual statistics, the plays he made, and the effect on the opposing offense — "
            "to explain why his performance was decisive in a way that other players' contributions were not."
        ),
        "type": "evidence_synthesis",
        "requires": ["argument construction", "comparative weighting"],
    },
    {
        "id": 5,
        "question": (
            "The 2002 FIFA World Cup saw unusual outcomes including the elimination of traditional "
            "powerhouses. Analyze the factors — team preparation, referee decisions, and emerging "
            "footballing nations — that made this tournament structurally different from prior World Cups."
        ),
        "type": "contextual_analysis",
        "requires": ["multi-cause reasoning", "historical comparison"],
    },
    {
        "id": 6,
        "question": (
            "Trace the evolution of the Kansas City Wizards across their seasons in the late 1990s. "
            "What patterns emerged in their performance, roster decisions, and competitive results "
            "that defined their identity as an MLS franchise during this period?"
        ),
        "type": "longitudinal_synthesis",
        "requires": ["cross-season comparison", "pattern recognition"],
    },
    {
        "id": 7,
        "question": (
            "How did the halftime show of Super Bowl 50 reflect broader trends in the music industry "
            "and the NFL's strategy for maximizing viewership? Consider the choice of performers, "
            "the cultural moment, and the show's reception relative to prior halftime shows."
        ),
        "type": "cultural_synthesis",
        "requires": ["cross-domain reasoning", "contextual inference"],
    },
    {
        "id": 8,
        "question": (
            "Cam Newton's performance in Super Bowl 50 has been described as a paradox: he led "
            "the Carolina Panthers to an elite regular season but struggled in the championship. "
            "Using evidence from the game, analyze what structural or situational factors best "
            "explain this contrast — was it the defense he faced, play-calling, or something else?"
        ),
        "type": "explanatory_reasoning",
        "requires": ["counter-factual thinking", "evidence weighing"],
    },
    {
        "id": 9,
        "question": (
            "The 1991 Rugby World Cup marked a turning point in the sport's global profile. "
            "Analyze how the tournament's structure, participating nations, and results reflected "
            "the state of rugby union at that time and foreshadowed its later professionalization."
        ),
        "type": "historical_synthesis",
        "requires": ["temporal reasoning", "structural analysis"],
    },
    {
        "id": 10,
        "question": (
            "Compare how two different sports franchises — one from American football and one from "
            "soccer — managed multi-season rebuilding periods. What common strategies and distinct "
            "approaches did they use, and what does this reveal about competitive team-building "
            "across different sports contexts?"
        ),
        "type": "cross_domain_comparison",
        "requires": ["cross-document synthesis", "abstract pattern recognition"],
    },
]

# ── LLM judge prompt ───────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator assessing responses to complex, reasoning-heavy questions.

Question: {question}

Answer 1 (none mode — standard LightRAG):
{answer1}

Answer 2 (llm_frames mode — frame-augmented):
{answer2}

Evaluate both answers on these criteria for COMPLEX REASONING questions:

1. **Depth** (0-5): Does the answer go beyond surface facts to provide genuine analysis?
2. **Synthesis** (0-5): Does it connect information from multiple sources/perspectives coherently?
3. **Causal reasoning** (0-5): Does it explain *why* and *how*, not just *what*?
4. **Accuracy** (0-5): Are the facts and inferences correct and supported?

Return ONLY valid JSON:
{{
  "answer1": {{"depth": <0-5>, "synthesis": <0-5>, "causal": <0-5>, "accuracy": <0-5>, "total": <sum>, "summary": "<one sentence>"}},
  "answer2": {{"depth": <0-5>, "synthesis": <0-5>, "causal": <0-5>, "accuracy": <0-5>, "total": <sum>, "summary": "<one sentence>"}},
  "winner": "<Answer 1 or Answer 2 or Tie>",
  "reasoning": "<why one is better for complex reasoning>"
}}"""


# ── LLM/embed factory ─────────────────────────────────────────────────────────
def make_llm_embed():
    import numpy as np
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, **kwargs):
        return await gpt_4o_mini_complete(prompt, api_key=OPENAI_KEY, **kwargs)

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8191)
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await openai_embed(texts, model=EMBED_MODEL, api_key=OPENAI_KEY)

    return llm_func, embed_func


def make_rag(mode: str, llm_func, embed_func):
    from lightrag import LightRAG
    return LightRAG(
        working_dir=STORAGE[mode],
        llm_model_func=llm_func,
        embedding_func=embed_func,
        llm_model_max_async=2,
        embedding_func_max_async=2,
    )


# ── Query one mode ─────────────────────────────────────────────────────────────
async def run_queries(mode: str) -> list[dict]:
    print(f"\n{'='*65}")
    print(f"  QUERYING: {mode.upper()}")
    print(f"{'='*65}")

    llm_func, embed_func = make_llm_embed()
    rag = make_rag(mode, llm_func, embed_func)
    await rag.initialize_storages()

    from lightrag import QueryParam
    results = []
    for q_data in COMPLEX_QUESTIONS:
        t0 = time.time()
        print(f"\n  [{q_data['id']}/10] {q_data['type']}")
        print(f"  Q: {q_data['question'][:80]}...")
        try:
            answer = await rag.aquery(
                q_data["question"],
                param=QueryParam(mode=QUERY_MODE, top_k=60)
            )
        except Exception as e:
            answer = f"[ERROR] {e}"
        elapsed = round(time.time() - t0, 2)
        print(f"  -> {elapsed}s | {len(str(answer))} chars")
        results.append({
            "id": q_data["id"],
            "question": q_data["question"],
            "type": q_data["type"],
            "requires": q_data["requires"],
            "answer": str(answer),
            "elapsed_s": elapsed,
        })

    await rag.finalize_storages()
    return results


# ── Judge ──────────────────────────────────────────────────────────────────────
async def judge_pair(q: str, a1: str, a2: str, llm_func) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=q,
        answer1=a1[:3000],
        answer2=a2[:3000],
    )
    try:
        raw = await llm_func(prompt)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ── Print results ─────────────────────────────────────────────────────────────
def print_results(comparisons: list[dict]):
    print(f"\n{'='*65}")
    print("  COMPLEX QA RESULTS — none vs llm_frames")
    print(f"{'='*65}")

    wins = {"Answer 1": 0, "Answer 2": 0, "Tie": 0}
    metric_totals = {"none": {"depth":0,"synthesis":0,"causal":0,"accuracy":0},
                     "llm_frames": {"depth":0,"synthesis":0,"causal":0,"accuracy":0}}
    valid = 0

    for c in comparisons:
        j = c.get("judgment", {})
        if "error" in j:
            continue
        valid += 1
        a1 = j.get("answer1", {})
        a2 = j.get("answer2", {})
        winner = j.get("winner", "Tie")
        wins[winner] = wins.get(winner, 0) + 1

        for m in ["depth","synthesis","causal","accuracy"]:
            metric_totals["none"][m] += a1.get(m, 0)
            metric_totals["llm_frames"][m] += a2.get(m, 0)

        print(f"\n  Q{c['id']} [{c['type']}]")
        print(f"  Q: {c['question'][:70]}...")
        print(f"  none:       total={a1.get('total','?')} | {a1.get('summary','')[:80]}")
        print(f"  llm_frames: total={a2.get('total','?')} | {a2.get('summary','')[:80]}")
        print(f"  Winner: {winner} — {j.get('reasoning','')[:100]}")

    print(f"\n{'-'*65}")
    print(f"  WINS: none={wins.get('Answer 1',0)}  llm_frames={wins.get('Answer 2',0)}  Tie={wins.get('Tie',0)}")
    print(f"\n  AVG SCORES (out of 5) over {valid} questions:")
    print(f"  {'Metric':<14} {'none':>8} {'llm_frames':>12} {'winner':>10}")
    print(f"  {'-'*46}")
    for m in ["depth","synthesis","causal","accuracy"]:
        n = metric_totals["none"][m] / max(valid,1)
        l = metric_totals["llm_frames"][m] / max(valid,1)
        w = "none*" if n > l else ("llm_frames*" if l > n else "tie")
        print(f"  {m:<14} {n:>8.2f} {l:>12.2f} {w:>10}")

    none_total = sum(metric_totals["none"].values()) / max(valid,1)
    llm_total  = sum(metric_totals["llm_frames"].values()) / max(valid,1)
    print(f"\n  OVERALL AVG: none={none_total:.2f}  llm_frames={llm_total:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("Complex QA Evaluation — none vs llm_frames")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"10 questions | query_mode={QUERY_MODE} | judge={LLM_MODEL}")

    # Must run modes in separate processes (FRAME_EXTRACTION_MODE is module-level const)
    import subprocess
    results_by_mode = {}

    for mode in ["none", "llm_frames"]:
        out_file = Path(f"_complex_qa_{mode}_tmp.json")
        cmd = [sys.executable, __file__, "--run-single", mode]
        env = {**os.environ, "LIGHTRAG_FRAME_EXTRACTION_MODE": mode, "PYTHONIOENCODING": "utf-8"}
        print(f"\n[subprocess] Starting mode={mode}...")
        subprocess.run(cmd, env=env)
        if out_file.exists():
            results_by_mode[mode] = json.loads(out_file.read_text(encoding="utf-8"))
            out_file.unlink()

    if len(results_by_mode) < 2:
        print("ERROR: Missing results for one or both modes")
        return

    # Judge all pairs
    print(f"\n{'='*65}")
    print("  JUDGING 10 pairs...")
    print(f"{'='*65}")

    llm_func, _ = make_llm_embed()
    comparisons = []
    none_results   = {r["id"]: r for r in results_by_mode["none"]}
    llm_results    = {r["id"]: r for r in results_by_mode["llm_frames"]}

    for q_data in COMPLEX_QUESTIONS:
        qid = q_data["id"]
        r1 = none_results.get(qid, {})
        r2 = llm_results.get(qid, {})
        print(f"  Judging Q{qid}...")
        judgment = await judge_pair(q_data["question"], r1.get("answer",""), r2.get("answer",""), llm_func)
        comparisons.append({
            "id": qid,
            "question": q_data["question"],
            "type": q_data["type"],
            "none_answer": r1.get("answer","")[:500],
            "llm_frames_answer": r2.get("answer","")[:500],
            "none_elapsed_s": r1.get("elapsed_s"),
            "llm_frames_elapsed_s": r2.get("elapsed_s"),
            "judgment": judgment,
        })

    print_results(comparisons)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"complex_qa_{ts}.json"
    out.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "n_questions": 10,
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


# ── Subprocess single-mode runner ─────────────────────────────────────────────
async def run_single(mode: str):
    results = await run_queries(mode)
    out = Path(f"_complex_qa_{mode}_tmp.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{mode}] Results written to {out}")


if __name__ == "__main__":
    if "--run-single" in sys.argv:
        mode = sys.argv[sys.argv.index("--run-single") + 1]
        os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = mode
        asyncio.run(run_single(mode))
    else:
        asyncio.run(main())
