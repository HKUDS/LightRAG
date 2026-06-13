#!/usr/bin/env python3
"""
FRAMES benchmark evaluation: 10 diverse multi-hop questions.
Downloads questions from google/frames-benchmark, fetches Wikipedia content,
indexes into both none and llm_frames storages, then compares answers.

Run: python eval_frames_benchmark.py
"""
from __future__ import annotations
import ast, asyncio, json, os, re, sys, time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env", override=False)

OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL   = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
QUERY_MODE  = "hybrid"

STORAGE = {
    "none":       "./rag_storage_frames_none",
    "llm_frames": "./rag_storage_frames_llm_frames",
}

RESULTS_DIR = Path("lightrag/evaluation/bench_results")
RESULTS_DIR.mkdir(exist_ok=True)

WIKI_FULL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "LightRAG-eval/1.0 (research; vietcuong2k182@gmail.com)"}


# ── Wikipedia fetchers ────────────────────────────────────────────────────────

def url_to_title(url: str) -> str:
    """Extract Wikipedia page title from URL."""
    # e.g. https://en.wikipedia.org/wiki/Super_Bowl_50 -> Super_Bowl_50
    m = re.search(r"wikipedia\.org/wiki/(.+)$", url)
    if m:
        return m.group(1).split("#")[0]
    return url


def fetch_wiki_full_text(title: str) -> str:
    """Fetch full Wikipedia article text via action=query API."""
    params = {
        "action": "query",
        "titles": title.replace("_", " "),
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "format": "json",
        "redirects": 1,
    }
    try:
        r = requests.get(WIKI_FULL, params=params, headers=WIKI_HEADERS, timeout=30)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            if text:
                return text[:15000]  # Cap at 15k chars per page
        return ""
    except Exception as e:
        print(f"  [WARN] fetch_wiki_full_text({title}): {e}")
        return ""


def collect_wiki_links(row: dict) -> list[str]:
    """Collect all wikipedia_link_* fields from a FRAMES row."""
    links = []
    # Try individual fields first
    for i in range(1, 20):
        key = f"wikipedia_link_{i}" if i < 11 else "wikipedia_link_11+"
        v = row.get(key, "")
        if v and isinstance(v, str) and v.startswith("http"):
            if v not in links:
                links.append(v)
    # Also try wiki_links string repr
    raw = row.get("wiki_links", "")
    if raw and isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            for u in parsed:
                if u and u not in links:
                    links.append(u)
        except Exception:
            pass
    return links


# ── Select 10 diverse samples ─────────────────────────────────────────────────

def select_10_samples() -> list[dict]:
    """Load FRAMES benchmark and pick 10 diverse, manageable samples."""
    print("Loading google/frames-benchmark from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("google/frames-benchmark", split="test")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    print(f"Total samples: {len(ds)}")

    # Score each sample by manageability (2-6 links) and reasoning diversity
    candidates = []
    for row in ds:
        row_dict = dict(row)
        links = collect_wiki_links(row_dict)
        n_links = len(links)
        if 2 <= n_links <= 6:
            candidates.append({**row_dict, "_links": links, "_n_links": n_links})

    print(f"Manageable samples (2-6 links): {len(candidates)}")

    # Pick diverse reasoning types
    reasoning_seen: dict[str, int] = {}
    selected = []
    type_priority = [
        "Multiple constraints", "Tabular", "Numerical",
        "Temporal", "Post-processing", "Inference",
    ]

    # First pass: one per reasoning type
    for rtype in type_priority:
        for c in candidates:
            rt = str(c.get("reasoning_types", ""))
            if rtype.lower() in rt.lower() and c not in selected:
                selected.append(c)
                reasoning_seen[rtype] = reasoning_seen.get(rtype, 0) + 1
                break

    # Fill up to 10 with remaining diverse samples
    for c in candidates:
        if len(selected) >= 10:
            break
        if c not in selected:
            selected.append(c)

    selected = selected[:10]
    print(f"Selected {len(selected)} samples")
    for i, s in enumerate(selected, 1):
        rt = s.get("reasoning_types", "?")
        n = s["_n_links"]
        print(f"  [{i}] {str(rt)[:50]} | {n} links | Q: {str(s.get('Prompt',''))[:60]}...")
    return selected


# ── Index documents ────────────────────────────────────────────────────────────

async def index_pages(mode: str, samples: list[dict]) -> None:
    """Fetch Wikipedia pages for all samples and index them."""
    from lightrag import LightRAG
    import numpy as np
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, **kwargs):
        return await gpt_4o_mini_complete(prompt, api_key=OPENAI_KEY, **kwargs)

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8191)
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await openai_embed(texts, model=EMBED_MODEL, api_key=OPENAI_KEY)

    rag = LightRAG(
        working_dir=STORAGE[mode],
        llm_model_func=llm_func,
        embedding_func=embed_func,
        llm_model_max_async=2,
        embedding_func_max_async=2,
    )
    await rag.initialize_storages()

    # Collect all unique Wikipedia pages needed
    all_links: dict[str, str] = {}  # title -> url
    for s in samples:
        for url in s["_links"]:
            title = url_to_title(url)
            if title and title not in all_links:
                all_links[title] = url

    print(f"\n[{mode}] Fetching & indexing {len(all_links)} unique Wikipedia pages...")
    docs = []
    file_paths = []
    for i, (title, url) in enumerate(all_links.items(), 1):
        print(f"  [{i}/{len(all_links)}] {title}")
        text = fetch_wiki_full_text(title)
        if text:
            docs.append(text)
            file_paths.append(url)
        time.sleep(0.3)  # Polite rate limit

    if docs:
        print(f"  Indexing {len(docs)} documents into {mode}...")
        await rag.ainsert(docs, file_paths=file_paths)
        print(f"  [OK] Indexing complete.")
    else:
        print(f"  [WARN] No documents fetched!")

    await rag.finalize_storages()


# ── Query ─────────────────────────────────────────────────────────────────────

async def run_queries(mode: str, samples: list[dict]) -> list[dict]:
    from lightrag import LightRAG, QueryParam
    import numpy as np
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, **kwargs):
        return await gpt_4o_mini_complete(prompt, api_key=OPENAI_KEY, **kwargs)

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8191)
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await openai_embed(texts, model=EMBED_MODEL, api_key=OPENAI_KEY)

    rag = LightRAG(
        working_dir=STORAGE[mode],
        llm_model_func=llm_func,
        embedding_func=embed_func,
        llm_model_max_async=2,
        embedding_func_max_async=2,
    )
    await rag.initialize_storages()

    print(f"\n{'='*65}")
    print(f"  QUERYING: {mode.upper()}")
    print(f"{'='*65}")

    results = []
    for i, s in enumerate(samples, 1):
        question = str(s.get("Prompt", ""))
        t0 = time.time()
        print(f"\n  [{i}/10] {str(s.get('reasoning_types',''))[:40]}")
        print(f"  Q: {question[:80]}...")
        try:
            answer = await rag.aquery(
                question,
                param=QueryParam(mode=QUERY_MODE, top_k=60)
            )
        except Exception as e:
            answer = f"[ERROR] {e}"
        elapsed = round(time.time() - t0, 2)
        print(f"  -> {elapsed}s | {len(str(answer))} chars")
        results.append({
            "id": i,
            "question": question,
            "ground_truth": str(s.get("Answer", "")),
            "reasoning_types": str(s.get("reasoning_types", "")),
            "n_links": s.get("_n_links", 0),
            "wiki_links": s.get("_links", []),
            "answer": str(answer),
            "elapsed_s": elapsed,
        })

    await rag.finalize_storages()
    return results


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator assessing multi-hop question-answering.

Question: {question}

Ground Truth Answer: {ground_truth}

Answer 1 (none mode — standard LightRAG):
{answer1}

Answer 2 (llm_frames mode — frame-augmented LightRAG):
{answer2}

Evaluate both answers on:
1. **Correctness** (0-5): Does it match or contain the ground truth?
2. **Multi-hop reasoning** (0-5): Does it correctly chain information from multiple sources?
3. **Completeness** (0-5): Does it address all parts of the question?
4. **Accuracy** (0-5): Are facts and inferences correct?

Return ONLY valid JSON:
{{
  "answer1": {{"correctness": <0-5>, "multihop": <0-5>, "completeness": <0-5>, "accuracy": <0-5>, "total": <sum>, "summary": "<one sentence>"}},
  "answer2": {{"correctness": <0-5>, "multihop": <0-5>, "completeness": <0-5>, "accuracy": <0-5>, "total": <sum>, "summary": "<one sentence>"}},
  "winner": "<Answer 1 or Answer 2 or Tie>",
  "reasoning": "<why one is better for multi-hop reasoning>"
}}"""


async def judge_pair(question: str, ground_truth: str, a1: str, a2: str) -> dict:
    import openai
    client = openai.AsyncOpenAI(api_key=OPENAI_KEY)
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth[:500],
        answer1=a1[:2500],
        answer2=a2[:2500],
    )
    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ── Print results ─────────────────────────────────────────────────────────────

def print_results(comparisons: list[dict]):
    print(f"\n{'='*65}")
    print("  FRAMES BENCHMARK -- none vs llm_frames")
    print(f"{'='*65}")

    wins = {"Answer 1": 0, "Answer 2": 0, "Tie": 0}
    metric_totals = {
        "none":       {"correctness": 0, "multihop": 0, "completeness": 0, "accuracy": 0},
        "llm_frames": {"correctness": 0, "multihop": 0, "completeness": 0, "accuracy": 0},
    }
    valid = 0

    for c in comparisons:
        j = c.get("judgment", {})
        if "error" in j:
            print(f"\n  Q{c['id']} ERROR: {j['error']}")
            continue
        valid += 1
        a1 = j.get("answer1", {})
        a2 = j.get("answer2", {})
        winner = j.get("winner", "Tie")
        wins[winner] = wins.get(winner, 0) + 1

        for m in ["correctness", "multihop", "completeness", "accuracy"]:
            metric_totals["none"][m] += a1.get(m, 0)
            metric_totals["llm_frames"][m] += a2.get(m, 0)

        q_short = c["question"][:65].encode("ascii", "replace").decode()
        gt_short = c["ground_truth"][:40].encode("ascii", "replace").decode()
        s1 = str(a1.get("summary", ""))[:70].encode("ascii", "replace").decode()
        s2 = str(a2.get("summary", ""))[:70].encode("ascii", "replace").decode()
        reason = str(j.get("reasoning", ""))[:90].encode("ascii", "replace").decode()

        print(f"\n  Q{c['id']} [{c.get('reasoning_types','')[:35]}]")
        print(f"  Q: {q_short}...")
        print(f"  GT: {gt_short}")
        print(f"  none:       total={a1.get('total','?')} | {s1}")
        print(f"  llm_frames: total={a2.get('total','?')} | {s2}")
        print(f"  Winner: {winner} -- {reason}")

    print(f"\n{'-'*65}")
    print(f"  WINS: none={wins.get('Answer 1',0)}  llm_frames={wins.get('Answer 2',0)}  Tie={wins.get('Tie',0)}")
    if valid > 0:
        print(f"\n  AVG SCORES (out of 5) over {valid} questions:")
        print(f"  {'Metric':<16} {'none':>8} {'llm_frames':>12} {'winner':>12}")
        print(f"  {'-'*50}")
        for m in ["correctness", "multihop", "completeness", "accuracy"]:
            n = metric_totals["none"][m] / valid
            l = metric_totals["llm_frames"][m] / valid
            w = "none*" if n > l else ("llm_frames*" if l > n else "tie")
            print(f"  {m:<16} {n:>8.2f} {l:>12.2f} {w:>12}")

        none_total = sum(metric_totals["none"].values()) / valid
        llm_total  = sum(metric_totals["llm_frames"].values()) / valid
        print(f"\n  OVERALL AVG: none={none_total:.2f}  llm_frames={llm_total:.2f}")


# ── Subprocess single-mode runner ─────────────────────────────────────────────

async def run_single(mode: str, samples_file: str):
    import json
    samples = json.loads(Path(samples_file).read_text(encoding="utf-8"))
    # Re-attach _links if not present
    for s in samples:
        if "_links" not in s:
            s["_links"] = s.get("wiki_links", [])
        if "_n_links" not in s:
            s["_n_links"] = len(s["_links"])

    await index_pages(mode, samples)
    results = await run_queries(mode, samples)
    out = Path(f"_frames_{mode}_tmp.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{mode}] Results written to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("FRAMES Benchmark Evaluation -- none vs llm_frames")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"10 questions | query_mode={QUERY_MODE} | judge={LLM_MODEL}")

    # Select 10 diverse samples
    samples = select_10_samples()

    # Save samples so subprocess can load them
    samples_file = Path("_frames_samples_tmp.json")
    # Remove internal keys that aren't serializable
    samples_clean = []
    for s in samples:
        sc = {k: v for k, v in s.items() if not k.startswith("_") and not hasattr(v, "__iter__") or isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        sc["_links"] = s["_links"]
        sc["_n_links"] = s["_n_links"]
        # Clean any non-serializable values
        clean = {}
        for k, v in sc.items():
            try:
                json.dumps(v)
                clean[k] = v
            except Exception:
                clean[k] = str(v)
        samples_clean.append(clean)

    samples_file.write_text(json.dumps(samples_clean, ensure_ascii=False, indent=2), encoding="utf-8")

    # Run each mode in a subprocess
    import subprocess
    results_by_mode = {}

    for mode in ["none", "llm_frames"]:
        out_file = Path(f"_frames_{mode}_tmp.json")
        cmd = [sys.executable, __file__, "--run-single", mode, str(samples_file)]
        env = {**os.environ, "LIGHTRAG_FRAME_EXTRACTION_MODE": mode, "PYTHONIOENCODING": "utf-8"}
        print(f"\n[subprocess] Starting mode={mode}...")
        subprocess.run(cmd, env=env)
        if out_file.exists():
            results_by_mode[mode] = json.loads(out_file.read_text(encoding="utf-8"))
            out_file.unlink()
        else:
            print(f"[WARN] No output file for mode={mode}")

    samples_file.unlink(missing_ok=True)

    if len(results_by_mode) < 2:
        print("ERROR: Missing results for one or both modes")
        return

    # Judge all pairs
    print(f"\n{'='*65}")
    print("  JUDGING 10 pairs...")
    print(f"{'='*65}")

    none_map    = {r["id"]: r for r in results_by_mode["none"]}
    frames_map  = {r["id"]: r for r in results_by_mode["llm_frames"]}
    comparisons = []

    for i in range(1, 11):
        r1 = none_map.get(i, {})
        r2 = frames_map.get(i, {})
        q  = r1.get("question", r2.get("question", ""))
        gt = r1.get("ground_truth", r2.get("ground_truth", ""))
        print(f"  Judging Q{i}...")
        judgment = await judge_pair(q, gt, r1.get("answer", ""), r2.get("answer", ""))
        comparisons.append({
            "id": i,
            "question": q,
            "ground_truth": gt,
            "reasoning_types": r1.get("reasoning_types", r2.get("reasoning_types", "")),
            "n_links": r1.get("n_links", r2.get("n_links", 0)),
            "none_answer": r1.get("answer", "")[:500],
            "llm_frames_answer": r2.get("answer", "")[:500],
            "none_elapsed_s": r1.get("elapsed_s"),
            "llm_frames_elapsed_s": r2.get("elapsed_s"),
            "judgment": judgment,
        })

    print_results(comparisons)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"frames_benchmark_{ts}.json"
    out.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "n_questions": 10,
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    if "--run-single" in sys.argv:
        idx = sys.argv.index("--run-single")
        mode = sys.argv[idx + 1]
        samples_file = sys.argv[idx + 2]
        os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = mode
        asyncio.run(run_single(mode, samples_file))
    else:
        asyncio.run(main())
