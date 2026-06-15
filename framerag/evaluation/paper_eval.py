"""Evaluate FrameRAG on ChronoQA using LLM-judge Likert scoring (E2RAG protocol).

Matches E2RAG paper (arXiv 2506.05939, Table 3):
  - 3 independent LLM judges, score 1-10 per answer
  - Score = (1/J) * sum_j[ (1/N) * sum_i s_ij ]
  - Results broken down by story and by reasoning facet

Reference scores from paper (Table 3, overall average):
  E2RAG (comb. extraction)   7.1257
  LightRAG hybrid            6.8804
  Vanilla RAG                6.6022

For multi-hop QA (HotpotQA / 2WikiMHQA / MuSiQue) use run_eval.py instead
(RAGAS metrics, matches LightRAG paper).

Usage:
    python -m framerag.evaluation.paper_eval \\
        --story_ids 1,2,5,6,7,8,9 \\
        --no_lightrag \\
        --llm_model gpt-4o-mini \\
        --judge_models gpt-4o-mini \\
        --output_file results_chronoqa.json

Note: Stories 3 and 4 (Harry Potter) have no excerpt text due to copyright
and are skipped automatically.

Environment variables:
    OPENAI_API_KEY              (required)
    EVAL_LLM_BINDING_API_KEY    (overrides OPENAI_API_KEY)
    EVAL_LLM_BINDING_HOST       (custom endpoint)
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
# Reference scores from E2RAG paper Table 3 (LLM-judge 1-10, 9 stories)
# ─────────────────────────────────────────────────────────────────────────────

E2RAG_REFERENCE = {
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

# Stories with available text (HP stories 3 & 4 omitted — copyright)
AVAILABLE_STORY_IDS = ["1", "2", "5", "6", "7", "8", "9"]

DEFAULT_JUDGE_MODELS = ["gpt-4.1-mini"]

# ─────────────────────────────────────────────────────────────────────────────
# LLM judge rubric (matches E2RAG paper Appendix D.2 spirit)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_RUBRIC = """\
You are an expert evaluator for question-answering systems.

Given:
- QUESTION: the question asked
- GROUND TRUTH: the reference correct answer
- SYSTEM RESPONSE: the answer produced by the RAG system under evaluation

Score the SYSTEM RESPONSE on a 1-10 Likert scale:
  10 : Perfectly accurate and comprehensive; matches the ground truth fully.
  8-9: Very accurate; captures all key information with minor gaps.
  6-7: Mostly correct; addresses the main points but has noticeable gaps.
  4-5: Partially correct; some relevant content but significant inaccuracies.
  2-3: Mostly incorrect or irrelevant; only superficial overlap with ground truth.
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

SYSTEM RESPONSE: {response}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM / Embed factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm(model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    key = api_key or os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("EVAL_LLM_BINDING_HOST")
    from openai import AsyncOpenAI
    kwargs: dict = {}
    if key:
        kwargs["api_key"] = key
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


def _make_embed(model: str = "text-embedding-3-small",
                api_key: Optional[str] = None, base_url: Optional[str] = None):
    import numpy as np
    key = api_key or os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("EVAL_LLM_BINDING_HOST")
    from openai import AsyncOpenAI
    kwargs: dict = {}
    if key:
        kwargs["api_key"] = key
    if url:
        kwargs["base_url"] = url
    client = AsyncOpenAI(**kwargs)

    async def _embed(texts: list[str], **_kw) -> np.ndarray:
        resp = await client.embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    return _embed


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge
# ─────────────────────────────────────────────────────────────────────────────

async def _judge_one(
    question: str,
    ground_truth: str,
    response: str,
    judge_llm,
) -> tuple[int, str]:
    """Score one (question, ground_truth, response) with one judge. Returns (score, reason)."""
    prompt = JUDGE_RUBRIC.format(
        question=question,
        ground_truth=ground_truth,
        response=response or "(no answer provided)",
    )
    try:
        raw = await judge_llm(prompt)
        raw = raw.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            data   = json.loads(raw[start:end])
            score  = int(data.get("score", 5))
            reason = str(data.get("reason", ""))
            return max(1, min(10, score)), reason
    except Exception as e:
        logger.warning(f"Judge parse error: {e} | raw: {repr(raw[:200]) if 'raw' in dir() else '?'}")
    return 5, "parse error"


async def _judge_response(
    question: str,
    ground_truth: str,
    response: str,
    judge_llms: list,
    sem: asyncio.Semaphore,
) -> dict:
    """Score with all judges concurrently. Returns avg score and per-judge details."""
    async def _one(j_llm):
        async with sem:
            score, reason = await _judge_one(question, ground_truth, response, j_llm)
            return {"score": score, "reason": reason}

    results = await asyncio.gather(*[_one(j) for j in judge_llms])
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"avg_score": avg, "judges": list(results)}


# ─────────────────────────────────────────────────────────────────────────────
# ChronoQA corpus helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_story(samples: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[str(s.get("story_id", s["id"]))].append(s)
    return dict(groups)


def _collect_story_corpus(story_samples: list[dict]) -> list[str]:
    """Build a single ordered document from all story passages, then return as [doc].

    Strategy:
      1. Collect (start_byte, end_byte, excerpt) from passages_raw across all QA pairs.
      2. Sort by start_byte so passage order matches the original novel.
      3. Merge overlapping / duplicate spans.
      4. Concatenate with a separator for any gaps, producing ONE big document.
      5. Fallback: if no byte info, fall back to the old per-excerpt list.

    Returning a single-element list lets the rest of the pipeline call ainsert once,
    so the chunker sees the full story and entity coref / causal edges can span passages.
    """
    # Collect spans with byte offsets from passages_raw
    spans: list[tuple[int, int, str]] = []  # (start_byte, end_byte, excerpt)
    for s in story_samples:
        for p in s.get("passages_raw") or []:
            if not isinstance(p, dict):
                continue
            excerpt = (p.get("excerpt") or "").strip()
            start = p.get("start_byte")
            end = p.get("end_byte")
            if excerpt and start is not None and end is not None:
                spans.append((int(start), int(end), excerpt))

    if not spans:
        # Fallback: no byte info — return deduplicated excerpts as before
        seen: set[str] = set()
        docs: list[str] = []
        for s in story_samples:
            for doc in s.get("all_excerpts") or s.get("supporting_docs", []):
                if doc and doc not in seen:
                    seen.add(doc)
                    docs.append(doc)
        return docs

    # Sort by start byte, then merge overlapping spans
    spans.sort(key=lambda x: x[0])
    merged: list[tuple[int, int, str]] = []
    for start, end, text in spans:
        if merged and start <= merged[-1][1]:
            # Overlapping — keep whichever end is further
            prev_start, prev_end, prev_text = merged[-1]
            if end > prev_end:
                # Extend: use the longer text
                merged[-1] = (prev_start, end, prev_text if len(prev_text) >= len(text) else text)
        else:
            merged.append((start, end, text))

    # Concatenate with gap marker between non-adjacent spans
    parts: list[str] = []
    for i, (start, end, text) in enumerate(merged):
        if i > 0:
            prev_end = merged[i - 1][1]
            if start > prev_end + 200:   # gap > ~200 bytes → mark the omission
                parts.append("\n\n[...]\n\n")
            else:
                parts.append("\n\n")
        parts.append(text.strip())

    full_doc = "".join(parts)
    return [full_doc]


# ─────────────────────────────────────────────────────────────────────────────
# Per-story RAG runners
# ─────────────────────────────────────────────────────────────────────────────

async def _run_story_framerag(
    story_id: str,
    story_samples: list[dict],
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int,
) -> list[dict]:
    """Index a story ONCE with FrameRAG, answer all its questions."""
    from framerag import FrameRAG

    corpus = _collect_story_corpus(story_samples)
    if not corpus:
        story_title = story_samples[0].get("story_title", story_id)
        logger.warning(
            f"[ChronoQA/FrameRAG] Story {story_id} '{story_title}' has no excerpt text "
            f"(likely Harry Potter — copyright). Skipping."
        )
        return [{**s, "response": "", "system": "FrameRAG"} for s in story_samples]

    story_title = story_samples[0].get("story_title", story_id)
    story_dir   = os.path.join(working_dir, "framerag", f"story_{story_id}")

    rag = FrameRAG(
        working_dir=story_dir,
        llm_func=llm_func,
        embed_func=embed_func,
        embedding_dim=embedding_dim,
        enable_llm_coref_verify=False,
    )
    await rag.initialize()

    logger.info(
        f"[ChronoQA/FrameRAG] Indexing story {story_id} '{story_title}': "
        f"{len(corpus)} excerpts, {len(story_samples)} questions"
    )
    # Document-level inserts are kept sequential (concurrency=1): each ainsert
    # mutates shared in-memory hypergraph adjacency / frame-DB state without a
    # lock, so parallel documents would race. Intra-document parallelism (all
    # chunks extracted concurrently, frame annotation parallel per event) already
    # provides the speedup.
    docs = [(i, d) for i, d in enumerate(corpus) if d.strip()]
    await rag.ainsert_batch(
        texts=[d for _, d in docs],
        source_docs=[f"story_{story_id}_p{i}" for i, _ in docs],
        concurrency=1,
    )

    # Parallel queries — hypergraph is read-only at this stage
    query_sem = asyncio.Semaphore(4)

    async def _safe_query(sample: dict) -> dict:
        async with query_sem:
            try:
                answer = await rag.aquery(sample["question"])
            except Exception as e:
                logger.error(f"[ChronoQA/FrameRAG] Query failed for {sample['id']}: {e}")
                answer = ""
        return {**sample, "response": answer, "system": "FrameRAG"}

    results = list(await asyncio.gather(*[_safe_query(s) for s in story_samples]))

    await rag.finalize()
    return results


async def _run_story_lightrag(
    story_id: str,
    story_samples: list[dict],
    llm_func,
    embed_func,
    working_dir: str,
    embedding_dim: int,
) -> list[dict]:
    """Index a story ONCE with LightRAG, answer all its questions."""
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        logger.warning("LightRAG not importable; skipping")
        return [{**s, "response": "", "system": "LightRAG"} for s in story_samples]

    corpus = _collect_story_corpus(story_samples)
    if not corpus:
        story_title = story_samples[0].get("story_title", story_id)
        logger.warning(
            f"[ChronoQA/LightRAG] Story {story_id} '{story_title}' has no excerpt text. Skipping."
        )
        return [{**s, "response": "", "system": "LightRAG"} for s in story_samples]

    story_title = story_samples[0].get("story_title", story_id)
    story_dir   = os.path.join(working_dir, "lightrag", f"story_{story_id}")
    os.makedirs(story_dir, exist_ok=True)

    raw_embed = embed_func.func if hasattr(embed_func, "func") else embed_func
    ef = EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=raw_embed)
    rag = LightRAG(working_dir=story_dir, llm_model_func=llm_func, embedding_func=ef)
    await rag.initialize_storages()

    logger.info(
        f"[ChronoQA/LightRAG] Indexing story {story_id} '{story_title}': "
        f"{len(corpus)} excerpts, {len(story_samples)} questions"
    )
    for doc in corpus:
        if doc.strip():
            await rag.ainsert(doc)

    results = []
    for sample in story_samples:
        try:
            answer = await rag.aquery(
                sample["question"], param=QueryParam(mode="hybrid", top_k=20)
            )
        except Exception as e:
            logger.error(f"[ChronoQA/LightRAG] Query failed for {sample['id']}: {e}")
            answer = ""
        results.append({**sample, "response": answer, "system": "LightRAG"})

    await rag.finalize_storages()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(results: list[dict]) -> dict:
    """Compute avg Likert score overall and by story/facet."""
    if not results:
        return {"avg_score": 0.0, "n": 0, "by_story": {}, "by_facet": {}}

    from collections import defaultdict
    avg = sum(r["avg_score"] for r in results) / len(results)

    by_story: dict[str, list[float]] = defaultdict(list)
    by_facet: dict[str, list[float]] = defaultdict(list)
    for r in results:
        sid   = str(r.get("story_id", ""))
        title = r.get("story_title", sid)
        key   = f"{sid}:{title}" if sid else "unknown"
        by_story[key].append(r["avg_score"])

        facet = r.get("category") or r.get("type") or "general"
        by_facet[facet].append(r["avg_score"])

    return {
        "avg_score": avg,
        "n":         len(results),
        "by_story":  {k: round(sum(v) / len(v), 4) for k, v in by_story.items()},
        "by_facet":  {k: round(sum(v) / len(v), 4) for k, v in by_facet.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_paper_evaluation(
    split: str = "train",
    max_samples: Optional[int] = None,
    story_ids: Optional[list[str]] = None,
    working_dir: str = "./eval_storage",
    llm_model: str = "gpt-4.1-mini",
    embed_model: str = "text-embedding-3-small",
    embedding_dim: int = 1536,
    judge_models: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    output_file: Optional[str] = None,
    concurrency: int = 1,
    judge_concurrency: int = 8,
    run_lightrag: bool = False,
) -> dict:
    """Evaluate FrameRAG on ChronoQA with LLM-judge Likert scoring (E2RAG protocol).

    Args:
        split:             Dataset split ('train' — ChronoQA has only one split).
        max_samples:       Limit total QA pairs loaded (None = all).
        story_ids:         Story IDs to evaluate (default: AVAILABLE_STORY_IDS).
                           Stories 3 and 4 (Harry Potter) have no text and are
                           automatically skipped even if included.
        working_dir:       Temp directory for per-story RAG indexes.
        llm_model:         LLM used by FrameRAG/LightRAG for answering.
        embed_model:       Embedding model.
        embedding_dim:     Embedding dimensionality.
        judge_models:      Judge LLM IDs (paper used 3 judges).
        api_key:           OpenAI-compatible API key.
        base_url:          Custom endpoint.
        output_file:       Path to save JSON results.
        concurrency:       Max parallel story evaluations (keep at 1 to avoid
                           JsonKVStorage namespace conflicts).
        judge_concurrency: Max parallel judge LLM calls.
        run_lightrag:      Also evaluate LightRAG as baseline.
    """
    from .datasets import load_chronoqa

    # Bootstrap LightRAG shared in-process storage (required by JsonKVStorage)
    try:
        from lightrag.kg.shared_storage import initialize_share_data
        initialize_share_data(workers=1)
    except (ImportError, AssertionError):
        pass

    judge_models = judge_models or DEFAULT_JUDGE_MODELS
    target_ids   = [str(s) for s in (story_ids or AVAILABLE_STORY_IDS)]

    samples = load_chronoqa(split=split, max_samples=max_samples)
    if not samples:
        logger.error("No ChronoQA samples loaded")
        return {}

    # Filter to requested stories only
    samples = [s for s in samples if str(s.get("story_id", "")) in target_ids]
    if not samples:
        logger.error(f"No samples found for story_ids={target_ids}")
        return {}

    story_groups = _group_by_story(samples)
    logger.info(
        f"ChronoQA eval: {len(story_groups)} stories, "
        f"{len(samples)} questions, {len(judge_models)} judges"
    )

    llm_func   = _make_llm(llm_model, api_key, base_url)
    embed_func = _make_embed(embed_model, api_key, base_url)
    judge_llms = [_make_llm(m, api_key, base_url) for m in judge_models]
    judge_sem  = asyncio.Semaphore(judge_concurrency)
    rag_sem    = asyncio.Semaphore(concurrency)
    os.makedirs(working_dir, exist_ok=True)

    # ── Generate answers (story by story) ────────────────────────────────────
    framerag_raw: list[dict] = []
    lightrag_raw: list[dict] = []

    async def _safe_story_fr(sid: str, grp: list[dict]) -> list[dict]:
        async with rag_sem:
            try:
                return await _run_story_framerag(
                    sid, grp, llm_func, embed_func, working_dir, embedding_dim)
            except Exception as e:
                logger.error(f"[FrameRAG/story {sid}] failed: {e}")
                return [{**s, "response": "", "system": "FrameRAG"} for s in grp]

    logger.info("=== FrameRAG: generating answers ===")
    for coro in asyncio.as_completed(
        [_safe_story_fr(sid, grp) for sid, grp in story_groups.items()]
    ):
        framerag_raw.extend(await coro)

    if run_lightrag:
        async def _safe_story_lr(sid: str, grp: list[dict]) -> list[dict]:
            async with rag_sem:
                try:
                    return await _run_story_lightrag(
                        sid, grp, llm_func, embed_func, working_dir, embedding_dim)
                except Exception as e:
                    logger.error(f"[LightRAG/story {sid}] failed: {e}")
                    return [{**s, "response": "", "system": "LightRAG"} for s in grp]

        logger.info("=== LightRAG: generating answers ===")
        for coro in asyncio.as_completed(
            [_safe_story_lr(sid, grp) for sid, grp in story_groups.items()]
        ):
            lightrag_raw.extend(await coro)

    # ── Judge all answers ─────────────────────────────────────────────────────
    logger.info(f"=== LLM judging ({len(judge_models)} judges) ===")

    async def _judge_sample(sr: dict) -> dict:
        response     = sr.get("response", "")
        gold_answers = sr.get("gold_answers") or [sr.get("ground_truth", "")]
        ground_truth = "; ".join(str(a) for a in gold_answers if a)
        verdict = await _judge_response(
            sr["question"], ground_truth, response, judge_llms, judge_sem
        )
        return {**sr, "avg_score": verdict["avg_score"], "judges": verdict["judges"]}

    all_to_judge = framerag_raw + lightrag_raw
    all_judged   = await asyncio.gather(*[_judge_sample(r) for r in all_to_judge])

    framerag_results = [r for r in all_judged if r.get("system") == "FrameRAG"]
    lightrag_results = [r for r in all_judged if r.get("system") == "LightRAG"]

    fr_summary = _summarise(framerag_results)
    lr_summary = _summarise(lightrag_results)

    _print_results(fr_summary, lr_summary, judge_models)

    output = {
        "dataset":      "chronoqa",
        "split":        split,
        "metric":       "LLM-judge Likert 1-10 (E2RAG paper protocol)",
        "judge_models": judge_models,
        "story_ids":    target_ids,
        "FrameRAG":     {"summary": fr_summary, "samples": framerag_results},
        "LightRAG":     {"summary": lr_summary, "samples": lightrag_results},
        "reference":    E2RAG_REFERENCE,
    }
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved -> {output_file}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(
    fr_summary: dict,
    lr_summary: dict,
    judge_models: list[str],
) -> None:
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"Dataset: ChronoQA  |  Metric: LLM-judge 1-10 (E2RAG paper)")
    print(f"Judges: {', '.join(judge_models)}")
    print(f"{'System':<32}{'Avg Score':>12}{'N':>8}")
    print(sep)

    print(f"  --- E2RAG paper reference (9 stories, n=497) ---")
    for sys_name, score in E2RAG_REFERENCE.items():
        print(f"  {sys_name:<30}{score:>12.4f}")

    print(sep)
    if lr_summary.get("n", 0) > 0:
        print(
            f"  {'LightRAG (ours)':<30}"
            f"{lr_summary['avg_score']:>12.4f}"
            f"{lr_summary['n']:>8}"
        )
    print(
        f"  {'FrameRAG (ours)':<30}"
        f"{fr_summary['avg_score']:>12.4f}"
        f"{fr_summary['n']:>8}"
    )

    if fr_summary.get("by_facet"):
        print(f"\n  FrameRAG by reasoning facet:")
        for facet, score in sorted(fr_summary["by_facet"].items()):
            print(f"    {facet:<38} {score:.4f}")

    if fr_summary.get("by_story"):
        print(f"\n  FrameRAG by story:")
        for story_key, score in sorted(
            fr_summary["by_story"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {story_key:<42} {score:.4f}")

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FrameRAG on ChronoQA with LLM-judge Likert (E2RAG protocol). "
                    "For multi-hop QA use run_eval.py instead."
    )
    parser.add_argument(
        "--story_ids", default=",".join(AVAILABLE_STORY_IDS),
        help=f"Comma-separated story IDs to evaluate. "
             f"Default: {','.join(AVAILABLE_STORY_IDS)} (all stories with text). "
             f"Stories 3,4 (Harry Potter) are skipped automatically.",
    )
    parser.add_argument("--split",             default="train",
                        help="Dataset split (ChronoQA only has 'train')")
    parser.add_argument("--max_samples",       type=int, default=None)
    parser.add_argument("--working_dir",       default="./eval_storage")
    parser.add_argument("--llm_model",         default="gpt-4.1-mini",
                        help="LLM for RAG answering")
    parser.add_argument("--embed_model",       default="text-embedding-3-small")
    parser.add_argument("--embedding_dim",     type=int, default=1536)
    parser.add_argument("--judge_models",      default="gpt-4.1-mini",
                        help="Comma-separated judge model IDs. "
                             "Paper uses 3 judges: e.g. gpt-4o-mini,gpt-4o,gpt-4o-mini")
    parser.add_argument("--api_key",           default=None)
    parser.add_argument("--base_url",          default=None)
    parser.add_argument("--output_file",       default=None)
    parser.add_argument("--concurrency",       type=int, default=1,
                        help="Parallel stories (keep at 1 to avoid storage conflicts)")
    parser.add_argument("--judge_concurrency", type=int, default=8)
    parser.add_argument("--no_lightrag",       action="store_true",
                        help="Skip LightRAG baseline, evaluate FrameRAG only")
    args = parser.parse_args()

    story_ids   = [s.strip() for s in args.story_ids.split(",") if s.strip()]
    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]

    asyncio.run(run_paper_evaluation(
        split             = args.split,
        max_samples       = args.max_samples,
        story_ids         = story_ids,
        working_dir       = args.working_dir,
        llm_model         = args.llm_model,
        embed_model       = args.embed_model,
        embedding_dim     = args.embedding_dim,
        judge_models      = judge_models,
        api_key           = args.api_key,
        base_url          = args.base_url,
        output_file       = args.output_file,
        concurrency       = args.concurrency,
        judge_concurrency = args.judge_concurrency,
        run_lightrag      = not args.no_lightrag,
    ))


if __name__ == "__main__":
    main()
