"""
Multi-mode query runner for the evaluation pipeline.

Loads questions from data/questions.json, runs each question through the
configured LightRAG query modes, and saves raw answers for later scoring.

Usage:
    python -m evaluation.run_queries                # all modes
    python -m evaluation.run_queries --modes hybrid mix naive
    python -m evaluation.run_queries --ids q001 q002   # specific questions
    python -m evaluation.run_queries --resume           # skip already-answered
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lightrag import LightRAG, QueryParam
from lightrag.utils import logger

from .config import (
    QUESTIONS_FILE,
    RESULTS_DIR,
    EvalConfig,
    DEFAULT_CONFIG,
    ALL_MODES,
)
from .ingest import _build_lightrag


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Question = Dict[str, Any]
# Minimal required keys: "id", "question"
# Optional keys:         "ground_truth", "hops", "evidence_docs", "metadata"


Answer = Dict[str, Any]
# Keys: "question_id", "question", "mode", "answer", "latency_s", "error"


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def load_questions(path: Path = QUESTIONS_FILE) -> List[Question]:
    """Load questions from the JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {path}\n"
            "Create it following the schema in data/questions.json."
        )
    data = json.loads(path.read_text())
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(
            f"No questions found in {path}. "
            "Add entries under the 'questions' key."
        )
    return questions


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

async def _run_single_query(
    rag: LightRAG,
    question: Question,
    mode: str,
    cfg: EvalConfig,
) -> Answer:
    """Run one question in one mode and return an Answer dict."""
    q_text = question["question"]
    q_id = question["id"]

    param = QueryParam(
        mode=mode,
        top_k=cfg.top_k,
        chunk_top_k=cfg.chunk_top_k,
        max_entity_tokens=cfg.max_entity_tokens,
        max_relation_tokens=cfg.max_relation_tokens,
        max_total_tokens=cfg.max_total_tokens,
        enable_rerank=cfg.enable_rerank,
    )

    start = time.perf_counter()
    try:
        answer_text = await rag.aquery(q_text, param=param)
        latency = time.perf_counter() - start
        return {
            "question_id": q_id,
            "question": q_text,
            "mode": mode,
            "answer": answer_text,
            "latency_s": round(latency, 3),
            "error": None,
        }
    except Exception as exc:
        latency = time.perf_counter() - start
        logger.error("Query failed [%s / %s]: %s", q_id, mode, exc)
        return {
            "question_id": q_id,
            "question": q_text,
            "mode": mode,
            "answer": None,
            "latency_s": round(latency, 3),
            "error": str(exc),
        }


async def run_queries(
    cfg: EvalConfig = DEFAULT_CONFIG,
    *,
    modes: Optional[List[str]] = None,
    question_ids: Optional[List[str]] = None,
    resume: bool = False,
    output_file: Optional[Path] = None,
) -> Path:
    """
    Run all questions through the specified query modes.

    Args:
        cfg: Evaluation configuration.
        modes: Which query modes to use. Defaults to cfg.query_modes.
        question_ids: If set, only run these question IDs.
        resume: Skip (question_id, mode) pairs already present in output_file.
        output_file: Where to write results. Auto-generated if None.

    Returns:
        Path to the JSON file containing all answers.
    """
    modes = modes or cfg.query_modes
    questions = load_questions()

    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
        if not questions:
            raise ValueError(f"No questions matched IDs: {question_ids}")

    # Determine output file
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"{ts}_answers.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    existing: Dict[str, Answer] = {}
    if resume and output_file.exists():
        existing_list: List[Answer] = json.loads(output_file.read_text())
        existing = {
            f"{a['question_id']}::{a['mode']}": a for a in existing_list
        }
        logger.info("Resuming — %d answer(s) already recorded.", len(existing))

    # Build work list
    work = [
        (q, mode)
        for q in questions
        for mode in modes
        if f"{q['id']}::{mode}" not in existing
    ]
    logger.info(
        "Running %d question(s) × %d mode(s) = %d queries.",
        len(questions),
        len(modes),
        len(work),
    )

    if not work:
        logger.info("Nothing to do — all answers already recorded.")
        return output_file

    # Initialise LightRAG
    rag = _build_lightrag(cfg)
    await rag.initialize_storages()

    # Run with a semaphore to cap concurrency
    sem = asyncio.Semaphore(cfg.query_concurrency)
    results: List[Answer] = list(existing.values())
    total = len(work)
    done = 0

    async def _bounded(q: Question, mode: str) -> None:
        nonlocal done
        async with sem:
            answer = await _run_single_query(rag, q, mode, cfg)
            results.append(answer)
            done += 1
            status = "OK" if answer["error"] is None else "ERR"
            logger.info(
                "[%d/%d] %s | mode=%-8s | %.2fs | %s",
                done,
                total,
                answer["question_id"],
                mode,
                answer["latency_s"],
                status,
            )

    await asyncio.gather(*[_bounded(q, m) for q, m in work])
    await rag.finalize_storages()

    # Sort for reproducibility
    results.sort(key=lambda a: (a["question_id"], a["mode"]))

    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Answers saved to %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Convenience: load answers back grouped by mode
# ---------------------------------------------------------------------------

def load_answers_by_mode(answers_file: Path) -> Dict[str, List[Answer]]:
    """Return {mode: [answer, ...]} from an answers JSON file."""
    answers: List[Answer] = json.loads(answers_file.read_text())
    by_mode: Dict[str, List[Answer]] = {}
    for a in answers:
        by_mode.setdefault(a["mode"], []).append(a)
    return by_mode


def load_answers_by_question(answers_file: Path) -> Dict[str, List[Answer]]:
    """Return {question_id: [answer, ...]} from an answers JSON file."""
    answers: List[Answer] = json.loads(answers_file.read_text())
    by_q: Dict[str, List[Answer]] = {}
    for a in answers:
        by_q.setdefault(a["question_id"], []).append(a)
    return by_q


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-mode LightRAG queries.")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=None,
        help="Query modes to run (default: all configured modes).",
    )
    p.add_argument(
        "--ids",
        nargs="+",
        metavar="QUESTION_ID",
        default=None,
        help="Only run these question IDs.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already answered in the latest output file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (auto-generated if not set).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output = asyncio.run(
        run_queries(
            modes=args.modes,
            question_ids=args.ids,
            resume=args.resume,
            output_file=args.output,
        )
    )
    print(f"Answers written to: {output}")


if __name__ == "__main__":
    main()
