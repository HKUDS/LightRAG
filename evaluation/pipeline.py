"""
Main orchestrator for the LightRAG evaluation pipeline.

Subcommands:
    ingest    — Insert documents from data/documents/ into LightRAG.
    query     — Run questions through all configured query modes.
    evaluate  — Score answers with RAGAS + LLM-as-judge.
    report    — Print a human-readable summary of the latest results.
    run       — Run the full pipeline: ingest → query → evaluate → report.

Quick-start:
    # 1. Set up .env (copy env.example and fill in LLM + embedding settings)
    cp env.example .env

    # 2. Place your corpus inside evaluation/data/documents/
    # 3. Define your questions in evaluation/data/questions.json

    # 4. Run the full pipeline
    python -m evaluation.pipeline run

    # Or step by step:
    python -m evaluation.pipeline ingest
    python -m evaluation.pipeline query --modes hybrid mix naive
    python -m evaluation.pipeline evaluate --answers results/XXXXXXXX_answers.json
    python -m evaluation.pipeline report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from lightrag.utils import logger

from .config import (
    RESULTS_DIR,
    ALL_MODES,
    EvalConfig,
    DEFAULT_CONFIG,
)
from .ingest import ingest_documents
from .run_queries import run_queries, load_answers_by_mode, load_answers_by_question
from .evaluate import run_ragas, run_llm_judge, save_ragas_scores, save_judge_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _latest_file(pattern: str) -> Optional[Path]:
    """Return the most-recently-created file matching a glob in RESULTS_DIR."""
    matches = sorted(RESULTS_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def print_report(
    answers_file: Optional[Path] = None,
    ragas_file: Optional[Path] = None,
    judge_file: Optional[Path] = None,
) -> None:
    """Print a human-readable summary to stdout."""
    answers_file = answers_file or _latest_file("*_answers.json")
    ragas_file = ragas_file or _latest_file("*_ragas_scores.json")
    judge_file = judge_file or _latest_file("*_judge_results.json")

    print("\n" + "=" * 60)
    print("  LightRAG Evaluation Report")
    print("=" * 60)

    # --- Answers summary ---
    if answers_file and answers_file.exists():
        answers = json.loads(answers_file.read_text())
        modes = sorted({a["mode"] for a in answers})
        total = len(answers)
        errors = sum(1 for a in answers if a["error"])
        avg_lat = (
            sum(a["latency_s"] for a in answers if a["latency_s"] is not None) / total
            if total
            else 0
        )
        print(f"\nAnswers: {answers_file.name}")
        print(f"  Total Q×mode pairs : {total}")
        print(f"  Modes tested       : {', '.join(modes)}")
        print(f"  Errors             : {errors}")
        print(f"  Avg latency        : {avg_lat:.2f}s")

        # Per-mode latency
        by_mode: dict = {}
        for a in answers:
            by_mode.setdefault(a["mode"], []).append(a["latency_s"] or 0)
        print("\n  Per-mode avg latency:")
        for m in sorted(by_mode):
            lats = by_mode[m]
            print(f"    {m:<10} {sum(lats)/len(lats):.2f}s")
    else:
        print("\nNo answers file found. Run: python -m evaluation.pipeline query")

    # --- RAGAS summary ---
    if ragas_file and ragas_file.exists():
        scores = json.loads(ragas_file.read_text())
        print(f"\nRAGAS scores: {ragas_file.name}")
        header = f"  {'Mode':<10} {'Faith':>8} {'AnswRel':>8} {'CtxRec':>8} {'CtxPrc':>8} {'N':>5}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for mode, s in sorted(scores.items()):
            if "error" in s:
                print(f"  {mode:<10}  ERROR: {s['error']}")
                continue
            print(
                f"  {mode:<10}"
                f" {s.get('faithfulness', 'n/a'):>8.3f}"
                f" {s.get('answer_relevancy', 'n/a'):>8.3f}"
                f" {s.get('context_recall', 'n/a'):>8.3f}"
                f" {s.get('context_precision', 'n/a'):>8.3f}"
                f" {s.get('n_samples', '?'):>5}"
            )
    else:
        print("\nNo RAGAS scores found. Run: python -m evaluation.pipeline evaluate")

    # --- Judge summary ---
    if judge_file and judge_file.exists():
        judge = json.loads(judge_file.read_text())
        print(f"\nLLM-as-judge: {judge_file.name}")
        for pairing, data in sorted(judge.items()):
            summary = data.get("_summary", {})
            if not summary:
                continue
            parts = [f"{k}={v}" for k, v in summary.items()]
            print(f"  {pairing}: {', '.join(parts)}")
    else:
        print("\nNo judge results found. Run: python -m evaluation.pipeline evaluate")

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def run_full_pipeline(
    cfg: EvalConfig = DEFAULT_CONFIG,
    *,
    modes: Optional[List[str]] = None,
    skip_ingest: bool = False,
    clear: bool = False,
    baseline_mode: str = "naive",
) -> None:
    """
    End-to-end: ingest → query → evaluate → report.

    Args:
        cfg: Evaluation configuration.
        modes: Query modes to run (defaults to cfg.query_modes).
        skip_ingest: Skip ingestion step (if documents are already indexed).
        clear: Clear RAG storage before ingesting.
        baseline_mode: Baseline for LLM-as-judge comparison.
    """
    # Step 1: Ingest
    if not skip_ingest:
        logger.info("Step 1/4: Ingesting documents…")
        await ingest_documents(cfg, clear=clear)
    else:
        logger.info("Step 1/4: Skipping ingestion (--skip-ingest).")

    # Step 2: Query
    logger.info("Step 2/4: Running queries…")
    answers_file = await run_queries(cfg, modes=modes)

    # Step 3: Evaluate
    logger.info("Step 3/4: Evaluating answers…")
    answers_by_mode = load_answers_by_mode(answers_file)
    answers_by_question = load_answers_by_question(answers_file)

    if cfg.run_ragas:
        try:
            ragas_scores = run_ragas(answers_by_mode, cfg)
            save_ragas_scores(ragas_scores)
        except ImportError as exc:
            logger.warning("RAGAS not available, skipping: %s", exc)

    if cfg.run_llm_judge:
        judge_results = await run_llm_judge(
            answers_by_question, cfg, baseline_mode=baseline_mode
        )
        save_judge_results(judge_results)

    # Step 4: Report
    logger.info("Step 4/4: Generating report…")
    print_report()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m evaluation.pipeline",
        description="LightRAG evaluation pipeline.",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ---- ingest ----
    p_ingest = sub.add_parser("ingest", help="Insert documents into LightRAG.")
    p_ingest.add_argument("--clear", action="store_true",
                          help="Wipe storage before ingesting.")
    p_ingest.add_argument("--no-skip", action="store_true",
                          help="Re-ingest already-processed files.")

    # ---- query ----
    p_query = sub.add_parser("query", help="Run questions through query modes.")
    p_query.add_argument("--modes", nargs="+", choices=ALL_MODES, default=None,
                         help="Modes to run (default: all).")
    p_query.add_argument("--ids", nargs="+", metavar="ID",
                         help="Run only these question IDs.")
    p_query.add_argument("--resume", action="store_true",
                         help="Skip already-answered pairs.")
    p_query.add_argument("--output", type=Path, default=None,
                         help="Output JSON file path.")

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Score answers with RAGAS + LLM-judge.")
    p_eval.add_argument("--answers", type=Path, default=None,
                        help="Answers file (defaults to latest in results/).")
    p_eval.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS scoring.")
    p_eval.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge scoring.")
    p_eval.add_argument("--baseline", default="naive",
                        help="Baseline mode for LLM-as-judge (default: naive).")

    # ---- report ----
    p_report = sub.add_parser("report", help="Print a summary of latest results.")
    p_report.add_argument("--answers", type=Path, default=None)
    p_report.add_argument("--ragas", type=Path, default=None)
    p_report.add_argument("--judge", type=Path, default=None)

    # ---- run (full pipeline) ----
    p_run = sub.add_parser("run", help="Run the full pipeline end-to-end.")
    p_run.add_argument("--modes", nargs="+", choices=ALL_MODES, default=None)
    p_run.add_argument("--skip-ingest", action="store_true",
                       help="Skip document ingestion.")
    p_run.add_argument("--clear", action="store_true",
                       help="Wipe RAG storage before ingesting.")
    p_run.add_argument("--baseline", default="naive",
                       help="Baseline mode for LLM-as-judge.")
    p_run.add_argument("--no-ragas", action="store_true")
    p_run.add_argument("--no-judge", action="store_true")

    return root


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = DEFAULT_CONFIG

    if args.command == "ingest":
        asyncio.run(
            ingest_documents(
                cfg,
                clear=args.clear,
                skip_already_ingested=not args.no_skip,
            )
        )

    elif args.command == "query":
        output = asyncio.run(
            run_queries(
                cfg,
                modes=args.modes,
                question_ids=args.ids,
                resume=args.resume,
                output_file=args.output,
            )
        )
        print(f"Answers written to: {output}")

    elif args.command == "evaluate":
        answers_path = args.answers or _latest_file("*_answers.json")
        if not answers_path:
            print("No answers file found. Run: python -m evaluation.pipeline query")
            sys.exit(1)

        answers_by_mode = load_answers_by_mode(answers_path)
        answers_by_question = load_answers_by_question(answers_path)

        if not args.no_ragas and cfg.run_ragas:
            try:
                ragas_scores = run_ragas(answers_by_mode, cfg)
                save_ragas_scores(ragas_scores)
            except ImportError as exc:
                logger.warning("RAGAS not available: %s", exc)

        if not args.no_judge and cfg.run_llm_judge:
            judge_results = asyncio.run(
                run_llm_judge(answers_by_question, cfg, baseline_mode=args.baseline)
            )
            save_judge_results(judge_results)

    elif args.command == "report":
        print_report(
            answers_file=args.answers,
            ragas_file=args.ragas,
            judge_file=args.judge,
        )

    elif args.command == "run":
        cfg.run_ragas = not args.no_ragas
        cfg.run_llm_judge = not args.no_judge
        asyncio.run(
            run_full_pipeline(
                cfg,
                modes=args.modes,
                skip_ingest=args.skip_ingest,
                clear=args.clear,
                baseline_mode=args.baseline,
            )
        )


if __name__ == "__main__":
    main()
