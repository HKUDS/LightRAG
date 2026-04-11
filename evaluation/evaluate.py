"""
Evaluation metrics for the LightRAG evaluation pipeline.

Two scorers are implemented:

1. RAGAS scorer  — uses Faithfulness, AnswerRelevancy, ContextRecall,
                   ContextPrecision.  Requires "ground_truth" in questions.json
                   and the `evaluation` extra to be installed:
                       uv sync --extra evaluation

2. LLM-as-judge  — pairwise comparison between every mode and a chosen
                   baseline (default: "naive").  Scores each mode on
                   Comprehensiveness, Diversity, and Empowerment.
                   Works without ground_truth.

Usage:
    python -m evaluation.evaluate --answers results/20240101_120000_answers.json
    python -m evaluation.evaluate --answers results/20240101_120000_answers.json --no-ragas
    python -m evaluation.evaluate --answers results/20240101_120000_answers.json --baseline hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lightrag.utils import logger

from .config import (
    QUESTIONS_FILE,
    RESULTS_DIR,
    EvalConfig,
    DEFAULT_CONFIG,
)
from .run_queries import load_answers_by_mode, load_answers_by_question, load_questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ground_truth_map(questions_file: Path = QUESTIONS_FILE) -> Dict[str, str]:
    """Return {question_id: ground_truth} for questions that have it."""
    questions = load_questions(questions_file)
    return {
        q["id"]: q["ground_truth"]
        for q in questions
        if q.get("ground_truth")
    }


# ---------------------------------------------------------------------------
# RAGAS scorer
# ---------------------------------------------------------------------------

def _ragas_score_mode(
    answers: List[Dict],
    ground_truth_map: Dict[str, str],
    cfg: EvalConfig,
) -> Optional[Dict[str, float]]:
    """
    Run RAGAS metrics on a list of answers for one query mode.

    Returns a dict with keys: faithfulness, answer_relevancy,
    context_recall, context_precision.
    Returns None if no answers have ground_truth.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "RAGAS dependencies are missing. Install with:\n"
            "    uv sync --extra evaluation\n"
            "or: pip install 'lightrag-hku[evaluation]'"
        )

    # Filter to answers that have ground_truth and a successful response
    rows = [
        a for a in answers
        if a["error"] is None
        and a["question_id"] in ground_truth_map
    ]
    if not rows:
        logger.warning("No answers with ground_truth found — skipping RAGAS.")
        return None

    dataset_dict: Dict[str, List] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for a in rows:
        dataset_dict["question"].append(a["question"])
        dataset_dict["answer"].append(a["answer"] or "")
        # LightRAG answers don't separate context; we pass the answer itself
        # as a proxy context. For richer evaluation, use aquery_data to get
        # actual retrieved chunks (see pipeline.py for the hook point).
        dataset_dict["contexts"].append([a["answer"] or ""])
        dataset_dict["ground_truth"].append(ground_truth_map[a["question_id"]])

    ds = Dataset.from_dict(dataset_dict)

    # Build judge LLM / embeddings
    judge_llm = ChatOpenAI(
        model=cfg.eval_llm_model,
        api_key=cfg.eval_api_key,
        base_url=(
            cfg.eval_llm_binding_host
            if cfg.eval_llm_binding_host != "https://api.openai.com/v1"
            else None
        ),
    )
    judge_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=cfg.eval_api_key,
        base_url=(
            cfg.eval_llm_binding_host
            if cfg.eval_llm_binding_host != "https://api.openai.com/v1"
            else None
        ),
    )

    metrics = [
        Faithfulness(llm=LangchainLLMWrapper(judge_llm)),
        AnswerRelevancy(
            llm=LangchainLLMWrapper(judge_llm),
            embeddings=judge_embeddings,
        ),
        ContextRecall(llm=LangchainLLMWrapper(judge_llm)),
        ContextPrecision(llm=LangchainLLMWrapper(judge_llm)),
    ]

    result = ragas_evaluate(dataset=ds, metrics=metrics)
    df = result.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "n_samples": len(rows),
    }


def run_ragas(
    answers_by_mode: Dict[str, List[Dict]],
    cfg: EvalConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict[str, float]]:
    """
    Score every mode with RAGAS.

    Returns {mode: {metric: score, ...}}.
    """
    ground_truth_map = _get_ground_truth_map()
    if not ground_truth_map:
        logger.warning(
            "No ground_truth found in %s. "
            "RAGAS context_recall and context_precision will not be meaningful.",
            QUESTIONS_FILE,
        )

    scores: Dict[str, Dict] = {}
    for mode, answers in answers_by_mode.items():
        logger.info("Running RAGAS for mode: %s (%d answers)", mode, len(answers))
        try:
            result = _ragas_score_mode(answers, ground_truth_map, cfg)
            scores[mode] = result or {}
        except Exception as exc:
            logger.error("RAGAS failed for mode %s: %s", mode, exc)
            scores[mode] = {"error": str(exc)}

    return scores


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI-generated answers. \
Assess answers objectively and return only the JSON specified."""

_JUDGE_PROMPT_TEMPLATE = """\
Evaluate two answers to the same question on three criteria:

**Comprehensiveness** — How thoroughly does the answer cover all aspects of the question?
**Diversity** — How varied and rich are the perspectives and insights provided?
**Empowerment** — How well does the answer help the reader understand the topic and make informed judgments?

Question:
{question}

Answer A ({mode_a}):
{answer_a}

Answer B ({mode_b}):
{answer_b}

For each criterion, pick the better answer (A or B) and explain why.
Then declare an overall winner.

Return ONLY valid JSON in this exact format:
{{
    "Comprehensiveness": {{"Winner": "A or B", "Explanation": "..."}},
    "Diversity":         {{"Winner": "A or B", "Explanation": "..."}},
    "Empowerment":       {{"Winner": "A or B", "Explanation": "..."}},
    "Overall":           {{"Winner": "A or B", "Explanation": "..."}}
}}"""


async def _judge_pair_async(
    question: str,
    mode_a: str,
    answer_a: str,
    mode_b: str,
    answer_b: str,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """Call the judge LLM for one question × one pair of modes."""
    try:
        from openai import AsyncOpenAI
        import json as _json

        client = AsyncOpenAI(
            api_key=cfg.eval_api_key,
            base_url=cfg.eval_llm_binding_host,
        )
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            mode_a=mode_a,
            answer_a=answer_a or "(no answer)",
            mode_b=mode_b,
            answer_b=answer_b or "(no answer)",
        )
        response = await client.chat.completions.create(
            model=cfg.eval_llm_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return _json.loads(raw)
    except Exception as exc:
        logger.error(
            "Judge failed [%s vs %s] for question '%s...': %s",
            mode_a,
            mode_b,
            question[:60],
            exc,
        )
        return {"error": str(exc)}


async def run_llm_judge(
    answers_by_question: Dict[str, List[Dict]],
    cfg: EvalConfig = DEFAULT_CONFIG,
    baseline_mode: str = "naive",
) -> Dict[str, Any]:
    """
    Pairwise LLM-as-judge comparison of every mode against baseline_mode.

    Returns a nested dict:
        {
            "<mode>_vs_<baseline>": {
                "<question_id>": {judgement_dict},
                ...
                "_summary": {"A_wins": int, "B_wins": int, "ties": int}
            },
            ...
        }
    """
    results: Dict[str, Any] = {}

    for question_id, answers in answers_by_question.items():
        answers_by_mode = {a["mode"]: a for a in answers}

        baseline = answers_by_mode.get(baseline_mode)
        if baseline is None:
            logger.warning(
                "Baseline mode %r not found for question %s — skipping.",
                baseline_mode,
                question_id,
            )
            continue

        for mode, answer in answers_by_mode.items():
            if mode == baseline_mode:
                continue

            key = f"{mode}_vs_{baseline_mode}"
            results.setdefault(key, {})

            judgment = await _judge_pair_async(
                question=answer["question"],
                mode_a=mode,
                answer_a=answer["answer"] or "",
                mode_b=baseline_mode,
                answer_b=baseline["answer"] or "",
                cfg=cfg,
            )
            results[key][question_id] = judgment

    # Compute per-pairing win summaries
    for key, judgments in results.items():
        mode_a = key.split("_vs_")[0]
        a_wins = b_wins = ties = 0
        for q_id, j in judgments.items():
            if isinstance(j, dict) and "Overall" in j:
                w = j["Overall"].get("Winner", "")
                if w == "A":
                    a_wins += 1
                elif w == "B":
                    b_wins += 1
                else:
                    ties += 1
        results[key]["_summary"] = {
            f"{mode_a}_wins": a_wins,
            f"{baseline_mode}_wins": b_wins,
            "ties": ties,
        }

    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_ragas_scores(scores: Dict, output_dir: Path = RESULTS_DIR) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{ts}_ragas_scores.json"
    out.write_text(json.dumps(scores, indent=2))
    logger.info("RAGAS scores saved to %s", out)

    # Also write a quick CSV
    try:
        import csv
        csv_path = output_dir / f"{ts}_ragas_scores.csv"
        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "n_samples"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode"] + metrics)
            w.writeheader()
            for mode, s in scores.items():
                row = {"mode": mode}
                for m in metrics:
                    row[m] = s.get(m, "")
                w.writerow(row)
        logger.info("RAGAS CSV saved to %s", csv_path)
    except Exception as exc:
        logger.warning("Could not write CSV: %s", exc)

    return out


def save_judge_results(results: Dict, output_dir: Path = RESULTS_DIR) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{ts}_judge_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Judge results saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score LightRAG answers.")
    p.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="Path to the answers JSON file produced by run_queries.py.",
    )
    p.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS scoring.",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring.",
    )
    p.add_argument(
        "--baseline",
        default="naive",
        help="Baseline mode for LLM-as-judge comparison (default: naive).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = DEFAULT_CONFIG

    answers_by_mode = load_answers_by_mode(args.answers)
    answers_by_question = load_answers_by_question(args.answers)

    if not args.no_ragas and cfg.run_ragas:
        ragas_scores = run_ragas(answers_by_mode, cfg)
        save_ragas_scores(ragas_scores)

    if not args.no_judge and cfg.run_llm_judge:
        judge_results = asyncio.run(
            run_llm_judge(answers_by_question, cfg, baseline_mode=args.baseline)
        )
        save_judge_results(judge_results)


if __name__ == "__main__":
    main()
