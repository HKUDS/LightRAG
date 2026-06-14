"""Evaluation metrics for FrameRAG.

Two metric families:

RAGAS (multi-hop QA — HotpotQA / 2WikiMHQA / MuSiQue):
  Matches lightrag/evaluation/eval_rag_quality.py for direct LightRAG comparison.
  - Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision, ragas_score

Likert (ChronoQA — temporal/causal narrative QA):
  Matches E²RAG paper (arXiv 2506.05939) for direct comparison.
  - 3 LLM judges, score 1–10 per answer
  - Formula: mean over judges × mean over samples
  - Dimensions: accuracy, consistency, completeness

Requires:
    pip install ragas datasets langchain-openai
  or:
    uv sync --extra evaluation
"""
from __future__ import annotations

import asyncio
import math
import os
from typing import Optional

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
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _require_ragas():
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS dependencies not installed.\n"
            "Install with:  uv sync --extra evaluation\n"
            "          or:  pip install ragas datasets langchain-openai"
        )


def make_ragas_evaluator(
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Return a configured (llm, embeddings) pair for RAGAS.

    Args:
        llm_model:       OpenAI-compatible chat model id.
        embedding_model: OpenAI-compatible embedding model id.
        api_key:         API key; falls back to OPENAI_API_KEY env var.
        base_url:        Custom endpoint URL (for local / proxy models).
    """
    _require_ragas()
    key = api_key or os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY to run RAGAS evaluation."
        )
    url = base_url or os.getenv("EVAL_LLM_BINDING_HOST")

    chat_kwargs = dict(model=llm_model, api_key=key)
    emb_kwargs  = dict(model=embedding_model, api_key=key)
    if url:
        chat_kwargs["base_url"] = url
        emb_kwargs["base_url"]  = url

    llm        = LangchainLLMWrapper(ChatOpenAI(**chat_kwargs))
    embeddings = OpenAIEmbeddings(**emb_kwargs)
    return llm, embeddings


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, float]:
    """Evaluate a batch with the 4 RAGAS metrics used by LightRAG.

    Args:
        questions:     One question per sample.
        answers:       RAG system's answer per sample.
        contexts:      List of retrieved text passages per sample (list[list[str]]).
        ground_truths: Gold reference answer per sample.
        llm_model:     Scoring LLM model id.
        embedding_model: Embedding model id for AnswerRelevancy.
        api_key:       OpenAI-compatible API key.
        base_url:      Custom endpoint (optional).

    Returns:
        dict with keys: faithfulness, answer_relevance, context_recall,
                        context_precision, ragas_score, n
    """
    _require_ragas()
    if not questions:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "ragas_score": 0.0,
            "n": 0,
        }

    llm, embeddings = make_ragas_evaluator(llm_model, embedding_model, api_key, base_url)

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
    result  = ragas_evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    scores = result.to_pandas()

    def _safe(col: str) -> float:
        if col not in scores.columns:
            return 0.0
        vals = [v for v in scores[col].tolist() if not _is_nan(v)]
        return sum(vals) / len(vals) if vals else 0.0

    faithfulness       = _safe("faithfulness")
    answer_relevance   = _safe("answer_relevancy")
    context_recall     = _safe("context_recall")
    context_precision  = _safe("context_precision")

    valid = [s for s in [faithfulness, answer_relevance, context_recall, context_precision]
             if s > 0]
    ragas_score = sum(valid) / len(valid) if valid else 0.0

    return {
        "faithfulness":       faithfulness,
        "answer_relevance":   answer_relevance,
        "context_recall":     context_recall,
        "context_precision":  context_precision,
        "ragas_score":        ragas_score,
        "n":                  len(questions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Likert scoring (ChronoQA — matches E²RAG paper arXiv 2506.05939)
# ─────────────────────────────────────────────────────────────────────────────

_LIKERT_SYSTEM = """\
You are a rigorous evaluator of question-answering systems operating on narrative text.
Score the answer on a scale from 1 to 10 using the rubric below.
Respond with ONLY a single integer between 1 and 10. No explanation.

Rubric:
 1–2  Completely wrong or irrelevant.
 3–4  Partially addresses the question but with major errors or omissions.
 5–6  Mostly correct but missing important details or has minor factual errors.
 7–8  Correct and well-grounded; minor incompleteness allowed.
 9–10 Fully correct, complete, temporally/causally consistent, and well-supported."""

_LIKERT_USER = """\
Question: {question}

Gold answer: {ground_truth}

System answer: {prediction}

Score (1–10):"""

_DEFAULT_NUM_JUDGES = 3


async def _judge_one(
    question: str,
    ground_truth: str,
    prediction: str,
    llm_func,
    judge_id: int,
) -> float:
    """Ask one LLM judge to score a single answer. Returns float 1–10."""
    prompt = _LIKERT_USER.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction or "(no answer)",
    )
    try:
        raw = await llm_func(prompt, system_prompt=_LIKERT_SYSTEM)
    except TypeError:
        # Some LLM wrappers don't accept system_prompt; fall back
        raw = await llm_func(_LIKERT_SYSTEM + "\n\n" + prompt)

    # Parse first integer found in response
    import re
    nums = re.findall(r"\b(10|[1-9])\b", (raw or "").strip())
    if nums:
        return float(nums[0])
    return 5.0  # neutral fallback if parse fails


async def _score_sample_async(
    question: str,
    ground_truth: str,
    prediction: str,
    llm_func,
    num_judges: int = _DEFAULT_NUM_JUDGES,
) -> dict:
    """Run num_judges judges concurrently for one sample. Returns per-judge scores."""
    tasks = [
        _judge_one(question, ground_truth, prediction, llm_func, j)
        for j in range(num_judges)
    ]
    scores = await asyncio.gather(*tasks)
    return {
        "scores": list(scores),
        "mean":   sum(scores) / len(scores),
    }


def compute_likert_metrics(
    questions: list[str],
    predictions: list[str],
    ground_truths: list[str],
    llm_func=None,
    llm_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    num_judges: int = _DEFAULT_NUM_JUDGES,
) -> dict:
    """LLM-judged Likert evaluation matching E²RAG paper (arXiv 2506.05939).

    Runs `num_judges` LLM judges per sample, averages within-sample then
    across samples:
        score = mean_samples( mean_judges( score_ij ) )

    Args:
        questions:     Questions.
        predictions:   System answers.
        ground_truths: Gold reference answers.
        llm_func:      Async callable (prompt, **kwargs) -> str.
                       If None, builds an OpenAI client from llm_model + api_key.
        llm_model:     Model id (used only when llm_func is None).
        api_key:       API key (falls back to EVAL_LLM_BINDING_API_KEY / OPENAI_API_KEY).
        base_url:      Custom endpoint (optional).
        num_judges:    Number of independent LLM judges (default 3, paper uses 3).

    Returns:
        dict with keys: likert_score (float 1–10), per_sample (list[dict]), n (int)
    """
    if not questions:
        return {"likert_score": 0.0, "per_sample": [], "n": 0}

    # Build llm_func if not provided
    if llm_func is None:
        key = api_key or os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("Set EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY.")
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(api_key=key, base_url=base_url or os.getenv("EVAL_LLM_BINDING_HOST"))
        model = llm_model

        async def _openai_llm(prompt: str, system_prompt: str = "", **_kw) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = await _client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=5,
            )
            return resp.choices[0].message.content or ""

        llm_func = _openai_llm

    async def _run_all():
        tasks = [
            _score_sample_async(q, gt, pred, llm_func, num_judges)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run_all())
                per_sample = future.result()
        else:
            per_sample = loop.run_until_complete(_run_all())
    except RuntimeError:
        per_sample = asyncio.run(_run_all())

    means = [s["mean"] for s in per_sample]
    likert_score = sum(means) / len(means) if means else 0.0

    return {
        "likert_score": likert_score,
        "per_sample":   [
            {"question": q, "ground_truth": gt, "prediction": pred, **s}
            for q, gt, pred, s in zip(questions, ground_truths, predictions, per_sample)
        ],
        "n": len(questions),
    }
