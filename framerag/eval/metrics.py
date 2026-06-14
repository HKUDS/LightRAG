"""RAGAS-based evaluation metrics for FrameRAG.

Uses the exact same 4 metrics as lightrag/evaluation/eval_rag_quality.py
so FrameRAG and LightRAG results are directly comparable:
  - Faithfulness          : answer stays faithful to retrieved contexts (LLM judge)
  - AnswerRelevancy       : answer is relevant to the question (LLM judge)
  - ContextRecall         : retrieved contexts cover all aspects of gold answer (LLM)
  - ContextPrecision      : retrieved contexts are precisely relevant (LLM judge)
  - ragas_score           : mean of all 4

Requires:
    pip install ragas datasets langchain-openai
  or:
    uv sync --extra evaluation
"""
from __future__ import annotations

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
