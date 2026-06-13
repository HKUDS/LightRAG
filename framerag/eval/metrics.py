"""Evaluation metrics for multi-hop QA."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_em(prediction: str, gold: str) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    return float(_normalize(prediction) == _normalize(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 after normalization."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em(prediction: str, gold_answers: list[str]) -> float:
    """EM against a list of acceptable gold answers (take max)."""
    return max(compute_em(prediction, g) for g in gold_answers) if gold_answers else 0.0


def best_f1(prediction: str, gold_answers: list[str]) -> float:
    """F1 against a list of acceptable gold answers (take max)."""
    return max(compute_f1(prediction, g) for g in gold_answers) if gold_answers else 0.0


def evaluate_answers(
    predictions: list[str],
    gold_answers: list[list[str]],
) -> dict[str, float]:
    """Evaluate a list of predictions against gold answers.

    Args:
        predictions: model answers, one per question
        gold_answers: list of acceptable answers per question (list of lists)

    Returns:
        dict with keys: em, f1, n (sample count)
    """
    assert len(predictions) == len(gold_answers), "length mismatch"
    em_scores = [best_em(p, g) for p, g in zip(predictions, gold_answers)]
    f1_scores = [best_f1(p, g) for p, g in zip(predictions, gold_answers)]
    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "n": len(predictions),
    }
