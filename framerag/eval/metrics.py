"""Evaluation metrics for multi-hop QA.

Implements:
  - Exact Match (EM): normalized string equality (SQuAD-style)
  - Token-level F1: token overlap after normalization (SQuAD-style)
  - ROUGE-1 / ROUGE-2 / ROUGE-L: recall-oriented overlap metrics
  - Coverage: fraction of gold answer tokens present in prediction
  - Hallucination proxy: fraction of prediction tokens NOT in supporting docs

All metrics handle multi-answer references by taking the max over alternatives.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# EM + F1 (SQuAD-style)
# ─────────────────────────────────────────────────────────────────────────────

def compute_em(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def compute_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em(prediction: str, gold_answers: list[str]) -> float:
    return max(compute_em(prediction, g) for g in gold_answers) if gold_answers else 0.0


def best_f1(prediction: str, gold_answers: list[str]) -> float:
    return max(compute_f1(prediction, g) for g in gold_answers) if gold_answers else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def compute_rouge_n(prediction: str, gold: str, n: int) -> dict[str, float]:
    pred_ng = _ngrams(_normalize(prediction).split(), n)
    gold_ng = _ngrams(_normalize(gold).split(), n)
    overlap = sum((pred_ng & gold_ng).values())
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall    = overlap / max(sum(gold_ng.values()), 1)
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j + 1]
            dp[j + 1] = prev + 1 if a[i] == b[j] else max(dp[j + 1], dp[j])
            prev = temp
    return dp[n]


def compute_rouge_l(prediction: str, gold: str) -> dict[str, float]:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / max(len(pred_tokens), 1)
    recall    = lcs / max(len(gold_tokens), 1)
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_rouge(
    prediction: str,
    gold_answers: list[str],
    n: int = 1,
    rouge_l: bool = False,
) -> dict[str, float]:
    """Return the best ROUGE-n (or ROUGE-L) F1 across all gold answers."""
    if not gold_answers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    fn = compute_rouge_l if rouge_l else lambda p, g: compute_rouge_n(p, g, n)
    results = [fn(prediction, g) for g in gold_answers]
    best = max(results, key=lambda r: r["f1"])
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Coverage / Hallucination proxy
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage(prediction: str, source_texts: list[str]) -> float:
    """Fraction of unique prediction tokens that appear in any source text."""
    pred_tokens = set(_normalize(prediction).split())
    if not pred_tokens:
        return 1.0
    source_tokens: set[str] = set()
    for src in source_texts:
        source_tokens.update(_normalize(src).split())
    return len(pred_tokens & source_tokens) / len(pred_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_answers(
    predictions: list[str],
    gold_answers: list[list[str]],
    source_texts: Optional[list[list[str]]] = None,
) -> dict:
    """Evaluate a list of predictions against multi-reference gold answers.

    Args:
        predictions: model answers, one per question
        gold_answers: list of gold answer sets per question (list of lists)
        source_texts: optional list of supporting document lists per question
                      (used for coverage computation)

    Returns:
        dict with keys: em, f1, rouge1, rouge2, rougeL, coverage (optional), n
    """
    if len(predictions) != len(gold_answers):
        raise ValueError(
            f"predictions ({len(predictions)}) vs gold_answers ({len(gold_answers)}) mismatch"
        )

    em_scores     = [best_em(p, g) for p, g in zip(predictions, gold_answers)]
    f1_scores     = [best_f1(p, g) for p, g in zip(predictions, gold_answers)]
    rouge1_scores = [best_rouge(p, g, n=1)["f1"]
                     for p, g in zip(predictions, gold_answers)]
    rouge2_scores = [best_rouge(p, g, n=2)["f1"]
                     for p, g in zip(predictions, gold_answers)]
    rougel_scores = [best_rouge(p, g, rouge_l=True)["f1"]
                     for p, g in zip(predictions, gold_answers)]

    def _avg(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    result: dict = {
        "em":      _avg(em_scores),
        "f1":      _avg(f1_scores),
        "rouge1":  _avg(rouge1_scores),
        "rouge2":  _avg(rouge2_scores),
        "rougeL":  _avg(rougel_scores),
        "n":       len(predictions),
    }

    if source_texts is not None:
        cov_scores = [
            compute_coverage(p, srcs)
            for p, srcs in zip(predictions, source_texts)
        ]
        result["coverage"] = _avg(cov_scores)

    return result


def evaluate_by_type(
    results: list[dict],
    type_field: str = "type",
) -> dict[str, dict]:
    """Compute per-type metrics from a list of result dicts.

    Each result dict must have 'prediction', 'gold_answers', and type_field.
    Returns {type_name: metrics_dict}.
    """
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        qtype = r.get(type_field, "unknown")
        by_type[qtype].append(r)

    type_metrics: dict[str, dict] = {}
    for qtype, type_results in by_type.items():
        preds = [r["prediction"] for r in type_results]
        golds = [r["gold_answers"] for r in type_results]
        type_metrics[qtype] = evaluate_answers(preds, golds)

    return type_metrics
