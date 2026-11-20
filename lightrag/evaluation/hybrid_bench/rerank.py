"""Reranker survival of lexical-only chunk hits.

Rules:
- sentence-transformers is imported LAZILY inside ``rerank_survival`` only;
  importing this module never requires it or torch. At runtime it is an
  ephemeral uv overlay, never a repo dependency.
- Survival is the fraction of qrels-positive lexical hits whose CrossEncoder
  score for the (query, chunk content) pair reaches ``min_score``.
- A missing reranker stack yields a skipped result with the install hint,
  not a crash and not a fabricated score.
"""

from __future__ import annotations

from dataclasses import dataclass

_INSTALL_HINT = (
    "uv run --no-sync --with sentence-transformers python -m "
    "lightrag.evaluation.hybrid_bench ... (ephemeral overlay; pyproject and "
    "uv.lock stay untouched)"
)


@dataclass(frozen=True, slots=True)
class RerankResult:
    status: str
    survival: float | None
    scored_hits: int
    surviving_hits: int
    min_score: float
    model: str
    detail: str


def rerank_survival(
    pairs: list[tuple[str, str]],
    min_score: float,
    model_name: str,
) -> RerankResult:
    """Score (query, chunk content) pairs; report the survival fraction."""
    if not pairs:
        return RerankResult(
            status="not-run",
            survival=None,
            scored_hits=0,
            surviving_hits=0,
            min_score=min_score,
            model=model_name,
            detail="no qrels-positive lexical hits to score",
        )
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        return RerankResult(
            status="skipped",
            survival=None,
            scored_hits=0,
            surviving_hits=0,
            min_score=min_score,
            model=model_name,
            detail=f"sentence-transformers unavailable: {exc}; {_INSTALL_HINT}",
        )
    model = CrossEncoder(model_name)
    scores = model.predict(pairs)
    surviving = sum(1 for score in scores if float(score) >= min_score)
    return RerankResult(
        status="scored",
        survival=surviving / len(pairs),
        scored_hits=len(pairs),
        surviving_hits=surviving,
        min_score=min_score,
        model=model_name,
        detail=f"{surviving}/{len(pairs)} lexical hits score >= {min_score}",
    )