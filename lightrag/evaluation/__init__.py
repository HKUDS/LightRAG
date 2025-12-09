"""
LightRAG Evaluation Module

RAGAS-based evaluation framework for assessing RAG system quality.

Usage:
    from lightrag.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    results = await evaluator.run()

Note: RAGEvaluator is imported lazily to avoid import errors
when ragas/datasets are not installed.
"""

from typing import Any

__all__ = ['RAGEvaluator']

# Stub to satisfy static analyzers; lazily loaded in __getattr__
RAGEvaluator: Any


def __getattr__(name: str) -> Any:
    """Lazy import to avoid dependency errors when ragas is not installed."""
    if name == 'RAGEvaluator':
        from .eval_rag_quality import RAGEvaluator

        return RAGEvaluator
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
