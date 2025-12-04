"""
LightRAG Evaluation Module

RAGAS-based evaluation framework for assessing RAG system quality.

Usage:
    from lightrag.evaluation.eval_rag_quality import RAGEvaluator

    evaluator = RAGEvaluator()
    results = await evaluator.run()

Note: RAGEvaluator is imported dynamically to avoid import errors
when ragas/datasets are not installed.
"""

__all__ = ["RAGEvaluator"]
