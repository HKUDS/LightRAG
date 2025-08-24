"""
Evaluation module implementing the 4 key RAG metrics from AI News & Strategy Daily.

Metrics implemented:
1. Relevance: Are the right chunks retrieved?
2. Fidelity: Is the response based on real sources?  
3. Quality: Would a human rate it as correct?
4. Latency: Is it fast enough (couple of seconds)?
"""

from .base import RAGEvaluator, EvaluationMetrics, EvaluationResult
from .relevance_evaluator import RelevanceEvaluator
from .fidelity_evaluator import FidelityEvaluator
from .quality_evaluator import QualityEvaluator
from .latency_evaluator import LatencyEvaluator

__all__ = [
    "RAGEvaluator",
    "EvaluationMetrics", 
    "EvaluationResult",
    "RelevanceEvaluator",
    "FidelityEvaluator",
    "QualityEvaluator", 
    "LatencyEvaluator",
]