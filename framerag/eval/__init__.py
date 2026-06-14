"""Evaluation pipeline for FrameRAG — RAGAS metrics (same as LightRAG)."""
from .metrics import compute_ragas_metrics, make_ragas_evaluator, RAGAS_AVAILABLE
from .datasets import load_hotpotqa, load_2wikimultihopqa, load_musique, load_chronoqa

__all__ = [
    "compute_ragas_metrics",
    "make_ragas_evaluator",
    "RAGAS_AVAILABLE",
    "load_hotpotqa",
    "load_2wikimultihopqa",
    "load_musique",
    "load_chronoqa",
]
