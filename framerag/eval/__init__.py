"""Evaluation pipeline for FrameRAG on multi-hop QA benchmarks."""
from .metrics import compute_em, compute_f1, evaluate_answers
from .datasets import load_hotpotqa, load_2wikimultihopqa, load_musique, load_chronoqa

__all__ = [
    "compute_em", "compute_f1", "evaluate_answers",
    "load_hotpotqa", "load_2wikimultihopqa", "load_musique", "load_chronoqa",
]
