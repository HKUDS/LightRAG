"""Reproducible hybrid-retrieval benchmark harness for the PG hybrid path.

Extends lightrag/evaluation with a repeatable measurement protocol: ingest a
normalized dataset into LightRAG, run dense / hybrid / lexical-only retrieval
arms, score chunk recall@k and nDCG@k plus derived entity-seeding hit@k with
fixed-seed bootstrap confidence intervals, measure retrieval and answer
latency, write amplification, backfill throughput and reranker survival, and
print gate verdicts against the proposal thresholds.

Rules:
- Run via ``python -m lightrag.evaluation.hybrid_bench``; every knob is a CLI
  argument, nothing is read from a repo .env.
- Gate thresholds are PROPOSALS: the harness prints verdicts and never
  adjusts a threshold.
- Fake hash embeddings and the fake LLM are smoke infrastructure only; any
  report produced with them is labeled SYNTHETIC and is not quality evidence.
- The reranker is imported lazily inside the reranker function; importing
  this package never requires sentence-transformers or torch.
"""

from lightrag.evaluation.hybrid_bench.dataset import (
    BenchDataset,
    DatasetError,
    discover_datasets,
    load_dataset,
)

__all__ = [
    "BenchDataset",
    "DatasetError",
    "discover_datasets",
    "load_dataset",
]
