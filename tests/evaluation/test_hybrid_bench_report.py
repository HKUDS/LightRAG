"""Report rendering follows the CONFIGURED top-k in every metric key and
heading (chunk table, entity hit, ablation), never a hardcoded @10.
"""

from __future__ import annotations

import pytest

from lightrag.evaluation.hybrid_bench.report import render_markdown

pytestmark = pytest.mark.offline


def _ci(mean: float) -> dict:
    return {"mean": mean, "ci95": [mean, mean], "n_queries": 1}


def _results(top_k: int) -> dict:
    return {
        "synthetic": True,
        "config": {"seed": 42, "resamples": 1000, "top_k": top_k},
        "hardware": {"platform": "test"},
        "gates": [],
        "datasets": [
            {
                "name": "tiny",
                "meta": {
                    "domain": "general",
                    "language": "en",
                    "source": "https://example.org",
                    "license": "test",
                    "subset_protocol": "test",
                },
                "ingest": {
                    "chunk_count": 1,
                    "extracted_doc_ids": [],
                    "chunks_only_doc_ids": ["d1"],
                    "extraction_cap": 0,
                    "recovery_rounds_used": 0,
                    "recovered_doc_ids": [],
                    "persistently_failed_doc_ids": [],
                },
                "zero_positive_query_ids": [],
                "entity_truth": {"zero_truth_query_ids": []},
                "empty_retrieval_query_ids": {},
                "chunk_metrics": {
                    "dense": {
                        f"recall@{top_k}": _ci(0.5),
                        f"ndcg@{top_k}": _ci(0.4),
                    }
                },
                "entity_metrics": {"dense": {f"hit@{top_k}": _ci(1.0)}},
                "latency": None,
                "ablation": {
                    "both": {
                        f"recall@{top_k}": 0.5,
                        f"ndcg@{top_k}": 0.4,
                        f"entity_hit@{top_k}": 1.0,
                    }
                },
                "write_amp": None,
                "index_sizes": None,
                "backfill_chunks_per_min": None,
                "rerank": None,
                "skipped": {},
            }
        ],
    }


@pytest.mark.parametrize("top_k", [5, 10])
def test_render_uses_configured_top_k_everywhere(top_k):
    markdown = render_markdown(_results(top_k))
    assert f"| arm | recall@{top_k} [95% CI] | nDCG@{top_k} [95% CI] |" in markdown
    assert f"hit@{top_k} 1.0000" in markdown
    assert (
        f"| combination | recall@{top_k} | nDCG@{top_k} "
        f"| entity hit@{top_k} |" in markdown
    )
    assert "| both | 0.5000 | 0.4000 | 1.0000 |" in markdown


def test_non_default_top_k_leaves_no_hardcoded_ten():
    markdown = render_markdown(_results(5))
    assert "recall@10" not in markdown
    assert "ndcg@10" not in markdown
    assert "hit@10" not in markdown
