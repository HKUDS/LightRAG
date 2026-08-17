"""
aggregate_chunk_scores must safely ignore chunk_results with None index or None relevance_score.
"""

from lightrag.rerank import aggregate_chunk_scores


def test_aggregate_chunk_scores_handles_none_index_and_score():
    """None-valued rerank results must not corrupt score aggregation."""
    chunk_results = [
        {"index": None, "relevance_score": 0.9},
        {"index": 0, "relevance_score": None},
        {"index": 0, "relevance_score": 0.8},
        {"index": "invalid", "relevance_score": 0.7},
    ]
    doc_indices = [0]
    num_original_docs = 1

    res = aggregate_chunk_scores(
        chunk_results=chunk_results,
        doc_indices=doc_indices,
        num_original_docs=num_original_docs,
        aggregation="mean",
    )
    assert len(res) == 1
    assert res[0]["index"] == 0
    assert res[0]["relevance_score"] == 0.8
