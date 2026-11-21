"""Regression tests for validation at the query rerank boundary."""

import pytest

from lightrag.base import QueryParam
from lightrag.utils import apply_rerank_if_enabled, process_chunks_unified

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_boolean_index_is_not_treated_as_document_one():
    async def rerank_func(**_kwargs):
        return [{"index": True, "relevance_score": 0.9}]

    documents = [{"content": "first"}, {"content": "second"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == documents


@pytest.mark.asyncio
async def test_valid_results_survive_malformed_items():
    async def rerank_func(**_kwargs):
        return [
            None,
            {"index": 1, "relevance_score": "0.8"},
            {"index": 0},
        ]

    documents = [{"content": "first"}, {"content": "second"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == [{"content": "second", "rerank_score": 0.8}]


@pytest.mark.asyncio
async def test_duplicate_index_keeps_its_best_score_and_one_chunk_top_k_slot():
    """A repeated index must take one chunk_top_k slot, with its highest score
    even when a lower one comes first: keeping the first would sort the
    document down and let min_rerank_score drop it."""

    async def rerank_func(**_kwargs):
        return [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.78},
            {"index": 0, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.85},
        ]

    result = await process_chunks_unified(
        query="query",
        unique_chunks=[{"content": "first"}, {"content": "second"}],
        query_param=QueryParam(enable_rerank=True, chunk_top_k=2),
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == [
        {"content": "first", "rerank_score": 0.99, "id": "DC1"},
        {"content": "second", "rerank_score": 0.78, "id": "DC2"},
    ]


@pytest.mark.asyncio
async def test_results_are_sorted_by_score_not_provider_order():
    """Callers (chunk_top_k truncation) slice the returned list positionally,
    so it must be sorted by relevance_score descending regardless of what
    order the provider returned results in."""

    async def rerank_func(**_kwargs):
        return [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]

    documents = [{"content": "low"}, {"content": "high"}, {"content": "mid"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
    )

    assert [d["content"] for d in result] == ["high", "mid", "low"]
    assert [d["rerank_score"] for d in result] == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_legacy_documents_are_not_inferred_from_later_index_metadata():
    legacy_results = [
        {"content": "ranked first"},
        {"content": "ranked second", "index": "source-index"},
    ]

    async def rerank_func(**_kwargs):
        return legacy_results

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=[{"content": "original"}],
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == legacy_results


@pytest.mark.asyncio
async def test_negative_top_n_keeps_original_documents_without_calling_rerank():
    """A negative top_n must not reach a custom rerank function, whose legacy
    result format would otherwise be sliced with it and silently lose the
    last-ranked documents."""
    calls = []

    async def rerank_func(**kwargs):
        calls.append(kwargs)
        return [{"content": "b"}, {"content": "a"}]

    documents = [{"content": "a"}, {"content": "b"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
        top_n=-1,
    )

    assert result == documents
    assert calls == []
