"""Regression tests for validation at the query rerank boundary."""

import pytest

from lightrag.utils import apply_rerank_if_enabled

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
