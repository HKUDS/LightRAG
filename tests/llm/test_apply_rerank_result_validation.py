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
