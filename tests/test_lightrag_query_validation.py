from types import SimpleNamespace

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.query_validation import EmptyQueryError, RAGQueryTooShortError


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["aquery_llm", "aquery_data"])
async def test_core_rag_query_entry_points_reject_short_queries_before_storage_access(
    method_name,
):
    rag = SimpleNamespace()

    with pytest.raises(RAGQueryTooShortError):
        await getattr(LightRAG, method_name)(
            rag,
            "中",
            param=QueryParam(mode="mix"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["aquery_llm", "aquery_data"])
@pytest.mark.parametrize("mode", ["mix", "naive", "bypass"])
@pytest.mark.parametrize("query", ["", "   "])
async def test_core_query_entry_points_reject_empty_queries_in_every_mode(
    method_name, mode, query
):
    """`bypass` skips retrieval, so it skips the RAG minimum — but an empty
    prompt is nothing to send to an LLM, and reaching the provider with one
    only buys an upstream error or an answer to nothing."""
    rag = SimpleNamespace()

    with pytest.raises(EmptyQueryError):
        await getattr(LightRAG, method_name)(
            rag,
            query,
            param=QueryParam(mode=mode),
        )
