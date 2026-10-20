from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam, QueryResult
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["mix", "naive"])
async def test_data_query_preserves_cache_friendly_budget_setting(mode, monkeypatch):
    captured = []

    async def query(*args, **kwargs):
        captured.append(args[2] if mode == "naive" else args[5])
        return QueryResult(content="context", raw_data={"data": {}})

    monkeypatch.setattr("lightrag.lightrag.kg_query", query)
    monkeypatch.setattr("lightrag.lightrag.naive_query", query)
    rag = SimpleNamespace(
        _build_global_config=lambda: {},
        _query_done=AsyncMock(),
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks=None,
        chunks_vdb=None,
        llm_response_cache=None,
    )
    param = QueryParam(mode=mode, cache_friendly_prompt=True)
    await LightRAG.aquery_data(rag, "What is this about?", param)
    assert captured[0].cache_friendly_prompt is True
    assert captured[0].only_need_context is True
    assert param.only_need_context is False


@pytest.mark.asyncio
async def test_bypass_ignores_cache_friendly_layout():
    model = AsyncMock(return_value="Direct answer")
    rag = SimpleNamespace(
        _build_global_config=lambda: {"role_llm_funcs": {"query": model}}
    )
    param = QueryParam(
        mode="bypass",
        cache_friendly_prompt=True,
        conversation_history=[{"role": "user", "content": "Earlier"}],
    )
    result = await LightRAG.aquery_llm(
        rag, "Direct question", param, system_prompt="Static policy"
    )
    assert result["llm_response"]["content"] == "Direct answer"
    assert model.await_args.args == ("Direct question",)
    assert model.await_args.kwargs["system_prompt"] == "Static policy"
    assert model.await_args.kwargs["history_messages"] == param.conversation_history
