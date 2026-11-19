"""Final filtering must use the same no-results path as an empty search."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _CharacterTokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def query_setup(monkeypatch):
    chunk = {
        "id": "chunk-1",
        "chunk_id": "chunk-1",
        "content": "Retrieved evidence.",
        "file_path": "source.md",
    }
    search = {
        "final_entities": [],
        "final_relations": [],
        "vector_chunks": [chunk],
        "chunk_tracking": {"chunk-1": {"source": "C", "frequency": 1, "order": 1}},
        "query_embedding": None,
    }
    # Replace search transport only; keep all truncation and context stages real.
    monkeypatch.setattr(
        "lightrag.operate._perform_kg_search", AsyncMock(return_value=search)
    )
    model = AsyncMock(return_value="Grounded answer.")
    reranker = AsyncMock(return_value=[{"index": 0, "relevance_score": 0.1}])
    config = {
        "tokenizer": Tokenizer("character", _CharacterTokenizer()),
        "role_llm_funcs": {"query": model},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.5,
        "rerank_model_func": reranker,
    }
    cache = SimpleNamespace(
        global_config={"enable_llm_cache": True},
        get_by_id=AsyncMock(return_value=None),
        upsert=AsyncMock(),
    )
    rag = SimpleNamespace(
        _build_global_config=lambda: config,
        _query_done=AsyncMock(),
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks=SimpleNamespace(global_config=config),
        chunks_vdb=SimpleNamespace(
            cosine_better_than_threshold=0.0, query=AsyncMock(return_value=[chunk])
        ),
        llm_response_cache=cache,
    )
    return rag, search, model, cache, reranker


async def _query(rag, mode, **options):
    slot = "content_data" if mode == "naive" else "context_data"
    return await LightRAG.aquery_llm(
        rag,
        "Find evidence",
        param=QueryParam(mode=mode, ll_keywords=["evidence"], **options),
        system_prompt="Answer using: {" + slot + "}",
    )


@pytest.mark.parametrize("mode", ["naive", "mix"])
@pytest.mark.parametrize("cause", ["rerank", "budget"])
@pytest.mark.parametrize("output", ["answer", "stream", "context", "prompt"])
async def test_all_final_evidence_removed_returns_no_results(
    query_setup, mode, cause, output
):
    rag, _, model, cache, reranker = query_setup
    result = await _query(
        rag,
        mode,
        enable_rerank=cause == "rerank",
        max_total_tokens=4096 if cause == "rerank" else 120,
        stream=output == "stream",
        only_need_context=output == "context",
        only_need_prompt=output == "prompt",
    )

    assert result["status"] == "failure"
    assert result["metadata"]["failure_reason"] == "no_results"
    assert result["llm_response"] == {
        "content": PROMPTS["fail_response"],
        "response_iterator": None,
        "is_streaming": False,
        "llm_generated": False,
    }
    model.assert_not_awaited()
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()
    if cause == "rerank":
        reranker.assert_awaited_once()


@pytest.mark.parametrize("mode", ["naive", "mix"])
@pytest.mark.parametrize("cause", ["rerank", "budget"])
async def test_data_only_query_reports_final_empty_evidence(query_setup, mode, cause):
    rag, _, model, cache, _ = query_setup
    result = await LightRAG.aquery_data(
        rag,
        "Find evidence",
        param=QueryParam(
            mode=mode,
            ll_keywords=["evidence"],
            enable_rerank=cause == "rerank",
            max_total_tokens=4096 if cause == "rerank" else 120,
        ),
    )

    assert result["status"] == "failure"
    assert result["metadata"]["failure_reason"] == "no_results"
    assert result["data"] == {}
    model.assert_not_awaited()
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()


@pytest.mark.parametrize("mode", ["hybrid", "mix"])
@pytest.mark.parametrize("evidence", ["entity", "relationship"])
async def test_filtered_chunks_preserve_graph_evidence(query_setup, mode, evidence):
    rag, search, model, _, _ = query_setup
    if evidence == "entity":
        search["final_entities"] = [
            {"entity_name": "Evidence", "description": "A surviving graph fact."}
        ]
    else:
        search["final_relations"] = [
            {"src_id": "Evidence", "tgt_id": "Fact", "description": "Related facts."}
        ]

    result = await _query(rag, mode, enable_rerank=True, max_total_tokens=4096)

    assert result["status"] == "success"
    assert result["data"]["chunks"] == []
    graph_key = "entities" if evidence == "entity" else "relationships"
    assert len(result["data"][graph_key]) == 1
    assert result["llm_response"]["content"] == "Grounded answer."
    assert result["llm_response"]["llm_generated"] is True
    model.assert_awaited_once()


@pytest.mark.parametrize("mode", ["naive", "mix"])
async def test_retained_chunks_still_generate_answers(query_setup, mode):
    rag, _, model, _, _ = query_setup
    result = await _query(rag, mode, enable_rerank=False, max_total_tokens=4096)

    assert result["status"] == "success"
    assert len(result["data"]["chunks"]) == 1
    assert result["llm_response"]["llm_generated"] is True
    model.assert_awaited_once()
