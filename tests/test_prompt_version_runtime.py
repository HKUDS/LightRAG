from copy import deepcopy

import numpy as np
import pytest

import lightrag.operate as operate_module
from lightrag.base import QueryContextResult, QueryParam
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


class _DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text)


def _build_rag(tmp_path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
    )


def test_resolve_active_prompt_groups_returns_none_without_active_version(tmp_path):
    rag = _build_rag(tmp_path)

    assert rag._resolve_active_prompt_groups() == {
        "indexing": None,
        "retrieval": None,
    }


def test_resolve_active_prompt_groups_returns_active_retrieval_payload(tmp_path):
    rag = _build_rag(tmp_path)
    registry = rag.prompt_version_store.initialize(locale="zh")
    active_version = registry["retrieval"]["versions"][0]

    rag.prompt_version_store.activate_version("retrieval", active_version["version_id"])

    assert rag._resolve_active_prompt_groups()["retrieval"] == active_version["payload"]


def test_resolve_indexing_runtime_addon_params_falls_back_to_env_values():
    addon_params = operate_module._resolve_indexing_runtime_addon_params(
        {
            "addon_params": {
                "language": "English",
                "entity_types": ["Person"],
            },
            "active_prompt_groups": {"indexing": None},
        }
    )

    assert addon_params == {
        "language": "English",
        "entity_types": ["Person"],
    }


def test_resolve_indexing_runtime_addon_params_prefers_active_payload():
    addon_params = operate_module._resolve_indexing_runtime_addon_params(
        {
            "addon_params": {
                "language": "English",
                "entity_types": ["Person"],
            },
            "active_prompt_groups": {
                "indexing": {
                    "summary_language": "Chinese",
                    "entity_types": ["Organization"],
                }
            },
        }
    )

    assert addon_params == {
        "language": "Chinese",
        "entity_types": ["Organization"],
    }


@pytest.mark.asyncio
async def test_kg_query_uses_active_retrieval_version_before_request_override(
    monkeypatch,
):
    async def _fake_get_keywords_from_query(*args, **kwargs):
        return ["hl"], ["ll"]

    async def _fake_build_query_context(*args, **kwargs):
        return QueryContextResult(
            context="context-body",
            raw_data={"status": "success", "message": "ok", "data": {}},
        )

    monkeypatch.setattr(
        operate_module, "get_keywords_from_query", _fake_get_keywords_from_query
    )
    monkeypatch.setattr(operate_module, "_build_query_context", _fake_build_query_context)

    result = await operate_module.kg_query(
        "what is this",
        knowledge_graph_inst=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_db=None,
        query_param=QueryParam(mode="mix", only_need_prompt=True),
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
            "active_prompt_groups": {
                "retrieval": {"query": {"rag_response": "ACTIVE={context_data}"}}
            },
        },
    )

    assert result is not None
    assert "ACTIVE=context-body" in result.content


@pytest.mark.asyncio
async def test_request_prompt_overrides_still_win_over_active_retrieval_version(
    monkeypatch,
):
    async def _fake_get_keywords_from_query(*args, **kwargs):
        return ["hl"], ["ll"]

    async def _fake_build_query_context(*args, **kwargs):
        return QueryContextResult(
            context="context-body",
            raw_data={"status": "success", "message": "ok", "data": {}},
        )

    monkeypatch.setattr(
        operate_module, "get_keywords_from_query", _fake_get_keywords_from_query
    )
    monkeypatch.setattr(operate_module, "_build_query_context", _fake_build_query_context)

    query_param = QueryParam(
        mode="mix",
        only_need_prompt=True,
        prompt_overrides={"query": {"rag_response": "REQUEST={context_data}"}},
    )
    result = await operate_module.kg_query(
        "what is this",
        knowledge_graph_inst=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_db=None,
        query_param=query_param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
            "active_prompt_groups": {
                "retrieval": {"query": {"rag_response": "ACTIVE={context_data}"}}
            },
        },
    )

    assert result is not None
    assert "REQUEST=context-body" in result.content
