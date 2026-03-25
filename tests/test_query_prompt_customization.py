from dataclasses import asdict

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
import lightrag.operate as operate_module
from lightrag.base import QueryContextResult, QueryParam, QueryResult
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


class _DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text)


class _DummyHashingKV:
    def __init__(self, enable_llm_cache: bool = False):
        self.global_config = {"enable_llm_cache": enable_llm_cache}


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


def test_query_param_accepts_prompt_overrides():
    param = QueryParam(prompt_overrides={"query": {"rag_response": "{context_data}"}})
    assert "query" in param.prompt_overrides


def test_lightrag_global_config_carries_prompt_config(tmp_path):
    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        prompt_config={"query": {"rag_response": "{context_data}"}},
    )
    global_config = asdict(rag)
    assert "prompt_config" in global_config
    assert global_config["prompt_config"]["query"]["rag_response"] == "{context_data}"


@pytest.mark.asyncio
async def test_aquery_data_copies_prompt_overrides(tmp_path, monkeypatch):
    rag = _build_rag(tmp_path)

    async def _fake_query_done():
        return None

    captured: dict[str, QueryParam] = {}

    async def _fake_kg_query(
        query,
        chunk_entity_relation_graph,
        entities_vdb,
        relationships_vdb,
        text_chunks,
        param,
        global_config,
        hashing_kv=None,
        system_prompt=None,
        chunks_vdb=None,
    ):
        captured["param"] = param
        return QueryResult(
            content="",
            raw_data={"status": "success", "message": "ok", "data": {}},
        )

    monkeypatch.setattr(rag, "_query_done", _fake_query_done)
    monkeypatch.setattr(lightrag_module, "kg_query", _fake_kg_query)

    param = QueryParam(
        mode="mix",
        prompt_overrides={"query": {"rag_response": "{context_data}"}},
    )
    await rag.aquery_data("test query", param)

    assert captured["param"].prompt_overrides == param.prompt_overrides


@pytest.mark.asyncio
async def test_kg_query_uses_query_prompt_override_in_only_need_prompt_mode(monkeypatch):
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

    param = QueryParam(
        mode="mix",
        only_need_prompt=True,
        prompt_overrides={"query": {"rag_response": "CTX={context_data}"}},
    )
    result = await operate_module.kg_query(
        "what is this",
        knowledge_graph_inst=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_db=None,
        query_param=param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
        },
    )

    assert result is not None
    assert "CTX=context-body" in result.content


@pytest.mark.asyncio
async def test_user_prompt_is_appended_when_custom_template_omits_placeholder(
    monkeypatch,
):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )

    param = QueryParam(
        mode="naive",
        only_need_prompt=True,
        user_prompt="Use bullet points",
        prompt_overrides={"query": {"naive_rag_response": "{content_data}"}},
    )
    result = await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
            "max_total_tokens": 4000,
        },
    )

    assert result is not None
    assert "Use bullet points" in result.content


@pytest.mark.asyncio
async def test_query_cache_hash_changes_with_prompt_override(monkeypatch):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    captured_hashes: list[str] = []

    async def _fake_handle_cache(hashing_kv, args_hash, *args, **kwargs):
        captured_hashes.append(args_hash)
        return "cached", 0

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )
    monkeypatch.setattr(operate_module, "handle_cache", _fake_handle_cache)

    base_config = {
        "llm_model_func": _dummy_llm,
        "tokenizer": _DummyTokenizer(),
        "prompt_config": {},
        "addon_params": {},
        "max_total_tokens": 4000,
    }
    param_a = QueryParam(
        mode="naive",
        prompt_overrides={"query": {"naive_rag_response": "A {content_data}"}},
    )
    param_b = QueryParam(
        mode="naive",
        prompt_overrides={"query": {"naive_rag_response": "B {content_data}"}},
    )

    await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param_a,
        global_config=base_config,
    )
    await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param_b,
        global_config=base_config,
    )

    assert len(captured_hashes) == 2
    assert captured_hashes[0] != captured_hashes[1]


@pytest.mark.asyncio
async def test_bypass_mode_rejects_prompt_overrides(tmp_path):
    rag = _build_rag(tmp_path)
    param = QueryParam(
        mode="bypass",
        prompt_overrides={"query": {"rag_response": "{context_data}"}},
    )

    result = await rag.aquery_llm("test query", param)
    assert result["status"] == "failure"
    assert "bypass mode" in result["message"]


@pytest.mark.asyncio
async def test_query_cache_hash_changes_with_system_prompt_override(monkeypatch):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    captured_hashes: list[str] = []

    async def _fake_handle_cache(hashing_kv, args_hash, *args, **kwargs):
        captured_hashes.append(args_hash)
        return "cached", 0

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )
    monkeypatch.setattr(operate_module, "handle_cache", _fake_handle_cache)

    base_config = {
        "llm_model_func": _dummy_llm,
        "tokenizer": _DummyTokenizer(),
        "prompt_config": {},
        "addon_params": {},
        "max_total_tokens": 4000,
    }
    param = QueryParam(mode="naive")

    await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param,
        global_config=base_config,
        system_prompt="SP1 {content_data}",
    )
    await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param,
        global_config=base_config,
        system_prompt="SP2 {content_data}",
    )

    assert len(captured_hashes) == 2
    assert captured_hashes[0] != captured_hashes[1]


@pytest.mark.asyncio
async def test_kg_query_context_prompt_override_is_applied(monkeypatch):
    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )

    param = QueryParam(
        mode="mix",
        prompt_overrides={
            "query": {
                "kg_query_context": "KG={entities_str}|{relations_str}|{reference_list_str}"
            }
        },
    )
    context_content, _ = await operate_module._build_context_str(
        entities_context=[{"entity_name": "A"}],
        relations_context=[{"src_id": "A", "tgt_id": "B"}],
        merged_chunks=[{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}],
        query="q",
        query_param=param,
        global_config={
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "max_total_tokens": 4000,
        },
    )
    assert "KG=" in context_content


@pytest.mark.asyncio
async def test_naive_query_context_prompt_override_is_applied(monkeypatch):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )

    param = QueryParam(
        mode="naive",
        only_need_context=True,
        prompt_overrides={"query": {"naive_query_context": "NC={text_chunks_str}"}},
    )
    result = await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
            "max_total_tokens": 4000,
        },
    )
    assert result is not None
    assert result.content.startswith("NC=")


@pytest.mark.asyncio
async def test_keywords_prompt_and_examples_overrides_are_applied():
    captured_prompts: list[str] = []

    async def _fake_model_func(prompt: str, **kwargs) -> str:
        captured_prompts.append(prompt)
        return '{"high_level_keywords":["A"],"low_level_keywords":["B"]}'

    param = QueryParam(
        mode="mix",
        model_func=_fake_model_func,
        prompt_overrides={
            "keywords": {
                "keywords_extraction": "Q={query};E={examples};L={language}",
                "keywords_extraction_examples": ["EX-1", "EX-2"],
            }
        },
    )
    hl, ll = await operate_module.extract_keywords_only(
        "hello",
        param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
        },
        hashing_kv=_DummyHashingKV(enable_llm_cache=False),
    )

    assert hl == ["A"]
    assert ll == ["B"]
    assert len(captured_prompts) == 1
    assert "Q=hello" in captured_prompts[0]
    assert "E=EX-1\nEX-2" in captured_prompts[0]


@pytest.mark.asyncio
async def test_keywords_cache_hash_changes_with_prompt_override(monkeypatch):
    captured_hashes: list[str] = []

    async def _fake_handle_cache(hashing_kv, args_hash, *args, **kwargs):
        captured_hashes.append(args_hash)
        return '{"high_level_keywords":["A"],"low_level_keywords":["B"]}', 0

    monkeypatch.setattr(operate_module, "handle_cache", _fake_handle_cache)

    config = {
        "llm_model_func": _dummy_llm,
        "tokenizer": _DummyTokenizer(),
        "prompt_config": {},
        "addon_params": {},
    }
    p1 = QueryParam(
        mode="mix",
        prompt_overrides={"keywords": {"keywords_extraction": "A {query} {examples}"}},
    )
    p2 = QueryParam(
        mode="mix",
        prompt_overrides={"keywords": {"keywords_extraction": "B {query} {examples}"}},
    )

    await operate_module.extract_keywords_only("hello", p1, config, hashing_kv=None)
    await operate_module.extract_keywords_only("hello", p2, config, hashing_kv=None)

    assert len(captured_hashes) == 2
    assert captured_hashes[0] != captured_hashes[1]


@pytest.mark.asyncio
async def test_request_time_prompt_override_rejects_non_query_keywords_family(monkeypatch):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )

    param = QueryParam(
        mode="naive",
        prompt_overrides={"entity_extraction": {"system_prompt": "bad"}},
    )
    with pytest.raises(ValueError):
        await operate_module.naive_query(
            "test query",
            chunks_vdb=None,
            query_param=param,
            global_config={
                "llm_model_func": _dummy_llm,
                "tokenizer": _DummyTokenizer(),
                "prompt_config": {},
                "addon_params": {},
                "max_total_tokens": 4000,
            },
        )


@pytest.mark.asyncio
async def test_prompt_overrides_not_mutated_in_place(monkeypatch):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )

    overrides = {"query": {"naive_rag_response": "{content_data}"}}
    snapshot = {"query": {"naive_rag_response": "{content_data}"}}
    param = QueryParam(mode="naive", only_need_prompt=True, prompt_overrides=overrides)

    await operate_module.naive_query(
        "test query",
        chunks_vdb=None,
        query_param=param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {},
            "addon_params": {},
            "max_total_tokens": 4000,
        },
    )

    assert overrides == snapshot


@pytest.mark.asyncio
async def test_request_prompt_overrides_non_none_invalid_type_raises(monkeypatch):
    async def _fake_get_vector_context(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    async def _fake_process_chunks_unified(*args, **kwargs):
        return [{"content": "chunk-a", "file_path": "doc-a", "chunk_id": "c1"}]

    monkeypatch.setattr(operate_module, "_get_vector_context", _fake_get_vector_context)
    monkeypatch.setattr(
        operate_module, "process_chunks_unified", _fake_process_chunks_unified
    )

    param = QueryParam(mode="naive")
    param.prompt_overrides = False  # type: ignore[assignment]

    with pytest.raises(ValueError):
        await operate_module.naive_query(
            "test query",
            chunks_vdb=None,
            query_param=param,
            global_config={
                "llm_model_func": _dummy_llm,
                "tokenizer": _DummyTokenizer(),
                "prompt_config": {},
                "addon_params": {},
                "max_total_tokens": 4000,
            },
        )


@pytest.mark.asyncio
async def test_request_prompt_overrides_take_precedence_over_global_prompt_config(
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

    param = QueryParam(
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
        query_param=param,
        global_config={
            "llm_model_func": _dummy_llm,
            "tokenizer": _DummyTokenizer(),
            "prompt_config": {"query": {"rag_response": "GLOBAL={context_data}"}},
            "addon_params": {},
        },
    )

    assert result is not None
    assert "REQUEST=context-body" in result.content
    assert "GLOBAL=context-body" not in result.content
