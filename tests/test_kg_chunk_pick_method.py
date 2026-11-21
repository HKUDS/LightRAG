"""Tests for ``kg_chunk_pick_method`` validation, normalization, and VECTOR->WEIGHT fallback."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import (
    _find_related_text_unit_from_entities,
    _find_related_text_unit_from_relations,
    _perform_kg_search,
    kg_query,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _FakeTokenizer:
    """Minimal byte-level tokenizer stub for LightRAG construction and cache tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _unused_llm(*_args, **_kwargs) -> str:
    return ""


async def _unused_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8), dtype=np.float32)


def _build_rag(tmp_path, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_unused_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_unused_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _FakeTokenizer()),
        **kwargs,
    )


def _text_chunks_db(pick_method, *, embedding_func=None) -> SimpleNamespace:
    return SimpleNamespace(
        global_config={
            "kg_chunk_pick_method": pick_method,
            "related_chunk_number": 2,
        },
        embedding_func=embedding_func,
        get_by_ids=AsyncMock(
            side_effect=lambda chunk_ids: [
                {"content": f"content-{chunk_id}"} for chunk_id in chunk_ids
            ]
        ),
    )


async def _select_from_entities(
    text_chunks_db: SimpleNamespace, *, query, chunks_vdb
) -> list[dict]:
    return await _find_related_text_unit_from_entities(
        [{"entity_name": "entity-1", "source_id": "chunk-1"}],
        QueryParam(),
        text_chunks_db,
        SimpleNamespace(),
        query=query,
        chunks_vdb=chunks_vdb,
    )


async def _select_from_relations(
    text_chunks_db: SimpleNamespace, *, query, chunks_vdb
) -> list[dict]:
    return await _find_related_text_unit_from_relations(
        [{"src_id": "src-1", "tgt_id": "tgt-1", "source_id": "chunk-1"}],
        QueryParam(),
        text_chunks_db,
        entity_chunks=[],
        query=query,
        chunks_vdb=chunks_vdb,
    )


@pytest.mark.parametrize(
    "invalid_value",
    ["WEIGTH", "", None],
    ids=["typo", "empty_string", "none"],
)
def test_kg_chunk_pick_method_rejects_invalid_constructor_value(
    tmp_path, invalid_value
):
    """A mistyped or non-string method fails at construction instead of dropping all KG chunks."""
    with pytest.raises(ValueError, match="KG_CHUNK_PICK_METHOD"):
        _build_rag(tmp_path, kg_chunk_pick_method=invalid_value)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("vector", "VECTOR"), (" Weight ", "WEIGHT")],
    ids=["lowercase_vector", "padded_mixed_weight"],
)
def test_kg_chunk_pick_method_canonicalized_at_construction(tmp_path, given, expected):
    rag = _build_rag(tmp_path, kg_chunk_pick_method=given)

    assert rag.kg_chunk_pick_method == expected


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("   ", "VECTOR"), (" weight ", "WEIGHT")],
    ids=["blank_env_defaults_to_vector", "lowercase_env_canonicalized"],
)
def test_kg_chunk_pick_method_reads_live_env_default(
    tmp_path, monkeypatch, env_value, expected
):
    """Blank env values fall back to VECTOR; valid lowercase env values normalize."""
    monkeypatch.setenv("KG_CHUNK_PICK_METHOD", env_value)

    rag = _build_rag(tmp_path)

    assert rag.kg_chunk_pick_method == expected


def test_kg_chunk_pick_method_rejects_invalid_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_CHUNK_PICK_METHOD", "WEIGTH")

    with pytest.raises(ValueError, match="KG_CHUNK_PICK_METHOD"):
        _build_rag(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("select", "query", "chunks_vdb", "expected_missing"),
    [
        (_select_from_entities, "what is lightrag?", None, "chunks_vdb"),
        (_select_from_entities, "   ", SimpleNamespace(), "query"),
        (_select_from_entities, None, None, "query and chunks_vdb"),
        (_select_from_relations, "what is lightrag?", None, "chunks_vdb"),
    ],
    ids=[
        "entities_missing_chunks_vdb",
        "entities_blank_query",
        "entities_missing_both",
        "relations_missing_chunks_vdb",
    ],
)
async def test_vector_pick_method_falls_back_to_weight_when_prerequisites_missing(
    monkeypatch, lightrag_log_records, select, query, chunks_vdb, expected_missing
):
    """VECTOR mode without chunks_vdb or query must fall back to WEIGHT rather than returning []."""
    vector_calls: list[dict] = []

    async def _forbidden_vector_picker(**kwargs):
        vector_calls.append(kwargs)
        raise AssertionError(
            "pick_by_vector_similarity must not run without prerequisites"
        )

    monkeypatch.setattr(
        "lightrag.operate.pick_by_vector_similarity", _forbidden_vector_picker
    )

    result = await select(
        _text_chunks_db("VECTOR", embedding_func=AsyncMock()),
        query=query,
        chunks_vdb=chunks_vdb,
    )

    assert vector_calls == []
    assert [chunk["chunk_id"] for chunk in result] == ["chunk-1"]
    warning_messages = [
        record.getMessage()
        for record in lightrag_log_records
        if record.levelno == logging.WARNING
    ]
    assert any(
        expected_missing in msg and "WEIGHT" in msg for msg in warning_messages
    ), f"Expected warning naming {expected_missing!r}, got {warning_messages!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "select",
    [_select_from_entities, _select_from_relations],
    ids=["entities", "relations"],
)
async def test_selectors_normalize_lowercase_method(select):
    """Direct operate callers normalize lowercase config values instead of returning []."""
    chunks = await select(
        _text_chunks_db(" weight "), query="q", chunks_vdb=SimpleNamespace()
    )

    assert [chunk["chunk_id"] for chunk in chunks] == ["chunk-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "select",
    [_select_from_entities, _select_from_relations],
    ids=["entities", "relations"],
)
async def test_selectors_reject_invalid_method(select):
    """Direct operate callers fail fast on unknown values instead of returning []."""
    with pytest.raises(ValueError, match="KG_CHUNK_PICK_METHOD"):
        await select(_text_chunks_db("WEIGTH"), query="q", chunks_vdb=SimpleNamespace())


@pytest.mark.asyncio
async def test_perform_kg_search_normalizes_lowercase_vector_for_batch_embedding():
    """Lowercase 'vector' in text_chunks_db.global_config still triggers query batch embedding."""
    embed_calls: list[list[str]] = []

    async def _embed(texts, **_kwargs):
        embed_calls.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    vdb = AsyncMock()
    vdb.query = AsyncMock(return_value=[])
    vdb.cosine_better_than_threshold = 0.2

    await _perform_kg_search(
        query="test query",
        ll_keywords="entity",
        hl_keywords="",
        knowledge_graph_inst=AsyncMock(),
        entities_vdb=vdb,
        relationships_vdb=vdb,
        text_chunks_db=_text_chunks_db("vector", embedding_func=_embed),
        query_param=QueryParam(mode="local", top_k=5),
    )

    assert embed_calls == [["test query", "entity"]]


@pytest.mark.asyncio
async def test_kg_query_cache_key_canonicalizes_kg_chunk_pick_method(monkeypatch):
    """'WEIGHT' and ' weight ' produce the same answer-cache key instead of splitting the cache."""
    calls = 0

    async def _model(*_args, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        return f"answer-{calls}"

    class _Cache:
        """Minimal in-memory KV cache stub."""

        def __init__(self) -> None:
            self.global_config = {"enable_llm_cache": True}
            self.store: dict[str, dict] = {}

        async def get_by_id(self, key: str):
            return self.store.get(key)

        async def upsert(self, entries: dict[str, dict]):
            self.store.update(entries)

    async def _fake_keywords(*_args, **_kwargs):
        return "", "topic"

    async def _fake_context(*_args, **_kwargs):
        return QueryContextResult(context="ctx", raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", _fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", _fake_context)

    config = {
        "tokenizer": Tokenizer("fake", _FakeTokenizer()),
        "role_llm_funcs": {"query": _model},
        "addon_params": {"language": "en"},
        "related_chunk_number": 1,
        "kg_chunk_pick_method": "WEIGHT",
    }
    cache = _Cache()
    text_chunks = SimpleNamespace(global_config=config)
    param = QueryParam(mode="local", enable_rerank=False, ll_keywords=["topic"])

    first = await kg_query(
        "query", None, None, None, text_chunks, param, config, hashing_kv=cache
    )
    config["kg_chunk_pick_method"] = " weight "
    second = await kg_query(
        "query", None, None, None, text_chunks, param, config, hashing_kv=cache
    )

    assert first.content == second.content == "answer-1"
    assert calls == 1
