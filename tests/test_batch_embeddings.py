"""
Tests for batch embedding pre-computation in _perform_kg_search().

Verifies that kg_query batches all needed embeddings (query, ll_keywords,
hl_keywords) into a single embedding API call instead of 3 sequential calls.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag.base import QueryParam


def _make_mock_embedding_func(dim=1536):
    """Create a mock async embedding function that returns distinct vectors per input."""

    async def _embed(texts, **kwargs):
        return np.array([np.full(dim, i + 1, dtype=np.float32) for i in range(len(texts))])

    mock = AsyncMock(side_effect=_embed)
    return mock


def _make_mock_kv_storage(embedding_func, global_config=None):
    mock = MagicMock()
    mock.embedding_func = embedding_func
    mock.global_config = global_config or {"kg_chunk_pick_method": "VECTOR"}
    return mock


def _make_mock_vdb():
    """Create a mock VDB whose query() records the query_embedding it receives."""
    mock = AsyncMock()
    mock.query = AsyncMock(return_value=[])
    mock.cosine_better_than_threshold = 0.2
    return mock


def _make_mock_graph():
    mock = AsyncMock()
    return mock


@pytest.mark.offline
@pytest.mark.asyncio
async def test_hybrid_mode_batches_embeddings():
    """In hybrid mode with both keywords, embedding_func should be called exactly once."""
    from lightrag.operate import _perform_kg_search

    embed_func = _make_mock_embedding_func()
    text_chunks_db = _make_mock_kv_storage(embed_func)
    entities_vdb = _make_mock_vdb()
    relationships_vdb = _make_mock_vdb()
    knowledge_graph = _make_mock_graph()

    query_param = QueryParam(mode="hybrid", top_k=5)

    await _perform_kg_search(
        query="test query",
        ll_keywords="entity1, entity2",
        hl_keywords="theme1, theme2",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
    )

    # The embedding function should be called exactly once with all 3 texts batched
    assert embed_func.call_count == 1, (
        f"Expected 1 batched embedding call, got {embed_func.call_count}"
    )
    call_args = embed_func.call_args[0][0]
    assert len(call_args) == 3, f"Expected 3 texts in batch, got {len(call_args)}"
    assert call_args == ["test query", "entity1, entity2", "theme1, theme2"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_hybrid_mode_passes_embeddings_to_vdbs():
    """Pre-computed embeddings should be forwarded to entities and relationships VDB queries."""
    from lightrag.operate import _perform_kg_search

    embed_func = _make_mock_embedding_func()
    text_chunks_db = _make_mock_kv_storage(embed_func)
    entities_vdb = _make_mock_vdb()
    relationships_vdb = _make_mock_vdb()
    knowledge_graph = _make_mock_graph()

    query_param = QueryParam(mode="hybrid", top_k=5)

    await _perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="theme keywords",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
    )

    # entities_vdb.query should receive ll_embedding (index 1 → all 2s)
    entities_call = entities_vdb.query.call_args
    assert entities_call is not None, "entities_vdb.query was not called"
    ll_embedding = entities_call.kwargs.get("query_embedding")
    assert ll_embedding is not None, "ll_embedding was not passed to entities_vdb.query"
    assert np.all(ll_embedding == 2.0), f"Expected ll_embedding=[2,2,...], got {ll_embedding[:3]}"

    # relationships_vdb.query should receive hl_embedding (index 2 → all 3s)
    rel_call = relationships_vdb.query.call_args
    assert rel_call is not None, "relationships_vdb.query was not called"
    hl_embedding = rel_call.kwargs.get("query_embedding")
    assert hl_embedding is not None, "hl_embedding was not passed to relationships_vdb.query"
    assert np.all(hl_embedding == 3.0), f"Expected hl_embedding=[3,3,...], got {hl_embedding[:3]}"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_local_mode_batches_query_and_ll():
    """In local mode, should batch query + ll_keywords (2 texts, not 3)."""
    from lightrag.operate import _perform_kg_search

    embed_func = _make_mock_embedding_func()
    text_chunks_db = _make_mock_kv_storage(embed_func)
    entities_vdb = _make_mock_vdb()
    relationships_vdb = _make_mock_vdb()
    knowledge_graph = _make_mock_graph()

    query_param = QueryParam(mode="local", top_k=5)

    await _perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
    )

    assert embed_func.call_count == 1
    call_args = embed_func.call_args[0][0]
    assert len(call_args) == 2, f"Expected 2 texts (query + ll), got {len(call_args)}"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_embedding_failure_falls_back_gracefully():
    """If batch embedding fails, VDB queries should still work (fallback to individual calls)."""
    from lightrag.operate import _perform_kg_search

    embed_func = AsyncMock(side_effect=RuntimeError("API error"))
    text_chunks_db = _make_mock_kv_storage(embed_func)
    entities_vdb = _make_mock_vdb()
    relationships_vdb = _make_mock_vdb()
    knowledge_graph = _make_mock_graph()

    query_param = QueryParam(mode="hybrid", top_k=5)

    # Should not raise — graceful degradation
    await _perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="theme keywords",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
    )

    # VDB queries should still be called (with query_embedding=None fallback)
    entities_call = entities_vdb.query.call_args
    assert entities_call is not None
    assert entities_call.kwargs.get("query_embedding") is None

    rel_call = relationships_vdb.query.call_args
    assert rel_call is not None
    assert rel_call.kwargs.get("query_embedding") is None
