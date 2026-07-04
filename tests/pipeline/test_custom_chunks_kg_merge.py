"""Regression tests for issue #3352.

``ainsert_custom_chunks`` must build the KG from the extracted entities and
relationships — i.e. capture the ``_process_extract_entities`` result and pass
it to ``merge_nodes_and_edges`` (mirroring the file pipeline) — instead of
firing extraction inside an ``asyncio.gather`` and discarding its return value.

It must also pass ``pipeline_status`` / ``pipeline_status_lock`` down to
extraction so the ``except`` block does not attempt ``async with None`` (which
would raise a ``TypeError`` that masks the real extraction error).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

CHUNKS = [
    "Marie Curie discovered polonium and radium.",
    "Pierre Curie collaborated with Marie Curie on radioactivity research.",
]


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"custom-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status(rag.workspace)
    return rag


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_merges_extracted_entities(tmp_path, monkeypatch):
    """The extracted chunk_results must be handed to merge_nodes_and_edges."""
    rag = await _build_rag(tmp_path)
    try:
        sentinel_results = [("nodes", "edges")]

        async def _fake_extract(chunks, *args, **kwargs):
            # A non-empty extraction result (real extraction is exercised
            # elsewhere; here we only assert the control flow around it).
            return sentinel_results

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        merge_spy = AsyncMock()
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_spy)

        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS), text_chunks=CHUNKS, doc_id="doc-merge"
        )

        merge_spy.assert_awaited_once()
        kwargs = merge_spy.await_args.kwargs
        assert kwargs["chunk_results"] is sentinel_results
        assert kwargs["doc_id"] == "doc-merge"
        assert kwargs["knowledge_graph_inst"] is rag.chunk_entity_relation_graph
        assert kwargs["entity_vdb"] is rag.entities_vdb
        assert kwargs["relationships_vdb"] is rag.relationships_vdb
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_skips_merge_when_extraction_empty(
    tmp_path, monkeypatch
):
    """No entities extracted -> merge_nodes_and_edges must not be called."""
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return []

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        merge_spy = AsyncMock()
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_spy)

        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS), text_chunks=CHUNKS, doc_id="doc-empty"
        )

        merge_spy.assert_not_awaited()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_process_extract_entities_surfaces_real_error_without_lock(
    tmp_path, monkeypatch
):
    """Secondary defect: with pipeline_status_lock=None the except block must
    surface the real extraction error, not a TypeError from ``async with None``.
    """
    rag = await _build_rag(tmp_path)
    try:

        async def _boom(*args, **kwargs):
            raise ValueError("extraction blew up")

        monkeypatch.setattr(lightrag_module, "extract_entities", _boom)

        # Called with the default pipeline_status / pipeline_status_lock = None.
        with pytest.raises(ValueError, match="extraction blew up"):
            await rag._process_extract_entities({"chunk-1": {"content": "x"}})
    finally:
        await rag.finalize_storages()
