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
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

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

        # A no-entity doc is still completed (PROCESSED), not left PROCESSING.
        status = await rag.doc_status.get_by_id("doc-empty")
        assert status is not None
        assert status["status"] == DocStatus.PROCESSED
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_retryable_after_merge_failure(
    tmp_path, monkeypatch
):
    """A KG merge failure must leave the insert retryable without leaking state.

    Mirroring the file pipeline, ``ainsert_custom_chunks`` marks the doc FAILED
    and DISCARDS (never flushes) the pending KG/base buffers on failure — it
    does not delete chunks. So:
    - the failed attempt leaves the doc FAILED (not PROCESSED);
    - the chunks survive (a later partial-KG source_id stays resolvable);
    - a retry (gated on doc_status, not on full_docs presence) reaches
      ``merge_nodes_and_edges`` again and, on success, marks the doc PROCESSED.
    """
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return [("nodes", "edges")]

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)

        async def _boom_merge(*args, **kwargs):
            raise RuntimeError("transient merge failure")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", _boom_merge)

        # First attempt: the merge fails and the error surfaces to the caller.
        with pytest.raises(RuntimeError, match="transient merge failure"):
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(CHUNKS),
                text_chunks=CHUNKS,
                doc_id="doc-retry",
            )

        # The failed attempt is recorded FAILED, not PROCESSED.
        status = await rag.doc_status.get_by_id("doc-retry")
        assert status is not None
        assert status["status"] == DocStatus.FAILED

        # Chunks are NOT deleted — the anti-orphan invariant. Deleting them (the
        # old delete-rollback) is what left partial-KG source_ids dangling.
        chunk_key = compute_mdhash_id(CHUNKS[0], prefix="chunk-")
        assert await rag.text_chunks.get_by_id(chunk_key) is not None

        # This mock wrote nothing to the graph and the failure path discards
        # rather than flushes, so no KG node is committed here.
        assert await rag.chunk_entity_relation_graph.get_node("Marie Curie") is None

        # Second attempt with a healthy merge must actually rebuild the KG,
        # not bail out at a duplicate check.
        merge_spy = AsyncMock()
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_spy)

        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS),
            text_chunks=CHUNKS,
            doc_id="doc-retry",
        )

        merge_spy.assert_awaited_once()

        # A successful retry marks the doc PROCESSED.
        status = await rag.doc_status.get_by_id("doc-retry")
        assert status is not None
        assert status["status"] == DocStatus.PROCESSED
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_partial_merge_is_not_orphaned(
    tmp_path, monkeypatch
):
    """A merge that writes partial graph state then fails must not orphan it.

    ``merge_nodes_and_edges`` does not roll back its own partial writes, so a
    caller that deleted chunks would leave those partial nodes' ``source_id``
    pointing at deleted chunks (the leak a Codex review confirmed on real
    merges). The doc_status model instead keeps chunks intact: the failed
    attempt is FAILED, the chunk survives so any partial node's ``source_id``
    still resolves, and a healthy retry rebuilds the KG idempotently.

    We deliberately do NOT assert the partial node is absent after the failure:
    on the in-memory NetworkX backend, ``_discard_pending_index_ops`` does not
    roll back already-materialized writes. The guarantee is "no dangling
    source_id + idempotent reprocess (+ nothing flushed to disk)", not "zero
    partial nodes in memory".
    """
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return [("nodes", "edges")]

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)

        chunk_key = compute_mdhash_id(CHUNKS[0], prefix="chunk-")

        async def _partial_then_boom(*args, **kwargs):
            # Write one graph node (as a real merge would), then fail.
            await rag.chunk_entity_relation_graph.upsert_node(
                "ENT",
                {
                    "entity_id": "ENT",
                    "entity_type": "PERSON",
                    "description": "partial",
                    "source_id": chunk_key,
                    "file_path": "unknown_source",
                },
            )
            raise RuntimeError("transient merge failure")

        monkeypatch.setattr(
            lightrag_module, "merge_nodes_and_edges", _partial_then_boom
        )

        with pytest.raises(RuntimeError, match="transient merge failure"):
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(CHUNKS),
                text_chunks=CHUNKS,
                doc_id="doc-partial",
            )

        # FAILED, and the chunk survives — so the partial node's source_id
        # resolves to a live chunk instead of dangling.
        status = await rag.doc_status.get_by_id("doc-partial")
        assert status is not None
        assert status["status"] == DocStatus.FAILED
        assert await rag.text_chunks.get_by_id(chunk_key) is not None

        # A healthy retry rebuilds the KG (reaches merge again).
        merge_spy = AsyncMock()
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_spy)
        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS),
            text_chunks=CHUNKS,
            doc_id="doc-partial",
        )
        merge_spy.assert_awaited_once()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_retry_purges_prior_chunks_and_kg(
    tmp_path, monkeypatch
):
    """Rebuilding a FAILED doc must purge the prior attempt's chunks + KG first.

    The retry must call _purge_doc_chunks_and_kg with the doc's stored
    chunks_list before re-extracting, mirroring the file pipeline's resume purge,
    so a completed-then-not-marked-PROCESSED merge rebuilds cleanly.

    Scope: this asserts purge is invoked with the right args. It does NOT cover
    the partial-merge case (merge writes some nodes then fails before its
    full_entities index row is written) — there purge finds no index and the
    retry re-merges, accumulating descriptions. That is a merge_nodes_and_edges
    limitation shared with the pipeline, tracked separately; see the note in
    ainsert_custom_chunks.
    """
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return [("nodes", "edges")]

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)

        async def _boom_merge(*args, **kwargs):
            raise RuntimeError("transient merge failure")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", _boom_merge)

        # First attempt fails and stores the doc's chunks_list under FAILED.
        with pytest.raises(RuntimeError, match="transient merge failure"):
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(CHUNKS),
                text_chunks=CHUNKS,
                doc_id="doc-purge",
            )

        status = await rag.doc_status.get_by_id("doc-purge")
        assert status is not None
        assert status["status"] == DocStatus.FAILED
        stored_chunks = status["chunks_list"]
        assert stored_chunks

        # On retry, the prior chunks_list must be purged before rebuild.
        purge_spy = AsyncMock()
        monkeypatch.setattr(rag, "_purge_doc_chunks_and_kg", purge_spy)
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", AsyncMock())

        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS),
            text_chunks=CHUNKS,
            doc_id="doc-purge",
        )

        purge_spy.assert_awaited_once()
        called_doc_id, called_chunks = purge_spy.await_args.args
        assert called_doc_id == "doc-purge"
        assert set(called_chunks) == set(stored_chunks)

        # And the doc completes.
        status = await rag.doc_status.get_by_id("doc-purge")
        assert status is not None
        assert status["status"] == DocStatus.PROCESSED
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_purge_failure_preserves_prior_chunks_list(
    tmp_path, monkeypatch
):
    """A purge failure on retry must preserve the prior chunks_list anchor.

    Purge runs BEFORE the PROCESSING upsert that overwrites doc_status with the
    new chunks_list. So if purge fails, doc_status still holds the PRIOR
    chunks_list — the next retry can still purge the old chunks/KG. If the
    PROCESSING write happened first (the bug), a purge failure would strand the
    old chunks/KG (a later retry would purge by the new ids and never reach them).
    """
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return [("nodes", "edges")]

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)

        # First attempt fails -> FAILED with its chunks_list (the "prior" anchor).
        async def _boom_merge(*args, **kwargs):
            raise RuntimeError("merge failure")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", _boom_merge)
        with pytest.raises(RuntimeError, match="merge failure"):
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(CHUNKS),
                text_chunks=CHUNKS,
                doc_id="doc-anchor",
            )
        prior_status = await rag.doc_status.get_by_id("doc-anchor")
        assert prior_status is not None
        prior_chunks = prior_status["chunks_list"]
        assert prior_chunks

        # Retry with DIFFERENT chunk text (new chunks_list differs) and make the
        # purge fail. The prior chunks_list must NOT be overwritten.
        async def _boom_purge(*args, **kwargs):
            raise RuntimeError("purge failure")

        monkeypatch.setattr(rag, "_purge_doc_chunks_and_kg", _boom_purge)

        new_chunks = ["An entirely different chunk about another subject."]
        with pytest.raises(RuntimeError, match="purge failure"):
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(new_chunks),
                text_chunks=new_chunks,
                doc_id="doc-anchor",
            )

        after = await rag.doc_status.get_by_id("doc-anchor")
        assert after is not None
        assert after["chunks_list"] == prior_chunks
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_first_insert_does_not_purge(tmp_path, monkeypatch):
    """A brand-new doc (no prior doc_status) must NOT call purge — there is
    nothing to purge, and purge is skipped when chunks_list is empty/absent."""
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return []

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", AsyncMock())

        purge_spy = AsyncMock()
        monkeypatch.setattr(rag, "_purge_doc_chunks_and_kg", purge_spy)

        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS),
            text_chunks=CHUNKS,
            doc_id="doc-fresh",
        )

        purge_spy.assert_not_awaited()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ainsert_custom_chunks_flush_failure_keeps_doc_rebuildable(
    tmp_path, monkeypatch
):
    """A success-path flush failure must not leave the doc PROCESSED-but-empty.

    PROCESSED is marked only AFTER a clean flush, so a flush error (e.g.
    IndexFlushError on a server backend) leaves the doc PROCESSING — which the
    reprocess gate treats as rebuildable — and a retry completes it, instead of
    a permanently "done but KG-missing" doc that the gate would skip forever.
    """
    rag = await _build_rag(tmp_path)
    try:

        async def _fake_extract(chunks, *args, **kwargs):
            return [("nodes", "edges")]

        monkeypatch.setattr(rag, "_process_extract_entities", _fake_extract)
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", AsyncMock())

        async def _boom_flush():
            raise RuntimeError("flush failed")

        monkeypatch.setattr(rag, "_insert_done_with_cleanup", _boom_flush)

        with pytest.raises(RuntimeError, match="flush failed"):
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(CHUNKS),
                text_chunks=CHUNKS,
                doc_id="doc-flush",
            )

        # The doc must NOT be PROCESSED — the KG was never flushed.
        status = await rag.doc_status.get_by_id("doc-flush")
        assert status is not None
        assert status["status"] != DocStatus.PROCESSED

        # A retry with a healthy flush completes it.
        monkeypatch.setattr(rag, "_insert_done_with_cleanup", AsyncMock())
        await rag.ainsert_custom_chunks(
            full_text="\n\n".join(CHUNKS),
            text_chunks=CHUNKS,
            doc_id="doc-flush",
        )
        status = await rag.doc_status.get_by_id("doc-flush")
        assert status is not None
        assert status["status"] == DocStatus.PROCESSED
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
