"""``ainsert_custom_chunks`` as a journaled, recoverable operation (issue
#3400, Phase 3).

Drives the real LightRAG object (JSON storages, offline) with extraction
monkeypatched to a deterministic fake (one entity per staged chunk), covering:

- create mode: full doc_status bookkeeping + recovery anchors;
- patch mode: chunks_list and anchors are UNIONED, never overwritten;
- idempotence: repeating committed input is a no-op;
- failure: FAILED + retained journal; the pipeline refuses to touch the row;
  the same call resumes and commits;
- conflict: a different operation is rejected while a journal is active;
- deletion: staged (uncommitted) patch chunks are cleaned up by
  ``adelete_by_doc_id``.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_pipeline import (
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    make_custom_chunk_id,
)

pytestmark = pytest.mark.offline


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
        workspace=f"ccpatch-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _fake_extraction(rag, monkeypatch):
    """Deterministic extraction: one entity per staged chunk, named after the
    chunk's first word (uppercased), attributed to the staged chunk id."""

    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id, payload in chunks.items():
            name = payload["content"].split()[0].upper()
            results.append(
                (
                    {
                        name: [
                            {
                                "entity_name": name,
                                "entity_type": "person",
                                "description": f"{name} description",
                                "source_id": chunk_id,
                                "file_path": "custom",
                                "timestamp": 1,
                            }
                        ]
                    },
                    {},
                )
            )
        return results

    monkeypatch.setattr(rag, "_process_extract_entities", fake_extract)


def _status_text(row: dict) -> str:
    raw = row.get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw)


def _journal(row: dict) -> dict | None:
    return (row.get("metadata") or {}).get(CUSTOM_CHUNK_PATCH_METADATA_KEY)


def _chunk_id(doc_key: str, content: str) -> str:
    return make_custom_chunk_id(doc_key, content)


@pytest.mark.offline
def test_chunk_id_doc_and_text_are_unambiguous():
    """Codex review (PR #3416): plain concatenation hashed doc_id="a" +
    chunk="bc" and doc_id="ab" + chunk="c" identically, letting two documents
    share (and clobber) one chunk row. The encoding must keep them distinct."""
    assert make_custom_chunk_id("a", "bc") != make_custom_chunk_id("ab", "c")
    # Deterministic for the same logical input.
    assert make_custom_chunk_id("a", "bc") == make_custom_chunk_id("a", "bc")


@pytest.mark.asyncio
async def test_create_mode_writes_doc_status_and_anchors(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks(
            "base document", ["alice is here"], doc_id="doc-1"
        )

        row = await rag.doc_status.get_by_id("doc-1")
        assert row is not None, "create mode must write a doc_status row"
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None, "journal must be cleared at commit"
        assert row["chunks_list"] == [_chunk_id("doc-1", "alice is here")]

        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors and anchors["entity_names"] == ["ALICE"]
        assert await rag.full_docs.get_by_id("doc-1") is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_repeating_committed_input_is_noop(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")
        row_before = await rag.doc_status.get_by_id("doc-1")

        # Same logical input again: committed no-op (content dedup).
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")
        row_after = await rag.doc_status.get_by_id("doc-1")
        assert row_after["chunks_list"] == row_before["chunks_list"]
        assert _status_text(row_after) == DocStatus.PROCESSED.value
        assert _journal(row_after) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_patch_unions_chunks_and_anchors(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")
        await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert row["chunks_list"] == [
            _chunk_id("doc-1", "alice is here"),
            _chunk_id("doc-1", "bob is there"),
        ], "patch must UNION into chunks_list, preserving committed chunks"

        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors["entity_names"] == ["ALICE", "BOB"], (
            "patch must union anchors, not overwrite the base document's"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_patch_rejected_on_non_processed_document(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.doc_status.upsert(
            {
                "doc-1": {
                    "status": DocStatus.FAILED,
                    "content_summary": "s",
                    "content_length": 1,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "file_path": "f",
                    "metadata": {},
                }
            }
        )
        with pytest.raises(RuntimeError, match="only PROCESSED documents"):
            await rag.ainsert_custom_chunks("base", ["alice"], doc_id="doc-1")
    finally:
        await rag.finalize_storages()


async def _fail_one_merge_then_restore(monkeypatch):
    """Make lightrag.lightrag.merge_nodes_and_edges raise once, then behave."""
    calls = {"n": 0}
    orig_merge = lightrag_module.merge_nodes_and_edges

    async def merge_boom(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("merge boom")
        return await orig_merge(**kwargs)

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_boom)
    return calls


@pytest.mark.asyncio
async def test_failed_patch_keeps_journal_pipeline_skips_and_resume_commits(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        await _fail_one_merge_then_restore(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")

        # FAILED with the journal (and its write-ahead candidates) retained.
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        journal = _journal(row)
        assert journal is not None, "journal must survive the failure"
        assert journal["entity_names"] == ["BOB"]
        assert journal["chunk_ids"] == [_chunk_id("doc-1", "bob is there")]
        # Committed base state is untouched.
        assert row["chunks_list"] == [_chunk_id("doc-1", "alice is here")]

        # The ordinary pipeline must not touch (reset/reprocess) the row.
        await rag.apipeline_process_enqueue_documents()
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        assert _journal(row) is not None

        # The same SDK call resumes and commits.
        await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None
        assert row["chunks_list"] == [
            _chunk_id("doc-1", "alice is here"),
            _chunk_id("doc-1", "bob is there"),
        ]
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors["entity_names"] == ["ALICE", "BOB"]

        # Busy slot was released through all of it.
        from lightrag.kg.shared_storage import get_namespace_data

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_conflicting_operation_rejected_while_journal_active(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        await _fail_one_merge_then_restore(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")

        with pytest.raises(RuntimeError, match="unfinished custom-chunk"):
            await rag.ainsert_custom_chunks(
                "base", ["carol is elsewhere"], doc_id="doc-1"
            )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failed_create_records_failed_doc_status(tmp_path, monkeypatch):
    """The historical gap: a failed create left chunk/vector data with NO
    doc_status row at all. Now it must leave a FAILED row with the journal."""
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await _fail_one_merge_then_restore(monkeypatch)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks(
                "fresh doc", ["alice is here"], doc_id="doc-9"
            )

        row = await rag.doc_status.get_by_id("doc-9")
        assert row is not None, "failed create must be discoverable in doc_status"
        assert _status_text(row) == DocStatus.FAILED.value
        journal = _journal(row)
        assert journal is not None and journal["mode"] == "create"

        # Resume completes the create.
        await rag.ainsert_custom_chunks("fresh doc", ["alice is here"], doc_id="doc-9")
        row = await rag.doc_status.get_by_id("doc-9")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert _journal(row) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_covers_journal_only_graph_candidates(tmp_path, monkeypatch):
    """Codex review (PR #3416): a patch that fails AFTER the merge wrote graph
    objects — but before commit unioned the anchors — leaves candidates that
    exist only in the journal. Deleting the document must remove those graph
    objects too, not just the staged chunks; otherwise they survive as
    orphans pointing at deleted chunks."""
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        # Merge succeeds (BOB reaches graph/vdb/tracking), then the operation
        # fails before the commit union.
        orig_merge = lightrag_module.merge_nodes_and_edges
        calls = {"n": 0}

        async def merge_then_boom(**kwargs):
            result = await orig_merge(**kwargs)
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("post-merge boom")
            return result

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_then_boom)
        with pytest.raises(RuntimeError, match="post-merge boom"):
            await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")

        # BOB reached the graph but is anchored ONLY in the journal.
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is not None
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert "BOB" not in (anchors or {}).get("entity_names", [])

        result = await rag.adelete_by_doc_id("doc-1")
        assert result.status == "success"
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is None, (
            "journal-only graph candidate must be cleaned by document deletion"
        )
        assert await rag.entity_chunks.get_by_id("BOB") is None
        assert (
            await rag.text_chunks.get_by_id(_chunk_id("doc-1", "bob is there")) is None
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_includes_staged_patch_chunks(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        _fake_extraction(rag, monkeypatch)
        await rag.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        await _fail_one_merge_then_restore(monkeypatch)
        staged_id = _chunk_id("doc-1", "bob is there")
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")
        assert await rag.text_chunks.get_by_id(staged_id) is not None, (
            "staged chunk must be durable before merge (flushed staging)"
        )

        result = await rag.adelete_by_doc_id("doc-1")
        assert result.status == "success"
        assert await rag.text_chunks.get_by_id(staged_id) is None, (
            "deletion must clean staged (journal-only) patch chunks"
        )
        assert (
            await rag.text_chunks.get_by_id(_chunk_id("doc-1", "alice is here")) is None
        )
    finally:
        await rag.finalize_storages()
