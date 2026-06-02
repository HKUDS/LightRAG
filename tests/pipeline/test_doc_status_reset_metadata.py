"""Tests for directive-only metadata cleanup when (re)queuing to PENDING.

Covers:
  * ``doc_status_reset_metadata`` — keeps long-lived directives
    (``process_options`` / ``source_file``, legacy ``source_file_name``
    tolerant) and drops every per-attempt timing/result field.
  * ``doc_status_metadata_has_attempt_fields`` — the precise trigger used to
    decide whether an already-PENDING doc needs normalising.
  * ``_validate_and_fix_document_consistency`` — interrupted docs reset to a
    clean PENDING (storage AND in-memory), already-PENDING docs carrying stale
    per-attempt fields normalised in place, and the locked semantic for
    docs that hold non-attempt custom metadata.
"""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_pipeline import (
    doc_status_metadata_has_attempt_fields,
    doc_status_reset_metadata,
)

pytestmark = pytest.mark.offline


# A metadata blob carrying every per-attempt field a document could have
# accumulated across parse/analyze/process plus the two long-lived directives.
_FULL_ATTEMPT_METADATA = {
    "process_options": "iF",
    "source_file": "report.pdf",
    "parse_start_time": 100,
    "parse_end_time": 200,
    "parse_stage_skipped": True,
    "parse_warnings": {"docx": "missing paraId"},
    "analyzing_start_time": 300,
    "analyzing_end_time": 400,
    "analyzing_stage_skipped": True,
    "chunk_opts": "size=512, overlap=256",
    "process_start_time": 500,
    "process_end_time": 600,
    "parse_format": "raw",
    "parse_engine": "native",
    "chunk_method": "fixed_token",
    "skip_kg": True,
    "mm_chunks": 3,
    "hard_fallback_split": "256 -> 512",
}


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_rag(tmp_path, test_name: str) -> LightRAG:
    workspace = f"{test_name}_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / test_name),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


# ----------------------------------------------------------------------------
# doc_status_reset_metadata unit tests
# ----------------------------------------------------------------------------


def test_reset_metadata_keeps_directives_drops_attempt_fields():
    result = doc_status_reset_metadata({"metadata": dict(_FULL_ATTEMPT_METADATA)})
    assert result == {"process_options": "iF", "source_file": "report.pdf"}


def test_reset_metadata_normalizes_legacy_source_file_name():
    result = doc_status_reset_metadata(
        {"metadata": {"source_file_name": "legacy.docx", "process_options": "t"}}
    )
    assert result == {"process_options": "t", "source_file": "legacy.docx"}


def test_reset_metadata_prefers_new_source_file_over_legacy():
    result = doc_status_reset_metadata(
        {"metadata": {"source_file": "new.pdf", "source_file_name": "old.pdf"}}
    )
    assert result == {"source_file": "new.pdf"}


def test_reset_metadata_handles_empty_or_invalid_metadata():
    assert doc_status_reset_metadata({"metadata": {}}) == {}
    assert doc_status_reset_metadata({"metadata": None}) == {}
    assert doc_status_reset_metadata({}) == {}
    assert doc_status_reset_metadata(None) == {}
    # blank directive values are not carried
    assert (
        doc_status_reset_metadata(
            {"metadata": {"process_options": "", "source_file": ""}}
        )
        == {}
    )


def test_has_attempt_fields_trigger():
    assert doc_status_metadata_has_attempt_fields({"metadata": {"parse_end_time": 1}})
    assert doc_status_metadata_has_attempt_fields(
        {"metadata": {"process_options": "F", "chunk_opts": "x"}}
    )
    # only directives / custom non-attempt fields -> no trigger
    assert not doc_status_metadata_has_attempt_fields(
        {"metadata": {"process_options": "F", "source_file": "a.pdf"}}
    )
    assert not doc_status_metadata_has_attempt_fields(
        {"metadata": {"custom_field": "x"}}
    )
    assert not doc_status_metadata_has_attempt_fields({"metadata": {}})
    assert not doc_status_metadata_has_attempt_fields({"metadata": None})


# ----------------------------------------------------------------------------
# _validate_and_fix_document_consistency integration tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_reset_clears_attempt_metadata_in_storage_and_memory(
    tmp_path,
):
    rag = await _build_rag(tmp_path, "reset_clears_attempt_metadata")
    try:
        doc_id = "doc-interrupted"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {doc_id: {"content": "interrupted doc", "file_path": "report.pdf"}}
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "interrupted",
                    "content_length": 15,
                    "chunks_count": 1,
                    "chunks_list": ["c-1"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "report.pdf",
                    "track_id": "track-1",
                    "error_msg": "old error",
                    "metadata": dict(_FULL_ATTEMPT_METADATA),
                }
            }
        )

        to_process_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # Storage: directives only, error cleared, chunks preserved.
        stored = await rag.doc_status.get_by_id(doc_id)
        assert stored is not None
        assert _status_to_text(stored["status"]) == "pending"
        assert stored["metadata"] == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }
        assert stored["error_msg"] == ""
        assert stored["chunks_list"] == ["c-1"]

        # In-memory status_doc carried forward by workers is cleaned too.
        assert to_process_docs[doc_id].metadata == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }
        assert _status_to_text(to_process_docs[doc_id].status) == "pending"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_stale_pending_normalized_clean_pending_untouched(tmp_path):
    rag = await _build_rag(tmp_path, "stale_pending_normalized")
    try:
        stale_id = "doc-stale-pending"
        clean_id = "doc-clean-pending"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                stale_id: {"content": "stale", "file_path": "stale.pdf"},
                clean_id: {"content": "clean", "file_path": "clean.pdf"},
            }
        )
        await rag.doc_status.upsert(
            {
                stale_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "stale",
                    "content_length": 5,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "stale.pdf",
                    "track_id": "track-stale",
                    "error_msg": "",
                    "metadata": {
                        "process_options": "F",
                        "parse_end_time": 999,
                        "analyzing_start_time": 111,
                    },
                },
                clean_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "clean",
                    "content_length": 5,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "clean.pdf",
                    "track_id": "track-clean",
                    "error_msg": "",
                    "metadata": {"process_options": "F", "source_file": "clean.pdf"},
                },
            }
        )

        to_process_docs = await rag.doc_status.get_docs_by_status(DocStatus.PENDING)
        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # Stale PENDING normalized to directives-only (storage + memory).
        stale_stored = await rag.doc_status.get_by_id(stale_id)
        assert stale_stored["metadata"] == {"process_options": "F"}
        assert to_process_docs[stale_id].metadata == {"process_options": "F"}

        # Clean PENDING left untouched: no rewrite (updated_at unchanged).
        clean_stored = await rag.doc_status.get_by_id(clean_id)
        assert clean_stored["metadata"] == {
            "process_options": "F",
            "source_file": "clean.pdf",
        }
        assert clean_stored["updated_at"] == now
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pending_custom_metadata_semantic_lock(tmp_path):
    """Locks the chosen action semantic for already-PENDING docs:

    * only custom (non-attempt) metadata, no stale key -> left untouched;
    * stale key + custom field -> collapsed to directives-only (custom
      dropped), matching the interrupted-reset path.
    """
    rag = await _build_rag(tmp_path, "pending_custom_metadata_lock")
    try:
        custom_only_id = "doc-custom-only"
        stale_plus_custom_id = "doc-stale-plus-custom"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                custom_only_id: {"content": "c", "file_path": "custom.pdf"},
                stale_plus_custom_id: {"content": "s", "file_path": "mixed.pdf"},
            }
        )
        await rag.doc_status.upsert(
            {
                custom_only_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "c",
                    "content_length": 1,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "custom.pdf",
                    "track_id": "track-custom",
                    "error_msg": "",
                    "metadata": {"custom_field": "keep-me"},
                },
                stale_plus_custom_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "s",
                    "content_length": 1,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "mixed.pdf",
                    "track_id": "track-mixed",
                    "error_msg": "",
                    "metadata": {
                        "process_options": "F",
                        "custom_field": "drop-me",
                        "parse_start_time": 42,
                    },
                },
            }
        )

        to_process_docs = await rag.doc_status.get_docs_by_status(DocStatus.PENDING)
        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        custom_stored = await rag.doc_status.get_by_id(custom_only_id)
        assert custom_stored["metadata"] == {"custom_field": "keep-me"}
        assert custom_stored["updated_at"] == now

        mixed_stored = await rag.doc_status.get_by_id(stale_plus_custom_id)
        assert mixed_stored["metadata"] == {"process_options": "F"}
        assert "custom_field" not in mixed_stored["metadata"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reset_and_normalize_with_string_status(tmp_path):
    """Non-JSON backends (Redis/Mongo/OpenSearch/Postgres) hydrate
    ``DocProcessingStatus.status`` as a raw string (``"processing"`` /
    ``"pending"``) rather than a ``DocStatus`` member.  Because ``DocStatus``
    subclasses ``str``, the enum-membership checks in
    ``_validate_and_fix_document_consistency`` still match — this test pins
    that behaviour so a future change to ``DocStatus`` (e.g. dropping the
    ``str`` base) cannot silently stop string-status docs from being cleaned.

    Builds the ``to_process_docs`` objects directly with string statuses to
    exercise the string path (``get_docs_by_status`` would return enum-valued
    members from the in-memory store).
    """
    rag = await _build_rag(tmp_path, "reset_string_status")
    try:
        processing_id = "doc-string-processing"
        pending_id = "doc-string-pending"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                processing_id: {"content": "p", "file_path": "proc.pdf"},
                pending_id: {"content": "q", "file_path": "pend.pdf"},
            }
        )

        # Status is a plain string, exactly as a non-JSON backend hydrates it.
        to_process_docs = {
            processing_id: DocProcessingStatus(
                content_summary="p",
                content_length=1,
                file_path="proc.pdf",
                status="processing",
                created_at=now,
                updated_at=now,
                track_id="track-proc",
                chunks_count=1,
                chunks_list=["c-1"],
                error_msg="old error",
                metadata=dict(_FULL_ATTEMPT_METADATA),
            ),
            pending_id: DocProcessingStatus(
                content_summary="q",
                content_length=1,
                file_path="pend.pdf",
                status="pending",
                created_at=now,
                updated_at=now,
                track_id="track-pend",
                chunks_count=0,
                chunks_list=[],
                error_msg="",
                metadata={"process_options": "F", "analyzing_start_time": 7},
            ),
        }
        assert isinstance(to_process_docs[processing_id].status, str)
        assert not isinstance(to_process_docs[processing_id].status, DocStatus)

        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # String "processing" is reset to a clean PENDING.
        proc_stored = await rag.doc_status.get_by_id(processing_id)
        assert proc_stored is not None
        assert _status_to_text(proc_stored["status"]) == "pending"
        assert proc_stored["metadata"] == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }
        assert to_process_docs[processing_id].metadata == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }

        # String "pending" carrying a stale field is normalized in place.
        pend_stored = await rag.doc_status.get_by_id(pending_id)
        assert pend_stored is not None
        assert pend_stored["metadata"] == {"process_options": "F"}
        assert to_process_docs[pending_id].metadata == {"process_options": "F"}
    finally:
        await rag.finalize_storages()
