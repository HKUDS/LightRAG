"""Offline tests for the queue-payload slimming refactor.

To keep large document bodies out of the in-memory ``q_analyze`` / ``q_process``
buffers, the parsed body is no longer carried through the cascading pipeline
queues. Instead:

* ``_parse_worker`` computes ``content_summary`` / ``content_length`` while it
  still holds the body, stamps them on the ``status_doc``, then drops the
  ``"content"`` key from the payload it enqueues onto ``q_analyze``.
* ``process_single_document`` (Layer 3) re-reads the body from ``full_docs`` by
  ``doc_id`` and strips the ``{{LRdoc}}`` marker according to the stored
  ``parse_format``.

These tests pin that contract: the body leaves the queues, the summary/length
are populated by the parse stage, and the body is faithfully reconstructed
(format-aware) at the process stage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_RAW,
)
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.pipeline import _BatchRunContext
from lightrag.parser.registry import parser_specs_snapshot
from lightrag.utils import EmbeddingFunc, Tokenizer, get_content_summary
from lightrag.utils_pipeline import make_lightrag_doc_content


pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 8)


async def _noop_llm(prompt, **kwargs):  # pragma: no cover - never invoked
    return ""


def _build_rag(tmp_path: Path) -> LightRAG:
    role_configs = {spec.name: RoleLLMConfig() for spec in ROLES}
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"reread-{tmp_path.name}",
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        role_llm_configs=role_configs,
    )


async def _make_ctx(rag: LightRAG) -> _BatchRunContext:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status.clear()
    pipeline_status.update(
        {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
        }
    )
    return _BatchRunContext(
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        semaphore=asyncio.Semaphore(2),
        total_files=1,
        parse_queues={
            "native": asyncio.Queue(),
            "mineru": asyncio.Queue(),
            "docling": asyncio.Queue(),
        },
        parser_specs=parser_specs_snapshot(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )


def _make_status_doc(doc_id: str, *, content_hash: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        # Deliberately stale placeholders — the parse worker must overwrite
        # these from the actual parsed body.
        content_summary="stale-summary",
        content_length=999,
        file_path=f"{doc_id}.txt",
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id=None,
        content_hash=content_hash,
    )


async def _seed_doc_status(rag: LightRAG, doc_id: str, *, process_options: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PENDING.value,
                "content_summary": "stale-summary",
                "content_length": 999,
                "file_path": f"{doc_id}.txt",
                "created_at": now,
                "updated_at": now,
                "track_id": "t",
                "metadata": {"process_options": process_options},
            }
        }
    )


@pytest.mark.asyncio
async def test_parse_worker_drops_body_and_sets_summary_length(tmp_path):
    """After parsing, the q_analyze payload must NOT carry ``content``, and the
    status_doc must hold the body's summary/length — while full_docs keeps the
    body for the downstream re-read."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx = await _make_ctx(rag)
        doc_id = "doc-raw"
        body = "The quick brown fox jumps over the lazy dog. " * 4

        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": body,
                    "file_path": f"{doc_id}.txt",
                    "parse_format": FULL_DOCS_FORMAT_RAW,
                }
            }
        )
        await _seed_doc_status(rag, doc_id)
        await ctx.parse_queues["native"].put(
            (doc_id, _make_status_doc(doc_id, content_hash="hash-raw"))
        )

        worker = asyncio.create_task(rag._parse_worker("native", ctx.parse_queues["native"], ctx))
        try:
            await asyncio.wait_for(ctx.parse_queues["native"].join(), timeout=2.0)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        # The doc was handed off to q_analyze exactly once.
        assert ctx.q_analyze.qsize() == 1
        enq_doc_id, enq_status_doc, enq_parsed = ctx.q_analyze.get_nowait()

        assert enq_doc_id == doc_id
        # The heavy body must be gone from the queue payload.
        assert "content" not in enq_parsed
        # Light metadata still rides along.
        assert enq_parsed["blocks_path"] == ""
        assert enq_parsed["doc_id"] == doc_id

        # Summary / length were computed by the parse worker from the body.
        assert enq_status_doc.content_length == len(body)
        assert enq_status_doc.content_summary == get_content_summary(body)

        # full_docs still holds the body for Layer 3 to re-read.
        stored = await rag.full_docs.get_by_id(doc_id)
        assert stored is not None and stored["content"] == body
    finally:
        await rag.finalize_storages()


async def _drive_process_and_collect_chunks(
    rag: LightRAG, ctx: _BatchRunContext, doc_id: str
) -> tuple[dict, str]:
    """Run process_single_document for one doc and return (doc_status_row,
    concatenated chunk content)."""
    status_doc = _make_status_doc(doc_id, content_hash=f"hash-{doc_id}")
    # The new contract: the queue payload carries NO ``content`` key.
    parsed_data: dict[str, Any] = {
        "doc_id": doc_id,
        "file_path": f"{doc_id}.txt",
        "blocks_path": "",
    }
    await rag.process_single_document(
        doc_id=doc_id,
        status_doc=status_doc,
        parsed_data=parsed_data,
        ctx=ctx,
    )

    row = await rag.doc_status.get_by_id(doc_id)
    chunk_ids = (row or {}).get("chunks_list") or []
    chunks = await rag.text_chunks.get_by_ids(chunk_ids)
    joined = "".join((c or {}).get("content", "") for c in chunks)
    return row, joined


@pytest.mark.asyncio
async def test_process_reads_raw_body_from_full_docs(tmp_path):
    """process_single_document must reconstruct a RAW body from full_docs when
    the queue payload omits ``content`` (skip_kg avoids any LLM call)."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx = await _make_ctx(rag)
        doc_id = "doc-proc-raw"
        body = "Raw body sentence number {}. ".format
        full_body = "".join(body(i) for i in range(5))

        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": full_body,
                    "file_path": f"{doc_id}.txt",
                    "parse_format": FULL_DOCS_FORMAT_RAW,
                    "process_options": "!",  # skip KG → no extraction LLM
                }
            }
        )
        await _seed_doc_status(rag, doc_id, process_options="!")

        row, joined = await _drive_process_and_collect_chunks(rag, ctx, doc_id)

        assert row is not None
        assert row.get("status") == DocStatus.PROCESSED.value
        assert (row.get("chunks_count") or 0) >= 1
        # The chunked text came from the re-read body, not an empty payload.
        # (The chunker trims surrounding whitespace per chunk, so compare
        # the meaningful content rather than byte-exact.)
        assert joined.strip() == full_body.strip()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_process_strips_lightrag_marker_on_reread(tmp_path):
    """For a lightrag-format record, the re-read must strip the ``{{LRdoc}}``
    marker so chunk content is the bare body — never the prefix."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        ctx = await _make_ctx(rag)
        doc_id = "doc-proc-lrdoc"
        bare_body = "".join(f"Lightrag body line {i}. " for i in range(5))

        await rag.full_docs.upsert(
            {
                doc_id: {
                    # Stored WITH the marker, as parse persists lightrag docs.
                    "content": make_lightrag_doc_content(bare_body),
                    "file_path": f"{doc_id}.txt",
                    "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                    "process_options": "!",
                }
            }
        )
        await _seed_doc_status(rag, doc_id, process_options="!")

        row, joined = await _drive_process_and_collect_chunks(rag, ctx, doc_id)

        assert row is not None
        assert row.get("status") == DocStatus.PROCESSED.value
        assert (row.get("chunks_count") or 0) >= 1
        # Marker must be stripped; bare body recovered (modulo the chunker's
        # per-chunk whitespace trimming).
        assert joined.strip() == bare_body.strip()
        assert "{{LRdoc}}" not in joined
    finally:
        await rag.finalize_storages()
