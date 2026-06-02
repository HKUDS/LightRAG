"""Offline tests for orchestrator-level storage failures during batch enqueue.

Covers the fix that:

* guarantees the parse/analyze/process workers created in ``_run_pipeline_batch``
  are ALWAYS cancelled+awaited via a try/finally — so an exception escaping the
  enqueue loop (e.g. an orchestrator-level ``full_docs.get_by_id`` failing during
  a backend outage) can never orphan live worker tasks that keep draining the
  queues and appending to ``history_messages`` while the caller's finally has
  already cleared ``busy`` (busy=False but processing visibly continues); and
* isolates a single failing document's engine-routing ``get_by_id`` so that a
  transient/corrupt doc is marked FAILED and the rest of the batch still
  processes, instead of escaping and aborting the whole batch.

The orchestrator's ``full_docs.get_by_id`` at batch enqueue is the read under
test. ``_validate_and_fix_document_consistency`` (which also reads full_docs,
but BEFORE any worker is created — so an escape there is a clean stop, not an
orphan) is bypassed to target the enqueue path deterministically.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

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


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
    ]


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"orphan-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _live_worker_tasks() -> list[asyncio.Task]:
    """Pipeline worker tasks (parse/analyze/process) that are still pending.

    With the try/finally fix these are always cancelled+awaited before
    ``_run_pipeline_batch`` returns, so this is empty afterwards. Without it,
    an escape orphans them and they linger here.
    """
    live = []
    for task in asyncio.all_tasks():
        qualname = getattr(task.get_coro(), "__qualname__", "")
        if any(
            marker in qualname
            for marker in ("_parse_worker", "_analyze_worker", "_process_worker")
        ):
            if not task.done():
                live.append(task)
    return live


async def _passthrough_validate(self_docs, pipeline_status, pipeline_status_lock):
    # Skip the pre-enqueue full_docs read so the only get_by_id for the bad
    # doc is the orchestrator enqueue read under test.
    return self_docs


@pytest.mark.asyncio
async def test_single_doc_get_by_id_failure_is_isolated_no_orphans(
    tmp_path, monkeypatch
):
    """One doc's enqueue get_by_id fails -> that doc FAILED, the rest PROCESSED,
    busy released, and no worker tasks are orphaned."""
    rag = await _build_rag(tmp_path)
    try:
        bad_path = "bad.txt"
        good_path = "good.txt"
        bad_id = compute_mdhash_id(bad_path, prefix="doc-")
        good_id = compute_mdhash_id(good_path, prefix="doc-")
        await rag.apipeline_enqueue_documents(
            input=["bad content", "good content"],
            file_paths=[bad_path, good_path],
        )

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )

        orig_get = rag.full_docs.get_by_id

        async def flaky_get(doc_id):
            if doc_id == bad_id:
                raise ConnectionError("redis down for this doc")
            return await orig_get(doc_id)

        monkeypatch.setattr(rag.full_docs, "get_by_id", flaky_get)

        await rag.apipeline_process_enqueue_documents()
        # Give any (incorrectly) orphaned tasks a tick to surface.
        await asyncio.sleep(0)

        assert _live_worker_tasks() == [], "worker tasks were orphaned"

        bad_status = await rag.doc_status.get_by_id(bad_id)
        good_status = await rag.doc_status.get_by_id(good_id)
        assert _status_to_text(bad_status["status"]) == "failed"
        assert _status_to_text(good_status["status"]) == "processed"

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_full_outage_escape_still_cancels_workers(tmp_path, monkeypatch):
    """When even the FAILED-status write fails (full outage), the exception
    escapes the enqueue loop, but the try/finally still cancels all workers
    and the caller's finally releases busy — no orphans, busy=False."""
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            input="outage content", file_paths="outage.txt"
        )

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )

        async def dead_get(doc_id):
            raise ConnectionError("redis fully down")

        async def dead_upsert(*args, **kwargs):
            raise ConnectionError("redis fully down")

        monkeypatch.setattr(rag.full_docs, "get_by_id", dead_get)
        # _finalize_doc_failure's doc_status write also fails -> escape.
        monkeypatch.setattr(rag.doc_status, "upsert", dead_upsert)

        # apipeline_process_enqueue_documents lets the storage error propagate
        # out (its finally releases busy but does not swallow the exception).
        with pytest.raises(Exception):
            await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0)

        assert _live_worker_tasks() == [], "worker tasks were orphaned on escape"

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()
