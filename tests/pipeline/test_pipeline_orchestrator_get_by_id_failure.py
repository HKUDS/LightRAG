"""Offline tests for orchestrator-level storage failures during batch enqueue.

Covers the fix that guarantees the parse/analyze/process workers created in
``_run_pipeline_batch`` are ALWAYS cancelled+awaited via a try/finally — so an
exception escaping the enqueue loop (e.g. an orchestrator-level
``full_docs.get_by_id`` failing during a backend outage) can never orphan live
worker tasks that keep draining the queues and appending to ``history_messages``
while the caller's finally has already cleared ``busy`` (the observed
"busy=False but processing visibly continues" symptom). The same finally also
discards not-yet-flushed shared index buffers so stale records are not carried
into the next batch.

By design there is NO per-document isolation of this read: the same unguarded
``full_docs.get_by_id`` also runs in ``_validate_and_fix_document_consistency``
before the batch, so a failure aborts the whole batch CLEANLY (workers torn
down, busy released, docs left PENDING for retry) rather than being isolated.

``_validate_and_fix_document_consistency`` is bypassed in these tests so the
escape lands inside ``_run_pipeline_batch`` (where workers are live) — the path
that actually exercises worker teardown.
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
async def test_enqueue_get_by_id_failure_aborts_batch_cleanly(tmp_path, monkeypatch):
    """An orchestrator-level get_by_id failure during batch enqueue (after the
    workers exist) aborts the whole batch CLEANLY: the exception propagates,
    every worker task is cancelled (no orphans), pending index buffers are
    discarded, and the caller releases ``busy``. The document is left in a
    non-PROCESSED state for the next run (no per-doc isolation by design).

    ``_validate_and_fix_document_consistency`` is bypassed so the escape lands
    inside ``_run_pipeline_batch`` (where workers are live) rather than in the
    earlier validation read — the path that actually exercises worker teardown.
    """
    rag = await _build_rag(tmp_path)
    try:
        doc_path = "outage.txt"
        doc_id = compute_mdhash_id(doc_path, prefix="doc-")
        await rag.apipeline_enqueue_documents(
            input="outage content", file_paths=doc_path
        )

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )

        async def dead_get(doc_id):
            raise ConnectionError("redis down during enqueue")

        monkeypatch.setattr(rag.full_docs, "get_by_id", dead_get)

        # The batch's finally must discard not-yet-flushed shared buffers when an
        # exception escapes (not just on the cooperative internal-error flag), so
        # stale KG/vector records are not carried into the next batch.
        discard_calls = 0
        orig_discard = rag._discard_pending_index_ops

        async def spy_discard(*, skip_enqueue_owned=True):
            nonlocal discard_calls
            discard_calls += 1
            await orig_discard(skip_enqueue_owned=skip_enqueue_owned)

        monkeypatch.setattr(rag, "_discard_pending_index_ops", spy_discard)

        # apipeline_process_enqueue_documents lets the storage error propagate
        # out (its finally releases busy but does not swallow the exception).
        with pytest.raises(Exception):
            await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0)

        assert _live_worker_tasks() == [], "worker tasks were orphaned on escape"
        assert discard_calls >= 1, "pending index buffers were not discarded on escape"

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status.get("busy") is False

        # Doc was not marked PROCESSED — it stays queued for the next run.
        row = await rag.doc_status.get_by_id(doc_id)
        assert _status_to_text(row["status"]) != "processed"
    finally:
        await rag.finalize_storages()
