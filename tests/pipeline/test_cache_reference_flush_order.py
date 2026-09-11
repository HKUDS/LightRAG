"""The extract-stage failure epilogue must commit references before cache rows.

An extract cache row is reachable only through the owning chunk's
``llm_cache_list``, and ``use_llm_func_with_cache`` records that reference
before writing the row. On a deferred KV backend — ``JsonKVStorage``, the
default — ``upsert`` reaches shared memory only, so that write order is not yet
durable: the COMMIT order is what makes it so.

``_finalize_doc_failure`` used to commit ``llm_response_cache`` alone, through
the deliberately narrow ``_persist_llm_response_cache_best_effort``. That put
the row on disk while the only reference to it stayed in memory, and the next
crash stranded it — an unreachable row carrying the chunk text verbatim plus
the entities extracted from it. The path is not hypothetical: the extract-stage
``except`` runs this epilogue on exactly the sibling-cancellation case the
write ordering exists for.

So the rule holds at both layers, and the second commit is suppressed when the
first did not land: a lost cache entry is recomputed on the next run, an
unreachable row is permanent.

See *LLM extraction cache reachability* in docs/design/PurgeRecoveryContract.md.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.exceptions import IndexFlushError
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.parser.registry import parser_specs_snapshot
from lightrag.pipeline import _BatchRunContext
from lightrag.utils import EmbeddingFunc, Tokenizer

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
        workspace=f"flushorder-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )
    await rag.initialize_storages()
    return rag


def _make_status_doc(doc_id: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary=f"summary-{doc_id}",
        content_length=10,
        file_path=f"{doc_id}.txt",
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id=None,
        content_hash=f"hash-{doc_id}",
    )


def _record_commits(
    rag: LightRAG,
    order: list[str],
    *,
    chunks_fail: bool = False,
    chunks_decline: bool = False,
):
    """Tag each storage's ``index_done_callback`` so the commit order is visible.

    ``chunks_decline`` returns an explicit ``False`` instead of raising: the
    other way a commit does not land, where the storage reloaded a newer
    snapshot and discarded the pending mutation.
    """

    def _wrap(storage, label: str, fail: bool, decline: bool):
        original = storage.index_done_callback

        async def _tagged():
            order.append(label)
            if fail:
                raise RuntimeError(f"{label} commit is down")
            if decline:
                return False
            return await original()

        storage.index_done_callback = _tagged

    _wrap(rag.text_chunks, "text_chunks", chunks_fail, chunks_decline)
    _wrap(rag.llm_response_cache, "llm_response_cache", False, False)


async def _run_epilogue(
    rag: LightRAG, doc_id: str, error: BaseException | None = None
) -> None:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["history_messages"] = []
    ctx = _BatchRunContext(
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        semaphore=asyncio.Semaphore(1),
        total_files=1,
        parse_queues={"native": asyncio.Queue()},
        parser_specs=parser_specs_snapshot(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )
    await rag._finalize_doc_failure(
        doc_id=doc_id,
        status_doc=_make_status_doc(doc_id),
        file_path=f"{doc_id}.txt",
        error=error or RuntimeError("a sibling chunk exploded"),
        stage_label="extract",
        current_file_number=1,
        total_files=1,
        failed_chunks_snapshot=([], 0),
        pending_tasks=[],
        metadata_extra={},
        ctx=ctx,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
    )


@pytest.mark.asyncio
async def test_chunk_references_are_committed_before_the_cache_rows(tmp_path):
    """Cache rows must never reach disk ahead of the references that find them."""
    rag = await _build_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order)

        await _run_epilogue(rag, "doc-order")

        assert order == ["text_chunks", "llm_response_cache"], order
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_failed_reference_commit_suppresses_the_cache_commit(tmp_path):
    """If the references did not land, the rows they point at must not either."""
    rag = await _build_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order, chunks_fail=True)

        # The epilogue still completes: it is itself a failure path.
        await _run_epilogue(rag, "doc-suppressed")

        assert order == ["text_chunks"], order
        assert "llm_response_cache" not in order, (
            "the cache was committed while its chunk references stayed in "
            "memory — a crash now strands rows nothing can reach"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_failed_status_row_is_still_written(tmp_path):
    """Suppressing a commit must not cost the document its FAILED record."""
    rag = await _build_rag(tmp_path)
    try:
        _record_commits(rag, [], chunks_fail=True)

        await _run_epilogue(rag, "doc-status")

        row = await rag.doc_status.get_by_id("doc-status")
        status = row["status"]
        assert (status.value if hasattr(status, "value") else str(status)) == "failed"
        assert "a sibling chunk exploded" in row["error_msg"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_declined_reference_commit_also_suppresses_the_cache_commit(tmp_path):
    """A DECLINED commit discarded the mutation, so the references are not on disk.

    ``index_done_callback`` returning an explicit ``False`` is the other way a
    commit fails to land — the storage reloaded a newer snapshot and dropped the
    pending write. Identity, not truthiness: backends that commit normally
    return ``None``, which must not read as a decline.
    """
    rag = await _build_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order, chunks_decline=True)

        await _run_epilogue(rag, "doc-declined")

        assert order == ["text_chunks"], order
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_an_index_flush_error_on_chunks_suppresses_without_retrying(tmp_path):
    """A flush the backend already gave up on must not be re-read as success.

    ``OpenSearchKVStorage._flush_pending_kv_ops`` DROPS a permanently-failed
    bulk operation from its buffer before raising, so calling
    ``index_done_callback`` again finds an empty buffer and returns normally
    while the chunk reference is gone for good. The epilogue must believe the
    exception that brought it here, not the retry.
    """
    rag = await _build_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order)
        namespace = (
            getattr(rag.text_chunks, "final_namespace", None)
            or rag.text_chunks.namespace
        )

        await _run_epilogue(
            rag,
            "doc-flusherror",
            error=IndexFlushError(
                "OpenSearchKVStorage", namespace, RuntimeError("permanent bulk failure")
            ),
        )

        assert order == [], (
            "the epilogue retried a flush the backend had already discarded, "
            "then committed the cache rows behind a reference that is gone"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_an_index_flush_error_on_another_namespace_does_not_suppress(tmp_path):
    """Only a text_chunks flush failure is evidence the references did not land."""
    rag = await _build_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order)

        await _run_epilogue(
            rag,
            "doc-othernamespace",
            error=IndexFlushError(
                "OpenSearchVectorStorage", "entities", RuntimeError("unrelated")
            ),
        )

        assert order == ["text_chunks", "llm_response_cache"], order
    finally:
        await rag.finalize_storages()
