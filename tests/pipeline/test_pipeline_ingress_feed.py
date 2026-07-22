"""In-batch feeder (Phase 2): documents that arrive DURING a running batch are
routed into that batch's parse queues instead of waiting for the per-batch
barrier.

The feeder is a pure accelerator — anything it drops/skips is a PENDING
``doc_status`` row that ``request_pending`` (dual-written by enqueue on the
same busy pipeline) recovers on the next batch — so these tests pin the
latency win (fed doc completes while an earlier doc is still stuck) and the
safety invariants (dedup against inflight, no busy loop when only manual/auto
signals are pending), all on a real LightRAG with in-memory JSON storages.
"""

import asyncio
from collections import Counter
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import get_pipeline_ingress
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


def _chunking(tokenizer, content, *args) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


class _MarkerExtract:
    """Process-layer extraction that counts calls per document marker and can
    block whichever marker is armed until released."""

    def __init__(self):
        self.calls: Counter = Counter()
        self.block_marker: str | None = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, chunks, *args, **kwargs):
        content = " ".join(str(v.get("content", "")) for v in chunks.values())
        marker = content.split(" ", 1)[0] if content else ""
        self.calls[marker] += 1
        if self.block_marker and self.block_marker in content:
            self.started.set()
            await self.release.wait()
        return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path, extract, *, max_parallel_insert: int) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"feed-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=max_parallel_insert,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = extract
    return rag


def _status_text(row) -> str:
    status = row["status"]
    return getattr(status, "value", str(status)).replace("DocStatus.", "").lower()


async def _wait_status(rag, doc_id, target: str, timeout: float = 5.0) -> None:
    async def _poll():
        while True:
            row = await rag.doc_status.get_by_id(doc_id)
            if row and _status_text(row) == target:
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(_poll(), timeout=timeout)


def test_document_arriving_mid_batch_is_fed_without_waiting(tmp_path):
    """Fix-proof: doc A blocks in the process stage; doc B enqueued DURING A's
    batch is routed by the feeder and reaches PROCESSED while A is still stuck.

    Without the feeder, B would only be picked up after A's batch finishes
    (request_pending → next batch), but A's batch cannot finish while A is
    blocked — so B would never complete and this would time out."""

    async def _run():
        extract = _MarkerExtract()
        extract.block_marker = "AAAA"
        # Two process workers so A blocking one does not starve B.
        rag = await _build_rag(tmp_path, extract, max_parallel_insert=2)
        try:
            await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
            a_id = compute_mdhash_id("a.txt", prefix="doc-")
            b_id = compute_mdhash_id("b.txt", prefix="doc-")

            proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
            await asyncio.wait_for(extract.started.wait(), timeout=5)  # A blocked

            # Enqueue B while A's batch is running and busy.
            await rag.apipeline_enqueue_documents(input="BBBB body", file_paths="b.txt")

            # The feeder routes B into A's running batch; B finishes without
            # waiting for A.
            await _wait_status(rag, b_id, "processed")
            a_row = await rag.doc_status.get_by_id(a_id)
            assert _status_text(a_row) != "processed"  # A still stuck

            extract.release.set()
            await asyncio.wait_for(proc, timeout=5)
            assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
            assert extract.calls["AAAA"] == 1
            assert extract.calls["BBBB"] == 1
        finally:
            extract.release.set()
            await rag.finalize_storages()

    asyncio.run(_run())


def test_feeder_deduplicates_inflight_document_message(tmp_path):
    """A document message that echoes an id already inflight (the initial batch
    doc, or one the feeder already routed) is dropped, not re-routed — the doc
    is extracted exactly once."""

    async def _run():
        extract = _MarkerExtract()
        extract.block_marker = "AAAA"
        rag = await _build_rag(tmp_path, extract, max_parallel_insert=2)
        try:
            await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
            a_id = compute_mdhash_id("a.txt", prefix="doc-")

            proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
            await asyncio.wait_for(extract.started.wait(), timeout=5)  # A inflight

            # Publish a stale document notification for A (already inflight).
            ingress = await get_pipeline_ingress(rag.workspace)
            ingress.put_document(PipelineIngressMessage(kind="document", doc_id=a_id))
            # Let the feeder observe and drop it.
            await asyncio.sleep(0.1)

            extract.release.set()
            await asyncio.wait_for(proc, timeout=5)
            assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
            assert extract.calls["AAAA"] == 1  # not double-routed
        finally:
            extract.release.set()
            await rag.finalize_storages()

    asyncio.run(_run())


def test_pending_manual_signal_does_not_busy_loop_the_feeder(tmp_path):
    """The feeder waits ONLY on the document channel: a sticky manual-retry
    request pending on the ingress must not spin the feeder (which would starve
    the event loop). A's batch still completes normally and its extract runs
    exactly once — a busy loop would inflate the count or stall A."""

    async def _run():
        extract = _MarkerExtract()  # no block marker: A flows straight through
        rag = await _build_rag(tmp_path, extract, max_parallel_insert=1)
        try:
            await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
            a_id = compute_mdhash_id("a.txt", prefix="doc-")

            # Arm a sticky manual retry request on the ingress. It sets the
            # work_event; a feeder that waited on has_work() (three channels)
            # would busy-loop on it. The feeder waits document-only, so it stays
            # parked and A completes.
            ingress = await get_pipeline_ingress(rag.workspace)
            ingress.request_manual_retry(
                "req-x",
                PipelineIngressMessage(
                    kind="rescan", retry_failed=True, request_id="req-x"
                ),
            )

            await rag.apipeline_process_enqueue_documents()
            assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
            assert extract.calls["AAAA"] == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
