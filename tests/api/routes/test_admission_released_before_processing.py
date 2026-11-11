"""A bg task must not still hold its enqueue reservation when it drives the loop.

``/documents/upload``, ``/documents/text`` and ``/documents/texts`` all reserve a
pending-enqueue slot in the endpoint and hand it to a background task whose
``finally`` releases it. That task calls ``apipeline_enqueue_documents`` and then
``apipeline_process_enqueue_documents`` — and when the pipeline was idle the
second call BECOMES the processing run. Holding the slot across it made the run
own the very token a concurrent manual retry waits for:

* the retry sets ``manual_freeze_requested`` and ``manual_phase=drain_to_idle``;
* DRAIN_TO_IDLE returns ``CONTINUE_DRAIN_WAIT`` while ``pending_enqueues > 0``;
* the only reservation left is the running supervisor's own token, releasable
  only when the run returns, which it cannot do until the wait ends.

``busy`` then stays latched forever with nothing in flight. The fix releases the
slot between enqueue and processing (``_release_admission_after_enqueue``), so
these tests pin the ordering at both call sites: the documents must be in
``doc_status`` (released after the enqueue, not before) AND the reservation must
be gone by the time processing starts (released before it, not after).
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import (
    MANUAL_PHASE_DRAIN_TO_IDLE,
    get_pipeline_ingress,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

# ``lightrag.api.config`` parses argv at import time, so pytest's own flags must
# not be visible while the router module is loaded (same guard as
# ``test_admission_endpoints.py``).
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

_release_enqueue_slot = _document_routes._release_enqueue_slot
_reserve_enqueue_slot = _document_routes._reserve_enqueue_slot
pipeline_index_file = _document_routes.pipeline_index_file
pipeline_index_texts = _document_routes.pipeline_index_texts

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


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _build_rag(tmp_path, *, capacity: int) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"adm-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        max_pending_documents=capacity,
    )
    await rag.initialize_storages()
    return rag


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


def _spy_on_processing(rag, pipeline_status, lock, token: str) -> dict:
    """Record what a concurrent DRAIN_TO_IDLE would see once the run starts.

    Replaces the drive so no LLM runs: the question is not what the pipeline
    processes, it is what the reservation state looks like at the moment the bg
    task stops being an enqueuer and becomes the processing run.
    """
    seen: dict = {}

    async def _process():
        async with lock:
            seen["pending_enqueues"] = pipeline_status.get("pending_enqueues", 0)
            seen["token_registered"] = token in (
                pipeline_status.get("pending_enqueue_tokens") or {}
            )

    rag.apipeline_process_enqueue_documents = _process
    return seen


async def _freeze_as_manual_retry(pipeline_status, lock) -> None:
    """Put the workspace in the state a mid-run ``/reprocess_failed`` creates."""
    async with lock:
        pipeline_status["manual_freeze_requested"] = True
        pipeline_status["manual_phase"] = MANUAL_PHASE_DRAIN_TO_IDLE


# ---------------------------------------------------------------------------
# /documents/text and /documents/texts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_texts_releases_its_slot_before_driving_the_loop(tmp_path, capacity):
    """Fix-proof: with the release in the caller's ``finally`` only, the spy saw
    ``pending_enqueues == 1`` and a drain waiting on it could never reach 0."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await _status_handles(rag)
            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            seen = _spy_on_processing(rag, pipeline_status, lock, token)
            await _freeze_as_manual_retry(pipeline_status, lock)

            try:
                await pipeline_index_texts(
                    rag,
                    ["a document about admission tokens"],
                    file_sources=["admission.txt"],
                    track_id="t-texts",
                    admission_token=token,
                )
            finally:
                # The endpoint's own release: idempotent, so it must not be what
                # makes the assertions below pass.
                await _release_enqueue_slot(rag, token)

            assert seen["pending_enqueues"] == 0
            assert seen["token_registered"] is False
            # Released AFTER the enqueue, not instead of it: the drain's cohort
            # must contain the document this request was admitted for.
            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_texts_release_survives_the_endpoint_double_release(tmp_path):
    """Both releases run on every request; the second must not disturb a slot a
    sibling upload took in the meantime."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=0)
        try:
            pipeline_status, lock = await _status_handles(rag)
            token = uuid4().hex
            sibling = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            assert await _reserve_enqueue_slot(rag, sibling)
            _spy_on_processing(rag, pipeline_status, lock, token)

            await pipeline_index_texts(
                rag,
                ["another document"],
                file_sources=["sibling.txt"],
                track_id="t-sibling",
                admission_token=token,
            )
            await _release_enqueue_slot(rag, token)

            async with lock:
                assert pipeline_status["pending_enqueues"] == 1
                assert sibling in pipeline_status["pending_enqueue_tokens"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# /documents/upload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_upload_releases_its_slot_before_driving_the_loop(tmp_path, capacity):
    """The upload path deadlocks exactly like ``/texts`` — same enqueue-then-drive
    shape, same reservation held by the caller's ``finally``."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await _status_handles(rag)
            source: Path = tmp_path / "upload.txt"
            source.write_text("a document arriving through /documents/upload")

            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            seen = _spy_on_processing(rag, pipeline_status, lock, token)
            await _freeze_as_manual_retry(pipeline_status, lock)

            try:
                await pipeline_index_file(
                    rag, source, "t-upload", admission_token=token
                )
            finally:
                await _release_enqueue_slot(rag, token)

            assert seen["pending_enqueues"] == 0
            assert seen["token_registered"] is False
            # Enqueued as PENDING with the parse deferred to the worker, so
            # the drain's cohort carries it exactly like a /texts document.
            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_upload_releases_its_slot_even_when_the_enqueue_rejects_the_file(tmp_path):
    """A refused file drives nothing, so the slot is not load-bearing — but it
    must not be held for the rest of the bg task either, or a drain waits on a
    request that already failed."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=0)
        try:
            pipeline_status, lock = await _status_handles(rag)
            # Refused by the unsafe-document-source guard, before any I/O, so
            # the file need not exist.
            refused = tmp_path / "bad\nname.txt"

            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            seen = _spy_on_processing(rag, pipeline_status, lock, token)

            await pipeline_index_file(rag, refused, "t-rejected", admission_token=token)

            # Nothing drove the loop, so the spy never ran.
            assert seen == {}
            async with lock:
                assert pipeline_status["pending_enqueues"] == 0
                assert token not in pipeline_status["pending_enqueue_tokens"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# End to end: the reported deadlock
# ---------------------------------------------------------------------------


def _chunking(tokenizer, content, *a, **k) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


class _Extract:
    """Entity extraction stub with a hook that fires inside the running run."""

    def __init__(self) -> None:
        self.fail = False
        self.on_call = None

    async def __call__(self, chunks, *args, **kwargs):
        if self.on_call is not None:
            hook, self.on_call = self.on_call, None
            await hook()
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_processing_rag(tmp_path, extract: _Extract) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"e2e-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = extract
    return rag


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


def test_a_manual_retry_queued_mid_texts_run_completes(tmp_path):
    """The reported deadlock, end to end.

    A ``/documents/texts`` bg task whose ``apipeline_process_enqueue_documents``
    became the processing run, with ``POST /documents/reprocess_failed`` queued
    while it works. Before the fix this never returned: DRAIN_TO_IDLE waited for
    ``pending_enqueues`` to reach 0, and the only reservation left was the
    running task's own token.

    The timeout is the assertion — without it the test hangs instead of failing.
    """

    async def _run():
        extract = _Extract()
        rag = await _build_processing_rag(tmp_path, extract)
        try:
            # A FAILED document for the manual retry to claim.
            extract.fail = True
            await rag.apipeline_enqueue_documents(
                input="body of old.txt", file_paths="old.txt"
            )
            await rag.apipeline_process_enqueue_documents()
            failed_id = compute_mdhash_id("old.txt", prefix="doc-")
            assert await _status_of(rag, failed_id) == DocStatus.FAILED.value

            # The retry request lands while the /texts run is extracting.
            extract.fail = False
            ingress = await get_pipeline_ingress(rag.workspace)
            request_id = uuid4().hex

            async def _queue_manual_retry():
                ingress.request_manual_retry(
                    request_id,
                    PipelineIngressMessage(
                        kind="rescan", retry_failed=True, request_id=request_id
                    ),
                )

            extract.on_call = _queue_manual_retry

            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            try:
                await asyncio.wait_for(
                    pipeline_index_texts(
                        rag,
                        ["fresh body"],
                        file_sources=["new.txt"],
                        track_id="t-e2e",
                        admission_token=token,
                    ),
                    timeout=30,
                )
            finally:
                await _release_enqueue_slot(rag, token)

            # The run drained, reset the FAILED doc and processed everything.
            new_id = compute_mdhash_id("new.txt", prefix="doc-")
            assert await _status_of(rag, new_id) == DocStatus.PROCESSED.value
            assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
            assert ingress.snapshot_manual_retries() == []  # ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
