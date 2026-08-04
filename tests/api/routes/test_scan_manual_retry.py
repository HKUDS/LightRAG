"""Scan-driven manual retry intent (Phase 0 + LR2 Phase 4-b).

The scan endpoint publishes a sticky manual retry request AFTER its
reservation is granted; ``run_scanning_process`` then serves that request
ITSELF — the shared exclusive FAILED→PENDING reset runs BEFORE any file is
discovered (LR2 §8.1), so a file that fails during this scan cannot be
absorbed by this scan's own request. When the reset does not complete,
discovery is skipped and the request stays sticky for the standard drain
path; a fallback drive in the finally keeps the reset PENDING rows from
waiting for an unrelated trigger, and a cancellation skips that drive.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag import LightRAG  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.pipeline_ingress import (  # noqa: E402
    ManualRetryPublishResult,
    PipelineIngressMessage,
)
from lightrag.kg.shared_storage import (  # noqa: E402
    MANUAL_PHASE_IDLE,
    acquire_reservation,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    initialize_pipeline_status,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id  # noqa: E402

DocumentManager = _document_routes.DocumentManager
run_scanning_process = _document_routes.run_scanning_process

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


class _FlippableExtract:
    def __init__(self):
        self.fail = True

    async def __call__(self, chunks, *args, **kwargs):
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


class _SelectiveExtract:
    """Fail extraction for chunks whose content carries a marked substring.

    The marker set is mutated between phases so one document can fail first and
    succeed on retry while another always fails, and every attempt is recorded so
    a test can count how many times a given document was extracted."""

    def __init__(self, *markers: str):
        self.markers = set(markers)
        self.attempts: list[str] = []

    async def __call__(self, chunks, *args, **kwargs):
        items = chunks.values() if isinstance(chunks, dict) else chunks
        contents = [(chunk or {}).get("content", "") for chunk in items]
        self.attempts.extend(contents)
        if any(marker in content for content in contents for marker in self.markers):
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in contents]

    def attempts_for(self, marker: str) -> int:
        return sum(1 for content in self.attempts if marker in content)


async def _build_rag(tmp_path, extract) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"scanmr-{uuid4().hex[:8]}",
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


async def _make_failed_doc(rag, extract) -> str:
    extract.fail = True
    await rag.apipeline_enqueue_documents(input="doc body", file_paths="a.txt")
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id("a.txt", prefix="doc-")
    row = await rag.doc_status.get_by_id(doc_id)
    assert row["status"] == DocStatus.FAILED
    return doc_id


async def _publish_manual(rag) -> str:
    ingress = await get_pipeline_ingress(rag.workspace)
    request_id = uuid4().hex
    ingress.request_manual_retry(
        request_id,
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id=request_id),
    )
    return request_id


def test_scan_serves_manual_request_and_retries_failed(tmp_path):
    """No-new-files scan: the pre-discovery exclusive reset pulls the FAILED doc
    back in (the ONLY automatic-trigger path doing so) and ACKs the request; the
    scan's own drive then processes the resulting PENDING row."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # consumed + ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_classification_failure_fallback_drives_sticky_request(tmp_path):
    """Classification raising must not strand the published intent: the
    finally drives the queue once (storage-only) and the run consumes it."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            def boom():
                raise RuntimeError("classification boom")

            doc_manager.iter_new_files = boom
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_new_files_all_fail_to_enqueue_still_drives_sticky_request(
    tmp_path, monkeypatch
):
    """new_files are found but EVERY one fails to enqueue (duplicate / empty
    body / extraction error), so pipeline_index_files never drives the queue
    and returns False.  ``queue_drive_attempted`` must stay False so the
    classification-failure fallback still gives this scan's sticky manual
    request its run.

    Fix-proof: the branch used to set ``queue_drive_attempted = True`` before
    the call, so the fallback was skipped and the FAILED doc stayed stranded
    until an unrelated trigger."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            # A brand-new basename (no doc_status row) → classified as new;
            # the file never has to exist on disk because enqueue is stubbed
            # to fail for every file below.
            doc_manager.iter_new_files = lambda: [
                Path(str(tmp_path / "inputs" / "brand_new.txt"))
            ]

            async def _enqueue_fails(rag, file_path, track_id=None, from_scan=False):
                return (False, None)

            monkeypatch.setattr(
                _document_routes, "pipeline_enqueue_file", _enqueue_fails
            )

            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED  # fallback drove it
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # consumed + ACKed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_cancelled_scan_does_not_drive_processing_after_reset(tmp_path):
    """ANY cancellation (shutdown or explicit — indistinguishable here) skips
    the fallback drive.

    The exclusive reset already ran (it precedes discovery), so per LR2 §7.6
    the reset rows stay PENDING and the request is ACKed; what a cancellation
    must NOT do is start a full processing run while shutdown waits for us."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            def cancelled():
                raise asyncio.CancelledError()

            doc_manager.iter_new_files = cancelled
            with pytest.raises(asyncio.CancelledError):
                await run_scanning_process(
                    rag, doc_manager, manual_request_id=request_id
                )

            row = await rag.doc_status.get_by_id(doc_id)
            # Reset persisted, processing never started (PENDING, not PROCESSED).
            assert row["status"] == DocStatus.PENDING
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # ACKed by the reset
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_failed_reset_precedes_new_file_enqueue(tmp_path, monkeypatch):
    """LR2 §8.1 ordering invariant: the exclusive FAILED reset runs BEFORE this
    scan discovers/enqueues anything, so a file that fails DURING the scan is
    NOT absorbed by the scan's own manual request.

    Fix-proof: with the old order (enqueue → drive → drain → reset), the newly
    enqueued file was processed as part of the drain's AUTO backlog, so its
    fresh FAILED row was still visible to the same request's EXCLUSIVE_RESET
    and got retried immediately — two extraction attempts for one scan. The
    pre-existing FAILED doc must still be retried (once), proving the reset
    itself did run."""

    async def _run():
        extract = _SelectiveExtract("OLDDOC", "BOOM")
        rag = await _build_rag(tmp_path, extract)
        try:
            # A pre-existing FAILED doc WITH content (reset-eligible).
            await rag.apipeline_enqueue_documents(
                input="OLDDOC body", file_paths="old.txt"
            )
            await rag.apipeline_process_enqueue_documents()
            old_id = compute_mdhash_id("old.txt", prefix="doc-")
            assert (await rag.doc_status.get_by_id(old_id))[
                "status"
            ] == DocStatus.FAILED
            extract.markers.discard("OLDDOC")  # a retry of it now succeeds

            # A brand-new input file whose extraction ALWAYS fails.
            input_dir = tmp_path / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("INPUT_DIR", str(input_dir))
            (input_dir / "new.txt").write_text("BOOM payload", encoding="utf-8")

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(input_dir))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            assert (await rag.doc_status.get_by_id(old_id))[
                "status"
            ] == DocStatus.PROCESSED
            failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
            assert [doc.file_path for doc in failed.values()] == ["new.txt"]
            # The invariant: exactly ONE attempt at the scan's own new file.
            assert extract.attempts_for("BOOM") == 1
            assert extract.attempts_for("OLDDOC") == 2  # initial + reset retry
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []  # ACKed by the reset
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_abandoned_reset_skips_discovery_and_leaves_request_sticky(
    tmp_path, monkeypatch
):
    """The ordering rule under failure: a reset that does NOT complete must
    abort the scan before discovery — enqueuing a new file would put it ahead
    of the still-sticky request, which is exactly what §8.1 forbids. The
    finally's drive then serves the request through the standard drain path."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            input_dir = tmp_path / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("INPUT_DIR", str(input_dir))
            (input_dir / "new.txt").write_text("fresh body", encoding="utf-8")

            request_id = await _publish_manual(rag)

            async def _reset_abandoned(request, *, scan_owner_token=None):
                return False

            monkeypatch.setattr(
                rag, "apipeline_reset_failed_for_scan", _reset_abandoned
            )

            doc_manager = DocumentManager(str(input_dir))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            # Discovery never ran: no doc_status row for the new file, and the
            # source file is still on disk (not archived).
            rows = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
            assert all("new.txt" not in (doc.file_path or "") for doc in rows.values())
            assert (input_dir / "new.txt").exists()
            # The fallback drive served the sticky request via the standard path.
            assert (await rag.doc_status.get_by_id(doc_id))[
                "status"
            ] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_reset_takes_busy_under_its_own_exclusive_fence(tmp_path):
    """The reset needs ``busy`` while the scan holds ``scanning_exclusive``.

    The exemption is owner-checked: a foreign token is still fenced out (no
    reset, request stays sticky), the fence owner gets the slot, and the reset
    leaves NO freeze behind — uploads are refused only inside its window, while
    the scan's own reservation survives for the classification phase."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False
            request_id = await _publish_manual(rag)

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )
            scanning_token = uuid4().hex
            reservation = await acquire_reservation(
                pipeline_status,
                pipeline_status_lock,
                owner_key="scanning_owner",
                owner=scanning_token,
                owner_kind="scan",
                flags={"scanning": True, "scanning_exclusive": True},
                reject_when=(),
            )
            assert reservation.acquired

            ingress = await get_pipeline_ingress(rag.workspace)
            assert (
                await rag.apipeline_reset_failed_for_scan(
                    request_id, scan_owner_token=uuid4().hex
                )
                is False
            )
            assert (await rag.doc_status.get_by_id(doc_id))[
                "status"
            ] == DocStatus.FAILED
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                request_id
            ]

            assert (
                await rag.apipeline_reset_failed_for_scan(
                    request_id, scan_owner_token=scanning_token
                )
                is True
            )
            assert (await rag.doc_status.get_by_id(doc_id))[
                "status"
            ] == DocStatus.PENDING
            assert ingress.snapshot_manual_retries() == []
            assert pipeline_status["busy"] is False
            assert pipeline_status["busy_owner"] is None
            assert pipeline_status["manual_freeze_requested"] is False
            assert pipeline_status["manual_resetting"] is False
            assert pipeline_status["manual_owner"] is None
            assert pipeline_status["manual_phase"] == MANUAL_PHASE_IDLE
            # The scan still owns its reservation: classification runs next.
            assert pipeline_status["scanning"] is True
            assert pipeline_status["scanning_exclusive"] is True
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_endpoint_refused_while_earlier_manual_request_queued(tmp_path):
    """LR2 §8.1: a scan runs its own exclusive FAILED reset, so it must not be
    granted while an EARLIER un-ACKed manual request is queued — that would
    jump the manual FIFO and deadlock that request's run (the scan fence
    refuses every processing reservation). Refusal has zero side effects."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await initialize_pipeline_status(workspace=rag.workspace)
            request_id = await _publish_manual(rag)  # an earlier /reprocess_failed

            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            router = _document_routes.create_document_routes(rag, doc_manager)
            scan_endpoint = [
                route.endpoint
                for route in router.routes
                if getattr(route, "name", "") == "scan_for_new_documents"
            ][-1]

            response = await scan_endpoint(set())
            assert response.status == "scanning_skipped_pipeline_busy"
            assert request_id[:8] in response.message

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert pipeline_status["scanning"] is False
            assert pipeline_status["scanning_exclusive"] is False
            assert pipeline_status["scanning_owner"] is None
            ingress = await get_pipeline_ingress(rag.workspace)
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                request_id
            ]  # untouched: the earlier request keeps its place
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_refuses_when_the_manual_channel_is_full(tmp_path, monkeypatch):
    """LR2 §10.1: a scan whose retry intent cannot be published must not start —
    its exclusive FAILED reset would never be acknowledged. Capacity is the one
    publish refusal worth retrying unchanged (429), and the reservation plus the
    job record are compensated because nothing was handed off."""

    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await initialize_pipeline_status(workspace=rag.workspace)
            ingress = await get_pipeline_ingress(rag.workspace)
            monkeypatch.setattr(
                ingress,
                "request_manual_retry",
                lambda *args, **kwargs: ManualRetryPublishResult.CAPACITY_EXCEEDED,
            )

            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            router = _document_routes.create_document_routes(rag, doc_manager)
            scan_endpoint = [
                route.endpoint
                for route in router.routes
                if getattr(route, "name", "") == "scan_for_new_documents"
            ][-1]

            with pytest.raises(_document_routes.HTTPException) as excinfo:
                await scan_endpoint(set())

            assert excinfo.value.status_code == 429
            assert excinfo.value.headers["Retry-After"]

            # Nothing owned: the scan reservation was released again.
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert pipeline_status["scanning"] is False
            assert pipeline_status["scanning_exclusive"] is False
            assert pipeline_status["busy"] is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
