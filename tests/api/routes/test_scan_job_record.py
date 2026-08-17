"""Bounded scan job record wired into /scan (LR2 Phase 4-b2, §8.6).

Startup order is reservation → job record → sticky manual intent → managed child
→ ``track_id``, with each step compensating the earlier ones on failure. The
record therefore exists before the response (an immediate status query cannot
404), the owning child finalises it (completed / failed / cancelled), and a
startup that never reaches takeover leaves a CANCELLED record plus a released
reservation — never an orphan RUNNING job holding capacity.
"""

import asyncio
import importlib
import sys
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.kg.scan_job_store import (  # noqa: E402
    SCAN_JOB_SAMPLE_LIMIT,
    AsyncioScanJobStore,
    ScanJobStatus,
)
from lightrag.kg.shared_storage import (  # noqa: E402
    get_namespace_data,
    get_pipeline_ingress,
    get_scan_job_store,
    initialize_pipeline_status,
)

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes
run_scanning_process = _document_routes.run_scanning_process

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """These rigs use stub rags (no ``initialize_storages``), so the module-level
    shared dicts — pipeline_status, the ingress mailbox and the scan job store
    registry — have to be set up explicitly."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield


class _StubRag:
    """Minimal rag surface the scan path touches with an EMPTY input dir."""

    def __init__(self):
        self.workspace = f"scanjob-{uuid4().hex[:8]}"
        self.process_calls = 0
        self.reset_calls = []
        self.drive_gate: asyncio.Event | None = None

    async def arollback_failed_custom_chunk_patches(self, **kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_reset_failed_for_scan(
        self, request_id, *, scan_owner_token=None
    ):
        self.reset_calls.append((request_id, scan_owner_token))
        return True

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1
        if self.drive_gate is not None:
            await self.drive_gate.wait()


def _endpoint(router, name):
    return [
        route.endpoint for route in router.routes if getattr(route, "name", "") == name
    ][-1]


async def _prepare(tmp_path):
    rag = _StubRag()
    await initialize_pipeline_status(workspace=rag.workspace)
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(rag, doc_manager)
    return rag, router


async def _await_managed(managed: set):
    for task in list(managed):
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_job_record_is_queryable_before_the_scan_finishes(tmp_path):
    """§8.6: the record must exist BEFORE the endpoint returns the track_id — an
    immediate status query cannot 404 — and the owning child finalises it."""

    async def _run():
        rag, router = await _prepare(tmp_path)
        scan = _endpoint(router, "scan_for_new_documents")
        status = _endpoint(router, "get_scan_job_status")

        # Park the child inside its queue drive so the job is observably RUNNING.
        rag.drive_gate = asyncio.Event()
        managed: set = set()
        response = await scan(managed)
        assert response.status == "scanning_started"

        running = await status(response.track_id)
        assert running.track_id == response.track_id
        assert running.status == "running"
        assert running.counts.get("discovered") == 0

        rag.drive_gate.set()
        await _await_managed(managed)

        done = await status(response.track_id)
        assert done.status == "completed"
        assert "0 discovered" in done.message
        # Bounded schema only: counters + the three fixed sample buckets.
        assert set(done.samples) == {"processed", "warning", "error"}

    asyncio.run(_run())


def test_capacity_exceeded_refuses_with_429_and_no_side_effects(tmp_path):
    """A full job store (every record a valid RUNNING job) refuses the scan with
    429 BEFORE any manual intent is published or any child started, and the
    reservation is released."""

    async def _run():
        rag, router = await _prepare(tmp_path)
        store = get_scan_job_store(rag.workspace)
        for index in range(200):  # past the per-workspace capacity
            store.create(f"filler-{index}", uuid4().hex)

        scan = _endpoint(router, "scan_for_new_documents")
        managed: set = set()
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            await scan(managed)
        assert excinfo.value.status_code == 429

        assert managed == set()
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status["scanning"] is False
        assert pipeline_status["scanning_exclusive"] is False
        assert pipeline_status["scanning_owner"] is None
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.snapshot_manual_retries() == []
        assert rag.reset_calls == []

    asyncio.run(_run())


def test_precommit_failure_cancels_the_job_and_releases_the_reservation(tmp_path):
    """Reverse compensation (§8.6 step 3): the manual-intent publish fails, so
    the job this endpoint created has no owner — it must end CANCELLED, not sit
    RUNNING holding capacity until its lease expires, and the scan reservation
    must be released."""

    async def _run():
        rag, router = await _prepare(tmp_path)
        ingress = await get_pipeline_ingress(rag.workspace)

        def _publish_boom(*args, **kwargs):
            raise RuntimeError("mailbox publish boom")

        ingress.request_manual_retry = _publish_boom

        scan = _endpoint(router, "scan_for_new_documents")
        managed: set = set()
        # The startup helper reports the failed takeover (the publish error
        # itself is logged by the helper).
        with pytest.raises(RuntimeError, match="failed to start"):
            await scan(managed)

        store = get_scan_job_store(rag.workspace)
        records = store.snapshot()
        assert [record["status"] for record in records] == ["cancelled"]
        assert "aborted before" in records[0]["message"]

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status["scanning"] is False
        assert pipeline_status["scanning_owner"] is None
        assert rag.reset_calls == []  # the child never ran

    asyncio.run(_run())


def test_classification_failure_marks_the_job_failed_with_an_error_sample(tmp_path):
    """A crashing classification phase must leave a FAILED record carrying a
    bounded error sample — not a RUNNING one waiting for the lease reaper."""

    async def _run():
        rag = _StubRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        store = get_scan_job_store(rag.workspace)
        track_id, job_token = "scan-fail", uuid4().hex
        assert store.create(track_id, job_token).record is not None

        doc_manager = DocumentManager(str(tmp_path))

        def _boom():
            raise RuntimeError("classification boom")

        doc_manager.iter_new_files = _boom
        await run_scanning_process(
            rag,
            doc_manager,
            track_id,
            manual_request_id=uuid4().hex,
            job_owner_token=job_token,
        )

        record = store.get(track_id)
        assert record["status"] == "failed"
        assert "classification boom" in record["message"]
        assert any(
            "classification boom" in item
            for item in record["samples"]["error"]["items"]
        )

    asyncio.run(_run())


def test_cancelled_scan_marks_the_job_cancelled(tmp_path):
    """A cancellation (shutdown / explicit) finalises the record synchronously —
    the finally never awaits to do it — so no RUNNING job is left behind."""

    async def _run():
        rag = _StubRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        store = get_scan_job_store(rag.workspace)
        track_id, job_token = "scan-cancel", uuid4().hex
        store.create(track_id, job_token)

        doc_manager = DocumentManager(str(tmp_path))

        def _cancelled():
            raise asyncio.CancelledError()

        doc_manager.iter_new_files = _cancelled
        with pytest.raises(asyncio.CancelledError):
            await run_scanning_process(
                rag,
                doc_manager,
                track_id,
                manual_request_id=uuid4().hex,
                job_owner_token=job_token,
            )

        assert store.get(track_id)["status"] == "cancelled"

    asyncio.run(_run())


def test_status_endpoint_404s_for_an_unknown_track_id(tmp_path):
    async def _run():
        _rag, router = await _prepare(tmp_path)
        status = _endpoint(router, "get_scan_job_status")
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            await status("scan-does-not-exist")
        assert excinfo.value.status_code == 404

    asyncio.run(_run())


def test_a_long_phase_does_not_lose_the_lease_under_a_live_owner(tmp_path, monkeypatch):
    """Fix-proof: the reporter renewed the lease only when the SCAN touched the
    record — per discovered file, per batch flush — and the final processing run
    touches it never. A drive longer than ``SCAN_JOB_LEASE_SECONDS`` therefore
    let the store reap a perfectly live owner's job to ABANDONED, and the scan's
    own COMPLETED transition then lost to that state: a finished scan reported
    as "owner presumed dead"."""

    async def _run():
        rag = _StubRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        # Compressed timings (lease 0.3s, renewal 0.05s) keep the real-time
        # window testable; the production ratio (renew = lease/3) is unchanged.
        store = AsyncioScanJobStore(rag.workspace, lease_seconds=0.3)
        monkeypatch.setattr(
            _document_routes, "_resolve_scan_job_store", lambda _rag: store
        )
        monkeypatch.setattr(_document_routes, "_SCAN_JOB_RENEW_SECONDS", 0.05)

        track_id, job_token = "scan-slow", uuid4().hex
        store.create(track_id, job_token)
        doc_manager = DocumentManager(str(tmp_path))

        # Park the scan inside the processing drive — the phase with no reporter
        # call of its own — for three lease periods.
        rag.drive_gate = asyncio.Event()
        scan = asyncio.create_task(
            run_scanning_process(
                rag,
                doc_manager,
                track_id,
                manual_request_id=uuid4().hex,
                job_owner_token=job_token,
            )
        )
        await asyncio.sleep(0.9)
        assert rag.process_calls == 1  # really parked in the drive
        assert store.get(track_id)["status"] == "running"

        rag.drive_gate.set()
        await scan
        assert store.get(track_id)["status"] == "completed"

    asyncio.run(_run())


def test_the_lease_heartbeat_never_outlives_its_scan(monkeypatch):
    """A heartbeat that survived its owner would renew a DEAD scan's record for
    ever, defeating the reaper it exists to work with. It retires on the scan
    task's completion, so the explicit cancel in the ``finally`` is a prompt
    exit, not the only one."""

    async def _run():
        monkeypatch.setattr(_document_routes, "_SCAN_JOB_RENEW_SECONDS", 0.02)
        store = AsyncioScanJobStore("heartbeat-test", lease_seconds=0.2)
        store.create("job", "owner")
        reporter = _document_routes._ScanJobReporter(store, "job", "owner")

        async def _scan_that_already_ended():
            return None

        scan_task = asyncio.create_task(_scan_that_already_ended())
        await scan_task

        # Started as if the finally never reached its cancel.
        heartbeat = asyncio.create_task(
            _document_routes._renew_scan_job_lease(reporter, scan_task)
        )
        await asyncio.wait_for(heartbeat, timeout=1.0)  # retires on its own

        # And the abandoned record is reapable, not held alive.
        await asyncio.sleep(0.25)
        assert store.get("job")["status"] == "abandoned"

    asyncio.run(_run())


def test_reporter_bounds_sample_calls_and_resyncs_the_cas_version():
    """The reporter's own budget keeps a million-file scan from costing one store
    call per warning: only ``SCAN_JOB_SAMPLE_LIMIT`` samples per bucket are ever
    sent, the rest are counted; and a concurrent version bump is re-synced once
    instead of silently dropping the update."""
    store = AsyncioScanJobStore("reporter-test")
    token = uuid4().hex
    store.create("job", token)
    reporter = _document_routes._ScanJobReporter(store, "job", token)

    for index in range(SCAN_JOB_SAMPLE_LIMIT + 5):
        reporter.sample("warning", f"warning {index}")
    reporter.count("discovered", 3)
    reporter.flush()

    record = store.get("job")
    assert len(record["samples"]["warning"]["items"]) == SCAN_JOB_SAMPLE_LIMIT
    assert record["samples"]["warning"]["dropped"] == 0  # never over-sent
    assert record["counts"]["warning_samples_suppressed"] == 5
    assert record["counts"]["discovered"] == 3

    # A concurrent writer bumps the version; the next update re-syncs and lands.
    store.update(
        "job", token, count_deltas={"other": 1}, expected_version=record["version"]
    )
    reporter.count("discovered", 1)
    reporter.flush()
    assert store.get("job")["counts"]["discovered"] == 4

    # A terminal record refuses further updates and disables the reporter.
    reporter.finish(ScanJobStatus.COMPLETED, "done")
    assert store.get("job")["status"] == "completed"
    assert reporter.enabled is False
    reporter.count("discovered", 99)
    reporter.flush()
    assert store.get("job")["counts"]["discovered"] == 4
