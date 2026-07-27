"""The scheduler's memory does not grow with the backlog (LR2 §14, §18 item 6).

This is the claim the whole bounded-scheduling rework exists to make, and until
now it was supported only by structural argument: pages are bounded, so memory
must be. Structure is the right argument, but it does not catch the way this
regresses in practice — one accumulating list, one `list(...)` around a generator,
one forgotten `dict` keyed by doc id, and the sweep is O(backlog) again while
every test still passes because the RESULTS are unchanged.

So this measures it. `tracemalloc` rather than RSS: RSS is dominated by the
allocator's arena behaviour and by whatever else the process has touched, so it is
far too noisy to assert on, while tracemalloc attributes Python allocations to the
code under test.

The assertion is a RATIO between two backlog sizes, never an absolute byte count.
An absolute threshold would encode this machine's interpreter and this fixture's
row width, and would be re-tuned into meaninglessness the first time it failed.
A ratio has a defensible meaning: sweeping 10x more documents at the same page
size must not cost 10x more memory.

The loop under measurement is the PRODUCTION one. An earlier version of this file
hand-rolled its own `get_docs_by_statuses_page` loop, which made it a test of the
measurement rather than of the scheduler: adding the very accumulator described
above to `_next_scheduling_page` would not have failed it. Both sweeps the
pipeline owns are driven here through their real methods —
`_next_scheduling_page` (the AUTO drain) and `_next_failed_page` +
`_reset_failed_page` (the manual EXCLUSIVE_RESET) — so page hydration, status
re-filtering and the reset's own bookkeeping are all inside the measurement.

The second, process-level acceptance layer lives in
`test_scheduler_memory_bounded_e2e.py`: it drives a full
`apipeline_process_enqueue_documents` run with the production supervisor,
feeder and all three worker queues, plus a production `run_scanning_process`
over a real large directory, in isolated child processes while sampling RSS.
Keep the two layers separate: this file pinpoints Python allocation regressions
inside the page/reset helpers; the companion catches composition, native
allocation and filesystem-lifecycle regressions.
"""

from __future__ import annotations

import asyncio
import tracemalloc
from datetime import datetime, timedelta, timezone

import pytest

from lightrag.base import CURSOR_END, CURSOR_START, CursorAfter, DocStatus
from lightrag.pipeline import _AUTO_RESUME_DOC_STATUSES, _PipelineMixin

pytestmark = pytest.mark.offline

_PAGE = 200
_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_TOKEN = "owner-token"


class _SyntheticDocStatus:
    """A doc_status that SERVES pages without materializing the backlog.

    Rows are generated per page from the cursor, so the backend itself uses O(1)
    memory for any backlog size. That is the point: any growth the measurement
    sees belongs to the scheduler, not to the fixture. A real backend is bounded
    the same way — the page query is index-ordered and returns `limit` rows (see
    tests/kg/test_scheduling_page_plans.py) — but going through one here would
    measure the driver's buffers instead.
    """

    # The production sweep probes this before reading full_docs content.
    supports_strict_point_reads = True

    def __init__(self, total: int, status: DocStatus = DocStatus.PENDING):
        self.total = total
        self.status = status
        self.pages_served = 0
        self.upserted = 0

    def _key(self, index: int) -> tuple[str, str]:
        return (
            (_BASE + timedelta(seconds=index)).isoformat(),
            f"doc-{index:09d}",
        )

    def _index_of(self, doc_id: str) -> int:
        return int(doc_id.rsplit("-", 1)[1])

    async def get_docs_by_statuses_page(
        self, statuses, *, limit, position=CURSOR_START, strict=False
    ):
        from lightrag.base import DocSchedulingRecord, DocStatusPage

        start = 0
        if isinstance(position, CursorAfter):
            start = int(position.opaque) + 1
        end = min(start + limit, self.total)
        docs = {}
        for i in range(start, end):
            created_at, doc_id = self._key(i)
            docs[doc_id] = DocSchedulingRecord(
                id=doc_id,
                status=self.status,
                created_at=created_at,
                updated_at=created_at,
                file_path=f"file-{i}.md",
                track_id="t",
                has_custom_chunk_journal=False,
            )
        self.pages_served += 1
        next_position = CURSOR_END if end >= self.total else CursorAfter(str(end - 1))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def get_full_docs_by_ids(self, ids, *, strict=False):
        """Hydration is part of the production page, so it is part of the
        measurement: one full row per id, discarded with the page."""
        from lightrag.base import DocProcessingStatus

        hydrated = {}
        for doc_id in ids:
            index = self._index_of(doc_id)
            created_at, _ = self._key(index)
            hydrated[doc_id] = DocProcessingStatus(
                content_summary=f"summary-{index}",
                content_length=64,
                status=self.status,
                created_at=created_at,
                updated_at=created_at,
                file_path=f"file-{index}.md",
                track_id="t",
                chunks_count=0,
                chunks_list=[],
                metadata={},
            )
        return hydrated

    async def get_by_id_strict(self, doc_id: str):
        # full_docs content probe for the FAILED reset.
        return {"content": "body", "file_path": f"file-{self._index_of(doc_id)}.md"}

    async def upsert(self, data: dict) -> None:
        # The reset's page write: counted, never retained.
        self.upserted += len(data)


def _pipeline_over(doc_status) -> _PipelineMixin:
    """The real mixin, bound to the synthetic backend.

    Only the attributes the two sweeps touch are provided — the point is to run
    `_next_scheduling_page` / `_next_failed_page` as written, not to build a
    LightRAG."""

    class _Pipeline(_PipelineMixin):
        def __init__(self):
            self.doc_status = doc_status
            self.full_docs = doc_status
            self.pipeline_scheduling_page_size = _PAGE

    return _Pipeline()


async def _sweep_peak_kib(total: int) -> tuple[float, int]:
    """Drive the production AUTO sweep over `total` documents, returning the peak
    traced allocation and the number of pages."""
    doc_status = _SyntheticDocStatus(total)
    pipeline = _pipeline_over(doc_status)

    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        position = CURSOR_START
        routed = 0
        while True:
            docs, position = await pipeline._next_scheduling_page(
                _AUTO_RESUME_DOC_STATUSES, position
            )
            # Stand in for routing: consume each record and keep NOTHING that
            # scales with the backlog — exactly the discipline under test.
            for doc_id in docs:
                routed += len(doc_id)
            if position is CURSOR_END:
                break
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert routed > 0
    return peak / 1024, doc_status.pages_served


async def _reset_peak_kib(total: int) -> tuple[float, int]:
    """Drive the production manual EXCLUSIVE_RESET sweep (`_next_failed_page` +
    `_reset_failed_page`) over `total` FAILED documents."""
    doc_status = _SyntheticDocStatus(total, status=DocStatus.FAILED)
    pipeline = _pipeline_over(doc_status)
    pipeline_status = {"busy_owner": {"token": _TOKEN}, "history_messages": []}
    lock = asyncio.Lock()

    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        position = CURSOR_START
        while True:
            docs, position = await pipeline._next_failed_page(position)
            if docs:
                await pipeline._reset_failed_page(docs, _TOKEN, pipeline_status, lock)
            if position is CURSOR_END:
                break
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert doc_status.upserted == total
    return peak / 1024, doc_status.pages_served


async def test_sweep_peak_memory_does_not_track_the_backlog():
    """10x the documents, same page size: memory must stay flat, not scale."""
    small_kib, small_pages = await _sweep_peak_kib(1_000)
    large_kib, large_pages = await _sweep_peak_kib(10_000)

    assert large_pages == 10 * small_pages  # the work really did grow 10x
    # Generous headroom: this asserts "does not scale", not a byte budget. A
    # per-document leak at 10x would blow well past 3x.
    assert large_kib < max(3.0 * small_kib, small_kib + 256), (
        f"peak grew with the backlog: {small_kib:.1f} KiB at 1k docs, "
        f"{large_kib:.1f} KiB at 10k docs"
    )


async def test_failed_reset_peak_memory_does_not_track_the_backlog():
    """The other production sweep: the manual retry's FAILED→PENDING reset builds
    a `docs_to_reset` dict per page, and it must be released with the page rather
    than accumulated across the walk."""
    small_kib, small_pages = await _reset_peak_kib(1_000)
    large_kib, large_pages = await _reset_peak_kib(10_000)

    assert large_pages == 10 * small_pages
    assert large_kib < max(3.0 * small_kib, small_kib + 256), (
        f"reset peak grew with the backlog: {small_kib:.1f} KiB at 1k FAILED, "
        f"{large_kib:.1f} KiB at 10k FAILED"
    )


async def test_page_size_not_backlog_size_sets_the_peak():
    """The complement: memory SHOULD grow with the page size. Without this, a
    scheduler that accidentally kept nothing at all would pass the test above
    while proving nothing about where the bound comes from."""
    global _PAGE
    original = _PAGE
    try:
        _PAGE = 50
        small_page_kib, _ = await _sweep_peak_kib(10_000)
        _PAGE = 2_000
        large_page_kib, _ = await _sweep_peak_kib(10_000)
    finally:
        _PAGE = original

    assert large_page_kib > small_page_kib, (
        "peak did not respond to page size at all, so the measurement is not "
        f"observing the page: {small_page_kib:.1f} vs {large_page_kib:.1f} KiB"
    )


async def test_a_backlog_sized_accumulator_is_caught():
    """Fix-proof for the measurement itself: the exact regression it exists to
    catch — one list that keeps every routed record — must fail the ratio."""

    async def _leaky_sweep_peak_kib(total: int) -> float:
        doc_status = _SyntheticDocStatus(total)
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            position = CURSOR_START
            accumulated = []  # the bug
            while True:
                page = await doc_status.get_docs_by_statuses_page(
                    [DocStatus.PENDING], limit=_PAGE, position=position, strict=True
                )
                accumulated.extend(page.docs.values())
                if page.next_position is CURSOR_END:
                    break
                position = page.next_position
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert accumulated
        return peak / 1024

    small_kib = await _leaky_sweep_peak_kib(1_000)
    large_kib = await _leaky_sweep_peak_kib(10_000)

    assert large_kib >= 3.0 * small_kib, (
        "the ratio assertion would not have caught a backlog-sized accumulator: "
        f"{small_kib:.1f} -> {large_kib:.1f} KiB"
    )
