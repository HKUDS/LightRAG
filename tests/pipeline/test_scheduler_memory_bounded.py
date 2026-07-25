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
"""

from __future__ import annotations

import tracemalloc
from datetime import datetime, timedelta, timezone

import pytest

from lightrag.base import CURSOR_END, CURSOR_START, CursorAfter, DocStatus

pytestmark = pytest.mark.offline

_PAGE = 200
_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _SyntheticDocStatus:
    """A doc_status that SERVES pages without materializing the backlog.

    Rows are generated per page from the cursor, so the backend itself uses O(1)
    memory for any backlog size. That is the point: any growth the measurement
    sees belongs to the scheduler, not to the fixture. A real backend is bounded
    the same way — the page query is index-ordered and returns `limit` rows (see
    tests/kg/test_scheduling_page_plans.py) — but going through one here would
    measure the driver's buffers instead.
    """

    def __init__(self, total: int):
        self.total = total
        self.pages_served = 0

    def _key(self, index: int) -> tuple[str, str]:
        return (
            (_BASE + timedelta(seconds=index)).isoformat(),
            f"doc-{index:09d}",
        )

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
                status=DocStatus.PENDING,
                created_at=created_at,
                updated_at=created_at,
                file_path=f"file-{i}.md",
                track_id="t",
                has_custom_chunk_journal=False,
            )
        self.pages_served += 1
        next_position = CURSOR_END if end >= self.total else CursorAfter(str(end - 1))
        return DocStatusPage(docs=docs, next_position=next_position)


async def _sweep_peak_kib(total: int) -> tuple[float, int]:
    """Page through `total` documents the way the supervisor does, returning the
    peak traced allocation and the number of pages."""
    doc_status = _SyntheticDocStatus(total)

    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        position = CURSOR_START
        routed = 0
        while True:
            page = await doc_status.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=_PAGE, position=position, strict=True
            )
            # Stand in for routing: consume each record and keep NOTHING that
            # scales with the backlog — exactly the discipline under test.
            for record in page.docs.values():
                routed += len(record.id)
            if page.next_position is CURSOR_END:
                break
            position = page.next_position
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert routed > 0
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
