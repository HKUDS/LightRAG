"""ONE shared scheduling contract suite for all five doc_status backends (LR2 §13.1).

Before this, each backend had its own hand-written file. Every one of them is
thorough about the things its author was thinking about, and that is exactly the
problem: the assertion sets drift, so a backend can be green in its own file while
violating an invariant only some other file happens to check. §13.1 asks for a
shared parametrized suite for that reason, and this is it — every case below runs
against every backend, or the backend is reported as unavailable.

JSON needs no service and always runs. To run the other four:

    docker run -d -p 6379:6379 redis:latest
    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=... -e POSTGRES_DB=... \\
        gzdaniel/postgres-for-rag:pg18-age-pgvector
    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local:8
    docker run -d -p 9200:9200 -e discovery.type=single-node \\
        -e DISABLE_SECURITY_PLUGIN=true opensearchproject/opensearch:3

Each backend gets a fresh random workspace per test: the contract covers derived
indexes and source multimaps, so leakage between cases would manufacture
false Conflicts.

Deliberately NOT duplicated here: backend-internal mechanics (Redis WATCH/MULTI
half-commit behaviour, PG SQL text, OpenSearch mapping shape). Those are properly
per-backend and stay in the per-backend files; this suite covers only what every
backend must agree on.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.exceptions import SourceConflictRepairCASError

from .conftest_scheduling_backends import BACKEND_PARAMS


@pytest.fixture(params=BACKEND_PARAMS)
async def doc_status(request, tmp_path):
    spec = request.param
    spec.skip_unless_available()
    storage = await spec.build(tmp_path)
    try:
        yield storage
    finally:
        try:
            await storage.drop()
        finally:
            await storage.finalize()


# ---------------------------------------------------------------------------
# Fixture data helpers.
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _stamp(offset_seconds: int) -> str:
    return (_BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()


def _row(
    *,
    status: DocStatus,
    created_at: str | None,
    file_path: str = "doc.md",
    summary: str = "body",
    **extra,
) -> dict:
    """A doc_status row shaped exactly like the pipeline writes one.

    No ``content`` key: doc_status stores ``content_summary`` /
    ``content_length`` and the body lives in full_docs. Writing a field the
    pipeline never writes would test the backends' tolerance for foreign keys
    instead of the contract (and OpenSearch legitimately rejects it).

    ``created_at=None`` OMITS the key — the "missing" case the sort contract
    talks about. An empty STRING is a different thing: PostgreSQL and OpenSearch
    type that column as a date and reject ``''`` outright, which is a stronger
    guarantee than ordering it.
    """
    row = {
        "status": status.value,
        "updated_at": created_at or _stamp(0),
        "content_summary": summary[:32],
        "content_length": len(summary),
        "file_path": file_path,
        "chunks_list": [],
        "chunks_count": 0,
        "track_id": "track-1",
    }
    if created_at is not None:
        row["created_at"] = created_at
    row.update(extra)
    return row


async def _seed(doc_status, rows: dict[str, dict]) -> None:
    await doc_status.upsert(rows)
    await doc_status.index_done_callback()


async def _drain(doc_status, statuses, *, limit) -> list[str]:
    """Walk the whole sweep, returning ids in the order the pages produced them.

    Termination is ``next_position is CURSOR_END`` and NOTHING else — an empty
    page is not the end (a page may be fully filtered yet not exhausted), which
    is the single most commonly broken part of this contract.
    """
    seen: list[str] = []
    position = CURSOR_START
    for _ in range(200):  # guard against a cursor that never advances
        page = await doc_status.get_docs_by_statuses_page(
            statuses, limit=limit, position=position, strict=True
        )
        seen.extend(page.docs.keys())
        if page.next_position is CURSOR_END:
            return seen
        assert isinstance(page.next_position, CursorAfter)
        position = page.next_position
    pytest.fail(f"sweep did not terminate after 200 pages; collected {len(seen)}")


# ---------------------------------------------------------------------------
# Sort order and cursor semantics.
# ---------------------------------------------------------------------------


async def test_equal_timestamps_are_ordered_by_id(doc_status):
    """``created_at`` alone is not a total order; without the id tiebreaker a
    keyset page can skip or repeat a record."""
    same = _stamp(0)
    await _seed(
        doc_status,
        {
            "doc-c": _row(status=DocStatus.PENDING, created_at=same, file_path="c.md"),
            "doc-a": _row(status=DocStatus.PENDING, created_at=same, file_path="a.md"),
            "doc-b": _row(status=DocStatus.PENDING, created_at=same, file_path="b.md"),
        },
    )

    assert await _drain(doc_status, [DocStatus.PENDING], limit=10) == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]


async def test_limit_one_paging_loses_and_repeats_nothing(doc_status):
    ids = {
        f"doc-{i:02d}": _row(status=DocStatus.PENDING, created_at=_stamp(i))
        for i in range(7)
    }
    await _seed(doc_status, ids)

    walked = await _drain(doc_status, [DocStatus.PENDING], limit=1)

    assert walked == sorted(ids)  # every record, exactly once, in key order


async def test_multi_status_pages_merge_into_one_key_order(doc_status):
    """The sweep is ordered by ``(created_at, id)`` ACROSS statuses, not
    concatenated per status."""
    await _seed(
        doc_status,
        {
            "doc-1": _row(status=DocStatus.PENDING, created_at=_stamp(1)),
            "doc-2": _row(status=DocStatus.FAILED, created_at=_stamp(2)),
            "doc-3": _row(status=DocStatus.PENDING, created_at=_stamp(3)),
            "doc-4": _row(status=DocStatus.FAILED, created_at=_stamp(4)),
        },
    )

    walked = await _drain(doc_status, [DocStatus.PENDING, DocStatus.FAILED], limit=2)

    assert walked == ["doc-1", "doc-2", "doc-3", "doc-4"]


async def test_a_row_with_no_created_at_fails_the_strict_page(doc_status):
    """Only legacy rows and external edits can lack ``created_at`` — all three
    write paths stamp it and ``update_doc_status_fields`` refuses to change it.

    All five backends treat such a row as UNUSABLE rather than as a
    first-sorted one, and that is the right call: ``DocSchedulingRecord`` requires
    ``created_at``, so there is no record to return, and §4.5 says a strict query
    is complete or it raises. Silently dropping it would let a corrupt row sit
    unprocessed and unreported forever.
    """
    await _seed(
        doc_status,
        {
            "doc-good": _row(status=DocStatus.PENDING, created_at=_stamp(10)),
            "doc-nostamp": _row(status=DocStatus.PENDING, created_at=None),
        },
    )

    with pytest.raises(Exception) as excinfo:
        await doc_status.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=10, position=CURSOR_START, strict=True
        )
    assert "created_at" in str(excinfo.value) or "doc-nostamp" in str(excinfo.value)


async def test_a_relaxed_sweep_skips_the_unusable_row_and_still_advances(doc_status):
    """Relaxed mode is the diagnostic path (admin listings), and there the bad row
    must be CONSUMED, not re-read: a cursor that sticks on it would loop forever
    instead of showing the operator the rest of the backlog."""
    await _seed(
        doc_status,
        {
            "doc-good": _row(status=DocStatus.PENDING, created_at=_stamp(10)),
            "doc-nostamp": _row(status=DocStatus.PENDING, created_at=None),
        },
    )

    seen: list[str] = []
    position = CURSOR_START
    for _ in range(20):
        page = await doc_status.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=position, strict=False
        )
        seen.extend(page.docs)
        if page.next_position is CURSOR_END:
            break
        position = page.next_position
    else:  # pragma: no cover - the loop guard
        pytest.fail("relaxed sweep stuck on the unusable row")

    assert seen == ["doc-good"]


async def test_the_end_is_the_cursor_not_an_empty_page(doc_status):
    await _seed(
        doc_status,
        {"doc-1": _row(status=DocStatus.PENDING, created_at=_stamp(1))},
    )

    first = await doc_status.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, position=CURSOR_START, strict=True
    )
    assert set(first.docs) == {"doc-1"}

    # Whether the backend already knows it is done or needs one more call, the
    # sweep must reach CURSOR_END without ever returning doc-1 twice.
    position = first.next_position
    extra: list[str] = []
    while position is not CURSOR_END:
        page = await doc_status.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=position, strict=True
        )
        extra.extend(page.docs)
        position = page.next_position
    assert extra == []


async def test_an_empty_status_set_terminates_immediately(doc_status):
    await _seed(
        doc_status,
        {"doc-1": _row(status=DocStatus.PROCESSED, created_at=_stamp(1))},
    )

    page = await doc_status.get_docs_by_statuses_page(
        [DocStatus.FAILED], limit=5, position=CURSOR_START, strict=True
    )

    assert page.docs == {}
    assert page.next_position is CURSOR_END


# ---------------------------------------------------------------------------
# created_at immutability — the sort key a sweep depends on.
# ---------------------------------------------------------------------------


async def test_a_status_transition_preserves_created_at(doc_status):
    created = _stamp(5)
    await _seed(
        doc_status, {"doc-1": _row(status=DocStatus.FAILED, created_at=created)}
    )

    await doc_status.update_doc_status_fields(
        "doc-1", {"status": DocStatus.PENDING.value, "error_msg": None}
    )

    page = await doc_status.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=5, position=CURSOR_START, strict=True
    )
    assert page.docs["doc-1"].created_at == created


async def test_update_fields_refuses_to_move_the_sort_key(doc_status):
    created = _stamp(5)
    await _seed(
        doc_status, {"doc-1": _row(status=DocStatus.PENDING, created_at=created)}
    )

    with pytest.raises(ValueError):
        await doc_status.update_doc_status_fields("doc-1", {"created_at": _stamp(999)})

    page = await doc_status.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=5, position=CURSOR_START, strict=True
    )
    assert page.docs["doc-1"].created_at == created


# ---------------------------------------------------------------------------
# Strict batch reads.
# ---------------------------------------------------------------------------


async def test_batch_read_returns_only_confirmed_ids(doc_status):
    await _seed(
        doc_status,
        {
            "doc-1": _row(status=DocStatus.PENDING, created_at=_stamp(1)),
            "doc-2": _row(status=DocStatus.PENDING, created_at=_stamp(2)),
        },
    )

    found = await doc_status.get_docs_by_ids(
        ["doc-1", "doc-missing", "doc-2"], strict=True
    )

    assert set(found) == {"doc-1", "doc-2"}
    assert found["doc-1"].status is DocStatus.PENDING


async def test_batch_read_of_nothing_is_not_an_error(doc_status):
    assert await doc_status.get_docs_by_ids([], strict=True) == {}
    assert await doc_status.get_full_docs_by_ids([], strict=True) == {}


async def test_the_scheduling_projection_omits_the_big_fields(doc_status):
    """A page must stay O(page_size × small constant); the projection is the
    reason it does."""
    await _seed(
        doc_status,
        {
            "doc-1": _row(
                status=DocStatus.PENDING,
                created_at=_stamp(1),
                content="x" * 5000,
                chunks_list=[f"chunk-{i}" for i in range(50)],
                chunks_count=50,
            )
        },
    )

    record = (await doc_status.get_docs_by_ids(["doc-1"], strict=True))["doc-1"]

    assert not hasattr(record, "content")
    assert not hasattr(record, "chunks_list")


async def test_full_hydration_carries_what_the_projection_dropped(doc_status):
    await _seed(
        doc_status,
        {
            "doc-1": _row(
                status=DocStatus.PENDING,
                created_at=_stamp(1),
                chunks_list=["chunk-a"],
                chunks_count=1,
            )
        },
    )

    full = await doc_status.get_full_docs_by_ids(["doc-1"], strict=True)

    assert full["doc-1"].chunks_list == ["chunk-a"]
    assert full["doc-1"].created_at == _stamp(1)


# ---------------------------------------------------------------------------
# Strict point read and strict count.
# ---------------------------------------------------------------------------


async def test_strict_point_read_distinguishes_absent_from_unknown(doc_status):
    await _seed(
        doc_status, {"doc-1": _row(status=DocStatus.PENDING, created_at=_stamp(1))}
    )

    assert (await doc_status.get_by_id_strict("doc-1")) is not None
    # A confirmed absence is None; an unconfirmable one would raise (per backend).
    assert (await doc_status.get_by_id_strict("doc-missing")) is None


async def test_strict_count_counts_the_requested_statuses_only(doc_status):
    await _seed(
        doc_status,
        {
            "doc-1": _row(status=DocStatus.PENDING, created_at=_stamp(1)),
            "doc-2": _row(status=DocStatus.PROCESSING, created_at=_stamp(2)),
            "doc-3": _row(status=DocStatus.PROCESSED, created_at=_stamp(3)),
            "doc-4": _row(status=DocStatus.FAILED, created_at=_stamp(4)),
        },
    )

    active = await doc_status.count_docs_by_statuses(
        [DocStatus.PENDING, DocStatus.PROCESSING], strict=True
    )

    assert active == 2
    assert await doc_status.count_docs_by_statuses([], strict=True) == 0


async def test_count_and_page_agree_after_a_transition(doc_status):
    """The count and the sweep read the same index; a backend that maintains one
    and not the other drifts silently."""
    await _seed(
        doc_status,
        {
            f"doc-{i}": _row(status=DocStatus.FAILED, created_at=_stamp(i))
            for i in range(5)
        },
    )

    await doc_status.update_doc_status_fields(
        "doc-2", {"status": DocStatus.PENDING.value}
    )

    assert await doc_status.count_docs_by_statuses([DocStatus.FAILED], strict=True) == 4
    assert (
        await doc_status.count_docs_by_statuses([DocStatus.PENDING], strict=True) == 1
    )
    assert await _drain(doc_status, [DocStatus.PENDING], limit=10) == ["doc-2"]
    assert "doc-2" not in await _drain(doc_status, [DocStatus.FAILED], limit=10)


# ---------------------------------------------------------------------------
# Typed source resolution.
# ---------------------------------------------------------------------------


async def test_source_absent_when_nothing_claims_the_key(doc_status):
    assert isinstance(
        await doc_status.resolve_doc_source_strict("nobody.md"), SourceAbsent
    )


async def test_source_unique_returns_the_doc_id_to_use(doc_status):
    await _seed(
        doc_status,
        {
            "doc-custom-id": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="report.md"
            )
        },
    )

    resolution = await doc_status.resolve_doc_source_strict("report.md")

    assert isinstance(resolution, SourceUnique)
    # The custom id is found through the source key — a basename lookup that
    # assumed doc ids derive from filenames would miss this row entirely.
    assert resolution.doc_id == "doc-custom-id"


async def test_two_primaries_on_one_key_are_a_conflict(doc_status):
    await _seed(
        doc_status,
        {
            "doc-old": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="dup.md"
            ),
            "doc-new": _row(
                status=DocStatus.PENDING, created_at=_stamp(2), file_path="dup.md"
            ),
        },
    )

    resolution = await doc_status.resolve_doc_source_strict("dup.md")

    assert isinstance(resolution, SourceConflict)
    assert set(resolution.sample_doc_ids) <= {"doc-old", "doc-new"}
    assert len(resolution.sample_doc_ids) >= 2
    if resolution.candidate_count is not None:
        assert resolution.candidate_count == 2


async def test_a_duplicate_marked_row_is_not_a_primary_candidate(doc_status):
    """``is_duplicate`` is what keeps a resolved conflict resolved."""
    await _seed(
        doc_status,
        {
            "doc-keep": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="dup.md"
            ),
            "doc-dup": _row(
                status=DocStatus.PROCESSED,
                created_at=_stamp(2),
                file_path="dup.md",
                metadata={"is_duplicate": True, "original_doc_id": "doc-keep"},
            ),
        },
    )

    resolution = await doc_status.resolve_doc_source_strict("dup.md")

    assert isinstance(resolution, SourceUnique)
    assert resolution.doc_id == "doc-keep"


# ---------------------------------------------------------------------------
# Derived-index consistency across the mutating paths.
# ---------------------------------------------------------------------------


async def test_deleting_a_row_removes_it_from_every_index(doc_status):
    await _seed(
        doc_status,
        {
            "doc-1": _row(
                status=DocStatus.PENDING, created_at=_stamp(1), file_path="gone.md"
            ),
            "doc-2": _row(
                status=DocStatus.PENDING, created_at=_stamp(2), file_path="stays.md"
            ),
        },
    )

    await doc_status.delete(["doc-1"])
    await doc_status.index_done_callback()

    assert (
        await doc_status.count_docs_by_statuses([DocStatus.PENDING], strict=True) == 1
    )
    assert await _drain(doc_status, [DocStatus.PENDING], limit=10) == ["doc-2"]
    assert isinstance(
        await doc_status.resolve_doc_source_strict("gone.md"), SourceAbsent
    )
    assert await doc_status.get_docs_by_ids(["doc-1"], strict=True) == {}


async def test_moving_a_source_key_moves_it_in_the_index(doc_status):
    await _seed(
        doc_status,
        {
            "doc-1": _row(
                status=DocStatus.PENDING, created_at=_stamp(1), file_path="before.md"
            )
        },
    )

    await doc_status.update_doc_status_fields("doc-1", {"file_path": "after.md"})

    assert isinstance(
        await doc_status.resolve_doc_source_strict("before.md"), SourceAbsent
    )
    moved = await doc_status.resolve_doc_source_strict("after.md")
    assert isinstance(moved, SourceUnique) and moved.doc_id == "doc-1"


async def test_upserting_over_a_row_keeps_the_indexes_single_valued(doc_status):
    await _seed(
        doc_status,
        {"doc-1": _row(status=DocStatus.PENDING, created_at=_stamp(1))},
    )
    await _seed(
        doc_status,
        {"doc-1": _row(status=DocStatus.PROCESSED, created_at=_stamp(1))},
    )

    assert (
        await doc_status.count_docs_by_statuses([DocStatus.PENDING], strict=True) == 0
    )
    assert (
        await doc_status.count_docs_by_statuses([DocStatus.PROCESSED], strict=True) == 1
    )
    assert await _drain(doc_status, [DocStatus.PROCESSED], limit=10) == ["doc-1"]


# ---------------------------------------------------------------------------
# Operator conflict listing + CAS repair.
# ---------------------------------------------------------------------------


async def test_conflict_listing_finds_the_conflicting_key(doc_status):
    await _seed(
        doc_status,
        {
            "doc-a": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="dup.md"
            ),
            "doc-b": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(2), file_path="dup.md"
            ),
            "doc-c": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(3), file_path="fine.md"
            ),
        },
    )

    page = await doc_status.list_source_conflicts_page(limit=10, position=CURSOR_START)

    keys = {summary.canonical_source_key for summary in page.conflicts}
    assert keys == {"dup.md"}


async def test_repair_is_dry_run_by_default_then_commits_under_cas(doc_status):
    await _seed(
        doc_status,
        {
            "doc-keep": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="dup.md"
            ),
            "doc-lose": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(2), file_path="dup.md"
            ),
        },
    )

    preview = await doc_status.repair_source_conflict(
        "dup.md",
        primary_doc_id="doc-keep",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )
    assert preview.committed is False
    assert preview.candidate_count == 2
    # The dry run changed nothing.
    assert isinstance(
        await doc_status.resolve_doc_source_strict("dup.md"), SourceConflict
    )

    committed = await doc_status.repair_source_conflict(
        "dup.md",
        primary_doc_id="doc-keep",
        expected_candidate_count=preview.candidate_count,
        expected_candidate_fingerprint=preview.fingerprint,
        dry_run=False,
    )
    assert committed.committed is True

    resolved = await doc_status.resolve_doc_source_strict("dup.md")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-keep"


async def test_a_stale_cas_token_refuses_instead_of_overwriting(doc_status):
    await _seed(
        doc_status,
        {
            "doc-keep": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="dup.md"
            ),
            "doc-lose": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(2), file_path="dup.md"
            ),
        },
    )
    preview = await doc_status.repair_source_conflict(
        "dup.md",
        primary_doc_id="doc-keep",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )

    # A third candidate appears between read and commit.
    await _seed(
        doc_status,
        {
            "doc-third": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(3), file_path="dup.md"
            )
        },
    )

    with pytest.raises(SourceConflictRepairCASError):
        await doc_status.repair_source_conflict(
            "dup.md",
            primary_doc_id="doc-keep",
            expected_candidate_count=preview.candidate_count,
            expected_candidate_fingerprint=preview.fingerprint,
            dry_run=False,
        )


async def test_repairing_onto_a_non_candidate_is_a_value_error(doc_status):
    await _seed(
        doc_status,
        {
            "doc-a": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(1), file_path="dup.md"
            ),
            "doc-b": _row(
                status=DocStatus.PROCESSED, created_at=_stamp(2), file_path="dup.md"
            ),
        },
    )
    preview = await doc_status.repair_source_conflict(
        "dup.md",
        primary_doc_id="doc-a",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )

    with pytest.raises(ValueError):
        await doc_status.repair_source_conflict(
            "dup.md",
            primary_doc_id="doc-not-a-candidate",
            expected_candidate_count=preview.candidate_count,
            expected_candidate_fingerprint=preview.fingerprint,
            dry_run=False,
        )


# ---------------------------------------------------------------------------
# The sweep under concurrent writes (live view, no snapshot claimed).
# ---------------------------------------------------------------------------


async def test_a_sweep_running_against_concurrent_writes_still_terminates(doc_status):
    """The contract claims a live view, not snapshot isolation — but it must
    still terminate and never return a record twice."""
    await _seed(
        doc_status,
        {
            f"doc-{i:02d}": _row(status=DocStatus.PENDING, created_at=_stamp(i))
            for i in range(6)
        },
    )

    seen: list[str] = []
    position = CURSOR_START
    appended = False
    for _ in range(50):
        page = await doc_status.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=2, position=position, strict=True
        )
        seen.extend(page.docs)
        if not appended:
            # A new record with a LATER key: it may or may not be observed.
            await _seed(
                doc_status,
                {"doc-99": _row(status=DocStatus.PENDING, created_at=_stamp(99))},
            )
            appended = True
        if page.next_position is CURSOR_END:
            break
        position = page.next_position
    else:  # pragma: no cover - the loop guard
        pytest.fail("sweep did not terminate under concurrent writes")

    assert len(seen) == len(set(seen))  # no record twice
    assert {f"doc-{i:02d}" for i in range(6)} <= set(seen)  # nothing pre-existing lost


async def test_concurrent_page_reads_do_not_interfere(doc_status):
    await _seed(
        doc_status,
        {
            f"doc-{i:02d}": _row(status=DocStatus.PENDING, created_at=_stamp(i))
            for i in range(4)
        },
    )

    pages = await asyncio.gather(
        *[
            doc_status.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=4, position=CURSOR_START, strict=True
            )
            for _ in range(4)
        ]
    )

    for page in pages:
        assert set(page.docs) == {f"doc-{i:02d}" for i in range(4)}
