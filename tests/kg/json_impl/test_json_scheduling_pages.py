"""JsonDocStatusStorage Phase 1 scheduling contract tests (LR2 redesign).

Covers: stable ``(created_at, id)`` keyset order with id tie-break, a
fully-filtered page advancing the cursor without a false End, CURSOR_END
termination, lightweight projection, strict count, targeted field updates
(created_at immutable), strict batch reads, typed source resolution
(Absent/Unique/Conflict), the explicit CAS source-conflict repair, and the
primary-row basename contract (duplicate markers invisible). No
failure-generation machinery remains.
"""

import json

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
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


def _doc(
    status: str,
    file_path: str = "a.pdf",
    created_at: str = "2026-01-01T00:00:00+00:00",
    **extra,
) -> dict:
    row = {
        "content_summary": "s",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": {},
        "error_msg": None,
        "chunks_list": [],
    }
    row.update(extra)
    return row


async def _storage(tmp_path, rows: dict | None = None) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()
    if rows:
        async with storage._storage_lock:
            storage._data.update(rows)
    return storage


async def _sweep_ids(storage, statuses, *, limit):
    """Drive a full sweep, returning ids in consumption order + page count."""
    ids: list[str] = []
    pages = 0
    position = CURSOR_START
    while True:
        page = await storage.get_docs_by_statuses_page(
            statuses,
            limit=limit,
            position=position,
            strict=True,
        )
        pages += 1
        ids.extend(page.docs.keys())
        if page.next_position is CURSOR_END:
            return ids, pages
        position = page.next_position
        assert isinstance(position, CursorAfter)
        assert pages < 100, "sweep failed to terminate"


@pytest.mark.asyncio
async def test_page_order_created_at_then_id_tiebreak(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-b": _doc("pending", created_at="2026-01-02T00:00:00+00:00"),
            "doc-a": _doc("pending", created_at="2026-01-02T00:00:00+00:00"),
            "doc-c": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
        },
    )
    ids, pages = await _sweep_ids(storage, [DocStatus.PENDING], limit=1)
    # Oldest created_at first; same-timestamp rows tie-break by id ASC.
    assert ids == ["doc-c", "doc-a", "doc-b"]
    assert pages >= 3


@pytest.mark.asyncio
async def test_missing_created_at_sorts_first_and_stays_reachable(tmp_path):
    """A missing/empty created_at row sorts into the "" bucket first and must
    stay reachable across a limit=1 sweep (never stranded behind a cursor)."""
    rows = {
        "doc-real": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
        "doc-null": _doc("pending", created_at=""),
    }
    storage = await _storage(tmp_path, rows)
    ids, _ = await _sweep_ids(storage, [DocStatus.PENDING], limit=1)
    assert ids == ["doc-null", "doc-real"]


@pytest.mark.asyncio
async def test_multi_status_kway_merge_no_gaps_or_repeats(tmp_path):
    rows = {
        f"doc-{i:02d}": _doc(
            "pending" if i % 2 == 0 else "failed",
            created_at=f"2026-01-{(i % 9) + 1:02d}T00:00:00+00:00",
        )
        for i in range(9)
    }
    storage = await _storage(tmp_path, rows)
    ids, _ = await _sweep_ids(storage, [DocStatus.PENDING, DocStatus.FAILED], limit=2)
    assert sorted(ids) == sorted(rows)  # no gaps
    assert len(ids) == len(set(ids))  # no repeats


@pytest.mark.asyncio
async def test_page_projection_is_lightweight_and_mixed_status(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc(
                "pending", chunks_list=["c"] * 500, created_at="2026-01-01T00:00:00"
            ),
            "doc-2": _doc("processing", created_at="2026-01-02T00:00:00"),
            "doc-3": _doc("processed", created_at="2026-01-03T00:00:00"),
        },
    )
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.PROCESSING], limit=10, strict=True
    )
    assert set(page.docs) == {"doc-1", "doc-2"}
    assert page.next_position is CURSOR_END
    record = page.docs["doc-1"]
    assert not hasattr(record, "chunks_list")
    assert record.status is DocStatus.PENDING
    assert record.has_custom_chunk_journal is False


@pytest.mark.asyncio
async def test_count_docs_by_statuses_strict(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc("pending"),
            "doc-2": _doc("pending"),
            "doc-3": _doc("failed"),
        },
    )
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 2
    assert (
        await storage.count_docs_by_statuses([DocStatus.PENDING, DocStatus.FAILED]) == 3
    )


@pytest.mark.asyncio
async def test_update_doc_status_fields_targeted(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("failed")})
    await storage.update_doc_status_fields("doc-1", {"status": "pending"})
    row = await storage.get_by_id("doc-1")
    assert row["status"] == "pending"
    assert row["file_path"] == "a.pdf"  # untouched fields preserved
    with pytest.raises(ValueError, match="created_at"):
        await storage.update_doc_status_fields("doc-1", {"created_at": "2030-01-01"})
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("missing", {"status": "pending"})
    # missing_ok swallows the unknown id.
    await storage.update_doc_status_fields(
        "missing", {"status": "pending"}, missing_ok=True
    )


@pytest.mark.asyncio
async def test_get_docs_by_ids_batch_present_and_missing(tmp_path):
    storage = await _storage(
        tmp_path,
        {"doc-1": _doc("pending"), "doc-2": _doc("failed")},
    )
    result = await storage.get_docs_by_ids(["doc-1", "doc-2", "ghost"], strict=True)
    assert set(result) == {"doc-1", "doc-2"}  # missing omitted, confirmed absent
    assert result["doc-2"].status is DocStatus.FAILED
    assert not hasattr(result["doc-1"], "chunks_list")


@pytest.mark.asyncio
async def test_get_full_docs_by_ids_hydrates_full_status(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc(
                "pending",
                content_summary="hello",
                content_length=42,
                chunks_list=["c1", "c2"],
                track_id="t9",
            ),
            "doc-2": _doc("failed"),
        },
    )
    result = await storage.get_full_docs_by_ids(
        ["doc-1", "doc-2", "ghost"], strict=True
    )
    assert set(result) == {"doc-1", "doc-2"}  # missing omitted, confirmed absent
    # FULL projection (unlike get_docs_by_ids): every heavy field is present.
    doc1 = result["doc-1"]
    assert doc1.content_summary == "hello"
    assert doc1.content_length == 42
    assert doc1.chunks_list == ["c1", "c2"]
    assert doc1.track_id == "t9"
    # Full status mirrors get_docs_by_statuses: status stays the raw str value
    # (a str-enum member equals its value), NOT converted to the DocStatus obj.
    assert result["doc-2"].status == DocStatus.FAILED


@pytest.mark.asyncio
async def test_get_full_docs_by_ids_strict_raises_on_malformed(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("pending")})
    async with storage._storage_lock:
        storage._data["bad"] = {"status": "pending"}  # missing required fields
    with pytest.raises((KeyError, TypeError)):
        await storage.get_full_docs_by_ids(["doc-1", "bad"], strict=True)
    # relaxed: the malformed row is skipped, the good one still hydrates.
    relaxed = await storage.get_full_docs_by_ids(["doc-1", "bad"], strict=False)
    assert set(relaxed) == {"doc-1"}


@pytest.mark.asyncio
async def test_resolve_doc_source_absent_and_unique(tmp_path):
    storage = await _storage(
        tmp_path, {"doc-primary": _doc("pending", file_path="a.pdf")}
    )
    assert isinstance(
        await storage.resolve_doc_source_strict("missing.pdf"), SourceAbsent
    )
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique)
    assert resolved.doc_id == "doc-primary"
    assert resolved.doc.file_path == "a.pdf"


@pytest.mark.asyncio
async def test_resolve_doc_source_ignores_duplicate_markers(tmp_path):
    dup = _doc("failed", file_path="a.pdf")
    dup["metadata"] = {"is_duplicate": True}
    storage = await _storage(
        tmp_path,
        {"dup-1": dup, "doc-primary": _doc("pending", file_path="a.pdf")},
    )
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-primary"


@pytest.mark.asyncio
async def test_resolve_doc_source_conflict(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc("pending", file_path="a.pdf"),
            "doc-2": _doc("failed", file_path="a.pdf"),
        },
    )
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceConflict)
    assert set(resolved.sample_doc_ids) == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_list_and_repair_source_conflict(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc("pending", file_path="a.pdf"),
            "doc-2": _doc("failed", file_path="a.pdf"),
            "doc-3": _doc("pending", file_path="a.pdf"),
            "solo": _doc("pending", file_path="b.pdf"),
        },
    )
    page = await storage.list_source_conflicts_page(limit=10)
    assert len(page.conflicts) == 1
    conflict = page.conflicts[0]
    assert conflict.canonical_source_key == "a.pdf"
    assert conflict.candidate_count == 3

    # dry-run reports without mutating.
    dry = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="ignored-in-dry-run",
    )
    assert dry.committed is False
    assert dry.candidate_count == 3
    still = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(still, SourceConflict)

    # stale expectation → CAS failure.
    with pytest.raises(StorageControlPlaneError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=99,
            expected_candidate_fingerprint=dry.fingerprint,
            dry_run=False,
        )

    # correct expectation → commit; resolver returns the chosen primary.
    result = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=dry.candidate_count,
        expected_candidate_fingerprint=dry.fingerprint,
        dry_run=False,
    )
    assert result.committed is True
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
    # losers demoted to duplicates, content intact.
    for loser in ("doc-1", "doc-3"):
        row = await storage.get_by_id(loser)
        assert row["metadata"]["is_duplicate"] is True
        assert row["metadata"]["original_doc_id"] == "doc-2"


@pytest.mark.asyncio
async def test_repair_source_conflict_repeats_converge(tmp_path):
    """Repeat safety (LR2 §5.5): a committed repair leaves ONE primary, so
    replaying the same request fails CAS (its token described the old set) while
    a fresh dry-run → commit converges to a no-op. This is also the resume path
    after a repair dies mid-way — no repair marker is involved."""
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc("pending", file_path="a.pdf"),
            "doc-2": _doc("failed", file_path="a.pdf"),
        },
    )
    first = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
    )
    committed = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=first.candidate_count,
        expected_candidate_fingerprint=first.fingerprint,
        dry_run=False,
    )
    assert committed.committed is True

    # Replaying the very same commit: the candidate set is now {doc-2}, so the
    # echoed token is stale and the repair refuses instead of re-demoting.
    with pytest.raises(SourceConflictRepairCASError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=first.candidate_count,
            expected_candidate_fingerprint=first.fingerprint,
            dry_run=False,
        )

    # A fresh dry-run → commit is a no-op that still succeeds (idempotent
    # convergence): one candidate, nothing left to demote.
    again = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
    )
    assert again.candidate_count == 1
    assert again.demoted_sample_doc_ids == ()
    final = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=again.candidate_count,
        expected_candidate_fingerprint=again.fingerprint,
        dry_run=False,
    )
    assert final.committed is True
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
    # The loser demoted by the first commit is untouched by the replay.
    row = await storage.get_by_id("doc-1")
    assert row["metadata"]["original_doc_id"] == "doc-2"


@pytest.mark.asyncio
async def test_basename_lookup_returns_primary_only(tmp_path):
    dup = _doc("failed", file_path="a.pdf")
    dup["metadata"] = {"is_duplicate": True, "duplicate_kind": "filename"}
    storage = await _storage(
        tmp_path,
        {
            "dup-1": dup,
            "doc-primary": _doc("pending", file_path="a.pdf"),
        },
    )
    match = await storage.get_doc_by_file_basename("a.pdf")
    assert match is not None and match[0] == "doc-primary"
    # Only the duplicate marker left → the basename is free again.
    await storage.delete(["doc-primary"])
    assert await storage.get_doc_by_file_basename("a.pdf") is None


@pytest.mark.asyncio
async def test_get_by_id_strict_confirmed_absence(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("pending")})
    assert (await storage.get_by_id_strict("doc-1"))["status"] == "pending"
    assert await storage.get_by_id_strict("missing") is None


@pytest.mark.asyncio
async def test_malformed_cursor_raises_control_plane_error(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("pending")})
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING],
            limit=1,
            position=CursorAfter(json.dumps({"not": "a-pair"})),
        )


@pytest.mark.asyncio
async def test_content_hash_lookup_returns_the_earliest_holder(tmp_path):
    """Dict insertion order is not creation order once rows are reloaded or
    rewritten, so returning the first match made the original_doc_id recorded on
    a duplicate depend on file layout. The base contract asks for the EARLIEST by
    (created_at, id)."""
    storage = await _storage(
        tmp_path,
        {
            # Inserted late-first AND with ids sorting opposite to created_at, so
            # neither dict order nor id order accidentally yields the right answer.
            "doc-a": _doc(
                "processed",
                file_path="late.pdf",
                created_at="2026-05-05T00:00:00+00:00",
                content_hash="dup",
            ),
            "doc-z": _doc(
                "processed",
                file_path="early.pdf",
                created_at="2024-01-01T00:00:00+00:00",
                content_hash="dup",
            ),
        },
    )

    result = await storage.get_doc_by_content_hash("dup")
    assert result is not None and result[0] == "doc-z"

    # Excluding the earliest falls through to the next one, still deterministically.
    result = await storage.get_doc_by_content_hash("dup", exclude_doc_id="doc-z")
    assert result is not None and result[0] == "doc-a"
