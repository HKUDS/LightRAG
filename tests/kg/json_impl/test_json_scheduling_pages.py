"""JsonDocStatusStorage Phase 1 scheduling contract tests.

Covers: stable ``(created_at, id)`` keyset order with id tie-break, the
consumed-position advance (a fully generation-filtered page advances the
cursor instead of terminating or re-reading), CURSOR_END termination,
lightweight projection, strict count, targeted field updates, the
failure-generation write side (reserve-before-publish holes, per-attempt
idempotency, counter calibration on restore), mode-marker semantics
(missing/mismatched marker reads MIGRATING — never LEGACY), and the
primary-row basename contract (duplicate markers invisible).
"""

import json

import pytest

from lightrag.base import (
    CURSOR_END,
    CursorAfter,
    DocStatus,
    FailureGenerationMode,
)
from lightrag.exceptions import (
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import load_json, write_json

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


async def _sweep_ids(storage, statuses, *, limit, max_failure_generation=None):
    """Drive a full sweep, returning ids in consumption order + page count."""
    ids: list[str] = []
    pages = 0
    position = None
    from lightrag.base import CURSOR_START

    position = CURSOR_START
    while True:
        page = await storage.get_docs_by_statuses_page(
            statuses,
            limit=limit,
            position=position,
            max_failure_generation=max_failure_generation,
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
async def test_fully_filtered_page_advances_without_false_end(tmp_path):
    """Rows dropped by the generation predicate are CONSUMED: the cursor
    advances past a fully-filtered page and still reaches the eligible row
    behind it — and never re-reads the filtered rows."""
    rows = {
        f"doc-{i:02d}": _doc(
            "failed",
            created_at=f"2026-01-01T00:00:0{i % 10}+00:00",
            failure_generation=100,
        )
        for i in range(4)
    }
    rows["doc-99"] = _doc(
        "failed", created_at="2026-01-05T00:00:00+00:00", failure_generation=1
    )
    storage = await _storage(tmp_path, rows)
    ids, pages = await _sweep_ids(
        storage, [DocStatus.FAILED], limit=2, max_failure_generation=5
    )
    assert ids == ["doc-99"]
    assert pages >= 3  # filtered pages advanced instead of terminating


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


@pytest.mark.asyncio
async def test_mode_enforced_after_init_and_migrating_on_marker_loss(tmp_path):
    storage = await _storage(tmp_path)
    assert await storage.get_failure_generation_mode() is FailureGenerationMode.ENFORCED
    # A lost/corrupt marker on an initialized workspace reads MIGRATING —
    # never LEGACY (which would reopen the full-snapshot path).
    write_json({"schema_version": 999, "mode": "enforced"}, storage._ctrl_file_name)
    assert (
        await storage.get_failure_generation_mode() is FailureGenerationMode.MIGRATING
    )
    import os

    os.unlink(storage._ctrl_file_name)
    assert (
        await storage.get_failure_generation_mode() is FailureGenerationMode.MIGRATING
    )
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()


@pytest.mark.asyncio
async def test_reserve_holes_never_reused_and_persisted(tmp_path):
    storage = await _storage(tmp_path)
    g1 = await storage.reserve_failure_generation()
    g2 = await storage.reserve_failure_generation()
    assert g2 == g1 + 1
    # A reservation whose FAILED write never happened stays a hole: the next
    # reservation continues past it (and survives the persisted counter).
    ctrl = load_json(storage._ctrl_file_name)
    assert ctrl["failure_generation_counter"] == g2


@pytest.mark.asyncio
async def test_mark_doc_failed_assigns_generation_and_is_attempt_idempotent(
    tmp_path,
):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc(
                "processing",
                created_at="2026-01-01T00:00:00+00:00",
                processing_attempt_id="attempt-1",
            )
        },
    )
    g1 = await storage.mark_doc_failed(
        "doc-1",
        {"error_msg": "boom", "created_at": "2030-12-31T00:00:00+00:00"},
    )
    row = await storage.get_by_id("doc-1")
    assert row["status"] == "failed"
    assert row["failure_generation"] == g1 and g1 >= 1
    assert row["failure_attempt_id"] == "attempt-1"
    # Immutable sort key: the caller-supplied created_at was ignored.
    assert row["created_at"] == "2026-01-01T00:00:00+00:00"

    # Same attempt failing again (idempotent retry of the failure write):
    # the generation must NOT advance.
    g_again = await storage.mark_doc_failed("doc-1", {"error_msg": "boom2"})
    assert g_again == g1
    assert (await storage.get_by_id("doc-1"))["failure_generation"] == g1

    # A NEW attempt failing later gets a fresh, larger generation.
    await storage.update_doc_status_fields(
        "doc-1", {"status": "processing", "processing_attempt_id": "attempt-2"}
    )
    g2 = await storage.mark_doc_failed("doc-1", {"error_msg": "boom3"})
    assert g2 > g1

    # Missing row: conditional create (pre-PENDING enqueue errors).
    g3 = await storage.mark_doc_failed(
        "ghost",
        {
            "error_msg": "early",
            "created_at": "2026-02-01T00:00:00+00:00",
            "processing_attempt_id": "attempt-x",
            "content_summary": "s",
            "content_length": 0,
            "file_path": "g.pdf",
            "updated_at": "2026-02-01T00:00:00+00:00",
        },
    )
    ghost = await storage.get_by_id("ghost")
    assert ghost["status"] == "failed" and ghost["failure_generation"] == g3 > g2


@pytest.mark.asyncio
async def test_counter_recalibrates_on_restore(tmp_path):
    """Restore-from-backup: rows carry generations ahead of the ctrl counter
    → init recalibrates to max(counter, max persisted), keeping reservations
    monotonic."""
    storage = await _storage(tmp_path)
    # upsert flushes the row to disk immediately (doc-status recovery anchor).
    await storage.upsert({"doc-1": _doc("failed", failure_generation=41)})
    # Simulate the restored, stale ctrl file.
    write_json(
        {
            "schema_version": JsonDocStatusStorage._CTRL_SCHEMA_VERSION,
            "mode": "enforced",
            "failure_generation_counter": 3,
        },
        storage._ctrl_file_name,
    )

    finalize_share_data()
    initialize_share_data()
    restored = await _storage(tmp_path)
    assert await restored.reserve_failure_generation() == 42


@pytest.mark.asyncio
async def test_ensure_processing_attempt_id_atomic(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("pending")})
    first = await storage.ensure_processing_attempt_id("doc-1")
    assert first and (await storage.ensure_processing_attempt_id("doc-1")) == first
    with pytest.raises(StorageRecordNotFoundError):
        await storage.ensure_processing_attempt_id("missing")


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
    strict_match = await storage.get_doc_by_file_basename_strict("a.pdf")
    assert strict_match is not None and strict_match[0] == "doc-primary"

    # Only the duplicate marker left → the basename is free again.
    await storage.delete(["doc-primary"])
    assert await storage.get_doc_by_file_basename("a.pdf") is None
    assert await storage.get_doc_by_file_basename_strict("a.pdf") is None


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
