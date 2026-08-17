"""PGDocStatusStorage memory-bounding scheduling API (Phase 1) — offline.

Covers the PG slice of the cursor-page / strict-read contract against a mocked
``self.db`` (same ``__new__`` + fake-db pattern as
tests/kg/test_doc_status_strict_parse_branches.py — no live PostgreSQL):

* cursor encode/decode round-trip + malformed cursor fails closed;
* keyset page SQL shape (ORDER BY, row-value comparison) and
  CURSOR_END/CursorAfter termination semantics;
* count_docs_by_statuses fail-closed;
* update_doc_status_fields immutable created_at + missing-row contract;
* get_docs_by_ids strict batch (present/missing/error/unusable);
* resolve_doc_source_strict Absent/Unique/Conflict (conflict-aware, exact
  count on the indexed predicate);
* list_source_conflicts_page SQL shape + bounded per-key sample + keyset
  termination;
* repair_source_conflict dry-run / CAS mismatch / commit-demotes-losers /
  bad-primary;
* legacy basename lookup excludes duplicate-marker rows;
* strict point reads + capability flags.
"""

import datetime
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
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.kg.postgres_impl import PGDocStatusStorage, PGKVStorage

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeDB:
    """Records query/execute calls; serves queued results (or raises them)."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls: list[tuple[str, list, bool]] = []
        self.execute_calls: list[tuple[str, dict | None]] = []

    async def query(self, sql, params=None, multirows=False, **kwargs):
        self.calls.append((sql, params, multirows))
        if self.results:
            result = self.results.pop(0)
        else:
            result = [] if multirows else None
        if isinstance(result, Exception):
            raise result
        return result

    async def execute(self, sql, data=None, **kwargs):
        self.execute_calls.append((sql, data))


class _ExplodingDB:
    """Any DB touch is a test failure (used to prove pre-DB validation)."""

    async def query(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("db.query must not be called")

    async def execute(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("db.execute must not be called")

    async def _run_with_retry(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("db._run_with_retry must not be called")


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeRepairConnection:
    """asyncpg-connection stub for the repair_source_conflict transaction.

    ``fetch`` returns the queued candidate rows (the FOR UPDATE re-read);
    ``execute`` records the demotion UPDATE so tests can assert its shape.
    """

    def __init__(self, candidate_ids):
        self._candidate_ids = list(candidate_ids)
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []

    def transaction(self):
        return _FakeTransaction()

    async def fetch(self, sql, *args):
        self.fetch_calls.append((sql, args))
        return [{"id": cid} for cid in self._candidate_ids]

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        return "UPDATE"


class _FakeDBWithConnection:
    def __init__(self, connection):
        self.connection = connection

    async def _run_with_retry(self, operation, **kwargs):
        return await operation(self.connection)


def _storage(db=None) -> PGDocStatusStorage:
    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.workspace = "ws"
    storage.namespace = "doc_status"
    storage.db = db if db is not None else _FakeDB()
    return storage


_TS = datetime.datetime(2026, 1, 2, 3, 4, 5, 123456)


def _page_row(doc_id="doc-1", created_at=_TS, **overrides):
    row = {
        "id": doc_id,
        "status": "pending",
        "created_at": created_at,
        "updated_at": created_at,
        "file_path": f"{doc_id}.pdf",
        "track_id": None,
        "metadata": "{}",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Cursor encode/decode
# ---------------------------------------------------------------------------


async def test_cursor_round_trip_preserves_exact_key():
    storage = _storage()
    opaque = storage._encode_cursor(_page_row())
    created, doc_id = storage._decode_cursor(opaque)
    # Naive-UTC datetime as stored in the TIMESTAMP column, microseconds kept.
    assert created == _TS
    assert created.tzinfo is None
    assert doc_id == "doc-1"


async def test_decode_cursor_normalizes_timezone_aware_iso():
    storage = _storage()
    aware = "2026-01-02T04:04:05.123456+01:00"
    created, _ = storage._decode_cursor(json.dumps([aware, "doc-1"]))
    assert created == _TS
    assert created.tzinfo is None


@pytest.mark.parametrize(
    "opaque",
    [
        "not json",
        '["only-one"]',
        '["2026-01-01T00:00:00+00:00", 5]',
        '[123, "doc-1"]',
        '["not-a-date", "doc-1"]',
        '["2026-01-01T00:00:00", "a", "extra"]',
        "{}",
    ],
)
async def test_malformed_cursor_raises_control_plane_error(opaque):
    storage = _storage(_ExplodingDB())
    with pytest.raises(StorageControlPlaneError):
        storage._decode_cursor(opaque)
    # And through the page API: rejected before any DB round-trip.
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=10, position=CursorAfter(opaque), strict=True
        )


async def test_null_created_at_encodes_null_bucket_cursor():
    """NULL created_at rows (corrupt writes) sort FIRST and stay reachable:
    the cursor encodes [null, id] instead of failing closed — a row-value
    comparison would otherwise starve them out of every later page."""
    storage = _storage()
    opaque = storage._encode_cursor(_page_row(doc_id="doc-n", created_at=None))
    assert json.loads(opaque) == [None, "doc-n"]
    assert storage._decode_cursor(opaque) == (None, "doc-n")


async def test_null_bucket_cursor_resumes_through_null_rows():
    """Cursor inside the NULL bucket: remaining NULL rows continue by id,
    then all real-timestamp rows — nothing is silently excluded."""
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING],
        limit=10,
        position=CursorAfter(json.dumps([None, "doc-n"])),
        strict=True,
    )
    sql, params, _ = db.calls[0]
    assert "(created_at IS NULL AND id > $2) OR created_at IS NOT NULL" in sql
    assert "ORDER BY created_at ASC NULLS FIRST, id ASC" in sql
    assert params[1] == "doc-n"


async def test_string_cursor_excludes_consumed_null_bucket():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING],
        limit=10,
        position=CursorAfter(json.dumps(["2026-01-01T00:00:00", "doc-1"])),
        strict=True,
    )
    sql, _, _ = db.calls[0]
    assert "created_at IS NOT NULL AND (created_at, id) >" in sql
    assert "NULLS FIRST" in sql


async def test_null_created_at_row_reachable_across_page_boundary():
    """Fix-proof for the starvation bug: a NULL created_at row that fills a
    page still yields a cursor that reaches the NEXT rows (bucket-aware
    keyset), instead of a NULL-poisoned row-value comparison."""
    null_row = _page_row(doc_id="doc-null", created_at=None)
    real_row = _page_row(doc_id="doc-real")
    db = _FakeDB(results=[[null_row], [real_row], []])
    storage = _storage(db)
    page1 = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, strict=False
    )
    # The corrupt row is consumed (skipped from the projection in relaxed
    # mode) and the cursor advances into the NULL bucket.
    assert page1.docs == {}
    assert isinstance(page1.next_position, CursorAfter)
    page2 = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, position=page1.next_position, strict=False
    )
    assert set(page2.docs) == {"doc-real"}
    sql2, _, _ = db.calls[1]
    assert "created_at IS NULL AND id >" in sql2


# ---------------------------------------------------------------------------
# Page SQL shape + termination
# ---------------------------------------------------------------------------


async def test_page_sql_keyset_shape_without_cursor():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED], limit=10, strict=True
    )
    assert page.docs == {}
    assert page.next_position is CURSOR_END

    sql, params, multirows = db.calls[0]
    assert multirows is True
    assert "ORDER BY created_at ASC NULLS FIRST, id ASC" in sql
    # ONE BRANCH PER STATUS with equality, not `status = ANY(...)`: a ScalarArrayOp
    # cannot produce ordered output, so the single-query form made the planner
    # Seq Scan the table and sort it on every page (see
    # tests/kg/test_scheduling_page_plans.py, which EXPLAINs this for real).
    assert "status = ANY(" not in sql
    assert sql.count("status=$") == 2
    assert " UNION ALL " in sql
    assert "(created_at, id) >" not in sql
    # The removed failure-generation cohort predicate must never reappear.
    assert "failure_generation" not in sql
    # $1 workspace, $2 limit (shared by every branch and the wrapper), then one
    # parameter per status.
    assert params == ["ws", 10, "pending", "failed"]


async def test_page_sql_with_cursor():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    opaque = storage._encode_cursor(_page_row())
    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED],
        limit=10,
        position=CursorAfter(opaque),
        strict=True,
    )
    sql, params, _ = db.calls[0]
    # The keyset predicate is built once and repeated verbatim in every branch,
    # so the cursor parameters come before the per-status ones.
    assert "(created_at, id) > ($2::timestamp, $3)" in sql
    assert sql.count("(created_at, id) > ($2::timestamp, $3)") == 2
    assert "ORDER BY created_at ASC NULLS FIRST, id ASC LIMIT $4" in sql
    assert "failure_generation" not in sql
    assert params[1] == _TS  # decoded back to the naive-UTC stored form
    assert params[2] == "doc-1"
    assert params[3] == 10  # the shared limit
    assert params[4:] == ["pending", "failed"]


async def test_page_full_returns_cursor_after_last_returned_row():
    rows = [_page_row("doc-1"), _page_row("doc-2", created_at=_TS.replace(hour=9))]
    db = _FakeDB(results=[rows])
    storage = _storage(db)
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, strict=True
    )
    assert set(page.docs) == {"doc-1", "doc-2"}
    assert isinstance(page.next_position, CursorAfter)
    created, doc_id = storage._decode_cursor(page.next_position.opaque)
    assert (created, doc_id) == (_TS.replace(hour=9), "doc-2")
    # Projection sanity: lightweight record, ISO timestamps with tz info.
    record = page.docs["doc-1"]
    assert record.status is DocStatus.PENDING
    assert record.created_at == "2026-01-02T03:04:05.123456+00:00"
    assert record.file_path == "doc-1.pdf"
    assert record.has_custom_chunk_journal is False


async def test_page_short_read_terminates_with_cursor_end():
    db = _FakeDB(results=[[_page_row("doc-1")]])
    storage = _storage(db)
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, strict=True
    )
    assert set(page.docs) == {"doc-1"}
    assert page.next_position is CURSOR_END


async def test_page_relaxed_skips_unusable_row_but_row_stays_consumed():
    bad = _page_row("doc-bad", created_at=None)
    good = _page_row("doc-good", created_at=_TS.replace(hour=9))
    db = _FakeDB(results=[[bad, good]])
    storage = _storage(db)
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, strict=False
    )
    assert set(page.docs) == {"doc-good"}
    # Cursor still advances past the whole SQL-returned frontier.
    assert isinstance(page.next_position, CursorAfter)
    _, doc_id = storage._decode_cursor(page.next_position.opaque)
    assert doc_id == "doc-good"


async def test_page_strict_raises_on_unusable_row_without_cursor():
    bad = _page_row("doc-bad", created_at=None)
    db = _FakeDB(results=[[bad, _page_row("doc-good")]])
    storage = _storage(db)
    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=2, strict=True
        )


async def test_page_argument_validation_before_db():
    storage = _storage(_ExplodingDB())
    with pytest.raises(ValueError):
        await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=0)
    empty = await storage.get_docs_by_statuses_page([], limit=5)
    assert empty.docs == {} and empty.next_position is CURSOR_END
    ended = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=5, position=CURSOR_END
    )
    assert ended.docs == {} and ended.next_position is CURSOR_END


async def test_page_db_error_propagates():
    db = _FakeDB(results=[RuntimeError("boom")])
    storage = _storage(db)
    with pytest.raises(RuntimeError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=5, position=CURSOR_START, strict=True
        )


# ---------------------------------------------------------------------------
# count_docs_by_statuses
# ---------------------------------------------------------------------------


async def test_count_docs_by_statuses_returns_int_and_fails_closed():
    db = _FakeDB(results=[{"count": 3}])
    storage = _storage(db)
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 3
    sql, params, _ = db.calls[0]
    assert "COUNT(*)" in sql and "status = ANY($2)" in sql
    assert params == ["ws", ["pending"]]

    assert await storage.count_docs_by_statuses([]) == 0

    # queue exhausted → fake returns None → fail-closed control-plane error
    with pytest.raises(StorageControlPlaneError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])


async def test_count_docs_by_statuses_db_error_propagates():
    db = _FakeDB(results=[ConnectionError("down")])
    storage = _storage(db)
    with pytest.raises(ConnectionError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])


# ---------------------------------------------------------------------------
# update_doc_status_fields
# ---------------------------------------------------------------------------


async def test_update_fields_rejects_created_at_and_unknown_columns():
    storage = _storage(_ExplodingDB())
    with pytest.raises(ValueError, match="created_at"):
        await storage.update_doc_status_fields("d1", {"created_at": "2026"})
    with pytest.raises(ValueError, match="unknown"):
        await storage.update_doc_status_fields("d1", {"evil; DROP": "x"})


async def test_update_fields_sql_shape_and_serialization():
    db = _FakeDB(results=[{"id": "d1"}])
    storage = _storage(db)
    await storage.update_doc_status_fields(
        "d1", {"status": "processing", "metadata": {"k": 1}, "chunks_list": ["c1"]}
    )
    sql, params, _ = db.calls[0]
    assert sql.startswith("UPDATE LIGHTRAG_DOC_STATUS SET ")
    assert "WHERE workspace=$1 AND id=$2 RETURNING id" in sql
    assert params[0] == "ws" and params[1] == "d1"
    assert params[2] == "processing"
    assert params[3] == json.dumps({"k": 1})  # JSONB serialized like upsert
    assert params[4] == json.dumps(["c1"])


async def test_update_fields_missing_row_contract():
    storage = _storage(_FakeDB(results=[None, None]))
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("ghost", {"status": "failed"})
    # missing_ok suppresses only the not-found outcome
    await storage.update_doc_status_fields(
        "ghost", {"status": "failed"}, missing_ok=True
    )


# ---------------------------------------------------------------------------
# get_docs_by_ids (strict batch read)
# ---------------------------------------------------------------------------


async def test_get_docs_by_ids_present_and_missing():
    rows = [_page_row("doc-1"), _page_row("doc-2", status="failed")]
    db = _FakeDB(results=[rows])
    storage = _storage(db)
    result = await storage.get_docs_by_ids(["doc-1", "doc-2", "ghost"], strict=True)
    assert set(result) == {"doc-1", "doc-2"}  # ghost omitted (confirmed absent)
    assert result["doc-2"].status is DocStatus.FAILED
    assert not hasattr(result["doc-1"], "chunks_list")  # lightweight projection
    sql, params, multirows = db.calls[0]
    assert "id = ANY($2)" in sql
    assert multirows is True
    assert params == ["ws", ["doc-1", "doc-2", "ghost"]]


async def test_get_docs_by_ids_empty_short_circuits():
    storage = _storage(_ExplodingDB())
    assert await storage.get_docs_by_ids([]) == {}


async def test_get_docs_by_ids_strict_db_error_propagates():
    db = _FakeDB(results=[ConnectionError("down")])
    storage = _storage(db)
    with pytest.raises(ConnectionError):
        await storage.get_docs_by_ids(["doc-1"], strict=True)


async def test_get_docs_by_ids_strict_raises_on_unusable_row():
    bad = _page_row("doc-bad", created_at=None)
    db = _FakeDB(results=[[bad]])
    storage = _storage(db)
    with pytest.raises(TypeError):
        await storage.get_docs_by_ids(["doc-bad"], strict=True)


# ---------------------------------------------------------------------------
# get_full_docs_by_ids (strict batch hydration to FULL DocProcessingStatus)
# ---------------------------------------------------------------------------


def _full_row(doc_id="d1", **overrides):
    """A FULL doc_status row (superset of the scheduling projection)."""
    row = _page_row(doc_id)
    row.update(
        {
            "content_summary": f"summary of {doc_id}",
            "content_length": 4242,
            "chunks_count": 3,
            "chunks_list": json.dumps([f"{doc_id}-c1", f"{doc_id}-c2"]),
            "content_hash": "hash-of-" + doc_id,
            "error_msg": None,
        }
    )
    row.update(overrides)
    return row


async def test_get_full_docs_by_ids_present_and_missing():
    rows = [_full_row("d1"), _full_row("d2", status="failed")]
    db = _FakeDB(results=[rows])
    storage = _storage(db)
    result = await storage.get_full_docs_by_ids(["d1", "d2", "ghost"], strict=True)
    assert set(result) == {"d1", "d2"}  # ghost omitted (confirmed absent)
    # status is the raw enum-str value, so compare by == (not is)
    assert result["d2"].status == DocStatus.FAILED
    # FULL projection: fields absent from the lightweight scheduling record
    assert result["d1"].content_summary == "summary of d1"
    assert result["d1"].content_length == 4242
    assert result["d1"].chunks_list == ["d1-c1", "d1-c2"]
    sql, params, multirows = db.calls[0]
    assert "SELECT *" in sql
    assert "id = ANY($2)" in sql
    assert multirows is True
    assert params == ["ws", ["d1", "d2", "ghost"]]


async def test_get_full_docs_by_ids_empty_short_circuits():
    storage = _storage(_ExplodingDB())
    assert await storage.get_full_docs_by_ids([]) == {}


async def test_get_full_docs_by_ids_strict_db_error_propagates():
    db = _FakeDB(results=[ConnectionError("down")])
    storage = _storage(db)
    with pytest.raises(ConnectionError):
        await storage.get_full_docs_by_ids(["d1"], strict=True)


async def test_get_full_docs_by_ids_relaxed_skips_malformed_row():
    bad = _full_row("d-bad")
    del bad["content_summary"]  # required field -> KeyError in the row projection
    db = _FakeDB(results=[[_full_row("d1"), bad]])
    storage = _storage(db)
    result = await storage.get_full_docs_by_ids(["d1", "d-bad"], strict=False)
    assert set(result) == {"d1"}  # malformed row skipped, good one kept


async def test_get_full_docs_by_ids_strict_raises_on_malformed_row():
    bad = _full_row("d-bad")
    del bad["content_summary"]
    db = _FakeDB(results=[[bad]])
    storage = _storage(db)
    with pytest.raises(KeyError):
        await storage.get_full_docs_by_ids(["d-bad"], strict=True)


# ---------------------------------------------------------------------------
# resolve_doc_source_strict (conflict-aware source resolution)
# ---------------------------------------------------------------------------


async def test_resolve_doc_source_short_circuits_sentinel_keys():
    storage = _storage(_ExplodingDB())
    assert isinstance(await storage.resolve_doc_source_strict(""), SourceAbsent)
    assert isinstance(
        await storage.resolve_doc_source_strict("unknown_source"), SourceAbsent
    )


async def test_resolve_doc_source_absent():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    assert isinstance(
        await storage.resolve_doc_source_strict("missing.pdf"), SourceAbsent
    )
    sql, params, multirows = db.calls[0]
    assert "COALESCE((metadata->>'is_duplicate')::boolean, false) = false" in sql
    assert "LIMIT 2" in sql
    assert multirows is True
    assert params == ["ws", "missing.pdf"]


async def test_resolve_doc_source_unique():
    db = _FakeDB(results=[[_page_row("doc-primary", file_path="a.pdf")]])
    storage = _storage(db)
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique)
    assert resolved.doc_id == "doc-primary"
    assert resolved.doc.file_path == "a.pdf"


async def test_resolve_doc_source_conflict_uses_exact_count():
    rows = [
        _page_row("doc-1", file_path="a.pdf"),
        _page_row("doc-2", file_path="a.pdf"),
    ]
    db = _FakeDB(results=[rows, {"c": 5}])
    storage = _storage(db)
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceConflict)
    assert resolved.candidate_count == 5  # exact COUNT(*), not the two-row sample
    assert set(resolved.sample_doc_ids) == {"doc-1", "doc-2"}
    # second query is the exact COUNT(*) on the same primary predicate
    count_sql, count_params, _ = db.calls[1]
    assert "COUNT(*)" in count_sql
    assert "is_duplicate" in count_sql
    assert count_params == ["ws", "a.pdf"]


async def test_resolve_doc_source_db_error_propagates():
    db = _FakeDB(results=[ConnectionError("down")])
    storage = _storage(db)
    with pytest.raises(ConnectionError):
        await storage.resolve_doc_source_strict("a.pdf")  # never degrades to Absent


# ---------------------------------------------------------------------------
# list_source_conflicts_page
# ---------------------------------------------------------------------------


async def test_list_source_conflicts_page_sql_and_projection():
    group_rows = [{"file_path": "a.pdf", "c": 3}]
    sample_rows = [{"id": "doc-1"}, {"id": "doc-2"}, {"id": "doc-3"}]
    db = _FakeDB(results=[group_rows, sample_rows])
    storage = _storage(db)
    page = await storage.list_source_conflicts_page(limit=10)
    assert len(page.conflicts) == 1
    conflict = page.conflicts[0]
    assert conflict.canonical_source_key == "a.pdf"
    assert conflict.candidate_count == 3
    assert set(conflict.sample_doc_ids) == {"doc-1", "doc-2", "doc-3"}
    assert page.next_position is CURSOR_END  # short read (< limit)

    group_sql, group_params, _ = db.calls[0]
    assert "GROUP BY file_path HAVING COUNT(*) >= 2" in group_sql
    assert "COALESCE((metadata->>'is_duplicate')::boolean, false) = false" in group_sql
    assert "NOT IN ('', 'unknown_source', 'no-file-path')" in group_sql
    assert group_params == ["ws", 10]
    sample_sql, sample_params, _ = db.calls[1]
    assert "ORDER BY id ASC LIMIT $3" in sample_sql
    assert sample_params == ["ws", "a.pdf", PGDocStatusStorage._CONFLICT_SAMPLE_CAP]


async def test_list_source_conflicts_page_full_returns_cursor():
    group_rows = [{"file_path": "a.pdf", "c": 2}]
    sample_rows = [{"id": "doc-1"}, {"id": "doc-2"}]
    db = _FakeDB(results=[group_rows, sample_rows])
    storage = _storage(db)
    page = await storage.list_source_conflicts_page(limit=1)  # full page
    assert isinstance(page.next_position, CursorAfter)
    assert storage._decode_conflict_cursor(page.next_position.opaque) == "a.pdf"


async def test_list_source_conflicts_page_cursor_predicate():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    await storage.list_source_conflicts_page(
        limit=10, position=CursorAfter(json.dumps("a.pdf"))
    )
    sql, params, _ = db.calls[0]
    assert "file_path > $2" in sql
    assert params == ["ws", "a.pdf", 10]


async def test_list_source_conflicts_page_argument_validation():
    storage = _storage(_ExplodingDB())
    with pytest.raises(ValueError):
        await storage.list_source_conflicts_page(limit=0)
    ended = await storage.list_source_conflicts_page(limit=5, position=CURSOR_END)
    assert ended.conflicts == () and ended.next_position is CURSOR_END


async def test_malformed_conflict_cursor_raises():
    storage = _storage(_ExplodingDB())
    with pytest.raises(StorageControlPlaneError):
        storage._decode_conflict_cursor("not json")
    # Non-string decoded value is also rejected, before any DB round-trip.
    with pytest.raises(StorageControlPlaneError):
        await storage.list_source_conflicts_page(limit=10, position=CursorAfter("123"))


# ---------------------------------------------------------------------------
# repair_source_conflict (explicit CAS repair)
# ---------------------------------------------------------------------------


async def test_repair_source_conflict_dry_run_reports_without_mutation():
    conn = _FakeRepairConnection(["doc-1", "doc-2", "doc-3"])
    storage = _storage(_FakeDBWithConnection(conn))
    result = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,  # ignored in dry-run
        expected_candidate_fingerprint="ignored",
    )
    assert result.committed is False
    assert result.candidate_count == 3
    assert set(result.demoted_sample_doc_ids) == {"doc-1", "doc-3"}
    # Candidate set re-read FOR UPDATE, no mutation issued.
    assert conn.fetch_calls and "FOR UPDATE" in conn.fetch_calls[0][0]
    assert conn.execute_calls == []


async def test_repair_source_conflict_cas_mismatch_refuses():
    conn = _FakeRepairConnection(["doc-1", "doc-2", "doc-3"])
    storage = _storage(_FakeDBWithConnection(conn))
    with pytest.raises(StorageControlPlaneError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=99,  # stale
            expected_candidate_fingerprint="whatever",
            dry_run=False,
        )
    assert conn.execute_calls == []  # CAS fails before any mutation


async def test_repair_source_conflict_commit_demotes_losers():
    # dry-run first to obtain the matching CAS fingerprint.
    dry_conn = _FakeRepairConnection(["doc-1", "doc-2", "doc-3"])
    dry = await _storage(_FakeDBWithConnection(dry_conn)).repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="ignored",
    )

    conn = _FakeRepairConnection(["doc-1", "doc-2", "doc-3"])
    storage = _storage(_FakeDBWithConnection(conn))
    result = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=dry.candidate_count,
        expected_candidate_fingerprint=dry.fingerprint,
        dry_run=False,
    )
    assert result.committed is True
    assert result.candidate_count == 3
    # One demotion UPDATE marking the losers is_duplicate + original_doc_id.
    assert len(conn.execute_calls) == 1
    sql, args = conn.execute_calls[0]
    assert "is_duplicate" in sql and "original_doc_id" in sql
    assert args[0] == "ws"
    assert set(args[1]) == {"doc-1", "doc-3"}  # all-but-primary demoted
    assert args[2] == "doc-2"  # original_doc_id = chosen primary


async def test_repair_source_conflict_primary_not_in_candidates_raises():
    conn = _FakeRepairConnection(["doc-1", "doc-2"])
    storage = _storage(_FakeDBWithConnection(conn))
    with pytest.raises(ValueError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="ghost",
            expected_candidate_count=2,
            expected_candidate_fingerprint="x",
        )


# ---------------------------------------------------------------------------
# legacy basename lookup: primary-row-only
# ---------------------------------------------------------------------------


async def test_basename_query_excludes_duplicate_marker_rows():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    assert await storage.get_doc_by_file_basename("report.pdf") is None
    sql, params, _ = db.calls[0]
    assert "COALESCE((metadata->>'is_duplicate')::boolean, false) = false" in sql
    assert "ORDER BY created_at ASC, id ASC LIMIT 1" in sql
    assert params == ["ws", "report.pdf"]


# ---------------------------------------------------------------------------
# strict point reads + capability flags
# ---------------------------------------------------------------------------


async def test_doc_status_get_by_id_strict_confirmed_absent_or_raise():
    storage = _storage(_FakeDB(results=[None]))
    assert await storage.get_by_id_strict("ghost") is None

    failing = _storage(_FakeDB(results=[ConnectionError("down")]))
    with pytest.raises(ConnectionError):
        await failing.get_by_id_strict("d1")


def test_scheduling_api_is_mandatory_no_flags():
    # The doc_status scheduling flags were removed: the paging/batch/resolver
    # API is @abstractmethod, so an instantiable backend implements it by
    # definition. get_by_id_strict stays an optional KV capability flag.
    for name in (
        "supports_bounded_scheduling_pages",
        "supports_strict_doc_status_batch_reads",
        "supports_strict_doc_source_resolution",
        "supports_failure_generation",
    ):
        assert not hasattr(PGDocStatusStorage, name)
        assert not hasattr(PGKVStorage, name)
    # strict point reads remain a declared capability.
    assert PGDocStatusStorage.supports_strict_point_reads is True
    assert PGKVStorage.supports_strict_point_reads is True
    # Fully concrete → no abstract methods left → instantiable.
    assert not PGDocStatusStorage.__abstractmethods__
    assert not PGKVStorage.__abstractmethods__
