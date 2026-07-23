"""PGDocStatusStorage memory-bounding scheduling API (Phase 1) — offline.

Covers the PG slice of the cursor-page contract against a mocked ``self.db``
(same ``__new__`` + fake-db pattern as
tests/kg/test_doc_status_strict_parse_branches.py — no live PostgreSQL):

* cursor encode/decode round-trip + malformed cursor fails closed;
* keyset page SQL shape (ORDER BY, row-value comparison, generation
  predicate only when a cutoff is given) and CURSOR_END/CursorAfter
  termination semantics;
* update_doc_status_fields immutable created_at + missing-row contract;
* mark_doc_failed idempotency / one-transaction reserve+publish / immutable
  created_at / conditional create;
* failure-generation mode mapping (missing row / bad schema_version / value
  parse / DB error propagation);
* basename lookup excludes duplicate-marker rows;
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
    FailureGenerationMode,
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


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConnection:
    """Minimal asyncpg connection for the mark_doc_failed transaction."""

    def __init__(self, status_row=None, next_generation=5):
        self.status_row = status_row
        self.next_generation = next_generation
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []

    def transaction(self):
        return _FakeTransaction()

    async def fetchrow(self, sql, *args):
        self.fetchrow_calls.append((sql, args))
        if "FROM LIGHTRAG_DOC_STATUS" in sql:
            return self.status_row
        if "LIGHTRAG_DOC_SCHEDULING_CTRL" in sql:
            return {"failure_generation_counter": self.next_generation}
        raise AssertionError(f"unexpected fetchrow: {sql}")

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        return "UPDATE 1"

    def counter_updates(self):
        return [
            c for c in self.fetchrow_calls if "LIGHTRAG_DOC_SCHEDULING_CTRL" in c[0]
        ]


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
        "failure_generation": 0,
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


async def test_encode_cursor_null_created_at_fails_closed():
    storage = _storage()
    with pytest.raises(StorageControlPlaneError):
        storage._encode_cursor(_page_row(created_at=None))


# ---------------------------------------------------------------------------
# Page SQL shape + termination
# ---------------------------------------------------------------------------


async def test_page_sql_keyset_shape_without_cursor_or_cutoff():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED], limit=10, strict=True
    )
    assert page.docs == {}
    assert page.next_position is CURSOR_END

    sql, params, multirows = db.calls[0]
    assert multirows is True
    assert "ORDER BY created_at ASC, id ASC" in sql
    assert "status = ANY($2)" in sql
    assert "(created_at, id) >" not in sql
    # Generation predicate must appear ONLY when a cutoff is given.
    assert "COALESCE(failure_generation, 0)" not in sql
    assert params == ["ws", ["pending", "failed"], 10]


async def test_page_sql_with_cursor_and_generation_cutoff():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    opaque = storage._encode_cursor(_page_row())
    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED],
        limit=10,
        position=CursorAfter(opaque),
        max_failure_generation=7,
        strict=True,
    )
    sql, params, _ = db.calls[0]
    assert "(created_at, id) > ($3::timestamp, $4)" in sql
    assert "(status != $5 OR COALESCE(failure_generation, 0) <= $6)" in sql
    assert "ORDER BY created_at ASC, id ASC LIMIT $7" in sql
    assert params[2] == _TS  # decoded back to the naive-UTC stored form
    assert params[3] == "doc-1"
    assert params[4] == DocStatus.FAILED.value
    assert params[5] == 7
    assert params[6] == 10


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
# mark_doc_failed
# ---------------------------------------------------------------------------


async def test_mark_doc_failed_idempotent_same_attempt_keeps_generation():
    conn = _FakeConnection(
        status_row={
            "status": "failed",
            "processing_attempt_id": "a1",
            "failure_attempt_id": "a1",
            "failure_generation": 7,
        }
    )
    storage = _storage(_FakeDBWithConnection(conn))
    generation = await storage.mark_doc_failed(
        "d1", {"processing_attempt_id": "a1", "error_msg": "x"}
    )
    assert generation == 7
    assert conn.counter_updates() == []  # no counter touch
    assert conn.execute_calls == []  # no row rewrite either


async def test_mark_doc_failed_existing_row_reserves_and_updates():
    conn = _FakeConnection(
        status_row={
            "status": "processing",
            "processing_attempt_id": "a1",
            "failure_attempt_id": None,
            "failure_generation": 0,
        },
        next_generation=9,
    )
    storage = _storage(_FakeDBWithConnection(conn))
    generation = await storage.mark_doc_failed(
        "d1", {"error_msg": "x", "created_at": "2020-01-01T00:00:00+00:00"}
    )
    assert generation == 9
    assert len(conn.counter_updates()) == 1
    sql, args = conn.execute_calls[0]
    assert sql.startswith("UPDATE LIGHTRAG_DOC_STATUS SET ")
    # Immutable sort key: caller-supplied created_at is IGNORED.
    assert "created_at" not in sql
    assert "status = $" in sql and "failure_generation = $" in sql
    assert "failure_attempt_id = $" in sql
    assert DocStatus.FAILED.value in args
    assert 9 in args
    assert "a1" in args  # failure_attempt_id = the row's persisted attempt


async def test_mark_doc_failed_missing_row_conditional_create():
    conn = _FakeConnection(status_row=None, next_generation=3)
    storage = _storage(_FakeDBWithConnection(conn))
    generation = await storage.mark_doc_failed(
        "d1",
        {
            "processing_attempt_id": "a9",
            "created_at": "2026-01-01T00:00:00+00:00",
            "error_msg": "parse blew up",
        },
    )
    assert generation == 3
    sql, args = conn.execute_calls[0]
    assert sql.startswith("INSERT INTO LIGHTRAG_DOC_STATUS")
    assert "ON CONFLICT (id, workspace) DO UPDATE SET" in sql
    # created_at IS inserted for a brand-new row ...
    assert "created_at" in sql.split("ON CONFLICT")[0]
    # ... but never overwritten when a concurrent insert won the race.
    assert "created_at = EXCLUDED.created_at" not in sql
    assert "failure_attempt_id" in sql
    assert DocStatus.FAILED.value in args


async def test_mark_doc_failed_rejects_unknown_columns():
    storage = _storage(_ExplodingDB())
    with pytest.raises(ValueError, match="unknown"):
        await storage.mark_doc_failed("d1", {"nope": 1})


# ---------------------------------------------------------------------------
# failure-generation mode + reservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ctrl_row", "expected"),
    [
        (None, FailureGenerationMode.MIGRATING),  # missing row after init
        (
            {"schema_version": 99, "mode": "enforced"},
            FailureGenerationMode.MIGRATING,
        ),  # unknown marker version
        (
            {"schema_version": 1, "mode": "enforced"},
            FailureGenerationMode.ENFORCED,
        ),
        (
            {"schema_version": 1, "mode": "legacy"},
            FailureGenerationMode.LEGACY,
        ),
        (
            {"schema_version": 1, "mode": "totally-bogus"},
            FailureGenerationMode.MIGRATING,
        ),  # unparsable value
    ],
)
async def test_get_failure_generation_mode_mapping(ctrl_row, expected):
    storage = _storage(_FakeDB(results=[ctrl_row]))
    assert await storage.get_failure_generation_mode() is expected


async def test_get_failure_generation_mode_db_error_propagates():
    storage = _storage(_FakeDB(results=[ConnectionError("marker read failed")]))
    with pytest.raises(ConnectionError):
        await storage.get_failure_generation_mode()  # never degrades to LEGACY


async def test_reserve_failure_generation_counter_row_not_sequence():
    db = _FakeDB(results=[{"failure_generation_counter": 42}])
    storage = _storage(db)
    assert await storage.reserve_failure_generation() == 42
    sql, params, _ = db.calls[0]
    assert "UPDATE LIGHTRAG_DOC_SCHEDULING_CTRL" in sql
    assert "RETURNING failure_generation_counter" in sql
    assert "nextval" not in sql  # review ruling: counter row, never a sequence
    assert params == ["ws", 1]


async def test_reserve_failure_generation_missing_marker_fails_closed():
    storage = _storage(_FakeDB(results=[None]))
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()


# ---------------------------------------------------------------------------
# ensure_processing_attempt_id
# ---------------------------------------------------------------------------


async def test_ensure_processing_attempt_id_cas_and_missing():
    db = _FakeDB(results=[{"processing_attempt_id": "persisted"}, None])
    storage = _storage(db)
    assert await storage.ensure_processing_attempt_id("d1") == "persisted"
    sql, params, _ = db.calls[0]
    assert "COALESCE(processing_attempt_id, $3)" in sql
    assert "RETURNING processing_attempt_id" in sql
    assert params[0] == "ws" and params[1] == "d1" and params[2]

    with pytest.raises(StorageRecordNotFoundError):
        await storage.ensure_processing_attempt_id("ghost")


# ---------------------------------------------------------------------------
# basename lookup: primary-row-only + strict variants
# ---------------------------------------------------------------------------


async def test_basename_query_excludes_duplicate_marker_rows():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    assert await storage.get_doc_by_file_basename("report.pdf") is None
    sql, params, _ = db.calls[0]
    assert "COALESCE((metadata->>'is_duplicate')::boolean, false) = false" in sql
    assert "ORDER BY created_at ASC, id ASC LIMIT 1" in sql
    assert params == ["ws", "report.pdf"]


async def test_basename_strict_delegates_and_propagates_errors():
    db = _FakeDB(results=[[]])
    storage = _storage(db)
    assert await storage.get_doc_by_file_basename_strict("report.pdf") is None
    assert "is_duplicate" in db.calls[0][0]

    failing = _storage(_FakeDB(results=[ConnectionError("down")]))
    with pytest.raises(ConnectionError):
        await failing.get_doc_by_file_basename_strict("report.pdf")


# ---------------------------------------------------------------------------
# strict point reads + capability flags
# ---------------------------------------------------------------------------


async def test_doc_status_get_by_id_strict_confirmed_absent_or_raise():
    storage = _storage(_FakeDB(results=[None]))
    assert await storage.get_by_id_strict("ghost") is None

    failing = _storage(_FakeDB(results=[ConnectionError("down")]))
    with pytest.raises(ConnectionError):
        await failing.get_by_id_strict("d1")


def test_capability_flags():
    assert PGDocStatusStorage.supports_bounded_scheduling_pages is True
    assert PGDocStatusStorage.supports_failure_generation is True
    assert PGDocStatusStorage.supports_strict_doc_identity_lookup is True
    assert PGDocStatusStorage.supports_strict_point_reads is True
    assert PGKVStorage.supports_strict_point_reads is True
