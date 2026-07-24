"""Offline tests for the Mongo Phase 1 scheduling surface.

Covers the bounded keyset page API (query/sort/limit shape, generation
predicate, consumed-position cursor advance, malformed-cursor rejection),
the failure-generation control plane (mode mapping, atomic reservation,
mark_doc_failed idempotency/CAS shape) and the strict basename/point-read
alignment. The pymongo collection is driven through minimal in-process
fakes — no live services.
"""

import json

import pytest

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from pymongo.errors import PyMongoError

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    FailureGenerationMode,
)
from lightrag.exceptions import StorageControlPlaneError, StorageRecordNotFoundError
from lightrag.kg.mongo_impl import MongoDocStatusStorage

pytestmark = pytest.mark.offline

_ROW_1 = {
    "_id": "doc-1",
    "status": "pending",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
    "file_path": "a.txt",
}
_ROW_2 = {
    "_id": "doc-2",
    "status": "failed",
    "created_at": "2026-01-02T00:00:00+00:00",
    "updated_at": "2026-01-02T00:00:00+00:00",
    "file_path": "b.txt",
}


class _UpdateResult:
    def __init__(self, matched_count=0, modified_count=0, upserted_id=None):
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.upserted_id = upserted_id


class _FakeFindCursor:
    """Records the find(...).sort(...).limit(...).to_list(...) chain."""

    def __init__(self, docs, error=None):
        self._docs = list(docs)
        self._error = error
        self.sort_spec = None
        self.limit_value = None
        self.to_list_length = None

    def sort(self, spec):
        self.sort_spec = spec
        return self

    def limit(self, n):
        self.limit_value = n
        return self

    async def to_list(self, length=None):
        self.to_list_length = length
        if self._error is not None:
            raise self._error
        return self._docs


class _FakeCollection:
    """Minimal AsyncCollection stand-in recording every call."""

    def __init__(
        self,
        find_docs=(),
        find_error=None,
        find_one_result=None,
        find_one_error=None,
        update_result=None,
        find_one_and_update_result=None,
        find_one_and_update_error=None,
    ):
        self._find_docs = find_docs
        self._find_error = find_error
        self.find_one_result = find_one_result
        self.find_one_error = find_one_error
        self.update_result = update_result or _UpdateResult()
        self.find_one_and_update_result = find_one_and_update_result
        self.find_one_and_update_error = find_one_and_update_error

        self.find_queries = []
        self.find_cursors = []
        self.find_one_calls = []
        self.update_one_calls = []
        self.find_one_and_update_calls = []
        self.count_documents_calls = []

    def find(self, query, projection=None):
        self.find_queries.append(query)
        cursor = _FakeFindCursor(self._find_docs, error=self._find_error)
        self.find_cursors.append(cursor)
        return cursor

    async def find_one(self, query, projection=None):
        self.find_one_calls.append((query, projection))
        if self.find_one_error is not None:
            raise self.find_one_error
        return self.find_one_result

    async def update_one(self, filter, update, upsert=False):
        self.update_one_calls.append((filter, update, upsert))
        return self.update_result

    async def find_one_and_update(self, filter, update, upsert=False, **kwargs):
        self.find_one_and_update_calls.append((filter, update, upsert, kwargs))
        if self.find_one_and_update_error is not None:
            raise self.find_one_and_update_error
        return self.find_one_and_update_result

    async def count_documents(self, query, **kwargs):
        self.count_documents_calls.append(query)
        return 42


def _storage(data=None, ctrl=None) -> MongoDocStatusStorage:
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.workspace = "t"
    storage.namespace = "doc_status"
    storage._collection_name = "t_doc_status"
    storage._data = data if data is not None else _FakeCollection()
    storage._ctrl = ctrl if ctrl is not None else _FakeCollection()
    return storage


# ---------------------------------------------------------------------------
# get_docs_by_statuses_page: query shape / sort / limit / cursor advance
# ---------------------------------------------------------------------------


async def test_page_start_query_shape_and_cursor_end():
    data = _FakeCollection(find_docs=[_ROW_1, _ROW_2])
    storage = _storage(data=data)

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED, DocStatus.PENDING], limit=5, position=CURSOR_START
    )

    # Query: status $in only — no keyset resume, no generation predicate.
    assert data.find_queries == [{"status": {"$in": ["failed", "pending"]}}]
    cursor = data.find_cursors[0]
    assert cursor.sort_spec == [("created_at", 1), ("_id", 1)]
    assert cursor.limit_value == 5
    assert cursor.to_list_length == 5

    assert set(page.docs) == {"doc-1", "doc-2"}
    # returned (2) < limit (5) proves exhaustion.
    assert page.next_position is CURSOR_END


async def test_page_keyset_and_generation_predicates():
    data = _FakeCollection(find_docs=[])
    storage = _storage(data=data)
    opaque = json.dumps(["2026-01-01T00:00:00+00:00", "doc-1"])

    await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED],
        limit=2,
        position=CursorAfter(opaque),
        max_failure_generation=7,
    )

    query = data.find_queries[0]
    assert query["status"] == {"$in": ["failed"]}
    keyset_or, generation_or = query["$and"]
    assert keyset_or == {
        "$or": [
            {"created_at": {"$gt": "2026-01-01T00:00:00+00:00"}},
            {"created_at": "2026-01-01T00:00:00+00:00", "_id": {"$gt": "doc-1"}},
        ]
    }
    # Generation predicate constrains ONLY failed rows; a missing field is
    # logical 0 and therefore always eligible.
    assert generation_or == {
        "$or": [
            {"status": {"$ne": "failed"}},
            {"failure_generation": {"$lte": 7}},
            {"failure_generation": {"$exists": False}},
        ]
    }


async def test_page_no_generation_predicate_without_cutoff():
    data = _FakeCollection(find_docs=[])
    storage = _storage(data=data)

    await storage.get_docs_by_statuses_page([DocStatus.FAILED], limit=2)

    assert "$and" not in data.find_queries[0]


async def test_page_full_page_advances_to_last_returned_key():
    data = _FakeCollection(find_docs=[_ROW_1, _ROW_2])
    storage = _storage(data=data)

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED, DocStatus.PENDING], limit=2
    )

    assert isinstance(page.next_position, CursorAfter)
    assert json.loads(page.next_position.opaque) == [
        "2026-01-02T00:00:00+00:00",
        "doc-2",
    ]


async def test_page_relaxed_skip_is_still_consumed_strict_raises():
    bad_row = {
        # Missing "status": unconvertible, but query-returned hence consumed.
        "_id": "doc-bad",
        "created_at": "2026-01-03T00:00:00+00:00",
    }
    data = _FakeCollection(find_docs=[_ROW_1, bad_row])
    storage = _storage(data=data)

    page = await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=2)

    # Relaxed: skipped from the page, but the cursor advances past the RAW
    # last returned doc — never re-read, never falsely terminal.
    assert set(page.docs) == {"doc-1"}
    assert isinstance(page.next_position, CursorAfter)
    assert json.loads(page.next_position.opaque) == [
        "2026-01-03T00:00:00+00:00",
        "doc-bad",
    ]

    with pytest.raises(KeyError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=2, strict=True
        )


async def test_missing_created_at_encodes_null_bucket_cursor():
    """A doc whose created_at field is entirely ABSENT keys as (None, _id):
    encoding it as "" would break the resume filter — {"created_at": ""}
    matches neither a missing field nor a null value, so a second corrupt
    doc past a page boundary would silently fall out of the sweep."""
    from lightrag.kg.mongo_impl import MongoDocStatusStorage

    assert MongoDocStatusStorage._doc_cursor_key({"_id": "doc-x"}) == (None, "doc-x")
    assert MongoDocStatusStorage._doc_cursor_key(
        {"_id": "doc-y", "created_at": None}
    ) == (None, "doc-y")


async def test_null_bucket_cursor_resumes_with_missing_matching_predicate():
    """Cursor inside the missing/null bucket: the resume filter must use
    {"created_at": None} (matches BOTH missing and null per Mongo $eq:null
    semantics) plus the $ne:None arm for every later bucket."""
    data = _FakeCollection(find_docs=[])
    storage = _storage(data=data)

    await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED],
        limit=2,
        position=CursorAfter(json.dumps([None, "doc-null-1"])),
    )

    query = data.find_queries[0]
    (keyset_or,) = query["$and"]
    assert keyset_or == {
        "$or": [
            {"created_at": None, "_id": {"$gt": "doc-null-1"}},
            {"created_at": {"$ne": None}},
        ]
    }


async def test_page_malformed_cursor_raises_control_plane_error():
    storage = _storage()

    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.FAILED], limit=2, position=CursorAfter("not-json")
        )
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.FAILED],
            limit=2,
            position=CursorAfter(json.dumps(["2026-01-01", 3])),
        )


async def test_page_transport_error_propagates():
    data = _FakeCollection(find_error=PyMongoError("boom"))
    storage = _storage(data=data)

    with pytest.raises(PyMongoError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.FAILED], limit=2, strict=True
        )


async def test_page_invalid_limit_and_terminal_position():
    storage = _storage()

    with pytest.raises(ValueError):
        await storage.get_docs_by_statuses_page([DocStatus.FAILED], limit=0)

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED], limit=2, position=CURSOR_END
    )
    assert page.docs == {} and page.next_position is CURSOR_END
    assert storage._data.find_queries == []  # no query issued


# ---------------------------------------------------------------------------
# count_docs_by_statuses
# ---------------------------------------------------------------------------


async def test_count_docs_by_statuses_counts_server_side():
    data = _FakeCollection()
    storage = _storage(data=data)

    count = await storage.count_docs_by_statuses(
        [DocStatus.PENDING, DocStatus.PROCESSING]
    )

    assert count == 42
    assert data.count_documents_calls == [
        {"status": {"$in": ["pending", "processing"]}}
    ]
    assert await storage.count_docs_by_statuses([]) == 0


# ---------------------------------------------------------------------------
# update_doc_status_fields
# ---------------------------------------------------------------------------


async def test_update_fields_rejects_created_at_without_db_call():
    data = _FakeCollection()
    storage = _storage(data=data)

    with pytest.raises(ValueError):
        await storage.update_doc_status_fields("doc-1", {"created_at": "x"})
    assert data.update_one_calls == []


async def test_update_fields_missing_row_raises_unless_missing_ok():
    data = _FakeCollection(update_result=_UpdateResult(matched_count=0))
    storage = _storage(data=data)

    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("doc-x", {"status": "pending"})

    await storage.update_doc_status_fields(
        "doc-x", {"status": "pending"}, missing_ok=True
    )

    data.update_result = _UpdateResult(matched_count=1)
    await storage.update_doc_status_fields("doc-1", {"status": "pending"})
    assert data.update_one_calls[-1] == (
        {"_id": "doc-1"},
        {"$set": {"status": "pending"}},
        False,
    )


# ---------------------------------------------------------------------------
# mark_doc_failed
# ---------------------------------------------------------------------------


async def test_mark_doc_failed_idempotent_same_attempt_no_reserve():
    existing = {
        "_id": "doc-1",
        "status": "failed",
        "processing_attempt_id": "att-1",
        "failure_attempt_id": "att-1",
        "failure_generation": 5,
    }
    data = _FakeCollection(find_one_result=existing)
    ctrl = _FakeCollection()
    storage = _storage(data=data, ctrl=ctrl)

    generation = await storage.mark_doc_failed("doc-1", {"error_msg": "boom"})

    assert generation == 5
    assert ctrl.find_one_and_update_calls == []  # no $inc reserved
    assert data.find_one_and_update_calls == []  # no publish issued


async def test_mark_doc_failed_reserves_and_publishes_conditionally():
    data = _FakeCollection(find_one_result=None)
    ctrl = _FakeCollection(
        find_one_and_update_result={
            "schema_version": 1,
            "mode": "enforced",
            "failure_generation_counter": 9,
        }
    )
    storage = _storage(data=data, ctrl=ctrl)

    generation = await storage.mark_doc_failed(
        "doc-new",
        {
            "processing_attempt_id": "att-2",
            "created_at": "2026-01-05T00:00:00+00:00",
            "error_msg": "boom",
        },
    )

    assert generation == 9
    # Reservation is an atomic $inc on the version-guarded ctrl doc.
    ctrl_filter, ctrl_update, _, _ = ctrl.find_one_and_update_calls[0]
    assert ctrl_filter == {"_id": "scheduling_ctrl", "schema_version": 1}
    assert ctrl_update == {"$inc": {"failure_generation_counter": 1}}

    publish_filter, publish_update, upsert, _ = data.find_one_and_update_calls[0]
    assert upsert is True
    # CAS guard against a concurrent FAILED publish of the same attempt.
    assert publish_filter == {
        "_id": "doc-new",
        "$nor": [{"status": "failed", "failure_attempt_id": "att-2"}],
    }
    set_fields = publish_update["$set"]
    assert set_fields["status"] == "failed"
    assert set_fields["failure_generation"] == 9
    assert set_fields["failure_attempt_id"] == "att-2"
    assert "created_at" not in set_fields  # immutable for existing rows
    # created_at lands only via $setOnInsert (conditional create path).
    assert publish_update["$setOnInsert"]["created_at"] == ("2026-01-05T00:00:00+00:00")
    assert publish_update["$setOnInsert"]["chunks_list"] == []


# ---------------------------------------------------------------------------
# failure-generation mode + reservation
# ---------------------------------------------------------------------------


async def test_mode_mapping_never_degrades_to_legacy():
    storage = _storage(ctrl=_FakeCollection(find_one_result=None))
    assert await storage.get_failure_generation_mode() == (
        FailureGenerationMode.MIGRATING
    )

    storage = _storage(
        ctrl=_FakeCollection(find_one_result={"schema_version": 99, "mode": "enforced"})
    )
    assert await storage.get_failure_generation_mode() == (
        FailureGenerationMode.MIGRATING
    )

    storage = _storage(
        ctrl=_FakeCollection(find_one_result={"schema_version": 1, "mode": "weird"})
    )
    assert await storage.get_failure_generation_mode() == (
        FailureGenerationMode.MIGRATING
    )

    storage = _storage(
        ctrl=_FakeCollection(find_one_result={"schema_version": 1, "mode": "enforced"})
    )
    assert await storage.get_failure_generation_mode() == (
        FailureGenerationMode.ENFORCED
    )


async def test_mode_transport_error_propagates():
    storage = _storage(ctrl=_FakeCollection(find_one_error=PyMongoError("down")))
    with pytest.raises(PyMongoError):
        await storage.get_failure_generation_mode()


async def test_reserve_missing_or_corrupt_ctrl_raises_control_plane_error():
    # find_one_and_update returning None == ctrl missing or version mismatch.
    storage = _storage(ctrl=_FakeCollection(find_one_and_update_result=None))
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()

    storage = _storage(
        ctrl=_FakeCollection(
            find_one_and_update_result={
                "schema_version": 1,
                "failure_generation_counter": "corrupt",
            }
        )
    )
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()

    storage = _storage(
        ctrl=_FakeCollection(
            find_one_and_update_result={
                "schema_version": 1,
                "failure_generation_counter": 4,
            }
        )
    )
    assert await storage.reserve_failure_generation() == 4


# ---------------------------------------------------------------------------
# ensure_processing_attempt_id
# ---------------------------------------------------------------------------


async def test_ensure_attempt_id_mints_when_conditional_write_matches():
    data = _FakeCollection(update_result=_UpdateResult(matched_count=1))
    storage = _storage(data=data)

    attempt = await storage.ensure_processing_attempt_id("doc-1")

    conditional_filter, update, _ = data.update_one_calls[0]
    assert conditional_filter["_id"] == "doc-1"
    assert conditional_filter["$or"] == [
        {"processing_attempt_id": {"$exists": False}},
        {"processing_attempt_id": {"$in": [None, ""]}},
    ]
    assert update == {"$set": {"processing_attempt_id": attempt}}


async def test_ensure_attempt_id_reuses_existing_and_raises_on_missing():
    data = _FakeCollection(
        update_result=_UpdateResult(matched_count=0),
        find_one_result={"_id": "doc-1", "processing_attempt_id": "att-old"},
    )
    storage = _storage(data=data)
    assert await storage.ensure_processing_attempt_id("doc-1") == "att-old"

    data = _FakeCollection(
        update_result=_UpdateResult(matched_count=0), find_one_result=None
    )
    storage = _storage(data=data)
    with pytest.raises(StorageRecordNotFoundError):
        await storage.ensure_processing_attempt_id("doc-missing")


# ---------------------------------------------------------------------------
# basename lookup: primary-only filter, strict vs legacy error semantics
# ---------------------------------------------------------------------------


async def test_basename_filter_excludes_duplicates():
    data = _FakeCollection(find_one_result={"_id": "doc-1", "file_path": "a.txt"})
    storage = _storage(data=data)

    result = await storage.get_doc_by_file_basename_strict("a.txt")

    assert result == ("doc-1", {"_id": "doc-1", "file_path": "a.txt"})
    query, _ = data.find_one_calls[0]
    assert query == {"file_path": "a.txt", "metadata.is_duplicate": {"$ne": True}}

    # Legacy method issues the identical primary-only query.
    await storage.get_doc_by_file_basename("a.txt")
    assert data.find_one_calls[1][0] == query


async def test_basename_strict_propagates_legacy_swallows():
    data = _FakeCollection(find_one_error=PyMongoError("down"))
    storage = _storage(data=data)

    with pytest.raises(PyMongoError):
        await storage.get_doc_by_file_basename_strict("a.txt")

    # Legacy compat path: same failure reads as a best-effort miss.
    assert await storage.get_doc_by_file_basename("a.txt") is None
