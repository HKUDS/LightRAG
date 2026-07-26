"""Offline tests for the Mongo Phase 1 scheduling surface.

Covers the bounded keyset page API (query/sort/limit shape, consumed-position
cursor advance, null-bucket resume, malformed-cursor rejection), the strict
batch read (:meth:`get_docs_by_ids`), the conflict-aware source resolver
(:meth:`resolve_doc_source_strict`), and the source-conflict listing / explicit
CAS repair. The pymongo collection is driven through minimal in-process fakes —
no live services.
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
    SourceAbsent,
    SourceConflict,
    SourceUnique,
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


class _FakeAggCursor:
    """Records the aggregate(...).to_list(...) chain."""

    def __init__(self, docs, error=None):
        self._docs = list(docs)
        self._error = error
        self.to_list_length = None

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
        update_many_result=None,
        count_value=42,
        agg_docs=(),
        agg_error=None,
    ):
        self._find_docs = find_docs
        self._find_error = find_error
        self.find_one_result = find_one_result
        self.find_one_error = find_one_error
        self.update_result = update_result or _UpdateResult()
        self.update_many_result = update_many_result or _UpdateResult()
        self._count_value = count_value
        self._agg_docs = agg_docs
        self._agg_error = agg_error

        self.find_queries = []
        self.find_cursors = []
        self.find_one_calls = []
        self.update_one_calls = []
        self.update_many_calls = []
        self.count_documents_calls = []
        self.aggregate_calls = []
        self.agg_cursors = []

    def find(self, query, projection=None, session=None):
        self.find_queries.append(query)
        # find_docs may be a callable so a test can route per query (the
        # conflict listing fetches a bounded sample per surfaced source key).
        docs = self._find_docs(query) if callable(self._find_docs) else self._find_docs
        cursor = _FakeFindCursor(docs, error=self._find_error)
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

    async def update_many(self, filter, update, session=None):
        self.update_many_calls.append((filter, update, session))
        return self.update_many_result

    async def count_documents(self, query, **kwargs):
        self.count_documents_calls.append(query)
        return self._count_value

    async def aggregate(self, pipeline, **kwargs):
        self.aggregate_calls.append((pipeline, kwargs))
        cursor = _FakeAggCursor(self._agg_docs, error=self._agg_error)
        self.agg_cursors.append(cursor)
        return cursor


class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self):
        self.started = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def start_transaction(self):
        self.started = True
        return _FakeTxn()


class _FakeClient:
    def __init__(self, session):
        self._session = session

    def start_session(self):
        return self._session


class _FakeDb:
    def __init__(self, client):
        self.client = client


def _storage(data=None, db=None) -> MongoDocStatusStorage:
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.workspace = "t"
    storage.namespace = "doc_status"
    storage._collection_name = "t_doc_status"
    storage._data = data if data is not None else _FakeCollection()
    storage.db = db
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

    # Query: status $in only — no keyset resume clause on a fresh sweep.
    assert data.find_queries == [{"status": {"$in": ["failed", "pending"]}}]
    cursor = data.find_cursors[0]
    assert cursor.sort_spec == [("created_at", 1), ("_id", 1)]
    assert cursor.limit_value == 5
    assert cursor.to_list_length == 5

    assert set(page.docs) == {"doc-1", "doc-2"}
    # returned (2) < limit (5) proves exhaustion.
    assert page.next_position is CURSOR_END


async def test_page_keyset_predicate():
    data = _FakeCollection(find_docs=[])
    storage = _storage(data=data)
    opaque = json.dumps(["2026-01-01T00:00:00+00:00", "doc-1"])

    await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED],
        limit=2,
        position=CursorAfter(opaque),
    )

    query = data.find_queries[0]
    assert query["status"] == {"$in": ["failed"]}
    # Only the keyset resume clause — no generation cohort predicate.
    (keyset_or,) = query["$and"]
    assert keyset_or == {
        "$or": [
            {"created_at": {"$gt": "2026-01-01T00:00:00+00:00"}},
            {"created_at": "2026-01-01T00:00:00+00:00", "_id": {"$gt": "doc-1"}},
        ]
    }


async def test_page_no_and_clause_on_fresh_sweep():
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
# get_docs_by_ids: strict batch read
# ---------------------------------------------------------------------------


async def test_get_docs_by_ids_returns_present_omits_missing():
    data = _FakeCollection(find_docs=[_ROW_1, _ROW_2])
    storage = _storage(data=data)

    result = await storage.get_docs_by_ids(["doc-1", "doc-2", "doc-missing"])

    # One indexed $in query; missing id positively absent from the result set.
    assert data.find_queries == [{"_id": {"$in": ["doc-1", "doc-2", "doc-missing"]}}]
    assert set(result) == {"doc-1", "doc-2"}
    assert result["doc-1"].status is DocStatus.PENDING


async def test_get_docs_by_ids_empty_input_issues_no_query():
    data = _FakeCollection()
    storage = _storage(data=data)

    assert await storage.get_docs_by_ids([]) == {}
    assert data.find_queries == []


async def test_get_docs_by_ids_relaxed_skips_bad_row_strict_raises():
    bad = {"_id": "doc-bad", "created_at": "2026-01-03T00:00:00+00:00"}  # no status
    data = _FakeCollection(find_docs=[_ROW_1, bad])
    storage = _storage(data=data)

    relaxed = await storage.get_docs_by_ids(["doc-1", "doc-bad"])
    assert set(relaxed) == {"doc-1"}  # bad row dropped, present ids kept

    with pytest.raises(KeyError):
        await storage.get_docs_by_ids(["doc-1", "doc-bad"], strict=True)


# ---------------------------------------------------------------------------
# get_full_docs_by_ids: FULL DocProcessingStatus hydration
# ---------------------------------------------------------------------------

_FULL_ROW_1 = {
    "_id": "d1",
    "status": "pending",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
    "file_path": "a.txt",
    "content_summary": "summary of a",
    "content_length": 123,
    "chunks_list": ["chunk-1", "chunk-2"],
    "metadata": {"k": "v"},
}
_FULL_ROW_2 = {
    "_id": "d2",
    "status": "failed",
    "created_at": "2026-01-02T00:00:00+00:00",
    "updated_at": "2026-01-02T00:00:00+00:00",
    "file_path": "b.txt",
    "content_summary": "summary of b",
    "content_length": 456,
    "chunks_list": ["chunk-3"],
    "metadata": {},
    "error_msg": "boom",
}


async def test_get_full_docs_by_ids_present_omits_missing_full_projection():
    data = _FakeCollection(find_docs=[_FULL_ROW_1, _FULL_ROW_2])
    storage = _storage(data=data)

    result = await storage.get_full_docs_by_ids(["d1", "d2", "ghost"], strict=True)

    # One indexed $in query; the missing id is positively absent (omitted).
    assert data.find_queries == [{"_id": {"$in": ["d1", "d2", "ghost"]}}]
    assert set(result) == {"d1", "d2"}
    # FULL projection: fields excluded from DocSchedulingRecord are populated.
    assert result["d1"].content_summary == "summary of a"
    assert result["d1"].content_length == 123
    assert result["d1"].chunks_list == ["chunk-1", "chunk-2"]
    assert result["d1"].metadata == {"k": "v"}
    # DocProcessingStatus keeps status as the raw str value (str-enum): ==, not is.
    assert result["d2"].status == DocStatus.FAILED


async def test_get_full_docs_by_ids_empty_input_issues_no_query():
    data = _FakeCollection()
    storage = _storage(data=data)

    assert await storage.get_full_docs_by_ids([]) == {}
    assert data.find_queries == []


async def test_get_full_docs_by_ids_strict_raises_on_transport_error():
    data = _FakeCollection(find_error=PyMongoError("down"))
    storage = _storage(data=data)

    with pytest.raises(PyMongoError):
        await storage.get_full_docs_by_ids(["d1"], strict=True)


async def test_get_full_docs_by_ids_relaxed_skips_bad_doc_strict_raises():
    bad = {  # missing required content_summary/content_length -> unconvertible
        "_id": "d-bad",
        "status": "pending",
        "created_at": "2026-01-03T00:00:00+00:00",
        "updated_at": "2026-01-03T00:00:00+00:00",
        "file_path": "c.txt",
    }
    data = _FakeCollection(find_docs=[_FULL_ROW_1, bad])
    storage = _storage(data=data)

    relaxed = await storage.get_full_docs_by_ids(["d1", "d-bad"])
    assert set(relaxed) == {"d1"}  # bad doc dropped, present id kept

    with pytest.raises(TypeError):
        await storage.get_full_docs_by_ids(["d1", "d-bad"], strict=True)


# ---------------------------------------------------------------------------
# resolve_doc_source_strict: conflict-aware typed resolution
# ---------------------------------------------------------------------------

_PRIMARY_QUERY = {"file_path": "a.txt", "metadata.is_duplicate": {"$ne": True}}


async def test_resolve_source_absent_for_empty_and_sentinels():
    data = _FakeCollection(find_docs=[])
    storage = _storage(data=data)

    assert isinstance(await storage.resolve_doc_source_strict("a.txt"), SourceAbsent)
    # Sentinels short-circuit without touching the collection.
    assert isinstance(await storage.resolve_doc_source_strict(""), SourceAbsent)
    assert isinstance(
        await storage.resolve_doc_source_strict("unknown_source"), SourceAbsent
    )
    assert data.find_queries == [_PRIMARY_QUERY]


async def test_resolve_source_unique():
    data = _FakeCollection(find_docs=[_ROW_1])
    storage = _storage(data=data)

    resolution = await storage.resolve_doc_source_strict("a.txt")

    assert isinstance(resolution, SourceUnique)
    assert resolution.doc_id == "doc-1"
    assert resolution.doc.id == "doc-1"
    assert resolution.doc.status is DocStatus.PENDING
    # limit(2) is what proves uniqueness cheaply.
    assert data.find_cursors[0].limit_value == 2
    assert data.count_documents_calls == []  # no exact count needed for unique


async def test_resolve_source_conflict_reports_exact_count_and_sample():
    data = _FakeCollection(find_docs=[_ROW_2, _ROW_1], count_value=7)
    storage = _storage(data=data)

    resolution = await storage.resolve_doc_source_strict("a.txt")

    assert isinstance(resolution, SourceConflict)
    assert resolution.candidate_count == 7  # exact count via count_documents
    assert resolution.sample_doc_ids == ("doc-1", "doc-2")  # sorted sample
    assert data.count_documents_calls == [_PRIMARY_QUERY]


async def test_resolve_transport_error_propagates():
    data = _FakeCollection(find_error=PyMongoError("down"))
    storage = _storage(data=data)

    with pytest.raises(PyMongoError):
        await storage.resolve_doc_source_strict("a.txt")


# ---------------------------------------------------------------------------
# legacy get_doc_by_file_basename: primary-only query, best-effort errors
# ---------------------------------------------------------------------------


async def test_legacy_basename_primary_only_query():
    data = _FakeCollection(find_one_result={"_id": "doc-1", "file_path": "a.txt"})
    storage = _storage(data=data)

    result = await storage.get_doc_by_file_basename("a.txt")

    assert result == ("doc-1", {"_id": "doc-1", "file_path": "a.txt"})
    query, _ = data.find_one_calls[0]
    assert query == _PRIMARY_QUERY


async def test_legacy_basename_swallows_transport_error():
    data = _FakeCollection(find_one_error=PyMongoError("down"))
    storage = _storage(data=data)

    # Legacy compat path: a transport failure reads as a best-effort miss.
    assert await storage.get_doc_by_file_basename("a.txt") is None


# ---------------------------------------------------------------------------
# list_source_conflicts_page
# ---------------------------------------------------------------------------


async def test_list_conflicts_maps_groups_and_terminates():
    # The aggregation is COUNT-ONLY: $push-ing every id would allocate an array
    # per distinct file_path in the collection, conflicting or not.
    groups = [
        {"_id": "a.txt", "candidate_count": 3},
        {"_id": "b.txt", "candidate_count": 2},
    ]
    samples = {
        "a.txt": [{"_id": "doc-1"}, {"_id": "doc-2"}, {"_id": "doc-3"}],
        "b.txt": [{"_id": "doc-8"}, {"_id": "doc-9"}],
    }
    data = _FakeCollection(agg_docs=groups, find_docs=lambda q: samples[q["file_path"]])
    storage = _storage(data=data)

    page = await storage.list_source_conflicts_page(limit=5)

    pipeline = data.aggregate_calls[0][0]
    assert pipeline[0] == {
        "$match": {
            "file_path": {
                "$type": "string",
                "$nin": ["", "unknown_source", "no-file-path"],
            },
            "metadata.is_duplicate": {"$ne": True},
        }
    }
    assert pipeline[1]["$group"] == {
        "_id": "$file_path",
        "candidate_count": {"$sum": 1},
    }
    assert pipeline[2] == {"$match": {"candidate_count": {"$gte": 2}}}
    assert pipeline[3] == {"$sort": {"_id": 1}}
    assert pipeline[4] == {"$limit": 5}

    assert [c.canonical_source_key for c in page.conflicts] == ["a.txt", "b.txt"]
    assert page.conflicts[0].candidate_count == 3
    assert page.conflicts[0].sample_doc_ids == ("doc-1", "doc-2", "doc-3")  # sorted
    assert page.conflicts[1].sample_doc_ids == ("doc-8", "doc-9")
    # One bounded sample query per SURFACED key — server-side sorted + capped,
    # so a key with a pathological primary count still ships a fixed sample.
    assert [q["file_path"] for q in data.find_queries] == ["a.txt", "b.txt"]
    for cursor in data.find_cursors:
        assert cursor.sort_spec == [("_id", 1)]
        assert cursor.limit_value == storage._CONFLICT_SAMPLE_CAP
        assert cursor.to_list_length == storage._CONFLICT_SAMPLE_CAP
    # returned (2) < limit (5) proves exhaustion.
    assert page.next_position is CURSOR_END


async def test_list_conflicts_full_page_advances_cursor():
    groups = [
        {"_id": "a.txt", "candidate_count": 2},
        {"_id": "b.txt", "candidate_count": 2},
    ]
    data = _FakeCollection(agg_docs=groups)
    storage = _storage(data=data)

    page = await storage.list_source_conflicts_page(
        limit=2, position=CursorAfter(json.dumps("0.txt"))
    )

    # Cursor resumes strictly after the last canonical key.
    match = data.aggregate_calls[0][0][0]["$match"]
    assert match["file_path"]["$gt"] == "0.txt"
    # full page (2 == limit) → resume after the last group.
    assert isinstance(page.next_position, CursorAfter)
    assert json.loads(page.next_position.opaque) == "b.txt"


async def test_list_conflicts_terminal_and_invalid_limit():
    storage = _storage()

    with pytest.raises(ValueError):
        await storage.list_source_conflicts_page(limit=0)

    page = await storage.list_source_conflicts_page(limit=2, position=CURSOR_END)
    assert page.conflicts == () and page.next_position is CURSOR_END
    assert storage._data.aggregate_calls == []


# ---------------------------------------------------------------------------
# repair_source_conflict: dry-run + CAS commit
# ---------------------------------------------------------------------------

_CANDIDATES = [{"_id": "doc-1"}, {"_id": "doc-2"}, {"_id": "doc-3"}]
_FINGERPRINT = MongoDocStatusStorage._conflict_fingerprint(["doc-1", "doc-2", "doc-3"])


async def test_repair_dry_run_reports_fingerprint_without_mutation():
    data = _FakeCollection(find_docs=_CANDIDATES)
    storage = _storage(data=data)

    result = await storage.repair_source_conflict(
        "a.txt",
        primary_doc_id="doc-1",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )

    assert result.committed is False
    assert result.candidate_count == 3
    assert result.fingerprint == _FINGERPRINT
    assert result.demoted_sample_doc_ids == ("doc-2", "doc-3")
    assert data.update_many_calls == []  # dry run never mutates


async def test_repair_commit_cas_success_demotes_losers_in_txn():
    data = _FakeCollection(
        find_docs=_CANDIDATES, update_many_result=_UpdateResult(modified_count=2)
    )
    session = _FakeSession()
    storage = _storage(data=data, db=_FakeDb(_FakeClient(session)))

    result = await storage.repair_source_conflict(
        "a.txt",
        primary_doc_id="doc-1",
        expected_candidate_count=3,
        expected_candidate_fingerprint=_FINGERPRINT,
        dry_run=False,
    )

    assert result.committed is True
    assert result.demoted_sample_doc_ids == ("doc-2", "doc-3")
    assert session.started is True  # ran inside a transaction

    filt, update, sess = data.update_many_calls[0]
    assert filt == {"_id": {"$in": ["doc-2", "doc-3"]}}
    assert update == {
        "$set": {
            "metadata.is_duplicate": True,
            "metadata.original_doc_id": "doc-1",
        }
    }
    assert sess is session  # demotion runs in the same session/txn


async def test_repair_commit_cas_mismatch_raises_without_mutation():
    data = _FakeCollection(find_docs=_CANDIDATES)
    session = _FakeSession()
    storage = _storage(data=data, db=_FakeDb(_FakeClient(session)))

    with pytest.raises(StorageControlPlaneError):
        await storage.repair_source_conflict(
            "a.txt",
            primary_doc_id="doc-1",
            expected_candidate_count=2,  # stale count
            expected_candidate_fingerprint="stale",
            dry_run=False,
        )
    assert data.update_many_calls == []  # CAS refused the overwrite


async def test_repair_primary_not_in_candidates_raises_value_error():
    data = _FakeCollection(find_docs=[{"_id": "doc-2"}, {"_id": "doc-3"}])
    storage = _storage(data=data)

    with pytest.raises(ValueError):
        await storage.repair_source_conflict(
            "a.txt",
            primary_doc_id="doc-1",
            expected_candidate_count=0,
            expected_candidate_fingerprint="",
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_exclude_doc_id_adds_ne_filter():
    # LR2 Phase 2.5: exclude_doc_id becomes an in-query _id $ne, still served
    # by the partial content_hash index — never a scan. Sorted + limited so the
    # EARLIEST other holder wins deterministically (base contract).
    data = _FakeCollection(find_docs=[{"_id": "doc-2", "content_hash": "h"}])
    s = _storage(data=data)

    result = await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")
    assert result is not None and result[0] == "doc-2"
    # The id half of the exclusion; the pointer half is asserted in
    # test_get_doc_by_content_hash_excludes_rows_pointing_at_the_excluded_id.
    assert data.find_queries[0]["content_hash"] == "h"
    assert data.find_queries[0]["_id"] == {"$ne": "doc-1"}
    assert data.find_cursors[0].sort_spec == [("created_at", 1), ("_id", 1)]
    assert data.find_cursors[0].limit_value == 1

    # No exclusion → plain content_hash filter.
    await s.get_doc_by_content_hash("h")
    assert data.find_queries[-1] == {"content_hash": "h"}


async def test_get_doc_by_content_hash_propagates_query_failure():
    """Fail-proof: the error used to become None, which the dedup callers read
    as "no duplicate" — enqueuing a duplicate row on a transport blip."""
    data = _FakeCollection(find_docs=[], find_error=PyMongoError("boom"))
    s = _storage(data=data)
    with pytest.raises(PyMongoError):
        await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_excludes_rows_pointing_at_the_excluded_id():
    """Second half of ``exclude_doc_id`` (base contract): a ``$nor`` clause drops
    rows marked ``is_duplicate`` that name the excluded id as their original —
    such a row records that the content belongs to the asking document, so
    returning it would close an is_duplicate cycle and leave their shared source
    with no primary. In-query, so the sort+limit still yield the earliest
    surviving holder rather than a post-filtered single row."""
    data = _FakeCollection(find_docs=[{"_id": "doc-3", "content_hash": "h"}])
    s = _storage(data=data)

    result = await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")

    assert result is not None and result[0] == "doc-3"
    assert data.find_queries == [
        {
            "content_hash": "h",
            "_id": {"$ne": "doc-1"},
            "$nor": [
                {
                    "metadata.is_duplicate": True,
                    "metadata.original_doc_id": "doc-1",
                }
            ],
        }
    ]
    assert data.find_cursors[0].sort_spec == [("created_at", 1), ("_id", 1)]
    assert data.find_cursors[0].limit_value == 1

    # No exclusion → no $nor clause either.
    await s.get_doc_by_content_hash("h")
    assert "$nor" not in data.find_queries[-1]
