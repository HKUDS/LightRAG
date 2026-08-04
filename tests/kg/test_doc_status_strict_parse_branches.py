"""Per-record strict/relaxed parse branches for Mongo/Redis/Postgres doc-status.

Cross-backend companion to the json/OpenSearch strict suites: every
``get_docs_by_statuses`` implementation must SKIP an undeserializable record
in relaxed mode (the historical contract — note ``DocProcessingStatus(**data)``
raises ``TypeError``, not ``KeyError``, on missing required fields) and RAISE
it under ``strict=True``.  Backends are driven through minimal in-process
fakes of their storage handles — no live services.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lightrag.base import DocStatus
from lightrag.kg.mongo_impl import MongoDocStatusStorage
from lightrag.kg.postgres_impl import PGDocStatusStorage
from lightrag.kg.redis_impl import RedisDocStatusStorage

pytestmark = pytest.mark.offline

_GOOD_FIELDS = {
    "content_summary": "summary",
    "content_length": 10,
    "file_path": "good.txt",
    "status": "failed",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


# ---------------------------------------------------------------------------
# Mongo
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        return self._docs


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, query):
        wanted = set(query["status"]["$in"])
        return _FakeCursor([d for d in self._docs if d["status"] in wanted])


def _mongo_storage(docs) -> MongoDocStatusStorage:
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.workspace = "t"
    storage._data = _FakeCollection(docs)
    return storage


async def test_mongo_relaxed_skips_bad_record_strict_raises():
    docs = [
        {"_id": "good", **_GOOD_FIELDS},
        {"_id": "bad", "status": "failed"},  # missing required fields
    ]
    storage = _mongo_storage(docs)

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}  # skipped, not crashed (TypeError caught)

    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self, entries):
        self._entries = entries
        self._keys = []

    def get(self, key):
        self._keys.append(key)

    async def execute(self):
        return [self._entries[k] for k in self._keys]


class _FakeRedis:
    def __init__(self, entries):
        self._entries = entries

    async def scan(self, cursor, match=None, count=None):
        return 0, list(self._entries)

    def pipeline(self):
        return _FakePipe(self._entries)


def _redis_storage(entries) -> RedisDocStatusStorage:
    storage = RedisDocStatusStorage.__new__(RedisDocStatusStorage)
    storage.workspace = "t"
    storage.final_namespace = "ns"

    @asynccontextmanager
    async def _conn():
        yield _FakeRedis(entries)

    storage._get_redis_connection = _conn
    return storage


async def test_redis_relaxed_skips_bad_record_strict_raises():
    entries = {
        "ns:good": json.dumps({**_GOOD_FIELDS, "metadata": {}, "error_msg": None}),
        "ns:bad": json.dumps({"status": "failed"}),  # missing required fields
    }
    storage = _redis_storage(entries)

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}  # skipped, not crashed (TypeError caught)

    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


def _pg_storage(rows) -> PGDocStatusStorage:
    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.workspace = "t"

    async def _query(sql, params, multirows=False):
        return rows

    storage.db = SimpleNamespace(query=_query)
    return storage


async def test_postgres_relaxed_skips_bad_row_strict_raises():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good_row = {
        "id": "good",
        "content_summary": "summary",
        "content_length": 10,
        "file_path": "good.txt",
        "status": "failed",
        "created_at": ts,
        "updated_at": ts,
        "chunks_count": 0,
        "chunks_list": [],
        "metadata": {},
        "error_msg": None,
    }
    bad_row = {"id": "bad", "status": "failed"}  # missing required columns
    storage = _pg_storage([good_row, bad_row])

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}

    with pytest.raises(KeyError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)
