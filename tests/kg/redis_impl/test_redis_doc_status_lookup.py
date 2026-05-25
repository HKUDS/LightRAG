"""Unit tests for RedisDocStatusStorage basename / content_hash lookups.

These tests do NOT require a live Redis instance — the Redis client is
substituted with an in-memory fake that mirrors just enough of the
``redis.asyncio`` surface used by ``RedisDocStatusStorage`` (``scan``,
``pipeline().get/set/exists/delete`` and ``execute``). This keeps the suite
offline-safe and fast.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from lightrag.base import DocStatus
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(status: str, file_path: str, content_hash: str | None = None) -> dict:
    payload = {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
    }
    if content_hash is not None:
        payload["content_hash"] = content_hash
    return payload


class _FakePipeline:
    """Mimics redis.asyncio pipeline: commands are queued synchronously and
    executed in batch via ``await execute()``."""

    def __init__(self, store: dict[str, str]):
        self._store = store
        self._ops: list[tuple] = []

    def get(self, key: str) -> None:
        self._ops.append(("get", key))

    def set(self, key: str, value: str) -> None:
        self._ops.append(("set", key, value))

    def exists(self, key: str) -> None:
        self._ops.append(("exists", key))

    def delete(self, key: str) -> None:
        self._ops.append(("delete", key))

    async def execute(self) -> list:
        results = []
        for op in self._ops:
            kind = op[0]
            if kind == "get":
                results.append(self._store.get(op[1]))
            elif kind == "set":
                self._store[op[1]] = op[2]
                results.append(True)
            elif kind == "exists":
                results.append(1 if op[1] in self._store else 0)
            elif kind == "delete":
                existed = op[1] in self._store
                self._store.pop(op[1], None)
                results.append(1 if existed else 0)
        self._ops.clear()
        return results


class _FakeRedis:
    """Tiny in-memory stand-in for the bits of ``redis.asyncio.Redis`` that
    ``RedisDocStatusStorage`` actually calls."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def ping(self):
        return True

    async def scan(self, *args, **kwargs):
        # Signature: scan(cursor, match=..., count=...). args holds the cursor
        # positional; we ignore it and return single-shot results (cursor=0)
        # so callers stop looping.
        _ = args
        match = kwargs.get("match", "")
        if match.endswith("*"):
            prefix = match[:-1]
            keys = [k for k in self.store if k.startswith(prefix)]
        else:
            keys = [k for k in self.store if k == match]
        return 0, keys

    def scan_iter(self, **kwargs):
        # Used by is_empty(); returns an async iterator.
        match = kwargs.get("match", "")
        prefix = match[:-1] if match.endswith("*") else match
        keys = [k for k in self.store if k.startswith(prefix)]

        async def _aiter():
            for k in keys:
                yield k

        return _aiter()

    def pipeline(self):
        return _FakePipeline(self.store)

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str):
        self.store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self.store:
                self.store.pop(k)
                count += 1
        return count


@pytest.fixture
def redis_doc_status(monkeypatch):
    """Construct RedisDocStatusStorage with its Redis client replaced by a
    fake in-memory store. No network I/O occurs."""
    fake = _FakeRedis()

    # Stub out the connection pool factory so __post_init__ does not invoke
    # the real redis-py ConnectionPool.from_url (which is lazy but still
    # parses URLs and caches state we don't want).
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.release_pool",
        lambda redis_url: None,
    )
    # Swap the Redis client class used in __post_init__ so any call site that
    # reaches self._redis hits the fake.
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: fake
    )

    from lightrag.kg.redis_impl import RedisDocStatusStorage

    storage = RedisDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    storage._initialized = True  # skip the real ping in initialize()
    return storage


def _store_raw(storage, doc_id: str, payload: dict) -> None:
    """Write a record directly into the fake redis backing store, bypassing
    ``upsert`` so we control the serialized shape (e.g. legacy rows without
    a content_hash field)."""
    key = f"{storage.final_namespace}:{doc_id}"
    storage._redis.store[key] = json.dumps(payload)


async def test_get_doc_by_file_basename_returns_tuple_on_hit(redis_doc_status):
    _store_raw(redis_doc_status, "doc-1", _doc(DocStatus.PROCESSED.value, "report.pdf"))

    result = await redis_doc_status.get_doc_by_file_basename("report.pdf")

    assert result is not None
    doc_id, doc_data = result
    assert doc_id == "doc-1"
    assert doc_data["file_path"] == "report.pdf"


async def test_get_doc_by_file_basename_misses_when_not_present(redis_doc_status):
    _store_raw(redis_doc_status, "doc-1", _doc(DocStatus.PROCESSED.value, "report.pdf"))

    assert await redis_doc_status.get_doc_by_file_basename("other.pdf") is None


async def test_get_doc_by_file_basename_empty_returns_none(redis_doc_status):
    _store_raw(redis_doc_status, "doc-1", _doc(DocStatus.PROCESSED.value, "report.pdf"))

    assert await redis_doc_status.get_doc_by_file_basename("") is None


async def test_get_doc_by_file_basename_unknown_source_sentinel(redis_doc_status):
    # A record whose file_path itself is the sentinel must not be returned by
    # a basename lookup for "unknown_source" — otherwise every unsourced doc
    # would collide.
    _store_raw(
        redis_doc_status, "doc-1", _doc(DocStatus.PROCESSED.value, "unknown_source")
    )

    assert await redis_doc_status.get_doc_by_file_basename("unknown_source") is None


async def test_get_doc_by_content_hash_returns_tuple_on_hit(redis_doc_status):
    _store_raw(
        redis_doc_status,
        "doc-1",
        _doc(DocStatus.PROCESSED.value, "report.pdf", content_hash="abc123"),
    )

    result = await redis_doc_status.get_doc_by_content_hash("abc123")

    assert result is not None
    doc_id, doc_data = result
    assert doc_id == "doc-1"
    assert doc_data["content_hash"] == "abc123"


async def test_get_doc_by_content_hash_misses_when_not_present(redis_doc_status):
    _store_raw(
        redis_doc_status,
        "doc-1",
        _doc(DocStatus.PROCESSED.value, "report.pdf", content_hash="abc123"),
    )

    assert await redis_doc_status.get_doc_by_content_hash("zzz999") is None


async def test_get_doc_by_content_hash_empty_returns_none_even_with_legacy_rows(
    redis_doc_status,
):
    # Legacy row written before the content_hash field existed; an empty-string
    # query must not match it. The early-return guard protects against this.
    _store_raw(
        redis_doc_status, "doc-legacy", _doc(DocStatus.PROCESSED.value, "old.pdf")
    )

    assert await redis_doc_status.get_doc_by_content_hash("") is None


async def test_get_doc_by_content_hash_ignores_legacy_rows(redis_doc_status):
    # A legacy row (no content_hash field) must not be returned when querying
    # any non-empty hash, because doc_data.get("content_hash") is None and
    # None != "abc123".
    _store_raw(
        redis_doc_status, "doc-legacy", _doc(DocStatus.PROCESSED.value, "old.pdf")
    )
    _store_raw(
        redis_doc_status,
        "doc-new",
        _doc(DocStatus.PROCESSED.value, "new.pdf", content_hash="abc123"),
    )

    result = await redis_doc_status.get_doc_by_content_hash("abc123")

    assert result is not None
    doc_id, _ = result
    assert doc_id == "doc-new"
