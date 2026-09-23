"""Regression test: ``get_status_counts`` must not double-count a document
whose key SCAN hands back more than once.

Redis's own SCAN contract only guarantees that every key present for the
whole iteration is returned AT LEAST once, not exactly once: a concurrent
rehash of the keyspace (routine under write load, since new documents are
being inserted while this sweep runs) can hand the same key back in a later
batch. ``get_status_counts`` folds each returned value straight into a plain
per-status counter with no per-doc_id key to dedupe against, so a repeated
key silently inflates that document's status count -- the number the WebUI
dashboard shows.

The client is a small purpose-built fake (no live Redis) that replays this
exact duplicate-return shape: cursor 0 returns one key, and the following
(final) batch returns that same key again alongside a second, genuinely new
one.
"""

from __future__ import annotations

import json

import pytest

from lightrag.kg.redis_impl import RedisDocStatusStorage

pytestmark = pytest.mark.offline

_WORKSPACE = "ws"
_NAMESPACE = "doc_status"
_FINAL_NAMESPACE = f"{_WORKSPACE}_{_NAMESPACE}"


class _DuplicateScanPipeline:
    """Minimal buffered-GET pipeline over a fixed key/value store."""

    def __init__(self, rows: dict[str, str]):
        self._rows = rows
        self._keys: list[str] = []

    def get(self, key: str):
        self._keys.append(key)
        return self

    async def execute(self):
        results = [self._rows.get(k) for k in self._keys]
        self._keys = []
        return results


class _DuplicateScanRedis:
    """Replays a SCAN that returns one key twice across two batches --
    exactly what a concurrent keyspace rehash produces on real Redis."""

    def __init__(self, rows: dict[str, str]):
        self._rows = rows
        self.scan_calls = 0

    async def scan(self, cursor: int = 0, match: str = "", count: int = 1000):
        self.scan_calls += 1
        keys = list(self._rows.keys())
        if cursor == 0:
            # First batch: only the first key.
            return 1, [keys[0]]
        if cursor == 1:
            # Final batch: the same first key again (duplicate SCAN return)
            # plus the genuinely new second key.
            return 0, keys
        raise AssertionError(f"unexpected cursor {cursor}")  # pragma: no cover

    def pipeline(self, transaction: bool = True):
        return _DuplicateScanPipeline(self._rows)


def _new_storage(rows: dict[str, str]) -> RedisDocStatusStorage:
    storage = RedisDocStatusStorage.__new__(RedisDocStatusStorage)
    storage.workspace = _WORKSPACE
    storage.namespace = _NAMESPACE
    storage.final_namespace = _FINAL_NAMESPACE
    storage._redis = _DuplicateScanRedis(rows)
    return storage


@pytest.mark.asyncio
async def test_status_counts_does_not_double_count_a_duplicate_scan_return():
    rows = {
        f"{_FINAL_NAMESPACE}:doc-1": json.dumps({"status": "processed"}),
        f"{_FINAL_NAMESPACE}:doc-2": json.dumps({"status": "pending"}),
    }
    storage = _new_storage(rows)

    counts = await storage.get_status_counts()

    # Two distinct documents on the wire -- doc-1's key is merely returned
    # twice by SCAN. The count must reflect two documents, not three.
    assert counts["processed"] == 1
    assert counts["pending"] == 1
    assert sum(counts.values()) == 2
