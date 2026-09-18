"""``MongoKVStorage.iter_rows``: the enumeration surface (scenario 16 in
docs/design/ConfigurationStorage.md) on the MongoDB backend.

A server cursor with ``batch_size`` bounding what is in memory; ``_id`` is the
document key already and the time defaults match ``get_by_ids``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lightrag.exceptions import StorageNotInitializedError
from lightrag.kg.mongo_impl import MongoKVStorage

pytestmark = pytest.mark.offline


class _FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        async def _gen():
            for doc in self._docs:
                yield dict(doc)

        return _gen()


def _storage(docs):
    storage = MongoKVStorage.__new__(MongoKVStorage)
    storage.namespace = "config"
    storage.workspace = "iterws"
    storage.global_config = {}
    collection = MagicMock()
    collection.find = MagicMock(return_value=_FakeCursor(docs))
    storage._data = collection
    storage.db = MagicMock()
    return storage, collection


async def test_rows_stream_through_a_bounded_cursor():
    docs = [{"_id": f"k{i}", "value": {"n": i}} for i in range(5)]
    storage, collection = _storage(docs)

    rows = [row async for row in storage.iter_rows(page_size=2)]

    assert [r["_id"] for r in rows] == [f"k{i}" for i in range(5)]
    assert all(r["create_time"] == 0 and r["update_time"] == 0 for r in rows)
    collection.find.assert_called_once_with({}, batch_size=2)


async def test_an_empty_collection_yields_nothing():
    storage, _ = _storage([])
    assert [row async for row in storage.iter_rows()] == []


async def test_an_uninitialized_storage_raises():
    storage, _ = _storage([])
    storage._data = None
    with pytest.raises(StorageNotInitializedError):
        async for _ in storage.iter_rows():
            pass
