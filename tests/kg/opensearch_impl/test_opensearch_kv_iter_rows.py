"""``OpenSearchKVStorage.iter_rows``: the enumeration surface (scenario 16 in
docs/design/ConfigurationStorage.md) on the OpenSearch backend.

A PIT + search_after scan, read-your-writes against the process-local buffer:
a buffered upsert is yielded in place of its indexed version, a buffered
delete hides its row, and a buffered row the index has never seen is yielded
after the scan.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.exceptions import StorageNotInitializedError
from lightrag.kg.opensearch_impl import OpenSearchKVStorage

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def patch_shard_doc_supported():
    with patch("lightrag.kg.opensearch_impl._shard_doc_supported", True):
        yield


def _hits(ids_and_sources):
    return [
        {
            "_id": doc_id,
            "_source": {**source, "__mirrored_id": doc_id},
            "sort": [doc_id],
        }
        for doc_id, source in ids_and_sources
    ]


def _storage(pages, *, pending=None, deleted=()):
    storage = OpenSearchKVStorage.__new__(OpenSearchKVStorage)
    storage.namespace = "config"
    storage.workspace = "iterws"
    storage.final_namespace = "iterws_config"
    storage._index_name = "iterws_config"
    storage._index_ready = True
    storage._pending_upserts = dict(pending or {})
    storage._pending_kv_deletes = set(deleted)
    storage._flush_lock = asyncio.Lock()
    storage._write_generation = 0
    storage._refreshed_generation = 0
    client = MagicMock()
    client.indices = MagicMock()
    client.indices.refresh = AsyncMock()
    client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    client.delete_pit = AsyncMock()
    responses = [{"hits": {"hits": page}} for page in pages] + [{"hits": {"hits": []}}]
    client.search = AsyncMock(side_effect=responses)
    storage.client = client
    return storage, client


async def test_rows_stream_across_pit_pages_in_the_point_read_shape():
    storage, client = _storage(
        [
            _hits([("a", {"value": {"n": 1}}), ("b", {"value": {"n": 2}})]),
            _hits([("c", {"value": {"n": 3}})]),
        ]
    )

    rows = [row async for row in storage.iter_rows(page_size=2)]

    assert [r["_id"] for r in rows] == ["a", "b", "c"]
    assert all("__mirrored_id" not in r for r in rows)
    assert all(r["create_time"] == 0 and r["update_time"] == 0 for r in rows)
    client.create_pit.assert_awaited_once()
    client.delete_pit.assert_awaited_once()
    # Every search asked for the page size, never the whole index.
    assert all(
        call.kwargs["body"]["size"] == 2 for call in client.search.call_args_list
    )


async def test_the_pending_buffer_is_read_your_writes():
    storage, _ = _storage(
        [
            _hits(
                [
                    ("indexed", {"value": 1}),
                    ("replaced", {"value": "old"}),
                    ("gone", {"value": 1}),
                ]
            )
        ],
        pending={"replaced": {"value": "new"}, "fresh": {"value": "buffered"}},
        deleted={"gone"},
    )

    rows = {r["_id"]: r for r in [row async for row in storage.iter_rows()]}

    assert set(rows) == {"indexed", "replaced", "fresh"}
    assert rows["replaced"]["value"] == "new"
    assert rows["fresh"]["value"] == "buffered"


async def test_an_uninitialized_storage_raises():
    storage, _ = _storage([])
    storage._flush_lock = None
    with pytest.raises(StorageNotInitializedError):
        async for _ in storage.iter_rows():
            pass
