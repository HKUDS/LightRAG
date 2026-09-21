"""``OpenSearchKVStorage.iter_rows``: the enumeration surface (scenario 16 in
docs/design/ConfigurationStorage.md) on the OpenSearch backend.

A PIT + search_after scan, read-your-writes against the process-local buffer:
a buffered upsert is yielded in place of its indexed version, a buffered
delete hides its row, and a buffered row the index has never seen is yielded
after the scan.

A missing index RAISES rather than ending the scan: the base contract forbids
presenting a partial listing as a complete one, and ``_chunk_source_is_populated``
turns a clean end into "confirmed empty", which is a durable ``origin=empty``
baseline write.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opensearchpy.exceptions import OpenSearchException

from lightrag.exceptions import StorageControlPlaneError, StorageNotInitializedError
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


async def test_a_dropped_index_raises_instead_of_ending_the_scan():
    """``_index_ready=False`` (post-drop) cannot answer "empty".

    Before this was fixed the scan returned zero rows and ended cleanly, which
    ``_chunk_source_is_populated`` reads as a confirmed-empty chunk source --
    the verdict that records an ``origin=empty`` embedding baseline for a
    container nobody could read.
    """
    storage, client = _storage([])
    storage._index_ready = False

    rows = []
    with pytest.raises(StorageControlPlaneError, match="not ready"):
        async for row in storage.iter_rows():
            rows.append(row)

    assert rows == []
    client.create_pit.assert_not_awaited()


async def test_an_index_that_vanishes_mid_scan_raises_and_marks_it_missing():
    """A live ``index_not_found`` is data loss, not emptiness.

    After ``initialize()`` the index always exists, so it disappearing under a
    scan (restore, concurrent drop) is indistinguishable from data loss -- the
    same argument ``get_by_id_strict`` makes for a point read.
    """
    storage, client = _storage([])
    client.create_pit = AsyncMock(
        side_effect=OpenSearchException("index_not_found_exception: iterws_config")
    )

    with pytest.raises(StorageControlPlaneError, match="unexpectedly missing"):
        async for _ in storage.iter_rows():
            pass

    assert storage._index_ready is False


async def test_the_pending_buffer_alone_is_not_yielded_as_the_listing():
    """A buffer is not the namespace, so it cannot stand in for one.

    ``_iter_raw_docs`` does not flush -- ``_refresh_for_search`` is
    best-effort and returns on a missing index -- so with the index gone the
    indexed side is unknown. The old code skipped the scan and yielded the
    buffered rows after it, a partial listing that ended cleanly; now the
    refusal precedes every row.
    """
    storage, _ = _storage([], pending={"buffered": {"value": 1}})
    storage._index_ready = False

    rows = []
    with pytest.raises(StorageControlPlaneError):
        async for row in storage.iter_rows():
            rows.append(row)

    assert rows == []


async def test_a_failed_refresh_raises_instead_of_scanning_a_stale_view():
    """The refresh before the PIT is what makes the frozen view complete.

    A row already durable but not yet in a searchable segment is missed by
    every page of the scan, so swallowing the refresh failure hands back a
    clean, empty listing over rows that exist. ``_chunk_source_is_populated``
    reads a clean end as CONFIRMED empty and the startup writes an
    ``origin=empty`` baseline from it, while the coverage refusal that should
    have fired (source populated, container empty) never does.
    """
    storage, client = _storage([])
    client.indices.refresh = AsyncMock(
        side_effect=OpenSearchException("refresh rejected: too many requests")
    )

    with pytest.raises(StorageControlPlaneError, match="not yet searchable"):
        async for _ in storage.iter_rows(page_size=10):
            pass

    # And it refused BEFORE freezing a view it could not vouch for.
    client.create_pit.assert_not_awaited()


async def test_a_refresh_failure_outside_a_scan_is_still_best_effort():
    """Only the scan asks for strictness. Every other search-based reader
    keeps the pre-refresh view it has always had -- turning those into errors
    would trade a stale listing for a broken one."""
    storage, client = _storage([])
    client.indices.refresh = AsyncMock(
        side_effect=OpenSearchException("refresh rejected: too many requests")
    )

    await storage._refresh_for_search()  # no raise
