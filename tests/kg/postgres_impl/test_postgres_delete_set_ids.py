"""Regression test for Issue #3286.

``adelete_by_doc_id`` / ``_purge_doc_chunks_and_kg`` historically passed
``chunk_ids`` as a ``set`` to ``PGKVStorage.delete`` / ``PGDocStatusStorage.delete``.
After delete-batching was introduced, the chunked path slices the id collection
(``ids[i : i + chunk]``); slicing a ``set`` raises ``'set' object is not
subscriptable``, which the broad ``except Exception`` in ``delete`` then swallows
as a log line -- so the records were silently NOT deleted.

These tests feed a ``set`` straight into ``delete`` with a positive batch cap
(forcing the chunked path) and assert that every id is actually handed to the
SQL layer. Asserting "no exception" alone is insufficient because the bug is
swallowed; we must assert the records really got deleted.
"""

import pytest

pytest.importorskip("asyncpg")

from lightrag.kg.postgres_impl import (  # noqa: E402
    PGDocStatusStorage,
    PGKVStorage,
)
from lightrag.namespace import NameSpace  # noqa: E402


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingConnection:
    """Captures the id-array argument of each ``execute`` call."""

    def __init__(self):
        self.deleted_id_batches: list[list[str]] = []

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, workspace, id_slice):
        # The chunked path slices ids; a slice of a list is a list. Record it.
        assert isinstance(id_slice, list)
        self.deleted_id_batches.append(id_slice)


class _FakeDB:
    def __init__(self, connection):
        self._connection = connection

    async def _run_with_retry(self, operation):
        # Mirror the real helper: invoke the closure against a live connection.
        return await operation(self._connection)


def _make_storage(cls, namespace, batch_cap):
    """Build a storage instance exercising only the attributes ``delete`` reads."""
    storage = object.__new__(cls)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage._max_delete_records_per_batch = batch_cap
    connection = _RecordingConnection()
    storage.db = _FakeDB(connection)
    return storage, connection


@pytest.mark.asyncio
async def test_pgkv_delete_accepts_set_and_chunks_all_ids():
    ids = {f"chunk-{i}" for i in range(5)}
    # Positive cap < len(ids) forces the chunked slicing path that broke on sets.
    storage, connection = _make_storage(
        PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS, batch_cap=2
    )

    await storage.delete(ids)

    flattened = [cid for batch in connection.deleted_id_batches for cid in batch]
    assert set(flattened) == ids
    assert len(flattened) == len(ids)  # no duplicates / drops
    assert all(len(batch) <= 2 for batch in connection.deleted_id_batches)


@pytest.mark.asyncio
async def test_pgdocstatus_delete_accepts_set_and_chunks_all_ids():
    ids = {f"doc-{i}" for i in range(5)}
    storage, connection = _make_storage(
        PGDocStatusStorage, NameSpace.DOC_STATUS, batch_cap=2
    )

    await storage.delete(ids)

    flattened = [did for batch in connection.deleted_id_batches for did in batch]
    assert set(flattened) == ids
    assert len(flattened) == len(ids)
    assert all(len(batch) <= 2 for batch in connection.deleted_id_batches)
