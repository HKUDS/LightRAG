"""Regression test: PGKVStorage.delete / PGDocStatusStorage.delete must not
swallow a failed SQL DELETE as a log line.

Both methods wrap the batched DELETE in a bare ``except Exception: log(...)``
with no ``raise``, so any SQL failure (constraint violation, a transient
error that exhausts ``_run_with_retry``'s attempts, anything) returns to the
caller exactly as a successful delete would -- the two are indistinguishable
from the outside.

This defeats callers that rely on the delete failing loudly. For example
lightrag.py's "create" rollback path calls ``self.full_docs.delete([doc_id])``
/ ``self.doc_status.delete([doc_id])`` and treats a clean return as proof the
rollback happened, explicitly to avoid "reporting a rollback as successful
while the FAILED journal row can reappear after a crash" -- a false success
that a swallowed exception here reproduces on the PostgreSQL backend.
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


class _FailingConnection:
    """Mimics a connection whose DELETE statement fails server side."""

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, workspace, id_slice):
        raise RuntimeError("simulated DELETE failure")


class _FakeDB:
    def __init__(self, connection):
        self._connection = connection

    async def _run_with_retry(self, operation):
        return await operation(self._connection)


def _make_storage(cls, namespace):
    """Build a storage instance exercising only the attributes ``delete`` reads."""
    storage = object.__new__(cls)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage._max_delete_records_per_batch = 10
    storage.db = _FakeDB(_FailingConnection())
    return storage


@pytest.mark.asyncio
async def test_pgkv_delete_raises_on_sql_failure():
    storage = _make_storage(PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS)

    with pytest.raises(RuntimeError, match="simulated DELETE failure"):
        await storage.delete(["chunk-1"])


@pytest.mark.asyncio
async def test_pgdocstatus_delete_raises_on_sql_failure():
    storage = _make_storage(PGDocStatusStorage, NameSpace.DOC_STATUS)

    with pytest.raises(RuntimeError, match="simulated DELETE failure"):
        await storage.delete(["doc-1"])
