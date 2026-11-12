"""Stateful fake reproduces batch commits; no PostgreSQL server is involved."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag.exceptions import DataMigrationError
from lightrag.kg.postgres_impl import PGVectorStorage, PostgreSQLDB

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class CommittedStore:
    def __init__(self):
        self.source = [
            dict(id=f"row-{i:04}", workspace="tenant", content_vector=[1.0, 0.0, 0.0])
            for i in range(750)
        ]
        self.target = {}
        self.batch_calls = 0
        self.acquisitions = 0
        self.fail_second = True

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        yield self

    @asynccontextmanager
    async def transaction(self, *, isolation):
        assert isolation == "repeatable_read"
        before = dict(self.target)
        try:
            yield
        except BaseException:
            self.target = before
            raise

    async def fetchval(self, sql, workspace):
        assert workspace == "tenant"
        return len(self.source) if "FROM legacy" in sql else len(self.target)

    async def prepare(self, sql):
        assert sql == "SELECT * FROM legacy LIMIT 0"
        return SimpleNamespace(
            get_attributes=lambda: [
                SimpleNamespace(name=name) for name in self.source[0]
            ]
        )

    async def execute(self, sql, workspace):
        assert 'INSERT INTO target ("id", "workspace", "content_vector")' in sql
        assert 'SELECT "id", "workspace", "content_vector" FROM legacy' in sql
        assert "WHERE workspace = $1" in sql
        self.batch_calls += 1
        for row in self.source:
            self.target.setdefault(row["id"], (workspace, row["content_vector"]))
            if self.fail_second and len(self.target) == 500:
                raise ValueError("injected copy failure")


def make_db(store):
    db = SimpleNamespace(
        vector_index_type=None,
        pool=store,
        _ensure_pool=AsyncMock(),
        connection_retry_backoff=0,
        connection_retry_backoff_max=0,
        connection_retry_attempts=1,
        _transient_exceptions=(),
        _before_sleep=AsyncMock(),
        check_table_exists=AsyncMock(return_value=True),
        _create_vector_index=AsyncMock(),
    )
    # Exercise the actual acquire/retry boundary instead of mocking it away.
    db._run_with_retry = PostgreSQLDB._run_with_retry.__get__(db)

    async def query(sql, params=None, multirows=False):
        if multirows:
            assert "FROM legacy" in sql
            rows = store.source
            if "id >" in sql:
                rows = [row for row in rows if row["id"] > params[1]]
            return rows[: params[-1]]
        assert "COUNT(*)" in sql
        return {
            "count": len(store.source) if "FROM legacy" in sql else len(store.target)
        }

    db.query = query
    return db


async def test_restart_must_not_accept_prior_committed_partial_batches():
    store = CommittedStore()
    args = dict(
        table_name="target",
        workspace="tenant",
        embedding_dim=3,
        legacy_table_name="legacy",
        base_table="LIGHTRAG_VDB_CHUNKS",
    )
    with pytest.raises(DataMigrationError):
        await PGVectorStorage.setup_table(db=make_db(store), **args)
    assert store.batch_calls == 1
    assert store.acquisitions == 1
    assert len(store.target) == 0
    assert len(store.source) == 750

    # A new DB facade shares only committed server state, not migration state.
    store.fail_second = False
    await PGVectorStorage.setup_table(db=make_db(store), **args)
    assert len(store.target) == 750, (
        "Restart accepted 500 committed rows as a complete 750-row migration"
    )


async def test_transaction_verification_failure_rolls_back(monkeypatch):
    store = CommittedStore()
    store.fail_second = False
    fetchval = store.fetchval

    async def wrong_count(sql, workspace):
        count = await fetchval(sql, workspace)
        return count - 1 if "FROM target" in sql and count else count

    monkeypatch.setattr(store, "fetchval", wrong_count)
    with pytest.raises(DataMigrationError, match="verification failed"):
        await PGVectorStorage._pg_migrate_workspace_data(
            make_db(store), "legacy", "target", "tenant", 750, 3
        )
    assert store.target == {}


async def test_populated_target_is_preserved_without_copy():
    store = CommittedStore()
    store.target["custom"] = ("tenant", [9.0, 8.0, 7.0])
    result = await PGVectorStorage._pg_migrate_workspace_data(
        make_db(store), "legacy", "target", "tenant", 750, 3
    )
    assert result is None
    assert store.batch_calls == 0
    assert store.target == {"custom": ("tenant", [9.0, 8.0, 7.0])}


async def test_cancellation_before_commit_rolls_back(monkeypatch):
    import asyncio

    store = CommittedStore()
    store.fail_second = False
    execute = store.execute

    async def cancelled(*args):
        await execute(*args)
        raise asyncio.CancelledError()

    monkeypatch.setattr(store, "execute", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await PGVectorStorage._pg_migrate_workspace_data(
            make_db(store), "legacy", "target", "tenant", 750, 3
        )
    assert store.target == {}


async def test_lost_commit_response_retry_does_not_recopy(monkeypatch):
    store = CommittedStore()
    store.fail_second = False
    attempts = 0

    @asynccontextmanager
    async def transaction(*, isolation):
        nonlocal attempts
        attempts += 1
        yield
        if attempts == 1:
            raise ConnectionError("COMMIT applied but response lost")

    monkeypatch.setattr(store, "transaction", transaction)
    db = make_db(store)
    db._transient_exceptions = (ConnectionError,)
    db.connection_retry_attempts = 2
    result = await PGVectorStorage._pg_migrate_workspace_data(
        db, "legacy", "target", "tenant", 750, 3
    )
    assert result is None
    assert len(store.target) == 750
    assert store.batch_calls == 1
    assert attempts == 2


async def test_helper_rejects_empty_workspace_without_acquiring_connection():
    store = CommittedStore()
    with pytest.raises(ValueError, match="workspace must be provided"):
        await PGVectorStorage._pg_migrate_workspace_data(
            make_db(store), "legacy", "target", "", 750, 3
        )
    assert store.acquisitions == 0


@pytest.mark.parametrize("bad", [None, -1, True, "0"])
@pytest.mark.parametrize("bad_read", [1, 2, 3])
async def test_invalid_count_is_not_treated_as_empty(monkeypatch, bad, bad_read):
    store = CommittedStore()
    store.fail_second = False
    fetchval = store.fetchval
    reads = 0

    async def invalid(*args):
        nonlocal reads
        reads += 1
        return bad if reads == bad_read else await fetchval(*args)

    monkeypatch.setattr(store, "fetchval", invalid)
    with pytest.raises(DataMigrationError, match="Invalid migration count"):
        await PGVectorStorage._pg_migrate_workspace_data(
            make_db(store), "legacy", "target", "tenant", 750, 3
        )
    assert store.target == {}


async def test_explicit_columns_preserve_metadata_order_and_quote_names(monkeypatch):
    store = CommittedStore()
    store.fail_second = False
    names = ["content_vector", "workspace", "id", 'odd"column']

    async def prepare(sql):
        return SimpleNamespace(
            get_attributes=lambda: [SimpleNamespace(name=name) for name in names]
        )

    async def execute(sql, workspace):
        expected = '"content_vector", "workspace", "id", "odd""column"'
        assert f"INSERT INTO target ({expected}) SELECT {expected} FROM legacy" in sql
        assert "WHERE workspace = $1 ON CONFLICT (workspace, id) DO NOTHING" in sql
        assert workspace == "tenant"
        store.target.update(
            {row["id"]: (workspace, row["content_vector"]) for row in store.source}
        )

    monkeypatch.setattr(store, "prepare", prepare)
    monkeypatch.setattr(store, "execute", execute)
    assert (
        await PGVectorStorage._pg_migrate_workspace_data(
            make_db(store), "legacy", "target", "tenant", 750, 3
        )
        == 750
    )
