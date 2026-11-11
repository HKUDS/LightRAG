from collections import Counter

import pytest

from tests.kg.hologres_impl.conftest import (
    HologresLiveCleanupError,
    _cleanup_live_schema_and_close,
)


SCHEMA = "lightrag_test_cleanup123"
CATALOG_DESCRIPTORS = {
    "live.schema.cleanup.enumerate.foreign_keys",
    "live.schema.cleanup.enumerate.relations",
    "live.schema.cleanup.enumerate.routines",
    "live.schema.cleanup.enumerate.types",
}


class CleanupClient:
    def __init__(
        self,
        *,
        catalog_rows=None,
        drop_failures=None,
        close_error=None,
    ):
        self.catalog_rows = catalog_rows or {}
        self.drop_failures = dict(drop_failures or {})
        self.close_error = close_error
        self.fetch_calls = []
        self.drop_calls = []
        self.drop_attempts = Counter()
        self.closed = False

    async def fetch_all(self, sql, *values, descriptor):
        self.fetch_calls.append((sql, values, descriptor))
        return self.catalog_rows.get(descriptor, ())

    async def execute_one(self, sql, *, descriptor, replay_safe):
        self.drop_calls.append((sql, descriptor, replay_safe))
        self.drop_attempts[sql] += 1
        for marker, remaining_failures in self.drop_failures.items():
            if marker not in sql:
                continue
            if remaining_failures is None:
                raise RuntimeError(
                    "server cleanup failed; password=supersecret; SQL=" + sql
                )
            if remaining_failures > 0:
                self.drop_failures[marker] = remaining_failures - 1
                raise RuntimeError(
                    "server cleanup failed; password=supersecret; SQL=" + sql
                )

    async def close(self):
        self.closed = True
        if self.close_error is not None:
            raise RuntimeError(self.close_error)


def _drop_sql(client):
    return [sql for sql, _descriptor, _replay_safe in client.drop_calls]


async def test_live_schema_cleanup_covers_catalogs_and_scopes_every_drop():
    client = CleanupClient(
        catalog_rows={
            "live.schema.cleanup.enumerate.foreign_keys": [
                {
                    "table_name": "event_children",
                    "constraint_name": "event_children_parent_fk",
                }
            ],
            "live.schema.cleanup.enumerate.relations": [
                {"relname": "events_view", "relkind": "v"},
                {"relname": "events_rollup", "relkind": "m"},
                {"relname": "foreign_events", "relkind": "f"},
                {"relname": "events_partitioned", "relkind": "p"},
                {"relname": "events", "relkind": "r"},
                {"relname": "events_id_seq", "relkind": "S"},
                {"relname": "events_idx", "relkind": "i"},
                {"relname": "event_payload", "relkind": "c"},
            ],
            "live.schema.cleanup.enumerate.routines": [
                {
                    "proname": "refresh_events",
                    "prokind": "f",
                    "identity_args": "integer, text",
                },
                {
                    "proname": "summarize_events",
                    "prokind": "f",
                    "identity_args": "",
                },
                {
                    "proname": "archive_events",
                    "prokind": "p",
                    "identity_args": "",
                },
                {
                    "proname": "event_count",
                    "prokind": "a",
                    "identity_args": "",
                },
                {
                    "proname": "event_total",
                    "prokind": "a",
                    "identity_args": "integer",
                },
            ],
            "live.schema.cleanup.enumerate.types": [
                {"typname": "event_status", "typtype": "e"},
                {"typname": "event_code", "typtype": "d"},
                {"typname": "event_window", "typtype": "r"},
            ],
        }
    )

    await _cleanup_live_schema_and_close(client, SCHEMA)

    assert {descriptor for _sql, _values, descriptor in client.fetch_calls} == (
        CATALOG_DESCRIPTORS
    )
    assert len(client.fetch_calls) == 4
    for sql, values, _descriptor in client.fetch_calls:
        assert "$1" in sql
        assert values == (SCHEMA,)
        assert SCHEMA not in sql

    catalog_sql = {
        descriptor: sql for sql, _values, descriptor in client.fetch_calls
    }
    foreign_key_sql = catalog_sql["live.schema.cleanup.enumerate.foreign_keys"]
    assert "pg_catalog.pg_constraint" in foreign_key_sql
    assert "contype = 'f'" in foreign_key_sql
    assert foreign_key_sql.count("nspname = $1") >= 2

    assert "pg_catalog.pg_class" in catalog_sql[
        "live.schema.cleanup.enumerate.relations"
    ]

    routine_sql = catalog_sql["live.schema.cleanup.enumerate.routines"]
    assert "pg_catalog.pg_proc" in routine_sql
    assert "pg_catalog.pg_get_function_identity_arguments" in routine_sql

    type_sql = catalog_sql["live.schema.cleanup.enumerate.types"]
    assert "pg_catalog.pg_type" in type_sql
    assert "t.typtype IN ('e', 'd', 'r')" in type_sql
    assert "t.typrelid = 0" in type_sql
    assert "t.typelem" in type_sql
    assert "t.typarray" in type_sql

    assert _drop_sql(client) == [
        f'ALTER TABLE IF EXISTS "{SCHEMA}"."event_children" '
        'DROP CONSTRAINT IF EXISTS "event_children_parent_fk"',
        f'DROP VIEW IF EXISTS "{SCHEMA}"."events_view"',
        f'DROP MATERIALIZED VIEW IF EXISTS "{SCHEMA}"."events_rollup"',
        f'DROP FOREIGN TABLE IF EXISTS "{SCHEMA}"."foreign_events"',
        f'DROP TABLE IF EXISTS "{SCHEMA}"."events_partitioned"',
        f'DROP TABLE IF EXISTS "{SCHEMA}"."events"',
        f'DROP SEQUENCE IF EXISTS "{SCHEMA}"."events_id_seq"',
        f'DROP INDEX IF EXISTS "{SCHEMA}"."events_idx"',
        f'DROP TYPE IF EXISTS "{SCHEMA}"."event_payload"',
        f'DROP FUNCTION IF EXISTS "{SCHEMA}"."refresh_events"(integer, text)',
        f'DROP FUNCTION IF EXISTS "{SCHEMA}"."summarize_events"()',
        f'DROP PROCEDURE IF EXISTS "{SCHEMA}"."archive_events"()',
        f'DROP AGGREGATE IF EXISTS "{SCHEMA}"."event_count"(*)',
        f'DROP AGGREGATE IF EXISTS "{SCHEMA}"."event_total"(integer)',
        f'DROP TYPE IF EXISTS "{SCHEMA}"."event_status"',
        f'DROP DOMAIN IF EXISTS "{SCHEMA}"."event_code"',
        f'DROP TYPE IF EXISTS "{SCHEMA}"."event_window"',
        f'DROP SCHEMA IF EXISTS "{SCHEMA}"',
    ]
    for sql, _descriptor, replay_safe in client.drop_calls:
        assert "CASCADE" not in sql
        assert ";" not in sql
        assert f'"{SCHEMA}"' in sql
        assert replay_safe is True
    assert client.closed is True


async def test_live_schema_cleanup_retries_dependency_failure_until_success():
    blocked_view_sql = f'DROP VIEW IF EXISTS "{SCHEMA}"."blocked_view"'
    materialized_view_sql = (
        f'DROP MATERIALIZED VIEW IF EXISTS "{SCHEMA}"."dependent_rollup"'
    )
    client = CleanupClient(
        catalog_rows={
            "live.schema.cleanup.enumerate.relations": [
                {"relname": "blocked_view", "relkind": "v"},
                {"relname": "dependent_rollup", "relkind": "m"},
            ]
        },
        drop_failures={blocked_view_sql: 1},
    )

    await _cleanup_live_schema_and_close(client, SCHEMA)

    dropped_sql = _drop_sql(client)
    assert dropped_sql[:2] == [blocked_view_sql, materialized_view_sql]
    assert client.drop_attempts[blocked_view_sql] == 2
    assert dropped_sql[-1] == f'DROP SCHEMA IF EXISTS "{SCHEMA}"'
    assert client.closed is True


async def test_live_schema_cleanup_stops_after_no_progress_and_finishes_teardown():
    long_name = "blocked_" + "x" * 55
    blocked_table_sql = f'DROP TABLE IF EXISTS "{SCHEMA}"."{long_name}"'
    client = CleanupClient(
        catalog_rows={
            "live.schema.cleanup.enumerate.relations": [
                {"relname": long_name, "relkind": "r"}
            ]
        },
        drop_failures={blocked_table_sql: None},
    )

    with pytest.raises(HologresLiveCleanupError) as exc_info:
        await _cleanup_live_schema_and_close(client, SCHEMA)

    assert client.drop_attempts[blocked_table_sql] == 1
    assert _drop_sql(client)[-1] == f'DROP SCHEMA IF EXISTS "{SCHEMA}"'
    assert client.closed is True
    failure_stage = exc_info.value.failures[0][0]
    assert failure_stage.startswith("relation drop [TABLE:")
    assert len(failure_stage) <= 96
    assert "password=supersecret" not in str(exc_info.value)
    assert blocked_table_sql not in str(exc_info.value)
    assert "supersecret" not in repr(exc_info.value.failures)
    assert blocked_table_sql not in repr(exc_info.value.failures)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


async def test_live_schema_cleanup_surfaces_schema_drop_only_failure_and_closes():
    client = CleanupClient(
        drop_failures={f'DROP SCHEMA IF EXISTS "{SCHEMA}"': None}
    )

    with pytest.raises(HologresLiveCleanupError, match="schema drop") as exc_info:
        await _cleanup_live_schema_and_close(client, SCHEMA)

    assert [stage for stage, _error in exc_info.value.failures] == ["schema drop"]
    assert client.closed is True


async def test_live_schema_cleanup_surfaces_close_only_failure_without_secrets():
    client = CleanupClient(close_error="password=close-secret")

    with pytest.raises(HologresLiveCleanupError, match="client close") as exc_info:
        await _cleanup_live_schema_and_close(client, SCHEMA)

    assert [stage for stage, _error in exc_info.value.failures] == ["client close"]
    assert "close-secret" not in str(exc_info.value)
    assert "close-secret" not in repr(exc_info.value.failures)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert client.closed is True


async def test_live_schema_cleanup_combines_object_and_close_failures():
    blocked_table_sql = f'DROP TABLE IF EXISTS "{SCHEMA}"."blocked_table"'
    client = CleanupClient(
        catalog_rows={
            "live.schema.cleanup.enumerate.relations": [
                {"relname": "blocked_table", "relkind": "r"}
            ]
        },
        drop_failures={blocked_table_sql: None},
        close_error="password=close-secret",
    )

    with pytest.raises(HologresLiveCleanupError) as exc_info:
        await _cleanup_live_schema_and_close(client, SCHEMA)

    stages = [stage for stage, _error in exc_info.value.failures]
    assert stages[0].startswith("relation drop [TABLE:blocked_table]")
    assert stages[-1] == "client close"
    assert "supersecret" not in str(exc_info.value)
    assert "close-secret" not in str(exc_info.value)
    assert "supersecret" not in repr(exc_info.value.failures)
    assert "close-secret" not in repr(exc_info.value.failures)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert client.closed is True


async def test_live_schema_cleanup_rejects_non_test_schema_but_still_closes_client():
    client = CleanupClient()

    with pytest.raises(ValueError, match="unique lightrag_test"):
        await _cleanup_live_schema_and_close(client, "public")

    assert client.fetch_calls == []
    assert client.drop_calls == []
    assert client.closed is True
