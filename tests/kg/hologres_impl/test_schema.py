import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re

import pytest

from lightrag.kg.hologres.client import (
    HologresClient,
    OperationKind,
)
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.schema import (
    LEDGER_TABLE_NAME,
    HologresOfflineMigrationRequired,
    HologresSchemaBusyError,
    HologresSchemaDefinitionError,
    HologresSchemaDriftError,
    HologresSchemaError,
    HologresSchemaManager,
    HologresSchemaStateError,
    SchemaDescriptor,
    SchemaState,
    bootstrap_descriptors,
    claim_statement,
    inspect_statement,
    load_statement,
    transition_statement,
)


SCHEMA = "lightrag_probe"
PROBE_TABLE = "lightrag_probe_events"
EPOCH = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)

TABLE_EXISTS_SQL = (
    "SELECT EXISTS ("
    "SELECT 1 FROM pg_catalog.pg_class c "
    "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
    "WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'r')"
)


def make_descriptor(
    *,
    name="probe_events",
    component="probe",
    version=1,
    step=1,
    table=PROBE_TABLE,
    replay_safe=True,
    precondition_sql=None,
    precondition_args=(),
    postcondition_sql=TABLE_EXISTS_SQL,
    postcondition_args=None,
    sql=None,
):
    return SchemaDescriptor(
        name=name,
        component=component,
        version=version,
        step=step,
        sql=sql
        or (
            f'CREATE TABLE IF NOT EXISTS "{SCHEMA}"."{table}" '
            "(id text NOT NULL, payload text, PRIMARY KEY (id))"
        ),
        postcondition_sql=postcondition_sql,
        postcondition_args=(
            (SCHEMA, table) if postcondition_args is None else postcondition_args
        ),
        precondition_sql=precondition_sql,
        precondition_args=precondition_args,
        replay_safe=replay_safe,
    )


class LedgerClient:
    """In-memory double exposing only the restricted Hologres client API.

    The double interprets calls by operation descriptor and relies on the
    documented positional argument layout of the schema module's statements:

    * ``schema.ledger.claim``
      ``(component, version, step, name, digest, owner, lease, now)``
    * ``schema.ledger.transition``
      ``(state, error_summary, owner, lease, now, component, version, step,
      name, digest, expected_owner, expected_state)``
    * ``schema.ledger.inspect`` ``(component, version, step, name)``
    * ``schema.ledger.load`` ``(components,)``
    """

    def __init__(
        self,
        *,
        descriptors=(),
        rows=(),
        catalog=(),
        ddl_errors=None,
        postcondition_results=None,
        precondition_results=None,
        clock=None,
    ):
        self.calls = []
        self.statement_calls = []
        self.executed = []
        self.rows = {}
        self.catalog = {tuple(entry) for entry in catalog}
        self.clock = clock
        self._postcondition_results = dict(postcondition_results or {})
        self._precondition_results = dict(precondition_results or {})
        self._ddl_errors = dict(ddl_errors or {})
        self._effects = {}
        for descriptor in (*bootstrap_descriptors(SCHEMA), *descriptors):
            self._effects[descriptor.sql] = tuple(descriptor.postcondition_args)
        for row in rows:
            self.rows[
                (
                    row["component"],
                    row["version"],
                    row["step"],
                    row["descriptor_name"],
                )
            ] = dict(row)

    def _record(self, descriptor, kind, replay_safe, sql, values):
        self.calls.append((descriptor, kind, replay_safe))
        self.statement_calls.append((descriptor, sql, values))

    @property
    def descriptors(self):
        return tuple(descriptor for descriptor, _kind, _replay in self.calls)

    async def execute_one(
        self, sql, *values, descriptor, replay_safe=False, timeout=None
    ):
        await asyncio.sleep(0)
        self._record(descriptor, OperationKind.WRITE, replay_safe, sql, values)
        self.executed.append((descriptor, sql, values))
        error = self._ddl_errors.get(sql)
        if error is not None:
            raise error
        effect = self._effects.get(sql)
        if effect is not None:
            self.catalog.add(effect)
        return "CREATE TABLE"

    async def fetch_value(
        self,
        sql,
        *values,
        descriptor,
        operation_kind=OperationKind.READ,
        replay_safe=None,
        timeout=None,
    ):
        await asyncio.sleep(0)
        self._record(descriptor, operation_kind, replay_safe, sql, values)
        if descriptor == "schema.ledger.transition":
            return self._transition(values)
        if descriptor == "schema.descriptor.precondition":
            if tuple(values) in self._precondition_results:
                return self._precondition_results[tuple(values)]
            return tuple(values) in self.catalog
        if descriptor in {
            "schema.bootstrap.verify",
            "schema.descriptor.postcondition",
        }:
            if tuple(values) in self._postcondition_results:
                return self._postcondition_results[tuple(values)]
            return tuple(values) in self.catalog
        raise AssertionError(f"Unexpected fetch_value descriptor: {descriptor}")

    async def fetch_one(
        self,
        sql,
        *values,
        descriptor,
        operation_kind=OperationKind.READ,
        replay_safe=None,
        timeout=None,
    ):
        await asyncio.sleep(0)
        self._record(descriptor, operation_kind, replay_safe, sql, values)
        if descriptor == "schema.ledger.claim":
            return self._claim(values)
        if descriptor == "schema.ledger.inspect":
            row = self.rows.get(tuple(values[:4]))
            return None if row is None else dict(row)
        raise AssertionError(f"Unexpected fetch_one descriptor: {descriptor}")

    async def fetch_all(
        self,
        sql,
        *values,
        descriptor,
        operation_kind=OperationKind.READ,
        replay_safe=None,
        timeout=None,
    ):
        await asyncio.sleep(0)
        self._record(descriptor, operation_kind, replay_safe, sql, values)
        if descriptor != "schema.ledger.load":
            raise AssertionError(f"Unexpected fetch_all descriptor: {descriptor}")
        components = set(values[0])
        return [
            dict(row)
            for key, row in sorted(self.rows.items())
            if key[0] in components
        ]

    def _claim(self, values):
        identity = tuple(values[:4])
        digest, owner, lease, now = values[4], values[5], values[6], values[7]
        row = self.rows.get(identity)
        if row is None:
            self.rows[identity] = {
                "component": identity[0],
                "version": identity[1],
                "step": identity[2],
                "descriptor_name": identity[3],
                "digest": digest,
                "state": SchemaState.PREPARED.value,
                "previous_state": None,
                "owner_token": owner,
                "lease_expires_at": lease,
                "error_summary": None,
                "created_at": now,
                "updated_at": now,
            }
            return dict(self.rows[identity])
        if row["digest"] != digest:
            return None
        if row["state"] == SchemaState.APPLIED.value:
            return None
        holder = row["owner_token"]
        held = row["lease_expires_at"]
        if (
            holder is not None
            and holder != owner
            and held is not None
            and held > now
        ):
            return None
        if row["state"] != SchemaState.PREPARED.value:
            # Mirrors the CASE expression in claim_statement: the marker is
            # sticky so a second consecutive crash cannot erase the evidence
            # that the statement was already attempted once.
            row["previous_state"] = row["state"]
        row["state"] = SchemaState.PREPARED.value
        row["owner_token"] = owner
        row["lease_expires_at"] = lease
        row["error_summary"] = None
        row["updated_at"] = now
        return dict(row)

    def _transition(self, values):
        (
            state,
            error_summary,
            owner,
            lease,
            now,
            component,
            version,
            step,
            name,
            digest,
            expected_owner,
            expected_state,
        ) = values
        row = self.rows.get((component, version, step, name))
        if row is None:
            return None
        if row["digest"] != digest:
            return None
        if row["owner_token"] != expected_owner:
            return None
        if row["state"] != expected_state:
            return None
        if row["lease_expires_at"] is None or row["lease_expires_at"] <= now:
            return None
        row["previous_state"] = expected_state
        row["state"] = state
        row["error_summary"] = error_summary
        row["owner_token"] = owner
        row["lease_expires_at"] = lease
        row["updated_at"] = now
        return 1


def make_manager(client, **kwargs):
    kwargs.setdefault("owner_token", "owner-a")
    kwargs.setdefault("schema", SCHEMA)
    kwargs.setdefault("now_provider", lambda: EPOCH)
    kwargs.setdefault("sleep", _no_sleep)
    return HologresSchemaManager(client, **kwargs)


async def _no_sleep(_delay):
    await asyncio.sleep(0)


def ledger_row(descriptor, **overrides):
    row = {
        "component": descriptor.component,
        "version": descriptor.version,
        "step": descriptor.step,
        "descriptor_name": descriptor.name,
        "digest": descriptor.digest,
        "state": SchemaState.PREPARED.value,
        "previous_state": None,
        "owner_token": None,
        "lease_expires_at": None,
        "error_summary": None,
        "created_at": EPOCH,
        "updated_at": EPOCH,
    }
    row.update(overrides)
    return row


def bootstrapped_catalog():
    return {(SCHEMA,), (SCHEMA, LEDGER_TABLE_NAME)}


# --------------------------------------------------------------------------
# descriptor identity, digest, and construction-time validation
# --------------------------------------------------------------------------


def test_descriptor_digest_is_stable_and_identity_sensitive():
    first = make_descriptor()
    second = make_descriptor()

    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.digest != make_descriptor(version=2).digest
    assert first.digest != make_descriptor(step=2).digest
    assert first.digest != make_descriptor(name="other_events").digest
    assert first.digest != make_descriptor(component="other").digest
    assert first.digest != make_descriptor(table="lightrag_probe_other").digest


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Probe_Events"},
        {"name": "1probe"},
        {"name": ""},
        {"component": "Probe"},
        {"component": "probe-events"},
        {"version": 0},
        {"version": -1},
        {"step": 0},
        {"version": True},
        {"step": "1"},
    ],
)
def test_descriptor_rejects_invalid_identity_fields(overrides):
    with pytest.raises(HologresSchemaDefinitionError):
        make_descriptor(**overrides)


@pytest.mark.parametrize(
    "sql",
    [
        'CREATE TABLE IF NOT EXISTS "s"."a" (id text); '
        'CREATE TABLE IF NOT EXISTS "s"."b" (id text)',
        "BEGIN",
        "COMMIT",
        "START TRANSACTION",
        "ROLLBACK",
        "SAVEPOINT probe",
        "CALL set_table_property('s.a', 'orientation', 'row')",
        "DO $$ SELECT 1 $$",
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
    ],
)
def test_descriptor_rejects_scripts_and_transaction_control(sql):
    with pytest.raises(HologresSchemaDefinitionError):
        make_descriptor(sql=sql)


@pytest.mark.parametrize(
    "sql",
    [
        'DROP TABLE "s"."a"',
        'DROP TABLE IF EXISTS "s"."a"',
        'DROP SCHEMA IF EXISTS "s"',
        'TRUNCATE TABLE "s"."a"',
        'DELETE FROM "s"."a"',
        'UPDATE "s"."a" SET id = \'x\'',
        'ALTER TABLE "s"."a" DROP COLUMN "id"',
        'ALTER TABLE "s"."a" RENAME COLUMN "id" TO "key"',
        'ALTER TABLE "s"."a" RENAME TO "b"',
        'ALTER TABLE "s"."a" ALTER COLUMN "id" TYPE integer',
        'CREATE TABLE IF NOT EXISTS "s"."b" AS SELECT * FROM "s"."a"',
        'INSERT INTO "s"."b" SELECT * FROM "s"."a"',
        'WITH moved AS (DELETE FROM "s"."a" RETURNING *) '
        'INSERT INTO "s"."b" SELECT * FROM moved',
        'CREATE TABLE "s"."a" (id text)',
        'CREATE OR REPLACE VIEW "s"."v" AS SELECT 1',
        'GRANT SELECT ON "s"."a" TO PUBLIC',
        "SELECT 1",
    ],
)
def test_descriptor_rejects_destructive_and_rebuild_shapes(sql):
    with pytest.raises(HologresOfflineMigrationRequired) as exc_info:
        make_descriptor(sql=sql)

    assert "offline migration" in str(exc_info.value).lower()
    assert issubclass(HologresOfflineMigrationRequired, HologresSchemaDefinitionError)


@pytest.mark.parametrize(
    "sql",
    [
        'CREATE SCHEMA IF NOT EXISTS "lightrag_probe"',
        'CREATE TABLE IF NOT EXISTS "s"."a" (id text NOT NULL, PRIMARY KEY (id))',
        'CREATE INDEX IF NOT EXISTS "a_idx" ON "s"."a" ("id")',
        'CREATE UNIQUE INDEX IF NOT EXISTS "a_uidx" ON "s"."a" ("id")',
        'ALTER TABLE "s"."a" ADD COLUMN IF NOT EXISTS "payload" text',
        'COMMENT ON TABLE "s"."a" IS \'probe\'',
    ],
)
def test_descriptor_accepts_additive_shapes(sql):
    descriptor = make_descriptor(sql=sql)

    assert descriptor.sql == sql


@pytest.mark.parametrize(
    "postcondition_sql",
    [
        'DELETE FROM "s"."a" RETURNING 1',
        'INSERT INTO "s"."a"(id) VALUES ($1) RETURNING 1',
        'UPDATE "s"."a" SET id = $1 RETURNING 1',
    ],
)
def test_descriptor_requires_a_read_only_postcondition(postcondition_sql):
    with pytest.raises(HologresSchemaDefinitionError):
        make_descriptor(
            postcondition_sql=postcondition_sql, postcondition_args=("x",)
        )


def test_descriptor_requires_a_read_only_precondition():
    with pytest.raises(HologresSchemaDefinitionError):
        make_descriptor(
            precondition_sql='DELETE FROM "s"."a" RETURNING 1',
            precondition_args=(),
        )


def test_descriptor_requires_an_explicit_boolean_replay_decision():
    with pytest.raises(HologresSchemaDefinitionError):
        make_descriptor(replay_safe="yes")


# --------------------------------------------------------------------------
# validation happens before any pool acquisition
# --------------------------------------------------------------------------


class GuardConnection:
    async def execute(self, *_args, **_kwargs):
        raise AssertionError("Schema validation must precede pool acquisition")

    async def fetchrow(self, *_args, **_kwargs):
        raise AssertionError("Schema validation must precede pool acquisition")

    async def fetch(self, *_args, **_kwargs):
        raise AssertionError("Schema validation must precede pool acquisition")

    async def fetchval(self, *_args, **_kwargs):
        raise AssertionError("Schema validation must precede pool acquisition")


class CountingPool:
    def __init__(self):
        self.acquire_count = 0

    @asynccontextmanager
    async def acquire(self, *, timeout=None):
        self.acquire_count += 1
        yield GuardConnection()

    async def close(self):
        return None


def guard_client():
    config = HologresConfig.from_env(
        {
            "HOLOGRES_HOST": "example.hologres.aliyuncs.com",
            "HOLOGRES_PORT": "80",
            "HOLOGRES_USER": "test_user",
            "HOLOGRES_PASSWORD": "secret",
            "HOLOGRES_DATABASE": "analytics",
            "HOLOGRES_SCHEMA": SCHEMA,
        }
    )
    pool = CountingPool()
    return HologresClient(config, pool=pool), pool


@pytest.mark.parametrize(
    "sql",
    [
        'DROP TABLE "s"."a"',
        'CREATE TABLE IF NOT EXISTS "s"."a" (id text); SELECT 1',
        "BEGIN",
    ],
)
async def test_rejected_descriptors_never_acquire_a_connection(sql):
    client, pool = guard_client()
    manager = HologresSchemaManager(
        client, schema=SCHEMA, owner_token="owner-a", now_provider=lambda: EPOCH
    )

    with pytest.raises(HologresSchemaDefinitionError):
        await manager.initialize([make_descriptor(sql=sql)])

    assert pool.acquire_count == 0


async def test_duplicate_identities_are_refused_before_any_connection():
    client, pool = guard_client()
    manager = HologresSchemaManager(
        client, schema=SCHEMA, owner_token="owner-a", now_provider=lambda: EPOCH
    )
    descriptor = make_descriptor()

    with pytest.raises(HologresSchemaDefinitionError, match="duplicate"):
        await manager.initialize([descriptor, descriptor])

    assert pool.acquire_count == 0


def test_manager_rejects_invalid_schema_identifiers():
    client, _pool = guard_client()

    with pytest.raises(HologresSchemaDefinitionError):
        HologresSchemaManager(client, schema='bad"schema')


# --------------------------------------------------------------------------
# ledger control table and bootstrap
# --------------------------------------------------------------------------


def test_ledger_descriptor_is_hybrid_oriented_shared_and_unpartitioned():
    schema_descriptor, ledger_descriptor = bootstrap_descriptors(SCHEMA)

    assert 'CREATE SCHEMA IF NOT EXISTS "lightrag_probe"' in schema_descriptor.sql
    assert (
        f'"{SCHEMA}"."{LEDGER_TABLE_NAME}"' in ledger_descriptor.sql
    )
    assert LEDGER_TABLE_NAME == "lightrag_hologres_schema_ledger"
    assert "CREATE TABLE IF NOT EXISTS" in ledger_descriptor.sql
    assert "orientation = 'row,column'" in ledger_descriptor.sql
    assert "PARTITION BY" not in ledger_descriptor.sql.upper()
    assert "workspace" not in ledger_descriptor.sql
    for column in (
        "component",
        "version",
        "step",
        "descriptor_name",
        "digest",
        "state",
        "previous_state",
        "owner_token",
        "lease_expires_at",
        "error_summary",
        "created_at",
        "updated_at",
    ):
        assert column in ledger_descriptor.sql
    assert schema_descriptor.replay_safe is True
    assert ledger_descriptor.replay_safe is True


async def test_bootstrap_runs_one_idempotent_statement_per_call_and_verifies():
    client = LedgerClient()
    manager = make_manager(client)

    await manager.bootstrap()

    assert [descriptor for descriptor, _sql, _values in client.executed] == [
        "schema.bootstrap.apply",
        "schema.bootstrap.apply",
    ]
    for _descriptor, sql, _values in client.executed:
        assert ";" not in sql
        assert "IF NOT EXISTS" in sql
    assert client.descriptors == (
        "schema.bootstrap.apply",
        "schema.bootstrap.verify",
        "schema.bootstrap.apply",
        "schema.bootstrap.verify",
    )
    verifications = [
        (kind, replay_safe)
        for descriptor, kind, replay_safe in client.calls
        if descriptor == "schema.bootstrap.verify"
    ]
    assert verifications == [(OperationKind.READ, None), (OperationKind.READ, None)]
    assert client.catalog == bootstrapped_catalog()


async def test_bootstrap_is_idempotent_across_repeated_calls():
    client = LedgerClient(catalog=bootstrapped_catalog())
    manager = make_manager(client)

    await manager.bootstrap()
    await manager.bootstrap()

    assert len(client.executed) == 4
    assert client.catalog == bootstrapped_catalog()


async def test_bootstrap_fails_closed_when_catalog_verification_is_false():
    client = LedgerClient(
        postcondition_results={(SCHEMA,): False},
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="bootstrap"):
        await manager.bootstrap()

    assert client.descriptors == (
        "schema.bootstrap.apply",
        "schema.bootstrap.verify",
    )


async def test_bootstrap_fails_closed_on_malformed_catalog_verification():
    client = LedgerClient(
        postcondition_results={(SCHEMA, LEDGER_TABLE_NAME): "true"},
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError):
        await manager.bootstrap()


# --------------------------------------------------------------------------
# claim / transition statement shapes
# --------------------------------------------------------------------------


def test_claim_is_one_cas_upsert_statement_with_returning():
    statement = claim_statement(SCHEMA)

    assert statement.count("INSERT INTO") == 1
    assert "ON CONFLICT (component, version, step, descriptor_name) DO UPDATE" in statement
    assert " WHERE " in statement
    assert "ledger.digest = EXCLUDED.digest" in statement
    assert "ledger.state <> 'applied'" in statement
    for lease_branch in (
        "ledger.owner_token = EXCLUDED.owner_token",
        "ledger.owner_token IS NULL",
        "ledger.lease_expires_at IS NULL",
        "ledger.lease_expires_at <= EXCLUDED.updated_at",
    ):
        assert lease_branch in statement
    # Re-claiming a row that is already parked in 'prepared' must not overwrite
    # an earlier interrupted-apply marker with 'prepared', otherwise a second
    # crash erases the evidence that the DDL was already attempted once.
    assert (
        "previous_state = CASE WHEN ledger.state = 'prepared' "
        "THEN ledger.previous_state ELSE ledger.state END"
    ) in statement
    assert statement.endswith(
        "RETURNING ledger.state, ledger.previous_state, ledger.owner_token"
    )
    assert ";" not in statement
    assert "pg_advisory" not in statement
    assert f'"{SCHEMA}"."{LEDGER_TABLE_NAME}"' in statement


def test_transition_statement_is_owner_digest_state_and_lease_guarded():
    statement = transition_statement(SCHEMA)

    assert statement.startswith("UPDATE")
    assert "RETURNING" in statement
    assert "owner_token = $11" in statement
    assert "digest = $10" in statement
    assert "state = $12" in statement
    assert "lease_expires_at > $5" in statement
    assert ";" not in statement


def _placeholder_numbers(sql):
    return {int(number) for number in re.findall(r"\$(\d+)", sql)}


async def test_ledger_statement_placeholders_match_every_production_call():
    descriptor = make_descriptor()
    successful_client = LedgerClient(descriptors=(descriptor,))
    await make_manager(successful_client).initialize([descriptor])

    blocked_client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-b",
                lease_expires_at=EPOCH + timedelta(minutes=5),
            )
        ],
    )
    with pytest.raises(HologresSchemaBusyError):
        await make_manager(blocked_client, busy_retries=0).initialize([descriptor])

    expected_statements = {
        "schema.ledger.claim": claim_statement(SCHEMA),
        "schema.ledger.transition": transition_statement(SCHEMA),
        "schema.ledger.load": load_statement(SCHEMA),
        "schema.ledger.inspect": inspect_statement(SCHEMA),
    }
    observed = {
        operation: []
        for operation in expected_statements
    }
    for operation, sql, values in (
        *successful_client.statement_calls,
        *blocked_client.statement_calls,
    ):
        if operation in observed:
            observed[operation].append((sql, values))

    assert all(observed.values())
    for operation, calls in observed.items():
        for sql, values in calls:
            assert sql == expected_statements[operation]
            assert _placeholder_numbers(sql) == set(range(1, len(values) + 1))


# --------------------------------------------------------------------------
# happy-path state machine
# --------------------------------------------------------------------------


async def test_initialize_progresses_through_the_full_state_machine():
    descriptor = make_descriptor()
    client = LedgerClient(descriptors=(descriptor,))
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert [result.state for result in results] == [SchemaState.APPLIED]
    assert results[0].executed_ddl is True
    assert results[0].resumed is False
    assert results[0].descriptor_name == descriptor.name

    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.APPLIED.value
    assert row["owner_token"] is None
    assert row["lease_expires_at"] is None
    assert row["error_summary"] is None
    assert row["digest"] == descriptor.digest

    assert client.descriptors == (
        "schema.bootstrap.apply",
        "schema.bootstrap.verify",
        "schema.bootstrap.apply",
        "schema.bootstrap.verify",
        "schema.ledger.load",
        "schema.ledger.claim",
        "schema.descriptor.postcondition",
        "schema.ledger.transition",
        "schema.descriptor.apply",
        "schema.ledger.transition",
        "schema.descriptor.postcondition",
        "schema.ledger.transition",
    )
    assert [
        row["previous_state"]
        for row in [client.rows[(descriptor.component, 1, 1, descriptor.name)]]
    ] == [SchemaState.VERIFYING.value]


async def test_claim_and_transitions_declare_explicit_write_replay_safety():
    descriptor = make_descriptor()
    client = LedgerClient(descriptors=(descriptor,))
    manager = make_manager(client)

    await manager.initialize([descriptor])

    row_writes = [
        (name, kind, replay_safe)
        for name, kind, replay_safe in client.calls
        if name in {"schema.ledger.claim", "schema.ledger.transition"}
    ]
    assert row_writes
    for _name, kind, replay_safe in row_writes:
        assert kind is OperationKind.WRITE
        assert replay_safe is True

    ddl_writes = [
        replay_safe
        for name, _kind, replay_safe in client.calls
        if name == "schema.descriptor.apply"
    ]
    assert ddl_writes == [descriptor.replay_safe]


async def test_descriptor_ddl_replay_flag_follows_the_descriptor_decision():
    descriptor = make_descriptor(replay_safe=False)
    client = LedgerClient(descriptors=(descriptor,))
    manager = make_manager(client)

    await manager.initialize([descriptor])

    assert [
        replay_safe
        for name, _kind, replay_safe in client.calls
        if name == "schema.descriptor.apply"
    ] == [False]


async def test_initialize_applies_descriptors_in_deterministic_order():
    first = make_descriptor(name="a_first", version=1, step=1, table="t_a")
    second = make_descriptor(name="b_second", version=1, step=2, table="t_b")
    third = make_descriptor(name="c_third", version=2, step=1, table="t_c")
    other = make_descriptor(
        name="a_other", component="other", version=1, step=1, table="t_o"
    )
    plan = (third, other, second, first)
    client = LedgerClient(descriptors=plan)
    manager = make_manager(client)

    results = await manager.initialize(plan)

    assert [result.descriptor_name for result in results] == [
        "a_other",
        "a_first",
        "b_second",
        "c_third",
    ]
    applied_tables = [
        sql
        for descriptor, sql, _values in client.executed
        if descriptor == "schema.descriptor.apply"
    ]
    for table, sql in zip(("t_o", "t_a", "t_b", "t_c"), applied_tables):
        assert table in sql


async def test_already_applied_descriptors_are_verified_and_skipped():
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        catalog={*bootstrapped_catalog(), (SCHEMA, PROBE_TABLE)},
        rows=[ledger_row(descriptor, state=SchemaState.APPLIED.value)],
    )
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].state is SchemaState.APPLIED
    assert results[0].executed_ddl is False
    assert "schema.descriptor.apply" not in client.descriptors
    assert "schema.ledger.claim" not in client.descriptors


async def test_applied_row_with_missing_object_fails_closed():
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        catalog=bootstrapped_catalog(),
        rows=[ledger_row(descriptor, state=SchemaState.APPLIED.value)],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors


# --------------------------------------------------------------------------
# claim coordination
# --------------------------------------------------------------------------


async def test_active_lease_owned_by_another_initializer_is_refused():
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-b",
                lease_expires_at=EPOCH + timedelta(minutes=5),
            )
        ],
    )
    manager = make_manager(client, busy_retries=0)

    with pytest.raises(HologresSchemaBusyError):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors
    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["owner_token"] == "owner-b"
    assert row["state"] == SchemaState.APPLYING.value


async def test_expired_lease_is_taken_over_by_the_next_initializer():
    descriptor = make_descriptor(replay_safe=True)
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-b",
                lease_expires_at=EPOCH - timedelta(seconds=1),
            )
        ],
    )
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].state is SchemaState.APPLIED
    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.APPLIED.value


async def test_transitions_refuse_a_stolen_row():
    descriptor = make_descriptor()
    client = LedgerClient(descriptors=(descriptor,))
    manager = make_manager(client)
    identity = (descriptor.component, 1, 1, descriptor.name)

    original_execute = client.execute_one

    async def steal_then_execute(sql, *values, **kwargs):
        result = await original_execute(sql, *values, **kwargs)
        if kwargs.get("descriptor") == "schema.descriptor.apply":
            client.rows[identity]["owner_token"] = "owner-thief"
        return result

    client.execute_one = steal_then_execute

    with pytest.raises(HologresSchemaStateError, match="owner"):
        await manager.initialize([descriptor])

    assert client.rows[identity]["state"] != SchemaState.APPLIED.value


async def test_concurrent_initializers_apply_each_descriptor_once():
    descriptor = make_descriptor()
    client = LedgerClient(descriptors=(descriptor,))
    first = make_manager(client, owner_token="owner-a")
    second = make_manager(client, owner_token="owner-b")

    outcomes = await asyncio.gather(
        first.initialize([descriptor]), second.initialize([descriptor])
    )

    assert [outcome[0].state for outcome in outcomes] == [
        SchemaState.APPLIED,
        SchemaState.APPLIED,
    ]
    ddl_calls = [
        name for name, _kind, _replay in client.calls if name == "schema.descriptor.apply"
    ]
    assert len(ddl_calls) == 1


# --------------------------------------------------------------------------
# digest drift and inconsistent ledger rows
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        SchemaState.PREPARED.value,
        SchemaState.APPLYING.value,
        SchemaState.VERIFYING.value,
        SchemaState.APPLIED.value,
        SchemaState.FAILED.value,
    ],
)
async def test_digest_drift_fails_closed_before_any_ddl(state):
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        catalog={*bootstrapped_catalog(), (SCHEMA, PROBE_TABLE)},
        rows=[ledger_row(descriptor, digest="0" * 64, state=state)],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaDriftError):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors
    assert "schema.ledger.claim" not in client.descriptors


async def test_unknown_ledger_state_is_not_silently_skipped():
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[ledger_row(descriptor, state="mystery")],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="state"):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors


async def test_ledger_rows_missing_from_the_plan_are_not_silently_skipped():
    descriptor = make_descriptor(name="probe_events", step=1)
    stale = make_descriptor(name="probe_dropped", step=2, table="t_dropped")
    client = LedgerClient(
        descriptors=(descriptor, stale),
        rows=[ledger_row(stale, state=SchemaState.APPLIED.value)],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="unknown"):
        await manager.initialize([descriptor])


async def test_downgrade_relative_to_the_ledger_is_refused():
    current = make_descriptor(name="probe_events", version=2, step=1)
    older = make_descriptor(name="probe_events", version=1, step=1)
    client = LedgerClient(
        descriptors=(current, older),
        rows=[ledger_row(current, state=SchemaState.APPLIED.value)],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaDefinitionError, match="downgrade"):
        await manager.initialize([older])


# --------------------------------------------------------------------------
# crash recovery
# --------------------------------------------------------------------------


async def test_crash_after_ddl_resumes_without_reapplying():
    descriptor = make_descriptor(replay_safe=False)
    client = LedgerClient(
        descriptors=(descriptor,),
        catalog={*bootstrapped_catalog(), (SCHEMA, PROBE_TABLE)},
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-crashed",
                lease_expires_at=EPOCH - timedelta(seconds=1),
            )
        ],
    )
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].state is SchemaState.APPLIED
    assert results[0].executed_ddl is False
    assert results[0].resumed is True
    assert "schema.descriptor.apply" not in client.descriptors
    assert client.rows[(descriptor.component, 1, 1, descriptor.name)]["state"] == (
        SchemaState.APPLIED.value
    )


async def test_unsatisfied_postcondition_replays_only_replay_safe_descriptors():
    descriptor = make_descriptor(replay_safe=True)
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-crashed",
                lease_expires_at=EPOCH - timedelta(seconds=1),
            )
        ],
    )
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].state is SchemaState.APPLIED
    assert results[0].executed_ddl is True


async def test_unsatisfied_postcondition_refuses_replay_when_not_replay_safe():
    descriptor = make_descriptor(replay_safe=False)
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-crashed",
                lease_expires_at=EPOCH - timedelta(seconds=1),
            )
        ],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="replay"):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors
    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    assert row["error_summary"] == "replay_not_permitted"


async def test_resumed_non_replay_safe_refusal_precedes_precondition():
    descriptor = make_descriptor(
        replay_safe=False,
        precondition_sql=TABLE_EXISTS_SQL,
        precondition_args=(SCHEMA, "missing_dependency"),
    )
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.APPLYING.value,
                owner_token="owner-crashed",
                lease_expires_at=EPOCH - timedelta(seconds=1),
            )
        ],
    )

    with pytest.raises(HologresSchemaStateError, match="replay"):
        await make_manager(client).initialize([descriptor])

    assert "schema.descriptor.precondition" not in client.descriptors
    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    assert row["error_summary"] == "replay_not_permitted"


async def test_second_crash_does_not_erase_the_interrupted_apply_marker():
    # A takeover claim parks the row in 'prepared' and records the interrupted
    # state in previous_state. If that takeover crashes too, the next claim must
    # still see the descriptor as resumed rather than as a fresh apply.
    descriptor = make_descriptor(replay_safe=False)
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.PREPARED.value,
                previous_state=SchemaState.APPLYING.value,
                owner_token="owner-crashed-twice",
                lease_expires_at=EPOCH - timedelta(seconds=1),
            )
        ],
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="replay"):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors
    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    assert row["error_summary"] == "replay_not_permitted"


async def test_fresh_rows_apply_ddl_even_when_not_replay_safe():
    descriptor = make_descriptor(replay_safe=False)
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[ledger_row(descriptor, state=SchemaState.PREPARED.value)],
    )
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].executed_ddl is True
    assert results[0].state is SchemaState.APPLIED


# --------------------------------------------------------------------------
# failure recording
# --------------------------------------------------------------------------


async def test_false_postcondition_records_failed_and_never_marks_applied():
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        postcondition_results={(SCHEMA, PROBE_TABLE): False},
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="postcondition"):
        await manager.initialize([descriptor])

    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    assert row["error_summary"] == "postcondition_unsatisfied"
    assert row["owner_token"] is None
    assert row["lease_expires_at"] is None


@pytest.mark.parametrize("malformed", ["true", 1, None, object()])
async def test_malformed_postcondition_result_records_failed(malformed):
    descriptor = make_descriptor()
    client = LedgerClient(
        descriptors=(descriptor,),
        postcondition_results={(SCHEMA, PROBE_TABLE): malformed},
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError):
        await manager.initialize([descriptor])

    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    assert row["error_summary"] in {
        "postcondition_malformed",
        "postcondition_unsatisfied",
    }


async def test_ddl_failure_records_failed_with_a_sanitized_bounded_summary():
    descriptor = make_descriptor()
    secret = (
        "password=hunter2 dsn=postgres://user:hunter2@host/db "
        "INSERT INTO secrets VALUES ('token') " + "x" * 500
    )
    client = LedgerClient(
        descriptors=(descriptor,),
        ddl_errors={descriptor.sql: RuntimeError(secret)},
    )
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError):
        await manager.initialize([descriptor])

    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    summary = row["error_summary"]
    assert len(summary) <= 200
    assert "hunter2" not in summary
    assert "postgres://" not in summary
    assert "INSERT" not in summary
    assert "ddl_execution_failed" in summary
    assert "RuntimeError" in summary


async def test_failed_precondition_records_failed_and_skips_ddl():
    descriptor = make_descriptor(
        precondition_sql=TABLE_EXISTS_SQL,
        precondition_args=(SCHEMA, "missing_dependency"),
    )
    client = LedgerClient(descriptors=(descriptor,))
    manager = make_manager(client)

    with pytest.raises(HologresSchemaStateError, match="precondition"):
        await manager.initialize([descriptor])

    assert "schema.descriptor.apply" not in client.descriptors
    row = client.rows[(descriptor.component, 1, 1, descriptor.name)]
    assert row["state"] == SchemaState.FAILED.value
    assert row["error_summary"] == "precondition_unsatisfied"


async def test_failed_rows_can_be_retried_by_a_later_initializer():
    descriptor = make_descriptor(replay_safe=True)
    client = LedgerClient(
        descriptors=(descriptor,),
        rows=[
            ledger_row(
                descriptor,
                state=SchemaState.FAILED.value,
                error_summary="ddl_execution_failed|RuntimeError",
            )
        ],
    )
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].state is SchemaState.APPLIED
    assert client.rows[(descriptor.component, 1, 1, descriptor.name)][
        "error_summary"
    ] is None


# --------------------------------------------------------------------------
# restricted API and secret hygiene
# --------------------------------------------------------------------------


def test_schema_module_never_touches_raw_connections_or_transactions(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    source = (
        Path(__file__).resolve().parents[3] / "lightrag/kg/hologres/schema.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "asyncpg",
        "transaction(",
        "executemany",
        ".acquire(",
        "pg_advisory",
        "copy_rows",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
    ):
        assert forbidden not in source, forbidden


class RestrictedClient(LedgerClient):
    def __getattr__(self, name):
        raise AssertionError(f"Schema manager used a forbidden client API: {name}")


async def test_manager_uses_only_the_four_restricted_client_methods():
    descriptor = make_descriptor()
    client = RestrictedClient(descriptors=(descriptor,))
    manager = make_manager(client)

    results = await manager.initialize([descriptor])

    assert results[0].state is SchemaState.APPLIED


async def test_manager_never_logs_owner_tokens_secrets_or_full_sql():
    descriptor = make_descriptor()
    secret_owner = "owner-token-3f9d21"
    client = LedgerClient(
        descriptors=(descriptor,),
        ddl_errors={
            descriptor.sql: RuntimeError("password=hunter2 CREATE TABLE secrets")
        },
    )
    manager = make_manager(client, owner_token=secret_owner)

    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collector()
    logger = logging.getLogger("lightrag")
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(HologresSchemaError):
            await manager.initialize([descriptor])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    joined = "\n".join(records)
    assert secret_owner not in joined
    assert "hunter2" not in joined
    assert "CREATE TABLE" not in joined
    assert descriptor.sql not in joined


async def test_raised_errors_never_leak_owner_tokens_or_sql():
    descriptor = make_descriptor()
    secret_owner = "owner-token-9a71bc"
    client = LedgerClient(
        descriptors=(descriptor,),
        ddl_errors={
            descriptor.sql: RuntimeError("password=hunter2 CREATE TABLE secrets")
        },
    )
    manager = make_manager(client, owner_token=secret_owner)

    with pytest.raises(HologresSchemaError) as exc_info:
        await manager.initialize([descriptor])

    message = str(exc_info.value)
    assert secret_owner not in message
    assert "hunter2" not in message
    assert descriptor.sql not in message
