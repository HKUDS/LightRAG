import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from lightrag.kg.hologres.capabilities import (
    ProbeKind,
    probe_production_capabilities,
    run_initial_isolated_probes,
)
from lightrag.kg.hologres.client import quote_qualified_identifier
from lightrag.kg.hologres.schema import (
    LEDGER_TABLE_NAME,
    HologresSchemaManager,
    SchemaDescriptor,
    SchemaState,
)


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_initial_hologres_capabilities(hologres_live_client):
    client, schema = hologres_live_client

    production_report = await probe_production_capabilities(client)
    isolated_report = await run_initial_isolated_probes(client, schema)

    assert production_report.version.major >= 5
    assert isolated_report.supports(ProbeKind.SINGLE_AUTOCOMMIT_DDL)
    assert isolated_report.supports(ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS)
    assert isolated_report.supports(ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT)


async def test_resumable_schema_management(hologres_live_client):
    client, schema = hologres_live_client
    ledger = quote_qualified_identifier(schema, LEDGER_TABLE_NAME)
    stale_table = "schema_stale_probe"
    concurrent_table = "schema_concurrent_probe"

    def descriptor(name, table):
        qualified = quote_qualified_identifier(schema, table)
        return SchemaDescriptor(
            name=name,
            component="live_schema",
            version=1,
            step=1 if table == stale_table else 2,
            sql=f"CREATE TABLE IF NOT EXISTS {qualified} (id text PRIMARY KEY)",
            postcondition_sql=(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'r')"
            ),
            postcondition_args=(schema, table),
            replay_safe=False,
        )

    class CrashAfterDdlClient:
        def __init__(self, delegate):
            self.delegate = delegate

        async def execute_one(self, sql, *values, **kwargs):
            result = await self.delegate.execute_one(sql, *values, **kwargs)
            if kwargs.get("descriptor") == "schema.descriptor.apply":
                raise asyncio.CancelledError("simulated crash after DDL")
            return result

        async def fetch_one(self, sql, *values, **kwargs):
            return await self.delegate.fetch_one(sql, *values, **kwargs)

        async def fetch_all(self, sql, *values, **kwargs):
            return await self.delegate.fetch_all(sql, *values, **kwargs)

        async def fetch_value(self, sql, *values, **kwargs):
            return await self.delegate.fetch_value(sql, *values, **kwargs)

    stale = descriptor("stale_probe", stale_table)
    concurrent = descriptor("concurrent_probe", concurrent_table)

    expired_now = datetime.now(timezone.utc) - timedelta(minutes=5)
    interrupted = HologresSchemaManager(
        CrashAfterDdlClient(client),
        schema=schema,
        owner_token="live-interrupted-owner",
        lease_seconds=1,
        now_provider=lambda: expired_now,
    )
    with pytest.raises(asyncio.CancelledError, match="simulated crash"):
        await interrupted.initialize([stale])

    recovered = HologresSchemaManager(
        client,
        schema=schema,
        owner_token="live-recovery-owner",
    )
    recovered_result = await recovered.initialize([stale])
    assert recovered_result[0].state is SchemaState.APPLIED
    assert recovered_result[0].resumed is True
    assert recovered_result[0].executed_ddl is False

    first = HologresSchemaManager(
        client, schema=schema, owner_token="live-concurrent-a"
    )
    second = HologresSchemaManager(
        client, schema=schema, owner_token="live-concurrent-b"
    )
    concurrent_results = await asyncio.gather(
        first.initialize([stale, concurrent]),
        second.initialize([stale, concurrent]),
    )
    assert all(
        result[-1].state is SchemaState.APPLIED for result in concurrent_results
    )
    assert sum(result[-1].executed_ddl for result in concurrent_results) == 1

    rows = await client.fetch_all(
        f"SELECT descriptor_name, state FROM {ledger} "
        "WHERE component = $1 ORDER BY step",
        "live_schema",
        descriptor="live.schema.ledger.read",
    )
    assert [(row["descriptor_name"], row["state"]) for row in rows] == [
        ("stale_probe", SchemaState.APPLIED.value),
        ("concurrent_probe", SchemaState.APPLIED.value),
    ]
