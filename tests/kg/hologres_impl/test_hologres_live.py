import asyncio
from datetime import datetime, timedelta, timezone
import uuid

import pytest

from lightrag.kg.hologres.capabilities import (
    ProbeKind,
    probe_production_capabilities,
    run_initial_isolated_probes,
)
from lightrag.kg.hologres.client import quote_qualified_identifier
from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.kg.hologres.schema import (
    LEDGER_TABLE_NAME,
    HologresSchemaManager,
    SchemaDescriptor,
    SchemaState,
    kv_schema_descriptors,
)
from lightrag.namespace import NameSpace


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


async def test_hologres_kv_logical_partitions_and_crud(hologres_live_client):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_kv_a_{suffix}"
    workspace_b = f"lightrag_test_kv_b_{suffix}"

    def storage(namespace, workspace):
        return HologresKVStorage(
            namespace=namespace,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=client.config,
            client=client,
        )

    primary = storage(NameSpace.KV_STORE_TEXT_CHUNKS, workspace_a)
    isolated = storage(NameSpace.KV_STORE_TEXT_CHUNKS, workspace_b)
    documents = storage(NameSpace.KV_STORE_FULL_DOCS, workspace_a)
    tracking = storage(NameSpace.KV_STORE_ENTITY_CHUNKS, workspace_a)
    entity_anchors = storage(NameSpace.KV_STORE_FULL_ENTITIES, workspace_a)
    relation_anchors = storage(NameSpace.KV_STORE_FULL_RELATIONS, workspace_a)
    storages = (
        primary,
        isolated,
        documents,
        tracking,
        entity_anchors,
        relation_anchors,
    )
    initialized = []

    try:
        for item in storages:
            await item.initialize()
            initialized.append(item)

        (descriptor,) = kv_schema_descriptors(schema)
        assert (
            await client.fetch_value(
                descriptor.postcondition_sql,
                *descriptor.postcondition_args,
                descriptor="live.kv.catalog",
            )
            is True
        )

        await primary.upsert({"shared": {"value": "a", "old": True}})
        await isolated.upsert({"shared": {"value": "b"}})
        assert await primary.get_by_id_strict("shared") == {
            "value": "a",
            "old": True,
        }
        assert await isolated.get_by_id_strict("shared") == {"value": "b"}

        await primary.upsert({"shared": {"value": "replaced"}, "empty": {}})
        assert await primary.get_by_id_strict("shared") == {"value": "replaced"}
        assert await primary.get_by_ids(["shared", "missing", "shared"]) == [
            {"value": "replaced"},
            None,
            {"value": "replaced"},
        ]
        assert await primary.filter_keys({"shared", "missing"}) == {"missing"}

        protected_a = {
            "sidecar_location": "sidecar-a",
            "parse_format": "markdown",
            "content_hash": "hash-a",
            "process_options": {"mode": "a"},
            "parse_engine": "native",
            "chunk_options": {"size": 100},
        }
        await documents.upsert(
            {
                "doc": {
                    "content": "old",
                    **protected_a,
                    "ordinary": "old",
                }
            }
        )
        await documents.upsert(
            {
                "doc": {
                    "content": "",
                    "sidecar_location": None,
                    "parse_format": "",
                    "content_hash": "",
                    "process_options": None,
                    "parse_engine": "",
                    "chunk_options": {},
                    "ordinary": "new",
                }
            }
        )
        assert await documents.get_by_id_strict("doc") == {
            "content": "",
            **protected_a,
            "ordinary": "new",
        }

        await documents.upsert({"doc": {"ordinary": "newer"}})
        assert await documents.get_by_id_strict("doc") == {
            "content": "",
            **protected_a,
            "ordinary": "newer",
        }

        protected_b = {
            "sidecar_location": "sidecar-b",
            "parse_format": "html",
            "content_hash": "hash-b",
            "process_options": {"mode": "b"},
            "parse_engine": "docling",
            "chunk_options": {"size": 200},
        }
        await documents.upsert({"doc": protected_b})
        assert await documents.get_by_id_strict("doc") == {
            "content": "",
            **protected_b,
            "ordinary": "newer",
        }

        await documents.upsert(
            {
                "new-doc": {
                    "content": "new",
                    "sidecar_location": None,
                    "parse_format": "",
                    "content_hash": "",
                    "process_options": None,
                    "parse_engine": "",
                    "chunk_options": {},
                    "ordinary": "inserted",
                }
            }
        )
        assert await documents.get_by_id_strict("new-doc") == {
            "content": "new",
            "ordinary": "inserted",
        }

        await tracking.upsert({"tracking": {"chunk_ids": []}})
        await entity_anchors.upsert(
            {"entity-anchor": {"entity_names": [], "metadata": {}}}
        )
        await relation_anchors.upsert(
            {"relation-anchor": {"relation_pairs": [], "metadata": {}}}
        )
        assert await tracking.get_by_id_strict("tracking") == {"chunk_ids": []}
        assert await entity_anchors.get_by_id_strict("entity-anchor") == {
            "entity_names": [],
            "metadata": {},
        }
        assert await relation_anchors.get_by_id_strict("relation-anchor") == {
            "relation_pairs": [],
            "metadata": {},
        }

        await primary.delete(["empty"])
        assert await primary.get_by_id_strict("empty") is None
        assert await primary.is_empty() is False
        await primary.drop()
        assert await primary.is_empty() is True
        assert await isolated.get_by_id_strict("shared") == {"value": "b"}
        assert await documents.get_by_id_strict("doc") is not None
    finally:
        for item in initialized:
            try:
                await item.drop()
            finally:
                await item.finalize()
