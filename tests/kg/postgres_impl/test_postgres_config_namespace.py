"""The ``config`` namespace on PostgreSQL: the ``LIGHTRAG_CONFIG`` table, the
per-namespace SQL templates, the row shaping in both directions, and
``iter_rows`` (scenario 16 in docs/design/ConfigurationStorage.md) as a
keyset-paged id walk read through ``get_by_ids``.

The ``workspace`` column is the CONTAINER's workspace; the workspace a row is
ABOUT travels inside the JSONB payload with the rest of the uniform row shape.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.postgres_impl import (
    NAMESPACE_TABLE_MAP,
    SQL_TEMPLATES,
    TABLES,
    PGKVStorage,
    _config_row_from_pg,
    _config_row_payload,
    namespace_to_table_name,
)
from lightrag.namespace import CONFIG_WORKSPACE, NameSpace
from lightrag.utils import _grant_reserved_workspace

pytestmark = pytest.mark.offline


def test_the_namespace_maps_to_its_own_table_and_ddl():
    assert NAMESPACE_TABLE_MAP[NameSpace.KV_STORE_CONFIG] == "LIGHTRAG_CONFIG"
    assert namespace_to_table_name("config") == "LIGHTRAG_CONFIG"
    ddl = TABLES["LIGHTRAG_CONFIG"]["ddl"]
    assert "value JSONB NOT NULL" in ddl
    assert "PRIMARY KEY (workspace, id)" in ddl
    # No other namespace may fold onto it through the endswith match.
    for namespace in (
        "full_docs",
        "text_chunks",
        "llm_response_cache",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "doc_status",
        "entities",
        "relationships",
        "chunks",
    ):
        assert namespace_to_table_name(namespace) != "LIGHTRAG_CONFIG", namespace


def test_the_templates_read_and_write_the_config_table():
    assert (
        "FROM LIGHTRAG_CONFIG WHERE workspace=$1 AND id=$2"
        in SQL_TEMPLATES["get_by_id_config"]
    )
    assert "id = ANY($2)" in SQL_TEMPLATES["get_by_ids_config"]
    upsert = SQL_TEMPLATES["upsert_config"]
    assert "INSERT INTO LIGHTRAG_CONFIG (workspace, id, value)" in upsert
    assert "ON CONFLICT (workspace,id) DO UPDATE" in upsert
    assert "update_time = CURRENT_TIMESTAMP" in upsert


def test_row_shaping_round_trips_the_payload():
    row = {
        "_id": "ws/embedding/entities",
        "create_time": 1,
        "update_time": 2,
        "schema_version": 1,
        "workspace": "ws",
        "updated_at": "now",
        "updated_by": "test",
        "value": {"model": "m", "dim": 8, "origin": "probe"},
    }
    payload = _config_row_payload(row)
    assert set(payload) == {
        "schema_version",
        "workspace",
        "updated_at",
        "updated_by",
        "value",
    }

    rebuilt = _config_row_from_pg(
        {
            "id": "ws/embedding/entities",
            "value": json.dumps(payload),
            "create_time": 5,
            "update_time": 0,
        }
    )
    assert rebuilt["value"] == {"model": "m", "dim": 8, "origin": "probe"}
    assert rebuilt["workspace"] == "ws"
    assert rebuilt["_id"] == rebuilt["id"] == "ws/embedding/entities"
    assert rebuilt["create_time"] == 5 and rebuilt["update_time"] == 5

    # asyncpg may hand JSONB back already decoded; a corrupt payload reads as
    # an empty mapping rather than raising in the shaping layer.
    assert _config_row_from_pg({"id": "k", "value": {"value": {}}})["value"] == {}
    assert _config_row_from_pg({"id": "k", "value": "not json"}).get("value") is None


def _storage(query_side_effect):
    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = "config"
    storage.workspace = CONFIG_WORKSPACE
    storage.global_config = {}
    db = MagicMock()
    db.query = AsyncMock(side_effect=query_side_effect)
    db.workspace = None
    storage.db = db
    # The reserved container workspace only passes validation under the
    # factory's grant; this helper stands in for the factory.
    with _grant_reserved_workspace(CONFIG_WORKSPACE):
        storage.__post_init__()
    return storage, db


async def test_upsert_builds_the_config_tuple():
    captured = []

    async def fake_run_with_retry(operation, **kwargs):
        conn = AsyncMock()
        tx = AsyncMock()
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = False
        conn.transaction = MagicMock(return_value=tx)
        await operation(conn)
        for call in conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))

    storage, db = _storage(None)
    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    await storage.upsert(
        {
            "ws/embedding/chunks": {
                "schema_version": 1,
                "workspace": "ws",
                "updated_at": "now",
                "updated_by": "t",
                "value": {"model": "m", "dim": 8, "origin": "rebuild"},
                "_id": "ws/embedding/chunks",
                "create_time": 1,
                "update_time": 2,
            }
        }
    )

    assert len(captured) == 1
    sql, rows = captured[0]
    assert sql == SQL_TEMPLATES["upsert_config"]
    workspace, key, payload = rows[0]
    assert (workspace, key) == (CONFIG_WORKSPACE, "ws/embedding/chunks")
    decoded = json.loads(payload)
    assert decoded["value"] == {"model": "m", "dim": 8, "origin": "rebuild"}
    assert "create_time" not in decoded and "_id" not in decoded


async def test_iter_rows_pages_the_ids_and_reads_each_page_through_get_by_ids():
    ids = [f"ws{i}/embedding/entities" for i in range(5)]
    table = {
        key: {
            "id": key,
            "value": json.dumps({"workspace": f"ws{i}", "value": {"n": i}}),
            "create_time": 1,
            "update_time": 1,
        }
        for i, key in enumerate(ids)
    }
    page_calls = []

    async def query(sql, params=None, multirows=False, **kwargs):
        if sql.startswith("SELECT id FROM LIGHTRAG_CONFIG"):
            workspace, last_id, limit = params
            assert workspace == CONFIG_WORKSPACE
            page_calls.append((last_id, limit))
            remaining = [k for k in sorted(table) if last_id is None or k > last_id]
            return [{"id": k} for k in remaining[:limit]]
        assert sql == SQL_TEMPLATES["get_by_ids_config"]
        workspace, wanted = params
        return [dict(table[k]) for k in wanted]

    storage, _ = _storage(query)

    rows = [row async for row in storage.iter_rows(page_size=2)]

    assert [r["_id"] for r in rows] == sorted(ids)
    assert [r["workspace"] for r in rows] == [f"ws{i}" for i in range(5)]
    # Three id pages of two, the last one short, and never a whole-table read.
    assert page_calls == [(None, 2), (sorted(ids)[1], 2), (sorted(ids)[3], 2)]


async def test_iter_rows_on_an_empty_table_yields_nothing():
    storage, _ = _storage(lambda *a, **k: [])
    assert [row async for row in storage.iter_rows()] == []
