"""Opt-in No-DDL tests in disposable databases with a non-owner runtime role.

Uses POSTGRES_* administrator credentials, with CREATEDB and CREATEROLE. No
existing application tables or roles are altered. Vector cases require pgvector;
the pg_smoke cases run on plain PostgreSQL without it.
"""

import os
import re
import secrets
from uuid import uuid4

import asyncpg
import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.pgtable_impl import _DDL
from lightrag.exceptions import VectorSpaceMismatchError
from lightrag.kg.postgres_impl import (
    TABLES,
    ClientManager,
    PGVectorStorage,
    PostgreSQLDB,
)
from lightrag.kg.postgres_schema import COMMON_TABLES, VECTOR_TABLES
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

_MODEL = "no_ddl_fixture"
_TABLE = "LIGHTRAG_VDB_CHUNKS_no_ddl_fixture_8d"


async def _embed(texts, **kwargs):
    # Nonzero and deterministic, with a distinct vector for each input.
    return np.array(
        [[1.0, len(t) / 100, *([0.25] * 6)] for t in texts], dtype=np.float32
    )


async def _llm(*args, **kwargs):
    return ""


@pytest.fixture
async def provisioned(monkeypatch, request):
    if not os.environ.get("POSTGRES_PASSWORD"):
        pytest.skip("POSTGRES_PASSWORD administrator credential is not configured")
    initialize_share_data(workers=1)
    with_vectors = getattr(request, "param", True)
    config = ClientManager.get_config("PGVectorStorage" if with_vectors else None)
    if config["workspace"]:
        pytest.skip("POSTGRES_WORKSPACE / config.ini would override test isolation")
    assert ClientManager._instances["db"] is None
    admin_params = {
        k: config[k] for k in ("host", "port", "user", "password", "database")
    }
    admin_params["port"] = int(admin_params["port"])
    admin = await asyncpg.connect(**admin_params)
    if with_vectors and not await admin.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name='vector')"
    ):
        await admin.close()
        pytest.skip("pgvector is not available on this PostgreSQL server")
    suffix = uuid4().hex[:12]
    database, role = f"lr_noddl_{suffix}", f"lr_runtime_{suffix}"
    password = secrets.token_hex(24)
    db = None
    created_database = created_role = False
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
        created_database = True
        await admin.execute(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
        created_role = True
        config.update(
            database=database,
            auto_migrate=True,
            vector_index_type="HNSW",
            workspace=None,
        )
        db = PostgreSQLDB(config)
        await db.initdb()
        await db.check_tables()
        await db.execute(_DDL)
        for base in (
            ()
            if not with_vectors
            else (
                "LIGHTRAG_VDB_CHUNKS",
                "LIGHTRAG_VDB_ENTITY",
                "LIGHTRAG_VDB_RELATION",
            )
        ):
            await PGVectorStorage.setup_table(
                db, f"{base}_{_MODEL}_8d", "tenant_a", 8, base, base
            )
        await db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        await db.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
        await db.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{role}"'
        )
        monkeypatch.setenv("POSTGRES_DATABASE", database)
        monkeypatch.setenv("POSTGRES_USER", role)
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
        monkeypatch.setenv("POSTGRES_VECTOR_INDEX_TYPE", "HNSW")
        monkeypatch.setenv("POSTGRES_AUTO_MIGRATE", "false")
        yield db, role
    finally:
        client = ClientManager._instances["db"]
        if client is not None:
            await client._reset_pool()
            ClientManager._instances = {
                "db": None,
                "ref_count": 0,
                "vector_signature": None,
            }
        if db is not None:
            await db._reset_pool()
        if created_database:
            await admin.execute(f'DROP DATABASE "{database}" WITH (FORCE)')
        if created_role:
            await admin.execute(f'DROP ROLE "{role}"')
        await admin.close()


@pytest.fixture
async def runtime_trace(monkeypatch):
    attempted = []
    # Record before execution: a broad catch in production cannot hide a call.
    forbidden = re.compile(
        r"(?:^|;)\s*(CREATE|ALTER|DROP|TRUNCATE|REINDEX|COMMENT|GRANT|REVOKE|DO)\b",
        re.I,
    )
    for name in ("execute", "executemany", "fetch", "fetchrow", "fetchval"):
        original = getattr(asyncpg.Connection, name)

        async def traced(self, query, *args, _original=original, **kwargs):
            if forbidden.search(query):
                attempted.append(query)
            return await _original(self, query, *args, **kwargs)

        monkeypatch.setattr(asyncpg.Connection, name, traced)
    yield attempted
    assert attempted == []


def _rag(tmp_path, workspace):
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGTableGraphStorage",
        config_storage="PGKVStorage",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=1024, model_name=_MODEL, func=_embed
        ),
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.1},
    )


async def test_restricted_runtime_crud_and_reconnect(
    provisioned, runtime_trace, tmp_path
):
    admin, role = provisioned
    # Compare against what automatic initialization actually provisioned. This
    # catches stale/omitted validator columns without parsing the DDL strings.
    assert set(COMMON_TABLES) == set(TABLES) - set(VECTOR_TABLES)
    expected_columns = {
        **{t.lower(): set(columns) for t, columns in COMMON_TABLES.items()},
        **{
            f"{t}_{_MODEL}_8d".lower(): set(columns) | {"content_vector"}
            for t, columns in VECTOR_TABLES.items()
        },
    }
    actual_columns = {}
    for row in await admin.query(
        "SELECT c.relname, a.attname FROM pg_class c JOIN pg_attribute a "
        "ON a.attrelid=c.oid WHERE a.attnum>0 AND NOT a.attisdropped AND c.oid IN "
        "(SELECT to_regclass(t) FROM unnest($1::text[]) AS names(t))",
        [list(expected_columns)],
        multirows=True,
    ):
        actual_columns.setdefault(row["relname"], set()).add(row["attname"])
    assert actual_columns == expected_columns
    permissions = await admin.query(
        "SELECT has_schema_privilege($1, 'public', 'CREATE') AS can_create, "
        "(SELECT relowner::regrole::text FROM pg_class WHERE oid='lightrag_doc_full'::regclass) AS owner",
        [role],
    )
    assert permissions == {"can_create": False, "owner": admin.user}
    first, second = _rag(tmp_path, "tenant_a"), _rag(tmp_path, "tenant_b")
    try:
        await first.initialize_storages()
        await second.initialize_storages()
        for rag, content in ((first, "alpha document"), (second, "beta document")):
            await rag.full_docs.upsert(
                {"same": {"content": content, "file_path": "sample.txt"}}
            )
            await rag.full_docs.upsert(
                {"same": {"content": content + " updated", "file_path": "sample.txt"}}
            )
            await rag.doc_status.upsert(
                {
                    "same": {
                        "content_summary": content,
                        "content_length": len(content),
                        "status": DocStatus.PENDING,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "file_path": "sample.txt",
                        "chunks_list": [],
                        "metadata": {},
                    }
                }
            )
            row = await rag.doc_status.get_by_id("same")
            assert row["status"] == DocStatus.PENDING
            await rag.doc_status.upsert(
                {"same": {**row, "status": DocStatus.PROCESSED}}
            )
            assert (await rag.doc_status.get_by_id("same"))[
                "status"
            ] == DocStatus.PROCESSED
            await rag.chunks_vdb.upsert(
                {
                    "same": {
                        "content": content,
                        "tokens": 2,
                        "chunk_order_index": 0,
                        "full_doc_id": "same",
                        "file_path": "sample.txt",
                    }
                }
            )
            await rag.chunks_vdb.index_done_callback()
            assert (await rag.chunks_vdb.get_by_id("same"))["content"] == content
            assert len((await rag.chunks_vdb.get_vectors_by_ids(["same"]))["same"]) == 8
            assert [r["id"] for r in await rag.chunks_vdb.query(content, top_k=1)] == [
                "same"
            ]
            graph = rag.chunk_entity_relation_graph
            await graph.upsert_node("A", {"entity_id": "A", "description": content})
            await graph.upsert_edge("A", "B", {"description": content, "weight": 0.5})
            assert (await graph.get_edge("B", "A"))["description"] == content
        assert (await first.full_docs.get_by_id("same"))[
            "content"
        ] == "alpha document updated"
        assert (await second.full_docs.get_by_id("same"))[
            "content"
        ] == "beta document updated"
        assert (await second.chunk_entity_relation_graph.get_node("A"))[
            "description"
        ] == "beta document"
        assert first.configuration_storage.workspace == "_lightrag_config"
        identity = await first.configuration_storage.get_by_id_strict(
            "$meta/storage_identity"
        )
        assert identity["value"]["uuid"]
        assert await first.configuration_storage.get_by_id_strict(
            "tenant_a/embedding/chunks"
        )
        assert await first.configuration_storage.get_by_id_strict(
            "tenant_b/embedding/chunks"
        )
        # A new physical pool must retain No-DDL policy and its vector codecs.
        await first.full_docs.db._reset_pool()
        assert (await first.full_docs.get_by_id("same"))[
            "content"
        ] == "alpha document updated"
        assert [
            r["id"] for r in await first.chunks_vdb.query("alpha document", top_k=1)
        ] == ["same"]
        await first.chunk_entity_relation_graph.delete_node("B")
        assert await first.chunk_entity_relation_graph.has_edge("A", "B") is False
        assert await second.chunk_entity_relation_graph.has_edge("A", "B") is True
        await first.full_docs.delete(["same"])
        await first.doc_status.delete(["same"])
        await first.chunks_vdb.delete(["same"])
        await first.chunks_vdb.index_done_callback()
        assert await first.full_docs.get_by_id("same") is None
        assert await first.doc_status.get_by_id("same") is None
        assert await first.chunks_vdb.get_by_id("same") is None
        assert (await second.chunks_vdb.get_by_id("same"))["content"] == "beta document"
        assert runtime_trace == []
    finally:
        await second.finalize_storages()
        await first.finalize_storages()
    assert ClientManager._instances["db"] is None
    assert ClientManager._instances["ref_count"] == 0


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("DROP TABLE lightrag_doc_full", "LIGHTRAG_DOC_FULL.*missing"),
        (
            "DROP TABLE lightrag_doc_full; CREATE VIEW lightrag_doc_full AS SELECT 1 AS id",
            "ordinary or partitioned table",
        ),
        (
            "ALTER TABLE lightrag_doc_full DROP CONSTRAINT lightrag_doc_full_pk; ALTER TABLE lightrag_doc_full ADD PRIMARY KEY(id)",
            "PRIMARY KEY",
        ),
        ("DROP INDEX idx_lightrag_graph_edges_namespace_tgt", "btree index.*tgt_id"),
        ("ALTER TABLE lightrag_doc_full DROP COLUMN parse_engine", "parse_engine"),
        (
            "ALTER TABLE lightrag_doc_full ALTER COLUMN chunk_options TYPE json USING chunk_options::json",
            "chunk_options.*jsonb",
        ),
        (
            "ALTER TABLE lightrag_doc_full DROP CONSTRAINT lightrag_doc_full_pk",
            "PRIMARY KEY",
        ),
        (
            f"ALTER TABLE {_TABLE} ALTER COLUMN content_vector TYPE vector(7)",
            "dimension",
        ),
        (
            f"ALTER TABLE {_TABLE} ALTER COLUMN content_vector TYPE halfvec(8)",
            "pgvector type vector",
        ),
        (f"DROP TABLE {_TABLE}", "LIGHTRAG_VDB_CHUNKS.*missing"),
        (
            "DROP INDEX idx_lightrag_vdb_chunks_no_ddl_fixture_8d_hnsw_cosine",
            "usable nonpartial hnsw",
        ),
        (
            "ALTER TABLE lightrag_graph_edges DROP CONSTRAINT fk_lightrag_graph_edges_src",
            "FOREIGN KEY.*src_id",
        ),
        (
            "ALTER TABLE lightrag_graph_edges DROP CONSTRAINT fk_lightrag_graph_edges_tgt; ALTER TABLE lightrag_graph_edges ADD FOREIGN KEY (workspace, namespace, tgt_id) REFERENCES lightrag_graph_nodes(workspace, namespace, id)",
            "ON DELETE CASCADE",
        ),
    ],
)
async def test_incompatible_schema_refuses_before_processing(
    provisioned, damage, message, tmp_path
):
    admin, _ = provisioned
    # Catalog mutations are confined to this test's disposable database.
    if "content_vector TYPE" in damage:
        await admin.execute(
            "DROP INDEX idx_lightrag_vdb_chunks_no_ddl_fixture_8d_hnsw_cosine"
        )
    await admin.execute(damage)
    physical_vector = "content_vector TYPE" in damage
    if physical_vector:
        dimension = 7 if "vector(7)" in damage else 8
        vector_literal = "[" + ",".join(["1"] * dimension) + "]"
        await admin.execute(
            f"INSERT INTO {_TABLE}(workspace,id,content_vector) VALUES ($1,$2,$3::text::{'halfvec' if 'halfvec' in damage else 'vector'})",
            {
                "workspace": "tenant_a",
                "id": "preserve-physical-schema-row",
                "vector": vector_literal,
            },
        )
    rag = _rag(tmp_path, "tenant_a")
    with pytest.raises((RuntimeError, ValueError), match=message) as error:
        await rag.initialize_storages()
    if physical_vector:
        assert not isinstance(error.value, VectorSpaceMismatchError)
        assert (
            await admin.query(f"SELECT id FROM {_TABLE} WHERE workspace='tenant_a'")
        )["id"] == "preserve-physical-schema-row"
    assert ClientManager._instances["db"] is None
    assert ClientManager._instances["ref_count"] == 0
    await rag.finalize_storages()


@pytest.mark.parametrize(
    ("legacy_workspace", "refuses"), [("tenant_a", True), ("other_tenant", False)]
)
async def test_graph_normalization_is_scoped_and_never_automatic(
    provisioned, legacy_workspace, refuses, tmp_path
):
    admin, _ = provisioned
    await admin.execute(
        "INSERT INTO lightrag_graph_nodes(workspace, namespace, id) VALUES ($1, 'chunk_entity_relation', 'A'), ($1, 'chunk_entity_relation', 'Z')",
        {"workspace": legacy_workspace},
    )
    await admin.execute(
        "INSERT INTO lightrag_graph_edges(workspace, namespace, src_id, tgt_id) VALUES ($1, 'chunk_entity_relation', 'Z', 'A')",
        {"workspace": legacy_workspace},
    )
    rag = _rag(tmp_path, "tenant_a")
    try:
        if refuses:
            with pytest.raises(RuntimeError, match="noncanonical edges.*tenant_a"):
                await rag.initialize_storages()
        else:
            await rag.initialize_storages()
        row = await admin.query(
            "SELECT src_id, tgt_id FROM lightrag_graph_edges WHERE workspace=$1",
            [legacy_workspace],
        )
        assert row == {"src_id": "Z", "tgt_id": "A"}
    finally:
        await rag.finalize_storages()
    assert ClientManager._instances["db"] is None


async def test_active_migration_state_refuses_without_copying(provisioned, tmp_path):
    admin, role = provisioned
    await PGVectorStorage._pg_create_table(
        admin, "LIGHTRAG_VDB_CHUNKS", "LIGHTRAG_VDB_CHUNKS", 8
    )
    await admin.execute(
        f'CREATE TABLE LIGHTRAG_VDB_MIGRATION_STATE (target_table text, workspace text, PRIMARY KEY(target_table, workspace)); GRANT SELECT ON LIGHTRAG_VDB_MIGRATION_STATE TO "{role}"; GRANT SELECT, DELETE ON LIGHTRAG_VDB_CHUNKS TO "{role}"'
    )
    await admin.execute(
        "INSERT INTO LIGHTRAG_VDB_CHUNKS(workspace, id, content_vector) VALUES ('tenant_a', 'uncopied', '[1,0,0,0,0,0,0,0]')"
    )
    await admin.execute(
        f"INSERT INTO {_TABLE}(workspace, id, content_vector) VALUES ('tenant_a', 'partial', '[1,0,0,0,0,0,0,0]')"
    )
    await admin.execute(
        f"INSERT INTO LIGHTRAG_VDB_MIGRATION_STATE VALUES ('{_TABLE.lower()}', 'tenant_a')"
    )
    rag = _rag(tmp_path, "tenant_a")
    with pytest.raises(RuntimeError, match="legacy vector migration.*incomplete"):
        await rag.initialize_storages()
    rows = await admin.query(
        f"SELECT id FROM {_TABLE} WHERE workspace='tenant_a'", multirows=True
    )
    assert rows == [{"id": "partial"}]
    assert (
        await admin.query("SELECT count(*) AS n FROM LIGHTRAG_VDB_MIGRATION_STATE")
    )["n"] == 1
    assert ClientManager._instances["db"] is None
    await rag.finalize_storages()


async def test_schema_resolution_follows_runtime_search_path(provisioned, monkeypatch):
    admin, role = provisioned
    await admin.execute(
        f'CREATE SCHEMA unrelated; CREATE TABLE unrelated.lightrag_doc_full (unrelated int); GRANT USAGE ON SCHEMA unrelated TO "{role}"; GRANT SELECT, INSERT, UPDATE, DELETE ON unrelated.lightrag_doc_full TO "{role}"'
    )
    # A same-named table outside the search path must not affect a valid attach.
    client = await ClientManager.get_client("PGVectorStorage")
    await ClientManager.release_client(client)
    monkeypatch.setenv("POSTGRES_SERVER_SETTINGS", "search_path=unrelated,public")
    with pytest.raises(
        RuntimeError, match="LIGHTRAG_DOC_FULL.*missing column workspace"
    ):
        await ClientManager.get_client("PGVectorStorage")
    assert ClientManager._instances["db"] is None
    assert ClientManager._instances["ref_count"] == 0


async def test_missing_extension_refuses_before_pool_publication(provisioned):
    admin, _ = provisioned
    await admin.execute("DROP EXTENSION vector CASCADE")
    with pytest.raises(RuntimeError, match="administrator install pgvector"):
        await ClientManager.get_client("PGVectorStorage")
    assert ClientManager._instances["db"] is None
    assert ClientManager._instances["ref_count"] == 0


@pytest.mark.pg_smoke
@pytest.mark.parametrize("provisioned", [False], indirect=True)
async def test_plain_postgres_no_ddl_does_not_require_pgvector(
    provisioned, runtime_trace
):
    from lightrag.kg.pgtable_impl import PGTableGraphStorage
    from lightrag.kg.postgres_impl import PGDocStatusStorage, PGKVStorage

    admin, _ = provisioned
    assert (
        await admin.query("SELECT 1 FROM pg_extension WHERE extname='vector'") is None
    )
    kv = PGKVStorage(
        namespace="full_docs", workspace="plain", global_config={}, embedding_func=None
    )
    status = PGDocStatusStorage(
        namespace="doc_status", workspace="plain", global_config={}, embedding_func=None
    )
    graph = PGTableGraphStorage(
        namespace="graph", workspace="plain", global_config={}, embedding_func=None
    )
    try:
        await kv.initialize()
        await status.initialize()
        await graph.initialize()
        assert kv.db.enable_vector is False
        await kv.upsert(
            {
                "plain-document": {
                    "content": "no vector extension",
                    "file_path": "plain.txt",
                }
            }
        )
        assert (await kv.get_by_id("plain-document"))[
            "content"
        ] == "no vector extension"
        await graph.upsert_edge("A", "B", {"description": "plain postgres"})
        assert await graph.has_edge("A", "B") is True
        await graph.delete_node("A")
        assert await graph.has_edge("A", "B") is False
        assert runtime_trace == []
    finally:
        await graph.finalize()
        await status.finalize()
        await kv.finalize()


async def test_halfvec_runtime_uses_existing_column_and_codec(provisioned, monkeypatch):
    admin, _ = provisioned
    admin.vector_index_type = "HNSW_HALFVEC"
    await admin._create_vector_index(_TABLE, 8)
    monkeypatch.setenv("POSTGRES_VECTOR_INDEX_TYPE", "HNSW_HALFVEC")
    storage = PGVectorStorage(
        namespace="chunks",
        workspace="tenant_a",
        global_config={
            "embedding_batch_num": 2,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.1},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, model_name=_MODEL, func=_embed),
    )
    try:
        await storage.initialize()
        await storage.upsert(
            {
                "halfvec": {
                    "content": "half vector",
                    "tokens": 2,
                    "chunk_order_index": 0,
                    "full_doc_id": "doc",
                    "file_path": "half.txt",
                }
            }
        )
        await storage.index_done_callback()
        assert len((await storage.get_vectors_by_ids(["halfvec"]))["halfvec"]) == 8
        assert [r["id"] for r in await storage.query("half vector", top_k=1)] == [
            "halfvec"
        ]
    finally:
        await storage.finalize()


@pytest.mark.pg_smoke
@pytest.mark.parametrize("provisioned", [False], indirect=True)
@pytest.mark.parametrize(
    ("damage", "message"),
    [
        pytest.param(
            "ALTER TABLE lightrag_doc_full ALTER COLUMN parse_engine SET NOT NULL; ALTER TABLE lightrag_doc_full ALTER COLUMN parse_engine SET DEFAULT 'legacy'",
            "parse_engine.*NULL",
            id="explicit-null-with-default",
        ),
        pytest.param(
            "ALTER TABLE lightrag_doc_full ALTER COLUMN meta SET NOT NULL",
            "meta.*default",
            id="omitted-required-column",
        ),
        pytest.param(
            "ALTER TABLE lightrag_doc_full DROP COLUMN content; ALTER TABLE lightrag_doc_full ADD COLUMN content text GENERATED ALWAYS AS ('computed') STORED",
            "content.*generated",
            id="generated-written-column",
        ),
        pytest.param(
            "ALTER TABLE lightrag_doc_full DROP COLUMN update_time; ALTER TABLE lightrag_doc_full ADD COLUMN update_time timestamp GENERATED ALWAYS AS ('2026-01-01'::timestamp) STORED",
            "update_time.*generated",
            id="generated-update-only-column",
        ),
        pytest.param(
            "ALTER TABLE lightrag_doc_chunks ALTER COLUMN tokens SET NOT NULL; ALTER TABLE lightrag_doc_chunks ALTER COLUMN tokens ADD GENERATED ALWAYS AS IDENTITY",
            "tokens.*identity",
            id="always-identity-written-column",
        ),
    ],
)
async def test_insert_column_requirements_reject_incompatible_schema(
    provisioned, damage, message
):
    admin, _ = provisioned
    await admin.execute(damage)
    with pytest.raises(RuntimeError, match=message):
        await ClientManager.get_client()
    assert ClientManager._instances["db"] is None
    assert ClientManager._instances["ref_count"] == 0


@pytest.mark.parametrize("family", ["legacy", "sibling"])
async def test_vector_cleanup_grants_are_checked_before_attach(provisioned, family):
    admin, role = provisioned
    base = "LIGHTRAG_VDB_CHUNKS"
    retained = base if family == "legacy" else f"{base}_previous_8d"
    await PGVectorStorage._pg_create_table(admin, retained, base, 8)
    await admin.execute(f'GRANT SELECT ON {retained} TO "{role}"')
    await admin.execute(
        f"INSERT INTO {retained}(workspace,id) VALUES ('tenant_a','kept'), ('other_tenant','other')"
    )
    await admin.execute(
        f"INSERT INTO {_TABLE}(workspace,id) VALUES ('tenant_a','active')"
    )

    def make_storage():
        return PGVectorStorage(
            namespace="chunks",
            workspace="tenant_a",
            global_config={
                "embedding_batch_num": 2,
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.1},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=8, model_name=_MODEL, func=_embed
            ),
        )

    storage = make_storage()
    try:
        with pytest.raises(RuntimeError, match=f"{retained.lower()}.*SELECT.*DELETE"):
            await storage.initialize()
    finally:
        await storage.finalize()
    assert (await admin.query(f"SELECT id FROM {_TABLE} WHERE workspace='tenant_a'"))[
        "id"
    ] == "active"
    assert (await admin.query(f"SELECT id FROM {retained} WHERE workspace='tenant_a'"))[
        "id"
    ] == "kept"
    # No INSERT/UPDATE grants are needed on retained tables or DML on migration state.
    await admin.execute(f'GRANT DELETE ON {retained} TO "{role}"')
    storage = make_storage()
    try:
        await storage.initialize()
        result = await storage.drop_all_models()
        assert result["status"] == "success", result
        assert (
            await admin.query(f"SELECT id FROM {retained} WHERE workspace='tenant_a'")
            is None
        )
        assert (
            await admin.query(f"SELECT id FROM {_TABLE} WHERE workspace='tenant_a'")
            is None
        )
        assert (
            await admin.query(
                f"SELECT id FROM {retained} WHERE workspace='other_tenant'"
            )
        )["id"] == "other"
    finally:
        await storage.finalize()


@pytest.mark.pg_smoke
@pytest.mark.parametrize("provisioned", [False], indirect=True)
async def test_compatible_column_variants_accept_runtime_writes(provisioned):
    from lightrag.kg.postgres_impl import PGKVStorage

    admin, _ = provisioned
    await admin.execute("""
        ALTER TABLE lightrag_doc_full
          ALTER COLUMN content SET NOT NULL,
          ALTER COLUMN chunk_options SET NOT NULL,
          ALTER COLUMN chunk_options DROP DEFAULT,
          ALTER COLUMN meta SET NOT NULL,
          ALTER COLUMN meta SET DEFAULT '{}'::jsonb,
          ALTER COLUMN create_time DROP DEFAULT,
          ADD COLUMN content_vector text GENERATED ALWAYS AS ('audit') STORED NOT NULL
    """)
    storage = PGKVStorage(
        namespace="full_docs",
        workspace="compatible",
        global_config={},
        embedding_func=None,
    )
    try:
        await storage.initialize()
        await storage.upsert({"doc": {"content": "original", "parse_engine": None}})
        await storage.upsert({"doc": {"content": "updated"}})
        assert (await storage.get_by_id("doc"))["content"] == "updated"
        assert await admin.query("""SELECT meta='{}'::jsonb AS default_used,
            content_vector AS generated_extra, create_time IS NULL AS empty_time,
            chunk_options='{}'::jsonb AS json_written
            FROM lightrag_doc_full WHERE workspace='compatible' AND id='doc'""") == {
            "default_used": True,
            "generated_extra": "audit",
            "empty_time": True,
            "json_written": True,
        }
    finally:
        await storage.finalize()
