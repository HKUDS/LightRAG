"""Read-only PostgreSQL attach checks. See docs/PostgreSQLNoDDL.md.

This is a validator for the SQL used by the PostgreSQL backends, not a schema
manager. Keep the column requirements aligned with postgres_impl.TABLES and
pgtable_impl._DDL. Resolve each unquoted runtime name through to_regclass first;
all subsequent catalog checks are scoped to that relation OID.
"""

from typing import Any

from ..utils import logger

_ROW = {
    "workspace": "varchar:255",
    "id": "varchar:255",
    "create_time": "timestamp",
    "update_time": "timestamp",
}
COMMON_TABLES = {
    "LIGHTRAG_DOC_FULL": {
        **_ROW,
        "doc_name": "varchar:1024",
        "content": "text",
        "meta": "jsonb",
        "sidecar_location": "text",
        "parse_format": "varchar:32",
        "content_hash": "text",
        "process_options": "text",
        "chunk_options": "jsonb",
        "parse_engine": "text",
    },
    "LIGHTRAG_DOC_CHUNKS": {
        **_ROW,
        "full_doc_id": "varchar:256",
        "chunk_order_index": "int4",
        "tokens": "int4",
        "content": "text",
        "file_path": "text",
        "llm_cache_list": "jsonb",
        "heading": "jsonb",
        "sidecar": "jsonb",
    },
    "LIGHTRAG_LLM_CACHE": {
        **_ROW,
        "original_prompt": "text",
        "return_value": "text",
        "chunk_id": "varchar:255",
        "cache_type": "varchar:32",
        "queryparam": "jsonb",
    },
    "LIGHTRAG_CONFIG": {**_ROW, "id": "text", "value": "jsonb"},
    "LIGHTRAG_DOC_STATUS": {
        "workspace": "varchar:255",
        "id": "varchar:255",
        "content_summary": "varchar:255",
        "content_length": "int4",
        "chunks_count": "int4",
        "status": "varchar:64",
        "file_path": "text",
        "chunks_list": "jsonb",
        "track_id": "varchar:255",
        "metadata": "jsonb",
        "error_msg": "text",
        "content_hash": "text",
        "created_at": "timestamp",
        "updated_at": "timestamp",
    },
    "LIGHTRAG_FULL_ENTITIES": {**_ROW, "entity_names": "jsonb", "count": "int4"},
    "LIGHTRAG_FULL_RELATIONS": {**_ROW, "relation_pairs": "jsonb", "count": "int4"},
    "LIGHTRAG_ENTITY_CHUNKS": {
        **_ROW,
        "id": "varchar:512",
        "chunk_ids": "jsonb",
        "count": "int4",
    },
    "LIGHTRAG_RELATION_CHUNKS": {
        **_ROW,
        "id": "varchar:512",
        "chunk_ids": "jsonb",
        "count": "int4",
    },
}
VECTOR_TABLES = {
    "LIGHTRAG_VDB_CHUNKS": {
        **_ROW,
        "full_doc_id": "varchar:256",
        "chunk_order_index": "int4",
        "tokens": "int4",
        "content": "text",
        "file_path": "text",
    },
    "LIGHTRAG_VDB_ENTITY": {
        **_ROW,
        "entity_name": "varchar:512",
        "content": "text",
        "chunk_ids": "varchar[]:255",
        "file_path": "text",
    },
    "LIGHTRAG_VDB_RELATION": {
        **_ROW,
        "source_id": "varchar:512",
        "target_id": "varchar:512",
        "content": "text",
        "chunk_ids": "varchar[]:255",
        "file_path": "text",
    },
}

# These are properties of the runtime INSERT parameters, not copies of the DDL.
# A DEFAULT cannot help a parameter that explicitly binds SQL NULL.
_NULLABLE_WRITES = {
    "LIGHTRAG_DOC_FULL": {
        "sidecar_location",
        "parse_format",
        "content_hash",
        "process_options",
        "parse_engine",
    },
    "LIGHTRAG_DOC_CHUNKS": {"file_path"},
    "LIGHTRAG_LLM_CACHE": {"chunk_id", "queryparam"},
    "LIGHTRAG_DOC_STATUS": {
        "chunks_count",
        "track_id",
        "error_msg",
        "content_hash",
        "created_at",
        "updated_at",
    },
    "LIGHTRAG_VDB_CHUNKS": {"file_path"},
    "LIGHTRAG_VDB_ENTITY": {"file_path"},
    "LIGHTRAG_VDB_RELATION": {"file_path"},
}
_INSERT_OMITTED_COLUMNS = {
    "LIGHTRAG_DOC_FULL": {"meta", "create_time", "update_time"},
    "LIGHTRAG_LLM_CACHE": {"create_time", "update_time"},
    "LIGHTRAG_CONFIG": {"create_time", "update_time"},
}

_RELATION_SQL = """
SELECT c.oid, c.relkind::text AS relkind,
       has_schema_privilege(c.relnamespace, 'USAGE') AND
       has_table_privilege(c.oid, 'SELECT') AND
       has_table_privilege(c.oid, 'INSERT') AND
       has_table_privilege(c.oid, 'UPDATE') AND
       has_table_privilege(c.oid, 'DELETE') AS permitted
FROM pg_class c WHERE c.oid = to_regclass($1)
"""
_COLUMNS_SQL = """
SELECT a.attname, t.typname, n.nspname AS type_schema, a.atttypmod, a.attnotnull, a.atthasdef,
       a.attgenerated::text AS generated, a.attidentity::text AS identity,
       EXISTS (SELECT 1 FROM pg_depend d JOIN pg_extension e ON e.oid = d.refobjid
               WHERE d.classid = 'pg_type'::regclass AND d.objid = t.oid
                 AND d.deptype = 'e' AND e.extname = 'vector') AS vector_type
FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE a.attrelid = $1 AND a.attnum > 0 AND NOT a.attisdropped
"""
_INDEXES_SQL = """
SELECT i.indexrelid, i.indisprimary, i.indisunique, i.indimmediate,
       i.indisvalid AND i.indisready AND i.indislive AS usable,
       i.indoption::smallint[] AS options,
       am.amname, pg_get_expr(i.indpred, i.indrelid) AS predicate,
       ARRAY(SELECT a.attname::text
             FROM unnest(i.indkey) WITH ORDINALITY k(attnum, n)
             LEFT JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
             WHERE n <= i.indnkeyatts ORDER BY n) AS columns,
       ARRAY(SELECT op.opcname::text FROM unnest(i.indclass) WITH ORDINALITY k(opid, n)
             JOIN pg_opclass op ON op.oid = k.opid ORDER BY n) AS opclasses
FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_am am ON am.oid = c.relam WHERE i.indrelid = $1
"""
_FOREIGN_KEYS_SQL = """
SELECT c.confrelid, c.confdeltype::text AS confdeltype, c.convalidated,
       ARRAY(SELECT a.attname::text FROM unnest(c.conkey) WITH ORDINALITY k(attnum, n)
             JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
             ORDER BY n) AS columns,
       ARRAY(SELECT a.attname::text FROM unnest(c.confkey) WITH ORDINALITY k(attnum, n)
             JOIN pg_attribute a ON a.attrelid = c.confrelid AND a.attnum = k.attnum
             ORDER BY n) AS referenced_columns
FROM pg_constraint c WHERE c.conrelid = $1 AND c.contype = 'f'
"""
_VECTOR_CLEANUP_SQL = """
SELECT c.relkind::text AS relkind,
       has_schema_privilege(c.relnamespace, 'USAGE') AND
       has_column_privilege(c.oid, a.attnum, 'SELECT') AND
       has_table_privilege(c.oid, 'DELETE') AS permitted
FROM pg_class c LEFT JOIN pg_attribute a
  ON a.attrelid = c.oid AND a.attname = 'workspace' AND NOT a.attisdropped
WHERE c.oid = to_regclass($1)
"""


def _incompatible(table: str, detail: str) -> RuntimeError:
    return RuntimeError(
        f"POSTGRES_AUTO_MIGRATE=false: {table}: {detail}. Have an administrator "
        "provision or migrate this schema using this LightRAG version with "
        "POSTGRES_AUTO_MIGRATE=true, then restart the runtime. See docs/PostgreSQLNoDDL.md."
    )


def _compatible_column(actual: dict, expected: str) -> bool:
    if actual["type_schema"] != "pg_catalog":
        return False
    kind, _, minimum = expected.partition(":")
    typ, modifier = actual["typname"], actual["atttypmod"]
    if kind in ("text", "varchar"):
        return typ == "text" or (
            typ == "varchar"
            and (modifier == -1 or (bool(minimum) and modifier - 4 >= int(minimum)))
        )
    if kind == "varchar[]":
        return typ == "_text" or (
            typ == "_varchar" and (modifier == -1 or modifier - 4 >= int(minimum))
        )
    return typ == kind


def _index_matches(
    index: dict, columns: tuple[str, ...], method: str = "btree"
) -> bool:
    return bool(
        index["usable"]
        and index["amname"] == method
        and index["predicate"] is None
        and tuple(index["columns"][: len(columns)]) == columns
    )


async def validate_table(
    db: Any,
    table: str,
    columns: dict[str, str],
    primary_key: tuple[str, ...] = ("workspace", "id"),
    *,
    schema_table: str | None = None,
) -> tuple[int, dict[str, dict], list[dict]]:
    """Require compatible columns, an immediate primary key and CRUD access.

    Wider text columns and extra nullable/defaulted/generated columns are compatible.
    Explicit NULL parameters require nullable columns; omitted NOT NULL columns
    need a default or generation. Written columns must accept explicit values.
    Names of constraints/indexes are irrelevant; definitions and OIDs decide.
    Table privileges do not certify row-level security policy compatibility.
    """
    relation = await db.query(_RELATION_SQL, [table.lower()])
    if not relation:
        raise _incompatible(
            table, "required table is missing on the runtime search_path"
        )
    if relation["relkind"] not in ("r", "p"):
        raise _incompatible(
            table,
            f"expected an ordinary or partitioned table, got relkind={relation['relkind']}",
        )
    if not relation["permitted"]:
        raise _incompatible(
            table,
            "requires schema USAGE and table SELECT, INSERT, UPDATE, DELETE grants",
        )
    oid = relation["oid"]
    attributes = await db.query(_COLUMNS_SQL, [oid], multirows=True)
    actual = {row["attname"]: row for row in attributes}
    schema_table = (schema_table or table).upper()
    nullable_writes = _NULLABLE_WRITES.get(schema_table, set())
    omitted = _INSERT_OMITTED_COLUMNS.get(schema_table, set())
    known_columns = set(columns)
    if schema_table in VECTOR_TABLES:
        known_columns.add("content_vector")
    for name, expected in columns.items():
        if name not in actual:
            raise _incompatible(table, f"missing column {name} (expected {expected})")
        if not _compatible_column(actual[name], expected):
            raise _incompatible(
                table,
                f"column {name} expects {expected}, got {actual[name]['typname']} with typmod {actual[name]['atttypmod']}",
            )
    for name, column in actual.items():
        # content_vector is validated separately for its extension type/dimension,
        # but is still an explicit INSERT parameter.
        written = name in known_columns and (
            name not in omitted or name == "update_time"
        )
        if written and column["generated"]:
            raise _incompatible(
                table,
                f"column {name} is generated but runtime SQL writes it explicitly",
            )
        if written and column["identity"] == "a":
            raise _incompatible(
                table,
                f"column {name} is GENERATED ALWAYS identity but runtime SQL writes it explicitly",
            )
        if name in nullable_writes and column["attnotnull"]:
            raise _incompatible(
                table,
                f"column {name} must allow NULL because runtime SQL can bind it explicitly",
            )
        if (
            name in omitted
            and column["attnotnull"]
            and not (column["atthasdef"] or column["generated"] or column["identity"])
        ):
            raise _incompatible(
                table, f"omitted NOT NULL column {name} needs a default or generation"
            )
    for name, column in actual.items():
        if (
            name not in known_columns
            and column["attnotnull"]
            and not column["atthasdef"]
            and not column["generated"]
            and not column["identity"]
        ):
            raise _incompatible(table, f"extra required column {name} has no default")
    indexes = await db.query(_INDEXES_SQL, [oid], multirows=True)
    if not any(
        i["indisprimary"]
        and i["indimmediate"]
        and i["usable"]
        and i["predicate"] is None
        and set(i["columns"]) == set(primary_key)
        for i in indexes
    ):
        raise _incompatible(
            table, f"requires a usable nondeferrable PRIMARY KEY {primary_key}"
        )
    if table.lower() == "lightrag_doc_status" and not any(
        _index_matches(i, ("workspace", "status", "created_at", "id"))
        and i["options"][:4] == [0, 0, 2, 0]
        for i in indexes
    ):
        # NULLS FIRST is part of the scheduling index's ordering contract.
        # As in automatic mode, absence degrades the plan but never correctness.
        logger.warning(
            "PostgreSQL: LIGHTRAG_DOC_STATUS scheduling index is missing or unusable; "
            "scheduling pages may scan and sort the workspace. Have an administrator "
            "create a btree index on (workspace, status, created_at NULLS FIRST, id)."
        )
    return oid, actual, indexes


async def validate_vector_table(
    db: Any,
    table: str,
    base_table: str,
    workspace: str,
    dimension: int,
    legacy: str,
) -> None:
    from .postgres_impl import PGVectorStorage, _pg_vector_sibling_tables

    _, columns, indexes = await validate_table(
        db, table, VECTOR_TABLES[base_table], schema_table=base_table
    )
    vector = columns.get("content_vector")
    expected_type = "halfvec" if db.vector_index_type == "HNSW_HALFVEC" else "vector"
    if not vector or not vector["vector_type"] or vector["typname"] != expected_type:
        raise _incompatible(
            table,
            f"content_vector must have pgvector type {expected_type}({dimension})",
        )
    if vector["atttypmod"] != dimension:
        # A physical definition cannot be repaired by deleting workspace rows.
        # The rebuild tool offers that destructive recovery for vector-space
        # refusals, so schema errors must never carry that exception type.
        raise _incompatible(
            table,
            f"content_vector requires declared dimension {dimension}, got {vector['atttypmod']}",
        )
    method = {
        "HNSW": "hnsw",
        "HNSW_HALFVEC": "hnsw",
        "IVFFLAT": "ivfflat",
        "VCHORDRQ": "vchordrq",
    }.get(db.vector_index_type)
    if db.vector_index_type and not method:
        raise _incompatible(
            table, f"unsupported POSTGRES_VECTOR_INDEX_TYPE={db.vector_index_type}"
        )
    if method and not any(
        _index_matches(i, ("content_vector",), method)
        and len(i["columns"]) == 1
        and i["opclasses"] == [f"{expected_type}_cosine_ops"]
        for i in indexes
    ):
        raise _incompatible(
            table,
            f"requires a usable nonpartial {method} index on content_vector with {expected_type}_cosine_ops",
        )
    # Legacy backups and inert markers are accepted exactly as in setup_table.
    # A live source with an empty target or a marker must be migrated by an admin.
    legacy_exists = bool(
        legacy
        and legacy.lower() != table.lower()
        and await db.check_table_exists(legacy)
    )
    cleanup_tables = await _pg_vector_sibling_tables(db, base_table, table)
    if legacy_exists:
        cleanup_tables.append(legacy.lower())
    for retained in cleanup_tables:
        access = await db.query(_VECTOR_CLEANUP_SQL, [retained])
        if not access or access["relkind"] not in ("r", "p") or not access["permitted"]:
            raise _incompatible(
                retained,
                "workspace cleanup requires a table with schema USAGE, SELECT on workspace and DELETE",
            )
    if legacy_exists:
        source = await db.query(
            f"SELECT EXISTS (SELECT 1 FROM {legacy} WHERE workspace=$1) AS populated",
            [workspace],
        )
        if source["populated"]:
            target = await db.query(
                f"SELECT EXISTS (SELECT 1 FROM {table} WHERE workspace=$1) AS populated",
                [workspace],
            )
            if not target[
                "populated"
            ] or await PGVectorStorage._pg_migration_in_progress(db, table, workspace):
                raise _incompatible(
                    table,
                    f"legacy vector migration from {legacy} is incomplete for workspace {workspace!r}",
                )
            logger.warning(
                "PostgreSQL: retained legacy table %s has rows for workspace %r; "
                "No-DDL mode leaves this backup unchanged. See docs/design/PostgresLegacyVectorMigration.md.",
                legacy,
                workspace,
            )


async def validate_graph_tables(db: Any, workspace: str, namespace: str) -> None:
    common = {
        "workspace": "text",
        "namespace": "text",
        "properties": "jsonb",
        "updated_at": "timestamptz",
    }
    nodes_oid, _, _ = await validate_table(
        db,
        "lightrag_graph_nodes",
        {**common, "id": "text"},
        ("workspace", "namespace", "id"),
    )
    edges_oid, _, indexes = await validate_table(
        db,
        "lightrag_graph_edges",
        {**common, "src_id": "text", "tgt_id": "text"},
        ("workspace", "namespace", "src_id", "tgt_id"),
    )
    constraints = await db.query(_FOREIGN_KEYS_SQL, [edges_oid], multirows=True)
    for endpoint in ("src_id", "tgt_id"):
        if not any(
            c["confrelid"] == nodes_oid
            and c["confdeltype"] == "c"
            and c["convalidated"]
            and dict(zip(c["columns"], c["referenced_columns"]))
            == {"workspace": "workspace", "namespace": "namespace", endpoint: "id"}
            for c in constraints
        ):
            raise _incompatible(
                "lightrag_graph_edges",
                f"requires a validated FOREIGN KEY (workspace, namespace, {endpoint}) referencing resolved lightrag_graph_nodes(workspace, namespace, id) ON DELETE CASCADE",
            )
    if not any(
        _index_matches(i, ("workspace", "namespace", "tgt_id")) for i in indexes
    ):
        raise _incompatible(
            "lightrag_graph_edges",
            "requires a usable btree index beginning with (workspace, namespace, tgt_id)",
        )
    noncanonical = await db.query(
        """SELECT EXISTS (
        SELECT 1 FROM lightrag_graph_edges WHERE workspace=$1 AND namespace=$2
        AND src_id COLLATE "C" > tgt_id COLLATE "C") AS needed""",
        [workspace, namespace],
    )
    if noncanonical["needed"]:
        raise _incompatible(
            "lightrag_graph_edges",
            f"noncanonical edges require normalization for workspace {workspace!r}, namespace {namespace!r}",
        )
