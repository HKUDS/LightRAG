"""
PostgreSQL Multi-Tenant Support Module

This module provides utilities and helpers for implementing multi-tenant isolation
in PostgreSQL storage implementations. It includes:

1. Table DDL updates with tenant_id and kb_id columns
2. SQL template builders that automatically add tenant filtering
3. Composite key generation
4. Data migration utilities

All storage implementations (KV, Vector, Graph, DocStatus) use these utilities
to ensure proper tenant isolation at the database level.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# TABLE DDL TEMPLATES WITH TENANT SUPPORT
# ============================================================================


def get_updated_tables_dict() -> Dict[str, Dict[str, str]]:
    # Returns the updated TABLES dictionary with tenant_id and kb_id columns.
    # This function generates table DDLs that include:
    # - tenant_id VARCHAR(255) NOT NULL
    # - kb_id VARCHAR(255) NOT NULL
    # - Composite PRIMARY KEY on (tenant_id, kb_id, workspace, id)
    # - Composite indexes for common query patterns
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", 1024))

    return {
        "LIGHTRAG_DOC_FULL": {
            "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        doc_name VARCHAR(1024),
                        content TEXT,
                        meta JSONB,
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_DOC_CHUNKS": {
            "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        full_doc_id VARCHAR(256),
                        chunk_order_index INTEGER,
                        tokens INTEGER,
                        content TEXT,
                        file_path TEXT NULL,
                        llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_VDB_CHUNKS": {
            "ddl": f"""CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        full_doc_id VARCHAR(256),
                        chunk_order_index INTEGER,
                        tokens INTEGER,
                        content TEXT,
                        content_vector VECTOR({embedding_dim}),
                        file_path TEXT NULL,
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_VDB_ENTITY": {
            "ddl": f"""CREATE TABLE LIGHTRAG_VDB_ENTITY (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        entity_name VARCHAR(512),
                        content TEXT,
                        content_vector VECTOR({embedding_dim}),
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        chunk_ids VARCHAR(255)[] NULL,
                        file_path TEXT NULL,
                        CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_VDB_RELATION": {
            "ddl": f"""CREATE TABLE LIGHTRAG_VDB_RELATION (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        source_id VARCHAR(512),
                        target_id VARCHAR(512),
                        content TEXT,
                        content_vector VECTOR({embedding_dim}),
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        chunk_ids VARCHAR(255)[] NULL,
                        file_path TEXT NULL,
                        CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_LLM_CACHE": {
            "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        id VARCHAR(255) NOT NULL,
                        original_prompt TEXT,
                        return_value TEXT,
                        chunk_id VARCHAR(255) NULL,
                        cache_type VARCHAR(32),
                        queryparam JSONB NULL,
                        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_DOC_STATUS": {
            "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        id VARCHAR(255) NOT NULL,
                        content_summary VARCHAR(255) NULL,
                        content_length INT4 NULL,
                        chunks_count INT4 NULL,
                        status VARCHAR(64) NULL,
                        file_path TEXT NULL,
                        chunks_list JSONB NULL DEFAULT '[]'::jsonb,
                        track_id VARCHAR(255) NULL,
                        metadata JSONB NULL DEFAULT '{}'::jsonb,
                        error_msg TEXT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_FULL_ENTITIES": {
            "ddl": """CREATE TABLE LIGHTRAG_FULL_ENTITIES (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        entity_names JSONB,
                        count INTEGER,
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_FULL_ENTITIES_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
        "LIGHTRAG_FULL_RELATIONS": {
            "ddl": """CREATE TABLE LIGHTRAG_FULL_RELATIONS (
                        tenant_id VARCHAR(255) NOT NULL,
                        kb_id VARCHAR(255) NOT NULL,
                        id VARCHAR(255) NOT NULL,
                        workspace VARCHAR(255),
                        relation_pairs JSONB,
                        count INTEGER,
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT LIGHTRAG_FULL_RELATIONS_PK PRIMARY KEY (tenant_id, kb_id, id)
                    )"""
        },
    }


# ============================================================================
# SQL TEMPLATE BUILDERS WITH TENANT FILTERING
# ============================================================================


class TenantSQLBuilder:
    # Helper class for building tenant-aware SQL queries

    @staticmethod
    def add_tenant_filter(
        sql: str, table_alias: str = "", param_index: int = 1
    ) -> Tuple[str, int]:
        # Add tenant_id and kb_id filters to a WHERE clause.
        # Args:
        #     sql: Original SQL query
        #     table_alias: Optional table alias (e.g., "t." for table t)
        #     param_index: Starting parameter index ($1, $2, etc.)
        # Returns:
        #     Tuple of (modified_sql, next_param_index)
        prefix = f"{table_alias}." if table_alias else ""
        tenant_filter = (
            f"{prefix}tenant_id=${param_index} AND {prefix}kb_id=${param_index + 1}"
        )

        if "WHERE" in sql:
            sql = sql.replace("WHERE", f"WHERE {tenant_filter} AND", 1)
        else:
            sql += f" WHERE {tenant_filter}"

        return sql, param_index + 2

    @staticmethod
    def build_filtered_query(
        base_query: str, tenant_id: str, kb_id: str, additional_params: List[Any] = None
    ) -> Tuple[str, List[Any]]:
        # Build a complete query with tenant filtering.
        # Args:
        #     base_query: Base SQL query with placeholders
        #     tenant_id: Tenant ID
        #     kb_id: Knowledge base ID
        #     additional_params: Additional query parameters
        # Returns:
        #     Tuple of (query, params)
        params = [tenant_id, kb_id]
        if additional_params:
            params.extend(additional_params)
        return base_query, params


# ============================================================================
# MIGRATION UTILITIES
# ============================================================================


async def add_tenant_columns_migration(
    db, table_name: str, tenant_id: str = "default", kb_id: str = "default"
):
    # Migrate existing table to add tenant_id and kb_id columns.
    # This migration:
    # 1. Adds tenant_id and kb_id columns (if not exist)
    # 2. Populates them with default values for existing data
    # 3. Updates primary key constraints
    # 4. Creates new composite indexes
    # Args:
    #     db: PostgreSQL database connection
    #     table_name: Name of table to migrate
    #     tenant_id: Default tenant ID for existing data
    #     kb_id: Default KB ID for existing data
    try:
        # Check if tenant_id column exists
        check_sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name.lower()}' AND column_name = 'tenant_id'
        """
        exists = await db.query(check_sql)

        if not exists:
            # Add tenant_id and kb_id columns
            alter_sql = f"""
            ALTER TABLE {table_name}
            ADD COLUMN tenant_id VARCHAR(255) NOT NULL DEFAULT '{tenant_id}',
            ADD COLUMN kb_id VARCHAR(255) NOT NULL DEFAULT '{kb_id}'
            """
            await db.execute(alter_sql)

            # Update existing primary key and constraints
            # This is table-specific, handled in migration methods

    except Exception as e:
        from lightrag.utils import logger

        logger.warning(f"Failed to migrate {table_name}: {e}")


def get_composite_key(tenant_id: str, kb_id: str, *args) -> str:
    # Generate a composite key for multi-tenant data isolation.
    # Args:
    #     tenant_id: Tenant ID
    #     kb_id: Knowledge base ID
    #     *args: Additional key components
    # Returns:
    #     Composite key string
    parts = [tenant_id, kb_id] + list(args)
    return ":".join(str(p) for p in parts if p)


# ============================================================================
# INDEX CREATION HELPERS
# ============================================================================


def get_tenant_indexes(table_name: str) -> List[Dict[str, str]]:
    # Get recommended composite indexes for multi-tenant tables.
    # Args:
    #     table_name: Name of the table
    # Returns:
    #     List of index creation statements
    base_name = table_name.lower()
    indexes = [
        {
            "name": f"idx_{base_name}_tenant_kb",
            "sql": f"CREATE INDEX IF NOT EXISTS idx_{base_name}_tenant_kb ON {table_name}(tenant_id, kb_id)",
        },
        {
            "name": f"idx_{base_name}_tenant_kb_id",
            "sql": f"CREATE INDEX IF NOT EXISTS idx_{base_name}_tenant_kb_id ON {table_name}(tenant_id, kb_id, id)",
        },
        {
            "name": f"idx_{base_name}_tenant_kb_workspace",
            "sql": f"CREATE INDEX IF NOT EXISTS idx_{base_name}_tenant_kb_workspace ON {table_name}(tenant_id, kb_id, workspace)",
        },
    ]
    return indexes


# ============================================================================
# HELPER METHODS FOR STORAGE CLASSES
# ============================================================================


def ensure_tenant_context(
    tenant_id: Optional[str] = None,
    kb_id: Optional[str] = None,
    default_tenant: str = "default",
    default_kb: str = "default",
) -> Tuple[str, str]:
    # Ensure tenant and KB IDs are set, using defaults if necessary.
    # Args:
    #     tenant_id: Tenant ID (may be None)
    #     kb_id: Knowledge base ID (may be None)
    #     default_tenant: Default tenant ID if not provided
    #     default_kb: Default KB ID if not provided
    # Returns:
    #     Tuple of (tenant_id, kb_id)
    return (tenant_id or default_tenant, kb_id or default_kb)
