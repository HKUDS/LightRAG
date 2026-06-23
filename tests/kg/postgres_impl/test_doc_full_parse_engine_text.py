"""LIGHTRAG_DOC_FULL.parse_engine must be TEXT (Phase 2).

The parse_engine field may carry an encoded engine-parameter directive
(``mineru(page_range=1-3,language=en)``) longer than the original VARCHAR(32),
so the CREATE DDL uses TEXT and the migration widens an existing VARCHAR(32)
column.  (A real overflow round-trip against live Postgres is the separate
@pytest.mark.integration test; these mocked assertions cannot run real SQL.)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import TABLES, PostgreSQLDB


def test_create_ddl_uses_text_for_parse_engine():
    ddl = TABLES["LIGHTRAG_DOC_FULL"]["ddl"]
    assert "parse_engine TEXT" in ddl
    assert "parse_engine VARCHAR(32)" not in ddl


@pytest.mark.asyncio
async def test_migration_widens_existing_varchar_column_to_text():
    db = PostgreSQLDB.__new__(PostgreSQLDB)  # skip __init__/connection setup

    executed: list[str] = []

    async def fake_query(sql, *args, **kwargs):
        # All columns already present -> ADD COLUMN loop is a no-op; the
        # parse_engine column reports the legacy VARCHAR type so the widening
        # ALTER fires.
        if "information_schema.columns" in sql and "column_name = ANY" in sql:
            return [
                {"column_name": c}
                for c in (
                    "sidecar_location",
                    "parse_format",
                    "content_hash",
                    "process_options",
                    "chunk_options",
                    "parse_engine",
                )
            ]
        if "information_schema.columns" in sql:  # the parse_engine type probe
            return {"data_type": "character varying"}
        return None

    async def fake_execute(sql, *args, **kwargs):
        executed.append(sql)

    db.query = AsyncMock(side_effect=fake_query)
    db.execute = AsyncMock(side_effect=fake_execute)

    await db._migrate_doc_full_add_pipeline_fields()

    assert any("ALTER COLUMN parse_engine TYPE TEXT" in sql for sql in executed), (
        executed
    )


@pytest.mark.asyncio
async def test_migration_skips_alter_when_already_text():
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    executed: list[str] = []

    async def fake_query(sql, *args, **kwargs):
        if "information_schema.columns" in sql and "column_name = ANY" in sql:
            return [
                {"column_name": c}
                for c in (
                    "sidecar_location",
                    "parse_format",
                    "content_hash",
                    "process_options",
                    "chunk_options",
                    "parse_engine",
                )
            ]
        if "information_schema.columns" in sql:
            return {"data_type": "text"}
        return None

    db.query = AsyncMock(side_effect=fake_query)
    db.execute = AsyncMock(side_effect=lambda sql, *a, **k: executed.append(sql))

    await db._migrate_doc_full_add_pipeline_fields()

    assert not any("ALTER COLUMN parse_engine" in sql for sql in executed)
