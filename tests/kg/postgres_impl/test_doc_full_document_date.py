"""PostgreSQL full-doc storage carries the optional document date."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import (
    SQL_TEMPLATES,
    TABLES,
    PGKVStorage,
    PostgreSQLDB,
)
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


def test_full_doc_schema_and_reads_preserve_reduced_precision_text():
    ddl = TABLES["LIGHTRAG_DOC_FULL"]["ddl"]
    assert "document_date VARCHAR(10) NULL" in ddl

    for key in ("get_by_id_full_docs", "get_by_ids_full_docs"):
        normalized = " ".join(SQL_TEMPLATES[key].split())
        assert "parse_engine, document_date FROM LIGHTRAG_DOC_FULL" in normalized
        assert "TO_CHAR(document_date" not in SQL_TEMPLATES[key]


@pytest.mark.asyncio
async def test_full_doc_migration_adds_document_date_to_existing_table():
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    executed: list[str] = []

    async def _query(sql, *_args, **_kwargs):
        if "column_name = ANY" in sql:
            return [
                {"column_name": column}
                for column in (
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

    async def _execute(sql, *_args, **_kwargs):
        executed.append(sql)

    db.query = AsyncMock(side_effect=_query)
    db.execute = AsyncMock(side_effect=_execute)

    await db._migrate_doc_full_add_pipeline_fields()

    assert any(
        "ALTER TABLE LIGHTRAG_DOC_FULL ADD COLUMN document_date VARCHAR(10) NULL" in sql
        for sql in executed
    )


@pytest.mark.asyncio
async def test_undated_full_doc_read_omits_document_date_key():
    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = NameSpace.KV_STORE_FULL_DOCS
    storage.workspace = "test"
    storage.db = SimpleNamespace(
        query=AsyncMock(
            return_value={
                "id": "doc-undated",
                "content": "legacy content",
                "chunk_options": {},
                "document_date": None,
            }
        )
    )

    result = await storage.get_by_id("doc-undated")

    assert "document_date" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("document_date", ["2018", "2018-10", "2018-10-01"])
async def test_dated_full_doc_read_preserves_precision(document_date):
    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = NameSpace.KV_STORE_FULL_DOCS
    storage.workspace = "test"
    storage.db = SimpleNamespace(
        query=AsyncMock(
            return_value={
                "id": "doc-dated",
                "content": "historical content",
                "chunk_options": {},
                "document_date": document_date,
            }
        )
    )

    result = await storage.get_by_id("doc-dated")

    assert result["document_date"] == document_date
