from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB


@pytest.mark.asyncio
async def test_check_table_exists_uses_search_path_visible_regclass():
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.query = AsyncMock(return_value={"exists": True})

    assert await db.check_table_exists("LIGHTRAG_DOC_FULL") is True

    db.query.assert_awaited_once_with(
        "SELECT to_regclass($1) IS NOT NULL AS exists",
        ["lightrag_doc_full"],
    )


@pytest.mark.asyncio
async def test_full_entity_relation_migration_reuses_table_exists_helper():
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.check_table_exists = AsyncMock(return_value=True)
    db.query = AsyncMock()
    db.execute = AsyncMock()

    await db._migrate_create_full_entities_relations_tables()

    assert [call.args[0] for call in db.check_table_exists.await_args_list] == [
        "LIGHTRAG_FULL_ENTITIES",
        "LIGHTRAG_FULL_RELATIONS",
    ]
    db.query.assert_not_called()
    db.execute.assert_not_called()
