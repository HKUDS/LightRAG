"""Pre-digest PostgreSQL vector tables are renamed onto the digested name.

The rename takes the indexes whose names are derived from the table, so the
first start after the digest does not build a second vector index. A rename
that loses the table to another model does not raise; that model keeps the
vectors and this one creates an empty table.
"""

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.kg.postgres_impl import (
    PGVectorStorage,
    _adopt_pre_digest_vector_table,
    _safe_index_name,
)
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


def _embedding(model_name="test_model", dim=768):
    async def embed(texts, **kwargs):
        return [[0.1] * dim for _ in texts]

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


def _storage(model_name="test_model", dim=768):
    return PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        workspace="test_ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
        },
        embedding_func=_embedding(model_name, dim),
    )


@pytest.fixture
def _init_lock():
    with patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock:
        mock_lock.return_value = AsyncMock()
        yield


@pytest.mark.asyncio
async def test_initialize_renames_pre_digest_table_and_indexes(_init_lock):
    storage = _storage()
    pre_digest = storage._pre_digest_table_name()
    digested = storage.table_name
    assert pre_digest and pre_digest != digested

    tables = {pre_digest.lower()}
    old_index = _safe_index_name(pre_digest, "hnsw_cosine")
    new_index = _safe_index_name(digested, "hnsw_cosine")
    indexes = {old_index}
    statements: list[str] = []

    db = AsyncMock()
    db.workspace = None
    db.vector_index_type = None

    async def check_table_exists(name):
        lowered = name.lower()
        return lowered in tables or lowered in indexes

    async def execute(sql, data=None, **kwargs):
        statements.append(sql)
        if sql.startswith("ALTER INDEX"):
            indexes.discard(old_index)
            indexes.add(new_index)
            return None
        if sql.startswith("ALTER TABLE"):
            tables.remove(pre_digest.lower())
            tables.add(digested.lower())
            return None
        return None

    db.check_table_exists = AsyncMock(side_effect=check_table_exists)
    db.execute = AsyncMock(side_effect=execute)
    db.query = AsyncMock(return_value={"count": 2})

    with patch(
        "lightrag.kg.postgres_impl.ClientManager.get_client",
        AsyncMock(return_value=db),
    ):
        await storage.initialize()

    index_at = next(i for i, sql in enumerate(statements) if "ALTER INDEX" in sql)
    table_at = next(i for i, sql in enumerate(statements) if "ALTER TABLE" in sql)
    assert index_at < table_at
    assert old_index in statements[index_at]
    assert new_index in statements[index_at]
    assert pre_digest in statements[table_at]
    assert digested in statements[table_at]
    assert digested.lower() in tables
    assert pre_digest.lower() not in tables


@pytest.mark.asyncio
async def test_existing_digested_table_is_not_renamed(_init_lock):
    storage = _storage()
    pre_digest = storage._pre_digest_table_name()
    digested = storage.table_name
    present = {pre_digest.lower(), digested.lower()}

    db = AsyncMock()
    db.workspace = None
    db.vector_index_type = None
    db.check_table_exists = AsyncMock(side_effect=lambda name: name.lower() in present)
    db.execute = AsyncMock()
    db.query = AsyncMock(return_value={"count": 1})

    with patch(
        "lightrag.kg.postgres_impl.ClientManager.get_client",
        AsyncMock(return_value=db),
    ):
        await storage.initialize()

    assert db.execute.await_count == 0


@pytest.mark.asyncio
async def test_lost_rename_leaves_the_caller_to_create_an_empty_table():
    storage = _storage("vendor/model:v1", dim=1024)
    pre_digest = storage._pre_digest_table_name()
    digested = storage.table_name
    state = {"old": True}

    async def check_table_exists(name):
        lowered = name.lower()
        if lowered == digested.lower():
            return False
        if lowered == pre_digest.lower():
            return state["old"]
        return False

    async def execute(sql, data=None, **kwargs):
        if sql.startswith("ALTER TABLE"):
            state["old"] = False
            raise RuntimeError("renamed by the other model")
        return None

    db = AsyncMock()
    db.check_table_exists = AsyncMock(side_effect=check_table_exists)
    db.execute = AsyncMock(side_effect=execute)

    adopted = await _adopt_pre_digest_vector_table(
        db, pre_digest, digested, "vendor/model:v1"
    )

    assert adopted is False
    assert state["old"] is False
