"""PGVectorStorage.get_by_id/get_by_ids must not leak the embedding column.

The SQL-fallback path used ``SELECT *`` and returned ``content_vector`` — a
pgvector value the asyncpg codec materializes as a numpy array, which is not
JSON-serializable. ``get_entity_info(include_vector_data=True)`` puts that row
into the API response, so ``POST /graph/entity/edit`` raised a 500 during
response serialization when every storage was PostgreSQL. Both read paths must
strip ``content_vector`` to match the buffered shape and the other vector
backends.
"""

import asyncio

import numpy as np
import pytest
from unittest.mock import AsyncMock

from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


async def _embed(texts, **kwargs):
    return np.array([[0.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def _make_storage(row):
    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
        workspace="ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=3, func=_embed, model_name="test_model"
        ),
    )

    async def _query(sql, params, multirows=False):
        return [row] if multirows else row

    storage.db = AsyncMock()
    storage.db.query = AsyncMock(side_effect=_query)
    storage._flush_lock = asyncio.Lock()
    return storage


def _row():
    # content_vector is a numpy array in production (pgvector codec).
    return {
        "id": "ent-1",
        "content": "Alice\ndescription",
        "entity_name": "Alice",
        "source_id": "chunk-1",
        "file_path": "doc.txt",
        "content_vector": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "created_at": 1,
    }


@pytest.mark.asyncio
async def test_get_by_id_strips_content_vector():
    storage = _make_storage(_row())

    result = await storage.get_by_id("ent-1")

    assert result is not None
    assert "content_vector" not in result
    assert result["content"] == "Alice\ndescription"
    assert result["id"] == "ent-1"


@pytest.mark.asyncio
async def test_get_by_ids_strips_content_vector():
    storage = _make_storage(_row())

    results = await storage.get_by_ids(["ent-1"])

    assert len(results) == 1
    assert results[0] is not None
    assert "content_vector" not in results[0]
    assert results[0]["content"] == "Alice\ndescription"
