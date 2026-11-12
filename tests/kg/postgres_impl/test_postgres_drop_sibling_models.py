"""PostgreSQL ``drop_sibling_models`` clears this workspace from other models' tables."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)


def _storage(rows):
    async def embed(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    storage = PGVectorStorage(
        namespace="chunks",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768, func=embed, model_name="model-b"
        ),
        workspace="ws",
    )
    storage.db = MagicMock()
    storage.db.query = AsyncMock(return_value=rows)
    storage.db.execute = AsyncMock()
    storage._flush_lock = asyncio.Lock()
    return storage


async def test_deletes_this_workspaces_rows_from_each_sibling_table():
    storage = _storage(
        [
            {"table_name": "lightrag_vdb_chunks_model_a_1536d"},
            {"table_name": "lightrag_vdb_chunks_model_c_1024d"},
        ]
    )

    result = await storage.drop_sibling_models()

    assert result["status"] == "success"
    sqls = [c.args[0] for c in storage.db.execute.call_args_list]
    assert [s.split()[2] for s in sqls] == [
        "lightrag_vdb_chunks_model_a_1536d",
        "lightrag_vdb_chunks_model_c_1024d",
    ]
    assert all(
        c.args[1] == {"workspace": "ws"} for c in storage.db.execute.call_args_list
    )
    # The active table is excluded in the query, not filtered afterwards.
    assert storage.db.query.call_args.args[1][1] == storage.table_name.lower()


async def test_ignores_tables_without_a_model_dimension_suffix():
    storage = _storage([{"table_name": "lightrag_vdb_chunks_backup"}])

    result = await storage.drop_sibling_models()

    assert result["status"] == "success"
    storage.db.execute.assert_not_called()


async def test_reports_a_backend_failure():
    storage = _storage([{"table_name": "lightrag_vdb_chunks_model_a_1536d"}])
    storage.db.execute.side_effect = RuntimeError("boom")

    result = await storage.drop_sibling_models()

    assert result["status"] == "error"
    assert "boom" in result["message"]
