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


def _storage(rows, model_name="model-b", dim=768):
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    storage = PGVectorStorage(
        namespace="chunks",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=dim, func=embed, model_name=model_name
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


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("lightrag_vdbxchunks_model_a_1536d", id="like-wildcard-match"),
        pytest.param('lightrag_vdb_chunks_a"; drop table x; --_8d', id="quoted-name"),
        pytest.param("lightrag_vdb_chunks_Model_8d", id="mixed-case"),
    ],
)
async def test_a_name_the_suffix_fold_cannot_produce_is_never_formatted_into_sql(name):
    """LIKE reads the base name's own underscores as wildcards, and the table
    name is formatted into the DELETE: only the exact prefix and the folded
    identifier shape may reach it."""
    storage = _storage([{"table_name": name}])

    result = await storage.drop_sibling_models()

    assert result["status"] == "success"
    storage.db.execute.assert_not_called()


async def test_does_not_wait_on_the_flush_lock():
    """No instance buffers writes for a sibling table, so a flush in progress
    on the active one is no reason to hold the sweep."""
    storage = _storage([{"table_name": "lightrag_vdb_chunks_model_a_1536d"}])
    await storage._flush_lock.acquire()
    try:
        result = await asyncio.wait_for(storage.drop_sibling_models(), timeout=5)
    finally:
        storage._flush_lock.release()

    assert result["status"] == "success"
    assert storage.db.execute.await_count == 1


def test_model_container_name_is_the_table_that_model_would_attach_to():
    """Compared as the catalog reports it: unquoted identifiers fold to lower case."""
    other = _storage([], model_name="Model-A", dim=1536)
    assert (
        _storage([]).model_container_name("Model-A", 1536) == other.table_name.lower()
    )


async def test_siblings_are_found_by_search_path_visibility_not_one_schema():
    """Every other statement names these tables unqualified, so they resolve
    through the whole search path. ``current_schema()`` is only its FIRST
    schema: with a ``"$user"`` schema or AGE's ``ag_catalog`` ahead of
    ``public`` it is not where the tables are, the listing comes back empty,
    and a sibling that was never looked at is reported as a clean sweep.
    Checked against a real PostgreSQL for the PR; this pins the predicate."""
    storage = _storage([])

    await storage.drop_sibling_models()

    sql = storage.db.query.call_args.args[0]
    assert "pg_table_is_visible" in sql
    assert "current_schema" not in sql
    assert "information_schema" not in sql
