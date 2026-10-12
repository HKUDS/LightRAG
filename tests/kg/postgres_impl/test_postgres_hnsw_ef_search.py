"""Regression test: PGVectorStorage.query must scale hnsw.ef_search with top_k.

pgvector's HNSW index only explores ``hnsw.ef_search`` candidates during an
ANN search (default 40), independent of the query's ``LIMIT``. Left unset,
PGVectorStorage.query() asked a Postgres HNSW index for more rows than the
index actually searched for whenever a caller requested ``top_k`` above 40
(the AGENTS.md example config uses ``top_k=60``), silently returning fewer
matches than exist under the cosine threshold instead of the requested
count -- the same failure shape as MongoDB's hardcoded ``numCandidates``.

The fix passes a ``SET LOCAL hnsw.ef_search`` preamble, scaled to
``max(top_k, 40)``, through to the same transaction as the fetch whenever an
HNSW-family index is configured, and leaves non-HNSW index types untouched.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, patch

from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc


@pytest.fixture
def mock_pg_db():
    """Mock PostgreSQL database connection, capturing db.query() kwargs."""
    db = AsyncMock()
    db.workspace = "test_workspace"
    db.vector_index_type = None

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if multirows:
            return []
        return {"exists": False, "count": 0}

    db.query = AsyncMock(side_effect=mock_query)
    db.execute = AsyncMock(return_value=None)
    db.check_table_exists = AsyncMock(return_value=True)

    return db


@pytest.fixture(autouse=True)
def mock_data_init_lock():
    with patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock:
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        yield mock_lock


@pytest.fixture
def mock_client_manager(mock_pg_db):
    with patch("lightrag.kg.postgres_impl.ClientManager") as mock_manager:
        mock_manager.get_client = AsyncMock(return_value=mock_pg_db)
        mock_manager.release_client = AsyncMock()
        yield mock_manager


@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    return EmbeddingFunc(embedding_dim=768, func=embed_func, model_name="test_model")


async def _make_storage(mock_embedding_func, index_type):
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }
    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )
    return storage


@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", ["HNSW", "HNSW_HALFVEC"])
@pytest.mark.parametrize(
    "top_k, expected_ef_search",
    [
        (60, 60),  # above pgvector's default: must be raised to top_k
        (100, 100),  # AGENTS.md example top_k
        (10, 40),  # below pgvector's default: floored at 40, never lowered
    ],
)
async def test_query_scales_ef_search_with_top_k(
    mock_client_manager,
    mock_pg_db,
    mock_embedding_func,
    index_type,
    top_k,
    expected_ef_search,
):
    mock_pg_db.vector_index_type = index_type
    storage = await _make_storage(mock_embedding_func, index_type)
    await storage.initialize()

    query_embedding = [0.1] * 768
    await storage.query("test query", top_k=top_k, query_embedding=query_embedding)

    assert mock_pg_db.query.called
    call_kwargs = mock_pg_db.query.call_args.kwargs
    preamble = call_kwargs.get("preamble")
    assert preamble is not None, (
        f"expected a hnsw.ef_search preamble for index type {index_type}, got none"
    )
    assert f"hnsw.ef_search = {expected_ef_search}" in preamble


@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", [None, "IVFFLAT", "VCHORDRQ"])
async def test_query_leaves_non_hnsw_index_types_untouched(
    mock_client_manager, mock_pg_db, mock_embedding_func, index_type
):
    """Non-HNSW index types must not receive an hnsw.ef_search preamble."""
    mock_pg_db.vector_index_type = index_type
    storage = await _make_storage(mock_embedding_func, index_type)
    await storage.initialize()

    query_embedding = [0.1] * 768
    await storage.query("test query", top_k=100, query_embedding=query_embedding)

    assert mock_pg_db.query.called
    call_kwargs = mock_pg_db.query.call_args.kwargs
    assert call_kwargs.get("preamble") is None
