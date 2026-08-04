from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import ClientManager


@pytest.fixture(autouse=True)
def reset_client_manager_state():
    """Reset the process-wide pool state BEFORE and AFTER each test.

    Resetting only on entry leaks this file's MagicMock pool (with ref_count=2
    and a pinned vector signature) into the rest of the session, where the next
    real PostgreSQL storage is then refused for "incompatible vector settings".
    """
    pristine = {"db": None, "ref_count": 0, "vector_signature": None}
    ClientManager._instances = dict(pristine)
    yield
    ClientManager._instances = dict(pristine)


def test_pg_vector_storage_enables_vector() -> None:
    config = ClientManager.get_config("PGVectorStorage")
    assert config["enable_vector"] is True


def test_non_pg_vector_storage_disables_vector() -> None:
    config = ClientManager.get_config("NanoVectorDBStorage")
    assert config["enable_vector"] is False


def test_milvus_storage_disables_vector() -> None:
    config = ClientManager.get_config("MilvusVectorDBStorage")
    assert config["enable_vector"] is False


def test_qdrant_storage_disables_vector() -> None:
    config = ClientManager.get_config("QdrantVectorDBStorage")
    assert config["enable_vector"] is False


def test_none_vector_storage_disables_vector() -> None:
    """An unspecified vector backend must NOT demand pgvector.

    This used to default to True — the last surviving default of the removed
    POSTGRES_ENABLE_VECTOR env var. It made "I don't know which vector backend is
    in use" mean "require an extension", which is why PGTableGraphStorage had to
    pass a sentinel backend name to stay installable on stock PostgreSQL and why
    tools/rebuild_vdb.py had to populate vector_storage defensively.
    """
    config = ClientManager.get_config(None)
    assert config["enable_vector"] is False


def test_no_args_disables_vector() -> None:
    config = ClientManager.get_config()
    assert config["enable_vector"] is False


def test_only_pg_vector_storage_enables_vector() -> None:
    """Exhaustive: pgvector is enabled for exactly one backend name."""
    from lightrag.kg import STORAGE_IMPLEMENTATIONS

    for name in STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]:
        expected = name == "PGVectorStorage"
        assert ClientManager.get_config(name)["enable_vector"] is expected, name


@pytest.mark.asyncio
async def test_get_client_reuses_shared_pool_for_same_vector_settings() -> None:
    db = MagicMock()
    db.initdb = AsyncMock()
    db.check_tables = AsyncMock()

    with patch("lightrag.kg.postgres_impl.PostgreSQLDB", return_value=db) as db_cls:
        first = await ClientManager.get_client("PGVectorStorage")
        second = await ClientManager.get_client("PGVectorStorage")

    assert first is db
    assert second is db
    assert ClientManager._instances["ref_count"] == 2
    db_cls.assert_called_once()
    db.initdb.assert_awaited_once()
    db.check_tables.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_client_rejects_conflicting_vector_storage_settings() -> None:
    db = MagicMock()
    db.initdb = AsyncMock()
    db.check_tables = AsyncMock()

    with patch("lightrag.kg.postgres_impl.PostgreSQLDB", return_value=db):
        await ClientManager.get_client("NanoVectorDBStorage")

        with pytest.raises(RuntimeError, match="process-wide"):
            await ClientManager.get_client("PGVectorStorage")


@pytest.mark.asyncio
async def test_pg_vector_storage_declares_its_own_pgvector_requirement() -> None:
    """PGVectorStorage must not ask global_config whether it needs pgvector.

    It IS the pgvector backend, so it always does. Since an unspecified vector
    backend no longer implies pgvector, reading the ambient value would resolve to
    None under a bare global_config and hand back a pool with no vector codec that
    only fails later, on the first vector query.
    """
    from lightrag.kg.postgres_impl import PGVectorStorage

    storage = object.__new__(PGVectorStorage)
    storage.db = None
    storage.workspace = "test"
    storage.namespace = "chunks"
    storage.global_config = {}  # deliberately names no vector backend
    # Only needed so setup_table's argument list can be evaluated; the call itself
    # is patched out below.
    storage.table_name = "LIGHTRAG_VDB_CHUNKS"
    storage.legacy_table_name = "LIGHTRAG_VDB_CHUNKS"
    storage.embedding_func = MagicMock(embedding_dim=8)
    storage._flush_lock = MagicMock()

    db = MagicMock()
    db.workspace = None

    @asynccontextmanager
    async def _lock():
        yield

    with (
        patch(
            "lightrag.kg.postgres_impl.ClientManager.get_client",
            new=AsyncMock(return_value=db),
        ) as get_client,
        patch("lightrag.kg.postgres_impl.get_data_init_lock", return_value=_lock()),
        patch.object(PGVectorStorage, "setup_table", new=AsyncMock(), create=True),
    ):
        await PGVectorStorage.initialize(storage)

    get_client.assert_awaited_once_with(vector_storage="PGVectorStorage")
    assert ClientManager.get_config("PGVectorStorage")["enable_vector"] is True
