from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import ClientManager


@pytest.fixture(autouse=True)
def reset_client_manager_state() -> None:
    ClientManager._instances = {"db": None, "ref_count": 0, "vector_signature": None}


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


def test_none_vector_storage_defaults_to_true() -> None:
    # Backward compatibility: when vector_storage is unknown (None), default to True.
    config = ClientManager.get_config(None)
    assert config["enable_vector"] is True


def test_no_args_defaults_to_true() -> None:
    # Backward compatibility: calling without arguments preserves prior behavior.
    config = ClientManager.get_config()
    assert config["enable_vector"] is True


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
