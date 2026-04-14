from lightrag.kg.postgres_impl import ClientManager


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
