"""
Tests for Milvus index creation behavior

This test suite validates:
1. P1: build_index_params uses compatibility helper
2. P2: Vector index creation failures are surfaced to callers
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from lightrag.kg.milvus_impl import MilvusVectorDBStorage, MilvusIndexConfig


@pytest.mark.offline
class TestMilvusIndexCreation:
    """Test index creation behavior and error handling"""

    def test_vector_index_creation_failure_is_raised(self):
        """Test that vector index creation failures are raised to the caller (P2 fix)"""
        # Setup storage instance
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client and _get_index_params
        mock_client = MagicMock()
        mock_index_params = MagicMock()

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        # Mock _get_index_params to return a valid IndexParams
        with patch.object(storage, "_get_index_params", return_value=mock_index_params):
            # Mock build_index_params to return the mock_index_params
            with patch.object(
                storage.index_config,
                "build_index_params",
                return_value=mock_index_params,
            ):
                # Mock create_index to raise an exception (simulating index creation failure)
                mock_client.create_index.side_effect = Exception(
                    "Index creation failed"
                )

                # Verify that the exception is raised (not caught and logged)
                with pytest.raises(Exception, match="Index creation failed"):
                    storage._create_indexes_after_collection()

    def test_scalar_index_creation_failure_is_logged_not_raised(self):
        """Test that scalar index creation failures are logged but not raised (existing behavior)"""
        # Setup storage instance
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "AUTOINDEX",  # No custom vector index
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client and _get_index_params
        mock_client = MagicMock()
        mock_index_params = MagicMock()

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        # Mock _get_index_params to return a valid IndexParams for scalar indexes
        with patch.object(storage, "_get_index_params", return_value=mock_index_params):
            # Let vector AUTOINDEX creation succeed, then fail on scalar index creation
            mock_client.create_index.side_effect = [
                None,
                Exception("Scalar index creation failed"),
            ]

            # Verify that the function completes without raising (scalar index failures are logged)
            # This should not raise an exception
            storage._create_indexes_after_collection()

            # The function should complete successfully even though scalar index creation failed

    def test_build_index_params_uses_passed_index_params(self):
        """Test that build_index_params uses the passed index_params parameter (P1 fix)"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
            hnsw_m=32,
            hnsw_ef_construction=256,
        )

        mock_index_params = MagicMock()

        # Call build_index_params with the mock_index_params
        result = config.build_index_params(mock_index_params)

        # Verify that it used the passed index_params
        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once()

    def test_build_index_params_raises_when_index_params_is_none_for_custom_type(self):
        """Test that build_index_params raises RuntimeError when index_params is None for custom types (P1 fix)"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
        )

        # Call with None (simulating compatibility helper returning None)
        # Should raise RuntimeError for non-AUTOINDEX types
        with pytest.raises(RuntimeError, match="IndexParams not available"):
            config.build_index_params(None)

    def test_build_index_params_returns_none_for_autoindex_when_index_params_is_none(
        self,
    ):
        """Test AUTOINDEX falls back to direct API parameters when IndexParams is unavailable."""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # AUTOINDEX should still produce direct API parameters
        result = config.build_index_params(None)
        assert result == {
            "field_name": "vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {},
        }

    def test_build_index_params_autoindex_uses_index_params_object(self):
        """Test AUTOINDEX still creates an explicit vector index when IndexParams is available."""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        mock_index_params = MagicMock()

        result = config.build_index_params(mock_index_params)

        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once_with(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={},
        )

    def test_create_indexes_uses_compatibility_helper(self):
        """Test that _create_indexes_after_collection uses _get_index_params (P1 fix)"""
        # Setup storage instance
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client
        mock_client = MagicMock()
        mock_index_params = MagicMock()

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        # Spy on _get_index_params to verify it's called
        with patch.object(
            storage, "_get_index_params", return_value=mock_index_params
        ) as mock_get_index_params:
            # Call the method
            storage._create_indexes_after_collection()

            # Verify that _get_index_params was called at least once
            assert mock_get_index_params.call_count >= 1

    def test_version_probing_only_for_hnsw_sq(self):
        """Test that get_server_version is only called when index type requires it (P2 fix)"""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Test with HNSW (no version requirement) - should NOT call get_server_version
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        storage._client = mock_client

        # Mock the init lock as an async context manager
        mock_lock = AsyncMock()

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=mock_lock
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        # get_server_version should NOT be called for HNSW
        mock_client.get_server_version.assert_not_called()

    def test_version_probing_called_for_hnsw_sq(self):
        """Test that get_server_version IS called when HNSW_SQ is configured (P2 fix)"""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW_SQ",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        mock_client.get_server_version.return_value = "2.6.9"
        storage._client = mock_client

        # Mock the init lock as an async context manager
        mock_lock = AsyncMock()

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=mock_lock
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        # get_server_version SHOULD be called for HNSW_SQ
        mock_client.get_server_version.assert_called_once()

    def test_initialize_creates_missing_database_before_collection_setup(self):
        """Test that initialize bootstraps a missing configured Milvus database."""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default"]
        mock_lock = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_DB_NAME": "lightrag",
            },
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ) as mock_client_cls:
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=mock_lock,
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        mock_client_cls.assert_called_once_with(
            uri="http://milvus:19530",
            user=None,
            password=None,
            token=None,
        )
        bootstrap_client.list_databases.assert_called_once_with()
        bootstrap_client.create_database.assert_called_once_with("lightrag")
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    def test_initialize_uses_existing_database_without_recreating_it(self):
        """Test that initialize switches to an existing configured Milvus database."""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default", "lightrag"]
        mock_lock = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_DB_NAME": "lightrag",
            },
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ):
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=mock_lock,
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        bootstrap_client.list_databases.assert_called_once_with()
        bootstrap_client.create_database.assert_not_called()
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    def test_model_suffix_is_used_for_milvus_collection_names(self):
        """Milvus should isolate collections by embedding model when model_name is available."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 2560
        mock_embedding_func.model_name = "qwen3-embedding:4b"

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        assert storage.legacy_namespace == "entities"
        assert storage.final_namespace == "entities_qwen3_embedding_4b_2560d"

    def test_compatible_legacy_collection_is_reused_when_suffix_available(self):
        """Compatible legacy collections should still be reused after suffixing is introduced."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 2560
        mock_embedding_func.model_name = "qwen3-embedding:4b"

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = (
            lambda name: name == storage.legacy_namespace
        )
        storage._client.describe_collection.return_value = {}

        with patch.object(storage, "_validate_collection_compatibility"):
            with patch.object(storage, "_ensure_collection_loaded"):
                storage._create_collection_if_not_exist()

        storage._client.create_collection.assert_not_called()
        assert storage.final_namespace == storage.legacy_namespace

    def test_legacy_dimension_mismatch_creates_isolated_collection_when_suffix_available(
        self,
    ):
        """A legacy dimension mismatch should create a new suffixed Milvus collection."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 2560
        mock_embedding_func.model_name = "qwen3-embedding:4b"

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = (
            lambda name: name == storage.legacy_namespace
        )
        storage._client.describe_collection.return_value = {}

        with patch.object(
            storage,
            "_validate_collection_compatibility",
            side_effect=ValueError(
                "Vector dimension mismatch for collection 'entities': existing=4096, current=2560"
            ),
        ):
            with patch.object(
                storage, "_create_schema_for_namespace", return_value="schema"
            ):
                with patch.object(storage, "_create_indexes_after_collection"):
                    with patch.object(storage, "_ensure_collection_loaded"):
                        storage._create_collection_if_not_exist()

        storage._client.create_collection.assert_called_once_with(
            collection_name="entities_qwen3_embedding_4b_2560d",
            schema="schema",
        )
        assert storage.final_namespace == "entities_qwen3_embedding_4b_2560d"

    def test_existing_collection_missing_vector_index_is_repaired(self):
        """Existing collections missing vector indexes should be repaired automatically."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        load_error = RuntimeError(
            "there is no vector index on field: [vector], please create index firstly"
        )

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(storage, "_validate_collection_compatibility"):
                with patch.object(
                    storage,
                    "_ensure_collection_loaded",
                    side_effect=[load_error, None],
                ) as mock_load:
                    with patch.object(
                        storage, "_repair_missing_vector_index"
                    ) as mock_repair:
                        storage._create_collection_if_not_exist()

        assert mock_load.call_count == 2
        mock_repair.assert_called_once_with()

    def test_existing_collection_index_repair_failure_has_precise_error(self):
        """Index repair failures should not be reported as collection validation failures."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        load_error = RuntimeError(
            "there is no vector index on field: [vector], please create index firstly"
        )

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(storage, "_validate_collection_compatibility"):
                with patch.object(
                    storage, "_ensure_collection_loaded", side_effect=load_error
                ):
                    with patch.object(
                        storage,
                        "_repair_missing_vector_index",
                        side_effect=RuntimeError("create index failed"),
                    ):
                        with pytest.raises(
                            RuntimeError,
                            match="Index repair failed for collection 'space1_entities'",
                        ):
                            storage._create_collection_if_not_exist()

    def test_existing_collection_non_index_validation_failure_still_raises(self):
        """Non-index validation failures should still stop initialization."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(
                storage,
                "_validate_collection_compatibility",
                side_effect=RuntimeError("dimension mismatch"),
            ):
                with pytest.raises(
                    RuntimeError,
                    match="Collection validation failed for 'space1_entities'",
                ):
                    storage._create_collection_if_not_exist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
