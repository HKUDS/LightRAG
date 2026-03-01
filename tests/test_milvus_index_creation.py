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
            # Mock the scalar index creation to fail
            mock_client.create_index.side_effect = Exception(
                "Scalar index creation failed"
            )

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
        """Test that build_index_params returns None for AUTOINDEX regardless of index_params"""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # AUTOINDEX should return None regardless
        result = config.build_index_params(None)
        assert result is None

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
