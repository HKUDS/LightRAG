"""
Tests for bridging vector_db_storage_cls_kwargs to MilvusIndexConfig

This test suite validates that MilvusIndexConfig parameters can be passed
through vector_db_storage_cls_kwargs and that backward compatibility is maintained.
"""

import pytest
from unittest.mock import patch, MagicMock
from lightrag.kg.milvus_impl import MilvusVectorDBStorage


@pytest.mark.offline
class TestMilvusKwargsParameterBridge:
    """Test parameter bridging from vector_db_storage_cls_kwargs to MilvusIndexConfig"""

    def test_kwargs_to_index_config_basic(self):
        """Test that basic HNSW parameters are passed from kwargs to MilvusIndexConfig"""
        # Mock the embedding function
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Create storage instance with custom index config parameters in kwargs
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "hnsw_m": 32,
                    "hnsw_ef": 256,
                    "hnsw_ef_construction": 300,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify that parameters were passed to index_config
        assert storage.index_config.hnsw_m == 32
        assert storage.index_config.hnsw_ef == 256
        assert storage.index_config.hnsw_ef_construction == 300

    def test_kwargs_to_index_config_index_and_metric_types(self):
        """Test that index_type and metric_type are passed from kwargs"""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "ivf_nlist": 2048,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify that parameters were passed to index_config
        assert storage.index_config.index_type == "IVF_FLAT"
        assert storage.index_config.metric_type == "L2"
        assert storage.index_config.ivf_nlist == 2048

    def test_kwargs_to_index_config_sq_parameters(self):
        """Test that HNSW_SQ parameters are passed from kwargs"""
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
                    "sq_type": "SQ8",
                    "sq_refine": True,
                    "sq_refine_type": "FP16",
                    "sq_refine_k": 20,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify that parameters were passed to index_config
        assert storage.index_config.index_type == "HNSW_SQ"
        assert storage.index_config.sq_type == "SQ8"
        assert storage.index_config.sq_refine is True
        assert storage.index_config.sq_refine_type == "FP16"
        assert storage.index_config.sq_refine_k == 20

    def test_backward_compatibility_no_index_params(self):
        """Test backward compatibility when no index parameters are provided in kwargs"""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Create storage without any index config parameters in kwargs
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify that default values are used (from environment variables or defaults)
        assert storage.index_config.index_type == "AUTOINDEX"  # Default
        assert storage.index_config.metric_type == "COSINE"  # Default
        assert storage.index_config.hnsw_m == 30  # Default
        assert storage.index_config.hnsw_ef_construction == 200  # Default

    def test_kwargs_params_override_environment_variables(self):
        """Test that kwargs parameters take precedence over environment variables"""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Set environment variables
        with patch.dict(
            "os.environ",
            {
                "MILVUS_INDEX_TYPE": "IVF_FLAT",
                "MILVUS_HNSW_M": "16",
            },
        ):
            # Create storage with kwargs parameters that should override env vars
            storage = MilvusVectorDBStorage(
                namespace="test_entities",
                workspace="test_workspace",
                global_config={
                    "embedding_batch_num": 100,
                    "vector_db_storage_cls_kwargs": {
                        "cosine_better_than_threshold": 0.3,
                        "index_type": "HNSW",
                        "hnsw_m": 64,
                    },
                },
                embedding_func=mock_embedding_func,
                meta_fields=set(),
            )

            # Verify that kwargs parameters override environment variables
            assert (
                storage.index_config.index_type == "HNSW"
            )  # From kwargs, not IVF_FLAT
            assert storage.index_config.hnsw_m == 64  # From kwargs, not 16

    def test_non_index_params_ignored(self):
        """Test that non-index-config parameters in kwargs are ignored"""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "hnsw_m": 32,
                    "some_other_param": "ignored",  # Should be ignored
                    "another_param": 123,  # Should be ignored
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify that valid parameter was passed
        assert storage.index_config.hnsw_m == 32
        # Verify that invalid parameters were ignored (no AttributeError)
        assert not hasattr(storage.index_config, "some_other_param")
        assert not hasattr(storage.index_config, "another_param")

    def test_fallback_uses_index_config(self):
        """Test that _create_vector_index_fallback uses index_config values"""
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
                    "metric_type": "L2",
                    "hnsw_m": 48,
                    "hnsw_ef_construction": 400,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client
        mock_client = MagicMock()
        storage._client = mock_client
        storage.workspace = "test_workspace"
        storage.final_namespace = "test_entities"

        # Call the fallback method
        storage._create_vector_index_fallback()

        # Verify that create_index was called with index_config values
        mock_client.create_index.assert_called_once()
        call_args = mock_client.create_index.call_args
        assert call_args[1]["collection_name"] == "test_entities"
        assert call_args[1]["field_name"] == "vector"
        assert call_args[1]["index_params"]["index_type"] == "HNSW"
        assert call_args[1]["index_params"]["metric_type"] == "L2"
        assert call_args[1]["index_params"]["params"]["M"] == 48
        assert call_args[1]["index_params"]["params"]["efConstruction"] == 400

    def test_fallback_autoindex_to_hnsw(self):
        """Test that fallback converts AUTOINDEX to HNSW"""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "AUTOINDEX",  # Should convert to HNSW in fallback
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client
        mock_client = MagicMock()
        storage._client = mock_client
        storage.workspace = "test_workspace"
        storage.final_namespace = "test_entities"

        # Call the fallback method
        storage._create_vector_index_fallback()

        # Verify that AUTOINDEX was converted to HNSW
        call_args = mock_client.create_index.call_args
        assert call_args[1]["index_params"]["index_type"] == "HNSW"

    def test_raganything_framework_integration_scenario(self):
        """Test configuration passing through frameworks like RAGAnything
        
        This test validates the use case where a framework (like RAGAnything) 
        sits on top of LightRAG and needs to pass Milvus index configuration 
        through to LightRAG without modifying environment variables.
        
        The framework can pass all index config parameters via 
        vector_db_storage_cls_kwargs, and they will be properly extracted 
        and applied to MilvusIndexConfig.
        """
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Simulate RAGAnything framework passing configuration to LightRAG
        # All index configuration parameters are passed through kwargs
        framework_config = {
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                # Required for vector storage
                "cosine_better_than_threshold": 0.2,
                
                # Milvus index configuration - all parameters supported
                "index_type": "HNSW",
                "metric_type": "L2",
                "hnsw_m": 48,
                "hnsw_ef_construction": 400,
                "hnsw_ef": 200,
                
                # Framework-specific parameters (should be ignored by Milvus)
                "framework_version": "1.0.0",
                "custom_setting": "value",
            },
        }

        # Create storage instance with framework configuration
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="raganything_workspace",
            global_config=framework_config,
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify all Milvus parameters were correctly extracted and applied
        assert storage.index_config.index_type == "HNSW"
        assert storage.index_config.metric_type == "L2"
        assert storage.index_config.hnsw_m == 48
        assert storage.index_config.hnsw_ef_construction == 400
        assert storage.index_config.hnsw_ef == 200
        
        # Verify framework-specific parameters were ignored
        assert not hasattr(storage.index_config, "framework_version")
        assert not hasattr(storage.index_config, "custom_setting")
        
        # Verify workspace isolation is maintained
        assert storage.workspace == "raganything_workspace"

    def test_all_milvus_parameters_supported_via_kwargs(self):
        """Test that all 11 MilvusIndexConfig parameters can be configured via kwargs
        
        This comprehensive test ensures that every single index configuration 
        parameter defined in MilvusIndexConfig can be passed through 
        vector_db_storage_cls_kwargs, which is critical for framework integration.
        """
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Pass ALL 11 MilvusIndexConfig parameters via kwargs
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    # All 11 MilvusIndexConfig parameters
                    "index_type": "HNSW_SQ",
                    "metric_type": "IP",
                    "hnsw_m": 64,
                    "hnsw_ef_construction": 512,
                    "hnsw_ef": 256,
                    "sq_type": "SQ8",
                    "sq_refine": True,
                    "sq_refine_type": "FP16",
                    "sq_refine_k": 30,
                    "ivf_nlist": 4096,
                    "ivf_nprobe": 64,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Verify EVERY parameter was correctly applied
        assert storage.index_config.index_type == "HNSW_SQ"
        assert storage.index_config.metric_type == "IP"
        assert storage.index_config.hnsw_m == 64
        assert storage.index_config.hnsw_ef_construction == 512
        assert storage.index_config.hnsw_ef == 256
        assert storage.index_config.sq_type == "SQ8"
        assert storage.index_config.sq_refine is True
        assert storage.index_config.sq_refine_type == "FP16"
        assert storage.index_config.sq_refine_k == 30
        assert storage.index_config.ivf_nlist == 4096
        assert storage.index_config.ivf_nprobe == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
