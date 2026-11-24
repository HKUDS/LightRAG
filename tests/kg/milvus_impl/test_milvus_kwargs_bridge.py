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
        assert storage.index_config.metric_type == "COSINE"  # deprecated, ignored
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
        assert storage.index_config.metric_type == "COSINE"  # deprecated, ignored
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
