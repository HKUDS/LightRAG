"""
Tests for Milvus index configuration

This test suite validates the MilvusIndexConfig class and its integration
with MilvusVectorDBStorage.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from lightrag.kg.milvus_impl import (
    MilvusIndexConfig,
    SUPPORTED_INDEX_TYPES,
    SUPPORTED_METRIC_TYPES,
    SUPPORTED_SQ_TYPES,
    SUPPORTED_REFINE_TYPES,
)


class TestMilvusIndexConfig:
    """MilvusIndexConfig unit tests"""

    def test_default_values(self):
        """Test default configuration"""
        config = MilvusIndexConfig()
        assert config.index_type == "AUTOINDEX"
        assert config.metric_type == "COSINE"
        assert config.hnsw_m == 30
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef == 100
        assert config.sq_type == "SQ8"
        assert config.sq_refine == False
        assert config.sq_refine_type == "FP32"
        assert config.sq_refine_k == 10
        assert config.ivf_nlist == 1024
        assert config.ivf_nprobe == 16

    def test_env_override(self):
        """Test environment variable override"""
        with patch.dict(
            os.environ,
            {
                "MILVUS_INDEX_TYPE": "HNSW",
                "MILVUS_METRIC_TYPE": "L2",
                "MILVUS_HNSW_M": "64",
            },
        ):
            config = MilvusIndexConfig()
            assert config.index_type == "HNSW"
            assert config.metric_type == "L2"
            assert config.hnsw_m == 64

    def test_init_param_priority(self):
        """Test initialization parameter priority over environment variables"""
        with patch.dict(os.environ, {"MILVUS_INDEX_TYPE": "IVF_FLAT"}):
            config = MilvusIndexConfig(index_type="HNSW")
            assert config.index_type == "HNSW"  # Init param takes precedence

    def test_case_insensitive_index_type(self):
        """Test that index type is case insensitive"""
        config = MilvusIndexConfig(index_type="hnsw")
        assert config.index_type == "HNSW"

    def test_case_insensitive_metric_type(self):
        """Test that metric type is case insensitive"""
        config = MilvusIndexConfig(metric_type="cosine")
        assert config.metric_type == "COSINE"

    def test_invalid_index_type(self):
        """Test invalid index type raises exception"""
        with pytest.raises(ValueError, match="Unsupported index type"):
            MilvusIndexConfig(index_type="INVALID_INDEX")

    def test_invalid_metric_type(self):
        """Test invalid metric type raises exception"""
        with pytest.raises(ValueError, match="Unsupported metric type"):
            MilvusIndexConfig(metric_type="INVALID_METRIC")

    def test_invalid_hnsw_m_range_low(self):
        """Test HNSW M parameter range validation (too low)"""
        with pytest.raises(ValueError, match="hnsw_m must be in"):
            MilvusIndexConfig(hnsw_m=1)  # Less than 2

    def test_invalid_hnsw_m_range_high(self):
        """Test HNSW M parameter range validation (too high)"""
        with pytest.raises(ValueError, match="hnsw_m must be in"):
            MilvusIndexConfig(hnsw_m=3000)  # Greater than 2048

    def test_valid_hnsw_m_boundary(self):
        """Test HNSW M parameter boundary values"""
        config_low = MilvusIndexConfig(hnsw_m=2)
        assert config_low.hnsw_m == 2

        config_high = MilvusIndexConfig(hnsw_m=2048)
        assert config_high.hnsw_m == 2048

    def test_invalid_hnsw_ef_construction(self):
        """Test HNSW efConstruction validation"""
        with pytest.raises(ValueError, match="hnsw_ef_construction must be"):
            MilvusIndexConfig(hnsw_ef_construction=0)

    def test_invalid_ivf_nlist_low(self):
        """Test IVF nlist parameter range validation (too low)"""
        with pytest.raises(ValueError, match="ivf_nlist must be in"):
            MilvusIndexConfig(ivf_nlist=0)

    def test_invalid_ivf_nlist_high(self):
        """Test IVF nlist parameter range validation (too high)"""
        with pytest.raises(ValueError, match="ivf_nlist must be in"):
            MilvusIndexConfig(ivf_nlist=70000)

    def test_invalid_sq_type(self):
        """Test invalid sq_type"""
        with pytest.raises(ValueError, match="Unsupported sq_type"):
            MilvusIndexConfig(index_type="HNSW_SQ", sq_type="INVALID")

    def test_invalid_refine_type(self):
        """Test invalid refine_type"""
        with pytest.raises(ValueError, match="Unsupported refine_type"):
            MilvusIndexConfig(
                index_type="HNSW_SQ", sq_refine=True, sq_refine_type="INVALID"
            )

    def test_version_validation_hnsw_sq_pass(self):
        """Test HNSW_SQ version validation passes with valid versions"""
        config = MilvusIndexConfig(index_type="HNSW_SQ")

        # Version meets requirement
        config.validate_milvus_version("2.6.8")  # Exactly required
        config.validate_milvus_version("2.6.9")  # Above requirement
        config.validate_milvus_version("2.7.0")  # Higher version

    def test_version_validation_hnsw_sq_fail(self):
        """Test HNSW_SQ version validation fails with invalid versions"""
        config = MilvusIndexConfig(index_type="HNSW_SQ")

        # Version does not meet requirement
        with pytest.raises(ValueError, match="HNSW_SQ requires Milvus 2.6.8"):
            config.validate_milvus_version("2.6.7")  # Below 2.6.8

        with pytest.raises(ValueError, match="HNSW_SQ requires Milvus 2.6.8"):
            config.validate_milvus_version("2.5.0")  # Much lower

    def test_version_validation_hnsw_sq_with_sq4u(self):
        """Test HNSW_SQ + SQ4U version validation"""
        config = MilvusIndexConfig(index_type="HNSW_SQ", sq_type="SQ4U")

        # Passes with valid version
        config.validate_milvus_version("2.6.9")

        # Fails with invalid version
        with pytest.raises(ValueError, match="HNSW_SQ requires Milvus 2.6.8"):
            config.validate_milvus_version("2.6.0")

    def test_version_validation_hnsw_no_requirement(self):
        """Test normal HNSW has no version restriction"""
        config = MilvusIndexConfig(index_type="HNSW")

        # No version restriction
        config.validate_milvus_version("2.4.0")  # Lower version OK
        config.validate_milvus_version("2.6.9")  # Higher version OK

    def test_version_validation_with_dev_suffix(self):
        """Test version validation handles dev suffixes"""
        config = MilvusIndexConfig(index_type="HNSW_SQ")

        # Should handle "2.6.9-dev" format
        config.validate_milvus_version("2.6.9-dev")

    def test_build_index_params_autoindex(self):
        """Test AUTOINDEX does not generate parameters"""
        config = MilvusIndexConfig(index_type="AUTOINDEX")
        mock_client = MagicMock()

        result = config.build_index_params(mock_client)
        assert result is None

    def test_build_index_params_hnsw(self):
        """Test HNSW index parameters construction"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
            hnsw_m=32,
            hnsw_ef_construction=256,
        )

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        result = config.build_index_params(mock_client)

        mock_index_params.add_index.assert_called_once()
        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["index_type"] == "HNSW"
        assert call_kwargs["metric_type"] == "COSINE"
        assert call_kwargs["params"]["M"] == 32
        assert call_kwargs["params"]["efConstruction"] == 256

    def test_build_index_params_hnsw_sq(self):
        """Test HNSW_SQ index parameters construction"""
        config = MilvusIndexConfig(
            index_type="HNSW_SQ",
            sq_type="SQ8",
            sq_refine=True,
            sq_refine_type="FP32",
        )

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        result = config.build_index_params(mock_client)

        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["index_type"] == "HNSW_SQ"
        assert call_kwargs["params"]["sq_type"] == "SQ8"
        assert call_kwargs["params"]["refine"] == True
        assert call_kwargs["params"]["refine_type"] == "FP32"

    def test_build_index_params_hnsw_sq_no_refine(self):
        """Test HNSW_SQ without refinement"""
        config = MilvusIndexConfig(
            index_type="HNSW_SQ", sq_type="SQ8", sq_refine=False
        )

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        result = config.build_index_params(mock_client)

        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["index_type"] == "HNSW_SQ"
        assert call_kwargs["params"]["sq_type"] == "SQ8"
        assert "refine" not in call_kwargs["params"]
        assert "refine_type" not in call_kwargs["params"]

    def test_build_index_params_ivf_flat(self):
        """Test IVF_FLAT index parameters construction"""
        config = MilvusIndexConfig(index_type="IVF_FLAT", ivf_nlist=2048)

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        result = config.build_index_params(mock_client)

        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["index_type"] == "IVF_FLAT"
        assert call_kwargs["params"]["nlist"] == 2048

    def test_build_search_params_hnsw(self):
        """Test HNSW search parameters construction"""
        config = MilvusIndexConfig(index_type="HNSW", hnsw_ef=150)
        params = config.build_search_params()
        assert params["params"]["ef"] == 150

    def test_build_search_params_hnsw_sq_with_refine(self):
        """Test HNSW_SQ with refinement search parameters"""
        config = MilvusIndexConfig(
            index_type="HNSW_SQ", hnsw_ef=200, sq_refine=True, sq_refine_k=20
        )
        params = config.build_search_params()
        assert params["params"]["ef"] == 200
        assert params["params"]["refine_k"] == 20

    def test_build_search_params_hnsw_sq_no_refine(self):
        """Test HNSW_SQ without refinement search parameters"""
        config = MilvusIndexConfig(
            index_type="HNSW_SQ", hnsw_ef=200, sq_refine=False
        )
        params = config.build_search_params()
        assert params["params"]["ef"] == 200
        assert "refine_k" not in params["params"]

    def test_build_search_params_ivf(self):
        """Test IVF search parameters construction"""
        config = MilvusIndexConfig(index_type="IVF_FLAT", ivf_nprobe=32)
        params = config.build_search_params()
        assert params["params"]["nprobe"] == 32

    def test_build_search_params_autoindex(self):
        """Test AUTOINDEX search parameters (empty)"""
        config = MilvusIndexConfig(index_type="AUTOINDEX")
        params = config.build_search_params()
        assert params == {}

    def test_to_dict_hnsw(self):
        """Test configuration export for HNSW"""
        config = MilvusIndexConfig(index_type="HNSW")
        d = config.to_dict()
        assert d["index_type"] == "HNSW"
        assert d["hnsw_m"] == 30
        assert d["sq_type"] is None  # Not HNSW_SQ
        assert d["ivf_nlist"] is None  # Not IVF

    def test_to_dict_hnsw_sq(self):
        """Test configuration export for HNSW_SQ"""
        config = MilvusIndexConfig(index_type="HNSW_SQ", sq_type="SQ8")
        d = config.to_dict()
        assert d["index_type"] == "HNSW_SQ"
        assert d["sq_type"] == "SQ8"
        assert d["ivf_nlist"] is None

    def test_to_dict_ivf(self):
        """Test configuration export for IVF"""
        config = MilvusIndexConfig(index_type="IVF_FLAT")
        d = config.to_dict()
        assert d["index_type"] == "IVF_FLAT"
        assert d["ivf_nlist"] == 1024
        assert d["sq_type"] is None

    def test_env_bool_parsing(self):
        """Test boolean environment variable parsing"""
        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "true"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert config.sq_refine == True

        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "false"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert config.sq_refine == False

        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "1"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert config.sq_refine == True

        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "0"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert config.sq_refine == False

    def test_env_int_parsing_invalid(self):
        """Test integer environment variable parsing with invalid value"""
        with patch.dict(os.environ, {"MILVUS_HNSW_M": "invalid"}):
            config = MilvusIndexConfig()
            assert config.hnsw_m == 30  # Falls back to default

    def test_all_index_types_supported(self):
        """Test all supported index types can be configured"""
        for index_type in SUPPORTED_INDEX_TYPES:
            if index_type == "HNSW_SQ":
                # HNSW_SQ requires special parameters
                config = MilvusIndexConfig(index_type=index_type, sq_type="SQ8")
            else:
                config = MilvusIndexConfig(index_type=index_type)
            assert config.index_type == index_type

    def test_all_metric_types_supported(self):
        """Test all supported metric types can be configured"""
        for metric_type in SUPPORTED_METRIC_TYPES:
            config = MilvusIndexConfig(metric_type=metric_type)
            assert config.metric_type == metric_type

    def test_all_sq_types_supported(self):
        """Test all supported sq_types can be configured"""
        for sq_type in SUPPORTED_SQ_TYPES:
            config = MilvusIndexConfig(index_type="HNSW_SQ", sq_type=sq_type)
            assert config.sq_type == sq_type

    def test_all_refine_types_supported(self):
        """Test all supported refine_types can be configured"""
        for refine_type in SUPPORTED_REFINE_TYPES:
            config = MilvusIndexConfig(
                index_type="HNSW_SQ", sq_refine=True, sq_refine_type=refine_type
            )
            assert config.sq_refine_type == refine_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
