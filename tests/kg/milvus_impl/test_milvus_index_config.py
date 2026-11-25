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
)


pytestmark = pytest.mark.offline


class TestMilvusIndexConfig:
    """MilvusIndexConfig unit tests"""

    def test_default_values(self):
        """Test default configuration"""
        config = MilvusIndexConfig()
        assert config.index_type == "AUTOINDEX"
        assert config.metric_type == "COSINE"
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 360
        assert config.hnsw_ef == 200
        assert config.sq_type == "SQ8"
        assert not config.sq_refine
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
                "MILVUS_HNSW_M": "64",
            },
        ):
            config = MilvusIndexConfig()
            assert config.index_type == "HNSW"
            assert config.hnsw_m == 64

    def test_case_insensitive_index_type(self):
        """Test that index type is case insensitive"""
        config = MilvusIndexConfig(index_type="hnsw")
        assert config.index_type == "HNSW"

    def test_invalid_index_type(self):
        """Test invalid index type raises exception"""
        with pytest.raises(ValueError, match="Unsupported index type"):
            MilvusIndexConfig(index_type="INVALID_INDEX")

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

    def test_build_index_params_hnsw(self):
        """Test HNSW index parameters construction"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
            hnsw_m=32,
            hnsw_ef_construction=256,
        )

        mock_index_params = MagicMock()

        config.build_index_params(mock_index_params)

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

        mock_index_params = MagicMock()

        config.build_index_params(mock_index_params)

        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["index_type"] == "HNSW_SQ"
        assert call_kwargs["params"]["sq_type"] == "SQ8"
        assert call_kwargs["params"]["refine"] is True
        assert call_kwargs["params"]["refine_type"] == "FP32"

    def test_build_index_params_hnsw_sq_no_refine(self):
        """Test HNSW_SQ without refinement"""
        config = MilvusIndexConfig(index_type="HNSW_SQ", sq_type="SQ8", sq_refine=False)

        mock_index_params = MagicMock()

        config.build_index_params(mock_index_params)

        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["index_type"] == "HNSW_SQ"
        assert call_kwargs["params"]["sq_type"] == "SQ8"
        assert "refine" not in call_kwargs["params"]
        assert "refine_type" not in call_kwargs["params"]

    def test_build_index_params_ivf_flat(self):
        """Test IVF_FLAT index parameters construction"""
        config = MilvusIndexConfig(index_type="IVF_FLAT", ivf_nlist=2048)

        mock_index_params = MagicMock()

        config.build_index_params(mock_index_params)

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
        config = MilvusIndexConfig(index_type="HNSW_SQ", hnsw_ef=200, sq_refine=False)
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
        assert d["hnsw_m"] == 16
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
            assert config.sq_refine is True

        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "false"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert not config.sq_refine

        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "1"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert config.sq_refine is True

        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": "0"}):
            config = MilvusIndexConfig(index_type="HNSW_SQ")
            assert not config.sq_refine

    def test_env_int_parsing_invalid(self):
        """Test integer environment variable parsing with invalid value"""
        with patch.dict(os.environ, {"MILVUS_HNSW_M": "invalid"}):
            config = MilvusIndexConfig()
            assert config.hnsw_m == 16  # Falls back to default (Milvus 2.4+)

    def test_get_config_field_names(self):
        """Test get_config_field_names() returns all dataclass fields"""
        field_names = MilvusIndexConfig.get_config_field_names()

        # Check that it's a set
        assert isinstance(field_names, set)

        # Check that all expected fields are present
        expected_fields = {
            "index_type",
            "metric_type",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef",
            "sq_type",
            "sq_refine",
            "sq_refine_type",
            "sq_refine_k",
            "ivf_nlist",
            "ivf_nprobe",
        }
        assert field_names == expected_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
