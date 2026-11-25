"""
Unit tests for PostgreSQL safe index name generation.

This module tests the _safe_index_name helper function which prevents
PostgreSQL's silent 63-byte identifier truncation from causing index
lookup failures.
"""

import pytest

# Mark all tests as offline (no external dependencies)
pytestmark = pytest.mark.offline


class TestSafeIndexName:
    """Test suite for _safe_index_name function."""

    def test_short_name_unchanged(self):
        """Short index names should remain unchanged."""
        from lightrag.kg.postgres_impl import _safe_index_name

        # Short table name - should return unchanged
        result = _safe_index_name("lightrag_vdb_entity", "hnsw_cosine")
        assert result == "idx_lightrag_vdb_entity_hnsw_cosine"
        assert len(result.encode("utf-8")) <= 63

    def test_long_name_gets_hashed(self):
        """Long table names exceeding 63 bytes should get hashed."""
        from lightrag.kg.postgres_impl import _safe_index_name

        # Long table name that would exceed 63 bytes
        long_table_name = "LIGHTRAG_VDB_ENTITY_text_embedding_3_large_3072d"
        result = _safe_index_name(long_table_name, "hnsw_cosine")

        # Should be within 63 bytes
        assert len(result.encode("utf-8")) <= 63

        # Should start with idx_ prefix
        assert result.startswith("idx_")

        # Should contain the suffix
        assert result.endswith("_hnsw_cosine")

        # Should NOT be the naive concatenation (which would be truncated)
        naive_name = f"idx_{long_table_name.lower()}_hnsw_cosine"
        assert result != naive_name

    def test_case_insensitive(self):
        """Table names should be normalized to lowercase."""
        from lightrag.kg.postgres_impl import _safe_index_name

        result_upper = _safe_index_name("LIGHTRAG_VDB_ENTITY", "hnsw_cosine")
        result_lower = _safe_index_name("lightrag_vdb_entity", "hnsw_cosine")

        assert result_upper == result_lower

    def test_boundary_case_exactly_63_bytes(self):
        """Test boundary case where name is exactly at 63-byte limit."""
        from lightrag.kg.postgres_impl import _safe_index_name

        # Create a table name that results in exactly 63 bytes
        # idx_ (4) + table_name + _ (1) + suffix = 63
        # So table_name + suffix = 58

        # Test a name that's just under the limit (should remain unchanged)
        short_suffix = "id"
        # idx_ (4) + 56 chars + _ (1) + id (2) = 63
        table_56 = "a" * 56
        result = _safe_index_name(table_56, short_suffix)
        expected = f"idx_{table_56}_{short_suffix}"
        assert result == expected
        assert len(result.encode("utf-8")) == 63
