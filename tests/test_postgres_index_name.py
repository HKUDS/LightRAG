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

    def test_deterministic_output(self):
        """Same input should always produce same output (deterministic)."""
        from lightrag.kg.postgres_impl import _safe_index_name

        table_name = "LIGHTRAG_VDB_CHUNKS_text_embedding_3_large_3072d"
        suffix = "hnsw_cosine"

        result1 = _safe_index_name(table_name, suffix)
        result2 = _safe_index_name(table_name, suffix)

        assert result1 == result2

    def test_different_suffixes_different_results(self):
        """Different suffixes should produce different index names."""
        from lightrag.kg.postgres_impl import _safe_index_name

        table_name = "LIGHTRAG_VDB_ENTITY_text_embedding_3_large_3072d"

        result1 = _safe_index_name(table_name, "hnsw_cosine")
        result2 = _safe_index_name(table_name, "ivfflat_cosine")

        assert result1 != result2

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

    def test_unicode_handling(self):
        """Unicode characters should be properly handled (bytes, not chars)."""
        from lightrag.kg.postgres_impl import _safe_index_name

        # Unicode characters can take more bytes than visible chars
        # Chinese characters are 3 bytes each in UTF-8
        table_name = "lightrag_测试_table"  # Contains Chinese chars
        result = _safe_index_name(table_name, "hnsw_cosine")

        # Should always be within 63 bytes
        assert len(result.encode("utf-8")) <= 63

    def test_real_world_model_names(self):
        """Test with real-world embedding model names that cause issues."""
        from lightrag.kg.postgres_impl import _safe_index_name

        # These are actual model names that have caused issues
        test_cases = [
            ("LIGHTRAG_VDB_CHUNKS_text_embedding_3_large_3072d", "hnsw_cosine"),
            ("LIGHTRAG_VDB_ENTITY_text_embedding_3_large_3072d", "hnsw_cosine"),
            ("LIGHTRAG_VDB_RELATION_text_embedding_3_large_3072d", "hnsw_cosine"),
            (
                "LIGHTRAG_VDB_ENTITY_bge_m3_1024d",
                "hnsw_cosine",
            ),  # Shorter model name
            (
                "LIGHTRAG_VDB_CHUNKS_nomic_embed_text_v1_768d",
                "ivfflat_cosine",
            ),  # Different index type
        ]

        for table_name, suffix in test_cases:
            result = _safe_index_name(table_name, suffix)

            # Critical: must be within PostgreSQL's 63-byte limit
            assert (
                len(result.encode("utf-8")) <= 63
            ), f"Index name too long: {result} for table {table_name}"

            # Must have consistent format
            assert result.startswith("idx_"), f"Missing idx_ prefix: {result}"
            assert result.endswith(f"_{suffix}"), f"Missing suffix {suffix}: {result}"

    def test_hash_uniqueness_for_similar_tables(self):
        """Similar but different table names should produce different hashes."""
        from lightrag.kg.postgres_impl import _safe_index_name

        # These tables have similar names but should have different hashes
        tables = [
            "LIGHTRAG_VDB_CHUNKS_model_a_1024d",
            "LIGHTRAG_VDB_CHUNKS_model_b_1024d",
            "LIGHTRAG_VDB_ENTITY_model_a_1024d",
        ]

        results = [_safe_index_name(t, "hnsw_cosine") for t in tables]

        # All results should be unique
        assert len(set(results)) == len(results), "Hash collision detected!"


class TestIndexNameIntegration:
    """Integration-style tests for index name usage patterns."""

    def test_pg_indexes_lookup_compatibility(self):
        """
        Test that the generated index name will work with pg_indexes lookup.

        This is the core problem: PostgreSQL stores the truncated name,
        but we were looking up the untruncated name. Our fix ensures we
        always use a name that fits within 63 bytes.
        """
        from lightrag.kg.postgres_impl import _safe_index_name

        table_name = "LIGHTRAG_VDB_CHUNKS_text_embedding_3_large_3072d"
        suffix = "hnsw_cosine"

        # Generate the index name
        index_name = _safe_index_name(table_name, suffix)

        # Simulate what PostgreSQL would store (truncate at 63 bytes)
        stored_name = index_name.encode("utf-8")[:63].decode("utf-8", errors="ignore")

        # The key fix: our generated name should equal the stored name
        # because it's already within the 63-byte limit
        assert (
            index_name == stored_name
        ), "Index name would be truncated by PostgreSQL, causing lookup failures!"

    def test_backward_compatibility_short_names(self):
        """
        Ensure backward compatibility with existing short index names.

        For tables that have existing indexes with short names (pre-model-suffix era),
        the function should not change their names.
        """
        from lightrag.kg.postgres_impl import _safe_index_name

        # Legacy table names without model suffix
        legacy_tables = [
            "LIGHTRAG_VDB_ENTITY",
            "LIGHTRAG_VDB_RELATION",
            "LIGHTRAG_VDB_CHUNKS",
        ]

        for table in legacy_tables:
            for suffix in ["hnsw_cosine", "ivfflat_cosine", "id"]:
                result = _safe_index_name(table, suffix)
                expected = f"idx_{table.lower()}_{suffix}"

                # Short names should remain unchanged for backward compatibility
                if len(expected.encode("utf-8")) <= 63:
                    assert (
                        result == expected
                    ), f"Short name changed unexpectedly: {result} != {expected}"
