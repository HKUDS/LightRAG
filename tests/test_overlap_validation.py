"""
Test for overlap_tokens validation to prevent infinite loop.

This test validates the fix for the bug where overlap_tokens >= max_tokens
causes an infinite loop in the chunking function.
"""

import pytest

from lightrag.rerank import chunk_documents_for_rerank


@pytest.mark.offline
class TestOverlapValidation:
    """Test suite for overlap_tokens validation"""

    def test_overlap_greater_than_max_tokens(self):
        """Test that overlap_tokens > max_tokens is clamped and doesn't hang"""
        documents = [" ".join([f"word{i}" for i in range(100)])]

        # This should clamp overlap_tokens to 29 (max_tokens - 1)
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=30, overlap_tokens=32
        )

        # Should complete without hanging
        assert len(chunked_docs) > 0
        assert all(idx == 0 for idx in doc_indices)

    def test_overlap_equal_to_max_tokens(self):
        """Test that overlap_tokens == max_tokens is clamped and doesn't hang"""
        documents = [" ".join([f"word{i}" for i in range(100)])]

        # This should clamp overlap_tokens to 29 (max_tokens - 1)
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=30, overlap_tokens=30
        )

        # Should complete without hanging
        assert len(chunked_docs) > 0
        assert all(idx == 0 for idx in doc_indices)

    def test_overlap_slightly_less_than_max_tokens(self):
        """Test that overlap_tokens < max_tokens works normally"""
        documents = [" ".join([f"word{i}" for i in range(100)])]

        # This should work without clamping
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=30, overlap_tokens=29
        )

        # Should complete successfully
        assert len(chunked_docs) > 0
        assert all(idx == 0 for idx in doc_indices)

    def test_small_max_tokens_with_large_overlap(self):
        """Test edge case with very small max_tokens"""
        documents = [" ".join([f"word{i}" for i in range(50)])]

        # max_tokens=5, overlap_tokens=10 should clamp to 4
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=5, overlap_tokens=10
        )

        # Should complete without hanging
        assert len(chunked_docs) > 0
        assert all(idx == 0 for idx in doc_indices)

    def test_multiple_documents_with_invalid_overlap(self):
        """Test multiple documents with overlap_tokens >= max_tokens"""
        documents = [
            " ".join([f"word{i}" for i in range(50)]),
            "short document",
            " ".join([f"word{i}" for i in range(75)]),
        ]

        # overlap_tokens > max_tokens
        chunked_docs, _ = chunk_documents_for_rerank(
            documents, max_tokens=25, overlap_tokens=30
        )

        # Should complete successfully and chunk the long documents
        assert len(chunked_docs) >= len(documents)
        # Short document should not be chunked
        assert "short document" in chunked_docs

    def test_normal_operation_unaffected(self):
        """Test that normal cases continue to work correctly"""
        documents = [
            " ".join([f"word{i}" for i in range(100)]),
            "short doc",
        ]

        # Normal case: overlap_tokens (10) < max_tokens (50)
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=50, overlap_tokens=10
        )

        # Long document should be chunked, short one should not
        assert len(chunked_docs) > 2  # At least 3 chunks (2 from long doc + 1 short)
        assert "short doc" in chunked_docs
        # Verify doc_indices maps correctly
        assert doc_indices[-1] == 1  # Last chunk is from second document

    def test_edge_case_max_tokens_one(self):
        """Test edge case where max_tokens=1"""
        documents = [" ".join([f"word{i}" for i in range(20)])]

        # max_tokens=1, overlap_tokens=5 should clamp to 0
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=1, overlap_tokens=5
        )

        # Should complete without hanging
        assert len(chunked_docs) > 0
        assert all(idx == 0 for idx in doc_indices)
