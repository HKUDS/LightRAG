"""Tests for citation utility functions in lightrag/utils.py.

This module tests the helper functions used for generating citations
and reference lists from document chunks.
"""

import pytest

from lightrag.utils import (
    _extract_document_title,
    _generate_excerpt,
    generate_reference_list_from_chunks,
)

# ============================================================================
# Tests for _extract_document_title()
# ============================================================================


class TestExtractDocumentTitle:
    """Tests for _extract_document_title function."""

    @pytest.mark.offline
    def test_regular_path(self):
        """Test extracting title from regular file path."""
        assert _extract_document_title('/path/to/document.pdf') == 'document.pdf'

    @pytest.mark.offline
    def test_nested_path(self):
        """Test extracting title from deeply nested path."""
        assert _extract_document_title('/a/b/c/d/e/report.docx') == 'report.docx'

    @pytest.mark.offline
    def test_s3_path(self):
        """Test extracting title from S3 URL."""
        assert _extract_document_title('s3://bucket/archive/default/doc123/report.pdf') == 'report.pdf'

    @pytest.mark.offline
    def test_s3_path_simple(self):
        """Test extracting title from simple S3 URL."""
        assert _extract_document_title('s3://mybucket/file.txt') == 'file.txt'

    @pytest.mark.offline
    def test_empty_string(self):
        """Test with empty string returns empty."""
        assert _extract_document_title('') == ''

    @pytest.mark.offline
    def test_trailing_slash(self):
        """Test path with trailing slash returns empty."""
        assert _extract_document_title('/path/to/dir/') == ''

    @pytest.mark.offline
    def test_filename_only(self):
        """Test with just a filename (no path)."""
        assert _extract_document_title('document.pdf') == 'document.pdf'

    @pytest.mark.offline
    def test_no_extension(self):
        """Test filename without extension."""
        assert _extract_document_title('/path/to/README') == 'README'

    @pytest.mark.offline
    def test_windows_style_path(self):
        """Test Windows-style path (backslashes)."""
        # os.path.basename handles this correctly on Unix
        result = _extract_document_title('C:\\Users\\docs\\file.pdf')
        # On Unix, this returns the whole string as basename doesn't split on backslash
        assert 'file.pdf' in result or result == 'C:\\Users\\docs\\file.pdf'

    @pytest.mark.offline
    def test_special_characters(self):
        """Test filename with special characters."""
        assert _extract_document_title('/path/to/my file (1).pdf') == 'my file (1).pdf'

    @pytest.mark.offline
    def test_unicode_filename(self):
        """Test filename with unicode characters."""
        assert _extract_document_title('/path/to/文档.pdf') == '文档.pdf'


# ============================================================================
# Tests for _generate_excerpt()
# ============================================================================


class TestGenerateExcerpt:
    """Tests for _generate_excerpt function."""

    @pytest.mark.offline
    def test_short_content(self):
        """Test content shorter than max_length."""
        assert _generate_excerpt('Hello world') == 'Hello world'

    @pytest.mark.offline
    def test_exact_length(self):
        """Test content exactly at max_length."""
        content = 'a' * 150
        result = _generate_excerpt(content, max_length=150)
        assert result == content  # No ellipsis for exact length

    @pytest.mark.offline
    def test_long_content_truncated(self):
        """Test long content is truncated with ellipsis."""
        content = 'a' * 200
        result = _generate_excerpt(content, max_length=150)
        assert len(result) == 153  # 150 chars + '...'
        assert result.endswith('...')

    @pytest.mark.offline
    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _generate_excerpt('') == ''

    @pytest.mark.offline
    def test_whitespace_stripped(self):
        """Test leading/trailing whitespace is stripped."""
        assert _generate_excerpt('  hello world  ') == 'hello world'

    @pytest.mark.offline
    def test_whitespace_only(self):
        """Test whitespace-only content returns empty."""
        assert _generate_excerpt('   \n\t  ') == ''

    @pytest.mark.offline
    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        content = 'This is a test sentence for excerpts.'
        result = _generate_excerpt(content, max_length=10)
        # Note: rstrip() removes trailing space before adding ellipsis
        assert result == 'This is a...'

    @pytest.mark.offline
    def test_unicode_content(self):
        """Test unicode content handling."""
        content = '日本語テキスト' * 50  # 350 chars
        result = _generate_excerpt(content, max_length=150)
        assert len(result) == 153  # 150 chars + '...'

    @pytest.mark.offline
    def test_newlines_preserved(self):
        """Test that newlines within content are preserved."""
        content = 'Line 1\nLine 2'
        result = _generate_excerpt(content)
        assert result == 'Line 1\nLine 2'

    @pytest.mark.offline
    def test_very_short_max_length(self):
        """Test with very short max_length."""
        result = _generate_excerpt('Hello world', max_length=5)
        assert result == 'Hello...'


# ============================================================================
# Tests for generate_reference_list_from_chunks()
# ============================================================================


class TestGenerateReferenceListFromChunks:
    """Tests for generate_reference_list_from_chunks function."""

    @pytest.mark.offline
    def test_empty_chunks(self):
        """Test with empty chunk list."""
        ref_list, updated_chunks = generate_reference_list_from_chunks([])
        assert ref_list == []
        assert updated_chunks == []

    @pytest.mark.offline
    def test_single_chunk(self):
        """Test with a single chunk."""
        chunks = [
            {
                'file_path': '/path/to/doc.pdf',
                'content': 'This is the content.',
                's3_key': 'archive/doc.pdf',
            }
        ]
        ref_list, updated_chunks = generate_reference_list_from_chunks(chunks)

        assert len(ref_list) == 1
        assert ref_list[0]['reference_id'] == '1'
        assert ref_list[0]['file_path'] == '/path/to/doc.pdf'
        assert ref_list[0]['document_title'] == 'doc.pdf'
        assert ref_list[0]['s3_key'] == 'archive/doc.pdf'
        assert ref_list[0]['excerpt'] == 'This is the content.'

        assert len(updated_chunks) == 1
        assert updated_chunks[0]['reference_id'] == '1'

    @pytest.mark.offline
    def test_multiple_chunks_same_file(self):
        """Test multiple chunks from same file get same reference_id."""
        chunks = [
            {'file_path': '/path/doc.pdf', 'content': 'Chunk 1'},
            {'file_path': '/path/doc.pdf', 'content': 'Chunk 2'},
            {'file_path': '/path/doc.pdf', 'content': 'Chunk 3'},
        ]
        ref_list, updated_chunks = generate_reference_list_from_chunks(chunks)

        assert len(ref_list) == 1
        assert ref_list[0]['reference_id'] == '1'
        # All chunks should have same reference_id
        for chunk in updated_chunks:
            assert chunk['reference_id'] == '1'

    @pytest.mark.offline
    def test_multiple_files_deduplication(self):
        """Test multiple files are deduplicated with unique reference_ids."""
        chunks = [
            {'file_path': '/path/doc1.pdf', 'content': 'Content 1'},
            {'file_path': '/path/doc2.pdf', 'content': 'Content 2'},
            {'file_path': '/path/doc1.pdf', 'content': 'Content 1 more'},
        ]
        ref_list, _updated_chunks = generate_reference_list_from_chunks(chunks)

        assert len(ref_list) == 2
        # doc1 appears twice, so should be reference_id '1' (higher frequency)
        # doc2 appears once, so should be reference_id '2'
        ref_ids = {r['file_path']: r['reference_id'] for r in ref_list}
        assert ref_ids['/path/doc1.pdf'] == '1'
        assert ref_ids['/path/doc2.pdf'] == '2'

    @pytest.mark.offline
    def test_prioritization_by_frequency(self):
        """Test that references are prioritized by frequency."""
        chunks = [
            {'file_path': '/rare.pdf', 'content': 'Rare'},
            {'file_path': '/common.pdf', 'content': 'Common 1'},
            {'file_path': '/common.pdf', 'content': 'Common 2'},
            {'file_path': '/common.pdf', 'content': 'Common 3'},
            {'file_path': '/rare.pdf', 'content': 'Rare 2'},
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)

        # common.pdf appears 3 times, rare.pdf appears 2 times
        # common.pdf should get reference_id '1'
        assert ref_list[0]['file_path'] == '/common.pdf'
        assert ref_list[0]['reference_id'] == '1'
        assert ref_list[1]['file_path'] == '/rare.pdf'
        assert ref_list[1]['reference_id'] == '2'

    @pytest.mark.offline
    def test_unknown_source_filtered(self):
        """Test that 'unknown_source' file paths are filtered out."""
        chunks = [
            {'file_path': '/path/doc.pdf', 'content': 'Valid'},
            {'file_path': 'unknown_source', 'content': 'Unknown'},
            {'file_path': '/path/doc2.pdf', 'content': 'Valid 2'},
        ]
        ref_list, updated_chunks = generate_reference_list_from_chunks(chunks)

        # unknown_source should not be in reference list
        assert len(ref_list) == 2
        file_paths = [r['file_path'] for r in ref_list]
        assert 'unknown_source' not in file_paths

        # Chunk with unknown_source should have empty reference_id
        assert updated_chunks[1]['reference_id'] == ''

    @pytest.mark.offline
    def test_empty_file_path_filtered(self):
        """Test that empty file paths are filtered out."""
        chunks = [
            {'file_path': '/path/doc.pdf', 'content': 'Valid'},
            {'file_path': '', 'content': 'No path'},
            {'content': 'Missing path key'},
        ]
        ref_list, _updated_chunks = generate_reference_list_from_chunks(chunks)

        assert len(ref_list) == 1
        assert ref_list[0]['file_path'] == '/path/doc.pdf'

    @pytest.mark.offline
    def test_s3_key_included(self):
        """Test that s3_key is included in reference list."""
        chunks = [
            {
                'file_path': 's3://bucket/archive/doc.pdf',
                'content': 'S3 content',
                's3_key': 'archive/doc.pdf',
            }
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)

        assert ref_list[0]['s3_key'] == 'archive/doc.pdf'
        assert ref_list[0]['document_title'] == 'doc.pdf'

    @pytest.mark.offline
    def test_excerpt_generated_from_first_chunk(self):
        """Test that excerpt is generated from first chunk of each file."""
        chunks = [
            {'file_path': '/doc.pdf', 'content': 'First chunk content'},
            {'file_path': '/doc.pdf', 'content': 'Second chunk different'},
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)

        # Excerpt should be from first chunk
        assert ref_list[0]['excerpt'] == 'First chunk content'

    @pytest.mark.offline
    def test_excerpt_added_to_each_chunk(self):
        """Test that each updated chunk has its own excerpt."""
        chunks = [
            {'file_path': '/doc.pdf', 'content': 'First chunk'},
            {'file_path': '/doc.pdf', 'content': 'Second chunk'},
        ]
        _, updated_chunks = generate_reference_list_from_chunks(chunks)

        assert updated_chunks[0]['excerpt'] == 'First chunk'
        assert updated_chunks[1]['excerpt'] == 'Second chunk'

    @pytest.mark.offline
    def test_original_chunks_not_modified(self):
        """Test that original chunks are not modified (returns copies)."""
        original_chunks = [
            {'file_path': '/doc.pdf', 'content': 'Content'},
        ]
        _, updated_chunks = generate_reference_list_from_chunks(original_chunks)

        # Original should not have reference_id
        assert 'reference_id' not in original_chunks[0]
        # Updated should have reference_id
        assert 'reference_id' in updated_chunks[0]

    @pytest.mark.offline
    def test_missing_s3_key_is_none(self):
        """Test that missing s3_key results in None."""
        chunks = [
            {'file_path': '/local/doc.pdf', 'content': 'Local file'},
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)

        assert ref_list[0]['s3_key'] is None

    @pytest.mark.offline
    def test_tie_breaking_by_first_appearance(self):
        """Test that same-frequency files are ordered by first appearance."""
        chunks = [
            {'file_path': '/doc_b.pdf', 'content': 'B first'},
            {'file_path': '/doc_a.pdf', 'content': 'A second'},
            {'file_path': '/doc_b.pdf', 'content': 'B again'},
            {'file_path': '/doc_a.pdf', 'content': 'A again'},
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)

        # Both files appear twice, but doc_b appeared first
        assert ref_list[0]['file_path'] == '/doc_b.pdf'
        assert ref_list[0]['reference_id'] == '1'
        assert ref_list[1]['file_path'] == '/doc_a.pdf'
        assert ref_list[1]['reference_id'] == '2'
