import pytest

from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.operate import chunking_by_token_size
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="dummy", tokenizer=DummyTokenizer())


# ============================================================================
# Tests for split_by_character_only=True (raises error on oversized chunks)
# ============================================================================


@pytest.mark.offline
def test_split_by_character_only_within_limit():
    """Test chunking when all chunks are within token limit."""
    tokenizer = make_tokenizer()

    chunks = chunking_by_token_size(
        tokenizer,
        "alpha\n\nbeta",
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert [chunk["content"] for chunk in chunks] == ["alpha", "beta"]


@pytest.mark.offline
def test_split_by_character_only_exceeding_limit_raises():
    """Test that oversized chunks raise ChunkTokenLimitExceededError."""
    tokenizer = make_tokenizer()
    oversized = "a" * 12

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            oversized,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=5,
        )

    err = excinfo.value
    assert err.chunk_tokens == len(oversized)
    assert err.chunk_token_limit == 5


@pytest.mark.offline
def test_chunk_error_includes_preview():
    """Test that error message includes chunk preview."""
    tokenizer = make_tokenizer()
    oversized = "x" * 100

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            oversized,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=10,
        )

    err = excinfo.value
    # Preview should be first 80 chars of a 100-char string
    assert err.chunk_preview == "x" * 80
    assert "Preview:" in str(err)


@pytest.mark.offline
def test_split_by_character_only_at_exact_limit():
    """Test chunking when chunk is exactly at token limit."""
    tokenizer = make_tokenizer()
    exact_size = "a" * 10

    chunks = chunking_by_token_size(
        tokenizer,
        exact_size,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == exact_size
    assert chunks[0]["tokens"] == 10


@pytest.mark.offline
def test_split_by_character_only_one_over_limit():
    """Test that chunk with one token over limit raises error."""
    tokenizer = make_tokenizer()
    one_over = "a" * 11

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            one_over,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=10,
        )

    err = excinfo.value
    assert err.chunk_tokens == 11
    assert err.chunk_token_limit == 10


# ============================================================================
# Tests for split_by_character_only=False (recursive splitting)
# ============================================================================


@pytest.mark.offline
def test_split_recursive_oversized_chunk():
    """Test recursive splitting of oversized chunk with split_by_character_only=False."""
    tokenizer = make_tokenizer()
    # 30 chars - should split into chunks of size 10
    oversized = "a" * 30

    chunks = chunking_by_token_size(
        tokenizer,
        oversized,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    # Should create 3 chunks of 10 tokens each
    assert len(chunks) == 3
    assert all(chunk["tokens"] == 10 for chunk in chunks)
    assert all(chunk["content"] == "a" * 10 for chunk in chunks)


@pytest.mark.offline
def test_split_with_chunk_overlap():
    """Test chunk splitting with overlap."""
    tokenizer = make_tokenizer()
    # 25 chars
    content = "a" * 25

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=3,
    )

    # With overlap of 3, chunks start at: 0, 7, 14, 21
    # Chunk 1: [0:10] = 10 tokens
    # Chunk 2: [7:17] = 10 tokens
    # Chunk 3: [14:24] = 10 tokens
    # Chunk 4: [21:25] = 4 tokens
    assert len(chunks) == 4
    assert chunks[0]["tokens"] == 10
    assert chunks[1]["tokens"] == 10
    assert chunks[2]["tokens"] == 10
    assert chunks[3]["tokens"] == 4


@pytest.mark.offline
def test_split_multiple_chunks_with_mixed_sizes():
    """Test splitting text with multiple chunks of different sizes."""
    tokenizer = make_tokenizer()
    # "small\n\nlarge_chunk_here\n\nmedium"
    # small: 5 tokens, large_chunk_here: 16 tokens, medium: 6 tokens
    content = "small\n\n" + "a" * 16 + "\n\nmedium"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=2,
    )

    # First chunk "small" should be kept as is (5 tokens)
    # Second chunk (16 tokens) should be split into 2 chunks
    # Third chunk "medium" should be kept as is (6 tokens)
    assert len(chunks) == 4
    assert chunks[0]["content"] == "small"
    assert chunks[0]["tokens"] == 5


@pytest.mark.offline
def test_split_exact_boundary():
    """Test splitting at exact chunk boundaries."""
    tokenizer = make_tokenizer()
    # Exactly 20 chars, should split into 2 chunks of 10
    content = "a" * 20

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    assert len(chunks) == 2
    assert chunks[0]["tokens"] == 10
    assert chunks[1]["tokens"] == 10


@pytest.mark.offline
def test_split_very_large_text():
    """Test splitting very large text into multiple chunks."""
    tokenizer = make_tokenizer()
    # 100 chars should create 10 chunks with chunk_size=10, overlap=0
    content = "a" * 100

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    assert len(chunks) == 10
    assert all(chunk["tokens"] == 10 for chunk in chunks)


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.offline
def test_empty_content():
    """Test chunking with empty content."""
    tokenizer = make_tokenizer()

    chunks = chunking_by_token_size(
        tokenizer,
        "",
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == ""
    assert chunks[0]["tokens"] == 0


@pytest.mark.offline
def test_single_character():
    """Test chunking with single character."""
    tokenizer = make_tokenizer()

    chunks = chunking_by_token_size(
        tokenizer,
        "a",
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == "a"
    assert chunks[0]["tokens"] == 1


@pytest.mark.offline
def test_no_delimiter_in_content():
    """Test chunking when content has no delimiter."""
    tokenizer = make_tokenizer()
    content = "a" * 30

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",  # Delimiter not in content
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    # Should still split based on token size
    assert len(chunks) == 3
    assert all(chunk["tokens"] == 10 for chunk in chunks)


@pytest.mark.offline
def test_no_split_character():
    """Test chunking without split_by_character (None)."""
    tokenizer = make_tokenizer()
    content = "a" * 30

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=None,
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    # Should split based purely on token size
    assert len(chunks) == 3
    assert all(chunk["tokens"] == 10 for chunk in chunks)


# ============================================================================
# Parameter Combinations
# ============================================================================


@pytest.mark.offline
def test_different_delimiter_newline():
    """Test with single newline delimiter."""
    tokenizer = make_tokenizer()
    content = "alpha\nbeta\ngamma"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 3
    assert [c["content"] for c in chunks] == ["alpha", "beta", "gamma"]


@pytest.mark.offline
def test_different_delimiter_comma():
    """Test with comma delimiter."""
    tokenizer = make_tokenizer()
    content = "one,two,three"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=",",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 3
    assert [c["content"] for c in chunks] == ["one", "two", "three"]


@pytest.mark.offline
def test_zero_overlap():
    """Test with zero overlap (no overlap)."""
    tokenizer = make_tokenizer()
    content = "a" * 20

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=None,
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    # Should create exactly 2 chunks with no overlap
    assert len(chunks) == 2
    assert chunks[0]["tokens"] == 10
    assert chunks[1]["tokens"] == 10


@pytest.mark.offline
def test_large_overlap():
    """Test with overlap close to chunk size."""
    tokenizer = make_tokenizer()
    content = "a" * 30

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=None,
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=9,
    )

    # With overlap=9, chunks start at: 0, 1, 2, 3...
    # Step size = chunk_size - overlap = 10 - 9 = 1
    # So we get: [0:10], [1:11], [2:12], ..., [29:30]
    # range(0, 30, 1) = 0 to 29, so 30 chunks total
    assert len(chunks) == 30


# ============================================================================
# Chunk Order Index Tests
# ============================================================================


@pytest.mark.offline
def test_chunk_order_index_simple():
    """Test that chunk_order_index is correctly assigned."""
    tokenizer = make_tokenizer()
    content = "a\n\nb\n\nc"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 3
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[1]["chunk_order_index"] == 1
    assert chunks[2]["chunk_order_index"] == 2


@pytest.mark.offline
def test_chunk_order_index_with_splitting():
    """Test chunk_order_index with recursive splitting."""
    tokenizer = make_tokenizer()
    content = "a" * 30

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=None,
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    assert len(chunks) == 3
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[1]["chunk_order_index"] == 1
    assert chunks[2]["chunk_order_index"] == 2


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.offline
def test_mixed_size_chunks_no_error():
    """Test that mixed size chunks work without error in recursive mode."""
    tokenizer = make_tokenizer()
    # Mix of small and large chunks
    content = "small\n\n" + "a" * 50 + "\n\nmedium"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=2,
    )

    # Should handle all chunks without error
    assert len(chunks) > 0
    # Small chunk should remain intact
    assert chunks[0]["content"] == "small"
    # Large chunk should be split into multiple pieces
    assert any(chunk["content"] == "a" * 10 for chunk in chunks)
    # Last chunk should contain "medium"
    assert any("medium" in chunk["content"] for chunk in chunks)


@pytest.mark.offline
def test_whitespace_handling():
    """Test that whitespace is properly handled in chunk content."""
    tokenizer = make_tokenizer()
    content = "  alpha  \n\n  beta  "

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=20,
    )

    # Content should be stripped
    assert chunks[0]["content"] == "alpha"
    assert chunks[1]["content"] == "beta"


@pytest.mark.offline
def test_consecutive_delimiters():
    """Test handling of consecutive delimiters."""
    tokenizer = make_tokenizer()
    content = "alpha\n\n\n\nbeta"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=20,
    )

    # Should split on delimiter and include empty chunks
    assert len(chunks) >= 2
    assert "alpha" in [c["content"] for c in chunks]
    assert "beta" in [c["content"] for c in chunks]
