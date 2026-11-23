import pytest

from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.chunker import chunking_by_token_size
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class MultiTokenCharacterTokenizer(TokenizerInterface):
    """
    Tokenizer where character-to-token ratio is non-uniform.
    This helps catch bugs where code incorrectly counts characters instead of tokens.

    Mapping:
    - Uppercase letters: 2 tokens each
    - Punctuation (!, ?, .): 3 tokens each
    - Other characters: 1 token each
    """

    def encode(self, content: str):
        tokens = []
        for ch in content:
            if ch.isupper():  # Uppercase = 2 tokens
                tokens.extend([ord(ch), ord(ch) + 1000])
            elif ch in ["!", "?", "."]:  # Punctuation = 3 tokens
                tokens.extend([ord(ch), ord(ch) + 2000, ord(ch) + 3000])
            else:  # Regular chars = 1 token
                tokens.append(ord(ch))
        return tokens

    def decode(self, tokens):
        # Simplified decode for testing
        result = []
        i = 0
        while i < len(tokens):
            base_token = tokens[i]
            # Check if this is part of a multi-token sequence
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] == base_token + 2000
                and tokens[i + 2] == base_token + 3000
            ):
                # 3-token punctuation
                result.append(chr(base_token))
                i += 3
            elif i + 1 < len(tokens) and tokens[i + 1] == base_token + 1000:
                # 2-token uppercase
                result.append(chr(base_token))
                i += 2
            else:
                # Single token
                result.append(chr(base_token))
                i += 1
        return "".join(result)


def make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="dummy", tokenizer=DummyTokenizer())


def make_multi_token_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="multi", tokenizer=MultiTokenCharacterTokenizer())


# ============================================================================
# Tests for split_by_character_only=True (raises error on oversized chunks)
# ============================================================================


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
def test_split_with_chunk_overlap():
    """
    Test chunk splitting with overlap using distinctive content.

    With distinctive characters, we can verify overlap positions are exact.
    Misaligned overlap would produce wrong content and fail the test.
    """
    tokenizer = make_tokenizer()
    # Each character is unique - enables exact position verification
    content = "0123456789abcdefghijklmno"  # 25 chars

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=3,
    )

    # With overlap=3, step size = chunk_size - overlap = 10 - 3 = 7
    # Chunks start at positions: 0, 7, 14, 21
    assert len(chunks) == 4

    # Verify exact content and token counts
    assert chunks[0]["tokens"] == 10
    assert chunks[0]["content"] == "0123456789"  # [0:10]

    assert chunks[1]["tokens"] == 10
    assert chunks[1]["content"] == "789abcdefg"  # [7:17] - overlaps with "789"

    assert chunks[2]["tokens"] == 10
    assert chunks[2]["content"] == "efghijklmn"  # [14:24] - overlaps with "efg"

    assert chunks[3]["tokens"] == 4
    assert chunks[3]["content"] == "lmno"  # [21:25] - overlaps with "lmn"


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


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.offline
def test_overlap_greater_than_chunk_size_raises():
    """Overlap larger than chunk size makes the stride negative, which yields an
    empty range and silently drops the entire document. Fail closed instead."""
    tokenizer = make_tokenizer()
    content = "abcdefghijklmnopqrstuvwxyz0123456789"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=50,
            chunk_overlap_token_size=100,
        )

    message = str(excinfo.value)
    assert "chunk_overlap_token_size (100)" in message
    assert "chunk_token_size (50)" in message


@pytest.mark.offline
def test_overlap_equal_to_chunk_size_raises():
    """Overlap equal to chunk size makes the stride zero, which otherwise raises
    an opaque ``range() arg 3 must not be zero``. Raise a clear message instead."""
    tokenizer = make_tokenizer()
    content = "abcdefghijklmnopqrstuvwxyz0123456789"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=10,
            chunk_overlap_token_size=10,
        )

    message = str(excinfo.value)
    assert "chunk_overlap_token_size (10)" in message
    assert "chunk_token_size (10)" in message


@pytest.mark.offline
def test_negative_overlap_raises():
    """Negative overlap makes the stride larger than chunk_token_size (e.g.
    step = 10 - (-1) = 11 for a 10-token chunk), silently skipping a token
    between every window instead of the intended overlap. Fail closed,
    matching the >= chunk_size case above."""
    tokenizer = make_tokenizer()
    content = "abcdefghijklmnopqrstuvwxyz0123456789"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=10,
            chunk_overlap_token_size=-1,
        )

    message = str(excinfo.value)
    assert "chunk_overlap_token_size must be non-negative" in message
    assert "-1" in message


@pytest.mark.offline
@pytest.mark.parametrize("chunk_overlap_token_size", [0, 5])
def test_valid_overlap_still_covers_every_token(chunk_overlap_token_size):
    """Control: zero and a valid positive overlap must still produce
    complete, gap-free coverage of the source content -- guards the
    negative-overlap rejection above against accidentally tightening or
    loosening a legitimate case.

    DummyTokenizer maps each character to its own token (ord(ch)), so a
    string of distinct printable characters lets coverage be checked by
    set membership without collisions.
    """
    tokenizer = make_tokenizer()
    content = "".join(chr(33 + i) for i in range(90))

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        chunk_token_size=10,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )

    covered = set()
    for chunk in chunks:
        covered.update(chunk["content"])

    assert covered == set(content)


@pytest.mark.offline
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_no_trailing_window_inside_previous_window(split_by_character):
    """A window that reaches the end of the text is the last one.

    17 tokens, size 10, overlap 3 (step 7): the windows [0:10] and [7:17]
    already cover everything. The next start (14) used to emit [14:17],
    which is the tail of [7:17] and adds no text. The "|" case sends the
    same text through the split_by_character path as one oversize segment.
    """
    tokenizer = make_tokenizer()
    content = "0123456789abcdefg"  # 17 chars

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=split_by_character,
        chunk_token_size=10,
        chunk_overlap_token_size=3,
    )

    assert [chunk["content"] for chunk in chunks] == ["0123456789", "789abcdefg"]


@pytest.mark.offline
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_every_window_adds_new_text(split_by_character):
    """Each window must end past the previous one, the last must reach the end
    of the text, and the count must be the minimum that covers it.

    A value sweep (see *Failures name the case* in
    docs/design/BackendTestingGuide.md): ``split_by_character`` selects the
    code path and stays a node; length and overlap only change the window
    arithmetic, so every combination is checked against the formula below
    and every mismatch is reported at once.
    """
    tokenizer = make_tokenizer()
    chunk_token_size = 10
    mismatches = []
    for chunk_overlap_token_size in (0, 3, 6, 9):
        step = chunk_token_size - chunk_overlap_token_size
        for length in range(1, 40):
            content = "".join(chr(33 + i) for i in range(length))
            chunks = chunking_by_token_size(
                tokenizer,
                content,
                split_by_character=split_by_character,
                chunk_token_size=chunk_token_size,
                chunk_overlap_token_size=chunk_overlap_token_size,
                _emit_source_span=True,
            )
            ends = [chunk["_source_span"]["end"] for chunk in chunks]
            expected_count = 1 + max(0, -(-(length - chunk_token_size) // step))
            if (
                not ends
                or ends != sorted(set(ends))
                or ends[-1] != length
                or len(chunks) != expected_count
            ):
                mismatches.append(
                    (length, chunk_overlap_token_size, ends, expected_count)
                )

    assert mismatches == []


@pytest.mark.offline
@pytest.mark.parametrize("split_by_character_only", [False, True])
def test_negative_overlap_raises_with_split_by_character(split_by_character_only):
    """Negative overlap must be rejected even when every split_by_character
    segment stays under chunk_token_size -- that path never reaches
    _window_step()'s inline calls, so the check has to run up front."""
    tokenizer = make_tokenizer()
    content = "abc|def|ghi"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            split_by_character="|",
            split_by_character_only=split_by_character_only,
            chunk_token_size=10,
            chunk_overlap_token_size=-1,
        )

    assert "chunk_overlap_token_size must be non-negative" in str(excinfo.value)


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


# ============================================================================
# Parameter Combinations
# ============================================================================


@pytest.mark.offline
def test_multi_character_delimiter_splitting():
    """
    Verify that multi-character delimiters are correctly recognized and not partially matched.

    Tests various multi-character delimiter scenarios to ensure the entire delimiter
    sequence is used for splitting, not individual characters.
    """
    tokenizer = make_tokenizer()

    # Test 1: Multi-character delimiter that contains single chars also present elsewhere
    content = "data<SEP>more<SEP>final"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="<SEP>",
        split_by_character_only=True,
        chunk_token_size=50,
    )

    assert len(chunks) == 3
    assert chunks[0]["content"] == "data"
    assert chunks[1]["content"] == "more"
    assert chunks[2]["content"] == "final"
    # Verify full delimiter is not in chunks, not just parts
    for chunk in chunks:
        assert "<SEP>" not in chunk["content"]

    # Test 2: Delimiter appears in middle of content
    content = "first><second><third"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="><",  # Multi-char delimiter
        split_by_character_only=True,
        chunk_token_size=50,
    )

    # Should split at "><" delimiter
    assert len(chunks) == 3
    assert chunks[0]["content"] == "first"
    assert chunks[1]["content"] == "second"
    assert chunks[2]["content"] == "third"

    # Test 3: Three-character delimiter
    content = "section1[***]section2[***]section3"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="[***]",
        split_by_character_only=True,
        chunk_token_size=50,
    )

    assert len(chunks) == 3
    assert chunks[0]["content"] == "section1"
    assert chunks[1]["content"] == "section2"
    assert chunks[2]["content"] == "section3"

    # Test 4: Delimiter with special regex characters (should be treated literally)
    content = "partA...partB...partC"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="...",
        split_by_character_only=True,
        chunk_token_size=50,
    )

    assert len(chunks) == 3
    assert chunks[0]["content"] == "partA"
    assert chunks[1]["content"] == "partB"
    assert chunks[2]["content"] == "partC"


@pytest.mark.offline
def test_delimiter_partial_match_not_split():
    """
    Verify that partial matches of multi-character delimiters don't cause splits.

    Only the complete delimiter sequence should trigger a split.
    """
    tokenizer = make_tokenizer()

    # Content contains "||" delimiter but also contains single "|"
    content = "data|single||data|with|pipes||final"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="||",  # Only split on double pipe
        split_by_character_only=True,
        chunk_token_size=50,
    )

    # Should split only at "||", not at single "|"
    assert len(chunks) == 3
    assert chunks[0]["content"] == "data|single"
    assert chunks[1]["content"] == "data|with|pipes"
    assert chunks[2]["content"] == "final"

    # Single "|" should remain in content, but not double "||"
    assert "|" in chunks[0]["content"]
    assert "|" in chunks[1]["content"]
    assert "||" not in chunks[0]["content"]
    assert "||" not in chunks[1]["content"]


@pytest.mark.offline
def test_no_delimiter_forces_token_based_split():
    """
    Verify that when split_by_character doesn't appear in content,
    chunking falls back to token-based splitting.
    """
    tokenizer = make_tokenizer()

    # Content without the specified delimiter
    content = "0123456789abcdefghijklmnop"  # 26 chars, no "\n\n"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",  # Delimiter not in content
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    # Should fall back to token-based splitting
    assert len(chunks) == 3
    assert chunks[0]["content"] == "0123456789"  # [0:10]
    assert chunks[1]["content"] == "abcdefghij"  # [10:20]
    assert chunks[2]["content"] == "klmnop"  # [20:26]

    # Verify it didn't somehow split at the delimiter that doesn't exist
    for chunk in chunks:
        assert "\n\n" not in chunk["content"]


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


# ============================================================================
# Token vs Character Counting Tests (Multi-Token Characters)
# ============================================================================


@pytest.mark.offline
def test_token_counting_not_character_counting():
    """
    Verify chunking uses token count, not character count.

    With MultiTokenCharacterTokenizer:
    - "aXa" = 3 chars but 4 tokens (a=1, X=2, a=1)

    This test would PASS if code incorrectly used character count (3 <= 3)
    but correctly FAILS because token count (4 > 3).
    """
    tokenizer = make_multi_token_tokenizer()

    # "aXa" = 3 characters, 4 tokens
    content = "aXa"

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=3,  # 3 token limit
        )

    err = excinfo.value
    assert err.chunk_tokens == 4  # Should be 4 tokens, not 3 characters
    assert err.chunk_token_limit == 3


@pytest.mark.offline
def test_recursive_split_with_multi_token_chars():
    """
    Test recursive splitting respects token boundaries, not character boundaries.

    "AAAAA" = 5 chars but 10 tokens (each A = 2 tokens)
    With chunk_size=6, should split at token positions, not character positions.
    """
    tokenizer = make_multi_token_tokenizer()

    # "AAAAA" = 5 characters, 10 tokens
    content = "AAAAA"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=6,
        chunk_overlap_token_size=0,
    )

    # Should split into: [0:6]=3 chars, [6:10]=2 chars
    # Not [0:3]=6 tokens, [3:5]=4 tokens (character-based would be wrong)
    assert len(chunks) == 2
    assert chunks[0]["tokens"] == 6
    assert chunks[1]["tokens"] == 4


@pytest.mark.offline
def test_overlap_uses_token_count():
    """
    Verify overlap calculation uses token count, not character count.

    "aAaAa" = 5 chars, 7 tokens (a=1, A=2, a=1, A=2, a=1)
    """
    tokenizer = make_multi_token_tokenizer()

    # "aAaAa" = 5 characters, 7 tokens
    content = "aAaAa"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=4,
        chunk_overlap_token_size=2,
    )

    # Chunks start at token positions: 0, 2, 4. [4:7] reaches the end, so the
    # start at 6 (a tail of [4:7]) is not used.
    # [0:4]=2 chars, [2:6]=2.5 chars, [4:7]=1.5 chars
    assert len(chunks) == 3
    assert chunks[0]["tokens"] == 4
    assert chunks[1]["tokens"] == 4
    assert chunks[2]["tokens"] == 3


@pytest.mark.offline
def test_mixed_multi_token_content():
    """Test chunking with mixed single and multi-token characters."""
    tokenizer = make_multi_token_tokenizer()

    # "hello\n\nWORLD!" = 12 chars
    # hello = 5 tokens, WORLD = 10 tokens (5 chars × 2), ! = 3 tokens
    # Total = 18 tokens
    content = "hello\n\nWORLD!"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=20,
    )

    assert len(chunks) == 2
    assert chunks[0]["content"] == "hello"
    assert chunks[0]["tokens"] == 5
    assert chunks[1]["content"] == "WORLD!"
    assert chunks[1]["tokens"] == 13  # 10 + 3


# ============================================================================
# Tests: the split_by_character path must not encode the whole document
# (regression guard for the deferred whole-document encode)
# ============================================================================


class CountingTokenizer(DummyTokenizer):
    """DummyTokenizer that records every string passed to encode()."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, content: str):
        self.encoded.append(content)
        return super().encode(content)


def _make_counting_tokenizer() -> tuple[CountingTokenizer, Tokenizer]:
    inner = CountingTokenizer()
    return inner, Tokenizer(model_name="counting", tokenizer=inner)


@pytest.mark.offline
def test_split_by_character_does_not_encode_whole_document():
    """The split_by_character path only encodes each segment, never the
    whole document (the fixed-window ``else`` branch is the only consumer
    of the whole-document token list)."""
    inner, tokenizer = _make_counting_tokenizer()
    content = "alpha\n\nbeta\n\ngamma"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        chunk_token_size=10,
    )

    assert [chunk["content"] for chunk in chunks] == ["alpha", "beta", "gamma"]
    # Exactly one encode per split segment; in particular no call with the
    # full document, whose token list the split path never reads.
    assert inner.encoded == ["alpha", "beta", "gamma"]


@pytest.mark.offline
def test_split_by_character_only_does_not_encode_whole_document():
    """Same guarantee for split_by_character_only=True."""
    inner, tokenizer = _make_counting_tokenizer()
    content = "one\ntwo\nthree"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert [chunk["content"] for chunk in chunks] == ["one", "two", "three"]
    assert inner.encoded == ["one", "two", "three"]


@pytest.mark.offline
def test_fixed_window_path_still_encodes_whole_document():
    """Without split_by_character the whole-document encode still happens."""
    inner, tokenizer = _make_counting_tokenizer()
    content = "abcdefghij"

    chunking_by_token_size(
        tokenizer, content, chunk_token_size=4, chunk_overlap_token_size=0
    )

    assert inner.encoded == [content]
