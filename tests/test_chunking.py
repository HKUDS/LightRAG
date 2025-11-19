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
