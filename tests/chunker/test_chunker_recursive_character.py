"""Unit tests for ``chunking_by_recursive_character`` (process_options=R)."""

import pytest

pytest.importorskip("langchain_text_splitters")

from lightrag.chunker import chunking_by_recursive_character  # noqa: E402
from lightrag.utils import Tokenizer, TokenizerInterface  # noqa: E402


class _CharTokenizer(TokenizerInterface):
    """1 char ≈ 1 token; lets assertions reason in terms of input length."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


@pytest.mark.offline
def test_empty_input_returns_empty_list():
    chunks = chunking_by_recursive_character(_tok(), "")
    assert chunks == []


@pytest.mark.offline
def test_short_input_single_chunk():
    body = "Para A.\n\nPara B."
    chunks = chunking_by_recursive_character(_tok(), body, chunk_token_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["content"] == body
    assert chunks[0]["_source_span"] == {"start": 0, "end": len(body)}
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[0]["tokens"] == len(body)


@pytest.mark.offline
def test_paragraph_separator_used_first():
    """``\\n\\n`` is the first separator in the default cascade — three
    paragraphs that each fit under the cap should split exactly there."""
    body = "Alpha section.\n\nBeta section.\n\nGamma section."
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=20,
        chunk_overlap_token_size=0,
    )

    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["content"].strip() for c in chunks)
    # Reconstructed (joined with the splitter's separator semantics) must
    # at least contain each original paragraph as a substring.
    joined = "\n\n".join(c["content"] for c in chunks)
    for para in ("Alpha section.", "Beta section.", "Gamma section."):
        assert para in joined


@pytest.mark.offline
def test_token_field_matches_tokenizer_encode_length():
    chunks = chunking_by_recursive_character(
        _tok(),
        "X" * 50 + "\n\n" + "Y" * 50,
        chunk_token_size=40,
        chunk_overlap_token_size=5,
    )
    tok = _tok()
    for c in chunks:
        assert c["tokens"] == len(tok.encode(c["content"]))


@pytest.mark.offline
def test_custom_separators_are_honored():
    body = "alpha|beta|gamma|delta"
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
        separators=["|", ""],
    )
    contents = [c["content"] for c in chunks]
    # With "|" as the primary separator and a 10-token cap, each 5-char
    # token name must land in its own chunk.
    assert any("alpha" in c for c in contents)
    assert any("delta" in c for c in contents)
    # Every chunk fits the cap.
    for c in chunks:
        assert c["tokens"] <= 10


@pytest.mark.offline
def test_recursive_chunks_carry_exact_source_spans_with_overlap():
    body = "Alpha section.\n\nBeta section.\n\nGamma section."
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=22,
        chunk_overlap_token_size=6,
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]
