"""Unit tests for ``chunking_by_recursive_character`` (process_options=R)."""

import pytest

pytest.importorskip("langchain_text_splitters")
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from lightrag.chunker import chunking_by_recursive_character  # noqa: E402
from lightrag.chunker.recursive_character import _split_text_with_spans  # noqa: E402
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


@pytest.mark.offline
def test_long_unique_text_keeps_every_source_span():
    """Every chunk of a long, non-repeating document must carry a span.

    Regression for the LangChain ``add_start_index`` unit mismatch: with a
    token-based length function its search cursor overshot each chunk's true
    start, dropping ``_source_span`` (and failing ``require_source_span``
    backfill) on roughly half of a unique document's chunks.
    """
    body = " ".join(f"word{i}" for i in range(800))
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=50,
        chunk_overlap_token_size=10,
    )

    assert len(chunks) > 5
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_repeated_text_spans_tile_whole_document():
    """Spans over heavily repeated text must tile the document, not cluster.

    Regression for the repeated-content ambiguity: a naive forward ``find``
    matches every identical overlapping chunk to the nearest repetition, so all
    spans collapse into the document head and the tail gets no provenance. The
    offset-aware splitter mirror must instead advance spans across the whole text.
    """
    unit = "ABCDE fghij. "
    body = unit * 60
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=40,
        chunk_overlap_token_size=10,
    )

    spans = [c["_source_span"] for c in chunks]
    assert len(spans) == len(chunks)  # no chunk lost its span

    # Each span is exact and starts are monotonically non-decreasing.
    for chunk, span in zip(chunks, spans):
        assert body[span["start"] : span["end"]] == chunk["content"]
    starts = [s["start"] for s in spans]
    assert starts == sorted(starts)

    # Spans reach the document tail rather than clustering at the head. Under
    # the old nearest-repetition behaviour the furthest end stalled near the
    # first few hundred chars; tiling reaches within one ``unit`` of the end.
    assert max(s["end"] for s in spans) >= len(body) - len(unit)


@pytest.mark.offline
def test_span_mirror_matches_langchain_split_text():
    """Drift guard for the offset-aware mirror of LangChain's splitter."""
    body = (
        "Alpha repeats.\n\n"
        "ff hh aa hh ee\n"
        "ff hh aa hh ee\n\n"
        "Tail repeats. Tail repeats. Tail repeats."
    )
    tok = _tok()

    def length_function(text: str) -> int:
        return len(tok.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=36,
        chunk_overlap=14,
        length_function=length_function,
        strip_whitespace=True,
    )

    mirrored = _split_text_with_spans(
        body,
        base_offset=0,
        separators=splitter._separators,
        chunk_size=splitter._chunk_size,
        chunk_overlap=splitter._chunk_overlap,
        length_function=length_function,
        keep_separator=splitter._keep_separator,
        is_separator_regex=splitter._is_separator_regex,
        strip_whitespace=splitter._strip_whitespace,
    )

    assert [piece for piece, _, _ in mirrored] == splitter.split_text(body)
    for piece, start, end in mirrored:
        assert body[start:end] == piece


@pytest.mark.offline
def test_chunk_repeating_an_earlier_block_maps_to_its_own_block():
    """A chunk whose text also appears in an earlier, different block must map
    to *its own* occurrence, not the earlier copy.

    Regression for cross-block duplicate provenance: merged text ``"ff aa\\n\\naa"``
    splits into ``["ff aa", "aa"]``. The second chunk ``"aa"`` is block 2, so its
    span must point at the *second* ``"aa"`` (offset 7), not the ``"aa"`` inside
    ``"ff aa"`` (offset 3). The earlier locator anchored on a predicted offset
    and snapped to the nearest copy, silently emitting the wrong — but
    legal-looking — span that strict backfill would trust.
    """
    merged = "ff aa\n\naa"
    chunks = chunking_by_recursive_character(
        _tok(),
        merged,
        chunk_token_size=5,
        chunk_overlap_token_size=1,
    )

    assert [c["content"] for c in chunks] == ["ff aa", "aa"]
    assert chunks[0]["_source_span"] == {"start": 0, "end": 5}
    # The second "aa" lives in block 2 at offset 7, after the "\n\n" separator —
    # not the "aa" embedded in "ff aa" at offset 3.
    assert chunks[1]["_source_span"] == {"start": 7, "end": 9}
    assert merged[7:9] == "aa"


@pytest.mark.offline
def test_repeated_block_window_does_not_shift_to_previous_duplicate():
    """A repeated multi-block chunk must not slide back into the prior duplicate."""
    block = "ff hh aa hh ee"
    merged = "\n\n".join([block] * 4)
    chunks = chunking_by_recursive_character(
        _tok(),
        merged,
        chunk_token_size=36,
        chunk_overlap_token_size=14,
    )

    assert [c["content"] for c in chunks] == [
        f"{block}\n\n{block}",
        f"{block}\n\n{block}",
    ]
    assert chunks[0]["_source_span"] == {"start": 0, "end": 30}
    assert chunks[1]["_source_span"] == {"start": 32, "end": 62}
