"""Regression tests for the O(N) delta-decode in ``_token_window_source_span``.

The fixed-token chunker maps each decoded token window back to its exact source
span. The offset of a window's start used to be recomputed with a full
``decode(tokens[:start_token])`` prefix decode on every window — O(N) per window,
O(N²) over a document. It now decodes only the delta since the previous *verified*
anchor (O(N) total). These tests pin three properties:

1. **Equivalence** — the delta path yields byte-identical spans to a reference
   full-prefix implementation, window for window (incl. a non-1:1 tokenizer and a
   real tiktoken BPE tokenizer).
2. **Exactness** — every emitted span is a verbatim slice of the source.
3. **Linear decode budget** — total tokens handed to ``decode`` scales ~O(N), not
   O(N²); the pre-optimization prefix decode would blow well past the bound.
"""

from __future__ import annotations

import pytest

from lightrag.chunker import chunking_by_fixed_token
from lightrag.chunker.token_size import _source_span, _token_window_source_span
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    """1:1 char-per-token; ``decode(encode(x)) == x`` so windows are verbatim."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _MultiTokenTokenizer(TokenizerInterface):
    """Non-uniform char→token ratio: uppercase = 2 tokens, ``.?!`` = 3, else 1.

    Exercises the offset arithmetic where token count != char count, so a bug that
    confuses the two surfaces immediately.
    """

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        for ch in content:
            if ch.isupper():
                tokens.extend([ord(ch), ord(ch) + 1000])
            elif ch in (".", "?", "!"):
                tokens.extend([ord(ch), ord(ch) + 2000, ord(ch) + 3000])
            else:
                tokens.append(ord(ch))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        result: list[str] = []
        i = 0
        while i < len(tokens):
            base = tokens[i]
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] == base + 2000
                and tokens[i + 2] == base + 3000
            ):
                result.append(chr(base))
                i += 3
            elif i + 1 < len(tokens) and tokens[i + 1] == base + 1000:
                result.append(chr(base))
                i += 2
            else:
                result.append(chr(base))
                i += 1
        return "".join(result)


class _CountingTokenizer(TokenizerInterface):
    """Wraps a char tokenizer and tallies every token handed to ``decode``."""

    def __init__(self) -> None:
        self.decoded_tokens = 0

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        self.decoded_tokens += len(tokens)
        return "".join(chr(t) for t in tokens)


def _ref_full_prefix_span(
    tokenizer: Tokenizer,
    content: str,
    tokens: list[int],
    start_token: int,
    end_token: int,
) -> dict[str, int] | None:
    """The pre-optimization implementation: full ``decode(tokens[:start_token])``."""
    window = tokenizer.decode(tokens[start_token:end_token])
    start = len(tokenizer.decode(tokens[:start_token]))
    end = start + len(window)
    if content[start:end] != window:
        found = content.find(
            window, max(0, start - 32), min(len(content), end + 32 + len(window))
        )
        if found < 0:
            return None
        start, end = found, found + len(window)
    return _source_span(content, start, end)


def _windows(n_tokens: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    step = chunk_size - overlap
    return [
        (start, min(start + chunk_size, n_tokens))
        for start in range(0, n_tokens, step)
    ]


@pytest.mark.offline
@pytest.mark.parametrize(
    "content",
    [
        " ".join(f"word{i:03d}" for i in range(400)),
        "Alpha. BETA gamma? Delta! " * 120,
        "Repeated phrase. " * 200,  # heavily repeated — exact offsets must not drift
    ],
)
def test_delta_decode_matches_full_prefix_reference(content: str) -> None:
    for impl in (_CharTokenizer(), _MultiTokenTokenizer()):
        tok = Tokenizer(model_name="t", tokenizer=impl)
        tokens = tok.encode(content)
        chunk_size, overlap = 60, 12
        anchor = (0, 0)
        for start_token, end_token in _windows(len(tokens), chunk_size, overlap):
            got, anchor = _token_window_source_span(
                tok, content, tokens, start_token, end_token, anchor=anchor
            )
            ref = _ref_full_prefix_span(tok, content, tokens, start_token, end_token)
            assert got == ref


@pytest.mark.offline
def test_delta_decode_matches_full_prefix_with_tiktoken() -> None:
    pytest.importorskip("tiktoken")
    from lightrag.utils import TiktokenTokenizer

    tok = TiktokenTokenizer()
    content = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    ) * 40
    tokens = tok.encode(content)
    anchor = (0, 0)
    for start_token, end_token in _windows(len(tokens), 24, 6):
        got, anchor = _token_window_source_span(
            tok, content, tokens, start_token, end_token, anchor=anchor
        )
        ref = _ref_full_prefix_span(tok, content, tokens, start_token, end_token)
        assert got == ref


@pytest.mark.offline
def test_fixed_token_spans_are_exact_on_long_doc() -> None:
    content = " ".join(f"token{i:04d}" for i in range(1500))
    tok = Tokenizer(model_name="char", tokenizer=_CharTokenizer())

    chunks = chunking_by_fixed_token(
        tok,
        content,
        chunk_token_size=120,
        chunk_overlap_token_size=20,
        _emit_source_span=True,
    )

    assert len(chunks) > 5
    for chunk in chunks:
        span = chunk["_source_span"]
        assert content[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_decode_budget_is_linear_not_quadratic() -> None:
    # char tokenizer => len(tokens) == len(content). With the old full-prefix
    # decode the helper alone would decode ~sum(starts) == N^2/(2*step) tokens;
    # the delta path decodes ~one step per window plus each window once. Assert
    # the total stays under a small linear multiple of N, a bound the quadratic
    # implementation cannot meet.
    content = "x" * 8000
    counting = _CountingTokenizer()
    tok = Tokenizer(model_name="counting", tokenizer=counting)
    n = len(tok.encode(content))  # 8000

    chunking_by_fixed_token(
        tok,
        content,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        _emit_source_span=True,
    )

    # Empirically ~3.2*N for the delta path; the old prefix decode is ~24*N here.
    assert counting.decoded_tokens <= 6 * n
