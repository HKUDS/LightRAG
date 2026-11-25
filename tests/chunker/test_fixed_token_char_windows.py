"""Fixed-token (F) windows are character slices of the source.

F cuts its windows with ``Tokenizer.split_by_token_limit``, so a window ends
at a character boundary whatever the tokenizer does with bytes: a chunk's
content is the source text of its ``_source_span`` exactly, never a decoded
token window with U+FFFD where a multi-byte character was split, and never a
lossy decoder's rewrite of the source. ``tokens`` counts that content.
"""

from __future__ import annotations

import random
import re

import pytest

from lightrag.chunker import chunking_by_fixed_token
from lightrag.utils import TiktokenTokenizer, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _MultiTokenTokenizer(TokenizerInterface):
    """Uppercase letters are 2 tokens, ``.?!`` 3, everything else 1."""

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        for ch in content:
            width = 2 if ch.isupper() else 3 if ch in ".?!" else 1
            tokens.extend([ord(ch)] * width)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        raise AssertionError("F must not decode token windows")


class _ByteTokenizer(TokenizerInterface):
    """One token per UTF-8 byte; a split character decodes to U+FFFD."""

    errors = "replace"

    def encode(self, content: str) -> list[int]:
        return list(content.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors=self.errors)


class _DropInvalidTokenizer(_ByteTokenizer):
    """Byte tokens; the decoder silently drops incomplete characters."""

    errors = "ignore"


class _LossyMetaspaceTokenizer(TokenizerInterface):
    """Word pieces whose decoder drops a leading space and rewrites ``□``."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.pieces: list[str] = []

    def encode(self, content: str) -> list[int]:
        tokens = []
        for piece in re.findall(r" ?\S+|\s", content):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.pieces)
                self.pieces.append(piece)
            tokens.append(self.vocab[piece])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        decoded = "".join(self.pieces[token] for token in tokens)
        return decoded.removeprefix(" ").replace("□", "[UNK]")


def _tokenizer(kind: str) -> Tokenizer:
    if kind == "tiktoken":
        return TiktokenTokenizer()
    impl = {
        "char": _CharTokenizer,
        "multi-token": _MultiTokenTokenizer,
        "bytes-replace": _ByteTokenizer,
        "bytes-drop": _DropInvalidTokenizer,
        "lossy-metaspace": _LossyMetaspaceTokenizer,
    }[kind]()
    return Tokenizer(kind, impl)


# Multi-byte characters, ZWJ sequences, flags, a literal U+FFFD of the
# source's own, repeated runs and whitespace (stripped window edges).
_ALPHABET = [
    "a", "a", "b", "A", ".", " ", " ", "\n", "x", "😀", "中", "é", "�",
    "the ", "👨‍👩", "中文", "🇨🇳", "→", "□",
]  # fmt: skip


def _random_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.4:
            parts.append(rng.choice(_ALPHABET) * rng.randint(1, 30))
        else:
            parts.append("".join(rng.choice(_ALPHABET) for _ in range(20)))
    return "".join(parts)


@pytest.mark.parametrize(
    "kind",
    [
        "char",
        "multi-token",
        "bytes-replace",
        "bytes-drop",
        "lossy-metaspace",
        "tiktoken",
    ],
)
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_every_window_is_a_character_slice_of_the_source(
    kind: str, split_by_character: str | None
) -> None:
    """Seeded audit against the source itself: every chunk is the exact text
    at its span, counts its own tokens, and together the chunks lose no
    non-whitespace character."""
    tok = _tokenizer(kind)
    rng = random.Random(4623)
    mismatches = []
    for case in range(60):
        segments = [_random_text(rng) for _ in range(2 if split_by_character else 1)]
        text = "|".join(segments)
        # 4 tokens is the widest character under a byte tokenizer.
        size = rng.randint(4, 24)
        overlap = rng.randint(0, size - 1)
        chunks = chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=overlap,
            split_by_character=split_by_character,
            _emit_source_span=True,
        )
        covered = set()
        for k, chunk in enumerate(chunks):
            body, span = chunk["content"], chunk.get("_source_span")
            where = (case, k, text, size, overlap)
            if not body:
                if span is not None or chunk["tokens"] != 0:
                    mismatches.append(("empty", where, chunk))
                continue
            if span is None or text[span["start"] : span["end"]] != body:
                mismatches.append(("slice", where, chunk))
                continue
            if chunk["tokens"] != len(tok.encode(body)):
                mismatches.append(("tokens", where, chunk))
            # A stripped BPE word may take a token more on its own; the other
            # tokenizers never grow when whitespace is stripped.
            if kind != "tiktoken" and chunk["tokens"] > size:
                mismatches.append(("size", where, chunk))
            covered.update(range(span["start"], span["end"]))
        lost = [
            i
            for i, ch in enumerate(text)
            if not ch.isspace() and ch != "|" and i not in covered
        ]
        if lost:
            mismatches.append(("lost", (case, text, size, overlap), lost[:5]))
    assert mismatches == []
