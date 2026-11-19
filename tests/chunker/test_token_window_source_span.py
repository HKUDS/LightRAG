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

import re

import pytest

from lightrag.chunker import chunking_by_fixed_token
from lightrag.chunker.token_size import (
    _recover_boundary_window,
    _source_span,
    _token_window_source_span,
)
from lightrag.sidecar.backfill import _normalize_text
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
    # Mirrors the chunker's stop rule: no window after one reaches the end.
    step = chunk_size - overlap
    windows = []
    for start in range(0, n_tokens, step):
        windows.append((start, min(start + chunk_size, n_tokens)))
        if start + chunk_size >= n_tokens:
            break
    return windows


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


def _source_span_matches_chunk_body(source_text: str, body: str) -> bool:
    if source_text == body:
        return True
    if _normalize_text(source_text) == _normalize_text(body):
        return True
    cleaned = body.strip("\ufffd")
    if cleaned == body:
        return False
    if source_text == cleaned:
        return True
    return _normalize_text(source_text) == _normalize_text(cleaned)


@pytest.mark.offline
def test_emoji_heavy_fixed_token_windows_keep_source_span_coverage() -> None:
    pytest.importorskip("tiktoken")
    from lightrag.utils import TiktokenTokenizer

    tok = TiktokenTokenizer()
    para = (
        "Release notes: 🎉 new chunker ✅ tested 🚀 shipped. "
        "Emoji families 👨👩👧👦 👩🏽🚀 and flags 🇨🇳 🇯🇵 🇩🇪 appear in user text. "
    )
    text = para * 120

    chunks = chunking_by_fixed_token(
        tok,
        text,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        _emit_source_span=True,
    )

    covered = bytearray(len(text))
    for chunk in chunks:
        span = chunk.get("_source_span")
        assert span is not None, chunk["chunk_order_index"]
        slice_text = text[span["start"] : span["end"]]
        body = chunk["content"]
        assert _source_span_matches_chunk_body(slice_text, body)
        for j in range(span["start"], span["end"]):
            covered[j] = 1

    uncovered = len(text) - sum(covered)
    assert uncovered / len(text) < 0.01


class _ByteTokenizer(TokenizerInterface):
    """One token per UTF-8 byte; decode replaces partial characters with U+FFFD."""

    def encode(self, content: str) -> list[int]:
        return list(content.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


def _byte_window_char_span(text: str, b0: int, b1: int) -> tuple[int, int] | None:
    """Char span of the characters lying wholly inside bytes ``[b0, b1)``."""
    inside = []
    pos = 0
    for i, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        if pos >= b0 and pos + width <= b1:
            inside.append(i)
        pos += width
    return (inside[0], inside[-1] + 1) if inside else None


@pytest.mark.offline
@pytest.mark.parametrize(
    ("text", "size", "overlap"),
    [
        ("😀" * 64 + " and then a unique ascii tail sentence here.", 9, 0),
        ("x😀y" * 40 + " tail end", 9, 0),
        ("é中😀" * 50, 7, 2),
    ],
)
def test_repetitive_boundary_windows_do_not_drift_the_anchor(
    text: str, size: int, overlap: int, tmp_path
) -> None:
    """A stripped probe that matches several places in repetitive text must
    not become the anchor: the leftmost pick dragged every later prediction
    backwards until clean windows lost their span and backfill failed the
    document. Every span must be the window's own position."""
    import json

    from lightrag.sidecar.backfill import backfill_chunk_sidecars

    chunks = chunking_by_fixed_token(
        Tokenizer("bytes", _ByteTokenizer()),
        text,
        chunk_token_size=size,
        chunk_overlap_token_size=overlap,
        _emit_source_span=True,
    )

    step = size - overlap
    for k, chunk in enumerate(chunks):
        expected = _byte_window_char_span(text, k * step, k * step + size)
        assert expected is not None
        assert chunk.get("_source_span") == _source_span(text, *expected), k

    blocks_path = tmp_path / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})
        + "\n"
        + json.dumps({"type": "content", "blockid": "b1", "content": text})
        + "\n",
        encoding="utf-8",
    )
    backfill_chunk_sidecars(chunks, str(blocks_path))
    assert all(chunk.get("sidecar") for chunk in chunks)


@pytest.mark.offline
def test_leading_trimmed_window_does_not_shift_later_spans(tmp_path) -> None:
    """A window that began inside a character is located after it, so its
    match is not a verified (token, char) anchor. Anchoring on it shifted the
    next prediction by the leftover bytes; over repeated text the shifted
    exact-match check still passed and a clean chunk was attributed to the
    wrong blocks."""
    import json

    from lightrag.sidecar.backfill import backfill_chunk_sidecars

    blocks = [("b0", "abcde😀qr😀a")] + [(f"b{i}", "a" * 12) for i in range(1, 7)]
    text = "\n\n".join(content for _, content in blocks)
    chunks = chunking_by_fixed_token(
        Tokenizer("bytes", _ByteTokenizer()),
        text,
        chunk_token_size=6,
        chunk_overlap_token_size=0,
        _emit_source_span=True,
    )
    for k, chunk in enumerate(chunks):
        expected = _byte_window_char_span(text, k * 6, k * 6 + 6)
        assert expected is not None
        assert chunk.get("_source_span") == _source_span(text, *expected), k

    blocks_path = tmp_path / "doc.blocks.jsonl"
    rows = [json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})]
    rows += [
        json.dumps({"type": "content", "blockid": blockid, "content": content})
        for blockid, content in blocks
    ]
    blocks_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    backfill_chunk_sidecars(chunks, str(blocks_path))
    refs = [[r["id"] for r in chunk["sidecar"]["refs"]] for chunk in chunks]
    # Shifted anchoring attributed chunk #7 to ["b1", "b2"] and every later
    # chunk one block too early.
    assert refs == [
        ["b0"], ["b0"], ["b0"], ["b1"], ["b1"], ["b2"], ["b2"], ["b2", "b3"],
        ["b3"], ["b3"], ["b4"], ["b4"], ["b5"], ["b5"], ["b5", "b6"], ["b6"], ["b6"],
    ]  # fmt: skip


_BYTE_AUDIT_ALPHABET = ["a", "a", "a", "b", " ", "😀", "中", "é", "\n", "x"]
# Multi-character pieces so BPE merges produce tokens that end mid-character.
_BPE_AUDIT_ALPHABET = _BYTE_AUDIT_ALPHABET + ["the ", "👨\u200d👩", "中文", "🇨🇳", "→"]


def _random_boundary_text(rng, alphabet: list[str]) -> str:
    parts = []
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.4:  # a repeated run, where shifted matches hide
            parts.append(rng.choice(alphabet) * rng.randint(1, 30))
        else:
            parts.append(
                "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 20)))
            )
    return "".join(parts)


@pytest.mark.offline
def test_every_span_is_the_window_position_under_byte_tokenizer() -> None:
    """Seeded differential audit against ground truth: with one token per
    byte, the characters wholly inside each window's bytes are known exactly,
    so every emitted span must equal that position -- not merely text that
    happens to match somewhere else in repetitive input."""
    import random

    rng = random.Random(7)
    tok = Tokenizer("bytes", _ByteTokenizer())
    for case in range(400):
        text = _random_boundary_text(rng, _BYTE_AUDIT_ALPHABET)
        if not text.strip():
            continue
        size = rng.randint(2, 24)
        overlap = rng.randint(0, size - 1)
        chunks = chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=overlap,
            _emit_source_span=True,
        )
        step = size - overlap
        for k, chunk in enumerate(chunks):
            expected = _byte_window_char_span(text, k * step, k * step + size)
            want = _source_span(text, *expected) if expected else None
            assert chunk.get("_source_span") == want, (case, k, text, size, overlap)


@pytest.mark.offline
def test_every_span_is_the_window_position_under_tiktoken() -> None:
    """The same audit with a real byte-level BPE tokenizer; ground truth comes
    from each token's byte length."""
    pytest.importorskip("tiktoken")
    import random

    from lightrag.utils import TiktokenTokenizer

    tok = TiktokenTokenizer()
    encoding = tok.tokenizer
    rng = random.Random(11)
    for case in range(150):
        text = _random_boundary_text(rng, _BPE_AUDIT_ALPHABET)
        if not text.strip():
            continue
        tokens = encoding.encode(text)
        byte_offsets = [0]
        for token in tokens:
            byte_offsets.append(
                byte_offsets[-1] + len(encoding.decode_single_token_bytes(token))
            )
        size = rng.randint(2, 24)
        overlap = rng.randint(0, size - 1)
        chunks = chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=overlap,
            _emit_source_span=True,
        )
        step = size - overlap
        for k, chunk in enumerate(chunks):
            b0 = byte_offsets[k * step]
            b1 = byte_offsets[min(len(tokens), k * step + size)]
            expected = _byte_window_char_span(text, b0, b1)
            want = _source_span(text, *expected) if expected else None
            assert chunk.get("_source_span") == want, (case, k, text, size, overlap)


class _PerByteReplacementTokenizer(_ByteTokenizer):
    """Byte tokens; decode emits one U+FFFD per undecodable byte.

    Python and Rust emit one U+FFFD for the whole head of a broken character;
    other decoders emit one per byte. Recovery must not depend on which.
    """

    def decode(self, tokens: list[int]) -> str:
        data = bytes(tokens)
        out: list[str] = []
        i = 0
        while i < len(data):
            for width in (1, 2, 3, 4):
                try:
                    out.append(data[i : i + width].decode("utf-8"))
                except UnicodeDecodeError:
                    continue
                i += width
                break
            else:
                out.append("\ufffd")
                i += 1
        return "".join(out)


@pytest.mark.offline
def test_every_span_is_the_window_position_when_decoder_replaces_per_byte() -> None:
    """Counting the U+FFFD a decoder emits for a broken head put the start one
    or two characters late with a per-byte decoder; over a repeated run the
    late position still matched. The start is counted from whole characters."""
    import random

    rng = random.Random(5)
    tok = Tokenizer("per-byte", _PerByteReplacementTokenizer())
    for case in range(400):
        text = _random_boundary_text(rng, _BYTE_AUDIT_ALPHABET)
        if not text.strip():
            continue
        size = rng.randint(2, 24)
        overlap = rng.randint(0, size - 1)
        chunks = chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=overlap,
            _emit_source_span=True,
        )
        step = size - overlap
        for k, chunk in enumerate(chunks):
            expected = _byte_window_char_span(text, k * step, k * step + size)
            want = _source_span(text, *expected) if expected else None
            assert chunk.get("_source_span") == want, (case, k, text, size, overlap)


@pytest.mark.offline
def test_source_replacement_near_boundary_window_gets_no_span(tmp_path) -> None:
    """When the source has U+FFFD of its own, a window's edge U+FFFD cannot be
    told apart. Stripping them all left "x", which was then found in another
    block; a document with U+FFFD gets no recovery, so the window has no span."""
    import json

    from lightrag.sidecar.backfill import backfill_chunk_sidecars

    blocks = [("b0", "x"), ("b1", "\ufffd" * 4 + "x😀")]
    text = "\n\n".join(content for _, content in blocks)
    chunks = chunking_by_fixed_token(
        Tokenizer("bytes", _ByteTokenizer()),
        text,
        chunk_token_size=14,
        chunk_overlap_token_size=11,
        _emit_source_span=True,
    )
    # Window 1 is bytes [3, 17): all of b1's U+FFFD, "x" and the emoji's head.
    assert chunks[1]["content"] == "\ufffd" * 4 + "x\ufffd"
    assert "_source_span" not in chunks[1]

    blocks_path = tmp_path / "doc.blocks.jsonl"
    rows = [json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})]
    rows += [
        json.dumps({"type": "content", "blockid": blockid, "content": content})
        for blockid, content in blocks
    ]
    blocks_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    backfill_chunk_sidecars(chunks, str(blocks_path))
    assert "sidecar" not in chunks[1]
    for chunk in chunks:
        if "sidecar" in chunk and chunk["chunk_order_index"] > 0:
            assert [r["id"] for r in chunk["sidecar"]["refs"]] == ["b1"]


@pytest.mark.offline
@pytest.mark.parametrize("head", ["\ufffd", "\ufffd\ufffd\ufffd"])
def test_recover_boundary_window_counts_characters_not_replacements(head: str) -> None:
    # "ab" then a split emoji, then "cd": the window holds the emoji's tail.
    content = "ab😀cd😀cd"
    assert _recover_boundary_window(
        content, "\ufffd\ufffdcd\ufffd", "ab" + head, 0
    ) == (
        3,
        5,
    )


@pytest.mark.offline
def test_recover_boundary_window_declines_instead_of_searching() -> None:
    content = "ab😀cd😀cd"
    # The text is not at the computed offset; it does occur two characters on.
    assert _recover_boundary_window(content, "\ufffdd😀c\ufffd", "ab\ufffd", 0) is None
    # A window with no edge U+FFFD, or nothing but U+FFFD, is not this case.
    assert _recover_boundary_window(content, "cd", "ab😀", 0) is None
    assert _recover_boundary_window(content, "\ufffd\ufffd", "ab", 0) is None


@pytest.mark.offline
def test_source_replacement_far_from_window_still_disables_recovery(tmp_path) -> None:
    """The exact-match probe can anchor on a split character that matched
    source U+FFFD. Recovery then computed from that anchor, and far from the
    U+FFFD a repeated run verified the shifted offset: a chunk holding only
    b1's bytes got a b0 span. Any U+FFFD in the document disables recovery."""
    import json

    from lightrag.sidecar.backfill import backfill_chunk_sidecars

    blocks = [
        ("b0", "\ufffd" * 7 + "é" * 12 + "😀" * 25 + "é" * 6),
        ("b1", "é" * 6),
    ]
    text = "\n\n".join(content for _, content in blocks)
    chunks = chunking_by_fixed_token(
        Tokenizer("bytes", _ByteTokenizer()),
        text,
        chunk_token_size=15,
        chunk_overlap_token_size=5,
        _emit_source_span=True,
    )
    # Chunk 16 is bytes [160, 171), the last window: the tail of one "é"
    # and five whole ones, all in b1 and more than 32 characters past the
    # last source U+FFFD.
    assert len(chunks) == 17
    assert chunks[16]["content"] == "\ufffdééééé"
    assert "_source_span" not in chunks[16]

    blocks_path = tmp_path / "doc.blocks.jsonl"
    rows = [json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})]
    rows += [
        json.dumps({"type": "content", "blockid": blockid, "content": content})
        for blockid, content in blocks
    ]
    blocks_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    backfill_chunk_sidecars(chunks, str(blocks_path))
    assert "sidecar" not in chunks[16]


@pytest.mark.offline
def test_recovery_never_runs_on_content_with_replacement_characters(
    monkeypatch,
) -> None:
    """Pins the rule itself rather than one input: content that holds U+FFFD
    never reaches the recovery helper, on either chunking path; content
    without it does, including a clean segment beside a garbled one."""
    import random

    from lightrag.chunker import token_size

    calls: list[str] = []
    real = token_size._recover_boundary_window

    def recording(content, window, prefix, prefix_start):
        calls.append(content)
        return real(content, window, prefix, prefix_start)

    monkeypatch.setattr(token_size, "_recover_boundary_window", recording)
    tok = Tokenizer("bytes", _ByteTokenizer())
    rng = random.Random(77)
    for _ in range(200):
        text = _random_boundary_text(rng, _BYTE_AUDIT_ALPHABET + ["\ufffd"])
        if "\ufffd" not in text or not text.strip():
            continue
        size = rng.randint(2, 24)
        chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=rng.randint(0, size - 1),
            _emit_source_span=True,
        )
    assert calls == []

    clean, garbled = "😀" * 12, "\ufffd" + "😀" * 12
    chunking_by_fixed_token(
        tok,
        clean + "|" + garbled,
        chunk_token_size=6,
        chunk_overlap_token_size=0,
        split_by_character="|",
        _emit_source_span=True,
    )
    assert calls and set(calls) == {clean}


class _DropInvalidTokenizer(_ByteTokenizer):
    """Byte tokenizer whose decoder silently drops incomplete UTF-8 characters."""

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


@pytest.mark.offline
@pytest.mark.parametrize("kind", ["literal-tiktoken", "literal-bytes", "drop-bytes"])
def test_ambiguous_boundary_windows_keep_their_own_position(kind: str) -> None:
    """Matching repeated text must never establish a false positional anchor."""
    from lightrag.utils import TiktokenTokenizer

    if kind == "literal-tiktoken":
        tok = TiktokenTokenizer()
        text, size, overlap = "🇨🇳🇨🇳��", 2, 1
    elif kind == "literal-bytes":
        tok = Tokenizer("bytes", _ByteTokenizer())
        text, size, overlap = "🇨🇳🇨🇳��", 4, 0
    else:
        tok = Tokenizer("bytes", _DropInvalidTokenizer())
        text, size, overlap = "😀" * 6 + "x", 6, 0

    tokens = tok.encode(text)
    widths = (
        [len(tok.tokenizer.decode_single_token_bytes(t)) for t in tokens]
        if kind == "literal-tiktoken"
        else [1] * len(tokens)
    )
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    chunks = chunking_by_fixed_token(
        tok,
        text,
        chunk_token_size=size,
        chunk_overlap_token_size=overlap,
        _emit_source_span=True,
    )
    mismatches = []
    located = []
    for k, chunk in enumerate(chunks):
        start = k * (size - overlap)
        expected = _byte_window_char_span(
            text, offsets[start], offsets[min(start + size, len(tokens))]
        )
        want = _source_span(text, *expected) if expected else None
        got = chunk.get("_source_span")
        if got is not None:
            located.append(k)
        if (got is not None or kind == "drop-bytes") and got != want:
            mismatches.append((k, got, want))
    assert mismatches == [], "source spans must follow token byte positions"
    assert located, "already-valid windows must retain provenance"
    if kind == "drop-bytes":
        assert chunks[-1].get("_source_span") == {"start": 6, "end": 7}
    else:
        clean_index = 4 if kind == "literal-tiktoken" else 2
        assert chunks[clean_index].get("_source_span") == {"start": 2, "end": 3}


@pytest.mark.offline
@pytest.mark.parametrize("kind", ["replace", "per-byte", "drop", "tiktoken"])
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_literal_replacement_and_dropping_decoder_span_audit(
    kind: str, split_by_character: str | None
) -> None:
    """Audit offsets, including segment-relative anchors, against token bytes."""
    import random

    from lightrag.utils import TiktokenTokenizer

    if kind == "tiktoken":
        tok = TiktokenTokenizer()
    else:
        impl = {
            "replace": _ByteTokenizer,
            "per-byte": _PerByteReplacementTokenizer,
            "drop": _DropInvalidTokenizer,
        }[kind]()
        tok = Tokenizer(kind, impl)
    rng = random.Random(4443)
    for _ in range(100):
        segments = [
            _random_boundary_text(rng, _BPE_AUDIT_ALPHABET + ["\ufffd", "\ufffd\ufffd"])
            for _ in range(2 if split_by_character else 1)
        ]
        text = "|".join(segments)
        size = rng.randint(2, 24)
        overlap = rng.randint(0, size - 1)
        chunks = chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=overlap,
            split_by_character=split_by_character,
            _emit_source_span=True,
        )
        expected = []
        segment_start = 0
        for segment in segments:
            tokens = tok.encode(segment)
            offsets = [0]
            for token in tokens:
                width = (
                    len(tok.tokenizer.decode_single_token_bytes(token))
                    if kind == "tiktoken"
                    else 1
                )
                offsets.append(offsets[-1] + width)
            windows = (
                [(0, len(tokens))]
                if split_by_character and len(tokens) <= size
                else _windows(len(tokens), size, overlap)
            )
            for start, end in windows:
                span = _byte_window_char_span(segment, offsets[start], offsets[end])
                expected.append(
                    _source_span(text, segment_start + span[0], segment_start + span[1])
                    if span
                    else None
                )
            segment_start += len(segment) + 1
        assert len(chunks) == len(expected)
        mismatches = [
            (k, chunk.get("_source_span"), want)
            for k, (chunk, want) in enumerate(zip(chunks, expected))
            if chunk.get("_source_span") is not None and chunk["_source_span"] != want
        ]
        assert mismatches == [], (text, size, overlap)


@pytest.mark.offline
@pytest.mark.parametrize("kind", ["replace", "drop", "tiktoken"])
async def test_boundary_spans_survive_hard_split_backfill_and_storage(
    kind, tmp_path
) -> None:
    """Use the chunker's actual dictionaries through the downstream copy path."""
    import json

    from lightrag.sidecar.backfill import backfill_chunk_sidecars
    from lightrag.utils import (
        TiktokenTokenizer,
        enforce_chunk_token_limit_before_embedding,
    )
    from lightrag.utils_pipeline import (
        build_chunks_dict_from_chunking_result,
        load_lightrag_document_content,
    )

    tok = (
        TiktokenTokenizer()
        if kind == "tiktoken"
        else Tokenizer(
            kind, _DropInvalidTokenizer() if kind == "drop" else _ByteTokenizer()
        )
    )
    blocks = [("b0", "🇨🇳🇨🇳��"), ("b1", "😀" * 6 + "x")]
    text = "\n\n".join(body for _, body in blocks)
    path = tmp_path / "doc.blocks.jsonl"
    rows = [{"type": "meta", "format": "lightrag", "version": "1.0"}]
    rows += [
        {"type": "content", "blockid": blockid, "content": body}
        for blockid, body in blocks
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    merged, blocks_path = await load_lightrag_document_content(tmp_path.as_uri())
    assert merged == text
    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
        _emit_source_span=True,
    )
    children = enforce_chunk_token_limit_before_embedding(
        chunking_result=chunks,
        tokenizer=tok,
        max_tokens=4,
        source_content=text,
    )
    assert len(children) > len(chunks)
    backfill_chunk_sidecars(children, blocks_path)
    located = []
    for child in children:
        if "sidecar" not in child:
            continue
        span = child["_source_span"]
        expected_refs = []
        cursor = 0
        for blockid, body in blocks:
            if cursor < span["end"] and span["start"] < cursor + len(body):
                expected_refs.append(blockid)
            cursor += len(body) + 2
        assert [ref["id"] for ref in child["sidecar"]["refs"]] == expected_refs
        located.append(child)
    assert located
    stored = build_chunks_dict_from_chunking_result(
        children, doc_id="doc-boundaries", file_path="doc.txt"
    )
    assert stored
    assert all("_source_span" not in row for row in stored.values())
    assert all("_source_span_unavailable" not in row for row in stored.values())
    assert any(row.get("sidecar") for row in stored.values())


class _MetaspaceTokenizer(TokenizerInterface):
    """Count decodes that strip one leading space, as in Metaspace decoding."""

    def __init__(self, *, lossy: bool = False) -> None:
        self.vocab: dict[str, int] = {}
        self.pieces: list[str] = []
        self.decoded_tokens = 0
        self.lossy = lossy

    def encode(self, content: str) -> list[int]:
        tokens = []
        for piece in re.findall(r" ?\S+|\s", content):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.pieces)
                self.pieces.append(piece)
            tokens.append(self.vocab[piece])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        self.decoded_tokens += len(tokens)
        decoded = "".join(self.pieces[token] for token in tokens).removeprefix(" ")
        return decoded.replace("□", "[UNK]") if self.lossy else decoded


@pytest.mark.offline
@pytest.mark.parametrize("lossy", [False, True])
def test_metaspace_windows_keep_linear_decode_budget(lossy: bool) -> None:
    words = ["alpha", "beta", "gamma", "delta"] * 1024
    if lossy:
        words[70] = "□"
    content = " ".join(words)
    tokenizer = _MetaspaceTokenizer(lossy=lossy)
    chunks = chunking_by_fixed_token(
        Tokenizer("metaspace", tokenizer),
        content,
        chunk_token_size=32,
        chunk_overlap_token_size=8,
        _emit_source_span=True,
    )
    assert tokenizer.decoded_tokens <= 7 * len(words)

    offsets = [match.start() for match in re.finditer(r"\S+", content)]
    windows = _windows(len(words), 32, 8)
    assert len(chunks) == len(windows)
    for chunk, (start, end) in zip(chunks, windows):
        assert chunk["content"] == " ".join(words[start:end]).replace("□", "[UNK]")
        if lossy and end > 70:
            assert "_source_span" not in chunk
        else:
            assert chunk["_source_span"] == {
                "start": offsets[start],
                "end": offsets[end - 1] + len(words[end - 1]),
            }


@pytest.mark.offline
@pytest.mark.parametrize(
    ("content", "split_by_character", "expected"),
    [
        pytest.param(
            "intro. alpha beta gamma delta",
            ".",
            [("intro", 0, 5), ("alpha beta", 7, 17), ("gamma delta", 18, 29)],
            id="split-segment-starts-with-space",
        ),
        pytest.param(
            "intro.\n alpha beta gamma delta",
            "\n",
            [("intro.", 0, 6), ("alpha beta", 8, 18), ("gamma delta", 19, 30)],
            id="split-on-newline",
        ),
        pytest.param(
            " alpha beta gamma delta",
            None,
            [("alpha beta", 1, 11), ("gamma delta", 12, 23)],
            id="document-starts-with-space",
        ),
        pytest.param(
            "  alpha beta gamma delta",
            None,
            [("alpha", 2, 7), ("beta gamma", 8, 18), ("delta", 19, 24)],
            id="only-one-of-two-spaces-dropped",
        ),
        pytest.param(
            "     b   word b   word",
            None,
            [("b", 5, 6), ("word", 9, 13), ("b", 14, 15), ("word", 18, 22)],
            id="joint-also-starts-with-whitespace",
        ),
    ],
)
def test_metaspace_origin_skips_dropped_leading_whitespace(
    tmp_path, content: str, split_by_character: str | None, expected
) -> None:
    """A Metaspace decoder drops the space that starts a segment, so token 0
    does not decode from offset 0. The origin must move past exactly the
    dropped whitespace: left at 0, every window of the segment lost its span
    and sidecar backfill failed the whole document; moved by a prefix match
    inside a run of whitespace, it sat one character early and the anchor
    died later in the segment."""
    import json

    from lightrag.sidecar.backfill import backfill_chunk_sidecars

    chunks = chunking_by_fixed_token(
        Tokenizer("metaspace", _MetaspaceTokenizer()),
        content,
        chunk_token_size=2,
        chunk_overlap_token_size=0,
        split_by_character=split_by_character,
        _emit_source_span=True,
    )
    located = [c for c in chunks if c["content"].strip()]
    assert [
        (c["content"], c["_source_span"]["start"], c["_source_span"]["end"])
        for c in located
    ] == expected

    blocks_path = tmp_path / "doc.blocks.jsonl"
    rows = [
        json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"}),
        json.dumps({"type": "content", "blockid": "b0", "content": content}),
    ]
    blocks_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    backfill_chunk_sidecars(chunks, str(blocks_path))
    assert [chunk["sidecar"]["id"] for chunk in located] == ["b0"] * len(expected)
