"""Application-side lexical tokenizer for PostgreSQL hybrid retrieval.

Pure string functions: no database driver, no async, no I/O. The write side
(``to_tsvector('simple', ...)``) and the query side (``to_tsquery('simple',
...)``) MUST both go through this module so their token sequences agree.

RULES

* Normalization: NFKC, then lowercase, before tokenizing. Both sides
  normalize identically. PG's ``simple`` dictionary lowercases again, which
  is idempotent on already-lowercased text, so lexemes agree under any
  database locale.
* CJK runs (the ranges in ``_CJK_RANGES``) become overlapping bigrams; a
  one-character run becomes a unigram. Bigrams overlap by one character
  (standard n-gram indexing); no other token expansion is emitted.
* Every other alphanumeric run is ONE word token, truncated to
  ``MAX_TOKEN_LENGTH`` characters (prefix kept). Punctuation, symbols,
  whitespace, combining marks and the tsquery reserved characters
  ``&``, ``|``, ``!``, ``(``, ``)``, ``<->``, ``'``, ``\\``, ``:`` and ``*``
  are all separators. Scripts outside the CJK ranges (Cyrillic, accented
  Latin, ...) are word tokens, not bigrammed.
* Output is bounded: at most ``MAX_TOKENS`` tokens per call; scanning stops
  at the cap. The query string is therefore at most
  ``MAX_TOKENS * MAX_TOKEN_LENGTH + 3 * (MAX_TOKENS - 1)`` characters.
* ``tokenize_for_index`` joins tokens with single spaces. Each emitted
  token is one atomic lexeme under PG's default parser (asciiword /
  numword / word), so ``to_tsvector('simple', tokenized)`` stores exactly
  the emitted tokens.
* ``tokenize_for_query`` joins the SAME tokens with `` & `` (AND). Tokens
  contain only alphanumeric characters by construction, so the result can
  never carry a tsquery operator character: no syntax error, no injection.
  Phrase proximity is a follow-up, not part of this contract.
* Empty or separator-only input yields ``""`` on both sides. An empty
  query string means "no lexical content": the caller must skip the
  lexical leg, not feed ``""`` to ``to_tsquery``.
* Changing any rule (ranges, caps, normalization) changes stored lexemes:
  it requires re-tokenizing existing rows, not just new writes.
"""

from __future__ import annotations

import unicodedata
from typing import Final

#: Maximum tokens emitted per call, on BOTH sides (write/query consistency).
MAX_TOKENS: Final = 4096

#: Maximum characters per word token; longer runs keep their prefix.
MAX_TOKEN_LENGTH: Final = 128

# CJK ideographs, kana and Hangul: runs of these are bigrammed. Sorted,
# disjoint. U+3005..U+3007 are the letter-like iteration marks (々〇〆) that
# sit inside the CJK punctuation block. Astral extension planes (U+20000+)
# are out of scope for bigramming: those characters are category Lo, so
# ch.isalnum() holds and they are ordinary word-token material instead.
_CJK_RANGES: Final = (
    (0x3005, 0x3007),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def tokenize(text: str) -> tuple[str, ...]:
    """Return the one token sequence both wrappers share (module RULES).

    Write/query token consistency holds by construction: both public
    wrappers consume this function and nothing else.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    word: list[str] = []
    cjk_prev = ""
    cjk_len = 0

    def flush_word() -> None:
        nonlocal word
        if word:
            tokens.append("".join(word))
            word = []

    def flush_cjk() -> None:
        nonlocal cjk_prev, cjk_len
        if cjk_len == 1:
            tokens.append(cjk_prev)
        cjk_prev = ""
        cjk_len = 0

    for ch in normalized:
        if len(tokens) >= MAX_TOKENS:
            break
        if _is_cjk(ch):
            flush_word()
            if cjk_len:
                tokens.append(cjk_prev + ch)
            cjk_prev = ch
            cjk_len += 1
        elif ch.isalnum():
            flush_cjk()
            if len(word) < MAX_TOKEN_LENGTH:
                word.append(ch)
        else:
            flush_word()
            flush_cjk()
    else:
        flush_word()
        flush_cjk()
    return tuple(tokens[:MAX_TOKENS])


def tokenize_for_index(text: str) -> str:
    """Space-joined tokens, ready for ``to_tsvector('simple', tokenized)``."""
    return " ".join(tokenize(text))


def tokenize_for_query(text: str) -> str:
    """AND-joined tsquery expression; ``""`` means skip the lexical leg."""
    return " & ".join(tokenize(text))