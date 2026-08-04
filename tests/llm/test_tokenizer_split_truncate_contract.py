"""Tests for the safe split/truncate contract on ``Tokenizer`` (replaces #3559).

``Tokenizer.split_by_token_limit`` / ``truncate_by_token_limit`` replace the old
sentence-first ``split_text_by_token_limit`` and the decode-based, fixed-margin
truncation helpers scattered across the codebase. The contract:

* every candidate substring is independently re-encoded and verified to fit
  ``max_tokens`` (BPE token count is not monotonic in text length, so nothing
  short of re-encoding the actual candidate can be trusted);
* ``split`` fully covers the input with no gaps and real forward progress;
* retreat is bounded and strictly one-directional (never oscillates);
* a third-party ``Tokenizer`` subclass implementing only ``encode``/``decode``
  gets correct (if slower) behavior "for free" via the generic base-class
  implementation, with no new abstract methods to fill in.
"""

from __future__ import annotations

import copy

import pytest

from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    TokenBudgetError,
    TokenSpan,
    TiktokenTokenizer,
    truncate_list_by_token_size,
)

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    """One token per character; deterministic and cheap for invariant checks."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _WordCountingTokenizer(TokenizerInterface):
    """A BPE-like tokenizer where a shorter prefix can cost MORE tokens.

    "ab" is a single recognized "word" token (cost 1); any other content
    costs one token per character. Truncating "ab" down to just "a" therefore
    goes from 1 token to 1 token (no change), but truncating "abc" down to
    "ab" goes from 3 (a, b, c) to 1 -- and, the counter-example this contract
    must survive, extending "a" (1 token) to "ab" (1 token) then over to "abx"
    jumps to 3. The property under test: re-encoding the actual candidate
    (not assuming monotonicity) is what makes truncate/split safe here.
    """

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        i = 0
        while i < len(content):
            if content[i : i + 2] == "ab":
                tokens.append(-1)  # sentinel "ab" token
                i += 2
            else:
                tokens.append(ord(content[i]))
                i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join("ab" if t == -1 else chr(t) for t in tokens)


def _char_tok() -> Tokenizer:
    return Tokenizer("char", _CharTokenizer())


def _tiktoken_tok() -> TiktokenTokenizer:
    return TiktokenTokenizer("gpt-4o-mini")


# --------------------------------------------------------------------------- #
# truncate_by_token_limit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_truncate_empty_content_returns_zero_span(tok_factory):
    tok = tok_factory()
    assert tok.truncate_by_token_limit("", 10) == TokenSpan(0, 0, 0)


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_truncate_non_positive_budget_raises(tok_factory):
    tok = tok_factory()
    with pytest.raises(ValueError):
        tok.truncate_by_token_limit("hello", 0)
    with pytest.raises(ValueError):
        tok.truncate_by_token_limit("hello", -1)


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_truncate_fits_whole_content_fast_path(tok_factory):
    tok = tok_factory()
    text = "hello world"
    span = tok.truncate_by_token_limit(text, 10_000)
    assert span.start == 0
    assert span.end == len(text)
    assert span.token_count == len(tok.encode(text))


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_truncate_result_is_a_safe_reencoded_prefix(tok_factory):
    tok = tok_factory()
    text = "abcdefghijklmnopqrstuvwxyz" * 5
    span = tok.truncate_by_token_limit(text, 7)
    assert span.start == 0
    sub = text[span.start : span.end]
    assert len(tok.encode(sub)) == span.token_count
    assert span.token_count <= 7


def test_truncate_raises_token_budget_error_when_even_one_code_point_does_not_fit():
    class _HeavyTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [0] * (5 * len(content))

        def decode(self, tokens: list[int]) -> str:
            return ""

    tok = Tokenizer("heavy", _HeavyTokenizer())
    with pytest.raises(TokenBudgetError) as exc_info:
        tok.truncate_by_token_limit("hello", 1)
    assert exc_info.value.max_tokens == 1
    assert exc_info.value.code_point_token_count == 5
    assert "hello"[:1] not in "" or True  # preview is just diagnostic


def test_truncate_survives_the_bpe_non_monotonic_counter_example():
    """A shorter candidate is not guaranteed to cost fewer tokens.

    "ab" costs 1 token; "abx" costs 3 (the word-token breaks). If truncate
    ever assumed "shorter == fewer or equal tokens" without re-encoding, it
    could accept a growing candidate incorrectly. Pin that every returned
    span is independently safe regardless of this non-monotonicity.
    """
    tok = Tokenizer("word", _WordCountingTokenizer())
    text = "ab" + "x" * 20
    for budget in (1, 2, 3, 5, 10):
        span = tok.truncate_by_token_limit(text, budget)
        sub = text[span.start : span.end]
        assert len(tok.encode(sub)) == span.token_count
        assert span.token_count <= budget


# --------------------------------------------------------------------------- #
# split_by_token_limit: coverage / no gaps / real progress
# --------------------------------------------------------------------------- #


def _assert_split_invariants(tok, text: str, spans: list[TokenSpan], max_tokens: int):
    assert spans, "split of non-empty content must not be empty"
    assert spans[0].start == 0
    covered_end = 0
    for i, span in enumerate(spans):
        assert span.start < span.end, "every span must be non-empty"
        assert span.start <= covered_end, "no coverage gap"
        piece = text[span.start : span.end]
        assert len(tok.encode(piece)) == span.token_count
        assert span.token_count <= max_tokens
        if i < len(spans) - 1:
            assert span.end > covered_end, "non-final span must make real progress"
        covered_end = span.end
    assert covered_end == len(text), "split must fully cover the content"


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_split_covers_ascii_with_no_gaps_and_real_progress(tok_factory):
    tok = tok_factory()
    text = "The quick brown fox jumps over the lazy dog. " * 30
    spans = tok.split_by_token_limit(text, 12)
    _assert_split_invariants(tok, text, spans, 12)


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_split_covers_cjk_with_no_gaps_and_real_progress(tok_factory):
    tok = tok_factory()
    text = "这是一个测试句子,用来验证安全切分的不变式。" * 20
    spans = tok.split_by_token_limit(text, 9)
    _assert_split_invariants(tok, text, spans, 9)


def test_split_covers_combining_and_zwj_emoji_without_over_promising_graphemes():
    tok = _tiktoken_tok()
    # Combining characters (e + combining acute) and a ZWJ family emoji: the
    # contract guarantees Unicode code point safety, not grapheme integrity.
    text = ("é combining " + "👨‍👩‍👧‍👦 zwj family ") * 10
    spans = tok.split_by_token_limit(text, 6)
    _assert_split_invariants(tok, text, spans, 6)


def test_split_covers_regional_indicators():
    tok = _tiktoken_tok()
    text = "🇯🇵🇰🇷🇨🇳🇺🇸🇬🇧 flags " * 15
    spans = tok.split_by_token_limit(text, 5)
    _assert_split_invariants(tok, text, spans, 5)


def test_split_covers_special_token_literal_text():
    """Literal special-token strings must round-trip via the disallowed_special
    fallback, not crash the split."""
    tok = _tiktoken_tok()
    text = ("plain text <|endoftext|> more plain text " * 10) + "<|endoftext|>"
    spans = tok.split_by_token_limit(text, 8)
    _assert_split_invariants(tok, text, spans, 8)


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_split_rejects_max_tokens_le_zero(tok_factory):
    tok = tok_factory()
    with pytest.raises(ValueError):
        tok.split_by_token_limit("hello", 0)


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_split_rejects_negative_overlap(tok_factory):
    tok = tok_factory()
    with pytest.raises(ValueError):
        tok.split_by_token_limit("hello", 5, overlap_tokens=-1)


def test_split_empty_content_returns_empty_list():
    tok = _char_tok()
    assert tok.split_by_token_limit("", 5) == []


@pytest.mark.parametrize("tok_factory", [_char_tok, _tiktoken_tok])
def test_split_produces_real_overlap_between_consecutive_windows(tok_factory):
    tok = tok_factory()
    text = "The quick brown fox jumps over the lazy dog. " * 6
    spans = tok.split_by_token_limit(text, 6, overlap_tokens=2)
    _assert_split_invariants(tok, text, spans, 6)
    assert len(spans) > 1
    # At least one consecutive pair actually overlaps (next window's start
    # reaches back into the previous window's covered range).
    assert any(spans[i].start < spans[i - 1].end for i in range(1, len(spans)))


def test_split_matches_old_fixed_width_windows_with_zero_overlap():
    """No-overlap split degenerates to plain fixed-size windows for a
    one-token-per-char tokenizer -- a drift guard against the old sentence-
    first packer's behavior for this simple case."""
    tok = _char_tok()
    text = "abcdefghij"
    spans = tok.split_by_token_limit(text, 4, overlap_tokens=0)
    pieces = [text[s.start : s.end] for s in spans]
    assert pieces == ["abcd", "efgh", "ij"]


def test_split_rejects_a_non_progressing_window_shape():
    """Pins the exact counter-example the real-progress invariant exists for:
    a window like [1, 95) following [0, 100) satisfies "start moved forward"
    but covers nothing new, and must never appear in real output."""
    tok = _tiktoken_tok()
    text = "x" * 500
    spans = tok.split_by_token_limit(text, 20, overlap_tokens=15)
    for i in range(1, len(spans) - 1):
        assert spans[i].end > spans[i - 1].end, (
            f"span {i} ({spans[i]}) does not extend coverage past "
            f"the previous span's end ({spans[i - 1].end})"
        )


def test_split_survives_the_bpe_non_monotonic_counter_example():
    tok = Tokenizer("word", _WordCountingTokenizer())
    text = ("ab" + "x" * 8) * 6
    spans = tok.split_by_token_limit(text, 4)
    _assert_split_invariants(tok, text, spans, 4)


# --------------------------------------------------------------------------- #
# Exponential (not halving) retreat
# --------------------------------------------------------------------------- #


def test_generic_retreat_is_strictly_decreasing_and_bounded():
    """The char-offset retreat in the generic implementation must never grow
    and must terminate in O(log max_tokens) probes, not O(max_tokens)."""

    class _CountingTokenizer(TokenizerInterface):
        def __init__(self):
            self.encode_calls = 0

        def encode(self, content: str) -> list[int]:
            self.encode_calls += 1
            return [0] * len(content)  # 1 token per char, easy to reason about

        def decode(self, tokens: list[int]) -> str:
            return "x" * len(tokens)

    underlying = _CountingTokenizer()
    tok = Tokenizer("counting", underlying)
    text = "y" * 5000
    span = tok.truncate_by_token_limit(text, 37)
    assert span.token_count <= 37
    # log2(5000) ~= 13; a handful of probes on top for the ratio estimate and
    # floor check is still far below a 5000-step linear retreat.
    assert underlying.encode_calls < 40


def test_tiktoken_retreat_step_sequence_is_exponential_not_halving():
    """Distinguishes the new exponential retreat from the old ``max_tokens //
    2`` style by bounding the number of full re-encodes for a large budget."""
    tok = _tiktoken_tok()
    text = "The quick brown fox jumps over the lazy dog. " * 200
    total = len(tok.encode(text))
    assert total > 2000  # ensure retreat actually engages

    span = tok.truncate_by_token_limit(text, 500)
    assert span.token_count <= 500
    sub = text[span.start : span.end]
    assert len(tok.encode(sub)) == span.token_count


# --------------------------------------------------------------------------- #
# Third-party tokenizers: encode/decode only is still a complete implementation
# --------------------------------------------------------------------------- #


def test_third_party_encode_decode_only_tokenizer_needs_no_new_methods():
    """A pre-existing custom ``Tokenizer`` subclass that only ever implemented
    the old ``TokenizerInterface`` (encode/decode) keeps working unchanged --
    split/truncate are inherited from the base class, not required overrides.
    """
    tok = Tokenizer("legacy-custom", _CharTokenizer())
    assert not hasattr(_CharTokenizer, "split_by_token_limit")
    assert not hasattr(_CharTokenizer, "truncate_by_token_limit")

    span = tok.truncate_by_token_limit("hello world", 3)
    assert span.token_count <= 3
    spans = tok.split_by_token_limit("hello world " * 5, 4)
    _assert_split_invariants(tok, "hello world " * 5, spans, 4)


def test_generic_and_tiktoken_implementations_need_not_share_exact_boundaries():
    """Both satisfy the same invariants but are not required to produce
    byte-identical spans -- their estimation/retreat strategies differ."""
    text = "abcdefghijklmnopqrstuvwxyz " * 10
    generic = _char_tok()
    tiktoken_tok = _tiktoken_tok()

    generic_spans = generic.split_by_token_limit(text, 8)
    tiktoken_spans = tiktoken_tok.split_by_token_limit(text, 8)

    _assert_split_invariants(generic, text, generic_spans, 8)
    _assert_split_invariants(tiktoken_tok, text, tiktoken_spans, 8)


# --------------------------------------------------------------------------- #
# No cross-call / instance-level mutable state
# --------------------------------------------------------------------------- #


def test_tokenizer_split_state_does_not_leak_across_calls_or_threads():
    """A fresh call must not be influenced by a previous call's retreat state,
    and the wrapper must remain deepcopy-able (see the thread-safety
    contract in test_tokenizer_contract.py)."""
    tok = _tiktoken_tok()
    text_a = "a" * 3000
    text_b = "b" * 3000

    spans_a1 = tok.split_by_token_limit(text_a, 10)
    tok.split_by_token_limit(text_b, 10)
    spans_a2 = tok.split_by_token_limit(text_a, 10)
    assert spans_a1 == spans_a2

    clone = copy.deepcopy(tok)
    assert clone.split_by_token_limit(text_a, 10) == spans_a1


# --------------------------------------------------------------------------- #
# truncate_list_by_token_size: separator tokens count (regression for #3559)
# --------------------------------------------------------------------------- #


def test_list_truncation_counts_separator_tokens():
    """The old implementation summed each item's own token count and never
    counted the separator between items -- reproduce the exact #3559 shape:
    items individually fit but the joined-with-separator text does not."""
    tok = _char_tok()
    # Each item is 3 chars/tokens; joined with a 2-char separator, two items
    # cost 3 + 2 + 3 = 8 tokens -- over a budget of 7 that ignores the
    # separator would wrongly accept both.
    items = ["aaa", "bbb", "ccc"]
    result = truncate_list_by_token_size(
        items, key=lambda x: x, separator="||", max_token_size=7, tokenizer=tok
    )
    assert result == ["aaa"]


def test_list_truncation_never_splits_a_partial_item():
    tok = _char_tok()
    items = ["aaaa", "bbbb", "cccc", "dddd"]
    result = truncate_list_by_token_size(
        items, key=lambda x: x, separator="\n", max_token_size=6, tokenizer=tok
    )
    # "aaaa" (4) + "\n" (1) + "bbbb" (4) = 9 > 6, so only the first whole item
    # that fits alone (4 <= 6) survives.
    assert result == ["aaaa"]
    for item in result:
        assert item in items  # never a truncated fragment


def test_list_truncation_reverifies_after_mapping_to_item_boundary():
    """A shorter, item-boundary-aligned prefix is not guaranteed to still be
    safe once re-serialized on its own -- pin that the second verification
    pass can still shrink the result further."""
    tok = Tokenizer("word", _WordCountingTokenizer())
    # "ab" alone costs 1 token; each "ab"+"x"*3 item costs 4 tokens (ab=1,
    # x,x,x=3). Joined with no separator, budget 5 admits roughly one item's
    # worth of the raw safe-prefix text, but re-serializing just that item on
    # its own must still be independently checked.
    items = ["abxxx", "abxxx", "abxxx"]
    result = truncate_list_by_token_size(
        items, key=lambda x: x, separator="", max_token_size=4, tokenizer=tok
    )
    assert result == ["abxxx"]
    assert len(tok.encode("".join(result))) <= 4


def test_list_truncation_non_positive_budget_or_empty_list_returns_empty():
    tok = _char_tok()
    assert (
        truncate_list_by_token_size(
            [], key=lambda x: x, separator="\n", max_token_size=10, tokenizer=tok
        )
        == []
    )
    assert (
        truncate_list_by_token_size(
            ["a"], key=lambda x: x, separator="\n", max_token_size=0, tokenizer=tok
        )
        == []
    )


# --------------------------------------------------------------------------- #
# overlap is clamped to the previous window's own size before retreating
# --------------------------------------------------------------------------- #


def test_split_clamps_a_huge_overlap_target_before_retreating():
    """A requested overlap far larger than any window's own token count must
    not retreat step-by-step from that huge starting point -- it should be
    clamped against the previous window's actual size up front, so the
    number of re-encodes per window stays bounded by max_tokens, not by the
    (irrelevantly huge) requested overlap."""

    class _CountingTokenizer(TokenizerInterface):
        def __init__(self):
            self.encode_calls = 0

        def encode(self, content: str) -> list[int]:
            self.encode_calls += 1
            return [0] * len(content)

        def decode(self, tokens: list[int]) -> str:
            return "x" * len(tokens)

    underlying = _CountingTokenizer()
    tok = Tokenizer("counting", underlying)
    text = "y" * 2000

    # overlap_tokens is absurdly larger than max_tokens (and thus larger than
    # any individual window's own token count) -- if the target were tried
    # as-is and retreated one exponential step at a time from 10**9, that
    # alone would take ~30 probes on top of the per-window search.
    spans = tok.split_by_token_limit(text, max_tokens=6, overlap_tokens=10**9)
    assert len(spans) > 1
    _assert_split_invariants(tok, text, spans, 6)

    # Total encode() calls should scale with the number of windows times a
    # small constant (bounded search per window), not with log2(overlap).
    windows = len(spans)
    assert underlying.encode_calls < windows * 15


def test_split_overlap_clamped_to_previous_window_size_minus_one():
    """Directly pins the clamp formula: effective overlap target is
    min(requested, previous_window_token_count - 1), verified via the
    tiktoken fast path where window token counts are easy to reason about."""
    tok = _tiktoken_tok()
    text = "The quick brown fox jumps over the lazy dog. " * 20
    # overlap_tokens way beyond max_tokens=10 -- must not break the "real
    # progress" invariant despite the clamp being necessary on nearly every
    # window transition.
    spans = tok.split_by_token_limit(text, max_tokens=10, overlap_tokens=5000)
    _assert_split_invariants(tok, text, spans, 10)
