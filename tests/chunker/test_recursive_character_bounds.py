"""Separator-cascade bounds and the no-op amplification (GHSA-26pm-px5v-8c4w).

The recursive splitter descends one level per remaining separator and re-scans
the text at every level, so cost is ``O(len(separators) x len(text))`` with both
factors supplied by one request. Bounding the list in the request model covers
HTTP callers only — CHUNK_R_SEPARATORS, addon_params, direct SDK calls and per-doc
snapshots persisted before that cap existed all bypass it — so the chunker bounds
whatever it is handed.

Two of the bounding rules are there to avoid breaking things the naive version of
this fix would have broken, and are marked below.
"""

from __future__ import annotations

import pytest

from lightrag.chunker.recursive_character import (
    chunking_by_recursive_character,
    normalize_r_separators,
)
from lightrag.constants import (
    DEFAULT_R_SEPARATORS,
    MAX_R_SEPARATOR_CHARS,
    MAX_R_SEPARATORS,
)
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline


class _CountingTokenizer:
    """Counts encodes so cost can be asserted without timing anything."""

    def __init__(self):
        self.encodes = 0

    def encode(self, content: str):
        self.encodes += 1
        # One token per 4 characters, deterministic and cheap.
        return list(range(max(len(content) // 4, 1)))

    def decode(self, tokens):
        return "x" * (len(tokens) * 4)


def _tok() -> tuple[Tokenizer, _CountingTokenizer]:
    underlying = _CountingTokenizer()
    return Tokenizer("test-model", underlying), underlying


# --------------------------------------------------------------------------- #
# normalize_r_separators
# --------------------------------------------------------------------------- #


def test_none_passes_through_untouched():
    """``None`` means "use the splitter's own default cascade".

    That cascade is LangChain's four-entry English one, not the repo's nine-entry
    DEFAULT_R_SEPARATORS. Substituting one for the other here would silently
    change how a direct SDK caller's CJK text splits.
    """
    assert normalize_r_separators(None) is None


def test_a_cascade_within_the_limits_is_returned_verbatim():
    original = list(DEFAULT_R_SEPARATORS)
    assert normalize_r_separators(original) == original


def test_over_long_entries_are_dropped():
    separators = ["\n\n", "x" * (MAX_R_SEPARATOR_CHARS + 1), "\n"]
    assert normalize_r_separators(separators) == ["\n\n", "\n"]


def test_a_long_cascade_is_truncated():
    separators = [f"s{i}" for i in range(900)]
    bounded = normalize_r_separators(separators)
    assert len(bounded) == MAX_R_SEPARATORS
    assert bounded == separators[:MAX_R_SEPARATORS]


def test_truncation_preserves_the_char_level_sentinel():
    """Regression guard, not a plain bound.

    The empty string is the char-level fallback: ``_split_text_with_spans`` takes
    the ``if not candidate`` branch on it and splits anything. Taking the first N
    entries drops it whenever the cascade is long, so ``new_separators`` runs out
    early and a segment with no ordinary separator in it is emitted WHOLE — a
    data-quality regression introduced by a security fix.
    """
    separators = [f"s{i}" for i in range(MAX_R_SEPARATORS)] + [""]
    bounded = normalize_r_separators(separators)

    assert len(bounded) == MAX_R_SEPARATORS
    assert bounded[-1] == ""


def test_a_cascade_without_a_sentinel_does_not_gain_one():
    """``load_chunk_separators`` strips the sentinel deliberately."""
    separators = [f"s{i}" for i in range(900)]
    assert "" not in normalize_r_separators(separators)


def test_everything_over_long_normalizes_to_empty_not_to_a_substitute():
    """The normalizer only bounds; what to do with an empty result is the
    caller's decision, and the two callers decide differently."""
    separators = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3
    assert normalize_r_separators(separators) == []


def test_runtime_normalizer_is_silent_for_an_invalid_snapshot(monkeypatch):
    """The hot-path backstop must not recreate a warning per document.

    ``caplog`` cannot express this assertion: ``lightrag.utils.logger`` sets
    ``propagate = False``, so no record ever reaches pytest's root handler and
    ``assert not caplog.records`` would hold even while the function warns on
    every call. Intercept the logger the module actually uses instead.
    """
    import lightrag.chunker.recursive_character as recursive_character

    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    over_long = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3
    too_many = [f"s{index}" for index in range(MAX_R_SEPARATORS + 6)]

    # Both correction kinds, twice each: a re-processed snapshot must stay quiet.
    for _ in range(2):
        assert normalize_r_separators(over_long) == []
        assert len(normalize_r_separators(too_many)) == MAX_R_SEPARATORS

    assert warnings == []


# --------------------------------------------------------------------------- #
# chunking_by_recursive_character consumes the bounds
# --------------------------------------------------------------------------- #


def test_the_sentinel_still_splits_an_otherwise_unsplittable_text():
    """The other half of the sentinel guard, end to end.

    A long run with no ordinary separator can only be broken at the char level.
    If truncation dropped the sentinel this returns one oversized chunk.
    """
    tokenizer, _ = _tok()
    text = "A" * 4000
    separators = [f"s{i}" for i in range(MAX_R_SEPARATORS)] + [""]

    chunks = chunking_by_recursive_character(
        tokenizer, text, 100, chunk_overlap_token_size=0, separators=separators
    )

    assert len(chunks) > 1


def test_an_all_over_long_cascade_falls_back_instead_of_crashing():
    """``_split_text_with_spans`` reads ``separators[-1]`` on its first line, so
    an empty cascade is an IndexError, not a degraded split."""
    tokenizer, _ = _tok()
    separators = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3

    chunks = chunking_by_recursive_character(
        tokenizer,
        "hello world " * 200,
        50,
        chunk_overlap_token_size=0,
        separators=separators,
    )

    assert chunks


def test_the_chunker_entry_point_is_silent_across_repeated_calls(monkeypatch):
    """One bad SDK argument must not produce one WARNING per document.

    ``normalize_r_separators`` going quiet is not enough on its own: the
    all-dropped fallback inside ``chunking_by_recursive_character`` used to warn
    unconditionally, so an application calling the documented entry point in a
    loop still got a log line per document from a single unchanging argument.
    This asserts the real entry point, not just the normalizer.

    The branch is only reachable from a direct SDK call: the env and
    addon_params ingress points report the emptiness once when they cache it,
    and ``_PipelineMixin`` removes the ``separators`` key from a stale snapshot
    whose entries were all dropped, so a configured path hands this function
    ``None`` rather than ``[]``.
    """
    import lightrag.chunker.recursive_character as recursive_character

    warnings: list[object] = []
    monkeypatch.setattr(
        recursive_character.logger, "warning", lambda *a, **k: warnings.append(a)
    )

    tokenizer, _ = _tok()
    over_long = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3

    for _ in range(3):
        assert chunking_by_recursive_character(
            tokenizer,
            "hello world\n\nsecond block",
            1200,
            chunk_overlap_token_size=100,
            separators=over_long,
        )

    assert warnings == []


def test_load_chunk_separators_falls_back_to_the_repo_cascade(monkeypatch):
    """The OTHER consumer decides differently, and the docs say so.

    ``chunking_by_recursive_character`` falls back to the splitter's own
    four-entry cascade (the test above). ``load_chunk_separators`` has no
    "splitter default" available — its callers need a usable cascade — so it
    falls back to ``DEFAULT_R_SEPARATORS`` the same way it already does for a
    missing or malformed ``CHUNK_R_SEPARATORS``, minus the sentinel it strips on
    purpose. Pinning both halves because they are documented as differing; a
    change that collapsed them into one fallback would move split points on one
    of the two paths without touching the other.
    """
    import json

    import lightrag.chunker.recursive_character as recursive_character
    import lightrag.multimodal_context as multimodal_context

    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)
    monkeypatch.setattr(multimodal_context.logger, "warning", warnings.append)

    monkeypatch.setenv(
        "CHUNK_R_SEPARATORS",
        json.dumps(["multimodal-warning-cache" * (MAX_R_SEPARATOR_CHARS + 1)] * 3),
    )

    assert multimodal_context.load_chunk_separators() == [
        s for s in DEFAULT_R_SEPARATORS if s
    ]
    assert multimodal_context.load_chunk_separators() == [
        s for s in DEFAULT_R_SEPARATORS if s
    ]
    assert sum("[CHUNK_R_SEPARATORS] dropped" in message for message in warnings) == 1
    assert (
        sum(
            "[load_chunk_separators] no usable separators" in message
            for message in warnings
        )
        == 1
    )


def test_the_all_over_long_fallback_matches_separators_none():
    """It falls back to this function's own default, not to DEFAULT_R_SEPARATORS.

    Anything else would change the split points of a caller who supplied a
    garbage cascade — including, on CJK text, where sentences break.
    """
    text = "第一句。第二句！第三句？" * 40 + "\n\nEnglish paragraph here. " * 20
    tokenizer_a, _ = _tok()
    tokenizer_b, _ = _tok()

    fallback = chunking_by_recursive_character(
        tokenizer_a,
        text,
        60,
        chunk_overlap_token_size=0,
        separators=["x" * (MAX_R_SEPARATOR_CHARS + 1)],
    )
    default = chunking_by_recursive_character(
        tokenizer_b, text, 60, chunk_overlap_token_size=0, separators=None
    )

    assert [c["content"] for c in fallback] == [c["content"] for c in default]


def test_separators_none_is_not_rewritten_to_the_repo_cascade():
    """Direct SDK behaviour must not shift.

    LangChain's default has no CJK punctuation; DEFAULT_R_SEPARATORS does. If
    ``None`` were mapped onto the latter, Chinese text would split in different
    places than it used to.
    """
    text = "第一句。第二句！第三句？" * 40
    tokenizer_a, _ = _tok()
    tokenizer_b, _ = _tok()

    as_none = chunking_by_recursive_character(
        tokenizer_a, text, 60, chunk_overlap_token_size=0, separators=None
    )
    as_repo_cascade = chunking_by_recursive_character(
        tokenizer_b,
        text,
        60,
        chunk_overlap_token_size=0,
        separators=list(DEFAULT_R_SEPARATORS),
    )

    assert [c["content"] for c in as_none] != [c["content"] for c in as_repo_cascade]


# --------------------------------------------------------------------------- #
# The no-op cascade: where the amplification actually came from
# --------------------------------------------------------------------------- #


def _sep_bomb_text() -> str:
    """The advisory's payload shape: the separator occurs once, at offset 0."""
    return "Q" + ("X " * 4000)


@pytest.mark.parametrize("count", [1, 8, MAX_R_SEPARATORS])
def test_a_no_op_cascade_costs_the_same_regardless_of_its_length(count):
    """The load-bearing cost assertion.

    A separator matching only at offset 0 yields one piece identical to the
    input, so the recursion used to call ``length_function`` — a whole-text
    encode — once per level while splitting nothing. Encode count, not wall
    clock: timing is unreliable in CI and the encodes are what the cost was.
    """
    tokenizer, underlying = _tok()

    chunking_by_recursive_character(
        tokenizer, _sep_bomb_text(), 1200, separators=["Q"] * count
    )

    # Two: one for the separator length in the merge step, one for the piece.
    assert underlying.encodes <= 4


def test_the_no_op_skip_does_not_change_the_output():
    """Skipping a level in place must reach the same state the recursion did."""
    tokenizer, _ = _tok()
    text = _sep_bomb_text()

    with_bomb = chunking_by_recursive_character(
        tokenizer, text, 1200, separators=["Q"] * MAX_R_SEPARATORS
    )

    # No separator divides this text, so it is emitted whole — exactly what the
    # recursive form produced, at a fraction of the cost.
    assert len(with_bomb) == 1
    assert with_bomb[0]["content"].startswith("QX")


def test_a_separator_that_really_splits_is_unaffected():
    """The skip must only trigger on a genuine no-op."""
    tokenizer, _ = _tok()
    text = "alpha|beta|gamma|delta " * 50

    chunks = chunking_by_recursive_character(
        tokenizer, text, 20, chunk_overlap_token_size=0, separators=["|", ""]
    )

    assert len(chunks) > 1
    assert "".join(c["content"] for c in chunks).replace(" ", "").count("alpha") == 50


def test_default_cascade_output_is_unchanged_by_the_skip():
    """Sanity net over ordinary text: the optimisation is cost-only."""
    tokenizer, _ = _tok()
    text = "Para one.\n\nPara two is longer and wordier.\n\nPara three.\n" * 20

    chunks = chunking_by_recursive_character(
        tokenizer,
        text,
        40,
        chunk_overlap_token_size=0,
        separators=list(DEFAULT_R_SEPARATORS),
    )

    rejoined = "".join(c["content"] for c in chunks)
    assert "Para one." in rejoined and "Para three." in rejoined
    assert len(chunks) > 1
