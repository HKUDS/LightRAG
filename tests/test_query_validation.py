import pytest

from lightrag.query_validation import (
    _WIDE_CODEPOINT_RANGES,
    EmptyQueryError,
    RAGQueryTooShortError,
    meets_min_rag_query_weight,
    rag_query_weight,
    validate_query_not_empty,
    validate_rag_query,
)


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "query, expected",
    [
        ("ab", 2),
        ("中", 2),
        ("中a", 3),
        ("中文", 4),
        ("  abc  ", 3),
        ("〇", 2),
        ("𠀀", 2),
        # Kana, Hangul and Bopomofo weigh what Han does: a two-character
        # Japanese, Korean or zhuyin word carries as much retrieval signal as a
        # two-character Chinese one, and weighing only Han rejected all three.
        ("ねこ", 4),
        ("データ", 6),
        ("한글", 4),
        ("오늘", 4),
        ("ㄅㄆ", 4),
        # Halfwidth forms are the same letters in a legacy encoding, still
        # emitted by Japanese IMEs.
        ("ﾈｺ", 4),
        ("ｶﾅ", 4),
        # Iteration and repeat marks, in both writing directions.
        ("々々", 4),
        ("〳〵", 4),
        ("二〇", 4),
    ],
)
def test_rag_query_weight_counts_east_asian_characters_as_two(query, expected):
    assert rag_query_weight(query) == expected


@pytest.mark.parametrize("query", ["", "  ", "a", "ab", "中", "〇"])
def test_validate_rag_query_rejects_weight_below_three(query):
    with pytest.raises(ValueError):
        validate_rag_query(query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("abc", "abc"),
        ("中a", "中a"),
        (" 中文 ", "中文"),
        ("ねこ", "ねこ"),
        ("한글", "한글"),
        ("ﾈｺ", "ﾈｺ"),
    ],
)
def test_validate_rag_query_accepts_and_strips_valid_queries(query, expected):
    assert validate_rag_query(query) == expected


@pytest.mark.parametrize("halfwidth, fullwidth", [("ﾈｺ", "ネコ"), ("ｶﾅ", "カナ")])
def test_halfwidth_and_fullwidth_kana_weigh_the_same(halfwidth, fullwidth):
    """The same word must not hinge on which encoding the keyboard emits."""
    assert rag_query_weight(halfwidth) == rag_query_weight(fullwidth)
    assert validate_rag_query(halfwidth) == halfwidth


@pytest.mark.parametrize("query", ["", "   ", "\t\n "])
def test_validate_query_not_empty_rejects_blank_text(query):
    with pytest.raises(EmptyQueryError):
        validate_query_not_empty(query)


@pytest.mark.parametrize("query, expected", [("a", "a"), ("  中  ", "中")])
def test_validate_query_not_empty_allows_any_non_blank_text(query, expected):
    """The non-empty rule is mode-independent; the RAG minimum is not."""
    assert validate_query_not_empty(query) == expected


def test_an_empty_rag_query_is_reported_as_empty_not_as_too_short():
    """The user-facing distinction: nothing typed vs. not enough typed."""
    with pytest.raises(EmptyQueryError):
        validate_rag_query("   ")
    with pytest.raises(RAGQueryTooShortError):
        validate_rag_query("ab")


@pytest.mark.parametrize(
    "query, expected",
    [("", False), ("ab", False), ("中", False), ("abc", True), ("中a", True)],
)
def test_meets_min_rag_query_weight_agrees_with_the_full_walk(query, expected):
    assert meets_min_rag_query_weight(query) is expected
    assert meets_min_rag_query_weight(query) is (rag_query_weight(query) >= 3)


def test_the_threshold_check_does_not_walk_the_whole_query():
    """A 64 KiB query must not cost 65,536 iterations on the event loop.

    `rag_query_weight` walking the full string added ~69 ms of synchronous CPU
    to every max-size RAG request; only the comparison against the minimum is
    ever needed, and that is decided by the third character at the latest.
    """
    consumed = 0

    class _CountingStr(str):
        """Counts how far the validator actually reads."""

        def __iter__(self):
            nonlocal consumed
            for character in str(self):
                consumed += 1
                yield character

        def strip(self):
            return self

    assert meets_min_rag_query_weight(_CountingStr("a" * 65536)) is True
    assert consumed == 3


@pytest.mark.parametrize(
    "codepoint, block",
    [
        (0x4E00, "CJK Unified Ideographs"),
        (0x3400, "Extension A"),
        (0x20000, "Extension B"),
        (0x2A700, "Extension C"),
        (0x2EBF0, "Extension I"),
        (0x31350, "Extension H"),
        (0x323B0, "Extension J, assigned in Unicode 17"),
        (0x3FFFD, "the far end of plane 3 — extensions not yet assigned"),
    ],
)
def test_every_cjk_ideograph_plane_is_covered_without_an_edit(codepoint, block):
    """Why the table is coarse rather than per assigned character.

    Planes 2 and 3 are allocated to CJK ideographs, so one range covers every
    extension that exists and every one that will. Enumerating assigned
    characters instead meant Extension J shipped in Unicode 17 weighing 1, and
    needed an edit in two languages to fix — as would Extension K, and so on.
    """
    assert rag_query_weight(chr(codepoint) * 2) == 4, block


@pytest.mark.parametrize(
    "query, what",
    [
        ("・・", "U+30FB KATAKANA MIDDLE DOT"),
        ("゛゜", "U+309B/U+309C sound marks"),
        ("ㅤㅤ", "U+3164 HANGUL FILLER, which renders as nothing"),
        ("\U0001afff\U0001afff", "unassigned, inside Kana Extended-B"),
    ],
)
def test_the_coarse_table_over_counts_and_that_is_the_deal(query, what):
    """Pins the imprecision as deliberate, so it is not "fixed" back.

    The ranges are block- and plane-granular, so punctuation, fillers and
    unassigned code points inside them weigh 2, and two of them clear the
    minimum. Chasing that per character is what made the table need an edit, in
    two languages, on every Unicode release. The cost of getting it wrong is one
    retrieval that finds nothing, for input nobody types meaning to ask
    something — this check is a coarse heuristic for "did the user ask
    anything", not a correctness boundary.
    """
    assert rag_query_weight(query) == 4, what
    assert validate_rag_query(query) == query


def test_the_weighted_ranges_are_sorted_and_disjoint():
    for earlier, later in zip(_WIDE_CODEPOINT_RANGES, _WIDE_CODEPOINT_RANGES[1:]):
        assert earlier[0] <= earlier[1]
        assert earlier[1] < later[0]


def test_the_frontend_range_table_matches_this_one():
    """The two sides are duplicated by necessity and drift silently.

    The WebUI pre-validates so the user is told why before a round trip; the
    server validates because it is the authority. They only agree if the tables
    agree, and a comment asking for that is not a mechanism — the halfwidth kana
    gap was exactly this drift, present in both files at once.
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "lightrag_webui"
        / "src"
        / "utils"
        / "queryValidation.ts"
    ).read_text(encoding="utf-8")

    def parse(name: str) -> tuple[tuple[int, int], ...]:
        table = re.search(rf"{name}[^=]*=\s*\[(.*?)\n\]", source, re.DOTALL)
        assert table, f"could not locate {name} in queryValidation.ts"
        return tuple(
            (int(start, 16), int(end, 16))
            for start, end in re.findall(
                r"\[\s*0x([0-9a-fA-F]+)\s*,\s*0x([0-9a-fA-F]+)\s*\]", table.group(1)
            )
        )

    assert parse("WIDE_CODEPOINT_RANGES") == _WIDE_CODEPOINT_RANGES
