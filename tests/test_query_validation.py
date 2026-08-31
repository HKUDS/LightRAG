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
        # Kana and Hangul weigh the same as Han: a two-character Japanese or
        # Korean word carries as much retrieval signal as a two-character
        # Chinese one, and counting it as 2 rejected both.
        ("ねこ", 4),
        ("データ", 6),
        ("한글", 4),
        ("오늘", 4),
        # Halfwidth forms are the same letters in a legacy encoding — still
        # typed on Japanese systems — so they weigh what their fullwidth
        # counterparts do.
        ("ﾈｺ", 4),
        ("ｶﾅ", 4),
        ("ﾡﾢ", 4),
        # Modifier LETTERS spell real words and weigh 2 …
        ("コーヒー", 8),
        ("\U0001aff0\U0001aff1", 4),  # Kana Extended-B (Minnan tone marks)
        # … while punctuation, modifier symbols and fillers stay at 1.
        ("・・", 2),  # U+30FB KATAKANA MIDDLE DOT (Po)
        ("゛゜", 2),  # U+309B/U+309C voiced sound marks (Sk)
        ("゠゠", 2),  # U+30A0 KATAKANA-HIRAGANA DOUBLE HYPHEN (Pd)
        ("ㅤㅤ", 2),  # U+3164 HANGUL FILLER — renders as nothing at all
    ],
)
def test_rag_query_weight_counts_cjk_as_two(query, expected):
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
    ],
)
def test_validate_rag_query_accepts_and_strips_valid_queries(query, expected):
    assert validate_rag_query(query) == expected


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


@pytest.mark.parametrize("halfwidth, fullwidth", [("ﾈｺ", "ネコ"), ("ｶﾅ", "カナ")])
def test_halfwidth_and_fullwidth_kana_weigh_the_same(halfwidth, fullwidth):
    """The same word must not hinge on which encoding the keyboard emits.

    Halfwidth katakana renders narrow, but the rule counts retrieval signal,
    not display width — 'ﾈｺ' and 'ネコ' are one word.
    """
    assert rag_query_weight(halfwidth) == rag_query_weight(fullwidth)
    assert validate_rag_query(halfwidth) == halfwidth


def test_halfwidth_punctuation_and_sound_marks_are_not_letters():
    """Only letters weigh 2; the marks around them stay at 1."""
    assert rag_query_weight("｡") == 1  # HALFWIDTH IDEOGRAPHIC FULL STOP
    assert rag_query_weight("･") == 1  # HALFWIDTH KATAKANA MIDDLE DOT
    assert rag_query_weight("ﾞ") == 1  # HALFWIDTH KATAKANA VOICED SOUND MARK
    assert rag_query_weight("ﾟ") == 1


def test_the_frontend_range_table_matches_this_one():
    """The two tables are duplicated by necessity and drift silently.

    The WebUI pre-validates so the user is told why before a round trip; the
    server validates because it is the authority. They only agree if the ranges
    agree, and a comment asking for that is not a mechanism — the halfwidth
    kana gap was exactly this drift, present in both files at once.
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

    table = re.search(r"WIDE_CODEPOINT_RANGES[^=]*=\s*\[(.*?)\n\]", source, re.DOTALL)
    assert table, "could not locate WIDE_CODEPOINT_RANGES in queryValidation.ts"

    frontend = tuple(
        (int(start, 16), int(end, 16))
        for start, end in re.findall(
            r"\[\s*0x([0-9a-fA-F]+)\s*,\s*0x([0-9a-fA-F]+)\s*\]", table.group(1)
        )
    )

    assert frontend == _WIDE_CODEPOINT_RANGES

    # The lone code point outside the ranges has to be mirrored too.
    assert "codePoint === 0x3007" in source


def test_every_weighted_code_point_is_a_letter():
    """The rule for the table, enforced on the table.

    Doubling whole Unicode blocks is how punctuation got in: "・・" and "゛゜"
    weighed 4 and cleared the minimum, as did two HANGUL FILLERs, which render
    as nothing at all. Listing those as examples would only pin the ones we
    happened to notice; this pins the property, so the next block someone adds
    cannot quietly reopen the hole.
    """
    import unicodedata

    offenders = []
    for start, end in _WIDE_CODEPOINT_RANGES:
        for codepoint in range(start, end + 1):
            character = chr(codepoint)
            name = unicodedata.name(character, None)
            if name is None:
                continue  # unassigned: cannot be typed, harmless inside a range
            if not unicodedata.category(character).startswith("L"):
                offenders.append((codepoint, "not a letter", name))
            elif "FILLER" in name:
                offenders.append((codepoint, "filler", name))

    assert offenders == [], "\n".join(
        f"U+{cp:04X} {why}: {name}" for cp, why, name in offenders
    )


def test_the_weighted_ranges_are_sorted_and_disjoint():
    """Several blocks are split around excluded code points; keep it readable."""
    for earlier, later in zip(_WIDE_CODEPOINT_RANGES, _WIDE_CODEPOINT_RANGES[1:]):
        assert earlier[0] <= earlier[1]
        assert earlier[1] < later[0]


@pytest.mark.parametrize("query", ["・・", "゛゜", "゠゠", "ㅤㅤ", "。。"])
def test_punctuation_only_queries_never_clear_the_minimum(query):
    with pytest.raises(RAGQueryTooShortError):
        validate_rag_query(query)
