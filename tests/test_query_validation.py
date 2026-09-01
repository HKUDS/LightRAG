import pytest

from lightrag.query_validation import (
    _WIDE_CODEPOINT_RANGES,
    _WIDE_NUMERAL_RANGES,
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
        # Iteration and repeat marks are letters (Lm/Lo) and weigh 2, in both
        # writing directions — the vertical marks were missing while the
        # horizontal ones at U+309D-U+309F were already doubled.
        ("〳〵", 4),  # vertical kana repeat marks
        ("々々", 4),  # U+3005 ideographic iteration mark
        ("〆〆", 4),  # U+3006 ideographic closing mark
        ("〻〼", 4),  # U+303B vertical ideographic iteration, U+303C masu mark
        ("〡〢", 4),  # Hangzhou numerals, Nl like 〇
        ("\U00016fe1\U00016fe1", 4),  # NUSHU ITERATION MARK
        ("\U00016fe3\U00016fe3", 4),  # OLD CHINESE ITERATION MARK
        ("\U00016fe2\U00016fe2", 2),  # OLD CHINESE HOOK MARK is Po
        ("\U00016fe0\U00016fe0", 2),  # TANGUT ITERATION MARK writes another language
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


def test_the_frontend_range_tables_match_these_ones():
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
    assert parse("WIDE_NUMERAL_RANGES") == _WIDE_NUMERAL_RANGES


# Letters assigned after UCD 14.0.0, which the interpreter running these tests
# cannot see. This is the ONLY hand-maintained part of the weighted table's
# provenance; everything else comes from ``unicodedata``. Refresh it when the
# table is regenerated on a newer Python, and delete entries the interpreter has
# caught up on — ``test_the_post_ucd14_allowlist_has_no_stale_entries`` says
# which.
_POST_UCD_14_LETTERS = frozenset(
    {0x2B739}  # Unicode 15.0, CJK Extension C
    | {0x1B132, 0x1B155}  # Unicode 15.1, Small Kana Extension
    | set(range(0x2EBF0, 0x2EE5D + 1))  # Unicode 15.1, CJK Extension I (622)
    | set(range(0x31350, 0x323AF + 1))  # Unicode 15.0, CJK Extension H (4192)
)


def test_every_weighted_code_point_is_an_assigned_letter():
    """The table invariant, with no code point skipped.

    Doubling whole Unicode blocks is how things that say nothing got weighted:
    "・・" and "゛゜" cleared the minimum, as did two HANGUL FILLERs — which
    render as nothing — and unassigned tails such as U+1AFFF, U+2B73A and
    U+2CEA2. Listing those as examples would pin only the ones we happened to
    notice; this pins the property over all 110,000-odd weighted code points.

    An entry this interpreter calls unassigned is accepted ONLY if it appears in
    ``_POST_UCD_14_LETTERS`` — CPython 3.11 ships UCD 14.0.0, which predates CJK
    Extensions H and I. A JavaScript runtime is deliberately not consulted:
    Bun 1.3.11 reports U+2B73A-U+2B73F, U+2CEA2-U+2CEAD and U+323B0 (past the
    end of Extension H) as assigned letters, and generating the table from that
    is exactly how those tails got in.
    """
    import unicodedata

    offenders = []
    for start, end in _WIDE_CODEPOINT_RANGES:
        for codepoint in range(start, end + 1):
            character = chr(codepoint)
            name = unicodedata.name(character, None)
            if name is None:
                if codepoint not in _POST_UCD_14_LETTERS:
                    offenders.append((codepoint, "unassigned", "-"))
                continue
            if not unicodedata.category(character).startswith("L"):
                offenders.append((codepoint, "not a letter", name))
            elif "FILLER" in name:
                offenders.append((codepoint, "filler", name))

    assert offenders == [], "\n".join(
        f"U+{cp:04X} {why}: {name}" for cp, why, name in offenders
    )


def test_the_post_ucd14_allowlist_has_no_stale_entries():
    """On a newer Python the allowlist shrinks; keep it honest, not cumulative.

    Every entry must be either still-unknown to this interpreter, or a real
    letter it has since learned — never something that turned out not to be a
    letter at all.
    """
    import unicodedata

    wrong = [
        cp
        for cp in sorted(_POST_UCD_14_LETTERS)
        if unicodedata.name(chr(cp), None) is not None
        and not unicodedata.category(chr(cp)).startswith("L")
    ]
    assert wrong == [], [f"U+{cp:04X}" for cp in wrong]


def test_the_allowlist_covers_only_code_points_inside_the_table():
    """An allowlist entry outside the ranges would silently do nothing."""
    covered = {
        cp for start, end in _WIDE_CODEPOINT_RANGES for cp in range(start, end + 1)
    }
    assert _POST_UCD_14_LETTERS <= covered


@pytest.mark.parametrize(
    "codepoint",
    [
        0x1AFF4,
        0x1AFFC,
        0x1AFFF,  # Kana Extended-B gaps
        0x3040,
        0x3097,  # Hiragana block gaps
        0x2B73A,
        0x2B73F,  # CJK Extension C tail
        0x2CEA2,
        0x2CEAD,  # CJK Extension E tail
    ],
)
def test_unassigned_code_points_are_not_weighted(codepoint):
    """Two of anything unassigned must not add up to a meaningful query."""
    query = chr(codepoint) * 2
    assert rag_query_weight(query) == 2
    with pytest.raises(RAGQueryTooShortError):
        validate_rag_query(query)


def test_the_ideographic_zero_is_weighted_despite_not_being_a_letter():
    """U+3007 is category Nl, so it is a deliberate exception to the table.

    It is how a Chinese year is written — 二〇二五年 — and says as much as the
    digits it stands in for.
    """
    import unicodedata

    assert unicodedata.category("〇") == "Nl"
    assert rag_query_weight("二〇") == 4


def test_the_weighted_ranges_are_sorted_and_disjoint():
    """Several blocks are split around excluded code points; keep it readable."""
    for earlier, later in zip(_WIDE_CODEPOINT_RANGES, _WIDE_CODEPOINT_RANGES[1:]):
        assert earlier[0] <= earlier[1]
        assert earlier[1] < later[0]


@pytest.mark.parametrize("query", ["・・", "゛゜", "゠゠", "ㅤㅤ", "。。"])
def test_punctuation_only_queries_never_clear_the_minimum(query):
    with pytest.raises(RAGQueryTooShortError):
        validate_rag_query(query)


def test_the_numeral_ranges_are_exactly_the_block_nl_set():
    """The second table is closed, not a running list of the ones we noticed.

    〇 was originally a lone special case. It is category Nl rather than L*, so
    it cannot join the letter table without weakening that table's invariant —
    but every other Nl character in the CJK Symbols and Punctuation block is the
    same idea (a CJK numeral written as a character), and leaving them out while
    doubling 〇 was an inconsistency waiting to be reported.
    """
    import unicodedata

    declared = {
        cp for start, end in _WIDE_NUMERAL_RANGES for cp in range(start, end + 1)
    }
    block_nl = {
        cp
        for cp in range(0x3000, 0x3040)
        if unicodedata.name(chr(cp), None) and unicodedata.category(chr(cp)) == "Nl"
    }
    assert declared == block_nl


# Names of the scripts used to write Chinese, Japanese and Korean, matched as
# SUBSTRINGS of the Unicode character name. This replaces a hand-listed set of
# Unicode blocks, which was the third audit in a row whose own coverage was the
# defect: a name-prefix match missed "VERTICAL KANA REPEAT MARK", the block list
# that replaced it omitted Bopomofo, and the corrected block list still omitted
# Ideographic Symbols and Punctuation. A keyword scan over the WHOLE code space
# has no coverage to under-specify — the contract is the keyword list itself.
#
# Tangut and Khitan are absent on purpose: they are written with Han-derived
# characters but are not Chinese, Japanese or Korean, which is what the
# minimum's contract speaks about.
_CJK_SCRIPT_KEYWORDS = (
    "CJK",
    "HIRAGANA",
    "KATAKANA",
    "KANA",
    "HANGUL",
    "IDEOGRAPH",
    "BOPOMOFO",
    "NUSHU",
    "CHINESE",
    "KANBUN",
)

# Every letter the scan reaches that is deliberately NOT weighted. Small and
# fully reasoned by construction — if it ever needs a new entry, that is a
# decision someone is making on purpose.
_DELIBERATELY_UNWEIGHTED = {
    # Fillers render as nothing at all; two of them are not a query.
    0x115F: "HANGUL CHOSEONG FILLER",
    0x1160: "HANGUL JUNGSEONG FILLER",
    0x3164: "HANGUL FILLER",
    0xFFA0: "HALFWIDTH HANGUL FILLER",
    # The other half of the preceding letter — ｶ + ﾞ is the single syllable
    # fullwidth writes as ガ. Weighting the mark would cost 4 halfwidth against
    # 2 fullwidth, inverting the equivalence that put halfwidth kana in the
    # table at all.
    0xFF9E: "HALFWIDTH KATAKANA VOICED SOUND MARK",
    0xFF9F: "HALFWIDTH KATAKANA SEMI-VOICED SOUND MARK",
    # Not CJK: the keywords appear inside an unrelated word.
    0x16D2: "RUNIC LETTER BERKANAN BEORC BJARKAN B",
    0x10094: "LINEAR B MONOGRAM B128 KANAKO",
}


def test_no_cjk_letter_lives_outside_the_two_tables():
    """Closes the class the vertical kana repeat marks belonged to.

    Reporting them one at a time — 〳〵, then ㄅㄆ, then the Nushu iteration
    mark — never ends. Every assigned letter of every CJK-writing script in the
    whole code space is either weighted or listed above with a reason.
    """
    import unicodedata

    weighted = {
        cp
        for start, end in _WIDE_CODEPOINT_RANGES + _WIDE_NUMERAL_RANGES
        for cp in range(start, end + 1)
    }

    missing = []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:  # surrogates are not characters
            continue
        if codepoint in weighted or codepoint in _DELIBERATELY_UNWEIGHTED:
            continue
        character = chr(codepoint)
        name = unicodedata.name(character, None)
        if not name or not unicodedata.category(character).startswith("L"):
            continue
        if any(keyword in name for keyword in _CJK_SCRIPT_KEYWORDS):
            missing.append((codepoint, name))

    assert missing == [], "\n".join(f"U+{cp:04X} {name}" for cp, name in missing)


def test_the_exclusion_list_is_not_a_dumping_ground():
    """Each exclusion must still be the letter its comment claims it is.

    A stale entry would silently hide a real gap the moment Unicode reuses or
    renames nothing — the point is that the list stays readable and each line
    keeps earning its place.
    """
    import unicodedata

    for codepoint, expected_name in _DELIBERATELY_UNWEIGHTED.items():
        character = chr(codepoint)
        assert unicodedata.name(character, None) == expected_name, f"U+{codepoint:04X}"
        assert unicodedata.category(character).startswith("L")


@pytest.mark.parametrize("codepoint", [0xFF9E, 0xFF9F])
def test_halfwidth_sound_marks_stay_at_one(codepoint):
    """A deliberate exception to the letters rule, pinned so it stays deliberate.

    U+FF9E/U+FF9F are Lm, so the rule alone would double them — but halfwidth
    ｶﾞ is the single syllable fullwidth writes as ガ. Weighting the mark would
    make that syllable weigh 4 halfwidth against 2 fullwidth.
    """
    import unicodedata

    assert unicodedata.category(chr(codepoint)) == "Lm"
    assert rag_query_weight(chr(codepoint)) == 1
    assert rag_query_weight("ｶ" + chr(codepoint)) == 3
