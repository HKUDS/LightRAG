"""Numbering styleKey classification and FS_base statistics (spec §2.2.2/§2.2.3).

Twelve numbering shapes, each with a unique ``styleKey``, a priority order
(smaller = shallower level when font sizes tie), optional unit-word sub-order
(篇/部/编/卷 > 章 > 节; 条 > 款 > 项; volume/part > chapter > section), and an
"allow empty title" flag. Classification is pure text analysis — candidacy
(font size / outline / bold gates) and homophone defenses (NER, blacklists)
live in the algorithm layer.

Matching order: ``MultiLevelNum`` first (so ``1.1.2`` is not clipped by the
single-number rule), then the table order top-down, first hit wins
(``Section 3`` stops at EnChapter and never tries EnClause). Single-char
roman numerals (``I.`` / ``v.``) default to EnAlpha and are reclassified to
RomanNum by a deferred second scan only when a multi-char/Unicode RomanNum
companion exists in the same sub-document.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

# --- styleKey constants ------------------------------------------------------

CN_CHAPTER = "CnChapter"
EN_CHAPTER = "EnChapter"
MULTI_LEVEL_NUM = "MultiLevelNum"
CN_CLAUSE = "CnClause"
EN_CLAUSE = "EnClause"
CN_NUM = "CnNum"
CN_PARENT_NUM = "CnParentNum"
ROMAN_NUM = "RomanNum"
EN_NUM = "EnNum"
EN_ALPHA = "EnAlpha"
EN_DOUBLE_PAREN = "EnDoubleParen"
EN_SINGLE_PAREN = "EnSingleParen"

#: styleKey -> level priority (smaller = shallower); spec §2.2.3 table.
STYLE_KEY_PRIORITY: dict[str, int] = {
    CN_CHAPTER: 1,
    EN_CHAPTER: 1,
    MULTI_LEVEL_NUM: 2,
    CN_CLAUSE: 3,
    EN_CLAUSE: 3,
    CN_NUM: 4,
    CN_PARENT_NUM: 5,
    ROMAN_NUM: 6,
    EN_NUM: 7,
    EN_ALPHA: 8,
    EN_DOUBLE_PAREN: 9,
    EN_SINGLE_PAREN: 10,
}

#: styleKeys whose heading may consist of the numbering alone (spec table Y).
ALLOW_EMPTY_TITLE = frozenset({CN_CHAPTER, EN_CHAPTER, CN_CLAUSE, EN_CLAUSE})

_CN_ORD = "一二三四五六七八九十百千万"

# --- the twelve patterns (spec §2.2.3, transcribed exactly) ------------------
# Group 1 always captures the numbering prefix (label); the remainder of the
# paragraph after the full match is the title text.

_P_MULTI_LEVEL = re.compile(
    # Separator "." must not be followed by a digit — otherwise a bare
    # trailing component ("1.2.3" as a whole paragraph) would backtrack into
    # "1.2" + separator "." + title "3" and defeat the no-empty-title rule.
    r"^\s*((?:§+\s*)?\d+(?:\.\d+)+)(?:[、\s]|\.(?!\d)|(?=[一-龥]))"
)
# Shape probe: any multi-level number opener claims the paragraph for
# MultiLevelNum exclusively — when the full rule then rejects it (bare "1.2",
# "3.14"), the paragraph is body, and single-number rules (EnNum "1." +
# title "2") must NOT re-capture it.
_P_MULTI_LEVEL_SHAPE = re.compile(r"^\s*(?:§+\s*)?\d+(?:\.\d+)+")
_P_CN_CHAPTER = re.compile(
    rf"^\s*(第\s*[{_CN_ORD}\d]+\s*[章节篇卷部编])(?:\s|、|：|:|$|(?=[一-龥]))"
)
_P_EN_CHAPTER = re.compile(
    r"^\s*((?:Chapter|Section|Part|Volume)(?![A-Za-z])\s*(?:\d+|[A-Za-z]+))"
    r"(?:\s|:|[.、]|$)",
    re.IGNORECASE,
)
_P_CN_CLAUSE = re.compile(
    rf"^\s*(第\s*[{_CN_ORD}\d]+\s*[条款项])(?:\s|、|：|:|$|(?=[一-龥]))"
)
_P_EN_CLAUSE = re.compile(
    r"^\s*((?:(?:Art(?:icle)?|Sec(?:tion)?|Clause|Para(?:graph)?)(?![A-Za-z])\.?"
    r"|§+|¶+)\s*(?:\d+|[A-Za-z]+))(?:\s|:|[.、]|$)",
    re.IGNORECASE,
)
_P_CN_NUM = re.compile(rf"^\s*([{_CN_ORD}]+)[.、\s]")
_P_CN_PARENT = re.compile(rf"^\s*([（(][{_CN_ORD}]+[）)]|[{_CN_ORD}]+[）)])")
_P_ROMAN = re.compile(r"^\s*([IVX]{2,}|[ivx]{2,}|[Ⅰ-Ⅻⅰ-ⅻ])[.、]")
_P_EN_NUM = re.compile(r"^\s*(\d+)(?:\s|[.、]|(?=[一-龥]))")
_P_EN_ALPHA = re.compile(r"^\s*([A-Za-z])[.、]")
_P_EN_DOUBLE_PAREN = re.compile(r"^\s*([（(](?:\d+|[a-zA-Z])[）)])")
_P_EN_SINGLE_PAREN = re.compile(r"^\s*((?:\d+|[a-zA-Z]))[）)]")

#: Try order: MultiLevelNum first, then the table order top-down.
_MATCH_ORDER: tuple[tuple[str, re.Pattern], ...] = (
    (MULTI_LEVEL_NUM, _P_MULTI_LEVEL),
    (CN_CHAPTER, _P_CN_CHAPTER),
    (EN_CHAPTER, _P_EN_CHAPTER),
    (CN_CLAUSE, _P_CN_CLAUSE),
    (EN_CLAUSE, _P_EN_CLAUSE),
    (CN_NUM, _P_CN_NUM),
    (CN_PARENT_NUM, _P_CN_PARENT),
    (ROMAN_NUM, _P_ROMAN),
    (EN_NUM, _P_EN_NUM),
    (EN_ALPHA, _P_EN_ALPHA),
    (EN_DOUBLE_PAREN, _P_EN_DOUBLE_PAREN),
    (EN_SINGLE_PAREN, _P_EN_SINGLE_PAREN),
)


# --- ordinal / unit parsing ---------------------------------------------------

_CN_DIGIT_VALUES = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNIT_VALUES = {"十": 10, "百": 100, "千": 1000, "万": 10000}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}
_UNICODE_ROMAN_BASE = {
    # U+2160-216B Ⅰ..Ⅻ and U+2170-217B ⅰ..ⅻ map to 1..12
}
for i in range(12):
    _UNICODE_ROMAN_BASE[chr(0x2160 + i)] = i + 1
    _UNICODE_ROMAN_BASE[chr(0x2170 + i)] = i + 1


def parse_cn_ordinal(text: str) -> int | None:
    """Parse a Chinese numeral (no 零 forms — the regex class excludes it)."""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    section = 0  # value accumulated below the current 万
    num = 0
    for ch in text:
        if ch in _CN_DIGIT_VALUES:
            num = _CN_DIGIT_VALUES[ch]
        elif ch in _CN_UNIT_VALUES:
            unit = _CN_UNIT_VALUES[ch]
            if unit == 10000:
                section = (section + (num or 0)) or 1
                total += section * unit
                section = 0
                num = 0
            else:
                section += (num or 1) * unit
                num = 0
        else:
            return None
    return total + section + num or None


def parse_roman(text: str) -> int | None:
    """Strict-ish subtractive roman parse over I/V/X; invalid forms → None."""
    text = text.strip()
    if not text:
        return None
    if len(text) == 1 and text in _UNICODE_ROMAN_BASE:
        return _UNICODE_ROMAN_BASE[text]
    upper = text.upper()
    if any(ch not in _ROMAN_VALUES for ch in upper):
        return None
    total = 0
    prev = 0
    for ch in reversed(upper):
        val = _ROMAN_VALUES[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    # Round-trip check rejects malformed sequences like IIX.
    if _to_roman(total) != upper:
        return None
    return total


def _to_roman(num: int) -> str | None:
    if not 0 < num < 40:  # I..XXXIX covers the [IVX] alphabet
        return None
    out = []
    for value, sym in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while num >= value:
            out.append(sym)
            num -= value
    return "".join(out)


def parse_alpha_ordinal(text: str) -> int | None:
    text = text.strip()
    if len(text) == 1 and text.isalpha() and text.isascii():
        return ord(text.lower()) - ord("a") + 1
    return None


#: Normalized unit ranks (smaller = shallower). Spec: 篇/部/编/卷 > 章 > 节;
#: 条 > 款 > 项; volume/part > chapter > section. EnClause units carry no
#: defined sub-order.
_UNIT_RANKS: dict[str, dict[str, int]] = {
    CN_CHAPTER: {"篇": 0, "部": 0, "编": 0, "卷": 0, "章": 1, "节": 2},
    CN_CLAUSE: {"条": 0, "款": 1, "项": 2},
    EN_CHAPTER: {"volume": 0, "part": 0, "chapter": 1, "section": 2},
}

#: EnClause abbreviation normalization (case-insensitive): Art ≡ Article,
#: Sec ≡ Section; § / ¶ are distinct units from the keyword forms.
_EN_CLAUSE_CANONICAL = {
    "art": "article",
    "article": "article",
    "sec": "section",
    "section": "section",
    "clause": "clause",
    "para": "paragraph",
    "paragraph": "paragraph",
}


def unit_rank(style_key: str, unit: str | None) -> int | None:
    """Sub-order rank of a unit word within its styleKey (None = no order)."""
    if unit is None:
        return None
    return _UNIT_RANKS.get(style_key, {}).get(unit)


@dataclass(frozen=True)
class NumberingClassification:
    """Result of classifying a paragraph's leading numbering."""

    style_key: str
    label_text: str  # the numbering prefix as written
    title_text: str  # remainder after the full pattern match
    unit: str | None = None  # normalized unit word (章/条/chapter/§/…)
    ordinal: int | None = None  # parsed ordinal value, when parseable
    raw_level: int | None = None  # MultiLevelNum: dot count + 1
    top_ordinal: int | None = None  # MultiLevelNum: leading component value

    @property
    def priority(self) -> int:
        return STYLE_KEY_PRIORITY[self.style_key]

    def series_key(self) -> tuple:
        """Grouping key for the "same-series numbering" judgment (§2.2.3).

        Unit-bearing styleKeys require the same normalized unit;
        MultiLevelNum groups by raw level.
        """
        if self.style_key == MULTI_LEVEL_NUM:
            return (self.style_key, self.raw_level)
        if self.style_key in (CN_CHAPTER, EN_CHAPTER, CN_CLAUSE, EN_CLAUSE):
            return (self.style_key, self.unit)
        return (self.style_key,)


def _extract_unit_and_ordinal(
    style_key: str, label: str
) -> tuple[str | None, int | None]:
    if style_key in (CN_CHAPTER, CN_CLAUSE):
        m = re.match(rf"^第\s*([{_CN_ORD}\d]+)\s*(.)$", label.strip())
        if not m:
            return None, None
        return m.group(2), parse_cn_ordinal(m.group(1))
    if style_key == EN_CHAPTER:
        m = re.match(r"^([A-Za-z]+)\s*(.+)$", label.strip())
        if not m:
            return None, None
        unit = m.group(1).lower()
        return unit, _parse_latin_ordinal(m.group(2))
    if style_key == EN_CLAUSE:
        stripped = label.strip()
        sym = re.match(r"^([§¶]+)\s*(.+)$", stripped)
        if sym:
            return sym.group(1)[0], _parse_latin_ordinal(sym.group(2))
        m = re.match(r"^([A-Za-z]+)\.?\s*(.+)$", stripped)
        if not m:
            return None, None
        unit = _EN_CLAUSE_CANONICAL.get(m.group(1).lower(), m.group(1).lower())
        return unit, _parse_latin_ordinal(m.group(2))
    if style_key == CN_NUM:
        return None, parse_cn_ordinal(label)
    if style_key == CN_PARENT_NUM:
        return None, parse_cn_ordinal(re.sub(r"[（()）]", "", label))
    if style_key == ROMAN_NUM:
        return None, parse_roman(label)
    if style_key == EN_NUM:
        return None, int(label) if label.strip().isdigit() else None
    if style_key == EN_ALPHA:
        return None, parse_alpha_ordinal(label)
    if style_key in (EN_DOUBLE_PAREN, EN_SINGLE_PAREN):
        inner = re.sub(r"[（()）]", "", label).strip()
        if inner.isdigit():
            return None, int(inner)
        return None, parse_alpha_ordinal(inner)
    return None, None


def _parse_latin_ordinal(text: str) -> int | None:
    text = text.strip()
    if text.isdigit():
        return int(text)
    roman = parse_roman(text)
    if roman is not None:
        return roman
    return parse_alpha_ordinal(text)


def classify_numbering(text: str) -> NumberingClassification | None:
    """Classify the leading numbering of a paragraph (first hit wins).

    Returns None when no pattern matches OR when the matched styleKey does
    not allow an empty title and nothing follows the numbering — such a
    paragraph "falls back to body", it does NOT retry later patterns.
    """
    if not text:
        return None
    multi_level_shape = _P_MULTI_LEVEL_SHAPE.match(text) is not None
    for style_key, pattern in _MATCH_ORDER:
        if multi_level_shape and style_key != MULTI_LEVEL_NUM:
            # A multi-level opener is claimed by MultiLevelNum exclusively;
            # if its full rule rejected the text, the paragraph is body.
            return None
        m = pattern.match(text)
        if m is None:
            continue
        label = m.group(1)
        title = text[m.end() :].strip()
        if not title and style_key not in ALLOW_EMPTY_TITLE:
            return None
        unit, ordinal = _extract_unit_and_ordinal(style_key, label)
        raw_level = None
        top_ordinal = None
        if style_key == MULTI_LEVEL_NUM:
            digits = re.sub(r"[§\s]", "", label)
            parts = digits.split(".")
            raw_level = len(parts)
            try:
                top_ordinal = int(parts[0])
            except ValueError:
                top_ordinal = None
        return NumberingClassification(
            style_key=style_key,
            label_text=label,
            title_text=title,
            unit=unit,
            ordinal=ordinal,
            raw_level=raw_level,
            top_ordinal=top_ordinal,
        )
    return None


_SINGLE_ROMAN_CHARS = frozenset("IVXivx")


def reclassify_single_char_romans(
    items: Sequence[NumberingClassification | None],
) -> list[NumberingClassification | None]:
    """Deferred second scan (spec §2.2.3): promote ``I.``-style EnAlpha items
    to RomanNum when the same sub-document already contains at least one
    multi-char or Unicode RomanNum companion."""
    has_roman_companion = any(
        item is not None and item.style_key == ROMAN_NUM for item in items
    )
    if not has_roman_companion:
        return list(items)
    out: list[NumberingClassification | None] = []
    for item in items:
        if (
            item is not None
            and item.style_key == EN_ALPHA
            and item.label_text in _SINGLE_ROMAN_CHARS
        ):
            out.append(
                replace(
                    item,
                    style_key=ROMAN_NUM,
                    ordinal=parse_roman(item.label_text),
                )
            )
        else:
            out.append(item)
    return out


# --- FS_base ------------------------------------------------------------------


@dataclass(frozen=True)
class FsBase:
    """Char-weighted dominant body font size and its confidence (§2.2.2)."""

    size_pt: float | None
    dominant_ratio: float  # weight share of the dominant size, 0.0-1.0
    confidence_high: bool  # dominant_ratio >= confidence threshold (CB5)


def compute_fs_base(
    weighted_sizes: Iterable[tuple[float, int]],
    *,
    confidence_ratio: float = 0.60,
) -> FsBase:
    """Compute FS_base from ``(size_pt, char_weight)`` pairs.

    The dominant size is the one covering the most characters; ties break
    toward the LARGER size. Confidence is "high" when the dominant size
    covers at least ``confidence_ratio`` of all weighted characters.
    """
    weights: dict[float, int] = {}
    total = 0
    for size_pt, chars in weighted_sizes:
        if size_pt is None or chars <= 0:
            continue
        weights[size_pt] = weights.get(size_pt, 0) + chars
        total += chars
    if not weights or total <= 0:
        return FsBase(size_pt=None, dominant_ratio=0.0, confidence_high=False)
    size, weight = max(weights.items(), key=lambda kv: (kv[1], kv[0]))
    ratio = weight / total
    return FsBase(
        size_pt=size,
        dominant_ratio=ratio,
        confidence_high=ratio >= confidence_ratio,
    )
