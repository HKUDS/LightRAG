"""Defensive judgments for smart heading discovery (spec §2.2.5 / §2.3.4).

Pure-function layer: consumed by ``title_block`` / ``heading_flow`` /
``extract``, depends only on :mod:`.nlp`, :mod:`.style_key`, and
:mod:`.features` (the last for the TOC third channel's font-size termination) —
never the reverse. Every rule here votes AGAINST heading-ness (do-no-harm: ties
break toward "not a heading").

Env-tunable knobs follow the live-read pattern of the CHUNK_P_REFERENCES_*
constants: DEFAULT_* lives in ``lightrag.constants``; the matching env var
(same name minus DEFAULT_) is read at call time.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lightrag.constants import (
    DEFAULT_DOCX_SMART_CAPTION_PREFIXES,
    DEFAULT_DOCX_SMART_ENNUM_BLACKLIST,
    DEFAULT_DOCX_SMART_HEADING_MAX_CHARS,
    DEFAULT_DOCX_SMART_IMPRINT_CLOSER_PREFIXES,
    DEFAULT_DOCX_SMART_IMPRINT_CLOSER_TRAILING,
    DEFAULT_DOCX_SMART_IMPRINT_COLON_PREFIXES,
    DEFAULT_DOCX_SMART_TOC_MIN_LINES,
    MAX_HEADING_LENGTH,
)

from . import nlp
from .features import effective_font_size_pt
from .style_key import (
    EN_NUM,
    MULTI_LEVEL_NUM,
    NumberingClassification,
    classify_numbering,
)

# Sentence-terminal characters (P4): CJK terminators plus both semicolons.
# An English period is NOT here — it may mark an abbreviation and goes
# through the spaCy re-segmentation check instead.
SENTENCE_END_CHARS = frozenset("。？！…；;")
# Closing wrappers to step over before the terminal check (P4): a sentence
# may legitimately end inside quotes/brackets — 他说："明天见。"
_CLOSING_WRAPPERS = frozenset("\"'”’」』〉》】）)]}")
# Internal sentence-boundary evidence. Unlike ``SENTENCE_END_CHARS`` this
# includes ASCII ``.!?`` because the boundary is only accepted when more
# semantic text follows; spaCy has already supplied the multi-sentence vote.
_INTERNAL_SENTENCE_END_CHARS = SENTENCE_END_CHARS | frozenset(".!?")


def _env_items(env_name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(env_name) or default
    return tuple(
        item.strip() for item in re.split(r"[|,，]", raw) if item and item.strip()
    )


def _env_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, "") or default)
    except ValueError:
        return default


def weighted_char_length(text: str) -> int:
    """English-equivalent length: 1 CJK char counts as 3 (spec §2.2.4)."""
    total = 0
    for ch in text:
        total += 3 if "一" <= ch <= "鿿" else 1
    return total


def heading_max_chars() -> int:
    """Strong-body / heading weighted length cap (en-equivalent), env-overridable
    via ``DOCX_SMART_HEADING_MAX_CHARS``. Single source for
    :func:`strong_body_reason` and :func:`truncate_to_heading_length`.

    ``cap < 3`` (too small to even hold a ``...`` ellipsis) falls back to the
    default — a runtime BACKSTOP. The loud check is :func:`validate_heading_max_chars_env`
    (raises on the same ``< 3`` boundary), run at every smart_heading entry point
    (API startup AND ``run_smart_heading``); this floor only matters for callers
    that bypass both (tests / library embedding)."""
    cap = _env_int("DOCX_SMART_HEADING_MAX_CHARS", DEFAULT_DOCX_SMART_HEADING_MAX_CHARS)
    return cap if cap >= 3 else DEFAULT_DOCX_SMART_HEADING_MAX_CHARS


def validate_heading_max_chars_env() -> int | None:
    """Structurally validate ``DOCX_SMART_HEADING_MAX_CHARS`` and return its
    parsed value (``None`` when unset).

    Raises ``ValueError`` on a non-integer or a value ``< 3`` (too small to even
    hold the ``...`` truncation marker — the same boundary :func:`heading_max_chars`
    floors at runtime). This is the SINGLE source shared by every smart_heading
    entry point — the API startup check (``routing._validate_smart_heading_max_chars``,
    which re-raises as a config error and adds a startup-only low-value warning)
    and the parse-time entry (``run_smart_heading``, where it hard-fails the
    parse) — so a bad cap is rejected identically regardless of how smart_heading
    was enabled. ``heading_max_chars`` keeps its floor only as a last-resort
    backstop for callers that bypass both.
    """
    raw = os.getenv("DOCX_SMART_HEADING_MAX_CHARS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"Invalid DOCX_SMART_HEADING_MAX_CHARS value {raw!r}; expected a "
            f"positive integer (en-equivalent chars, default "
            f"{DEFAULT_DOCX_SMART_HEADING_MAX_CHARS})"
        ) from None
    if value < 3:
        raise ValueError(
            f"DOCX_SMART_HEADING_MAX_CHARS={value} is too small; the minimum is 3 "
            f"(the width of the '...' truncation marker). Use the default "
            f"{DEFAULT_DOCX_SMART_HEADING_MAX_CHARS} unless you have a specific reason."
        )
    return value


def truncate_to_heading_length(text: str) -> str:
    """Bound a synthesized heading to BOTH the strong-body weighted cap
    (:func:`heading_max_chars`) AND the hard raw ceiling ``MAX_HEADING_LENGTH``,
    appending ``...`` when either is exceeded. The raw ceiling keeps
    ``parse_document.truncate_heading`` a no-op so the four landing sites of a
    title block (H1 / doc_title / first_heading / every descendant's
    parent_headings root) never split, even when the env cap is set above 200."""
    cap = heading_max_chars()
    if weighted_char_length(text) <= cap and len(text) <= MAX_HEADING_LENGTH:
        return text
    w_budget = max(cap - 3, 0)  # reserve the "..." (weight 3)
    r_budget = MAX_HEADING_LENGTH - 3
    used_w = used_r = 0
    out: list[str] = []
    for ch in text:
        w = 3 if "一" <= ch <= "鿿" else 1
        if used_w + w > w_budget or used_r + 1 > r_budget:
            break
        out.append(ch)
        used_w += w
        used_r += 1
    # rstrip so a cut landing right after the "  " separator does not leave
    # "词  ..." with a dangling double space.
    return "".join(out).rstrip() + "..."


# ---------------------------------------------------------------------------
# Strong-body features (spec §2.2.5 step 1; hard demotion evidence)
# ---------------------------------------------------------------------------


def _strip_leading_numbering(text: str) -> str:
    """Text with a leading numbering label removed ("3.1.1 X" → "X").

    A numbering label is structural, not prose. Fed to spaCy its dots
    mis-segment a numbered heading into pseudo-sentences ("3.1.1 桌面…" →
    "3.1." | "1 桌面…"), so a legitimate numbered heading falsely reads as ≥2
    sentences. Reuses the numbering machinery (:func:`classify_numbering`) as
    the single source of what the label is; a no-op when there is none.
    """
    cls = classify_numbering(text)
    if cls is None or not cls.label_text:
        return text
    head = text.lstrip()
    if head.startswith(cls.label_text):
        # Drop the label plus its trailing separator run (". 、:）" etc.).
        # \s (not a literal-space whitelist) so NBSP/full-width spaces go too —
        # a leading \xa0 survives into the prose otherwise, and zh spaCy counts
        # it as its own "sentence", falsely demoting the heading (test8 第二章).
        return re.sub(r"^[\s、.。:：)）]+", "", head[len(cls.label_text) :])
    return text


def has_explicit_internal_sentence_boundary(text: str) -> bool:
    """Whether title prose contains a visible sentence boundary before more text.

    This is deterministic corroboration for spaCy's ``>=2`` sentence vote,
    not a sentence splitter of its own. Leading structural numbering is
    removed first, decimal dots are ignored, and a terminator only counts
    when later alphanumeric/CJK content exists. Quotes, brackets, whitespace,
    and punctuation may sit between the terminator and that later content.
    """
    prose = _strip_leading_numbering(text.strip())
    for pos, char in enumerate(prose):
        if char not in _INTERNAL_SENTENCE_END_CHARS:
            continue
        if (
            char == "."
            and pos > 0
            and pos + 1 < len(prose)
            and prose[pos - 1].isdigit()
            and prose[pos + 1].isdigit()
        ):
            continue
        if any(ch.isalnum() for ch in prose[pos + 1 :]):
            return True
    return False


def strong_body_reason(text: str) -> str | None:
    """Return the rule id when ``text`` carries a strong body feature.

    Rules (any one suffices):
      - ``imprint_marker``: opens with a 公文版记 (imprint) ANCHOR — a colon
        prefix (抄送：/ 主题词：) — checked first, before any length / spaCy
        rule, so an anchor line keeps its imprint identity however long it
        runs. (The 印发-family CLOSER is region-scoped and deliberately NOT
        part of strong_body; see :func:`imprint_closer_reason`.);
      - ``strong_body_length``: weighted length > 180 en-equivalent chars;
      - ``strong_body_multi_sentence``: spaCy sees ≥2 sentences AND a visible
        internal sentence terminator corroborates the split — the spaCy vote
        alone is not trusted (the pinned zh model hallucinates mid-word breaks
        on short title fragments: 索菲|亚, 承诺|书);
      - ``strong_body_sentence_end``: ends with a sentence terminator
        (stepping over closing quotes/brackets); a trailing English period
        counts only when spaCy segmentation says it closes a sentence.

    Returns None when no rule fires (the paragraph MAY be a heading).
    """
    reason = imprint_marker_reason(text)
    if reason is not None:
        return reason
    stripped = text.strip()
    if not stripped:
        return None
    if weighted_char_length(stripped) > heading_max_chars():
        return "strong_body_length"

    tail = stripped
    while tail and tail[-1] in _CLOSING_WRAPPERS:
        tail = tail[:-1].rstrip()
    if tail and tail[-1] in SENTENCE_END_CHARS:
        return "strong_body_sentence_end"
    if tail.endswith(".") and nlp.ends_with_sentence_period(tail):
        return "strong_body_sentence_end"

    # Judge sentence shape over the title PROSE, not a leading numbering label:
    # spaCy splits multi-level numbers ("3.1.1") into pseudo-sentences, which
    # would falsely demote every dotted-numbered heading to body.
    #
    # The spaCy ≥2 vote is NECESSARY but not SUFFICIENT: the pinned zh model's
    # dependency parser hallucinates sentence boundaries mid-word on short
    # heading fragments (广州市增城区“7.19”索菲亚… splits at 索菲|亚, 承诺书 at
    # 承诺|书), and a bigger model just relocates the errors (zh_core_web_lg is
    # no better than _sm). So a DETERMINISTIC corroborator CONSTRAINS (does not
    # replace) the model vote: still require spaCy's ≥2, but trust it only when
    # a visible internal sentence terminator (。！？…；;.!?, decimals excluded)
    # corroborates the split — a mid-word pseudo-split never produces one.
    # Known tradeoff: punctuation-less multi-clause prose (e.g. a lead-in ending
    # in “：”) also loses the verdict; accepted as body-vs-heading rarely hinges
    # on it and length / sentence-end rules still cover genuine long prose.
    prose = _strip_leading_numbering(stripped)
    if (
        prose
        and nlp.sentence_count(prose) >= 2
        and has_explicit_internal_sentence_boundary(stripped)
    ):
        return "strong_body_multi_sentence"
    return None


# ---------------------------------------------------------------------------
# Numbering homophone vetoes (NER + P1) — revoke NUMBERING identity only;
# the paragraph's size/bold/outline paths stay open.
# ---------------------------------------------------------------------------

# A version-number shape "3.14 版" — a MultiLevelNum opener whose trailing
# unit word is 版. The negative lookahead keeps 公文 headings "5.2 版面" /
# "7.2 版头" / "7.4 版记" OUT of the veto: those have a CJK character right
# after 版 (版面/版头/版记), so 版 is the head of a real word, not a bare
# version-unit token. "3.14 版" (end of line) and "3.14版 v2" still veto.
_VERSION_SHAPE_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s*版(?![一-龥])")

#: MLN NER escape ceiling: a MultiLevelNum whose leading component is <= this
#: is a plausible section number (chapters rarely exceed 99); a larger leading
#: component (2026.3.5) is almost certainly a real date and keeps its veto.
_MLN_NER_QUANTITY_ESCAPE_MAX_TOP = 99

#: An EnNum ordinal terminated by a dot / 、 ("4." / "12." / "2026." / "4、").
#: classify_numbering gives MultiLevelNum priority over "N.N", so an EnNum whose
#: number is followed by a dot always has a NON-digit after it — a list-ordinal
#: shape, never a date/time (which use 年/月/日, "-"/"/"/":"). The discriminator
#: is the DOT, not the digit count.
_EN_NUM_DOT_ORDINAL_RE = re.compile(r"^\s*\d+[.、]")


def numbering_homophone_reason(
    classification: NumberingClassification, text: str
) -> str | None:
    """Why this "numbering" is actually a date/amount/version/unit phrase.

    - NER veto: the paragraph opens with a DATE/TIME/MONEY/PERCENT/QUANTITY
      entity (2026年3月5日…, $100…) — spec §2.2.5;
    - version-token veto: the token after a leading number is 版;
    - P1 blacklist (NER blind-spot backstop): an EnNum digit run directly
      followed by a blacklisted CJK unit/measure word (2026年 / 1个 / 1条);
      MultiLevelNum version shapes (3.14 版).

    Only digit-led styleKeys are exposed to this veto — 一、/（一）forms
    with their mandatory separators are already unambiguous.
    """
    if classification.style_key not in (EN_NUM, MULTI_LEVEL_NUM):
        return None

    if classification.style_key == MULTI_LEVEL_NUM:
        if _VERSION_SHAPE_RE.match(text):
            return "homophone_version_shape"
    else:
        lstripped = text.lstrip()
        m = re.match(r"^\d+", lstripped)
        if m:
            after = lstripped[m.end() :]
            blacklist = _env_items(
                "DOCX_SMART_ENNUM_BLACKLIST", DEFAULT_DOCX_SMART_ENNUM_BLACKLIST
            )
            # startswith (not a single-char membership test) so a user-added
            # multi-char unit ("小时" / "公斤" / "平方米") matches too, not only
            # the single-char defaults.
            if after and any(after.startswith(item) for item in blacklist):
                return "homophone_unit_blacklist"

    label = nlp.leading_entity_label(text)
    if label in nlp.HOMOPHONE_ENTITY_LABELS:
        # spaCy mislabels EnNum dot-ordinals ("4. 制定实施方案"→DATE,
        # "1. 建立…制度"→PERCENT) as homophone number-phrases, revoking a real
        # list heading's numbering identity. That shape — number + dot +
        # NON-digit — is structurally never a date/time/money/percent/quantity
        # (MultiLevelNum already claimed "N.N"), so trust the structure over ANY
        # NER label. Genuine homophones (2026年…, 100元…) are caught upstream by
        # the unit blacklist and never reach here; real "%"/"$" phrases aren't
        # EnNum dot-ordinals. MultiLevelNum keeps the NER veto.
        if (
            classification.style_key == EN_NUM
            and _EN_NUM_DOT_ORDINAL_RE.match(text) is not None
        ):
            return None
        # A MultiLevelNum with a small leading component ("7.2.1 份号",
        # "7.2 版头") that spaCy tags QUANTITY is a real section number, not a
        # measure: 版/份/号 read as quantifier units in isolation, so the NER
        # sees "7.2 <unit>". Trust the numbering structure and let the size /
        # bold / series channels judge it. DATE/TIME/MONEY/PERCENT keep their
        # veto — a two-digit-year date ("12.31.25 项目日期") or a point-time
        # ("12.30.45 会议纪要") also classifies as MultiLevelNum top<=99 and is
        # structurally indistinguishable from a section number, so only the
        # QUANTITY label (the one that empirically misfires on 公文 headings)
        # is exempted.
        if (
            classification.style_key == MULTI_LEVEL_NUM
            and label == "QUANTITY"
            and classification.top_ordinal is not None
            and classification.top_ordinal <= _MLN_NER_QUANTITY_ESCAPE_MAX_TOP
        ):
            return None
        return "homophone_ner_entity"
    token_after = nlp.token_following_leading_number(text)
    if token_after == "版":
        return "homophone_version_token"
    return None


# ---------------------------------------------------------------------------
# P3 caption-prefix blacklist
# ---------------------------------------------------------------------------


def caption_prefix_reason(text: str) -> str | None:
    """Figure/table/equation captions are never headings (P3).

    Requires prefix + a numbering shape (图1 / 表 2-1 / Figure 3 / Fig. 4(a))
    so a heading that merely starts with the word (图书管理系统) survives.
    """
    stripped = text.lstrip()
    lowered = stripped.lower()
    for prefix in _env_items(
        "DOCX_SMART_CAPTION_PREFIXES", DEFAULT_DOCX_SMART_CAPTION_PREFIXES
    ):
        # Case-insensitive so手工 lowercased English captions ("figure 3",
        # "table 2-1") are vetoed like their capitalized forms; CJK prefixes
        # (图 / 表 / 公式) are unaffected by casefolding.
        if not lowered.startswith(prefix.lower()):
            continue
        after = stripped[len(prefix) :].lstrip()
        if re.match(r"^[0-9０-９一二三四五六七八九十]+([-.–][0-9]+)*", after):
            return "caption_prefix"
    return None


# ---------------------------------------------------------------------------
# 公文版记 (imprint) lines — metadata, never headings. An ANCHOR (抄送 / 主题词)
# opens a 版记 region and vetoes title-block membership for itself + its 2
# preceding non-blank paragraphs; a CLOSER (印发-family, incl. 印发机关) ends the
# region but is only recognized region-side, in an anchor's forward window. See
# title_block.detect_imprint_regions.
# ---------------------------------------------------------------------------

#: Whitespace allowed to interleave/pad a prefix: justified official-document
#: typesetting stretches a label with plain/ideographic spaces (抄　送：).
_IMPRINT_GAP = r"[ \t　]*"


def imprint_marker_reason(text: str) -> str | None:
    """A 公文版记 (imprint) ANCHOR line — 抄送：… / 主题词：… — is metadata.

    A single colon class (env-tunable, comma/pipe-separated): the prefix
    followed by a full/half-width colon. The issuing-organ / print line that
    ENDS a 版记 (印发-family, incl. 印发机关) is a CLOSER, not an anchor — see
    :func:`imprint_closer_reason`.

    Operates on the RAW text (leading whitespace ignored, the rest kept); a
    bare label with no colon after it does not match.
    """
    head = text.lstrip()
    if not head:
        return None
    for prefix in _env_items(
        "DOCX_SMART_IMPRINT_COLON_PREFIXES",
        DEFAULT_DOCX_SMART_IMPRINT_COLON_PREFIXES,
    ):
        # Per-char escape: prefixes are env-configurable, so a user item
        # carrying regex metachars must still match literally.
        body = _IMPRINT_GAP.join(re.escape(ch) for ch in prefix)
        if re.match(body + _IMPRINT_GAP + "[：:]", head):
            return "imprint_marker"
    return None


def imprint_closer_reason(text: str) -> str | None:
    """A 版记 region CLOSER line — 印发-family — is metadata.

    Deliberately NOT a standalone per-line rule and NOT part of
    :func:`strong_body_reason`: a line-final ``印发`` occurs in body prose
    (…已印发) too, so callers invoke this ONLY inside the forward window of an
    imprint anchor (:func:`title_block.detect_imprint_regions`). That scoping
    is exactly how the "印发 must follow 抄送/主题词" constraint is enforced.

    Two forms (env-tunable, comma/pipe-separated):
      - prefix class (印发 / 印发机关): the line OPENS with the marker + a colon
        or any whitespace (印发：XX办公厅 / 印发 XX办公厅 / 印发机关　XX办公厅). ``\\s``
        includes a soft line break and the ideographic space, so a two-line
        ``印发机关\\n机构名`` still hits;
      - trailing class (印发): the line ENDS with the marker (某某办公室
        2026年6月30日 印发) — the GB/T layout with the issuer/date first. A
        trailing period stops the match (已印发。 does not close a region).

    Operates on the RAW text: the leading/trailing whitespace is evidence.
    """
    head = text.lstrip()
    if not head:
        return None
    for prefix in _env_items(
        "DOCX_SMART_IMPRINT_CLOSER_PREFIXES",
        DEFAULT_DOCX_SMART_IMPRINT_CLOSER_PREFIXES,
    ):
        body = _IMPRINT_GAP.join(re.escape(ch) for ch in prefix)
        # Prefix marker followed by a (optionally gap-padded) colon OR any
        # whitespace — mirrors the anchor tails but for the 印发 opener.
        if re.match(body + r"(?:" + _IMPRINT_GAP + r"[：:]|\s)", head):
            return "imprint_closer"
    tail = text.rstrip()
    for marker in _env_items(
        "DOCX_SMART_IMPRINT_CLOSER_TRAILING",
        DEFAULT_DOCX_SMART_IMPRINT_CLOSER_TRAILING,
    ):
        body = _IMPRINT_GAP.join(re.escape(ch) for ch in marker)
        m = re.search(body + r"$", tail)
        # The GB/T trailing form carries the issuer/date BEFORE 印发, so a bare
        # label (印发 alone) must not fire — require something to precede it.
        if m is not None and m.start() > 0:
            return "imprint_closer"
    return None


#: A whole-line 成文日期 (document date): CJK-numeral (二○○九年七月六日),
#: Arabic (2009年7月6日), or separator-style (2026.7.31 / 2026/7/31 /
#: 2026-7-31; the separator must be consistent within one date). Zero is
#: written 〇 / ○ (circle) / 零 in practice.
_CN_DATE_DIGIT = "〇○零一二三四五六七八九十两廿"
_DOCUMENT_DATE_CN = re.compile(
    rf"^[{_CN_DATE_DIGIT}]{{2,4}}年[{_CN_DATE_DIGIT}]{{1,3}}月[{_CN_DATE_DIGIT}]{{1,3}}日$"
)
_DOCUMENT_DATE_AR = re.compile(r"^\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日$")
_DOCUMENT_DATE_SEP = re.compile(r"^\d{4}([./\-])\d{1,2}\1\d{1,2}$")


def is_document_date(text: str) -> bool:
    """True when the WHOLE line is a 成文日期 (a bare document date), e.g.
    ``二○○九年七月六日``, ``2009年7月6日``, ``2026.7.31``, ``2026/7/31`` or
    ``2026-7-31``.

    A line that merely CONTAINS a date (第十六条…自2009年7月1日起施行。) is NOT a
    document date — the ``^…$`` anchors demand the date be the entire line.
    Two consumers: pulling a mis-ordered 成文日期 trailing a 版记 into the
    region (:func:`title_block.detect_imprint_regions`), and excluding a bare
    date line from the solo centered-heading channel (§2.2.5 gate) — a
    centered 落款 date must never read as a heading.
    """
    s = text.strip()
    return bool(
        _DOCUMENT_DATE_CN.match(s)
        or _DOCUMENT_DATE_AR.match(s)
        or _DOCUMENT_DATE_SEP.match(s)
    )


def is_symbolic_line(text: str) -> bool:
    """True when the line carries no letter at all — page numbers (``- 1 -``),
    separators (``***`` / ``——``), bare figures. Positive detection: look for
    a letter (``str.isalpha`` covers CJK ideographs and Latin alike) instead
    of stripping a punctuation set, so full-width symbols can never slip
    through an incomplete list. Used by the solo centered-heading channel
    (§2.2.5 gate) — a centered decoration line must never read as a heading.
    """
    return not any(ch.isalpha() for ch in text)


# ---------------------------------------------------------------------------
# Landing guardrails: canonicalization, I1/I2/I3 machine checks, TOC (§2.3)
# ---------------------------------------------------------------------------

_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")
# Dot leader: ≥3 ASCII/middle dots, or any run holding an ellipsis char
# (one … renders as three dots; Chinese TOCs typically use ……).
_TOC_DOT_LEADER_RE = re.compile(r"(?:…[.…·]*|[.·]{3,})\s*\d+\s*$")

# --- TOC third channel: a plain (leaderless) numbered run under a 目次 title ---
# A standalone 目次 / 目录 / (Table of) Contents line anchors the channel.
_TOC_TITLE_RE = re.compile(
    r"^\s*(?:目\s*次|目\s*录|(?:table\s+of\s+)?contents)\s*$", re.IGNORECASE
)
# Trailing page number after ≥1 space/tab/ideographic-space, stripped before the
# duplicate-line comparison (a TOC entry "1 范围　3" matches body "1 范围").
_TOC_TRAILING_PAGENO_RE = re.compile(r"[ \t　]+\d{1,4}\s*$")
# Anchor front-part limit: a 目次 title is only accepted while fewer than this
# many long (body) paragraphs have been seen — a mid/late-document "Contents"
# heading must not anchor the channel.
_TOC_ANCHOR_MAX_LONG_PARAS_BEFORE = 3
# The TOC's max font size is established from this many leading members; a later
# member strictly larger than that ends the run (the body's first heading).
_TOC_SIZE_PROBE_MEMBERS = 3
# Placeholder marking the elided tail of a retained TOC (§2.3 TOC retention).
# Shared by the assembler (emits it as a body line) and I1 (subtracts it from
# the output multiset): a single source-of-truth prevents the two from drifting
# apart, which would silently defeat the I1 subtraction.
TOC_ELLIPSIS = "……"


def canonicalize_paragraph_text(text: str) -> str:
    """Whitespace-free, markdown-prefix-free canonical form (I1 comparisons).

    ALL whitespace goes (the sidecar-backfill lesson: collapsing to single
    spaces broke zero-width CJK splits), and a leading markdown ``#`` run is
    stripped so rendered heading lines compare equal to their source
    paragraphs. Placeholder tokens are left as-is — table/equation/drawing
    payloads are covered by their own asset integrity checks, not I1.
    """
    stripped = _HEADING_PREFIX_RE.sub("", (text or "").strip())
    return re.sub(r"\s+", "", stripped)


def _visible_text_lines(text: str | None) -> list[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


def _visible_lines(rec: Any) -> list[str]:
    return _visible_text_lines(rec.text)


def _detect_plain_numbered_toc(
    records: Sequence[Any], min_lines: int
) -> tuple[set[int], list[dict]]:
    """Third TOC channel: a leaderless run of short numbered lines under a
    standalone 目次 / 目录 / (Table of) Contents title, confirmed by every line
    having a duplicate later in the body (a GB/T-style TOC typed as plain text
    with no dot leaders). See the four safety constraints below — this channel
    deletes real content (``toc_indices`` is the I1 whitelist), so it errs
    toward false negatives (a short/unusual TOC is left as candidates, which
    CB1 handles) rather than deleting body.
    """
    # Line-level inverted index: canon(line).casefold() -> record indices where
    # it appears. Line-level (not paragraph-level) so soft-broken TOCs match,
    # and casefold so an English TOC entry matches a differently-cased heading.
    canon_at: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        if rec.kind != "para":
            continue
        for ln in _visible_lines(rec):
            key = canonicalize_paragraph_text(ln).casefold()
            if key:
                canon_at.setdefault(key, []).append(i)

    def _copy_after(line: str, after: int) -> bool:
        for variant in (line, _TOC_TRAILING_PAGENO_RE.sub("", line)):
            key = canonicalize_paragraph_text(variant).casefold()
            if key and any(idx > after for idx in canon_at.get(key, ())):
                return True
        return False

    cap = heading_max_chars()
    toc: set[int] = set()
    events: list[dict] = []
    long_paras_seen = 0
    n = len(records)
    i = 0
    while i < n:
        rec = records[i]
        if rec.kind == "para":
            lines = _visible_lines(rec)
            is_long = any(weighted_char_length(ln) > cap for ln in lines)
            if is_long:
                long_paras_seen += 1
            is_anchor = (
                len(lines) == 1
                and _TOC_TITLE_RE.match(lines[0]) is not None
                and long_paras_seen < _TOC_ANCHOR_MAX_LONG_PARAS_BEFORE
            )
            if not is_anchor:
                i += 1
                continue

            # Collect the maximal run of short, forward-duplicated members after
            # the anchor. Stop at the first paragraph that breaks any rule —
            # incrementally, so the body's first heading (whose only copy is the
            # TOC entry ABOVE it) ends the run instead of being swallowed.
            block: list[int] = []
            block_lines = 0
            probe_sizes: list[float] = []
            toc_max_size: float | None = None
            j = i + 1
            while j < n:
                r = records[j]
                if r.kind == "empty_para":
                    j += 1
                    continue
                if r.kind != "para":
                    break
                mlines = _visible_lines(r)
                if not mlines or any(weighted_char_length(ln) > cap for ln in mlines):
                    break
                # Font-size termination: a member strictly larger than the TOC's
                # established max size is the body's first heading. The max is
                # locked from the first probe MEMBERS' known sizes; if all of
                # those are None, size termination stays disabled (other rules
                # gate).
                size = effective_font_size_pt(r)
                if (
                    toc_max_size is not None
                    and size is not None
                    and size > toc_max_size
                ):
                    break
                if not all(_copy_after(ln, j) for ln in mlines):
                    break
                block.append(j)
                block_lines += len(mlines)
                if toc_max_size is None and len(block) <= _TOC_SIZE_PROBE_MEMBERS:
                    if size is not None:
                        probe_sizes.append(size)
                    if len(block) == _TOC_SIZE_PROBE_MEMBERS:
                        toc_max_size = max(probe_sizes) if probe_sizes else None
                j += 1

            # all-or-nothing: every member line must have a copy strictly after
            # the whole block, and the block must clear the line threshold.
            end = block[-1] if block else i
            confirmed = block_lines >= min_lines and all(
                _copy_after(ln, end) for k in block for ln in _visible_lines(records[k])
            )
            if confirmed:
                toc.update(block)
                events.append(
                    {
                        "rule": "toc_plain_numbered_run",
                        "anchor": i,
                        "start": block[0],
                        "end": end,
                    }
                )
                i = end + 1
                continue
        i += 1
    return toc, events


def detect_toc_records(records: Sequence[Any]) -> tuple[set[int], list[dict]]:
    """Three-channel TOC detection (§2.2.2). Returns ``(indices, events)`` — the
    record indices to remove from smart output, and third-channel audit events.

    1. Structural: TOC field instructions / _Toc bookmark links on a paragraph.
    2. Dot-leader run: ≥ DOCX_SMART_TOC_MIN_LINES consecutive leader LINES
       ending in a dot leader + page number. Lines are counted, not paragraphs:
       the run spans standalone paragraphs AND the soft-break lines inside one
       paragraph (§2.2.2 "含独立段落或段内软回车拆出的行"), so a TOC typed as a
       single multi-line paragraph is caught too. An isolated single line is
       never evidence.
    3. Plain numbered run: a leaderless run of short numbered lines under a
       目次 / 目录 / (Table of) Contents title, every line duplicated later in
       the body (see :func:`_detect_plain_numbered_toc`).

    Detection only — the ``smart_toc_removed_paragraphs`` warning is a
    content claim, so the caller records it once the smart output (which
    actually drops these records) is accepted, not here.
    """
    min_lines = _env_int("DOCX_SMART_TOC_MIN_LINES", DEFAULT_DOCX_SMART_TOC_MIN_LINES)
    toc: set[int] = set()
    for i, rec in enumerate(records):
        if rec.kind == "para" and (rec.is_toc_field or rec.is_toc_link):
            toc.add(i)

    run: list[int] = []
    run_lines = 0

    def _flush_run() -> None:
        nonlocal run, run_lines
        if run_lines >= min_lines:
            toc.update(run)
        run = []
        run_lines = 0

    for i, rec in enumerate(records):
        if rec.kind == "empty_para":
            continue  # blank lines do not break a leader run
        if rec.kind == "para":
            lines = [ln.strip() for ln in rec.text.split("\n") if ln.strip()]
            # A paragraph joins the run only if EVERY visible line is a dot
            # leader; its soft-break lines each count toward the line total.
            if lines and all(_TOC_DOT_LEADER_RE.search(ln) for ln in lines):
                run.append(i)
                run_lines += len(lines)
                continue
        _flush_run()
    _flush_run()

    plain, events = _detect_plain_numbered_toc(records, min_lines)
    toc.update(plain)
    return toc, events


@dataclass(frozen=True)
class TocOutputPlan:
    """How a detected TOC is rendered into body output (§2.3 TOC retention).

    ``toc_indices`` stays the full heading-pipeline exclusion set; this plan
    only decides which of those excluded lines are nonetheless re-emitted as
    body, so a 目录 heading keeps a few of its entries instead of being
    orphaned. Budget is counted in VISIBLE LINES (soft-break lines count
    individually), matching DOCX_SMART_TOC_KEEP_LINES.
    """

    kept_text: dict[
        int, str
    ]  # record index -> body text (its first kept visible lines)
    ellipsis_anchor: (
        int | None
    )  # emit one TOC_ELLIPSIS body line after this record; None = nothing elided
    kept_lines: int  # visible lines retained as body
    removed_lines: int  # visible lines elided (straddle tails + demoted_body_text)
    fully_removed_records: int  # records with visible text and NOTHING kept


def plan_toc_output(
    records: Sequence[Any], toc_indices: set[int], *, keep_lines: int
) -> TocOutputPlan:
    """Decide which leading TOC lines to keep as body and where the single
    ``TOC_ELLIPSIS`` marker goes; count the line-level keep/drop.

    Pure over ``(records, toc_indices, keep_lines)`` — the env read lives in
    the caller (``run_smart_heading``), so this stays trivially unit-testable
    with an explicit budget. Records absent from ``kept_text`` are dropped
    whole; a straddling record (one crossing the budget boundary) keeps its
    first lines and counts the remainder in ``removed_lines``. Invisible
    (whitespace-only) TOC records are skipped entirely — neither kept nor
    counted nor eligible as the ellipsis anchor. A record's
    ``demoted_body_text`` tail (oversize soft-break remainder from the read
    pass) is never re-emitted by the TOC branch, so its visible lines always
    count as removed — an elided tail alone still raises the ellipsis.

    This is the single source for every disposition count the caller reports
    (``fully_removed_records`` included) — keep any new count here rather than
    re-deriving it at a call site, or the visibility definitions drift.
    """
    ordered = [i for i in sorted(toc_indices) if records[i].kind == "para"]
    kept_text: dict[int, str] = {}
    remaining = max(0, keep_lines)
    kept_lines = 0
    removed_lines = 0
    fully_removed = 0
    last_kept: int | None = None
    first_visible: int | None = None
    for i in ordered:
        removed_lines += len(_visible_text_lines(records[i].demoted_body_text))
        lines = _visible_lines(records[i])
        if not lines:
            continue
        if first_visible is None:
            first_visible = i
        if remaining <= 0:
            removed_lines += len(lines)
            fully_removed += 1
            continue
        take = min(len(lines), remaining)
        kept_text[i] = "\n".join(lines[:take])
        kept_lines += take
        removed_lines += len(lines) - take
        last_kept = i
        remaining -= take
    if removed_lines > 0:
        ellipsis_anchor = last_kept if last_kept is not None else first_visible
    else:
        ellipsis_anchor = None
    return TocOutputPlan(
        kept_text, ellipsis_anchor, kept_lines, removed_lines, fully_removed
    )


def toc_audit_entries(records: Sequence[Any], toc_indices: set[int]) -> list[dict]:
    """Audit rows for detected TOC paragraphs (count + text hash, §2.3.5).

    Legacy-named: since TOC retention (§2.3) these indices are the DETECTED
    TOC records, some of which may be re-emitted as body (see
    :func:`plan_toc_output`) — they are no longer all "removed".
    """
    import hashlib

    entries = []
    for i in sorted(toc_indices):
        text = records[i].text or ""
        entries.append(
            {
                "record_index": i,
                "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            }
        )
    return entries


def verify_content_preservation(
    records: Sequence[Any],
    blocks: Sequence[dict],
    *,
    toc_indices: set[int] = frozenset(),
    ignored_output_texts: Sequence[str] = (),
) -> list[str]:
    """I1 multiset check: every pass1-read paragraph's text must appear in
    the union of the output blocks' content∪heading. Returns the canonical
    forms that are MISSING (empty list = pass).

    Exact line-level multiset matching first; leftovers get one merge
    tolerance pass — every leftover OUTPUT line must be exactly the
    concatenation of consecutive leftover source paragraphs in document
    order, each consumed at most once (covers §2.2.7 heading merges).
    One-directional substring containment is not accepted: it would let a
    genuinely lost short paragraph hide inside an unrelated longer line.
    Only the TOC whitelist (indices from the three-channel judgment) may be
    absent entirely.

    ``ignored_output_texts`` are output lines this call injected that do NOT
    come from any record (retained TOC lines + the ``TOC_ELLIPSIS`` marker);
    they are subtracted from the output multiset before matching so an injected
    copy cannot mask a genuinely lost source line (a leaderless TOC entry can
    canonicalize identically to its body heading).

    Source paragraphs are split on ``\n`` before canonicalizing so a body
    paragraph carrying soft breaks (``w:br`` → ``"\n"``, emitted verbatim as
    multiple output content lines) compares symmetrically with the output;
    without the split the whole paragraph would canonicalize to one piece
    that no single output line equals, forcing a spurious fallback. Heading
    merges / soft-break heading joins still resolve through the tolerance
    pass (several source lines → one joined output line).
    """
    source: list[str] = []
    for i, rec in enumerate(records):
        if rec.kind != "para" or i in toc_indices:
            continue
        for piece in (rec.text, rec.demoted_body_text):
            if not piece:
                continue
            for line in piece.split("\n"):
                canon = canonicalize_paragraph_text(line)
                if canon:
                    source.append(canon)

    out_lines: dict[str, int] = {}
    for block in blocks:
        for line in (block.get("content") or "").split("\n"):
            canon = canonicalize_paragraph_text(line)
            if canon:
                out_lines[canon] = out_lines.get(canon, 0) + 1

    # Subtract the lines this call injected (retained TOC body + the ellipsis)
    # at the same \n-split granularity as out_lines, so an injected copy cannot
    # satisfy a source line the assembler actually lost.
    for text in ignored_output_texts:
        for line in (text or "").split("\n"):
            canon = canonicalize_paragraph_text(line)
            if canon and out_lines.get(canon, 0) > 0:
                out_lines[canon] -= 1

    missing: list[str] = []
    leftovers_src: list[str] = []
    for canon in source:
        count = out_lines.get(canon, 0)
        if count > 0:
            out_lines[canon] = count - 1
        else:
            leftovers_src.append(canon)

    if leftovers_src:
        leftover_out: list[str] = []
        for line, count in out_lines.items():
            leftover_out.extend([line] * count)
        used = [False] * len(leftovers_src)
        for line in leftover_out:
            for s in range(len(leftovers_src)):
                if used[s] or not line.startswith(leftovers_src[s]):
                    continue
                acc_len = 0
                picked: list[int] = []
                k = s
                while k < len(leftovers_src) and acc_len < len(line):
                    piece = leftovers_src[k]
                    if used[k] or not line.startswith(piece, acc_len):
                        break
                    picked.append(k)
                    acc_len += len(piece)
                    k += 1
                if acc_len == len(line):
                    for p in picked:
                        used[p] = True
                    break
        missing.extend(
            leftovers_src[s] for s in range(len(leftovers_src)) if not used[s]
        )
    return missing


def verify_baseline_heading_retention(
    records: Sequence[Any],
    decisions: Sequence[Any],
    *,
    demotion_rules: frozenset[str] = frozenset(
        {
            "strong_body_demoted",
            "clamp_gt9_demoted",
            "placeholder_demoted",
            "toc_removed",
            "title_block_member",
            "merged_absorbed",
        }
    ),
) -> list[int]:
    """I2: the smart heading set must cover every baseline (outlineLvl)
    heading except explicit rule-tagged demotions. Returns violating record
    indices (empty = pass). Table-cell paragraphs never become records, so
    that exemption is structural.
    """
    by_record = {d.record_index: d for d in decisions}
    violations: list[int] = []
    for i, rec in enumerate(records):
        if rec.kind != "para" or rec.outline_level is None:
            continue
        d = by_record.get(i)
        if d is None:
            violations.append(i)
            continue
        if d.is_heading or d.is_title_block:
            continue
        if not any(rule in demotion_rules for rule in d.rule_trail):
            violations.append(i)
    return violations


def verify_anchor_semantics(decisions: Sequence[Any]) -> list[int]:
    """I3 (decidable subset): a non-numbered anchored heading that survived
    must sit exactly at outlineLvl+1; and level 0 is reserved for title-block
    roots — a non-title-block heading at level 0 is a construction bug (the
    single-root-per-sub-document invariant, machine-checked as an assertion
    per §2.3.2). Returns violating record indices."""
    violations: list[int] = []
    for d in decisions:
        if (
            d.is_heading
            and not d.is_title_block
            and d.anchored
            and d.numbering is None
            and d.outline_level is not None
            and d.level != d.outline_level + 1
        ):
            violations.append(d.record_index)
        elif d.is_heading and not d.is_title_block and d.level == 0:
            # Only title blocks may root a sub-document at level 0.
            violations.append(d.record_index)
    return violations


def smart_output_length_ok(
    smart_blocks: Sequence[dict],
    baseline_blocks: Sequence[dict],
    *,
    min_ratio: float = 0.30,
) -> bool:
    """§2.3.6: smart content shorter than 30% of baseline → fall back."""
    smart_len = sum(len(b.get("content") or "") for b in smart_blocks)
    base_len = sum(len(b.get("content") or "") for b in baseline_blocks)
    if base_len <= 0:
        return True
    return smart_len >= min_ratio * base_len


def shadow_baseline_diff(
    smart_blocks: Sequence[dict], baseline_blocks: Sequence[dict]
) -> dict:
    """Shadow A/B summary for metrics (§2.3.5): heading-count change, level
    distribution change, and total content char delta."""

    def _levels(blocks: Sequence[dict]) -> dict[int, int]:
        out: dict[int, int] = {}
        for b in blocks:
            lv = int(b.get("level") or 0)
            out[lv] = out.get(lv, 0) + 1
        return out

    smart_chars = sum(len(b.get("content") or "") for b in smart_blocks)
    base_chars = sum(len(b.get("content") or "") for b in baseline_blocks)
    return {
        "heading_count_smart": len(smart_blocks),
        "heading_count_baseline": len(baseline_blocks),
        "level_distribution_smart": _levels(smart_blocks),
        "level_distribution_baseline": _levels(baseline_blocks),
        "content_char_delta": smart_chars - base_chars,
    }
