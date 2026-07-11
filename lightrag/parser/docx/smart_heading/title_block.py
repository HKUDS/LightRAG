"""Title-block (document main title) discovery and LLM judgment (spec §2.2.4).

Heuristics find CANDIDATE windows — runs of paragraphs lacking strong-body
features that contain a visually dominant line — plus a single-paragraph
channel limited to the document's first eligible paragraph. An LLM (via the
synchronous judge callable; this module never touches asyncio) then confirms
and decomposes each candidate.

LLM output is STRICTLY validated: the main/sub title must be locatable in
the window text (concatenation allowed), and paragraphs carrying a physical
outline level are never demoted by an LLM "body" vote (invariant I2). A
non-title verdict's headings/body partition must be well-formed; a MALFORMED
partition (missing/null field, out-of-range index, a duplicate within a list,
or the same index voted both heading and body) raises
:class:`TitleBlockLLMError`, as does any unparseable or non-locatable answer —
those failures are loud, never silently degraded. An UNDER-SPECIFIED partition
(two valid lists whose union merely omits some window indices) is instead
recovered: the unmentioned paragraphs abstain (named in neither list, so
neither audited as headings nor vetoed) and re-enter the normal flow, logged
via ``title_block_partition_incomplete`` so a single local omission never
fails the whole document. Of the two partition lists only the BODY side
carries force (a veto revoking candidate identity); the headings side is
retained for audit/reference and never admits a heading by itself.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
    DEFAULT_DOCX_SMART_IMPRINT_FORWARD_PARAS,
    DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS,
    DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA,
    DEFAULT_DOCX_SMART_TITLE_HEAD_ZONE_RECORDS,
)
from lightrag.utils import logger

from . import guardrails
from ..drawing_image_extractor import DRAWING_TAG_PATTERN
from .features import effective_font_size_pt
from .style_key import classify_numbering

#: Synchronous LLM judge protocol — matches SyncLLMBridge.__call__.
LLMJudge = Callable[..., str]

#: Max weighted chars for a candidate title line (30 CJK / 90 en; §2.2.4).
TITLE_LINE_MAX_WEIGHTED_CHARS = 90

_FLANK_WINDOW = 20  # paragraphs scanned per flank for nearby heading dominance
_SINGLE_CONTEXT = 2  # paragraphs of context around a single candidate


def _cover_semantic_text(rec: Any) -> str:
    """The paragraph text with its ``<drawing.../>`` tags stripped — what the
    cover-material judgments (length / strong-body / imprint / title-line)
    see, AND what the LLM window shows / locate-back matches against. Images
    carry no judgeable text, so they are removed from the prompt entirely
    (least noise, no reliance on the LLM obeying a marker convention); a
    placeholder's own attribute text (id/name/path) must never count toward
    the length cap or feed text rules. Block ASSEMBLY always keeps the
    original ``rec.text`` — the image survives in the output."""
    return DRAWING_TAG_PATTERN.sub("", rec.text or "").strip()


class TitleBlockLLMError(ValueError):
    """The LLM answer was unparseable or failed locate-back validation."""


def _env_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, "") or default)
    except ValueError:
        return default


def _env_float(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, "") or default)
    except ValueError:
        return default


_TIKTOKEN_ENCODER = None


def _estimate_tokens(text: str) -> int:
    """Deterministic token estimate (cl100k_base; tiktoken is a core dep)."""
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        import tiktoken

        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_TIKTOKEN_ENCODER.encode(text))


def _canon(text: str) -> str:
    """Whitespace-free canonical form for locate-back comparisons."""
    return re.sub(r"\s+", "", text or "")


def _is_cjk_char(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def _join_heading_texts(left: str, right: str) -> str:
    """CJK-aware join: no space between CJK boundaries, a space otherwise."""
    if left and right and _is_cjk_char(left[-1]) and _is_cjk_char(right[0]):
        return left + right
    return f"{left} {right}"


def flatten_heading_line(text: str) -> str:
    """Render a heading that carries line breaks as ONE line.

    A heading is a single title no matter how its source was line-wrapped
    (soft ``<w:br>`` breaks, an LLM echoing a multi-paragraph cover title) —
    left multi-line it leaks ``\\n`` into every descendant's
    ``parent_headings``. ``splitlines()`` covers ``\\r\\n``/U+2028-style
    breaks, not just ``\\n``; lines join CJK-aware so Chinese titles gain no
    stray spaces.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    joined = lines[0]
    for ln in lines[1:]:
        joined = _join_heading_texts(joined, ln)
    return joined


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TitleBlockCandidate:
    start: int  # record index range [start, end) covering the window
    end: int
    single: bool
    trigger: str  # audit rule id
    #: §2.2.4 table channel: a cover laid out inside tables; tables render
    #: cell-by-cell for the LLM. ``members`` below carries the member records.
    table: bool = False
    #: Table-channel member records in source order: the qualifying tables
    #: PLUS any absorbed cover-material / image-only paragraphs between or
    #: after them. Filled by the table-run scan; paragraph channels leave it
    #: empty (render then falls back to scanning the range for tables).
    members: tuple[int, ...] = ()


@dataclass(frozen=True)
class TitleBlockDecision:
    candidate: TitleBlockCandidate
    is_title_block: bool
    member_indices: tuple[int, ...] = ()  # record indices forming the block
    main_title: str | None = None
    sub_title: str | None = None
    doc_number: str | None = None
    classification: str | None = None
    publisher: str | None = None
    date: str | None = None
    # Non-title verdicts. ``heading_indices`` is the LLM's headings partition,
    # retained for downstream audit/reference only — NOT an admission rule
    # (the gate re-judges those paragraphs on its normal signals; a vote that
    # matches none is logged as ``llm_grant_rejected``). ``body_indices`` is
    # the veto side, which DOES carry force (candidate identity revoked).
    heading_indices: tuple[int, ...] = ()
    body_indices: tuple[int, ...] = ()
    raw_response: str = field(default="", repr=False)


def compose_title_heading(decision: TitleBlockDecision) -> str:
    """``<主标题> — <副标题> — <发文字号>(<分类>, <出版单位>, <日期>)``."""
    parts = [
        p for p in (decision.main_title, decision.sub_title, decision.doc_number) if p
    ]
    head = " — ".join(parts)
    meta = [
        m for m in (decision.classification, decision.publisher, decision.date) if m
    ]
    if meta:
        head += f"({', '.join(meta)})"
    return head


def _has_real_numbering(
    text: str, numbering_veto: Callable[[Any, str], str | None]
) -> bool:
    """True when the paragraph opens with a GENUINE numbering prefix.

    Shares the §2.2.5 homophone exclusion: a date/amount/version opener
    (2026年度工作报告) has its numbering identity revoked first, so it does
    NOT disqualify a title candidate.
    """
    classification = classify_numbering(text)
    if classification is None:
        return False
    return numbering_veto(classification, text) is None


#: A title line must clear FS_base by at least this many points. BOTH
#: channels require +2pt: the multi-paragraph window used to admit the
#: §2.2.2 strong-signal tier (+1pt), but that deliberately let ordinary
#: section-heading-sized lines open windows (A7/B17). Tightening to +2pt —
#: matching the single-paragraph channel (DOCX_SMART_TITLE_BLOCK_MIN_DELTA) —
#: demands genuine cover-title dominance over body and aligns with the
#: §2.2.4 conservative-preference principle (a title-block false positive is
#: leveraged into regional structural errors, so ties go to "not a title").
MULTI_WINDOW_TITLE_DELTA_PT = 2.0

#: 裸附件标记行（附件 / 附件： / 附件三：）——mid-document 封面窗口的第二类
#: 文档边界证据（公文附件封面）。semantic 整行匹配（drawing tag 已剥），
#: "附件：见下表"式正文绝不匹配。
_ATTACHMENT_OPENER = re.compile(
    r"^附\s*件\s*[一二三四五六七八九十0-9１-９]{0,3}\s*[:：]?$"
)

#: Sentence-terminal punctuation marking a BODY line (head-zone terminator):
#: cover material (titles, issuers, doc numbers, dates) is never punctuated
#: like prose. Deterministic on purpose — spaCy falsely splits decimal-bearing
#: cover titles into "multi-sentence" strong verdicts, so strong_body must
#: not close the zone.
_BODY_SENTENCE_ENDERS = ("。", "！", "？", "；", ".", "!", "?", ";")


@dataclass
class ImprintRegion:
    """A 公文版记 span anchored on an imprint marker (抄送 / 主题词).

    - ``anchor``: the marker record index.
    - ``closer``: the 印发-family line that ends the region, or ``None`` when
      none was found within the forward window (then the region is just the
      anchor — the pre-region fallback).
    - ``members``: anchor→closer span, inclusive, paragraphs only — barred
      from every title-block channel and force-demoted to body when a valid
      title block immediately follows the region.
    - ``preceding``: up to 2 non-blank paragraphs above the anchor — the
      signature/date lines a tail window would otherwise absorb; VETOED from
      title blocks but NEVER demoted (they are the previous document's own
      content, not 版记 metadata).
    """

    anchor: int
    closer: int | None
    members: set[int]
    preceding: set[int]


def _page_boundary_between(prev: Any, cur: Any) -> bool:
    """页/节边界落在 prev 与 cur 之间：title block 是单页单元（§2.2.4）。

    cur 侧看"段前"证据（pageBreakBefore / 行首换页），prev 侧看"段后"证据
    （正文之后的换页 / 段落级 sectPr）。行首换页只算 cur 侧——
    ``has_nonleading_page_break_run`` 才是 prev 侧证据，防止同一个换页符
    既在标题前断窗、又在其副标题前再断一次。table 记录的边界字段默认
    False，证据来自相邻 paragraph 一侧即生效。
    """
    return bool(
        cur.page_break_before
        or cur.has_leading_page_break_run
        or prev.has_nonleading_page_break_run
        or prev.ends_section
    )


def is_content_record(idx: int, rec: Any, skip_indices: set[int]) -> bool:
    """A record that carries document CONTENT: a table, or a paragraph with
    non-empty SEMANTIC text. Blank/whitespace paragraphs, empty tables,
    section breaks, TOC lines, caller-skipped records and pure ``<drawing/>``
    logo/seal paragraphs are all transparent. The head-zone measure, the
    ``imprint_single`` forward walk AND the imprint confirm scan in
    ``heading_flow._demote_confirmed_imprint_regions`` share this single
    predicate — a cover the walk can find must be a cover the confirm scan
    can also reach, or a 版记 region silently stays undemoted."""
    if idx in skip_indices:
        return False
    if rec.kind == "table":
        return True
    return (
        rec.kind == "para"
        and not rec.is_toc_field
        and not rec.is_toc_link
        and bool(_cover_semantic_text(rec))
    )


def _blank_or_skipped(rec: Any, idx: int, skip_indices: set[int]) -> bool:
    """A record the imprint walks step over WITHOUT spending budget: a blank
    paragraph or a TOC line (``skip_indices``). TOC lines never count toward
    the 2-preceding / forward window and never join a region (N2)."""
    return (
        rec.kind == "empty_para"
        or (rec.kind == "para" and not rec.text.strip())
        or idx in skip_indices
    )


def detect_imprint_regions(
    records: Sequence[Any],
    *,
    imprint_marker: Callable[[str], str | None],
    imprint_closer: Callable[[str], str | None],
    document_date: Callable[[str], bool] | None = None,
    skip_indices: set[int] = frozenset(),
) -> list[ImprintRegion]:
    """Map every 公文版记 region: an imprint anchor with its neighbourhood.

    For each anchor (:func:`guardrails.imprint_marker_reason`, e.g. 抄送 /
    主题词; TOC lines cannot anchor) two bounded walks run:

    - BACKWARD: up to 2 non-blank paragraphs → ``preceding`` (the signature /
      date lines above the 版记).
    - FORWARD: up to ``DOCX_SMART_IMPRINT_FORWARD_PARAS`` (default 3) non-blank
      paragraphs, closing on the first 印发-family CLOSER
      (:func:`guardrails.imprint_closer_reason`, only ever recognized here, in
      an anchor's forward window). Another anchor (抄送/主题词) encountered mid-
      walk is MIDDLE content, not a closer — a 主题词-opened region runs THROUGH
      a following 抄送 to reach the 印发 closer. When a closer is found the
      anchor→closer span (middle lines included) becomes ``members``; otherwise
      ``members`` is just the anchor.
    - After the closer, up to 2 bare 成文日期 lines (:func:`guardrails.
      is_document_date`) immediately following it are ALSO pulled into
      ``members``. Some documents mis-order 版记 THEN 成文日期 (not the GB/T
      order); that trailing date belongs to THIS document, so absorbing it
      keeps the NEXT cover from seeding a title window on it.

    All walks step over blank AND TOC lines without spending budget, and stop
    at any non-paragraph record (a table / section break is a structural
    boundary a ruling must not leak across). Overlapping anchors' regions are
    unioned by the caller.
    """
    document_date = document_date or guardrails.is_document_date
    forward_paras = _env_int(
        "DOCX_SMART_IMPRINT_FORWARD_PARAS", DEFAULT_DOCX_SMART_IMPRINT_FORWARD_PARAS
    )
    n = len(records)
    regions: list[ImprintRegion] = []
    for i, rec in enumerate(records):
        if rec.kind != "para" or i in skip_indices:
            continue
        if imprint_marker(rec.text) is None:
            continue

        preceding: set[int] = set()
        remaining = 2
        k = i - 1
        while k >= 0 and remaining:
            r = records[k]
            if _blank_or_skipped(r, k, skip_indices):
                k -= 1
                continue
            if r.kind != "para":
                break
            preceding.add(k)
            remaining -= 1
            k -= 1

        members: set[int] = {i}
        closer: int | None = None
        seen = 0
        j = i + 1
        while j < n and seen < forward_paras:
            r = records[j]
            if _blank_or_skipped(r, j, skip_indices):
                j += 1
                continue
            if r.kind != "para":
                break
            seen += 1
            members.add(j)
            # Only a 印发-family CLOSER ends the region; a start marker (抄送 /
            # 主题词) here is middle content, so the walk runs through it.
            if imprint_closer(r.text) is not None:
                closer = j
                break
            j += 1
        if closer is None:
            members = {i}  # no closer within the window → anchor only
        else:
            # Absorb a mis-ordered 成文日期 trailing the closer (see docstring).
            date_budget = 2
            m = closer + 1
            while m < n and date_budget:
                r = records[m]
                if _blank_or_skipped(r, m, skip_indices):
                    m += 1
                    continue
                if r.kind != "para" or not document_date(r.text):
                    break
                members.add(m)
                date_budget -= 1
                m += 1

        regions.append(
            ImprintRegion(anchor=i, closer=closer, members=members, preceding=preceding)
        )
    return regions


def find_title_block_candidates(
    records: Sequence[Any],
    *,
    fs_base_pt: float | None,
    strong_body: Callable[[str], str | None] | None = None,
    numbering_veto: Callable[[Any, str], str | None] | None = None,
    imprint_marker: Callable[[str], str | None] | None = None,
    imprint_closer: Callable[[str], str | None] | None = None,
    document_date: Callable[[str], bool] | None = None,
    imprint_excluded: set[int] | None = None,
    imprint_boundary_indices: set[int] | None = None,
    skip_indices: set[int] = frozenset(),
    suppressed_events: list[dict] | None = None,
    head_zone_records: int | None = None,
) -> list[TitleBlockCandidate]:
    """Find multi-paragraph windows and single-paragraph title candidates.

    ``strong_body`` / ``numbering_veto`` / ``imprint_marker`` / ``imprint_closer``
    / ``document_date`` default to the guardrails implementations and are
    injectable for NLP-free tests. ``imprint_excluded`` (the 版记 veto set)
    and ``imprint_boundary_indices`` (each region's LAST member — closer or
    its absorbed 成文日期 line, the 公文汇编 document-boundary evidence) are
    both derived from ONE internal :func:`detect_imprint_regions` call when
    either is None, so test and production callers share the same semantics;
    ``run_smart_heading`` passes precomputed sets to reuse its regions.

    Mid-document gate (§2.2.4): a multi/table window whose start lies past
    the document HEAD ZONE — more than ``head_zone_records`` CONTENT records
    (semantic-text paragraphs / tables; blanks, TOC lines, section breaks
    and pure-drawing logo paragraphs don't count) precede it — is only
    admitted when it carries document-boundary evidence: its preceding
    non-blank record is an imprint-region tail or a bare 附件 marker line,
    or the window opens on such a marker. A rejected window is appended to
    ``suppressed_events`` (when provided) instead of becoming a candidate;
    it never reaches the LLM and never joins ``covered`` (the single channel
    keeps its take-over chance). ``head_zone_records`` defaults to the
    ``DOCX_SMART_TITLE_HEAD_ZONE_RECORDS`` env knob — real covers may sit
    behind a few leading tables or long title lines, never ~a page of
    content.
    """
    strong_body = strong_body or guardrails.strong_body_reason
    numbering_veto = numbering_veto or guardrails.numbering_homophone_reason
    imprint_marker = imprint_marker or guardrails.imprint_marker_reason
    imprint_closer = imprint_closer or guardrails.imprint_closer_reason
    document_date = document_date or guardrails.is_document_date
    delta = _env_float(
        "DOCX_SMART_TITLE_BLOCK_MIN_DELTA", DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA
    )
    if imprint_excluded is None or imprint_boundary_indices is None:
        regions = detect_imprint_regions(
            records,
            imprint_marker=imprint_marker,
            imprint_closer=imprint_closer,
            document_date=document_date,
            skip_indices=skip_indices,
        )
        if imprint_excluded is None:
            imprint_excluded = set()
            for reg in regions:
                imprint_excluded |= reg.members
                imprint_excluded |= reg.preceding
        if imprint_boundary_indices is None:
            # A region's LAST member (closer or the trailing 成文日期 it
            # absorbed) — same walk origin as the confirm scan in
            # _demote_confirmed_imprint_regions (max(members) + 1).
            imprint_boundary_indices = {
                max(reg.members) for reg in regions if reg.closer is not None
            }

    para_indices = [
        i
        for i, r in enumerate(records)
        if r.kind == "para"
        and bool((r.text or "").strip())
        and i not in skip_indices
        and not r.is_toc_field
        and not r.is_toc_link
    ]

    n = len(records)

    def _is_content_record(idx: int, r: Any) -> bool:
        return is_content_record(idx, r, skip_indices)

    if head_zone_records is None:
        head_zone_records = _env_int(
            "DOCX_SMART_TITLE_HEAD_ZONE_RECORDS",
            DEFAULT_DOCX_SMART_TITLE_HEAD_ZONE_RECORDS,
        )
    # Prefix counts of content records: content_before[i] = how many content
    # records precede record i.
    content_before = [0] * (n + 1)
    for i, r in enumerate(records):
        content_before[i + 1] = content_before[i] + (
            1 if _is_content_record(i, r) else 0
        )

    def _is_attachment_marker(idx: int) -> bool:
        r = records[idx]
        return r.kind == "para" and bool(
            _ATTACHMENT_OPENER.match(_cover_semantic_text(r).strip())
        )

    def _table_member_ok(idx: int) -> bool:
        """Membership gate: EVERY non-empty cell reads as title material —
        no strong-body feature, no 版记 (imprint) marker OR 印发-family closer,
        within the title length cap, no physical outline. A data table
        (sentence / long cells) can never join, which is what makes absorbing
        member tables content-safe. Defined ahead of the head-zone block: the
        body-signal scan uses its INVERSE (a table that cannot be cover
        material is body evidence)."""
        cf = records[idx].table_cell_features
        if cf is None:
            return False
        for row in cf:
            for text, _size, has_outline in row:
                t = (text or "").strip()
                if not t:
                    continue
                if has_outline:
                    return False
                if guardrails.weighted_char_length(t) > TITLE_LINE_MAX_WEIGHTED_CHARS:
                    return False
                if strong_body(t) is not None:
                    return False
                # Imprint check on the RAW cell text: the whitespace after a
                # closer prefix (印发机关␣…) is the match evidence and a stripped
                # copy would erase it. A short imprint cell is not necessarily
                # caught by the strong-body rules. BOTH the anchor (抄送 / 主题词)
                # and closer (印发 / …印发) shapes are checked per-cell — a table
                # channel has no "anchor above" context, so a 版记 cell must veto
                # on its own here.
                if imprint_marker(text) is not None or imprint_closer(text) is not None:
                    return False
        return True

    # First BODY-SIGNAL content record: a sentence-terminated paragraph, a
    # real (outline/genuinely numbered) heading, or a DATA table (one that
    # cannot be cover material — sentence/long/outline cells) marks where
    # document prose / structure starts — a cover never follows it. The
    # paragraph check is DELIBERATELY punctuation-based, not strong_body-
    # based: spaCy falsely splits decimal-bearing cover titles (“7.19”) into
    # "multi-sentence" strong verdicts, and one false-strong line at the top
    # would close the zone on a real cover. Computed lazily and cached;
    # ``n`` = no signal anywhere (tiny cover-only documents).
    _body_signal_cache: list[int | None] = [None]

    def _first_body_signal() -> int:
        if _body_signal_cache[0] is None:
            sig = n
            for i2, r2 in enumerate(records):
                if not _is_content_record(i2, r2):
                    continue
                if r2.kind == "table":
                    if not _table_member_ok(i2):
                        sig = i2
                        break
                    continue
                if r2.kind == "para" and (
                    _cover_semantic_text(r2).rstrip().endswith(_BODY_SENTENCE_ENDERS)
                    or _is_real_heading(i2)
                ):
                    sig = i2
                    break
            _body_signal_cache[0] = sig
        return _body_signal_cache[0]

    def _window_position_allowed(start: int) -> bool:
        """§2.2.4 mid-document gate: a multi/table window opens freely in the
        document head zone — fewer than ``head_zone_records`` content records
        before it (blanks/TOC/section breaks/logo paragraphs don't count) AND
        no body signal (sentence-punctuated line / real heading / data table)
        yet: a couple of leading cover tables or long title lines keep the
        zone open, but a single body sentence or data table at the top closes
        it (a short preamble must not reopen the door the record-count cap
        guards on long documents). Past the zone a window needs document-boundary evidence —
        it starts on a bare 附件 marker line, or its preceding non-blank
        record (non-content records skipped, same eyes as the imprint confirm
        scan) is an imprint-region tail or an attachment marker paragraph."""
        if content_before[start] < head_zone_records and start <= _first_body_signal():
            return True
        if _is_attachment_marker(start):
            return True
        k = start - 1
        while k >= 0:
            if not _is_content_record(k, records[k]):
                k -= 1
                continue
            return k in imprint_boundary_indices or _is_attachment_marker(k)
        return True

    def _suppress(trigger: str, start: int, end: int) -> None:
        if suppressed_events is not None:
            suppressed_events.append(
                {
                    "rule": "mid_document_window_suppressed",
                    "trigger": trigger,
                    "start": start,
                    "end": end,
                }
            )

    strong_cache: dict[int, str | None] = {}

    def _is_strong(idx: int) -> bool:
        if idx not in strong_cache:
            strong_cache[idx] = strong_body(records[idx].text)
        return strong_cache[idx] is not None

    real_heading_cache: dict[int, bool] = {}

    def _is_real_heading(idx: int) -> bool:
        """A paragraph carrying a physical outline level or GENUINE numbering
        is a structural section heading, never title-block cover text — so
        BOTH channels exclude it: it terminates (and stays out of) a
        multi-paragraph window, and it is never a single-paragraph candidate.

        Enforces invariant I2 at candidate-formation time: an outline
        paragraph is a heading and must never be routed into a title block —
        a multi-window member's text is lost (absorbed members become empty
        sentinels) and a single-line title would demote it to a level-0
        cover heading. ``outline_level_raw`` (pre-policy) is used so a
        length-demoted heading (``outline_level`` None, raw set) is also kept
        out, matching the physical-outline test in :func:`judge_title_block`.
        """
        if idx not in real_heading_cache:
            r = records[idx]
            real_heading_cache[idx] = (
                r.outline_level_raw is not None
                or _has_real_numbering(r.text, numbering_veto)
            )
        return real_heading_cache[idx]

    def _is_title_line(rec: Any) -> bool:
        size = effective_font_size_pt(rec)
        return (
            fs_base_pt is not None
            and size is not None
            and size >= fs_base_pt + MULTI_WINDOW_TITLE_DELTA_PT
            and guardrails.weighted_char_length(rec.text.strip())
            <= TITLE_LINE_MAX_WEIGHTED_CHARS
            and not rec.is_toc_field
            and not rec.is_toc_link
        )

    def _dominates_neighbor_headings(
        win_max: float | None, start: int, end: int
    ) -> bool:
        """Visual-dominance gate (Rule 2): the block's biggest line must
        out-size a neighbouring section heading on at least one flank.

        Scans up to ``_FLANK_WINDOW`` paragraphs outward from each window edge
        for the nearest real section heading (physical outline OR genuine
        numbering, via ``_is_real_heading``) and compares its effective size.
        A genuine cover title sits ABOVE the section-heading tier; a mid-body
        emphasis cluster that never beats the surrounding headings is not a
        title block. When no such heading flanks the window (e.g. a document
        with no outline/numbering structure, or a size-less neighbour) the
        rule cannot be evaluated and passes — its absolute counterpart
        (``_is_title_line``'s +2pt over body) still guards the candidate.
        """
        if win_max is None:
            return False
        total = len(records)

        def _nearest_heading_size(step: int, frm: int) -> float | None:
            seen = 0
            k = frm
            while 0 <= k < total and seen < _FLANK_WINDOW:
                r = records[k]
                if r.kind == "para":
                    seen += 1
                    if _is_real_heading(k):
                        return effective_font_size_pt(r)
                k += step
            return None

        prev = _nearest_heading_size(-1, start - 1)
        nxt = _nearest_heading_size(1, end)
        if prev is None and nxt is None:
            return True  # no comparable neighbour → rule inapplicable, pass
        return (prev is not None and win_max > prev) or (
            nxt is not None and win_max > nxt
        )

    candidates: list[TitleBlockCandidate] = []
    covered: set[int] = set()

    # --- multi-paragraph windows -----------------------------------------
    i = 0
    while i < n:
        rec = records[i]
        if (
            rec.kind != "para"
            or i in skip_indices
            or i in imprint_excluded
            or _is_strong(i)
            or _is_real_heading(i)
            or rec.is_toc_field
            or rec.is_toc_link
        ):
            i += 1
            continue
        # Grow the window: consecutive paragraphs without strong-body
        # features; empty paragraphs stay inside, tables / section breaks /
        # TOC lines / strong-body paragraphs / real section headings
        # (physical outline or genuine numbering) end it. A page/section
        # boundary BETWEEN scanned records also ends it (a title block is a
        # single-page unit); the boundary-carrying record becomes the next
        # seed via ``i = max(end, i + 1)`` — reset, not swallowed. The seed
        # itself carrying a leading break is fine (a 公文汇编 second cover
        # opens on a page break).
        start = i
        window_paras = []
        j = i
        prev_rec = None
        while j < n:
            r = records[j]
            if (
                window_paras
                and prev_rec is not None
                and _page_boundary_between(prev_rec, r)
            ):
                break
            if r.kind == "empty_para":
                prev_rec = r
                j += 1
                continue
            if (
                r.kind != "para"
                or j in skip_indices
                or j in imprint_excluded
                or r.is_toc_field
                or r.is_toc_link
                or _is_strong(j)
                or _is_real_heading(j)
            ):
                break
            window_paras.append(j)
            prev_rec = r
            j += 1
        end = j
        title_line_sizes = [
            effective_font_size_pt(records[k])
            for k in window_paras
            if _is_title_line(records[k])
        ]
        if (
            len(window_paras) >= 2
            and title_line_sizes
            and _dominates_neighbor_headings(max(title_line_sizes), start, end)
        ):
            if _window_position_allowed(start):
                candidates.append(
                    TitleBlockCandidate(
                        start=start, end=end, single=False, trigger="multi_window"
                    )
                )
                covered.update(range(start, end))
            else:
                _suppress("multi_window", start, end)
        i = max(end, i + 1)

    # --- table windows (§2.2.4 table channel) ------------------------------
    # A cover laid out INSIDE tables — possibly interleaved with cover-material
    # paragraphs (a mixed table/paragraph cover: 档号表 + 主标题段 + 发文机关表)
    # or image-only paragraphs (a 印章 / logo between cover tables). Scan each
    # run of consecutive tables and absorbable paragraphs (empty paragraphs may
    # sit between them); the first non-qualifying record ends both the window
    # AND the run (the remainder never joins a title block). The run is a
    # candidate when >=1 member table carries a title row OR >=1 absorbed
    # paragraph is a cover title line.

    def _para_absorbable_in_cover(idx: int) -> bool:
        """A run paragraph that reads as cover material: absorbed into the
        table run as a member (source order) instead of breaking it. Pure
        image placeholder(s) (印章/logo/二维码) always qualify; otherwise the
        same bar a table cell clears in ``_table_member_ok`` — short,
        non-body, no physical outline / real numbering, not a 版记 line — all
        judged on the SEMANTIC text (drawing tags stripped): a mixed
        "logo + 机关名" paragraph judges on the 机关名 alone, while rendering
        keeps the original text so the image survives in the block."""
        rec = records[idx]
        if rec.kind != "para" or idx in skip_indices or idx in imprint_excluded:
            # imprint_excluded: 版记 region members/preceding — the same
            # neighbourhood protection the paragraph channels apply.
            return False
        if rec.is_toc_field or rec.is_toc_link or _is_real_heading(idx):
            return False
        if not (rec.text or "").strip():
            return False
        semantic = _cover_semantic_text(rec)
        if not semantic:
            return True  # pure image placeholder(s), any length
        # Real-numbering check on the SEMANTIC text: _is_real_heading above
        # classifies the RAW text, which a leading <drawing/> tag defeats —
        # "<drawing/> 第一章 总则" is a genuine section heading and must break
        # the run, not join the cover.
        if _has_real_numbering(semantic, numbering_veto):
            return False
        if guardrails.weighted_char_length(semantic) > TITLE_LINE_MAX_WEIGHTED_CHARS:
            return False
        if strong_body(semantic) is not None:
            return False
        # Per-line 版记 backstop for lines outside any detected region (e.g.
        # an isolated closer with no anchor above).
        if imprint_marker(semantic) is not None or imprint_closer(semantic) is not None:
            return False
        return True

    def _is_cover_title_line(idx: int) -> bool:
        """An absorbed paragraph that can CARRY the cover title (so a mixed
        cover with the main title in a paragraph, not a table row, still
        stands). Length/emptiness judge the semantic text — a long drawing
        tag sharing the line must not push the title over the cap; the size
        is the paragraph's real effective size. A pure-image paragraph
        (empty semantic) is decoration, never a title line."""
        rec = records[idx]
        size = effective_font_size_pt(rec)
        semantic = _cover_semantic_text(rec)
        return (
            fs_base_pt is not None
            and size is not None
            and size >= fs_base_pt + MULTI_WINDOW_TITLE_DELTA_PT
            and bool(semantic)
            and guardrails.weighted_char_length(semantic)
            <= TITLE_LINE_MAX_WEIGHTED_CHARS
            and not rec.is_toc_field
            and not rec.is_toc_link
        )

    # _table_member_ok is defined earlier (above the head-zone block): the
    # body-signal scan needs its inverse — a table that cannot be cover
    # material is body evidence.

    def _table_title_size(idx: int) -> float | None:
        """Max size over this table's title rows (None = no title row).

        A title row is a single-PHYSICAL-cell row (a gridSpan full-width
        merge is ONE ``w:tc``) whose text is non-empty, unnumbered (same
        genuine-numbering exclusion as the paragraph channels) and sized at
        the +2pt tier over the global FS_base."""
        if fs_base_pt is None:
            return None
        best = None
        for row in records[idx].table_cell_features or ():
            if len(row) != 1:
                continue
            text, size, _outline = row[0]
            t = (text or "").strip()
            if (
                t
                and size is not None
                and size >= fs_base_pt + MULTI_WINDOW_TITLE_DELTA_PT
                and not _has_real_numbering(t, numbering_veto)
            ):
                if best is None or size > best:
                    best = size
        return best

    i = 0
    while i < n:
        if records[i].kind != "table" or i in skip_indices:
            i += 1
            continue
        members: list[int] = []
        j = i
        prev_rec = None
        boundary_break = False
        while j < n:
            r = records[j]
            # Page/section boundary between adjacent scanned records ends the
            # run BEFORE the boundary record — which then reseeds a new run
            # (checked on every prev/cur pair incl. table↔para adjacency).
            if prev_rec is not None and _page_boundary_between(prev_rec, r):
                boundary_break = True
                break
            if r.kind == "empty_para":
                prev_rec = r
                j += 1
                continue
            if r.kind == "table" and j not in skip_indices and _table_member_ok(j):
                members.append(j)
            elif r.kind == "para" and _para_absorbable_in_cover(j):
                members.append(j)  # absorbed cover material / image, in order
            else:
                break
            prev_rec = r
            j += 1
        # The rest of the run (from the first non-qualifying record) is
        # skipped outright — it never seeds another window. A boundary break
        # skips the sweep entirely (the boundary record starts a NEW run),
        # and the sweep itself also stops at boundaries for the same reason.
        run_end = j
        if not boundary_break:
            while run_end < n and (
                records[run_end].kind in ("table", "empty_para")
                or _para_absorbable_in_cover(run_end)
            ):
                if run_end > i and _page_boundary_between(
                    records[run_end - 1], records[run_end]
                ):
                    break
                run_end += 1
        if members:
            # Title evidence from EITHER side of the mixed cover: a table
            # title row, or an absorbed paragraph that is a cover title line.
            sizes = []
            for m in members:
                if records[m].kind == "table":
                    s = _table_title_size(m)
                else:
                    s = (
                        effective_font_size_pt(records[m])
                        if _is_cover_title_line(m)
                        else None
                    )
                if s is not None:
                    sizes.append(s)
            start, end = members[0], members[-1] + 1
            if sizes and _dominates_neighbor_headings(max(sizes), start, end):
                if _window_position_allowed(start):
                    candidates.append(
                        TitleBlockCandidate(
                            start=start,
                            end=end,
                            single=False,
                            trigger="table_window",
                            table=True,
                            members=tuple(members),
                        )
                    )
                    covered.update(range(start, end))
                else:
                    _suppress("table_window", start, end)
        i = max(run_end, i + 1)

    # --- imprint-single channel (版记后单行标题) ----------------------------
    # An imprint-region tail is STRONG document-boundary evidence (user
    # ruling): the 公文汇编 next-document cover right after it may be a
    # SINGLE big line (page/section breaks and logo paragraphs absorbed by
    # the walk), exempt from the multi-window >=2-member requirement. The
    # candidate reuses single=True judge semantics (± context window; a
    # false verdict's partition carries no force). Windows the multi/table
    # scans already covered are not re-emitted.
    for b in sorted(imprint_boundary_indices):
        f = b + 1
        while f < n and not _is_content_record(f, records[f]):
            f += 1
        if f >= n or f in covered:
            continue
        r = records[f]
        if (
            r.kind == "para"
            and f not in imprint_excluded
            and not _is_strong(f)
            and not _is_real_heading(f)
            and _is_title_line(r)
        ):
            candidates.append(
                TitleBlockCandidate(
                    start=f, end=f + 1, single=True, trigger="imprint_single"
                )
            )
            covered.update({f})

    # --- single-paragraph channel -----------------------------------------
    # Only the document's first eligible paragraph may represent its main
    # title.  Never scan forward for a replacement: page/section boundaries,
    # centered blank-flanked lines, and body-font changes are common inside a
    # document and promoting those lines to level-0 roots damages the chapter
    # hierarchy.  ``para_indices`` already excludes leading empty paragraphs,
    # TOC records, and caller-supplied skip indices.
    if para_indices:
        idx = para_indices[0]
        rec = records[idx]
        single_size = effective_font_size_pt(rec)
        if (
            idx not in covered
            and not _is_strong(idx)
            and idx not in imprint_excluded
            and guardrails.weighted_char_length(rec.text.strip())
            <= TITLE_LINE_MAX_WEIGHTED_CHARS
            and not _is_real_heading(idx)
            and fs_base_pt is not None
            and single_size is not None
            and single_size >= fs_base_pt + delta
            and _dominates_neighbor_headings(single_size, idx, idx + 1)
        ):
            candidates.append(
                TitleBlockCandidate(
                    start=idx, end=idx + 1, single=True, trigger="single_line"
                )
            )
    candidates.sort(key=lambda c: c.start)
    return candidates


# ---------------------------------------------------------------------------
# LLM judgment
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a document-structure analyst. You are shown consecutive "
    "paragraphs from a document (typically its opening, or a segment right "
    "after a page break). Decide whether they form a TITLE BLOCK: the cover "
    "area carrying the document's main title plus optional subtitle, "
    "document number, classification, publisher and date — as opposed to "
    "ordinary section headings or body text. A title block must contain the "
    "document's own MAIN TITLE; metadata lines alone (a standard/document "
    "number, a date) without the main title are NOT a title block. When "
    "font-size evidence is given, weigh it: a cover main title is normally "
    "set in the largest font, and small same-sized lines around it are "
    "usually form labels or metadata — their presence does not disqualify a "
    "dominant title line. If unsure, answer false. Answer with a single raw "
    "JSON object — no markdown fences, no commentary."
)

_USER_TEMPLATE = """Paragraphs (indexed; [BLANK] marks an empty line in the original):

{window}

{dominance}Rules:
- First decide "is_title_block".
- If true: fill "main_title" (required; NEVER a front-matter/bookkeeping heading — 目录 / 目次 / 更改记录 / 修订记录 / 修改记录 / 变更记录 / 版本记录 / 更改页 / 修订页 / 前言 / 引言 / 摘要 / 编制说明, or English equivalents like Contents / Table of Contents / Revision History / Change Record / Change Log / Document History / Foreword / Abstract, case-insensitive, ignoring whitespace between Chinese characters; a longer title merely containing such a word, e.g. 更改记录管理规范, is acceptable) plus any of "sub_title" / "doc_number" / "classification" / "publisher" / "date" that are present — each must be verbatim text taken from the paragraphs above (the main title may concatenate consecutive paragraphs); use null for absent fields, and set "headings" and "body" to [].
- If false: set all six text fields to null, and classify EVERY index — {indices} — into exactly one of "headings" (a real section heading, e.g. 前言 / 第一章 / 目录 / 更改记录) or "body" (everything else; document numbers, dates and publisher lines are NOT headings — put them in "body"). Each index must appear in exactly one of the two arrays, as an integer (not a string).

Respond with JSON matching:
{{"is_title_block": true|false, "main_title": string|null, "sub_title": string|null, "doc_number": string|null, "classification": string|null, "publisher": string|null, "date": string|null, "headings": [int, ...], "body": [int, ...]}}"""


def _dominance_legend(
    line_sizes: Sequence[float | None], fs_base_pt: float | None
) -> str:
    """Deterministic font-size evidence for the LLM window (§2.2.4).

    ``line_sizes[k]`` is the effective size of prompt line ``[k]`` — ONLY
    lines actually emitted have an entry, so a token-truncated line can never
    appear in the legend. Lines at the title tier (FS_base + 2pt, the same
    bar the candidate gates use) are listed, tiered so the LARGEST size —
    the cue a human uses to spot the cover title — reads stronger than
    merely-enlarged lines (e.g. an absorbed next-page section heading).
    Rendered as a standalone line, never inside the indexed text lines, so
    locate-back canons and verbatim echoes are untouched. Empty when no
    baseline or no dominant line — the prompt is then byte-identical to the
    pre-evidence form.
    """
    if fs_base_pt is None:
        return ""
    dominant = [
        (k, s)
        for k, s in enumerate(line_sizes)
        if s is not None and s >= fs_base_pt + MULTI_WINDOW_TITLE_DELTA_PT
    ]
    if not dominant:
        return ""
    top = max(s for _, s in dominant)

    def _fmt(items: list[tuple[int, float]]) -> str:
        return ", ".join(f"[{k}]={s:g}pt" for k, s in items)

    legend = "Font-size evidence — largest lines: " + _fmt(
        [(k, s) for k, s in dominant if s == top]
    )
    others = [(k, s) for k, s in dominant if s != top]
    if others:
        legend += "; other enlarged lines: " + _fmt(others)
    legend += (
        f" (body ≈ {fs_base_pt:g}pt). The visually largest lines are "
        "likely title-bearing if the window is a cover title block."
    )
    return legend


def _render_window(
    records: Sequence[Any], candidate: TitleBlockCandidate, warnings: dict | None
) -> tuple[str, list[int], list[int]]:
    """Render the candidate window for the LLM; returns
    ``(text, index_map, image_paras)``.

    ``index_map[k]`` is the record index of window line ``[k]``. The window
    is token-capped (env DOCX_SMART_LLM_WINDOW_TOKENS); overflow truncates
    from the tail with a warning.

    ``image_paras`` are the window's image-only paragraphs: they render NO
    line (nothing judgeable to show, and skipping keeps a non-title partition
    from having to classify an empty line), but a multi-window title verdict
    must still count them as members — dropping them would re-emit the image
    AFTER the block's members, breaking source order in assembly.
    """
    mandatory: int | None = None
    if candidate.single:
        # The single paragraph plus N context paragraphs on each side. The
        # candidate itself is MANDATORY: context is reference-only, so tail
        # truncation must never drop the candidate line (review D2) — else a
        # true verdict would compose a level-0 heading from context alone.
        para_positions = [i for i, r in enumerate(records) if r.kind == "para"]
        pos = para_positions.index(candidate.start)
        lo = para_positions[max(0, pos - _SINGLE_CONTEXT)]
        hi_pos = min(len(para_positions) - 1, pos + _SINGLE_CONTEXT)
        hi = para_positions[hi_pos] + 1
        span = range(lo, hi)
        mandatory = candidate.start
    else:
        span = range(candidate.start, candidate.end)

    cap = _env_int("DOCX_SMART_LLM_WINDOW_TOKENS", DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS)
    # Reserve the mandatory candidate line's budget up front so surrounding
    # context can be trimmed without ever evicting the candidate.
    reserve = (
        _estimate_tokens(_cover_semantic_text(records[mandatory]))
        if mandatory is not None
        else 0
    )
    lines: list[str] = []
    index_map: list[int] = []
    image_paras: list[int] = []
    used = 0
    truncated = False
    for i in span:
        rec = records[i]
        if rec.kind == "empty_para":
            lines.append("[BLANK]")
            continue
        if rec.kind != "para":
            continue
        # The LLM sees the SEMANTIC text (drawing tags removed entirely —
        # images carry no judgeable content, and a marker would only invite
        # echoes); locate-back canons match the same text. An image-only
        # paragraph thus renders nothing: skip it (no index_map entry, so a
        # non-title partition never has to classify an empty line) but track
        # it for membership — EXCEPT the mandatory single candidate, which
        # must always be emitted.
        semantic = _cover_semantic_text(rec)
        if not semantic and i != mandatory:
            image_paras.append(i)
            continue
        line = f"[{len(index_map)}] {semantic}"
        cost = _estimate_tokens(line)
        if i == mandatory:
            # Always emit the candidate; it was pre-reserved from the cap.
            used += cost
            reserve = 0  # candidate spent; free the reservation for context
            index_map.append(i)
            lines.append(line)
            continue
        if used + cost + reserve > cap and index_map:
            truncated = True
            if mandatory is None:
                break  # multi-paragraph window: tail truncation (spec §2.2.4)
            # Single window: skip this reference-only context line but keep
            # scanning so the mandatory candidate (and smaller context) land.
            continue
        used += cost
        index_map.append(i)
        lines.append(line)
    if truncated:
        if warnings is not None:
            warnings["title_block_window_truncated"] = (
                warnings.get("title_block_window_truncated", 0) + 1
            )
        logger.warning(
            "[smart_heading] title-block candidate window exceeded %d tokens; "
            "tail content not shown to the LLM",
            cap,
        )
    return "\n".join(lines), index_map, image_paras


def _render_table_window(
    records: Sequence[Any], candidate: TitleBlockCandidate, warnings: dict | None
) -> tuple[str, list[int], str, list[float | None]]:
    """Render a TABLE candidate window for the LLM (§2.2.4 table channel).

    One indexed line per non-empty PHYSICAL cell, row by row across the
    member tables — plus one line per absorbed cover-material paragraph (its
    whole text: a mixed cover may carry the MAIN TITLE in a paragraph, so it
    must reach the LLM window and the locate-back canon). Returns
    ``(window_text, member_record_indices, window_canon, line_sizes)`` — the
    canon concatenates the rendered texts because a table record's own
    ``text`` is a ``<table>{json}</table>`` placeholder and useless for
    locate-back; ``line_sizes`` is parallel to the rendered lines (cell size
    from ``table_cell_features``, paragraph size via
    :func:`effective_font_size_pt`) and feeds the font-size legend.
    Token-capped from the tail (same cap and warning as the multi-paragraph
    window).

    Members come from ``candidate.members`` (filled by the table-run scan,
    source order). An empty ``members`` falls back to scanning the range for
    tables — the pre-``members`` behaviour, kept for hand-built candidates.
    """
    cap = _env_int("DOCX_SMART_LLM_WINDOW_TOKENS", DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS)
    lines: list[str] = []
    line_sizes: list[float | None] = []
    canon_parts: list[str] = []
    members: list[int] = []
    used = 0
    truncated = False

    def _emit(t: str, size: float | None) -> None:
        nonlocal used, truncated
        if not t or truncated:
            return
        line = f"[{len(lines)}] {t}"
        cost = _estimate_tokens(line)
        if used + cost > cap and lines:
            truncated = True
            return
        used += cost
        lines.append(line)
        line_sizes.append(size)
        canon_parts.append(_canon(t))

    member_iter = candidate.members or tuple(
        i for i in range(candidate.start, candidate.end) if records[i].kind == "table"
    )
    for i in member_iter:
        rec = records[i]
        if rec.kind == "table":
            if rec.table_cell_features is None:
                continue
            members.append(i)
            for row in rec.table_cell_features:
                for text, size, _outline in row:
                    _emit((text or "").strip(), size)
        elif rec.kind == "para":
            # Absorbed cover-material / image paragraph: keep it as a member
            # (source-order assembly stays lossless) but show the LLM only the
            # SEMANTIC text — drawing tags are removed entirely, so a pure
            # image paragraph contributes NO window line (least prompt noise,
            # nothing to echo back) while a mixed line reads clean
            # ("某某管理办法") and locate-back matches it contiguously.
            members.append(i)
            _emit(_cover_semantic_text(rec), effective_font_size_pt(rec))
    if truncated:
        if warnings is not None:
            warnings["title_block_window_truncated"] = (
                warnings.get("title_block_window_truncated", 0) + 1
            )
        logger.warning(
            "[smart_heading] title-block candidate window exceeded %d tokens; "
            "tail content not shown to the LLM",
            cap,
        )
    return "\n".join(lines), members, "".join(canon_parts), line_sizes


def _parse_llm_json(raw: str) -> dict:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise TitleBlockLLMError(
            f"title-block LLM answer carries no JSON object: {text[:200]!r}"
        )
    payload = m.group(0)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        try:
            import json_repair

            data = json_repair.loads(payload)
        except Exception as exc:
            raise TitleBlockLLMError(
                f"title-block LLM answer is not valid JSON: {payload[:200]!r}"
            ) from exc
    if not isinstance(data, dict):
        raise TitleBlockLLMError(
            f"title-block LLM answer is not a JSON object: {payload[:200]!r}"
        )
    return data


def _locate(text: str | None, window_canon: str) -> bool:
    if text is None:
        return True
    canon = _canon(text)
    return bool(canon) and canon in window_canon


def judge_title_block(
    candidate: TitleBlockCandidate,
    records: Sequence[Any],
    llm_judge: LLMJudge,
    *,
    warnings: dict | None = None,
    fs_base_pt: float | None = None,
) -> TitleBlockDecision:
    """Run one candidate through the LLM and strictly validate the verdict.

    ``fs_base_pt`` (the global FS_base initial value, §2.2.2) turns on the
    font-size evidence legend in the prompt — the visual-dominance cue a
    human uses to spot a cover title, which the plain text lines cannot
    carry. ``None`` (hand-built candidates, legacy tests) renders the
    legend-free prompt.
    """
    if llm_judge is None:
        raise TitleBlockLLMError(
            "smart_heading needs an LLM to judge a title-block candidate but "
            "none is configured (debug runs: build_debug_rag(extract_llm_func=…))"
        )
    if candidate.table:
        # §2.2.4 table channel: the window is cell texts (plus any absorbed
        # cover-material paragraphs), and locate-back must run against them
        # (a table record's text is a <table> placeholder).
        window_text, member_records, window_canon, line_sizes = _render_table_window(
            records, candidate, warnings
        )
        index_map = []
        # Table windows render one indexed line per emitted cell/paragraph and
        # never a [BLANK] line, so the rendered line count IS the index count.
        line_count = len(window_text.splitlines())
    else:
        window_text, index_map, image_paras = _render_window(
            records, candidate, warnings
        )
        # Semantic canon (drawing tags stripped): the window showed tag-free
        # text, so the answer is tag-free — a raw canon would reject a title
        # split by a mid-line image as non-contiguous.
        window_canon = "".join(
            _canon(_cover_semantic_text(records[i])) for i in index_map
        )
        member_records = []
        line_count = len(index_map)
        # Paragraph windows label lines by index_map position, so the size
        # list is derived from it directly — emitted lines only, matching
        # the table channel's emit-time collection.
        line_sizes = [effective_font_size_pt(records[i]) for i in index_map]
        if candidate.single:
            # Single windows: the ±context lines are reference-only and a
            # true verdict's main_title is validated against the CANDIDATE
            # line alone — advertising a bigger context line as the largest
            # title-bearing line would steer the LLM straight into a
            # guaranteed locate-back failure. Only the candidate line may
            # carry font-size evidence.
            line_sizes = [
                s if index_map[k] == candidate.start else None
                for k, s in enumerate(line_sizes)
            ]
    # Spell out the exact index set the partition must cover — a copyable
    # list is far harder to drop an index from than an implicit "EVERY".
    dominance = _dominance_legend(line_sizes, fs_base_pt)
    prompt = _USER_TEMPLATE.format(
        window=window_text,
        dominance=f"{dominance}\n\n" if dominance else "",
        indices=", ".join(str(k) for k in range(line_count)),
    )
    raw = llm_judge(prompt, system_prompt=_SYSTEM_PROMPT)
    data = _parse_llm_json(raw)

    is_title = bool(data.get("is_title_block"))

    def _opt_str(key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TitleBlockLLMError(f"title-block field {key!r} must be a string")
        # Title material must be single-line: the prompt lets the LLM
        # concatenate consecutive paragraphs, and an echoed line break would
        # ride main_title into the heading stack and every descendant's
        # parent_headings. Locate-back below is unaffected (_canon strips
        # ALL whitespace).
        return flatten_heading_line(value) or None

    if is_title:
        main_title = _opt_str("main_title")
        if not main_title:
            raise TitleBlockLLMError(
                "title-block verdict lacks a main_title (required)"
            )
        # A single-line candidate's block is ONLY that line — the window
        # context around it was shown to the LLM for judgment, not for
        # inclusion; its heading is the main title alone (§2.2.4).
        if candidate.single:
            member_indices = (candidate.start,)
            sub_title = doc_number = None
            classification = publisher = date = None
            # Hallucination guard scoped to the CANDIDATE paragraph only: the
            # heading is that line's text, so a main_title copied from a
            # reference-only context paragraph must not validate (review D4).
            # Semantic form for the same reason as the window canon above.
            locate_scope = _canon(_cover_semantic_text(records[candidate.start]))
        else:
            # Table windows: members are the member records (tables + any
            # absorbed cover-material paragraphs), in source order. Paragraph
            # windows: the rendered lines PLUS the window's image-only
            # paragraphs — unrendered, but membership keeps them emitted in
            # source order by assembly instead of trailing the block.
            member_indices = (
                tuple(member_records)
                if candidate.table
                else tuple(sorted([*index_map, *image_paras]))
            )
            sub_title = _opt_str("sub_title")
            doc_number = _opt_str("doc_number")
            classification = _opt_str("classification")
            publisher = _opt_str("publisher")
            date = _opt_str("date")
            locate_scope = window_canon
        for label, value in (
            ("main_title", main_title),
            ("sub_title", sub_title),
            ("doc_number", doc_number),
        ):
            if not _locate(value, locate_scope):
                raise TitleBlockLLMError(
                    f"title-block {label} {value!r} cannot be located in the "
                    "candidate window (LLM hallucination guard)"
                )
        return TitleBlockDecision(
            candidate=candidate,
            is_title_block=True,
            member_indices=member_indices,
            main_title=main_title,
            sub_title=sub_title,
            doc_number=doc_number,
            classification=classification,
            publisher=publisher,
            date=date,
            raw_response=raw,
        )

    # Non-title verdict for a SINGLE candidate: the ±context was reference
    # only, so none of it is named in either list (review D3). The candidate
    # simply re-enters the normal §2.2.5 flow on its own signals — it is not
    # demoted, and the surrounding context keeps whatever standing the gate
    # gives it. The LLM's headings/body partition over the context is ignored.
    #
    # A TABLE candidate takes the same path: cells are not paragraph records,
    # so a headings/body partition has nothing to land on — the member tables
    # simply stay ordinary body tables.
    if candidate.single or candidate.table:
        return TitleBlockDecision(
            candidate=candidate,
            is_title_block=False,
            heading_indices=(),
            body_indices=(),
            raw_response=raw,
        )

    # Non-title verdict: the LLM partitions the window into headings (audit
    # reference only — no admission force) and body (vetoed). MALFORMED
    # output is loud —
    # a missing/null/non-int field, an out-of-range index, a duplicate inside
    # one list, or the same index voted both heading AND body are corruption
    # that cannot be reconciled. But an UNDER-SPECIFIED partition (two valid
    # lists whose union merely misses some window indices) is recoverable: the
    # LLM abstained on those paragraphs, so — exactly like the single/table
    # non-title path (review D3) — they are named in neither list and
    # re-enter the normal §2.2.5 flow. Loud (warning) but not fatal.
    def _indices(key: str) -> list[int]:
        value = data.get(key)
        # Must be present and a list of plain ints. bool is a subclass of int —
        # reject True/False masquerading as indexes. A missing key or null is
        # NOT an empty list here (no `or []`): a non-title verdict must state
        # both partitions, or it is malformed.
        if not isinstance(value, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise TitleBlockLLMError(f"title-block field {key!r} must be [int]")
        return list(value)

    headings = _indices("headings")
    body = _indices("body")
    h_set, b_set = set(headings), set(body)
    all_indices = set(range(len(index_map)))
    # Malformed (unreconcilable) → hard fail. The length checks catch [0, 0]
    # duplicates that the set operations would otherwise hide.
    if (
        (h_set | b_set) - all_indices  # out-of-range index
        or h_set & b_set  # same index in both lists
        or len(headings) != len(h_set)  # duplicate within headings
        or len(body) != len(b_set)  # duplicate within body
    ):
        raise TitleBlockLLMError(
            "non-title verdict partition is malformed "
            f"(headings={headings}, body={body}, window indices "
            f"{sorted(all_indices)})"
        )
    # Under-specification is recoverable: unmentioned indices abstain (they
    # enter neither heading_records nor body_records below, so downstream
    # sees neither an audit reference nor a veto for them).
    missing = all_indices - h_set - b_set
    if missing:
        if warnings is not None:
            warnings["title_block_partition_incomplete"] = (
                warnings.get("title_block_partition_incomplete", 0) + 1
            )
        logger.warning(
            "[smart_heading] non-title verdict left %d window paragraph(s) "
            "unclassified in candidate [%d, %d); leaving them to normal flow "
            "(window indices %s, records %s)",
            len(missing),
            candidate.start,
            candidate.end,
            sorted(missing),
            [index_map[k] for k in sorted(missing)],
        )

    heading_records: list[int] = [index_map[k] for k in headings]
    body_records: list[int] = []
    for k in body:
        rec_idx = index_map[k]
        if records[rec_idx].outline_level_raw is not None:
            # I2: a physical-outline paragraph is never demoted by an LLM
            # body vote — it keeps heading standing.
            if warnings is not None:
                warnings["title_block_llm_outline_demotion_blocked"] = (
                    warnings.get("title_block_llm_outline_demotion_blocked", 0) + 1
                )
            heading_records.append(rec_idx)
        else:
            body_records.append(rec_idx)

    return TitleBlockDecision(
        candidate=candidate,
        is_title_block=False,
        heading_indices=tuple(sorted(heading_records)),
        body_indices=tuple(sorted(body_records)),
        raw_response=raw,
    )
