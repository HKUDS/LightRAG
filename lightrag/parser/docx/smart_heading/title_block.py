"""Title-block (document main title) discovery and LLM judgment (spec §2.2.4).

Heuristics find CANDIDATE windows — runs of paragraphs lacking strong-body
features that contain a visually dominant line — plus a single-paragraph
channel for spliced multi-article documents whose per-article title is one
isolated big line. An LLM (via the synchronous judge callable; this module
never touches asyncio) then confirms and decomposes each candidate.

LLM output is STRICTLY validated: the main/sub title must be locatable in
the window text (concatenation allowed), and paragraphs carrying a physical
outline level are never demoted by an LLM "body" vote (invariant I2). A
non-title verdict's headings/body partition must be well-formed; a MALFORMED
partition (missing/null field, out-of-range index, a duplicate within a list,
or the same index voted both heading and body) raises
:class:`TitleBlockLLMError`, as does any unparseable or non-locatable answer —
those failures are loud, never silently degraded. An UNDER-SPECIFIED partition
(two valid lists whose union merely omits some window indices) is instead
recovered: the unmentioned paragraphs abstain (neither granted nor vetoed) and
re-enter the normal flow, logged via ``title_block_partition_incomplete`` so a
single local omission never fails the whole document.
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
    DEFAULT_DOCX_SMART_SINGLE_TITLE_LLM_MAX,
    DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA,
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

_FLANK_WINDOW = 20  # K paragraphs on each side for the size-divergence gate
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
    # non-title verdicts: LLM-granted plain-heading identity / body text
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


def _window_mode_size(records: Sequence[Any], indices: Sequence[int]) -> float | None:
    weights: dict[float, int] = {}
    for idx in indices:
        rec = records[idx]
        if rec.font_size_pt is None:
            continue
        w = len((rec.text or "").strip())
        if w <= 0:
            continue
        weights[rec.font_size_pt] = weights.get(rec.font_size_pt, 0) + w
    if not weights:
        return None
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


#: A title line must clear FS_base by at least this many points. BOTH
#: channels require +2pt: the multi-paragraph window used to admit the
#: §2.2.2 strong-signal tier (+1pt), but that deliberately let ordinary
#: section-heading-sized lines open windows (A7/B17). Tightening to +2pt —
#: matching the single-paragraph channel (DOCX_SMART_TITLE_BLOCK_MIN_DELTA) —
#: demands genuine cover-title dominance over body and aligns with the
#: §2.2.4 conservative-preference principle (a title-block false positive is
#: leveraged into regional structural errors, so ties go to "not a title").
MULTI_WINDOW_TITLE_DELTA_PT = 2.0


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


def _imprint_veto_indices(
    records: Sequence[Any],
    imprint_marker: Callable[[str], str | None],
    imprint_closer: Callable[[str], str | None],
    document_date: Callable[[str], bool],
    skip_indices: set[int],
) -> set[int]:
    """Record indices barred from every title-block channel by 公文版记 —
    the union of every region's members and preceding neighbours."""
    veto: set[int] = set()
    for reg in detect_imprint_regions(
        records,
        imprint_marker=imprint_marker,
        imprint_closer=imprint_closer,
        document_date=document_date,
        skip_indices=skip_indices,
    ):
        veto |= reg.members
        veto |= reg.preceding
    return veto


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
    warnings: dict | None = None,
    skip_indices: set[int] = frozenset(),
) -> list[TitleBlockCandidate]:
    """Find multi-paragraph windows and single-paragraph title candidates.

    ``strong_body`` / ``numbering_veto`` / ``imprint_marker`` / ``imprint_closer``
    / ``document_date`` default to the guardrails implementations and are
    injectable for NLP-free tests. ``imprint_excluded`` (the 版记 veto set) is
    computed internally via :func:`detect_imprint_regions` when not supplied;
    ``run_smart_heading`` passes a precomputed set so it can reuse the same
    regions for demotion.
    """
    strong_body = strong_body or guardrails.strong_body_reason
    numbering_veto = numbering_veto or guardrails.numbering_homophone_reason
    imprint_marker = imprint_marker or guardrails.imprint_marker_reason
    imprint_closer = imprint_closer or guardrails.imprint_closer_reason
    document_date = document_date or guardrails.is_document_date
    delta = _env_float(
        "DOCX_SMART_TITLE_BLOCK_MIN_DELTA", DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA
    )
    single_cap = _env_int(
        "DOCX_SMART_SINGLE_TITLE_LLM_MAX", DEFAULT_DOCX_SMART_SINGLE_TITLE_LLM_MAX
    )
    if imprint_excluded is None:
        imprint_excluded = _imprint_veto_indices(
            records, imprint_marker, imprint_closer, document_date, skip_indices
        )

    para_indices = [
        i for i, r in enumerate(records) if r.kind == "para" and i not in skip_indices
    ]
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
    n = len(records)
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
        # (physical outline or genuine numbering) end it.
        start = i
        window_paras = []
        j = i
        while j < n:
            r = records[j]
            if r.kind == "empty_para":
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
            candidates.append(
                TitleBlockCandidate(
                    start=start, end=end, single=False, trigger="multi_window"
                )
            )
            covered.update(range(start, end))
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

    def _table_member_ok(idx: int) -> bool:
        """Membership gate: EVERY non-empty cell reads as title material —
        no strong-body feature, no 版记 (imprint) marker OR 印发-family closer,
        within the title length cap, no physical outline. A data table
        (sentence / long cells) can never join, which is what makes absorbing
        member tables content-safe."""
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
        while j < n:
            r = records[j]
            if r.kind == "empty_para":
                j += 1
                continue
            if r.kind == "table" and j not in skip_indices and _table_member_ok(j):
                members.append(j)
            elif r.kind == "para" and _para_absorbable_in_cover(j):
                members.append(j)  # absorbed cover material / image, in order
            else:
                break
            j += 1
        # The rest of the run (from the first non-qualifying record) is
        # skipped outright — it never seeds another window.
        run_end = j
        while run_end < n and (
            records[run_end].kind in ("table", "empty_para")
            or _para_absorbable_in_cover(run_end)
        ):
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
        i = max(run_end, i + 1)

    # --- single-paragraph channel -----------------------------------------
    single_found = 0
    single_truncated = 0
    for pos, idx in enumerate(para_indices):
        if idx in covered:
            continue
        rec = records[idx]
        if rec.is_toc_field or rec.is_toc_link:
            continue
        if _is_strong(idx):
            continue
        # Imprint veto before the LLM-review cap below: a vetoed line must
        # not consume the per-document single-candidate budget.
        if idx in imprint_excluded:
            continue
        if (
            guardrails.weighted_char_length(rec.text.strip())
            > TITLE_LINE_MAX_WEIGHTED_CHARS
        ):
            continue
        if _is_real_heading(idx):
            continue
        single_size = effective_font_size_pt(rec)
        if not (
            fs_base_pt is not None
            and single_size is not None
            and single_size >= fs_base_pt + delta
        ):
            continue
        if not _dominates_neighbor_headings(single_size, idx, idx + 1):
            continue
        if not _single_boundary_evidence(records, idx, pos, para_indices):
            continue
        if single_found >= single_cap:
            single_truncated += 1
            continue
        single_found += 1
        candidates.append(
            TitleBlockCandidate(
                start=idx, end=idx + 1, single=True, trigger="single_line"
            )
        )
    if single_truncated and warnings is not None:
        warnings["title_block_single_candidates_truncated"] = (
            warnings.get("title_block_single_candidates_truncated", 0)
            + single_truncated
        )
        logger.warning(
            "[smart_heading] %d single-line title candidates beyond the "
            "per-document LLM review cap were skipped",
            single_truncated,
        )
    candidates.sort(key=lambda c: c.start)
    return candidates


def _single_boundary_evidence(
    records: Sequence[Any], idx: int, para_pos: int, para_indices: list[int]
) -> bool:
    """At least one hard boundary proof (§2.2.4 single-paragraph gate)."""
    rec = records[idx]
    # (a) first non-empty paragraph of the document
    first_para = next(
        (
            i
            for i in para_indices
            if records[i].text.strip() or records[i].kind == "para"
        ),
        None,
    )
    if idx == first_para:
        return True
    # (b) right after an explicit page/section boundary: own pageBreakBefore,
    # an own LEADING page-break run (Ctrl+Enter then typing on), a page-break
    # run in the previous paragraph, or a section break (§2.2.4 evidence b).
    if rec.page_break_before or getattr(rec, "has_leading_page_break_run", False):
        return True
    k = idx - 1
    while k >= 0 and records[k].kind == "empty_para":
        k -= 1
    if k >= 0 and (
        records[k].kind == "section_break"
        or getattr(records[k], "ends_section", False)
        or getattr(records[k], "has_page_break_run", False)
    ):
        return True
    # (c) centered with >=1 empty paragraph on both sides
    if rec.alignment == "center":
        prev_empty = idx > 0 and records[idx - 1].kind == "empty_para"
        next_empty = idx + 1 < len(records) and records[idx + 1].kind == "empty_para"
        if prev_empty and next_empty:
            return True
    # (d) body font-size divergence across the flanks (K=20 paragraphs each)
    before = para_indices[max(0, para_pos - _FLANK_WINDOW) : para_pos]
    after = para_indices[para_pos + 1 : para_pos + 1 + _FLANK_WINDOW]
    if before and after:
        mode_before = _window_mode_size(records, before)
        mode_after = _window_mode_size(records, after)
        if (
            mode_before is not None
            and mode_after is not None
            and mode_before != mode_after
        ):
            return True
    return False


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
    "number, a date) without the main title are NOT a title block. If "
    "unsure, answer false. Answer with a single raw JSON object — no "
    "markdown fences, no commentary."
)

_USER_TEMPLATE = """Paragraphs (indexed; [BLANK] marks an empty line in the original):

{window}

Rules:
- First decide "is_title_block".
- If true: fill "main_title" (required) plus any of "sub_title" / "doc_number" / "classification" / "publisher" / "date" that are present — each must be verbatim text taken from the paragraphs above (the main title may concatenate consecutive paragraphs); use null for absent fields, and set "headings" and "body" to [].
- If false: set all six text fields to null, and classify EVERY index — {indices} — into exactly one of "headings" (a real section heading, e.g. 前言 / 第一章) or "body" (everything else; document numbers, dates and publisher lines are NOT headings — put them in "body"). Each index must appear in exactly one of the two arrays, as an integer (not a string).

Respond with JSON matching:
{{"is_title_block": true|false, "main_title": string|null, "sub_title": string|null, "doc_number": string|null, "classification": string|null, "publisher": string|null, "date": string|null, "headings": [int, ...], "body": [int, ...]}}"""


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
) -> tuple[str, list[int], str]:
    """Render a TABLE candidate window for the LLM (§2.2.4 table channel).

    One indexed line per non-empty PHYSICAL cell, row by row across the
    member tables — plus one line per absorbed cover-material paragraph (its
    whole text: a mixed cover may carry the MAIN TITLE in a paragraph, so it
    must reach the LLM window and the locate-back canon). Returns
    ``(window_text, member_record_indices, window_canon)`` — the canon
    concatenates the rendered texts because a table record's own ``text`` is
    a ``<table>{json}</table>`` placeholder and useless for locate-back.
    Token-capped from the tail (same cap and warning as the multi-paragraph
    window).

    Members come from ``candidate.members`` (filled by the table-run scan,
    source order). An empty ``members`` falls back to scanning the range for
    tables — the pre-``members`` behaviour, kept for hand-built candidates.
    """
    cap = _env_int("DOCX_SMART_LLM_WINDOW_TOKENS", DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS)
    lines: list[str] = []
    canon_parts: list[str] = []
    members: list[int] = []
    used = 0
    truncated = False

    def _emit(t: str) -> None:
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
                for text, _size, _outline in row:
                    _emit((text or "").strip())
        elif rec.kind == "para":
            # Absorbed cover-material / image paragraph: keep it as a member
            # (source-order assembly stays lossless) but show the LLM only the
            # SEMANTIC text — drawing tags are removed entirely, so a pure
            # image paragraph contributes NO window line (least prompt noise,
            # nothing to echo back) while a mixed line reads clean
            # ("某某管理办法") and locate-back matches it contiguously.
            members.append(i)
            _emit(_cover_semantic_text(rec))
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
    return "\n".join(lines), members, "".join(canon_parts)


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
) -> TitleBlockDecision:
    """Run one candidate through the LLM and strictly validate the verdict."""
    if llm_judge is None:
        raise TitleBlockLLMError(
            "smart_heading needs an LLM to judge a title-block candidate but "
            "none is configured (debug runs: build_debug_rag(extract_llm_func=…))"
        )
    if candidate.table:
        # §2.2.4 table channel: the window is cell texts (plus any absorbed
        # cover-material paragraphs), and locate-back must run against them
        # (a table record's text is a <table> placeholder).
        window_text, member_records, window_canon = _render_table_window(
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
    # Spell out the exact index set the partition must cover — a copyable
    # list is far harder to drop an index from than an implicit "EVERY".
    prompt = _USER_TEMPLATE.format(
        window=window_text,
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
        return value.strip() or None

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
    # only, so we neither grant nor veto any of it (review D3). The candidate
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

    # Non-title verdict: the LLM partitions the window into headings (granted
    # plain-heading identity) and body (vetoed). MALFORMED output is loud —
    # a missing/null/non-int field, an out-of-range index, a duplicate inside
    # one list, or the same index voted both heading AND body are corruption
    # that cannot be reconciled. But an UNDER-SPECIFIED partition (two valid
    # lists whose union merely misses some window indices) is recoverable: the
    # LLM abstained on those paragraphs, so — exactly like the single/table
    # non-title path (review D3) — we neither grant nor veto them and they
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
    # enter neither heading_records nor body_records below, so they are
    # neither granted nor vetoed downstream).
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
