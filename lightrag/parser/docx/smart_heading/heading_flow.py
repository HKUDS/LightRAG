"""Heading candidacy, font-size leveling, and numbering alignment (§2.2.5).

This module hosts the per-sub-document pipeline steps:

  step 1  candidate gate — outline / font-size tiers / numbering / bold /
          centered-companion channels, strong-body demotion, homophone and
          caption vetoes, CB5 low-confidence tier gating, CB1 density
          re-estimation;
  step 2  pure font-size leveling (size bands → consecutive levels; intra-
          band order: centered-unnumbered < numbered (priority / unit
          sub-order / MultiLevelNum raw level on an independent track) <
          uncentered-unnumbered);
  step 3  top-level backfill (CnChapter/EnChapter/EnNum absorbed as
          MultiLevelNum raw level 1 on ordinal-linkage evidence);
  step 4  same-series numbering level alignment (open-interval series,
          level mode with shallow tie-break, subtree shift).

Anchoring (two-round outline locking, §2.2.6), merge/demote (§2.2.7) and
skeleton correction (§2.2.8) build on these decisions in later steps.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
    DEFAULT_DOCX_SMART_CB2_BODY_RATIO,
    DEFAULT_DOCX_SMART_CB2_OUTLINE_RATIO,
    DEFAULT_DOCX_SMART_CONFIDENCE_RATIO,
    DEFAULT_DOCX_SMART_DENSITY_BASELINE_MARGIN,
    DEFAULT_DOCX_SMART_DENSITY_MAX,
    DEFAULT_DOCX_SMART_HEADING_MAX_CHARS,
    DEFAULT_DOCX_SMART_MIN_INTER_HEADING_CHARS,
    DEFAULT_DOCX_SMART_SEQ_BREAK_PARAS,
)
from lightrag.utils import get_content_summary, logger

from . import guardrails
from .features import effective_font_size_pt
from .style_key import (
    CN_CHAPTER,
    EN_CHAPTER,
    EN_NUM,
    MULTI_LEVEL_NUM,
    FsBase,
    NumberingClassification,
    classify_numbering,
    compute_fs_base,
    reclassify_single_char_romans,
    unit_rank,
)

_CENTER_RUN_MAX = 4  # anti-poetry: longer centered runs lose the channel

#: §2.3.5 deviation: the re-judgment ledger normally stores only a content
#: hash, never plaintext. This is the max character length of the plaintext
#: preview added alongside that hash for human-readable auditing.
_AUDIT_SUMMARY_CHARS = 60


def _env_float(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, "") or default)
    except ValueError:
        return default


def _env_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, "") or default)
    except ValueError:
        return default


def _para_hash(text: str) -> str:
    """Audit hash for one paragraph (same shape as the TOC audit rows)."""
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


@dataclass
class HeadingDecision:
    """Mutable per-record heading verdict threaded through the pipeline."""

    record_index: int
    text: str
    is_heading: bool = False
    level: int | None = None
    font_size_pt: float | None = None
    outline_level: int | None = None  # 0-based physical outline
    numbering: NumberingClassification | None = None
    centered: bool = False
    all_bold: bool = False
    anchored: bool = False  # outline-locked (set by §2.2.6)
    pre_anchor_level: int | None = None  # level before round-1 anchoring
    is_title_block: bool = False
    composed_heading: str | None = None  # title block: the compound heading
    title_parts: tuple[str, ...] = ()
    member_indices: tuple[int, ...] = ()
    use_raw_text: bool = False
    absorbed: bool = False  # merged into a preceding heading (§2.2.7)
    rule_trail: list[str] = field(default_factory=list)

    def note(self, rule: str) -> None:
        self.rule_trail.append(rule)

    @property
    def anchor_delta(self) -> int:
        """Round-1 level shift (post - pre); 0 when not anchored."""
        if not self.anchored or self.pre_anchor_level is None or self.level is None:
            return 0
        return self.level - self.pre_anchor_level


# ---------------------------------------------------------------------------
# FS_base statistics over records
# ---------------------------------------------------------------------------


def _record_weight(rec: Any) -> int:
    """Char weight for FS_base statistics (§2.2.2).

    Prefers the visible source-text count (w:t only) so auto-numbering labels
    and ``<sup>``/``<equation>``/``<drawing>``/``<table>`` placeholder markup
    never skew the body-size mode. Falls back to the stripped assembly text
    when the count was not computed (manually built records / smart-off)."""
    vc = getattr(rec, "visible_char_count", None)
    if vc is not None:
        return vc
    return len((rec.text or "").strip())


def _stat_paragraphs(records: Sequence[Any], indices: Sequence[int]) -> list[int]:
    """Record indices that participate in FS_base statistics (§2.2.2):
    paragraphs only — tables/cells, empty paragraphs and TOC lines are out."""
    out = []
    for i in indices:
        rec = records[i]
        if rec.kind != "para" or rec.is_toc_field or rec.is_toc_link:
            continue
        if _record_weight(rec) <= 0:
            continue
        out.append(i)
    return out


def document_fs_base(
    records: Sequence[Any],
    indices: Sequence[int],
    *,
    confidence_ratio: float | None = None,
) -> FsBase:
    ratio = (
        confidence_ratio
        if confidence_ratio is not None
        else _env_float(
            "DOCX_SMART_CONFIDENCE_RATIO", DEFAULT_DOCX_SMART_CONFIDENCE_RATIO
        )
    )
    pairs = [
        (records[i].font_size_pt, _record_weight(records[i]))
        for i in _stat_paragraphs(records, indices)
        if records[i].font_size_pt is not None
    ]
    return compute_fs_base(pairs, confidence_ratio=ratio)


def _log_physical_feature_summary(
    records: Sequence[Any], body_indices: Sequence[int], doc_fs: FsBase
) -> None:
    """Emit a one-line document-level summary once physical features are
    extracted: the body FS_base (over body paragraphs, its natural scope),
    the total paragraph count and the per-outline-level histogram (over all
    ``para`` records, the physical view). Outline levels are shown 1-based
    (``L1`` == outline_level 0, i.e. Word "Level 1"); paragraphs without an
    outline fall in the ``none`` bucket."""
    total_paras = 0
    outline_hist: dict[int | None, int] = {}
    for rec in records:
        if rec.kind != "para":
            continue
        total_paras += 1
        lv = rec.outline_level
        outline_hist[lv] = outline_hist.get(lv, 0) + 1

    parts = [
        f"L{lv + 1}: {outline_hist[lv]}"
        for lv in sorted(k for k in outline_hist if k is not None)
    ]
    if None in outline_hist:
        parts.append(f"none: {outline_hist[None]}")
    levels_str = ", ".join(parts) if parts else "(none)"

    fs_str = f"{doc_fs.size_pt}pt" if doc_fs.size_pt is not None else "unknown"
    confidence = "high" if doc_fs.confidence_high else "low"
    logger.info(
        "[smart_heading] physical features: FS_base=%s (confidence=%s), "
        "%d paragraphs {%s}",
        fs_str,
        confidence,
        total_paras,
        levels_str,
    )


# ---------------------------------------------------------------------------
# step 1 — candidate gate
# ---------------------------------------------------------------------------

#: styleKeys whose strong-body check runs at recognition time; the rest
#: defer to the post-merge sweep (§2.2.5 note / §2.2.7).
EARLY_STRONG_BODY_KEYS = frozenset({CN_CHAPTER, EN_CHAPTER, "CnClause", "EnClause"})


@dataclass
class GateResult:
    decisions: list[HeadingDecision]  # admitted candidates, document order
    fs_base: FsBase
    density: float  # candidates / non-empty paragraphs
    non_empty: int = 0  # denominator behind ``density`` (non-empty, non-TOC paras)
    cb1_reestimated: bool = False
    cb1_tripped: bool = False  # still too dense after one re-estimation
    #: CB1 spared the sub-document without re-estimation: the raw first-pass
    #: density was over threshold, but a look-ahead of the downstream
    #: strong-body + CB2 series-propagation sweep projected it back under the
    #: bar, so the trip was withheld (§2.3.3). Real demotion is left to the
    #: normal post-merge sweep.
    cb1_strong_body_recovered: bool = False
    cb1_new_fs: float | None = None  # re-estimated FS_base (audit, §2.3.5)
    #: Second-pass mean body chars between adjacent candidates (observational
    #: only — the §2.3.3 recheck is density-only; None when < 2 candidates).
    cb1_new_inter_chars: float | None = None
    #: Recognition-time strong-body demotions. Kept apart from ``decisions``
    #: (they must not be leveled/anchored) but carry a rule-tagged
    #: ``HeadingDecision`` so every demotion is explicit and audited, never a
    #: silent drop. OUTLINE members bear the §2.3.2 I2 weight — they were
    #: baseline headings, so they re-render from ``full_text_raw``
    #: (``use_raw_text``) and the retention check sees the rule trail (review
    #: C2). NON-outline members were never baseline headings: they exist purely
    #: so the per-paragraph audit ledger matches the demotion counter, and stay
    #: output-neutral (``use_raw_text`` off → same body rendering as no
    #: decision).
    demoted: list[HeadingDecision] = field(default_factory=list)


def _effective_size(rec: Any) -> float | None:
    """Candidate size (shared with title_block via features)."""
    return effective_font_size_pt(rec)


def gate_candidates(
    records: Sequence[Any],
    indices: Sequence[int],
    *,
    fs_base: FsBase,
    strong_body: Callable[[str], str | None] | None = None,
    numbering_veto: Callable[[Any, str], str | None] | None = None,
    caption_veto: Callable[[str], str | None] | None = None,
    warnings: dict | None = None,
    force_low_confidence: bool = False,
    cb1_reestimate: bool = False,
    llm_heading_grants: set[int] = frozenset(),
    llm_body_vetoes: set[int] = frozenset(),
) -> GateResult:
    """§2.2.5 step 1 over one sub-document (``indices`` into ``records``).

    ``llm_heading_grants`` are record indices the title-block LLM classified
    as plain headings — identity only, admitted as candidates but leveled by
    the same downstream flow and still subject to strong-body demotion.
    ``llm_body_vetoes`` are indices the LLM classified as body — candidate
    identity revoked (§2.2.4 "赋予或撤销"); outline paragraphs never appear
    here (title_block already reroutes them per I2).

    ``cb1_reestimate`` switches to the CB1 second-pass semantics (§2.3.3):
    same-size composite paths (series / bold / centered companions) are all
    disabled, while sizes strictly above the re-estimated FS_base remain
    auto-admitted; outline pass-through and LLM grants are unaffected.
    """
    strong_body = strong_body or guardrails.strong_body_reason
    numbering_veto = numbering_veto or guardrails.numbering_homophone_reason
    caption_veto = caption_veto or guardrails.caption_prefix_reason

    high_conf = fs_base.confidence_high and not force_low_confidence
    fs = fs_base.size_pt

    para_indices = [
        i
        for i in indices
        if records[i].kind == "para"
        and not records[i].is_toc_field
        and not records[i].is_toc_link
    ]

    # Pre-classify numbering (with homophone veto) for every paragraph.
    numbering: dict[int, NumberingClassification | None] = {}
    veto_notes: dict[int, str] = {}
    for i in para_indices:
        cls = classify_numbering(records[i].text)
        if cls is not None:
            veto = numbering_veto(cls, records[i].text)
            if veto is not None:
                veto_notes[i] = veto
                cls = None
        numbering[i] = cls
    # Deferred single-char roman reclassification (needs the whole list).
    reclassified = reclassify_single_char_romans([numbering[i] for i in para_indices])
    for i, cls in zip(para_indices, reclassified):
        numbering[i] = cls

    # Series population per key (含自身): the "成套编号" gate.
    series_count: dict[tuple, int] = {}
    for i in para_indices:
        cls = numbering[i]
        if cls is not None:
            series_count[cls.series_key()] = series_count.get(cls.series_key(), 0) + 1

    def _in_series(i: int) -> bool:
        cls = numbering[i]
        return cls is not None and series_count.get(cls.series_key(), 0) >= 2

    strong_cache: dict[int, str | None] = {}

    def _strong(i: int) -> str | None:
        if i not in strong_cache:
            strong_cache[i] = strong_body(records[i].text)
        return strong_cache[i]

    def _short(i: int) -> bool:
        # Candidate shortness reuses the strong-body length cap.
        try:
            cap = int(
                os.getenv("DOCX_SMART_HEADING_MAX_CHARS", "")
                or DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
            )
        except ValueError:
            cap = DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
        return guardrails.weighted_char_length(records[i].text.strip()) <= cap

    def _title_short(i: int) -> bool:
        from .title_block import TITLE_LINE_MAX_WEIGHTED_CHARS

        return (
            guardrails.weighted_char_length(records[i].text.strip())
            <= TITLE_LINE_MAX_WEIGHTED_CHARS
        )

    # Centered-companion channel: base-shape lines grouped into runs
    # (consecutive without an intervening body paragraph); runs longer than
    # _CENTER_RUN_MAX lose the channel; companions must come from a
    # DIFFERENT run (adjacent centered lines are never companions).
    centered_shape: dict[int, bool] = {}
    for i in para_indices:
        rec = records[i]
        centered_shape[i] = (
            rec.alignment == "center"
            and i not in llm_body_vetoes  # confirmed body is no companion
            and _title_short(i)
            and _strong(i) is None
            and numbering[i] is None
            and caption_veto(records[i].text) is None
        )
    runs: list[list[int]] = []
    current_run: list[int] = []
    for i in para_indices:
        if centered_shape[i]:
            current_run.append(i)
        elif records[i].kind == "para" and _record_weight(records[i]) > 0:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)
    run_of: dict[int, int] = {}
    for ridx, run in enumerate(runs):
        for i in run:
            run_of[i] = ridx

    def _centered_channel(i: int) -> bool:
        if not centered_shape.get(i):
            return False
        my_run = run_of.get(i)
        if my_run is None or len(runs[my_run]) > _CENTER_RUN_MAX:
            return False
        my_size = _effective_size(records[i])
        for ridx, run in enumerate(runs):
            if ridx == my_run or len(run) > _CENTER_RUN_MAX:
                continue
            for j in run:
                if _effective_size(records[j]) == my_size:
                    return True
        return False

    # Weak-signal companions (P2): count of same-size weak-tier paragraphs.
    # A strong-body paragraph will be demoted and so is NOT a companion —
    # spec §2.3.4 counts only "不含强正文特征的弱信号段落". Excluding it here
    # prevents a lone +0.5 paragraph from falsely pairing with a same-size
    # sentence/long paragraph that is about to be dropped.
    weak_size_count: dict[float, int] = {}
    if fs is not None:
        for i in para_indices:
            size = _effective_size(records[i])
            if size is not None and size == fs + 0.5 and _strong(i) is None:
                weak_size_count[size] = weak_size_count.get(size, 0) + 1

    decisions: list[HeadingDecision] = []
    demoted: list[HeadingDecision] = []
    admitted: set[int] = set()

    for i in para_indices:
        rec = records[i]
        text = rec.text
        size = _effective_size(rec)
        cls = numbering[i]
        rule: str | None = None

        if rec.outline_level is not None:
            # P3 never vetoes an outline paragraph: silently skipping it here
            # would strip a baseline heading without a decision/warning and
            # trip the I2 machine check (A5). Wrong captions with an outline
            # fall to the strong-body / guardrail path like any outline para.
            rule = "outline"
        elif i in llm_body_vetoes:
            continue  # §2.2.4: LLM body vote revokes candidate identity
        elif caption_veto(text) is not None:
            continue  # P3: never a heading
        elif i in llm_heading_grants:
            rule = "llm_grant"
        elif fs is None or size is None:
            rule = None
        elif cb1_reestimate:
            # CB1 second pass: only a size strictly above the re-estimated
            # body size auto-admits; all same-size composite paths are off.
            if size > fs:
                rule = "cb1_size"
        elif high_conf:
            if size >= fs + 1.0:
                rule = "size_strong"
            elif size == fs + 0.5:
                if _in_series(i):
                    rule = "weak_plus_series"
                elif rec.all_bold and _short(i):
                    rule = "weak_plus_bold"
                elif _centered_channel(i):
                    rule = "weak_plus_center"
                elif weak_size_count.get(size, 0) >= 2:
                    rule = "weak_pair"  # P2 companions
            elif size == fs:
                if _in_series(i):
                    rule = "base_series"
                elif rec.all_bold and _short(i):
                    rule = "base_bold"
                elif _centered_channel(i):
                    rule = "base_center"
            # size < fs without outline: never a heading (§2.2.2 necessary
            # condition under high confidence)
        else:
            # Low confidence (CB5 / CB1): no size tier stands alone; require
            # size >= fs AND a composite signal.
            if size >= fs:
                if _in_series(i):
                    rule = "lowconf_series"
                elif rec.all_bold and _short(i):
                    rule = "lowconf_bold"
                elif _centered_channel(i):
                    rule = "lowconf_center"

        if rule is None:
            continue

        # Strong-body demotion at recognition time: unnumbered candidates
        # and chapter/clause-class numbered ones; other numbered styleKeys
        # defer to the post-merge sweep (§2.2.7).
        check_now = cls is None or cls.style_key in EARLY_STRONG_BODY_KEYS
        if check_now:
            reason = _strong(i)
            if reason is not None:
                if warnings is not None:
                    warnings["smart_strong_body_demotions"] = (
                        warnings.get("smart_strong_body_demotions", 0) + 1
                    )
                # Every recognition-time strong-body demotion leaves an
                # EXPLICIT, rule-tagged, per-paragraph decision — never a
                # silent drop — so the per-paragraph audit ledger matches the
                # demotion counter above. The demoted decision is kept out of
                # the leveled candidate list. Outline members bear the I2
                # (§2.3.2) weight: they were baseline headings, so they
                # re-render from full_text_raw (``use_raw_text``), the retention
                # check sees the rule trail (review C2), and the demotion is
                # warned. Non-outline candidates were never baseline headings —
                # they stay output-neutral (``use_raw_text`` off → same body
                # rendering as no decision) and emit no I2 warning.
                is_outline = rec.outline_level is not None
                dem = HeadingDecision(
                    record_index=i,
                    text=text,
                    is_heading=False,
                    outline_level=rec.outline_level,
                    numbering=cls,
                    use_raw_text=is_outline,
                )
                dem.note(reason)
                dem.note("strong_body_demoted")
                if i in veto_notes:
                    dem.note(veto_notes[i])
                demoted.append(dem)
                if is_outline:
                    logger.warning(
                        "[smart_heading] I2: outline paragraph demoted to body "
                        "by strong-body feature (%s)",
                        reason,
                    )
                continue

        decision = HeadingDecision(
            record_index=i,
            text=text,
            is_heading=True,
            font_size_pt=size,
            outline_level=rec.outline_level,
            numbering=cls,
            centered=rec.alignment == "center",
            all_bold=rec.all_bold,
        )
        decision.note(rule)
        if i in veto_notes:
            decision.note(veto_notes[i])
        decisions.append(decision)
        admitted.add(i)

    non_empty = len(para_indices)
    density = (len(decisions) / non_empty) if non_empty else 0.0
    return GateResult(
        decisions=decisions,
        fs_base=fs_base,
        density=density,
        non_empty=non_empty,
        demoted=demoted,
    )


def _avg_inter_heading_body_chars(
    records: Sequence[Any], indices: Sequence[int], result: GateResult
) -> float | None:
    """Mean visible body chars between adjacent admitted candidates.

    ``None`` when fewer than two candidates (no adjacent pair exists).
    """
    cand = sorted(d.record_index for d in result.decisions)
    if len(cand) < 2:
        return None
    admitted = set(cand)
    body_chars = 0
    for i in indices:
        if cand[0] < i < cand[-1] and i not in admitted:
            rec = records[i]
            if rec.kind == "para":
                # CJK-weighted (1 Chinese = 3 English), same yardstick as
                # every other length threshold in the spec (§2.2.4/§2.2.5).
                body_chars += guardrails.weighted_char_length(rec.text)
    return body_chars / (len(cand) - 1)


def gate_with_cb1(
    records: Sequence[Any],
    indices: Sequence[int],
    *,
    fs_base: FsBase,
    baseline_density: float = 0.0,
    warnings: dict | None = None,
    **gate_kwargs: Any,
) -> GateResult:
    """Run the gate; on CB1 overflow (density > threshold, or the average
    body chars between adjacent candidates below
    ``DOCX_SMART_MIN_INTER_HEADING_CHARS`` — trigger side only, §2.3.3):

    1. Look-ahead: project the downstream strong-body + CB2 series-propagation
       sweep onto a throwaway copy of the candidates and re-measure density.
       The raw first-pass density counts numbered body-in-disguise (e.g. a run
       of ``CnParentNum`` "(一)…。" clauses) whose strong-body demotion is
       deferred to that post-merge sweep — and the sweep is skipped once the
       breaker trips. If the projected density falls back under threshold, the
       sub-document is not truly over-detected: withhold the trip, keep the
       first-pass decisions (``cb1_strong_body_recovered``), and let the normal
       sweep do the real demotion. Genuine same-size numbered headings (``一、``)
       survive instead of being wiped by the blanket re-estimation below.
    2. Otherwise re-estimate FS_base once by folding the candidates' dominant
       size into the body and re-gate in ``cb1_reestimate`` mode (composite
       same-size paths off, sizes above the new body size still auto-admit).
       Still over the density threshold → mark the breaker tripped.

    The density threshold is baseline-aware: ``max(floor, baseline_density +
    margin)`` where the floor is ``DOCX_SMART_DENSITY_MAX`` and the margin is
    ``DOCX_SMART_DENSITY_BASELINE_MARGIN``. A richly-outlined sub-document
    admits more candidates by construction, so its baseline outline density
    (plus the margin) raises the ceiling above the flat floor. The SAME
    threshold gates both the first-pass trip and the re-gate recheck.
    """
    density_floor = _env_float("DOCX_SMART_DENSITY_MAX", DEFAULT_DOCX_SMART_DENSITY_MAX)
    baseline_margin = _env_float(
        "DOCX_SMART_DENSITY_BASELINE_MARGIN", DEFAULT_DOCX_SMART_DENSITY_BASELINE_MARGIN
    )
    threshold = max(density_floor, baseline_density + baseline_margin)
    min_inter = _env_int(
        "DOCX_SMART_MIN_INTER_HEADING_CHARS",
        DEFAULT_DOCX_SMART_MIN_INTER_HEADING_CHARS,
    )
    result = gate_candidates(
        records, indices, fs_base=fs_base, warnings=warnings, **gate_kwargs
    )
    inter_chars = _avg_inter_heading_body_chars(records, indices, result)
    sparse_body = inter_chars is not None and inter_chars < min_inter
    if result.density <= threshold and not sparse_body:
        return result

    # CB1 look-ahead (§2.3.3): the raw first-pass density is over the bar, but
    # numbered body-in-disguise (e.g. a run of ``CnParentNum`` "(一)…。" clauses)
    # is still counted as candidates here — its strong-body demotion is deferred
    # to the post-merge sweep, which runs LATER and is skipped entirely once the
    # breaker trips. Project that sweep forward on a throwaway copy (strong-body
    # + CB2 series propagation) and re-measure: if the projected density falls
    # back under threshold, the sub-document is not truly over-detected once
    # body-in-disguise is removed. Withhold the trip and let the normal
    # downstream sweep do the real demotion — this keeps genuine same-size
    # numbered headings (``一、``) that the blanket re-estimation would wipe out.
    #
    # The projection runs the FULL strong-body check (so multi-sentence body is
    # caught too), but ``strong_body`` is a per-parse memo (run_smart_heading):
    # every verdict computed here is cached by paragraph text, so the
    # authoritative post-merge sweep reads them back instead of re-invoking
    # spaCy. Net spaCy cost stays one pass per unique paragraph across the gate,
    # this look-ahead, and the downstream sweep.
    scratch = [copy.deepcopy(d) for d in result.decisions]
    assign_levels_by_size(scratch)  # CB2 series-group intervals need d.level
    demote_strong_body_headings(
        scratch, strong_body=gate_kwargs.get("strong_body"), warnings=None
    )
    survivors = [d for d in scratch if d.is_heading]
    proj_density = (len(survivors) / result.non_empty) if result.non_empty else 0.0
    proj = GateResult(decisions=survivors, fs_base=fs_base, density=proj_density)
    proj_inter = _avg_inter_heading_body_chars(records, indices, proj)
    proj_sparse = proj_inter is not None and proj_inter < min_inter
    if proj_density <= threshold and not proj_sparse:
        result.cb1_strong_body_recovered = True
        if warnings is not None:
            warnings["smart_cb1_strong_body_recovered"] = (
                warnings.get("smart_cb1_strong_body_recovered", 0) + 1
            )
        logger.info(
            "[smart_heading] CB1: raw density %.0f%% over threshold %.0f%%, but "
            "strong-body + series propagation projects %.0f%% — not tripping "
            "(deferred to the post-merge demotion sweep)",
            result.density * 100,
            threshold * 100,
            proj_density * 100,
        )
        return result

    # CB1: the dominant "heading" size is body in disguise. Fold it in:
    # the new FS_base is the candidates' char-weighted dominant size when
    # larger than the current one.
    cand_pairs = [
        (d.font_size_pt, _record_weight(records[d.record_index]))
        for d in result.decisions
        if d.font_size_pt is not None
    ]
    cand_mode = compute_fs_base(cand_pairs).size_pt
    new_size = fs_base.size_pt
    if cand_mode is not None and (new_size is None or cand_mode > new_size):
        new_size = cand_mode
    reestimated = FsBase(
        size_pt=new_size,
        dominant_ratio=fs_base.dominant_ratio,
        confidence_high=fs_base.confidence_high,
    )
    if warnings is not None:
        warnings["smart_cb1_reestimated"] = warnings.get("smart_cb1_reestimated", 0) + 1
    logger.warning(
        "[smart_heading] CB1 tripped (density %.0f%% vs threshold %.0f%%%s); "
        "re-estimated FS_base to %s and disabled same-size composite paths",
        result.density * 100,
        threshold * 100,
        (
            f", avg body chars between headings {inter_chars:.0f} < {min_inter}"
            if sparse_body
            else ""
        ),
        new_size,
    )
    second = gate_candidates(
        records,
        indices,
        fs_base=reestimated,
        warnings=warnings,
        cb1_reestimate=True,
        **gate_kwargs,
    )
    second.cb1_reestimated = True
    second.cb1_new_fs = new_size
    # Second-pass inter-heading spacing: observational only (the §2.3.3
    # recheck stays density-only — this does NOT feed the trip decision).
    second.cb1_new_inter_chars = _avg_inter_heading_body_chars(records, indices, second)
    if second.density > threshold:
        second.cb1_tripped = True
        if warnings is not None:
            warnings["smart_cb1_tripped"] = warnings.get("smart_cb1_tripped", 0) + 1
    # Re-gate outcome: the first warning above fires BEFORE this second pass
    # runs, so it can only report the first-pass density. Surface the
    # post-re-estimation density AND inter-heading spacing here — WARNING when
    # it stayed over the bar (this sub-document falls back to outline-only),
    # INFO when it converged.
    log = logger.warning if second.cb1_tripped else logger.info
    inter_str = (
        f"{second.cb1_new_inter_chars:.0f}"
        if second.cb1_new_inter_chars is not None
        else "n/a"
    )
    log(
        "[smart_heading] CB1 re-gate: density %.0f%%->%.0f%% (threshold %.0f%%), "
        "avg body chars between headings %s, %s",
        result.density * 100,
        second.density * 100,
        threshold * 100,
        inter_str,
        (
            "still over -> outline-only fallback for this sub-document"
            if second.cb1_tripped
            else "converged, kept re-estimated pass"
        ),
    )
    return second


# ---------------------------------------------------------------------------
# step 2 — pure font-size leveling
# ---------------------------------------------------------------------------


def assign_levels_by_size(decisions: list[HeadingDecision]) -> None:
    """Assign levels from size bands (§2.2.5 step 2), in place.

    Bands are the distinct effective sizes, descending; each band expands
    into consecutive levels: centered-unnumbered first (shallowest), then
    the numbering classes — non-MultiLevelNum keys ordered by (priority,
    unit sub-order) on one track while MultiLevelNum raw levels advance an
    independent track anchored at the numbering base (collisions between
    the two tracks are legal) — and uncentered-unnumbered last (deepest).
    The next band starts one below the previous band's deepest level.
    """
    if not decisions:
        return
    sized = [d for d in decisions if d.font_size_pt is not None]
    unsized = [d for d in decisions if d.font_size_pt is None]
    bands = sorted({d.font_size_pt for d in sized}, reverse=True)

    next_level = 1
    band_levels: dict[float, dict[str, Any]] = {}
    for band in bands:
        members = [d for d in sized if d.font_size_pt == band]
        has_centered_plain = any(d.centered and d.numbering is None for d in members)
        has_uncentered_plain = any(
            (not d.centered) and d.numbering is None for d in members
        )
        numbered = [d for d in members if d.numbering is not None]

        level_map: dict[str, Any] = {}
        cursor = next_level
        if has_centered_plain:
            level_map["centered_plain"] = cursor
            cursor += 1

        # Numbering classes present in this band, ordered by
        # (priority, unit sub-order); MultiLevelNum occupies ONE slot on
        # this track while its raw levels advance independently.
        class_keys: list[tuple] = []
        seen_keys: set[tuple] = set()
        for d in numbered:
            cls = d.numbering
            if cls.style_key == MULTI_LEVEL_NUM:
                key = (MULTI_LEVEL_NUM,)
            else:
                key = (cls.style_key, cls.unit)
            if key not in seen_keys:
                seen_keys.add(key)
                class_keys.append(key)

        def _class_sort(key: tuple):
            from .style_key import STYLE_KEY_PRIORITY

            style = key[0]
            rank = unit_rank(style, key[1]) if len(key) > 1 else None
            return (STYLE_KEY_PRIORITY[style], rank if rank is not None else 99)

        class_keys.sort(key=_class_sort)

        numbering_base = cursor
        class_level: dict[tuple, int] = {}
        for offset, key in enumerate(class_keys):
            class_level[key] = numbering_base + offset
        cursor = numbering_base + len(class_keys) if class_keys else cursor

        mln_raws = sorted(
            {
                d.numbering.raw_level
                for d in numbered
                if d.numbering.style_key == MULTI_LEVEL_NUM
                and d.numbering.raw_level is not None
            }
        )
        mln_base_level = class_level.get((MULTI_LEVEL_NUM,))
        mln_raw_level: dict[int, int] = {}
        if mln_base_level is not None and mln_raws:
            min_raw = mln_raws[0]
            for raw in mln_raws:
                mln_raw_level[raw] = mln_base_level + (raw - min_raw)
            cursor = max(cursor, max(mln_raw_level.values()) + 1)

        if has_uncentered_plain:
            level_map["uncentered_plain"] = cursor
            cursor += 1

        band_levels[band] = {
            "map": level_map,
            "classes": class_level,
            "mln": mln_raw_level,
        }
        next_level = cursor

    for d in sized:
        info = band_levels[d.font_size_pt]
        cls = d.numbering
        if cls is None:
            key = "centered_plain" if d.centered else "uncentered_plain"
            d.level = info["map"][key]
        elif cls.style_key == MULTI_LEVEL_NUM and cls.raw_level in info["mln"]:
            d.level = info["mln"][cls.raw_level]
        elif cls.style_key == MULTI_LEVEL_NUM:
            d.level = info["classes"][(MULTI_LEVEL_NUM,)]
        else:
            d.level = info["classes"][(cls.style_key, cls.unit)]
    # Size-traceless candidates park at the deepest assigned level.
    if unsized:
        deepest = max((d.level or 1) for d in sized) if sized else 1
        for d in unsized:
            d.level = deepest


# ---------------------------------------------------------------------------
# step 3 — top-level backfill
# ---------------------------------------------------------------------------

_BACKFILL_KEYS = (EN_NUM, CN_CHAPTER, EN_CHAPTER)


def backfill_top_level(
    decisions: list[HeadingDecision], *, warnings: dict | None = None
) -> None:
    """Absorb a parent numbering class as MultiLevelNum raw level 1 (§2.2.5
    step 3), in place. Runs per main-title scope (callers pass decisions of
    one scope). No-op without multi-level numbering."""
    mln = [
        d
        for d in decisions
        if d.numbering is not None and d.numbering.style_key == MULTI_LEVEL_NUM
    ]
    if not mln:
        return
    min_raw = min(
        (d.numbering.raw_level for d in mln if d.numbering.raw_level is not None),
        default=None,
    )
    if min_raw is None:
        return
    shallowest = [d for d in mln if d.numbering.raw_level == min_raw]
    # Size threshold: the mode of the shallowest-raw sizes; ties toward the
    # SMALLER size (spec: 以字号众数最小者为准).
    size_weights: dict[float, int] = {}
    for d in shallowest:
        if d.font_size_pt is not None:
            size_weights[d.font_size_pt] = size_weights.get(d.font_size_pt, 0) + 1
    if not size_weights:
        return
    threshold = min(
        (s for s, w in size_weights.items() if w == max(size_weights.values())),
    )
    shallow_level = min(
        (d.level for d in shallowest if d.level is not None), default=None
    )

    # Group ALL parent-class headings; the size threshold is one of two
    # admission channels below (ordinal linkage can waive it — parents typed
    # smaller than their children are a real-world anomaly, §2.2.5 step 3).
    groups: dict[tuple, list[HeadingDecision]] = {}
    for d in decisions:
        cls = d.numbering
        if cls is not None and cls.style_key in _BACKFILL_KEYS:
            groups.setdefault(cls.series_key(), []).append(d)
    if not groups:
        return

    def _ordinal_linkage(members: list[HeadingDecision]) -> int:
        """Count parent ordinals whose scope holds top-value-matching
        MultiLevelNum members; 0 on any counter-example."""
        ordered = sorted(members, key=lambda d: d.record_index)
        linked = set()
        for pos, parent in enumerate(ordered):
            n = parent.numbering.ordinal
            if n is None:
                continue
            scope_start = parent.record_index
            scope_end = (
                ordered[pos + 1].record_index
                if pos + 1 < len(ordered)
                else float("inf")
            )
            tops = {
                d.numbering.top_ordinal
                for d in mln
                if scope_start < d.record_index < scope_end
                and d.numbering.top_ordinal is not None
            }
            if not tops:
                continue
            if tops == {n}:
                linked.add(n)
            else:
                return 0  # counter-example: foreign top value in scope
        return len(linked)

    from .style_key import STYLE_KEY_PRIORITY

    # Two admission channels per class: size threshold OR ≥2 ordinal-linkage
    # pairs (waives the size gate). Exactly one linked pair never waives —
    # too weak, surfaced as a warning.
    eligible: list[tuple[tuple, list[HeadingDecision], int]] = []
    for key, members in groups.items():
        sizes = [d.font_size_pt for d in members if d.font_size_pt is not None]
        size_ok = bool(sizes) and max(sizes) >= threshold
        linked = _ordinal_linkage(members)
        if size_ok or linked >= 2:
            eligible.append((key, members, linked))
        elif linked == 1:
            if warnings is not None:
                warnings["smart_backfill_single_linkage"] = (
                    warnings.get("smart_backfill_single_linkage", 0) + 1
                )
            logger.warning(
                "[smart_heading] top-level backfill: only one ordinal "
                "linkage pair for %s — size threshold not waived",
                key,
            )
    if not eligible:
        return

    def _preference(item):
        key, members, linked = item
        max_size = max((d.font_size_pt or 0.0) for d in members)
        return (
            0 if (linked >= 2 and key[0] == EN_NUM) else 1,
            0 if linked >= 2 else 1,
            STYLE_KEY_PRIORITY[key[0]],
            -max_size,
        )

    eligible.sort(key=_preference)
    _key, members, _linked = eligible[0]

    from dataclasses import replace as _replace

    for d in members:
        d.numbering = _replace(
            d.numbering,
            style_key=MULTI_LEVEL_NUM,
            raw_level=1,
            top_ordinal=d.numbering.ordinal,
        )
        d.note("backfill_top_level")
        if shallow_level is not None and d.level is not None:
            if d.level >= shallow_level:
                new_level = max(1, shallow_level - 1)
                if shallow_level - 1 < 1 and warnings is not None:
                    warnings["smart_backfill_cannot_raise"] = (
                        warnings.get("smart_backfill_cannot_raise", 0) + 1
                    )
                if new_level != d.level:
                    d.level = new_level


# ---------------------------------------------------------------------------
# step 4 — same-series level alignment
# ---------------------------------------------------------------------------


def _collect_series_group(
    decisions: list[HeadingDecision], start_pos: int
) -> list[int]:
    """Open-interval members of the series starting at ``start_pos``.

    Walk forward from the series head: same-key headings join; any heading
    STRICTLY shallower than every current member closes the interval
    (level returned to an ancestor scope); title blocks (level 0) are hard
    boundaries. Deeper/equal headings of other kinds are allowed inside.
    """
    head = decisions[start_pos]
    key = head.numbering.series_key()
    group = [start_pos]
    min_level = head.level if head.level is not None else 1
    # §2.2.3 / DOCX_SMART_SEQ_BREAK_PARAS: a long run of body paragraphs
    # between same-key members closes the open sequence. Record-index gaps
    # are the proxy for "consecutive body paragraphs" (records between two
    # heading decisions are body/tables). 0 (the default) disables the rule.
    try:
        seq_break = int(
            os.getenv("DOCX_SMART_SEQ_BREAK_PARAS", "")
            or DEFAULT_DOCX_SMART_SEQ_BREAK_PARAS
        )
    except ValueError:
        seq_break = DEFAULT_DOCX_SMART_SEQ_BREAK_PARAS
    last_member_record = head.record_index
    for pos in range(start_pos + 1, len(decisions)):
        d = decisions[pos]
        if not d.is_heading:
            continue  # demoted/absorbed members are transparent
        if d.is_title_block or d.level == 0:
            break
        if d.numbering is not None and d.numbering.series_key() == key:
            if seq_break > 0 and d.record_index - last_member_record - 1 > seq_break:
                break  # the intervening body run closed this sequence
            group.append(pos)
            last_member_record = d.record_index
            if d.level is not None:
                min_level = min(min_level, d.level)
            continue
        if d.level is not None and d.level < min_level:
            break
    return group


def _shift_subtree(
    decisions: list[HeadingDecision], pos: int, old_level: int, delta: int
) -> None:
    """Shift the heading at ``pos`` by ``delta`` and its subtree with it."""
    decisions[pos].level = decisions[pos].level + delta
    for k in range(pos + 1, len(decisions)):
        d = decisions[k]
        if d.level is None or d.level <= old_level or d.is_title_block:
            break
        d.level += delta


def align_numbering_series(
    decisions: list[HeadingDecision], *, skip_anchored: bool = False
) -> None:
    """§2.2.5 step 4: align every same-series numbering group to its level
    mode (ties toward the SHALLOWER level), shifting subtrees along, in
    place. Scanning is head-to-tail; each group aligns once.

    ``skip_anchored=True`` keeps outline-locked members fixed (used by the
    §2.2.8 smoothing pass: unanchored members align to the anchored mode).
    """
    aligned: set[int] = set()
    pos = 0
    while pos < len(decisions):
        d = decisions[pos]
        if (
            d.numbering is None
            or pos in aligned
            or d.level is None
            or d.is_title_block
            or not d.is_heading
        ):
            pos += 1
            continue
        group = _collect_series_group(decisions, pos)
        aligned.update(group)
        anchored_levels = [decisions[g].level for g in group if decisions[g].anchored]
        pool = (
            anchored_levels
            if (skip_anchored and anchored_levels)
            else [decisions[g].level for g in group]
        )
        weights: dict[int, int] = {}
        for lv in pool:
            weights[lv] = weights.get(lv, 0) + 1
        top = max(weights.values())
        mode = min(lv for lv, w in weights.items() if w == top)
        # Align members (deepest first so subtree walks see pre-shift levels
        # of later members consistently; document order within same depth).
        for g in group:
            member = decisions[g]
            if member.level == mode:
                continue
            if skip_anchored and member.anchored:
                continue
            old = member.level
            _shift_subtree(decisions, g, old, mode - old)
            member.note("series_align")
        pos += 1


# ---------------------------------------------------------------------------
# §2.2.6 — two-round physical-outline anchoring
# ---------------------------------------------------------------------------


def _all_series_groups(decisions: list[HeadingDecision]) -> list[list[int]]:
    """All same-series open-interval groups, in first-member order."""
    seen: set[int] = set()
    groups: list[list[int]] = []
    for pos, d in enumerate(decisions):
        if d.numbering is None or pos in seen or d.is_title_block or not d.is_heading:
            continue
        group = _collect_series_group(decisions, pos)
        seen.update(group)
        groups.append(group)
    return groups


def anchor_outline_levels(
    decisions: list[HeadingDecision], *, warnings: dict | None = None
) -> bool:
    """§2.2.6: re-anchor size-derived levels onto the physical outline.

    Round 1 — non-numbered outlined headings snap to ``outlineLvl + 1``;
    a numbered series with ≥1 outlined member snaps the WHOLE series to the
    mode of its members' outline levels (ties toward the shallower level;
    CB3 warns when fewer than half the members carry an outline but still
    propagates). Anchored headings record their pre-anchor level.

    Round 2 — every maximal run of unanchored headings between anchors is
    adjusted: the leading anchor's shift is followed by its immediate
    subtree (stopping at the first heading not deeper than the anchor's
    new level); when the trailing anchor moved SHALLOWER (R2 < R1), the
    remaining sub-window shifts with it — as a whole when it still holds
    levels ≥ R1, else just enough (max(0, L0 - R2 + 1)) to close the gap.
    A trailing anchor that moved deeper adjusts nothing. Levels floor at 1
    (a floored shift records a compression warning). Every heading ends
    the pass outline-locked.

    Returns False (and changes nothing) when no heading carries an outline.
    """
    if not any(d.outline_level is not None and not d.is_title_block for d in decisions):
        return False

    # --- round 1 -----------------------------------------------------------
    for d in decisions:
        if d.is_title_block or d.numbering is not None:
            continue
        if d.outline_level is not None:
            d.pre_anchor_level = d.level
            d.level = d.outline_level + 1
            d.anchored = True
            d.note("anchor_outline_direct")

    for group in _all_series_groups(decisions):
        members = [decisions[p] for p in group]
        outlined = [m for m in members if m.outline_level is not None]
        if not outlined:
            continue
        weights: dict[int, int] = {}
        for m in outlined:
            lv = m.outline_level + 1
            weights[lv] = weights.get(lv, 0) + 1
        top = max(weights.values())
        mode = min(lv for lv, w in weights.items() if w == top)
        if len(outlined) * 2 < len(members):
            if warnings is not None:
                warnings["smart_cb3_low_outline_ratio"] = (
                    warnings.get("smart_cb3_low_outline_ratio", 0) + 1
                )
            logger.warning(
                "[smart_heading] CB3: only %d/%d members of a numbering "
                "series carry a physical outline; propagating level %d anyway",
                len(outlined),
                len(members),
                mode,
            )
        for m in members:
            m.pre_anchor_level = m.level
            m.level = mode
            m.anchored = True
            m.note("anchor_outline_series")

    # --- round 2 -----------------------------------------------------------
    def _floor_shift(d: HeadingDecision, delta: int) -> None:
        new_level = d.level + delta
        if new_level < 1:
            new_level = 1
            if warnings is not None:
                warnings["smart_anchor_floor_compressions"] = (
                    warnings.get("smart_anchor_floor_compressions", 0) + 1
                )
        d.level = new_level

    seq = [d for d in decisions if not d.is_title_block and d.level is not None]
    pos = 0
    n = len(seq)
    while pos < n:
        if seq[pos].anchored:
            pos += 1
            continue
        window_start = pos
        window_end = pos
        while window_end < n and not seq[window_end].anchored:
            window_end += 1
        window = seq[window_start:window_end]
        front = seq[window_start - 1] if window_start > 0 else None
        back = seq[window_end] if window_end < n else None

        cut = 0  # boundary between follow-zone and the trailing sub-window
        if front is not None and front.anchor_delta != 0:
            delta = front.anchor_delta
            # §2.2.6 round 2: the follow stops at a heading whose level is
            # equal to or shallower than the front anchor's PRE-adjust level
            # (the back-anchor branch already uses pre; A3 restores symmetry).
            stop_level = (
                front.pre_anchor_level
                if front.pre_anchor_level is not None
                else front.level
            )
            for k, d in enumerate(window):
                if d.level <= stop_level:
                    break
                _floor_shift(d, delta)
                d.note("anchor_follow_front")
                cut = k + 1

        if cut and warnings is not None:  # §2.3.5: window shift events
            warnings["smart_anchor_window_shifts"] = (
                warnings.get("smart_anchor_window_shifts", 0) + 1
            )

        sub = window[cut:]
        if back is not None and sub:
            r1 = (
                back.pre_anchor_level
                if back.pre_anchor_level is not None
                else back.level
            )
            r2 = back.level
            if r2 < r1:  # trailing anchor moved shallower
                if any(d.level >= r1 for d in sub):
                    delta = r2 - r1
                else:
                    deepest = max(d.level for d in sub)
                    delta = -max(0, deepest - r2 + 1)
                if delta:
                    for d in sub:
                        _floor_shift(d, delta)
                        d.note("anchor_follow_back")
                    if warnings is not None:
                        warnings["smart_anchor_window_shifts"] = (
                            warnings.get("smart_anchor_window_shifts", 0) + 1
                        )

        for d in window:
            d.anchored = True
        pos = window_end
    return True


# ---------------------------------------------------------------------------
# §2.2.7 — heading merge and strong-body demotion sweep
# ---------------------------------------------------------------------------

_MERGE_MAX_LINES = 4


def _is_cjk_char(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def _join_heading_texts(left: str, right: str) -> str:
    """CJK-aware join: no space between CJK boundaries, a space otherwise."""
    if left and right and _is_cjk_char(left[-1]) and _is_cjk_char(right[0]):
        return left + right
    return f"{left} {right}"


def merge_split_headings(
    decisions: list[HeadingDecision],
    records: Sequence[Any],
    *,
    warnings: dict | None = None,
) -> list[HeadingDecision]:
    """§2.2.7: re-join headings the author split for line-spacing looks.

    Adjacent same-level headings with the same font size (bold ignored)
    merge — "adjacent" tolerates ONE empty paragraph between them; a
    numbered heading may only START a merge (never be absorbed); the merged
    text (soft-break lines + paragraph splits) is capped at 4 lines.
    Returns the compacted decision list (absorbed members removed).
    """
    out: list[HeadingDecision] = []
    pos = 0
    while pos < len(decisions):
        cur = decisions[pos]
        if not cur.is_heading or cur.is_title_block:
            out.append(cur)
            pos += 1
            continue
        lines = cur.text.count("\n") + 1
        merged_members = [cur.record_index]
        k = pos + 1
        while k < len(decisions):
            nxt = decisions[k]
            if not nxt.is_heading or nxt.is_title_block:
                break
            if nxt.numbering is not None:
                break  # a numbered heading only ever starts a merge
            if nxt.level != cur.level or nxt.font_size_pt != cur.font_size_pt:
                break
            # Adjacency: between the two records only empty paragraphs, at
            # most one.
            between = records[merged_members[-1] + 1 : nxt.record_index]
            if len(between) > 1 or any(r.kind != "empty_para" for r in between):
                break
            nxt_lines = nxt.text.count("\n") + 1
            if lines + nxt_lines > _MERGE_MAX_LINES:
                break
            cur.text = _join_heading_texts(cur.text, nxt.text)
            lines += nxt_lines
            merged_members.append(nxt.record_index)
            cur.note("heading_merge")
            if warnings is not None:
                warnings["smart_heading_merges"] = (
                    warnings.get("smart_heading_merges", 0) + 1
                )
            k += 1
        if len(merged_members) > 1:
            cur.member_indices = tuple(merged_members)
        out.append(cur)
        pos = k if k > pos + 1 else pos + 1
    return out


def demote_strong_body_headings(
    decisions: list[HeadingDecision],
    *,
    strong_body: Callable[[str], str | None] | None = None,
    warnings: dict | None = None,
) -> None:
    """§2.2.7 post-merge sweep: every remaining heading is re-checked for
    strong-body features; hits demote to body. A numbered hit propagates the
    demotion to its whole same-series group unless CB2 trips (fewer than 20%
    of the group hit, or ≥50% of the group carries a physical outline) — a
    tripped breaker demotes only the hits themselves, with a warning."""
    strong_body = strong_body or guardrails.strong_body_reason
    body_ratio = _env_float(
        "DOCX_SMART_CB2_BODY_RATIO", DEFAULT_DOCX_SMART_CB2_BODY_RATIO
    )
    outline_ratio = _env_float(
        "DOCX_SMART_CB2_OUTLINE_RATIO", DEFAULT_DOCX_SMART_CB2_OUTLINE_RATIO
    )

    hits: set[int] = set()
    for pos, d in enumerate(decisions):
        if not d.is_heading or d.is_title_block:
            continue
        reason = strong_body(d.text)
        if reason is not None:
            hits.add(pos)
            d.note(reason)

    demote: set[int] = set(hits)
    handled_groups: set[int] = set()
    for pos in sorted(hits):
        d = decisions[pos]
        if d.numbering is None:
            continue
        group = _collect_series_group_containing(decisions, pos)
        gid = min(group)
        if gid in handled_groups:
            continue
        handled_groups.add(gid)
        group_hits = [g for g in group if g in hits]
        hit_share = len(group_hits) / len(group)
        outlined_share = sum(
            1 for g in group if decisions[g].outline_level is not None
        ) / len(group)
        if hit_share < body_ratio or outlined_share >= outline_ratio:
            if warnings is not None:
                warnings["smart_cb2_propagation_stopped"] = (
                    warnings.get("smart_cb2_propagation_stopped", 0) + 1
                )
            logger.warning(
                "[smart_heading] CB2: demotion NOT propagated to a numbering "
                "series (hit share %.0f%%, outlined share %.0f%%)",
                hit_share * 100,
                outlined_share * 100,
            )
            continue
        demote.update(group)
        if warnings is not None:  # §2.3.5: successful propagation events
            warnings["smart_cb2_propagations"] = (
                warnings.get("smart_cb2_propagations", 0) + 1
            )

    for pos in demote:
        d = decisions[pos]
        d.is_heading = False
        d.use_raw_text = True
        d.note("strong_body_demoted")
        if warnings is not None:
            warnings["smart_strong_body_demotions"] = (
                warnings.get("smart_strong_body_demotions", 0) + 1
            )


def _collect_series_group_containing(
    decisions: list[HeadingDecision], pos: int
) -> list[int]:
    """The full open-interval group that contains position ``pos``."""
    for group in _all_series_groups(decisions):
        if pos in group:
            return group
    return [pos]


# ---------------------------------------------------------------------------
# §2.2.8 — numbering skeleton correction, smoothing, clamping
# ---------------------------------------------------------------------------

#: Cross-class unit hierarchy for nesting-evidence edges (legal documents:
#: 章 holds 条 even though 条 numbering runs through chapters).
_GLOBAL_UNIT_ORDER = {
    "篇": 0,
    "部": 0,
    "编": 0,
    "卷": 0,
    "章": 1,
    "节": 2,
    "条": 3,
    "款": 4,
    "项": 5,
    "volume": 0,
    "part": 0,
    "chapter": 1,
    "section": 2,
}


@dataclass
class _SkeletonNode:
    key: tuple
    positions: list[int]  # positions into the decisions list
    fixed: bool = False
    snapshot: int = 1

    @property
    def first_pos(self) -> int:
        return self.positions[0]


def _habitual_shallower(a: _SkeletonNode, b: _SkeletonNode, decisions) -> bool:
    """Whether class ``a`` habitually sits shallower than ``b``."""
    from .style_key import STYLE_KEY_PRIORITY

    ca = decisions[a.first_pos].numbering
    cb = decisions[b.first_pos].numbering
    if ca.style_key == MULTI_LEVEL_NUM and cb.style_key == MULTI_LEVEL_NUM:
        return (ca.raw_level or 0) < (cb.raw_level or 0)
    ua = _GLOBAL_UNIT_ORDER.get(ca.unit) if ca.unit else None
    ub = _GLOBAL_UNIT_ORDER.get(cb.unit) if cb.unit else None
    if ua is not None and ub is not None and ua != ub:
        return ua < ub
    return STYLE_KEY_PRIORITY[ca.style_key] < STYLE_KEY_PRIORITY[cb.style_key]


def _nesting_evidence(
    a: _SkeletonNode, b: _SkeletonNode, decisions: list[HeadingDecision]
) -> bool:
    """All-three nesting evidence for edge a→b (§2.2.8)."""
    a_pos = sorted(a.positions)
    b_pos = sorted(b.positions)
    if len(a_pos) < 2:
        return False
    # (iii) no counter-example: a b member before the first a member is an
    # orphan.
    if b_pos[0] < a_pos[0]:
        return False
    # (i) alternating containment: every a scope holds ≥1 b member.
    scopes: list[tuple[int, float, list[int]]] = []
    for idx, start in enumerate(a_pos):
        end = a_pos[idx + 1] if idx + 1 < len(a_pos) else float("inf")
        inside = [p for p in b_pos if start < p < end]
        if not inside:
            return False
        scopes.append((start, end, inside))
    # (ii) ordinal shape, any one of:
    ca = decisions[a.first_pos].numbering
    cb = decisions[b.first_pos].numbering
    # (ii-a) b ordinals reset/start inside every a scope
    resets = all(
        decisions[inside[0]].numbering.ordinal == 1 for _s, _e, inside in scopes
    )
    if resets:
        return True
    # (ii-b) b is MultiLevelNum whose top value links to a's ordinal
    if cb.style_key == MULTI_LEVEL_NUM:
        links = all(
            decisions[a_start].numbering.ordinal is not None
            and {
                decisions[p].numbering.top_ordinal
                for p in inside
                if decisions[p].numbering.top_ordinal is not None
            }
            == {decisions[a_start].numbering.ordinal}
            for a_start, _e, inside in scopes
        )
        if links:
            return True
    # (ii-c) both carry unit words with a defined strong hierarchy
    ua = _GLOBAL_UNIT_ORDER.get(ca.unit) if ca.unit else None
    ub = _GLOBAL_UNIT_ORDER.get(cb.unit) if cb.unit else None
    return ua is not None and ub is not None and ua < ub


def correct_numbering_skeleton(
    decisions: list[HeadingDecision], *, warnings: dict | None = None
) -> list[dict]:
    """§2.2.8 step 1: single-shot skeleton solve over numbering families.

    Nodes: MultiLevelNum gets ONE node per raw level (document-wide); every
    other styleKey gets one node per open-interval series. A node holding an
    anchored member is FIXED. Edges: MultiLevelNum intrinsic raw k → k+1
    plus nesting-evidence edges. A cycle abandons the whole correction.
    Solve once in topological order (ties by first appearance):
    solved(b) = max(snapshot(b), max(solved(a)+1)); fixed nodes never move
    (a conflicting demand only warns — anchors out-rank size/evidence).
    Write-back uses floor semantics (members only ever get DEEPER), and each
    deepened skeleton heading entrains the unlocked non-skeleton headings
    behind it that were deeper than its snapshot.

    Returns audit entries [{rule, position, before, after}].
    """
    audit: list[dict] = []
    numbered_positions = [
        pos
        for pos, d in enumerate(decisions)
        if d.is_heading and d.numbering is not None and not d.is_title_block
    ]
    if not numbered_positions:
        return audit

    pre_levels = {pos: decisions[pos].level for pos in range(len(decisions))}

    # --- nodes -------------------------------------------------------------
    # Grouping is by series_key over the WHOLE sub-document (MultiLevelNum:
    # one node per raw level) — NOT by open interval: the level inversions
    # this pass exists to fix are exactly what would shatter interval
    # grouping into singletons. Mis-grouping safety rests on the all-three
    # evidence rule below, not on interval walls.
    nodes: list[_SkeletonNode] = []
    node_by_key: dict[tuple, _SkeletonNode] = {}
    mln_by_raw: dict[int, _SkeletonNode] = {}
    for pos in numbered_positions:
        cls = decisions[pos].numbering
        key = cls.series_key()
        node = node_by_key.get(key)
        if node is None:
            node = _SkeletonNode(key=key, positions=[])
            node_by_key[key] = node
            nodes.append(node)
            if cls.style_key == MULTI_LEVEL_NUM and cls.raw_level is not None:
                mln_by_raw[cls.raw_level] = node
        node.positions.append(pos)

    for node in nodes:
        levels = [pre_levels[p] for p in node.positions if pre_levels[p] is not None]
        if not levels:
            continue
        # Snapshot = the SHALLOWEST pre-correction member level (§2.2.8):
        # window shifts may have pushed part of a series deeper; the
        # shallowest member represents the series' un-displaced position.
        # Fixed nodes were already unified to the anchored mode by §2.2.6,
        # where min == mode.
        node.snapshot = min(levels)
        node.fixed = any(decisions[p].anchored for p in node.positions)

    # --- edges -------------------------------------------------------------
    index_of = {id(n): i for i, n in enumerate(nodes)}
    edges: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    raws = sorted(mln_by_raw)
    for raw in raws:
        if raw + 1 in mln_by_raw:
            edges[index_of[id(mln_by_raw[raw])]].add(index_of[id(mln_by_raw[raw + 1])])
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j or j in edges[i]:
                continue
            if not _habitual_shallower(a, b, decisions):
                continue
            if _nesting_evidence(a, b, decisions):
                edges[i].add(j)
            elif (
                a.snapshot is not None
                and b.snapshot is not None
                and b.snapshot <= a.snapshot
            ):
                # An inversion against the habitual order without enough
                # evidence to correct it — observed, never acted on (§2.3.5).
                if warnings is not None:
                    warnings["smart_skeleton_inversion_suspected"] = (
                        warnings.get("smart_skeleton_inversion_suspected", 0) + 1
                    )

    # --- cycle check + topological order ------------------------------------
    indeg = {i: 0 for i in range(len(nodes))}
    for i, outs in edges.items():
        for j in outs:
            indeg[j] += 1
    ready = sorted(
        (i for i in indeg if indeg[i] == 0), key=lambda i: nodes[i].first_pos
    )
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(edges[i], key=lambda j: nodes[j].first_pos):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
        ready.sort(key=lambda i: nodes[i].first_pos)
    if len(order) != len(nodes):
        if warnings is not None:
            warnings["smart_skeleton_cycle_abandoned"] = (
                warnings.get("smart_skeleton_cycle_abandoned", 0) + 1
            )
        logger.warning(
            "[smart_heading] numbering skeleton has contradictory evidence "
            "(cycle); correction abandoned for this sub-document"
        )
        return audit

    # --- solve --------------------------------------------------------------
    solved: dict[int, int] = {}
    preds: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for i, outs in edges.items():
        for j in outs:
            preds[j].append(i)
    for i in order:
        node = nodes[i]
        demanded = max((solved[p] + 1 for p in preds[i]), default=None)
        if node.fixed:
            solved[i] = node.snapshot
            if demanded is not None and demanded > node.snapshot:
                if warnings is not None:
                    warnings["smart_skeleton_anchor_conflict"] = (
                        warnings.get("smart_skeleton_anchor_conflict", 0) + 1
                    )
        else:
            solved[i] = max(node.snapshot, demanded or 0)

    # --- write-back with floor semantics + entrainment ----------------------
    skeleton_positions = {p for n in nodes for p in n.positions}
    for i in order:
        node = nodes[i]
        if node.fixed:
            continue
        for p in sorted(node.positions):
            d = decisions[p]
            snap_member = pre_levels[p] if pre_levels[p] is not None else node.snapshot
            final = max(snap_member, solved[i])
            delta = final - snap_member
            if final != d.level:
                audit.append(
                    {
                        "rule": "skeleton_correct",
                        "position": p,
                        "before": d.level,
                        "after": final,
                    }
                )
                d.level = final
                d.note("skeleton_correct")
            # Entrainment: unlocked, non-skeleton headings behind this member
            # that were deeper than its SNAPSHOT follow the push down.
            if delta > 0:
                for k in range(p + 1, len(decisions)):
                    e = decisions[k]
                    if not e.is_heading:
                        continue
                    if k in skeleton_positions or e.anchored or e.is_title_block:
                        break
                    if pre_levels[k] is not None and pre_levels[k] > (
                        pre_levels[p] or 0
                    ):
                        audit.append(
                            {
                                "rule": "skeleton_entrain",
                                "position": k,
                                "before": e.level,
                                "after": pre_levels[k] + delta,
                            }
                        )
                        e.level = pre_levels[k] + delta
                        e.note("skeleton_entrain")
                    else:
                        break
    return audit


def clamp_deep_levels(
    decisions: list[HeadingDecision], *, warnings: dict | None = None
) -> None:
    """§2.2.8 step 3: headings deeper than level 9 demote to body (no
    clamping-to-9 — that would break same-series level equality)."""
    for d in decisions:
        if d.is_heading and not d.is_title_block and (d.level or 0) > 9:
            d.is_heading = False
            d.use_raw_text = True
            d.note("clamp_gt9_demoted")
            if warnings is not None:
                warnings["smart_clamp_demotions"] = (
                    warnings.get("smart_clamp_demotions", 0) + 1
                )


# ---------------------------------------------------------------------------
# whole-document orchestration
# ---------------------------------------------------------------------------


@dataclass
class SmartHeadingResult:
    """Per-record decisions plus document-level artifacts for assembly."""

    decisions: dict[int, HeadingDecision]
    toc_indices: set[int]
    doc_title: str | None
    audit: dict
    fallback_sub_docs: int = 0


def _estimate_record_tokens(records: Sequence[Any], indices: Sequence[int]) -> int:
    from .title_block import _estimate_tokens

    return sum(
        _estimate_tokens(records[i].text)
        for i in indices
        if records[i].kind == "para" and records[i].text
    )


def _outline_only_decisions(
    records: Sequence[Any],
    indices: Sequence[int],
    *,
    imprint_marker: Callable[[str], str | None] | None = None,
    warnings: dict | None = None,
) -> list[HeadingDecision]:
    """Sub-document fallback: internal levels revert to outlineLvl-only
    (the baseline rule); global actions (title blocks, TOC removal,
    doc_title) are NOT undone (§3.4 mixed-output rules).

    公文版记 (imprint) lines are the one exception to the baseline revert: an
    outline paragraph opening with an imprint marker is demoted to body with
    an explicit rule-tagged decision — skipping it silently would read as an
    I2 violation (a baseline heading with no decision). Only the pure-regex
    imprint rule applies here; the fallback must not grow the full NLP
    strong-body demotion surface.
    """
    imprint_marker = imprint_marker or guardrails.imprint_marker_reason
    out = []
    for i in indices:
        rec = records[i]
        if rec.kind != "para" or rec.outline_level is None:
            continue
        reason = imprint_marker(rec.text)
        if reason is not None:
            if warnings is not None:
                warnings["smart_strong_body_demotions"] = (
                    warnings.get("smart_strong_body_demotions", 0) + 1
                )
            dem = HeadingDecision(
                record_index=i,
                text=rec.text,
                is_heading=False,
                outline_level=rec.outline_level,
                use_raw_text=True,
            )
            dem.note(reason)
            dem.note("strong_body_demoted")
            logger.warning(
                "[smart_heading] imprint marker demoted outline paragraph to "
                "body in sub-document fallback (%s)",
                reason,
            )
            out.append(dem)
            continue
        d = HeadingDecision(
            record_index=i,
            text=rec.text,
            is_heading=True,
            level=rec.outline_level + 1,
            outline_level=rec.outline_level,
        )
        d.note("subdoc_fallback_outline_only")
        out.append(d)
    return out


def _demote_confirmed_imprint_regions(
    records: Sequence[Any],
    decisions: dict[int, HeadingDecision],
    regions: Sequence[Any],
    title_starts: Sequence[int],
    warnings: dict | None,
) -> None:
    """Force a confirmed 公文版记 region's lines to body (§版记 conditional).

    A region is CONFIRMED — 100% 版记, not a false imprint hit — only when its
    closer is IMMEDIATELY followed by a valid title block: the first structural
    record after the closer (blank paragraphs / section breaks skipped) starts
    a title block, i.e. the next document's cover in a 公文汇编. Then every
    region member that is STILL a heading is demoted to body with a rule-tagged
    decision (``strong_body_demoted`` keeps invariant I2 green); members already
    body are output-neutral and left untouched. The preceding signature/date
    lines (``reg.preceding``) are NEVER demoted — they are the previous
    document's own content, only vetoed from title-block absorption.
    """
    valid_starts = set(title_starts)
    n = len(records)
    for reg in regions:
        if reg.closer is None:
            continue
        # Scan from the region's LAST member (a trailing 成文日期 absorbed after
        # the closer extends the region past reg.closer), not the closer itself.
        j = max(reg.members) + 1
        while j < n and records[j].kind in (
            "empty_para",
            "empty_table",
            "section_break",
        ):
            j += 1
        if j >= n or j not in valid_starts:
            continue  # no title block right after → veto-only, not confirmed
        for idx in sorted(reg.members):
            existing = decisions.get(idx)
            if existing is None or not existing.is_heading:
                continue  # already body — output-neutral, don't re-tag/re-count
            rec = records[idx]
            dem = HeadingDecision(
                record_index=idx,
                text=rec.text,
                is_heading=False,
                outline_level=rec.outline_level,
                use_raw_text=True,
            )
            dem.note("imprint_region")
            dem.note("strong_body_demoted")
            decisions[idx] = dem
            if warnings is not None:
                warnings["smart_imprint_region_demotions"] = (
                    warnings.get("smart_imprint_region_demotions", 0) + 1
                )
            logger.warning(
                "[smart_heading] imprint region line demoted to body — a title "
                "block follows the 版记 (record %d)",
                idx,
            )


def _memoized_strong_body(
    fn: Callable[[str], str | None],
) -> Callable[[str], str | None]:
    """Per-parse memo for the strong-body verdict. The verdict is a pure
    function of the paragraph text within one parse (the length env is
    constant), so the gate, the CB1 look-ahead, and the post-merge demotion
    sweep can share ONE spaCy pass per unique paragraph rather than each
    re-running ``sentence_count`` on the same text. The cache is scoped to this
    wrapper instance (one per ``run_smart_heading`` call), so it never leaks
    across documents or survives an env change between parses."""
    cache: dict[str, str | None] = {}

    def _cached(text: str) -> str | None:
        if text not in cache:
            cache[text] = fn(text)
        return cache[text]

    return _cached


def run_smart_heading(
    records: Sequence[Any],
    *,
    llm_judge: Any,
    warnings: dict,
    strong_body: Callable[[str], str | None] | None = None,
    numbering_veto: Callable[[Any, str], str | None] | None = None,
    caption_veto: Callable[[str], str | None] | None = None,
) -> SmartHeadingResult | None:
    """Run the full §2.2 pipeline over one document's records.

    Returns ``None`` when the CB4 whole-document gate rejects (too short:
    smart is skipped BEFORE any title-block work, zero LLM calls, output is
    the untouched baseline — TOC removal included in the skip). Otherwise a
    :class:`SmartHeadingResult`; per-sub-document breaker trips fall back to
    outlineLvl-only levels for that sub-document only.
    """
    from lightrag.constants import (
        DEFAULT_DOCX_SMART_MIN_TOKENS,
        DEFAULT_DOCX_SMART_SUBDOC_MIN_TOKENS,
    )

    from . import guardrails as g
    from . import title_block as tb

    # Two CB4 gates: the whole-document gate (below) decides whether smart runs
    # at all; the per-sub-document gate (in the sub-document loop) decides
    # whether an individual sub-document gets size-based leveling or falls back
    # to outline-only. The sub-document gate DEFAULTS to
    # ``min(DEFAULT_DOCX_SMART_SUBDOC_MIN_TOKENS, min_tokens)`` so that lowering
    # DOCX_SMART_MIN_TOKENS alone (the "run smart on short documents" knob) also
    # pulls the sub-document floor down — otherwise a short document would clear
    # the whole-document gate only to have its sub-documents silently fall back
    # to outline-only. An explicit DOCX_SMART_SUBDOC_MIN_TOKENS is honored as-is
    # (a large document may legitimately want to level only its big sections).
    min_tokens = _env_int("DOCX_SMART_MIN_TOKENS", DEFAULT_DOCX_SMART_MIN_TOKENS)
    subdoc_min_tokens = _env_int(
        "DOCX_SMART_SUBDOC_MIN_TOKENS",
        min(DEFAULT_DOCX_SMART_SUBDOC_MIN_TOKENS, min_tokens),
    )

    # One spaCy pass per unique paragraph across the whole document: the gate,
    # the CB1 look-ahead, and the post-merge sweep all share this memo.
    strong_body = _memoized_strong_body(strong_body or g.strong_body_reason)

    audit: dict[str, Any] = {"sub_documents": [], "rule_events": []}

    # Whole-document TOC judgment feeds BOTH the CB4 counting scope and the
    # smart output removal set (§2.3.3: the token count excludes TOC lines
    # so a long-TOC + short-body doc takes the same short path).
    toc_indices = g.detect_toc_records(records)
    audit["toc_removed"] = g.toc_audit_entries(records, toc_indices)

    body_indices = [
        i
        for i in range(len(records))
        if records[i].kind == "para" and i not in toc_indices
    ]

    # Physical features are all extracted by now: log a document-level
    # summary before the CB4 gate can short-circuit (so even skipped short
    # docs surface the diagnostic). ``doc_fs`` is reused as the title-block
    # baseline below to avoid recomputing the global FS_base.
    doc_fs = document_fs_base(records, body_indices)
    _log_physical_feature_summary(records, body_indices, doc_fs)

    if _estimate_record_tokens(records, body_indices) < min_tokens:
        return None  # CB4 whole-document gate — smart never ran

    # --- title blocks -------------------------------------------------------
    # §2.2.4: the title-block gate baseline is the GLOBAL FS_base initial
    # value — the char-weighted dominant size (§2.2.2), not a weighted mean.
    fs_initial = doc_fs.size_pt
    # §版记: map imprint regions ONCE (抄送 anchor → 印发 closer, + preceding
    # signature/date lines). The union is the title-block veto set; the regions
    # themselves are reused after title judgment for the conditional demotion
    # (a region followed by a valid title block is a 公文汇编 boundary).
    imprint_regions = tb.detect_imprint_regions(
        records,
        imprint_marker=g.imprint_marker_reason,
        imprint_closer=g.imprint_closer_reason,
        skip_indices=toc_indices,
    )
    imprint_excluded: set[int] = set()
    for reg in imprint_regions:
        imprint_excluded |= reg.members
        imprint_excluded |= reg.preceding
    candidates = tb.find_title_block_candidates(
        records,
        fs_base_pt=fs_initial,
        strong_body=strong_body,
        numbering_veto=numbering_veto,
        imprint_excluded=imprint_excluded,
        warnings=warnings,
        skip_indices=toc_indices,
    )
    decisions: dict[int, HeadingDecision] = {}
    llm_grants: set[int] = set()
    llm_body_vetoes: set[int] = set()
    title_starts: list[int] = []
    doc_title: str | None = None

    # §2.3.5: LLM call count + per-candidate verdicts are structured metrics.
    llm_calls = {"n": 0}
    judge = llm_judge
    if judge is not None:

        def _counted_judge(prompt: str, **kw: Any) -> str:
            llm_calls["n"] += 1
            return llm_judge(prompt, **kw)

        judge = _counted_judge
    audit["title_block_candidates"] = []

    for cand in candidates:
        verdict = tb.judge_title_block(cand, records, judge, warnings=warnings)
        audit["title_block_candidates"].append(
            {
                "trigger": cand.trigger,
                "start": cand.start,
                "single": cand.single,
                "is_title_block": verdict.is_title_block,
            }
        )
        if verdict.is_title_block:
            heading = tb.compose_title_heading(verdict)
            main_pos = verdict.member_indices[0]
            d = HeadingDecision(
                record_index=main_pos,
                text=records[main_pos].text,
                is_heading=True,
                is_title_block=True,
                level=0,
                composed_heading=heading,
                title_parts=tuple(
                    p
                    for p in (
                        verdict.main_title,
                        verdict.sub_title,
                        verdict.doc_number,
                        verdict.classification,
                        verdict.publisher,
                        verdict.date,
                    )
                    if p
                ),
                member_indices=verdict.member_indices,
            )
            d.note(f"title_block:{cand.trigger}")
            decisions[main_pos] = d
            title_starts.append(main_pos)
            if doc_title is None:
                doc_title = verdict.main_title
        else:
            llm_grants.update(verdict.heading_indices)
            # §2.2.4 "赋予或撤销": body votes revoke candidate identity
            # (outline paragraphs never land here — title_block reroutes
            # them to heading_indices per I2).
            llm_body_vetoes.update(verdict.body_indices)

    title_members: set[int] = set()
    for pos in title_starts:
        title_members.update(decisions[pos].member_indices)

    # --- sub-documents ------------------------------------------------------
    bounds = [0] + sorted(title_starts) + [len(records)]
    seen_start: set[int] = set()
    spans: list[tuple[int, int]] = []
    for k in range(len(bounds) - 1):
        start, end = bounds[k], bounds[k + 1]
        if start in seen_start or start >= end:
            continue
        seen_start.add(start)
        spans.append((start, end))

    fallback_subs = 0
    #: record_index -> the FS_base (body size) of the sub-document it lives
    #: in, so the per-paragraph audit ledger can report each paragraph's
    #: sub-document baseline. Populated for EVERY sub-document below,
    #: including the CB4 short-doc fallback.
    sub_fs_by_index: dict[int, float | None] = {}
    for start, end in spans:
        sub_indices = [
            i
            for i in range(start, end)
            if records[i].kind == "para"
            and i not in toc_indices
            and i not in title_members
        ]
        if not sub_indices:
            continue
        sub_audit: dict[str, Any] = {"range": [start, end]}
        audit["sub_documents"].append(sub_audit)

        # Per-sub-document FS_base is computed and recorded up front (before
        # the CB4 gate can short-circuit) so its value is available for the
        # audit ledger of every sub-document, fallbacks included.
        fs = document_fs_base(records, sub_indices)
        sub_audit["fs_base"] = fs.size_pt
        sub_audit["fs_confidence_high"] = fs.confidence_high
        sub_audit["fs_dominant_ratio"] = round(fs.dominant_ratio, 4)
        for i in sub_indices:
            sub_fs_by_index[i] = fs.size_pt

        baseline_headings = sum(
            1 for i in sub_indices if records[i].outline_level is not None
        )
        baseline_density = baseline_headings / len(sub_indices)

        if _estimate_record_tokens(records, sub_indices) < subdoc_min_tokens:
            sub_audit["fallback"] = "cb4_short_subdoc"
            fallback_subs += 1
            for d in _outline_only_decisions(records, sub_indices, warnings=warnings):
                decisions[d.record_index] = d
            continue

        if not fs.confidence_high:  # §2.3.5: CB5 trips per sub-document
            warnings["smart_cb5_low_confidence"] = (
                warnings.get("smart_cb5_low_confidence", 0) + 1
            )

        gate = gate_with_cb1(
            records,
            sub_indices,
            fs_base=fs,
            baseline_density=baseline_density,
            warnings=warnings,
            strong_body=strong_body,
            numbering_veto=numbering_veto,
            caption_veto=caption_veto,
            llm_heading_grants=llm_grants,
            llm_body_vetoes=llm_body_vetoes,
        )
        if gate.cb1_reestimated:  # §2.3.5: FS_base + density after re-estimation
            sub_audit["cb1_reestimated_fs"] = gate.cb1_new_fs
            sub_audit["cb1_reestimated_density"] = round(gate.density, 4)
            sub_audit["cb1_reestimated_inter_chars"] = (
                round(gate.cb1_new_inter_chars, 1)
                if gate.cb1_new_inter_chars is not None
                else None
            )
        if gate.cb1_strong_body_recovered:  # §2.3.3 look-ahead spared the trip
            sub_audit["cb1_strong_body_recovered"] = True
        if gate.cb1_tripped:
            sub_audit["fallback"] = "cb1_density"
            fallback_subs += 1
            for d in _outline_only_decisions(records, sub_indices, warnings=warnings):
                decisions[d.record_index] = d
            continue

        ds = gate.decisions
        assign_levels_by_size(ds)
        backfill_top_level(ds, warnings=warnings)
        align_numbering_series(ds)
        anchor_outline_levels(ds, warnings=warnings)
        ds = merge_split_headings(ds, records, warnings=warnings)
        demote_strong_body_headings(ds, strong_body=strong_body, warnings=warnings)
        skeleton_audit = correct_numbering_skeleton(ds, warnings=warnings)
        if skeleton_audit:
            # §2.3.5: audit rows key on the paragraph HASH, not a transient
            # position into this sub-document's decision list.
            for event in skeleton_audit:
                pos = event.pop("position", None)
                if pos is not None:
                    event["hash"] = _para_hash(ds[pos].text)
            audit["rule_events"].extend(skeleton_audit)
        align_numbering_series(ds, skip_anchored=True)  # §2.2.8 smoothing
        clamp_deep_levels(ds, warnings=warnings)

        sub_audit["headings"] = sum(1 for d in ds if d.is_heading)
        for d in ds:
            decisions[d.record_index] = d
            # Absorbed merge members: mark trailing member records so the
            # assembler skips their standalone output — but ONLY while the
            # merged heading survives. If the post-merge sweep or clamping
            # demoted it, the joined text is never rendered, so the members
            # must fall back to emitting their own paragraph text; otherwise
            # their content vanishes and I1 trips a whole-document fallback
            # (review C4).
            if d.member_indices and not d.is_title_block and d.is_heading:
                for m in d.member_indices[1:]:
                    absorbed = HeadingDecision(record_index=m, text="", absorbed=True)
                    absorbed.note("merged_absorbed")
                    decisions[m] = absorbed
        # I2 audit trail for recognition-time outline demotions (review C2):
        # merged in AFTER the candidate loop so they never reach leveling /
        # anchoring, yet appear in the final decision map + audit ledger.
        for dem in gate.demoted:
            decisions.setdefault(dem.record_index, dem)

    # Sentinel decisions so the I2 retention check can tell a rule-tagged
    # absence from a silent drop: TOC-removed and title-block-member
    # paragraphs are legitimate non-headings.
    for i in toc_indices:
        if i not in decisions:
            sentinel = HeadingDecision(record_index=i, text="")
            sentinel.note("toc_removed")
            decisions[i] = sentinel
    for pos in title_starts:
        for m in decisions[pos].member_indices:
            if m not in decisions:
                sentinel = HeadingDecision(record_index=m, text="")
                sentinel.note("title_block_member")
                decisions[m] = sentinel

    # §版记: a 抄送…印发 region whose closer is immediately followed by a valid
    # title block is a confirmed 公文汇编 boundary — force its lines to body.
    _demote_confirmed_imprint_regions(
        records, decisions, imprint_regions, title_starts, warnings
    )

    # §2.3.5: the full re-judgment ledger — one row per paragraph that any
    # rule touched (hash + rule trail + final level), replayable offline.
    audit["llm_calls"] = llm_calls["n"]
    audit["decisions"] = [
        {
            "hash": _para_hash(d.text),
            # §2.3.5 deviation: a plaintext preview alongside the hash for
            # human-readable auditing (see _AUDIT_SUMMARY_CHARS).
            "summary": get_content_summary(d.text, max_length=_AUDIT_SUMMARY_CHARS),
            # Paragraph font size = the char-weighted MODE off the record
            # (NOT the effective size on the decision), plus the body
            # baseline of the sub-document this paragraph belongs to.
            "font_size_pt": records[d.record_index].font_size_pt,
            "sub_fs_base": sub_fs_by_index.get(d.record_index),
            "rules": list(d.rule_trail),
            "level": d.level,
            "is_heading": d.is_heading,
        }
        for _, d in sorted(decisions.items())
        if d.rule_trail
    ]

    return SmartHeadingResult(
        decisions=decisions,
        toc_indices=toc_indices,
        doc_title=doc_title,
        audit=audit,
        fallback_sub_docs=fallback_subs,
    )
