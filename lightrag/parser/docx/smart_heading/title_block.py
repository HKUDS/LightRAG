"""Title-block (document main title) discovery and LLM judgment (spec §2.2.4).

Heuristics find CANDIDATE windows — runs of paragraphs lacking strong-body
features that contain a visually dominant line — plus a single-paragraph
channel for spliced multi-article documents whose per-article title is one
isolated big line. An LLM (via the synchronous judge callable; this module
never touches asyncio) then confirms and decomposes each candidate.

LLM output is STRICTLY validated: the main/sub title must be locatable in
the window text (concatenation allowed), a non-title verdict must classify
every window paragraph, and paragraphs carrying a physical outline level are
never demoted by an LLM "body" vote (invariant I2). Any unparseable or
non-locatable answer raises :class:`TitleBlockLLMError` — LLM failures are
loud, never silently degraded.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
    DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS,
    DEFAULT_DOCX_SMART_SINGLE_TITLE_LLM_MAX,
    DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA,
)
from lightrag.utils import logger

from . import guardrails
from .features import effective_font_size_pt
from .style_key import classify_numbering

#: Synchronous LLM judge protocol — matches SyncLLMBridge.__call__.
LLMJudge = Callable[..., str]

#: Max weighted chars for a candidate title line (30 CJK / 90 en; §2.2.4).
TITLE_LINE_MAX_WEIGHTED_CHARS = 90

_FLANK_WINDOW = 20  # K paragraphs on each side for the size-divergence gate
_SINGLE_CONTEXT = 2  # paragraphs of context around a single candidate


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


#: Multi-paragraph windows admit a title line at the §2.2.2 strong-signal
#: tier (+1pt over the global FS_base initial value); the single-paragraph
#: channel is one tier stricter (+2pt via DOCX_SMART_TITLE_BLOCK_MIN_DELTA).
MULTI_WINDOW_TITLE_DELTA_PT = 1.0


def find_title_block_candidates(
    records: Sequence[Any],
    *,
    fs_base_pt: float | None,
    strong_body: Callable[[str], str | None] | None = None,
    numbering_veto: Callable[[Any, str], str | None] | None = None,
    warnings: dict | None = None,
    skip_indices: set[int] = frozenset(),
) -> list[TitleBlockCandidate]:
    """Find multi-paragraph windows and single-paragraph title candidates.

    ``strong_body`` / ``numbering_veto`` default to the guardrails
    implementations and are injectable for NLP-free tests.
    """
    strong_body = strong_body or guardrails.strong_body_reason
    numbering_veto = numbering_veto or guardrails.numbering_homophone_reason
    delta = _env_float(
        "DOCX_SMART_TITLE_BLOCK_MIN_DELTA", DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA
    )
    single_cap = _env_int(
        "DOCX_SMART_SINGLE_TITLE_LLM_MAX", DEFAULT_DOCX_SMART_SINGLE_TITLE_LLM_MAX
    )

    para_indices = [
        i for i, r in enumerate(records) if r.kind == "para" and i not in skip_indices
    ]
    strong_cache: dict[int, str | None] = {}

    def _is_strong(idx: int) -> bool:
        if idx not in strong_cache:
            strong_cache[idx] = strong_body(records[idx].text)
        return strong_cache[idx] is not None

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
            or _is_strong(i)
            or rec.is_toc_field
            or rec.is_toc_link
        ):
            i += 1
            continue
        # Grow the window: consecutive paragraphs without strong-body
        # features; empty paragraphs stay inside, tables / section breaks /
        # TOC lines / strong-body paragraphs end it.
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
                or r.is_toc_field
                or r.is_toc_link
                or _is_strong(j)
            ):
                break
            window_paras.append(j)
            j += 1
        end = j
        if len(window_paras) >= 2 and any(
            _is_title_line(records[k]) for k in window_paras
        ):
            candidates.append(
                TitleBlockCandidate(
                    start=start, end=end, single=False, trigger="multi_window"
                )
            )
            covered.update(range(start, end))
        i = max(end, i + 1)

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
        if (
            guardrails.weighted_char_length(rec.text.strip())
            > TITLE_LINE_MAX_WEIGHTED_CHARS
        ):
            continue
        if _has_real_numbering(rec.text, numbering_veto):
            continue
        single_size = effective_font_size_pt(rec)
        if not (
            fs_base_pt is not None
            and single_size is not None
            and single_size >= fs_base_pt + delta
        ):
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
    "ordinary section headings or body text. Answer with a single JSON "
    "object and nothing else."
)

_USER_TEMPLATE = """Paragraphs (indexed; [BLANK] marks an empty line in the original):

{window}

Rules:
- "main_title" / "sub_title" / "doc_number" must be verbatim text taken from the paragraphs above (the main title may concatenate consecutive paragraphs).
- If this is NOT a title block, classify EVERY index into exactly one of "headings" (a real section heading) or "body".
- Use null for fields that are absent.

Respond with JSON matching:
{{"is_title_block": true|false, "main_title": string|null, "sub_title": string|null, "doc_number": string|null, "classification": string|null, "publisher": string|null, "date": string|null, "headings": [int, ...], "body": [int, ...]}}"""


def _render_window(
    records: Sequence[Any], candidate: TitleBlockCandidate, warnings: dict | None
) -> tuple[str, list[int]]:
    """Render the candidate window for the LLM; returns (text, index_map).

    ``index_map[k]`` is the record index of window line ``[k]``. The window
    is token-capped (env DOCX_SMART_LLM_WINDOW_TOKENS); overflow truncates
    from the tail with a warning.
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
    reserve = _estimate_tokens(records[mandatory].text) if mandatory is not None else 0
    lines: list[str] = []
    index_map: list[int] = []
    used = 0
    truncated = False
    for i in span:
        rec = records[i]
        if rec.kind == "empty_para":
            lines.append("[BLANK]")
            continue
        if rec.kind != "para":
            continue
        line = f"[{len(index_map)}] {rec.text}"
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
    return "\n".join(lines), index_map


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
    window_text, index_map = _render_window(records, candidate, warnings)
    prompt = _USER_TEMPLATE.format(window=window_text)
    raw = llm_judge(prompt, system_prompt=_SYSTEM_PROMPT)
    data = _parse_llm_json(raw)

    is_title = bool(data.get("is_title_block"))
    window_canon = "".join(_canon(records[i].text) for i in index_map)

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
            locate_scope = _canon(records[candidate.start].text)
        else:
            member_indices = tuple(index_map)
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
    if candidate.single:
        return TitleBlockDecision(
            candidate=candidate,
            is_title_block=False,
            heading_indices=(),
            body_indices=(),
            raw_response=raw,
        )

    # Non-title verdict: every window index must be classified exactly once.
    def _indices(key: str) -> list[int]:
        value = data.get(key) or []
        # bool is a subclass of int — reject True/False masquerading as indexes.
        if not isinstance(value, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise TitleBlockLLMError(f"title-block field {key!r} must be [int]")
        return list(value)

    headings = _indices("headings")
    body = _indices("body")
    all_indices = set(range(len(index_map)))
    # A true partition: exact cover, disjoint, AND no duplicates within a list
    # (the length check catches [0, 0] which the set operations would hide).
    if (
        set(headings) | set(body) != all_indices
        or set(headings) & set(body)
        or len(headings) + len(body) != len(all_indices)
    ):
        raise TitleBlockLLMError(
            "non-title verdict must classify every window paragraph exactly "
            f"once (headings={headings}, body={body}, expected {sorted(all_indices)})"
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
