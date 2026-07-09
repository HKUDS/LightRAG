"""Defensive judgments for smart heading discovery (spec §2.2.5 / §2.3.4).

Pure-function layer: consumed by ``title_block`` / ``heading_flow`` /
``extract``, depends only on :mod:`.nlp` and :mod:`.style_key` — never the
reverse. Every rule here votes AGAINST heading-ness (do-no-harm: ties break
toward "not a heading").

Env-tunable knobs follow the live-read pattern of the CHUNK_P_REFERENCES_*
constants: DEFAULT_* lives in ``lightrag.constants``; the matching env var
(same name minus DEFAULT_) is read at call time.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

from lightrag.constants import (
    DEFAULT_DOCX_SMART_CAPTION_PREFIXES,
    DEFAULT_DOCX_SMART_ENNUM_BLACKLIST,
    DEFAULT_DOCX_SMART_HEADING_MAX_CHARS,
    DEFAULT_DOCX_SMART_IMPRINT_COLON_PREFIXES,
    DEFAULT_DOCX_SMART_IMPRINT_SPACE_PREFIXES,
    DEFAULT_DOCX_SMART_TOC_MIN_LINES,
)

from . import nlp
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
        return head[len(cls.label_text) :].lstrip(" \t、.。:：)）")
    return text


def strong_body_reason(text: str) -> str | None:
    """Return the rule id when ``text`` carries a strong body feature.

    Rules (any one suffices):
      - ``imprint_marker``: opens with a 公文版记 (imprint) marker — checked
        first, on the RAW text (an imprint line keeps its identity however
        long it runs, and the whitespace after a space-class prefix is
        evidence that ``strip()`` would destroy);
      - ``strong_body_length``: weighted length > 180 en-equivalent chars;
      - ``strong_body_multi_sentence``: spaCy sees ≥2 sentences;
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
    max_chars = _env_int(
        "DOCX_SMART_HEADING_MAX_CHARS", DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    )
    if weighted_char_length(stripped) > max_chars:
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
    prose = _strip_leading_numbering(stripped)
    if prose and nlp.sentence_count(prose) >= 2:
        return "strong_body_multi_sentence"
    return None


# ---------------------------------------------------------------------------
# Numbering homophone vetoes (NER + P1) — revoke NUMBERING identity only;
# the paragraph's size/bold/outline paths stay open.
# ---------------------------------------------------------------------------

_VERSION_SHAPE_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s*版")


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
# 公文版记 (imprint) marker lines — metadata, never headings (they also veto
# title-block membership for themselves and their neighbourhood; see
# title_block._imprint_veto_indices)
# ---------------------------------------------------------------------------

#: Whitespace allowed to interleave/pad a prefix: justified official-document
#: typesetting stretches a label with plain/ideographic spaces (抄　送：).
_IMPRINT_GAP = r"[ \t　]*"


def imprint_marker_reason(text: str) -> str | None:
    """A 公文版记 (imprint) marker line — 抄送：… / 印发机关␣… — is metadata.

    Two prefix classes (env-tunable, comma/pipe-separated):
      - colon class (抄送): prefix followed by a full/half-width colon;
      - space class (印发机关): prefix followed by any whitespace — ``\\s``
        includes a soft line break and the ideographic space, so a two-line
        ``印发机关\\n机构名`` paragraph still hits.

    Operates on the RAW text (leading whitespace ignored, the rest kept):
    the whitespace after a space-class prefix is the match evidence, so a
    ``strip()``-ed copy must never be passed in. A bare label with nothing
    after it (印发机关) does not match.
    """
    head = text.lstrip()
    if not head:
        return None
    for env, default, tail in (
        (
            "DOCX_SMART_IMPRINT_COLON_PREFIXES",
            DEFAULT_DOCX_SMART_IMPRINT_COLON_PREFIXES,
            _IMPRINT_GAP + "[：:]",
        ),
        (
            "DOCX_SMART_IMPRINT_SPACE_PREFIXES",
            DEFAULT_DOCX_SMART_IMPRINT_SPACE_PREFIXES,
            r"\s",
        ),
    ):
        for prefix in _env_items(env, default):
            # Per-char escape: prefixes are env-configurable, so a user item
            # carrying regex metachars must still match literally.
            body = _IMPRINT_GAP.join(re.escape(ch) for ch in prefix)
            if re.match(body + tail, head):
                return "imprint_marker"
    return None


# ---------------------------------------------------------------------------
# Landing guardrails: canonicalization, I1/I2/I3 machine checks, TOC (§2.3)
# ---------------------------------------------------------------------------

_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")
# Dot leader: ≥3 ASCII/middle dots, or any run holding an ellipsis char
# (one … renders as three dots; Chinese TOCs typically use ……).
_TOC_DOT_LEADER_RE = re.compile(r"(?:…[.…·]*|[.·]{3,})\s*\d+\s*$")


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


def detect_toc_records(records: Sequence[Any]) -> set[int]:
    """Two-channel TOC detection (§2.2.2): structural evidence (TOC field
    instructions / _Toc bookmark links on the paragraph) plus the heuristic
    channel — a run of ≥ DOCX_SMART_TOC_MIN_LINES consecutive leader LINES
    ending in a dot leader + page number. Lines are counted, not paragraphs:
    the run spans standalone paragraphs AND the soft-break lines inside one
    paragraph (§2.2.2 "含独立段落或段内软回车拆出的行"), so a TOC typed as a
    single multi-line paragraph is caught too. An isolated single line is
    never evidence. Returns the record indices to remove from smart output.

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

    return toc


def toc_audit_entries(records: Sequence[Any], toc_indices: set[int]) -> list[dict]:
    """Audit rows for removed TOC paragraphs (count + text hash, §2.3.5)."""
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
    Only the TOC whitelist (indices from the two-channel judgment) may be
    absent entirely.

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
