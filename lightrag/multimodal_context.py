"""Surrounding-context enrichment for native multimodal sidecars.

See ``docs/NativeMultimodalSurroundingContextPlan-zh.md``.

For each entry in ``drawings.json`` / ``tables.json`` / ``equations.json``,
this module locates the matching ``<drawing … id="…" … />``,
``<table … id="…" …>…</table>`` / table ``<cite refid="…">`` or
``<equation … id="…" …>…</equation>`` inside the *single*
``blocks.jsonl`` content row referenced by the entry's ``blockid``, then
extracts up to ``max_tokens`` of leading and trailing text from the same
row (without crossing block rows).

Sidecar entries gain an optional ``surrounding`` field:

    {
      "leading":  "…",
      "trailing": "…"
    }

with both halves capped at ``max_tokens`` tokens (default 2000).
Truncation prefers paragraph / sentence / clause boundaries (using the
recursive separator cascade from ``CHUNK_R_SEPARATORS`` / falling back
to :data:`lightrag.constants.DEFAULT_R_SEPARATORS`); only when a single
closest segment alone exceeds the budget does the splitter fall through
to a character-level binary search.

Multimodal tags (``<drawing/>``, ``<equation>…</equation>``,
``<table>…</table>``) inside the candidate text are treated as atomic so
the splitter cannot cut a tag in half.  For ``tables.json`` entries —
where the surrounding should describe text around the target table
without dragging other tables along — every ``<table>…</table>`` is
removed from the candidate text *before* token counting and
segmentation, so the saved surrounding string and the tokens budgeted
against it stay in sync.  For ``drawings.json`` / ``equations.json``
entries the table tags are preserved when they fit; oversized JSON or
HTML tables are row-trimmed (tail rows for leading, head rows for
trailing) so the surrounding keeps the rows physically closest to the
target.
"""

from __future__ import annotations

import json
import logging
import os
import re
from html import escape as html_escape
from html import unescape as html_unescape
from pathlib import Path
from lightrag.constants import DEFAULT_R_SEPARATORS
from lightrag.table_markup import (
    TABLE_TAG_RE,
    detect_table_format,
    parse_table_tag,
    serialize_html_rows,
    split_html_rows,
)
from lightrag.utils import Tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tag scanner — atomises a string into a list of ``(kind, text)`` pieces so
# the recursive splitter can treat ``<drawing/>``, ``<equation>…</equation>``
# and ``<table>…</table>`` as indivisible.
# ---------------------------------------------------------------------------

_MM_TAG_RE = re.compile(
    r"<drawing\b[^>]*/>"
    r"|<table\b[^>]*>.*?</table>"
    r"|<equation\b[^>]*>.*?</equation>",
    re.DOTALL,
)

_TABLE_CITE_RE = re.compile(
    r'<cite\b(?=[^>]*\btype\s*=\s*"table")[^>]*>.*?</cite>',
    re.DOTALL,
)


def _atomize(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``(kind, content)`` atoms.

    ``kind`` ∈ ``{"text", "drawing", "equation", "table"}``.
    Concatenating all atom contents reproduces ``text`` verbatim.
    """
    atoms: list[tuple[str, str]] = []
    pos = 0
    for match in _MM_TAG_RE.finditer(text):
        if match.start() > pos:
            atoms.append(("text", text[pos : match.start()]))
        tag_text = match.group(0)
        if tag_text.startswith("<drawing"):
            kind = "drawing"
        elif tag_text.startswith("<table"):
            kind = "table"
        else:
            kind = "equation"
        atoms.append((kind, tag_text))
        pos = match.end()
    if pos < len(text):
        atoms.append(("text", text[pos:]))
    return atoms


# ---------------------------------------------------------------------------
# Target-tag locators.  Each builds a regex that matches a complete tag
# carrying the requested ``id`` attribute, regardless of attribute order.
# ---------------------------------------------------------------------------


def _drawing_pattern(item_id: str) -> re.Pattern[str]:
    esc = re.escape(item_id)
    return re.compile(
        rf'<drawing\b[^>]*?\bid\s*=\s*"{esc}"[^>]*?/>',
        re.DOTALL,
    )


def _table_pattern(item_id: str) -> re.Pattern[str]:
    esc = re.escape(item_id)
    return re.compile(
        rf'<table\b[^>]*?\bid\s*=\s*"{esc}"[^>]*?>.*?</table>'
        rf'|<cite\b(?=[^>]*\btype\s*=\s*"table")'
        rf'(?=[^>]*\brefid\s*=\s*"{esc}")[^>]*>.*?</cite>',
        re.DOTALL,
    )


def _equation_pattern(item_id: str) -> re.Pattern[str]:
    esc = re.escape(item_id)
    return re.compile(
        rf'<equation\b[^>]*?\bid\s*=\s*"{esc}"[^>]*?>.*?</equation>',
        re.DOTALL,
    )


def find_target_span(
    kind: str, item_id: str, block_content: str
) -> tuple[int, int] | None:
    """Locate the target multimodal marker with the given ``id`` inside
    ``block_content``.

    Returns ``(start, end)`` byte offsets, or ``None`` if not found.
    ``kind`` is the sidecar root key — ``"drawings"`` / ``"tables"`` /
    ``"equations"``.
    """
    if kind == "drawings":
        pattern = _drawing_pattern(item_id)
    elif kind == "tables":
        pattern = _table_pattern(item_id)
    elif kind == "equations":
        pattern = _equation_pattern(item_id)
    else:
        return None
    match = pattern.search(block_content)
    if not match:
        return None
    return match.start(), match.end()


# ---------------------------------------------------------------------------
# Recursive splitter that respects multimodal tag atoms.
# ---------------------------------------------------------------------------


def _split_text_segment(text: str, separators: list[str]) -> tuple[list[str], int]:
    """Split ``text`` using the first separator that produces >1 pieces.

    Returns ``(segments, sep_index)`` where ``segments`` reproduces
    ``text`` verbatim when concatenated and ``sep_index`` is the index
    in ``separators`` of the separator that was used.  When no listed
    separator yields >1 piece the original string is returned as a
    single-element list with ``sep_index = len(separators)`` — the
    caller is responsible for any further char-level fallback.

    The separator is kept attached to the preceding segment so the
    assembled accumulator preserves whitespace boundaries.
    """
    if not text:
        return [text], len(separators)
    for idx, sep in enumerate(separators):
        if not sep:
            continue
        if sep in text:
            parts = text.split(sep)
            assembled: list[str] = []
            for j, part in enumerate(parts):
                if j < len(parts) - 1:
                    assembled.append(part + sep)
                else:
                    if part:
                        assembled.append(part)
            if len(assembled) > 1:
                return assembled, idx
    return [text], len(separators)


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def _char_trim_leading(text: str, max_tokens: int, tokenizer: Tokenizer) -> str:
    """Drop characters from the head until the token count fits.

    Used as the final char-level fallback for the ``leading`` half — we
    want to keep the *tail* of the text (closest to the target).
    """
    if _count_tokens(tokenizer, text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _count_tokens(tokenizer, text[mid:]) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def _char_trim_trailing(text: str, max_tokens: int, tokenizer: Tokenizer) -> str:
    """Drop characters from the tail until the token count fits.

    Used as the final char-level fallback for the ``trailing`` half — we
    keep the *head* (closest to the target).
    """
    if _count_tokens(tokenizer, text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens(tokenizer, text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


# ---------------------------------------------------------------------------
# Row-aware table trimming for drawings / equations surrounding.
# ---------------------------------------------------------------------------


def _row_trim_table_leading(
    tag_text: str, max_tokens: int, tokenizer: Tokenizer
) -> str | None:
    """Return a smaller ``<table>…</table>`` whose tail rows fit ``max_tokens``.

    For a JSON table, takes the last ``k`` rows (closest to the target)
    such that the re-wrapped tag still fits.  For an HTML table, takes
    the last ``k`` ``<tr>``s with their wrapper context.  Returns
    ``None`` when no row-bounded trim fits.
    """
    match = TABLE_TAG_RE.match(tag_text.strip())
    if not match:
        return None
    attrs = match.group("attrs")
    body = match.group("body")
    fmt = detect_table_format(attrs, body)
    if fmt == "json":
        parsed = parse_table_tag(tag_text)
        if not parsed:
            return None
        attrs_str, rows = parsed
        for k in range(len(rows) - 1, 0, -1):
            candidate = (
                f"<table {attrs_str}>"
                f"{json.dumps(rows[-k:], ensure_ascii=False)}"
                f"</table>"
            )
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                return candidate
        return _char_fallback_json_table(
            attrs_str,
            json.dumps(rows[-1], ensure_ascii=False) if rows else body,
            max_tokens,
            tokenizer,
            keep_tail=True,
        )
    if fmt == "html":
        rows = split_html_rows(body)
        if not rows:
            return None
        for k in range(len(rows) - 1, 0, -1):
            inner = serialize_html_rows(rows[-k:])
            candidate = f"<table {attrs}>{inner}</table>"
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                return candidate
        return _char_fallback_html_table(
            attrs,
            rows[-1][1] if rows else body,
            max_tokens,
            tokenizer,
            keep_tail=True,
        )
    return None


def _row_trim_table_trailing(
    tag_text: str, max_tokens: int, tokenizer: Tokenizer
) -> str | None:
    """Return a smaller ``<table>…</table>`` whose head rows fit ``max_tokens``."""
    match = TABLE_TAG_RE.match(tag_text.strip())
    if not match:
        return None
    attrs = match.group("attrs")
    body = match.group("body")
    fmt = detect_table_format(attrs, body)
    if fmt == "json":
        parsed = parse_table_tag(tag_text)
        if not parsed:
            return None
        attrs_str, rows = parsed
        for k in range(len(rows) - 1, 0, -1):
            candidate = (
                f"<table {attrs_str}>"
                f"{json.dumps(rows[:k], ensure_ascii=False)}"
                f"</table>"
            )
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                return candidate
        return _char_fallback_json_table(
            attrs_str,
            json.dumps(rows[0], ensure_ascii=False) if rows else body,
            max_tokens,
            tokenizer,
            keep_tail=False,
        )
    if fmt == "html":
        rows = split_html_rows(body)
        if not rows:
            return None
        for k in range(len(rows) - 1, 0, -1):
            inner = serialize_html_rows(rows[:k])
            candidate = f"<table {attrs}>{inner}</table>"
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                return candidate
        return _char_fallback_html_table(
            attrs,
            rows[0][1] if rows else body,
            max_tokens,
            tokenizer,
            keep_tail=False,
        )
    return None


def _empty_table(attrs: str) -> str:
    return f"<table {attrs}></table>"


def _char_fallback_json_table(
    attrs: str,
    source_text: str,
    max_tokens: int,
    tokenizer: Tokenizer,
    *,
    keep_tail: bool,
) -> str | None:
    """Fit one oversized JSON table row while keeping a valid table tag.

    The fallback stores the truncated serialized row text as a JSON string
    inside a one-row table.  That preserves JSON validity and keeps the
    closest side of the oversized row when no complete row can fit.
    """
    empty = _empty_table(attrs)
    if _count_tokens(tokenizer, empty) > max_tokens:
        return None

    def candidate(chars: int) -> str:
        snippet = source_text[-chars:] if keep_tail and chars else source_text[:chars]
        if not chars:
            return empty
        body = json.dumps([[snippet]], ensure_ascii=False)
        return f"<table {attrs}>{body}</table>"

    if _count_tokens(tokenizer, candidate(len(source_text))) <= max_tokens:
        return candidate(len(source_text))

    lo, hi = 0, len(source_text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens(tokenizer, candidate(mid)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return candidate(lo)


def _char_fallback_html_table(
    attrs: str,
    row_html: str,
    max_tokens: int,
    tokenizer: Tokenizer,
    *,
    keep_tail: bool,
) -> str | None:
    """Fit one oversized HTML row without emitting broken table markup."""
    empty = _empty_table(attrs)
    if _count_tokens(tokenizer, empty) > max_tokens:
        return None

    text = html_unescape(re.sub(r"<[^>]+>", "", row_html or ""))

    def candidate(chars: int) -> str:
        snippet = text[-chars:] if keep_tail and chars else text[:chars]
        if not chars:
            return empty
        return f"<table {attrs}><tr><td>{html_escape(snippet)}</td></tr></table>"

    if _count_tokens(tokenizer, candidate(len(text))) <= max_tokens:
        return candidate(len(text))

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens(tokenizer, candidate(mid)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return candidate(lo)


def remove_table_tags(text: str) -> str:
    """Strip every table marker from ``text``.

    Used to pre-clean candidate text for ``tables.json`` surroundings:
    we never include sibling tables, so they must be dropped *before*
    token counting and segmentation so the budget matches the persisted
    string exactly.
    """
    return _TABLE_CITE_RE.sub("", TABLE_TAG_RE.sub("", text))


# ---------------------------------------------------------------------------
# Core leading / trailing builders.
# ---------------------------------------------------------------------------


def _build_leading(
    source: str,
    *,
    kind: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    separators: list[str],
) -> str:
    """Build the ``leading`` half: suffix of ``source`` within budget."""
    if not source or max_tokens <= 0:
        return ""
    if kind == "tables":
        source = remove_table_tags(source)
        if not source:
            return ""
    accumulated = ""
    atoms = _atomize(source)
    for atom_idx in range(len(atoms) - 1, -1, -1):
        atom_kind, atom_text = atoms[atom_idx]
        if not atom_text:
            continue
        if atom_kind in {"drawing", "equation"}:
            candidate = atom_text + accumulated
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                accumulated = candidate
                continue
            break
        if atom_kind == "table":
            # Only reached for drawings/equations surroundings — table
            # tags are pre-stripped for the ``tables`` kind above.
            candidate = atom_text + accumulated
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                accumulated = candidate
                continue
            remaining = max_tokens - _count_tokens(tokenizer, accumulated)
            if remaining > 0:
                trimmed = _row_trim_table_leading(atom_text, remaining, tokenizer)
                if trimmed is not None:
                    accumulated = trimmed + accumulated
            break
        # Plain text atom — segment with separator cascade and accumulate
        # from the right.
        addition = _accumulate_text_leading(
            atom_text,
            existing=accumulated,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            separators=separators,
        )
        if addition is None:
            # Even a partial fit was not possible; we stop here.
            break
        accumulated = addition + accumulated
        if _count_tokens(tokenizer, accumulated) >= max_tokens:
            break
    return accumulated


def _accumulate_text_leading(
    text: str,
    *,
    existing: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    separators: list[str],
) -> str | None:
    """Add as much of ``text`` (suffix) as fits into the remaining budget.

    Returns the chunk to prepend to ``existing``, or ``None`` to signal
    "stop walking earlier atoms" (i.e. budget exhausted with no useful
    addition).
    """
    segments, sep_idx = _split_text_segment(text, separators)
    if not segments:
        return None
    # Try to add whole segments from the right.  ``buf`` is what we will
    # prepend to ``existing``.
    buf = ""
    for i in range(len(segments) - 1, -1, -1):
        candidate = segments[i] + buf
        # Total tokens once we prepend ``candidate`` to ``existing``.
        if _count_tokens(tokenizer, candidate + existing) <= max_tokens:
            buf = candidate
            continue
        # Cannot fit segment ``i`` whole.  Two cases:
        if buf:
            # We already added at least one segment — stop here without
            # char-truncating a more-distant segment.
            return buf
        # ``buf`` is empty: the closest segment alone overflows. Recurse
        # into the next separator level so we try a finer split before
        # falling back to characters.
        weaker = separators[sep_idx + 1 :] if sep_idx < len(separators) else []
        if weaker:
            return _accumulate_text_leading(
                segments[i],
                existing=existing,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                separators=weaker,
            )
        # Char-level fallback: take the longest suffix of this segment
        # that fits the remaining budget.
        remaining = max_tokens - _count_tokens(tokenizer, existing)
        if remaining <= 0:
            return None
        trimmed = _char_trim_leading(segments[i], remaining, tokenizer)
        return trimmed if trimmed else None
    return buf if buf else None


def _build_trailing(
    source: str,
    *,
    kind: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    separators: list[str],
) -> str:
    """Build the ``trailing`` half: prefix of ``source`` within budget."""
    if not source or max_tokens <= 0:
        return ""
    if kind == "tables":
        source = remove_table_tags(source)
        if not source:
            return ""
    accumulated = ""
    atoms = _atomize(source)
    for atom_kind, atom_text in atoms:
        if not atom_text:
            continue
        if atom_kind in {"drawing", "equation"}:
            candidate = accumulated + atom_text
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                accumulated = candidate
                continue
            break
        if atom_kind == "table":
            candidate = accumulated + atom_text
            if _count_tokens(tokenizer, candidate) <= max_tokens:
                accumulated = candidate
                continue
            remaining = max_tokens - _count_tokens(tokenizer, accumulated)
            if remaining > 0:
                trimmed = _row_trim_table_trailing(atom_text, remaining, tokenizer)
                if trimmed is not None:
                    accumulated = accumulated + trimmed
            break
        addition = _accumulate_text_trailing(
            atom_text,
            existing=accumulated,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            separators=separators,
        )
        if addition is None:
            break
        accumulated = accumulated + addition
        if _count_tokens(tokenizer, accumulated) >= max_tokens:
            break
    return accumulated


def _accumulate_text_trailing(
    text: str,
    *,
    existing: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    separators: list[str],
) -> str | None:
    segments, sep_idx = _split_text_segment(text, separators)
    if not segments:
        return None
    buf = ""
    for i, seg in enumerate(segments):
        candidate = buf + seg
        if _count_tokens(tokenizer, existing + candidate) <= max_tokens:
            buf = candidate
            continue
        if buf:
            return buf
        weaker = separators[sep_idx + 1 :] if sep_idx < len(separators) else []
        if weaker:
            return _accumulate_text_trailing(
                seg,
                existing=existing,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                separators=weaker,
            )
        remaining = max_tokens - _count_tokens(tokenizer, existing)
        if remaining <= 0:
            return None
        trimmed = _char_trim_trailing(seg, remaining, tokenizer)
        return trimmed if trimmed else None
    return buf if buf else None


# ---------------------------------------------------------------------------
# Public entrypoints.
# ---------------------------------------------------------------------------


def load_chunk_separators() -> list[str]:
    """Resolve the recursive-character separator cascade.

    Reads ``CHUNK_R_SEPARATORS`` and falls back to
    :data:`lightrag.constants.DEFAULT_R_SEPARATORS` on missing / invalid
    JSON.  The returned list always has the empty-string sentinel
    dropped — char fallback is signalled separately by the caller.
    """
    raw = os.getenv("CHUNK_R_SEPARATORS")
    separators: list[str]
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                separators = parsed
            else:
                separators = list(DEFAULT_R_SEPARATORS)
        except json.JSONDecodeError:
            separators = list(DEFAULT_R_SEPARATORS)
    else:
        separators = list(DEFAULT_R_SEPARATORS)
    return [s for s in separators if s]


def load_content_rows_by_blockid(blocks_path: str) -> dict[str, str]:
    """Read ``blocks.jsonl`` and return ``{blockid: content_str}``.

    Only ``type == "content"`` rows are kept.  When the same blockid
    appears multiple times, the first occurrence wins.
    """
    rows: dict[str, str] = {}
    path = Path(blocks_path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") != "content":
                continue
            blockid = obj.get("blockid")
            if not isinstance(blockid, str) or not blockid:
                continue
            if blockid in rows:
                continue
            content = obj.get("content")
            if isinstance(content, str):
                rows[blockid] = content
    return rows


DEFAULT_SURROUNDING_MAX_TOKENS = 2000


def _resolve_surrounding_budget(
    leading_max_tokens: int | None,
    trailing_max_tokens: int | None,
) -> tuple[int, int]:
    """Resolve per-half token budgets, defaulting to env vars then 2000.

    Reads ``SURROUNDING_LEADING_MAX_TOKENS`` / ``SURROUNDING_TRAILING_MAX_TOKENS``
    when the caller passes ``None``.  Invalid env values fall back to
    :data:`DEFAULT_SURROUNDING_MAX_TOKENS`.
    """

    def _from_env(env_var: str) -> int:
        raw = os.getenv(env_var)
        if raw is None or not raw.strip():
            return DEFAULT_SURROUNDING_MAX_TOKENS
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "[multimodal_context] invalid %s=%r; falling back to %d",
                env_var,
                raw,
                DEFAULT_SURROUNDING_MAX_TOKENS,
            )
            return DEFAULT_SURROUNDING_MAX_TOKENS
        return max(0, value)

    leading = (
        leading_max_tokens
        if leading_max_tokens is not None
        else _from_env("SURROUNDING_LEADING_MAX_TOKENS")
    )
    trailing = (
        trailing_max_tokens
        if trailing_max_tokens is not None
        else _from_env("SURROUNDING_TRAILING_MAX_TOKENS")
    )
    return leading, trailing


def build_surrounding(
    *,
    kind: str,
    block_content: str,
    span: tuple[int, int],
    tokenizer: Tokenizer,
    leading_max_tokens: int,
    trailing_max_tokens: int,
    separators: list[str],
) -> dict[str, str]:
    """Compute ``{"leading": …, "trailing": …}`` for one sidecar entry.

    ``leading_max_tokens`` and ``trailing_max_tokens`` are independent
    per-half caps so deployments can tune the two contexts separately
    via ``SURROUNDING_LEADING_MAX_TOKENS`` / ``SURROUNDING_TRAILING_MAX_TOKENS``.
    """
    start, end = span
    leading_src = block_content[:start]
    trailing_src = block_content[end:]
    leading = _build_leading(
        leading_src,
        kind=kind,
        tokenizer=tokenizer,
        max_tokens=leading_max_tokens,
        separators=separators,
    )
    trailing = _build_trailing(
        trailing_src,
        kind=kind,
        tokenizer=tokenizer,
        max_tokens=trailing_max_tokens,
        separators=separators,
    )
    return {"leading": leading, "trailing": trailing}


def enrich_sidecars_with_surrounding(
    *,
    blocks_path: str,
    enabled_modalities: set[str],
    tokenizer: Tokenizer,
    leading_max_tokens: int | None = None,
    trailing_max_tokens: int | None = None,
    separators: list[str] | None = None,
) -> dict[str, int]:
    """Backfill ``surrounding`` on enabled-modality sidecars.

    Args:
        blocks_path: path to the ``…blocks.jsonl`` artifact.
        enabled_modalities: subset of ``{"drawings", "tables",
            "equations"}`` reflecting the document's ``process_options``.
        tokenizer: tokenizer used to enforce the per-half token budget.
        leading_max_tokens: leading-half cap.  ``None`` reads
            ``SURROUNDING_LEADING_MAX_TOKENS`` (default 2000).
        trailing_max_tokens: trailing-half cap.  ``None`` reads
            ``SURROUNDING_TRAILING_MAX_TOKENS`` (default 2000).
        separators: explicit separator cascade.  Defaults to the cascade
            resolved from ``CHUNK_R_SEPARATORS`` (or
            ``DEFAULT_R_SEPARATORS``).

    Returns:
        ``{modality: updated_entries}`` for diagnostics.  Modalities
        without a sidecar on disk are silently skipped (consistent with
        the rest of the multimodal pipeline).
    """
    counts = {"drawings": 0, "tables": 0, "equations": 0}
    if not enabled_modalities:
        return counts

    blocks_file = Path(blocks_path)
    if not blocks_file.exists():
        return counts

    content_by_blockid = load_content_rows_by_blockid(blocks_path)
    if separators is None:
        separators = load_chunk_separators()

    leading_tokens, trailing_tokens = _resolve_surrounding_budget(
        leading_max_tokens, trailing_max_tokens
    )

    base = str(blocks_file)
    if base.endswith(".blocks.jsonl"):
        base = base[: -len(".blocks.jsonl")]

    for root_key in ("drawings", "tables", "equations"):
        if root_key not in enabled_modalities:
            continue
        sidecar_path = Path(base + f".{root_key}.json")
        if not sidecar_path.exists():
            continue
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[multimodal_context] failed to read %s: %s",
                sidecar_path,
                exc,
            )
            continue
        items = payload.get(root_key)
        if not isinstance(items, dict):
            continue

        updated = 0
        for item_id, item in items.items():
            if not isinstance(item, dict):
                continue
            blockid = item.get("blockid")
            if not isinstance(blockid, str) or not blockid:
                continue
            block_content = content_by_blockid.get(blockid)
            if block_content is None:
                continue
            span = find_target_span(root_key, item_id, block_content)
            if span is None:
                logger.debug(
                    "[multimodal_context] %s/%s: id not found in block %s",
                    root_key,
                    item_id,
                    blockid,
                )
                continue
            surrounding = build_surrounding(
                kind=root_key,
                block_content=block_content,
                span=span,
                tokenizer=tokenizer,
                leading_max_tokens=leading_tokens,
                trailing_max_tokens=trailing_tokens,
                separators=separators,
            )
            item["surrounding"] = surrounding
            updated += 1

        counts[root_key] = updated
        try:
            sidecar_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "[multimodal_context] failed to write %s: %s",
                sidecar_path,
                exc,
            )
            continue
        logger.debug(
            "[multimodal_context] %s: surrounding written for %d entries",
            root_key,
            updated,
        )

    return counts


__all__ = [
    "DEFAULT_SURROUNDING_MAX_TOKENS",
    "build_surrounding",
    "enrich_sidecars_with_surrounding",
    "find_target_span",
    "load_chunk_separators",
    "load_content_rows_by_blockid",
    "remove_table_tags",
]
