"""Chunk schema helpers shared across the chunking + extraction pipeline.

Three responsibilities live here so chunker implementations and the pipeline
both consume identical normalization rules:

- :func:`normalize_chunk_heading` collapses the legacy flat
  ``heading``/``parent_headings``/``level`` triple and the new nested form
  into the canonical ``{"level", "heading", "parent_headings"}`` dict.
- :func:`normalize_chunk_sidecar` validates the new ``sidecar`` payload and
  ensures ``refs`` is always present as a list (single-source items may omit
  it before normalization; we materialize a single-element list for the
  storage layer).
- :func:`strip_internal_multimodal_markup_for_extraction` rewrites
  ``<cite>`` / ``<drawing>`` / ``<equation>`` markup so the entity-extraction
  LLM sees a clean text body. The original ``chunk["content"]`` is never
  mutated; the cleaned string is only used to build the extraction prompt.

The clean function is intentionally conservative: it only strips
parser-emitted identifier attributes that have no business reaching the LLM
(``id``, ``refid``, ``path``, ``src``). Visible captions and equation bodies
are preserved so the extracted entities can still ground against them.
"""

from __future__ import annotations

import re
from typing import Any


_SIDECAR_TYPES = frozenset({"block", "drawing", "table", "equation"})


def normalize_chunk_heading(dp: dict[str, Any]) -> dict[str, Any] | None:
    """Return the canonical nested heading dict or ``None`` when absent.

    Accepts:

    - ``dp["heading"]`` already a dict ``{"level", "heading", "parent_headings"}``.
    - Legacy flat fields ``heading: str`` + ``parent_headings: list[str]`` +
      ``level: int``.

    Empty / missing inputs collapse to ``None`` so callers can simply omit
    the field when writing the chunk record.
    """
    nested = dp.get("heading")
    if isinstance(nested, dict):
        heading_text = str(nested.get("heading") or "").strip()
        parents_raw = nested.get("parent_headings") or []
        level_raw = nested.get("level", 0)
    else:
        heading_text = str(nested or "").strip()
        parents_raw = dp.get("parent_headings") or []
        level_raw = dp.get("level", 0)

    parent_headings: list[str] = []
    if isinstance(parents_raw, list):
        for entry in parents_raw:
            text = str(entry or "").strip()
            if text:
                parent_headings.append(text)

    try:
        level = int(level_raw or 0)
    except (TypeError, ValueError):
        level = 0

    if not heading_text and not parent_headings and level == 0:
        return None

    return {
        "level": level,
        "heading": heading_text,
        "parent_headings": parent_headings,
    }


def normalize_chunk_sidecar(dp: dict[str, Any]) -> dict[str, Any] | None:
    """Return the canonical sidecar dict or ``None`` when absent / invalid.

    Output shape::

        {"type": <one of block|drawing|table|equation>,
         "id":   <primary source id>,
         "refs": [{"type": ..., "id": ...}, ...]}

    ``refs`` is always materialized as a list with at least the primary id.
    Single-source chunks therefore land in storage with ``refs=[{type,id}]``
    so downstream consumers don't need to special-case the field's presence.
    """
    sidecar = dp.get("sidecar")
    if not isinstance(sidecar, dict):
        return None
    sidecar_type = str(sidecar.get("type") or "").strip()
    sidecar_id = str(sidecar.get("id") or "").strip()
    if sidecar_type not in _SIDECAR_TYPES or not sidecar_id:
        return None

    refs_raw = sidecar.get("refs")
    refs: list[dict[str, str]] = []
    if isinstance(refs_raw, list):
        for entry in refs_raw:
            if not isinstance(entry, dict):
                continue
            ref_type = str(entry.get("type") or "").strip()
            ref_id = str(entry.get("id") or "").strip()
            if ref_type in _SIDECAR_TYPES and ref_id:
                refs.append({"type": ref_type, "id": ref_id})
    if not refs:
        refs = [{"type": sidecar_type, "id": sidecar_id}]

    return {"type": sidecar_type, "id": sidecar_id, "refs": refs}


# `<cite type="..." refid="...">visible text</cite>` → `visible text`.
_CITE_RE = re.compile(
    r"<cite\b[^>]*>(.*?)</cite>",
    flags=re.IGNORECASE | re.DOTALL,
)

# Self-closing `<drawing ...>` placeholder.  We keep `caption` (visible) and
# drop `id`, `path`, `src`, `format`, etc.  Tags without any caption are
# removed entirely so they don't pollute extraction input.
_DRAWING_RE = re.compile(
    r"<drawing\b([^>]*)/>",
    flags=re.IGNORECASE,
)

# Container `<equation id="..." format="...">latex</equation>`.  Strip
# identifier attributes; preserve the body and the `format` attribute so
# extraction still sees the equation is a structured element.
_EQUATION_RE = re.compile(
    r"<equation\b([^>]*)>(.*?)</equation>",
    flags=re.IGNORECASE | re.DOTALL,
)

# Match attribute pairs like ``caption="text with \"escapes\""``.  We treat
# only the safe identifier-style attributes; complex quoting is rare in
# parser output.
_ATTR_RE = re.compile(
    r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"',
)


def _attrs_to_dict(attr_string: str) -> dict[str, str]:
    return {
        match.group(1).lower(): match.group(2)
        for match in _ATTR_RE.finditer(attr_string)
    }


def _format_attrs(pairs: list[tuple[str, str]]) -> str:
    return "".join(f' {k}="{v}"' for k, v in pairs if v)


def _replace_drawing(match: re.Match[str]) -> str:
    attrs = _attrs_to_dict(match.group(1))
    caption = attrs.get("caption", "")
    if not caption.strip():
        return ""
    return f"<drawing{_format_attrs([('caption', caption)])} />"


def _replace_equation(match: re.Match[str]) -> str:
    attrs = _attrs_to_dict(match.group(1))
    body = match.group(2)
    keep: list[tuple[str, str]] = []
    fmt = attrs.get("format", "")
    if fmt:
        keep.append(("format", fmt))
    caption = attrs.get("caption", "")
    if caption.strip():
        keep.append(("caption", caption))
    return f"<equation{_format_attrs(keep)}>{body}</equation>"


def strip_internal_multimodal_markup_for_extraction(content: str) -> str:
    """Strip parser-internal identifiers from a chunk content string.

    Only the entity-extraction prompt should receive the cleaned form;
    callers must NOT mutate the stored chunk ``content`` so query-time
    citations still resolve back to the original parser output.

    Transformations:

    - ``<cite type="…" refid="…">Table 1</cite>`` → ``Table 1``
    - ``<drawing id="dr-…" path="…" src="…" caption="Fig 1" />``
        → ``<drawing caption="Fig 1" />``
        (drops the entire tag when no caption is present)
    - ``<equation id="eq-…" format="latex">…</equation>``
        → ``<equation format="latex">…</equation>``
    """
    if not content:
        return content
    cleaned = _CITE_RE.sub(lambda m: m.group(1), content)
    cleaned = _DRAWING_RE.sub(_replace_drawing, cleaned)
    cleaned = _EQUATION_RE.sub(_replace_equation, cleaned)
    return cleaned


__all__ = [
    "normalize_chunk_heading",
    "normalize_chunk_sidecar",
    "strip_internal_multimodal_markup_for_extraction",
]
