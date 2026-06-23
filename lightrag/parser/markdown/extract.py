"""Pure markdown → block-list extraction for the native markdown engine.

This is the engine-private counterpart of ``parser/docx/parse_document.py``:
it turns raw markdown text into the same shape the native IR builder consumes
(a list of heading-split block dicts whose ``content`` carries placeholder
markers), plus side tables describing the tables / equations / images those
markers stand for.

Placeholder protocol (two-stage, mirroring the docx parser):

- ``extract_markdown`` embeds **self-closing temporary markers** in block
  ``content`` — ``<mdtable ref="t0"/>`` / ``<mdequation ref="e0"/>`` /
  ``<mddrawing ref="d0"/>`` — and never builds IR objects itself.
- :class:`lightrag.parser.markdown.ir_builder.NativeMarkdownIRBuilder` later
  rewrites each marker into the IR placeholder token (``{{TBL:k}}`` /
  ``{{EQ:k}}`` / ``{{IMG:k}}``) and builds the matching IR item from the side
  tables.

The actual table/equation/image payloads live in side tables keyed by the
marker ref (NOT inside the marker string). This keeps a captured HTML
``<table>`` out of the content stream — so its inner ``</table>`` can never
truncate a naive ``<mdtable>…</mdtable>`` wrapper — and keeps image bytes off
the content string entirely.

Supported subset (NOT full CommonMark/GFM, by design — see the parser plan):
ATX headings, simple pipe tables (with a header row), block-level ``$$`` math,
inline ``![alt](src)`` images, and HTML ``<table>`` blocks. Reference-style
images, escaped pipes, nested tables, setext headings and list/quote-nested
structures are left as verbatim text rather than misrecognised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from lightrag.parser._html_table import starts_with_html_tag
from lightrag.parser._markdown import render_heading_line

PREFACE_HEADING = "Preface/Uncategorized"

# --- placeholder marker protocol (shared with the IR builder) --------------
_TABLE_MARKER = '<mdtable ref="{ref}"/>'
_EQUATION_MARKER = '<mdequation ref="{ref}"/>'
_DRAWING_MARKER = '<mddrawing ref="{ref}"/>'

TABLE_MARKER_RE = re.compile(r'<mdtable ref="([^"]+)"/>')
EQUATION_MARKER_RE = re.compile(r'<mdequation ref="([^"]+)"/>')
DRAWING_MARKER_RE = re.compile(r'<mddrawing ref="([^"]+)"/>')

# --- markdown token patterns ----------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# Trailing closing-hash run of an ATX heading (``## Foo ##`` → ``Foo``).
_HEADING_TRAILING_HASHES_RE = re.compile(r"\s+#+\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
# A GFM delimiter row: one or more ``---`` cells with optional ``:`` alignment.
_DELIMITER_ROW_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
# A single delimiter cell (after splitting on ``|``): ``---`` with optional
# ``:`` alignment markers, nothing else.
_DELIMITER_CELL_RE = re.compile(r"^:?-+:?$")
# Inline image: ``![alt](src "optional title")``. ``src`` may be wrapped in
# angle brackets. base64 data URLs and bare URLs/paths (no spaces, no ``)``).
_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\(\s*(?P<src><[^>]*>|[^)\s]+)(?:\s+"[^"]*")?\s*\)'
)


def table_marker(ref: str) -> str:
    return _TABLE_MARKER.format(ref=ref)


def equation_marker(ref: str) -> str:
    return _EQUATION_MARKER.format(ref=ref)


def drawing_marker(ref: str) -> str:
    return _DRAWING_MARKER.format(ref=ref)


@dataclass
class ResolvedImage:
    """Outcome of resolving one ``![](src)`` reference.

    ``kind``:
      * ``"local"`` — bytes available; ``asset_ref`` is a stable identity used
        to deduplicate (same identity ⇒ one on-disk asset shared by every
        occurrence). ``data`` / ``suggested_name`` / ``fmt`` describe it.
      * ``"external"`` — keep as an external link; ``url`` is rendered verbatim
        into the drawing's ``path_override`` (no bytes materialized).
      * ``"skip"`` — drop the image (resolver already logged / counted it).
    """

    kind: str
    asset_ref: str = ""
    data: bytes | None = None
    suggested_name: str = ""
    fmt: str = ""
    url: str = ""


class ImageResolver(Protocol):
    """Resolves a markdown image ``src`` to bytes / link / skip.

    Implementations own all I/O (base64 decode, HTTP download, textpack asset
    read) and any deduplication caching, plus bumping warning counters for
    skipped / failed images. ``extract_markdown`` stays pure and only records
    what the resolver returns.
    """

    def resolve(self, src: str) -> ResolvedImage: ...


@dataclass
class MarkdownExtraction:
    """Result of :func:`extract_markdown`.

    ``blocks`` mirrors the docx block-dict shape (``heading`` / ``level`` /
    ``parent_headings`` / ``content``). The side tables are keyed by marker
    ref; ``assets`` is keyed by :attr:`ResolvedImage.asset_ref` (deduped).
    """

    blocks: list[dict] = field(default_factory=list)
    tables: dict[str, dict] = field(default_factory=dict)
    equations: dict[str, str] = field(default_factory=dict)
    drawings: dict[str, dict] = field(default_factory=dict)
    assets: dict[str, dict] = field(default_factory=dict)


def _clean_heading(text: str) -> str:
    return _HEADING_TRAILING_HASHES_RE.sub("", text).strip()


def _split_pipe_row(line: str) -> list[str]:
    """Split a pipe-table row into trimmed cells (no escaped-pipe handling)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def _is_pipe_table_delimiter(header_line: str, delim_line: str) -> bool:
    """True iff ``delim_line`` is a GFM delimiter row matching ``header_line``.

    Beyond the row-shape regex, every delimiter cell must be a bare ``---`` (no
    stray text) and the column count must equal the header's. This rejects a
    bare ``---`` (a thematic break or setext underline) sitting under a
    pipe-containing paragraph — that has one column versus the header's many, so
    it is not a table, matching GFM's column-count rule."""
    if not _DELIMITER_ROW_RE.match(delim_line):
        return False
    delim_cells = _split_pipe_row(delim_line)
    if not all(_DELIMITER_CELL_RE.match(cell) for cell in delim_cells):
        return False
    return len(delim_cells) == len(_split_pipe_row(header_line))


def extract_markdown(
    text: str,
    *,
    image_resolver: ImageResolver,
) -> MarkdownExtraction:
    """Extract markdown ``text`` into heading-split blocks + side tables.

    Image I/O and any warning counting are delegated to ``image_resolver``;
    this function stays pure (no filesystem / network).
    """
    out = MarkdownExtraction()
    lines = text.splitlines()

    heading_stack: list[tuple[int, str]] = []  # (level, clean_text)
    counters = {"t": 0, "e": 0, "d": 0}

    cur_heading = PREFACE_HEADING
    cur_level = 0
    cur_parents: list[str] = []
    cur_lines: list[str] = []
    has_block_payload = False  # any marker emitted in the current block

    def _flush() -> None:
        nonlocal cur_lines, has_block_payload
        content = "\n".join(cur_lines).strip()
        if not content and not has_block_payload:
            cur_lines = []
            has_block_payload = False
            return
        out.blocks.append(
            {
                "heading": cur_heading,
                "level": cur_level,
                "parent_headings": list(cur_parents),
                "content": "\n".join(cur_lines).rstrip(),
            }
        )
        cur_lines = []
        has_block_payload = False

    def _open(level: int, clean: str, raw: str, parents: list[str]) -> None:
        nonlocal cur_heading, cur_level, cur_parents
        cur_heading = clean
        cur_level = level
        cur_parents = parents
        cur_lines.append(render_heading_line(level, raw))

    def _next_ref(kind: str) -> str:
        counters[kind] += 1
        return f"{kind}{counters[kind]}"

    def _resolve_image(match: re.Match[str]) -> str:
        src = match.group("src").strip()
        if src.startswith("<") and src.endswith(">"):
            src = src[1:-1].strip()
        resolved = image_resolver.resolve(src)
        if resolved.kind == "skip":
            # Resolver already warned/counted; drop the image, keep nothing.
            return ""
        # A base64 data URL carries no meaningful reference name and would
        # bloat the sidecar (the bytes are already materialized as an asset),
        # so it is not echoed into ``src``.
        display_src = "" if src.lower().startswith("data:") else src
        ref = _next_ref("d")
        if resolved.kind == "local":
            asset_ref = resolved.asset_ref
            if asset_ref not in out.assets:
                out.assets[asset_ref] = {
                    "suggested_name": resolved.suggested_name,
                    "data": resolved.data,
                    "fmt": resolved.fmt,
                }
            out.drawings[ref] = {
                "kind": "local",
                "asset_ref": asset_ref,
                "fmt": resolved.fmt,
                "src": display_src,
            }
        else:  # external
            out.drawings[ref] = {
                "kind": "external",
                "url": resolved.url or src,
                "fmt": resolved.fmt,
                "src": display_src,
            }
        return drawing_marker(ref)

    def _emit_inline(line: str) -> None:
        nonlocal has_block_payload
        before = len(out.drawings)
        new_line = _IMAGE_RE.sub(_resolve_image, line)
        if len(out.drawings) != before:
            has_block_payload = True
        cur_lines.append(new_line)

    n = len(lines)
    i = 0
    fence: tuple[str, int, bool] | None = None  # (char, length, is_open)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- fenced code blocks: verbatim, suppress all detection ----------
        fence_match = _FENCE_RE.match(stripped)
        if fence is not None:
            cur_lines.append(line)
            if fence_match:
                ch, run = fence_match.group(1)[0], len(fence_match.group(1))
                # A closing fence is the same char, length >= opener, no info.
                if ch == fence[0] and run >= fence[1] and not fence_match.group(2):
                    fence = None
            i += 1
            continue
        if fence_match:
            fence = (fence_match.group(1)[0], len(fence_match.group(1)), True)
            cur_lines.append(line)
            i += 1
            continue

        # --- ATX heading ---------------------------------------------------
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            raw = heading_match.group(2)
            clean = _clean_heading(raw)
            heading_stack[:] = heading_stack[: max(level - 1, 0)]
            parents = [h for _, h in heading_stack if h]
            heading_stack.append((level, clean))
            _flush()
            _open(level, clean, raw, parents)
            i += 1
            continue

        # --- block equation ($$ … $$) --------------------------------------
        if stripped.startswith("$$"):
            consumed, latex = _consume_block_equation(lines, i)
            if consumed > 0:
                ref = _next_ref("e")
                out.equations[ref] = latex
                cur_lines.append(equation_marker(ref))
                has_block_payload = True
                i += consumed
                continue

        # --- HTML <table> block --------------------------------------------
        if starts_with_html_tag(stripped.lower(), "table"):
            consumed, html = _consume_html_table(lines, i)
            if consumed > 0:
                ref = _next_ref("t")
                out.tables[ref] = {"kind": "html", "html": html}
                cur_lines.append(table_marker(ref))
                has_block_payload = True
                i += consumed
                continue

        # --- pipe table ----------------------------------------------------
        if "|" in line and i + 1 < n and _is_pipe_table_delimiter(line, lines[i + 1]):
            consumed, rows, header = _consume_pipe_table(lines, i)
            if consumed > 0:
                ref = _next_ref("t")
                out.tables[ref] = {"kind": "pipe", "rows": rows, "header": header}
                cur_lines.append(table_marker(ref))
                has_block_payload = True
                i += consumed
                continue

        # --- plain text line (inline images resolved here) -----------------
        _emit_inline(line)
        i += 1

    _flush()
    return out


def _consume_block_equation(lines: list[str], start: int) -> tuple[int, str]:
    """Parse a ``$$``-delimited block equation starting at ``lines[start]``.

    Returns ``(lines_consumed, latex)`` or ``(0, "")`` when the block is not
    closed (treated as plain text by the caller). Only paragraph-level math is
    recognised: the opening line's stripped text must start with ``$$``.
    """
    first = lines[start].strip()
    inner_first = first[2:]
    # Single-line ``$$ … $$``.
    if inner_first.rstrip().endswith("$$") and len(inner_first.rstrip()) >= 2:
        latex = inner_first.rstrip()[:-2].strip()
        return 1, latex
    # Multi-line: collect until a line whose stripped text ends with ``$$``.
    body: list[str] = []
    if inner_first.strip():
        body.append(inner_first.strip())
    j = start + 1
    while j < len(lines):
        s = lines[j].strip()
        if s.endswith("$$"):
            tail = s[:-2].strip()
            if tail:
                body.append(tail)
            return (j - start + 1), "\n".join(body).strip()
        body.append(lines[j])
        j += 1
    return 0, ""


def _consume_html_table(lines: list[str], start: int) -> tuple[int, str]:
    """Collect a ``<table>…</table>`` block (line-spanning). ``(consumed, html)``
    or ``(0, "")`` when no closing ``</table>`` is found."""
    buf: list[str] = []
    j = start
    while j < len(lines):
        buf.append(lines[j])
        if "</table>" in lines[j].lower():
            return (j - start + 1), "\n".join(buf).strip()
        j += 1
    return 0, ""


def _consume_pipe_table(
    lines: list[str], start: int
) -> tuple[int, list[list[str]], list[list[str]] | None]:
    """Parse a GFM pipe table whose header is ``lines[start]`` and delimiter is
    ``lines[start+1]``. Returns ``(consumed, body_rows, header_grid)``."""
    header = _split_pipe_row(lines[start])
    body: list[list[str]] = []
    j = start + 2  # skip header + delimiter
    while j < len(lines):
        s = lines[j].strip()
        if not s or "|" not in s:
            break
        body.append(_split_pipe_row(lines[j]))
        j += 1
    return (j - start), body, [header] if header else None
