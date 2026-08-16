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
from collections.abc import Callable
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
      * ``"external"`` — keep as an external link; ``url`` is carried in the
        drawing's ``src`` while ``path`` stays empty (no bytes materialized).
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


class _ForwardFinder:
    """Cache a forward search over one monotonic query stream.

    Each finder belongs to one parser phase whose query starts advance
    left-to-right. Keeping its last answer lets malformed candidates share the
    scan that reached their common delimiter instead of each scanning the rest
    of a long line again.
    """

    def __init__(self, line: str, predicate: Callable[[str], bool]) -> None:
        self._line = line
        self._predicate = predicate
        self._match: int | None = None
        self._exhausted = False

    def find(self, start: int) -> int | None:
        if self._match is not None and self._match >= start:
            return self._match
        if self._exhausted:
            return None

        for index in range(start, len(self._line)):
            if self._predicate(self._line[index]):
                self._match = index
                return index

        self._exhausted = True
        return None


def _replace_inline_images(line: str, replace: Callable[[str], str]) -> str:
    """Replace supported inline images without retrying a regex at each ``![``.

    The native Markdown subset accepts ``![alt](src "optional title")`` with
    either an angle-bracketed or whitespace-free source.  A global regular
    expression replacement tried the complete expression at every ``![``.
    Malformed input with many candidate prefixes consequently rescanned the
    same suffix quadratically. This scanner visits candidates left-to-right and
    keeps every cached forward search in its own monotonic parser phase,
    making its work linear in the line length while retaining the supported
    grammar.
    """

    if "![" not in line:
        return line

    close_bracket = _ForwardFinder(line, lambda char: char == "]")
    close_paren = _ForwardFinder(line, lambda char: char == ")")
    close_angle = _ForwardFinder(line, lambda char: char == ">")
    whitespace = _ForwardFinder(line, str.isspace)
    source_non_whitespace = _ForwardFinder(line, lambda char: not char.isspace())

    # The angle-bracket and bare-source alternatives inspect tails in a
    # different order. Title handling makes two non-whitespace queries per
    # alternative, so each phase needs its own monotonic cache as well.
    angle_tail_non_whitespace = _ForwardFinder(line, lambda char: not char.isspace())
    angle_after_title_non_whitespace = _ForwardFinder(
        line, lambda char: not char.isspace()
    )
    angle_quote = _ForwardFinder(line, lambda char: char == '"')
    bare_tail_non_whitespace = _ForwardFinder(line, lambda char: not char.isspace())
    bare_after_title_non_whitespace = _ForwardFinder(
        line, lambda char: not char.isspace()
    )
    bare_quote = _ForwardFinder(line, lambda char: char == '"')

    def _tail_end(
        source_end: int,
        *,
        tail_non_whitespace: _ForwardFinder,
        after_title_non_whitespace: _ForwardFinder,
        quote: _ForwardFinder,
    ) -> int | None:
        """Return the end of an optional title plus the required ``)``."""

        next_char = tail_non_whitespace.find(source_end)
        if next_char is None:
            return None
        if line[next_char] == ")":
            return next_char + 1

        # The title is optional, but when present it needs at least one
        # whitespace character before its opening quote.
        if next_char == source_end or line[next_char] != '"':
            return None
        title_end = quote.find(next_char + 1)
        if title_end is None:
            return None
        closing_paren = after_title_non_whitespace.find(title_end + 1)
        if closing_paren is not None and line[closing_paren] == ")":
            return closing_paren + 1
        return None

    def _match_end(start: int) -> tuple[int, str] | None:
        alt_end = close_bracket.find(start + 2)
        if alt_end is None or alt_end + 1 >= len(line) or line[alt_end + 1] != "(":
            return None
        source_start = source_non_whitespace.find(alt_end + 2)
        if source_start is None:
            return None

        # Preserve the original alternation order: a valid angle-bracketed
        # source wins over the bare-source fallback.
        if line[source_start] == "<":
            angle_end = close_angle.find(source_start + 1)
            if angle_end is not None:
                end = _tail_end(
                    angle_end + 1,
                    tail_non_whitespace=angle_tail_non_whitespace,
                    after_title_non_whitespace=angle_after_title_non_whitespace,
                    quote=angle_quote,
                )
                if end is not None:
                    return end, line[source_start : angle_end + 1]

        whitespace_end = whitespace.find(source_start)
        paren_end = close_paren.find(source_start)
        source_end = min(
            len(line) if whitespace_end is None else whitespace_end,
            len(line) if paren_end is None else paren_end,
        )
        if source_end == source_start:
            return None
        end = _tail_end(
            source_end,
            tail_non_whitespace=bare_tail_non_whitespace,
            after_title_non_whitespace=bare_after_title_non_whitespace,
            quote=bare_quote,
        )
        if end is not None:
            return end, line[source_start:source_end]
        return None

    parts: list[str] = []
    position = 0
    while True:
        start = line.find("![", position)
        if start < 0:
            parts.append(line[position:])
            return "".join(parts)
        matched = _match_end(start)
        if matched is None:
            # No match begins at this prefix.  Advance only past ``![`` so a
            # nested, valid image remains visible to the next iteration.
            parts.append(line[position : start + 2])
            position = start + 2
            continue
        end, src = matched
        parts.append(line[position:start])
        parts.append(replace(src))
        position = end


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

    def _resolve_image(src: str) -> str:
        src = src.strip()
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
        new_line = _replace_inline_images(line, _resolve_image)
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
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
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
