"""Intermediate representation (IR) handed by parser adapters to the writer.

Parser engines do not write spec-shaped JSON directly. Each engine adapter
produces an :class:`IRDoc`; :func:`lightrag.sidecar.writer.write_sidecar`
turns that into ``*.parsed/`` files matching ``LightRAGSidecarFormat-zh.md``.

Why an in-process IR (not a serialized intermediate):

- One executable spec point. ``writer.py`` is the only place that knows id
  formats, placeholder tags, blockid computation, ``asset_dir`` truth value.
- Engine adapters only translate; they never embed knowledge of the on-disk
  format.
- The dataclasses below cover the spec contract plus an ``extras`` escape
  hatch on item-level objects so engine-specific signals (rowspan, OCR
  confidence, ...) can be passed through without spec churn.

Placeholder convention used by :attr:`IRBlock.content_template`:

- ``{{TBL:k}}`` — k is the placeholder key declared on the IRTable object
- ``{{IMG:k}}`` — IRDrawing
- ``{{EQ:k}}``  — block-level IREquation (``is_block=True``)
- ``{{EQI:k}}`` — inline IREquation (``is_block=False``); rendered without an
  id, never enters ``equations.json``

The writer expands these templates after id allocation. Adapters MUST emit
exactly one placeholder per item; multiple in-content placeholders sharing
the same key are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IRPosition:
    """Block-level position. See spec §八.

    ``type`` values: ``"paraid"`` (docx) / ``"bbox"`` (pdf) /
    ``"heading"`` (md) / ``"absolute"`` (text).
    """

    type: str
    anchor: Any = None
    range: list | None = None
    charspan: list[int] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.anchor is not None:
            out["anchor"] = self.anchor
        if self.range is not None:
            out["range"] = list(self.range)
        if self.charspan is not None:
            out["charspan"] = list(self.charspan)
        return out


@dataclass
class IRTable:
    """Spec §五. ``rows`` (preferred) or ``html`` describes the body.

    The writer renders ``{{TBL:placeholder_key}}`` in IRBlock.content_template
    as ``<table id="tb-..." format="json|html">body</table>``; ``format``
    is chosen by which payload the adapter populated.
    """

    placeholder_key: str
    rows: list[list[str]] | None = None
    html: str | None = None
    num_rows: int = 0
    num_cols: int = 0
    caption: str = ""
    footnotes: list[str] = field(default_factory=list)
    table_header: list[list[str]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    # Optional verbatim body to render inside the ``<table …>…</table>`` tag
    # in ``blocks.jsonl``. When set, the writer uses this string in the block
    # text instead of re-encoding ``rows`` via ``json.dumps`` — preserving
    # the parser's original whitespace/escaping when byte-equivalence with a
    # pre-existing output is required (currently: the native docx adapter
    # which keeps the docx-extracted JSON string verbatim). The
    # ``tables.json`` ``content`` field is unaffected and remains the
    # canonical ``json.dumps(rows, ensure_ascii=False)`` encoding.
    body_override: str | None = None


@dataclass
class IRDrawing:
    """Spec §四. ``asset_ref`` points to an :class:`AssetSpec` in IRDoc."""

    placeholder_key: str
    asset_ref: str
    fmt: str = ""
    caption: str = ""
    footnotes: list[str] = field(default_factory=list)
    src: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    # Optional verbatim path. When set, the writer emits this string in
    # both the ``blocks.jsonl`` ``<drawing path>`` attribute and the
    # ``drawings.json`` ``path`` field as-is — bypassing
    # ``asset_paths`` resolution and the ``block_drawing_path_style``
    # transformation. Used for linked / external image references (e.g.
    # ``<drawing path="https://…/img.png" />``) that point at bytes not
    # materialized into ``<base>.blocks.assets/``.
    path_override: str | None = None


@dataclass
class IREquation:
    """Spec §六. ``is_block=False`` ⇒ inline; not allocated an id, not written
    to ``equations.json``; rendered as ``<equation format="latex">…</equation>``
    in block text.
    """

    placeholder_key: str
    latex: str
    is_block: bool = True
    caption: str = ""
    footnotes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class IRBlock:
    """One content block (spec §3.2).

    ``content_template`` is the final block text with placeholder tokens
    embedded. The writer expands tokens once ids are assigned.
    """

    content_template: str
    heading: str = ""
    level: int = 0
    parent_headings: list[str] = field(default_factory=list)
    session_type: str = "body"
    table_slice: str = "none"
    table_header: str | None = None
    positions: list[IRPosition] = field(default_factory=list)
    tables: list[IRTable] = field(default_factory=list)
    drawings: list[IRDrawing] = field(default_factory=list)
    equations: list[IREquation] = field(default_factory=list)


@dataclass
class AssetSpec:
    """Describes one file that lands in ``<base>.blocks.assets/``.

    ``source`` may be:

    - :class:`pathlib.Path` to an existing file on disk (writer copies it);
    - :class:`bytes` payload (writer dumps it);
    - ``None`` when the file is already in place at ``<assets_dir>/<suggested_name>``
      (e.g. native docx parser writes assets during extraction); the writer
      then records its size without touching it.

    Carrier protocol: a drawing references the asset by :attr:`ref`; the
    writer resolves that to a concrete filename inside the assets dir and
    writes the result to both ``drawings.json`` (full relative path) and
    the ``<drawing path>`` attribute in ``blocks.jsonl``.
    """

    ref: str
    suggested_name: str
    source: Path | bytes | None = None


@dataclass
class IRDoc:
    """Top-level IR — the input to :func:`write_sidecar`."""

    document_name: str
    document_format: str
    doc_title: str
    split_option: dict[str, Any]
    blocks: list[IRBlock]
    assets: list[AssetSpec] = field(default_factory=list)
    bbox_attributes: dict[str, Any] | None = None
