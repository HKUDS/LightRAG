"""Scenario fixtures for the native docx → SidecarWriter migration tests.

Each scenario describes:

- ``blocks``      — what ``extract_docx_blocks`` would return (the synthetic
  block dicts that the adapter consumes).
- ``parse_metadata`` — the dict the upstream parser fills in (the adapter
  consumes ``first_heading``, and ``doc_title`` — the smart assembler's
  explicit verdict, empty string allowed — when the key is present).
- ``assets``      — files the upstream extractor would have written into
  ``<base>.blocks.assets/`` before the IR builder runs. Maps relative names
  inside the asset dir → byte content.
- ``doc_id``      — fixed so blockid + sidecar ids are deterministic.
- ``file_path``   — used for canonical basename / doc_title fallback.

The captured outputs (``blocks.jsonl`` + per-modality JSONs + assets) live
under ``tests/parser/docx/golden/native_docx/<scenario>/``. The
production path (``LightRAG.parse_native``) must produce byte-identical
bytes vs those fixtures; the regen script under ``scripts/`` rewrites
them when the format intentionally changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _block(
    content: str,
    *,
    heading: str = "",
    level: int = 0,
    parent: list[str] | None = None,
    uuid: str = "p1",
    uuid_end: str | None = None,
    table_headers: list[Any] | None = None,
    table_chunk_role: str = "none",
) -> dict[str, Any]:
    """Build a synthetic block matching ``extract_docx_blocks`` output."""
    out: dict[str, Any] = {
        "uuid": uuid,
        "uuid_end": uuid_end if uuid_end is not None else uuid,
        "heading": heading,
        "content": content,
        "type": "text",
        "parent_headings": list(parent or []),
        "level": level,
        "table_chunk_role": table_chunk_role,
    }
    if table_headers is not None:
        out["table_headers"] = table_headers
    return out


@dataclass
class Scenario:
    name: str
    doc_id: str
    file_path: str  # canonical-ish; what the pipeline would pass
    blocks: list[dict[str, Any]]
    parse_metadata: dict[str, Any] = field(default_factory=dict)
    assets: dict[str, bytes] = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    # --- 1: text-only, multi-heading -----------------------------------
    Scenario(
        name="text_only_hierarchy",
        doc_id="doc-aaaa111122223333aaaa111122223333",
        file_path="paper.docx",
        parse_metadata={"first_heading": "Introduction"},
        blocks=[
            _block(
                "# Introduction",
                heading="Introduction",
                level=1,
                uuid="h1",
            ),
            _block(
                "Body paragraph one.",
                heading="Introduction",
                level=1,
                uuid="p1",
                uuid_end="p2",
            ),
            _block(
                "## Background",
                heading="Background",
                level=2,
                parent=["Introduction"],
                uuid="h2",
            ),
            _block(
                "Sub body.",
                heading="Background",
                level=2,
                parent=["Introduction"],
                uuid="p3",
            ),
        ],
    ),
    # --- 2: block + inline equations -----------------------------------
    Scenario(
        name="equations_block_and_inline",
        doc_id="doc-bbbb222233334444bbbb222233334444",
        file_path="formulas.docx",
        parse_metadata={"first_heading": "Equations"},
        blocks=[
            _block(
                "# Equations",
                heading="Equations",
                level=1,
                uuid="h1",
            ),
            _block(
                # Inline equation (no surrounding \n on either side)
                "Energy is <equation>E=mc^2</equation> per Einstein.",
                heading="Equations",
                level=1,
                uuid="p1",
            ),
            _block(
                # Block equation (wedged between newlines)
                "Consider:\n<equation>x^2 + y^2 = r^2</equation>\nThe circle equation.",
                heading="Equations",
                level=1,
                uuid="p2",
            ),
            _block(
                # Block at content edge (start == 0)
                "<equation>a + b = c</equation>\ntext after",
                heading="Equations",
                level=1,
                uuid="p3",
            ),
        ],
    ),
    # --- 3: tables with and without table_headers ----------------------
    Scenario(
        name="tables_mixed",
        doc_id="doc-cccc333344445555cccc333344445555",
        file_path="report.docx",
        parse_metadata={"first_heading": "Report"},
        blocks=[
            _block(
                "# Report",
                heading="Report",
                level=1,
                uuid="h1",
            ),
            _block(
                # Table with table_headers (cross-page repeating)
                'See table:\n<table>[["X","Y"],["1","2"],["3","4"]]</table>',
                heading="Report",
                level=1,
                uuid="t1",
                table_headers=[[["X", "Y"]]],  # one table, one header row
            ),
            _block(
                # Table without table_headers
                'Plain table:\n<table>[["a","b"]]</table>',
                heading="Report",
                level=1,
                uuid="t2",
            ),
            _block(
                # Two tables in one block
                '<table>[["p"]]</table>\nthen\n<table>[["q","r"],["s","t"]]</table>',
                heading="Report",
                level=1,
                uuid="t3",
                table_headers=[None, [["q", "r"]]],
            ),
        ],
    ),
    # --- 4: drawings + assets ------------------------------------------
    Scenario(
        name="drawings_with_assets",
        doc_id="doc-dddd444455556666dddd444455556666",
        file_path="diagrams.docx",
        parse_metadata={"first_heading": "Diagrams"},
        assets={
            "fig1.png": b"\x89PNG\r\n\x1a\n-fig1-fake",
            "fig2.jpg": b"\xff\xd8\xff\xe0-fig2-fake",
        },
        blocks=[
            _block(
                "# Diagrams",
                heading="Diagrams",
                level=1,
                uuid="h1",
            ),
            _block(
                "Figure one:\n"
                '<drawing id="x" format="png" '
                'path="diagrams.blocks.assets/fig1.png" '
                'src="docx://image1" />\n'
                "Figure two:\n"
                '<drawing id="y" format="jpg" '
                'path="diagrams.blocks.assets/fig2.jpg" '
                'src="docx://image2" />',
                heading="Diagrams",
                level=1,
                uuid="p1",
            ),
        ],
    ),
    # --- 5: all modalities mixed ---------------------------------------
    Scenario(
        name="all_modalities",
        doc_id="doc-eeee555566667777eeee555566667777",
        file_path="combo.docx",
        parse_metadata={"first_heading": "Combined"},
        assets={"pic.png": b"PNG-combo"},
        blocks=[
            _block(
                "# Combined",
                heading="Combined",
                level=1,
                uuid="h1",
            ),
            _block(
                "Look at this figure:\n"
                '<drawing id="z" format="png" '
                'path="combo.blocks.assets/pic.png" '
                'src="docx://img" />\n'
                "Plus a table:\n"
                '<table>[["α","β"],["γ","δ"]]</table>\n'
                "And a block equation:\n"
                "<equation>F = ma</equation>\n"
                "And an inline <equation>v=d/t</equation> here.",
                heading="Combined",
                level=1,
                uuid="p1",
            ),
        ],
    ),
    # --- 6: empty block dropped ----------------------------------------
    Scenario(
        name="empty_block_dropped",
        doc_id="doc-ffff666677778888ffff666677778888",
        file_path="sparse.docx",
        parse_metadata={"first_heading": "Sparse"},
        blocks=[
            _block(
                "# Sparse",
                heading="Sparse",
                level=1,
                uuid="h1",
            ),
            _block(
                "   \n   ",  # strips to empty — must be dropped
                heading="Sparse",
                level=1,
                uuid="p_empty",
            ),
            _block(
                "Real content after empty.",
                heading="Sparse",
                level=1,
                uuid="p_real",
            ),
        ],
    ),
    # --- 7: external / linked image references ------------------------
    # DOCX can carry ``<a:blip r:link="rId…"/>`` references to image
    # targets that live outside the package — the upstream extractor
    # then emits ``<drawing path="<external URL or unresolved path>" />``
    # WITHOUT writing bytes into ``<base>.blocks.assets/``. The adapter
    # must pass those paths through verbatim (both in ``blocks.jsonl``
    # and ``drawings.json``); turning them into AssetSpecs with
    # ``source=None`` would make the writer warn-and-skip → ``path=""``,
    # losing the only reference downstream consumers have.
    Scenario(
        name="external_image_link",
        doc_id="doc-1111aaaa2222bbbb1111aaaa2222bbbb",
        file_path="linked.docx",
        parse_metadata={"first_heading": "Linked"},
        # No on-disk assets — the path points elsewhere.
        assets={},
        blocks=[
            _block(
                "# Linked",
                heading="Linked",
                level=1,
                uuid="h1",
            ),
            _block(
                "See the diagram online:\n"
                '<drawing id="z" format="png" '
                'path="https://example.com/diagrams/architecture.png" '
                'src="docx://external" />\n'
                "And a relative-but-not-asset path:\n"
                '<drawing id="z2" format="gif" '
                'path="../images/legacy.gif" '
                'src="docx://legacy" />',
                heading="Linked",
                level=1,
                uuid="p1",
            ),
        ],
    ),
    # --- 8: missing paraid ---------------------------------------------
    Scenario(
        name="missing_paraid",
        doc_id="doc-99990000111122229999000011112222",
        file_path="legacy.docx",
        parse_metadata={"first_heading": ""},  # no headings at all
        blocks=[
            _block(
                "Just plain text without a heading.",
                heading="",
                level=0,
                uuid="",  # missing
                uuid_end="",
            ),
            _block(
                "Another paragraph with no paraId.",
                heading="",
                level=0,
                uuid="",
                uuid_end="",
            ),
        ],
    ),
]


__all__ = ["Scenario", "SCENARIOS", "_block"]
