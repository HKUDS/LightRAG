"""Word document content extraction with structure awareness.

Implements §2 of docx-extraction-guide-zh.md:
- Heading level via outlineLvl (with style inheritance)
- Automatic numbering restoration
- Table extraction (merged cells, cross-page headers)
- Special content: images, formulas (OMML→LaTeX), super/subscripts
- Paragraph anchor (w14:paraId)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, List, Optional
from xml.etree import ElementTree as ET

from .constants import MAX_HEADING_LENGTH
from .token_estimation import estimate_tokens

# ── Word XML namespaces ──────────────────────────────────────────────
_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

for prefix, uri in _NS.items():
    ET.register_namespace(prefix, uri)


# ── Data structures ──────────────────────────────────────────────────
@dataclass
class Paragraph:
    """A single paragraph extracted from .docx."""

    text: str
    para_id: str = ""  # w14:paraId
    outline_level: int = 9  # 0-8 = heading, 9 = body
    is_table: bool = False
    table_json: Optional[List[List[str]]] = None
    table_header_rows: Optional[List[List[str]]] = None  # cross-page header
    has_drawing: bool = False
    drawing_id: str = ""
    drawing_name: str = ""


@dataclass
class HeadingInfo:
    """Tracks current heading context during extraction."""

    title: str = ""
    level: int = 0
    parent_chain: list[str] = field(default_factory=list)


# ── Numbering Resolver ───────────────────────────────────────────────
class NumberingResolver:
    """Restore Word automatic numbering from numbering.xml."""

    def __init__(self, numbering_part: Any | None):
        self._counters: dict[str, dict[int, int]] = {}
        self._abstract: dict[str, dict] = {}
        self._num_map: dict[str, str] = {}
        if numbering_part is None:
            return
        try:
            tree = ET.fromstring(numbering_part.blob)
        except Exception:
            return

        # Parse abstractNum definitions
        for abstract in tree.findall("w:abstractNum", _NS):
            abstract_id = abstract.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}abstractNumId", "")
            levels: dict[int, dict] = {}
            for lvl in abstract.findall("w:lvl", _NS):
                ilvl = int(lvl.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl", "0"))
                fmt_el = lvl.find("w:numFmt", _NS)
                fmt = fmt_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "decimal") if fmt_el is not None else "decimal"
                txt_el = lvl.find("w:lvlText", _NS)
                txt = txt_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "") if txt_el is not None else ""
                start_el = lvl.find("w:start", _NS)
                start = int(start_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "1")) if start_el is not None else 1
                levels[ilvl] = {"fmt": fmt, "text": txt, "start": start}
            self._abstract[abstract_id] = levels

        # Parse num → abstractNum mapping
        for num in tree.findall("w:num", _NS):
            num_id = num.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numId", "")
            ref = num.find("w:abstractNumId", _NS)
            if ref is not None:
                self._num_map[num_id] = ref.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "")

    def resolve(self, num_id: str, ilvl: int) -> str:
        """Return rendered numbering prefix like '2.1' or '(a)'."""
        abstract_id = self._num_map.get(num_id, "")
        levels = self._abstract.get(abstract_id, {})
        if ilvl not in levels:
            return ""

        key = f"{num_id}"
        if key not in self._counters:
            self._counters[key] = {}

        counters = self._counters[key]
        if ilvl not in counters:
            counters[ilvl] = levels[ilvl].get("start", 1)
        else:
            counters[ilvl] += 1

        # Reset deeper levels
        for deeper in list(counters.keys()):
            if deeper > ilvl:
                del counters[deeper]

        template = levels[ilvl].get("text", "")
        fmt = levels[ilvl].get("fmt", "decimal")

        result = template
        for lvl_idx in range(ilvl + 1):
            val = counters.get(lvl_idx, levels.get(lvl_idx, {}).get("start", 1))
            replacement = self._format_number(val, fmt if lvl_idx == ilvl else "decimal")
            result = result.replace(f"%{lvl_idx + 1}", replacement)
        return result

    @staticmethod
    def _format_number(val: int, fmt: str) -> str:
        if fmt == "decimal":
            return str(val)
        if fmt == "lowerLetter":
            return chr(96 + val) if 1 <= val <= 26 else str(val)
        if fmt == "upperLetter":
            return chr(64 + val) if 1 <= val <= 26 else str(val)
        if fmt == "lowerRoman":
            romans = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
            return romans[val] if val < len(romans) else str(val)
        if fmt == "upperRoman":
            romans = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            return romans[val] if val < len(romans) else str(val)
        if fmt in ("chineseCounting", "ideographDigital"):
            cn = "零一二三四五六七八九十"
            return cn[val] if val < len(cn) else str(val)
        return str(val)


# ── Heading level detection ──────────────────────────────────────────
def _get_outline_level(para_element: ET.Element, styles_map: dict[str, int]) -> int:
    """Detect heading level via outlineLvl with style inheritance."""
    pPr = para_element.find("w:pPr", _NS)
    if pPr is not None:
        ol = pPr.find("w:outlineLvl", _NS)
        if ol is not None:
            val = ol.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "9")
            lvl = int(val)
            if lvl <= 8:
                return lvl

    # Fallback: check style reference
    if pPr is not None:
        style_el = pPr.find("w:pStyle", _NS)
        if style_el is not None:
            style_id = style_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "")
            if style_id in styles_map:
                return styles_map[style_id]
    return 9  # body text


def _build_styles_map(styles_part: Any | None) -> dict[str, int]:
    """Build style_id → outline_level map from styles.xml (with basedOn)."""
    result: dict[str, int] = {}
    if styles_part is None:
        return result
    try:
        tree = ET.fromstring(styles_part.blob)
    except Exception:
        return result

    # First pass: direct outlineLvl
    based_on: dict[str, str] = {}
    for style in tree.findall("w:style", _NS):
        sid = style.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId", "")
        pPr = style.find("w:pPr", _NS)
        if pPr is not None:
            ol = pPr.find("w:outlineLvl", _NS)
            if ol is not None:
                val = int(ol.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "9"))
                if val <= 8:
                    result[sid] = val

        bo = style.find("w:basedOn", _NS)
        if bo is not None:
            based_on[sid] = bo.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "")

    # Second pass: inherit from basedOn
    for sid, parent_sid in based_on.items():
        if sid not in result and parent_sid in result:
            result[sid] = result[parent_sid]

    return result


# ── Paragraph text extraction with special content ───────────────────
def _extract_paragraph_text(para_element: ET.Element) -> str:
    """Extract paragraph text preserving images, formulas, super/subscripts."""
    parts: list[str] = []

    for child in para_element.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        # Regular text
        if tag == "t":
            if child.text:
                parts.append(child.text)

        # Superscript / subscript
        elif tag == "vertAlign":
            val = child.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "")
            # Handled via run properties – text comes from <w:t>
            pass

        # Drawing (image)
        elif tag == "drawing":
            img_id, img_name = _extract_drawing_info(child)
            if img_id:
                parts.append(f'<drawing id="{img_id}" name="{img_name}" />')

        # OMML formula
        elif tag == "oMath" or tag == "oMathPara":
            latex = _omml_to_text(child)
            if latex:
                parts.append(f"<equation>{latex}</equation>")

    return "".join(parts)


def _extract_paragraph_text_with_formatting(para_element: ET.Element) -> str:
    """Extract paragraph text with super/subscript tags."""
    parts: list[str] = []

    for run in para_element.findall(".//w:r", _NS):
        rPr = run.find("w:rPr", _NS)
        vert = None
        if rPr is not None:
            va = rPr.find("w:vertAlign", _NS)
            if va is not None:
                vert = va.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "")

        for t in run.findall("w:t", _NS):
            text = t.text or ""
            if vert == "superscript":
                parts.append(f"<sup>{text}</sup>")
            elif vert == "subscript":
                parts.append(f"<sub>{text}</sub>")
            else:
                parts.append(text)

        # Drawing inside run
        for drawing in run.findall(".//w:drawing", _NS):
            img_id, img_name = _extract_drawing_info(drawing)
            if img_id:
                parts.append(f'<drawing id="{img_id}" name="{img_name}" />')

    # Math at paragraph level
    for math_el in para_element.findall(".//m:oMath", _NS):
        latex = _omml_to_text(math_el)
        if latex:
            parts.append(f"<equation>{latex}</equation>")

    return "".join(parts)


def _extract_drawing_info(drawing_el: ET.Element) -> tuple[str, str]:
    """Extract image id and name from a drawing element."""
    # Try docPr
    for dp in drawing_el.iter():
        tag = dp.tag.split("}")[-1] if "}" in dp.tag else dp.tag
        if tag == "docPr":
            img_id = dp.get("id", "")
            img_name = dp.get("name", "")
            return img_id, img_name
    return "", ""


def _omml_to_text(math_el: ET.Element) -> str:
    """Simple OMML → text extraction (best-effort, not full LaTeX)."""
    texts: list[str] = []
    for el in math_el.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag == "t" and el.text:
            texts.append(el.text)
    return " ".join(texts)


def _get_para_id(para_element: ET.Element) -> str:
    """Get w14:paraId from paragraph element."""
    for attr_name, val in para_element.attrib.items():
        if "paraId" in attr_name:
            return val
    return ""


def _infer_heading_level_by_text(text: str) -> int | None:
    """Fallback heading detection from paragraph text.

    Returns 1-based heading level (1..9) or None when not detected.
    Used when Word heading styles/outlineLvl are missing.
    """
    t = (text or "").strip()
    if not t:
        return None
    if len(t) > MAX_HEADING_LENGTH:
        return None

    # Chinese chapter style: "第三章 ...", "第1章 ..."
    if re.match(r"^第[一二三四五六七八九十百千0-9]+章\b", t):
        return 1

    # Numeric heading: 1 / 1.2 / 1.2.3 / 3.1.4.2 ...
    m = re.match(r"^(\d+(?:\.\d+){0,8})\s+", t)
    if m:
        level = m.group(1).count(".") + 1
        # Avoid treating likely list items as headings
        if level == 1 and len(t) > 60:
            return None
        return min(max(level, 1), 9)

    # Chinese section style: "一、", "（一）", "(一)"
    if re.match(r"^[一二三四五六七八九十]+、", t):
        return 2
    if re.match(r"^[（(][一二三四五六七八九十]+[）)]", t):
        return 3

    return None


# ── Table extraction ─────────────────────────────────────────────────
def _extract_table(table_element: ET.Element) -> tuple[list[list[str]], list[list[str]]]:
    """Extract table as JSON 2D array + cross-page header rows.

    Returns:
        (all_rows, header_rows) where each is list[list[str]]
    """
    rows: list[list[str]] = []
    header_rows: list[list[str]] = []

    for tr in table_element.findall("w:tr", _NS):
        cells: list[str] = []
        is_header = False

        # Check for table header (tblHeader)
        trPr = tr.find("w:trPr", _NS)
        if trPr is not None:
            th = trPr.find("w:tblHeader", _NS)
            if th is not None:
                is_header = True

        for tc in tr.findall("w:tc", _NS):
            cell_parts: list[str] = []
            for p in tc.findall("w:p", _NS):
                t = _extract_paragraph_text_with_formatting(p)
                if t:
                    cell_parts.append(t)
            cells.append("\n".join(cell_parts))

        rows.append(cells)
        if is_header:
            header_rows.append(cells)

    return rows, header_rows


def _table_to_content(rows: list[list[str]]) -> str:
    """Wrap table rows as <table>JSON</table>."""
    return f"<table>{json.dumps(rows, ensure_ascii=False)}</table>"


# ── Main extraction entry point ──────────────────────────────────────
def extract_docx_paragraphs(file_bytes: bytes) -> list[Paragraph]:
    """Extract all paragraphs from a .docx file with structure awareness.

    Returns a list of Paragraph objects in document reading order.
    """
    from docx import Document as DocxDocument  # type: ignore

    docx_file = BytesIO(file_bytes)
    doc = DocxDocument(docx_file)

    # Build styles map for heading detection
    styles_part = doc.part.part_related_by("http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles") if doc.part else None
    styles_map = _build_styles_map(styles_part)

    # Build numbering resolver
    numbering_part = None
    try:
        numbering_part = doc.part.part_related_by("http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering")
    except Exception:
        pass
    numbering = NumberingResolver(numbering_part)

    paragraphs: list[Paragraph] = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            para_id = _get_para_id(element)
            outline_level = _get_outline_level(element, styles_map)
            text = _extract_paragraph_text_with_formatting(element)

            # Numbering prefix
            pPr = element.find("w:pPr", _NS)
            if pPr is not None:
                numPr = pPr.find("w:numPr", _NS)
                if numPr is not None:
                    num_id_el = numPr.find("w:numId", _NS)
                    ilvl_el = numPr.find("w:ilvl", _NS)
                    if num_id_el is not None and ilvl_el is not None:
                        num_id = num_id_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "0")
                        ilvl = int(ilvl_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "0"))
                        if num_id != "0":
                            prefix = numbering.resolve(num_id, ilvl)
                            if prefix:
                                text = f"{prefix} {text}"

            # Fallback heading detection by text pattern when style-based detection fails
            if outline_level >= 9:
                inferred = _infer_heading_level_by_text(text)
                if inferred is not None:
                    outline_level = inferred - 1  # convert to 0-based (0..8)

            # Check for drawings
            has_drawing = len(element.findall(".//w:drawing", _NS)) > 0

            paragraphs.append(Paragraph(
                text=text,
                para_id=para_id,
                outline_level=outline_level,
                is_table=False,
                has_drawing=has_drawing,
            ))

        elif tag == "tbl":
            rows, header_rows = _extract_table(element)
            if not rows:
                continue

            # Get first and last paraIds from table cells
            all_paras = element.findall(".//w:p", _NS)
            first_id = _get_para_id(all_paras[0]) if all_paras else ""
            last_id = _get_para_id(all_paras[-1]) if all_paras else ""

            content = _table_to_content(rows)
            paragraphs.append(Paragraph(
                text=content,
                para_id=first_id,
                outline_level=9,
                is_table=True,
                table_json=rows,
                table_header_rows=header_rows if header_rows else None,
            ))
            # Store last_id on same paragraph for uuid_end
            if last_id:
                paragraphs[-1].drawing_id = last_id  # reuse field for table end id

    return paragraphs
