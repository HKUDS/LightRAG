#!/usr/bin/env python3
"""
ABOUTME: Shared drawing/image extraction utilities for DOCX parsing and editing
ABOUTME: Resolves w:drawing -> a:blip relationships, exports embedded images, builds placeholders
"""

from __future__ import annotations

import posixpath
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from html import escape, unescape
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

try:
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover
    from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPE_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
SOURCE_DOCUMENT_PART = "/word/document.xml"

# Match old and new drawing placeholders (requires id/name, allows extra attributes)
DRAWING_PATTERN = re.compile(
    r'<drawing\b(?=[^>]*\bid="[^"]*")(?=[^>]*\bname="[^"]*")[^>]*/>'
)
DRAWING_TAG_PATTERN = re.compile(r"<drawing\b[^>]*/>")
DRAWING_ATTR_PATTERN = re.compile(r'([a-zA-Z_][\w:.-]*)="([^"]*)"')


@dataclass
class DrawingRelationship:
    """Relationship metadata for a single relationship ID."""

    rel_id: str
    target: str
    target_mode: str
    rel_type: str
    part_name: Optional[str] = None
    content_type: Optional[str] = None
    image_format: Optional[str] = None


@dataclass
class DrawingExtractionContext:
    """Context used to resolve and export drawing images for one DOCX file."""

    docx_path: Path
    blocks_output_path: Optional[Path] = None
    export_dir_name: Optional[str] = None
    export_dir_path: Optional[Path] = None
    relationships: Dict[str, DrawingRelationship] = field(default_factory=dict)
    _exported_part_to_relpath: Dict[str, str] = field(default_factory=dict)
    _used_filenames: Dict[str, str] = field(default_factory=dict)

    def resolve_relationship(self, rel_id: str) -> Optional[DrawingRelationship]:
        return self.relationships.get(rel_id)

    def export_embedded_image(self, rel: DrawingRelationship) -> Optional[str]:
        """
        Export an embedded image relationship target to export_dir.

        Returns:
            Relative path like "<blocks_stem>.image/image1.png" if exported,
            or None when export is not applicable.
        """
        if not self.export_dir_path or not self.export_dir_name:
            return None
        if rel.target_mode.lower() == "external":
            return None
        if not rel.part_name:
            return None
        if rel.part_name in self._exported_part_to_relpath:
            return self._exported_part_to_relpath[rel.part_name]

        zip_member = rel.part_name.lstrip("/")
        try:
            with zipfile.ZipFile(self.docx_path, "r") as zf:
                blob = zf.read(zip_member)
        except Exception:
            return None

        filename = self._dedupe_filename(PurePosixPath(rel.part_name).name or "image")
        output_file = self.export_dir_path / filename
        output_file.write_bytes(blob)

        rel_path = str(PurePosixPath(self.export_dir_name) / filename)
        self._exported_part_to_relpath[rel.part_name] = rel_path
        return rel_path

    def _dedupe_filename(self, base_name: str) -> str:
        if base_name not in self._used_filenames:
            self._used_filenames[base_name] = base_name
            return base_name

        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        index = 2
        while True:
            candidate = f"{stem}_{index}{suffix}"
            if candidate not in self._used_filenames:
                self._used_filenames[candidate] = candidate
                return candidate
            index += 1


def _normalize_image_format(ext_or_type: str) -> Optional[str]:
    if not ext_or_type:
        return None
    value = ext_or_type.strip().lower()

    # Content-Type
    if value.startswith("image/"):
        value = value.split("/", 1)[1]
        if "+" in value:
            value = value.split("+", 1)[0]
        if value.startswith("x-"):
            value = value[2:]

    # Extension (with or without leading dot)
    value = value.lstrip(".")
    if value == "jpg":
        return "jpeg"
    if value in {"jpeg", "png", "gif", "bmp", "tiff", "webp", "svg", "emf", "wmf"}:
        return value
    return value or None


def _infer_format_from_target(target: str) -> Optional[str]:
    if not target:
        return None
    parsed = urlparse(target)
    path = parsed.path if parsed.scheme else target
    suffix = PurePosixPath(path).suffix
    return _normalize_image_format(suffix)


def _resolve_part_name(source_part_name: str, target: str) -> str:
    if target.startswith("/"):
        return posixpath.normpath(target)
    source_dir = posixpath.dirname(source_part_name)
    joined = posixpath.join(source_dir, target)
    normalized = posixpath.normpath(joined)
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized


def create_drawing_context(
    docx_path: str,
    blocks_output_path: Optional[str] = None,
) -> DrawingExtractionContext:
    """
    Create extraction context for a DOCX file.

    If blocks_output_path is provided, this also prepares `<blocks_stem>.image/`
    beside the blocks file and clears any previous content.
    """
    docx_file = Path(docx_path)
    ctx = DrawingExtractionContext(docx_path=docx_file)

    if blocks_output_path:
        output_path = Path(blocks_output_path)
        export_dir_name = f"{output_path.stem}.image"
        export_dir_path = output_path.parent / export_dir_name
        if export_dir_path.exists():
            shutil.rmtree(export_dir_path)
        export_dir_path.mkdir(parents=True, exist_ok=True)
        ctx.blocks_output_path = output_path
        ctx.export_dir_name = export_dir_name
        ctx.export_dir_path = export_dir_path

    _load_relationships(ctx)
    return ctx


def _load_relationships(ctx: DrawingExtractionContext) -> None:
    rels_xml = "word/_rels/document.xml.rels"
    content_types_xml = "[Content_Types].xml"

    overrides: Dict[str, str] = {}
    defaults: Dict[str, str] = {}

    try:
        with zipfile.ZipFile(ctx.docx_path, "r") as zf:
            if content_types_xml in zf.namelist():
                ct_root = ET.parse(zf.open(content_types_xml)).getroot()
                for node in ct_root.findall(f".//{{{CONTENT_TYPE_NS}}}Override"):
                    part_name = node.get("PartName")
                    content_type = node.get("ContentType")
                    if part_name and content_type:
                        overrides[part_name] = content_type
                for node in ct_root.findall(f".//{{{CONTENT_TYPE_NS}}}Default"):
                    ext = node.get("Extension")
                    content_type = node.get("ContentType")
                    if ext and content_type:
                        defaults[ext.lower()] = content_type

            if rels_xml not in zf.namelist():
                return
            rels_root = ET.parse(zf.open(rels_xml)).getroot()
    except Exception:
        return

    for rel in rels_root.findall(f".//{{{REL_NS}}}Relationship"):
        rel_id = rel.get("Id")
        target = rel.get("Target", "")
        target_mode = rel.get("TargetMode", "")
        rel_type = rel.get("Type", "")
        if not rel_id:
            continue

        part_name = None
        content_type = None
        image_format = None

        if target_mode.lower() != "external":
            part_name = _resolve_part_name(SOURCE_DOCUMENT_PART, target)
            if part_name:
                content_type = overrides.get(part_name)
                if not content_type:
                    ext = PurePosixPath(part_name).suffix.lower().lstrip(".")
                    content_type = defaults.get(ext)
                image_format = _normalize_image_format(content_type or _infer_format_from_target(part_name))
        else:
            image_format = _normalize_image_format(_infer_format_from_target(target))

        ctx.relationships[rel_id] = DrawingRelationship(
            rel_id=rel_id,
            target=target,
            target_mode=target_mode,
            rel_type=rel_type,
            part_name=part_name,
            content_type=content_type,
            image_format=image_format,
        )


def _extract_blip_relationship(drawing_elem) -> Optional[Tuple[str, str]]:
    for blip in drawing_elem.findall(".//a:blip", NS):
        # Prefer explicit external links when both link/embed are present on one blip.
        # Word may keep an embedded cache for linked pictures.
        rel_link = blip.get(f"{{{NS['r']}}}link")
        if rel_link:
            return "link", rel_link
        rel_embed = blip.get(f"{{{NS['r']}}}embed")
        if rel_embed:
            return "embed", rel_embed
    return None


def _build_placeholder(attrs: Dict[str, str]) -> str:
    ordered_keys = ["id", "name", "path", "format"]
    pieces = []
    for key in ordered_keys:
        if key in attrs and attrs[key] is not None:
            pieces.append(f'{key}="{escape(str(attrs[key]), quote=True)}"')

    # Preserve extra attributes deterministically (sorted by name)
    for key in sorted(k for k in attrs.keys() if k not in ordered_keys):
        value = attrs[key]
        if value is not None:
            pieces.append(f'{key}="{escape(str(value), quote=True)}"')

    return f"<drawing {' '.join(pieces)} />"


def extract_drawing_placeholder_from_element(
    drawing_elem,
    context: Optional[DrawingExtractionContext] = None,
    include_extended_attrs: bool = True,
) -> str:
    """
    Build a <drawing ... /> placeholder from a w:drawing element.

    Behavior:
    - Always emits id/name from wp:docPr when present.
    - For embedded images (a:blip@r:embed): exports image and sets path/format.
    - For linked images (a:blip@r:link): does not download; path is original link target.
    - When no image reference exists (e.g. chart drawing): keeps id/name only.
    """
    doc_pr = drawing_elem.find(".//wp:docPr", NS)
    attrs = {
        "id": doc_pr.get("id", "") if doc_pr is not None else "",
        "name": doc_pr.get("name", "") if doc_pr is not None else "",
    }

    if include_extended_attrs:
        rel_ref = _extract_blip_relationship(drawing_elem)
        if rel_ref is not None and context is not None:
            rel_kind, rel_id = rel_ref
            rel = context.resolve_relationship(rel_id)
            if rel is not None:
                if rel_kind == "embed" and rel.rel_type == IMAGE_REL_TYPE:
                    rel_path = context.export_embedded_image(rel)
                    if rel_path:
                        attrs["path"] = rel_path
                    if rel.image_format:
                        attrs["format"] = rel.image_format
                elif rel_kind == "link":
                    if rel.target:
                        attrs["path"] = rel.target
                    if rel.image_format:
                        attrs["format"] = rel.image_format

    return _build_placeholder(attrs)


def parse_drawing_attributes(placeholder: str) -> Dict[str, str]:
    """Parse attributes from a <drawing ... /> placeholder."""
    return {
        name: unescape(value)
        for name, value in DRAWING_ATTR_PATTERN.findall(placeholder)
    }


def normalize_drawing_placeholder(
    placeholder: str,
    include_extended_attrs: bool = False,
) -> str:
    """
    Normalize one drawing placeholder into canonical attribute order.

    Args:
        placeholder: Input placeholder string
        include_extended_attrs: If False, keeps only id/name.
    """
    attrs = parse_drawing_attributes(placeholder)
    normalized = {
        "id": attrs.get("id", ""),
        "name": attrs.get("name", ""),
    }
    if include_extended_attrs:
        if "path" in attrs:
            normalized["path"] = attrs["path"]
        if "format" in attrs:
            normalized["format"] = attrs["format"]
        for key, value in attrs.items():
            if key not in {"id", "name", "path", "format"}:
                normalized[key] = value
    return _build_placeholder(normalized)


def normalize_drawing_placeholders_in_text(
    text: str,
    include_extended_attrs: bool = False,
) -> str:
    """Normalize all drawing placeholders inside a text blob."""
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        return normalize_drawing_placeholder(
            match.group(0),
            include_extended_attrs=include_extended_attrs,
        )

    return DRAWING_TAG_PATTERN.sub(_replace, text)
