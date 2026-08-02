"""Optional multimodal and graph additions for query responses."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any

from lightrag.api.assets import WorkspaceAssetCatalog
from lightrag.api.graph_projection import _relation_id


_DRAWING_TAG_RE = re.compile(r"<drawing\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*/?>", re.I)
_TABLE_TAG_RE = re.compile(r"<table\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*>(.*?)</table>", re.I | re.S)
_EQUATION_TAG_RE = re.compile(r"<equation\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*>(.*?)</equation>", re.I | re.S)


class QueryEnricher:
    def __init__(self, document_manager: Any):
        self.catalog = WorkspaceAssetCatalog(document_manager)
        self._items: dict[str, dict[str, Any]] | None = None
        self._block_pages: dict[str, int | None] = {}

    def _load_items(self) -> dict[str, dict[str, Any]]:
        if self._items is not None:
            return self._items
        items: dict[str, dict[str, Any]] = {}
        root, _workspace = self.catalog._root_and_workspace()
        for parsed_dir in self.catalog._sidecar_directories(root):
            blocks = sorted(parsed_dir.glob("*.blocks.jsonl"))
            for blocks_path in blocks:
                _doc_id, block_map = self.catalog._read_meta(blocks_path)
                for block_id, block in block_map.items():
                    self._block_pages[block_id] = self.catalog._page_from_block(block)
                for suffix, key in (("tables.json", "tables"), ("equations.json", "equations"), ("drawings.json", "drawings")):
                    sidecar = blocks_path.with_name(blocks_path.name.removesuffix(".blocks.jsonl") + f".{suffix}")
                    try:
                        payload = json.loads(sidecar.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        continue
                    values = payload.get(key) if isinstance(payload, dict) else None
                    if isinstance(values, dict):
                        items.update({str(item_id): dict(item) for item_id, item in values.items() if isinstance(item, dict)})
        self._items = items
        return items

    def _page_for_item(self, item: dict[str, Any]) -> int | None:
        for key in ("page_idx", "page", "page_no", "page_number"):
            value = item.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return self._block_pages.get(str(item.get("blockid") or ""))

    @staticmethod
    def _video_fields(item: dict[str, Any]) -> dict[str, Any]:
        extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
        if str(extras.get("media_type") or item.get("media_type") or "") != "video_contact_sheet":
            return {}
        fields: dict[str, Any] = {"media_type": "video_contact_sheet"}
        for key in ("scene_id", "start_ms", "end_ms", "frame_timestamps_ms", "manifest_ref"):
            if key in extras:
                fields[key] = extras[key]
        return fields

    @staticmethod
    def _chunk_document_id(chunk: dict[str, Any]) -> str | None:
        value = chunk.get("full_doc_id") or chunk.get("document_id")
        return str(value) if value else None

    @staticmethod
    def _display_name(path: str | None) -> str | None:
        if not path:
            return None
        return Path(path).name

    def enrich_references(
        self, references: list[dict[str, Any]], chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for reference in references:
            result = dict(reference)
            ref_id = str(reference.get("reference_id") or "")
            matching = [chunk for chunk in chunks if str(chunk.get("reference_id") or "") == ref_id]
            if not matching and reference.get("file_path"):
                matching = [chunk for chunk in chunks if chunk.get("file_path") == reference.get("file_path")]
            document_id = next((self._chunk_document_id(chunk) for chunk in matching if self._chunk_document_id(chunk)), None)
            file_path = str(reference.get("file_path") or (matching[0].get("file_path") if matching else ""))
            if document_id:
                result.setdefault("document_id", document_id)
            if file_path:
                result.setdefault("display_name", self._display_name(file_path))
            pages: list[int] = []
            video_ranges: list[dict[str, Any]] = []
            for chunk in matching:
                for match in _DRAWING_TAG_RE.finditer(str(chunk.get("content") or "")):
                    item = self._load_items().get(match.group(1))
                    page = self._page_for_item(item or {})
                    if page is not None:
                        pages.append(page)
                    video = self._video_fields(item or {})
                    if video:
                        video_ranges.append(video)
            if pages:
                result.setdefault("page", min(pages))
            if video_ranges:
                result.setdefault("video_scenes", video_ranges)
            enriched.append(result)
        return enriched

    def content_blocks(
        self, response: str, chunks: list[dict[str, Any]], *, include_graph_context: bool = False, data: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        blocks: list[dict[str, Any]] = [{"type": "markdown", "markdown": response}]
        seen: set[str] = set()
        items = self._load_items()
        for chunk in chunks:
            content = str(chunk.get("content") or "")
            for match in _DRAWING_TAG_RE.finditer(content):
                drawing_id = match.group(1)
                if drawing_id in seen:
                    continue
                seen.add(drawing_id)
                asset = self.catalog.by_drawing_id(drawing_id)
                if asset is None:
                    continue
                item = items.get(drawing_id) or {}
                video = self._video_fields(item)
                if video:
                    blocks.append({
                        "type": "video_scene",
                        "asset_id": asset.asset_id,
                        "caption": asset.caption,
                        **video,
                    })
                else:
                    blocks.append({
                        "type": "image",
                        "asset_id": asset.asset_id,
                        "caption": asset.caption,
                        "page": asset.page,
                    })
            for match in _TABLE_TAG_RE.finditer(content):
                table_id = match.group(1)
                if table_id in seen:
                    continue
                seen.add(table_id)
                item = items.get(table_id) or {}
                table_content = str(item.get("content") or match.group(2) or "")
                if str(item.get("format")) == "json":
                    try:
                        rows = json.loads(table_content)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        rows = []
                    if isinstance(rows, list):
                        table_content = "<table>" + "".join(
                            "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
                            for row in rows if isinstance(row, list)
                        ) + "</table>"
                blocks.append({"type": "table", "html": table_content, "caption": item.get("caption") or None})
            for match in _EQUATION_TAG_RE.finditer(content):
                equation_id = match.group(1)
                if equation_id in seen:
                    continue
                seen.add(equation_id)
                item = items.get(equation_id) or {}
                blocks.append({"type": "equation", "latex": str(item.get("latex") or item.get("content") or match.group(2) or "")})

        graph_context: dict[str, Any] | None = None
        if include_graph_context:
            data = data or {}
            graph_nodes: list[str] = []
            for entity in data.get("entities") or []:
                if isinstance(entity, dict):
                    value = entity.get("entity_name") or entity.get("name") or entity.get("id")
                else:
                    value = entity
                if value is not None and str(value) not in graph_nodes:
                    graph_nodes.append(str(value))
            graph_edges: list[str] = []
            for relation in data.get("relationships") or []:
                if not isinstance(relation, dict):
                    continue
                source = str(relation.get("src_id") or relation.get("source") or "")
                target = str(relation.get("tgt_id") or relation.get("target") or "")
                if source and target:
                    graph_edges.append(_relation_id(relation, source, target))
            graph_context = {"nodes": graph_nodes, "edges": graph_edges}
        return blocks, graph_context


__all__ = ["QueryEnricher"]
