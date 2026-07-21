"""Workspace-scoped multimodal asset discovery and safe file serving.

The parser sidecars intentionally keep bytes on disk and refer to them using
relative paths.  This module is the API boundary for those files: callers see
opaque, deterministic ``ast_`` identifiers and never receive a filesystem
path.  The catalog is rebuilt from the selected workspace's sidecars on each
request; this keeps it correct after parsing, retries and document deletion
without introducing a second metadata database.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightrag.constants import PARSED_DIR_NAME


_DRAWING_ID_RE = re.compile(r"\bim-([0-9a-f]{32})-(\d{4})\b")
_PAGE_KEYS = ("page_idx", "page", "page_no", "page_number")


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    document_id: str
    path: Path
    filename: str
    mime_type: str
    page: int | None = None
    caption: str | None = None
    drawing_id: str | None = None


def _workspace_doc_manager(value: Any) -> Any:
    resolver = getattr(value, "_resolve_workspace_value", None)
    return resolver() if callable(resolver) else value


def _stable_asset_id(workspace: str, document_id: str, relative_path: str) -> str:
    payload = f"{workspace}\0{document_id}\0{relative_path}".encode("utf-8")
    # Hex is deliberately used instead of exposing a path or a sequential
    # counter.  It is opaque, URL-safe and stable across process restarts.
    return "ast_" + hashlib.sha256(payload).hexdigest()[:26]


def _safe_relative_path(root: Path, candidate: Path) -> Path | None:
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except (OSError, ValueError):
        return None


def _item_page(item: dict[str, Any]) -> int | None:
    for key in _PAGE_KEYS:
        value = item.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


class WorkspaceAssetCatalog:
    """Resolve multimodal assets for the request-selected workspace."""

    def __init__(self, document_manager: Any):
        self.document_manager = document_manager
        self._records_cache: list[AssetRecord] | None = None

    def _root_and_workspace(self) -> tuple[Path, str]:
        manager = _workspace_doc_manager(self.document_manager)
        root = Path(getattr(manager, "input_dir", ""))
        workspace = str(getattr(manager, "workspace", "") or "")
        return root, workspace

    def _sidecar_directories(self, root: Path) -> list[Path]:
        parsed_root = root / PARSED_DIR_NAME
        if not parsed_root.is_dir():
            return []
        # Parsed artifacts are one level below __parsed__, but recurse one
        # level as well for deployments that preserve parser bundles.
        candidates = [p for p in parsed_root.glob("*.parsed*") if p.is_dir()]
        candidates.extend(p for p in parsed_root.glob("*/*.parsed*") if p.is_dir())
        return sorted(set(candidates))

    @staticmethod
    def _read_meta(blocks_path: Path) -> tuple[str | None, dict[str, dict[str, Any]]]:
        block_pages: dict[str, dict[str, Any]] = {}
        document_id: str | None = None
        try:
            lines = blocks_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None, block_pages
        if lines:
            try:
                meta = json.loads(lines[0])
                if isinstance(meta, dict):
                    raw_id = meta.get("doc_id")
                    document_id = str(raw_id) if raw_id else None
            except json.JSONDecodeError:
                pass
        for line in lines[1:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict) or item.get("type") != "content":
                continue
            block_id = str(item.get("blockid") or "")
            if block_id:
                block_pages[block_id] = item
        return document_id, block_pages

    @staticmethod
    def _page_from_block(item: dict[str, Any] | None) -> int | None:
        if not item:
            return None
        for position in item.get("positions") or []:
            if not isinstance(position, dict):
                continue
            if str(position.get("type") or "") == "bbox":
                anchor_value = position.get("anchor")
                try:
                    if anchor_value is not None:
                        return int(anchor_value)
                except (TypeError, ValueError):
                    pass
            for key in _PAGE_KEYS:
                value = position.get(key)
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
            anchor = position.get("anchor")
            if isinstance(anchor, dict):
                for key in _PAGE_KEYS:
                    value = anchor.get(key)
                    if value is not None:
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            pass
        return None

    def records(self) -> list[AssetRecord]:
        if self._records_cache is not None:
            return self._records_cache
        root, workspace = self._root_and_workspace()
        if not root.is_dir():
            self._records_cache = []
            return self._records_cache
        records: list[AssetRecord] = []
        for parsed_dir in self._sidecar_directories(root):
            blocks_candidates = sorted(parsed_dir.glob("*.blocks.jsonl"))
            for blocks_path in blocks_candidates:
                document_id, block_map = self._read_meta(blocks_path)
                drawings_path = blocks_path.with_name(
                    blocks_path.name.removesuffix(".blocks.jsonl") + ".drawings.json"
                )
                try:
                    payload = json.loads(drawings_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                drawings = payload.get("drawings") if isinstance(payload, dict) else None
                if not isinstance(drawings, dict):
                    continue
                for drawing_id, raw in drawings.items():
                    if not isinstance(raw, dict):
                        continue
                    path_value = str(raw.get("path") or "")
                    if not path_value or "://" in path_value:
                        continue
                    # The sidecar path is relative to the parsed directory and
                    # is validated again before serving.
                    candidate = _safe_relative_path(parsed_dir, parsed_dir / path_value)
                    if candidate is None or not candidate.is_file():
                        continue
                    if document_id is None:
                        match = _DRAWING_ID_RE.search(str(drawing_id))
                        document_id = f"doc-{match.group(1)}" if match else None
                    if not document_id:
                        continue
                    relative_path = candidate.relative_to(parsed_dir.resolve()).as_posix()
                    asset_id = _stable_asset_id(workspace, document_id, relative_path)
                    mime_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                    records.append(
                        AssetRecord(
                            asset_id=asset_id,
                            document_id=document_id,
                            path=candidate,
                            filename=candidate.name,
                            mime_type=mime_type,
                            page=_item_page(raw) or self._page_from_block(block_map.get(raw.get("blockid"))),
                            caption=(str(raw.get("caption")) if raw.get("caption") else None),
                            drawing_id=str(drawing_id),
                        )
                    )
        # A malformed parser output must not create ambiguous IDs.  Keep the
        # first deterministic occurrence if a stale archived sidecar repeats it.
        unique: dict[str, AssetRecord] = {}
        for record in records:
            unique.setdefault(record.asset_id, record)
        self._records_cache = list(unique.values())
        return self._records_cache

    def get(self, asset_id: str) -> AssetRecord | None:
        if not asset_id or not asset_id.startswith("ast_"):
            return None
        return next((item for item in self.records() if item.asset_id == asset_id), None)

    def by_drawing_id(self, drawing_id: str) -> AssetRecord | None:
        return next((item for item in self.records() if item.drawing_id == drawing_id), None)

    def for_documents(self, document_ids: set[str]) -> list[AssetRecord]:
        if not document_ids:
            return []
        return [item for item in self.records() if item.document_id in document_ids]

    @staticmethod
    def variant_path(record: AssetRecord, variant: str) -> Path:
        if variant == "original":
            return record.path
        # Parsers may provide a thumbnail beside the original.  If not, the
        # original is the lossless and safe fallback; the API still honours the
        # variant contract without requiring Pillow or another image library.
        suffix = record.path.suffix
        candidates = (
            record.path.with_name(record.path.stem + ".thumbnail" + suffix),
            record.path.with_name(record.path.stem + ".thumb" + suffix),
        )
        return next((candidate for candidate in candidates if candidate.is_file()), record.path)


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(max(offset, 0)).encode()).decode().rstrip("=")


def decode_cursor(value: str | None) -> int:
    if not value:
        return 0
    try:
        padded = value + "=" * (-len(value) % 4)
        offset = int(base64.urlsafe_b64decode(padded.encode()).decode())
        return max(offset, 0)
    except (ValueError, UnicodeError, base64.binascii.Error):
        raise ValueError("cursor must be a valid opaque graph cursor")


__all__ = ["AssetRecord", "WorkspaceAssetCatalog", "decode_cursor", "encode_cursor"]
