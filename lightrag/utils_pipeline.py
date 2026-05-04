"""Pipeline-specific helpers for document status, identity, and content.

These helpers are shared by the LightRAG pipeline mixin (lightrag/pipeline.py)
and by other LightRAG methods that touch the document ingestion paths
(custom-chunks ingest, deletion, etc.). They are kept out of utils.py because
they are tied to the doc_status / full_docs domain rather than to general
text/token utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.constants import LIGHTRAG_DOC_CONTENT_PREFIX, PARSED_DIR_NAME
from lightrag.parser_routing import canonicalize_parser_hinted_basename
from lightrag.utils import (
    compute_mdhash_id,
    get_content_summary,
    logger,
    move_file_to_parsed_dir,
)


PLACEHOLDER_DOCUMENT_SOURCES = {"", "no-file-path", "unknown_source"}


def chunk_fields_from_status_doc(
    status_doc: DocProcessingStatus,
) -> tuple[list[str], int]:
    """Return (chunks_list, chunks_count) preserved from a status document.

    Filters out any non-string or empty chunk IDs.  When chunks_count is
    absent or invalid, it is inferred from the length of chunks_list.
    """
    chunks_list: list[str] = []
    if isinstance(status_doc.chunks_list, list):
        chunks_list = [
            chunk_id
            for chunk_id in status_doc.chunks_list
            if isinstance(chunk_id, str) and chunk_id
        ]

    if isinstance(status_doc.chunks_count, int) and status_doc.chunks_count >= 0:
        return chunks_list, status_doc.chunks_count

    return chunks_list, len(chunks_list)


def resolve_doc_file_path(
    status_doc: DocProcessingStatus | None = None,
    content_data: dict[str, Any] | None = None,
) -> str:
    """Resolve the best available document file path.

    Prefer a non-placeholder path from doc_status, then fall back to full_docs.
    This avoids overwriting historical file paths with placeholder values during
    retries or early-cancellation paths.
    """

    placeholder_paths = {"", "no-file-path", "unknown_source"}

    def _normalize_path(candidate: Any) -> str | None:
        if not isinstance(candidate, str):
            return None

        normalized = candidate.strip()
        if not normalized:
            return None

        return normalized

    candidates = [
        _normalize_path(getattr(status_doc, "file_path", None)),
        _normalize_path(content_data.get("file_path") if content_data else None),
    ]

    for candidate in candidates:
        if candidate and candidate not in placeholder_paths:
            return candidate

    for candidate in candidates:
        if candidate:
            return "unknown_source" if candidate == "no-file-path" else candidate

    return "unknown_source"


def document_source_key(file_path: Any) -> str:
    """Return the filename-level key used for document uniqueness."""
    source = str(file_path or "").strip()
    if source in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    filename = canonicalize_parser_hinted_basename(source).strip()
    if filename in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    return filename or "unknown_source"


def has_known_document_source(source_key: str) -> bool:
    return source_key not in PLACEHOLDER_DOCUMENT_SOURCES


def doc_status_field(doc: Any, field: str, default: Any = "") -> Any:
    if isinstance(doc, dict):
        return doc.get(field, default)
    return getattr(doc, field, default)


def doc_status_value(doc: Any) -> str:
    status = doc_status_field(doc, "status", "")
    if isinstance(status, DocStatus):
        return status.value
    return str(status or "")


def compute_text_content_hash(content: str) -> str:
    """MD5 hex digest of text content used for cross-filename dedup."""
    return compute_mdhash_id(content, prefix="")


def compute_file_content_hash(path_str: str) -> str | None:
    """Stream-compute MD5 of a file's bytes; returns None if unreadable.

    Resolves the LightRAG ``*.blocks.jsonl`` conventions used by
    ``_load_lightrag_document_content`` so the hash matches the actual
    document body regardless of whether ``path_str`` points at the blocks
    file directly or its parent directory/base name.
    """
    if not path_str:
        return None
    try:
        path = Path(path_str)
        if path.is_dir():
            candidates = sorted(path.glob("*.blocks.jsonl"))
            if not candidates:
                return None
            path = candidates[0]
        elif not (path.exists() and path.is_file()):
            blocks_path = Path(path_str + ".blocks.jsonl")
            if blocks_path.exists() and blocks_path.is_file():
                path = blocks_path
            else:
                return None
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute file content hash for {path_str}: {e}")
        return None


def configured_input_dir() -> Path:
    input_dir = os.getenv("INPUT_DIR", "").strip()
    return Path(input_dir) if input_dir else Path.cwd() / "inputs"


async def get_existing_doc_by_file_basename(
    doc_status: DocStatusStorage, file_path: Any
) -> tuple[str, Any] | None:
    """Find an existing doc_status record by file basename.

    Both write and lookup paths feed file_path through ``document_source_key``
    first, so stored basenames are already canonical (parser hints stripped).
    Storage backends therefore compare canonical-vs-canonical and do not need
    to re-run any normalization themselves.
    """
    basename = document_source_key(file_path)
    if basename == "unknown_source":
        return None
    return await doc_status.get_doc_by_file_basename(basename)


async def get_existing_doc_by_content_hash(
    doc_status: DocStatusStorage, content_hash: str
) -> tuple[str, Any] | None:
    """Find an existing doc_status record by content hash."""
    if not content_hash:
        return None
    return await doc_status.get_doc_by_content_hash(content_hash)


async def get_duplicate_doc_by_content_hash(
    doc_status: DocStatusStorage, content_hash: str, current_doc_id: str
) -> tuple[str, Any] | None:
    """Find another doc_status record with the same content hash."""
    if not content_hash:
        return None

    match = await doc_status.get_doc_by_content_hash(content_hash)
    if match and match[0] != current_doc_id:
        return match

    try:
        docs = await doc_status.get_docs_by_statuses(list(DocStatus))
    except Exception:
        return None
    for doc_id, doc in docs.items():
        if doc_id == current_doc_id:
            continue
        if doc_status_field(doc, "content_hash", "") == content_hash:
            return doc_id, doc
    return None


def make_lightrag_doc_content(merged_text: str, max_length: int = 250) -> str:
    """Build the ``full_docs.content`` value for ``format=lightrag`` records.

    The result has shape ``"{{LRdoc}}<summary>"`` where ``<summary>`` is the
    same leading-text snippet that paginated APIs return in
    ``content_summary`` (see ``get_content_summary``). This keeps the
    behaviour mandated by ``docs/FileProcessingConfiguration-zh.md``.
    """
    summary = get_content_summary(merged_text or "", max_length=max_length)
    return f"{LIGHTRAG_DOC_CONTENT_PREFIX}{summary}"


# ---------------------------------------------------------------------------
# Document path / artifact helpers (moved from _PipelineMixin)
# ---------------------------------------------------------------------------


def input_dir_path() -> Path:
    return configured_input_dir()


def parsed_dir_for_source(source_path: str | None = None) -> Path:
    if not source_path:
        return input_dir_path() / PARSED_DIR_NAME

    source = Path(source_path)
    if source.is_absolute():
        if source.parent.name == PARSED_DIR_NAME:
            return source.parent
        return source.parent / PARSED_DIR_NAME

    source_parent = source.parent
    if str(source_parent) == ".":
        return input_dir_path() / PARSED_DIR_NAME
    if source_parent.name == PARSED_DIR_NAME:
        return input_dir_path() / source_parent
    return input_dir_path() / source_parent / PARSED_DIR_NAME


def parsed_artifact_dir_for_source(
    source_path: str | None = None, file_path: str | None = None
) -> Path:
    parsed_dir = parsed_dir_for_source(source_path)
    source_name = (
        canonicalize_parser_hinted_basename(source_path or file_path or "document")
        or "document"
    )
    artifact_name = f"{source_name}.parsed"
    artifact_dir = parsed_dir / artifact_name
    if not artifact_dir.exists() or artifact_dir.is_dir():
        return artifact_dir

    for i in range(1, 1000):
        candidate = parsed_dir / f"{artifact_name}_{i:03d}"
        if not candidate.exists() or candidate.is_dir():
            return candidate

    return parsed_dir / f"{artifact_name}_{int(time.time())}"


def resolve_lightrag_document_path(document_path: str) -> str:
    path = Path(document_path)
    if path.is_absolute():
        return str(path)
    return str(input_dir_path() / path)


# ---------------------------------------------------------------------------
# Source archive helpers
# ---------------------------------------------------------------------------


async def archive_docx_source_after_full_docs_sync(source_path: str) -> str | None:
    source = Path(source_path)
    try:
        target = await move_file_to_parsed_dir(source, skip_if_already_parsed=True)
    except Exception as e:
        logger.warning(
            f"[parse] Source archive skipped after full_docs sync: {source_path}: {e}"
        )
        return None
    if target is None:
        return None
    if target != source:
        logger.debug(
            f"[parse] Archived DOCX source after full_docs sync: {source} -> {target}"
        )
    return str(target)


async def archive_source_after_full_docs_sync(source_path: str) -> str | None:
    return await archive_docx_source_after_full_docs_sync(source_path)


# ---------------------------------------------------------------------------
# LightRAG Document blocks loader
# ---------------------------------------------------------------------------


async def load_lightrag_document_content(
    lightrag_document_path: str,
) -> tuple[str, str]:
    """Load LightRAG Document blocks and return (merged_text, blocks_path)."""
    path = Path(lightrag_document_path)
    candidates: list[Path] = []
    if path.suffix == ".jsonl" and path.name.endswith(".blocks.jsonl"):
        candidates.append(path)
    else:
        candidates.append(Path(str(path) + ".blocks.jsonl"))
        if path.is_dir():
            candidates.extend(path.glob("*.blocks.jsonl"))
        else:
            candidates.append(path)

    blocks_path = None
    for c in candidates:
        if c.exists() and c.is_file():
            blocks_path = c
            break
    if blocks_path is None:
        raise FileNotFoundError(
            f"LightRAG blocks file not found from path: {lightrag_document_path}"
        )

    merged_parts: list[str] = []
    with blocks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if i == 0:
                continue
            if obj.get("type") != "content":
                continue
            content = obj.get("content", "")
            if isinstance(content, str) and content.strip():
                merged_parts.append(content)

    return "\n\n".join(merged_parts), str(blocks_path)


# ---------------------------------------------------------------------------
# Payload introspection helpers (parser response normalization)
# ---------------------------------------------------------------------------


def get_by_path(payload: Any, path: str) -> Any:
    if not path:
        return None
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def extract_content_list_from_payload(
    payload: Any,
) -> list[dict[str, Any]] | None:
    """Try to find a MinerU/Docling-like content list from arbitrary JSON payload."""
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            first = payload[0]
            if "type" in first or "label" in first or "text" in first:
                return cast(list[dict[str, Any]], payload)
        return None
    if not isinstance(payload, dict):
        return None

    # Common direct keys first
    for key in ("content_list", "content", "items", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            extracted = extract_content_list_from_payload(value)
            if extracted is not None:
                return extracted
        elif isinstance(value, dict):
            extracted = extract_content_list_from_payload(value)
            if extracted is not None:
                return extracted

    # Deep search as fallback
    for value in payload.values():
        extracted = extract_content_list_from_payload(value)
        if extracted is not None:
            return extracted
    return None


def normalize_parser_result_to_content_list(
    parser_result: str | list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalize parser result to structured content list if possible."""
    if parser_result is None:
        return None
    if isinstance(parser_result, list):
        return extract_content_list_from_payload(parser_result)
    if isinstance(parser_result, dict):
        return extract_content_list_from_payload(parser_result)
    text = str(parser_result).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return extract_content_list_from_payload(payload)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multimodal entity augmentation
# ---------------------------------------------------------------------------


def augment_chunk_results_with_mm_entities(
    chunk_results: list,
    mm_specs: list[dict[str, Any]],
    file_path: str,
) -> list:
    """Inject modality object entities and relations into merge inputs."""
    if not mm_specs:
        return chunk_results

    extracted_by_chunk: dict[str, set[str]] = {}
    for maybe_nodes, _ in chunk_results:
        if not isinstance(maybe_nodes, dict):
            continue
        for entity_name, entity_records in maybe_nodes.items():
            if not isinstance(entity_records, list):
                continue
            for rec in entity_records:
                if not isinstance(rec, dict):
                    continue
                source_id = str(rec.get("source_id") or "")
                if not source_id:
                    continue
                extracted_by_chunk.setdefault(source_id, set()).add(str(entity_name))

    now_ts = int(time.time())
    mm_nodes: dict[str, list[dict[str, Any]]] = {}
    mm_edges: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for spec in mm_specs:
        src = str(spec["entity_name"])
        chunk_id = str(spec["chunk_id"])
        kind = str(spec["kind"])
        title = str(spec.get("name") or src)
        caption_text = str(spec.get("caption_text") or "").strip()
        heading = str(spec.get("heading") or "").strip()
        summary = str(spec.get("summary") or "").strip()

        mm_nodes.setdefault(src, []).append(
            {
                "entity_name": src,
                "entity_type": kind,
                "description": summary or f"{kind} object: {title}",
                "source_id": chunk_id,
                "file_path": file_path,
                "timestamp": now_ts,
            }
        )

        targets = extracted_by_chunk.get(chunk_id, set())
        for tgt in sorted(targets):
            if tgt == src:
                continue
            desc = (
                f"Entity `{tgt}` is associated with {kind} `{title}` "
                f"in section `{heading or 'unknown'}`."
            )
            if caption_text:
                desc += f" Captions: {caption_text}."
            edge_key = tuple(sorted((src, tgt)))
            mm_edges.setdefault(edge_key, []).append(
                {
                    "src_id": src,
                    "tgt_id": tgt,
                    "weight": 1.0,
                    "description": desc,
                    "keywords": "belongs to,part of,contained in",
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "timestamp": now_ts,
                }
            )

    if mm_nodes or mm_edges:
        chunk_results = list(chunk_results) + [(mm_nodes, mm_edges)]
    return chunk_results
