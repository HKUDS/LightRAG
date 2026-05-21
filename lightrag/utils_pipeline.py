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
import re
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, urlsplit

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    LIGHTRAG_DOC_CONTENT_PREFIX,
    PARSED_DIR_NAME,
)
from lightrag.parser_routing import canonicalize_parser_hinted_basename
from lightrag.utils import (
    compute_mdhash_id,
    logger,
    move_file_to_parsed_dir,
)


PLACEHOLDER_DOCUMENT_SOURCES = {"", "no-file-path", "unknown_source"}
SIDECAR_LOCATION_UNKNOWN = "unknown_source"


def build_chunks_dict_from_chunking_result(
    chunking_result: list[dict[str, Any]],
    *,
    doc_id: str,
    file_path: str,
) -> dict[str, dict[str, Any]]:
    """Assemble the per-doc chunks dict written into chunks_vdb / text_chunks.

    Resolves a stable ``chunk_key`` for each entry — preferring an explicit
    ``chunk_id`` (auto-prefixed with ``doc_id-`` if not already), falling back
    to a positional ``chunk-NNN`` derived from ``chunk_order_index``, and
    finally hashing on collision so two entries inside one document never
    overwrite each other.
    """
    chunks: dict[str, dict[str, Any]] = {}
    for dp in chunking_result:
        chunk_content = dp.get("content", "")
        if not chunk_content:
            continue
        raw_chunk_id = dp.get("chunk_id", "")
        order = dp.get("chunk_order_index")
        if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
            chunk_key = (
                raw_chunk_id
                if raw_chunk_id.startswith(f"{doc_id}-")
                else f"{doc_id}-{raw_chunk_id}"
            )
        elif isinstance(order, int):
            chunk_key = f"{doc_id}-chunk-{order:03d}"
        else:
            chunk_key = compute_mdhash_id(f"{doc_id}:{chunk_content}", prefix="chunk-")

        # Hard collision guard (same chunk_id inside one document).
        if chunk_key in chunks:
            chunk_key = compute_mdhash_id(
                f"{doc_id}:{order}:{chunk_content}",
                prefix="chunk-",
            )
        # Preserve any pre-populated cache ids on dp (multimodal chunks
        # arrive with analysis cache ids already attached so document
        # deletion can find them via the per-chunk llm_cache_list).
        existing_cache_list = dp.get("llm_cache_list")
        seed_cache_list: list[str] = []
        if isinstance(existing_cache_list, list):
            seen: set[str] = set()
            for entry in existing_cache_list:
                key = str(entry or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    seed_cache_list.append(key)
        chunks[chunk_key] = {
            **dp,
            "full_doc_id": doc_id,
            "file_path": file_path,
            "llm_cache_list": seed_cache_list,
        }
    return chunks


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

    Returns the first non-placeholder ``file_path`` from doc_status, then
    full_docs. Both are already canonicalized at write time, so this only
    has to skip placeholder sentinels.
    """
    for source in (
        getattr(status_doc, "file_path", None),
        content_data.get("file_path") if content_data else None,
    ):
        if not isinstance(source, str):
            continue
        candidate = source.strip()
        if candidate and candidate not in PLACEHOLDER_DOCUMENT_SOURCES:
            return candidate
    return "unknown_source"


def normalize_document_file_path(file_path: Any) -> str:
    """Return the canonical basename stored as ``file_path``.

    Strips any supported ``[hint]`` segment so ``abc.docx`` and
    ``abc.[native-iet].docx`` map to the same key. Collapses placeholders to
    ``"unknown_source"``. Idempotent.
    """
    source = str(file_path or "").strip()
    if source in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    canonical = canonicalize_parser_hinted_basename(source).strip()
    if canonical in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    return canonical or "unknown_source"


# Back-compat alias retained until call sites that import the old name are
# all switched over (the public surface is ``normalize_document_file_path``).
document_canonical_key = normalize_document_file_path


def has_known_document_source(source_key: str) -> bool:
    return source_key not in PLACEHOLDER_DOCUMENT_SOURCES


def doc_status_field(doc: Any, field: str, default: Any = "") -> Any:
    if isinstance(doc, dict):
        return doc.get(field, default)
    return getattr(doc, field, default)


# Long-lived per-document metadata fields that must survive every
# doc_status state transition.  ``process_options`` records the user's
# per-file processing strategy at enqueue time and is read by analyze /
# chunk / KG-skip stages and by admin/list APIs throughout the document's
# lifetime, so we cannot let an intermediate transition (PARSING /
# ANALYZING / PROCESSING / PROCESSED / FAILED upsert) clobber it.
# ``parse_warnings`` records non-fatal parser warnings (e.g. legacy docx
# tables missing ``w14:paraId``) that admins should be able to surface
# alongside the document record after PROCESSED.
# ``chunk_opts`` is written when entering PROCESSING (via ``extraction_meta``)
# and records the actual chunker params used for that document in the same
# format as the ``Chunking <strategy>: ...`` log line (params portion only).
# Carrying it forward keeps the value visible after PROCESSING -> FAILED,
# whose ``metadata_extra`` only carries timing fields.
# ``parsing_start_time`` / ``analyzing_start_time`` are Unix epoch seconds
# stamped at the entry of ``_parse_worker`` / ``_analyze_worker`` (mirrors
# the existing ``processing_start_time`` set when entering PROCESSING) so
# per-stage durations can be derived from doc_status post-mortem.
# ``parse_stage_skipped`` is written by ``parse_mineru`` / ``parse_docling``
# when the raw bundle cache is valid and the parse stage round trip is
# skipped; absence == not skipped (e.g. native parser, or cache miss).
# ``source_file_name`` records the original pending-parse source basename used
# by parser workers; it is intentionally separate from canonical ``file_path``.
#
# The order of this tuple is the rendering order of metadata fields in
# the WebUI ``DocumentStatusDetailsDialog`` (carry-over builds the new
# metadata dict by iterating this tuple, and dict / JSON / JSX preserve
# insertion order all the way to the rendered output). Keep fields
# grouped by stage: parse-stage fields together, analyze-stage fields
# together, etc., so the dialog reads top-to-bottom along the pipeline.
_DOC_STATUS_METADATA_CARRY_OVER_KEYS: tuple[str, ...] = (
    "process_options",
    "source_file_name",
    "parse_warnings",
    "chunk_opts",
    "parsing_start_time",
    "parse_stage_skipped",
    "analyzing_start_time",
)


def doc_status_metadata_carry_over(status_doc: Any) -> dict[str, Any]:
    """Return the subset of ``status_doc.metadata`` to preserve across upserts.

    ``doc_status`` storage backends generally treat the ``metadata`` field
    as an opaque blob and **replace** it on every upsert, so callers must
    explicitly carry forward fields they want to keep.  This helper centralises
    the list of fields we always carry: today only ``process_options``, but
    new long-lived metadata can be added by extending
    ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``.
    """
    if status_doc is None:
        return {}
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return {}
    carry: dict[str, Any] = {}
    for key in _DOC_STATUS_METADATA_CARRY_OVER_KEYS:
        if key in raw_metadata and raw_metadata[key] not in (None, ""):
            carry[key] = raw_metadata[key]
    return carry


def doc_status_transition_metadata(
    status_doc: Any,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a doc_status ``metadata`` payload that preserves carry-over fields.

    Use at every state-transition upsert site so the user's
    ``process_options`` (and any future long-lived metadata fields) survive
    PENDING → PARSING → ANALYZING → PROCESSING → PROCESSED / FAILED.
    """
    payload = doc_status_metadata_carry_over(status_doc)
    if extra:
        payload.update(extra)
    return payload


def doc_status_value(doc: Any) -> str:
    status = doc_status_field(doc, "status", "")
    if isinstance(status, DocStatus):
        return status.value
    return str(status or "")


# Sidecar item ids embed ``doc_hash`` (= doc_id without the ``doc-`` prefix),
# and for pending_parse uploads doc_id derives from the filename — so the
# same content under two filenames renders with different ids in
# ``merged_text``. Strip those surfaces before hashing so cross-filename
# content_hash dedup actually fires.
_SIDECAR_ID_PATTERN = re.compile(r"\b(tb|im|eq)-[0-9a-f]{32}-(\d{4})\b")
_ASSET_PATH_PATTERN = re.compile(r'(?<=path=")[^"]*\.blocks\.assets/')


def normalize_merged_text_for_hash(content: str) -> str:
    """Strip filename-derived prefixes from sidecar ids and asset paths.

    Idempotent and safe on plain text (matches the doc_hash literal only —
    32 lowercase hex digits between the modality prefix and a 4-digit
    sequence). RAW text bodies without sidecar markup pass through
    unchanged.
    """
    if not content:
        return content
    content = _SIDECAR_ID_PATTERN.sub(r"\1-<DOC>-\2", content)
    content = _ASSET_PATH_PATTERN.sub("<ASSETS>/", content)
    return content


def compute_text_content_hash(content: str) -> str:
    """MD5 hex digest of text content used for cross-filename dedup.

    Input is normalized via :func:`normalize_merged_text_for_hash` first so
    sidecar-rendered bodies dedupe across filenames despite carrying
    filename-derived item ids and asset paths.
    """
    return compute_mdhash_id(normalize_merged_text_for_hash(content), prefix="")


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
    """Find an existing doc_status record by canonical file basename.

    Inputs are normalized via :func:`normalize_document_file_path` so callers
    may pass either the bare canonical name (``abc.docx``) or a hint-bearing
    variant (``abc.[native-iet].docx``); both resolve to the same logical
    document.
    """
    basename = normalize_document_file_path(file_path)
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


def make_lightrag_doc_content(merged_text: str) -> str:
    """Build the ``full_docs.content`` value for ``format=lightrag`` records.

    The result has shape ``"{{LRdoc}}<merged_text>"`` — the marker prefix
    distinguishes lightrag-format full_docs from raw-format ones, and the
    body is the complete merged text from the ``.blocks.jsonl`` content
    lines so F-chunking can run identically on raw and lightrag inputs
    (the prefix is stripped at chunking time via
    ``strip_lightrag_doc_prefix``).
    """
    return f"{LIGHTRAG_DOC_CONTENT_PREFIX}{merged_text or ''}"


def strip_lightrag_doc_prefix(content: str | None, parse_format: str | None) -> str:
    """Return the bare body for a stored ``full_docs.content`` value.

    The ``{{LRdoc}}`` marker is stripped **only** when ``parse_format``
    indicates the record is in lightrag format.  Any other ``parse_format``
    (``raw``, ``pending_parse``, ``None`` ...) returns the content
    unchanged so a raw document whose literal body happens to start with
    ``{{LRdoc}}`` is never silently truncated.

    Centralizing the format check here turns "must check format before
    stripping" from a caller-side discipline into a structural property of
    the function: any future call site that forgets to gate is protected
    automatically.
    """
    if (
        parse_format == FULL_DOCS_FORMAT_LIGHTRAG
        and isinstance(content, str)
        and content.startswith(LIGHTRAG_DOC_CONTENT_PREFIX)
    ):
        return content[len(LIGHTRAG_DOC_CONTENT_PREFIX) :]
    return content or ""


# ---------------------------------------------------------------------------
# Document path / artifact helpers (moved from _PipelineMixin)
# ---------------------------------------------------------------------------


def input_dir_path() -> Path:
    return configured_input_dir()


def parsed_dir() -> Path:
    """Return the project-wide parsed-artifact root: ``<input_dir>/__parsed__``."""
    return input_dir_path() / PARSED_DIR_NAME


def parsed_artifact_dir_for(
    file_path: str, *, parent_hint: Path | str | None = None
) -> Path:
    """Return the per-document sidecar directory for ``file_path``.

    ``file_path`` must already be canonical (run ``normalize_document_file_path``
    first if unsure). When ``parent_hint`` is supplied (e.g. the live source
    file's parent), the sidecar is placed next to it under ``__parsed__/``
    rather than under the global ``input_dir``; this keeps test isolation
    intact when the source lives outside ``INPUT_DIR``. On collision with an
    existing non-directory entry, the helper appends ``_001``..``_999`` and
    finally a unix timestamp suffix.
    """
    if parent_hint is not None:
        hint = Path(parent_hint)
        # ``hint`` may already point at a ``__parsed__/`` dir (e.g. when the
        # caller re-archived a source); reuse it in place rather than nesting.
        root = hint if hint.name == PARSED_DIR_NAME else hint / PARSED_DIR_NAME
    else:
        root = parsed_dir()
    source_name = (
        canonicalize_parser_hinted_basename(file_path or "document") or "document"
    )
    artifact_name = f"{source_name}.parsed"
    artifact_dir = root / artifact_name
    if not artifact_dir.exists() or artifact_dir.is_dir():
        return artifact_dir

    for i in range(1, 1000):
        candidate = root / f"{artifact_name}_{i:03d}"
        if not candidate.exists() or candidate.is_dir():
            return candidate

    return root / f"{artifact_name}_{int(time.time())}"


# ---------------------------------------------------------------------------
# Sidecar URI helpers (``full_docs.sidecar_location``)
# ---------------------------------------------------------------------------
#
# Sidecar URI scheme conventions:
#   - Local:  ``file:///abs/path/to/abc.parsed/``   (trailing slash required)
#   - Remote: ``s3://bucket/workspace/abc.parsed/`` (future; resolver returns
#             None today so local readers gracefully skip)
#   - Unknown sentinel: literal string ``"unknown_source"``


def sidecar_uri_for(parsed_artifact_dir: Path | str) -> str:
    """Build the canonical sidecar URI for a local artifact directory.

    The result always ends with ``/`` so a reader can distinguish a directory
    from a file at the URI level. Non-ASCII characters are percent-encoded.
    """
    p = Path(parsed_artifact_dir).resolve()
    encoded = quote(str(p), safe="/")
    return f"file://{encoded}/"


def resolve_sidecar_uri(uri: str | None) -> Path | None:
    """Decode a sidecar URI into a local filesystem Path.

    Returns None for the unknown sentinel, empty input, or any non-``file://``
    scheme (remote schemes will get their own resolvers).
    """
    if not uri or uri == SIDECAR_LOCATION_UNKNOWN:
        return None
    parts = urlsplit(uri)
    if parts.scheme != "file":
        return None
    path_str = unquote(parts.path)
    if path_str.endswith("/") and len(path_str) > 1:
        path_str = path_str[:-1]
    return Path(path_str)


def sidecar_blocks_path(uri: str | None) -> str | None:
    """Locate the first ``*.blocks.jsonl`` file inside a sidecar URI.

    Returns the absolute path as a string, or None when the URI cannot be
    resolved locally or the directory holds no blocks file.
    """
    d = resolve_sidecar_uri(uri)
    if d is None or not d.is_dir():
        return None
    candidates = sorted(d.glob("*.blocks.jsonl"))
    return str(candidates[0]) if candidates else None


def sidecar_modality_path(uri: str | None, modality: str) -> str | None:
    """Return the path for a sidecar modality JSON (drawings/tables/equations).

    Does not require the file to exist — callers check. Returns None when the
    sidecar URI cannot be resolved or has no blocks file to anchor the name.
    """
    blocks = sidecar_blocks_path(uri)
    if not blocks:
        return None
    return f"{blocks[: -len('.blocks.jsonl')]}.{modality}.json"


def sidecar_assets_dir_for_uri(uri: str | None) -> Path | None:
    """Return the ``*.blocks.assets/`` directory Path for a sidecar URI.

    The directory may not exist; callers create it on first asset write.
    """
    blocks = sidecar_blocks_path(uri)
    if not blocks:
        return None
    return Path(f"{blocks[: -len('.blocks.jsonl')]}.blocks.assets")


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


async def load_lightrag_document_content(sidecar_uri: str) -> tuple[str, str]:
    """Load LightRAG Document blocks and return ``(merged_text, blocks_path)``.

    ``sidecar_uri`` is a sidecar location URI (see ``sidecar_uri_for``); this
    locates the ``*.blocks.jsonl`` file inside it, reads the content lines
    (skipping the meta header at index 0 and any non-content entries), and
    returns the merged body plus the absolute blocks path.
    """
    resolved = sidecar_blocks_path(sidecar_uri)
    if resolved is None:
        raise FileNotFoundError(
            f"LightRAG blocks file not found from sidecar uri: {sidecar_uri}"
        )
    blocks_path = Path(resolved)

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


# Multimodal entity injection used to live here as a centralized post-pass
# over all chunk_results. It has been moved into
# :func:`lightrag.operate.extract_entities._process_single_content` so each
# multimodal chunk injects its own entity/relation records while still under
# its concurrency slot.  The chunk's ``sidecar.type`` (drawing/table/equation)
# is the dispatch key; see operate.py for the new logic.
