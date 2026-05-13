"""Document ingestion pipeline mixin for the LightRAG class.

This module isolates the document parse/enqueue/extraction pipeline so that
``lightrag.py`` stays focused on storage management, querying, and editing.
The mixin is wired into :class:`lightrag.LightRAG` via multiple inheritance
and relies on attributes/methods that the main class provides
(``self.full_docs``, ``self.doc_status``, ``self.tokenizer``,
``self.parse_native``-related fields, ``self._insert_done``,
``self._process_extract_entities``, etc.).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json

import json_repair
import mimetypes
import os
from copy import deepcopy
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSED_DIR_NAME,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)
from lightrag.exceptions import MultimodalAnalysisError, PipelineCancelledException
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.operate import merge_nodes_and_edges
from lightrag.parser_routing import (
    canonicalize_parser_hinted_basename,
    resolve_file_parser_directives,
    resolve_stored_document_parser_engine,
)
from lightrag.utils import (
    CacheData,
    _serialize_cache_variant,
    compute_args_hash,
    compute_mdhash_id,
    enforce_chunk_token_limit_before_embedding,
    generate_cache_key,
    generate_track_id,
    get_content_summary,
    get_llm_cache_identity,
    handle_cache,
    logger,
    sanitize_text_for_encoding,
    save_to_cache,
    serialize_llm_cache_identity,
)
from lightrag.utils_pipeline import (
    archive_docx_source_after_full_docs_sync,
    archive_source_after_full_docs_sync,
    build_chunks_dict_from_chunking_result,
    chunk_fields_from_status_doc,
    compute_file_content_hash,
    compute_text_content_hash,
    doc_status_field,
    doc_status_transition_metadata,
    document_canonical_key,
    document_source_key,
    get_by_path,
    get_duplicate_doc_by_content_hash,
    get_existing_doc_by_content_hash,
    get_existing_doc_by_file_basename,
    has_known_document_source,
    input_dir_path,
    load_lightrag_document_content,
    make_lightrag_doc_content,
    normalize_parser_result_to_content_list,
    parsed_artifact_dir_for_source,
    resolve_doc_file_path,
    resolve_lightrag_blocks_path,
    resolve_lightrag_document_path,
    strip_lightrag_doc_prefix,
)


# Document statuses the pipeline considers "in-flight or pending" — used by
# both the initial snapshot and every refetch after a request_pending
# continuation.  Module-level so we don't reconstruct the list on every
# pipeline entry.
_INFLIGHT_DOC_STATUSES = (
    DocStatus.PROCESSING,
    DocStatus.FAILED,
    DocStatus.PENDING,
    DocStatus.PARSING,
    DocStatus.ANALYZING,
)


# Map ``process_options.chunking`` selector → ``extraction_meta.chunking_method``
# string used by the pipeline observability layer and the resume path.
_CHUNKING_METHOD_LABELS: dict[str, str] = {
    "F": "fixed_token",
    "R": "recursive_character",
    "V": "semantic_vector",
    "P": "paragraph_semantic",
}


_CHUNK_LOG_KEY_ALIASES: dict[str, str] = {
    "chunk_overlap_token_size": "overlap",
    "breakpoint_threshold_type": "break",
    "breakpoint_threshold_amount": "amount",
    "buffer_size": "buf",
    "split_by_character": "split_by",
    "split_by_character_only": "split_only",
    "separators": "seps",
    "sentence_split_regex": "regex",
}


def _format_chunking_params(
    chunk_size: int,
    params: dict[str, Any],
) -> str:
    """Format the ``size=..., key=value, ...`` portion shared by the chunking
    start log line and ``doc_status.metadata['chunk_opts']``.

    Drops keys with ``None``/empty values so the line stays scannable;
    callers pass the strategy-specific kwargs they're about to splat
    into the chunker so the output mirrors the actual call.  Long keys are
    aliased to short forms via ``_CHUNK_LOG_KEY_ALIASES``.
    """
    pieces = [f"size={chunk_size}"]
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            continue
        short = _CHUNK_LOG_KEY_ALIASES.get(key, key)
        pieces.append(
            f"{short}={value!r}" if isinstance(value, str) else f"{short}={value}"
        )
    return ", ".join(pieces)


@dataclass
class _BatchRunContext:
    """Per-batch shared state for the parse/analyze/process worker pipeline.

    Bundles the cross-cutting handles (pipeline_status, locks, queues,
    semaphore) so worker methods accept a single ``ctx`` argument instead of
    ~8 individually plumbed parameters.  ``processed_count`` mutates inside
    each batch and is always read/written under ``pipeline_status_lock``.
    """

    pipeline_status: dict
    pipeline_status_lock: Any
    semaphore: asyncio.Semaphore
    total_files: int
    q_native: asyncio.Queue
    q_mineru: asyncio.Queue
    q_docling: asyncio.Queue
    q_analyze: asyncio.Queue
    q_process: asyncio.Queue
    processed_count: int = 0


class _PipelineMixin:
    """Mixin providing document ingestion pipeline methods for LightRAG.

    Designed to be combined as a base of LightRAG only.  Relies on
    LightRAG-provided attributes (``self.full_docs``, ``self.doc_status``,
    ``self.tokenizer``, ``self.parser_*``, ``self.workspace`` ...) and on the
    shared methods ``self._insert_done`` / ``self._process_extract_entities``
    which remain in the main class and are resolved through MRO.
    """

    # ============================================================
    # Public document ingestion API (entry points)
    # ============================================================

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
        docs_format: str = FULL_DOCS_FORMAT_RAW,
        lightrag_document_paths: str | list[str] | None = None,
        parse_engine: str | list[str] | None = None,
        process_options: str | list[str] | None = None,
        chunk_options: dict | list[dict] | None = None,
        from_scan: bool = False,
    ) -> str:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents (skip content dedup when format is lightrag)
        2. Generate document initial status
        3. Filter out already processed documents
        4. Enqueue document in status

        Args:
            input: Single document string or list of document strings (can be empty when docs_format is lightrag)
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated (from content or file_path when lightrag)
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status
            docs_format: "raw" (default) or "lightrag"; when "lightrag" content may be empty and content-dedup is skipped
            lightrag_document_paths: paths to LightRAG Document (e.g. .blocks.jsonl dir or base path), when docs_format is lightrag
            parse_engine: file extraction engine already used or target engine for pending_parse
            process_options: per-document processing options string (i/t/e/!/F/R/V/P);
                accepted as a single string broadcast to every input or as a list
                aligned with ``input``. Stored verbatim on ``full_docs`` and
                mirrored to ``doc_status.metadata['process_options']``.
            chunk_options: per-document chunker parameter snapshot.
                Accepted as ``dict`` (broadcast to every input) or
                ``list[dict]`` (aligned with ``input``).  When ``None``,
                each doc's snapshot is built via
                :func:`lightrag.parser_routing.resolve_chunk_options`
                from ``self.addon_params['chunker']``.  Persisted to
                ``full_docs[doc_id]['chunk_options']`` and consumed by
                :meth:`process_single_document` to drive the file
                chunkers (F / R / V / P).  Callers that need to bake
                F-strategy runtime args (``split_by_character`` /
                ``split_by_character_only``) into the snapshot — e.g.
                :meth:`LightRAG.ainsert` — should call
                :func:`resolve_chunk_options` themselves and pass the
                result here; this function is intentionally chunker-
                config agnostic.  See
                ``docs/FileProcessingConfiguration-zh.md`` for the schema.
            from_scan: when True, the caller is the scan-owned background task
                that already holds ``pipeline_status["scanning"]``.  Scan
                does additional doc_status reads during its classification
                phase (PROCESSED detection, FAILED-stub deletion, etc.)
                so external writers are blocked via
                ``scanning_exclusive``.  Scan's own enqueues happen in
                its processing phase, after classification has cleared
                ``scanning_exclusive``, but ``from_scan=True`` is still
                forwarded as a defence-in-depth bypass so an unexpected
                scan-owned write inside the classification window is
                allowed through.  External callers must leave this False.

        Returns:
            str: tracking ID for monitoring processing status

        Raises:
            RuntimeError: if a scan is in progress (and ``from_scan`` is
                False), or if a destructive job (clear / delete) is in
                flight.  Concurrent indexing (``busy=True`` from the
                processing loop) is permitted — the running loop is
                notified via ``request_pending`` and picks up the
                newly-enqueued doc after its current batch finishes.
        """
        # Concurrency contract: enqueue may proceed concurrently with the
        # processing loop because (a) full_docs is upserted before
        # doc_status, so a consistency check never sees a ghost row, and
        # (b) the running loop re-queries doc_status by status after each
        # batch and sets ``request_pending`` whenever new work arrives
        # while busy.  Two states still block enqueue:
        #   * ``scanning_exclusive`` — scan task is in its CLASSIFICATION
        #     phase, reading doc_status to classify files and possibly
        #     deleting stale stubs.  Concurrent enqueue would race
        #     against scan's reads / mutations.  ``from_scan=True``
        #     lifts this guard for the scan task's own enqueues.
        #     ``scanning`` alone (the processing phase) does NOT block,
        #     identical to the upload-during-busy case.
        #   * ``destructive_busy`` — clear / delete is dropping storages
        #     or removing input files; a concurrent write would be
        #     silently clobbered.
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )
        async with pipeline_status_lock:
            if not from_scan and pipeline_status.get("scanning_exclusive"):
                raise RuntimeError(
                    "Cannot enqueue while scan is classifying files; "
                    "wait for the classification phase to finish "
                    "before retrying."
                )
            if pipeline_status.get("destructive_busy"):
                raise RuntimeError(
                    "Cannot enqueue while pipeline is clearing or "
                    "deleting documents; wait for the running job to "
                    "finish before retrying."
                )

        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("enqueue")
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        if isinstance(lightrag_document_paths, str):
            lightrag_document_paths = (
                [lightrag_document_paths] if lightrag_document_paths else None
            )
        if isinstance(parse_engine, str):
            parse_engine = [parse_engine] * len(input)
        if isinstance(process_options, str):
            process_options = [process_options] * len(input)
        if isinstance(chunk_options, dict):
            chunk_options = [chunk_options] * len(input)

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
            file_paths = [
                path.strip() if isinstance(path, str) else "" for path in file_paths
            ]
            file_paths = [path if path else "unknown_source" for path in file_paths]
        else:
            file_paths = ["unknown_source"] * len(input)

        is_lightrag_format = docs_format == FULL_DOCS_FORMAT_LIGHTRAG
        if is_lightrag_format and lightrag_document_paths is not None:
            if len(lightrag_document_paths) != len(input):
                raise ValueError(
                    "Number of lightrag_document_paths must match the number of documents"
                )
        if parse_engine is not None and len(parse_engine) != len(input):
            raise ValueError(
                "Number of parse engines must match the number of documents"
            )
        if process_options is not None and len(process_options) != len(input):
            raise ValueError(
                "Number of process options must match the number of documents"
            )
        if chunk_options is not None and len(chunk_options) != len(input):
            raise ValueError(
                "Number of chunk_options dicts must match the number of documents"
            )

        def _parse_engine_at(index: int) -> str | None:
            if parse_engine is None:
                return None
            engine = str(parse_engine[index] or "").strip().lower()
            return engine or None

        def _process_options_at(index: int) -> str:
            if process_options is None:
                return ""
            from lightrag.parser_routing import sanitize_process_options

            return sanitize_process_options(process_options[index])

        def _chunk_options_at(index: int) -> dict[str, Any]:
            """Resolve the per-doc chunk_options snapshot.

            When the caller supplied ``chunk_options`` we take a deep
            copy so two docs in the same batch (broadcast from a single
            dict) cannot share mutable inner sub-dicts; otherwise we
            build a fresh snapshot from ``self.addon_params['chunker']``.

            F-strategy runtime args (``split_by_character`` /
            ``split_by_character_only`` from :meth:`LightRAG.ainsert`)
            are baked into the snapshot upstream — ainsert calls
            :func:`lightrag.parser_routing.resolve_chunk_options` itself
            and passes the result via ``chunk_options=``.  This function
            is purely a persistence helper; chunker-config construction
            is not its concern.
            """
            from lightrag.parser_routing import resolve_chunk_options

            if chunk_options is not None:
                return deepcopy(chunk_options[index])
            return resolve_chunk_options(self.addon_params)

        # 1. Validate ids and build contents (when lightrag: no content dedup, content may be empty)
        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

        # Two basenames per file:
        #  - source_keys: user-visible name preserved verbatim, written as
        #    full_docs.file_path / doc_status.file_path so the UI can render
        #    the user's original ``[hint]`` choice.
        #  - canonical_keys: parser-hint stripped basename used for filename
        #    dedup and as the seed for deterministic doc_ids; written as
        #    full_docs.canonical_basename / doc_status.canonical_basename.
        source_keys = [document_source_key(path) for path in file_paths]
        canonical_keys = [document_canonical_key(path) for path in file_paths]
        contents: dict[str, dict[str, Any]] = {}
        source_to_doc_id: dict[str, str] = {}
        content_hash_to_doc_id: dict[str, str] = {}
        duplicate_attempts: list[dict[str, Any]] = []

        def _add_content(
            index: int,
            content: str,
            doc_format: str,
            *,
            lightrag_document_path: str | None = None,
        ) -> None:
            source_key = source_keys[index]
            canonical_key = canonical_keys[index]
            source_path = file_paths[index]

            # Body length excludes the {{LRdoc}} marker so duplicate-attempt
            # bookkeeping reports the same units as raw documents.
            # strip_lightrag_doc_prefix is a no-op for non-lightrag formats.
            body_length = len(strip_lightrag_doc_prefix(content, doc_format))

            # Compute content hash: skip for pending_parse (content extracted later).
            content_hash: str | None = None
            if doc_format == FULL_DOCS_FORMAT_RAW:
                content_hash = compute_text_content_hash(content or "")
            elif doc_format == FULL_DOCS_FORMAT_LIGHTRAG and lightrag_document_path:
                content_hash = compute_file_content_hash(
                    resolve_lightrag_document_path(lightrag_document_path)
                )

            known_source = has_known_document_source(canonical_key)
            if ids is not None:
                doc_id = ids[index]
            elif known_source:
                doc_id = compute_mdhash_id(canonical_key, prefix="doc-")
            elif doc_format == FULL_DOCS_FORMAT_RAW:
                doc_id = compute_mdhash_id(content or "", prefix="doc-")
            elif content_hash:
                doc_id = compute_mdhash_id(content_hash, prefix="doc-")
            else:
                doc_id = compute_mdhash_id(
                    f"{canonical_key}-{track_id}-{index}", prefix="doc-"
                )

            if known_source and canonical_key in source_to_doc_id:
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": source_to_doc_id[canonical_key],
                        "file_path": source_key,
                        "content_length": body_length,
                        "existing_status": "batch_duplicate",
                        "existing_track_id": "",
                        "duplicate_kind": "filename",
                    }
                )
                return

            if content_hash and content_hash in content_hash_to_doc_id:
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": content_hash_to_doc_id[content_hash],
                        "file_path": source_key,
                        "content_length": body_length,
                        "existing_status": "batch_duplicate",
                        "existing_track_id": "",
                        "duplicate_kind": "content_hash",
                    }
                )
                return

            if known_source:
                source_to_doc_id[canonical_key] = doc_id
            if content_hash:
                content_hash_to_doc_id[content_hash] = doc_id

            content_data: dict[str, Any] = {
                "content": content,
                "file_path": source_key,
                "canonical_basename": canonical_key,
                "parse_format": doc_format,
            }
            if content_hash:
                content_data["content_hash"] = content_hash
            # Persist the original path only when it actually carries directory
            # information (absolute path or contains a separator); a plain
            # basename is already captured by ``file_path``.
            raw_source = str(source_path).strip()
            if raw_source and (os.sep in raw_source or "/" in raw_source):
                content_data["source_path"] = source_path
            if lightrag_document_path:
                content_data["lightrag_document_path"] = lightrag_document_path
            if engine := _parse_engine_at(index):
                content_data["parse_engine"] = engine
            options_str = _process_options_at(index)
            if options_str:
                content_data["process_options"] = options_str
            # Always snapshot chunk_options at enqueue time — independent
            # of whether process_options selected a specific strategy —
            # so the per-doc parameters are frozen even when ``F``
            # (default) is used.
            content_data["chunk_options"] = _chunk_options_at(index)
            contents[doc_id] = content_data

        if is_lightrag_format:
            # LightRAG Document: no content hash dedup; content may be empty
            for i in range(len(file_paths)):
                path = file_paths[i]
                lightrag_path = (
                    lightrag_document_paths[i] if lightrag_document_paths else ""
                ) or path
                # Per docs/FileProcessingConfiguration-zh.md, full_docs.content
                # for format=lightrag must be "{{LRdoc}}" + a leading summary.
                # Read the blocks file and derive the summary; if the file is
                # not yet readable (rare, e.g. mid-rotation), fall back to an
                # empty summary so enqueue is never blocked by I/O issues.
                try:
                    resolved_path = str(
                        Path(resolve_lightrag_document_path(str(lightrag_path)))
                    )
                    merged_text, _ = await load_lightrag_document_content(resolved_path)
                except Exception as exc:
                    logger.warning(
                        f"[apipeline_enqueue] failed to load LightRAG Document "
                        f"for summary ({lightrag_path}): {exc}"
                    )
                    merged_text = ""
                summary_content = make_lightrag_doc_content(merged_text)
                _add_content(
                    i,
                    summary_content,
                    FULL_DOCS_FORMAT_LIGHTRAG,
                    lightrag_document_path=lightrag_path,
                )
        elif ids is not None:
            for i, doc in enumerate(input):
                cleaned_content = sanitize_text_for_encoding(doc)
                _add_content(
                    i,
                    cleaned_content,
                    FULL_DOCS_FORMAT_RAW,
                )
        elif docs_format == FULL_DOCS_FORMAT_PENDING_PARSE:
            for i, doc in enumerate(input):
                _add_content(
                    i,
                    doc or "",
                    FULL_DOCS_FORMAT_PENDING_PARSE,
                )
        else:
            for i, doc in enumerate(input):
                cleaned_content = sanitize_text_for_encoding(doc)
                _add_content(i, cleaned_content, FULL_DOCS_FORMAT_RAW)

        # 2. Generate document initial status (without content)
        def _initial_doc_status(content_data: dict[str, Any]) -> dict[str, Any]:
            # For lightrag-format full_docs the persisted content carries the
            # ``{{LRdoc}}`` marker; strip it so summary/length match raw
            # semantics (the marker is full_docs internal bookkeeping and
            # must not leak into doc_status).  strip_lightrag_doc_prefix
            # internally checks parse_format, so non-lightrag formats pass
            # through untouched.
            body_text = strip_lightrag_doc_prefix(
                content_data.get("content", ""),
                content_data.get("parse_format"),
            )
            base: dict[str, Any] = {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(body_text),
                "content_length": len(body_text),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data["file_path"],
                "canonical_basename": content_data.get("canonical_basename"),
                "track_id": track_id,
            }
            if content_data.get("content_hash"):
                base["content_hash"] = content_data["content_hash"]
            options_str = content_data.get("process_options") or ""
            if options_str:
                # Mirror process_options into doc_status.metadata so admin UIs
                # can surface the per-document strategy without a full_docs lookup.
                base["metadata"] = {"process_options": options_str}
            return base

        new_docs: dict[str, Any] = {
            id_: _initial_doc_status(content_data)
            for id_, content_data in contents.items()
        }

        # Serialise the dedup-read-then-upsert critical section across
        # concurrent enqueue calls within the same workspace.  Without
        # this, two enqueues for the same content (e.g. /upload during
        # scan's processing phase, or two uploads via /text + /upload)
        # can both read doc_status before either upserts, both miss the
        # content_hash dedup, and both end up writing PENDING rows for
        # the same content — bypassing the dedup that's supposed to
        # land one of them as ``duplicate_kind=content_hash`` FAILED.
        #
        # The lock is workspace-scoped and only spans steps 3-4 below
        # (filter_keys → upserts).  It does NOT block concurrent
        # processing (``apipeline_process_enqueue_documents`` reads
        # doc_status independently) or scan classification
        # (``scanning_exclusive`` already gates concurrent enqueue).
        # Lock order: enqueue_serialize → pipeline_status_lock (the
        # request_pending nudge inside is fine; no caller holds
        # pipeline_status_lock first then needs enqueue_serialize).
        enqueue_serialize_lock = get_namespace_lock(
            "enqueue_serialize", workspace=self.workspace
        )

        async with enqueue_serialize_lock:
            # 3. Filter out already processed documents
            # Get docs ids
            all_new_doc_ids = set(new_docs.keys())
            # Exclude IDs of documents that are already enqueued.  The previous
            # ``reprocess_existing_non_processed`` flag has been removed: any
            # same-name record (regardless of status) is treated as a duplicate
            # here.  Recovering half-processed documents is now the job of the
            # pipeline's resume logic, which runs in apipeline_process_enqueue_documents
            # rather than this enqueue path.
            unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

            for doc_id in list(unique_new_doc_ids):
                content_data = contents[doc_id]

                # 3a. Filename-based dedup: same basename always treated as duplicate.
                match = await get_existing_doc_by_file_basename(
                    self.doc_status, content_data["file_path"]
                )
                if match:
                    existing_doc_id, existing_doc = match
                    unique_new_doc_ids.discard(doc_id)
                    duplicate_attempts.append(
                        {
                            "doc_id": doc_id,
                            "original_doc_id": existing_doc_id,
                            "file_path": content_data["file_path"],
                            "content_length": new_docs.get(doc_id, {}).get(
                                "content_length", 0
                            ),
                            "existing_status": doc_status_field(
                                existing_doc, "status", "unknown"
                            ),
                            "existing_track_id": doc_status_field(
                                existing_doc, "track_id", ""
                            ),
                            "duplicate_kind": "filename",
                        }
                    )
                    continue

                # 3b. Content-hash dedup: different filename but same body still dupes.
                content_hash = content_data.get("content_hash")
                if not content_hash:
                    continue
                hash_match = await get_existing_doc_by_content_hash(
                    self.doc_status, content_hash
                )
                if hash_match:
                    existing_doc_id, existing_doc = hash_match
                    unique_new_doc_ids.discard(doc_id)
                    duplicate_attempts.append(
                        {
                            "doc_id": doc_id,
                            "original_doc_id": existing_doc_id,
                            "file_path": content_data["file_path"],
                            "content_length": new_docs.get(doc_id, {}).get(
                                "content_length", 0
                            ),
                            "existing_status": doc_status_field(
                                existing_doc, "status", "unknown"
                            ),
                            "existing_track_id": doc_status_field(
                                existing_doc, "track_id", ""
                            ),
                            "duplicate_kind": "content_hash",
                        }
                    )

            # Handle duplicate documents - create trackable records with current track_id
            ignored_ids = list(all_new_doc_ids - unique_new_doc_ids)
            for doc_id in ignored_ids:
                if any(
                    attempt.get("doc_id") == doc_id for attempt in duplicate_attempts
                ):
                    continue
                existing_doc = await self.doc_status.get_by_id(doc_id)
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": doc_id,
                        "file_path": new_docs.get(doc_id, {}).get(
                            "file_path", "unknown_source"
                        ),
                        "content_length": new_docs.get(doc_id, {}).get(
                            "content_length", 0
                        ),
                        "existing_status": (
                            existing_doc.get("status", "unknown")
                            if existing_doc
                            else "unknown"
                        ),
                        "existing_track_id": (
                            existing_doc.get("track_id", "") if existing_doc else ""
                        ),
                        "duplicate_kind": "filename",
                    }
                )

            if duplicate_attempts:
                duplicate_docs: dict[str, Any] = {}
                for index, attempt in enumerate(duplicate_attempts):
                    doc_id = attempt["doc_id"]
                    file_path = attempt.get("file_path") or "unknown_source"
                    duplicate_kind = attempt.get("duplicate_kind") or "filename"
                    logger.warning(
                        f"Duplicate document detected ({duplicate_kind}): "
                        f"{doc_id} ({file_path})"
                    )

                    # Create a new record with unique ID for this duplicate attempt
                    dup_record_id = compute_mdhash_id(
                        f"{doc_id}-{track_id}-{index}-{file_path}", prefix="dup-"
                    )
                    if duplicate_kind == "content_hash":
                        error_prefix = (
                            "Identical content already exists under another filename."
                        )
                    else:
                        error_prefix = "File name already exists."
                    duplicate_docs[dup_record_id] = {
                        "status": DocStatus.FAILED,
                        "content_summary": (
                            f"[DUPLICATE:{duplicate_kind}] Original document: "
                            f"{attempt.get('original_doc_id', doc_id)}"
                        ),
                        "content_length": attempt.get("content_length", 0),
                        "chunks_count": 0,
                        "chunks_list": [],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": file_path,
                        "track_id": track_id,  # Use current track_id for tracking
                        "error_msg": (
                            f"{error_prefix} "
                            f"Original doc_id: {attempt.get('original_doc_id', doc_id)}, "
                            f"Status: {attempt.get('existing_status', 'unknown')}"
                        ),
                        "metadata": {
                            "is_duplicate": True,
                            "duplicate_kind": duplicate_kind,
                            "original_doc_id": attempt.get("original_doc_id", doc_id),
                            "original_track_id": attempt.get("existing_track_id", ""),
                        },
                    }

                # Store duplicate records in doc_status
                if duplicate_docs:
                    await self.doc_status.upsert(duplicate_docs)
                    logger.info(
                        f"Created {len(duplicate_docs)} duplicate document records with track_id: {track_id}"
                    )

            # Filter new_docs to only include documents with unique IDs
            new_docs = {
                doc_id: new_docs[doc_id]
                for doc_id in unique_new_doc_ids
                if doc_id in new_docs
            }

            if not new_docs:
                logger.warning("No new unique documents were found.")
                return

            # 4. Store document content in full_docs and status in doc_status
            full_docs_data = {
                doc_id: {
                    "content": contents[doc_id].get("content", ""),
                    "file_path": contents[doc_id]["file_path"],
                    "canonical_basename": contents[doc_id].get("canonical_basename"),
                    "parse_format": contents[doc_id].get(
                        "parse_format", FULL_DOCS_FORMAT_RAW
                    ),
                }
                for doc_id in new_docs.keys()
            }
            for doc_id in new_docs.keys():
                if contents[doc_id].get("content_hash"):
                    full_docs_data[doc_id]["content_hash"] = contents[doc_id][
                        "content_hash"
                    ]
                if contents[doc_id].get("source_path"):
                    full_docs_data[doc_id]["source_path"] = contents[doc_id][
                        "source_path"
                    ]
                if contents[doc_id].get("lightrag_document_path"):
                    full_docs_data[doc_id]["lightrag_document_path"] = contents[doc_id][
                        "lightrag_document_path"
                    ]
                if contents[doc_id].get("parse_engine"):
                    full_docs_data[doc_id]["parse_engine"] = contents[doc_id][
                        "parse_engine"
                    ]
                if contents[doc_id].get("process_options"):
                    full_docs_data[doc_id]["process_options"] = contents[doc_id][
                        "process_options"
                    ]
                # ``chunk_options`` is always populated by ``_add_content``
                # at enqueue time so it's persisted unconditionally.
                if contents[doc_id].get("chunk_options") is not None:
                    full_docs_data[doc_id]["chunk_options"] = contents[doc_id][
                        "chunk_options"
                    ]
            await self.full_docs.upsert(full_docs_data)
            # Persist data to disk immediately
            await self.full_docs.index_done_callback()

            # Store document status (without content)
            await self.doc_status.upsert(new_docs)
            logger.debug(f"Stored {len(new_docs)} new unique documents")

        # Notify any in-flight processing loop that new work has arrived.
        # The loop checks ``request_pending`` after each batch and will
        # re-query doc_status to pick up these PENDING rows.  Without
        # this nudge a caller that does not subsequently call
        # ``apipeline_process_enqueue_documents`` (or whose call races
        # with the loop's just-finished batch) could leave the new docs
        # stranded until the next unrelated trigger.
        async with pipeline_status_lock:
            if pipeline_status.get("busy"):
                pipeline_status["request_pending"] = True

        return track_id

    async def apipeline_enqueue_error_documents(
        self,
        error_files: list[dict[str, Any]],
        track_id: str | None = None,
    ) -> None:
        """
        Record file extraction errors in doc_status storage.

        This function creates error document entries in the doc_status storage for files
        that failed during the extraction process. Each error entry contains information
        about the failure to help with debugging and monitoring.

        Args:
            error_files: List of dictionaries containing error information for each failed file.
                Each dictionary should contain:
                - file_path: Original file name/path
                - error_description: Brief error description (for content_summary)
                - original_error: Full error message (for error_msg)
                - file_size: File size in bytes (for content_length, 0 if unknown)
            track_id: Optional tracking ID for grouping related operations

        Returns:
            None
        """
        if not error_files:
            logger.debug("No error files to record")
            return

        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("error")

        error_docs: dict[str, Any] = {}
        current_time = datetime.now(timezone.utc).isoformat()

        for error_file in error_files:
            file_path = error_file.get("file_path", "unknown_file")
            error_description = error_file.get(
                "error_description", "File extraction failed"
            )
            original_error = error_file.get("original_error", "Unknown error")
            file_size = error_file.get("file_size", 0)

            # Generate unique doc_id with "error-" prefix
            doc_id_content = f"{file_path}-{error_description}"
            doc_id = compute_mdhash_id(doc_id_content, prefix="error-")

            error_docs[doc_id] = {
                "status": DocStatus.FAILED,
                "content_summary": error_description,
                "content_length": file_size,
                "error_msg": original_error,
                "chunks_count": 0,  # No chunks for failed files
                "chunks_list": [],
                "created_at": current_time,
                "updated_at": current_time,
                "file_path": file_path,
                "track_id": track_id,
                "metadata": {
                    "error_type": "file_extraction_error",
                },
            }

        # Store error documents in doc_status
        if error_docs:
            await self.doc_status.upsert(error_docs)
            # Log each error for debugging
            for doc_id, error_doc in error_docs.items():
                logger.error(
                    f"File processing error: - ID: {doc_id} {error_doc['file_path']}"
                )

    async def apipeline_process_enqueue_documents(self) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Validate document data consistency and fix any issues
        3. Split document content into chunks
        4. Process each chunk for entity and relation extraction
        5. Update the document status
        """
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                to_process_docs: dict[
                    str, DocProcessingStatus
                ] = await self.doc_status.get_docs_by_statuses(
                    list(_INFLIGHT_DOC_STATUSES)
                )

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Default Job",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 0,
                        "batchs": 0,  # Total number of files to be processed
                        "cur_batch": 0,  # Number of files already processed
                        "request_pending": False,  # Clear any previous request
                        "cancellation_requested": False,  # Initialize cancellation flag
                        "latest_message": "",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy, just set request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        # Tracks whether the loop has already released ``busy`` under
        # the same critical section that observed request_pending=False.
        # This makes the exit handoff atomic: a concurrent enqueue can
        # either set request_pending BEFORE we release (in which case
        # the loop continues with a fresh snapshot) or AFTER (in which
        # case it sees busy=False and starts a new loop via its own
        # process_enqueue call).  Without this, a small window between
        # "loop reads request_pending=False" and "finally clears busy"
        # could strand newly-enqueued PENDING docs.
        busy_released_in_loop = False

        try:
            # Process documents until no more documents or requests
            while True:
                # Check for cancellation request at the start of main loop
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        pipeline_status["request_pending"] = False
                        pipeline_status["cancellation_requested"] = False

                        log_message = "Pipeline cancelled by user"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # Exit directly, skipping request_pending check
                        return

                if not to_process_docs:
                    log_message = "All enqueued documents have been processed"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    if await self._atomic_release_busy_or_consume_pending(
                        pipeline_status, pipeline_status_lock
                    ):
                        busy_released_in_loop = True
                        break
                    to_process_docs = await self.doc_status.get_docs_by_statuses(
                        list(_INFLIGHT_DOC_STATUSES)
                    )
                    continue

                # Validate document data consistency and fix any issues
                to_process_docs = await self._validate_and_fix_document_consistency(
                    to_process_docs, pipeline_status, pipeline_status_lock
                )

                if not to_process_docs:
                    log_message = (
                        "No valid documents to process after consistency check"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    if await self._atomic_release_busy_or_consume_pending(
                        pipeline_status, pipeline_status_lock
                    ):
                        busy_released_in_loop = True
                        break
                    to_process_docs = await self.doc_status.get_docs_by_statuses(
                        list(_INFLIGHT_DOC_STATUSES)
                    )
                    continue

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                await self._run_pipeline_batch(
                    to_process_docs,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                )

                # Atomic exit handoff: if request_pending was set during
                # this batch (e.g. a concurrent enqueue while busy=True),
                # clear it and refetch.  Otherwise release ``busy`` under
                # the SAME lock so a concurrent enqueue cannot squeeze a
                # request_pending=True past us into a now-stranded state.
                if await self._atomic_release_busy_or_consume_pending(
                    pipeline_status, pipeline_status_lock
                ):
                    busy_released_in_loop = True
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                to_process_docs = await self.doc_status.get_docs_by_statuses(
                    list(_INFLIGHT_DOC_STATUSES)
                )

        finally:
            log_message = "Enqueued document processing pipeline stopped"
            logger.info(log_message)
            # If the loop already released ``busy`` under the atomic exit
            # check, don't clobber it here — a concurrent enqueue may have
            # observed busy=False and started a new processing pass that
            # has set busy=True for itself.  Cancellation flag and log
            # bookkeeping are always safe to update.
            async with pipeline_status_lock:
                if not busy_released_in_loop:
                    pipeline_status["busy"] = False
                pipeline_status["cancellation_requested"] = (
                    False  # Always reset cancellation flag
                )
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    # ============================================================
    # Pipeline orchestration
    # ============================================================

    async def _run_pipeline_batch(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
        *,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Run one batch of pending documents through the parse → analyze →
        process queues.

        Three cascading layers of queues:
          - Layer 1: Content Parsing  (parse_native / parse_mineru / parse_docling)
          - Layer 2: Multimodal Analyze  (analyze_multimodal)
          - Layer 3: Entity / Relation Extraction  (process_single_document)
        """
        total_files = len(to_process_docs)
        pipeline_status["job_name"] = self._format_job_name(
            to_process_docs, total_files
        )

        ctx = _BatchRunContext(
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            semaphore=asyncio.Semaphore(self.max_parallel_insert),
            total_files=total_files,
            q_native=asyncio.Queue(maxsize=self.queue_size_default),
            q_mineru=asyncio.Queue(maxsize=self.queue_size_default),
            q_docling=asyncio.Queue(maxsize=self.queue_size_default),
            q_analyze=asyncio.Queue(maxsize=self.queue_size_default),
            q_process=asyncio.Queue(maxsize=self.queue_size_insert),
        )

        workers: list[asyncio.Task] = []
        for _ in range(max(1, self.max_parallel_parse_native)):
            workers.append(
                asyncio.create_task(self._parse_worker("native", ctx.q_native, ctx))
            )
        for _ in range(max(1, self.max_parallel_parse_mineru)):
            workers.append(
                asyncio.create_task(self._parse_worker("mineru", ctx.q_mineru, ctx))
            )
        for _ in range(max(1, self.max_parallel_parse_docling)):
            workers.append(
                asyncio.create_task(self._parse_worker("docling", ctx.q_docling, ctx))
            )
        for _ in range(max(1, self.max_parallel_analyze)):
            workers.append(asyncio.create_task(self._analyze_worker(ctx)))
        for _ in range(max(1, self.max_parallel_insert)):
            workers.append(asyncio.create_task(self._process_worker(ctx)))

        # Add pending files to the correct parsing queue
        for doc_id, status_doc in to_process_docs.items():
            content_data = await self.full_docs.get_by_id(doc_id) or {}
            engine = resolve_stored_document_parser_engine(
                file_path=getattr(status_doc, "file_path", "unknown_source"),
                content_data=content_data,
            )
            if engine == "mineru":
                await ctx.q_mineru.put((doc_id, status_doc))
            elif engine == "docling":
                await ctx.q_docling.put((doc_id, status_doc))
            else:
                await ctx.q_native.put((doc_id, status_doc))

        await asyncio.gather(
            ctx.q_native.join(), ctx.q_mineru.join(), ctx.q_docling.join()
        )
        await ctx.q_analyze.join()
        await ctx.q_process.join()

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def _validate_and_fix_document_consistency(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
        pipeline_status: dict,
        pipeline_status_lock: asyncio.Lock,
    ) -> dict[str, DocProcessingStatus]:
        """Validate and fix document data consistency by deleting inconsistent entries, but preserve failed documents"""
        inconsistent_docs = []
        failed_docs_to_preserve = []
        successful_deletions = 0

        # Check each document's data consistency
        for doc_id, status_doc in to_process_docs.items():
            # Check if corresponding content exists in full_docs
            content_data = await self.full_docs.get_by_id(doc_id)
            if not content_data:
                # Check if this is a failed document that should be preserved
                if (
                    hasattr(status_doc, "status")
                    and status_doc.status == DocStatus.FAILED
                ):
                    failed_docs_to_preserve.append(doc_id)
                else:
                    inconsistent_docs.append(doc_id)

        # Log information about failed documents that will be preserved
        if failed_docs_to_preserve:
            async with pipeline_status_lock:
                preserve_message = f"Preserving {len(failed_docs_to_preserve)} failed document entries for manual review"
                logger.info(preserve_message)
                pipeline_status["latest_message"] = preserve_message
                pipeline_status["history_messages"].append(preserve_message)

            # Remove failed documents from processing list but keep them in doc_status
            for doc_id in failed_docs_to_preserve:
                to_process_docs.pop(doc_id, None)

        # Delete inconsistent document entries(excluding failed documents)
        if inconsistent_docs:
            async with pipeline_status_lock:
                summary_message = (
                    f"Inconsistent document entries found: {len(inconsistent_docs)}"
                )
                logger.info(summary_message)
                pipeline_status["latest_message"] = summary_message
                pipeline_status["history_messages"].append(summary_message)

            successful_deletions = 0
            for doc_id in inconsistent_docs:
                try:
                    status_doc = to_process_docs[doc_id]
                    file_path = resolve_doc_file_path(status_doc=status_doc)

                    # Delete doc_status entry
                    await self.doc_status.delete([doc_id])
                    successful_deletions += 1

                    # Log successful deletion
                    async with pipeline_status_lock:
                        log_message = (
                            f"Deleted inconsistent entry: {doc_id} ({file_path})"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                    # Remove from processing list
                    to_process_docs.pop(doc_id, None)

                except Exception as e:
                    # Log deletion failure
                    async with pipeline_status_lock:
                        error_message = f"Failed to delete entry: {doc_id} - {str(e)}"
                        logger.error(error_message)
                        pipeline_status["latest_message"] = error_message
                        pipeline_status["history_messages"].append(error_message)

        # Final summary log
        # async with pipeline_status_lock:
        #     final_message = f"Successfully deleted {successful_deletions} inconsistent entries, preserved {len(failed_docs_to_preserve)} failed documents"
        #     logger.info(final_message)
        #     pipeline_status["latest_message"] = final_message
        #     pipeline_status["history_messages"].append(final_message)

        # Reset interrupted documents that pass consistency checks to PENDING status
        docs_to_reset = {}
        reset_count = 0

        for doc_id, status_doc in to_process_docs.items():
            # Check if document has corresponding content in full_docs (consistency check)
            content_data = await self.full_docs.get_by_id(doc_id)
            if content_data:  # Document passes consistency check
                # Check if document is in interrupted status
                if hasattr(status_doc, "status") and status_doc.status in [
                    DocStatus.PROCESSING,
                    DocStatus.FAILED,
                    DocStatus.PARSING,
                    DocStatus.ANALYZING,
                ]:
                    preserved_chunks_list, preserved_chunks_count = (
                        chunk_fields_from_status_doc(status_doc)
                    )
                    resolved_file_path = resolve_doc_file_path(
                        status_doc=status_doc,
                        content_data=content_data,
                    )
                    # Prepare document for status reset to PENDING
                    docs_to_reset[doc_id] = {
                        "status": DocStatus.PENDING,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "chunks_count": preserved_chunks_count,
                        "chunks_list": preserved_chunks_list,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": resolved_file_path,
                        "track_id": getattr(status_doc, "track_id", ""),
                        "content_hash": getattr(status_doc, "content_hash", None),
                        # Clear transient error / processing fields but preserve
                        # long-lived per-doc metadata (process_options) seeded
                        # at enqueue time.
                        "error_msg": "",
                        "metadata": doc_status_transition_metadata(status_doc),
                    }

                    # Update the status in to_process_docs as well
                    status_doc.status = DocStatus.PENDING
                    status_doc.file_path = resolved_file_path
                    reset_count += 1

        # Update doc_status storage if there are documents to reset
        if docs_to_reset:
            await self.doc_status.upsert(docs_to_reset)

            async with pipeline_status_lock:
                reset_message = (
                    f"Reset {reset_count} documents from "
                    "PARSING/ANALYZING/PROCESSING/FAILED to PENDING status"
                )
                logger.info(reset_message)
                pipeline_status["latest_message"] = reset_message
                pipeline_status["history_messages"].append(reset_message)

        return to_process_docs

    async def _atomic_release_busy_or_consume_pending(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """Atomically decide whether to release ``busy`` or consume a
        pending request.

        Closes the loop-exit handoff race: a concurrent enqueue that
        sets ``request_pending`` while the processing loop is on its
        way out will be observed in the same critical section that
        releases ``busy``, so the loop sees it and refetches instead
        of stranding the new doc in PENDING.

        Returns:
            True when ``busy`` has been cleared under the same lock
            that observed ``request_pending=False`` — caller must
            break out of the loop and skip clearing ``busy`` in its
            finally block.

            False when ``request_pending`` was set: the flag is
            cleared and the caller must refetch ``doc_status`` and
            continue the loop.
        """
        async with pipeline_status_lock:
            if pipeline_status.get("request_pending", False):
                pipeline_status["request_pending"] = False
                return False
            pipeline_status["busy"] = False
            return True

    @staticmethod
    def _format_job_name(
        to_process_docs: dict[str, DocProcessingStatus],
        total_files: int,
    ) -> str:
        """Build the ``job_name`` shown in pipeline_status for one batch."""
        first_doc = next(iter(to_process_docs.values()))
        first_doc_path = first_doc.file_path
        if first_doc_path:
            path_prefix = first_doc_path[:20] + (
                "..." if len(first_doc_path) > 20 else ""
            )
        else:
            path_prefix = "unknown_source"
        return f"{path_prefix}[{total_files} files]"

    # ============================================================
    # Cascading queue workers (Layer 1 -> 2 -> 3)
    # ============================================================

    async def _parse_worker(
        self,
        engine: str,
        in_q: asyncio.Queue,
        ctx: _BatchRunContext,
    ) -> None:
        """Layer 1 worker: consume (doc_id, status_doc) and emit parsed data.

        Marks PARSING, runs the engine-specific parser (mineru / docling /
        native), refreshes ``content_hash`` if the parser patched it, and
        either short-circuits via ``_mark_duplicate_after_parse`` or hands
        off to ``q_analyze``.  Writes FAILED on exception.
        """
        while True:
            item = await in_q.get()
            try:
                doc_id_w, status_doc_w = item
                file_path_w = getattr(status_doc_w, "file_path", "unknown_source")
                content_data_w = await self.full_docs.get_by_id(doc_id_w)
                if not content_data_w:
                    raise Exception(
                        f"Document content not found in full_docs for doc_id: {doc_id_w}"
                    )
                await self._upsert_doc_status_transition(
                    doc_id=doc_id_w,
                    status=DocStatus.PARSING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )
                if engine == "mineru":
                    parsed_data_w = await self.parse_mineru(
                        doc_id_w, file_path_w, content_data_w
                    )
                elif engine == "docling":
                    parsed_data_w = await self.parse_docling(
                        doc_id_w, file_path_w, content_data_w
                    )
                else:
                    parsed_data_w = await self.parse_native(
                        doc_id_w, file_path_w, content_data_w
                    )

                # Mirror non-fatal parser warnings (e.g. legacy docx tables
                # missing w14:paraId) onto the in-memory status_doc so the
                # ANALYZING / PROCESSING / PROCESSED / FAILED upserts carry
                # the field through ``doc_status_transition_metadata``.
                parse_warnings_payload_w = parsed_data_w.get("parse_warnings")
                if parse_warnings_payload_w:
                    if not isinstance(status_doc_w.metadata, dict):
                        status_doc_w.metadata = {}
                    status_doc_w.metadata["parse_warnings"] = parse_warnings_payload_w

                # parse_* may have patched content_hash for
                # pending_parse → raw transitions.
                refreshed = await self.doc_status.get_by_id(doc_id_w)
                if refreshed:
                    refreshed_hash = (
                        refreshed.get("content_hash")
                        if isinstance(refreshed, dict)
                        else getattr(refreshed, "content_hash", None)
                    )
                    if refreshed_hash:
                        status_doc_w.content_hash = refreshed_hash

                if await self._mark_duplicate_after_parse(
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                    content_hash=status_doc_w.content_hash,
                    content_length=len(parsed_data_w.get("content", "")),
                    content_data=content_data_w,
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                ):
                    continue

                await ctx.q_analyze.put((doc_id_w, status_doc_w, parsed_data_w))
            except Exception as e:
                logger.error(f"Parse worker failed ({engine}): {e}")
                try:
                    await self._upsert_doc_status_transition(
                        doc_id=doc_id_w,
                        status=DocStatus.FAILED,
                        status_doc=status_doc_w,
                        file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                        extra_fields={"error_msg": str(e)},
                    )
                except Exception:
                    pass
            finally:
                in_q.task_done()

    async def _analyze_worker(self, ctx: _BatchRunContext) -> None:
        """Layer 2 worker: run multimodal analysis (VLM) and feed q_process.

        Refreshes ``content_summary`` / ``content_length`` from the parsed
        body (pending_parse → lightrag / raw documents start with empty
        summary / zero length at enqueue) so PROCESSING / PROCESSED upserts
        end up with real values.
        """
        while True:
            item = await ctx.q_analyze.get()
            try:
                doc_id_w, status_doc_w, parsed_data_w = item
                file_path_w = getattr(status_doc_w, "file_path", "unknown_source")
                refreshed_content_w = parsed_data_w.get("content", "") or ""
                refreshed_summary_w = get_content_summary(refreshed_content_w)
                refreshed_length_w = len(refreshed_content_w)
                status_doc_w.content_summary = refreshed_summary_w
                status_doc_w.content_length = refreshed_length_w
                await self._upsert_doc_status_transition(
                    doc_id=doc_id_w,
                    status=DocStatus.ANALYZING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )
                analyzed = await self.analyze_multimodal(
                    doc_id=doc_id_w,
                    file_path=file_path_w,
                    parsed_data=parsed_data_w,
                )
                await ctx.q_process.put((doc_id_w, status_doc_w, analyzed))
            except Exception as e:
                # Mirror _parse_worker: failures here must transition the
                # document to FAILED with a diagnostic ``error_msg``, otherwise
                # MultimodalAnalysisError (raised by analyze_multimodal under
                # the new hard-failure contract) would leave the doc stuck in
                # ANALYZING forever.
                logger.error(f"Analyze worker failed: {e}")
                try:
                    await self._upsert_doc_status_transition(
                        doc_id=doc_id_w,
                        status=DocStatus.FAILED,
                        status_doc=status_doc_w,
                        file_path=getattr(
                            status_doc_w, "file_path", "unknown_source"
                        ),
                        extra_fields={"error_msg": str(e)},
                    )
                except Exception:
                    pass
            finally:
                ctx.q_analyze.task_done()

    async def _process_worker(self, ctx: _BatchRunContext) -> None:
        """Layer 3 worker: dispatch each ready document to single-doc processing."""
        while True:
            item = await ctx.q_process.get()
            try:
                doc_id_w, status_doc_w, parsed_data_w = item
                await self.process_single_document(
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    parsed_data=parsed_data_w,
                    ctx=ctx,
                )
            finally:
                ctx.q_process.task_done()

    # ============================================================
    # Single-document state machine
    # ============================================================

    async def process_single_document(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        parsed_data: dict[str, Any],
        ctx: _BatchRunContext,
    ) -> None:
        """Single-document state machine: chunking → KG extraction → merge.

        Always invoked from ``_process_worker`` with ``parsed_data`` already
        populated by ``_parse_worker`` + ``_analyze_worker``.  Drives the
        PROCESSING → PROCESSED state machine, with FAILED fallbacks at both
        the extract and merge stage boundaries.
        """
        from lightrag.parser_routing import parse_process_options

        file_path = resolve_doc_file_path(status_doc=status_doc)
        current_file_number = 0
        file_extraction_stage_ok = False
        processing_start_time = int(time.time())
        first_stage_tasks: list[asyncio.Task] = []
        entity_relation_task: asyncio.Task | None = None
        chunks: dict[str, Any] = {}
        content_data: dict[str, Any] | None = None
        extraction_meta: dict[str, Any] = {}
        chunk_results: list = []
        doc_process_opts = parse_process_options("")

        def get_failed_chunk_snapshot() -> tuple[list[str], int]:
            if chunks:
                chunk_ids = list(chunks.keys())
                return chunk_ids, len(chunk_ids)
            return chunk_fields_from_status_doc(status_doc)

        async with ctx.semaphore:
            try:
                # Resolve file_path from full_docs before honoring a queued
                # cancellation so corrupted doc_status placeholders do not
                # get written back again during retry/cancel flows.
                content_data = await self.full_docs.get_by_id(doc_id)
                if content_data:
                    file_path = resolve_doc_file_path(
                        status_doc=status_doc,
                        content_data=content_data,
                    )
                    status_doc.file_path = file_path

                # Check for cancellation before starting document processing.
                # file_path is resolved before this check so queued documents
                # do not lose their source path on early cancellation.
                await self._raise_if_cancelled(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                )

                async with ctx.pipeline_status_lock:
                    ctx.processed_count += 1
                    current_file_number = ctx.processed_count
                    ctx.pipeline_status["cur_batch"] = ctx.processed_count

                    log_message = (
                        f"Extracting stage {current_file_number}/"
                        f"{ctx.total_files}: {file_path}"
                    )
                    logger.info(log_message)
                    ctx.pipeline_status["history_messages"].append(log_message)
                    log_message = f"Processing d-id: {doc_id}"
                    logger.info(log_message)
                    ctx.pipeline_status["latest_message"] = log_message
                    ctx.pipeline_status["history_messages"].append(log_message)

                    # Prevent memory growth: keep only latest 5000 messages
                    # when exceeding 10000.  Trim in place so Manager.list-
                    # backed shared state remains appendable and visible
                    # across processes.
                    if len(ctx.pipeline_status["history_messages"]) > 10000:
                        logger.info(
                            f"Trimming pipeline history from {len(ctx.pipeline_status['history_messages'])} to 5000 messages"
                        )
                        del ctx.pipeline_status["history_messages"][:-5000]

                content = parsed_data.get("content", "")

                # Decode per-document processing options once; later stages
                # (multimodal hook / KG extraction) re-read them from
                # full_docs as well.
                doc_process_opts = parse_process_options(
                    (content_data or {}).get("process_options", "")
                )

                # Resume guard: if content was already extracted under
                # earlier process_options, purge stale chunks + KG before
                # rebuilding.
                await self._purge_stale_extraction_if_resuming(
                    doc_id=doc_id,
                    status_doc=status_doc,
                    file_path=file_path,
                    content_data=content_data,
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                )

                # Chunker dispatch is driven by whether ``process_options``
                # explicitly named a chunking strategy:
                #   - Explicit selector (F/R/V/P present in the raw
                #     options string): dispatch to a chunker that
                #     follows the standardized file-chunker contract
                #     ``(tokenizer, content, chunk_token_size, *,
                #     <strategy kwargs>)``, with kwargs supplied from
                #     the per-doc ``chunk_options`` snapshot persisted
                #     at enqueue time.
                #   - No selector supplied: honor the
                #     externally-customizable ``self.chunking_func``
                #     with its legacy 6-arg signature so existing
                #     callers (typically :meth:`ainsert` for raw text)
                #     keep working unchanged.  Legacy callers still
                #     read parameters from ``chunk_options`` first
                #     (per-doc snapshot), with ctx values as fallback
                #     for already-enqueued docs predating chunk_options.
                chunk_opts = (content_data or {}).get("chunk_options")
                if not isinstance(chunk_opts, dict) or not chunk_opts:
                    # Backwards compatibility: rows enqueued before the
                    # chunk_options snapshot was added fall back to a
                    # fresh build from current addon_params['chunker'].
                    # F-strategy split args fall back to whatever lives
                    # in addon_params['chunker']['fixed_token']; runtime
                    # overrides are an ainsert-time concern and don't
                    # apply at process time for legacy rows.
                    from lightrag.parser_routing import resolve_chunk_options

                    chunk_opts = resolve_chunk_options(self.addon_params)
                resolved_chunk_size = int(
                    chunk_opts.get("chunk_token_size") or self.chunk_token_size
                )

                # Captured per-strategy below; persisted to
                # ``doc_status.metadata['chunk_opts']`` via ``extraction_meta``
                # so admin/list APIs can see the actual chunker params used.
                chunk_opts_str: str = ""

                if doc_process_opts.chunking_explicit:
                    from lightrag.chunker import (
                        chunking_by_fixed_token,
                        chunking_by_paragraph_semantic,
                        chunking_by_recursive_character,
                        chunking_by_semantic_vector,
                    )

                    strategy = doc_process_opts.chunking
                    if strategy == "P":
                        # P carries its own optional ``chunk_token_size``
                        # override (CHUNK_P_SIZE env or
                        # ``addon_params['chunker']['paragraph_semantic']``);
                        # pop it out of the kwargs so we don't pass it
                        # both positionally and via ``**`` splat (which
                        # would TypeError).  Fall back to the shared
                        # top-level resolved size when unset.
                        p_opts = dict(chunk_opts.get("paragraph_semantic") or {})
                        p_chunk_size = int(
                            p_opts.pop("chunk_token_size", resolved_chunk_size)
                        )
                        p_blocks_path = (
                            str(parsed_data.get("blocks_path") or "").strip() or None
                        )
                        chunk_opts_str = _format_chunking_params(p_chunk_size, p_opts)
                        logger.info(f"Chunking P: {chunk_opts_str}, file: {file_path}")
                        chunking_result = chunking_by_paragraph_semantic(
                            self.tokenizer,
                            content,
                            p_chunk_size,
                            blocks_path=p_blocks_path,
                            **p_opts,
                        )
                    elif strategy == "R":
                        # R carries its own optional ``chunk_token_size``
                        # override (CHUNK_R_SIZE env or
                        # ``addon_params['chunker']['recursive_character']``);
                        # pop it out of the kwargs so we don't pass it
                        # both positionally and via ``**`` splat (which
                        # would TypeError).  Fall back to the shared
                        # top-level resolved size when unset.
                        r_opts = dict(chunk_opts.get("recursive_character") or {})
                        r_chunk_size = int(
                            r_opts.pop("chunk_token_size", resolved_chunk_size)
                        )
                        chunk_opts_str = _format_chunking_params(r_chunk_size, r_opts)
                        logger.info(f"Chunking R: {chunk_opts_str}, file: {file_path}")
                        chunking_result = chunking_by_recursive_character(
                            self.tokenizer,
                            content,
                            r_chunk_size,
                            **r_opts,
                        )
                    elif strategy == "V":
                        # V carries its own optional ``chunk_token_size``
                        # advisory ceiling override (CHUNK_V_SIZE env or
                        # ``addon_params['chunker']['semantic_vector']``);
                        # same pop-then-splat pattern as P/R.
                        v_opts = dict(chunk_opts.get("semantic_vector") or {})
                        v_chunk_size = int(
                            v_opts.pop("chunk_token_size", resolved_chunk_size)
                        )
                        chunk_opts_str = _format_chunking_params(v_chunk_size, v_opts)
                        logger.info(f"Chunking V: {chunk_opts_str}, file: {file_path}")
                        chunking_result = await chunking_by_semantic_vector(
                            self.tokenizer,
                            content,
                            v_chunk_size,
                            embedding_func=self.embedding_func,
                            **v_opts,
                        )
                    else:  # "F"
                        f_opts = chunk_opts.get("fixed_token") or {}
                        chunk_opts_str = _format_chunking_params(
                            resolved_chunk_size, f_opts
                        )
                        logger.info(f"Chunking F: {chunk_opts_str}, file: {file_path}")
                        chunking_result = chunking_by_fixed_token(
                            self.tokenizer,
                            content,
                            resolved_chunk_size,
                            **f_opts,
                        )
                else:
                    f_opts = chunk_opts.get("fixed_token") or {}
                    chunk_opts_str = _format_chunking_params(
                        resolved_chunk_size,
                        {
                            "split_by_character": f_opts.get("split_by_character"),
                            "split_by_character_only": f_opts.get(
                                "split_by_character_only", False
                            ),
                            "overlap": f_opts.get(
                                "chunk_overlap_token_size",
                                self.chunk_overlap_token_size,
                            ),
                        },
                    )
                    logger.info(
                        f"Chunking F(legacy): {chunk_opts_str}, file: {file_path}"
                    )
                    chunking_result = self.chunking_func(
                        self.tokenizer,
                        content,
                        f_opts.get("split_by_character"),
                        f_opts.get("split_by_character_only", False),
                        f_opts.get(
                            "chunk_overlap_token_size",
                            self.chunk_overlap_token_size,
                        ),
                        resolved_chunk_size,
                    )
                if inspect.isawaitable(chunking_result):
                    chunking_result = await chunking_result

                if not isinstance(chunking_result, (list, tuple)):
                    raise TypeError(
                        f"chunking_func must return a list or tuple of dicts, "
                        f"got {type(chunking_result)}"
                    )

                # Reflect the format actually persisted in full_docs.
                # Previously a structured-parse fallback always tagged
                # parse_format=raw, which silently disabled
                # _run_multimodal_postprocess_hook for lightrag documents.
                persisted_format = (
                    content_data.get("parse_format")
                    if isinstance(content_data, dict)
                    else FULL_DOCS_FORMAT_RAW
                ) or FULL_DOCS_FORMAT_RAW
                persisted_engine = (
                    content_data.get("parse_engine")
                    if isinstance(content_data, dict)
                    else None
                )
                extraction_meta = {
                    "parse_format": persisted_format,
                    "parse_engine": persisted_engine
                    or (
                        "native"
                        if persisted_format == FULL_DOCS_FORMAT_LIGHTRAG
                        else "legacy"
                    ),
                    "chunking_method": (
                        # Explicit selector in process_options: reflect
                        # the dispatched strategy.  ``fixed_token_fallback``
                        # is preserved as a defensive label in case a
                        # future selector char slips past the validator.
                        _CHUNKING_METHOD_LABELS.get(
                            doc_process_opts.chunking, "fixed_token_fallback"
                        )
                        if doc_process_opts.chunking_explicit
                        # No selector: chunking_func was invoked, which
                        # defaults to chunking_by_token_size but may be
                        # customized by the caller.
                        else "legacy_chunking_func"
                    ),
                    # Mirrors the chunking start log line (params portion only,
                    # without the strategy prefix or file path) so admins can
                    # see the actual chunker params used.  Carried across
                    # transitions via ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``.
                    "chunk_opts": chunk_opts_str,
                }

                # Multimodal post-process hook entrypoint:
                # runs after chunking and before entity extraction.
                chunking_result = await self._run_multimodal_postprocess_hook(
                    doc_id=doc_id,
                    file_path=file_path,
                    chunking_result=chunking_result,
                    extraction_meta=extraction_meta,
                )

                blocks_path = str(parsed_data.get("blocks_path") or "").strip()
                if blocks_path:
                    max_order = -1
                    for ch in chunking_result:
                        if isinstance(ch, dict) and isinstance(
                            ch.get("chunk_order_index"), int
                        ):
                            max_order = max(max_order, int(ch["chunk_order_index"]))
                    mm_chunks = self._build_mm_chunks_from_sidecars(
                        doc_id=doc_id,
                        file_path=file_path,
                        blocks_path=blocks_path,
                        base_order_index=max_order + 1,
                        process_options=(content_data or {}).get(
                            "process_options"
                        ),
                    )
                    if mm_chunks:
                        chunking_result = list(chunking_result) + mm_chunks
                        extraction_meta["mm_chunks"] = len(mm_chunks)

                # Final hard guard before embedding: split any oversize
                # chunk while preserving heading hierarchy metadata.
                if (
                    self.embedding_token_limit is not None
                    and self.embedding_token_limit > 0
                ):
                    original_chunk_count = len(chunking_result)
                    chunking_result = enforce_chunk_token_limit_before_embedding(
                        chunking_result=chunking_result,
                        tokenizer=self.tokenizer,
                        max_tokens=self.embedding_token_limit,
                    )
                    if len(chunking_result) != original_chunk_count:
                        logger.info(
                            "Applied hard fallback split before embedding for "
                            f"d-id: {doc_id}, chunks {original_chunk_count} -> {len(chunking_result)} "
                            f"(limit={self.embedding_token_limit})"
                        )
                        # Compact "pre -> post" summary mirrors the log
                        # middle segment.  Field is only present when a
                        # hard split actually occurred, so its presence
                        # alone signals the trigger.
                        extraction_meta["hard_fallback_split"] = (
                            f"{original_chunk_count} -> {len(chunking_result)}"
                        )

                chunks = build_chunks_dict_from_chunking_result(
                    chunking_result, doc_id=doc_id, file_path=file_path
                )

                if not chunks:
                    logger.warning("No document chunks to process")

                processing_start_time = int(time.time())

                await self._raise_if_cancelled(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                )

                # Stage 1: persist doc_status PROCESSING + chunks in parallel.
                doc_status_task = asyncio.create_task(
                    self._upsert_doc_status_transition(
                        doc_id=doc_id,
                        status=DocStatus.PROCESSING,
                        status_doc=status_doc,
                        file_path=file_path,
                        extra_fields={
                            "chunks_count": len(chunks),
                            "chunks_list": list(chunks.keys()),
                        },
                        metadata_extra={
                            "processing_start_time": processing_start_time,
                            **extraction_meta,
                        },
                    )
                )
                chunks_vdb_task = asyncio.create_task(self.chunks_vdb.upsert(chunks))
                text_chunks_task = asyncio.create_task(self.text_chunks.upsert(chunks))
                first_stage_tasks = [
                    doc_status_task,
                    chunks_vdb_task,
                    text_chunks_task,
                ]
                entity_relation_task = None

                await asyncio.gather(*first_stage_tasks)

                # Stage 2: entity/relation extraction (after text_chunks are
                # saved).  When the user opted out via process_options '!',
                # skip extraction entirely; chunks remain in the vector
                # store so naive / mix retrieval still works.
                if doc_process_opts.skip_kg:
                    logger.info(
                        f"[skip_kg] process_options '!' set for d-id: {doc_id}; "
                        f"skipping entity/relation extraction"
                    )
                    chunk_results = []
                    extraction_meta["skip_kg"] = True
                else:
                    entity_relation_task = asyncio.create_task(
                        self._process_extract_entities(
                            chunks,
                            ctx.pipeline_status,
                            ctx.pipeline_status_lock,
                        )
                    )
                    chunk_results = await entity_relation_task
                file_extraction_stage_ok = True

            except Exception as e:
                pending_tasks = first_stage_tasks + (
                    [entity_relation_task] if entity_relation_task else []
                )
                await self._finalize_doc_failure(
                    doc_id=doc_id,
                    status_doc=status_doc,
                    file_path=file_path,
                    error=e,
                    stage_label="extract",
                    current_file_number=current_file_number,
                    total_files=ctx.total_files,
                    failed_chunks_snapshot=get_failed_chunk_snapshot(),
                    pending_tasks=pending_tasks,
                    metadata_extra={
                        "processing_start_time": processing_start_time,
                        "processing_end_time": int(time.time()),
                    },
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                )

            # Concurrency is controlled by keyed lock for individual
            # entities and relationships.
            if file_extraction_stage_ok:
                try:
                    await self._raise_if_cancelled(
                        ctx.pipeline_status, ctx.pipeline_status_lock
                    )

                    # Use chunk_results from entity_relation_task.  When
                    # skip_kg is set, chunk_results is empty so there are no
                    # nodes/edges to merge — but we still need to flush the
                    # chunks_vdb / text_chunks writes (already done above)
                    # and reach PROCESSED.
                    if not doc_process_opts.skip_kg:
                        await merge_nodes_and_edges(
                            chunk_results=chunk_results,
                            knowledge_graph_inst=self.chunk_entity_relation_graph,
                            entity_vdb=self.entities_vdb,
                            relationships_vdb=self.relationships_vdb,
                            global_config=self._build_global_config(),
                            full_entities_storage=self.full_entities,
                            full_relations_storage=self.full_relations,
                            doc_id=doc_id,
                            pipeline_status=ctx.pipeline_status,
                            pipeline_status_lock=ctx.pipeline_status_lock,
                            llm_response_cache=self.llm_response_cache,
                            entity_chunks_storage=self.entity_chunks,
                            relation_chunks_storage=self.relation_chunks,
                            current_file_number=current_file_number,
                            total_files=ctx.total_files,
                            file_path=file_path,
                        )

                    processing_end_time = int(time.time())
                    await self._upsert_doc_status_transition(
                        doc_id=doc_id,
                        status=DocStatus.PROCESSED,
                        status_doc=status_doc,
                        file_path=file_path,
                        extra_fields={
                            "chunks_count": len(chunks),
                            "chunks_list": list(chunks.keys()),
                        },
                        metadata_extra={
                            "processing_start_time": processing_start_time,
                            "processing_end_time": processing_end_time,
                            **extraction_meta,
                        },
                    )

                    await self._insert_done()

                    async with ctx.pipeline_status_lock:
                        log_message = (
                            f"Completed processing file "
                            f"{current_file_number}/{ctx.total_files}: "
                            f"{file_path}"
                        )
                        logger.info(log_message)
                        ctx.pipeline_status["latest_message"] = log_message
                        ctx.pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    await self._finalize_doc_failure(
                        doc_id=doc_id,
                        status_doc=status_doc,
                        file_path=file_path,
                        error=e,
                        stage_label="merge",
                        current_file_number=current_file_number,
                        total_files=ctx.total_files,
                        failed_chunks_snapshot=get_failed_chunk_snapshot(),
                        pending_tasks=[],
                        metadata_extra={
                            "processing_start_time": processing_start_time,
                            "processing_end_time": int(time.time()),
                            **extraction_meta,
                        },
                        pipeline_status=ctx.pipeline_status,
                        pipeline_status_lock=ctx.pipeline_status_lock,
                    )

    async def _purge_stale_extraction_if_resuming(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        content_data: dict[str, Any] | None,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """If the document already has extracted content, purge stale chunks
        and KG contributions before re-running chunking + entity extraction
        under the current ``process_options``.

        Mutates ``status_doc.chunks_list`` / ``chunks_count`` to reflect the
        purge so subsequent state-machine upserts don't write back stale IDs.
        Also emits an engine-mismatch warning when the filename hint disagrees
        with the stored ``parse_engine`` — the extracted content is the source
        of truth, so the user must delete + re-upload to switch engines.
        """
        content_already_extracted = isinstance(content_data, dict) and (
            (
                content_data.get("parse_format") == FULL_DOCS_FORMAT_LIGHTRAG
                and content_data.get("lightrag_document_path")
            )
            or (
                content_data.get("parse_format") == FULL_DOCS_FORMAT_RAW
                and (content_data.get("content") or "").strip()
            )
        )
        if not content_already_extracted:
            return

        intended_engine, _ = resolve_file_parser_directives(file_path)
        stored_engine = (content_data.get("parse_engine") or "").lower()
        if intended_engine and stored_engine and intended_engine != stored_engine:
            log_message = (
                f"[resume] {doc_id}: filename hint / "
                f"LIGHTRAG_PARSER implies engine="
                f"{intended_engine!r} but full_docs "
                f"already has parse_engine="
                f"{stored_engine!r}; keeping the existing "
                f"extraction.  Delete + re-upload to "
                f"switch engines."
            )
            logger.warning(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        stored_chunk_ids = {
            chunk_id
            for chunk_id in (status_doc.chunks_list or [])
            if isinstance(chunk_id, str) and chunk_id
        }
        if not stored_chunk_ids:
            return

        log_message = (
            f"[resume] {doc_id}: purging "
            f"{len(stored_chunk_ids)} chunk(s) and "
            f"associated KG entries from a previous run "
            f"before rebuilding under current "
            f"process_options"
        )
        logger.info(log_message)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
        await self._purge_doc_chunks_and_kg(
            doc_id,
            stored_chunk_ids,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
        )
        # The status_doc carries chunks_list / chunks_count from the prior
        # run; clear them so subsequent state-machine upserts don't write
        # back stale IDs.
        status_doc.chunks_list = []
        status_doc.chunks_count = 0

    # ============================================================
    # doc_status state-machine helpers (shared by all layers)
    # ============================================================

    async def _upsert_doc_status_transition(
        self,
        doc_id: str,
        status: DocStatus,
        status_doc: DocProcessingStatus,
        file_path: str,
        *,
        extra_fields: dict[str, Any] | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        """Single source of truth for doc_status state-transition upserts.

        Mirrors the field set used at every PARSING / ANALYZING / PROCESSING /
        PROCESSED / FAILED transition.  ``extra_fields`` carries
        ``chunks_count`` / ``chunks_list`` / ``error_msg``; ``metadata_extra``
        is forwarded to ``doc_status_transition_metadata`` so carry-over
        fields (e.g. ``process_options``) survive every state change.
        """
        payload: dict[str, Any] = {
            "status": status,
            "content_summary": status_doc.content_summary,
            "content_length": status_doc.content_length,
            "created_at": status_doc.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "file_path": file_path,
            "track_id": status_doc.track_id,
            "content_hash": status_doc.content_hash,
            "metadata": doc_status_transition_metadata(
                status_doc, extra=metadata_extra
            ),
        }
        if extra_fields:
            payload.update(extra_fields)
        await self.doc_status.upsert({doc_id: payload})

    async def _raise_if_cancelled(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Raise ``PipelineCancelledException`` if the user has requested cancel."""
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled")

    async def _finalize_doc_failure(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        error: BaseException,
        stage_label: str,
        current_file_number: int,
        total_files: int,
        failed_chunks_snapshot: tuple[list[str], int],
        pending_tasks: list[asyncio.Task],
        metadata_extra: dict[str, Any],
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Common epilogue for an extract / merge stage failure.

        Logs the error (or cancellation), cancels any pending stage tasks,
        flushes the LLM response cache, and writes a FAILED status row that
        preserves the failed chunks snapshot and processing-time metadata.
        """
        if isinstance(error, PipelineCancelledException):
            if stage_label == "merge":
                error_msg = (
                    f"User cancelled during merge {current_file_number}/"
                    f"{total_files}: {file_path}"
                )
            else:
                error_msg = (
                    f"User cancelled {current_file_number}/"
                    f"{total_files}: {file_path}"
                )
            logger.warning(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
        else:
            logger.error(traceback.format_exc())
            if stage_label == "merge":
                error_msg = (
                    f"Merging stage failed in document "
                    f"{current_file_number}/{total_files}: {file_path}"
                )
            else:
                error_msg = (
                    f"Failed to extract document "
                    f"{current_file_number}/{total_files}: {file_path}"
                )
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(traceback.format_exc())
                pipeline_status["history_messages"].append(error_msg)

        for task in pending_tasks:
            if task and not task.done():
                task.cancel()

        if self.llm_response_cache:
            try:
                await self.llm_response_cache.index_done_callback()
            except Exception as persist_error:
                logger.error(f"Failed to persist LLM cache: {persist_error}")

        failed_chunks_list, failed_chunks_count = failed_chunks_snapshot
        await self._upsert_doc_status_transition(
            doc_id=doc_id,
            status=DocStatus.FAILED,
            status_doc=status_doc,
            file_path=file_path,
            extra_fields={
                "error_msg": str(error),
                "chunks_count": failed_chunks_count,
                "chunks_list": failed_chunks_list,
            },
            metadata_extra=metadata_extra,
        )

    # ============================================================
    # Parser engines (also called by tests directly)
    # ============================================================

    async def parse_native(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Phase 1 parse for native/raw, lightrag and pending_parse formats."""
        doc_format = content_data.get("parse_format", FULL_DOCS_FORMAT_RAW)
        if doc_format == FULL_DOCS_FORMAT_LIGHTRAG:
            # full_docs.content carries the merged text with the {{LRdoc}}
            # marker; strip it so the chunking path is identical to raw.
            # blocks_path is still resolved for downstream multimodal
            # sidecar reads (_build_mm_chunks_from_sidecars).
            merged_text = strip_lightrag_doc_prefix(
                content_data.get("content"), doc_format
            )
            raw_doc_path = content_data.get("lightrag_document_path") or file_path
            resolved_doc_path = resolve_lightrag_document_path(str(raw_doc_path))
            blocks_path = resolve_lightrag_blocks_path(resolved_doc_path) or ""

            return {
                "doc_id": doc_id,
                "file_path": file_path,
                "parse_format": doc_format,
                "content": merged_text,
                "blocks_path": blocks_path,
            }

        if doc_format == FULL_DOCS_FORMAT_PENDING_PARSE:
            source_path = self._resolve_source_file_for_parser(
                str(content_data.get("source_path") or file_path)
            )
            p = Path(source_path)
            if p.exists() and p.is_file() and p.suffix.lower() == ".docx":
                from lightrag.native_parser.docx import (
                    parse_docx_to_lightrag_document,
                )

                file_bytes = await asyncio.to_thread(p.read_bytes)
                parsed_data = await parse_docx_to_lightrag_document(
                    file_bytes=file_bytes,
                    file_path=file_path,
                    doc_id=doc_id,
                    source_path=str(p),
                )

                blocks_path = Path(parsed_data["blocks_path"])
                self._enrich_parsed_sidecars_with_surrounding(
                    str(blocks_path),
                    doc_id=doc_id,
                    file_path=file_path,
                )
                stored_blocks_path = str(blocks_path)
                try:
                    stored_blocks_path = str(blocks_path.relative_to(input_dir_path()))
                except ValueError:
                    pass
                await self._persist_parsed_full_docs(
                    doc_id,
                    {
                        "content": make_lightrag_doc_content(parsed_data["content"]),
                        "file_path": file_path,
                        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                        "lightrag_document_path": stored_blocks_path,
                        "parse_engine": PARSER_ENGINE_NATIVE,
                        "update_time": int(time.time()),
                    },
                )
                await archive_docx_source_after_full_docs_sync(str(p))
                logger.info(
                    f"[parse_native] pending_parse completed for {file_path} "
                    f"via native_parser/docx"
                )
                return parsed_data
            raise ValueError(
                f"Native parser does not support pending file: {file_path}"
            )

        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "content": content_data.get("content", ""),
            "blocks_path": "",
        }

    async def parse_mineru(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        endpoint = os.getenv("MINERU_ENDPOINT", "").strip()
        if not endpoint:
            raise ValueError("MINERU_ENDPOINT is required for MinerU parsing")
        protocol = {
            "upload_url": endpoint,
            "poll_url_template": os.getenv(
                "MINERU_POLL_ENDPOINT",
                endpoint + "/{trace_id}",
            ),
            "poll_method": os.getenv("MINERU_POLL_METHOD", "GET"),
            "id_field": os.getenv("MINERU_ID_FIELD", "trace_id"),
            "status_field": os.getenv("MINERU_STATUS_FIELD", "status"),
            "result_url_field": os.getenv("MINERU_RESULT_URL_FIELD", "result_url"),
            "content_field": os.getenv("MINERU_CONTENT_FIELD", "content"),
            "success_values": os.getenv(
                "MINERU_SUCCESS_VALUES",
                "done,success,succeeded,completed,finished",
            ),
            "failed_values": os.getenv("MINERU_FAILED_VALUES", "failed,error"),
            "poll_interval_seconds": float(
                os.getenv("MINERU_POLL_INTERVAL_SECONDS", "2")
            ),
            "max_polls": int(os.getenv("MINERU_MAX_POLLS", "180")),
        }
        source_file_path = self._resolve_source_file_for_parser(
            str(content_data.get("source_path") or file_path)
        )
        result_text = await self._call_protocol_parse_service(
            protocol=protocol,
            file_path=source_file_path,
        )
        content_list = normalize_parser_result_to_content_list(result_text)
        if content_list:
            return await self._write_lightrag_document_from_content_list(
                doc_id=doc_id,
                file_path=file_path,
                content_list=content_list,
                engine=PARSER_ENGINE_MINERU,
                source_path=source_file_path,
            )
        if not result_text:
            raise ValueError(f"MinerU parser returned empty content for {file_path}")

        await self._persist_parsed_full_docs(
            doc_id,
            {
                "content": str(result_text),
                "file_path": file_path,
                "parse_format": FULL_DOCS_FORMAT_RAW,
                "parse_engine": PARSER_ENGINE_MINERU,
                "update_time": int(time.time()),
            },
        )
        await archive_docx_source_after_full_docs_sync(source_file_path)
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "content": str(result_text),
            "blocks_path": "",
        }

    async def parse_docling(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        endpoint = os.getenv("DOCLING_ENDPOINT", "").strip()
        if not endpoint:
            raise ValueError("DOCLING_ENDPOINT is required for Docling parsing")
        protocol = {
            "upload_url": endpoint,
            "poll_url_template": os.getenv(
                "DOCLING_POLL_ENDPOINT",
                endpoint + "/{task_id}",
            ),
            "poll_method": os.getenv("DOCLING_POLL_METHOD", "GET"),
            "id_field": os.getenv("DOCLING_ID_FIELD", "task_id"),
            "status_field": os.getenv("DOCLING_STATUS_FIELD", "status"),
            "result_url_field": os.getenv("DOCLING_RESULT_URL_FIELD", "result_url"),
            "content_field": os.getenv("DOCLING_CONTENT_FIELD", "content"),
            "success_values": os.getenv(
                "DOCLING_SUCCESS_VALUES",
                "done,success,succeeded,completed,finished",
            ),
            "failed_values": os.getenv("DOCLING_FAILED_VALUES", "failed,error"),
            "poll_interval_seconds": float(
                os.getenv("DOCLING_POLL_INTERVAL_SECONDS", "2")
            ),
            "max_polls": int(os.getenv("DOCLING_MAX_POLLS", "180")),
        }
        source_file_path = self._resolve_source_file_for_parser(
            str(content_data.get("source_path") or file_path)
        )
        result_text = await self._call_protocol_parse_service(
            protocol=protocol,
            file_path=source_file_path,
        )
        content_list = normalize_parser_result_to_content_list(result_text)
        if content_list:
            return await self._write_lightrag_document_from_content_list(
                doc_id=doc_id,
                file_path=file_path,
                content_list=content_list,
                engine=PARSER_ENGINE_DOCLING,
                source_path=source_file_path,
            )
        if not result_text:
            raise ValueError(f"Docling parser returned empty content for {file_path}")

        await self._persist_parsed_full_docs(
            doc_id,
            {
                "content": str(result_text),
                "file_path": file_path,
                "parse_format": FULL_DOCS_FORMAT_RAW,
                "parse_engine": PARSER_ENGINE_DOCLING,
                "update_time": int(time.time()),
            },
        )
        await archive_docx_source_after_full_docs_sync(source_file_path)
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "content": str(result_text),
            "blocks_path": "",
        }

    # ============================================================
    # Parser internals
    # ============================================================

    async def _call_protocol_parse_service(
        self, protocol: dict[str, Any], file_path: str
    ) -> str | None:
        """Protocol-driven async parse call for MinerU/Docling."""
        upload_url = str(protocol.get("upload_url") or "").strip()
        if not upload_url:
            return None
        if httpx is None:
            logger.warning("httpx not installed, skip async parse service call")
            return None

        id_field = str(protocol.get("id_field", "id"))
        status_field = str(protocol.get("status_field", "status"))
        result_url_field = str(protocol.get("result_url_field", "result_url"))
        content_field = str(protocol.get("content_field", "content"))
        poll_url_tpl = str(protocol.get("poll_url_template", "")).strip()
        poll_method = str(protocol.get("poll_method", "GET")).upper()
        poll_interval = float(protocol.get("poll_interval_seconds", 2.0))
        max_polls = int(protocol.get("max_polls", 120))
        success_values = set(
            x.strip().lower()
            for x in str(
                protocol.get(
                    "success_values", "done,success,succeeded,completed,finished"
                )
            ).split(",")
            if x.strip()
        )
        failed_values = set(
            x.strip().lower()
            for x in str(protocol.get("failed_values", "failed,error")).split(",")
            if x.strip()
        )

        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    upload_url, files={"file": (Path(file_path).name, f)}
                )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Parse service upload failed: {resp.status_code} {resp.text[:400]}"
                )
            upload_payload = resp.json() if resp.text else {}
            task_id = get_by_path(upload_payload, id_field)
            if not task_id:
                direct_content = get_by_path(upload_payload, content_field)
                return str(direct_content) if direct_content else None
            task_id = str(task_id)

            poll_url = (
                poll_url_tpl.format(task_id=task_id, trace_id=task_id, id=task_id)
                if poll_url_tpl
                else upload_url
            )
            poll_params = {"task_id": task_id, "trace_id": task_id, "id": task_id}
            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)
                if poll_method == "POST":
                    poll_resp = await client.post(poll_url, json=poll_params)
                else:
                    poll_resp = await client.get(poll_url, params=poll_params)
                poll_payload = poll_resp.json() if poll_resp.text else {}
                status_raw = get_by_path(poll_payload, status_field)
                status_val = str(status_raw).lower() if status_raw is not None else ""

                if status_val in success_values:
                    result_url = get_by_path(poll_payload, result_url_field)
                    if result_url:
                        dl = await client.get(str(result_url))
                        dl.raise_for_status()
                        return dl.text
                    content_val = get_by_path(poll_payload, content_field)
                    return str(content_val) if content_val else None
                if status_val in failed_values:
                    raise RuntimeError(
                        f"Parse service failed for task {task_id}: {poll_payload}"
                    )
        raise TimeoutError(f"Parse service polling timeout for task: {task_id}")

    async def _persist_parsed_full_docs(
        self,
        doc_id: str,
        record: dict[str, Any],
    ) -> str | None:
        """Write a parse-result record to ``full_docs`` and sync ``content_hash``.

        Computes ``content_hash`` from the actual extracted body so subsequent
        ``get_doc_by_content_hash`` lookups can dedupe across pending_parse
        records that did not have a hash at enqueue time. Also patches the
        existing ``doc_status`` row so both storages stay aligned.

        The original ``pending_parse`` record carries metadata seeded at
        enqueue time (``process_options``, ``canonical_basename``,
        ``source_path``, ...) that downstream stages still need after parsing.
        ``full_docs`` upserts overwrite the entire row, so we merge the
        existing record with the new ``record`` payload before upserting:
        fresh fields from ``record`` (``content`` / ``parse_format`` /
        ``lightrag_document_path`` / ``parse_engine`` / ``update_time``)
        take precedence, while pre-existing fields are preserved.
        """
        fmt = record.get("parse_format")
        content_hash: str | None = None
        if fmt == FULL_DOCS_FORMAT_RAW:
            content_hash = compute_text_content_hash(record.get("content") or "")
        elif fmt == FULL_DOCS_FORMAT_LIGHTRAG:
            blocks_path = record.get("lightrag_document_path") or ""
            if blocks_path:
                content_hash = compute_file_content_hash(
                    resolve_lightrag_document_path(blocks_path)
                )

        existing = await self.full_docs.get_by_id(doc_id)
        if isinstance(existing, dict):
            payload = {**existing, **record}
        else:
            payload = dict(record)
        if content_hash:
            payload["content_hash"] = content_hash

        await self.full_docs.upsert({doc_id: payload})
        await self.full_docs.index_done_callback()

        if content_hash:
            existing_status = await self.doc_status.get_by_id(doc_id)
            if existing_status:
                patched = dict(existing_status)
                patched["content_hash"] = content_hash
                patched["updated_at"] = datetime.now(timezone.utc).isoformat()
                await self.doc_status.upsert({doc_id: patched})
        return content_hash

    async def _mark_duplicate_after_parse(
        self,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        content_hash: str | None,
        content_length: int,
        content_data: dict[str, Any] | None = None,
        pipeline_status: dict | None = None,
        pipeline_status_lock: asyncio.Lock | None = None,
    ) -> bool:
        """Mark post-parse content duplicates and stop further processing."""
        if not content_hash:
            return False

        match = await get_duplicate_doc_by_content_hash(
            self.doc_status, content_hash, doc_id
        )
        if not match:
            return False

        original_doc_id, original_doc = match
        original_track_id = doc_status_field(original_doc, "track_id", "")
        original_status = doc_status_field(original_doc, "status", "unknown")
        now = datetime.now(timezone.utc).isoformat()
        message = (
            "Identical content already exists under another filename. "
            f"Original doc_id: {original_doc_id}, Status: {original_status}"
        )

        await self.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": (
                        f"[DUPLICATE:content_hash] Original document: {original_doc_id}"
                    ),
                    "content_length": content_length,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": status_doc.created_at,
                    "updated_at": now,
                    "file_path": file_path,
                    "track_id": status_doc.track_id,
                    "content_hash": content_hash,
                    "error_msg": message,
                    "metadata": doc_status_transition_metadata(
                        status_doc,
                        extra={
                            "is_duplicate": True,
                            "duplicate_kind": "content_hash",
                            "original_doc_id": original_doc_id,
                            "original_track_id": original_track_id,
                        },
                    ),
                }
            }
        )
        try:
            await self.full_docs.delete([doc_id])
            await self.full_docs.index_done_callback()
        except Exception as e:
            logger.warning(f"Failed to remove duplicate full_docs entry {doc_id}: {e}")

        source_path = str((content_data or {}).get("source_path") or file_path)
        archived = await archive_source_after_full_docs_sync(source_path)
        archive_msg = f"; archived to {archived}" if archived else ""
        warning = f"Duplicate content skipped after parsing: {file_path}{archive_msg}"
        logger.warning(warning)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = warning
                pipeline_status["history_messages"].append(warning)
        return True

    def _resolve_source_file_for_parser(self, file_path: str) -> str:
        """Resolve a readable source file path for parser upload."""
        p = Path(file_path)
        if p.exists() and p.is_file():
            return str(p)

        name = p.name
        candidates: list[Path] = []
        input_path = input_dir_path()
        candidates.append(input_path / name)
        candidates.append(input_path / PARSED_DIR_NAME / name)

        # Common local defaults used by API server.
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "inputs" / name,
                cwd / "inputs" / PARSED_DIR_NAME / name,
                cwd / PARSED_DIR_NAME / name,
            ]
        )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return file_path

    def _enrich_parsed_sidecars_with_surrounding(
        self,
        blocks_path: str | None,
        *,
        doc_id: str,
        file_path: str,
    ) -> dict[str, int]:
        """Backfill parse-stage sidecar ``surrounding`` fields.

        This runs immediately after a LightRAG Document writer emits
        ``.blocks.jsonl`` and sidecars, before ``full_docs`` is updated and
        before the VLM analysis stage.  Keeping the enrichment here makes
        the parse artifacts stable regardless of later ``process_options``
        choices.
        """
        path = str(blocks_path or "").strip()
        tokenizer = getattr(self, "tokenizer", None)
        if not path or tokenizer is None:
            return {"drawings": 0, "tables": 0, "equations": 0}
        try:
            from lightrag.multimodal_context import (
                enrich_sidecars_with_surrounding,
            )

            counts = enrich_sidecars_with_surrounding(
                blocks_path=path,
                enabled_modalities={"drawings", "tables", "equations"},
                tokenizer=tokenizer,
            )
            if any(counts.values()):
                logger.info(
                    f"[parse_native] surrounding backfilled for d-id: {doc_id}, "
                    f"file: {file_path}: "
                    + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
                )
            return counts
        except Exception as enrich_err:
            logger.warning(
                f"[parse_native] surrounding enrichment failed for "
                f"d-id: {doc_id}, file: {file_path}: {enrich_err}"
            )
            return {"drawings": 0, "tables": 0, "equations": 0}

    async def _write_lightrag_document_from_content_list(
        self,
        doc_id: str,
        file_path: str,
        content_list: list[dict[str, Any]],
        engine: str,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        """Convert parser content list to LightRAG Document files and return parsed_data."""
        parsed_dir = parsed_artifact_dir_for_source(source_path, file_path)
        if parsed_dir.exists():
            shutil.rmtree(parsed_dir)
        parsed_dir.mkdir(parents=True, exist_ok=True)

        source_name = canonicalize_parser_hinted_basename(file_path) or f"{doc_id}.bin"
        base_name = Path(source_name).stem or source_name
        blocks_path = parsed_dir / f"{base_name}.blocks.jsonl"
        tables_path = parsed_dir / f"{base_name}.tables.json"
        drawings_path = parsed_dir / f"{base_name}.drawings.json"
        equations_path = parsed_dir / f"{base_name}.equations.json"

        blocks_lines: list[str] = []
        merged_parts: list[str] = []
        block_idx = 0
        table_idx = 0
        drawing_idx = 0
        equation_idx = 0

        tables: dict[str, Any] = {}
        drawings: dict[str, Any] = {}
        equations: dict[str, Any] = {}

        def _to_list_str(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(x) for x in value if str(x).strip()]
            text_val = str(value).strip()
            return [text_val] if text_val else []

        def _parse_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except Exception:
                return default

        def _normalize_grid_rows(grid: Any) -> list[list[str]]:
            normalized_rows: list[list[str]] = []
            if not isinstance(grid, list):
                return normalized_rows
            for row in grid:
                if not isinstance(row, list):
                    continue
                normalized_row: list[str] = []
                for cell in row:
                    if isinstance(cell, dict):
                        normalized_row.append(str(cell.get("text", "")).strip())
                    else:
                        normalized_row.append(str(cell).strip())
                normalized_rows.append(normalized_row)
            return normalized_rows

        def _coerce_table_rows(
            value: Any,
        ) -> tuple[str, Any, list[list[str]], int, int]:
            raw_value = value
            if isinstance(raw_value, str):
                stripped = raw_value.strip()
                if not stripped:
                    return "html", "", [], 0, 0
                parsed_value = None
                try:
                    parsed_value = json.loads(stripped)
                except Exception:
                    try:
                        import ast

                        parsed_value = ast.literal_eval(stripped)
                    except Exception:
                        parsed_value = None
                if parsed_value is None:
                    return "html", raw_value, [], 0, 0
                raw_value = parsed_value

            if isinstance(raw_value, list):
                rows = _normalize_grid_rows(raw_value)
                return (
                    "json",
                    json.dumps(rows, ensure_ascii=False),
                    rows,
                    len(rows),
                    max((len(r) for r in rows), default=0),
                )

            if isinstance(raw_value, dict):
                rows = _normalize_grid_rows(raw_value.get("grid"))
                if not rows and isinstance(raw_value.get("rows"), list):
                    rows = _normalize_grid_rows(raw_value.get("rows"))
                num_rows = _parse_int(
                    raw_value.get("num_rows"), len(rows) if rows else 0
                )
                num_cols = _parse_int(
                    raw_value.get("num_cols"),
                    max((len(r) for r in rows), default=0),
                )
                if rows:
                    return (
                        "json",
                        json.dumps(rows, ensure_ascii=False),
                        rows,
                        num_rows,
                        num_cols,
                    )
                return (
                    "html",
                    json.dumps(raw_value, ensure_ascii=False),
                    [],
                    num_rows,
                    num_cols,
                )

            text_value = str(raw_value or "").strip()
            return "html", text_value, [], 0, 0

        heading_stack: list[str] = []

        def _update_heading_context(
            heading_text: str, level: int
        ) -> tuple[str, int, list[str]]:
            nonlocal heading_stack
            clean_heading = str(heading_text or "").strip()
            clean_level = max(_parse_int(level, 1), 1)
            heading_stack = heading_stack[: max(clean_level - 1, 0)]
            parent_chain = [x for x in heading_stack if x]
            heading_stack.append(clean_heading)
            return clean_heading, clean_level, parent_chain

        def _append_block(
            content_text: str,
            heading: str = "",
            level: int = 0,
            parent_headings: list[str] | None = None,
        ) -> str:
            nonlocal block_idx
            content_text = str(content_text or "").strip()
            if not content_text:
                return ""
            blockid = hashlib.md5(
                f"{doc_id}:{block_idx}:{heading}:{content_text}".encode("utf-8")
            ).hexdigest()
            blocks_lines.append(
                json.dumps(
                    {
                        "type": "content",
                        "blockid": blockid,
                        "format": "plain_text",
                        "content": content_text,
                        "heading": heading,
                        "parent_headings": list(parent_headings or []),
                        "level": level,
                        "session_type": "body",
                        "table_slice": "none",
                        "positions": [],
                    },
                    ensure_ascii=False,
                )
            )
            merged_parts.append(content_text)
            block_idx += 1
            return blockid

        current_heading = ""
        current_level = 0
        current_parent_headings: list[str] = []

        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or item.get("label") or "").lower()

            if item_type in {"text", "title", "section_header", "list", "code"}:
                text = (
                    item.get("text")
                    or item.get("content")
                    or "\n".join(
                        item.get("list_items", [])
                        if isinstance(item.get("list_items"), list)
                        else []
                    )
                    or item.get("code_body")
                    or ""
                )
                if not str(text).strip():
                    continue
                inferred_level = int(item.get("text_level", 0) or 0)
                if item_type in {"title", "section_header"} and inferred_level <= 0:
                    inferred_level = int(item.get("level", 1) or 1)
                if inferred_level > 0:
                    (
                        current_heading,
                        current_level,
                        current_parent_headings,
                    ) = _update_heading_context(str(text), inferred_level)
                _append_block(
                    str(text),
                    heading=current_heading,
                    level=current_level,
                    parent_headings=current_parent_headings,
                )
                continue

            if item_type == "equation":
                equation_idx += 1
                eq_id = str(item.get("id") or f"eq-{doc_id}-{equation_idx:04d}")
                caption = str(item.get("caption") or f"公式{equation_idx}")
                footnotes = _to_list_str(
                    item.get("equation_footnote") or item.get("footnotes")
                )
                eq_text = str(item.get("text") or item.get("content") or "").strip()
                wrapped = (
                    f'<equation id="{eq_id}" format="latex" caption="{caption}">{eq_text}</equation>'
                    if eq_text
                    else f'<cite type="equation" refid="{eq_id}">公式{equation_idx}</cite>'
                )
                blockid = _append_block(
                    wrapped,
                    heading=current_heading,
                    level=current_level,
                    parent_headings=current_parent_headings,
                )
                equations[eq_id] = {
                    "id": eq_id,
                    "blockid": blockid,
                    "heading": current_heading,
                    "format": "latex",
                    "content": eq_text,
                    "index": equation_idx - 1,
                    "caption": caption,
                    "footnotes": footnotes,
                }
                continue

            if item_type == "table":
                table_idx += 1
                table_id = str(item.get("id") or f"tb-{doc_id}-{table_idx:04d}")
                caption = str(item.get("caption") or f"表格{table_idx}")
                table_caption = _to_list_str(item.get("table_caption"))
                if table_caption and not item.get("caption"):
                    caption = table_caption[0]
                footnotes = _to_list_str(
                    item.get("table_footnote") or item.get("footnotes")
                )
                table_body = item.get("table_body") or item.get("content") or ""
                rows = item.get("rows") if isinstance(item.get("rows"), list) else None
                (
                    fmt,
                    table_content,
                    normalized_rows,
                    inferred_num_rows,
                    inferred_num_cols,
                ) = _coerce_table_rows(rows if rows is not None else table_body)
                rows = normalized_rows or (rows if isinstance(rows, list) else [])
                cite_text = (
                    f'<cite type="table" refid="{table_id}">表{table_idx}</cite>'
                )
                blockid = _append_block(
                    cite_text,
                    heading=current_heading,
                    level=current_level,
                    parent_headings=current_parent_headings,
                )
                tables[table_id] = {
                    "id": table_id,
                    "blockid": blockid,
                    "heading": current_heading,
                    "dimension": [
                        _parse_int(item.get("num_rows"), inferred_num_rows),
                        _parse_int(item.get("num_cols"), inferred_num_cols),
                    ],
                    "format": fmt,
                    "content": table_content,
                    "index": table_idx - 1,
                    "caption": caption,
                    "footnotes": footnotes,
                    "image": item.get("img_path") or item.get("image"),
                }
                continue

            if item_type in {"image", "picture", "drawing"}:
                drawing_idx += 1
                drawing_id = str(item.get("id") or f"dr-{doc_id}-{drawing_idx:04d}")
                image_caption = _to_list_str(
                    item.get("image_caption") or item.get("captions")
                )
                caption = str(
                    item.get("caption")
                    or (image_caption[0] if image_caption else f"图{drawing_idx}")
                )
                footnotes = _to_list_str(
                    item.get("image_footnote") or item.get("footnotes")
                )
                path_val = str(item.get("img_path") or item.get("path") or "")
                src_val = str(item.get("src") or "")
                fmt = (
                    Path(path_val).suffix.lower().lstrip(".")
                    if path_val
                    else str(item.get("format") or "")
                )
                drawing_tag = (
                    f'<drawing id="{drawing_id}" format="{fmt}" caption="{caption}" '
                    f'path="{path_val}" src="{src_val}" />'
                )
                blockid = _append_block(
                    drawing_tag,
                    heading=current_heading,
                    level=current_level,
                    parent_headings=current_parent_headings,
                )
                drawings[drawing_id] = {
                    "id": drawing_id,
                    "blockid": blockid,
                    "heading": current_heading,
                    "format": fmt,
                    "path": path_val,
                    "src": src_val,
                    "caption": caption,
                    "footnotes": footnotes,
                }
                continue

            # Fallback: serialize unknown item to text for robustness.
            fallback_text = str(item.get("text") or item.get("content") or "").strip()
            if fallback_text:
                _append_block(
                    fallback_text,
                    heading=current_heading,
                    level=current_level,
                    parent_headings=current_parent_headings,
                )

        merged_text = "\n\n".join([x for x in merged_parts if x.strip()])
        doc_hash = hashlib.sha256(merged_text.encode("utf-8")).hexdigest()
        parse_time = datetime.now(timezone.utc).isoformat()
        meta = {
            "type": "meta",
            "format": "lightrag",
            "version": "1.0",
            "document_name": source_name,
            "document_format": Path(source_name).suffix.lower().lstrip("."),
            "document_hash": f"sha256:{doc_hash}",
            "table_file": bool(tables),
            "equation_file": bool(equations),
            "drawing_file": bool(drawings),
            "asset_dir": False,
            "split_option": {},
            "blocks": len(blocks_lines),
            "doc_id": doc_id,
            "parse_engine": engine,
            "parse_time": parse_time,
            "doc_title": Path(source_name).stem or source_name,
        }
        blocks_path.write_text(
            "\n".join([json.dumps(meta, ensure_ascii=False)] + blocks_lines) + "\n",
            encoding="utf-8",
        )

        if tables:
            tables_path.write_text(
                json.dumps(
                    {"version": "1.0", "tables": tables}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        if drawings:
            drawings_path.write_text(
                json.dumps(
                    {"version": "1.0", "drawings": drawings},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if equations:
            equations_path.write_text(
                json.dumps(
                    {"version": "1.0", "equations": equations},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        self._enrich_parsed_sidecars_with_surrounding(
            str(blocks_path),
            doc_id=doc_id,
            file_path=file_path,
        )

        # Keep full_docs in sync so restart/reprocess can directly use LightRAG Document.
        stored_blocks_path = str(blocks_path)
        try:
            stored_blocks_path = str(blocks_path.relative_to(input_dir_path()))
        except ValueError:
            pass
        await self._persist_parsed_full_docs(
            doc_id,
            {
                "content": make_lightrag_doc_content(merged_text),
                "file_path": file_path,
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "lightrag_document_path": stored_blocks_path,
                "parse_engine": engine,
                "update_time": int(time.time()),
            },
        )
        await archive_docx_source_after_full_docs_sync(
            source_path or self._resolve_source_file_for_parser(file_path)
        )
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
            "content": merged_text,
            "blocks_path": str(blocks_path),
        }

    # ============================================================
    # Multimodal / VLM
    # ============================================================

    async def analyze_multimodal(
        self,
        doc_id: str,
        file_path: str,
        parsed_data: dict[str, Any],
        *,
        process_options: str | None = None,
    ) -> dict[str, Any]:
        """Phase 2: Multimodal analysis (VLM). Writes llm_analyze_result to LightRAG Document.

        Per-document ``i`` / ``t`` / ``e`` flags from
        ``full_docs.process_options`` decide which modalities are sent to the
        VLM.  Sidecars are always written by the parser regardless of these
        flags so toggling options later does not require re-parsing — only
        the ``llm_analyze_result`` payload is gated here.

        Idempotent by design: per-item ``llm_analyze_result`` already
        present is not re-computed.  This lets users incrementally enable
        new modalities (e.g. add ``t`` after a prior ``i``-only pass) and
        re-trigger analysis without redundant VLM calls or losing prior
        results.

        Args:
            process_options: Optional override that bypasses the
                ``full_docs.process_options`` lookup; primarily used by unit
                tests that exercise the VLM analysis path without going
                through the enqueue pipeline.
        """
        from lightrag.parser_routing import parse_process_options

        blocks_path = parsed_data.get("blocks_path")
        if not blocks_path:
            return parsed_data

        block_file = Path(blocks_path)
        if not block_file.exists():
            return parsed_data

        # Resolve which modalities the user opted into for this document.
        if process_options is None:
            try:
                content_data = await self.full_docs.get_by_id(doc_id) or {}
            except Exception:
                content_data = {}
            options_str = (
                content_data.get("process_options")
                if isinstance(content_data, dict)
                else ""
            ) or ""
        else:
            options_str = process_options
        process_opts = parse_process_options(options_str)
        if not (process_opts.images or process_opts.tables or process_opts.equations):
            logger.debug(
                f"[analyze_multimodal] no i/t/e options set for d-id: {doc_id}; "
                f"skipping VLM analysis"
            )
            return parsed_data

        # Diagnose opt-in vs sidecar mismatch up-front so users investigating
        # "why did VLM not run on my images" see a one-line INFO per document
        # instead of silent skips.  Empty sidecars are a normal outcome
        # (some documents simply have no images/tables/equations), so this is
        # informational rather than a warning.
        sidecar_base = str(block_file)
        if sidecar_base.endswith(".blocks.jsonl"):
            sidecar_base = sidecar_base[: -len(".blocks.jsonl")]
        opt_in_missing: list[str] = []
        for opt_char, modality, suffix in (
            ("i", "drawings", ".drawings.json"),
            ("t", "tables", ".tables.json"),
            ("e", "equations", ".equations.json"),
        ):
            enabled = {
                "i": process_opts.images,
                "t": process_opts.tables,
                "e": process_opts.equations,
            }[opt_char]
            if enabled and not Path(sidecar_base + suffix).exists():
                opt_in_missing.append(f"{opt_char}:{modality}")
        if opt_in_missing:
            logger.info(
                f"[analyze_multimodal] process_options opted into "
                f"{','.join(opt_in_missing)} for d-id: {doc_id} (file={file_path}), "
                f"but the parser produced no such sidecar; VLM analysis skipped "
                f"for those modalities."
            )

        try:
            lines = block_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                return parsed_data
            meta = json.loads(lines[0])
            if not isinstance(meta, dict) or meta.get("type") != "meta":
                return parsed_data

            from lightrag.llm._vision_utils import (
                image_audit_metadata,
                image_cache_metadata,
                normalize_image_inputs,
                read_image_dimensions,
            )
            from lightrag.prompt_multimodal import (
                IMAGE_TYPE_ENUM,
                IMAGE_TYPE_FALLBACK,
                MULTIMODAL_PROMPTS,
            )
            from lightrag.constants import (
                DEFAULT_MM_ANALYSIS_PRIORITY,
                DEFAULT_MM_IMAGE_MIN_PIXEL,
                DEFAULT_SUMMARY_LANGUAGE,
            )

            global_config = self._build_global_config()
            addon_params = global_config.get("addon_params") or {}
            language = (
                global_config.get("_resolved_summary_language")
                or addon_params.get("language")
                or DEFAULT_SUMMARY_LANGUAGE
            )
            vlm_process_enable = bool(global_config.get("vlm_process_enable", False))
            max_image_bytes = max(
                256 * 1024,
                int(os.getenv("VLM_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))),
            )
            min_image_pixel = max(
                1,
                int(os.getenv("VLM_MIN_IMAGE_PIXEL", str(DEFAULT_MM_IMAGE_MIN_PIXEL))),
            )

            use_vlm_func = self.role_llm_funcs.get("vlm")
            use_extract_func = self.role_llm_funcs.get("extract")
            vlm_cache_identity = get_llm_cache_identity(global_config, role="vlm")
            extract_cache_identity = get_llm_cache_identity(
                global_config, role="extract"
            )

            _IMAGE_TYPE_VALUES = set(IMAGE_TYPE_ENUM)
            _VLM_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

            def _json_extract(text: str) -> dict[str, Any]:
                """Tolerant JSON object recovery.

                Mirrors :func:`lightrag.operate._process_json_extraction_result`
                so weaker models that emit ```json ... ``` fenced output,
                trailing commas, or unquoted keys are still salvageable.
                The order of attempts is:

                1. Strip a leading ```json fence if present.
                2. Hand the cleaned string to ``json_repair.loads`` (handles
                   minor structural slips like trailing commas).
                3. Fall back to a greedy ``{...}`` regex slice for outputs
                   that wrap the JSON object in prose, then re-run
                   ``json_repair.loads`` on the slice.
                """
                if not text:
                    return {}
                candidate = text.strip()
                fence_match = re.match(
                    r"^```(?:json)?\s*\n(.*?)\n```$",
                    candidate,
                    re.DOTALL | re.IGNORECASE,
                )
                if fence_match:
                    candidate = fence_match.group(1).strip()
                try:
                    obj = json_repair.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
                m = re.search(r"\{[\s\S]*\}", candidate)
                if m:
                    try:
                        obj = json_repair.loads(m.group(0))
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                return {}

            def _normalize_text(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value.strip()
                if isinstance(value, (list, tuple)):
                    return "\n".join(str(v).strip() for v in value if str(v).strip())
                return str(value).strip()

            def _captions_value(item_obj: dict[str, Any]) -> str:
                return _normalize_text(item_obj.get("caption")) or "n/a"

            def _footnotes_value(item_obj: dict[str, Any]) -> str:
                raw = item_obj.get("footnotes")
                if isinstance(raw, (list, tuple)):
                    joined = "; ".join(str(v).strip() for v in raw if str(v).strip())
                    return joined or "n/a"
                text = _normalize_text(raw)
                return text or "n/a"

            def _surrounding_value(item_obj: dict[str, Any], key: str) -> str:
                surrounding = item_obj.get("surrounding") or {}
                if not isinstance(surrounding, dict):
                    return "n/a"
                value = _normalize_text(surrounding.get(key))
                return value or "n/a"

            def _resolve_image_path(
                path_str: str | None, sidecar_dir: Path
            ) -> Path | None:
                if not path_str:
                    return None
                candidate = Path(path_str)
                if not candidate.is_absolute():
                    sidecar_candidate = sidecar_dir / path_str
                    if sidecar_candidate.exists() and sidecar_candidate.is_file():
                        candidate = sidecar_candidate
                if candidate.exists() and candidate.is_file():
                    return candidate
                return None

            def _failure_result(message: str) -> dict[str, Any]:
                return {
                    "analyze_time": int(time.time()),
                    "status": "failure",
                    "message": message,
                }

            def _skipped_result(message: str) -> dict[str, Any]:
                return {
                    "analyze_time": int(time.time()),
                    "status": "skipped",
                    "message": message,
                }

            async def _analyze_drawing(
                item_id: str, item: dict[str, Any], sidecar_dir: Path
            ) -> tuple[dict[str, Any], str | None]:
                path_str = (
                    item.get("path") or item.get("img_path") or item.get("image_path")
                )
                candidate = _resolve_image_path(path_str, sidecar_dir)
                if candidate is None:
                    return (
                        _skipped_result(f"image file not found: {path_str or 'n/a'}"),
                        None,
                    )
                ext = candidate.suffix.lower()
                if ext not in _VLM_RASTER_EXTS:
                    return (
                        _skipped_result(f"unsupported image format: {ext}"),
                        None,
                    )
                dims = read_image_dimensions(candidate)
                if dims is not None and (
                    dims[0] < min_image_pixel or dims[1] < min_image_pixel
                ):
                    return (
                        _skipped_result(
                            f"image width or height is smaller than "
                            f"{min_image_pixel}px"
                        ),
                        None,
                    )
                if not vlm_process_enable or use_vlm_func is None:
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: VLM analysis required but "
                        "VLM role is not available "
                        "(VLM_PROCESS_ENABLE or vlm role config)"
                    )
                try:
                    raw = candidate.read_bytes()
                except OSError as exc:
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: cannot read image {candidate}: {exc}"
                    ) from exc
                if not raw:
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: image file is empty"
                    )
                if len(raw) > max_image_bytes:
                    return (
                        _skipped_result(
                            f"image too large: {len(raw)} bytes "
                            f"(limit {max_image_bytes})"
                        ),
                        None,
                    )
                mime, _ = mimetypes.guess_type(str(candidate))
                mime = mime or "image/png"
                img_payload = {
                    "base64": base64.b64encode(raw).decode("ascii"),
                    "mime_type": mime,
                    "source_id": item_id,
                    "source_file": str(candidate),
                    "modality": "image",
                    "doc_id": doc_id,
                }
                normalized_images = normalize_image_inputs([img_payload])
                prompt = MULTIMODAL_PROMPTS["image_analysis"].format(
                    language=language,
                    content="",
                    captions=_captions_value(item),
                    footnotes=_footnotes_value(item),
                    leading=_surrounding_value(item, "leading"),
                    trailing=_surrounding_value(item, "trailing"),
                    item_id=item_id,
                    file_path=file_path,
                )
                args_hash = compute_args_hash(
                    prompt,
                    "",
                    "",
                    serialize_llm_cache_identity(vlm_cache_identity),
                    _serialize_cache_variant({"type": "json_object"}),
                    _serialize_cache_variant(image_cache_metadata(normalized_images)),
                    "drawing",
                )
                cache_id = generate_cache_key("default", "analysis", args_hash)
                cached = await handle_cache(
                    self.llm_response_cache,
                    args_hash,
                    prompt,
                    mode="default",
                    cache_type="analysis",
                )
                if cached is not None:
                    result_text = cached[0]
                    fresh = False
                else:
                    try:
                        result_text = await use_vlm_func(
                            prompt,
                            stream=False,
                            image_inputs=[img_payload],
                            _priority=DEFAULT_MM_ANALYSIS_PRIORITY,
                        )
                    except Exception as exc:
                        raise MultimodalAnalysisError(
                            f"drawings/{item_id}: VLM call failed: {exc}"
                        ) from exc
                    fresh = True
                parsed = _json_extract(str(result_text))
                name = parsed.get("name")
                type_value = parsed.get("type")
                description = parsed.get("description")
                if not isinstance(name, str) or not name.strip():
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: missing or invalid field 'name'"
                    )
                if not isinstance(description, str) or not description.strip():
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: missing or invalid field 'description'"
                    )
                if not isinstance(type_value, str) or not type_value.strip():
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: missing or invalid field 'type'"
                    )
                if type_value not in _IMAGE_TYPE_VALUES:
                    type_value = IMAGE_TYPE_FALLBACK
                if fresh:
                    audit_blob = image_audit_metadata(normalized_images)
                    original_prompt = prompt + (
                        f"\n<vlm_images>"
                        f"{json.dumps(audit_blob, ensure_ascii=False)}"
                        "</vlm_images>"
                        if audit_blob
                        else ""
                    )
                    await save_to_cache(
                        self.llm_response_cache,
                        CacheData(
                            args_hash=args_hash,
                            content=str(result_text),
                            prompt=original_prompt,
                            mode="default",
                            cache_type="analysis",
                            chunk_id=None,
                        ),
                    )
                return (
                    {
                        "name": name.strip(),
                        "type": type_value,
                        "description": description.strip(),
                        "analyze_time": int(time.time()),
                        "status": "success",
                        "message": "",
                    },
                    cache_id,
                )

            async def _analyze_text_modality(
                kind: str, item_id: str, item: dict[str, Any]
            ) -> tuple[dict[str, Any], str | None]:
                if use_extract_func is None:
                    raise MultimodalAnalysisError(
                        f"{kind}/{item_id}: EXTRACT role is required but not configured"
                    )
                content_text = _normalize_text(item.get("content"))
                if not content_text:
                    raise MultimodalAnalysisError(
                        f"{kind}/{item_id}: missing {kind} content"
                    )
                template = MULTIMODAL_PROMPTS[f"{kind}_analysis"]
                prompt = template.format(
                    language=language,
                    content=content_text,
                    captions=_captions_value(item),
                    footnotes=_footnotes_value(item),
                    leading=_surrounding_value(item, "leading"),
                    trailing=_surrounding_value(item, "trailing"),
                    item_id=item_id,
                    file_path=file_path,
                )
                args_hash = compute_args_hash(
                    prompt,
                    "",
                    "",
                    serialize_llm_cache_identity(extract_cache_identity),
                    _serialize_cache_variant({"type": "json_object"}),
                    _serialize_cache_variant([]),
                    kind,
                )
                cache_id = generate_cache_key("default", "analysis", args_hash)
                cached = await handle_cache(
                    self.llm_response_cache,
                    args_hash,
                    prompt,
                    mode="default",
                    cache_type="analysis",
                )
                if cached is not None:
                    result_text = cached[0]
                    fresh = False
                else:
                    try:
                        result_text = await use_extract_func(
                            prompt,
                            stream=False,
                            response_format={"type": "json_object"},
                            _priority=DEFAULT_MM_ANALYSIS_PRIORITY,
                        )
                    except Exception as exc:
                        raise MultimodalAnalysisError(
                            f"{kind}/{item_id}: EXTRACT call failed: {exc}"
                        ) from exc
                    fresh = True
                parsed = _json_extract(str(result_text))
                name = parsed.get("name")
                description = parsed.get("description")
                if not isinstance(name, str) or not name.strip():
                    raise MultimodalAnalysisError(
                        f"{kind}/{item_id}: missing or invalid field 'name'"
                    )
                if not isinstance(description, str) or not description.strip():
                    raise MultimodalAnalysisError(
                        f"{kind}/{item_id}: missing or invalid field 'description'"
                    )
                result_obj: dict[str, Any] = {
                    "name": name.strip(),
                    "description": description.strip(),
                    "analyze_time": int(time.time()),
                    "status": "success",
                    "message": "",
                }
                if kind == "equation":
                    equation_value = parsed.get("equation")
                    if (
                        not isinstance(equation_value, str)
                        or not equation_value.strip()
                    ):
                        raise MultimodalAnalysisError(
                            f"equation/{item_id}: missing or invalid field 'equation'"
                        )
                    result_obj["equation"] = equation_value.strip()
                if fresh:
                    await save_to_cache(
                        self.llm_response_cache,
                        CacheData(
                            args_hash=args_hash,
                            content=str(result_text),
                            prompt=prompt,
                            mode="default",
                            cache_type="analysis",
                            chunk_id=None,
                        ),
                    )
                return (result_obj, cache_id)

            def _attach_cache_id(
                item_obj: dict[str, Any], cache_id: str | None
            ) -> None:
                if not cache_id:
                    return
                existing = item_obj.get("llm_cache_list")
                if not isinstance(existing, list):
                    existing = []
                if cache_id not in existing:
                    existing.append(cache_id)
                item_obj["llm_cache_list"] = existing

            base_name = str(block_file)
            if base_name.endswith(".blocks.jsonl"):
                base_name = base_name[: -len(".blocks.jsonl")]
            sidecars = [
                (
                    Path(base_name + ".drawings.json"),
                    "drawings",
                    "drawing",
                    process_opts.images,
                ),
                (
                    Path(base_name + ".tables.json"),
                    "tables",
                    "table",
                    process_opts.tables,
                ),
                (
                    Path(base_name + ".equations.json"),
                    "equations",
                    "equation",
                    process_opts.equations,
                ),
            ]
            for sidecar_path, root_key, kind, enabled in sidecars:
                if not enabled or not sidecar_path.exists():
                    continue
                try:
                    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise MultimodalAnalysisError(
                        f"failed to read sidecar {sidecar_path}: {exc}"
                    ) from exc
                items = payload.get(root_key, {})
                if not isinstance(items, dict):
                    continue

                valid_keys: list[str] = []
                analyze_tasks: list[Any] = []
                skipped_existing = 0
                for item_id, item in items.items():
                    if not isinstance(item, dict):
                        continue
                    existing = item.get("llm_analyze_result")
                    if isinstance(existing, dict):
                        existing_status = existing.get("status")
                        if existing_status in ("success", "skipped"):
                            skipped_existing += 1
                            continue
                        if existing_status == "failure":
                            raise MultimodalAnalysisError(
                                f"{root_key}/{item_id}: prior llm_analyze_result "
                                f"is failure ({existing.get('message') or 'no message'})"
                            )
                        # Unknown legacy status falls through to re-analysis.
                    valid_keys.append(item_id)
                    if kind == "drawing":
                        analyze_tasks.append(
                            _analyze_drawing(item_id, item, sidecar_path.parent)
                        )
                    else:
                        analyze_tasks.append(
                            _analyze_text_modality(kind, item_id, item)
                        )
                if skipped_existing:
                    logger.debug(
                        f"[analyze_multimodal] {root_key}: "
                        f"{skipped_existing} item(s) already analyzed, skipping; "
                        f"{len(analyze_tasks)} item(s) to analyze"
                    )

                analyzed = await asyncio.gather(*analyze_tasks, return_exceptions=True)

                failure_to_raise: MultimodalAnalysisError | None = None
                for idx, item_id in enumerate(valid_keys):
                    item = items.get(item_id)
                    if not isinstance(item, dict):
                        continue
                    outcome = analyzed[idx]
                    if isinstance(outcome, MultimodalAnalysisError):
                        item["llm_analyze_result"] = _failure_result(str(outcome))
                        if failure_to_raise is None:
                            failure_to_raise = outcome
                        continue
                    if isinstance(outcome, Exception):
                        item["llm_analyze_result"] = _failure_result(
                            f"unexpected error: {outcome}"
                        )
                        if failure_to_raise is None:
                            failure_to_raise = MultimodalAnalysisError(
                                f"{root_key}/{item_id}: unexpected error: {outcome}"
                            )
                        continue
                    result_obj, cache_id = outcome
                    item["llm_analyze_result"] = result_obj
                    _attach_cache_id(item, cache_id)
                try:
                    sidecar_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except OSError as exc:
                    logger.warning(
                        f"[analyze_multimodal] failed to write sidecar "
                        f"{sidecar_path}: {exc}"
                    )
                if failure_to_raise is not None:
                    raise failure_to_raise

            parsed_data["multimodal_processed"] = True
            logger.info(
                f"[analyze_multimodal] completed for d-id: {doc_id}, file: {file_path}"
            )
        except MultimodalAnalysisError:
            raise
        except Exception as e:
            logger.warning(f"[analyze_multimodal] failed for d-id: {doc_id}: {e}")
        return parsed_data

    async def _run_multimodal_postprocess_hook(
        self,
        doc_id: str,
        file_path: str,
        chunking_result: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        extraction_meta: dict[str, Any],
    ) -> list[dict[str, Any]] | tuple[dict[str, Any], ...]:
        """Multimodal post-process entrypoint.

        Placement:
            chunking -> [this hook] -> entity extraction

        Default behavior is no-op. This method defines a stable extension point
        for built-in multimodal processors.

        Activates when the per-document ``process_options`` opts into at least
        one of ``i`` / ``t`` / ``e``.  Per-modality work in subsequent steps
        (``_build_mm_chunks_from_sidecars``, ``analyze_multimodal``) decides
        whether to act based on whether ``drawings.json`` / ``tables.json`` /
        ``equations.json`` actually exist on disk — the parser declares
        modality availability by writing those sidecars, not by listing
        capabilities in meta.
        """
        from lightrag.parser_routing import parse_process_options

        parse_format = extraction_meta.get("parse_format")
        if parse_format != FULL_DOCS_FORMAT_LIGHTRAG:
            return chunking_result

        try:
            content_data = await self.full_docs.get_by_id(doc_id) or {}
        except Exception:
            content_data = {}
        process_opts = parse_process_options(
            content_data.get("process_options")
            if isinstance(content_data, dict)
            else ""
        )
        active = {
            ch
            for ch, enabled in (
                ("i", process_opts.images),
                ("t", process_opts.tables),
                ("e", process_opts.equations),
            )
            if enabled
        }
        if not active:
            return chunking_result

        logger.info(
            f"[multimodal-hook] enabled for d-id: {doc_id}, file: {file_path}, "
            f"parse_engine={extraction_meta.get('parse_engine')}, opts={sorted(active)}"
        )

        # TODO(multimodal pipeline):
        # 1) call modal processors using vlm_llm_model_func (VLM role)
        # 2) merge multimodal outputs back into chunk dicts
        # 3) keep chunk_order_index continuity and chunk_id stability
        return chunking_result

    def _build_mm_chunks_from_sidecars(
        self,
        doc_id: str,
        file_path: str,
        blocks_path: str,
        base_order_index: int,
        process_options: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build multimodal chunks from sidecars carrying analysis results.

        Only items whose ``llm_analyze_result.status == "success"`` produce
        chunks.  ``"skipped"`` items are silently ignored; ``"failure"``
        items raise :class:`MultimodalAnalysisError` so the document is
        marked failed (a failure should already have aborted the analyze
        phase — this is a defensive recheck).

        Each chunk follows the new schema: nested ``heading`` and
        ``sidecar`` dicts, no flat ``parent_headings`` / ``level`` /
        ``content_type`` fields.  ``llm_cache_list`` is merged from the
        underlying sidecar item so document deletion can clean up the
        ``cache_type="analysis"`` entries it created.

        ``process_options`` gates which modality sidecars are read: a
        document re-processed after opting out of ``i`` / ``t`` / ``e``
        must NOT pick up stale success results from a prior pass.  When
        ``None`` (e.g. ad-hoc unit tests), every modality is considered.

        Raises:
            MultimodalAnalysisError: when an item carries ``status="failure"``,
                or when the multimodal chunk cannot be fit under the
                extraction token budget even after truncating description
                to :data:`DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS`.
        """
        from lightrag.constants import (
            DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
            DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS,
        )
        from lightrag.parser_routing import parse_process_options

        block_file = Path(blocks_path)
        if not block_file.exists():
            return []

        base = str(block_file)
        if base.endswith(".blocks.jsonl"):
            base = base[: -len(".blocks.jsonl")]

        if process_options is None:
            allowed = {"drawing", "table", "equation"}
        else:
            opts = parse_process_options(process_options)
            allowed = set()
            if opts.images:
                allowed.add("drawing")
            if opts.tables:
                allowed.add("table")
            if opts.equations:
                allowed.add("equation")

        sidecar_defs = [
            (root, Path(base + suffix), kind)
            for root, suffix, kind in (
                ("drawings", ".drawings.json", "drawing"),
                ("tables", ".tables.json", "table"),
                ("equations", ".equations.json", "equation"),
            )
            if kind in allowed
        ]

        mm_chunks: list[dict[str, Any]] = []
        order = base_order_index

        def _norm_str_list(v: Any) -> list[str]:
            if v is None:
                return []
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            return [s] if s else []

        def _norm_parent_headings(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(p).strip() for p in value if str(p or "").strip()]

        def _build_heading_dict(item: dict[str, Any]) -> dict[str, Any] | None:
            heading_raw = item.get("heading")
            if isinstance(heading_raw, dict):
                heading_text = str(heading_raw.get("heading") or "").strip()
                parents = _norm_parent_headings(heading_raw.get("parent_headings"))
                try:
                    level = int(heading_raw.get("level") or 0)
                except (TypeError, ValueError):
                    level = 0
            else:
                heading_text = str(heading_raw or "").strip()
                parents = _norm_parent_headings(item.get("parent_headings"))
                try:
                    level = int(item.get("level") or 0)
                except (TypeError, ValueError):
                    level = 0
            if not heading_text and not parents and level == 0:
                return None
            return {
                "level": level,
                "heading": heading_text,
                "parent_headings": parents,
            }

        def _render(
            kind: str,
            name: str,
            description: str,
            footnotes_joined: str,
            equation_body: str,
        ) -> str:
            sections: list[str] = []
            if kind == "drawing":
                sections.append(f"- Image Name:\n{name}")
                # type field comes from caller via name interpolation: omit
                # when blank because the prompt enforces it as required so a
                # missing type should already have raised.
                # The image type was inserted at call-site by the caller via a
                # secondary call to this helper if needed.
                if description:
                    sections.append(f"- Image Description:\n{description}")
                if footnotes_joined:
                    sections.append(f"- Image Footnotes:\n{footnotes_joined}")
            elif kind == "table":
                sections.append(f"- Table Name:\n{name}")
                if description:
                    sections.append(f"- Table Description:\n{description}")
                if footnotes_joined:
                    sections.append(f"- Table Footnotes:\n{footnotes_joined}")
            else:
                sections.append(f"- Equation Name:\n{name}")
                if equation_body:
                    sections.append(f"- Equation Body:\n{equation_body}")
                if description:
                    sections.append(f"- Equation Description:\n{description}")
                if footnotes_joined:
                    sections.append(f"- Equation Footnotes:\n{footnotes_joined}")
            return "\n\n".join(sections).strip()

        def _render_image(
            name: str,
            image_type: str,
            description: str,
            footnotes_joined: str,
        ) -> str:
            sections = [
                f"- Image Name:\n{name}",
                f"- Image Type:\n{image_type}",
            ]
            if description:
                sections.append(f"- Image Description:\n{description}")
            if footnotes_joined:
                sections.append(f"- Image Footnotes:\n{footnotes_joined}")
            return "\n\n".join(sections).strip()

        max_tokens = DEFAULT_MAX_EXTRACT_INPUT_TOKENS
        min_desc_tokens = DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS

        for root_key, sidecar_path, kind in sidecar_defs:
            if not sidecar_path.exists():
                continue
            try:
                payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items = payload.get(root_key, {})
            if not isinstance(items, dict):
                continue

            for local_idx, (item_id, item) in enumerate(items.items()):
                if not isinstance(item, dict):
                    continue

                analysis = item.get("llm_analyze_result")
                if not isinstance(analysis, dict):
                    continue
                status = analysis.get("status")
                if status == "skipped":
                    continue
                if status == "failure":
                    raise MultimodalAnalysisError(
                        f"{root_key}/{item_id}: llm_analyze_result.status='failure' "
                        f"({analysis.get('message') or 'no message'})"
                    )
                if status != "success":
                    # Treat unknown / legacy status as missing — no chunk.
                    continue

                name = str(analysis.get("name") or "").strip()
                description = str(analysis.get("description") or "").strip()
                equation_body = str(analysis.get("equation") or "").strip()
                image_type = str(analysis.get("type") or "").strip()
                if not name:
                    raise MultimodalAnalysisError(
                        f"{root_key}/{item_id}: success result missing 'name'"
                    )
                if not description:
                    raise MultimodalAnalysisError(
                        f"{root_key}/{item_id}: success result missing 'description'"
                    )
                if kind == "drawing" and not image_type:
                    raise MultimodalAnalysisError(
                        f"drawings/{item_id}: success result missing 'type'"
                    )
                if kind == "equation" and not equation_body:
                    raise MultimodalAnalysisError(
                        f"equations/{item_id}: success result missing 'equation'"
                    )

                footnotes_list = _norm_str_list(item.get("footnotes"))
                footnotes_joined = "; ".join(footnotes_list)

                def _compose(desc: str) -> str:
                    if kind == "drawing":
                        return _render_image(
                            name=name,
                            image_type=image_type,
                            description=desc,
                            footnotes_joined=footnotes_joined,
                        )
                    return _render(
                        kind=kind,
                        name=name,
                        description=desc,
                        footnotes_joined=footnotes_joined,
                        equation_body=equation_body,
                    )

                chunk_content = _compose(description)
                tokens = len(self.tokenizer.encode(chunk_content))
                if tokens > max_tokens:
                    # Truncate only the description, never name/type/equation.
                    desc_tokens = self.tokenizer.encode(description)
                    overflow = tokens - max_tokens
                    keep = max(min_desc_tokens, len(desc_tokens) - overflow)
                    while True:
                        truncated_desc = self.tokenizer.decode(desc_tokens[:keep])
                        chunk_content = _compose(truncated_desc)
                        tokens = len(self.tokenizer.encode(chunk_content))
                        if tokens <= max_tokens or keep <= min_desc_tokens:
                            break
                        keep = max(min_desc_tokens, keep - (tokens - max_tokens))
                    if tokens > max_tokens:
                        raise MultimodalAnalysisError(
                            f"{root_key}/{item_id}: multimodal chunk exceeds "
                            f"{max_tokens} tokens even after truncating description "
                            f"to {min_desc_tokens} tokens"
                        )

                if not chunk_content:
                    continue

                heading_dict = _build_heading_dict(item)
                sidecar_block = {
                    "type": kind,
                    "id": str(item_id),
                    "refs": [{"type": kind, "id": str(item_id)}],
                }
                cache_list = item.get("llm_cache_list")
                cache_list = (
                    [str(c) for c in cache_list if str(c).strip()]
                    if isinstance(cache_list, list)
                    else []
                )

                chunk_dict: dict[str, Any] = {
                    "chunk_id": f"{doc_id}-mm-{kind}-{local_idx:03d}",
                    "chunk_order_index": order,
                    "content": chunk_content,
                    "tokens": tokens,
                    "sidecar": sidecar_block,
                    "llm_cache_list": cache_list,
                }
                if heading_dict is not None:
                    chunk_dict["heading"] = heading_dict
                mm_chunks.append(chunk_dict)
                order += 1

        return mm_chunks
