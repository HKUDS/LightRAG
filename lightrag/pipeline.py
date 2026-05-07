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
import mimetypes
import os
import re
import shutil
import time
import traceback
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
from lightrag.exceptions import PipelineCancelledException
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.operate import merge_nodes_and_edges
from lightrag.parser_routing import (
    canonicalize_parser_hinted_basename,
    resolve_file_parser_directives,
    resolve_stored_document_parser_engine,
)
from lightrag.utils import (
    compute_mdhash_id,
    enforce_chunk_token_limit_before_embedding,
    generate_track_id,
    get_content_summary,
    logger,
    sanitize_text_for_encoding,
)
from lightrag.utils_pipeline import (
    archive_docx_source_after_full_docs_sync,
    archive_source_after_full_docs_sync,
    augment_chunk_results_with_mm_entities,
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


class _PipelineMixin:
    """Mixin providing document ingestion pipeline methods for LightRAG.

    Designed to be combined as a base of LightRAG only.  Relies on
    LightRAG-provided attributes (``self.full_docs``, ``self.doc_status``,
    ``self.tokenizer``, ``self.parser_*``, ``self.workspace`` ...) and on the
    shared methods ``self._insert_done`` / ``self._process_extract_entities``
    which remain in the main class and are resolved through MRO.
    """

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
            process_options: per-document processing options string (i/t/e/!/F/R/S);
                accepted as a single string broadcast to every input or as a list
                aligned with ``input``. Stored verbatim on ``full_docs`` and
                mirrored to ``doc_status.metadata['process_options']``.
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

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
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

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Check if another process is already processing the queue
        # Statuses the pipeline considers "in-flight or pending"; used by
        # both the initial snapshot and every refetch after a
        # request_pending continuation.
        _processing_statuses = [
            DocStatus.PROCESSING,
            DocStatus.FAILED,
            DocStatus.PENDING,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
        ]

        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                to_process_docs: dict[
                    str, DocProcessingStatus
                ] = await self.doc_status.get_docs_by_statuses(_processing_statuses)

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

        async def _try_atomic_release() -> bool:
            """Thin wrapper that updates the local
            ``busy_released_in_loop`` flag based on the result of
            ``_atomic_release_busy_or_consume_pending``.
            """
            nonlocal busy_released_in_loop
            released = await self._atomic_release_busy_or_consume_pending(
                pipeline_status, pipeline_status_lock
            )
            if released:
                busy_released_in_loop = True
            return released

        try:
            # Process documents until no more documents or requests
            while True:
                # Check for cancellation request at the start of main loop
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        # Clear pending request
                        pipeline_status["request_pending"] = False
                        # Celar cancellation flag
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
                    if await _try_atomic_release():
                        break
                    to_process_docs = await self.doc_status.get_docs_by_statuses(
                        _processing_statuses
                    )
                    continue

                # Validate document data consistency and fix any issues as part of the pipeline
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
                    if await _try_atomic_release():
                        break
                    to_process_docs = await self.doc_status.get_docs_by_statuses(
                        _processing_statuses
                    )
                    continue

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                # Update pipeline_status, batchs now represents the total number of files to be processed
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Three cascading layers of queues:
                # Layer 1: Content Parsing - parse_native/parse_mineru/parse_docling
                # Layer 2: Multimoldal Ananlyze - analyze_multimodal
                # Layer 3: Entity and Relation Extraction - process_document

                # Content Parsing Queues
                q_native: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size_default)
                q_mineru: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size_default)
                q_docling: asyncio.Queue = asyncio.Queue(
                    maxsize=self.queue_size_default
                )
                # Multimoldal Anlyaze Queue
                q_analyze: asyncio.Queue = asyncio.Queue(
                    maxsize=self.queue_size_default
                )
                # Entity and Relation Extraction Queue
                q_process: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size_insert)

                workers: list[asyncio.Task] = []

                # Get first document's file path and total count for job name
                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path

                # Handle cases where first_doc_path is None
                if first_doc_path:
                    path_prefix = first_doc_path[:20] + (
                        "..." if len(first_doc_path) > 20 else ""
                    )
                else:
                    path_prefix = "unknown_source"

                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                # Create a counter to track the number of processed files
                processed_count = 0
                # Create a semaphore to limit the number of concurrent file processing
                semaphore = asyncio.Semaphore(self.max_parallel_insert)

                async def process_document(
                    doc_id: str,
                    status_doc: DocProcessingStatus,
                    split_by_character: str | None,
                    split_by_character_only: bool,
                    pipeline_status: dict,
                    pipeline_status_lock: asyncio.Lock,
                    semaphore: asyncio.Semaphore,
                    pre_parsed_data: dict[str, Any] | None = None,
                ) -> None:
                    """Process single document"""
                    # Initialize variables at the start to prevent UnboundLocalError in error handling
                    file_path = resolve_doc_file_path(status_doc=status_doc)
                    current_file_number = 0
                    file_extraction_stage_ok = False
                    processing_start_time = int(time.time())
                    first_stage_tasks = []
                    entity_relation_task = None
                    chunks: dict[str, Any] = {}
                    content_data: dict[str, Any] | None = None

                    def get_failed_chunk_snapshot() -> tuple[list[str], int]:
                        if chunks:
                            chunk_ids = list(chunks.keys())
                            return chunk_ids, len(chunk_ids)
                        return chunk_fields_from_status_doc(status_doc)

                    async with semaphore:
                        nonlocal processed_count
                        # Initialize to prevent UnboundLocalError in error handling
                        first_stage_tasks = []
                        entity_relation_task = None
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
                            async with pipeline_status_lock:
                                if pipeline_status.get("cancellation_requested", False):
                                    raise PipelineCancelledException("User cancelled")

                            async with pipeline_status_lock:
                                # Update processed file count and save current file number
                                processed_count += 1
                                current_file_number = (
                                    processed_count  # Save the current file number
                                )
                                pipeline_status["cur_batch"] = processed_count

                                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["history_messages"].append(log_message)
                                log_message = f"Processing d-id: {doc_id}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                                # Prevent memory growth: keep only latest 5000 messages when exceeding 10000
                                if len(pipeline_status["history_messages"]) > 10000:
                                    logger.info(
                                        f"Trimming pipeline history from {len(pipeline_status['history_messages'])} to 5000 messages"
                                    )
                                    # Trim in place so Manager.list-backed shared state
                                    # remains appendable and visible across processes.
                                    del pipeline_status["history_messages"][:-5000]

                            if pre_parsed_data is None:
                                # ---- Phase 1: PARSING ----
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PARSING,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,
                                            "content_hash": status_doc.content_hash,
                                            "metadata": doc_status_transition_metadata(
                                                status_doc
                                            ),
                                        }
                                    }
                                )

                                if not content_data:
                                    raise Exception(
                                        f"Document content not found in full_docs for doc_id: {doc_id}"
                                    )

                                parse_engine = resolve_stored_document_parser_engine(
                                    file_path=file_path, content_data=content_data
                                )
                                if parse_engine == "mineru":
                                    parsed_data = await self.parse_mineru(
                                        doc_id, file_path, content_data
                                    )
                                elif parse_engine == "docling":
                                    parsed_data = await self.parse_docling(
                                        doc_id, file_path, content_data
                                    )
                                else:
                                    parsed_data = await self.parse_native(
                                        doc_id, file_path, content_data
                                    )

                                content = parsed_data.get("content", "")

                                # parse_* may have patched doc_status with the
                                # content_hash that was missing for pending_parse.
                                # Refresh the in-memory dataclass so subsequent
                                # state-machine upserts preserve it.
                                refreshed_status = await self.doc_status.get_by_id(
                                    doc_id
                                )
                                if refreshed_status:
                                    refreshed_hash = (
                                        refreshed_status.get("content_hash")
                                        if isinstance(refreshed_status, dict)
                                        else getattr(
                                            refreshed_status, "content_hash", None
                                        )
                                    )
                                    if refreshed_hash:
                                        status_doc.content_hash = refreshed_hash

                                if await self._mark_duplicate_after_parse(
                                    doc_id=doc_id,
                                    status_doc=status_doc,
                                    file_path=file_path,
                                    content_hash=status_doc.content_hash,
                                    content_length=len(content),
                                    content_data=content_data,
                                    pipeline_status=pipeline_status,
                                    pipeline_status_lock=pipeline_status_lock,
                                ):
                                    return

                                # ---- Phase 2: ANALYZING ----
                                # Refresh content_summary / content_length from
                                # the parsed body so pending_parse → lightrag /
                                # raw documents (whose summary was empty and
                                # length was 0 at enqueue) end up with real
                                # values that propagate through every later
                                # state transition.  We mirror the values onto
                                # the in-memory status_doc dataclass — same
                                # pattern as the content_hash refresh above —
                                # so PROCESSING / PROCESSED upserts (which
                                # read from status_doc.content_summary /
                                # status_doc.content_length) preserve them.
                                refreshed_summary = get_content_summary(content)
                                refreshed_length = len(content)
                                status_doc.content_summary = refreshed_summary
                                status_doc.content_length = refreshed_length
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.ANALYZING,
                                            "content_summary": refreshed_summary,
                                            "content_length": refreshed_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,
                                            "content_hash": status_doc.content_hash,
                                            "metadata": doc_status_transition_metadata(
                                                status_doc
                                            ),
                                        }
                                    }
                                )
                                parsed_data = await self.analyze_multimodal(
                                    doc_id=doc_id,
                                    file_path=file_path,
                                    parsed_data=parsed_data,
                                )
                            else:
                                parsed_data = pre_parsed_data
                                content = parsed_data.get("content", "")

                            extraction_meta: dict[str, Any] = {}

                            # Decode per-document processing options once; later
                            # stages (multimodal hook / KG extraction) re-read
                            # them from full_docs as well.
                            from lightrag.parser_routing import (
                                parse_process_options,
                            )

                            doc_process_opts = parse_process_options(
                                (content_data or {}).get("process_options", "")
                            )

                            # ---- Resume guard ----
                            # When the pipeline picks up a non-fresh document whose
                            # content has already been extracted into full_docs, we
                            # must purge any stale chunks / entities / relations
                            # from a previous interrupted attempt BEFORE re-running
                            # chunking + entity extraction under the *current*
                            # process_options.  Skipping this would either leave
                            # orphaned chunk-IDs in the vector DB or mix old and
                            # new chunks together, neither of which is safe.
                            #
                            # Both pipeline entry points (worker-driven and inline)
                            # converge here, so this is the single canonical place
                            # to do the purge regardless of which path got us here.
                            content_already_extracted = isinstance(
                                content_data, dict
                            ) and (
                                (
                                    content_data.get("parse_format")
                                    == FULL_DOCS_FORMAT_LIGHTRAG
                                    and content_data.get("lightrag_document_path")
                                )
                                or (
                                    content_data.get("parse_format")
                                    == FULL_DOCS_FORMAT_RAW
                                    and (content_data.get("content") or "").strip()
                                )
                            )
                            stored_chunk_ids = set(
                                chunk_id
                                for chunk_id in (status_doc.chunks_list or [])
                                if isinstance(chunk_id, str) and chunk_id
                            )
                            if content_already_extracted:
                                # Engine-mismatch warning: changing the parser engine
                                # after extraction is *not* honoured — the extracted
                                # content is the source of truth.  Users wanting to
                                # re-extract with a new engine must delete +
                                # re-upload.
                                intended_engine, _ = resolve_file_parser_directives(
                                    file_path
                                )
                                stored_engine = (
                                    content_data.get("parse_engine") or ""
                                ).lower()
                                if (
                                    intended_engine
                                    and stored_engine
                                    and intended_engine != stored_engine
                                ):
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
                                        pipeline_status["history_messages"].append(
                                            log_message
                                        )

                                if stored_chunk_ids:
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
                                        pipeline_status["history_messages"].append(
                                            log_message
                                        )
                                    await self._purge_doc_chunks_and_kg(
                                        doc_id,
                                        stored_chunk_ids,
                                        pipeline_status=pipeline_status,
                                        pipeline_status_lock=pipeline_status_lock,
                                    )
                                    # The status_doc carries chunks_list / chunks_count
                                    # from the prior run; clear them so subsequent
                                    # state-machine upserts don't accidentally
                                    # re-write stale IDs.
                                    status_doc.chunks_list = []
                                    status_doc.chunks_count = 0

                            # F-chunking only — R/S strategies are deferred.
                            # ``content`` here is always the bare body —
                            # parse_native is the canonical place that strips
                            # the {{LRdoc}} marker for lightrag, and raw /
                            # pending-parse / mineru-fallback / docling-fallback
                            # paths return ``content_data["content"]`` verbatim,
                            # so a raw document whose literal text starts with
                            # ``{{LRdoc}}`` keeps that prefix intact.  Stripping
                            # again here would corrupt that case.
                            if doc_process_opts.chunking != "F":
                                logger.warning(
                                    f"[chunking] process_options chunking="
                                    f"{doc_process_opts.chunking!r} requested for d-id: "
                                    f"{doc_id}, file: {file_path}, but R/S strategies are "
                                    f"not yet implemented; falling back to fixed chunking "
                                    f"('F')."
                                )

                            chunking_result = self.chunking_func(
                                self.tokenizer,
                                content,
                                split_by_character,
                                split_by_character_only,
                                self.chunk_overlap_token_size,
                                self.chunk_token_size,
                            )
                            if inspect.isawaitable(chunking_result):
                                chunking_result = await chunking_result

                            if not isinstance(chunking_result, (list, tuple)):
                                raise TypeError(
                                    f"chunking_func must return a list or tuple of dicts, "
                                    f"got {type(chunking_result)}"
                                )

                            # Reflect the format actually persisted in
                            # full_docs.  Previously a structured-parse
                            # fallback always tagged parse_format=raw, which
                            # silently disabled _run_multimodal_postprocess_hook
                            # for lightrag documents (it gates on parse_format).
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
                                    "fixed_token_fallback"
                                    if doc_process_opts.chunking != "F"
                                    else "fixed_token"
                                ),
                            }

                            # Multimodal post-process hook entrypoint:
                            # runs after chunking and before entity extraction.
                            chunking_result = (
                                await self._run_multimodal_postprocess_hook(
                                    doc_id=doc_id,
                                    file_path=file_path,
                                    chunking_result=chunking_result,
                                    extraction_meta=extraction_meta,
                                )
                            )

                            mm_specs: list[dict[str, Any]] = []
                            blocks_path = str(
                                parsed_data.get("blocks_path") or ""
                            ).strip()
                            if blocks_path:
                                max_order = -1
                                for ch in chunking_result:
                                    if isinstance(ch, dict) and isinstance(
                                        ch.get("chunk_order_index"), int
                                    ):
                                        max_order = max(
                                            max_order, int(ch["chunk_order_index"])
                                        )
                                mm_chunks, mm_specs = (
                                    self._build_mm_chunks_from_sidecars(
                                        doc_id=doc_id,
                                        file_path=file_path,
                                        blocks_path=blocks_path,
                                        base_order_index=max_order + 1,
                                    )
                                )
                                if mm_chunks:
                                    chunking_result = list(chunking_result) + mm_chunks
                                    extraction_meta["mm_chunks"] = len(mm_chunks)

                            # Final hard guard before embedding:
                            # split any oversize chunk while preserving heading hierarchy metadata.
                            if (
                                self.embedding_token_limit is not None
                                and self.embedding_token_limit > 0
                            ):
                                original_chunk_count = len(chunking_result)
                                chunking_result = (
                                    enforce_chunk_token_limit_before_embedding(
                                        chunking_result=chunking_result,
                                        tokenizer=self.tokenizer,
                                        max_tokens=self.embedding_token_limit,
                                    )
                                )
                                if len(chunking_result) != original_chunk_count:
                                    logger.info(
                                        "Applied hard fallback split before embedding for "
                                        f"d-id: {doc_id}, chunks {original_chunk_count} -> {len(chunking_result)} "
                                        f"(limit={self.embedding_token_limit})"
                                    )
                                    extraction_meta["hard_fallback_split"] = True
                                    extraction_meta["pre_split_chunks"] = (
                                        original_chunk_count
                                    )
                                    extraction_meta["post_split_chunks"] = len(
                                        chunking_result
                                    )

                            # Build chunks dictionary
                            chunks: dict[str, Any] = {}
                            for dp in chunking_result:
                                chunk_content = dp.get("content", "")
                                if not chunk_content:
                                    continue
                                raw_chunk_id = dp.get("chunk_id", "")
                                order = dp.get("chunk_order_index")
                                if (
                                    isinstance(raw_chunk_id, str)
                                    and raw_chunk_id.strip()
                                ):
                                    if raw_chunk_id.startswith(f"{doc_id}-"):
                                        chunk_key = raw_chunk_id
                                    else:
                                        chunk_key = f"{doc_id}-{raw_chunk_id}"
                                elif isinstance(order, int):
                                    chunk_key = f"{doc_id}-chunk-{order:03d}"
                                else:
                                    chunk_key = compute_mdhash_id(
                                        f"{doc_id}:{chunk_content}", prefix="chunk-"
                                    )

                                # Hard collision guard (same chunk_id inside one document).
                                if chunk_key in chunks:
                                    chunk_key = compute_mdhash_id(
                                        f"{doc_id}:{order}:{chunk_content}",
                                        prefix="chunk-",
                                    )
                                chunks[chunk_key] = {
                                    **dp,
                                    "full_doc_id": doc_id,
                                    "file_path": file_path,
                                    "llm_cache_list": [],
                                }

                            if not chunks:
                                logger.warning("No document chunks to process")

                            # Record processing start time
                            processing_start_time = int(time.time())

                            # Check for cancellation before entity extraction
                            async with pipeline_status_lock:
                                if pipeline_status.get("cancellation_requested", False):
                                    raise PipelineCancelledException("User cancelled")

                            # Process document in two stages
                            # Stage 1: Process text chunks and docs (parallel execution)
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSING,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(
                                                chunks.keys()
                                            ),  # Save chunks list
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "content_hash": status_doc.content_hash,
                                            "metadata": doc_status_transition_metadata(
                                                status_doc,
                                                extra={
                                                    "processing_start_time": processing_start_time,
                                                    **extraction_meta,
                                                },
                                            ),
                                        }
                                    }
                                )
                            )
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks)
                            )
                            text_chunks_task = asyncio.create_task(
                                self.text_chunks.upsert(chunks)
                            )

                            # First stage tasks (parallel execution)
                            first_stage_tasks = [
                                doc_status_task,
                                chunks_vdb_task,
                                text_chunks_task,
                            ]
                            entity_relation_task = None

                            # Execute first stage tasks
                            await asyncio.gather(*first_stage_tasks)

                            # Stage 2: Process entity relation graph (after text_chunks are saved).
                            # When the user opted out via process_options '!', skip
                            # entity/relation extraction entirely; chunks remain in
                            # the vector store so naive / mix retrieval still works.
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
                                        chunks, pipeline_status, pipeline_status_lock
                                    )
                                )
                                chunk_results = await entity_relation_task
                                chunk_results = augment_chunk_results_with_mm_entities(
                                    chunk_results=chunk_results,
                                    mm_specs=mm_specs,
                                    file_path=file_path,
                                )
                            file_extraction_stage_ok = True

                        except Exception as e:
                            # Check if this is a user cancellation
                            if isinstance(e, PipelineCancelledException):
                                # User cancellation - log brief message only, no traceback
                                error_msg = f"User cancelled {current_file_number}/{total_files}: {file_path}"
                                logger.warning(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        error_msg
                                    )
                            else:
                                # Other exceptions - log with traceback
                                logger.error(traceback.format_exc())
                                error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                logger.error(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        traceback.format_exc()
                                    )
                                    pipeline_status["history_messages"].append(
                                        error_msg
                                    )

                            # Cancel tasks that are not yet completed
                            all_tasks = first_stage_tasks + (
                                [entity_relation_task] if entity_relation_task else []
                            )
                            for task in all_tasks:
                                if task and not task.done():
                                    task.cancel()

                            # Persistent llm cache with error handling
                            if self.llm_response_cache:
                                try:
                                    await self.llm_response_cache.index_done_callback()
                                except Exception as persist_error:
                                    logger.error(
                                        f"Failed to persist LLM cache: {persist_error}"
                                    )

                            # Record processing end time for failed case
                            processing_end_time = int(time.time())
                            failed_chunks_list, failed_chunks_count = (
                                get_failed_chunk_snapshot()
                            )

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error_msg": str(e),
                                        "chunks_count": failed_chunks_count,
                                        "chunks_list": failed_chunks_list,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                        "track_id": status_doc.track_id,  # Preserve existing track_id
                                        "content_hash": status_doc.content_hash,
                                        "metadata": doc_status_transition_metadata(
                                            status_doc,
                                            extra={
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        ),
                                    }
                                }
                            )

                        # Concurrency is controlled by keyed lock for individual entities and relationships
                        if file_extraction_stage_ok:
                            try:
                                # Check for cancellation before merge
                                async with pipeline_status_lock:
                                    if pipeline_status.get(
                                        "cancellation_requested", False
                                    ):
                                        raise PipelineCancelledException(
                                            "User cancelled"
                                        )

                                # Use chunk_results from entity_relation_task.
                                # When skip_kg is set, chunk_results is empty so
                                # there are no nodes/edges to merge — but we
                                # still need to flush the chunks_vdb / text_chunks
                                # writes (already done above) and reach PROCESSED.
                                if not doc_process_opts.skip_kg:
                                    await merge_nodes_and_edges(
                                        chunk_results=chunk_results,  # result collected from entity_relation_task
                                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                                        entity_vdb=self.entities_vdb,
                                        relationships_vdb=self.relationships_vdb,
                                        global_config=self._build_global_config(),
                                        full_entities_storage=self.full_entities,
                                        full_relations_storage=self.full_relations,
                                        doc_id=doc_id,
                                        pipeline_status=pipeline_status,
                                        pipeline_status_lock=pipeline_status_lock,
                                        llm_response_cache=self.llm_response_cache,
                                        entity_chunks_storage=self.entity_chunks,
                                        relation_chunks_storage=self.relation_chunks,
                                        current_file_number=current_file_number,
                                        total_files=total_files,
                                        file_path=file_path,
                                    )

                                # Record processing end time
                                processing_end_time = int(time.time())

                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSED,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(chunks.keys()),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "content_hash": status_doc.content_hash,
                                            "metadata": doc_status_transition_metadata(
                                                status_doc,
                                                extra={
                                                    "processing_start_time": processing_start_time,
                                                    "processing_end_time": processing_end_time,
                                                    **extraction_meta,
                                                },
                                            ),
                                        }
                                    }
                                )

                                # Call _insert_done after processing each file
                                await self._insert_done()

                                async with pipeline_status_lock:
                                    log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                    logger.info(log_message)
                                    pipeline_status["latest_message"] = log_message
                                    pipeline_status["history_messages"].append(
                                        log_message
                                    )

                            except Exception as e:
                                # Check if this is a user cancellation
                                if isinstance(e, PipelineCancelledException):
                                    # User cancellation - log brief message only, no traceback
                                    error_msg = f"User cancelled during merge {current_file_number}/{total_files}: {file_path}"
                                    logger.warning(error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )
                                else:
                                    # Other exceptions - log with traceback
                                    logger.error(traceback.format_exc())
                                    error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                                    logger.error(error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            traceback.format_exc()
                                        )
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )

                                # Persistent llm cache with error handling
                                if self.llm_response_cache:
                                    try:
                                        await self.llm_response_cache.index_done_callback()
                                    except Exception as persist_error:
                                        logger.error(
                                            f"Failed to persist LLM cache: {persist_error}"
                                        )

                                # Record processing end time for failed case
                                processing_end_time = int(time.time())
                                failed_chunks_list, failed_chunks_count = (
                                    get_failed_chunk_snapshot()
                                )

                                # Update document status to failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error_msg": str(e),
                                            "chunks_count": failed_chunks_count,
                                            "chunks_list": failed_chunks_list,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "content_hash": status_doc.content_hash,
                                            "metadata": doc_status_transition_metadata(
                                                status_doc,
                                                extra={
                                                    "processing_start_time": processing_start_time,
                                                    "processing_end_time": processing_end_time,
                                                    **extraction_meta,
                                                },
                                            ),
                                        }
                                    }
                                )

                async def parse_worker(engine: str, in_q: asyncio.Queue):
                    while True:
                        item = await in_q.get()
                        try:
                            doc_id_w, status_doc_w = item
                            file_path_w = getattr(
                                status_doc_w, "file_path", "unknown_source"
                            )
                            content_data_w = await self.full_docs.get_by_id(doc_id_w)
                            if not content_data_w:
                                raise Exception(
                                    f"Document content not found in full_docs for doc_id: {doc_id_w}"
                                )
                            await self.doc_status.upsert(
                                {
                                    doc_id_w: {
                                        "status": DocStatus.PARSING,
                                        "content_summary": status_doc_w.content_summary,
                                        "content_length": status_doc_w.content_length,
                                        "created_at": status_doc_w.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path_w,
                                        "track_id": status_doc_w.track_id,
                                        "content_hash": status_doc_w.content_hash,
                                        "metadata": doc_status_transition_metadata(
                                            status_doc_w
                                        ),
                                    }
                                }
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
                                pipeline_status=pipeline_status,
                                pipeline_status_lock=pipeline_status_lock,
                            ):
                                continue

                            await q_analyze.put((doc_id_w, status_doc_w, parsed_data_w))
                        except Exception as e:
                            logger.error(f"Parse worker failed ({engine}): {e}")
                            try:
                                await self.doc_status.upsert(
                                    {
                                        doc_id_w: {
                                            "status": DocStatus.FAILED,
                                            "error_msg": str(e),
                                            "content_summary": status_doc_w.content_summary,
                                            "content_length": status_doc_w.content_length,
                                            "created_at": status_doc_w.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": getattr(
                                                status_doc_w,
                                                "file_path",
                                                "unknown_source",
                                            ),
                                            "track_id": status_doc_w.track_id,
                                            "content_hash": status_doc_w.content_hash,
                                            "metadata": doc_status_transition_metadata(
                                                status_doc_w
                                            ),
                                        }
                                    }
                                )
                            except Exception:
                                pass
                        finally:
                            in_q.task_done()

                async def analyze_worker():
                    while True:
                        item = await q_analyze.get()
                        try:
                            doc_id_w, status_doc_w, parsed_data_w = item
                            file_path_w = getattr(
                                status_doc_w, "file_path", "unknown_source"
                            )
                            # Refresh content_summary / content_length from
                            # the parsed body so pending_parse → lightrag /
                            # raw documents (which start with empty summary
                            # and zero length at enqueue) end up with real
                            # values that propagate through every later
                            # state transition.  Mirrors the values onto the
                            # in-memory status_doc_w dataclass so PROCESSING /
                            # PROCESSED upserts (which read from
                            # status_doc_w.content_summary /
                            # status_doc_w.content_length) preserve them.
                            refreshed_content_w = parsed_data_w.get("content", "") or ""
                            refreshed_summary_w = get_content_summary(
                                refreshed_content_w
                            )
                            refreshed_length_w = len(refreshed_content_w)
                            status_doc_w.content_summary = refreshed_summary_w
                            status_doc_w.content_length = refreshed_length_w
                            await self.doc_status.upsert(
                                {
                                    doc_id_w: {
                                        "status": DocStatus.ANALYZING,
                                        "content_summary": refreshed_summary_w,
                                        "content_length": refreshed_length_w,
                                        "created_at": status_doc_w.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path_w,
                                        "track_id": status_doc_w.track_id,
                                        "content_hash": status_doc_w.content_hash,
                                        "metadata": doc_status_transition_metadata(
                                            status_doc_w
                                        ),
                                    }
                                }
                            )
                            analyzed = await self.analyze_multimodal(
                                doc_id=doc_id_w,
                                file_path=file_path_w,
                                parsed_data=parsed_data_w,
                            )
                            await q_process.put((doc_id_w, status_doc_w, analyzed))
                        except Exception as e:
                            logger.error(f"Analyze worker failed: {e}")
                        finally:
                            q_analyze.task_done()

                async def process_worker():
                    while True:
                        item = await q_process.get()
                        try:
                            doc_id_w, status_doc_w, parsed_data_w = item
                            await process_document(
                                doc_id_w,
                                status_doc_w,
                                split_by_character,
                                split_by_character_only,
                                pipeline_status,
                                pipeline_status_lock,
                                semaphore,
                                pre_parsed_data=parsed_data_w,
                            )
                        finally:
                            q_process.task_done()

                # Create workers for each queue of pipeline layer
                for _ in range(max(1, self.max_parallel_parse_native)):
                    workers.append(
                        asyncio.create_task(parse_worker("native", q_native))
                    )
                for _ in range(max(1, self.max_parallel_parse_mineru)):
                    workers.append(
                        asyncio.create_task(parse_worker("mineru", q_mineru))
                    )
                for _ in range(max(1, self.max_parallel_parse_docling)):
                    workers.append(
                        asyncio.create_task(parse_worker("docling", q_docling))
                    )
                for _ in range(max(1, self.max_parallel_analyze)):
                    workers.append(asyncio.create_task(analyze_worker()))
                for _ in range(max(1, self.max_parallel_insert)):
                    workers.append(asyncio.create_task(process_worker()))

                # Add pending files to the correct parsing queue
                for doc_id, status_doc in to_process_docs.items():
                    content_data = await self.full_docs.get_by_id(doc_id) or {}
                    engine = resolve_stored_document_parser_engine(
                        file_path=getattr(status_doc, "file_path", "unknown_source"),
                        content_data=content_data,
                    )
                    if engine == "mineru":
                        await q_mineru.put((doc_id, status_doc))
                    elif engine == "docling":
                        await q_docling.put((doc_id, status_doc))
                    else:
                        await q_native.put((doc_id, status_doc))

                await asyncio.gather(q_native.join(), q_mineru.join(), q_docling.join())
                await q_analyze.join()
                await q_process.join()

                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

                # Atomic exit handoff: if request_pending was set during
                # this batch (e.g. a concurrent enqueue while busy=True),
                # clear it and refetch.  Otherwise release ``busy`` under
                # the SAME lock so a concurrent enqueue cannot squeeze a
                # request_pending=True past us into a now-stranded state.
                if await _try_atomic_release():
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                to_process_docs = await self.doc_status.get_docs_by_statuses(
                    _processing_statuses
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

        Idempotent by design: ``meta.analyze_time`` is treated as the
        timestamp of the most recent successful pass rather than a
        "completed" sentinel, and per-item ``llm_analyze_result`` already
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

            # ``analyze_time`` is now the "most recent successful pass"
            # timestamp.  We refresh it after the body finishes successfully
            # rather than using it as an early-return gate, so re-triggering
            # analyze_multimodal with newly-enabled i/t/e options proceeds.
            now_iso = datetime.now(timezone.utc).isoformat()

            # Analyze sidecar multimodal items by VLM model role.
            use_vlm_func = self.role_llm_funcs["vlm"]
            effective_vlm_max_async = self._get_effective_role_llm_max_async("vlm")
            sem = asyncio.Semaphore(max(1, effective_vlm_max_async))
            analyze_retries = max(0, int(os.getenv("VLM_ANALYZE_RETRIES", "2")))
            max_image_bytes = max(
                256 * 1024, int(os.getenv("VLM_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
            )

            def _extract_json_obj(text: str) -> dict[str, Any]:
                if not text:
                    return {}
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                return {}

            async def _analyze_item(
                root_key: str, item_id: str, item: dict[str, Any]
            ) -> dict[str, Any]:
                def _conservative_result(reason: str) -> dict[str, Any]:
                    base_name = item.get("caption") or item_id
                    modality = (
                        "image"
                        if root_key == "drawings"
                        else "table"
                        if root_key == "tables"
                        else "equation"
                    )
                    conservative = {
                        "name": base_name,
                        "summary": (
                            f"Conservative summary only: unavailable or weak visual evidence for {modality}."
                        ),
                        "detail_description": (
                            f"No grounded visual evidence. Reason: {reason}. "
                            "Only metadata-level description is retained."
                        ),
                        "grounded": False,
                        "grounding_reason": reason,
                    }
                    if root_key == "drawings":
                        conservative["image_type"] = ""
                    return conservative

                def _build_image_data_url(path_str: str | None) -> str | None:
                    if not path_str:
                        return None
                    p = Path(path_str)
                    if not p.exists() or not p.is_file():
                        return None
                    try:
                        raw = p.read_bytes()
                    except Exception:
                        return None
                    if not raw:
                        return None
                    if len(raw) > max_image_bytes:
                        logger.warning(
                            f"[analyze_multimodal] image too large ({len(raw)} bytes) for {root_key}/{item_id}, skip image input"
                        )
                        return None
                    mime, _ = mimetypes.guess_type(str(p))
                    if not mime:
                        mime = "image/png"
                    b64 = base64.b64encode(raw).decode("ascii")
                    return f"data:{mime};base64,{b64}"

                def _normalize_text(value: Any) -> str:
                    if value is None:
                        return ""
                    if isinstance(value, str):
                        return value.strip()
                    if isinstance(value, (list, tuple)):
                        return "\n".join(
                            str(v).strip() for v in value if str(v).strip()
                        )
                    return str(value).strip()

                def _normalize_grounded_value(value: Any) -> Any:
                    if isinstance(value, bool) or value is None:
                        return value
                    if isinstance(value, str):
                        lowered = value.strip().lower()
                        if lowered == "true":
                            return True
                        if lowered == "false":
                            return False
                    if isinstance(value, (int, float)) and value in {0, 1}:
                        return bool(value)
                    return value

                default_result = {
                    "name": item.get("caption") or item_id,
                    "summary": "",
                    "detail_description": "",
                }
                if root_key == "drawings":
                    default_result["image_type"] = ""
                schema_hint = (
                    '{"name":"string","summary":"string","detail_description":"string","grounded":"boolean","grounding_reason":"string"}'
                    if root_key != "drawings"
                    else '{"name":"string","image_type":"string","summary":"string","detail_description":"string","grounded":"boolean","grounding_reason":"string"}'
                )
                image_data_url = _build_image_data_url(
                    item.get("path") or item.get("img_path") or item.get("image_path")
                )
                has_visual_evidence = bool(image_data_url)
                caption_text = _normalize_text(item.get("caption"))
                footnotes_text = _normalize_text(item.get("footnotes"))
                content_text = _normalize_text(item.get("content"))
                has_textual_evidence = root_key in {
                    "tables",
                    "equations",
                } and any((caption_text, footnotes_text, content_text))
                evidence_mode = (
                    "visual"
                    if has_visual_evidence
                    else "textual"
                    if has_textual_evidence
                    else "none"
                )
                for attempt in range(analyze_retries + 1):
                    prompt = (
                        "You are a multimodal analyzer.\n"
                        "Return ONLY one JSON object. No markdown. No explanation.\n"
                        "Grounding policy:\n"
                        "- Do NOT invent unseen objects, domains, diseases, or scenarios.\n"
                        "- Prefer the strongest available evidence source.\n"
                        "- For tables/equations without image evidence, analyze from content/caption/footnotes first.\n"
                        "- In textual-only mode, do not invent appearance/layout details that are not supported by the provided content.\n"
                        "- If evidence is missing/weak/uncertain, set grounded=false and keep summary/detail conservative.\n"
                        "- If grounded=false, avoid rich semantic claims; keep to metadata-level statements only.\n"
                        f"JSON schema example: {schema_hint}\n"
                        f"modality={root_key}\n"
                        f"item_id={item_id}\n"
                        f"caption={caption_text}\n"
                        f"footnotes={footnotes_text}\n"
                        f"content={content_text}\n"
                        f"has_visual_evidence={has_visual_evidence}\n"
                        f"has_textual_evidence={has_textual_evidence}\n"
                        f"evidence_mode={evidence_mode}\n"
                        "Constraints:\n"
                        "- summary: <= 120 words\n"
                        "- detail_description: <= 500 words\n"
                    )
                    messages = None
                    if image_data_url:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_data_url},
                                    },
                                ],
                            }
                        ]
                    async with sem:
                        if messages:
                            try:
                                result_text = await use_vlm_func(
                                    prompt, stream=False, messages=messages
                                )
                            except TypeError:
                                # Backward compatibility for providers that don't accept messages.
                                result_text = await use_vlm_func(prompt, stream=False)
                            except Exception as msg_err:
                                logger.warning(
                                    f"[analyze_multimodal] visual call failed for {root_key}/{item_id}: {msg_err}"
                                )
                                return _conservative_result("visual_call_failed")
                        else:
                            result_text = await use_vlm_func(prompt, stream=False)
                    parsed = _extract_json_obj(str(result_text))
                    if (
                        parsed
                        and isinstance(parsed.get("name"), str)
                        and isinstance(parsed.get("summary"), str)
                        and isinstance(parsed.get("detail_description"), str)
                    ):
                        if "grounded" in parsed:
                            parsed["grounded"] = _normalize_grounded_value(
                                parsed.get("grounded")
                            )
                        default_result.update(
                            {
                                k: v
                                for k, v in parsed.items()
                                if k
                                in {
                                    "name",
                                    "summary",
                                    "detail_description",
                                    "image_type",
                                    "grounded",
                                    "grounding_reason",
                                }
                            }
                        )
                        if evidence_mode == "none":
                            return _conservative_result("missing_image")
                        if parsed.get("grounded") is False:
                            reason = str(
                                parsed.get("grounding_reason")
                                or (
                                    "weak_visual_evidence"
                                    if evidence_mode == "visual"
                                    else "weak_textual_evidence"
                                )
                            )
                            return _conservative_result(reason)
                        if "grounded" not in default_result:
                            default_result["grounded"] = True
                        if not default_result.get("grounding_reason"):
                            default_result["grounding_reason"] = (
                                "visual_evidence"
                                if evidence_mode == "visual"
                                else "textual_content_only"
                            )
                        return default_result
                    if attempt < analyze_retries:
                        logger.warning(
                            f"[analyze_multimodal] invalid JSON, retry {attempt + 1}/{analyze_retries} for {root_key}/{item_id}"
                        )
                if evidence_mode == "none":
                    return _conservative_result("missing_image")
                return _conservative_result("analysis_failed")

            # Write back llm_analyze_result to multimodal sidecar files.
            base_name = str(block_file)
            if base_name.endswith(".blocks.jsonl"):
                base_name = base_name[: -len(".blocks.jsonl")]
            sidecars = [
                (Path(base_name + ".drawings.json"), "drawings", process_opts.images),
                (Path(base_name + ".tables.json"), "tables", process_opts.tables),
                (
                    Path(base_name + ".equations.json"),
                    "equations",
                    process_opts.equations,
                ),
            ]
            for sidecar_path, root_key, enabled in sidecars:
                if not enabled:
                    continue
                if not sidecar_path.exists():
                    continue
                try:
                    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    items = payload.get(root_key, {})
                    if isinstance(items, dict):
                        analyze_tasks = []
                        valid_keys = []
                        skipped_existing = 0
                        for item_id, item in items.items():
                            if not isinstance(item, dict):
                                continue
                            # Idempotency: skip items that already have a VLM
                            # result from a prior pass.  A user re-enabling
                            # additional modalities should not re-spend tokens
                            # on items that were already analyzed.
                            if isinstance(item.get("llm_analyze_result"), dict):
                                skipped_existing += 1
                                continue
                            valid_keys.append(item_id)
                            analyze_tasks.append(_analyze_item(root_key, item_id, item))
                        if skipped_existing:
                            logger.debug(
                                f"[analyze_multimodal] {root_key}: "
                                f"{skipped_existing} item(s) already have "
                                f"llm_analyze_result, skipping; "
                                f"{len(analyze_tasks)} item(s) to analyze"
                            )
                        analyzed_results = await asyncio.gather(
                            *analyze_tasks, return_exceptions=True
                        )
                        for idx, item_id in enumerate(valid_keys):
                            item = items.get(item_id)
                            if not isinstance(item, dict):
                                continue
                            result_obj = analyzed_results[idx]
                            if isinstance(result_obj, Exception):
                                logger.warning(
                                    f"[analyze_multimodal] item analyze failed: {root_key}/{item_id}: {result_obj}"
                                )
                                continue
                            item["llm_analyze_result"] = result_obj
                    sidecar_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as sidecar_error:
                    logger.warning(
                        f"[analyze_multimodal] failed to write sidecar {sidecar_path}: {sidecar_error}"
                    )

            # Refresh ``meta.analyze_time`` to record the most-recent successful
            # pass.  This happens after the sidecar loop so a crash mid-loop
            # does not falsely advertise completion; on the next run the same
            # already-analyzed items will be skipped anyway.
            meta["analyze_time"] = now_iso
            lines[0] = json.dumps(meta, ensure_ascii=False)
            block_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            parsed_data["analyze_time"] = now_iso
            parsed_data["multimodal_processed"] = True
            logger.info(
                f"[analyze_multimodal] marked analyze_time for d-id: {doc_id}, file: {file_path}"
            )
        except Exception as e:
            logger.warning(
                f"[analyze_multimodal] failed to update analyze_time for d-id: {doc_id}: {e}"
            )
        return parsed_data

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
            "split_method": "raw_paragraph",
            "blocks": len(blocks_lines),
            "doc_id": doc_id,
            "parse_engine": engine,
            "parse_time": parse_time,
            "analyze_time": "",
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build multimodal chunks and modality descriptors from sidecars."""
        block_file = Path(blocks_path)
        if not block_file.exists():
            return [], []

        base = str(block_file)
        if base.endswith(".blocks.jsonl"):
            base = base[: -len(".blocks.jsonl")]

        sidecar_defs = [
            ("drawings", Path(base + ".drawings.json"), "drawing"),
            ("tables", Path(base + ".tables.json"), "table"),
            ("equations", Path(base + ".equations.json"), "equation"),
        ]

        mm_chunks: list[dict[str, Any]] = []
        mm_specs: list[dict[str, Any]] = []
        order = base_order_index

        def _norm_list(v: Any) -> list[str]:
            if v is None:
                return []
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            return [s] if s else []

        def _mm_entity_name(kind: str, raw_payload: dict[str, Any]) -> str:
            payload = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True)
            digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
            return f"{kind}-{digest}"

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

                # mm_chunks are VLM-output containers: without llm_analyze_result
                # there is nothing meaningful to index. analyze_multimodal only
                # writes this field for modalities opted into via i/t/e in
                # process_options, so unopted modalities (and VLM failures)
                # naturally produce no chunks here.
                analysis = item.get("llm_analyze_result")
                if not isinstance(analysis, dict) or not analysis:
                    continue
                name = str(analysis.get("name") or item.get("caption") or item_id)
                summary = str(analysis.get("summary") or "").strip()
                detail = str(analysis.get("detail_description") or "").strip()
                heading = str(item.get("heading") or "").strip()
                captions = _norm_list(item.get("caption"))
                footnotes = _norm_list(item.get("footnotes"))
                image_type = str(analysis.get("image_type") or "").strip()

                raw_for_hash: dict[str, Any] = {
                    "kind": kind,
                    "name": name,
                    "summary": summary,
                    "detail": detail,
                    "content": item.get("content"),
                    "path": item.get("path"),
                    "src": item.get("src"),
                    "caption": item.get("caption"),
                }
                entity_name = _mm_entity_name(kind, raw_for_hash)
                chunk_id = f"{doc_id}-mm-{kind}-{local_idx:03d}"

                if kind == "drawing":
                    lines = [
                        f"Image_Name: {name}",
                    ]
                    if image_type:
                        lines.append(f"Image_Type: {image_type}")
                    lines.extend(
                        [
                            "Image_Location:",
                            f"  - Document_Name: {Path(file_path).name}",
                        ]
                    )
                    if heading:
                        lines.append(f"  - Session_Heading: {heading}")
                    if captions:
                        lines.append("Image_Captions:")
                        lines.extend([f"  - {x}" for x in captions])
                    if footnotes:
                        lines.append("Image_Footnotes:")
                        lines.extend([f"  - {x}" for x in footnotes])
                    if summary:
                        lines.append(f'Image_Summary: "{summary}"')
                    if detail:
                        lines.append(f'Image_Detail_Description: "{detail}"')
                elif kind == "table":
                    lines = [
                        f"Table_Name: {name}",
                        "Table_Location:",
                        f"  - Document_Name: {Path(file_path).name}",
                    ]
                    if heading:
                        lines.append(f"  - Session_Heading: {heading}")
                    if captions:
                        lines.append("Table_Captions:")
                        lines.extend([f"  - {x}" for x in captions])
                    if footnotes:
                        lines.append("Table_Footnotes:")
                        lines.extend([f"  - {x}" for x in footnotes])
                    if summary:
                        lines.append(f'Table_Summary: "{summary}"')
                    if detail:
                        lines.append(f'Table_Detail_Description: "{detail}"')
                else:
                    lines = [
                        f"Equation_Name: {name}",
                        "Equation_Location:",
                        f"  - Document_Name: {Path(file_path).name}",
                    ]
                    if heading:
                        lines.append(f"  - Session_Heading: {heading}")
                    if captions:
                        lines.append("Equation_Captions:")
                        lines.extend([f"  - {x}" for x in captions])
                    if footnotes:
                        lines.append("Equation_Footnotes:")
                        lines.extend([f"  - {x}" for x in footnotes])
                    if summary:
                        lines.append(f'Equation_Summary: "{summary}"')
                    if detail:
                        lines.append(f'Equation_Detail_Description: "{detail}"')

                chunk_content = "\n".join(lines).strip()
                if not chunk_content:
                    continue

                mm_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_order_index": order,
                        "content": chunk_content,
                        "tokens": len(self.tokenizer.encode(chunk_content)),
                        "content_type": kind,
                        "heading": heading,
                        "parent_headings": [],
                        "level": 0,
                    }
                )
                mm_specs.append(
                    {
                        "kind": kind,
                        "chunk_id": chunk_id,
                        "entity_name": entity_name,
                        "entity_type": kind,
                        "name": name,
                        "caption_text": "; ".join(captions),
                        "heading": heading,
                        "summary": summary,
                    }
                )
                order += 1

        return mm_chunks, mm_specs
