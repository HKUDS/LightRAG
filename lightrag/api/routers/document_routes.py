"""
This module contains all document-related routes for the LightRAG API.
"""

import asyncio
import base64
import binascii
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from lightrag.utils import (
    logger,
    get_pinyin_sort_key,
    performance_timing_log,
    safe_log_value,
    validate_workspace,
)
import aiofiles
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Any, Literal, Sequence
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lightrag import LightRAG
from lightrag.api.utils_api import internal_server_error
from lightrag.base import (
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    DocProcessingStatus,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.exceptions import (
    PipelineBackpressureError,
    SourceConflictPrimaryUnusableError,
    SourceConflictRepairCASError,
    StorageCapabilityError,
    StorageControlPlaneError,
    StorageNotInitializedError,
)
from lightrag.constants import (
    DEFAULT_SCAN_ENQUEUE_BATCH_SIZE,
    FILE_EXTRACTION_SUMMARY_PREFIX,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_ARTIFACT_DIR_SUFFIXES,
    PARSED_DIR_NAME,
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_PARAGRAH,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_CHUNK_VECTOR,
)
from lightrag.tools.source_conflict_repair import (
    refuse_an_unusable_primary,
    source_conflict_repair_lock,
    verify_repair_outcome,
)
from lightrag.kg.scan_job_store import (
    SAMPLE_BUCKETS,
    SCAN_JOB_LEASE_SECONDS,
    SCAN_JOB_SAMPLE_LIMIT,
    ScanJobCreateOutcome,
    ScanJobStatus,
    ScanJobUpdateConflict,
)
from lightrag.parser.routing import (
    FilenameParserHintError,
    canonicalize_parser_hinted_basename,
    chunk_strategy_key,
    encode_parse_engine,
    parse_process_options,
    resolve_chunk_options,
    resolve_parser_directives,
)
from lightrag.utils import (
    generate_track_id,
    move_file_to_parsed_dir,
)
from lightrag.kg.shared_storage import append_pipeline_history
from lightrag.utils_pipeline import count_active_documents, read_source_file_basename
from lightrag.api.admission import adopt_admission_ticket
from lightrag.api.utils_api import get_combined_auth_dependency
from ..config import global_args


# Function to format datetime to ISO format string with timezone information
def format_datetime(dt: Any) -> Optional[str]:
    """Format datetime to ISO format string with timezone information

    Args:
        dt: Datetime object, string, or None

    Returns:
        ISO format string with timezone information, or None if input is None
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt

    # Check if datetime object has timezone information
    if isinstance(dt, datetime):
        # If datetime object has no timezone info (naive datetime), add UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

    # Return ISO format string with timezone information
    return dt.isoformat()


# NOTE: the APIRouter instance is created INSIDE `create_document_routes`
# (not at module scope). A module-level router is shared across processes,
# and re-running the factory — which the test suite does to validate
# create_app for different `--api-prefix` values — would re-decorate the
# same router each time, accumulating duplicate routes and triggering
# FastAPI's "Duplicate Operation ID" warnings.

# Temporary file prefix
temp_prefix = "__tmp__"
UNKNOWN_FILE_SOURCE = "unknown_source"
LEGACY_EMPTY_FILE_PATH_SENTINELS = {"", "no-file-path"}
ARCHIVED_FILE_SUFFIX_RE = re.compile(r"_(?:\d{3}|\d{10,})$")

# ``Retry-After`` hint on an admission 429. Capacity frees up when a document
# finishes processing, which is a document-scale wait, not a request-scale one —
# a value low enough to invite an immediate retry storm would be worse than none.
_ADMISSION_RETRY_AFTER_SECONDS = 30


def normalize_file_path(file_path: str | None) -> str:
    """Normalize missing document sources to a single non-null sentinel."""
    if file_path is None:
        return UNKNOWN_FILE_SOURCE

    normalized = file_path.strip()
    if normalized in LEGACY_EMPTY_FILE_PATH_SENTINELS:
        return UNKNOWN_FILE_SOURCE

    return canonicalize_parser_hinted_basename(normalized) or UNKNOWN_FILE_SOURCE


def is_valid_file_source(file_source: str | None) -> bool:
    if file_source is None:
        return False
    return normalize_file_path(file_source) != UNKNOWN_FILE_SOURCE


def sanitize_filename(filename: str, input_dir: Path) -> str:
    """
    Sanitize uploaded filename to prevent Path Traversal attacks.

    Args:
        filename: The original filename from the upload
        input_dir: The target input directory

    Returns:
        str: Sanitized filename that is safe to use

    Raises:
        HTTPException: If the filename is unsafe or invalid
    """
    # Basic validation
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Remove path separators and traversal sequences
    clean_name = filename.replace("/", "").replace("\\", "")
    clean_name = clean_name.replace("..", "")

    # Remove control characters and null bytes
    clean_name = "".join(c for c in clean_name if ord(c) >= 32 and c != "\x7f")

    # Remove leading/trailing whitespace and dots
    clean_name = clean_name.strip().strip(".")

    # Check if anything is left after sanitization
    if not clean_name:
        raise HTTPException(
            status_code=400, detail="Invalid filename after sanitization"
        )

    # Verify the final path stays within the input directory
    try:
        final_path = (input_dir / clean_name).resolve()
        if not final_path.is_relative_to(input_dir.resolve()):
            raise HTTPException(status_code=400, detail="Unsafe filename detected")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    return clean_name


class ScanResponse(BaseModel):
    """Response model for document scanning operation

    Attributes:
        status: Status of the scanning operation.  ``scanning_started`` when
            a new background scan has been scheduled;
            ``scanning_skipped_pipeline_busy`` when the request was rejected
            because indexing or another scan is already running.
        message: Optional message with additional details
        track_id: Tracking ID for monitoring scanning progress
    """

    status: Literal["scanning_started", "scanning_skipped_pipeline_busy"] = Field(
        description="Status of the scanning operation"
    )
    message: Optional[str] = Field(
        default=None, description="Additional details about the scanning operation"
    )
    track_id: str = Field(description="Tracking ID for monitoring scanning progress")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "scanning_started",
                "message": "Scanning process has been initiated in the background",
                "track_id": "scan_20250729_170612_abc123",
            }
        }
    )


class ScanJobSampleBucket(BaseModel):
    """One bounded sample bucket of a scan job record (LR2 §8.6).

    ``items`` is capped in COUNT (per-bucket limit) and in SIZE (UTF-8 bytes per
    sample); ``truncated`` marks that at least one retained sample was clipped
    and ``dropped`` counts samples the store did not retain. There is no
    unbounded per-file list anywhere in this schema."""

    items: List[str] = Field(
        default_factory=list, description="Retained sample strings (bounded)"
    )
    truncated: bool = Field(
        default=False, description="At least one retained sample was byte-truncated"
    )
    dropped: int = Field(
        default=0, description="Samples not retained (count or record byte cap)"
    )


class ScanJobStatusResponse(BaseModel):
    """Bounded status of one scan job (``GET /documents/scan/status/{track_id}``).

    Aggregate counters plus the three fixed sample buckets — never the full file
    list. ``status`` is ``running`` until the owning task finalises it
    (``completed`` / ``failed`` / ``cancelled``), or ``abandoned`` when the
    record's lease expired because its owner died."""

    track_id: str = Field(description="Tracking ID of the scan job")
    status: Literal["running", "completed", "failed", "cancelled", "abandoned"] = Field(
        description="Lifecycle status of the scan job"
    )
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Aggregate classification/progress counters (bounded key set)",
    )
    counters_dropped: int = Field(
        default=0,
        description=(
            "Counter deltas the store refused (over-long key, distinct-key cap, "
            "or the record byte ceiling) — always 0 for an in-tree client"
        ),
    )
    samples: Dict[str, ScanJobSampleBucket] = Field(
        default_factory=dict,
        description="Bounded processed / warning / error sample buckets",
    )
    created_at: float = Field(description="Job creation time (epoch seconds)")
    updated_at: float = Field(description="Last update time (epoch seconds)")
    version: int = Field(description="CAS version of the record")
    message: str = Field(
        default="", description="Terminal or latest status message (byte-capped)"
    )


class SourceConflictItem(BaseModel):
    """One canonical source key claimed by more than one primary document.

    ``candidate_count`` is ``None`` when the backend only proved "at least two"
    without a cheap exact count; ``sample_doc_ids`` is a fixed-size sample, never
    the whole conflicting set (LR2 §5.5)."""

    canonical_source_key: str = Field(
        description="Canonical (hint-stripped) basename shared by the candidates"
    )
    candidate_count: Optional[int] = Field(
        default=None, description="Exact number of primary candidates, when known"
    )
    sample_doc_ids: List[str] = Field(
        default_factory=list, description="Bounded sample of candidate document IDs"
    )


class SourceConflictListResponse(BaseModel):
    """One page of source conflicts (``GET /documents/source_conflicts``)."""

    conflicts: List[SourceConflictItem] = Field(
        default_factory=list, description="Conflicts in this page"
    )
    next_cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor for the next page; null when the listing is exhausted"
        ),
    )


class SourceConflictRepairRequest(BaseModel):
    """Operator request to resolve one source conflict (LR2 §5.5).

    The winner is never chosen automatically: the operator names
    ``primary_doc_id``. ``dry_run`` (the default) mutates nothing and returns the
    ``candidate_count`` / ``fingerprint`` pair that must be echoed back to
    commit — a compare-and-set token, so a candidate set that changed in between
    fails the commit instead of being silently overwritten."""

    canonical_source_key: str = Field(
        min_length=1, description="Canonical source key to repair"
    )
    primary_doc_id: str = Field(
        min_length=1, description="Document ID to keep as the single primary"
    )
    expected_candidate_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="CAS token from the dry-run; required when dry_run is false",
    )
    expected_candidate_fingerprint: Optional[str] = Field(
        default=None,
        description="CAS token from the dry-run; required when dry_run is false",
    )
    dry_run: bool = Field(
        default=True, description="Report only (default) instead of committing"
    )

    @model_validator(mode="after")
    def _commit_requires_cas_tokens(self) -> "SourceConflictRepairRequest":
        # A commit without the echoed tokens could not be CAS-checked at all;
        # refuse it here (422) rather than let a backend compare against a
        # placeholder. dry-run ignores them by contract.
        if not self.dry_run and (
            self.expected_candidate_count is None
            or not self.expected_candidate_fingerprint
        ):
            raise ValueError(
                "expected_candidate_count and expected_candidate_fingerprint are "
                "required when dry_run is false; run a dry-run first and echo "
                "both values back"
            )
        return self


class SourceConflictRepairResponse(BaseModel):
    """Outcome of a source-conflict repair, dry-run or committed."""

    canonical_source_key: str = Field(description="Canonical source key")
    primary_doc_id: str = Field(description="Document ID kept as the single primary")
    candidate_count: int = Field(
        description="Primary candidates observed under the repair lock"
    )
    fingerprint: str = Field(
        description=(
            "Deterministic digest of the candidate IDs — the CAS token to echo "
            "back on commit"
        )
    )
    demoted_sample_doc_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Bounded sample of the documents that would be / were marked "
            "metadata.is_duplicate=true (content is never deleted)"
        ),
    )
    committed: bool = Field(
        description="False for a dry-run, true when the demotions were written"
    )


class ReprocessResponse(BaseModel):
    """Response model for reprocessing failed documents operation

    Attributes:
        status: Status of the reprocessing operation
        message: Message describing the operation result
        track_id: Always empty string. Reprocessed documents retain their original track_id.
    """

    status: Literal["reprocessing_started"] = Field(
        description="Status of the reprocessing operation"
    )
    message: str = Field(description="Human-readable message describing the operation")
    track_id: str = Field(
        default="",
        description="Always empty string. Reprocessed documents retain their original track_id from initial upload.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "reprocessing_started",
                "message": "Reprocessing of failed documents has been initiated in background",
                "track_id": "",
            }
        }
    )


class CancelPipelineResponse(BaseModel):
    """Response model for pipeline cancellation operation

    Attributes:
        status: Status of the cancellation request
        message: Message describing the operation result
    """

    status: Literal["cancellation_requested", "not_busy"] = Field(
        description="Status of the cancellation request"
    )
    message: str = Field(description="Human-readable message describing the operation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "cancellation_requested",
                "message": "Pipeline cancellation has been requested. Documents will be marked as FAILED.",
            }
        }
    )


TextChunkingStrategy = Literal[
    "fixed_token",
    "recursive_character",
    "semantic_vector",
    "paragraph_semantic",
]


class _StrictChunkParams(BaseModel):
    """Base for per-strategy chunking params.

    ``strict=True`` rejects the Pydantic-v2 lax coercions that would
    otherwise let malformed requests through and fail later in the
    background chunker: bool-as-int (``true`` -> 1), numeric strings
    (``"5"`` -> 5), float-as-int.  ``extra="forbid"`` turns unknown keys
    into a 422 (replacing a hand-rolled allow-list).  ``chunk_token_size``
    is shared by every strategy; ``None`` means "not supplied — fall back
    to ``addon_params``/env default at process time".
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_token_size: Optional[int] = Field(default=None, ge=1)


class _OverlapChunkParams(_StrictChunkParams):
    chunk_overlap_token_size: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> "_OverlapChunkParams":
        # Only enforceable when BOTH are explicit; when chunk_token_size
        # is None the effective size is resolved from addon_params/env at
        # process time and can't be compared against here.
        if (
            self.chunk_token_size is not None
            and self.chunk_overlap_token_size is not None
            and self.chunk_overlap_token_size >= self.chunk_token_size
        ):
            raise ValueError("chunk_overlap_token_size must be < chunk_token_size")
        return self


class FixedTokenChunkParams(_OverlapChunkParams):
    split_by_character: Optional[str] = None
    split_by_character_only: Optional[bool] = None


class RecursiveCharacterChunkParams(_OverlapChunkParams):
    separators: Optional[list[str]] = None


class ParagraphSemanticChunkParams(_OverlapChunkParams):
    # Drop the trailing reference section before chunking. ``None`` means
    # "not supplied — inherit the addon_params/env default at process time".
    # Detection-tuning knobs (tail window / heading prefixes) are env-only and
    # read live by the chunker, so they are intentionally not exposed here.
    drop_references: Optional[bool] = None


class SemanticVectorChunkParams(_StrictChunkParams):
    # Enum verified against the installed langchain_experimental
    # (text_splitter.py ``BreakpointThresholdType``), not from memory.
    breakpoint_threshold_type: Optional[
        Literal["percentile", "standard_deviation", "interquartile", "gradient"]
    ] = None
    # A strict ``float`` field still accepts an ``int`` (e.g. JSON ``95``) and
    # widens it losslessly to ``95.0`` — strict only rejects ``str`` / ``bool``
    # here, which is exactly what we want. Do NOT relax strict (that would let
    # numeric strings through) or switch to ``int | float`` (that would stop
    # normalizing ints to float). Locked by tests in test_document_routes_chunking.
    breakpoint_threshold_amount: Optional[float] = None
    buffer_size: Optional[int] = Field(default=None, ge=1)
    sentence_split_regex: Optional[str] = None

    @field_validator("sentence_split_regex")
    @classmethod
    def _valid_sentence_split_regex(cls, v: Optional[str]) -> Optional[str]:
        # The value is fed to LangChain's SemanticChunker and compiled during
        # split_text. A malformed pattern (e.g. "(") would only blow up in the
        # background, so compile it here to reject synchronously (HTTP 422).
        if v is None:
            return v
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(
                f"sentence_split_regex is not a valid regular expression: {exc}"
            ) from exc
        return v

    @model_validator(mode="after")
    def _amount_in_range(self) -> "SemanticVectorChunkParams":
        amt = self.breakpoint_threshold_amount
        if amt is None:
            return self
        # ``> 0`` is type-independent (every threshold type wants a positive
        # magnitude), so it is safe to enforce at parse time.
        if amt <= 0:
            raise ValueError("breakpoint_threshold_amount must be > 0")
        # The ``(0, 100]`` ceiling is percentile/gradient-specific (those feed
        # np.percentile, which requires q in [0, 100]). It depends on the
        # threshold TYPE, so only enforce it here when the type is supplied in
        # the SAME request. When the type is omitted, the effective type is
        # resolved from addon_params/env later — assuming "percentile" here
        # would wrongly 422 a partial override that inherits
        # standard_deviation/interquartile (which allow amounts > 100). The
        # ceiling against the merged type is applied by
        # ``_validate_effective_semantic_amount`` in ``_resolve_text_chunking``.
        if self.breakpoint_threshold_type in ("percentile", "gradient") and amt > 100:
            raise ValueError(
                "breakpoint_threshold_amount must be within (0, 100] "
                "for percentile/gradient"
            )
        return self


_CHUNKING_PARAMS_MODEL: dict[str, type[_StrictChunkParams]] = {
    "fixed_token": FixedTokenChunkParams,
    "recursive_character": RecursiveCharacterChunkParams,
    "semantic_vector": SemanticVectorChunkParams,
    "paragraph_semantic": ParagraphSemanticChunkParams,
}


class TextChunkingConfig(BaseModel):
    """Chunking strategy + strategy-specific params for a text insert.

    Validation is delegated to the per-strategy typed model so unknown
    keys, wrong types, and out-of-range values all raise synchronously
    during request parsing (HTTP 422) — never later in the background
    indexing task, where the HTTP response has already been sent.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: TextChunkingStrategy = "fixed_token"
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> "TextChunkingConfig":
        typed = _CHUNKING_PARAMS_MODEL[self.strategy].model_validate(self.params)
        # Normalize down to exactly the keys the caller supplied with a real
        # value (validated + coerced) so the enqueue-time merge overrides only
        # what was set. ``exclude_none`` additionally drops explicit nulls:
        # every param field means "inherit the addon_params/env default" when
        # None, so an explicit ``"chunk_token_size": null`` must NOT be merged
        # over the resolved default — otherwise the route would 200 and the
        # background chunker would do ``int(None)`` and fail the document.
        self.params = typed.model_dump(exclude_unset=True, exclude_none=True)
        return self


class InsertTextRequest(BaseModel):
    """Request model for inserting a single text document

    Attributes:
        text: The text content to be inserted into the RAG system
        file_source: Source of the text (optional)
        chunking: Optional chunking strategy + params; omit to keep the
            default fixed-token behavior and addon_params defaults.
    """

    text: str = Field(
        min_length=1,
        description="The text to insert",
    )
    file_source: Optional[str] = Field(
        default=None, min_length=0, description="File Source"
    )
    chunking: Optional[TextChunkingConfig] = Field(
        default=None,
        description="Chunking strategy and params; omit for default fixed-token chunking",
    )

    @field_validator("text", mode="after")
    @classmethod
    def strip_text_after(cls, text: str) -> str:
        # min_length runs before strip; reject whitespace-only so empty docs are not enqueued.
        stripped = text.strip()
        if not stripped:
            raise ValueError("text cannot be empty or whitespace-only")
        return stripped

    @field_validator("file_source", mode="before")
    @classmethod
    def normalize_source_before(cls, file_source: Optional[str]) -> str:
        return normalize_file_path(file_source)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This is a sample text to be inserted into the RAG system.",
                "file_source": "Source of the text (optional)",
                "chunking": {
                    "strategy": "fixed_token",
                    "params": {
                        "chunk_token_size": 1200,
                        "chunk_overlap_token_size": 100,
                        "split_by_character": "\n\n",
                        "split_by_character_only": True,
                    },
                },
            }
        }
    )


class InsertTextsRequest(BaseModel):
    """Request model for inserting multiple text documents

    Attributes:
        texts: List of text contents to be inserted into the RAG system
        file_sources: Sources of the texts (optional)
    """

    texts: list[str] = Field(
        min_length=1,
        description="The texts to insert",
    )
    file_sources: Optional[list[str]] = Field(
        default=None, min_length=0, description="Sources of the texts"
    )
    chunking: Optional[TextChunkingConfig] = Field(
        default=None,
        description="Shared chunking strategy and params for all texts; omit for default fixed-token chunking",
    )

    @field_validator("texts", mode="after")
    @classmethod
    def strip_texts_after(cls, texts: list[str]) -> list[str]:
        stripped = [text.strip() for text in texts]
        if any(not text for text in stripped):
            raise ValueError("texts cannot contain empty or whitespace-only entries")
        return stripped

    @field_validator("file_sources", mode="before")
    @classmethod
    def normalize_sources_before(
        cls, file_sources: Optional[list[str]]
    ) -> Optional[list[str]]:
        if file_sources is None:
            return None

        return [normalize_file_path(file_source) for file_source in file_sources]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "This is the first text to be inserted.",
                    "This is the second text to be inserted.",
                ],
                "file_sources": [
                    "First file source (optional)",
                ],
                "chunking": {
                    "strategy": "recursive_character",
                    "params": {"chunk_token_size": 1000},
                },
            }
        }
    )


class InsertResponse(BaseModel):
    """Response model for document insertion operations

    Attributes:
        status: Status of the operation (success, partial_success, failure).
            Same-name conflicts are rejected with HTTP 409 rather than being
            reported as a "duplicated" 200 response, so this field never
            takes that value any more.
        message: Detailed message describing the operation result
        track_id: Tracking ID for monitoring processing status
    """

    status: Literal["success", "partial_success", "failure"] = Field(
        description="Status of the operation"
    )
    message: str = Field(description="Message describing the operation result")
    track_id: str = Field(description="Tracking ID for monitoring processing status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "File 'document.pdf' uploaded successfully. Processing will continue in background.",
                "track_id": "upload_20250729_170612_abc123",
            }
        }
    )


class ClearDocumentsResponse(BaseModel):
    """Response model for document clearing operation

    Attributes:
        status: Status of the clear operation
        message: Detailed message describing the operation result
    """

    status: Literal["success", "partial_success", "busy", "fail"] = Field(
        description="Status of the clear operation"
    )
    message: str = Field(description="Message describing the operation result")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "All documents cleared successfully. Deleted 15 files.",
            }
        }
    )


class ForceResetRecoveryRequest(BaseModel):
    """Request model for force-clearing a ``recovery_required`` fence.

    The fence is raised when a worker dies mid custom_chunks/delete/clear, which
    may leave storage partially committed. Clearing it re-opens a possibly-
    inconsistent workspace, so an explicit ``confirm`` is required.
    """

    confirm: bool = Field(
        default=False,
        description=(
            "Must be true to force-reset. The workspace may be in a "
            "partially-committed state; only reset after verifying/repairing it."
        ),
    )


class ForceResetRecoveryResponse(BaseModel):
    status: Literal["reset", "no_recovery_required"] = Field(
        description="reset = fence cleared; no_recovery_required = nothing to clear"
    )
    message: str = Field(description="Human-readable result")
    cancelled_manual_retries: int = Field(
        default=0,
        description=(
            "Queued manual retry requests cancelled along with the fence. A "
            "sticky request makes /documents/scan refuse its reservation (it may "
            "not jump the manual FIFO), so clearing the fence without clearing "
            "these would leave the documented recovery path — force_reset then "
            "/documents/scan — still blocked. The failed documents are untouched; "
            "re-issue /documents/reprocess_failed, or let the scan's own FAILED "
            "reset cover them."
        ),
    )


class ClearCacheRequest(BaseModel):
    """Request model for clearing cache

    This model is kept for API compatibility but no longer accepts any parameters.
    All cache will be cleared regardless of the request content.
    """

    model_config = ConfigDict(json_schema_extra={"example": {}})


class ClearCacheResponse(BaseModel):
    """Response model for cache clearing operation

    Attributes:
        status: Status of the clear operation
        message: Detailed message describing the operation result
    """

    status: Literal["success", "fail"] = Field(
        description="Status of the clear operation"
    )
    message: str = Field(description="Message describing the operation result")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Successfully cleared cache for modes: ['default', 'naive']",
            }
        }
    )


"""Response model for document status

Attributes:
    id: Document identifier
    content_summary: Summary of document content
    content_length: Length of document content
    status: Current processing status
    created_at: Creation timestamp (ISO format string)
    updated_at: Last update timestamp (ISO format string)
    chunks_count: Number of chunks (optional)
    error: Error message if any (optional)
    metadata: Additional metadata (optional)
    file_path: Path to the document file
"""


class DeleteDocRequest(BaseModel):
    doc_ids: List[str] = Field(..., description="The IDs of the documents to delete.")
    delete_file: bool = Field(
        default=False,
        description="Whether to delete the corresponding file in the upload directory.",
    )
    delete_llm_cache: bool = Field(
        default=False,
        description="Whether to delete cached LLM extraction results for the documents.",
    )

    @field_validator("doc_ids", mode="after")
    @classmethod
    def validate_doc_ids(cls, doc_ids: List[str]) -> List[str]:
        if not doc_ids:
            raise ValueError("Document IDs list cannot be empty")

        validated_ids = []
        for doc_id in doc_ids:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
            validated_ids.append(doc_id.strip())

        # Check for duplicates
        if len(validated_ids) != len(set(validated_ids)):
            raise ValueError("Document IDs must be unique")

        return validated_ids


# doc_status.metadata keys that are internal pipeline bookkeeping and must
# never reach the frontend. smartheading_llm_cache_ids is a deletion-time
# LLM-cache purge list (written by the parse pipeline at pipeline.py:1748,
# consumed only by adelete_by_doc_id).
_INTERNAL_METADATA_KEYS = frozenset({"smartheading_llm_cache_ids"})


class DocStatusResponse(BaseModel):
    id: str = Field(description="Document identifier")
    content_summary: str = Field(description="Summary of document content")
    content_length: int = Field(description="Length of document content in characters")
    status: DocStatus = Field(description="Current processing status")
    created_at: str = Field(description="Creation timestamp (ISO format string)")
    updated_at: str = Field(description="Last update timestamp (ISO format string)")
    track_id: Optional[str] = Field(
        default=None, description="Tracking ID for monitoring progress"
    )
    chunks_count: Optional[int] = Field(
        default=None, description="Number of chunks the document was split into"
    )
    error_msg: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata about the document"
    )
    file_path: str = Field(description="Path to the document file")

    @field_validator("metadata", mode="after")
    @classmethod
    def _strip_internal_metadata(
        cls, metadata: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Never expose internal pipeline bookkeeping to API clients."""
        if not isinstance(metadata, dict):
            return metadata  # None / (defensively) non-dict pass through
        if not _INTERNAL_METADATA_KEYS.intersection(metadata):
            return metadata  # common case: nothing to strip, no copy
        # Copy-then-strip — the source DocProcessingStatus.metadata is shared
        # with the deletion path and carry-over; mutating it in place would
        # remove the field adelete_by_doc_id needs.
        return {k: v for k, v in metadata.items() if k not in _INTERNAL_METADATA_KEYS}

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123456",
                "content_summary": "Research paper on machine learning",
                "content_length": 15240,
                "status": "processed",
                "created_at": "2025-03-31T12:34:56",
                "updated_at": "2025-03-31T12:35:30",
                "track_id": "upload_20250729_170612_abc123",
                "chunks_count": 12,
                "error": None,
                "metadata": {"author": "John Doe", "year": 2025},
                "file_path": "research_paper.pdf",
            }
        }
    )


class DocsStatusesResponse(BaseModel):
    """Response model for document statuses

    Attributes:
        statuses: Dictionary mapping document status to lists of document status responses
    """

    statuses: Dict[DocStatus, List[DocStatusResponse]] = Field(
        default_factory=dict,
        description="Dictionary mapping document status to lists of document status responses",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "statuses": {
                    "PENDING": [
                        {
                            "id": "doc_123",
                            "content_summary": "Pending document",
                            "content_length": 5000,
                            "status": "pending",
                            "created_at": "2025-03-31T10:00:00",
                            "updated_at": "2025-03-31T10:00:00",
                            "track_id": "upload_20250331_100000_abc123",
                            "chunks_count": None,
                            "error": None,
                            "metadata": None,
                            "file_path": "pending_doc.pdf",
                        }
                    ],
                    "PREPROCESSED": [
                        {
                            "id": "doc_789",
                            "content_summary": "Document pending final indexing",
                            "content_length": 7200,
                            "status": "preprocessed",
                            "created_at": "2025-03-31T09:30:00",
                            "updated_at": "2025-03-31T09:35:00",
                            "track_id": "upload_20250331_093000_xyz789",
                            "chunks_count": 10,
                            "error": None,
                            "metadata": None,
                            "file_path": "preprocessed_doc.pdf",
                        }
                    ],
                    "PROCESSED": [
                        {
                            "id": "doc_456",
                            "content_summary": "Processed document",
                            "content_length": 8000,
                            "status": "processed",
                            "created_at": "2025-03-31T09:00:00",
                            "updated_at": "2025-03-31T09:05:00",
                            "track_id": "insert_20250331_090000_def456",
                            "chunks_count": 8,
                            "error": None,
                            "metadata": {"author": "John Doe"},
                            "file_path": "processed_doc.pdf",
                        }
                    ],
                }
            }
        }
    )


class TrackStatusResponse(BaseModel):
    """Response model for tracking document processing status by track_id

    Attributes:
        track_id: The tracking ID
        documents: List of documents associated with this track_id
        total_count: Total number of documents for this track_id
        status_summary: Count of documents by status
    """

    track_id: str = Field(description="The tracking ID")
    documents: List[DocStatusResponse] = Field(
        description="List of documents associated with this track_id"
    )
    total_count: int = Field(description="Total number of documents for this track_id")
    status_summary: Dict[str, int] = Field(description="Count of documents by status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "track_id": "upload_20250729_170612_abc123",
                "documents": [
                    {
                        "id": "doc_123456",
                        "content_summary": "Research paper on machine learning",
                        "content_length": 15240,
                        "status": "PROCESSED",
                        "created_at": "2025-03-31T12:34:56",
                        "updated_at": "2025-03-31T12:35:30",
                        "track_id": "upload_20250729_170612_abc123",
                        "chunks_count": 12,
                        "error": None,
                        "metadata": {"author": "John Doe", "year": 2025},
                        "file_path": "research_paper.pdf",
                    }
                ],
                "total_count": 1,
                "status_summary": {"PROCESSED": 1},
            }
        }
    )


class DocumentsRequest(BaseModel):
    """Request model for paginated document queries

    Attributes:
        status_filter: Legacy single-status filter, ignored when status_filters is set
        status_filters: Filter by multiple document statuses, None for all statuses
        page: Page number (1-based)
        page_size: Number of documents per page (10-200)
        sort_field: Field to sort by ('created_at', 'updated_at', 'id', 'file_path')
        sort_direction: Sort direction ('asc' or 'desc')
    """

    status_filter: Optional[DocStatus] = Field(
        default=None,
        description="Legacy single-status filter, ignored when status_filters is set",
    )
    status_filters: Optional[List[DocStatus]] = Field(
        default=None, description="Filter by multiple document statuses"
    )
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=10, le=200, description="Number of documents per page (10-200)"
    )
    sort_field: Literal["created_at", "updated_at", "id", "file_path"] = Field(
        default="updated_at", description="Field to sort by"
    )
    sort_direction: Literal["asc", "desc"] = Field(
        default="desc", description="Sort direction"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_filters": ["PREPROCESSED", "PARSING", "ANALYZING"],
                "page": 1,
                "page_size": 50,
                "sort_field": "updated_at",
                "sort_direction": "desc",
            }
        }
    )


class PaginationInfo(BaseModel):
    """Pagination information

    Attributes:
        page: Current page number
        page_size: Number of items per page
        total_count: Total number of items
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_prev: Whether there is a previous page
    """

    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")
    total_count: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 1,
                "page_size": 50,
                "total_count": 150,
                "total_pages": 3,
                "has_next": True,
                "has_prev": False,
            }
        }
    )


class PaginatedDocsResponse(BaseModel):
    """Response model for paginated document queries

    Attributes:
        documents: List of documents for the current page
        pagination: Pagination information
        status_counts: Count of documents by status for all documents
    """

    documents: List[DocStatusResponse] = Field(
        description="List of documents for the current page"
    )
    pagination: PaginationInfo = Field(description="Pagination information")
    status_counts: Dict[str, int] = Field(
        description="Count of documents by status for all documents"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "id": "doc_123456",
                        "content_summary": "Research paper on machine learning",
                        "content_length": 15240,
                        "status": "PROCESSED",
                        "created_at": "2025-03-31T12:34:56",
                        "updated_at": "2025-03-31T12:35:30",
                        "track_id": "upload_20250729_170612_abc123",
                        "chunks_count": 12,
                        "error_msg": None,
                        "metadata": {"author": "John Doe", "year": 2025},
                        "file_path": "research_paper.pdf",
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 50,
                    "total_count": 150,
                    "total_pages": 3,
                    "has_next": True,
                    "has_prev": False,
                },
                "status_counts": {
                    "PENDING": 10,
                    "PROCESSING": 5,
                    "PREPROCESSED": 5,
                    "PROCESSED": 130,
                    "FAILED": 5,
                },
            }
        }
    )


class StatusCountsResponse(BaseModel):
    """Response model for document status counts

    Attributes:
        status_counts: Count of documents by status
    """

    status_counts: Dict[str, int] = Field(description="Count of documents by status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_counts": {
                    "PENDING": 10,
                    "PROCESSING": 5,
                    "PREPROCESSED": 5,
                    "PROCESSED": 130,
                    "FAILED": 5,
                }
            }
        }
    )


class SupportedFileTypesResponse(BaseModel):
    """Response model for the upload allowlist and parser capability matrix

    Attributes:
        supported_extensions: Dotted lowercase suffixes accepted for a bare
            (unhinted) filename under the current parser routing rules
        engines: Usable parser engine -> dotted suffixes it can parse, for
            pre-validating filenames that carry a ``[engine]`` hint
    """

    supported_extensions: List[str] = Field(
        description="Suffixes accepted for an unhinted filename (dotted, lowercase)"
    )
    engines: Dict[str, List[str]] = Field(
        description="Usable parser engine -> dotted suffixes it can parse"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "supported_extensions": [".jpg", ".md", ".pdf", ".txt"],
                "engines": {
                    "legacy": [".md", ".pdf", ".txt"],
                    "mineru": [".jpg", ".pdf", ".png"],
                    "native": [".docx", ".md", ".textpack"],
                },
            }
        }
    )


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status

    Attributes:
        busy: Whether the pipeline is currently busy
        job_name: Current job name (e.g., indexing files/indexing texts)
        job_start: Job start time as ISO format string with timezone (optional)
        docs: Total number of documents to be indexed
        batchs: Number of batches for processing documents
        cur_batch: Current processing batch
        latest_message: Latest message from pipeline processing
        history_messages: List of history messages
        update_status: Status of update flags for all namespaces
        recovery_required: Whether the workspace is fenced pending recovery
            (every mutation is refused with 503 until it is cleared)
        recovery_kind: Coarse cause of the fence, when one is set
        recovery_message: Operator-facing explanation of the fence, including the
            bounded blocker sample where the cause provides one
    """

    busy: bool = False
    job_name: str = "Default Job"
    job_start: Optional[str] = None
    docs: int = 0
    batchs: int = 0
    cur_batch: int = 0
    latest_message: str = ""
    history_messages: Optional[List[str]] = None
    update_status: Optional[dict] = None
    # Sanitized fence projection (see shared_storage.describe_recovery_fence).
    # The raw ``recovery_required`` record stays internal — it sits alongside
    # owner records carrying PIDs and reservation tokens — but an operator still
    # needs a read-only way to see that the workspace is fenced and why.
    recovery_required: bool = False
    recovery_kind: Optional[str] = None
    recovery_message: Optional[str] = None

    @field_validator("job_start", mode="before")
    @classmethod
    def parse_job_start(cls, value):
        """Process datetime and return as ISO format string with timezone"""
        return format_datetime(value)

    model_config = ConfigDict(extra="allow")


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
        workspace: str = "",  # New parameter for workspace isolation
    ):
        # Reject path traversal before using workspace in the upload path
        validate_workspace(workspace)
        # Store the base input directory and workspace
        self.base_input_dir = Path(input_dir)
        self.workspace = workspace
        self.indexed_files = set()

        # Create workspace-specific input directory
        # If workspace is provided, create a subdirectory for data isolation
        if workspace:
            self.input_dir = self.base_input_dir / workspace
        else:
            self.input_dir = self.base_input_dir

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supported_extensions(self) -> tuple:
        """Suffixes accepted for an unhinted filename, derived live.

        A suffix is advertised only when it is *routable without extra
        directives*: the engine that ``resolve_file_parser_engine`` picks for
        a bare ``x.<suffix>`` (filename hint absent; ``LIGHTRAG_PARSER``
        rules + default apply) must itself support the suffix. This keeps
        "uploadable" aligned with "will actually parse": e.g. mineru's
        ``png`` joins only when its endpoint is configured AND a routing
        rule (or per-file hint, see ``is_supported_file``) sends pngs to it
        — otherwise the default ``legacy`` engine would fail the suffix gate
        at the parse stage. A default deployment equals the local engines'
        (legacy ∪ native) types; no hardcoded list to keep in sync.
        """
        from lightrag.parser.registry import available_engine_suffixes
        from lightrag.parser.routing import (
            parser_engine_supports_suffix,
            resolve_file_parser_engine,
        )

        out = []
        for s in sorted(available_engine_suffixes()):
            engine = resolve_file_parser_engine(f"x.{s}")
            if parser_engine_supports_suffix(engine, s):
                out.append(f".{s}")
        return tuple(out)

    @property
    def engine_capabilities(self) -> Dict[str, List[str]]:
        """Usable engine -> dotted suffixes it can parse, derived live.

        Covers user-selectable engines whose endpoint (if any) is configured
        — the same gate ``resolve_parser_directives`` applies to a filename
        hint. Lets the WebUI pre-validate hinted names (``img.[mineru].png``)
        locally instead of uploading the file only to receive a 400.
        """
        from lightrag.parser.registry import (
            engine_endpoint_configured,
            suffix_capabilities,
            supported_parser_engines,
        )

        return {
            engine: [f".{s}" for s in sorted(suffix_capabilities(engine))]
            for engine in sorted(supported_parser_engines())
            if engine_endpoint_configured(engine)
        }

    def iter_new_files(self) -> Iterator[Path]:
        """Yield new, routable input files ONE AT A TIME (LR2 §8.2).

        A single streaming pass: one ``iterdir()`` over the input directory, no
        whole-directory list and no whole-directory sort, so the scan's peak
        memory is set by its enqueue batch (``SCAN_ENQUEUE_BATCH_SIZE``) rather
        than by how many files the directory holds. An interrupted scan needs no
        in-memory resume state — the next scan re-discovers, and the persistent
        ``doc_status`` rows are the deduplication authority.

        Files are admitted on the *available* engine suffix surface (so a
        hint-carrying file like ``img.[mineru].png`` is discoverable even when
        bare ``.png`` is not advertised), then narrowed to those whose resolved
        engine actually supports them (``is_supported_file``).

        Deliberately unordered (LR2 §8.5): candidates are written one at a time
        to the disk-backed scan spool, whose mtime index supplies global order
        without a whole-directory Python list. The old hint-preferring
        group-by-canonical-name pass was exactly the O(files-in-directory)
        memory structure this replaces.
        """
        from lightrag.parser.registry import available_engine_suffixes
        from lightrag.parser.routing import FilenameParserHintError

        suffixes = {f".{s}" for s in available_engine_suffixes()}
        logger.debug(f"Streaming scan of {self.input_dir} for {len(suffixes)} suffixes")
        for file_path in self.input_dir.iterdir():
            # Suffix comparison is case-sensitive, matching the per-suffix glob
            # this replaced; ``__parsed__`` and any other directory is skipped.
            if file_path.suffix not in suffixes or not file_path.is_file():
                continue
            if file_path in self.indexed_files:
                continue
            try:
                if not self.is_supported_file(file_path.name):
                    continue
            except FilenameParserHintError:
                # Malformed hint: pass the file through — the enqueue path
                # reports a detailed error document, instead of the scan
                # silently ignoring the user's file.
                pass
            yield file_path

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        """True when THIS filename routes to an engine that can parse it.

        Resolves the engine for the concrete name — so a per-file hint
        (``img.[mineru].png``) is honoured — and checks the resolved engine
        supports the suffix. A bare suffix that would fall through to the
        default ``legacy`` engine is rejected here instead of failing later
        at the parse worker's suffix gate.

        Raises :class:`FilenameParserHintError` for a malformed hint —
        callers surface it (upload → HTTP 400 with the detailed message;
        scan passes the file through so enqueue emits an error document).
        """
        from lightrag.parser.routing import (
            parser_engine_supports_suffix,
            parser_suffix,
            resolve_file_parser_engine,
        )

        engine = resolve_file_parser_engine(filename)
        return parser_engine_supports_suffix(engine, parser_suffix(filename))


def validate_file_path_security(file_path_str: str, base_dir: Path) -> Optional[Path]:
    """
    Validate file path security to prevent Path Traversal attacks.

    Args:
        file_path_str: The file path string to validate
        base_dir: The base directory that the file must be within

    Returns:
        Path: Safe file path if valid, None if unsafe or invalid
    """
    if not file_path_str or not file_path_str.strip():
        return None

    try:
        # Clean the file path string
        clean_path_str = file_path_str.strip()

        # Check for obvious path traversal patterns before processing
        # This catches both Unix (..) and Windows (..\) style traversals
        if ".." in clean_path_str:
            # Additional check for Windows-style backslash traversal
            if (
                "\\..\\" in clean_path_str
                or clean_path_str.startswith("..\\")
                or clean_path_str.endswith("\\..")
            ):
                # logger.warning(
                #     f"Security violation: Windows path traversal attempt detected - {file_path_str}"
                # )
                return None

        # Normalize path separators (convert backslashes to forward slashes)
        # This helps handle Windows-style paths on Unix systems
        normalized_path = clean_path_str.replace("\\", "/")

        # Create path object and resolve it (handles symlinks and relative paths)
        candidate_path = (base_dir / normalized_path).resolve()
        base_dir_resolved = base_dir.resolve()

        # Check if the resolved path is within the base directory
        if not candidate_path.is_relative_to(base_dir_resolved):
            # logger.warning(
            #     f"Security violation: Path traversal attempt detected - {file_path_str}"
            # )
            return None

        return candidate_path

    except (OSError, ValueError, Exception) as e:
        logger.warning(f"Invalid file path detected: {file_path_str} - {str(e)}")
        return None


def get_doc_status_value(doc_status: Any) -> str:
    """Read status from dict or DocProcessingStatus-like objects."""
    status = (
        doc_status.get("status")
        if isinstance(doc_status, dict)
        else getattr(doc_status, "status", None)
    )
    if isinstance(status, DocStatus):
        return status.value
    return str(status or "")


def get_doc_track_id(doc_status: Any) -> str:
    """Read track_id from dict or DocProcessingStatus-like objects."""
    track_id = (
        doc_status.get("track_id")
        if isinstance(doc_status, dict)
        else getattr(doc_status, "track_id", None)
    )
    return str(track_id or "")


async def get_existing_doc_by_file_path_candidates(
    doc_status: Any, file_path: Path | str
) -> dict[str, Any] | None:
    """Find an existing document by canonical basename."""
    basename = normalize_file_path(str(file_path))
    if basename == UNKNOWN_FILE_SOURCE:
        return None
    match = await doc_status.get_doc_by_file_basename(basename)
    if not match:
        return None
    _, existing_doc_data = match
    return existing_doc_data


def _admission_capacity(rag: LightRAG) -> int:
    """``MAX_PENDING_DOCUMENTS`` for this instance; 0 disables admission."""
    capacity = getattr(rag, "max_pending_documents", 0)
    return capacity if isinstance(capacity, int) and capacity > 0 else 0


async def _strict_active_count(rag: LightRAG) -> int:
    """Strict count of admission-relevant documents, or 503.

    Fail-closed: a backend that cannot count (capability gap, control-plane
    failure) must not be read as "there is room" — that is exactly how an
    admission cap turns into an unbounded backlog.
    """
    try:
        return await count_active_documents(rag.doc_status)
    except Exception as count_error:
        logger.error(f"Admission active-document count failed: {count_error}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot determine how many documents are in flight, so new work "
                "cannot be admitted. Retry once document storage is healthy."
            ),
        )


def _backpressure_http_error(error: PipelineBackpressureError) -> HTTPException:
    """Render admission backpressure as 429 with the numbers a client needs."""
    return HTTPException(
        status_code=429,
        detail=(
            f"Pipeline is at capacity: {error.current} document(s) already "
            f"active or reserved, {error.requested} requested, capacity "
            f"{error.capacity}. Retry once documents finish processing."
        ),
        headers={"Retry-After": str(_ADMISSION_RETRY_AFTER_SECONDS)},
    )


# Pipeline states that refuse a NEW ordinary ingress reservation. Re-weighting a
# reservation taken before one of these appeared is deliberately NOT refused: a
# request admitted before a freeze is allowed to finish (LR2 §9.2).
_INGRESS_FENCES: tuple[tuple[str, str], ...] = (
    (
        "scanning_exclusive",
        "Document scan is classifying files. Wait for the classification "
        "phase to finish before submitting new work.",
    ),
    (
        "destructive_busy",
        "Pipeline is clearing or deleting documents. Wait for the running "
        "job to finish before submitting new work.",
    ),
    (
        "manual_freeze_requested",
        "A retry of failed documents is draining the pipeline. Wait for it "
        "to finish before submitting new work.",
    ),
)


async def _reserve_enqueue_slot(
    rag: LightRAG,
    token: str,
    *,
    weight: int = 1,
    reject_when: tuple[tuple[str, str], ...] = _INGRESS_FENCES,
) -> bool:
    """Atomically check exclusive-writer state and reserve a
    pending-enqueue slot keyed by ``token``, weighted for admission.

    ``token`` is generated by the caller before any await and added to the
    ``pending_enqueue_tokens`` set (whose length mirrors ``pending_enqueues``),
    so the release is idempotent and owner-identified — an endpoint and its
    background task may both release without over-decrementing a sibling's slot,
    and a dead owner's token can later be reaped.

    Concurrent enqueues are permitted while the processing loop is
    running — the loop is notified via the ingress mailbox and picks up
    newly-enqueued docs mid-batch or at the batch boundary.  This includes the
    scan task's processing phase: once classification is done, the
    scan transitions to driving the processing pipeline like any
    other enqueuer, and uploads can land alongside it.

    Three states block new uploads/inserts:

    - ``scanning_exclusive``: scan task is in its CLASSIFICATION
      phase — reading doc_status to classify files (PROCESSED →
      archive, FAILED-without-full_docs → retry-as-new, etc.) and
      possibly deleting stale stubs.  Concurrent enqueue would race
      against scan's reads / stub deletions.  ``scanning`` alone
      (the processing phase) does NOT block uploads.
    - ``destructive_busy``: a /documents/clear or per-doc delete is in
      flight.  These DROP storages and remove input files; an enqueue
      accepted in this window would write to a storage that is being
      torn down and silently lose the document after the client saw
      success.
    - ``manual_freeze_requested``: a manual retry (``/reprocess_failed``
      or ``/scan``) has frozen new ingress while it drains the pipeline
      to idle and exclusively resets FAILED→PENDING (LR2 §6.1/§7.2).
      An enqueue reserved BEFORE the freeze is allowed to finish; only
      new reservations are refused, for the bounded freeze window.

    ``pending_enqueues`` is incremented so the scan endpoint can refuse
    while bg tasks are mid-enqueue.  The counter does NOT gate
    ``apipeline_process_enqueue_documents`` — concurrent processing is
    explicitly allowed and is what makes "upload while pipeline is
    busy" possible.

    A workspace whose ``pipeline_status`` has never been initialised
    (mocked test rigs) is treated as idle; no slot is reserved.

    With ``MAX_PENDING_DOCUMENTS > 0`` the same critical section enforces the
    admission capacity (LR2 §9.1): ``weight`` documents are charged against the
    strict active count plus every other in-flight reservation's weight, and an
    over-capacity request is refused with 429 *before* the file is written to
    disk. ``weight`` is the caller's best estimate at reservation time (1 for
    ``/upload`` and ``/text``); the enqueue guard later re-weights the same token
    to the deduped count, and ``/texts`` adjusts it once the body is parsed.

    Returns:
        True when a slot was reserved (caller MUST pair with
        ``_release_enqueue_slot``); False when pipeline_status is not
        bootstrapped.

    Raises:
        HTTPException(409): when
            ``pipeline_status['scanning_exclusive']``,
            ``pipeline_status['destructive_busy']`` or
            ``pipeline_status['manual_freeze_requested']`` is set.
        HTTPException(429): when the admission capacity is exhausted.
        HTTPException(503): when the strict active count cannot be taken —
            admission fails closed rather than assuming there is room.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import (
        PipelineReservationConflict,
        acquire_enqueue_reservation,
        get_namespace_data,
        get_namespace_lock,
    )

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return False
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    capacity = _admission_capacity(rag)
    active_count = await _strict_active_count(rag) if capacity > 0 else None
    try:
        result = await acquire_enqueue_reservation(
            pipeline_status,
            pipeline_status_lock,
            token=token,
            reject_when=reject_when,
            weight=weight,
            capacity=capacity,
            active_count=active_count,
        )
    except PipelineBackpressureError as backpressure:
        raise _backpressure_http_error(backpressure)
    if not result.acquired:
        raise HTTPException(
            status_code=(
                503
                if result.conflict is PipelineReservationConflict.RECOVERY_REQUIRED
                else 409
            ),
            detail=result.message,
        )
    return True


async def check_pipeline_busy_or_raise(rag: LightRAG) -> None:
    """Refuse the request with HTTP 409 when the document pipeline is busy.

    Intended for short, fine-grained graph mutations (entity/relation
    edit/create/delete/merge). Reads ``pipeline_status['busy']`` under
    the namespace lock and raises immediately on contention -- it does
    NOT set any flag, so it cannot block the pipeline itself.

    ``busy`` is set by the processing loop and by destructive jobs
    (``/documents/clear`` / per-doc delete). Both paths concurrently
    write the same graph storages that these endpoints mutate, so a
    409 here mirrors the existing UI guard and tells clients to wait.

    A narrow race remains between this check and the underlying graph
    write: if the pipeline transitions to busy in that window, the
    per-edge/-node locks inside the storage layer are the last line of
    defense. That trade-off is deliberate -- holding ``busy`` here
    would serialise every UI edit against document ingestion, which is
    a worse user-visible failure mode than tolerating the race.

    No-op (returns silently) when ``pipeline_status`` was never
    bootstrapped, matching the behaviour of ``_acquire_destructive_busy``
    so test rigs without a real shared-storage Manager keep working.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import (
        PipelineReservationConflict,
        check_pipeline_status_mutation,
        get_namespace_data,
        get_namespace_lock,
    )

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    result = await check_pipeline_status_mutation(
        pipeline_status,
        pipeline_status_lock,
        reject_when=(
            (
                "busy",
                "Pipeline is busy with another operation. Wait for the running "
                "job to finish before editing the knowledge graph.",
            ),
        ),
    )
    if not result.acquired:
        raise HTTPException(
            status_code=(
                503
                if result.conflict is PipelineReservationConflict.RECOVERY_REQUIRED
                else 409
            ),
            detail=result.message,
        )


async def _acquire_destructive_busy(
    rag: LightRAG, token: str, *, kind: str, operation_record: dict
) -> tuple[bool, str | None]:
    """Atomically reserve the destructive busy slot for ``/documents/clear``
    or ``/documents/delete_document``.

    ``kind`` (``"clear"`` / ``"delete"``) and ``operation_record`` (a snapshot of
    the target, e.g. ``{"kind": "delete", "doc_ids": [...]}``) are stamped with
    the owner so a dead owner can be fenced (``recovery_required``) — clear/delete
    may partially drop storages / files and are not safe to blindly re-run.

    ``token`` is generated by the caller BEFORE any await and stored as
    ``busy_owner`` so the caller's finally can release the slot by owner even if
    this acquire is cancelled at the lock exit (it never returns the token).

    Both jobs DROP storages and (for clear) remove input files.  They
    must serialise against:

    - any other ``busy`` work (processing loop, another destructive job),
    - an in-flight ``scanning`` task that reads/writes doc_status and
      INPUT/, and
    - any ``pending_enqueues`` reservation whose bg task has not yet
      written to doc_status — accepting the destructive job in that
      window would drop storages while the enqueue is mid-write,
      losing a document the client already saw success for.

    All three checks happen inside a single ``pipeline_status_lock``
    critical section together with the flag write, so a concurrent
    enqueue/scan reservation cannot squeeze past us.

    Caller is responsible for clearing both flags in its finally block.

    Returns:
        (acquired, reason).  ``acquired=True`` and ``reason=None`` on
        success.  ``acquired=False`` with a human-readable ``reason``
        when another writer has the lock; the caller surfaces this to
        the client (HTTP 200 with status="busy" for these endpoints).

        For test rigs where ``pipeline_status`` was never bootstrapped,
        returns (True, None) — there is nothing to coordinate against.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import (
        acquire_reservation,
        get_namespace_data,
        get_namespace_lock,
    )

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return True, None
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    result = await acquire_reservation(
        pipeline_status,
        pipeline_status_lock,
        owner_key="busy_owner",
        owner=token,
        owner_kind=kind,
        flags={
            "busy": True,
            "destructive_busy": True,
            "operation_record": operation_record,
        },
        reject_when=(
            ("busy", "Pipeline is busy with another operation."),
            (
                "scanning",
                "Document scan is in progress. Wait for the scan to complete "
                "before clearing or deleting.",
            ),
            (
                "pending_enqueues",
                "An upload/insert or a source-conflict repair is in flight. "
                "Wait for it to complete before clearing or deleting.",
            ),
        ),
    )
    return result.acquired, result.message


def _release_destructive_action(status) -> None:
    status.update(
        {
            "busy": False,
            "destructive_busy": False,
            "busy_owner": None,
            "operation_record": None,
        }
    )


async def _release_destructive_busy(rag: LightRAG, token: str) -> None:
    """Release the destructive busy slot acquired by
    ``_acquire_destructive_busy``.  Never raises (except a re-raised
    cancellation after the release has completed).

    Owner-checked by ``token`` (a no-op if a later holder took the slot) and
    cancellation-resistant. Distinct from ``_release_enqueue_slot``: that helper
    clears ``pending_enqueues`` (the upload/insert reservation), this one clears
    ``busy + destructive_busy`` (the clear/delete reservation).

    The namespace fetch runs INSIDE ``run_to_completion`` (via
    ``release_owned_reservation``) so a cancellation delivered during the fetch
    or lock — e.g. a shutdown cancelling this background release — is
    retried/completed rather than leaking the slot.
    """
    from lightrag.kg.shared_storage import release_owned_reservation

    await release_owned_reservation(
        rag.workspace,
        owner_key="busy_owner",
        token=token,
        action=_release_destructive_action,
    )


def _adopt_or_new_enqueue_token(http_request: Any) -> tuple[str, bool]:
    """Resolve this request's pending-enqueue token.

    Returns ``(token, already_reserved)``. When the ASGI admission middleware ran
    (LR2 §9.3) it already reserved a slot before the body was read, so the route
    ADOPTS that token — reserving a second one would charge the same request
    twice. Otherwise the route mints one and reserves it itself, which is what
    keeps admission correct with the middleware absent (capacity disabled, a
    non-HTTP caller, or a test rig invoking the endpoint directly).

    Synchronous on purpose: callers need the token before entering the ``try``
    whose ``finally`` releases it.
    """
    ticket = adopt_admission_ticket(http_request)
    if ticket is not None:
        return ticket.token, True
    return uuid4().hex, False


async def _reweight_enqueue_slot(rag: LightRAG, token: str, weight: int) -> None:
    """Re-charge an already-held reservation for ``weight`` documents.

    ``/texts`` reserves before the body is parsed (it must, to keep the
    admission decision atomic) and only then learns how many documents the
    request carries. Re-registering the same token replaces its weight; the
    token's own previous weight is excluded from the capacity sum, so this is a
    replacement rather than an addition.

    A no-op when admission is disabled or ``pipeline_status`` was never
    bootstrapped (mocked rigs).

    Raises:
        HTTPException(429): the request does not fit at its true size.
        HTTPException(503): the strict active count failed.
    """
    if _admission_capacity(rag) <= 0:
        return
    await _reserve_enqueue_slot(rag, token, weight=weight, reject_when=())


async def _release_enqueue_slot(rag: LightRAG, token: str) -> None:
    """Release the slot ``token`` reserved by ``_reserve_enqueue_slot``.

    Removes ``token`` from ``pending_enqueue_tokens`` and mirrors the count into
    ``pending_enqueues`` in a single atomic update. Idempotent (a no-op if the
    token is absent, so an endpoint and its background task may both release)
    and cancellation-resistant. The bg task itself drives processing via
    ``apipeline_process_enqueue_documents`` after enqueue; no cross-task drain
    coordination is needed. Never raises (except a re-raised cancellation after
    the release has completed).

    The namespace fetch runs INSIDE ``run_to_completion`` (via
    ``release_token_set_reservation``) so a cancellation delivered during the
    fetch or lock — e.g. a shutdown cancelling this background release — is
    retried/completed rather than leaking the slot.
    """
    from lightrag.kg.shared_storage import release_token_set_reservation

    await release_token_set_reservation(
        rag.workspace,
        tokens_key="pending_enqueue_tokens",
        token=token,
    )


def _release_scanning_action(status) -> None:
    status.update(
        {"scanning": False, "scanning_exclusive": False, "scanning_owner": None}
    )


async def _release_scanning_reservation(rag: LightRAG, token: str) -> None:
    """Release the scanning slot reserved by the ``/documents/scan`` endpoint.

    Owner-checked by ``token`` (a no-op if a later scan took the slot) and
    cancellation-resistant. Used by the endpoint's finally to cover the window
    between reserving ``scanning``/``scanning_exclusive`` and handing the task
    off to ``run_scanning_process`` (which otherwise owns the release). Never
    raises (except a re-raised cancellation after the release has completed).

    The namespace fetch runs INSIDE ``run_to_completion`` (via
    ``release_owned_reservation``) so a cancellation delivered during the fetch
    or lock — e.g. a shutdown cancelling ``run_scanning_process`` — is
    retried/completed rather than leaking the slot.
    """
    from lightrag.kg.shared_storage import release_owned_reservation

    await release_owned_reservation(
        rag.workspace,
        owner_key="scanning_owner",
        token=token,
        action=_release_scanning_action,
    )


def get_managed_background_tasks(request: Request) -> set:
    """FastAPI dependency returning the app's managed background-task set.

    Reservation-holding background work (scan / delete / enqueue / reprocess) is
    started via ``start_reserved_background_task`` into this set — a real,
    tracked ``asyncio`` task rather than a Starlette ``BackgroundTasks`` callback
    (which runs only AFTER the response body is sent and is not tracked, so a
    request cancelled mid-send would drop the callback and strand the
    reservation). The lifespan drains this set on shutdown so every child's
    finally releases its reservation before shared state is torn down.

    Direct-call tests pass a plain ``set()`` for this argument.
    """
    return request.app.state.background_tasks


def find_existing_file_by_file_path(input_dir: Path, file_path: str) -> Path | None:
    """Find an input-dir file whose canonical basename matches ``file_path``.

    Callers pass the stored canonical ``file_path`` (already hint-stripped);
    on-disk filenames are normalized before comparison so a hint-bearing
    variant on disk still matches a canonical stored ``file_path``.
    """
    if not file_path or file_path == UNKNOWN_FILE_SOURCE:
        return None
    try:
        for candidate in input_dir.iterdir():
            if not candidate.is_file():
                continue
            if normalize_file_path(candidate.name) == file_path:
                return candidate
    except FileNotFoundError:
        return None
    return None


def canonicalize_archived_file_variant_basename(
    file_path: Path | str, *, strip_archive_suffix: bool = False
) -> str:
    """Canonical basename for original files and numbered archive variants."""
    name = Path(file_path).name
    path = Path(name)
    stem = (
        ARCHIVED_FILE_SUFFIX_RE.sub("", path.stem)
        if strip_archive_suffix
        else path.stem
    )
    return normalize_file_path(f"{stem}{path.suffix}")


def _file_path_for_parsed_artifact_dir(dir_name: str) -> str | None:
    """Return the canonical source basename for a parser artifact dir.

    Recognized layouts (suffix list in
    :data:`lightrag.constants.PARSED_ARTIFACT_DIR_SUFFIXES`):

    - ``<basename>.parsed[_NNN]/``        — sidecar output (every engine)
    - ``<basename>.mineru_raw[_NNN]/``    — MinerU preserved raw bundle
    - ``<basename>.docling_raw[_NNN]/``   — Docling preserved raw bundle

    Raw bundles are preserved across re-parses for cache reuse and on-demand
    diagnostics; they are cleaned only when the user deletes the document
    with ``delete_file=True`` so the raw artifacts and source file go away
    together.
    """
    stripped = ARCHIVED_FILE_SUFFIX_RE.sub("", dir_name)
    for suffix in PARSED_ARTIFACT_DIR_SUFFIXES:
        if stripped.endswith(suffix):
            basename = stripped[: -len(suffix)]
            if basename:
                return normalize_file_path(basename)
    return None


def delete_file_variants_by_file_path(
    input_dir: Path,
    file_path: str | None,
) -> tuple[list[str], list[str]]:
    """Delete input/__parsed__ source files matching a canonical ``file_path``."""
    if not file_path:
        return [], []
    canonical = normalize_file_path(file_path)
    if canonical == UNKNOWN_FILE_SOURCE:
        return [], []
    canonical_names = {canonical}

    deleted_files: list[str] = []
    errors: list[str] = []
    candidate_dirs = [input_dir, input_dir / PARSED_DIR_NAME]
    input_dir_resolved = input_dir.resolve()

    for candidate_dir in candidate_dirs:
        try:
            candidates = list(candidate_dir.iterdir())
        except FileNotFoundError:
            continue
        except Exception as e:
            errors.append(f"Failed to scan {candidate_dir}: {e}")
            continue

        in_parsed_dir = candidate_dir.name == PARSED_DIR_NAME
        for candidate in candidates:
            if candidate.is_file():
                if (
                    canonicalize_archived_file_variant_basename(
                        candidate.name,
                        strip_archive_suffix=in_parsed_dir,
                    )
                    not in canonical_names
                ):
                    continue

                safe_candidate = validate_file_path_security(
                    candidate.name, candidate_dir
                )
                if safe_candidate is None:
                    errors.append(f"Unsafe file path skipped: {candidate.name}")
                    continue

                try:
                    safe_candidate.unlink()
                    deleted_files.append(
                        str(safe_candidate.relative_to(input_dir_resolved))
                    )
                except Exception as e:
                    errors.append(f"Failed to delete {candidate.name}: {e}")
                continue

            if in_parsed_dir and candidate.is_dir():
                canonical_for_dir = _file_path_for_parsed_artifact_dir(candidate.name)
                if (
                    canonical_for_dir is None
                    or canonical_for_dir not in canonical_names
                ):
                    continue

                safe_candidate = validate_file_path_security(
                    candidate.name, candidate_dir
                )
                if safe_candidate is None:
                    errors.append(f"Unsafe artifact dir skipped: {candidate.name}")
                    continue

                try:
                    shutil.rmtree(safe_candidate)
                    deleted_files.append(
                        str(safe_candidate.relative_to(input_dir_resolved))
                    )
                except Exception as e:
                    errors.append(
                        f"Failed to delete artifact dir {candidate.name}: {e}"
                    )

    return deleted_files, errors


async def record_scan_warning(rag: LightRAG, message: str) -> None:
    logger.warning(message)
    try:
        from lightrag.kg import shared_storage

        if not getattr(shared_storage, "_initialized", False):
            return

        workspace = getattr(rag, "workspace", "")
        pipeline_status = await shared_storage.get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        pipeline_status_lock = shared_storage.get_namespace_lock(
            "pipeline_status", workspace=workspace
        )
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = message
            append_pipeline_history(pipeline_status, message)
    except Exception:
        pass


# Legacy text extractors moved to lightrag.parser.legacy.extractors; the
# legacy engine now extracts at the worker stage (LegacyParser), not here.


class _ScanCandidate(NamedTuple):
    """One spooled scan candidate: its path plus the size discovery already read.

    Carrying the size forward keeps the enqueue phase from re-``stat``-ing a
    file the spool just stat'ed — INPUT_DIR is frequently a network mount, where
    that second metadata round-trip is the expensive part.

    ``size`` is never ``None``: a candidate whose stat failed carries ``0``, the
    same "size unknown" value ``pipeline_enqueue_file`` has always reported for
    an unreadable file.  Keeping it an ``int`` is what makes "one metadata read
    per candidate" true unconditionally — ``None`` would mean "caller supplied
    nothing" and send the enqueue path back to the filesystem for exactly the
    vanished files where the second read is most likely to be slow.
    """

    path: Path
    size: int


async def pipeline_enqueue_file(
    rag: LightRAG,
    file_path: Path,
    track_id: str = None,
    from_scan: bool = False,
    admission_token: str | None = None,
    known_file_size: int | None = None,
) -> tuple[bool, str]:
    """Add a file to the queue for processing

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID, if not provided will be generated
        admission_token: the caller's pending-enqueue reservation, forwarded to
            the admission guard so it re-weights that token rather than
            registering a second one (LR2 §9.2)
        from_scan: True only when invoked by the scan-owned background task,
            which already holds ``pipeline_status["scanning"]``.  Forwarded to
            ``apipeline_enqueue_documents`` so the scan can enqueue the files
            it just discovered without tripping the scanning guard there.
        known_file_size: the size the caller already stat'ed, reused instead of
            issuing a second metadata read.  Scan supplies the value its
            candidate spool recorded at discovery time; it feeds error reports
            only, so a size that went stale between discovery and enqueue costs
            nothing.
    Returns:
        tuple: (success: bool, track_id: str)
    """

    # Generate track_id if not provided
    if track_id is None:
        track_id = generate_track_id("unknown")

    try:
        # File size is used only for error reporting. Scan-time mtime ordering
        # happens before this function, in the disk-backed candidate spool;
        # doc_status.created_at always remains the actual first-persist time.
        if known_file_size is not None:
            file_size = known_file_size
        else:
            try:
                stat = await asyncio.to_thread(file_path.stat)
                file_size = stat.st_size
            except Exception:
                file_size = 0

        try:
            directives = resolve_parser_directives(file_path)
        except FilenameParserHintError as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": FILE_EXTRACTION_SUMMARY_PREFIX
                    + "Filename hint error",
                    "original_error": str(e),
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Invalid filename hint in {file_path.name}: {e}"
            )
            return False, track_id

        extraction_engine = directives.engine
        process_options = directives.process_options
        api_process_options = process_options or PROCESS_OPTION_CHUNK_FIXED

        # Overlay any per-file chunk parameters (from the filename hint or a
        # LIGHTRAG_PARSER rule) onto the active strategy's chunk_options so the
        # parse worker chunks this document with them. Absent params keep the
        # legacy path (chunk_options built at enqueue time from addon_params).
        hint_chunk_options = None
        active_strategy = parse_process_options(api_process_options).chunking
        hint_chunk_params = directives.chunk_params.get(active_strategy)
        if hint_chunk_params:
            try:
                strategy_key = chunk_strategy_key(api_process_options)
                hint_chunk_options = resolve_chunk_options(
                    rag.addon_params, process_options=api_process_options
                )
                hint_chunk_options[strategy_key].update(hint_chunk_params)
                _validate_effective_chunk_overlap(
                    hint_chunk_options, strategy_key, strategy_key
                )
            except ValueError as e:
                error_files = [
                    {
                        "file_path": str(file_path.name),
                        "error_description": FILE_EXTRACTION_SUMMARY_PREFIX
                        + "Chunk parameter error",
                        "original_error": str(e),
                        "file_size": file_size,
                    }
                ]
                await rag.apipeline_enqueue_error_documents(error_files, track_id)
                logger.error(
                    f"[File Extraction]Invalid chunk parameters in "
                    f"{file_path.name}: {e}"
                )
                return False, track_id
        # All engines defer parsing to the worker stage: the file is already
        # saved on disk, so we enqueue PENDING_PARSE with the chosen engine.
        # Legacy now extracts at the worker (LegacyParser) instead of eagerly
        # here, so every engine shares one ingestion path.
        # Encode any per-file engine params into the parse_engine field
        # (e.g. "mineru(page_range=1-3,language=en)") so they ride the existing
        # persisted column to the parse worker. Bare engine when there are none.
        parse_engine_field = encode_parse_engine(
            extraction_engine, directives.engine_params
        )
        try:
            enqueue_kwargs = {
                "file_paths": str(file_path),
                "track_id": track_id,
                "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                "parse_engine": parse_engine_field,
                "process_options": api_process_options,
                "from_scan": from_scan,
            }
            if admission_token is not None:
                # Only sent when the caller actually holds a reservation; None
                # is the enqueue's own default and adding it would be noise.
                enqueue_kwargs["admission_token"] = admission_token
            if hint_chunk_options is not None:
                enqueue_kwargs["chunk_options"] = hint_chunk_options
            enqueue_result = await rag.apipeline_enqueue_documents("", **enqueue_kwargs)
            if enqueue_result is None:
                try:
                    await move_file_to_parsed_dir(file_path)
                except Exception as move_error:
                    logger.error(
                        f"Failed to move duplicate file {file_path.name} to {PARSED_DIR_NAME} directory: {move_error}"
                    )
                return False, track_id
            logger.info(
                f"[File Extraction]Deferred {file_path.name} to {extraction_engine} parser"
            )
            return True, track_id
        except Exception as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": FILE_EXTRACTION_SUMMARY_PREFIX
                    + "Parser enqueue error",
                    "original_error": f"Failed to enqueue file for parser: {str(e)}",
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Error enqueuing {file_path.name} for {extraction_engine}: {str(e)}"
            )
            return False, track_id

    except Exception as e:
        # Catch-all for any unexpected errors
        try:
            file_size = file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            file_size = 0

        error_files = [
            {
                "file_path": str(file_path.name),
                "error_description": "Unexpected processing error",
                "original_error": f"Unexpected error: {str(e)}",
                "file_size": file_size,
            }
        ]
        await rag.apipeline_enqueue_error_documents(error_files, track_id)
        logger.error(f"Enqueuing file {file_path.name} error: {str(e)}")
        logger.error(traceback.format_exc())
        return False, track_id
    finally:
        if file_path.name.startswith(temp_prefix):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")


async def pipeline_index_file(
    rag: LightRAG,
    file_path: Path,
    track_id: str = None,
    admission_token: str | None = None,
):
    """Index a file with track_id

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID
        admission_token: the endpoint's pending-enqueue reservation, forwarded
            so the admission guard re-weights THAT token to the deduped count
            instead of counting this request twice (LR2 §9.2)
    """
    try:
        success, _ = await pipeline_enqueue_file(
            rag, file_path, track_id, admission_token=admission_token
        )
        if success:
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        logger.error(f"Error indexing file {file_path.name}: {str(e)}")
        logger.error(traceback.format_exc())


async def pipeline_enqueue_scan_batch(
    rag: LightRAG,
    candidates: Sequence[_ScanCandidate],
    track_id: str = None,
) -> int:
    """Write ONE mtime-ordered, bounded scan batch — no processing drive (§8.2).

    Discovery first stages candidates in the disposable disk spool; its global
    mtime index emits batches in order while the scan still holds
    ``scanning_exclusive``. Processing runs exactly once afterwards, when the
    fence has dropped (§8.1). Enqueue is therefore separated from driving: a
    per-batch drive would be refused by that very fence and only set the
    deferred-processing flag.

    ``from_scan=True`` is implied — this path exists only for the scan-owned
    background task, whose own ``scanning`` flag would otherwise trip the guard
    inside ``apipeline_enqueue_documents``.

    Args:
        rag: LightRAG instance
        candidates: the batch's spooled candidates IN ENQUEUE ORDER, bounded by
            ``SCAN_ENQUEUE_BATCH_SIZE``
        track_id: tracking ID stamped on every document of this scan

    Returns:
        How many files actually landed a doc_status row. A per-file rejection
        (bad filename hint, empty body, content duplicate, ...) is reported by
        ``pipeline_enqueue_file`` as an error document / archive and does not
        abort the rest of the batch.
    """
    if not candidates:
        return 0
    enqueued = 0
    try:
        # Iterate in the order given. The disk spool emits candidates in global
        # ``(mtime, path)`` order and created_at=now() records that order into
        # doc_status one row at a time, so ANY re-sort here (the batch-local
        # pinyin sort this replaced included) silently breaks oldest-file-first.
        for candidate in candidates:
            success, _ = await pipeline_enqueue_file(
                rag,
                candidate.path,
                track_id,
                from_scan=True,
                known_file_size=candidate.size,
            )
            if success:
                enqueued += 1
        return enqueued
    except Exception as e:
        logger.error(f"Error enqueuing scan batch: {str(e)}")
        logger.error(traceback.format_exc())
        return enqueued


_STRATEGY_TO_PROCESS_OPTION: Dict[str, str] = {
    "fixed_token": PROCESS_OPTION_CHUNK_FIXED,
    "recursive_character": PROCESS_OPTION_CHUNK_RECURSIVE,
    "semantic_vector": PROCESS_OPTION_CHUNK_VECTOR,
    "paragraph_semantic": PROCESS_OPTION_CHUNK_PARAGRAH,
}


def _resolve_text_chunking(
    chunking: Optional[TextChunkingConfig], rag: LightRAG
) -> tuple[str, dict]:
    """Freeze a ``chunking`` request into ``(process_options, chunk_options)``.

    When ``chunking`` is ``None`` this reproduces today's behavior exactly:
    fixed-token strategy with the snapshot built from
    ``rag.addon_params['chunker']``.

    Otherwise the validated, strategy-specific params are merged into the
    selected strategy's sub-dict. ``chunk_token_size`` rides along inside
    ``params`` like any other key — every strategy (F included, after the
    ``process_single_document`` cleanup) reads its size from its own
    sub-dict, with the top-level snapshot value as the shared fallback.

    Raises:
        ValueError: when the request lowers ``chunk_token_size`` below the
            *effective* ``chunk_overlap_token_size``.  The overlap is often
            inherited from ``addon_params``/env (the overlay fills
            ``fixed_token``/``recursive_character``/``paragraph_semantic``
            overlap with ``CHUNK_*_OVERLAP_SIZE`` / ``CHUNK_OVERLAP_SIZE``),
            so this can only be checked here against the resolved snapshot,
            not in the request model.  Callers on the request path invoke
            this synchronously so the failure surfaces as HTTP 422 before any
            background work is scheduled.
    """
    if chunking is None:
        # No request-driven config: reproduce today's behavior verbatim,
        # including not introducing new validation on the default path.
        process_options = PROCESS_OPTION_CHUNK_FIXED
        return process_options, resolve_chunk_options(
            rag.addon_params, process_options=process_options
        )

    process_options = _STRATEGY_TO_PROCESS_OPTION[chunking.strategy]
    chunk_options = resolve_chunk_options(
        rag.addon_params, process_options=process_options
    )
    strategy_key = chunk_strategy_key(process_options)
    chunk_options[strategy_key].update(chunking.params)
    _validate_effective_chunk_overlap(chunk_options, strategy_key, chunking.strategy)
    _validate_effective_semantic_amount(chunk_options, strategy_key)
    return process_options, chunk_options


def _validate_effective_chunk_overlap(
    chunk_options: dict, strategy_key: str, strategy_name: str
) -> None:
    """Reject a resolved snapshot whose overlap is >= its chunk size.

    Operates on the fully-resolved ``chunk_options`` so it catches the case
    the request model cannot: ``chunk_token_size`` supplied in the request
    while ``chunk_overlap_token_size`` is inherited from addon_params/env
    (e.g. ``chunk_token_size=50`` with the default overlap ``100``).  The
    effective size is the strategy sub-dict value, falling back to the
    top-level snapshot size; the effective overlap is the sub-dict value
    (``semantic_vector`` carries none, so it is skipped).
    """
    sub = chunk_options.get(strategy_key) or {}
    # Fixed-token delimiter-only mode (split_by_character set AND
    # split_by_character_only=True) never applies overlap:
    # chunking_by_token_size only validates each delimiter segment against
    # chunk_token_size and raises on an oversize segment — the overlap field
    # is unused. Enforcing overlap < size there would wrongly 422 a valid
    # request such as paragraph splitting with a small chunk_token_size.
    # (split_by_character_only is itself a no-op when split_by_character is
    # falsy, so both must be effective for overlap to be skipped.)
    if (
        strategy_key == "fixed_token"
        and sub.get("split_by_character")
        and sub.get("split_by_character_only")
    ):
        return
    overlap = sub.get("chunk_overlap_token_size")
    if overlap is None:
        return
    size = sub.get("chunk_token_size")
    if size is None:
        size = chunk_options.get("chunk_token_size")
    if size is not None and overlap >= size:
        raise ValueError(
            f"chunking for strategy '{strategy_name}': effective "
            f"chunk_overlap_token_size ({overlap}) must be < chunk_token_size "
            f"({size}). The overlap is inherited from addon_params/env when "
            f"not set in the request; raise chunk_token_size or lower "
            f"chunk_overlap_token_size."
        )


def _validate_effective_semantic_amount(chunk_options: dict, strategy_key: str) -> None:
    """Reject a resolved semantic_vector snapshot whose breakpoint amount
    exceeds the percentile/gradient ceiling.

    Uses the *effective* ``breakpoint_threshold_type`` from the merged
    snapshot — the request model cannot, because the type may be inherited
    from ``addon_params``/``CHUNK_V_BREAKPOINT_THRESHOLD_TYPE`` while the
    request overrides only ``breakpoint_threshold_amount``. ``percentile`` /
    ``gradient`` feed ``np.percentile`` (q must be in ``[0, 100]``);
    ``standard_deviation`` / ``interquartile`` are multipliers with no upper
    bound, so a request amount > 100 is valid for them.
    """
    if strategy_key != "semantic_vector":
        return
    sub = chunk_options.get(strategy_key) or {}
    amt = sub.get("breakpoint_threshold_amount")
    if amt is None:
        return
    kind = sub.get("breakpoint_threshold_type") or "percentile"
    if kind in ("percentile", "gradient") and amt > 100:
        raise ValueError(
            f"chunking for strategy 'semantic_vector': "
            f"breakpoint_threshold_amount ({amt}) must be within (0, 100] for "
            f"breakpoint_threshold_type '{kind}'. The type is inherited from "
            f"addon_params/env when not set in the request."
        )


async def pipeline_index_texts(
    rag: LightRAG,
    texts: List[str],
    file_sources: List[str] = None,
    track_id: str = None,
    chunking: Optional[TextChunkingConfig] = None,
    admission_token: str | None = None,
):
    """Index a list of texts with track_id

    Args:
        rag: LightRAG instance
        texts: The texts to index
        file_sources: Sources of the texts
        track_id: Optional tracking ID
        chunking: Optional chunking strategy + params (already validated by
            the request model); when None, default fixed-token chunking is used
        admission_token: the endpoint's pending-enqueue reservation, forwarded so
            the admission guard re-weights that token to the deduped count
            (LR2 §9.2)
    """
    if not texts:
        return

    if not file_sources or len(file_sources) != len(texts):
        raise ValueError("A valid file source is required for each text")

    normalized_file_sources = [normalize_file_path(source) for source in file_sources]
    if any(source == UNKNOWN_FILE_SOURCE for source in normalized_file_sources):
        raise ValueError("A valid file source is required for each text")
    if len(set(normalized_file_sources)) != len(normalized_file_sources):
        raise ValueError("File sources must be unique by filename")

    process_options, chunk_options = _resolve_text_chunking(chunking, rag)
    enqueue_kwargs: dict[str, Any] = {
        "input": texts,
        "file_paths": normalized_file_sources,
        "track_id": track_id,
        "process_options": process_options,
        "chunk_options": chunk_options,
    }
    if admission_token is not None:
        # See pipeline_enqueue_file: only forwarded when a reservation exists.
        enqueue_kwargs["admission_token"] = admission_token
    await rag.apipeline_enqueue_documents(**enqueue_kwargs)
    await rag.apipeline_process_enqueue_documents()


# How long a scan may go without touching its job record before it is renewed
# for its own sake — by ``reporter.renew()`` inside the discovery loop, and by
# the independent ``_renew_scan_job_lease`` heartbeat during phases that never
# touch the record. A RUNNING job whose lease expires is reaped to ABANDONED
# (its owner presumed dead), so this must stay a fraction of the lease — a
# directory walk over a large tree, let alone a processing run, is easily
# longer than one lease period.
_SCAN_JOB_RENEW_SECONDS = SCAN_JOB_LEASE_SECONDS / 3


class _ScanJobReporter:
    """Client half of the bounded scan-job update protocol (LR2 §8.6).

    Submits ONLY ``count deltas + at most one bounded sample + the expected
    owner token/version`` — never a full record — and every bound is re-validated
    inside the store, so a bug here cannot write an O(total_files) object into
    the (possibly Manager-hosted) job map.

    Cost control: counts are accumulated locally and flushed in ONE call per
    phase (or lease renewal), and each bucket sends at most
    ``SCAN_JOB_SAMPLE_LIMIT`` samples — everything beyond that is counted into
    ``<bucket>_samples_suppressed`` instead of costing another RPC. A
    million-file scan therefore costs a bounded number of store calls.

    Every store call is SYNCHRONOUS (the store is ``threading.Lock``-guarded with
    no blocking waits), so the finally/cancellation path can still mark a job
    terminal without awaiting. A reporter with no store or token, or one whose
    record it no longer owns (job gone / superseded owner / already terminal —
    e.g. reaped to ABANDONED after a lease expiry), disables itself: progress
    reporting must never fail the scan.
    """

    def __init__(self, store: Any, track_id: str | None, owner_token: str | None):
        self._store = store if (store and track_id and owner_token) else None
        self._track_id = track_id
        self._owner_token = owner_token
        self._version = 1
        self._counts: Dict[str, int] = {}
        self._pending_samples: List[tuple[str, str]] = []
        self._samples_budget = {
            bucket: SCAN_JOB_SAMPLE_LIMIT for bucket in SAMPLE_BUCKETS
        }
        self._last_call = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def count(self, key: str, delta: int = 1) -> None:
        """Buffer a counter delta (flushed later)."""
        if self._store is None:
            return
        self._counts[key] = self._counts.get(key, 0) + delta

    def sample(self, bucket: str, text: str) -> None:
        """Buffer one bounded sample, or count it as suppressed once the bucket's
        send budget is spent (the store caps retained samples anyway)."""
        if self._store is None:
            return
        if self._samples_budget.get(bucket, 0) <= 0:
            self.count(f"{bucket}_samples_suppressed")
            return
        self._samples_budget[bucket] -= 1
        self._pending_samples.append((bucket, text))

    def flush(self) -> None:
        """Send the buffered deltas (one call) and any buffered samples (one call
        each). Renews the lease as a side effect of every accepted update."""
        if self._store is None:
            return
        counts, samples = self._counts, self._pending_samples
        self._counts, self._pending_samples = {}, []
        if not counts and not samples:
            self._submit(None, None)
            return
        first_sample = samples[0] if samples else None
        self._submit(counts or None, first_sample)
        for sample in samples[1:]:
            if self._store is None:  # a submit above disabled us mid-flush
                return
            self._submit(None, sample)

    def renew(self) -> None:
        """Flush if the lease is due for renewal; a cheap no-op otherwise."""
        if self._store is None:
            return
        if time.monotonic() - self._last_call < _SCAN_JOB_RENEW_SECONDS:
            return
        self.flush()

    def finish(self, status: ScanJobStatus, message: str = "") -> None:
        """Flush, then transition the job to a terminal status (owner-checked,
        CAS'd). A late transition that lost the record (reaped/superseded) is
        logged and dropped — it must not overwrite a newer state."""
        if self._store is None:
            return
        self.flush()
        if self._store is None:  # the flush disabled us
            return
        for attempt in range(2):
            try:
                result = self._store.set_status(
                    self._track_id,
                    self._owner_token,
                    status,
                    expected_version=self._version,
                    message=message,
                )
            except Exception as store_error:
                logger.warning(
                    f"Scan job {self._track_id} terminal transition failed: {store_error}"
                )
                break
            if result.ok:
                break
            if (
                attempt == 0
                and result.conflict is ScanJobUpdateConflict.VERSION
                and result.record
            ):
                self._version = result.record.get("version", self._version)
                continue
            logger.warning(
                f"Scan job {self._track_id} terminal transition refused "
                f"({getattr(result.conflict, 'value', result.conflict)})"
            )
            break
        self._store = None

    def _submit(
        self, counts: Optional[Dict[str, int]], sample: Optional[tuple[str, str]]
    ) -> None:
        """One CAS'd update, with a single re-sync retry on a version conflict."""
        for attempt in range(2):
            try:
                result = self._store.update(
                    self._track_id,
                    self._owner_token,
                    count_deltas=counts,
                    sample=sample,
                    expected_version=self._version,
                )
            except Exception as store_error:
                logger.warning(
                    f"Scan job {self._track_id} progress update failed "
                    f"(reporting disabled): {store_error}"
                )
                self._store = None
                return
            if result.ok:
                self._version = (result.record or {}).get("version", self._version + 1)
                self._last_call = time.monotonic()
                return
            if (
                attempt == 0
                and result.conflict is ScanJobUpdateConflict.VERSION
                and result.record
            ):
                self._version = result.record.get("version", self._version)
                continue
            # NOT_FOUND / OWNER / TERMINAL: this reporter is stale for good.
            logger.warning(
                f"Scan job {self._track_id} no longer accepts updates "
                f"({getattr(result.conflict, 'value', result.conflict)}); "
                "progress reporting disabled"
            )
            self._store = None
            return


async def _renew_scan_job_lease(reporter: _ScanJobReporter, scan_task: Any) -> None:
    """Renew a scan job's lease from an independent task (LR2 §8.6).

    The reporter only touches the record when the SCAN touches it: once per
    discovered file, on a batch flush, at a phase boundary. Three phases do
    neither, and none of them has a bounded duration — the custom-chunk
    rollback, the exclusive FAILED→PENDING reset, and the final
    ``apipeline_process_enqueue_documents`` (parse + LLM + index over
    everything this scan produced, plus the deferred drive in the finally).
    Any one of them outliving ``SCAN_JOB_LEASE_SECONDS`` let the store reap a
    perfectly healthy RUNNING job to ABANDONED; the scan's own terminal
    transition then lost to that state, so a scan that COMPLETED was reported
    as "owner presumed dead". Only a heartbeat that does not depend on the
    scan's own progress can distinguish "slow" from "dead".

    Sleeping half the renewal interval bounds the worst-case gap between store
    calls at 1.5x that interval — half the lease — since ``renew()`` skips a
    tick that a scan-driven call already covered.

    Exits when ``scan_task`` completes, so a heartbeat can never outlive its
    owner and keep a dead scan's record alive (that would defeat the very
    reaper this protects against). The task is also cancelled explicitly before
    the terminal transition; this check is what holds if the ``finally`` never
    gets there. Every reporter method is synchronous, so this task can only run
    BETWEEN them — no flush can be interleaved and no lock is needed.
    """
    try:
        while not scan_task.done():
            await asyncio.sleep(max(_SCAN_JOB_RENEW_SECONDS / 2, 0.01))
            reporter.renew()
    except asyncio.CancelledError:
        raise
    except Exception as heartbeat_error:  # never fail the scan over reporting
        logger.warning(f"Scan job lease heartbeat stopped: {heartbeat_error}")


class _ScanFileClass(str, Enum):
    """The seven mutually exclusive scan classification exits (LR2 §8.3).

    Values double as the scan job's counter keys, so a ``/scan/status`` reader
    sees the taxonomy verbatim."""

    CLAIMED_NEW = "claimed_new"
    SOURCE_CONFLICT = "source_conflict"
    PROCESSED = "processed"
    STALE_STUB = "stale_stub"
    SOURCE_IDENTITY_UNKNOWN = "source_identity_unknown"
    RESUME_SAME_PHYSICAL_SOURCE = "resume_same_physical_source"
    ALIAS_DUPLICATE = "alias_duplicate"


@dataclass(frozen=True)
class _ScanFileDecision:
    """One file's classification plus what the caller needs to act on it."""

    kind: _ScanFileClass
    doc_id: str | None = None
    detail: str = ""
    """Operator-facing reason, recorded as a bounded job sample when set."""


async def _confirm_full_docs_absent(rag: LightRAG, doc_id: str) -> bool | None:
    """Is the document's ``full_docs`` content CONFIRMED absent? (LR2 §8.3.D)

    Returns ``True`` (positively absent — the FAILED row is an unprocessable
    stub), ``False`` (content exists), or ``None`` when absence cannot be
    confirmed: the backend has no strict point read, or the read failed. Only
    ``True`` may drive the destructive stub deletion — a best-effort miss
    (``get_by_id`` swallowing a transport error) would delete the row of a
    document whose content actually exists.
    """
    store = rag.full_docs
    if not getattr(store, "supports_strict_point_reads", False):
        logger.warning(
            f"{type(store).__name__} has no strict point reads; keeping the "
            f"FAILED row for {doc_id} instead of trusting an unconfirmed miss"
        )
        return None
    try:
        return await store.get_by_id_strict(doc_id) is None
    except Exception as read_error:
        logger.warning(
            f"Strict full_docs point read failed for {doc_id}; keeping the "
            f"FAILED row: {read_error}"
        )
        return None


async def _row_source_file(rag: LightRAG, doc_id: str) -> str | None:
    """Read ``metadata.source_file`` — the ORIGINAL physical basename (hint
    included) that created this row — hydrating exactly one record.

    ``None`` means the row carries no source identity (custom-ID / legacy /
    non-scan-origin inserts) or vanished between resolution and hydration; per
    §8.3.E that is NOT evidence of a different physical file. A strict
    hydration failure propagates: identity decisions must fail closed.
    """
    rows = await rag.doc_status.get_full_docs_by_ids([doc_id], strict=True)
    row = rows.get(doc_id)
    if row is None:
        return None
    return read_source_file_basename(getattr(row, "metadata", None) or {})


async def classify_scan_file(
    rag: LightRAG, file_path: Path, canonical_source_key: str
) -> _ScanFileDecision:
    """Classify one physical file into the seven §8.3 exits.

    Identity is resolved with ``resolve_doc_source_strict``: the doc ID is the
    identity every later operation uses, and the canonical basename only LOCATES
    candidates — it is not assumed unique (custom-ID inserts, legacy ids and
    historical collisions can share one), which is why two primary candidates
    are a conflict to repair rather than a row to overwrite.

    Read-only by contract: every destructive consequence (enqueue, archive, stub
    deletion) is performed by the caller, so a decision can be logged, counted
    and sampled before anything on disk or in storage changes. The mutually
    exclusive order is the document's:

    1. ``SourceAbsent`` → CLAIMED_NEW
    2. ``SourceConflict`` → SOURCE_CONFLICT (never enqueue/delete/archive)
    3. PROCESSED → PROCESSED (archive the input file)
    4. FAILED with CONFIRMED-absent content → STALE_STUB (delete, retry as new)
    5. no ``metadata.source_file`` → SOURCE_IDENTITY_UNKNOWN (keep, warn)
    6. physical basename == ``source_file`` → RESUME_SAME_PHYSICAL_SOURCE
    7. otherwise → ALIAS_DUPLICATE (archive the alias, keep the row)
    """
    if canonical_source_key == UNKNOWN_FILE_SOURCE:
        # No usable canonical identity to resolve against (nothing to collide
        # with either); the enqueue path reports its own error document if the
        # name turns out to be unusable.
        return _ScanFileDecision(_ScanFileClass.CLAIMED_NEW)

    resolution = await rag.doc_status.resolve_doc_source_strict(canonical_source_key)

    if isinstance(resolution, SourceAbsent):
        return _ScanFileDecision(_ScanFileClass.CLAIMED_NEW)

    if isinstance(resolution, SourceConflict):
        count = resolution.candidate_count
        return _ScanFileDecision(
            _ScanFileClass.SOURCE_CONFLICT,
            detail=(
                f"{count if count is not None else 'Multiple'} primary documents "
                f"share canonical source '{canonical_source_key}'; repair them by "
                "doc id before this file can be classified (sample doc ids: "
                f"{', '.join(resolution.sample_doc_ids) or 'unavailable'})"
            ),
        )

    if not isinstance(resolution, SourceUnique):  # pragma: no cover - typed union
        raise TypeError(
            f"resolve_doc_source_strict returned {type(resolution).__name__}; "
            "expected SourceAbsent | SourceUnique | SourceConflict"
        )

    doc_id = resolution.doc_id
    status_value = get_doc_status_value(resolution.doc)

    if status_value == DocStatus.PROCESSED.value:
        return _ScanFileDecision(_ScanFileClass.PROCESSED, doc_id=doc_id)

    if status_value == DocStatus.FAILED.value:
        # An extraction-error stub (recorded by apipeline_enqueue_error_documents)
        # never wrote full_docs, and the consistency validator preserves it for
        # manual review — the resume path can never advance it. A re-scan of a
        # fixed file must therefore drop the stub and start over. Only a
        # CONFIRMED absence may do that; an unconfirmed one falls through to the
        # non-destructive exits below.
        if await _confirm_full_docs_absent(rag, doc_id) is True:
            return _ScanFileDecision(_ScanFileClass.STALE_STUB, doc_id=doc_id)

    source_file = await _row_source_file(rag, doc_id)
    if source_file is None:
        return _ScanFileDecision(
            _ScanFileClass.SOURCE_IDENTITY_UNKNOWN,
            doc_id=doc_id,
            detail=(
                f"Document {doc_id} (status {status_value}) shares canonical "
                f"source '{canonical_source_key}' but records no source file, so "
                f"{file_path.name} cannot be proven to be a different physical "
                "file; keeping it untouched"
            ),
        )
    if source_file == file_path.name:
        return _ScanFileDecision(
            _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE, doc_id=doc_id
        )
    return _ScanFileDecision(
        _ScanFileClass.ALIAS_DUPLICATE,
        doc_id=doc_id,
        detail=(
            f"Archiving alias {file_path.name}: canonical source "
            f"'{canonical_source_key}' already belongs to document {doc_id} "
            f"(source file '{source_file}', status {status_value})"
        ),
    )


def _scan_enqueue_batch_size() -> int:
    """How many claimed files one streaming scan batch holds (LR2 §8.2).

    Read per scan (not captured at import) so a restart-free config reload takes
    effect. ``initialize_config`` rejects a non-positive value at startup; the
    clamp here only covers a rig that bypassed it — never fall back to an
    unbounded batch, which is precisely what streaming discovery removes.
    """
    configured = getattr(global_args, "scan_enqueue_batch_size", None)
    if not isinstance(configured, int) or configured <= 0:
        logger.warning(
            "SCAN_ENQUEUE_BATCH_SIZE is not a positive integer "
            f"({configured!r}); falling back to {DEFAULT_SCAN_ENQUEUE_BATCH_SIZE}"
        )
        return DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
    return configured


_SCAN_SPOOL_PREFIX = "lightrag-scan-sort-"


def _scan_spool_base_dir(rag: LightRAG) -> Path | None:
    """Where this scan creates its disposable candidate spool.

    Resolution order: ``SCAN_SPOOL_DIR`` → ``WORKING_DIR/scan_spool``, each
    narrowed to a per-workspace subdirectory.  The spool holds
    O(files-in-INPUT_DIR) ordering rows, so it must land on real, writable,
    local disk: ``/tmp`` is a RAM-backed tmpfs on many Linux hosts, which would
    put the very state this design moves OUT of Python memory straight back into
    memory.  WORKING_DIR is already LightRAG's own writable local state volume,
    which makes it the right default.

    INPUT_DIR is deliberately NOT a candidate.  It is frequently a network mount
    (the slowest possible home for the one bulk-insert phase in the pipeline),
    it is legitimately mounted read-only in "scan but never write" deployments,
    and it is co-managed by sync tools that would copy or delete the spool
    mid-scan.

    The workspace subdirectory is what lets the spool have a FIXED name inside
    it (see :class:`_ScanCandidateSpool`): two workspaces sharing one
    WORKING_DIR must never reuse or delete each other's file.  That makes the
    workspace → directory mapping load-bearing, so it is INJECTIVE by
    construction — ``unnamed/`` for the empty workspace, ``named/<workspace>/``
    for every other.  A bare sentinel would not be: ``validate_workspace``
    accepts any single path component, so a workspace literally called
    ``_default`` would collide with the unnamed one.  The two hold different
    per-workspace scan locks and can therefore run concurrently, and the loser
    of that race has its live database deleted out from under it.

    Returns ``None`` only when no working directory can be determined at all —
    a test rig, never a real ``LightRAG``, whose ``working_dir`` always defaults
    to ``./rag_storage``.  There is no operator-chosen placement to honour in
    that case, so the caller falls back to the OS temp dir.
    """
    configured = getattr(global_args, "scan_spool_dir", None)
    if isinstance(configured, str) and configured.strip():
        base = Path(configured.strip())
    else:
        working_dir = getattr(rag, "working_dir", None)
        if not isinstance(working_dir, str) or not working_dir.strip():
            return None
        base = Path(working_dir.strip()) / "scan_spool"

    workspace = getattr(rag, "workspace", "") or ""
    if not workspace:
        return base / "unnamed"
    # Reuses the storage layer's rule rather than sanitizing: a workspace that
    # could escape its directory is a configuration error everywhere else too.
    return base / "named" / validate_workspace(workspace)


class _ScanCandidateSpool:
    """Disk-backed, globally mtime-ordered scan candidates.

    Exact oldest-file-first ordering over an unordered directory iterator needs
    O(number of files) state somewhere. Keeping that state in Python memory
    would undo LR2's bounded scan, while overloading ``doc_status.created_at``
    with the file's mtime makes a persistence timestamp lie. This disposable
    SQLite spool is the separation point:

    * discovery/classification inserts one lightweight row at a time;
    * a UNIQUE canonical key preserves first-physical-claim-wins before any
      candidate has reached doc_status;
    * an on-disk ``(mtime, path, sequence)`` index provides the global order;
    * :meth:`ordered_batches` fetches at most K candidates into Python memory;
    * once a path is enqueued, doc_status stamps its real creation time and the
      filesystem timestamp has no further role.

    A missing mtime does not lose the candidate. It sorts after every readable
    timestamp and the enqueue path reports a vanished/unreadable file normally.

    **Lifetime.** The database is rebuildable coordination state, never
    persistent LightRAG storage. It is deleted when the scan ends, and a process
    death simply leaves the source files for the next scan to rediscover. But it
    lives on a PERSISTENT volume, where nothing reclaims what ``kill -9`` leaves
    behind — so its name inside the per-workspace directory is FIXED rather than
    randomized, and the next scan deletes any leftover before opening its own.
    Residue is therefore capped at one spool, not one per crash. Reusing a fixed
    name is safe because ``scanning_exclusive`` already admits a single scan per
    workspace, which is the same guarantee that keeps two scans from fighting
    over INPUT_DIR.

    **Placement is fail-closed.** If the operator's directory cannot be used,
    the scan fails instead of quietly relocating to the OS temp dir. That
    fallback is not a degraded-but-correct mode: on a host where ``/tmp`` is
    tmpfs it silently restores the O(number of files) RAM cost this whole design
    exists to remove, and the symptom is an OOM kill halfway through a large
    scan rather than an error naming the misconfiguration.
    """

    _DB_NAME = "candidates.sqlite3"

    def __init__(self, commit_interval: int, base_dir: Path | None = None):
        self._spool_dir, self._temp_dir = self._prepare_dir(base_dir)
        self._db_path = self._spool_dir / self._DB_NAME
        try:
            self._discard_database_files()
        except OSError as residue_error:
            # Opening on top of residue we could not remove would mean reading
            # another scan's rows, so this is fail-closed like placement itself.
            raise RuntimeError(
                f"Scan candidate spool {self._db_path} could not be reclaimed "
                f"({residue_error}); refusing to reuse another scan's database."
            ) from residue_error
        self._connection = sqlite3.connect(self._db_path)
        # Bound SQLite's page cache and force any query scratch space to disk.
        # The database is disposable, so fsync durability buys nothing: a
        # process death simply makes the next /scan rediscover the source files.
        self._connection.execute("PRAGMA cache_size = -2048")
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute("PRAGMA synchronous = OFF")
        self._connection.executescript(
            """
            CREATE TABLE candidates (
                sequence INTEGER PRIMARY KEY,
                canonical_key TEXT,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_missing INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                path_sort_key TEXT NOT NULL
            );
            CREATE UNIQUE INDEX candidates_canonical
                ON candidates(canonical_key);
            CREATE INDEX candidates_schedule
                ON candidates(
                    mtime_missing,
                    mtime_ns,
                    path_sort_key,
                    sequence
                );
            """
        )
        self._commit_interval = max(1, int(commit_interval))
        self._pending_writes = 0

    @staticmethod
    def _prepare_dir(
        base_dir: Path | None,
    ) -> tuple[Path, tempfile.TemporaryDirectory | None]:
        """Resolve the spool directory, or raise (see "Placement is fail-closed").

        ``base_dir is None`` means no placement was determinable at all (a test
        rig without a ``working_dir``); there is nothing to honour, so a
        self-cleaning OS temp dir is used and returned for disposal.
        """
        if base_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix=_SCAN_SPOOL_PREFIX)
            return Path(temp_dir.name), temp_dir
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as spool_error:
            raise RuntimeError(
                f"Scan candidate spool directory {base_dir} is unusable "
                f"({spool_error}). Point SCAN_SPOOL_DIR at a writable local "
                "disk; the scan is refused rather than relocated, because the "
                "OS temp dir is a RAM-backed tmpfs on many hosts and would "
                "silently restore the memory cost the spool exists to avoid."
            ) from spool_error
        return base_dir, None

    def _discard_database_files(self) -> None:
        """Delete this workspace's spool database and any SQLite side files.

        Called before opening (reclaiming a crashed scan's residue) and after
        closing (ordinary disposal).
        """
        for path in self._spool_dir.glob(f"{self._DB_NAME}*"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    async def claim(
        self,
        file_path: Path,
        canonical_key: str,
        sequence: int,
    ) -> str | None:
        """Stage one candidate; return the earlier claimer's path on conflict."""

        try:
            # INPUT_DIR may be a network mount. Keep its metadata read off the
            # event loop so scan-job heartbeats and cancellation remain live.
            # This is the ONLY stat of the file: the size travels with the
            # candidate so the enqueue phase does not repeat the round-trip.
            stat = await asyncio.to_thread(file_path.stat)
            mtime_missing = 0
            # NOT ``getattr(stat, "st_mtime_ns", <fallback>)``: Python evaluates
            # a getattr default eagerly, so the fallback would read st_mtime on
            # every call and an object exposing only st_mtime_ns would raise.
            mtime_ns = getattr(stat, "st_mtime_ns", None)
            if mtime_ns is None:
                mtime_ns = int(stat.st_mtime * 1_000_000_000)
            file_size = stat.st_size
        except Exception:
            mtime_missing = 1
            mtime_ns = 0
            # 0, not NULL: the candidate's size is always "already read", so the
            # enqueue path never re-stats. 0 is exactly how pipeline_enqueue_file
            # has always reported a size it could not read, and the field feeds
            # error reports only.
            file_size = 0

        # SQLite UNIQUE permits multiple NULLs, matching unknown_source's
        # "identity cannot be claimed" semantics: every such candidate is
        # staged, none of them claims the others' slot.
        stored_canonical = (
            None if canonical_key == UNKNOWN_FILE_SOURCE else canonical_key
        )
        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO candidates(
                sequence,
                canonical_key,
                file_path,
                file_size,
                mtime_missing,
                mtime_ns,
                path_sort_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                stored_canonical,
                str(file_path),
                file_size,
                mtime_missing,
                int(mtime_ns),
                get_pinyin_sort_key(str(file_path)),
            ),
        )
        if cursor.rowcount == 0:
            # OR IGNORE swallows EVERY constraint violation, so a row that did
            # not land is only a duplicate canonical claim when we can name the
            # claimer. Anything else (a reused sequence, a NOT NULL breach)
            # would silently drop a discovered file — raise instead.
            row = (
                self._connection.execute(
                    "SELECT file_path FROM candidates WHERE canonical_key = ?",
                    (stored_canonical,),
                ).fetchone()
                if stored_canonical is not None
                else None
            )
            if row is None:
                raise RuntimeError(
                    f"scan candidate {file_path} (sequence {sequence}) was "
                    "rejected by the spool without a canonical claim to blame"
                )
            return str(row[0])

        self._pending_writes += 1
        if self._pending_writes >= self._commit_interval:
            self._connection.commit()
            self._pending_writes = 0
        return None

    def ordered_batches(self, batch_size: int) -> Iterator[list[_ScanCandidate]]:
        """Yield globally oldest-first batches, each bounded by ``batch_size``."""

        self._connection.commit()
        self._pending_writes = 0
        cursor = self._connection.execute(
            """
            SELECT file_path, file_size
            FROM candidates
            ORDER BY
                mtime_missing ASC,
                mtime_ns ASC,
                path_sort_key ASC,
                sequence ASC
            """
        )
        while rows := cursor.fetchmany(batch_size):
            yield [_ScanCandidate(Path(str(row[0])), int(row[1])) for row in rows]

    def close(self) -> None:
        try:
            self._connection.close()
        finally:
            try:
                self._discard_database_files()
            except OSError as cleanup_error:
                # The scan itself is done; leftover bytes are reclaimed by the
                # next scan's pre-open discard, so this must not fail the run.
                logger.warning(
                    f"Could not delete scan spool {self._db_path} "
                    f"({cleanup_error}); the next scan will reclaim it."
                )
            if self._temp_dir is not None:
                self._temp_dir.cleanup()

    def __enter__(self) -> "_ScanCandidateSpool":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self.close()


def _enforce_texts_per_request(count: int) -> None:
    """Refuse an oversized ``/documents/texts`` batch with 413 (LR2 §11).

    Read per request (not captured at import) so a config reload applies, and
    checked BEFORE the endpoint's per-text existence reads — those are one
    storage round-trip each, so a 100k-text batch would otherwise do 100k of them
    on its way to being refused.

    413 rather than 429: ``MAX_PENDING_DOCUMENTS`` says "the pipeline is full,
    come back later", but a request that carries more documents than one request
    may carry will never fit, however long the client waits. A non-positive or
    non-integer setting disables the check — ``initialize_config`` already
    rejects a negative one, and refusing every batch because a knob is malformed
    would be worse than not enforcing it.
    """
    limit = getattr(global_args, "max_texts_per_request", None)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return
    if count > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many texts in one request: {count} (maximum {limit}). "
                "Split the batch into smaller requests."
            ),
        )


# Upper bound on one source-conflict listing page. The projection is bounded
# per key (count + fixed sample), so this only caps the response size.
_SOURCE_CONFLICT_PAGE_MAX = 200


def _encode_source_conflict_cursor(position: CursorPosition) -> Optional[str]:
    """Wrap a backend continuation token as a client-facing opaque cursor.

    The backend's token is its own private format (JSON for some backends). It
    is base64url-wrapped here for two reasons: it keeps storage internals out of
    the query string, and it lets the endpoint reject a garbled cursor as a 400
    instead of letting the backend raise a control-plane error that would be
    reported as an unavailable service (503).
    """
    if not isinstance(position, CursorAfter):
        return None
    return base64.urlsafe_b64encode(position.opaque.encode("utf-8")).decode("ascii")


def _decode_source_conflict_cursor(raw: Optional[str]) -> CursorPosition:
    """Inverse of :func:`_encode_source_conflict_cursor`; 400 on a bad cursor.

    Boundary: only the envelope is validated here. A well-formed envelope whose
    payload the backend cannot parse (e.g. a cursor minted by a different
    backend) still surfaces as that backend's control-plane error.
    """
    if raw is None or raw == "":
        return CURSOR_START
    try:
        opaque = base64.b64decode(
            raw.encode("ascii"), altchars=b"-_", validate=True
        ).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError) as decode_error:
        logger.warning(
            f"Rejected malformed source-conflict cursor "
            f"'{safe_log_value(raw, 64)}': {decode_error}"
        )
        raise HTTPException(
            status_code=400,
            detail="Malformed cursor; pass back the next_cursor value verbatim.",
        )
    if not opaque:
        raise HTTPException(
            status_code=400,
            detail="Malformed cursor; pass back the next_cursor value verbatim.",
        )
    return CursorAfter(opaque)


def _resolve_scan_job_store(rag: LightRAG) -> Any:
    """Resolve this workspace's scan job store, or None when unavailable.

    Mocked rigs (and any path running before shared storage is initialised) have
    no store; the scan must still run, just without a job record."""
    try:
        from lightrag.kg.shared_storage import get_scan_job_store

        return get_scan_job_store(getattr(rag, "workspace", ""))
    except Exception as store_error:
        logger.debug(f"Scan job store unavailable: {store_error}")
        return None


def _cancel_scan_job(
    rag: LightRAG, track_id: str | None, owner_token: str | None
) -> None:
    """Owner-checked cancel of a job this endpoint created but never handed off.

    Part of the §8.6 reverse compensation chain: a job whose child never took
    over has nobody to finalise it. Owner-checked, so a late compensation can
    never cancel a SUCCESSOR's job; idempotent for a missing/terminal record."""
    if not track_id or not owner_token:
        return
    store = _resolve_scan_job_store(rag)
    if store is None:
        return
    try:
        store.cancel(
            track_id,
            owner_token,
            message="scan startup aborted before the background task took over",
        )
    except Exception as cancel_error:
        logger.warning(f"Scan job {track_id} startup cancel failed: {cancel_error}")


async def run_scanning_process(
    rag: LightRAG,
    doc_manager: DocumentManager,
    track_id: str = None,
    scanning_token: str | None = None,
    manual_request_id: str | None = None,
    job_owner_token: str | None = None,
):
    """Background task to scan and index documents

    Args:
        rag: LightRAG instance
        doc_manager: DocumentManager instance
        track_id: Optional tracking ID to pass to all scanned files
        manual_request_id: The sticky manual retry request the scan endpoint
            published for this scan (None on legacy/mocked paths). It is served
            BEFORE any file is discovered, by the shared exclusive FAILED reset
            (LR2 §8.1) — the only path that pulls FAILED documents back in — so
            a file that fails during THIS scan cannot be absorbed by this
            scan's own request. When the reset does not complete, discovery is
            skipped entirely and the request stays sticky for the standard
            drain path. Either way the finally drives the queue once if no
            branch above did, so the reset PENDING rows / the unserved request
            do not wait for an unrelated trigger; on ANY cancellation the drive
            is skipped and whatever was reset simply stays PENDING for the next
            trigger (a shutdown must not start a full processing run, and a
            cancellation's origin cannot be told apart here).
        job_owner_token: Owner token of the bounded scan job record the endpoint
            created before handing this task the reservation (None on
            legacy/mocked paths). This task owns that record from takeover on:
            it reports bounded progress into it and finalises it in the finally
            (COMPLETED / FAILED / CANCELLED), so the endpoint's compensation
            chain never has to (LR2 §8.6).
    """
    # The scan endpoint set ``scanning=True`` AND
    # ``scanning_exclusive=True`` synchronously before scheduling this
    # task.  ``scanning`` covers the whole lifecycle (refuses
    # overlapping scans); ``scanning_exclusive`` covers only the
    # classification phase below — we clear it before invoking
    # pipeline_index_files so concurrent uploads can land while the
    # scan-driven processing finishes.  Both MUST be cleared in
    # finally so subsequent uploads / scans can proceed even if the
    # body raises.  When pipeline_status is not initialised (mocked
    # test rigs), the flags were never set so there's nothing to
    # clear — track that here to skip the namespace fetch.
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
        has_scan_deferred_processing,
        transition_scanning_reservation,
    )

    pipeline_status = None
    pipeline_status_lock = None
    # Distinguish a normal exit from a cancellation (server shutdown / task
    # cancel): a cancelled scan MUST NOT kick off a full processing run in its
    # finally. Shutdown is waiting for THIS task to exit, and one cancel injects
    # CancelledError only once, so a post-release ``await`` here would otherwise
    # run parse/LLM/index to completion and stall the shutdown.
    was_cancelled = False
    # True once any drive branch below invoked the processing queue: the
    # manual-intent fallback in the finally only fires when EVERY branch was
    # skipped or classification failed — a drive that ran had its chance to
    # consume the scan's sticky request (and a busy-refused drive leaves it
    # for the running loop's quiescence peek).
    queue_drive_attempted = False
    # Bounded job-record reporting (LR2 §8.6). Disabled (no-op) when this task
    # owns no record — legacy/mocked paths and any run before shared storage is
    # initialised. The terminal transition happens in the finally; the default
    # below covers an exit path that sets nothing (it never leaves a RUNNING
    # record behind for the lease reaper to guess about).
    reporter = _ScanJobReporter(
        _resolve_scan_job_store(rag) if job_owner_token else None,
        track_id,
        job_owner_token,
    )
    job_status = ScanJobStatus.FAILED
    job_message = "scan ended without reporting an outcome"
    # Lease heartbeat, independent of scan progress: the phases below (rollback,
    # exclusive FAILED reset, processing) can each outlast the lease without
    # touching the record. Cancelled just before the terminal transition, after
    # the deferred drive in the finally — see _renew_scan_job_lease.
    lease_heartbeat = (
        asyncio.create_task(_renew_scan_job_lease(reporter, asyncio.current_task()))
        if reporter.enabled
        else None
    )
    try:
        # Fetch INSIDE the release try: the scan endpoint already reserved
        # ``scanning``/``scanning_exclusive`` before scheduling us, so a
        # cancellation delivered at this await must still reach the finally and
        # release. PipelineNotInitializedError = mocked test rig (nothing to
        # clear); a real CancelledError propagates to the finally.
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )
        except PipelineNotInitializedError:
            pass

        # Roll back failed/stale custom-chunk operations FIRST, while the
        # classification phase still holds ``scanning_exclusive`` (issue
        # #3400 Phase 4). Discovery is storage-driven — SDK operations may
        # have no scan-visible input file — and a failed rollback keeps the
        # journal/FAILED row for the next scan without aborting this one.
        if pipeline_status is not None and pipeline_status_lock is not None:
            try:
                await rag.arollback_failed_custom_chunk_patches(
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                )
            except Exception as rollback_error:
                logger.error(
                    f"Scan-time custom-chunk rollback failed: {rollback_error}"
                )

        # LR2 §8.1: retry the FAILED documents that already existed BEFORE this
        # scan discovers or enqueues anything. Running the exclusive
        # FAILED→PENDING reset first is what keeps a file that fails DURING this
        # scan out of this scan's own manual request (it would otherwise be
        # retried immediately, by the very request that admitted it). The reset
        # reuses the /reprocess_failed helper under this scan's owner token and
        # ACKs the request itself, so the drive below no longer consumes it.
        if manual_request_id is not None and pipeline_status is not None:
            if not await rag.apipeline_reset_failed_for_scan(
                manual_request_id, scan_owner_token=scanning_token
            ):
                # No reset, no discovery: enqueuing new files now would put them
                # ahead of a still-sticky manual request, which is exactly the
                # ordering §8.1 forbids. The finally releases the reservations and
                # drives the queue once so the standard drain path serves the
                # request; the next scan discovers the files.
                abort_message = (
                    "Scan aborted before file discovery: the exclusive FAILED "
                    "reset did not complete (manual request "
                    f"{manual_request_id[:8]} stays pending)"
                )
                logger.error(abort_message)
                job_status = ScanJobStatus.FAILED
                job_message = abort_message
                reporter.sample("error", abort_message)
                return

        # ---- streaming discovery + disk-backed ordering (LR2 §8.2/§8.4) -----
        # Discovery/classification stays a single pass with O(K) Python memory,
        # but exact global mtime order needs O(number of files) ordering state
        # somewhere. The disposable SQLite spool keeps that state on disk and
        # yields at most ``SCAN_ENQUEUE_BATCH_SIZE`` paths at a time after
        # discovery. Only then are rows written to doc_status, in oldest-file-
        # first order, with created_at stamped at the actual write time.
        #
        # The cost of global ordering: nothing is persisted until discovery
        # finishes, so an interrupted scan enqueues NOTHING (the source files
        # stay in INPUT_DIR for the next scan to rediscover). The one thing that
        # does not come back is a STALE_STUB row deleted below — the file is
        # re-enqueued as new next time, but its preserved-for-review FAILED stub
        # is gone.
        batch_size = _scan_enqueue_batch_size()
        discovered = 0
        enqueued_count = 0
        resumed_count = 0
        processed_count = 0

        async def _archive(file_path: Path, warning: str, counter_key: str) -> None:
            """Archive an input file (never delete it) and record the reason."""
            await record_scan_warning(rag, warning)
            reporter.count(counter_key)
            reporter.sample("warning", warning)
            try:
                await move_file_to_parsed_dir(file_path)
            except Exception as move_error:
                archive_error = (
                    f"Failed to move scan file {file_path.name} "
                    f"to {PARSED_DIR_NAME}: {move_error}"
                )
                logger.error(archive_error)
                reporter.count("errors")
                reporter.sample("error", archive_error)

        async def _claim_new_file(
            spool: _ScanCandidateSpool,
            file_path: Path,
            canonical_key: str,
            counter_key: str,
            sequence: int,
        ) -> None:
            """Stage one file under the scan-wide first canonical claim (§8.4)."""

            claimer = await spool.claim(file_path, canonical_key, sequence)
            if claimer is not None:
                # Full paths, not basenames: two same-named files in different
                # subdirectories share a canonical key, and "a.docx duplicates
                # a.docx" tells an operator nothing about which one won.
                await _archive(
                    file_path,
                    "Skipping duplicate file in scan: "
                    f"{file_path} duplicates {claimer} "
                    f"(canonical: {canonical_key})",
                    _ScanFileClass.ALIAS_DUPLICATE.value,
                )
                return
            reporter.count(counter_key)

        with _ScanCandidateSpool(
            batch_size, _scan_spool_base_dir(rag)
        ) as candidate_spool:
            for file_path in doc_manager.iter_new_files():
                discovered += 1
                # Classifying a large tree can outlast the job lease; renewing here
                # (time-based, a no-op most iterations) keeps the record from being
                # reaped to ABANDONED under a live owner.
                reporter.renew()
                filename = file_path.name
                canonical_key = normalize_file_path(str(file_path))
                decision = await classify_scan_file(rag, file_path, canonical_key)

                if decision.kind is _ScanFileClass.CLAIMED_NEW:
                    await _claim_new_file(
                        candidate_spool,
                        file_path,
                        canonical_key,
                        _ScanFileClass.CLAIMED_NEW.value,
                        discovered,
                    )
                    continue

                if decision.kind is _ScanFileClass.SOURCE_CONFLICT:
                    # §8.3.B: no enqueue, no doc_status delete, NO archive — the
                    # file stays put and an operator repairs the historical
                    # collision by doc id.
                    await record_scan_warning(rag, decision.detail)
                    reporter.count(_ScanFileClass.SOURCE_CONFLICT.value)
                    reporter.sample("warning", decision.detail)
                    continue

                if decision.kind is _ScanFileClass.PROCESSED:
                    processed_count += 1
                    await _archive(
                        file_path,
                        f"Skipping already processed file: {filename}",
                        _ScanFileClass.PROCESSED.value,
                    )
                    reporter.sample("processed", filename)
                    continue

                if decision.kind is _ScanFileClass.STALE_STUB:
                    # §8.3.D: content confirmed absent, so this FAILED row can
                    # never be resumed — drop it and retry the fixed file as new.
                    try:
                        await rag.doc_status.delete([decision.doc_id])
                    except Exception as delete_error:
                        stub_error = (
                            "Failed to delete stale failed-extraction doc_status "
                            f"stub {decision.doc_id} ({filename}): {delete_error}"
                        )
                        logger.error(stub_error)
                        reporter.count("errors")
                        reporter.sample("error", stub_error)
                        continue
                    logger.info(
                        "Retrying previously failed extraction; removed stale "
                        f"doc_status stub: {filename} (doc_id: {decision.doc_id})"
                    )
                    await _claim_new_file(
                        candidate_spool,
                        file_path,
                        canonical_key,
                        _ScanFileClass.STALE_STUB.value,
                        discovered,
                    )
                    continue

                if decision.kind is _ScanFileClass.SOURCE_IDENTITY_UNKNOWN:
                    # §8.3.E: missing source_file is not evidence of a different
                    # physical file; keep both file and row for review.
                    await record_scan_warning(rag, decision.detail)
                    reporter.count(_ScanFileClass.SOURCE_IDENTITY_UNKNOWN.value)
                    reporter.sample("warning", decision.detail)
                    continue

                if decision.kind is _ScanFileClass.ALIAS_DUPLICATE:
                    # §8.3.G: same canonical key, different physical file.
                    await _archive(
                        file_path,
                        decision.detail,
                        _ScanFileClass.ALIAS_DUPLICATE.value,
                    )
                    continue

                # §8.3.F RESUME_SAME_PHYSICAL_SOURCE: the same physical file
                # behind an unfinished row. Resume it from persistent state.
                logger.info(
                    f"Resuming previously unfinished file from scan: {filename} "
                    f"(doc_id: {decision.doc_id})"
                )
                resumed_count += 1
                reporter.count(_ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE.value)

            # Discovery is complete. Iterate the on-disk mtime index in bounded
            # pages; each successful enqueue stamps doc_status.created_at=now().
            for ordered_batch in candidate_spool.ordered_batches(batch_size):
                enqueued_count += await pipeline_enqueue_scan_batch(
                    rag, ordered_batch, track_id
                )
                # Publish this batch's counters/samples and renew the job lease.
                reporter.flush()

        reporter.count("discovered", discovered)
        reporter.flush()

        # Classification + enqueue complete — release ``scanning_exclusive`` so
        # concurrent uploads/inserts can land in doc_status while the scan-driven
        # processing finishes. ``scanning`` stays True for the rest of the task
        # lifecycle (released in the finally) so the /scan endpoint still refuses
        # overlapping scans. Any duplicate detected during the processing phase is
        # handled by apipeline_enqueue_documents' in-batch dedup, identical to the
        # upload-during-busy case.
        if pipeline_status is not None and pipeline_status_lock is not None:
            if scanning_token is not None:
                await transition_scanning_reservation(
                    pipeline_status,
                    pipeline_status_lock,
                    token=scanning_token,
                )
            else:
                async with pipeline_status_lock:
                    pipeline_status["scanning_exclusive"] = False

        # §8.1 final step: ONE processing run covers everything this scan
        # produced — the enqueued batches, the resume targets, and the FAILED rows
        # the pre-discovery reset turned into PENDING. Unconditional: a scan that
        # enqueued nothing may still have resume targets or reset rows with no
        # other trigger, and an empty queue makes this a cheap no-op.
        queue_drive_attempted = True
        await rag.apipeline_process_enqueue_documents()

        summary = (
            f"Scanning process completed: {discovered} discovered, "
            f"{enqueued_count} enqueued, {resumed_count} resuming, "
            f"{processed_count} already processed."
        )
        logger.info(summary)
        job_status = ScanJobStatus.COMPLETED
        job_message = summary

    except asyncio.CancelledError:
        # Shutdown / task cancel: skip the deferred drive below and leave
        # ``scan_deferred_processing`` set for the next scan / trigger.
        was_cancelled = True
        job_status = ScanJobStatus.CANCELLED
        job_message = "Scan cancelled (shutdown or explicit cancellation)."
        raise
    except Exception as e:
        logger.error(f"Error during scanning process: {str(e)}")
        logger.error(traceback.format_exc())
        job_status = ScanJobStatus.FAILED
        job_message = f"Scan failed: {e}"
        reporter.sample("error", f"Scan failed: {e}")
    finally:
        # Always release both scanning flags so future uploads / scans are not
        # blocked by a crashed / cancelled task.
        if scanning_token is not None:
            # Real scans: owner-checked + cancellation-resistant release that
            # fetches status FRESH, so a cancellation during the initial
            # namespace fetch above (which leaves pipeline_status_lock None)
            # still releases scanning. A no-op if a later scan took the slot or
            # the workspace was never initialised.
            await _release_scanning_reservation(rag, scanning_token)
        elif pipeline_status is not None and pipeline_status_lock is not None:
            # Legacy/test path (no token): best-effort in-place clear.
            async with pipeline_status_lock:
                pipeline_status.update(
                    {
                        "scanning": False,
                        "scanning_exclusive": False,
                        "scanning_owner": None,
                    }
                )

        # If this scan's ``scanning_exclusive`` fence turned away a concurrent
        # processing request that no run then picked up, its PENDING doc has no
        # other trigger — the branches above drive the queue only for new/resume
        # files or an empty scan directory, so an all-already-processed scan or a
        # classification error would strand it. Drain the queue once now that
        # scanning has fully released. The check is READ-ONLY; the drive's own
        # acquire clears the deferred flag, so a failed drive leaves it set for
        # the next scan to honour. Empty queue → silent no-op.
        #
        # Skipped on cancellation: a cancelled scan must not start a full
        # processing run while shutdown waits for it. The flag is never checked
        # or cleared here, so it stays set for the next scan / trigger.
        if (
            not was_cancelled
            and pipeline_status is not None
            and pipeline_status_lock is not None
        ):
            drive_needed = await has_scan_deferred_processing(
                pipeline_status, pipeline_status_lock
            )
            if (
                not drive_needed
                and manual_request_id is not None
                and not queue_drive_attempted
            ):
                # Manual-intent fallback: every drive branch above was skipped or
                # refused, so nothing has processed what this scan produced.
                # Either outcome of the pre-discovery reset needs one drive:
                #   * it completed — the FAILED rows are now PENDING with no
                #     other trigger (classification may then have raised, or
                #     found nothing to enqueue);
                #   * it did not — the request is still sticky and the drive's
                #     start peek runs the standard drain → reset → ACK path.
                # A drive with nothing to do is a cheap no-op ("No documents to
                # process"), so this needs no mailbox pre-check.
                drive_needed = True
            if drive_needed:
                try:
                    await rag.apipeline_process_enqueue_documents()
                except Exception as drive_error:
                    logger.error(
                        f"Deferred post-scan queue drive failed: {drive_error}"
                    )
                    job_status = ScanJobStatus.FAILED
                    job_message = f"Post-scan queue drive failed: {drive_error}"
                    reporter.sample("error", job_message)

        # The heartbeat has covered every phase including the deferred drive
        # above; stop it before the terminal transition so the record's last
        # write is the terminal one. Not awaited: the cancellation path must not
        # await here (one cancel is injected once), and the task's own
        # ``scan_task.done()`` check retires it regardless.
        if lease_heartbeat is not None:
            lease_heartbeat.cancel()

        # Finalise the job record LAST, so the terminal status covers the whole
        # task including the deferred drive above (LR2 §8.6). Synchronous — safe
        # even on the cancellation path, which must not await. A record that was
        # reaped or taken over meanwhile refuses this transition rather than
        # overwriting a newer state.
        reporter.finish(job_status, job_message)


async def background_delete_documents(
    rag: LightRAG,
    doc_manager: DocumentManager,
    doc_ids: List[str],
    delete_file: bool = False,
    delete_llm_cache: bool = False,
    token: str | None = None,
):
    """Background task to delete multiple documents"""
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
        get_pipeline_ingress,
        release_owned_reservation,
    )

    total_docs = len(doc_ids)
    successful_deletions = []
    failed_deletions = []
    pipeline_status = None
    pipeline_status_lock = None

    try:
        # Fetch + initial job-status write INSIDE the release try: the delete
        # endpoint already reserved busy + destructive_busy (owner=token) before
        # scheduling us, so a cancellation delivered at these awaits must still
        # reach the finally and release the slot.
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )

        # The /documents/delete_document endpoint has already reserved the
        # destructive slot synchronously: ``busy=True`` and
        # ``destructive_busy=True`` were set before the client got
        # ``deletion_started``, after checking busy + scanning +
        # pending_enqueues>0 atomically.  Here we only update the
        # job-info fields; the busy reservation was acquired by the
        # endpoint and is released in the finally block below.
        async with pipeline_status_lock:
            pipeline_status.update(
                {
                    # Job name can not be changed, it's verified in adelete_by_doc_id()
                    "job_name": f"Deleting {total_docs} Documents",
                    "job_start": datetime.now().isoformat(),
                    "docs": total_docs,
                    "batchs": total_docs,
                    "cur_batch": 0,
                    "latest_message": "Starting document deletion process",
                }
            )
            # Use slice assignment to clear the list in place
            pipeline_status["history_messages"][:] = [
                "Starting document deletion process"
            ]
            if delete_llm_cache:
                append_pipeline_history(
                    pipeline_status, "LLM cache cleanup requested for this deletion job"
                )

        # Loop through each document ID and delete them one by one
        for i, doc_id in enumerate(doc_ids, 1):
            # Check for cancellation at the start of each document deletion
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    cancel_msg = f"Deletion cancelled by user at document {i}/{total_docs}. {len(successful_deletions)} deleted, {total_docs - i + 1} remaining."
                    logger.info(cancel_msg)
                    pipeline_status["latest_message"] = cancel_msg
                    append_pipeline_history(pipeline_status, cancel_msg)
                    # Add remaining documents to failed list with cancellation reason
                    failed_deletions.extend(
                        doc_ids[i - 1 :]
                    )  # i-1 because enumerate starts at 1
                    break  # Exit the loop, remaining documents unchanged

                start_msg = f"Deleting document {i}/{total_docs}: {doc_id}"
                logger.info(start_msg)
                pipeline_status.update({"cur_batch": i, "latest_message": start_msg})
                append_pipeline_history(pipeline_status, start_msg)

            file_path = "#"
            try:
                result = await rag.adelete_by_doc_id(
                    doc_id, delete_llm_cache=delete_llm_cache
                )
                file_path = (
                    getattr(result, "file_path", "-") if "result" in locals() else "-"
                )
                if result.status == "success":
                    successful_deletions.append(doc_id)
                    success_msg = (
                        f"Document deleted {i}/{total_docs}: {doc_id}[{file_path}]"
                    )
                    logger.info(success_msg)
                    async with pipeline_status_lock:
                        append_pipeline_history(pipeline_status, success_msg)

                    # Handle file deletion if requested and source information is available
                    if (
                        delete_file
                        and result.file_path
                        and result.file_path != UNKNOWN_FILE_SOURCE
                    ):
                        try:
                            deleted_files, file_delete_errors = (
                                delete_file_variants_by_file_path(
                                    doc_manager.input_dir,
                                    result.file_path,
                                )
                            )
                            for file_delete_error in file_delete_errors:
                                logger.warning(file_delete_error)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = (
                                        file_delete_error
                                    )
                                    append_pipeline_history(
                                        pipeline_status, file_delete_error
                                    )

                            if deleted_files:
                                file_delete_msg = (
                                    "Successfully deleted source files: "
                                    + ", ".join(deleted_files)
                                )
                                logger.info(file_delete_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = file_delete_msg
                                    append_pipeline_history(
                                        pipeline_status, file_delete_msg
                                    )
                            else:
                                file_error_msg = (
                                    "File deletion skipped, missing or unsafe file: "
                                    f"{result.file_path}"
                                )
                                logger.warning(file_error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = file_error_msg
                                    append_pipeline_history(
                                        pipeline_status, file_error_msg
                                    )

                        except Exception as file_error:
                            file_error_msg = f"Failed to delete file {result.file_path}: {str(file_error)}"
                            logger.error(file_error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = file_error_msg
                                append_pipeline_history(pipeline_status, file_error_msg)
                    elif delete_file:
                        no_file_msg = (
                            f"File deletion skipped, missing file path: {doc_id}"
                        )
                        logger.warning(no_file_msg)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = no_file_msg
                            append_pipeline_history(pipeline_status, no_file_msg)
                else:
                    failed_deletions.append(doc_id)
                    error_msg = f"Failed to delete {i}/{total_docs}: {doc_id}[{file_path}] - {result.message}"
                    logger.error(error_msg)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = error_msg
                        append_pipeline_history(pipeline_status, error_msg)

            except Exception as e:
                failed_deletions.append(doc_id)
                error_msg = f"Error deleting document {i}/{total_docs}: {doc_id}[{file_path}] - {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    append_pipeline_history(pipeline_status, error_msg)

    except Exception as e:
        error_msg = f"Critical error during batch deletion: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                append_pipeline_history(pipeline_status, error_msg)
    finally:
        # Final summary + release the destructive slot, owner-checked +
        # cancellation-resistant so a cancel here cannot wedge the slot or
        # clobber a later holder. Returns whether an indexing request is pending.

        # Resolve the ingress handle BEFORE the release: the pending-work
        # probe runs inside the owner-checked critical section below, and a
        # Manager RPC handle must never be resolved lazily while the status
        # lock is held.  A resolution failure fails TOWARD driving — a
        # spurious drive is a busy-gated cheap no-op, while skipping it would
        # silently defer work that arrived during the delete to the next
        # unrelated trigger.  This await precedes the cancellation-resistant
        # release — safe only because get_pipeline_ingress is suspension-free
        # by contract (see its docstring), so a pending re-cancellation
        # cannot be delivered here and skip the release.
        try:
            delete_exit_ingress = await get_pipeline_ingress(rag.workspace)
        except Exception as ingress_error:
            delete_exit_ingress = None
            logger.warning(
                "batch-delete exit: pipeline ingress unavailable; failing "
                f"toward the post-delete queue drive: {ingress_error}"
            )

        def _delete_release(status):
            completion_msg = f"Deletion completed: {len(successful_deletions)} successful, {len(failed_deletions)} failed"
            status.update(
                {
                    "busy": False,
                    "destructive_busy": False,
                    "busy_owner": None,
                    "operation_record": None,
                    "cancellation_requested": False,
                    "latest_message": completion_msg,
                }
            )
            append_pipeline_history(status, completion_msg)
            # Probe the mailbox INSIDE the same critical section that releases
            # the slot: a processing request refused while we held busy armed
            # the auto-rescan flag under this very lock, so it is either seen
            # here (drive below) or was armed after busy=False — in which case
            # the arming caller's own process call takes the slot itself.
            if delete_exit_ingress is None:
                return True  # fail toward driving
            try:
                return bool(delete_exit_ingress.has_work())
            except Exception as probe_error:
                logger.warning(
                    "batch-delete exit: ingress has_work probe failed; "
                    f"failing toward the post-delete queue drive: {probe_error}"
                )
                return True

        # Release the destructive slot with the fetch, lock and owner-checked
        # write ALL inside run_to_completion (release_owned_reservation), so a
        # cancellation delivered during the release — including a re-cancellation
        # of the namespace fetch after the initial fetch was already cancelled —
        # is retried rather than leaving busy/destructive_busy stuck. Returns the
        # pending-work probe result (or None when uninitialised / a later holder
        # owns it).
        has_pending_request = await release_owned_reservation(
            rag.workspace,
            owner_key="busy_owner",
            token=token,
            action=_delete_release,
        )

        # If there are pending requests, start document processing pipeline
        if has_pending_request:
            try:
                logger.info(
                    "Processing pending document indexing requests after deletion"
                )
                await rag.apipeline_process_enqueue_documents()
            except Exception as e:
                logger.error(f"Error processing pending documents after deletion: {e}")


def create_document_routes(
    rag: LightRAG, doc_manager: DocumentManager, api_key: Optional[str] = None
):
    # Fresh router per call — see the note above the temp_prefix constant.
    router = APIRouter(
        prefix="/documents",
        tags=["documents"],
    )

    # Create combined auth dependency for document routes
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/scan", response_model=ScanResponse, dependencies=[Depends(combined_auth)]
    )
    async def scan_for_new_documents(
        managed_tasks: set = Depends(get_managed_background_tasks),
    ):
        """
        Trigger the scanning process for new documents.

        Refuses to start a new scan with
        ``status='scanning_skipped_pipeline_busy'`` (and does not
        schedule a background task) when any of these is set:

        - ``pipeline_status["busy"]`` — the processing loop or another
          destructive job is running.
        - ``pipeline_status["scanning"]`` — another scan is already
          running (any phase: classification or processing).
        - ``pipeline_status["pending_enqueues"] > 0`` — an /upload,
          /text or /texts endpoint has reserved a slot whose bg task
          has not yet written to doc_status; starting a scan now would
          race scan's classification reads against that pending write.
        - ``pipeline_status["manual_freeze_requested"]`` — a manual
          retry is draining the pipeline for its exclusive reset.
        - an earlier un-ACKed manual retry request is still queued in
          the ingress mailbox: a scan runs its OWN exclusive FAILED
          reset, so starting it now would jump the manual FIFO and
          deadlock that request (the scan fence refuses its run) —
          LR2 §8.1. Boundary: a request whose driving task died
          (worker SIGKILL under a live Manager) stays queued until
          SOME processing run peeks it, so scans keep refusing until
          then; ``/documents/reprocess_failed`` both publishes and
          drives, and a run serves the EARLIEST request first, so it
          is the one-call recovery for that state.

        Both ``scanning`` and ``scanning_exclusive`` are acquired
        synchronously here so a subsequent fast-follow request hits the
        guard rather than racing against the not-yet-started task.
        ``run_scanning_process`` runs the exclusive FAILED reset and
        classification under ``scanning_exclusive``, then clears it,
        allowing concurrent uploads to land while the scan-driven
        processing finishes.

        Startup order (LR2 §8.6), each step compensating the previous
        ones if it fails: reservation → bounded job record → sticky
        manual intent → managed child (start barrier) → ``track_id``.
        The record exists BEFORE the response, so an immediate
        ``/documents/scan/status/{track_id}`` cannot 404. Once the
        child has taken over it owns the record and finalises it; the
        endpoint never cancels a job from that point on.

        Returns:
            ScanResponse: A response object containing the scanning status and track_id

        Raises:
            HTTPException: 429 when the scan job store is at capacity
                (all records are valid RUNNING jobs), 409 on a track-id
                collision, 503 when the job store is unavailable.
        """
        from lightrag.exceptions import PipelineNotInitializedError
        from lightrag.kg.pipeline_ingress import (
            ManualRetryPublishResult,
            PipelineIngressMessage,
        )
        from lightrag.kg.shared_storage import (
            ManualIntentRefusal,
            ManualIntentRefused,
            PipelineReservationConflict,
            acquire_reservation,
            get_namespace_data,
            get_namespace_lock,
            get_pipeline_ingress,
            start_committed_background_task,
            start_reserved_background_task,
        )

        # Generate track_id with "scan" prefix for scanning operation
        track_id = generate_track_id("scan")
        # Owner token for the scanning reservation, generated before any await.
        scanning_token = uuid4().hex
        # The scan's manual retry intent: published ONLY after the scan
        # reservation is granted (a refused scan must have zero side effects),
        # inside the committed-startup child. The scan itself serves it — the
        # exclusive FAILED reset runs before file discovery and ACKs the request
        # (LR2 §8.1) — and it is the only path pulling FAILED docs back into the
        # pipeline. Un-ACKed (reset abandoned / crash), it stays sticky for the
        # standard drain path.
        manual_request_id = uuid4().hex
        # Owner token of the bounded scan job record. Held by the endpoint until
        # the child takes over, so a startup that aborts can owner-checked cancel
        # exactly the record it created (never a successor's).
        job_owner_token = uuid4().hex
        job_created = False
        # Endpoint-visible mirror of the commit state: set synchronously with
        # the publish, so the finally below knows ownership transferred even
        # when the caller was cancelled right after the commit.
        takeover = {"committed": False}

        async def _legacy_scan_work(started):
            # Mocked-rig path (no pipeline_status): started.set() first so the
            # start-barrier confirms takeover; no reservation, no ingress.
            started.set()
            await run_scanning_process(rag, doc_manager, track_id, scanning_token)

        async def _scan_backstop():
            # Reverse compensation (LR2 §8.6), owner-checked + idempotent, and
            # only ever reached when the child did NOT take over: the job record
            # this endpoint created has nobody left to finalise it, so cancel it
            # before releasing the reservation. Both carry their owner token, so
            # a late compensation can never clean up a successor's job/scan.
            if job_created:
                _cancel_scan_job(rag, track_id, job_owner_token)
            await _release_scanning_reservation(rag, scanning_token)

        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
        except PipelineNotInitializedError:
            # Workspace pipeline_status not yet bootstrapped (e.g. mocked
            # test rigs).  Treat as idle and allow the scan to proceed; the
            # scanning flag has nowhere to live so it is effectively skipped.
            # Still start it as a managed task so shutdown can drain it.
            await start_reserved_background_task(
                managed_tasks, work=_legacy_scan_work, backstop_release=_scan_backstop
            )
            return ScanResponse(
                status="scanning_started",
                message="Scanning process has been initiated in the background",
                track_id=track_id,
            )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )

        # Atomically acquire the scanning flag.  Scan is the exclusive
        # writer in this contract — it reads doc_status to make
        # classification decisions (PROCESSED / resume / retry-as-new /
        # archive) and would race with concurrent writers — so refuse if:
        #   * pipeline is processing (busy=True): scan + processing both
        #     read/mutate doc_status; serialise.
        #   * another scan is in flight (scanning=True).
        #   * any /upload, /text, /texts endpoint has reserved a
        #     pending-enqueue slot (see _reserve_enqueue_slot): the bg
        #     task has not yet written doc_status and we would otherwise
        #     race with its mid-flight write.
        #   * a manual retry holds the enqueue freeze, or an earlier
        #     un-ACKed manual request is queued in the mailbox (LR2
        #     §8.1): this scan runs its own exclusive FAILED reset and
        #     must not jump that FIFO.
        #
        # Resolved BEFORE the reservation: the manual-FIFO check is one mailbox
        # call made inside the acquire's critical section, which must never
        # trigger a lazy Manager lookup while the status lock is held. A
        # resolution failure aborts with zero side effects (nothing reserved).
        ingress = await get_pipeline_ingress(rag.workspace)
        reserved = False
        handed_off = False
        try:
            # Pre-arm owner-checked cleanup before acquire: cancellation at the
            # helper's lock exit may occur after the reservation update but before
            # the result is returned.
            reserved = True
            result = await acquire_reservation(
                pipeline_status,
                pipeline_status_lock,
                owner_key="scanning_owner",
                owner=scanning_token,
                owner_kind="scan",
                flags={"scanning": True, "scanning_exclusive": True},
                reject_when=(
                    (
                        "busy",
                        "Pipeline is currently busy processing documents. Wait for "
                        "the running job to finish before triggering another scan.",
                    ),
                    (
                        "scanning",
                        "Another scan is already in progress. Wait for it to finish "
                        "before triggering a new one.",
                    ),
                    (
                        "pending_enqueues",
                        "An upload/insert or a source-conflict repair is in "
                        "flight. Wait for it to complete before triggering a scan.",
                    ),
                    (
                        "manual_freeze_requested",
                        "A manual retry is draining the pipeline for its exclusive "
                        "reset. Wait for it to finish before triggering a scan.",
                    ),
                ),
                pipeline_ingress=ingress,
                refuse_when_manual_pending=True,
            )
            if not result.acquired:
                reserved = False
                if result.conflict is PipelineReservationConflict.BUSY:
                    logger.warning(
                        "Scan request skipped: pipeline is busy processing documents"
                    )
                elif result.conflict is PipelineReservationConflict.SCANNING:
                    logger.warning(
                        "Scan request skipped: another scan is already in progress"
                    )
                elif result.conflict is PipelineReservationConflict.PENDING_ENQUEUE:
                    pending_enqueues = (result.snapshot or {}).get(
                        "pending_enqueues", 0
                    )
                    logger.warning(
                        "Scan request skipped: "
                        f"{pending_enqueues} pending enqueue(s) reserved by "
                        "upload/insert endpoints"
                    )
                elif result.conflict is PipelineReservationConflict.MANUAL_FREEZE:
                    logger.warning(f"Scan request skipped: {result.message}")
                return ScanResponse(
                    status="scanning_skipped_pipeline_busy",
                    message=result.message or "Pipeline reservation is unavailable.",
                    track_id=track_id,
                )

            # Create the bounded job record BEFORE publishing the manual intent
            # or starting the child (LR2 §8.6): the track_id we return must be
            # queryable immediately, and a refusal here must leave no intent and
            # no child behind. Store unavailable → 503; capacity exhausted (every
            # record a valid RUNNING job) → 429; a track-id collision → 409. The
            # finally releases the reservation for all three.
            try:
                job_store = _resolve_scan_job_store(rag)
                if job_store is None:
                    raise RuntimeError("scan job store is not initialised")
                create_result = job_store.create(track_id, job_owner_token)
            except Exception as job_error:
                logger.error(f"Scan job record could not be created: {job_error}")
                raise HTTPException(
                    status_code=503,
                    detail="Scan job store is unavailable; retry shortly.",
                )
            if create_result.outcome is ScanJobCreateOutcome.CAPACITY_EXCEEDED:
                logger.warning(
                    "Scan request refused: the scan job store is at capacity "
                    "(all records are running jobs)"
                )
                raise HTTPException(
                    status_code=429,
                    detail="Too many scan jobs are in flight; retry once one finishes.",
                )
            if create_result.outcome is ScanJobCreateOutcome.INVALID_IDENTIFIER:
                # Unreachable from here (both identifiers are generated), but the
                # store validates server-side on principle, so map its refusal
                # instead of treating it as success.
                logger.error(
                    f"Scan job record refused: {create_result.message or 'invalid identifier'}"
                )
                raise HTTPException(
                    status_code=500,
                    detail="internal_server_error",
                )
            if create_result.outcome is ScanJobCreateOutcome.ALREADY_EXISTS:
                # Unreachable for a freshly minted track id; a future caller
                # reusing one must be refused rather than silently adopting (and
                # then finalising) somebody else's job record.
                logger.error(
                    f"Scan job {track_id} already exists; refusing to reuse a track id"
                )
                raise HTTPException(
                    status_code=409,
                    detail="A scan job with this track id already exists.",
                )
            job_created = True

            # Hand the reservation to a managed background task via the
            # two-state committed startup: the child publishes the scan's
            # sticky manual retry intent (commit) and only then runs the scan.
            # A cancellation BEFORE the commit cancels the child and releases
            # the reservation with zero side effects (and cancels the job record
            # via the backstop); AFTER the commit the child is never cancelled —
            # it owns the reservation (released in run_scanning_process's
            # finally, owner-checked by scanning_token), the published intent AND
            # the job record (finalised in that same finally).

            async def _scan_commit(state):
                # Fence already passed — this endpoint holds the scanning
                # reservation. The commit only publishes the intent; the
                # committed marker is set synchronously with the publish so a
                # cancellation landing in the lock release already counts as
                # committed on both sides.
                async with pipeline_status_lock:
                    publish = ingress.request_manual_retry(
                        manual_request_id,
                        PipelineIngressMessage(
                            kind="rescan",
                            retry_failed=True,
                            request_id=manual_request_id,
                        ),
                    )
                    if publish is not ManualRetryPublishResult.ACCEPTED:
                        # Nothing was published, so nothing is owned: refuse the
                        # commit instead of starting a scan whose FAILED reset
                        # would never be acknowledged (LR2 §10.1). The endpoint's
                        # compensation chain releases the scanning reservation and
                        # cancels the job record because ``handed_off`` stays
                        # False.
                        return ManualIntentRefusal(
                            "Could not queue this scan's retry intent "
                            f"({publish.value}); no scan was started.",
                            capacity_exceeded=(
                                publish is ManualRetryPublishResult.CAPACITY_EXCEEDED
                            ),
                        )
                    state["committed"] = True
                    takeover["committed"] = True
                return None

            async def _scan_run():
                await run_scanning_process(
                    rag,
                    doc_manager,
                    track_id,
                    scanning_token,
                    manual_request_id=manual_request_id,
                    job_owner_token=job_owner_token,
                )

            await start_committed_background_task(
                managed_tasks,
                commit=_scan_commit,
                work=_scan_run,
                backstop_release=_scan_backstop,
            )
            # Ownership of the slot transferred to the bg task — it releases in
            # its finally. The endpoint's finally must NOT release it again.
            handed_off = True
            return ScanResponse(
                status="scanning_started",
                message="Scanning process has been initiated in the background",
                track_id=track_id,
            )
        except ManualIntentRefused as refusal:
            # The commit refused BEFORE publishing, so nothing is owned and the
            # finally below undoes the reservation and the job record. Capacity is
            # the one refusal worth retrying unchanged.
            raise HTTPException(
                status_code=429 if refusal.capacity_exceeded else 503,
                detail=str(refusal),
                headers=(
                    {"Retry-After": str(_ADMISSION_RETRY_AFTER_SECONDS)}
                    if refusal.capacity_exceeded
                    else None
                ),
            )
        finally:
            # Release the scanning slot if we reserved it but a cancellation
            # arrived before the managed task took over (e.g. at the
            # pipeline_status_lock __aexit__ await, before hand-off). Owner-
            # checked + idempotent, so a no-op once the task owns it (or the
            # start helper's own backstop already released it). A commit that
            # already happened (takeover mirror) means the child owns the slot
            # even if the caller was cancelled before ``handed_off`` was set.
            #
            # The job record follows the SAME ownership rule: not handed off means
            # nobody will finalise it, so cancel it here too (owner-checked and
            # idempotent — the startup helper's backstop may already have done it;
            # after the commit the child owns it and this must not fire).
            if reserved and not handed_off and not takeover["committed"]:
                if job_created:
                    _cancel_scan_job(rag, track_id, job_owner_token)
                await _release_scanning_reservation(rag, scanning_token)

    @router.get(
        "/scan/status/{track_id}",
        response_model=ScanJobStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_scan_job_status(track_id: str):
        """
        Report the bounded status of one scan job (LR2 §8.6).

        ``/documents/scan`` returns as soon as the job record exists and the
        background task has taken over, long before the directory walk, the
        FAILED reset and document processing finish. This endpoint reports that
        job's progress from the SAME bounded schema the store enforces —
        aggregate counters plus three capped sample buckets, never an
        ``O(total_files)`` file list.

        A job whose owner died stays ``running`` only until its lease expires and
        the store reaps it to ``abandoned``. Terminal records are kept for a TTL
        (and evicted under capacity pressure), so an old track_id eventually 404s.

        Raises:
            HTTPException: 404 when no job record exists for ``track_id``; 503
                when the job store is unavailable.
        """
        store = _resolve_scan_job_store(rag)
        if store is None:
            raise HTTPException(
                status_code=503, detail="Scan job store is unavailable."
            )
        try:
            record = store.get(track_id)
        except Exception as store_error:
            logger.error(f"Scan job status lookup failed: {store_error}")
            raise HTTPException(
                status_code=503, detail="Scan job store is unavailable."
            )
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"No scan job found for track_id {track_id}"
            )
        return ScanJobStatusResponse(**record)

    @router.get(
        "/source_conflicts",
        response_model=SourceConflictListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_source_conflicts(
        limit: int = Query(
            50,
            ge=1,
            le=_SOURCE_CONFLICT_PAGE_MAX,
            description="Maximum conflicts to return in this page",
        ),
        cursor: Optional[str] = Query(
            None, description="next_cursor from a previous page (opaque)"
        ),
    ):
        """
        List canonical source keys claimed by more than one primary document.

        A scan classifies such a file as ``source_conflict``: it is never
        enqueued, no record is deleted and the file is left in place, because
        picking a winner automatically could silently retire the wrong document
        (LR2 §5.5). This endpoint enumerates the conflicts an operator has to
        settle, and ``POST /documents/source_conflicts/repair`` settles them.

        The projection is bounded per key — an exact candidate count when the
        backend can produce one cheaply, plus a fixed-size sample of candidate
        doc IDs — so a workspace with a pathological conflict never materialises
        an unbounded ID list into the response.

        Raises:
            HTTPException: 400 for a malformed cursor; 501 when the configured
                doc_status backend cannot enumerate conflicts (no strict source
                resolution); 503 when the storage control plane is unavailable
                (e.g. a derived index still rebuilding).
        """
        position = _decode_source_conflict_cursor(cursor)
        try:
            page = await rag.doc_status.list_source_conflicts_page(
                limit=limit, position=position
            )
        except StorageCapabilityError as capability_error:
            logger.warning(f"Source-conflict listing unsupported: {capability_error}")
            raise HTTPException(
                status_code=501,
                detail=(
                    "The configured doc_status backend cannot enumerate source "
                    "conflicts (no strict source resolution)."
                ),
            )
        except (StorageControlPlaneError, StorageNotInitializedError) as storage_error:
            logger.error(f"Source-conflict listing failed: {storage_error}")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Document status storage is unavailable; retry once it is ready."
                ),
            )
        except Exception as e:
            logger.error(f"Error listing source conflicts: {e}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

        return SourceConflictListResponse(
            conflicts=[
                SourceConflictItem(
                    canonical_source_key=summary.canonical_source_key,
                    candidate_count=summary.candidate_count,
                    sample_doc_ids=list(summary.sample_doc_ids),
                )
                for summary in page.conflicts
            ],
            next_cursor=_encode_source_conflict_cursor(page.next_position),
        )

    @router.post(
        "/source_conflicts/repair",
        response_model=SourceConflictRepairResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def repair_source_conflict(
        payload: SourceConflictRepairRequest,
        http_request: Request,
    ):
        """
        Settle one source conflict by naming the document that keeps the source.

        Two-step, compare-and-set protocol (LR2 §5.5):

        1. ``dry_run=true`` (the default) mutates nothing and returns the
           current ``candidate_count`` and ``fingerprint`` — a digest over the
           candidate doc IDs in stable order.
        2. Echo both back with ``dry_run=false`` to commit. The backend re-reads
           the candidate set and refuses (409) when it changed in between, so a
           concurrent enqueue/delete is never silently overwritten. The commit
           runs under the canonical-key + enqueue-serialize locks, which is what
           keeps a NEW primary from being inserted between that re-read and the
           demotions (no backend can block that phantom on its own — see
           ``DocStatusStorage.repair_source_conflict``), and what orders the
           commit against the processing stage's duplicate marking, which takes
           the same canonical-key lock.

        The commit marks every candidate other than ``primary_doc_id`` with
        ``metadata.is_duplicate=true`` and ``original_doc_id=<primary>``. Content
        is never deleted and the demoted rows keep their own status — they only
        lose their claim on the canonical source, which is what makes the strict
        resolver return a single primary afterwards.

        Repeat safety: a committed repair leaves exactly one primary, so
        replaying the SAME request 409s (its token is stale) while a fresh
        dry-run → commit converges to a no-op. A repair that dies mid-way is
        resumed the same way — the finished rows already express ``duplicate``
        and the unfinished ones still resolve as a conflict; there is no repair
        marker to reconcile.

        Every call is audited: dry-runs at INFO, commits and refusals at WARNING,
        with the caller's address and the operator-supplied identifiers.

        Raises:
            HTTPException: 409 when ``primary_doc_id`` is not a current primary
                candidate or the CAS tokens are stale; 501 when the backend
                cannot repair conflicts; 503 when storage is unavailable; 422
                when a commit omits the CAS tokens.
        """
        actor = http_request.client.host if http_request.client else "unknown"
        audit_key = safe_log_value(payload.canonical_source_key)
        audit_primary = safe_log_value(payload.primary_doc_id)
        action = "dry-run" if payload.dry_run else "COMMIT"
        try:
            if payload.dry_run:
                result = await rag.doc_status.repair_source_conflict(
                    payload.canonical_source_key,
                    primary_doc_id=payload.primary_doc_id,
                    expected_candidate_count=(
                        payload.expected_candidate_count
                        if payload.expected_candidate_count is not None
                        else 0
                    ),
                    expected_candidate_fingerprint=(
                        payload.expected_candidate_fingerprint or ""
                    ),
                    dry_run=True,
                )
            else:
                # Only the COMMIT needs the locks: the operator's two requests
                # are bridged by the CAS token, so what must be serialized is
                # the backend's internal re-read → demote span, not the gap
                # between the two HTTP calls.
                async with source_conflict_repair_lock(
                    rag.workspace, payload.canonical_source_key
                ):
                    # Two primaries can never keep the source: a stub with no
                    # content (a scan would delete it out from under the
                    # demotions) and one whose content already lives under
                    # another document (processing will mark it a duplicate).
                    # Refuse inside the lock, where the answers cannot go stale.
                    await refuse_an_unusable_primary(
                        rag.doc_status,
                        rag.full_docs,
                        payload.canonical_source_key,
                        payload.primary_doc_id,
                    )
                    result = await rag.doc_status.repair_source_conflict(
                        payload.canonical_source_key,
                        primary_doc_id=payload.primary_doc_id,
                        expected_candidate_count=(
                            payload.expected_candidate_count
                            if payload.expected_candidate_count is not None
                            else 0
                        ),
                        expected_candidate_fingerprint=(
                            payload.expected_candidate_fingerprint or ""
                        ),
                        dry_run=False,
                    )
                    # Verify the end state inside the locks rather than reporting
                    # a settled key on the strength of "the demotions landed".
                    # The locks exclude every in-deployment writer of this
                    # candidate set — enqueue, clear/delete + scan classification
                    # + manual reset (the reservation), and the processing
                    # stage's duplicate marking (the keyed source lock it takes
                    # too) — so this is a backstop for what stays outside them: a
                    # second deployment writing the same database. It raises when
                    # the key came out Absent or still conflicting.
                    await verify_repair_outcome(
                        rag.doc_status,
                        payload.canonical_source_key,
                        payload.primary_doc_id,
                        result,
                    )
        except SourceConflictPrimaryUnusableError as unusable_primary:
            # A DIFFERENT 409 from the one below: this primary IS a candidate, it
            # just cannot own the source. Reporting it as "not a candidate" would
            # send the operator to re-list a conflict that has not changed. The
            # detail is built here from sanitized values rather than forwarded
            # from the exception text — so it must branch on the exception's
            # REASON, or it states the wrong cause with total confidence (it used
            # to answer "has no full_docs content" for a document whose content
            # was fine and simply belonged to another document).
            logger.warning(
                f"[source-conflict repair] {action} REFUSED by {actor}: "
                f"key='{audit_key}' primary={audit_primary}: {unusable_primary}"
            )
            if (
                unusable_primary.reason
                == SourceConflictPrimaryUnusableError.REASON_CONTENT_ELSEWHERE
            ):
                holder = safe_log_value(unusable_primary.holder_doc_id)
                detail = (
                    f"Document {audit_primary} holds the same content as "
                    f"{holder or 'another document'}, which claims a different "
                    f"source, so it cannot own '{audit_key}': processing marks a "
                    "content duplicate FAILED and deletes its content, leaving "
                    "the source with no primary. Choose another primary, or "
                    "settle that document first."
                )
            else:
                detail = (
                    f"Document {audit_primary} has no full_docs content, so it "
                    f"cannot own source '{audit_key}'. Choose a primary that has "
                    "content."
                )
            raise HTTPException(status_code=409, detail=detail)
        except ValueError as not_a_candidate:
            # The named primary is not currently a primary candidate: either a
            # typo or a view that went stale. Same recovery either way — list the
            # conflict again — so it is a state conflict, not a 400.
            logger.warning(
                f"[source-conflict repair] {action} REFUSED by {actor}: "
                f"key='{audit_key}' primary={audit_primary}: {not_a_candidate}"
            )
            # Built from the (sanitized) request values rather than the storage
            # message: the exception text is not guaranteed to be client-safe.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Document {audit_primary} is not a current primary candidate "
                    f"for source '{audit_key}'; list the conflict again to see the "
                    "current candidates."
                ),
            )
        except SourceConflictRepairCASError as cas_error:
            # Subclass of StorageControlPlaneError, so it MUST be caught first:
            # the state here is known (the candidate set moved), which is a 409,
            # not the parent's "state unknown" 503.
            logger.warning(
                f"[source-conflict repair] {action} REFUSED by {actor}: "
                f"key='{audit_key}' primary={audit_primary}: {cas_error}"
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    "The candidate set changed since the dry-run; re-run the "
                    "dry-run and commit with the new count/fingerprint."
                ),
            )
        except StorageCapabilityError as capability_error:
            logger.warning(f"Source-conflict repair unsupported: {capability_error}")
            raise HTTPException(
                status_code=501,
                detail=(
                    "The configured doc_status backend cannot repair source "
                    "conflicts (no strict source resolution)."
                ),
            )
        except (StorageControlPlaneError, StorageNotInitializedError) as storage_error:
            logger.error(
                f"[source-conflict repair] {action} FAILED by {actor}: "
                f"key='{audit_key}' primary={audit_primary}: {storage_error}"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Document status storage is unavailable; retry once it is ready."
                ),
            )
        except Exception as e:
            logger.error(
                f"[source-conflict repair] {action} ERRORED by {actor}: "
                f"key='{audit_key}' primary={audit_primary}: {e}"
            )
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

        audit_line = (
            f"[source-conflict repair] {action} by {actor}: key='{audit_key}' "
            f"primary={audit_primary} candidates={result.candidate_count} "
            f"fingerprint={result.fingerprint} "
            f"demoted_sample={list(result.demoted_sample_doc_ids)}"
        )
        if result.committed:
            logger.warning(audit_line)
        else:
            logger.info(audit_line)

        return SourceConflictRepairResponse(
            canonical_source_key=result.canonical_source_key,
            primary_doc_id=result.primary_doc_id,
            candidate_count=result.candidate_count,
            fingerprint=result.fingerprint,
            demoted_sample_doc_ids=list(result.demoted_sample_doc_ids),
            committed=result.committed,
        )

    @router.post(
        "/upload", response_model=InsertResponse, dependencies=[Depends(combined_auth)]
    )
    async def upload_to_input_dir(
        managed_tasks: set = Depends(get_managed_background_tasks),
        file: UploadFile = File(...),
        http_request: Request = None,
    ):
        """
        Upload a file to the input directory and index it.

        This API endpoint accepts a file through an HTTP POST request, checks if the
        uploaded file is of a supported type, saves it in the specified input directory,
        indexes it for retrieval, and returns a success status with relevant details.

        **File Size Limit:**
        - Configurable via `MAX_UPLOAD_SIZE` environment variable (default: 100MB)
        - Set to `None` or `0` for unlimited upload size
        - Returns HTTP 413 (Request Entity Too Large) if file exceeds limit

        **Duplicate Detection Behavior:**

        This endpoint handles two types of duplicate scenarios differently:

        1. **Filename Duplicate (Synchronous Detection)**:
           - Detected immediately, before any file is written.
           - File name is treated as the unique document key.  Both
             ``doc_status`` and the INPUT directory are checked under the
             canonical (parser-hint stripped) basename so ``abc.docx`` and
             ``abc.[native].docx`` map to the same record.
           - **HTTP 409** is returned when a same-name record already exists.
             The response detail names the conflict source ("Document
             storage already contains ..." or "Input directory already
             contains ...").  Clients must delete the existing document
             (``DELETE /documents/{doc_id}``) before re-uploading; there is
             no longer a 200 ``status="duplicated"`` soft-fail response.

        2. **Content Duplicate (Asynchronous Detection)**:
           - Detected during background processing after content extraction
           - Returns `status="success"` with a new track_id immediately
           - The duplicate is detected later when processing the file content
           - Use `/documents/track_status/{track_id}` to check the final result:
             - Document will have `status="FAILED"`
             - `error_msg` contains "Content already exists. Original doc_id: xxx"
             - `metadata.is_duplicate=true` with reference to original document
             - `metadata.original_doc_id` points to the existing document
             - `metadata.original_track_id` shows the original upload's track_id

        **Why Different Behavior?**
        - Filename check is fast (simple lookup), done synchronously
        - Content extraction is expensive (PDF/DOCX parsing), done asynchronously
        - This design prevents blocking the client during expensive operations

        **Concurrency Constraint:**
        - The endpoint refuses with HTTP 409 only while one of the
          following exclusive-writer states is set:
          ``pipeline_status["scanning_exclusive"]`` (a scan is in its
          classification phase, reading and possibly mutating doc_status)
          or ``pipeline_status["destructive_busy"]`` (``/documents/clear``
          or per-doc delete is dropping storages / removing input files).
          Wait for the running job to finish before re-submitting.
        - ``busy=True`` from the processing loop, and a scan in its
          processing phase (``scanning=True`` with
          ``scanning_exclusive=False``), do NOT block uploads — uploads
          are accepted concurrently and the running pipeline picks them
          up via the ingress mailbox.

        Args:
            managed_tasks: injected managed background-task set
                (see get_managed_background_tasks) — the reservation-holding work
                runs as a tracked asyncio task, not a Starlette callback
            file (UploadFile): The file to be uploaded. It must have an allowed extension.

        Returns:
            InsertResponse: A response object containing the upload status and a message.
                - status="success": File accepted and queued for processing

        Raises:
            HTTPException: 400 unsupported file type, 409 same-name
                conflict or scan-classifying / destructive job in
                flight, 413 file too large, 500 other errors.
        """
        from lightrag.kg.shared_storage import start_reserved_background_task

        enqueue_token, admission_adopted = _adopt_or_new_enqueue_token(http_request)
        handed_off = False
        try:
            # Reject upload while a scan is in its CLASSIFICATION
            # phase or a destructive job (clear / per-doc delete) is
            # in flight, AND reserve a pending-enqueue slot so a scan
            # request that arrives before the bg task runs cannot
            # transition scanning_exclusive=True under us.  Concurrent
            # processing (``busy=True``) and a scan in its processing
            # phase (``scanning=True`` with
            # ``scanning_exclusive=False``) are permitted: the running
            # loop picks up our doc via the ingress mailbox, mid-batch
            # or at the batch boundary.
            if not admission_adopted:
                await _reserve_enqueue_slot(rag, enqueue_token)

            # Sanitize filename to prevent Path Traversal attacks
            safe_filename = sanitize_filename(file.filename, doc_manager.input_dir)

            try:
                filename_supported = doc_manager.is_supported_file(safe_filename)
            except FilenameParserHintError as hint_error:
                # Reject malformed hints synchronously with the detailed
                # message (previously surfaced asynchronously as an error
                # document after the upload was accepted).
                raise HTTPException(status_code=400, detail=str(hint_error))
            if not filename_supported:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            # Check file size limit (if configured)
            if (
                global_args.max_upload_size is not None
                and global_args.max_upload_size > 0
            ):
                # Safe access to file size (not available in older Starlette versions)
                file_size = getattr(file, "size", None)

                # Pre-flight size check (only if size is available)
                if file_size is not None:
                    if file_size > global_args.max_upload_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {global_args.max_upload_size / 1024 / 1024:.1f}MB, uploaded: {file_size / 1024 / 1024:.1f}MB",
                        )
                else:
                    # If size not available, we'll check during streaming
                    logger.debug(
                        f"File size not available in UploadFile for {safe_filename}, will check during streaming"
                    )

            file_path = doc_manager.input_dir / safe_filename

            # Strict name pre-check.  Both the INPUT directory and doc_status
            # must be free of any same-canonical-basename record before we
            # accept the upload.  Replacing an existing document requires an
            # explicit DELETE first; we no longer write a "duplicated" 200
            # response that silently no-ops.
            existing_doc_data = await get_existing_doc_by_file_path_candidates(
                rag.doc_status, file_path
            )
            if existing_doc_data:
                status = get_doc_status_value(existing_doc_data) or "unknown"
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Document storage already contains '{safe_filename}' "
                        f"(Status: {status}). Delete the existing record before re-uploading."
                    ),
                )

            # INPUT directory check, using canonical parser-hint names.
            # Fast path: exact filename match avoids iterdir on large input directories.
            canonical_filename = normalize_file_path(safe_filename)
            if file_path.exists():
                existing_input_file: Path | None = file_path
            else:
                existing_input_file = find_existing_file_by_file_path(
                    doc_manager.input_dir, canonical_filename
                )
            if existing_input_file:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Input directory already contains a file with the same "
                        f"canonical basename ('{existing_input_file.name}'). "
                        f"Remove or rename it before re-uploading."
                    ),
                )

            # Async streaming write with size check
            bytes_written = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            needs_cleanup = False

            async with aiofiles.open(file_path, "wb") as out_file:
                while True:
                    # Read chunk from upload stream
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break

                    # Check size limit during streaming (if not checked before)
                    if (
                        global_args.max_upload_size is not None
                        and global_args.max_upload_size > 0
                    ):
                        bytes_written += len(chunk)
                        if bytes_written > global_args.max_upload_size:
                            needs_cleanup = True
                            break

                    # Write chunk to file
                    await out_file.write(chunk)

            # Cleanup after file is closed
            if needs_cleanup:
                try:
                    file_path.unlink()
                except Exception as cleanup_error:
                    logger.error(
                        f"Error cleaning up oversized file {safe_filename}: {cleanup_error}"
                    )

                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {global_args.max_upload_size / 1024 / 1024:.1f}MB, uploaded: {bytes_written / 1024 / 1024:.1f}MB",
                )

            track_id = generate_track_id("upload")

            # Bg task: enqueue + trigger processing, then release the slot.
            # ``pipeline_index_file`` does both: it calls
            # ``pipeline_enqueue_file`` (writes doc_status / full_docs) and
            # then ``apipeline_process_enqueue_documents``.  The latter is
            # safe to invoke even when the loop is already busy — its
            # refused reservation arms the auto-rescan flag and returns,
            # so concurrent uploads/inserts cooperate via the running
            # loop's quiescence decision.
            async def _indexing_work(started):
                # started.set() first (no await before it) so the endpoint's
                # start-barrier confirms takeover before returning; a body-send
                # cancellation therefore cannot strand the enqueue slot.
                started.set()
                try:
                    await pipeline_index_file(
                        rag,
                        file_path,
                        track_id,
                        admission_token=enqueue_token,
                    )
                finally:
                    await _release_enqueue_slot(rag, enqueue_token)

            async def _enqueue_backstop():
                await _release_enqueue_slot(rag, enqueue_token)

            await start_reserved_background_task(
                managed_tasks,
                work=_indexing_work,
                backstop_release=_enqueue_backstop,
            )
            # Ownership of the slot transferred to the bg task — the
            # finally block below must NOT release it again.
            handed_off = True

            return InsertResponse(
                status="success",
                message=f"File '{safe_filename}' uploaded successfully. Processing will continue in background.",
                track_id=track_id,
            )

        except HTTPException:
            # Re-raise HTTP exceptions (400, 413, etc.)
            raise
        except Exception as e:
            logger.error(f"Error /documents/upload: {file.filename}: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)
        finally:
            # If we reserved a slot but never scheduled the bg task
            # (e.g. early validation rejection or streaming-write
            # failure), release here.  No drain coordination needed —
            # any sibling bg task triggers its own processing pass.
            if not handed_off:
                await _release_enqueue_slot(rag, enqueue_token)

    @router.post(
        "/text", response_model=InsertResponse, dependencies=[Depends(combined_auth)]
    )
    async def insert_text(
        request: InsertTextRequest,
        managed_tasks: set = Depends(get_managed_background_tasks),
        http_request: Request = None,
    ):
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.

        **Concurrency Constraint:**
        - Refuses with HTTP 409 only while
          ``pipeline_status["scanning_exclusive"]`` (a scan is in its
          classification phase) or ``pipeline_status["destructive_busy"]``
          (clear / per-doc delete is in flight) is set.  ``busy=True``
          from the processing loop, and a scan in its processing phase,
          do NOT block — the running pipeline picks up the new doc via
          the ingress mailbox.

        Args:
            request (InsertTextRequest): The request body containing the text to be inserted.
            managed_tasks: injected managed background-task set
                (see get_managed_background_tasks) — the reservation-holding work
                runs as a tracked asyncio task, not a Starlette callback

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: 400 invalid file_source, 409 same-name conflict
                or scan/destructive job in flight, 500 other errors.
        """
        from lightrag.kg.shared_storage import start_reserved_background_task

        enqueue_token, admission_adopted = _adopt_or_new_enqueue_token(http_request)
        handed_off = False
        try:
            # Reject text insertion while a scan is in progress AND reserve
            # a pending-enqueue slot — see /upload for the rationale.
            if not admission_adopted:
                await _reserve_enqueue_slot(rag, enqueue_token)

            # Check if file_source already exists in doc_status storage
            if not is_valid_file_source(request.file_source):
                raise HTTPException(
                    status_code=400,
                    detail="A valid file_source is required for text insertion",
                )

            normalized_file_source = normalize_file_path(request.file_source)
            existing_doc_data = await get_existing_doc_by_file_path_candidates(
                rag.doc_status, normalized_file_source
            )
            if existing_doc_data:
                status = get_doc_status_value(existing_doc_data) or "unknown"
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Document storage already contains '{normalized_file_source}' "
                        f"(Status: {status}). Delete the existing record before re-inserting."
                    ),
                )

            # Resolve + validate chunking synchronously so an invalid
            # effective config (e.g. chunk_token_size below the inherited
            # overlap) fails with HTTP 422 here, before any background work is
            # scheduled. pipeline_index_texts re-resolves from the same
            # addon_params inside the task.
            try:
                _resolve_text_chunking(request.chunking, rag)
            except ValueError as exc:
                # Controlled chunking-config validation message (numeric sizes
                # only, no internal detail); kept as client-facing 422 feedback
                # but wrapped so it is never a bare raw-exception passthrough.
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid chunking configuration: {exc}",
                )

            # Generate track_id for text insertion
            track_id = generate_track_id("insert")

            async def _indexing_work(started):
                # started.set() first (no await before it) so the endpoint's
                # start-barrier confirms takeover before returning; a body-send
                # cancellation therefore cannot strand the enqueue slot.
                started.set()
                try:
                    await pipeline_index_texts(
                        rag,
                        [request.text],
                        file_sources=[normalized_file_source],
                        track_id=track_id,
                        chunking=request.chunking,
                        admission_token=enqueue_token,
                    )
                finally:
                    await _release_enqueue_slot(rag, enqueue_token)

            async def _enqueue_backstop():
                await _release_enqueue_slot(rag, enqueue_token)

            await start_reserved_background_task(
                managed_tasks,
                work=_indexing_work,
                backstop_release=_enqueue_backstop,
            )
            handed_off = True

            return InsertResponse(
                status="success",
                message="Text successfully received. Processing will continue in background.",
                track_id=track_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error /documents/text: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)
        finally:
            if not handed_off:
                await _release_enqueue_slot(rag, enqueue_token)

    @router.post(
        "/texts",
        response_model=InsertResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def insert_texts(
        request: InsertTextsRequest,
        managed_tasks: set = Depends(get_managed_background_tasks),
        http_request: Request = None,
    ):
        """
        Insert multiple texts into the RAG system.

        This endpoint allows you to insert multiple text entries into the RAG system
        in a single request.

        **Concurrency Constraint:**
        - Refuses with HTTP 409 only while
          ``pipeline_status["scanning_exclusive"]`` (a scan is in its
          classification phase) or ``pipeline_status["destructive_busy"]``
          (clear / per-doc delete is in flight) is set.  ``busy=True``
          from the processing loop, and a scan in its processing phase,
          do NOT block — the running pipeline picks up the new docs via
          the ingress mailbox.

        Args:
            request (InsertTextsRequest): The request body containing the list of texts.
            managed_tasks: injected managed background-task set
                (see get_managed_background_tasks) — the reservation-holding work
                runs as a tracked asyncio task, not a Starlette callback

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: 400 invalid file_sources, 409 same-name
                conflict or scan/destructive job in flight, 413 more texts
                than ``MAX_TEXTS_PER_REQUEST`` allows, 500 other errors.
        """
        from lightrag.kg.shared_storage import start_reserved_background_task

        enqueue_token, admission_adopted = _adopt_or_new_enqueue_token(http_request)
        handed_off = False
        try:
            # Before anything that costs a storage round-trip or a reservation:
            # a batch over the per-request ceiling is refused on its shape alone,
            # independent of pipeline state.
            _enforce_texts_per_request(len(request.texts))

            # Reject batch text insertion while a scan is in progress AND
            # reserve a pending-enqueue slot — see /upload for the rationale.
            if not admission_adopted:
                await _reserve_enqueue_slot(rag, enqueue_token)

            # Check if any file_sources already exist in doc_status storage
            if not request.file_sources or len(request.file_sources) != len(
                request.texts
            ):
                raise HTTPException(
                    status_code=400,
                    detail="A valid file_source is required for each text",
                )

            normalized_file_sources = [
                normalize_file_path(file_source) for file_source in request.file_sources
            ]
            if any(
                file_source == UNKNOWN_FILE_SOURCE
                for file_source in normalized_file_sources
            ):
                raise HTTPException(
                    status_code=400,
                    detail="A valid file_source is required for each text",
                )
            if len(set(normalized_file_sources)) != len(normalized_file_sources):
                raise HTTPException(
                    status_code=400,
                    detail="file_sources must be unique by filename",
                )

            for file_source in normalized_file_sources:
                existing_doc_data = await get_existing_doc_by_file_path_candidates(
                    rag.doc_status, file_source
                )
                if existing_doc_data:
                    status = get_doc_status_value(existing_doc_data) or "unknown"
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Document storage already contains '{file_source}' "
                            f"(Status: {status}). Delete the existing record before re-inserting."
                        ),
                    )

            # Resolve + validate the shared chunking synchronously so an
            # invalid effective config (e.g. chunk_token_size below the
            # inherited overlap) fails with HTTP 422 here, before any
            # background work is scheduled. pipeline_index_texts re-resolves
            # from the same addon_params inside the task.
            try:
                _resolve_text_chunking(request.chunking, rag)
            except ValueError as exc:
                # Controlled chunking-config validation message (numeric sizes
                # only, no internal detail); kept as client-facing 422 feedback
                # but wrapped so it is never a bare raw-exception passthrough.
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid chunking configuration: {exc}",
                )

            # The reservation was taken for a single document before the body
            # was known; this request actually brings N. Re-weight the SAME
            # token now (LR2 §9.3 item 5) so a batch that does not fit is
            # refused here with 429 instead of being accepted and then
            # rejected inside a background task the client cannot observe.
            await _reweight_enqueue_slot(rag, enqueue_token, len(request.texts))

            # Generate track_id for texts insertion
            track_id = generate_track_id("insert")

            async def _indexing_work(started):
                # started.set() first (no await before it) so the endpoint's
                # start-barrier confirms takeover before returning; a body-send
                # cancellation therefore cannot strand the enqueue slot.
                started.set()
                try:
                    await pipeline_index_texts(
                        rag,
                        request.texts,
                        file_sources=normalized_file_sources,
                        track_id=track_id,
                        chunking=request.chunking,
                        admission_token=enqueue_token,
                    )
                finally:
                    await _release_enqueue_slot(rag, enqueue_token)

            async def _enqueue_backstop():
                await _release_enqueue_slot(rag, enqueue_token)

            await start_reserved_background_task(
                managed_tasks,
                work=_indexing_work,
                backstop_release=_enqueue_backstop,
            )
            handed_off = True

            return InsertResponse(
                status="success",
                message="Texts successfully received. Processing will continue in background.",
                track_id=track_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error /documents/texts: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)
        finally:
            if not handed_off:
                await _release_enqueue_slot(rag, enqueue_token)

    @router.delete(
        "", response_model=ClearDocumentsResponse, dependencies=[Depends(combined_auth)]
    )
    async def clear_documents():
        """
        Clear all documents from the RAG system.

        This endpoint deletes all documents, entities, relationships, and files from the system.
        It uses the storage drop methods to properly clean up all data and removes all files
        from the input directory.

        **Concurrency Constraint:**
        - Atomically reserves the destructive slot (sets ``busy=True``
          and ``destructive_busy=True``) before dropping anything.
          Refuses with ``status="busy"`` when ANY of these is set:
          ``pipeline_status["busy"]`` (processing loop or another
          destructive job in flight), ``pipeline_status["scanning"]``
          (a scan is anywhere in its lifecycle), or
          ``pipeline_status["pending_enqueues"] > 0`` (an /upload,
          /text or /texts has reserved a slot whose bg task has not
          yet written to doc_status).

        Returns:
            ClearDocumentsResponse: A response object containing the status and message.
                - status="success":           All documents and files were successfully cleared.
                - status="partial_success":   Document clear job exit with some errors.
                - status="busy":              Operation could not be completed because another
                  writer (busy / scanning / pending enqueue) holds the pipeline.
                - status="fail":              All storage drop operations failed, with message
                - message: Detailed information about the operation results, including counts
                  of deleted files and any errors encountered.

        Raises:
            HTTPException: Raised when a serious error occurs during the clearing process,
                          with status code 500 and error details in the detail field.
        """
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
            get_pipeline_ingress,
        )

        # Get pipeline status and lock
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )

        # Atomically reserve the destructive slot.  Checks busy +
        # scanning + pending_enqueues>0 in a single critical section
        # before flipping busy=True and destructive_busy=True together.
        # ``destructive_busy`` blocks reservation and the enqueue
        # last-line guard: clear is about to drop every storage and
        # remove every input file, so a concurrent upload accepted in
        # this window would write to storages mid-drop and silently
        # lose the document.
        destructive_token = uuid4().hex
        # Acquire INSIDE the try/finally so a cancellation delivered while
        # ``_acquire_destructive_busy`` is exiting its own lock — busy /
        # destructive_busy / busy_owner already written but ``acquired`` not yet
        # assigned — still runs the owner-checked release in ``finally`` (which
        # also covers the status-init lock await below). Mirrors the delete
        # endpoint. Release is owner-checked + idempotent: a no-op when the
        # acquire returned busy and never stamped our token.
        try:
            acquired, reason = await _acquire_destructive_busy(
                rag,
                destructive_token,
                kind="clear",
                operation_record={"kind": "clear", "scope": "all"},
            )
            if not acquired:
                return ClearDocumentsResponse(status="busy", message=reason)
            async with pipeline_status_lock:
                pipeline_status.update(
                    {
                        "job_name": "Clearing Documents",
                        "job_start": datetime.now().isoformat(),
                        "docs": 0,
                        "batchs": 0,
                        "cur_batch": 0,
                        "latest_message": "Starting document clearing process",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
                append_pipeline_history(
                    pipeline_status, "Starting document clearing process"
                )

            # We own busy+destructive: every document the mailbox refers to is
            # about to be dropped, so clear the ingress alongside the
            # job-status reset above — un-ACKed manual retry requests
            # are retired as CANCELLED_BY_CLEAR (a delayed replay of the same
            # request id is refused), and the document/auto channels are
            # emptied.  ``destructive_busy`` already refuses new enqueues, so
            # nothing repopulates the mailbox during the drop.  Clearing
            # BEFORE the drops means a later partial drop failure has already
            # retired manual requests for still-live FAILED docs — acceptable
            # under the destructive intent and surfaced via partial_success.
            # A failure here only degrades: residual stale messages are
            # compacted by the next run's consumption idempotence, and a
            # surviving sticky request strict-scans the emptied doc_status and
            # ACKs harmlessly.
            try:
                ingress = await get_pipeline_ingress(rag.workspace)
                ingress.clear()
            except Exception as ingress_clear_error:
                logger.warning(
                    "/documents/clear: failed to clear the pipeline ingress "
                    "mailbox; safe to continue — residual document messages "
                    "are compacted by the next run and a surviving manual "
                    "retry request ACKs harmlessly against the emptied "
                    f"doc_status: {ingress_clear_error}"
                )

            # Scan job records describe documents that are about to be dropped,
            # so retire the ones nobody owns: only TERMINAL (and lease-expired →
            # ABANDONED, which the snapshot reaps first) records are removed. A
            # still-valid RUNNING job is never removed out from under its owner
            # (LR2 §8.6) — a live scan cannot even coexist with this clear, since
            # the destructive reservation refuses while ``scanning`` is held.
            try:
                job_store = _resolve_scan_job_store(rag)
                if job_store is not None:
                    for record in job_store.snapshot():
                        if record.get("status") != ScanJobStatus.RUNNING.value:
                            job_store.remove_terminal(record["track_id"])
            except Exception as job_clear_error:
                logger.warning(
                    "/documents/clear: failed to retire finished scan job "
                    "records; safe to continue — they expire by TTL and are "
                    f"evicted under capacity pressure: {job_clear_error}"
                )

            # Use drop method to clear all data
            drop_tasks = []
            storages = [
                rag.text_chunks,
                rag.full_docs,
                rag.full_entities,
                rag.full_relations,
                rag.entity_chunks,
                rag.relation_chunks,
                rag.entities_vdb,
                rag.relationships_vdb,
                rag.chunks_vdb,
                rag.chunk_entity_relation_graph,
                rag.doc_status,
            ]

            # Log storage drop start
            append_pipeline_history(
                pipeline_status, "Starting to drop storage components"
            )

            for storage in storages:
                if storage is not None:
                    drop_tasks.append(storage.drop())

            # Wait for all drop tasks to complete
            drop_results = await asyncio.gather(*drop_tasks, return_exceptions=True)

            # Check for errors and log results
            errors = []
            storage_success_count = 0
            storage_error_count = 0

            for i, result in enumerate(drop_results):
                storage_name = storages[i].__class__.__name__
                if isinstance(result, Exception):
                    error_msg = f"Error dropping {storage_name}: {str(result)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    storage_error_count += 1
                elif isinstance(result, dict) and result.get("status") != "success":
                    # drop() reports a non-raising failure as {"status": "error"}
                    # (e.g. a backend that could not safely clear a kept legacy
                    # store). Honor it so the clear is not counted as successful
                    # while stale data remains and could be re-migrated/resurface.
                    error_msg = (
                        f"Error dropping {storage_name}: "
                        f"{result.get('message', 'unknown error')}"
                    )
                    errors.append(error_msg)
                    logger.error(error_msg)
                    storage_error_count += 1
                else:
                    namespace = storages[i].namespace
                    workspace = storages[i].workspace
                    logger.info(
                        f"Successfully dropped {storage_name}: {workspace}/{namespace}"
                    )
                    storage_success_count += 1

            # Log storage drop results
            if storage_error_count > 0:
                append_pipeline_history(
                    pipeline_status,
                    f"Dropped {storage_success_count} storage components with {storage_error_count} errors",
                )
            else:
                append_pipeline_history(
                    pipeline_status,
                    f"Successfully dropped all {storage_success_count} storage components",
                )

            # Some backends (OpenSearch) drop the doc_status index as a whole
            # physical container rather than just its rows, and gate every
            # subsequent STRICT read behind a readiness flag that only a
            # write path clears back to healthy. /documents/scan's very first
            # doc_status touch is a strict READ (the custom-chunk rollback,
            # then the exclusive FAILED->PENDING reset) — neither is a write,
            # so nothing would ever re-create the dropped index before the
            # first scan tries to read it, permanently wedging every scan on
            # this workspace until the process restarts. re-run the public,
            # idempotent initialize() right away so the backend is exactly as
            # ready as it was right after server startup; a backend without
            # this gap (Postgres/Mongo/Redis/JSON row-level drop) just no-ops.
            if rag.doc_status is not None:
                try:
                    await rag.doc_status.initialize()
                except Exception as reinit_error:
                    logger.error(
                        f"/documents/clear: failed to re-initialize doc_status "
                        f"after drop; the next /documents/scan may fail until "
                        f"a write recreates it: {reinit_error}"
                    )

            # If all storage operations failed, return error status and don't proceed with file deletion
            if storage_success_count == 0 and storage_error_count > 0:
                error_message = "All storage drop operations failed. Aborting document clearing process."
                logger.error(error_message)
                append_pipeline_history(pipeline_status, error_message)
                return ClearDocumentsResponse(status="fail", message=error_message)

            # Log file deletion start
            append_pipeline_history(
                pipeline_status, "Starting to delete files in input directory"
            )

            # Delete only files in the current directory, preserve files in subdirectories
            deleted_files_count = 0
            file_errors_count = 0

            for file_path in doc_manager.input_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_files_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
                        file_errors_count += 1

            # Log file deletion results
            if file_errors_count > 0:
                append_pipeline_history(
                    pipeline_status,
                    f"Deleted {deleted_files_count} files with {file_errors_count} errors",
                )
                errors.append(f"Failed to delete {file_errors_count} files")
            else:
                append_pipeline_history(
                    pipeline_status, f"Successfully deleted {deleted_files_count} files"
                )

            # Prepare final result message
            final_message = ""
            if errors:
                final_message = f"Cleared documents with some errors. Deleted {deleted_files_count} files."
                status = "partial_success"
            else:
                final_message = f"All documents cleared successfully. Deleted {deleted_files_count} files."
                status = "success"

            # Log final result
            append_pipeline_history(pipeline_status, final_message)

            # Return response based on results
            return ClearDocumentsResponse(status=status, message=final_message)
        except Exception as e:
            error_msg = f"Error clearing documents: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            append_pipeline_history(pipeline_status, error_msg)
            raise internal_server_error(e)
        finally:
            # Reset busy + destructive_busy after completion so the next
            # reservation / scan sees an idle pipeline. Owner-checked +
            # cancellation-resistant so a cancel here cannot leave the slot
            # wedged and cannot clobber a later holder.
            from lightrag.kg.shared_storage import with_reservation_lock

            def _clear_release(status):
                completion_msg = "Document clearing process completed"
                status.update(
                    {
                        "busy": False,
                        "destructive_busy": False,
                        "busy_owner": None,
                        "operation_record": None,
                        "latest_message": completion_msg,
                    }
                )
                append_pipeline_history(status, completion_msg)

            await with_reservation_lock(
                pipeline_status,
                pipeline_status_lock,
                owner_key="busy_owner",
                token=destructive_token,
                action=_clear_release,
            )

    @router.get(
        "/pipeline_status",
        dependencies=[Depends(combined_auth)],
        response_model=PipelineStatusResponse,
    )
    async def get_pipeline_status() -> PipelineStatusResponse:
        """
        Get the current status of the document indexing pipeline.

        This endpoint returns information about the current state of the document processing pipeline,
        including the processing status, progress information, and history messages.

        Returns:
            PipelineStatusResponse: A response object containing:
                - busy (bool): Whether the pipeline is currently busy
                - job_name (str): Current job name (e.g., indexing files/indexing texts)
                - job_start (str, optional): Job start time as ISO format string
                - docs (int): Total number of documents to be indexed
                - batchs (int): Number of batches for processing documents
                - cur_batch (int): Current processing batch
                - latest_message (str): Latest message from pipeline processing
                - history_messages (List[str], optional): List of history messages (limited to latest 1000 entries,
                  with truncation message if more than 1000 messages exist)

        Raises:
            HTTPException: If an error occurs while retrieving pipeline status (500)
        """
        try:
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
                get_all_update_flags_status,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            # Get update flags status for all namespaces
            update_status = await get_all_update_flags_status(workspace=rag.workspace)

            # Convert MutableBoolean objects to regular boolean values
            processed_update_status = {}
            for namespace, flags in update_status.items():
                processed_flags = []
                for flag in flags:
                    # Handle both multiprocess and single process cases
                    if hasattr(flag, "value"):
                        processed_flags.append(bool(flag.value))
                    else:
                        processed_flags.append(bool(flag))
                processed_update_status[namespace] = processed_flags

            async with pipeline_status_lock:
                # DictProxy.copy() is one Manager RPC; dict(proxy) may fetch
                # values individually through the mapping protocol.
                status_dict = pipeline_status.copy()

            # Sanitized fence projection BEFORE the internal fields are dropped
            # (``recovery_required`` is one of them): an operator needs a
            # read-only way to see that the workspace is fenced, its coarse cause
            # and the bounded blocker sample, without the PIDs / tokens / raw
            # operation_record the internal record carries.
            from lightrag.kg.shared_storage import (
                _INTERNAL_PIPELINE_STATUS_FIELDS,
                describe_recovery_fence,
            )

            fence_view = describe_recovery_fence(status_dict)

            # Drop internal reservation-ownership / dead-process-recovery
            # bookkeeping: these carry raw owner tokens, PIDs and per-token sets
            # that must never be exposed on the API (PipelineStatusResponse is
            # extra="allow", so unknown keys would otherwise pass through).
            for _internal_field in _INTERNAL_PIPELINE_STATUS_FIELDS:
                status_dict.pop(_internal_field, None)

            status_dict.update(fence_view)

            # Add processed update_status to the status dictionary
            status_dict["update_status"] = processed_update_status

            # Materialize history_messages to a regular list if it's a Manager
            # ListProxy, then limit to latest 1000 with a truncation banner. A
            # slice is one Manager RPC; list(proxy) walks the sequence protocol
            # with one __getitem__ RPC per element (history can hold 10k lines).
            if "history_messages" in status_dict:
                history_list = status_dict["history_messages"][:]
                total_count = len(history_list)

                if total_count > 1000:
                    # Calculate truncated message count
                    truncated_count = total_count - 1000

                    # Take only the latest 1000 messages
                    latest_messages = history_list[-1000:]

                    # Add truncation message at the beginning
                    truncation_message = (
                        f"[Truncated history messages: {truncated_count}/{total_count}]"
                    )
                    status_dict["history_messages"] = [
                        truncation_message
                    ] + latest_messages
                else:
                    # No truncation needed, return all messages
                    status_dict["history_messages"] = history_list

            # Ensure job_start is properly formatted as a string with timezone information
            if "job_start" in status_dict and status_dict["job_start"]:
                # Use format_datetime to ensure consistent formatting
                status_dict["job_start"] = format_datetime(status_dict["job_start"])

            return PipelineStatusResponse(**status_dict)
        except Exception as e:
            logger.error(f"Error getting pipeline status: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    # TODO: Deprecated, use /documents/paginated instead
    @router.get(
        "", response_model=DocsStatusesResponse, dependencies=[Depends(combined_auth)]
    )
    async def documents() -> DocsStatusesResponse:
        """
        Get the status of all documents in the system. This endpoint is deprecated; use /documents/paginated instead.
        To prevent excessive resource consumption, a maximum of 1,000 records is returned.

        This endpoint retrieves the current status of all documents, grouped by their
        processing status (PENDING, PROCESSING, PREPROCESSED, PROCESSED, FAILED). The results are
        limited to 1000 total documents with fair distribution across all statuses.

        Returns:
            DocsStatusesResponse: A response object containing a dictionary where keys are
                                DocStatus values and values are lists of DocStatusResponse
                                objects representing documents in each status category.
                                Maximum 1000 documents total will be returned.

        Raises:
            HTTPException: If an error occurs while retrieving document statuses (500).
        """
        try:
            statuses = (
                DocStatus.PENDING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
                DocStatus.PROCESSING,
                DocStatus.PREPROCESSED,
                DocStatus.PROCESSED,
                DocStatus.FAILED,
            )

            tasks = [rag.get_docs_by_status(status) for status in statuses]
            results: List[Dict[str, DocProcessingStatus]] = await asyncio.gather(*tasks)

            response = DocsStatusesResponse()
            total_documents = 0
            max_documents = 1000

            # Convert results to lists for easier processing
            status_documents = []
            for idx, result in enumerate(results):
                status = statuses[idx]
                docs_list = []
                for doc_id, doc_status in result.items():
                    docs_list.append((doc_id, doc_status))
                status_documents.append((status, docs_list))

            # Fair distribution: round-robin across statuses
            status_indices = [0] * len(
                status_documents
            )  # Track current index for each status
            current_status_idx = 0

            while total_documents < max_documents:
                # Check if we have any documents left to process
                has_remaining = False
                for status_idx, (status, docs_list) in enumerate(status_documents):
                    if status_indices[status_idx] < len(docs_list):
                        has_remaining = True
                        break

                if not has_remaining:
                    break

                # Try to get a document from the current status
                status, docs_list = status_documents[current_status_idx]
                current_index = status_indices[current_status_idx]

                if current_index < len(docs_list):
                    doc_id, doc_status = docs_list[current_index]

                    if status not in response.statuses:
                        response.statuses[status] = []

                    response.statuses[status].append(
                        DocStatusResponse(
                            id=doc_id,
                            content_summary=doc_status.content_summary,
                            content_length=doc_status.content_length,
                            status=doc_status.status,
                            created_at=format_datetime(doc_status.created_at),
                            updated_at=format_datetime(doc_status.updated_at),
                            track_id=doc_status.track_id,
                            chunks_count=doc_status.chunks_count,
                            error_msg=doc_status.error_msg,
                            metadata=doc_status.metadata,
                            file_path=normalize_file_path(doc_status.file_path),
                        )
                    )

                    status_indices[current_status_idx] += 1
                    total_documents += 1

                # Move to next status (round-robin)
                current_status_idx = (current_status_idx + 1) % len(status_documents)

            return response
        except Exception as e:
            logger.error(f"Error GET /documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    class DeleteDocByIdResponse(BaseModel):
        """Response model for single document deletion operation."""

        status: Literal["deletion_started", "busy", "not_allowed"] = Field(
            description="Status of the deletion operation"
        )
        message: str = Field(description="Message describing the operation result")
        doc_id: str = Field(description="The ID of the document to delete")

    @router.delete(
        "/delete_document",
        response_model=DeleteDocByIdResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete a document and all its associated data by its ID.",
    )
    async def delete_document(
        delete_request: DeleteDocRequest,
        managed_tasks: set = Depends(get_managed_background_tasks),
    ) -> DeleteDocByIdResponse:
        """
        Delete documents and all their associated data by their IDs using background processing.

        Deletes specific documents and all their associated data, including their status,
        text chunks, vector embeddings, and any related graph data. When requested,
        cached LLM extraction responses are removed after graph deletion/rebuild completes.
        The deletion process runs in the background to avoid blocking the client connection.

        This operation is irreversible and will interact with the pipeline status.

        **Concurrency Constraint:**
        - Atomically reserves the destructive slot (sets ``busy=True``
          and ``destructive_busy=True``) **synchronously** before
          returning ``deletion_started``, so a /scan or /upload that
          arrives before the bg task runs cannot race the delete.
          Refuses with ``status="busy"`` when ANY of these is set:
          ``pipeline_status["busy"]``, ``pipeline_status["scanning"]``,
          or ``pipeline_status["pending_enqueues"] > 0``.

        Args:
            delete_request (DeleteDocRequest): The request containing the document IDs and deletion options.
            managed_tasks: injected managed background-task set
                (see get_managed_background_tasks) — the reservation-holding work
                runs as a tracked asyncio task, not a Starlette callback

        Returns:
            DeleteDocByIdResponse: The result of the deletion operation.
                - status="deletion_started": The document deletion has been initiated in the background.
                - status="busy": Another writer (busy / scanning / pending enqueue) holds the
                  pipeline; nothing scheduled, retry after the running job finishes.

        Raises:
            HTTPException:
              - 500: If an unexpected internal error occurs during initialization.
        """
        from lightrag.kg.shared_storage import start_reserved_background_task

        doc_ids = delete_request.doc_ids

        # Owner token for the destructive slot, generated before any await so the
        # finally can release it by owner even if the acquire is cancelled.
        destructive_token = uuid4().hex

        async def _delete_work(started):
            # started.set() first (no await before it) so the endpoint's
            # start-barrier confirms takeover before returning; a body-send
            # cancellation therefore cannot strand the reservation.
            # background_delete_documents releases busy + destructive_busy
            # (owner-checked by destructive_token) in its own finally.
            started.set()
            await background_delete_documents(
                rag,
                doc_manager,
                doc_ids,
                delete_request.delete_file,
                delete_request.delete_llm_cache,
                destructive_token,
            )

        async def _delete_backstop():
            # Owner-checked + idempotent; runs only if the child never took over.
            await _release_destructive_busy(rag, destructive_token)

        handed_off = False
        try:
            # Atomically reserve the destructive slot BEFORE returning
            # ``deletion_started``.  Without this, the bg task would set
            # destructive_busy only when it later runs — leaving a
            # window where a /scan or /upload can race the delete after
            # the client has already received success.  The check
            # covers busy + scanning + pending_enqueues>0 in a single
            # critical section.
            acquired, reason = await _acquire_destructive_busy(
                rag,
                destructive_token,
                kind="delete",
                operation_record={"kind": "delete", "doc_ids": doc_ids},
            )
            if not acquired:
                return DeleteDocByIdResponse(
                    status="busy",
                    message=reason or "Cannot delete documents while pipeline is busy",
                    doc_id=", ".join(doc_ids),
                )

            # Hand the reservation to a managed background task. The start
            # barrier guarantees takeover (started.set) before we return, so a
            # cancellation while sending the response body cannot strand
            # busy/destructive_busy.
            await start_reserved_background_task(
                managed_tasks, work=_delete_work, backstop_release=_delete_backstop
            )
            # Ownership of the slot transferred to the bg task — it releases in
            # its finally. The endpoint's finally must NOT release it again.
            handed_off = True

            return DeleteDocByIdResponse(
                status="deletion_started",
                message=f"Document deletion for {len(doc_ids)} documents has been initiated. Processing will continue in background.",
                doc_id=", ".join(doc_ids),
            )

        except Exception as e:
            error_msg = f"Error initiating document deletion for {delete_request.doc_ids}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise internal_server_error(e)
        finally:
            # Release by the pre-generated token unless the managed task took
            # over. Covers a cancellation delivered while _acquire_destructive_busy
            # was exiting its lock (flags + owner already written, but ``acquired``
            # not yet assigned) or before hand-off: release is owner-checked +
            # idempotent, so it frees the slot we took and is a no-op if we never
            # acquired (or the start helper's backstop already released it).
            if not handed_off:
                await _release_destructive_busy(rag, destructive_token)

    @router.post(
        "/clear_cache",
        response_model=ClearCacheResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def clear_cache(request: ClearCacheRequest):
        """
        Clear all cache data from the LLM response cache storage.

        This endpoint clears all cached LLM responses regardless of mode.
        The request body is accepted for API compatibility but is ignored.

        Args:
            request (ClearCacheRequest): The request body (ignored for compatibility).

        Returns:
            ClearCacheResponse: A response object containing the status and message.

        Raises:
            HTTPException: If an error occurs during cache clearing (500).
        """
        try:
            # Call the aclear_cache method (no modes parameter)
            await rag.aclear_cache()

            # Prepare success message
            message = "Successfully cleared all cache"

            return ClearCacheResponse(status="success", message=message)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    @router.get(
        "/track_status/{track_id}",
        response_model=TrackStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_track_status(track_id: str) -> TrackStatusResponse:
        """
        Get the processing status of documents by tracking ID.

        This endpoint retrieves all documents associated with a specific tracking ID,
        allowing users to monitor the processing progress of their uploaded files or inserted texts.

        Args:
            track_id (str): The tracking ID returned from upload, text, or texts endpoints

        Returns:
            TrackStatusResponse: A response object containing:
                - track_id: The tracking ID
                - documents: List of documents associated with this track_id
                - total_count: Total number of documents for this track_id

        Raises:
            HTTPException: If track_id is invalid (400) or an error occurs (500).
        """
        try:
            # Validate track_id
            if not track_id or not track_id.strip():
                raise HTTPException(status_code=400, detail="Track ID cannot be empty")

            track_id = track_id.strip()

            # Get documents by track_id
            docs_by_track_id = await rag.aget_docs_by_track_id(track_id)

            # Convert to response format
            documents = []
            status_summary = {}

            for doc_id, doc_status in docs_by_track_id.items():
                documents.append(
                    DocStatusResponse(
                        id=doc_id,
                        content_summary=doc_status.content_summary,
                        content_length=doc_status.content_length,
                        status=doc_status.status,
                        created_at=format_datetime(doc_status.created_at),
                        updated_at=format_datetime(doc_status.updated_at),
                        track_id=doc_status.track_id,
                        chunks_count=doc_status.chunks_count,
                        error_msg=doc_status.error_msg,
                        metadata=doc_status.metadata,
                        file_path=normalize_file_path(doc_status.file_path),
                    )
                )

                # Build status summary
                # Handle both DocStatus enum and string cases for robust deserialization
                status_key = str(doc_status.status)
                status_summary[status_key] = status_summary.get(status_key, 0) + 1

            return TrackStatusResponse(
                track_id=track_id,
                documents=documents,
                total_count=len(documents),
                status_summary=status_summary,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting track status for {track_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    @router.post(
        "/paginated",
        response_model=PaginatedDocsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_documents_paginated(
        request: DocumentsRequest,
    ) -> PaginatedDocsResponse:
        """
        Get documents with pagination support.

        This endpoint retrieves documents with pagination, filtering, and sorting capabilities.
        It provides better performance for large document collections by loading only the
        requested page of data.

        Args:
            request (DocumentsRequest): The request body containing pagination parameters

        Returns:
            PaginatedDocsResponse: A response object containing:
                - documents: List of documents for the current page
                - pagination: Pagination information (page, total_count, etc.)
                - status_counts: Count of documents by status for all documents

        Raises:
            HTTPException: If an error occurs while retrieving documents (500).
        """
        trace_id = uuid4().hex[:8]
        request_start = time.perf_counter()
        status_filter_value = (
            request.status_filter.value if request.status_filter is not None else None
        )
        workspace = getattr(rag, "workspace", None)

        performance_timing_log(
            "[documents/paginated][%s] Request start workspace=%s status_filter=%s page=%s page_size=%s sort_field=%s sort_direction=%s",
            trace_id,
            workspace,
            status_filter_value,
            request.page,
            request.page_size,
            request.sort_field,
            request.sort_direction,
        )

        try:

            async def _timed_call(operation_name: str, operation):
                operation_start = time.perf_counter()
                performance_timing_log(
                    "[documents/paginated][%s] %s started",
                    trace_id,
                    operation_name,
                )
                try:
                    result = await operation
                except Exception:
                    elapsed = time.perf_counter() - operation_start
                    performance_timing_log(
                        "[documents/paginated][%s] %s failed after %.4fs",
                        trace_id,
                        operation_name,
                        elapsed,
                    )
                    raise

                elapsed = time.perf_counter() - operation_start
                performance_timing_log(
                    "[documents/paginated][%s] %s completed in %.4fs",
                    trace_id,
                    operation_name,
                    elapsed,
                )
                return result

            query_task_create_start = time.perf_counter()
            docs_task = asyncio.create_task(
                _timed_call(
                    "get_docs_paginated",
                    rag.doc_status.get_docs_paginated(
                        status_filter=request.status_filter,
                        status_filters=request.status_filters,
                        page=request.page,
                        page_size=request.page_size,
                        sort_field=request.sort_field,
                        sort_direction=request.sort_direction,
                    ),
                )
            )
            status_counts_task = asyncio.create_task(
                _timed_call(
                    "get_all_status_counts",
                    rag.doc_status.get_all_status_counts(),
                )
            )
            query_task_create_elapsed = time.perf_counter() - query_task_create_start
            performance_timing_log(
                "[documents/paginated][%s] Query tasks created in %.4fs",
                trace_id,
                query_task_create_elapsed,
            )

            query_await_start = time.perf_counter()
            (documents_with_ids, total_count), status_counts = await asyncio.gather(
                docs_task, status_counts_task
            )
            query_await_elapsed = time.perf_counter() - query_await_start
            performance_timing_log(
                "[documents/paginated][%s] Query tasks awaited in %.4fs",
                trace_id,
                query_await_elapsed,
            )

            # Convert documents to response format
            response_assembly_start = time.perf_counter()
            doc_responses = []
            for doc_id, doc in documents_with_ids:
                doc_responses.append(
                    DocStatusResponse(
                        id=doc_id,
                        content_summary=doc.content_summary,
                        content_length=doc.content_length,
                        status=doc.status,
                        created_at=format_datetime(doc.created_at),
                        updated_at=format_datetime(doc.updated_at),
                        track_id=doc.track_id,
                        chunks_count=doc.chunks_count,
                        error_msg=doc.error_msg,
                        metadata=doc.metadata,
                        file_path=normalize_file_path(doc.file_path),
                    )
                )

            # Calculate pagination info
            total_pages = (total_count + request.page_size - 1) // request.page_size
            has_next = request.page < total_pages
            has_prev = request.page > 1

            pagination = PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                total_count=total_count,
                total_pages=total_pages,
                has_next=has_next,
                has_prev=has_prev,
            )
            response = PaginatedDocsResponse(
                documents=doc_responses,
                pagination=pagination,
                status_counts=status_counts,
            )
            response_assembly_elapsed = time.perf_counter() - response_assembly_start
            total_elapsed = time.perf_counter() - request_start

            performance_timing_log(
                "[documents/paginated][%s] Response assembled in %.4fs",
                trace_id,
                response_assembly_elapsed,
            )
            performance_timing_log(
                "[documents/paginated][%s] Request completed in %.4fs returned_rows=%s total_count=%s status_count_keys=%s",
                trace_id,
                total_elapsed,
                len(doc_responses),
                total_count,
                sorted(status_counts.keys()),
            )

            return response

        except Exception as e:
            total_elapsed = time.perf_counter() - request_start
            performance_timing_log(
                "[documents/paginated][%s] Request failed after %.4fs",
                trace_id,
                total_elapsed,
            )
            logger.error(f"Error getting paginated documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    @router.get(
        "/status_counts",
        response_model=StatusCountsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_document_status_counts() -> StatusCountsResponse:
        """
        Get counts of documents by status.

        This endpoint retrieves the count of documents in each processing status
        (PENDING, PROCESSING, PROCESSED, FAILED) for all documents in the system.

        Returns:
            StatusCountsResponse: A response object containing status counts

        Raises:
            HTTPException: If an error occurs while retrieving status counts (500).
        """
        try:
            status_counts = await rag.doc_status.get_all_status_counts()
            return StatusCountsResponse(status_counts=status_counts)

        except Exception as e:
            logger.error(f"Error getting document status counts: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    @router.get(
        "/supported_file_types",
        response_model=SupportedFileTypesResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_supported_file_types(
        response: Response,
    ) -> SupportedFileTypesResponse:
        """
        Get the upload allowlist and the parser capability matrix.

        ``supported_extensions`` is the live allowlist for bare (unhinted)
        filenames — derived from the parser registry plus the current
        ``LIGHTRAG_PARSER`` routing rules, the same source the upload
        endpoint's 400 detail quotes. ``engines`` maps every usable engine
        (user-selectable, endpoint configured) to the suffixes it can parse,
        so the WebUI can pre-validate hinted filenames like
        ``img.[mineru].png`` without uploading the file first.

        Derived from process-level configuration today; when per-workspace
        parser rules land, this endpoint resolves the ``LIGHTRAG-WORKSPACE``
        header internally — the response contract stays unchanged (hence the
        ``Vary`` header, declared ahead of time so caches never reuse a
        response across workspaces).
        """
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "LIGHTRAG-WORKSPACE"
        return SupportedFileTypesResponse(
            supported_extensions=list(doc_manager.supported_extensions),
            engines=doc_manager.engine_capabilities,
        )

    @router.post(
        "/reprocess_failed",
        response_model=ReprocessResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def reprocess_failed_documents(
        managed_tasks: set = Depends(get_managed_background_tasks),
    ):
        """
        Reprocess existing failed, pending, or interrupted document records
        without scanning the input directory for new files.

        Pure storage-driven recovery: this endpoint publishes a sticky manual
        retry request into the workspace ingress and drives the processing
        pipeline. FAILED documents re-enter the pipeline ONLY through such an
        explicit request — each request grants at most ONE retry attempt per
        FAILED document; a document that fails again stays FAILED until the
        next explicit request. PENDING and interrupted documents
        (PROCESSING/PARSING/ANALYZING left by a crash) are picked up as well.

        This endpoint does NOT discover new files, archive anything, or run
        scan classification; a FAILED document with an unfinished
        custom-chunk journal is skipped and still requires ``/documents/scan``
        to roll back.

        If the pipeline is busy, the request is NOT lost: it stays queued in
        the workspace ingress and executes when the current run reaches its
        next quiescence point. The processing happens in the background and
        can be monitored via the pipeline status; reprocessed documents
        retain their original track_id from initial upload.

        Returns:
            ReprocessResponse: Response with status and message.
                track_id is always empty string because reprocessed documents
                retain their original track_id from initial upload.

        Raises:
            HTTPException: 503 when the workspace is fenced (recovery
                required, or a clear/delete is dropping storages); 500 on
                unexpected errors while initiating reprocessing.
        """
        from lightrag.exceptions import PipelineNotInitializedError
        from lightrag.kg.shared_storage import (
            ManualIntentRefused,
            commit_manual_retry_request,
            get_namespace_data,
            get_namespace_lock,
            get_pipeline_ingress,
            start_committed_background_task,
            start_reserved_background_task,
        )

        request_id = uuid4().hex

        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
        except PipelineNotInitializedError:
            pipeline_status = None

        if pipeline_status is None:
            # Workspace pipeline_status not bootstrapped (mocked test rigs):
            # there is no fence and no ingress to publish into — fall back to
            # a plain managed drive.
            async def _legacy_work(started):
                started.set()
                await rag.apipeline_process_enqueue_documents()

            async def _noop_backstop():
                return None

            try:
                await start_reserved_background_task(
                    managed_tasks, work=_legacy_work, backstop_release=_noop_backstop
                )
                return ReprocessResponse(
                    status="reprocessing_started",
                    message="Reprocessing of failed documents has been initiated "
                    "in background. Documents retain their original track_id.",
                )
            except Exception as e:
                logger.error(f"Error initiating reprocessing: {str(e)}")
                logger.error(traceback.format_exc())
                raise internal_server_error(e)

        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )
        # Resolve the ingress BEFORE any critical section (never lazily while
        # the pipeline status lock is held inside the commit).
        ingress = await get_pipeline_ingress(rag.workspace)

        # Two-state startup: the fence recheck and the sticky publish happen
        # in ONE pipeline_status_lock critical section inside the child
        # (commit), so a destructive clear cannot slip its reservation in
        # between; once committed, an endpoint cancellation no longer cancels
        # the child — the published intent has an owner driving it. Reprocess
        # holds no reservation of its own (apipeline_process_enqueue_documents
        # acquires/releases busy itself), so the backstop is a no-op.
        async def _commit(state):
            return await commit_manual_retry_request(
                pipeline_status,
                pipeline_status_lock,
                ingress,
                request_id,
                state,
            )

        async def _work():
            await rag.apipeline_process_enqueue_documents()

        async def _noop_backstop():
            return None

        try:
            await start_committed_background_task(
                managed_tasks,
                commit=_commit,
                work=_work,
                backstop_release=_noop_backstop,
            )
            logger.info(
                f"Reprocessing of failed documents initiated (request {request_id})"
            )
            return ReprocessResponse(
                status="reprocessing_started",
                message="Reprocessing of failed documents has been initiated in "
                "background (one retry attempt per failed document). If the "
                "pipeline is busy the request executes at its next quiescence "
                "point. Documents retain their original track_id.",
            )
        except ManualIntentRefused as refusal:
            # 429 only for a full manual channel (retry later works); every other
            # refusal — fence held, id already finalized — is 503 (LR2 §10.1).
            raise HTTPException(
                status_code=429 if refusal.capacity_exceeded else 503,
                detail=str(refusal),
                headers=(
                    {"Retry-After": str(_ADMISSION_RETRY_AFTER_SECONDS)}
                    if refusal.capacity_exceeded
                    else None
                ),
            )
        except Exception as e:
            logger.error(f"Error initiating reprocessing of failed documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    @router.post(
        "/recovery/force_reset",
        response_model=ForceResetRecoveryResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def force_reset_recovery(
        request: ForceResetRecoveryRequest,
    ) -> ForceResetRecoveryResponse:
        """Force-clear a ``recovery_required`` fence (UNSAFE, manual).

        The fence is raised when a worker is killed mid custom_chunks / delete /
        clear (Linux multi-worker), which may have left storage partially
        committed, or when a manual retry's drain cannot reach idle — so every
        mutation is refused until the workspace is recovered. This endpoint does
        NOT repair anything; it only drops the fence (and any lingering
        reservation flags), re-opening a possibly-inconsistent workspace. Requires
        ``confirm=true``. A true idempotent replay of the interrupted operation is
        a separate concern (core atomicity / #3400).

        It ALSO cancels the workspace's queued manual retry requests, and that is
        load-bearing rather than housekeeping: a sticky un-ACKed request makes
        ``/documents/scan`` refuse its reservation (a scan runs its own exclusive
        FAILED reset and may not jump the manual FIFO — LR2 §8.1). For the
        blocked-drain fence, ``/documents/scan`` is the remedy, so clearing the
        fence alone would leave the recovery path just as blocked as before. The
        failed documents are untouched by the cancellation — the scan's own FAILED
        reset covers them, or ``/documents/reprocess_failed`` can be re-issued —
        and the cancelled ids are retired as terminal, so a delayed replay of one
        cannot silently re-queue.

        Because both halves are required, this endpoint is **fail-closed and
        atomic**: the ingress is resolved first and a failure there returns 503
        with the fence untouched, then the cancellation and the fence drop happen
        in ONE ``pipeline_status_lock`` critical section with the cancellation
        first. A cancellation failure also returns 503 and keeps the fence. The
        only partial state this admits is "intents cancelled, fence still up",
        which a retry completes; the opposite order would admit "fence down,
        intents still queued", where ``/scan`` stays refused and nothing says so.
        """
        from lightrag.exceptions import PipelineNotInitializedError
        from lightrag.kg.shared_storage import (
            MANUAL_PHASE_IDLE,
            get_namespace_data,
            get_namespace_lock,
            get_pipeline_ingress,
        )

        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Set confirm=true to force-reset. The workspace may be in a "
                    "partially-committed state after a worker died mid "
                    "custom_chunks/delete/clear; verify or repair it first."
                ),
            )
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
        except PipelineNotInitializedError:
            return ForceResetRecoveryResponse(
                status="no_recovery_required",
                message="Pipeline status is not initialized; nothing to reset.",
            )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )
        # Resolved BEFORE the critical section (a Manager RPC handle must never be
        # looked up lazily while pipeline_status_lock is held) and FAIL CLOSED if
        # it cannot be reached. Cancelling the queued intents is one HALF of this
        # recovery, so a force-reset that cannot do it must not drop the fence and
        # report success: that leaves the sticky request in place, keeps
        # /documents/scan refused, and recreates the silent dead end this endpoint
        # exists to close. Keeping the fence is safely retryable; a false "reset"
        # is not.
        try:
            ingress = await get_pipeline_ingress(rag.workspace)
        except Exception as ingress_error:
            logger.error(
                "Force-reset refused: cannot reach the ingress mailbox to cancel "
                f"the queued manual retries ({ingress_error}); the fence is kept "
                "so the recovery can be retried rather than reported as done."
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cannot reach the ingress mailbox to cancel queued manual "
                    "retries; the recovery fence is unchanged. Retry shortly."
                ),
            )

        cancelled = 0
        async with pipeline_status_lock:
            if not pipeline_status.get("recovery_required"):
                return ForceResetRecoveryResponse(
                    status="no_recovery_required",
                    message="No recovery_required fence is set.",
                )

            # Cancel the queued intents FIRST, and inside the SAME critical
            # section that drops the fence. Two reasons, and the order is the
            # whole point:
            #
            # * failure atomicity — if the cancellation succeeds but the status
            #   update below fails, the workspace is left "intents cancelled,
            #   fence still up", which is safely retryable. The reverse order
            #   leaves "fence down, intents still queued": /scan stays refused
            #   and nothing says so.
            # * no window — dropping the fence and then cancelling outside the
            #   lock let a new processing run acquire ``busy`` in between, peek
            #   the still-sticky request and claim it in ``_begin_manual_drain``.
            #   That run holds the request id, so it could go on to run
            #   FAILED → PENDING for a request the operator had just cancelled.
            #   Both mutations under one lock hold close it: when the lock is
            #   released the fence is gone AND the mailbox is empty.
            #
            # The mailbox call is a single already-resolved RPC under the lock —
            # the same shape as the scan endpoint's publish and the reservation
            # helpers' manual-pending peek.
            try:
                cancelled = int(ingress.cancel_manual_retries() or 0)
            except Exception as cancel_error:
                logger.error(
                    "Force-reset refused: the recovery fence is kept because the "
                    "queued manual retries could not be cancelled "
                    f"({cancel_error}); clearing the fence alone would leave "
                    "/documents/scan refused."
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Could not cancel the queued manual retries; the recovery "
                        "fence is unchanged. Retry shortly."
                    ),
                )

            # Drop the fence and any lingering reservation state in one atomic
            # update. This is deliberately owner-agnostic — it is a manual
            # override, not a normal owner-checked release. The manual freeze goes
            # with it: it is held BY a run that this reset is abandoning, so
            # leaving it would wedge every upload behind an owner that is gone.
            pipeline_status.update(
                {
                    "recovery_required": None,
                    "operation_record": None,
                    "busy": False,
                    "destructive_busy": False,
                    "busy_owner": None,
                    "scanning": False,
                    "scanning_exclusive": False,
                    "scanning_owner": None,
                    "manual_freeze_requested": False,
                    "manual_freeze_started_at": None,
                    "manual_resetting": False,
                    "manual_phase": MANUAL_PHASE_IDLE,
                    "manual_owner": None,
                }
            )

        logger.warning(
            "recovery_required fence force-reset (unsafe manual override) for "
            f"workspace {rag.workspace}; cancelled {cancelled} queued manual "
            "retry request(s)"
        )
        return ForceResetRecoveryResponse(
            status="reset",
            cancelled_manual_retries=cancelled,
            message=(
                "recovery_required fence cleared"
                + (
                    f" and {cancelled} queued manual retry request(s) cancelled"
                    if cancelled
                    else ""
                )
                + ". The workspace may still be "
                "partially committed — reprocess/verify affected documents."
            ),
        )

    @router.post(
        "/cancel_pipeline",
        response_model=CancelPipelineResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def cancel_pipeline():
        """
        Request cancellation of the currently running pipeline.

        This endpoint sets a cancellation flag in the pipeline status. The pipeline will:
        1. Check this flag at key processing points
        2. Stop processing new documents
        3. Cancel all running document processing tasks
        4. Mark all PROCESSING documents as FAILED with reason "User cancelled"

        The cancellation is graceful and ensures data consistency. Documents that have
        completed processing will remain in PROCESSED status.

        Returns:
            CancelPipelineResponse: Response with status and message
                - status="cancellation_requested": Cancellation flag has been set
                - status="not_busy": Pipeline is not currently running

        Raises:
            HTTPException: If an error occurs while setting cancellation flag (500).
        """
        try:
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            async with pipeline_status_lock:
                if not pipeline_status.get("busy", False):
                    return CancelPipelineResponse(
                        status="not_busy",
                        message="Pipeline is not currently running. No cancellation needed.",
                    )

                # Set cancellation flag
                cancel_msg = "Pipeline cancellation requested by user"
                logger.info(cancel_msg)
                pipeline_status.update(
                    {
                        "cancellation_requested": True,
                        "latest_message": cancel_msg,
                    }
                )
                append_pipeline_history(pipeline_status, cancel_msg)

            return CancelPipelineResponse(
                status="cancellation_requested",
                message="Pipeline cancellation has been requested. Documents will be marked as FAILED.",
            )

        except Exception as e:
            logger.error(f"Error requesting pipeline cancellation: {str(e)}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    return router
