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
import inspect
import json

import mimetypes
import threading
import time
import traceback
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorPosition,
    DocProcessingStatus,
    DocStatus,
    SourceConflict,
    SourceUnique,
)
from lightrag.constants import (
    ENQUEUE_SERIALIZE_LOCK_NAMESPACE,
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSED_DIR_NAME,
)
from lightrag.exceptions import (
    MultimodalAnalysisError,
    PipelineCancelledException,
    PipelineRecoveryRequiredError,
    PipelineReservationConflictError,
    IndexFlushError,
)
from lightrag.kg.shared_storage import (
    MANUAL_PHASE_DRAIN_TO_IDLE,
    MANUAL_PHASE_EXCLUSIVE_RESET,
    MANUAL_PHASE_IDLE,
    PipelineReservationConflict,
    _reservation_owner_token,
    acquire_processing_reservation,
    append_pipeline_history,
    check_pipeline_status_mutation,
    fence_workspace_for_recovery,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    make_manual_owner_record,
    reap_dead_reservations_locked,
    run_to_completion,
    with_reservation_lock,
)
from lightrag import pipeline_metrics
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.operate import merge_nodes_and_edges
from lightrag.parser.base import ParseContext
from lightrag.parser.llm_bridge import LLMBridgePipelineCancelled
from lightrag.parser.registry import (
    get_parser,
    parser_specs_snapshot,
    supported_parser_engines,
    suffix_capabilities,
)
from lightrag.parser.routing import (
    parser_suffix,
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
    get_env_value,
    get_llm_cache_identity,
    handle_cache,
    is_truncated_response,
    logger,
    repair_vlm_json_escape_damage_nested,
    sanitize_text_for_encoding,
    save_to_cache,
    serialize_llm_cache_identity,
    strip_control_characters,
    tolerant_load_json_dict,
)
from lightrag.utils_pipeline import (
    # Re-exported through the pipeline namespace (not used by this module
    # directly): the parser layer resolves these as ``lightrag.pipeline.<name>``
    # and the parser CLI / base archive path patch them there.
    archive_docx_source_after_full_docs_sync,  # noqa: F401
    parsed_artifact_dir_for,  # noqa: F401
    archive_source_after_full_docs_sync,
    build_chunks_dict_from_chunking_result,
    chunk_fields_from_status_doc,
    compute_text_content_hash,
    doc_status_custom_chunk_patch,
    doc_status_field,
    doc_status_parse_failure_fields,
    doc_status_transition_metadata,
    get_duplicate_doc_by_content_hash,
    get_existing_doc_by_content_hash,
    has_known_document_source,
    input_dir_path,
    normalize_document_file_path,
    doc_status_metadata_has_attempt_fields,
    doc_status_reset_metadata,
    read_source_file_basename,
    resolve_existing_doc_source,
    resolve_doc_file_path,
    resolve_doc_status_parse_engine,
    source_candidate_set_lock,
    strip_lightrag_doc_prefix,
)


# Document statuses the pipeline resumes AUTOMATICALLY — the initial snapshot
# of every run and every quiescence refetch, AND the only statuses any
# ``_next_scheduling_page`` sweep ever carries.  PROCESSING/PARSING/ANALYZING
# are dead-process orphans (an interrupted run left them mid-flight) and must
# self-heal; FAILED is deliberately absent: a document that genuinely failed is
# retried only through an explicit manual request, whose EXCLUSIVE_RESET phase
# pages FAILED→PENDING via ``_next_failed_page`` with no worker running (LR2
# §7.3) — never as a side effect of an unrelated upload or rescan.
_AUTO_RESUME_DOC_STATUSES = (
    DocStatus.PENDING,
    DocStatus.PROCESSING,
    DocStatus.PARSING,
    DocStatus.ANALYZING,
)

# Multiprocess feeder poll interval: the Manager-backed ingress has no async
# ``get_document``, so the feeder blocks a worker thread in
# ``wait_for_documents(timeout)`` and re-checks between polls. Bounded by the
# mailbox's own MAX_MAILBOX_WAIT_SECONDS clamp; the single-process feeder is
# fully event-driven (``await get_document()``) and never polls.
_FEEDER_POLL_SECONDS = 0.5

# Max documents the feeder drains (and admits) per iteration before returning
# to its top-of-loop control check. Bounds the destructive-drain re-publish
# burst on a teardown; cancellation and a pending manual request are ALSO
# re-checked before every single admission, so the per-signal yield latency is
# one admission, not a whole drain.
_FEEDER_DRAIN_LIMIT = 256

# Manual DRAIN_TO_IDLE poll: how long to sleep between re-checks while waiting
# for pre-freeze in-flight enqueue reservations (pending_enqueues > 0) to
# finish (LR2 §7.2 step 5). Bounded by the background enqueue's own duration —
# the freeze admits no NEW reservations, so the count only decreases.
_MANUAL_DRAIN_POLL_SECONDS = 0.2

# How many ``_MANUAL_DRAIN_POLL_SECONDS`` re-checks a SCAN-driven reset gives the
# in-flight enqueue count before abandoning the reset (LR2 §8.1). Unlike the
# manual endpoint's unbounded drain wait, a scan was only granted its
# reservation while ``pending_enqueues == 0``, so a non-zero count here means a
# reservation slipped into the window before the freeze went up: wait it out
# briefly, then leave the request to the standard drain path rather than blocking
# file discovery behind it.
_SCAN_DRAIN_CONFIRM_ATTEMPTS = 25

# How many consecutive DRAIN_TO_IDLE confirmation rounds may report the SAME
# blocking active rows before the drain is declared stalled and the workspace is
# fenced (LR2 §7.2 "DRAIN_TO_IDLE 的前进性" / §13.2 case 16).
#
# One round is not evidence: the confirmation sweep legitimately returns rows the
# run then processes. But a round that changes nothing produces an identical id
# set, and re-sweeping identical rows can only produce the same result again — so
# a small count separates "still draining" from "cannot drain", with no wall-clock
# timeout to tune. Every routed page leaves the AUTO set (terminal or FAILED), and
# a shrinking/changing set resets the counter, so a large-but-finite backlog is
# never mistaken for a stall.
_MANUAL_DRAIN_STALL_ROUNDS = 3

# Upper bound on the blocking doc ids reported with the stall fence. The point is
# to name the offending rows, not to serialise a backlog into an error message.
_MANUAL_DRAIN_STALL_SAMPLE = 8

# Same bound for the drain's blocker report (active rows the drain cannot advance
# at all — see ``_confirm_auto_drained``). Kept separate from the stall sample so
# either report can be widened without silently widening the other.
_MANUAL_DRAIN_BLOCKER_SAMPLE = 8


class _ManualDrainProgress:
    """Forward-progress tracker for one run's DRAIN_TO_IDLE (LR2 §7.2).

    Run-scoped, not shared state: it only has to detect a spin WITHIN one
    processing run, because a run that exits releases the freeze and the next one
    re-drains from scratch.
    """

    __slots__ = ("_blocking_ids", "_repeats")

    def __init__(self) -> None:
        self._blocking_ids: frozenset[str] | None = None
        self._repeats = 0

    def observe(self, doc_ids) -> bool:
        """Record one "the drain is not idle yet" observation.

        Returns False once the SAME rows have blocked the drain for
        :data:`_MANUAL_DRAIN_STALL_ROUNDS` consecutive rounds — the caller must
        then fence rather than sweep again.
        """
        ids = frozenset(doc_ids)
        if ids != self._blocking_ids:
            self._blocking_ids = ids
            self._repeats = 1
            return True
        self._repeats += 1
        return self._repeats < _MANUAL_DRAIN_STALL_ROUNDS


class PipelineNextStep(Enum):
    """Outcome of the atomic quiescence decision (see
    :meth:`_PipelineMixin._decide_pipeline_next_step`)."""

    RELEASED = "released"
    CONTINUE_AUTO = "continue_auto"
    # A sticky manual retry is pending and the pipeline is IDLE: BEGIN the
    # manual protocol (LR2 §6.1/§7.2) — freeze new ingress, then drain the AUTO
    # backlog to idle.  No FAILED is swept here; FAILED is reset only in the
    # exclusive phase below.  The request stays sticky (ACKed after the reset).
    CONTINUE_MANUAL = "continue_manual"
    # Manual drain reached AUTO-empty but pre-freeze in-flight enqueue
    # reservations (pending_enqueues > 0) have not finished.  Bounded-wait for
    # them so their PENDING docs join this drain cohort before the reset
    # (LR2 §7.2 step 5 / §7.3 pending_enqueues=0 precondition).
    CONTINUE_DRAIN_WAIT = "continue_drain_wait"
    # Manual drain reached strict idle (no AUTO, no in-flight, no mailbox):
    # enter EXCLUSIVE_RESET — page FAILED→PENDING with NO worker running, then
    # ACK the request and resume normal processing (LR2 §7.3).
    BEGIN_EXCLUSIVE_RESET = "begin_exclusive_reset"
    # Document messages still resident in the mailbox at quiescence (e.g. an
    # enqueue that landed after the batch's feeder stopped, or a stale
    # notification for a doc that has since reached a terminal state).  The
    # refetch drains a bounded chunk and lets the strict scan decide: live
    # docs come back as the next batch, provably-stale messages are dropped.
    CONTINUE_DOCUMENT = "continue_document"
    # The in-progress bounded AUTO sweep has more pages (its keyset cursor is
    # not yet CURSOR_END).  No wake-up signal is consumed: the run simply
    # fetches the next page of the SAME sweep before honouring quiescence
    # signals, so a huge PENDING backlog drains page-by-page while cancellation
    # and manual retry still preempt at every page boundary (LR2 Phase 2 §6.3).
    CONTINUE_SWEEP_PAGE = "continue_sweep_page"
    # Cancellation pending: the run must stop WITHOUT consuming any wake-up
    # signal — the ingress is fully retained for the next explicit trigger.
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PipelineNextDecision:
    step: PipelineNextStep
    # Set only for CONTINUE_MANUAL: the sticky request this continuation
    # serves; ACKed after the FAILED→PENDING reset persists.
    manual_request_id: str | None = None
    # Set only for CANCELLED: True when the cancellation is an internal-error
    # abort, whose exit path surfaces the actionable halt message.
    internal_error: bool = False


def _vlm_image_budget_limits() -> tuple[int, int]:
    """Resolve VLM image byte/pixel floors from env (empty → documented defaults)."""
    from lightrag.constants import DEFAULT_MM_IMAGE_MIN_PIXEL

    max_image_bytes = max(
        256 * 1024,
        get_env_value("VLM_MAX_IMAGE_BYTES", 5 * 1024 * 1024, int),
    )
    min_image_pixel = max(
        1,
        get_env_value("VLM_MIN_IMAGE_PIXEL", DEFAULT_MM_IMAGE_MIN_PIXEL, int),
    )
    return max_image_bytes, min_image_pixel


@lru_cache(maxsize=64)
def _warn_content_budget_structurally_starved(
    *,
    max_extract: int,
    leading_cap: int,
    trailing_cap: int,
    frame_reserve: int,
    content_min: int,
) -> None:
    """Log once when the configured caps alone starve multimodal content.

    ``lru_cache`` keyed on the config tuple makes this emit a single WARNING
    per unique configuration per process — an operator heads-up so gross
    misconfiguration surfaces proactively instead of as a burst of per-item
    :class:`MultimodalAnalysisError` failures from ``_analyze_text_modality``.

    Conservative by design: ``frame_reserve`` counts only the fixed safety
    buffer, not the (item-specific, typically much larger) template frame, so
    ``static_room`` *overestimates* the room left for content.  It therefore
    warns only in egregious cases where the SURROUNDING caps by themselves
    already leave less than ``content_min``; the per-item budget floor in
    ``_analyze_text_modality`` is the true enforcement.
    """
    static_room = max_extract - leading_cap - trailing_cap - frame_reserve
    if static_room >= content_min:
        return
    logger.warning(
        "[analyze_multimodal] MAX_EXTRACT_INPUT_TOKENS=%d leaves only ~%d "
        "tokens for multimodal content after SURROUNDING caps (leading=%d, "
        "trailing=%d) and a %d-token reserve — below the content floor of %d "
        "(MM_EXTRACT_CONTENT_MIN_TOKENS). Items with non-trivial surrounding/"
        "captions will fail analysis. Raise MAX_EXTRACT_INPUT_TOKENS or lower "
        "SURROUNDING_LEADING_MAX_TOKENS / SURROUNDING_TRAILING_MAX_TOKENS.",
        max_extract,
        static_room,
        leading_cap,
        trailing_cap,
        frame_reserve,
        content_min,
    )


def _call_source_file_resolver(
    owner: Any,
    file_path: str,
    *,
    source_file: str | None = None,
    parser_engine: str | None = None,
) -> str:
    """Call parser source resolver while tolerating legacy test doubles."""
    resolver = owner._resolve_source_file_for_parser
    params = inspect.signature(resolver).parameters
    supports_context = "source_file" in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if supports_context:
        return resolver(
            file_path,
            source_file=source_file,
            parser_engine=parser_engine,
        )
    return resolver(source_file or file_path)


def _rearm_auto_rescan(ingress: Any) -> None:
    """Set the auto-rescan dirty flag and count the re-arm (LR2 Phase 6 item 3).

    Every re-arm costs a later strict sweep, so the rate explains sweep load that
    no upload accounts for. Counted here rather than inside the ingress because in
    multiprocess mode the mailbox lives in the Manager process, where a counter
    would be invisible to the worker answering /health.
    """
    ingress.request_auto_rescan()
    pipeline_metrics.increment(pipeline_metrics.AUTO_RESCAN_REARMS)


# Backward-compatible source-file reader.  Implementation lives in
# utils_pipeline so reset/normalisation helpers there can reuse it without a
# reverse import into this module; kept as a module-level alias for the
# existing call sites below.
_read_source_file = read_source_file_basename


# Map ``process_options.chunking`` selector → ``extraction_meta.chunk_method``
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
    "drop_references": "drop_rf",
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
    # Parse queues are dynamic: one per ParserSpec.queue_group (always at
    # least "native"). ``parser_specs`` is the batch snapshot threaded through
    # routing + the parse workers so a mid-batch register_parser cannot change
    # the engine set for this run.
    parse_queues: dict[str, asyncio.Queue]
    parser_specs: dict
    q_analyze: asyncio.Queue
    q_process: asyncio.Queue
    # Process-local bridge for the shared pipeline cancellation status. Native
    # parser hooks run in worker threads, so they cannot await the async status
    # lock directly while waiting on an LLM response.
    pipeline_cancel_event: threading.Event | None = None
    processed_count: int = 0
    # Phase 2 in-batch feeder bookkeeping (all read/written ONLY under
    # ``pipeline_status_lock``):
    #  * ``inflight_doc_ids`` — every doc_id currently routed into or moving
    #    through the queues. The initial batch is pre-registered before the
    #    feeder starts; the feeder adds an id right after its parse-queue put
    #    succeeds.
    #  * ``routing_doc_ids`` — ids being admitted right now (added before the
    #    ``put``, moved to inflight after it with no ``await`` between the two
    #    mutations) so the feeder's dedup sees an id the instant admission
    #    begins — there is never a window where an admitting id is in neither
    #    set.
    inflight_doc_ids: set = field(default_factory=set)
    routing_doc_ids: set = field(default_factory=set)
    # Reservation token of the run that owns ``busy`` for this batch (LR2 §7.7
    # items 3/4/7). Every worker status write verifies it is still the current
    # ``busy_owner`` before touching doc_status, so a run whose ownership was
    # reclaimed (dead-process reaper, Manager generation change) discards its
    # late results instead of writing a final status over the new owner's work.
    # ``None`` disables the check and exists only for call paths that construct
    # a context outside a reservation (tests).
    run_owner_token: str | None = None


class _PipelineMixin:
    """Mixin providing document ingestion pipeline methods for LightRAG.

    Designed to be combined as a base of LightRAG only.  Relies on
    LightRAG-provided attributes (``self.full_docs``, ``self.doc_status``,
    ``self.tokenizer``, ``self.parser_*``, ``self.workspace`` ...) and on the
    shared methods ``self._insert_done`` / ``self._process_extract_entities``
    which remain in the main class and are resolved through MRO.
    """

    # ============================================================
    # Admission control (LR2 §9)
    # ============================================================

    async def _reserve_ingress_slot(
        self,
        pipeline_status: dict[str, Any],
        pipeline_status_lock,
        *,
        admission_token: str | None,
        requested: int,
        reject_when,
    ) -> str | None:
        """Register this enqueue in ``pending_enqueue_tokens`` and charge
        ``requested`` new documents against the admission capacity (LR2 §9.2).

        Called inside ``enqueue_serialize_lock``, after dedup and before the
        FIRST storage write of the critical section — the duplicate ``dup-*``
        FAILED records included. That position matters three times over:

        * the strict active count it takes cannot be raced by a sibling enqueue
          (which is waiting on that same lock);
        * ``requested`` is the real deduped count, not the coarse pre-body
          estimate an endpoint had to reserve with;
        * **the reservation is what makes the write side mutually exclusive with
          a manual retry.** ``manual_freeze_requested`` refuses a NEW
          reservation here, and DRAIN_TO_IDLE waits for ``pending_enqueues`` to
          reach zero (LR2 §6.1/§7.2), so the exclusive FAILED→PENDING reset
          cannot begin between this check and the last write below. The fence
          check at the top of :meth:`apipeline_enqueue_documents` cannot do
          that job: it takes no reservation, so it leaves the whole validate →
          dedup → write span invisible to the drain.

        ``reject_when`` is passed unconditionally and
        :func:`acquire_enqueue_reservation` applies it only when the token is
        NOT already registered — i.e. only when this call mints a reservation
        for a reservation-less caller (SDK ``ainsert`` / direct
        ``apipeline_enqueue_documents``). When the caller already holds one (an
        endpoint reserved it, fences and all, before parsing the request body)
        this is a RE-WEIGHT, and a request admitted before a freeze appeared is
        deliberately allowed to finish — the drain is already waiting for it.
        Deciding that inside the reservation, from the same snapshot, is what
        keeps the exemption from being self-attested: passing a token string is
        not enough to skip a fence. A token's own previous weight is excluded
        from the capacity sum, so re-weighting never counts a request against
        itself.

        Returns the token this function owns and the caller must release (only
        for a minted one), or ``None`` when the reservation belongs to the
        caller.

        Raises:
            PipelineReservationConflictError: a mutual-exclusion fence or the
                recovery fence refused this reservation. Nothing has been written
                yet. The refusal REASON survives as data (LR2 §9.1): a caller
                reads ``.conflict`` / ``.recovery_required`` to tell a fenced
                workspace (→ 503) from a bounded manual-freeze / scan /
                destructive window it should retry (→ 409), rather than parsing
                the message. Subclasses ``RuntimeError``, so a caller written
                against the older coarse contract still catches it.
            PipelineBackpressureError: no capacity (→ 429).
            StorageCapabilityError / StorageControlPlaneError: the strict count
                failed. Deliberately propagated rather than treated as "room
                available" (→ 503): admission fails closed.
        """
        from lightrag.kg.shared_storage import acquire_enqueue_reservation
        from lightrag.utils_pipeline import count_active_documents

        capacity = self.max_pending_documents
        # Only the capacity math needs the count, and it is a strict full scan:
        # taking it with admission disabled would tax every enqueue for nothing.
        active_count = (
            await count_active_documents(self.doc_status) if capacity > 0 else None
        )
        token = admission_token or f"admission-{uuid.uuid4().hex}"
        result = await acquire_enqueue_reservation(
            pipeline_status,
            pipeline_status_lock,
            token=token,
            reject_when=tuple(reject_when),
            weight=requested,
            capacity=capacity,
            active_count=active_count,
        )
        if not result.acquired:
            raise PipelineReservationConflictError(
                result.message or "pipeline reservation is unavailable",
                conflict=result.conflict,
                fence=result.fence,
            )
        return None if admission_token else token

    async def _release_admission_reservation(self, token: str) -> None:
        """Drop a reservation minted by :meth:`_reserve_ingress_slot`."""
        from lightrag.kg.shared_storage import release_token_set_reservation

        await release_token_set_reservation(
            self.workspace,
            tokens_key="pending_enqueue_tokens",
            token=token,
        )

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
        parse_engine: str | list[str] | None = None,
        process_options: str | list[str] | None = None,
        chunk_options: dict | list[dict] | None = None,
        admission_token: str | None = None,
        from_scan: bool = False,
    ) -> str:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        2. Generate document initial status
        3. Filter out already processed documents
        4. Enqueue document in status

        Args:
            input: Single document string or list of document strings (can be empty when docs_format is pending_parse)
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated (from content or file_path).
                **Providing ``ids`` marks the SDK raw direct-insert path**
                (:meth:`LightRAG.ainsert`) and takes precedence over
                ``docs_format``: the documents are always enqueued as RAW
                — sanitized verbatim content, no parse-worker deferral —
                by design, not as an oversight. ``pending_parse`` is the
                server upload path, which never passes ``ids``.
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status
            docs_format: "raw" (default) or "pending_parse"; "pending_parse" defers
                extraction to the parse worker (content may be empty and
                content-dedup happens after parsing). Ignored when ``ids``
                is provided (see ``ids`` above).
            parse_engine: file extraction engine already used or target engine for pending_parse
            process_options: per-document processing options string (i/t/e/!/F/R/V/P);
                accepted as a single string broadcast to every input or as a list
                aligned with ``input``. Stored verbatim on ``full_docs`` and
                mirrored to ``doc_status.metadata['process_options']``.
            chunk_options: per-document chunker parameter snapshot.
                Accepted as ``dict`` (broadcast to every input) or
                ``list[dict]`` (aligned with ``input``).  When ``None``,
                each doc's snapshot is built via
                :func:`lightrag.parser.routing.resolve_chunk_options`
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
            admission_token: the pending-enqueue reservation the caller already
                holds (endpoints reserve one before reading the request body).
                With ``MAX_PENDING_DOCUMENTS > 0`` the admission guard
                re-weights THAT token to the deduped document count instead of
                registering a second reservation, so a request is never counted
                against itself. Leave ``None`` on the SDK path: the guard mints
                and releases its own short-lived reservation.
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
                notified via the ingress mailbox and picks up the
                newly-enqueued doc mid-batch (feeder) or at the batch
                boundary (quiescence decision).
        """
        # Concurrency contract: enqueue may proceed concurrently with the
        # processing loop because (a) full_docs is upserted before
        # doc_status, so a consistency check never sees a ghost row, and
        # (b) the running loop observes the ingress mailbox mid-batch and
        # at every batch boundary, so work that arrives while busy is
        # picked up without a new run.  Three states still block enqueue:
        #   * ``scanning_exclusive`` — scan task is in its CLASSIFICATION
        #     phase, reading doc_status to classify files and possibly
        #     deleting stale stubs.  Concurrent enqueue would race
        #     against scan's reads / mutations.  ``from_scan=True``
        #     lifts this guard for the scan task's own enqueues.
        #     ``scanning`` alone (the processing phase) does NOT block,
        #     identical to the upload-during-busy case.
        #   * ``manual_freeze_requested`` — a manual retry has frozen new
        #     ingress while it drains the pipeline to idle and exclusively
        #     resets FAILED→PENDING (LR2 §6.1/§7.2).  Also lifted by
        #     ``from_scan=True`` (the scan owns the manual operation).
        #   * ``destructive_busy`` — clear / delete is dropping storages
        #     or removing input files; a concurrent write would be
        #     silently clobbered.
        #
        # The check below is a FAST FAIL, not the guarantee: it registers
        # nothing, so a fence raised after it (a manual retry freezing ingress)
        # would not see this call at all.  The same ``reject_when`` list is
        # re-evaluated atomically with the pending-enqueue reservation that
        # covers the writes — see ``_reserve_ingress_slot``.  Checking here as
        # well is worth it because it refuses before the body is validated,
        # parsed and deduped.
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )
        reject_when = []
        if not from_scan:
            reject_when.append(
                (
                    "scanning_exclusive",
                    "Cannot enqueue while scan is classifying files; wait for the "
                    "classification phase to finish before retrying.",
                )
            )
            # A manual retry (LR2 §6.1/§7.2) froze new ingress while it drains
            # the pipeline and exclusively resets FAILED→PENDING. Lifted for
            # ``from_scan=True`` exactly like scanning_exclusive: the scan is the
            # manual operation's own driver, so its enqueues must not self-block.
            reject_when.append(
                (
                    "manual_freeze_requested",
                    "Cannot enqueue while a manual retry is draining the pipeline; "
                    "wait for it to finish before retrying.",
                )
            )
        reject_when.append(
            (
                "destructive_busy",
                "Cannot enqueue while pipeline is clearing or deleting documents; "
                "wait for the running job to finish before retrying.",
            )
        )
        mutation_result = await check_pipeline_status_mutation(
            pipeline_status,
            pipeline_status_lock,
            reject_when=reject_when,
            # A caller holding a registered pending-enqueue reservation was
            # admitted before any fence raised since, and the fence holder either
            # could not raise it while that reservation exists or is waiting for
            # it (LR2 §9.2). Refusing here would drop the work of a request whose
            # client was already told it was accepted — and for /text there is no
            # input file to rediscover, so the content would simply be lost.
            exempt_if_reserved=admission_token,
        )
        if not mutation_result.acquired:
            # Structured, not a bare message (LR2 §9.1): an SDK caller must be
            # able to tell a fenced workspace (503) from a bounded freeze/scan/
            # destructive window worth retrying (409) without string matching.
            raise PipelineReservationConflictError(
                mutation_result.message or "pipeline reservation is unavailable",
                conflict=mutation_result.conflict,
                fence=mutation_result.fence,
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

        if docs_format not in (FULL_DOCS_FORMAT_RAW, FULL_DOCS_FORMAT_PENDING_PARSE):
            raise ValueError(
                f"Unsupported docs_format {docs_format!r}; expected "
                f"{FULL_DOCS_FORMAT_RAW!r} or {FULL_DOCS_FORMAT_PENDING_PARSE!r}. "
                "The 'lightrag' enqueue format was removed; already-parsed "
                "documents are resumed via the full_docs parse_format marker "
                "and ReuseParser."
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

        def _parse_engine_at(index: int, doc_format: str) -> str | None:
            if parse_engine is None:
                return None
            raw = str(parse_engine[index] or "").strip()
            if not raw:
                return None
            # ``parse_engine`` may carry engine parameters encoded in hint
            # syntax (``mineru(page_range=1-3,language=en)``).  Decode +
            # validate + re-encode canonically so a direct SDK/API caller (who
            # bypasses ``resolve_parser_directives``) gets the same rejection /
            # coercion as the upload path; raise on a malformed directive.
            from lightrag.parser.routing import (
                decode_parse_engine,
                encode_parse_engine,
                seed_smart_heading_param,
            )

            engine, params, errs = decode_parse_engine(raw)
            if errs:
                raise ValueError(f"Invalid parse_engine {raw!r}: " + "; ".join(errs))
            if not engine:
                return None
            if doc_format == FULL_DOCS_FORMAT_PENDING_PARSE:
                # The DOCX_SMART_HEADING global default must materialize here
                # as well: a direct caller's bare ``native`` on a .docx
                # persists with the seed (same contract as an upload), and an
                # explicit ``native(smart_heading=false)`` stays the opt-out.
                # RAW docs are excluded — there parse_engine is a record of
                # the engine that ALREADY extracted the content, and no docx
                # parser runs, so injecting the seed would falsify it.
                seed_smart_heading_param(engine, params, file_paths[index])
            return encode_parse_engine(engine, params) if params else engine

        def _process_options_at(index: int) -> str:
            if process_options is None:
                return ""
            from lightrag.parser.routing import sanitize_process_options

            return sanitize_process_options(process_options[index])

        def _chunk_options_at(index: int) -> dict[str, Any]:
            """Resolve the per-doc slim chunk_options snapshot.

            Projects the chunker config down to the one strategy
            sub-dict selected by the doc's ``process_options`` (F by
            default) — the persisted ``full_docs[doc_id]['chunk_options']``
            carries only the params actually consumed at process time.

            When the caller supplied ``chunk_options`` we slim it
            against the per-doc options (deep-copying internally so two
            docs broadcast from a single dict cannot share mutable
            sub-dicts); otherwise we build a fresh snapshot from
            ``self.addon_params['chunker']``.

            F-strategy runtime args (``split_by_character`` /
            ``split_by_character_only`` from :meth:`LightRAG.ainsert`)
            are baked into the snapshot upstream — ainsert calls
            :func:`lightrag.parser.routing.resolve_chunk_options` itself
            and passes the result via ``chunk_options=``.  This function
            is purely a persistence helper; chunker-config construction
            is not its concern.
            """
            from lightrag.parser.routing import (
                resolve_chunk_options,
                slim_chunk_options,
            )

            doc_options = _process_options_at(index)
            if chunk_options is not None:
                return slim_chunk_options(chunk_options[index], doc_options)
            return resolve_chunk_options(self.addon_params, process_options=doc_options)

        # 1. Validate ids and build contents (when lightrag: no content dedup, content may be empty)
        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

        # Canonicalize every input filename once: the stored ``file_path``
        # is hint-stripped and serves UI display, filename dedup, and the
        # deterministic doc_id seed in one go.
        file_paths_canonical = [
            normalize_document_file_path(path) for path in file_paths
        ]
        contents: dict[str, dict[str, Any]] = {}
        source_to_doc_id: dict[str, str] = {}
        content_hash_to_doc_id: dict[str, str] = {}
        duplicate_attempts: list[dict[str, Any]] = []

        def _add_content(
            index: int,
            content: str,
            doc_format: str,
        ) -> None:
            file_path_canonical = file_paths_canonical[index]

            body_length = len(content)

            # Compute content hash: skip for pending_parse (content extracted later).
            content_hash: str | None = None
            if doc_format == FULL_DOCS_FORMAT_RAW:
                content_hash = compute_text_content_hash(content or "")

            known_source = has_known_document_source(file_path_canonical)
            if ids is not None:
                doc_id = ids[index]
            elif known_source:
                doc_id = compute_mdhash_id(file_path_canonical, prefix="doc-")
            elif doc_format == FULL_DOCS_FORMAT_RAW:
                doc_id = compute_mdhash_id(content or "", prefix="doc-")
            else:
                doc_id = compute_mdhash_id(
                    f"{file_path_canonical}-{track_id}-{index}", prefix="doc-"
                )

            if known_source and file_path_canonical in source_to_doc_id:
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": source_to_doc_id[file_path_canonical],
                        "file_path": file_path_canonical,
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
                        "file_path": file_path_canonical,
                        "content_length": body_length,
                        "existing_status": "batch_duplicate",
                        "existing_track_id": "",
                        "duplicate_kind": "content_hash",
                    }
                )
                return

            if known_source:
                source_to_doc_id[file_path_canonical] = doc_id
            if content_hash:
                content_hash_to_doc_id[content_hash] = doc_id

            content_data: dict[str, Any] = {
                "content": content,
                "file_path": file_path_canonical,
                "parse_format": doc_format,
            }
            if content_hash:
                content_data["content_hash"] = content_hash
            if engine := _parse_engine_at(index, doc_format):
                content_data["parse_engine"] = engine
            if doc_format == FULL_DOCS_FORMAT_PENDING_PARSE:
                source_file = Path(str(file_paths[index] or "").strip()).name
                if has_known_document_source(source_file):
                    content_data["source_file"] = source_file
            options_str = _process_options_at(index)
            if options_str:
                content_data["process_options"] = options_str
            # Always snapshot chunk_options at enqueue time — independent
            # of whether process_options selected a specific strategy —
            # so the per-doc parameters are frozen even when ``F``
            # (default) is used.
            content_data["chunk_options"] = _chunk_options_at(index)
            contents[doc_id] = content_data

        # ``ids`` outranks ``docs_format`` by design: explicit ids mark the
        # SDK raw direct-insert path (ainsert), which always enqueues the
        # sanitized body as RAW. pending_parse (server upload) never passes
        # ids, so the two never legitimately combine.
        if ids is not None:
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
        def _initial_doc_status(
            doc_id: str, content_data: dict[str, Any]
        ) -> dict[str, Any]:
            body_text = content_data.get("content", "")
            base: dict[str, Any] = {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(body_text),
                "content_length": len(body_text),
                # The row's immutable scheduling key is its actual first
                # persistence time. Scan mtime controls pre-persistence enqueue
                # order only; filesystem timestamps never masquerade as row
                # creation metadata.
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data["file_path"],
                "track_id": track_id,
            }
            if content_data.get("content_hash"):
                base["content_hash"] = content_data["content_hash"]
            metadata: dict[str, Any] = {}
            options_str = content_data.get("process_options") or ""
            if options_str:
                # Mirror process_options into doc_status.metadata so admin UIs
                # can surface the per-document strategy without a full_docs lookup.
                metadata["process_options"] = options_str
            source_file = _read_source_file(content_data)
            if source_file:
                metadata["source_file"] = source_file
            if metadata:
                base["metadata"] = metadata
            return base

        new_docs: dict[str, Any] = {
            id_: _initial_doc_status(id_, content_data)
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
        # ingress publish inside is fine; no caller holds
        # pipeline_status_lock first then needs enqueue_serialize).
        enqueue_serialize_lock = get_namespace_lock(
            ENQUEUE_SERIALIZE_LOCK_NAMESPACE, workspace=self.workspace
        )
        status_upsert_error: Exception | None = None
        process_after_status_error = False

        # The exit stack releases an admission reservation minted below (SDK
        # path) once the writes are done, on every exit including cancellation —
        # without indenting the whole critical section. It unwinds AFTER the
        # serialize lock, so the reservation outlives the doc_status upsert
        # exactly as §9.2 requires ("token 持续到 doc_status 可见").
        async with AsyncExitStack() as admission_stack, enqueue_serialize_lock:
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

                # 3a. Filename-based dedup: same basename always treated as
                # duplicate. Resolved STRICTLY — a storage failure raises rather
                # than reporting "no match", which this branch would read as
                # "not a duplicate" and admit a second row for a filename that
                # already exists.
                resolution = await resolve_existing_doc_source(
                    self.doc_status, content_data["file_path"]
                )
                if isinstance(resolution, SourceConflict):
                    # Several primaries already share this basename (a historical
                    # collision). Refuse rather than attach the new document to an
                    # arbitrary one of them: which row is "the original" is
                    # exactly what an operator has to decide, via the
                    # source-conflict repair flow. The record is trackable so the
                    # submitter sees why nothing was processed.
                    unique_new_doc_ids.discard(doc_id)
                    duplicate_attempts.append(
                        {
                            "doc_id": doc_id,
                            "file_path": content_data["file_path"],
                            "content_length": new_docs.get(doc_id, {}).get(
                                "content_length", 0
                            ),
                            "conflict_sample_doc_ids": resolution.sample_doc_ids,
                            "conflict_candidate_count": resolution.candidate_count,
                            "duplicate_kind": "filename_conflict",
                        }
                    )
                    continue
                if isinstance(resolution, SourceUnique):
                    unique_new_doc_ids.discard(doc_id)
                    duplicate_attempts.append(
                        {
                            "doc_id": doc_id,
                            "original_doc_id": resolution.doc_id,
                            "file_path": content_data["file_path"],
                            "content_length": new_docs.get(doc_id, {}).get(
                                "content_length", 0
                            ),
                            "existing_status": doc_status_field(
                                resolution.doc, "status", "unknown"
                            ),
                            "existing_track_id": doc_status_field(
                                resolution.doc, "track_id", ""
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
                    self.doc_status,
                    content_hash,
                    # This id does not exist yet (filter_keys removed the ones
                    # that do); passing it lets the lookup ignore a pointer row
                    # that names it as ITS original, which is what makes content
                    # whose primary row was deleted re-ingestible.
                    candidate_doc_id=doc_id,
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

            # The documents this call will actually write as new work. Computed
            # BEFORE the reservation below, because the reservation's weight is
            # the deduped count — and before the duplicate records are stored,
            # because the reservation has to cover every write in this section.
            admitted_doc_ids = [
                doc_id for doc_id in unique_new_doc_ids if doc_id in new_docs
            ]

            # Ingress reservation + admission (LR2 §9.2), at the ONE chokepoint
            # every entry point funnels through: endpoints, the SDK's
            # ainsert/apipeline_enqueue_documents, and any future caller. It sits
            # here — after dedup, before the first storage write of this critical
            # section — so a manual retry's exclusive FAILED reset cannot start
            # while this enqueue is on its way to the writes: a freeze refuses a
            # new reservation, and DRAIN_TO_IDLE waits for the reservations that
            # already exist. The entry fence check above is only a fast fail; it
            # registers nothing, so on its own it leaves this whole span open.
            #
            # Two callers legitimately skip it:
            #   * ``from_scan`` — exempt from the cap by design (§9.1: a scan may
            #     exceed it, and the active rows it creates are what makes
            #     ordinary uploads wait). Its exclusion is structural instead: it
            #     holds ``scanning_exclusive`` across its whole enqueue span,
            #     which refuses every processing reservation, so no manual reset
            #     can be draining alongside it.
            #   * a caller that already holds a reservation while admission is
            #     disabled — there is no capacity to charge, and the drain is
            #     already waiting for that token.
            if not from_scan and (
                admission_token is None or self.max_pending_documents > 0
            ):
                minted_token = await self._reserve_ingress_slot(
                    pipeline_status,
                    pipeline_status_lock,
                    admission_token=admission_token,
                    requested=len(admitted_doc_ids),
                    reject_when=reject_when,
                )
                if minted_token is not None:
                    admission_stack.push_async_callback(
                        self._release_admission_reservation, minted_token
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
                    if duplicate_kind == "filename_conflict":
                        # No original is named: several primaries already share
                        # this basename and choosing between them is the
                        # operator's decision (source-conflict repair), so the
                        # message carries the bounded candidate sample instead of
                        # asserting one of them is "the" original.
                        count = attempt.get("conflict_candidate_count")
                        sample = attempt.get("conflict_sample_doc_ids") or ()
                        error_msg = (
                            f"{count if count is not None else 'Multiple'} existing "
                            f"documents already share this file name, so this "
                            f"upload cannot be attributed to one of them. Repair "
                            f"the conflict by doc id first (sample doc ids: "
                            f"{', '.join(sample) or 'unavailable'})."
                        )
                        summary = (
                            f"[DUPLICATE:{duplicate_kind}] Unresolved file-name "
                            f"conflict; repair required"
                        )
                    else:
                        if duplicate_kind == "content_hash":
                            error_prefix = "Identical content already exists under another filename."
                        else:
                            error_prefix = "File name already exists."
                        error_msg = (
                            f"{error_prefix} "
                            f"Original doc_id: {attempt.get('original_doc_id', doc_id)}, "
                            f"Status: {attempt.get('existing_status', 'unknown')}"
                        )
                        summary = (
                            f"[DUPLICATE:{duplicate_kind}] Original document: "
                            f"{attempt.get('original_doc_id', doc_id)}"
                        )
                    duplicate_docs[dup_record_id] = {
                        "status": DocStatus.FAILED,
                        "content_summary": summary,
                        "content_length": attempt.get("content_length", 0),
                        "chunks_count": 0,
                        "chunks_list": [],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": file_path,
                        "track_id": track_id,  # Use current track_id for tracking
                        "error_msg": error_msg,
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

            # Filter new_docs to only include documents with unique IDs (the set
            # the reservation above was weighted for).
            new_docs = {doc_id: new_docs[doc_id] for doc_id in admitted_doc_ids}

            if not new_docs:
                logger.warning("No new unique documents were found.")
                return

            # Ingress handle for the publishes below: a document notification
            # per newly-stored doc (routed mid-batch by the feeder or at the
            # batch boundary) and an auto-rescan flag on a partial doc_status
            # commit.  The mailbox is the wake-up signal only — a dropped
            # message is still recovered by the PENDING doc_status row plus
            # the next run's initial strict scan.
            ingress = await get_pipeline_ingress(self.workspace)

            # 4. Store document content in full_docs and status in doc_status
            full_docs_data = {
                doc_id: {
                    "content": contents[doc_id].get("content", ""),
                    "file_path": contents[doc_id]["file_path"],
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
            try:
                await self.doc_status.upsert(new_docs)
            except Exception as exc:
                status_upsert_error = exc
                # A doc-status write may partially commit before failing (e.g.
                # OpenSearch bulk reports per-item failures after some PENDING
                # rows already landed). Decide how to avoid stranding committed
                # rows here, under the same lock the processing loop uses for
                # its atomic busy/exit handoff; the actual wake runs after the
                # lock is released (see below). The original storage error is
                # always re-raised.
                try:
                    async with pipeline_status_lock:
                        # Partial commit: some PENDING rows may have landed.
                        # Arm the auto-rescan flag (consumed at the next
                        # quiescence point under this same lock) so a busy run
                        # picks the rows up; when idle, run one inline pass
                        # below instead.  If the arm itself fails the rows stay
                        # PENDING until the next run's initial strict scan —
                        # the enqueue re-raises the storage error either way.
                        if not pipeline_status.get("busy"):
                            process_after_status_error = True
                        _rearm_auto_rescan(ingress)
                except Exception as wake_error:
                    logger.error(
                        "Failed to wake document processing after enqueue "
                        "status upsert error (partially-committed rows stay "
                        "PENDING until the next run's initial scan): "
                        f"{wake_error}"
                    )
            else:
                logger.debug(f"Stored {len(new_docs)} new unique documents")

        if status_upsert_error is not None:
            if process_after_status_error:
                # Best-effort recovery, NOT a confirmed partial-commit signal:
                # this fires on ANY upsert error while the pipeline is idle,
                # because neither doc_status.upsert nor this function can
                # cheaply tell whether rows actually landed (only some backends
                # like OpenSearch bulk commit partially before raising). Safe to
                # over-call — apipeline_process_enqueue_documents is busy-gated
                # and queries the real doc_status, so with nothing committed it
                # is a cheap no-op ("No documents to process"). The caller will
                # not trigger processing itself (it is about to get an
                # exception), so we run one inline pass here before re-raising.
                try:
                    await self.apipeline_process_enqueue_documents()
                except Exception as wake_error:
                    logger.warning(
                        "Failed to start document processing after enqueue "
                        f"status upsert error: {wake_error}"
                    )
            # Bare re-raise preserves the exception's original __traceback__:
            # only the local `exc` name was unbound at the end of the except
            # block; the exception object and its traceback survive via
            # status_upsert_error.
            raise status_upsert_error

        # Notify any in-flight processing loop that new work has arrived.
        # The document publish happens INSIDE the ``pipeline_status_lock``
        # critical section: the consumer's atomic exit decision
        # (:meth:`_decide_pipeline_next_step`) checks the mailbox and releases
        # ``busy`` under this very lock, so publishing outside it would let a
        # quiescing run observe an empty mailbox, release, and only then see
        # the messages land — stranding an enqueue-only doc in an idle
        # mailbox.  Serialized on the lock, a publish either lands before the
        # decision (which then sees it) or after the release (busy already
        # False — the designed idle-enqueue case).  ``put_documents`` is ONE
        # server-side batch call, so the lock is never held across N Manager
        # RPCs; overflow past the bounded channel coalesces into the
        # auto-rescan flag server-side.
        #
        # A wake-up side-effect must never fail an enqueue that already
        # durably committed doc_status: a multiprocess batch publish is a
        # Manager RPC that can raise on a Manager outage.  On a publish
        # failure, fall back to arming the auto-rescan flag (one smaller RPC,
        # same critical section); if that also fails, log — the docs are
        # PENDING rows and the next run's initial strict scan recovers them.
        async with pipeline_status_lock:
            try:
                ingress.put_documents(
                    [
                        PipelineIngressMessage(kind="document", doc_id=doc_id)
                        for doc_id in new_docs
                    ]
                )
            except Exception as publish_error:
                try:
                    _rearm_auto_rescan(ingress)
                    logger.warning(
                        "Ingress document notification failed after a "
                        "successful enqueue; armed the auto-rescan flag "
                        f"instead: {publish_error}"
                    )
                except Exception as rescan_error:
                    logger.error(
                        "Ingress document notification failed after a "
                        "successful enqueue and the auto-rescan fallback "
                        "failed too (docs stay PENDING until the next run's "
                        f"initial scan): publish={publish_error}; "
                        f"rescan={rescan_error}"
                    )

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
            file_path = normalize_document_file_path(
                error_file.get("file_path", "unknown_file")
            )
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

    async def apipeline_process_enqueue_documents(
        self, _holding_busy: bool = False, token: str | None = None
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

        ``_holding_busy`` (internal): when True the caller already owns the
        ``busy`` slot and hands it to this run atomically — used by
        ``ainsert_custom_chunks`` to drain mailbox work that arrived
        while it held ``busy``, without ever dropping ``busy`` to False (which
        would open a window for a clear/delete/scan reservation to start and
        drop the accepted-but-unprocessed document). This run takes over the
        existing slot instead of re-acquiring it, and always releases it — on
        the no-work path, on a get-docs failure, and via the loop's finally —
        so a handoff can never wedge the pipeline as permanently busy.
        """
        # Workspace snapshot: status, lock and ingress all resolve through the
        # value captured at entry, so one run can never straddle two
        # coordination namespaces if ``self.workspace`` were reassigned
        # mid-run.  This pins the coordination layer only — the storages were
        # bound at init and switching workspaces on a live instance stays
        # unsupported.
        run_workspace = self.workspace
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=run_workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=run_workspace
        )

        # Owner token for this run's ``busy`` reservation. A handed-off run
        # (``_holding_busy``) reuses the caller's token — it already owns the
        # slot — so the caller's release and this run's release refer to the
        # same owner; a normal run mints its own.
        if _holding_busy:
            if token is None:
                raise ValueError("_holding_busy requires the owner token")
        else:
            token = uuid.uuid4().hex

        # A handed-off slot is ours from entry; a normal run owns it only after
        # it sets busy=True below. The finally releases ``busy`` iff we own it.
        holds_busy = _holding_busy

        # Tracks whether the loop already released ``busy`` under the same
        # critical section that observed an empty mailbox. This makes the
        # exit handoff atomic: a concurrent enqueue either publishes BEFORE we
        # release (loop continues with a fresh snapshot) or AFTER (it sees
        # busy=False and starts a new loop via its own process_enqueue).
        busy_released_in_loop = False

        # Ownership of a consumed-but-not-yet-committed wake-up signal: True
        # while a destructively-consumed signal (the entry absorb of the auto
        # flag, or a CONTINUE_AUTO/CONTINUE_DOCUMENT quiescence consumption)
        # has produced docs that NO batch has taken over yet.  The finally's
        # cancellation-resistant bookkeeping re-arms auto-rescan whenever the
        # run exits with this still set — a cancel, a validation failure, or
        # ANY other escape in that window would otherwise strand the docs:
        # they are PENDING rows whose channel entries were already drained, so
        # no other signal is left.  Initialized (with the ingress handle)
        # BEFORE the try so the finally can always read them.
        uncommitted_wakeup = False
        ingress = None

        # Forward-progress tracker for this run's manual DRAIN_TO_IDLE (LR2 §7.2).
        # Run-scoped: a run that exits releases the freeze, and the next one
        # re-drains from scratch, so a stall only ever has to be detected within
        # one run.
        drain_progress = _ManualDrainProgress()

        try:
            # Resolve the ingress handle BEFORE the reservation: the acquire's
            # busy-refusal arms the auto-rescan flag inside its own critical
            # section, and a Manager RPC handle must never be resolved lazily
            # while ``pipeline_status_lock`` is held.  A resolution failure
            # aborts before the slot is taken (a handed-off slot is still
            # released by the finally, so a handoff can never wedge).
            ingress = await get_pipeline_ingress(run_workspace)

            # Acquire the processing slot BEFORE reading doc_status. The queue
            # read and all downstream work MUST run under ``busy`` so a concurrent
            # clear/delete (which takes ``busy`` then rewrites storage) and a scan
            # classification phase (``scanning_exclusive``, refused by
            # acquire_processing_reservation) are held off — reading doc_status
            # outside the reservation would race their storage rewrites. A
            # handed-off run (``_holding_busy``) already owns the slot and takes it
            # over rather than re-acquiring.
            holds_busy = True
            reservation = await acquire_processing_reservation(
                pipeline_status,
                pipeline_status_lock,
                token=token,
                already_held=_holding_busy,
                pipeline_ingress=ingress,
                flags={
                    "job_name": "Default Job",
                    "job_start": datetime.now(timezone.utc).isoformat(),
                    "docs": 0,
                    "batchs": 0,
                    "cur_batch": 0,
                    "cancellation_requested": False,
                    "cancellation_reason": None,
                    "cancellation_detail": None,
                    "latest_message": "",
                },
            )
            if not reservation.acquired:
                holds_busy = False
                if (
                    reservation.conflict
                    is PipelineReservationConflict.RECOVERY_REQUIRED
                ):
                    logger.warning(reservation.message)
                elif reservation.conflict is PipelineReservationConflict.SCANNING:
                    logger.info(
                        "Scan is classifying files; processing will resume once "
                        "the classification phase finishes."
                    )
                else:
                    logger.info(
                        "Another process is already processing the document "
                        "queue. Request queued."
                    )
                return

            # Run-start peek protocol: the ingress mailbox is the ONLY source
            # of manual retry intent (processing carries no manual params). The
            # earliest sticky request — if any — BEGINS the manual drain right
            # here: set the freeze + owner (owner-checked, tied to this run's
            # busy token) BEFORE any AUTO work, so no new ingress enters during
            # the drain (LR2 §7.2). The request stays sticky and is ACKed only
            # after the exclusive FAILED→PENDING reset persists (in
            # BEGIN_EXCLUSIVE_RESET). FAILED is never swept inline — the initial
            # scan drains the AUTO backlog only. With no manual request pending,
            # this run IS the rescan: absorb the auto-rescan dirty flag so
            # quiescence doesn't re-trigger for work this very scan already
            # picks up.
            initial_statuses = _AUTO_RESUME_DOC_STATUSES
            manual_msg = ingress.peek_next_manual_retry()
            if manual_msg is not None:
                await self._begin_manual_drain(
                    manual_msg.request_id, token, pipeline_status, pipeline_status_lock
                )
                absorbed_auto_rescan = False
            else:
                absorbed_auto_rescan = ingress.consume_auto_rescan()
            # Own the absorbed signal from the INSTANT of consumption (see the
            # definition above the try): if the strict scan below raises —
            # Exception, CancelledError, any BaseException — the finally's
            # bookkeeping re-arms the flag; a task cancellation delivered
            # inside the scan would slip past any local `except Exception`
            # compensation, so there is none.  (A manual request was only
            # peeked and stays sticky on its own.)
            uncommitted_wakeup = absorbed_auto_rescan

            # Bounded scheduling sweep (LR2 Phase 2): the initial scan is now
            # the first PAGE of a keyset sweep. ``sweep_statuses`` /
            # ``sweep_cursor`` are threaded through every quiescence decision so
            # the run drains the KNOWN backlog page-by-page (CONTINUE_SWEEP_PAGE)
            # before honouring auto-rescan/document signals. A manual sweep and
            # PIPELINE_SCHEDULING_PAGE_SIZE=0 both collapse to a single
            # CURSOR_END page (legacy single-scan; see _next_scheduling_page).
            sweep_statuses = initial_statuses
            to_process_docs: dict[str, DocProcessingStatus]
            (
                to_process_docs,
                sweep_cursor,
            ) = await self._next_scheduling_page(sweep_statuses, CURSOR_START)

            # The scan is complete: an empty result means the signal was fully
            # SERVED, not lost — release ownership.  A non-empty result keeps
            # it owned until the consistency validator takes the docs over
            # (the auto flag is the canonical lost-notification recovery
            # signal and drained document messages are represented by their
            # PENDING rows, same as channel overflow).
            uncommitted_wakeup = absorbed_auto_rescan and bool(to_process_docs)

            if not to_process_docs:
                # Empty AUTO scan. Release the slot we just took WITHOUT the
                # "pipeline stopped" bookkeeping — a run that did no work should
                # record nothing — but through the atomic quiescence decision
                # that also consumes a wake-up signal (manual/auto/document) set
                # by a concurrent publisher AFTER our read, so a doc that landed
                # in that gap is not stranded in PENDING. A pending manual is
                # served here too: the decision returns BEGIN_EXCLUSIVE_RESET
                # (phase already DRAIN_TO_IDLE), whose refetch runs the reset and
                # ACKs — never RELEASED while a manual is being served.
                logger.info("No documents to process")
                decision = await self._decide_pipeline_next_step(
                    pipeline_status, pipeline_status_lock, ingress, sweep_cursor
                )
                if decision.step is PipelineNextStep.RELEASED:
                    # No pending work: slot released silently, finally is a no-op.
                    holds_busy = False
                    return
                if decision.step is PipelineNextStep.CANCELLED:
                    # Nothing was processed and nothing was consumed: stop —
                    # the finally resets the cancellation state (surfacing the
                    # internal halt when applicable) and releases the slot.
                    return
                # A wake-up signal (or an unfinished sweep / manual reset) was
                # pending: re-fetch per the decision and fall through into the
                # loop to process it.
                (
                    to_process_docs,
                    sweep_statuses,
                    sweep_cursor,
                ) = await self._refetch_for_decision(
                    decision,
                    sweep_statuses,
                    sweep_cursor,
                    ingress,
                    token=token,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    drain_progress=drain_progress,
                )
                uncommitted_wakeup = decision.step in (
                    PipelineNextStep.CONTINUE_AUTO,
                    PipelineNextStep.CONTINUE_DOCUMENT,
                ) and bool(to_process_docs)

            # Process documents until no more documents or requests
            while True:
                # Check for cancellation request at the start of main loop
                async with pipeline_status_lock:
                    status_snapshot = pipeline_status.copy()
                    if status_snapshot.get("cancellation_requested", False):
                        # Read the cause BEFORE resetting reason/detail below.
                        is_internal = (
                            status_snapshot.get("cancellation_reason")
                            == "internal_error"
                        )
                        label = self._cancellation_label(status_snapshot)

                        if is_internal:
                            # Unrecoverable storage error: halting is intentional
                            # (auto-retry into a broken backend will not recover).
                            # Surface at error level with an actionable message;
                            # affected docs stay queued (PENDING/FAILED) and are
                            # picked up when processing is restarted after the
                            # storage issue is resolved.
                            log_message = self._internal_halt_message(label)
                            logger.error(log_message)
                        else:
                            log_message = f"Pipeline cancelled ({label})"
                            logger.info(log_message)
                        pipeline_status.update(
                            {
                                "cancellation_requested": False,
                                "cancellation_reason": None,
                                "cancellation_detail": None,
                                "latest_message": log_message,
                            }
                        )
                        append_pipeline_history(status_snapshot, log_message)

                        # A signal consumed by the immediately-preceding
                        # decision (or the entry absorb) that no batch took
                        # over is restored by the finally's bookkeeping — one
                        # restore point covers this exit and every non-cancel
                        # escape (e.g. a validation failure) alike.

                        # Exit directly, skipping the quiescence decision: a
                        # cancel stops the WHOLE run and the mailbox is fully
                        # retained for the next explicit trigger.
                        return

                if not to_process_docs:
                    log_message = "All enqueued documents have been processed"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    append_pipeline_history(pipeline_status, log_message)
                    decision = await self._decide_pipeline_next_step(
                        pipeline_status, pipeline_status_lock, ingress, sweep_cursor
                    )
                    if decision.step is PipelineNextStep.RELEASED:
                        busy_released_in_loop = True
                        break
                    if decision.step is PipelineNextStep.CANCELLED:
                        if decision.internal_error:
                            break  # finally surfaces the actionable halt
                        continue  # loop-top handler does the cancel bookkeeping
                    (
                        to_process_docs,
                        sweep_statuses,
                        sweep_cursor,
                    ) = await self._refetch_for_decision(
                        decision,
                        sweep_statuses,
                        sweep_cursor,
                        ingress,
                        token=token,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                        drain_progress=drain_progress,
                    )
                    uncommitted_wakeup = decision.step in (
                        PipelineNextStep.CONTINUE_AUTO,
                        PipelineNextStep.CONTINUE_DOCUMENT,
                    ) and bool(to_process_docs)
                    continue

                # Validate document data consistency and fix any issues. The
                # AUTO sweep never carries FAILED (manual resets happen in the
                # exclusive phase), so the validator only normalises interrupted
                # orphans / stale PENDING here — it no longer ACKs a manual
                # request.
                to_process_docs = await self._validate_and_fix_document_consistency(
                    to_process_docs, pipeline_status, pipeline_status_lock
                )
                # The validator has taken the docs over (its resets are
                # persisted): the consumed signal is committed, and from here
                # the batch/feeder teardown and the PENDING rows own recovery —
                # a later cancel must NOT re-arm.
                uncommitted_wakeup = False

                if not to_process_docs:
                    log_message = (
                        "No valid documents to process after consistency check"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    append_pipeline_history(pipeline_status, log_message)
                    decision = await self._decide_pipeline_next_step(
                        pipeline_status, pipeline_status_lock, ingress, sweep_cursor
                    )
                    if decision.step is PipelineNextStep.RELEASED:
                        busy_released_in_loop = True
                        break
                    if decision.step is PipelineNextStep.CANCELLED:
                        if decision.internal_error:
                            break  # finally surfaces the actionable halt
                        continue  # loop-top handler does the cancel bookkeeping
                    (
                        to_process_docs,
                        sweep_statuses,
                        sweep_cursor,
                    ) = await self._refetch_for_decision(
                        decision,
                        sweep_statuses,
                        sweep_cursor,
                        ingress,
                        token=token,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                        drain_progress=drain_progress,
                    )
                    uncommitted_wakeup = decision.step in (
                        PipelineNextStep.CONTINUE_AUTO,
                        PipelineNextStep.CONTINUE_DOCUMENT,
                    ) and bool(to_process_docs)
                    continue

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)
                pipeline_status.update(
                    {
                        "docs": len(to_process_docs),
                        "batchs": len(to_process_docs),
                        "cur_batch": 0,
                        "latest_message": log_message,
                    }
                )
                append_pipeline_history(pipeline_status, log_message)

                await self._run_pipeline_batch(
                    to_process_docs,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    ingress=ingress,
                    token=token,
                )

                # Atomic exit handoff: if a wake-up signal arrived during this
                # batch (a manual retry request, the auto-rescan flag, resident
                # document messages published by a concurrent enqueue while
                # busy=True), consume it and refetch.  Otherwise
                # release ``busy`` under the SAME lock so a concurrent
                # publisher cannot squeeze a signal past us into a
                # now-stranded state.  A cancellation observed during the batch
                # (user cancel or internal-error abort) wins INSIDE the same
                # critical section and consumes nothing: cancellation stops the
                # WHOLE run — no new epoch — with the ingress fully retained
                # for the next explicit trigger.
                decision = await self._decide_pipeline_next_step(
                    pipeline_status, pipeline_status_lock, ingress, sweep_cursor
                )
                if decision.step is PipelineNextStep.RELEASED:
                    busy_released_in_loop = True
                    break
                if decision.step is PipelineNextStep.CANCELLED:
                    if decision.internal_error:
                        # Break to the finally, whose bookkeeping surfaces the
                        # actionable halt message and resets the state.
                        break
                    # Loop back into the loop-top handler for its "Pipeline
                    # cancelled" bookkeeping.
                    continue

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                append_pipeline_history(pipeline_status, log_message)

                # Fetch the next batch: the next page of the current sweep
                # (CONTINUE_SWEEP_PAGE), a manual reset/drain (CONTINUE_MANUAL /
                # BEGIN_EXCLUSIVE_RESET) or a fresh sweep per the decision.
                (
                    to_process_docs,
                    sweep_statuses,
                    sweep_cursor,
                ) = await self._refetch_for_decision(
                    decision,
                    sweep_statuses,
                    sweep_cursor,
                    ingress,
                    token=token,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    drain_progress=drain_progress,
                )
                uncommitted_wakeup = decision.step in (
                    PipelineNextStep.CONTINUE_AUTO,
                    PipelineNextStep.CONTINUE_DOCUMENT,
                ) and bool(to_process_docs)

        finally:
            stopped_message = "Enqueued document processing pipeline stopped"

            async def _finalize():
                # Cancellation-resistant, owner-checked release + bookkeeping.
                # Only runs when THIS invocation held the slot AND did work: a
                # rejected acquire (busy/scanning/recovery) and an empty-queue run
                # both release their slot inline and clear holds_busy, so they skip
                # this — no spurious stop message, and we never touch another run's
                # state.
                if not holds_busy:
                    return
                lock = get_namespace_lock("pipeline_status", workspace=run_workspace)
                status = await get_namespace_data(
                    "pipeline_status", workspace=run_workspace
                )
                async with lock:
                    # Restore a wake-up signal a decision consumed but no
                    # batch took over — regardless of WHY the run is exiting
                    # (user cancel, a validation failure, any escape): the
                    # docs behind it are PENDING rows whose channel entries
                    # were already drained, so without this they would wait
                    # for an unrelated trigger.  Before the release below, so
                    # the next acquirer observes the signal; failure only
                    # degrades to next-explicit-trigger recovery.
                    if uncommitted_wakeup and ingress is not None:
                        try:
                            _rearm_auto_rescan(ingress)
                        except Exception as restore_error:
                            logger.error(
                                "[pipeline] failed to restore the consumed "
                                "wake-up signal on exit (docs stay PENDING "
                                "until the next explicit trigger): "
                                f"{restore_error}"
                            )

                    status_snapshot = status.copy()
                    current_owner = _reservation_owner_token(
                        status_snapshot.get("busy_owner")
                    )
                    updates = {}
                    released_here = False
                    # Release ``busy`` only if the loop did not already release it
                    # under the atomic exit check AND we still own the slot.
                    if not busy_released_in_loop and current_owner == token:
                        # Clear the manual freeze in the SAME atomic update: an
                        # abnormal exit (exception / cancel) mid drain-or-reset
                        # must never leave a live-but-gone owner's freeze wedging
                        # uploads forever (LR2 §6.1). The sticky manual request
                        # survives in the mailbox and is re-run from Start by the
                        # next owner (idempotent). A clean PROCESS-phase exit
                        # already cleared it via _end_manual_drain, so this is a
                        # no-op there.
                        updates.update(
                            {
                                "busy": False,
                                "busy_owner": None,
                                "manual_freeze_requested": False,
                                "manual_freeze_started_at": None,
                                "manual_resetting": False,
                                "manual_phase": MANUAL_PHASE_IDLE,
                                "manual_owner": None,
                            }
                        )
                        current_owner = None  # slot is now free
                        released_here = True

                    # The run OWNED the slot iff the loop released it atomically
                    # or the release above just did.  An acquire that raised
                    # before stamping the owner (e.g. the busy-refusal's
                    # auto-rescan arm, or a snapshot RPC failure on an idle
                    # pipeline) never owned it — no stop log and no bookkeeping
                    # for a run that never started, so an idle pipeline_status
                    # is not stamped with a "stopped" message and a live
                    # holder's log is not polluted with one.
                    if busy_released_in_loop or released_here:
                        logger.info(stopped_message)

                    # Bookkeeping (cancellation reset, stopped/halt messages) must
                    # NOT touch a slot a DIFFERENT owner has already taken: once the
                    # loop released busy, a concurrent enqueue can acquire and set
                    # its own cancellation state, which we must not clear. Only run
                    # bookkeeping while the slot is still ours or free.
                    if current_owner is not None and current_owner != token:
                        return
                    if not busy_released_in_loop and not released_here:
                        return

                    # An internal-error abort exits via the batch's break (not the
                    # loop-top handler), so surface the actionable halt reason here
                    # BEFORE clearing the reason/detail (read it first so
                    # _cancellation_label sees the cause).
                    internal_halt = None
                    if status_snapshot.get("cancellation_reason") == "internal_error":
                        internal_halt = self._internal_halt_message(
                            self._cancellation_label(status_snapshot)
                        )
                        logger.error(internal_halt)
                    updates.update(
                        {
                            "cancellation_requested": False,
                            "cancellation_reason": None,
                            "cancellation_detail": None,
                            "latest_message": internal_halt
                            if internal_halt
                            else stopped_message,
                        }
                    )
                    status.update(updates)
                    append_pipeline_history(status_snapshot, stopped_message)
                    if internal_halt is not None:
                        append_pipeline_history(status_snapshot, internal_halt)

            await run_to_completion(_finalize)

    # ============================================================
    # Pipeline orchestration
    # ============================================================

    async def _feeder_next_batch(self, ingress) -> list[PipelineIngressMessage]:
        """Block until at least one document message is available, then return
        up to ``_FEEDER_DRAIN_LIMIT`` messages from the channel.

        Two ingress flavors (see :mod:`lightrag.kg.pipeline_ingress`):

        * single-process ``AsyncioPipelineIngress`` — fully event-driven:
          ``await get_document()`` returns one message (pairing ``task_done``),
          then a bounded non-blocking ``drain_documents`` sweeps up to the
          remaining limit.
        * multiprocess ``ManagerPipelineIngress`` — no async get; block a
          worker thread in ``wait_for_documents(timeout)`` (a WAIT that never
          dequeues, so a cancelled/timed-out thread cannot steal a message),
          then a bounded ``drain_documents``.  An empty poll returns ``[]`` and
          the feeder loops.

        The drain is bounded so the feeder returns to its control-signal check
        instead of draining the whole 4096-deep channel in one pass, and so a
        teardown re-publish never has to restore more than one bounded drain.
        Cancellation propagates out of the ``await`` here (nothing dequeued
        yet); a message drained but not routed is carried in the feeder's
        deferred set and re-published on teardown.
        """
        if hasattr(ingress, "get_document"):
            first = await ingress.get_document()
            return [first, *ingress.drain_documents(limit=_FEEDER_DRAIN_LIMIT - 1)]
        await asyncio.to_thread(ingress.wait_for_documents, _FEEDER_POLL_SECONDS)
        return ingress.drain_documents(limit=_FEEDER_DRAIN_LIMIT)

    async def _admit_fed_document(
        self,
        ctx: "_BatchRunContext",
        doc_id: str,
        status_doc: DocProcessingStatus,
        content_data: dict,
    ) -> None:
        """Two-stage admission of a feeder-routed document into its parse queue.

        Fixes the ghost-inflight window: the id is registered in
        ``routing_doc_ids`` (and the batch counters bumped) UNDER the lock
        BEFORE the bounded-queue ``put`` — which may block and be cancelled —
        and is only moved into ``inflight_doc_ids`` AFTER the put succeeds, with
        no ``await`` between the discard and the add.  A cancelled/failed put
        rolls the registration and counters back so the document is neither
        double-counted nor permanently shadowed from re-admission; the message
        itself is put back to the mailbox by the feeder's teardown re-publish.

        ``content_data`` is the full_docs row the caller (``_route_one``) already
        read for its visibility check, reused here for engine routing so
        admission does not issue a second full_docs read.
        """
        file_path = getattr(status_doc, "file_path", "unknown_source")
        key = resolve_stored_document_parser_engine(
            file_path=file_path, content_data=content_data or {}
        )
        spec = ctx.parser_specs.get(key)
        group = spec.queue_group if spec is not None else "native"
        queue = ctx.parse_queues.get(group, ctx.parse_queues["native"])

        async with ctx.pipeline_status_lock:
            ctx.routing_doc_ids.add(doc_id)
            ctx.total_files += 1
            ctx.pipeline_status["docs"] = ctx.pipeline_status.get("docs", 0) + 1
            ctx.pipeline_status["batchs"] = ctx.pipeline_status.get("batchs", 0) + 1
        try:
            await queue.put((doc_id, status_doc))
        except BaseException:
            async with ctx.pipeline_status_lock:
                ctx.routing_doc_ids.discard(doc_id)
                ctx.total_files = max(0, ctx.total_files - 1)
                ctx.pipeline_status["docs"] = max(
                    0, ctx.pipeline_status.get("docs", 0) - 1
                )
                ctx.pipeline_status["batchs"] = max(
                    0, ctx.pipeline_status.get("batchs", 0) - 1
                )
            raise
        async with ctx.pipeline_status_lock:
            # No await between these two mutations: dedup never sees a gap.
            ctx.routing_doc_ids.discard(doc_id)
            ctx.inflight_doc_ids.add(doc_id)

    def _republish_documents(
        self, ingress, messages, *, source: str = "feeder"
    ) -> None:
        """Best-effort re-publish of drained-but-unresolved document messages.

        The document channel drain is destructive (``drain_documents`` removes
        the messages).  On a teardown — user cancel, internal error, or the
        exception backoff — the drained messages this feeder had not yet routed
        or confirmed stale/terminal must go back to the mailbox: a cancelled
        run consumes nothing further, so without the restore those PENDING
        docs would wait for an unrelated future trigger instead of the next
        run.  A multiprocess ``put_document`` is a Manager RPC; swallow
        failures — the docs are PENDING and still recovered by the next
        initial scan.

        ``source`` selects the log level: the feeder path stays at debug (its
        drops are per-message noise backed by auto-rescan / the next initial
        scan), while the supervisor's quiescence compensation
        (``"quiescence"``) logs a warning — there a swallowed drop is one of
        the last lines that can attribute why a doc waited for an unrelated
        trigger.
        """
        log = logger.debug if source == "feeder" else logger.warning
        for msg in messages:
            try:
                ingress.put_document(msg)
            except Exception as republish_error:
                log(
                    f"[{source}] document re-publish failed for "
                    f"{getattr(msg, 'doc_id', None)}: {republish_error}"
                )

    async def _route_one(
        self,
        ctx: "_BatchRunContext",
        msg: "PipelineIngressMessage",
        pending: dict,
    ) -> bool:
        """Resolve one fed document message.

        Returns True when the message is fully RESOLVED — admitted into a parse
        queue, or a confirmed stale/terminal / duplicate / journaled doc that is
        correctly dropped — and False when it was only SKIPPED for now (full_docs
        not yet visible, or a transient read) and so should be re-published if
        the batch tears down before a rescan can recover it.
        """
        doc_id = msg.doc_id
        if not doc_id:
            return True  # malformed notification: nothing to route
        async with ctx.pipeline_status_lock:
            duplicate = doc_id in ctx.routing_doc_ids or doc_id in ctx.inflight_doc_ids
        if duplicate:
            return True  # already routing/inflight — dedup
        status_doc = pending.get(doc_id)
        if status_doc is None:
            return True  # not PENDING (processed/failed/deleted) — stale/terminal
        if doc_status_custom_chunk_patch(status_doc) is not None:
            return True  # journaled → scan/custom-chunk recovery owns it
        try:
            full_doc = await self.full_docs.get_by_id(doc_id)
        except Exception as read_error:
            logger.debug(
                f"[feeder] full_docs read failed for {doc_id}, deferring "
                f"(auto-rescan recovers): {read_error}"
            )
            return False  # transient → skip, re-publish on teardown
        if not full_doc:
            return False  # not yet visible (ryw lag) → skip, re-publish on teardown
        await self._admit_fed_document(ctx, doc_id, status_doc, full_doc)
        return True  # admitted

    @staticmethod
    def _feeder_should_yield(
        ctx: "_BatchRunContext", ingress, *, check_auto: bool
    ) -> bool:
        """True when the feeder must stop admitting and let the batch reach its
        quiescence point:

        * cancellation requested (so ``/cancel_pipeline`` can complete);
        * a sticky manual retry request waiting (so an unbounded batch does not
          starve it) — both re-checked before every single admission, so the
          yield latency is one admission, not a whole drain;
        * (``check_auto``, at the top of each drain) the auto-rescan flag is
          dirty — a document-channel overflow dropped notifications, a partial
          upsert landed, or one of this feeder's own skips armed it. Yielding
          hands the batch to its boundary, where the supervisor consumes the
          flag and its strict rescan recovers those dropped/deferred PENDING
          docs; without it a sustained stream that never lets the batch reach a
          boundary would starve them. The feeder only PEEKS the flag (via
          ``counts``); the supervisor remains its sole consumer.

        NOT ``has_work()`` — that would busy-loop on a pending manual/auto entry.
        """
        if ctx.pipeline_cancel_event is not None and ctx.pipeline_cancel_event.is_set():
            return True
        if ingress.peek_next_manual_retry() is not None:
            return True
        if check_auto and ingress.counts().get("auto_rescan_pending"):
            return True
        return False

    def _feeder_epoch_full(self, ctx: "_BatchRunContext") -> bool:
        """True when this epoch has admitted ``PIPELINE_SCHEDULING_PAGE_SIZE``
        documents (``inflight`` + ``routing``) — the single-epoch bound of
        LR2 §6.4.

        ``inflight_doc_ids`` grows monotonically within a batch (it is the
        dedup set of everything routed into this epoch, never shrunk on
        completion), so this counts total admissions this epoch, not live
        concurrency. Once the count reaches the page size the feeder stops
        admitting and the remaining mailbox / doc_status work continues in the
        NEXT epoch (a fresh page via CONTINUE_SWEEP_PAGE, or a CONTINUE_DOCUMENT
        refetch). Disabled when paging is off (``page_size <= 0``) — the feeder
        then admits freely, exactly as before Phase 2. Read without the lock: a
        single-threaded ``len`` is a consistent snapshot and this is a soft
        backpressure gate, not a hard invariant.
        """
        page_size = getattr(self, "pipeline_scheduling_page_size", 0)
        if page_size <= 0:
            return False
        return len(ctx.inflight_doc_ids) + len(ctx.routing_doc_ids) >= page_size

    async def _pipeline_feeder(self, ctx: "_BatchRunContext", ingress) -> None:
        """Route documents that arrive DURING a batch into its parse queues so
        they process without waiting for the batch barrier.

        Pure accelerator: every message it discards or skips is a PENDING
        ``doc_status`` row that the supervisor's quiescence rescan or the
        next run's initial strict scan recovers.  The feeder never writes
        ``doc_status``; correctness is anchored by the doc_status source of
        truth plus the atomic quiescence decision — the feeder only removes
        latency.

        Per drained message: ROUTED (admitted), STALE/TERMINAL (dropped), or
        SKIPPED (full_docs not yet visible / a transient read — arms the
        auto-rescan flag; re-published on a teardown, see ``_route_one`` and
        ``_republish_documents``).

        **Bounded liveness (yields to control signals):** the feeder stops
        admitting — returns so the batch drains to its quiescence point — when
        cancellation is requested (otherwise ``/cancel_pipeline`` could never
        complete under a sustained upload stream that keeps the parse queue
        non-empty) or when a sticky manual retry request is waiting (it is only
        consumed at the batch boundary, so an unbounded batch would starve it).
        This is re-checked before the drain AND before EVERY admission
        (:meth:`_feeder_should_yield`), so a signal that lands while the feeder
        is admitting a full drain is honored within one admission, not after up
        to ``_FEEDER_DRAIN_LIMIT`` more.  It deliberately does NOT wait on
        ``has_work()`` — that would busy-loop on a pending manual/auto signal.

        **Deferred survives the whole feeder:** SKIPPED messages and a drain's
        unrouted tail accumulate in a feeder-lifetime ``deferred`` list and are
        re-published to the mailbox on ANY exit (the ``finally``) — the drain
        already removed them from the mailbox, so a skip forgotten from an
        earlier iteration would otherwise be lost on a cancelled run (which
        consumes nothing further).  An unexpected per-iteration failure logs,
        re-publishes just that drain's tail for retry, backs off, and continues
        rather than silently dying for the rest of the batch.
        """
        # Feeder-lifetime set: SKIPs (full_docs not yet visible / transient read)
        # and any drained-but-unrouted tail. Restored to the mailbox on exit.
        deferred: list = []
        try:
            while True:
                batch: list = []
                reached = 0
                try:
                    # Top-of-drain yield check is INSIDE the try so a transient
                    # manual-peek / counts RPC failure (multiprocess) is logged
                    # and retried, not left to silently kill the feeder. Includes
                    # the auto-rescan flag (overflow / partial upsert / this
                    # feeder's own skips) so a sustained stream still reaches a
                    # boundary where the supervisor's rescan recovers the dropped
                    # PENDING docs.
                    if self._feeder_should_yield(ctx, ingress, check_auto=True):
                        return
                    # Single-epoch bound (LR2 §6.4): once this batch holds
                    # PIPELINE_SCHEDULING_PAGE_SIZE docs the feeder stops
                    # admitting; the rest continues in the next epoch (a fresh
                    # page, or a CONTINUE_DOCUMENT refetch of the mailbox).
                    if self._feeder_epoch_full(ctx):
                        return
                    batch = await self._feeder_next_batch(ingress)
                    drained_ids = [msg.doc_id for msg in batch if msg.doc_id]
                    if drained_ids:
                        # Bounded to THIS drain's ids (not the whole PENDING
                        # set): hydrate full status, keep only rows still
                        # PENDING. A drained doc now processed/failed/deleted is
                        # simply absent, and _route_one drops it as stale.
                        hydrated = await self.doc_status.get_full_docs_by_ids(
                            drained_ids, strict=True
                        )
                        pending = {
                            doc_id: doc
                            for doc_id, doc in hydrated.items()
                            if getattr(doc, "status", None) == DocStatus.PENDING
                        }
                        for reached, msg in enumerate(batch):
                            # Honor an in-flight cancel/manual WITHIN one doc
                            # (auto-rescan is coarser — checked per drain above).
                            if self._feeder_should_yield(
                                ctx, ingress, check_auto=False
                            ):
                                deferred.extend(batch[reached:])  # unprocessed tail
                                return
                            # Re-check the epoch bound mid-drain: earlier
                            # admissions in THIS loop may have reached the cap.
                            if self._feeder_epoch_full(ctx):
                                deferred.extend(batch[reached:])
                                return
                            if not await self._route_one(ctx, msg, pending):
                                deferred.append(msg)  # SKIP — survives to teardown
                                # Arm the recovery signal so the skip is picked up
                                # by the supervisor's next rescan even if this
                                # feeder never otherwise reaches a boundary.
                                _rearm_auto_rescan(ingress)
                    reached = len(batch)  # every message reached a decision
                except asyncio.CancelledError:
                    deferred.extend(batch[reached:])  # unprocessed tail → finally
                    raise
                except Exception as feeder_error:
                    logger.warning(
                        "[feeder] in-batch feeder iteration failed (batch "
                        f"continues; re-published messages recover docs): {feeder_error}"
                    )
                    # Retry just this drain's unprocessed tail; the accumulated
                    # deferred skips are restored on exit, not bounced each error.
                    self._republish_documents(ingress, batch[reached:])
                    await asyncio.sleep(_FEEDER_POLL_SECONDS)
        finally:
            self._republish_documents(ingress, deferred)

    async def _run_pipeline_batch(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
        *,
        pipeline_status: dict,
        pipeline_status_lock,
        ingress,
        token: str | None = None,
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

        # Lock one registry snapshot for the whole batch; build one parse
        # queue per distinct queue_group (always includes "native").
        parser_specs = parser_specs_snapshot()
        queue_groups = {spec.queue_group for spec in parser_specs.values()}
        parse_queues = {
            group: asyncio.Queue(maxsize=self.queue_size_parse)
            for group in queue_groups
        }

        ctx = _BatchRunContext(
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            semaphore=asyncio.Semaphore(self.max_parallel_insert),
            total_files=total_files,
            parse_queues=parse_queues,
            parser_specs=parser_specs,
            q_analyze=asyncio.Queue(maxsize=self.queue_size_analyze),
            q_process=asyncio.Queue(maxsize=self.queue_size_insert),
            pipeline_cancel_event=threading.Event(),
            # Pre-register the whole initial batch as inflight BEFORE the feeder
            # starts, so a document message that echoes an initial-batch doc is
            # deduplicated the moment the feeder runs.
            run_owner_token=token,
            inflight_doc_ids=set(to_process_docs.keys()),
        )
        # Publish the run context on the instance for the ONE storage write that
        # cannot receive it as an argument: ``_persist_parsed_full_docs`` is
        # reached as ``ctx.rag._persist_parsed_full_docs`` from inside a parser
        # (in-tree and third-party alike), so a parameter would only guard the
        # parsers that opt in. ``busy`` is a single per-workspace slot and this
        # instance is per-workspace, so at most one run is ever published here;
        # the previous value is restored rather than cleared so a nested/handed
        # off run cannot blank an outer one.
        previous_run_ctx = getattr(self, "_active_run_ctx", None)
        self._active_run_ctx = ctx

        async def _watch_pipeline_cancellation() -> None:
            """Bridge shared cancellation state into native parser threads.

            The status is guarded by an async lock and may be set by an API
            request in another task. Native extractors are synchronous and may
            be blocked in ``SyncLLMBridge``, so a thread-safe event is the
            handoff point. This watcher belongs to this batch and is always
            cancelled in the batch teardown below.
            """
            while True:
                if await self._cancellation_requested(
                    pipeline_status, pipeline_status_lock
                ):
                    ctx.pipeline_cancel_event.set()
                    return
                await asyncio.sleep(0.5)

        def _group_concurrency(group: str) -> int:
            # Built-in groups keep their existing LightRAG fields (env +
            # programmatic overrides preserved). Third-party groups use the
            # owner spec's ``concurrency`` (the registrant baked in any env
            # override at registration); an unowned group shares native's.
            field_name = f"max_parallel_parse_{group}"
            if hasattr(self, field_name):
                # A spec declaring ``concurrency`` on a built-in group is a
                # plugin-author misconfig: the pool is sized by the instance
                # field, so surface the ignored value instead of silently
                # dropping it.
                ignored = [
                    s.engine_name
                    for s in parser_specs.values()
                    if s.queue_group == group and s.concurrency is not None
                ]
                if ignored:
                    logger.warning(
                        "[parse] queue_group %r is built-in (sized by %s=%d); "
                        "spec-level concurrency from %s is ignored",
                        group,
                        field_name,
                        getattr(self, field_name),
                        ignored,
                    )
                return getattr(self, field_name)
            owners = [
                s
                for s in parser_specs.values()
                if s.queue_group == group and s.concurrency is not None
            ]
            if len(owners) > 1:
                raise ValueError(
                    f"queue_group {group!r} has multiple concurrency owners: "
                    f"{[s.engine_name for s in owners]}"
                )
            if owners:
                return owners[0].concurrency
            return self.max_parallel_parse_native

        # Resolve every group's worker count BEFORE spawning any task:
        # _group_concurrency can still raise (a queue_group with multiple
        # concurrency owners). Raising here — while zero workers exist — avoids
        # orphaning already-spawned workers outside the try/finally below (they
        # would block forever on an empty queue, never cancelled).
        group_worker_counts = {
            group: max(1, _group_concurrency(group)) for group in parse_queues
        }

        cancellation_watcher = asyncio.create_task(_watch_pipeline_cancellation())
        workers: list[asyncio.Task] = []
        for group, queue in parse_queues.items():
            for _ in range(group_worker_counts[group]):
                workers.append(
                    asyncio.create_task(self._parse_worker(group, queue, ctx))
                )
        for _ in range(max(1, self.max_parallel_analyze)):
            workers.append(asyncio.create_task(self._analyze_worker(ctx)))
        for _ in range(max(1, self.max_parallel_insert)):
            workers.append(asyncio.create_task(self._process_worker(ctx)))

        # In-batch feeder (Phase 2): ``ingress`` comes from the caller — the
        # run resolves it once against its workspace snapshot — so a batch can
        # never straddle two coordination namespaces; its cancellation is
        # guaranteed by the finally alongside the workers.
        feeder_task: asyncio.Task | None = None

        # The workers above are live asyncio tasks; their cancellation MUST be
        # guaranteed even if enqueuing or a queue join raises (e.g. an orchestrator-
        # level storage call fails during a backend outage). Without this try/finally
        # an escape here would orphan the workers — they keep draining the queues and
        # appending to history_messages while the caller's finally has already cleared
        # ``busy`` — leaving busy=False while processing visibly continues.
        try:
            # Add pending files to the correct parsing queue
            for current_file_number, (doc_id, status_doc) in enumerate(
                to_process_docs.items(), start=1
            ):
                file_path = getattr(status_doc, "file_path", "unknown_source")
                # Per-document isolation: the engine-routing get_by_id is the only
                # orchestrator-level storage read in this loop. A transient/corrupt
                # single-doc failure must FAIL just that document and continue with
                # the rest of the batch — not escape and abort the whole batch.
                # During a full outage _finalize_doc_failure's own doc_status write
                # also raises; that escape is caught by the finally below (workers
                # are cleanly cancelled) and the batch aborts as a whole.
                try:
                    content_data = await self.full_docs.get_by_id(doc_id) or {}
                except Exception as e:
                    await self._finalize_doc_failure(
                        ctx=ctx,
                        doc_id=doc_id,
                        status_doc=status_doc,
                        file_path=file_path,
                        error=e,
                        stage_label="parse",
                        current_file_number=current_file_number,
                        total_files=total_files,
                        failed_chunks_snapshot=([], 0),
                        pending_tasks=[],
                        metadata_extra={},
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                    # This pre-registered doc failed before routing — drop it
                    # from inflight so the feeder's dedup does not shadow a
                    # later re-enqueue of the same id.
                    async with pipeline_status_lock:
                        ctx.inflight_doc_ids.discard(doc_id)
                    continue
                # Select the concurrency pool by the engine's queue_group
                # (snapshot). The worker re-resolves the actual parser per-doc;
                # this only picks which queue/pool the doc waits in. Unknown
                # group -> native pool (defensive; never KeyError).
                key = resolve_stored_document_parser_engine(
                    file_path=file_path,
                    content_data=content_data,
                )
                spec = parser_specs.get(key)
                group = spec.queue_group if spec is not None else "native"
                queue = ctx.parse_queues.get(group, ctx.parse_queues["native"])
                await queue.put((doc_id, status_doc))

            # Start the in-batch feeder now that the initial batch is queued: it
            # routes documents that arrive DURING this batch straight into the
            # parse queues (the latency win), so the joins below wait on both
            # the initial batch and everything the feeder adds.
            feeder_task = asyncio.create_task(self._pipeline_feeder(ctx, ingress))

            await asyncio.gather(*(q.join() for q in ctx.parse_queues.values()))
            await ctx.q_analyze.join()
            await ctx.q_process.join()

            # Queues drained. Stop the feeder (cancel = cooperative stop; a
            # cancelled parse-queue put rolls back in _admit_fed_document), then
            # run one more join cascade to absorb anything it routed in the stop
            # window. With the feeder stopped no new items enter, so this second
            # cascade is stable. A document message still sitting in the mailbox
            # stays there and is resolved by the quiescence decision
            # (CONTINUE_DOCUMENT) or the next batch's feeder.
            feeder_task.cancel()
            await asyncio.gather(feeder_task, return_exceptions=True)
            feeder_task = None
            await asyncio.gather(*(q.join() for q in ctx.parse_queues.values()))
            await ctx.q_analyze.join()
            await ctx.q_process.join()
        finally:
            # Wake native parser bridges before cancelling worker tasks. Their
            # underlying worker thread then stops waiting for an in-flight LLM
            # response even if task cancellation reaches run_in_executor first.
            ctx.pipeline_cancel_event.set()
            cancellation_watcher.cancel()
            if feeder_task is not None:
                feeder_task.cancel()
            for w in workers:
                w.cancel()
            teardown_tasks: list[asyncio.Task] = [*workers, cancellation_watcher]
            if feeder_task is not None:
                teardown_tasks.append(feeder_task)
            await asyncio.gather(*teardown_tasks, return_exceptions=True)
            # Every worker (hence every parser) is stopped and joined here, so
            # no further ``_persist_parsed_full_docs`` call belongs to this run.
            self._active_run_ctx = previous_run_ctx

        # If the batch aborted on an internal storage error, the shared
        # cross-file flush buffers may still hold records from the documents
        # that were marked FAILED. Discard them now (workers are stopped, so
        # this does not race a flush) so they are neither re-flushed nor
        # carried into the next batch — every affected document is reprocessed
        # on retry. See _discard_pending_index_ops / drop_pending_index_ops.
        async with pipeline_status_lock:
            status_snapshot = pipeline_status.copy()
            internal_abort = (
                status_snapshot.get("cancellation_requested", False)
                and status_snapshot.get("cancellation_reason") == "internal_error"
            )
        if internal_abort:
            await self._discard_pending_index_ops()

    def _build_pending_reset_update(
        self,
        status_doc: DocProcessingStatus,
        content_data: dict | None,
    ) -> tuple[dict, str, dict]:
        """Build the doc_status upsert that returns one interrupted/FAILED doc
        to a clean PENDING state, preserving the immutable ``created_at``.

        Shared by the batch consistency validator and the manual EXCLUSIVE_RESET
        (LR2 §7.3) so both scrub stale per-attempt metadata identically. Returns
        ``(update_dict, resolved_file_path, reset_metadata)`` — the latter two so
        the caller can also mirror them onto the in-memory ``status_doc`` when it
        carries the doc forward into workers (the reset path does not).
        """
        preserved_chunks_list, preserved_chunks_count = chunk_fields_from_status_doc(
            status_doc
        )
        resolved_file_path = resolve_doc_file_path(
            status_doc=status_doc,
            content_data=content_data,
        )
        # Directives-only metadata: drop per-attempt timing/result fields, keep
        # process_options / source_file (legacy source_file_name tolerant).
        reset_metadata = doc_status_reset_metadata(status_doc)
        update = {
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
            "error_msg": "",
            "metadata": reset_metadata,
        }
        return update, resolved_file_path, reset_metadata

    async def _validate_and_fix_document_consistency(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
        pipeline_status: dict,
        pipeline_status_lock: asyncio.Lock,
    ) -> dict[str, DocProcessingStatus]:
        """Validate and fix document data consistency by deleting inconsistent
        entries, but preserve failed documents.

        The ``full_docs`` probe is STRICT where the backend supports it, because
        a miss here is acted on DESTRUCTIVELY: an active row whose content is
        absent has its ``doc_status`` row deleted. A plain ``get_by_id`` may
        report a transport failure — or, on OpenSearch, an index that is merely
        not ready or transiently missing — as a best-effort ``None``, and this
        function would then delete every live scheduling row it swept during a
        storage blip. LR2 §5.4/§7.2 require the opposite: only a CONFIRMED
        absence may drive the deletion, and a backend that cannot confirm keeps
        the row and reports a bounded warning. Same rule, same helper shape as
        ``_reset_failed_page`` and scan's ``_confirm_full_docs_absent``.
        """
        # Documents carrying a custom-chunk patch journal belong to an
        # in-flight or failed ainsert_custom_chunks operation (issue #3400
        # Phase 3). Ordinary pipeline processing must not touch them: a reset
        # would strip the journal and rebuild the whole document, discarding
        # the operation's recovery anchor. They are resumed by the SDK caller
        # (same call) or rolled back by /documents/scan.
        #
        # DEFENSIVE since the journal filter moved into ``_next_scheduling_page``
        # (a journaled row popped here but left in an AUTO status made the manual
        # drain's confirmation sweep spin forever). Kept because this function is
        # also reachable with a caller-supplied map.
        journaled_patch_docs = [
            doc_id
            for doc_id, status_doc in to_process_docs.items()
            if doc_status_custom_chunk_patch(status_doc) is not None
        ]
        if journaled_patch_docs:
            for doc_id in journaled_patch_docs:
                to_process_docs.pop(doc_id, None)
            async with pipeline_status_lock:
                skip_message = (
                    f"Skipping {len(journaled_patch_docs)} document(s) with an "
                    "unfinished custom-chunk operation (retry the operation or "
                    "run a scan to roll it back)"
                )
                logger.info(skip_message)
                pipeline_status["latest_message"] = skip_message
                append_pipeline_history(pipeline_status, skip_message)

        inconsistent_docs = []
        failed_docs_to_preserve = []
        unconfirmed_docs = []
        successful_deletions = 0

        # One read per document, reused by the reset loop below: the content is
        # both the consistency verdict AND an input to the PENDING reset update,
        # and reading it twice let the two disagree (and doubled the round trips).
        content_by_doc: dict[str, dict | None] = {}
        strict_reads = getattr(self.full_docs, "supports_strict_point_reads", False)

        # Check each document's data consistency
        for doc_id, status_doc in to_process_docs.items():
            # Check if corresponding content exists in full_docs. Strict where
            # supported so "absent" is CONFIRMED absent before anything is
            # deleted (see this method's docstring).
            if strict_reads:
                content_data = await self.full_docs.get_by_id_strict(doc_id)
            else:
                content_data = await self.full_docs.get_by_id(doc_id)
            content_by_doc[doc_id] = content_data
            if not content_data:
                # Check if this is a failed document that should be preserved
                if (
                    hasattr(status_doc, "status")
                    and status_doc.status == DocStatus.FAILED
                ):
                    failed_docs_to_preserve.append(doc_id)
                elif strict_reads:
                    inconsistent_docs.append(doc_id)
                else:
                    # Cannot tell "content really gone" from "the read failed":
                    # keep the row, keep it out of this batch, and say so.
                    unconfirmed_docs.append(doc_id)

        if unconfirmed_docs:
            warning = (
                f"Keeping {len(unconfirmed_docs)} document(s) with no readable "
                f"content: {type(self.full_docs).__name__} cannot confirm whether "
                "the content is really absent (no strict point reads), so a "
                "storage failure is indistinguishable from an inconsistent entry"
            )
            logger.warning(warning)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = warning
                append_pipeline_history(pipeline_status, warning)
            for doc_id in unconfirmed_docs:
                to_process_docs.pop(doc_id, None)

        # Log information about failed documents that will be preserved
        if failed_docs_to_preserve:
            async with pipeline_status_lock:
                preserve_message = f"Preserving {len(failed_docs_to_preserve)} failed document entries for manual review"
                logger.info(preserve_message)
                pipeline_status["latest_message"] = preserve_message
                append_pipeline_history(pipeline_status, preserve_message)

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
                append_pipeline_history(pipeline_status, summary_message)

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
                        append_pipeline_history(pipeline_status, log_message)

                    # Remove from processing list
                    to_process_docs.pop(doc_id, None)

                except Exception as e:
                    # Log deletion failure
                    async with pipeline_status_lock:
                        error_message = f"Failed to delete entry: {doc_id} - {str(e)}"
                        logger.error(error_message)
                        pipeline_status["latest_message"] = error_message
                        append_pipeline_history(pipeline_status, error_message)

        # Final summary log
        # async with pipeline_status_lock:
        #     final_message = f"Successfully deleted {successful_deletions} inconsistent entries, preserved {len(failed_docs_to_preserve)} failed documents"
        #     logger.info(final_message)
        #     pipeline_status["latest_message"] = final_message
        #     pipeline_status["history_messages"].append(final_message)

        # Bring every to-be-processed document into a clean PENDING state.
        # Two cases are handled here so stale per-attempt metadata never
        # survives into the PENDING wait window (where the WebUI would render
        # last attempt's parse/analyze timings):
        #   * interrupted docs (PROCESSING/PARSING/ANALYZING/FAILED) are reset
        #     to PENDING, clearing error_msg and resetting metadata to the
        #     enqueue-time directives only;
        #   * docs that are ALREADY PENDING but still carry per-attempt fields
        #     (e.g. reset by an older build that preserved them) are normalised
        #     in place to directives-only.
        # In BOTH cases the cleaned metadata is mirrored back onto the in-memory
        # ``status_doc`` so the downstream parse worker — which no longer scrubs
        # stale keys itself — carries the clean dict forward through
        # ``doc_status_transition_metadata`` at every later transition.
        docs_to_reset = {}
        reset_count = 0
        normalized_count = 0

        for doc_id, status_doc in to_process_docs.items():
            # Content from the single consistency read above — no second probe,
            # which could otherwise disagree with the verdict already acted on.
            content_data = content_by_doc.get(doc_id)
            if not content_data:  # Fails consistency check; handled above
                continue
            status = getattr(status_doc, "status", None)
            # FAILED is DEFENSIVE here: post-Phase-3 the AUTO sweep that feeds
            # this validator never carries FAILED (those are reset only by the
            # manual EXCLUSIVE_RESET), so this branch handles orphan
            # PROCESSING/PARSING/ANALYZING in practice.
            is_interrupted = status in (
                DocStatus.PROCESSING,
                DocStatus.FAILED,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
            )
            # Only normalise an already-PENDING doc when it actually carries a
            # stale per-attempt field — a precise trigger so unrelated/custom
            # metadata on a clean PENDING is never rewritten or dropped.
            needs_pending_normalize = (
                status == DocStatus.PENDING
                and doc_status_metadata_has_attempt_fields(status_doc)
            )
            if not (is_interrupted or needs_pending_normalize):
                continue

            reset_update, resolved_file_path, reset_metadata = (
                self._build_pending_reset_update(status_doc, content_data)
            )
            docs_to_reset[doc_id] = reset_update

            # Mirror onto the in-memory status_doc so workers carry it forward.
            status_doc.status = DocStatus.PENDING
            status_doc.file_path = resolved_file_path
            status_doc.metadata = reset_metadata
            if is_interrupted:
                reset_count += 1
            else:
                normalized_count += 1

        # Update doc_status storage if there are documents to reset
        if docs_to_reset:
            await self.doc_status.upsert(docs_to_reset)

            async with pipeline_status_lock:
                reset_message = (
                    f"Reset {reset_count} interrupted document(s) from "
                    "PARSING/ANALYZING/PROCESSING to PENDING status"
                    + (
                        f"; normalized {normalized_count} PENDING document(s) "
                        "with stale metadata"
                        if normalized_count
                        else ""
                    )
                )
                logger.info(reset_message)
                pipeline_status["latest_message"] = reset_message
                append_pipeline_history(pipeline_status, reset_message)

        return to_process_docs

    # ============================================================
    # Manual retry protocol: freeze → DRAIN_TO_IDLE → EXCLUSIVE_RESET
    # (LR2 §6.1/§7). The busy processing run drives all three phases; the
    # freeze/phase/owner state is owner-checked against THIS run's busy token,
    # so a dead owner's reaper reclaim (which shares the busy pid) clears the
    # whole manual state and the sticky request is re-run from scratch.
    # ============================================================

    async def _begin_manual_drain(
        self,
        request_id: str,
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """Enter DRAIN_TO_IDLE for a manual retry (owner-checked, atomic).

        Sets ``manual_freeze_requested=True`` (rejects new enqueue reservations),
        ``manual_phase=DRAIN_TO_IDLE`` and ``manual_owner`` in ONE update, only
        while we still own ``busy`` — never a True freeze flag with no owner
        (LR2 §6.1). Returns False if we no longer own the slot."""
        owner = make_manual_owner_record(request_id, token)

        def _apply(status):
            status.update(
                {
                    "manual_freeze_requested": True,
                    "manual_freeze_started_at": time.time(),
                    "manual_resetting": False,
                    "manual_phase": MANUAL_PHASE_DRAIN_TO_IDLE,
                    "manual_owner": owner,
                }
            )
            return True

        return bool(
            await with_reservation_lock(
                pipeline_status,
                pipeline_status_lock,
                owner_key="busy_owner",
                token=token,
                action=_apply,
            )
        )

    async def _end_manual_drain(
        self,
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Clear the manual freeze/reset state back to idle (owner-checked).

        Called after the ACK at the PROCESS transition, and defensively in the
        run's finally: a live-but-exiting owner must never leave a freeze that
        wedges uploads forever. Owner-checked so a handed-off/new owner's state
        is never clobbered; the sticky request (if not yet ACKed) survives in
        the mailbox and is re-run."""

        def _apply(status):
            status.update(
                {
                    "manual_freeze_requested": False,
                    "manual_freeze_started_at": None,
                    "manual_resetting": False,
                    "manual_phase": MANUAL_PHASE_IDLE,
                    "manual_owner": None,
                }
            )
            return True

        await with_reservation_lock(
            pipeline_status,
            pipeline_status_lock,
            owner_key="busy_owner",
            token=token,
            action=_apply,
        )

    async def _still_run_owner(self, ctx: "_BatchRunContext") -> bool:
        """True iff this batch run still owns ``busy`` (LR2 §7.7 items 3/4/7).

        Guards every worker status write. ``run_owner_token is None`` means the
        context was built outside a reservation (tests) and skips the check.

        This is an in-memory owner check, deliberately NOT an expiring lease:
        LR2 §7.7 keeps stale writers out by never revoking a live owner, and
        notes that forcing takeover from a process that is alive but unreachable
        would require a storage-verifiable fencing token, which the no-new-field
        design does not have. So this rejects writes from a run whose ownership
        was reclaimed after its process was CONFIRMED dead (or after a Manager
        generation change) — it is not a partition-safe fence.
        """
        if ctx.run_owner_token is None:
            return True
        return await self._still_freeze_owner(
            ctx.run_owner_token, ctx.pipeline_status, ctx.pipeline_status_lock
        )

    async def _still_freeze_owner(
        self,
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """True iff THIS run still owns ``busy`` (and therefore the freeze).

        The manual protocol runs single-threaded in the busy owner's task, so
        the owner can only change via the dead-process reaper — impossible while
        this task is alive. This check is defense-in-depth (LR2 §7.7 item 8):
        every FAILED reset page verifies ownership before it writes."""

        async def _run() -> bool:
            async with pipeline_status_lock:
                return _reservation_owner_token(pipeline_status.get("busy_owner")) == (
                    token
                )

        return await run_to_completion(_run)

    async def _next_failed_page(
        self, position: CursorPosition
    ) -> tuple[dict[str, DocProcessingStatus], CursorPosition]:
        """Fetch ONE bounded page of FAILED docs, hydrated to full status.

        The manual EXCLUSIVE_RESET pages FAILED so a huge backlog (LR2 §13.2
        case 1: 10,000 FAILED) never materialises at once. ``page_size <= 0``
        collapses to the legacy single strict scan (paging disabled). Ordering
        is the immutable ``(created_at, id)`` keyset. A strict page/hydration
        error propagates so the caller neither advances the cursor nor drops a
        FAILED row (it stays FAILED for the next run)."""
        page_size = getattr(self, "pipeline_scheduling_page_size", 0)
        if page_size <= 0:
            docs = await self.doc_status.get_docs_by_statuses(
                [DocStatus.FAILED], strict=True
            )
            return docs, CURSOR_END

        page = await self.doc_status.get_docs_by_statuses_page(
            [DocStatus.FAILED],
            limit=page_size,
            position=position,
            strict=True,
        )
        if not page.docs:
            return {}, page.next_position
        hydrated = await self.doc_status.get_full_docs_by_ids(
            list(page.docs.keys()), strict=True
        )
        # Keep only rows still FAILED (a row raced out between page and
        # hydration is no longer this reset's concern). str-enum equality lets
        # a raw string or a DocStatus member both match.
        failed = {
            doc_id: doc
            for doc_id, doc in hydrated.items()
            if getattr(doc, "status", None) in {DocStatus.FAILED.value}
        }
        return failed, page.next_position

    async def _reset_failed_page(
        self,
        docs: dict[str, DocProcessingStatus],
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> int:
        """Reset one page of FAILED docs to PENDING (LR2 §7.3 per-FAILED rules).

        * journaled custom-chunk patch → skipped (an in-flight SDK operation
          owns it; a reset would strip its recovery anchor);
        * FAILED without ``full_docs`` content → skipped, left FAILED (an
          unprocessable stub; a reset would immediately re-fail — matches the
          batch validator's "preserve failed for manual review");
        * otherwise → reset to PENDING preserving the immutable ``created_at``.

        The ``full_docs`` probe is STRICT where the backend supports it: a plain
        ``get_by_id`` may report a transport failure (or, on OpenSearch, an index
        that is merely not ready) as a best-effort ``None``, which this loop
        would read as "unprocessable stub" and skip. Every doc would then be
        skipped, the sweep would still page to End, and the manual retry would be
        ACKed having reset nothing — the user's "retry my failed documents"
        silently doing nothing. A strict read raises instead, so the caller keeps
        the request sticky and does not advance the cursor. Backends without the
        capability keep the conservative skip but are COUNTED, so the operator
        sees the reset was not complete instead of inferring success from silence.

        The page upsert is owner-checked: ownership is verified immediately
        before the write, so a reaper reclaim (dead process) aborts the reset
        instead of a stale task writing doc_status. Returns the reset count.
        Raises on a storage read/write error so the caller keeps the request
        sticky and does not advance the cursor."""
        docs_to_reset: dict[str, dict] = {}
        unconfirmed = 0
        strict_reads = getattr(self.full_docs, "supports_strict_point_reads", False)
        for doc_id, status_doc in docs.items():
            if doc_status_custom_chunk_patch(status_doc) is not None:
                continue
            if strict_reads:
                content_data = await self.full_docs.get_by_id_strict(doc_id)
            else:
                content_data = await self.full_docs.get_by_id(doc_id)
                if not content_data:
                    unconfirmed += 1
            if not content_data:
                continue
            update, _, _ = self._build_pending_reset_update(status_doc, content_data)
            docs_to_reset[doc_id] = update
        if unconfirmed:
            warning = (
                f"Manual retry: {unconfirmed} FAILED document(s) left untouched — "
                f"{type(self.full_docs).__name__} cannot confirm whether their "
                f"content is really absent (no strict point reads), so a storage "
                f"failure is indistinguishable from an unprocessable stub"
            )
            logger.warning(warning)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = warning
                append_pipeline_history(pipeline_status, warning)
        if not docs_to_reset:
            return 0
        # Owner-checked page write (LR2 §7.7 item 8): confirm we still hold the
        # freeze before persisting, so a dead-owner reclaim cannot be overwritten
        # by this stale task.
        if not await self._still_freeze_owner(
            token, pipeline_status, pipeline_status_lock
        ):
            raise RuntimeError(
                "manual reset lost freeze ownership before a FAILED→PENDING page "
                "write; aborting so a new owner re-runs the reset from Start"
            )
        await self.doc_status.upsert(docs_to_reset)
        return len(docs_to_reset)

    async def _run_exclusive_failed_reset(
        self,
        request_id: str,
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """Phase two (LR2 §7.3): page FAILED→PENDING with NO worker running.

        Precondition (guaranteed by the caller): DRAIN_TO_IDLE reached strict
        idle — pending_enqueues/inflight/routing are 0 and no AUTO record
        remains — so this reset runs without any failed producer. Transitions
        ``manual_phase`` to EXCLUSIVE_RESET (``manual_resetting=True``,
        owner-checked), then pages FAILED from Start to End, resetting each page.

        Returns True when the reset reached End (caller ACKs); False if it lost
        ownership. A storage error propagates: the request stays sticky, the
        cursor does not advance, and the next run re-runs the reset from Start
        (already-reset rows are PENDING and no longer appear on a FAILED page,
        so no double reset — LR2 §7.3 "为什么不需要 generation/checkpoint")."""

        def _enter(status):
            # The freeze timestamp is wall clock because the freeze may have been
            # taken by another process; this is the DRAIN_TO_IDLE window — the one
            # an operator feels, since uploads are refused throughout it.
            froze_at = status.get("manual_freeze_started_at")
            if isinstance(froze_at, (int, float)) and froze_at > 0:
                pipeline_metrics.observe(
                    pipeline_metrics.MANUAL_DRAIN_SECONDS, time.time() - float(froze_at)
                )
            status.update(
                {
                    "manual_resetting": True,
                    "manual_phase": MANUAL_PHASE_EXCLUSIVE_RESET,
                }
            )
            return True

        if not await with_reservation_lock(
            pipeline_status,
            pipeline_status_lock,
            owner_key="busy_owner",
            token=token,
            action=_enter,
        ):
            return False

        total_reset = 0
        pages = 0
        reset_started = time.monotonic()
        cursor: CursorPosition = CURSOR_START
        while True:
            if not await self._still_freeze_owner(
                token, pipeline_status, pipeline_status_lock
            ):
                return False
            docs, cursor = await self._next_failed_page(cursor)
            pages += 1
            if docs:
                total_reset += await self._reset_failed_page(
                    docs, token, pipeline_status, pipeline_status_lock
                )
            if cursor is CURSOR_END:
                break

        # The reset runs with no worker, so its duration is exactly how long the
        # pipeline was held idle on purpose (LR2 Phase 6 item 3).
        pipeline_metrics.observe(
            pipeline_metrics.MANUAL_RESET_SECONDS, time.monotonic() - reset_started
        )
        pipeline_metrics.increment(pipeline_metrics.MANUAL_RESET_PAGES, pages)
        pipeline_metrics.increment(pipeline_metrics.MANUAL_RESET_DOCS, total_reset)

        reset_message = (
            f"Manual retry {request_id[:8]}: reset {total_reset} FAILED "
            "document(s) to PENDING (exclusive phase, no worker running)"
        )
        logger.info(reset_message)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = reset_message
            append_pipeline_history(pipeline_status, reset_message)
        return True

    async def _confirm_scan_drain_idle(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """DRAIN_TO_IDLE confirmation for a scan-driven reset (LR2 §8.1/§7.2).

        A scan reservation is only granted while ``busy`` is False, no other
        scan/destructive job holds the pipeline and ``pending_enqueues == 0``; the
        scan fence plus the manual freeze then refuse every new enqueue
        reservation and processing run. So the drain is already complete when this
        runs, and ``inflight``/``routing`` are structurally zero: this reset owns
        ``busy`` and starts no worker.

        The bounded re-check below is defense-in-depth for the one window that can
        still show a non-zero count — a reservation taken between the scan's
        acquire and the freeze — and it reaps confirmed-dead tokens first so a
        worker SIGKILLed mid-reserve cannot wedge it. Returns False when the count
        does not reach zero within the bound: the caller then abandons the reset
        (the sticky request survives for the standard drain path) rather than
        resetting FAILED with a live producer around."""
        for _ in range(_SCAN_DRAIN_CONFIRM_ATTEMPTS):
            async with pipeline_status_lock:
                pending = pipeline_status.get("pending_enqueues", 0)
            if not pending:
                return True
            await reap_dead_reservations_locked(pipeline_status, pipeline_status_lock)
            await asyncio.sleep(_MANUAL_DRAIN_POLL_SECONDS)
        async with pipeline_status_lock:
            pending = pipeline_status.get("pending_enqueues", 0)
        if not pending:
            return True
        logger.warning(
            f"Scan FAILED reset abandoned: {pending} enqueue reservation(s) still "
            "in flight after the drain wait; the manual request stays sticky for "
            "the standard drain path"
        )
        return False

    async def apipeline_reset_failed_for_scan(
        self,
        request_id: str,
        *,
        scan_owner_token: str | None = None,
    ) -> bool:
        """Run a scan's manual intent through the SHARED exclusive reset (LR2 §8.1).

        ``/scan`` is a composite operation: it retries the FAILED documents that
        existed when it started AND discovers new files. The order is fixed —
        reset FIRST, discover after — because enqueueing new files first would let
        the scan's OWN new failures be absorbed by its own manual request.

        Called by ``run_scanning_process`` while it holds
        ``scanning``/``scanning_exclusive``, so:

        * ``busy`` is taken with ``scan_reset_owner_token`` (the scan owns the
          fence that would otherwise refuse it) — the reset needs the slot to
          reuse the owner-checked manual helpers;
        * DRAIN_TO_IDLE is confirmed rather than performed
          (:meth:`_confirm_scan_drain_idle`) and the AUTO backlog is deliberately
          NOT processed here: with no worker running (this call starts none) the
          PENDING rows are inert, so they are not failed producers, and processing
          them under the exclusive fence would hold uploads at 409 for the whole
          backlog. They are processed after classification, by the scan's normal
          drive (LR2 §8.1 "释放 scanning_exclusive → 处理 PENDING");
        * the reset itself is :meth:`_run_exclusive_failed_reset` — the very helper
          ``/reprocess_failed`` uses, not a second retry algorithm.

        Returns True iff the reset reached End and the request was ACKed (the
        caller may then discover files). False — slot refused, ownership lost,
        drain not idle — leaves the request sticky and un-ACKed: the caller MUST
        skip discovery so no new file can be enqueued ahead of the reset, and the
        standard drain path (any later drive) serves the request instead. A
        storage error propagates for the same reason.
        """
        run_workspace = self.workspace
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=run_workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=run_workspace
        )
        ingress = await get_pipeline_ingress(run_workspace)

        token = uuid.uuid4().hex
        holds_busy = False
        try:
            reservation = await acquire_processing_reservation(
                pipeline_status,
                pipeline_status_lock,
                token=token,
                already_held=False,
                pipeline_ingress=ingress,
                scan_reset_owner_token=scan_owner_token,
                flags={
                    "job_name": "Scan: reset failed documents",
                    "job_start": datetime.now(timezone.utc).isoformat(),
                    "docs": 0,
                    "batchs": 0,
                    "cur_batch": 0,
                    "cancellation_requested": False,
                    "cancellation_reason": None,
                    "cancellation_detail": None,
                    "latest_message": "",
                },
            )
            if not reservation.acquired:
                holds_busy = False
                logger.warning(
                    "Scan FAILED reset skipped: "
                    f"{reservation.message or 'pipeline reservation unavailable'}"
                )
                return False
            holds_busy = True

            # Freeze new ingress + claim the manual owner record for THIS request,
            # tied to the busy token we just took (every reset page re-verifies it).
            if not await self._begin_manual_drain(
                request_id, token, pipeline_status, pipeline_status_lock
            ):
                return False
            if not await self._confirm_scan_drain_idle(
                pipeline_status, pipeline_status_lock
            ):
                return False
            if not await self._run_exclusive_failed_reset(
                request_id, token, pipeline_status, pipeline_status_lock
            ):
                return False
            # ACK only after every FAILED→PENDING write persisted (LR2 §7.3
            # completion point): a crash before this re-runs the reset from Start.
            ingress.ack_manual_retry(request_id)
            return True
        finally:
            if holds_busy:
                # Cancellation-resistant, owner-checked release: busy AND the
                # whole manual state go in ONE update so an abnormal exit can
                # never leave a freeze wedging uploads with no owner (LR2 §6.1).
                # The scan keeps ``scanning``/``scanning_exclusive`` — its
                # classification phase runs next, without ``busy``.
                async def _release():
                    def _apply(status):
                        status.update(
                            {
                                "busy": False,
                                "busy_owner": None,
                                "manual_freeze_requested": False,
                                "manual_freeze_started_at": None,
                                "manual_resetting": False,
                                "manual_phase": MANUAL_PHASE_IDLE,
                                "manual_owner": None,
                            }
                        )
                        return True

                    await with_reservation_lock(
                        pipeline_status,
                        pipeline_status_lock,
                        owner_key="busy_owner",
                        token=token,
                        action=_apply,
                    )

                await run_to_completion(_release)

    async def _next_scheduling_page(
        self,
        statuses,
        position: CursorPosition,
    ) -> tuple[dict[str, DocProcessingStatus], CursorPosition]:
        """Fetch ONE bounded scheduling page and hydrate it to full status.

        Returns ``(to_process_docs, next_position)`` — the full
        :class:`DocProcessingStatus` map for the page, plus the cursor to pass
        back for the next page (``next_position is CURSOR_END`` ends the
        sweep). Memory stays O(page_size): the lightweight keyset page only
        carries ids/status, and the survivors are hydrated to full rows via
        ``get_full_docs_by_ids`` right before routing.

        ``PIPELINE_SCHEDULING_PAGE_SIZE=0`` collapses to the LEGACY single-scan
        (one page terminating at ``CURSOR_END``), behaving byte-for-byte as
        before Phase 2. Every caller passes ``_AUTO_RESUME_DOC_STATUSES`` (no
        FAILED — those are reset only by the manual EXCLUSIVE_RESET's own
        ``_next_failed_page``), so this sweep is always a plain AUTO drain.

        The strict page + strict hydration are a complete-or-raise pair: any
        backend error propagates so the caller neither advances the cursor nor
        silently drops docs (they stay PENDING for the next sweep).

        Either path drops rows carrying an unfinished custom-chunk journal — they
        are not routable work; see the comment on the filter below.
        """
        page_size = getattr(self, "pipeline_scheduling_page_size", 0)
        if page_size <= 0:
            docs = await self.doc_status.get_docs_by_statuses(
                list(statuses), strict=True
            )
            return {
                doc_id: doc
                for doc_id, doc in docs.items()
                if doc_status_custom_chunk_patch(doc) is None
            }, CURSOR_END

        routable_ids, _journaled_ids, next_position = await self._fetch_scheduling_page(
            statuses, position, limit=page_size
        )
        if not routable_ids:
            # A fully-filtered page is not termination — the cursor still
            # advances (or is CURSOR_END); return the empty map and let the
            # caller loop on next_position.
            return {}, next_position

        return await self._hydrate_scheduling_page(
            routable_ids, statuses
        ), next_position

    async def _fetch_scheduling_page(
        self,
        statuses,
        position: CursorPosition,
        *,
        limit: int,
    ) -> tuple[list[str], list[str], CursorPosition]:
        """One strict keyset page, split into routable and journaled doc ids.

        Rows carrying an unfinished custom-chunk journal are NOT routable work:
        they belong to an in-flight (or crash-interrupted) ``ainsert_custom_chunks``
        operation, and only the SDK caller or ``/documents/scan``'s rollback may
        advance them (see ``DocSchedulingRecord.has_custom_chunk_journal``, and the
        identical drop in ``_route_one``). Splitting them out HERE — from the
        lightweight page, before hydration — keeps a routing sweep from carrying a
        row it must never route, while still REPORTING them, which is what the
        manual drain's idle proof needs (see :meth:`_confirm_auto_drained`): the
        two callers ask different questions of the same page and a filter that
        only dropped rows could answer just one of them.
        """
        page_started = time.monotonic()
        page = await self.doc_status.get_docs_by_statuses_page(
            list(statuses),
            limit=limit,
            position=position,
            strict=True,
        )
        # Timed around the keyset query alone (not the hydration): a slow page
        # here means the backend's (created_at, id) index, and that is the one
        # thing the bounded sweep cannot work around.
        pipeline_metrics.observe(
            pipeline_metrics.SCHEDULING_PAGE_SECONDS, time.monotonic() - page_started
        )
        routable_ids: list[str] = []
        journaled_ids: list[str] = []
        for doc_id, record in page.docs.items():
            if record.has_custom_chunk_journal:
                journaled_ids.append(doc_id)
            else:
                routable_ids.append(doc_id)
        return routable_ids, journaled_ids, page.next_position

    async def _hydrate_scheduling_page(
        self,
        doc_ids: list[str],
        statuses,
    ) -> dict[str, DocProcessingStatus]:
        """Hydrate one page's routable ids to full rows, still in ``statuses``."""
        hydrated = await self.doc_status.get_full_docs_by_ids(doc_ids, strict=True)
        # Keep only rows still in the requested statuses (matches the legacy
        # get_docs_by_statuses view): a row raced out of the set between the
        # page and its hydration is no longer routable. str-enum equality lets
        # a raw string or a DocStatus member both match.
        status_values = {s.value for s in statuses}
        return {
            doc_id: doc
            for doc_id, doc in hydrated.items()
            if getattr(doc, "status", None) in status_values
        }

    async def _confirm_auto_drained(
        self,
    ) -> tuple[dict[str, DocProcessingStatus], CursorPosition, tuple[str, ...]]:
        """Is the PERSISTENT AUTO set drained? (LR2 §4.2 / §7.2)

        The manual drain's idle proof, and deliberately NOT
        :meth:`_next_scheduling_page`, which answers a different question — "what
        may this run route?" — and drops rows it must never route. Reusing that
        filtered answer as the idle proof is fail-open in a way the spin it
        replaced was not: the exclusive ``FAILED → PENDING`` reset would start,
        and ACK, while the persistent AUTO set is NOT empty. §4.2 requires a
        strict confirmation that the AUTO set is empty, and §7.2 requires an
        active row that cannot be advanced to be REPORTED as a blocker rather
        than silently ignored so the manual retry can start.

        It also pages past fully-filtered pages instead of reading one as
        termination. An empty ``docs`` with a live cursor means "call again" (the
        page contract is explicit that ``CURSOR_END`` is the only termination
        signal), so stopping at the first empty page could start the reset with
        routable AUTO rows still ahead of the cursor — reachable whenever page one
        happens to be all journaled rows, and before that whenever a page was
        fully consumed by the status re-filter.

        Returns ``(routable, cursor, blocking_ids)``:

        * ``routable`` non-empty → work remains; the caller keeps draining, and
          ``cursor`` is that page's continuation.
        * ``routable`` empty and ``blocking_ids`` non-empty → the AUTO set is not
          empty, but nothing left in it can be routed by this run: a drain
          blocker. ``blocking_ids`` is a BOUNDED sample.
        * both empty → genuinely drained; the reset may start.
        """
        page_size = getattr(self, "pipeline_scheduling_page_size", 0)
        if page_size <= 0:
            # Legacy single scan (paging disabled).
            docs = await self.doc_status.get_docs_by_statuses(
                list(_AUTO_RESUME_DOC_STATUSES), strict=True
            )
            routable = {
                doc_id: doc
                for doc_id, doc in docs.items()
                if doc_status_custom_chunk_patch(doc) is None
            }
            if routable:
                return routable, CURSOR_END, ()
            blocked = sorted(doc_id for doc_id in docs if doc_id not in routable)
            return {}, CURSOR_END, tuple(blocked[:_MANUAL_DRAIN_BLOCKER_SAMPLE])

        blocking: list[str] = []
        position: CursorPosition = CURSOR_START
        while True:
            (
                routable_ids,
                journaled_ids,
                position,
            ) = await self._fetch_scheduling_page(
                _AUTO_RESUME_DOC_STATUSES, position, limit=page_size
            )
            if routable_ids:
                hydrated = await self._hydrate_scheduling_page(
                    routable_ids, _AUTO_RESUME_DOC_STATUSES
                )
                if hydrated:
                    # Routable work found: return immediately so the drain keeps
                    # its current latency and O(page) memory. Rows behind this
                    # page are simply not reached yet.
                    return hydrated, position, ()
                # Every routable id raced out of the AUTO set between the page and
                # its hydration — not a blocker, just a page that emptied itself.
            for doc_id in journaled_ids:
                if len(blocking) < _MANUAL_DRAIN_BLOCKER_SAMPLE:
                    blocking.append(doc_id)
            if position is CURSOR_END:
                # Only CURSOR_END proves the sweep saw the whole AUTO set.
                return {}, CURSOR_END, tuple(blocking)

    async def _decide_pipeline_next_step(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
        ingress,
        sweep_cursor: CursorPosition = CURSOR_END,
    ) -> PipelineNextDecision:
        """Atomic quiescence decision: continue for a pending wake-up signal
        or release ``busy`` in the same critical section.

        Closes the loop-exit handoff race: a publisher that lands a signal
        (sticky manual request, auto-rescan flag, or a document message)
        while the processing loop is on its way out is observed in the same
        critical section that would release ``busy``, so the loop refetches
        instead of stranding the new work.

        Fixed priority (LR2 §6.2): cancellation → advancement of the CURRENT
        manual drain/reset → earliest (new) manual request → auto-rescan →
        document mailbox → release. The lock covers only this decision (mailbox
        calls are single in-memory/Manager RPCs; storage queries happen strictly
        outside, in :meth:`_refetch_for_decision`).

        The decision is ``manual_phase``-aware (read from ``pipeline_status``
        under this same lock):

        * **DRAIN_TO_IDLE** — this run already holds the freeze and is draining
          for a manual retry. Advancing THAT drain is top priority after cancel:
          finish the AUTO sweep pages, consume auto-rescan / document backlog,
          then wait for pre-freeze in-flight enqueues (CONTINUE_DRAIN_WAIT), and
          once strictly idle enter the exclusive reset (BEGIN_EXCLUSIVE_RESET).
          A second manual is NOT peeked here — it waits FIFO until this one ACKs
          and the phase returns to IDLE (LR2 §7.5).
        * **IDLE** — normal quiescence:
          0. **Cancellation** (consumes NOTHING): the cancel endpoint sets the
             flag under this same lock, so a cancel never lands between its
             check and a consuming step below.
          1. **Manual retry** (peeked, NOT consumed): earliest sticky request;
             BEGINS a drain (CONTINUE_MANUAL sets the freeze). It preempts an
             in-progress AUTO sweep (the drain re-scans AUTO from Start, so no
             doc is lost). Stays sticky, ACKed only after the exclusive reset.
          2. **Sweep page** — the in-progress bounded AUTO sweep has more pages.
          3. **Auto rescan** (consumed atomically): sole consumer of the dirty
             flag; re-armed on a failed follow-up query in the refetch.
          4. **Document channel non-empty** (peeked, NOT drained).
          5. Nothing pending: release the slot with its owner token — RELEASED.

        Boundary (documented, not solved here): this closes every
        published-but-missed handoff, but an enqueue that crashes BETWEEN its
        doc_status persist and its publish leaves no signal at all — that doc
        is recovered by the next run's initial strict scan, never by this
        decision.
        """
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                return PipelineNextDecision(
                    PipelineNextStep.CANCELLED,
                    internal_error=pipeline_status.get("cancellation_reason")
                    == "internal_error",
                )
            if (
                pipeline_status.get("manual_phase", MANUAL_PHASE_IDLE)
                == MANUAL_PHASE_DRAIN_TO_IDLE
            ):
                # Advance the manual drain we already own (freeze is set). Drain
                # every remaining page / signal, wait out pre-freeze in-flight
                # enqueues, then run the exclusive reset. Do NOT peek a new
                # manual — a second request waits FIFO (LR2 §7.5).
                if sweep_cursor is not CURSOR_END:
                    return PipelineNextDecision(PipelineNextStep.CONTINUE_SWEEP_PAGE)
                if ingress.consume_auto_rescan():
                    return PipelineNextDecision(PipelineNextStep.CONTINUE_AUTO)
                if ingress.counts().get("documents"):
                    return PipelineNextDecision(PipelineNextStep.CONTINUE_DOCUMENT)
                if pipeline_status.get("pending_enqueues", 0) > 0:
                    # Enqueues reserved BEFORE the freeze must finish (their
                    # PENDING docs join this cohort) before the reset can start
                    # with no failed producer (LR2 §7.2 step 5 / §7.3). The
                    # CONTINUE_DRAIN_WAIT refetch reaps confirmed-dead tokens
                    # before it waits, so a worker SIGKILLed mid-reserve (Linux
                    # multi-worker) cannot leave a phantom count that wedges this
                    # drain (the freeze blocks the uploads that would reap it).
                    return PipelineNextDecision(PipelineNextStep.CONTINUE_DRAIN_WAIT)
                return PipelineNextDecision(PipelineNextStep.BEGIN_EXCLUSIVE_RESET)
            manual_msg = ingress.peek_next_manual_retry()
            if manual_msg is not None:
                return PipelineNextDecision(
                    PipelineNextStep.CONTINUE_MANUAL,
                    manual_request_id=manual_msg.request_id,
                )
            # The in-progress bounded sweep has more pages: finish draining the
            # KNOWN backlog before consuming any quiescence signal. Preempted
            # by cancellation and manual retry above (LR2 §6.3 "无 manual 时继续
            # cursor"); consumes nothing, so auto-rescan/document stay pending
            # for when the cursor reaches CURSOR_END. Collapses to a no-op when
            # paging is off (cursor is CURSOR_END after a single-scan page).
            if sweep_cursor is not CURSOR_END:
                return PipelineNextDecision(PipelineNextStep.CONTINUE_SWEEP_PAGE)
            if ingress.consume_auto_rescan():
                return PipelineNextDecision(PipelineNextStep.CONTINUE_AUTO)
            if ingress.counts().get("documents"):
                return PipelineNextDecision(PipelineNextStep.CONTINUE_DOCUMENT)
            pipeline_status.update({"busy": False, "busy_owner": None})
            return PipelineNextDecision(PipelineNextStep.RELEASED)

    async def _refetch_for_decision(
        self,
        decision: PipelineNextDecision,
        sweep_statuses,
        sweep_cursor: CursorPosition,
        ingress,
        *,
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
        drain_progress: "_ManualDrainProgress",
    ) -> tuple[dict[str, DocProcessingStatus], tuple, CursorPosition]:
        """Fetch the next batch for a CONTINUE_*/BEGIN_* decision (bounded).

        Returns ``(docs, next_statuses, next_cursor)`` — the hydrated
        full-status map to process plus the sweep state the caller threads into
        the next ``_decide``/refetch (``next_cursor is CURSOR_END`` ends the
        sweep). All manual ACKing is internal to BEGIN_EXCLUSIVE_RESET, so no
        ACK id is returned.

        The continuations:

        * **CONTINUE_SWEEP_PAGE** — advance the SAME sweep to its next page. A
          page-fetch failure does NOT advance the cursor and arms auto-rescan so
          the remaining backlog is recovered (LR2 §6.3).
        * **CONTINUE_MANUAL** — BEGIN a manual drain (LR2 §7.2): set the freeze +
          owner (owner-checked, tied to this run's busy token), phase →
          DRAIN_TO_IDLE, then start a fresh AUTO sweep from ``CURSOR_START`` (the
          drain processes the AUTO backlog; FAILED is reset only in the exclusive
          phase). The manual request stays sticky — ACKed after the reset.
        * **CONTINUE_DRAIN_WAIT** — bounded async sleep waiting for pre-freeze
          in-flight enqueues to finish, then re-decide (empty batch).
        * **BEGIN_EXCLUSIVE_RESET** — the drain reached strict idle. Do a FINAL
          strict AUTO confirmation sweep (a doc a since-finished in-flight
          enqueue added after the previous sweep passed); if non-empty, process
          it and stay in the drain. If empty, run the exclusive FAILED→PENDING
          reset (no worker), ACK the request, clear the freeze (→ PROCESS), then
          fetch a fresh AUTO sweep so the just-reset PENDING docs get processed.
        * **CONTINUE_AUTO** — start a fresh bounded AUTO sweep from
          ``CURSOR_START``. The signal was consumed in the decision, so a
          first-page failure re-arms auto-rescan before propagating.
        * **CONTINUE_DOCUMENT** — bounded destructive drain of resident
          document messages FIRST, then a fresh bounded AUTO sweep from
          ``CURSOR_START``. Drain-before-scan is the correctness anchor: every
          drained message's doc that is still live (its PENDING row persisted
          before its publish, which preceded the drain) is a PENDING row the
          multi-page sweep necessarily reaches, so a drained message never
          needs re-publishing; a provably-stale one (terminal/deleted) is
          compacted. A sweep-start failure re-publishes the drained messages
          and arms auto-rescan before propagating.
        """
        step = decision.step

        # (A) Continue the current bounded sweep — same statuses, next page.
        if step is PipelineNextStep.CONTINUE_SWEEP_PAGE:
            try:
                docs, next_cursor = await self._next_scheduling_page(
                    sweep_statuses, sweep_cursor
                )
            except BaseException:
                try:
                    _rearm_auto_rescan(ingress)
                except BaseException as compensation_error:
                    logger.warning(
                        "[pipeline] failed to arm auto-rescan after a paged "
                        "sweep refetch failure; the remaining backlog is "
                        f"recovered by the NEXT explicit trigger: {compensation_error}"
                    )
                raise
            return docs, sweep_statuses, next_cursor

        # (A2) Manual drain: hold the freeze while the pre-freeze in-flight
        # enqueues finish. Reap confirmed-dead reservation tokens FIRST — a
        # worker SIGKILLed mid-reserve (Linux multi-worker) would otherwise leave
        # a phantom pending_enqueues count that wedges this drain forever, since
        # the freeze blocks the uploads whose acquire would normally reap it
        # (no-op off Linux-multiworker). Then a bounded async sleep (not a
        # busy-loop); the next decision drains any doc they publish, then
        # re-checks pending_enqueues.
        if step is PipelineNextStep.CONTINUE_DRAIN_WAIT:
            await reap_dead_reservations_locked(pipeline_status, pipeline_status_lock)
            await asyncio.sleep(_MANUAL_DRAIN_POLL_SECONDS)
            return {}, sweep_statuses, sweep_cursor

        # (A3) The manual drain reached strict idle — run the exclusive reset.
        if step is PipelineNextStep.BEGIN_EXCLUSIVE_RESET:
            return await self._refetch_begin_exclusive_reset(
                ingress,
                token=token,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                drain_progress=drain_progress,
            )

        # (B) Start a FRESH sweep from CURSOR_START. CONTINUE_MANUAL begins the
        # freeze first; CONTINUE_DOCUMENT drains the mailbox first (bounded).
        drained: list[PipelineIngressMessage] = []
        if step is PipelineNextStep.CONTINUE_MANUAL:
            # Set the freeze + owner BEFORE draining so no new ingress enters
            # (owner-checked; a lost slot winds the run down via an empty batch).
            if not await self._begin_manual_drain(
                decision.manual_request_id,
                token,
                pipeline_status,
                pipeline_status_lock,
            ):
                return {}, _AUTO_RESUME_DOC_STATUSES, CURSOR_END
        if step is PipelineNextStep.CONTINUE_DOCUMENT:
            drained = ingress.drain_documents(limit=_FEEDER_DRAIN_LIMIT)
        try:
            docs, next_cursor = await self._next_scheduling_page(
                _AUTO_RESUME_DOC_STATUSES, CURSOR_START
            )
        except BaseException:
            # Compensations must not raise over the original error (multiproc
            # mailbox calls are RPCs; BaseException here so even an interrupt
            # in a compensation call cannot displace it): the docs behind a
            # lost signal are still PENDING rows, recovered by the next run's
            # initial scan. A manual request stays sticky on its own (the freeze
            # is cleared by the run's finally), so it needs no re-arm.
            try:
                if step is PipelineNextStep.CONTINUE_AUTO:
                    _rearm_auto_rescan(ingress)
                elif drained:
                    self._republish_documents(ingress, drained, source="quiescence")
                    # Backstop for re-publishes swallowed above: the strict
                    # rescan behind the flag recovers those PENDING docs.
                    _rearm_auto_rescan(ingress)
            except BaseException as compensation_error:
                logger.warning(
                    "[pipeline] failed to restore the consumed wake-up signal "
                    f"after a strict refetch failure ({len(drained)} drained "
                    f"document message(s), first ids "
                    f"{[m.doc_id for m in drained[:8]]}); the docs stay "
                    "PENDING and are recovered by the NEXT explicit trigger, "
                    f"not automatically: {compensation_error}"
                )
            raise
        return docs, _AUTO_RESUME_DOC_STATUSES, next_cursor

    async def _refetch_begin_exclusive_reset(
        self,
        ingress,
        *,
        token: str,
        pipeline_status: dict,
        pipeline_status_lock,
        drain_progress: "_ManualDrainProgress",
    ) -> tuple[dict[str, DocProcessingStatus], tuple, CursorPosition]:
        """BEGIN_EXCLUSIVE_RESET handler (LR2 §7.2 step 7-8 → §7.3 → §7.4).

        A final strict AUTO sweep confirms the drain is complete; if it finds a
        doc (an in-flight enqueue added it after the previous sweep passed), the
        run processes it and stays in DRAIN_TO_IDLE. Otherwise the exclusive
        FAILED→PENDING reset runs (no worker), the request is ACKed, the freeze
        is cleared (→ PROCESS), and a fresh AUTO sweep returns the just-reset
        PENDING docs for normal processing.

        This confirmation is also the drain's forward-progress checkpoint: it is
        the single place that decides "not drained yet, sweep again", so it is
        where a drain that CANNOT advance has to be caught (LR2 §7.2). Two
        distinct failures are caught here:

        * an active row the drain can never advance at all (an unfinished
          custom-chunk journal) makes the AUTO set non-empty forever, so the
          reset must NOT start — :meth:`_fence_blocked_manual_drain`;
        * rows that look routable yet keep coming back unchanged fence the
          workspace rather than being swept forever —
          :meth:`_fence_stalled_manual_drain`."""
        # (1) Final strict confirmation that the PERSISTENT AUTO set is drained.
        # Unfiltered and paged to CURSOR_END (see _confirm_auto_drained): the
        # routing sweep's filtered view is not an idle proof, and one empty page
        # is not termination.
        docs, next_cursor, blocking_ids = await self._confirm_auto_drained()
        if docs:
            if not drain_progress.observe(docs.keys()):
                await self._fence_stalled_manual_drain(
                    docs, pipeline_status, pipeline_status_lock
                )
            return docs, _AUTO_RESUME_DOC_STATUSES, next_cursor
        if blocking_ids:
            await self._fence_blocked_manual_drain(
                blocking_ids, pipeline_status, pipeline_status_lock
            )

        # (2) Read the request id off the freeze owner, then run the reset. A
        # missing owner means a concurrent reaper cleared the freeze (dead
        # process) — wind the run down via an empty PROCESS sweep.
        async with pipeline_status_lock:
            manual_owner = pipeline_status.get("manual_owner")
        request_id = (manual_owner or {}).get("request_id")
        if request_id is None:
            return {}, _AUTO_RESUME_DOC_STATUSES, CURSOR_END

        if await self._run_exclusive_failed_reset(
            request_id, token, pipeline_status, pipeline_status_lock
        ):
            # ACK only after every FAILED→PENDING reset persisted (LR2 §7.3
            # completion point): a crash before this re-runs the reset; a crash
            # after is recovered by the now-persistent PENDING rows.
            ingress.ack_manual_retry(request_id)
            await self._end_manual_drain(token, pipeline_status, pipeline_status_lock)

        # (3) PROCESS: the reset PENDING docs (and any admitted since the freeze
        # cleared) are picked up by a fresh AUTO sweep.
        docs, next_cursor = await self._next_scheduling_page(
            _AUTO_RESUME_DOC_STATUSES, CURSOR_START
        )
        return docs, _AUTO_RESUME_DOC_STATUSES, next_cursor

    async def _fence_blocked_manual_drain(
        self,
        blocking_ids: tuple[str, ...],
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Fence the workspace: the AUTO set holds rows the drain cannot advance.

        Rows with an unfinished custom-chunk journal — only ``/documents/scan``'s
        rollback (or the owning SDK call) resolves those. LR2 §4.2 requires the
        AUTO set to be confirmed empty before ``EXCLUSIVE_RESET`` and §7.2 forbids
        silently ignoring an active row to let the manual retry start, so the reset
        does not run and the request is NOT ACKed: no FAILED document is consumed
        by an attempt that could not happen.

        This sets ``recovery_required`` — §7.2's literal remedy — after an attempt
        to avoid it did not survive review. Reporting the blocker WITHOUT fencing
        looked better (a fence refuses every mutation, including the very
        ``/documents/scan`` that fixes this) but the reasoning was wrong: a sticky
        un-ACKed manual request already makes ``/scan`` refuse its reservation
        (``refuse_when_manual_pending``, LR2 §8.1, so a scan cannot jump the manual
        FIFO), and the blocker leaves exactly that request queued. So the
        "unfenced" path had no reachable remedy either — it was a silent dead end
        instead of a loud one, which is strictly worse.

        The way out is therefore the same for both halves of the deadlock and is
        documented on the fence: ``POST /documents/recovery/force_reset`` clears the
        fence AND cancels the queued manual intents that were blocking ``/scan``,
        after which ``POST /documents/scan`` rolls the journal back and runs the
        FAILED reset itself.

        Raises :class:`PipelineRecoveryRequiredError`, which unwinds the run
        through its ``finally``: freeze and ``busy`` are released (owner-checked) so
        a live-but-exiting owner never wedges the pipeline, while the fence
        survives to refuse mutations with 503.
        """
        detail = (
            f"manual retry cannot start: {len(blocking_ids)} or more active "
            "document(s) hold an unfinished custom-chunk operation, so the "
            "pipeline is not idle and the exclusive FAILED reset must not begin. "
            "Clear this with POST /documents/recovery/force_reset (which also "
            "cancels the queued retry that is blocking /documents/scan), then POST "
            "/documents/scan to roll the operation back — the scan runs the FAILED "
            f"reset itself (blocked doc id sample: {', '.join(blocking_ids) or 'unavailable'})."
        )
        logger.error(detail)
        await fence_workspace_for_recovery(
            pipeline_status,
            pipeline_status_lock,
            kind="manual_drain_blocked",
            message=detail,
            operation_record={"scope": ", ".join(blocking_ids)},
        )
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = detail
            append_pipeline_history(pipeline_status, detail)
        raise PipelineRecoveryRequiredError(detail, blocked_doc_ids=tuple(blocking_ids))

    async def _fence_stalled_manual_drain(
        self,
        docs: dict[str, DocProcessingStatus],
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Fence the workspace for a DRAIN_TO_IDLE that cannot advance (LR2 §7.2).

        The same active rows have now blocked the drain for
        :data:`_MANUAL_DRAIN_STALL_ROUNDS` consecutive confirmation rounds without
        changing state. Sweeping again is guaranteed to return them again, so the
        design's rule applies: set ``recovery_required``, report a BOUNDED sample
        of the blocking doc ids, and stop — never spin, and never silently drop
        the active rows to let the manual retry start.

        Raises :class:`PipelineRecoveryRequiredError`, which unwinds the run
        through its ``finally``: the freeze and ``busy`` are released
        (owner-checked) so nothing is wedged on a live-but-exiting owner, while
        the fence survives and refuses every later mutation with 503 until an
        operator resolves the rows or calls ``/documents/recovery/force_reset``.
        The manual request stays sticky and un-ACKed, so no FAILED document is
        consumed by an attempt that never ran.
        """
        blocked = sorted(docs)[:_MANUAL_DRAIN_STALL_SAMPLE]
        detail = (
            f"manual retry drain stalled: {len(docs)} active document(s) have "
            f"blocked DRAIN_TO_IDLE for {_MANUAL_DRAIN_STALL_ROUNDS} consecutive "
            f"rounds without changing state (blocked doc id sample: "
            f"{', '.join(blocked) or 'unavailable'})."
        )
        logger.error(detail)
        await fence_workspace_for_recovery(
            pipeline_status,
            pipeline_status_lock,
            kind="manual_drain_stalled",
            message=detail,
            operation_record={"scope": ", ".join(blocked)},
        )
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = detail
            append_pipeline_history(pipeline_status, detail)
        raise PipelineRecoveryRequiredError(detail, blocked_doc_ids=tuple(blocked))

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
            # Best-effort engine attribution for the FAILED metadata when the
            # failure happens before the per-doc engine is resolved below.
            resolved_engine_w: str | None = None
            # doc_status ``parse_engine`` value, computed once below and used at
            # BOTH the success stamp and the failure engine_hint so the field
            # never jumps across transitions. Encoded (engine+params) when the
            # engine that runs matches the stored engine, else bare effective.
            status_engine_w: str | None = None
            try:
                doc_id_w, status_doc_w = item
                file_path_w = getattr(status_doc_w, "file_path", "unknown_source")
                # Boundary cancellation check: skip parsing the next queued doc
                # without invoking the engine, mark it FAILED with a friendly
                # "User cancelled" message, and let the finally task_done()
                # drain the queue so q.join() in _run_pipeline_batch returns.
                if await self._cancellation_requested(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                ):
                    await self._mark_doc_cancelled_in_stage(
                        ctx=ctx,
                        doc_id=doc_id_w,
                        status_doc=status_doc_w,
                        file_path=file_path_w,
                        stage_label="parse",
                        pipeline_status=ctx.pipeline_status,
                        pipeline_status_lock=ctx.pipeline_status_lock,
                    )
                    continue
                content_data_w = await self.full_docs.get_by_id(doc_id_w)
                if not content_data_w:
                    raise Exception(
                        f"Document content not found in full_docs for doc_id: {doc_id_w}"
                    )
                if isinstance(status_doc_w.metadata, dict):
                    source_file_w = _read_source_file(status_doc_w.metadata)
                    if source_file_w:
                        # Normalize the legacy ``source_file_name`` onto the new
                        # key in the in-memory status metadata so the carry-over
                        # allowlist (which no longer lists ``source_file_name``)
                        # preserves it through the PARSING upsert below. Without
                        # this, a retry after a parse failure — before full_docs
                        # is rewritten — would no longer resolve the hinted
                        # source file. Idempotent when the new key already exists.
                        status_doc_w.metadata["source_file"] = source_file_w
                        if not _read_source_file(content_data_w):
                            content_data_w["source_file"] = source_file_w
                # Stamp parse_start_time on the in-memory status_doc so
                # carry-over (_DOC_STATUS_METADATA_CARRY_OVER_KEYS) writes it
                # into doc_status here and preserves it across every
                # subsequent state transition for stage-duration analysis.
                if not isinstance(status_doc_w.metadata, dict):
                    status_doc_w.metadata = {}
                # Stale per-attempt fields (parse_end_time / *_stage_skipped /
                # analyzing_*) from a prior failed/retried attempt are already
                # scrubbed when the document is brought to PENDING in
                # _validate_and_fix_document_consistency (the single cleanup
                # point), so they are not carried into this PARSING upsert.
                status_doc_w.metadata["parse_start_time"] = int(time.time())
                await self._upsert_doc_status_transition(
                    ctx=ctx,
                    doc_id=doc_id_w,
                    status=DocStatus.PARSING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )
                async with ctx.pipeline_status_lock:
                    log_message = f"Parsing ({engine}): {doc_id_w}"
                    logger.info(log_message)
                    ctx.pipeline_status["latest_message"] = log_message
                    append_pipeline_history(ctx.pipeline_status, log_message)
                # Resolve the actual parser per-doc from the batch snapshot
                # (snapshot-consistent: a mid-batch register_parser cannot be
                # picked up here). ``engine`` is only the queue-group/pool id.
                specs = ctx.parser_specs
                doc_format_w = content_data_w.get("parse_format", FULL_DOCS_FORMAT_RAW)
                key = resolve_stored_document_parser_engine(
                    file_path=file_path_w, content_data=content_data_w
                )
                # PENDING_PARSE must resolve to a real (user-selectable) engine;
                # an internal key (reuse/passthrough) wrongly stored as
                # parse_engine is corrupt -> fail just this doc.
                if doc_format_w == FULL_DOCS_FORMAT_PENDING_PARSE:
                    key_spec = specs.get(key)
                    if key_spec is not None and not key_spec.user_selectable:
                        raise ValueError(
                            f"internal parser {key!r} is not a valid "
                            f"PENDING_PARSE engine: doc_id={doc_id_w}"
                        )
                parser = get_parser(key, specs=specs)
                if parser is None:
                    logger.warning(
                        "[parse] engine %r not registered; falling back to legacy",
                        key,
                    )
                effective_key = key if parser is not None else "legacy"
                resolved_engine_w = effective_key
                # When the stored parse_engine carries engine params AND the
                # engine that actually ran matches it, preserve the encoded
                # directive for the doc_status stamp so the per-file params stay
                # user-visible. Left None otherwise (no params, or an engine
                # mismatch / internal passthrough-reuse key) so the existing
                # resolver logic decides the displayed engine unchanged.
                _stored_pe_w = (
                    content_data_w.get("parse_engine")
                    if isinstance(content_data_w, dict)
                    else None
                )
                if _stored_pe_w and "(" in str(_stored_pe_w):
                    from lightrag.parser.routing import (
                        decode_parse_engine,
                        encode_parse_engine,
                        normalize_parser_engine,
                    )

                    if normalize_parser_engine(_stored_pe_w) == effective_key:
                        _, _stored_params_w, _ = decode_parse_engine(_stored_pe_w)
                        if _stored_params_w:
                            status_engine_w = encode_parse_engine(
                                effective_key, _stored_params_w
                            )
                parser = parser or get_parser("legacy", specs=specs)
                # Suffix gate only for real engines on a PENDING_PARSE parse;
                # reuse/passthrough (raw/lightrag/unknown_source) are skipped.
                if (
                    doc_format_w == FULL_DOCS_FORMAT_PENDING_PARSE
                    and effective_key in supported_parser_engines(specs)
                ):
                    suffix_w = parser_suffix(file_path_w)
                    if suffix_w not in suffix_capabilities(effective_key, specs):
                        raise ValueError(
                            f"engine {effective_key!r} does not support "
                            f".{suffix_w or '<no suffix>'}: doc_id={doc_id_w}"
                        )
                parsed_data_w = (
                    await parser.parse(
                        ParseContext(
                            self,
                            doc_id_w,
                            file_path_w,
                            content_data_w,
                            pipeline_cancel_event=ctx.pipeline_cancel_event,
                        )
                    )
                ).to_dict()

                # Mirror parse-stage LLM cache keys before the post-parse
                # cancellation check. A cancellation flushes completed cache
                # rows, and the FAILED status must retain their IDs so document
                # deletion can purge them instead of leaving orphaned entries.
                smartheading_ids_w = parsed_data_w.get("smartheading_llm_cache_ids")
                if smartheading_ids_w:
                    if not isinstance(status_doc_w.metadata, dict):
                        status_doc_w.metadata = {}
                    status_doc_w.metadata["smartheading_llm_cache_ids"] = (
                        smartheading_ids_w
                    )

                # A cancellation can arrive after a title-block LLM call has
                # completed but before its native parser returns. Do not stamp
                # or enqueue a successful result once the batch is cancelled.
                if await self._cancellation_requested(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                ):
                    await self._mark_doc_cancelled_in_stage(
                        ctx=ctx,
                        doc_id=doc_id_w,
                        status_doc=status_doc_w,
                        file_path=file_path_w,
                        stage_label="parse",
                        pipeline_status=ctx.pipeline_status,
                        pipeline_status_lock=ctx.pipeline_status_lock,
                    )
                    continue

                # Align the in-memory body with the sanitized copy that
                # _persist_parsed_full_docs wrote to full_docs: a parser may
                # return ParseResult(content=...) carrying the pre-clean text
                # (e.g. legacy returns the raw extraction verbatim). Downstream
                # this body feeds content_summary / content_length on doc_status
                # and the duplicate-check length, so leaving C0 control chars
                # (incl. NUL, which breaks PostgreSQL text writes) here would let
                # them reach doc_status. No-op for sidecar engines (already
                # cleaned at write_sidecar) and for already-clean content.
                if isinstance(parsed_data_w.get("content"), str):
                    parsed_data_w["content"] = strip_control_characters(
                        parsed_data_w["content"]
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

                # Mirror raw-bundle cache-hit flag from mineru/docling; cache-
                # miss runs (including parse_native, which has no cache
                # concept) stamp ``parse_end_time`` instead so post-mortem
                # can derive the parse-stage duration. The two fields are
                # mutually exclusive per attempt. Both are persisted right
                # below (before the doc enters q_analyze) so doc_status
                # reflects the parse end immediately; carry-over keeps them
                # visible across every later transition.
                if not isinstance(status_doc_w.metadata, dict):
                    status_doc_w.metadata = {}
                if parsed_data_w.get("parse_stage_skipped"):
                    status_doc_w.metadata["parse_stage_skipped"] = True
                else:
                    status_doc_w.metadata["parse_end_time"] = int(time.time())

                # Stamp the parse-stage extraction metadata (parse_format /
                # parse_engine) now that the engine has run and reported its
                # actual format/engine. These are determined here, so record
                # them at the PARSING upsert below instead of deferring to
                # PROCESSING; carry-over (_DOC_STATUS_METADATA_CARRY_OVER_KEYS)
                # then preserves them across ANALYZING → PROCESSING → PROCESSED.
                # ``resolve_doc_status_parse_engine`` is the shared resolver
                # used by process_single_document too, so the value never jumps
                # between the early and final writes. The engine source order
                # mirrors the process stage's read from full_docs: the parser's
                # own report wins, then the enqueue-time directive on
                # content_data (raw passthrough), then the format-based default.
                parse_format_w = (
                    parsed_data_w.get("parse_format") or FULL_DOCS_FORMAT_RAW
                )
                # ``status_engine_w`` (computed pre-parse) is the encoded
                # directive for the engine that actually ran; prefer it so the
                # recorded value keeps the per-file params, then the parser's
                # own bare report, then the stored value.
                explicit_engine_w = (
                    status_engine_w
                    or parsed_data_w.get("parse_engine")
                    or (
                        content_data_w.get("parse_engine")
                        if isinstance(content_data_w, dict)
                        else None
                    )
                )
                status_doc_w.metadata["parse_format"] = parse_format_w
                status_doc_w.metadata["parse_engine"] = resolve_doc_status_parse_engine(
                    parse_format_w, explicit_engine_w
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
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                    ctx=ctx,
                ):
                    continue

                # Compute content-derived fields here while the parse worker
                # still holds the body, and stamp them on status_doc so they
                # are persisted at the PARSING transition below. Downstream
                # stages (analyze / process) re-read the body from full_docs by
                # doc_id instead of carrying it through q_analyze / q_process,
                # keeping large documents out of those in-memory buffers. Parse
                # has already persisted the parsed body to full_docs (lightrag /
                # raw), so the re-read is guaranteed to find it.
                parsed_content_w = parsed_data_w.get("content", "") or ""
                status_doc_w.content_summary = get_content_summary(parsed_content_w)
                status_doc_w.content_length = len(parsed_content_w)

                # Persist the parse-stage outcome to doc_status now, before the
                # doc waits in q_analyze, so parse_end_time / parse_stage_skipped
                # reflect the actual end of parsing instead of only landing at the
                # ANALYZING transition via carry-over. content_hash is already
                # refreshed and duplicates are filtered out by this point.
                await self._upsert_doc_status_transition(
                    ctx=ctx,
                    doc_id=doc_id_w,
                    status=DocStatus.PARSING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )

                # smart-heading cache rows are created in the parse stage and
                # otherwise wait for the much later PROCESS-level _insert_done.
                # Commit them after their IDs are durable in doc_status, but
                # keep a cache flush failure non-fatal to the parsed document.
                if smartheading_ids_w:
                    await self._persist_llm_response_cache_best_effort(
                        stage_label="smart-heading parse",
                        doc_id=doc_id_w,
                    )

                # Drop the heavy body from the queue payload; q_analyze /
                # q_process now carry only lightweight metadata (blocks_path,
                # parse_format, flags). process_single_document re-reads the
                # body from full_docs by doc_id.
                parsed_data_w.pop("content", None)
                await ctx.q_analyze.put((doc_id_w, status_doc_w, parsed_data_w))
            except (PipelineCancelledException, LLMBridgePipelineCancelled):
                # Cancellation raised from inside the parse engine (future-
                # proofing — engines do not generally call
                # _raise_if_cancelled, but native smart-heading's bridge raises
                # LLMBridgePipelineCancelled while waiting on the batch cancel
                # event. Parser-executor shutdown is intentionally a distinct
                # exception and remains a generic parse failure for audit.
                await self._mark_doc_cancelled_in_stage(
                    ctx=ctx,
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                    stage_label="parse",
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                )
            except Exception as e:
                logger.error(f"Parse worker failed ({engine}): {e}")
                # Mirror the pre-deferral enqueue-time error documents:
                # content_summary (when empty) + metadata error_type /
                # error_stage / parse_engine, so the WebUI list and detail
                # views describe the failure instead of showing a blank row.
                extra_fields_w, metadata_extra_w = doc_status_parse_failure_fields(
                    e,
                    status_doc=status_doc_w,
                    engine_hint=status_engine_w or resolved_engine_w or engine,
                )
                try:
                    await self._upsert_doc_status_transition(
                        ctx=ctx,
                        doc_id=doc_id_w,
                        status=DocStatus.FAILED,
                        status_doc=status_doc_w,
                        file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                        extra_fields=extra_fields_w,
                        metadata_extra=metadata_extra_w,
                    )
                except Exception as upsert_err:
                    # The storage backend may be unavailable too (e.g. the same
                    # outage that failed the parse). Don't re-raise — that would
                    # take down the worker — but log so the doc stuck in PARSING
                    # is diagnosable instead of failing silently.
                    logger.error(
                        f"Failed to record FAILED status for {doc_id_w}: {upsert_err}"
                    )
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
                # Boundary cancellation check: same pattern as _parse_worker.
                # Items already past PARSING that are still queued for analyze
                # are short-circuited to FAILED here so the multimodal VLM
                # path is not entered after the user clicked cancel.
                if await self._cancellation_requested(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                ):
                    await self._mark_doc_cancelled_in_stage(
                        ctx=ctx,
                        doc_id=doc_id_w,
                        status_doc=status_doc_w,
                        file_path=file_path_w,
                        stage_label="analyze",
                        pipeline_status=ctx.pipeline_status,
                        pipeline_status_lock=ctx.pipeline_status_lock,
                    )
                    continue
                # content_summary / content_length were computed by the parse
                # worker (which held the body) and are already set on this
                # status_doc; the body is no longer carried through the queue,
                # and analyze_multimodal works off the on-disk sidecar
                # (blocks_path), not the body, so no re-read is needed here.
                # Stamp analyzing_start_time so per-stage durations stay
                # derivable from doc_status even after PROCESSED / FAILED;
                # carry-over preserves it across later upserts.
                if not isinstance(status_doc_w.metadata, dict):
                    status_doc_w.metadata = {}
                status_doc_w.metadata["analyzing_start_time"] = int(time.time())
                await self._upsert_doc_status_transition(
                    ctx=ctx,
                    doc_id=doc_id_w,
                    status=DocStatus.ANALYZING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )
                analyzed = await self.analyze_multimodal(
                    doc_id=doc_id_w,
                    file_path=file_path_w,
                    parsed_data=parsed_data_w,
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                )
                # Mirror analyze-stage outcome as a 3-way decision so the
                # ``analyzing_end_time`` stamp only ever lands on attempts
                # that genuinely completed:
                #   - ``analyzing_stage_skipped`` (set by analyze_multimodal at
                #     its three early-return branches: no blocks_path, blocks
                #     file missing, no i/t/e options) → user/config skipped;
                #     stamp the skipped flag.
                #   - ``multimodal_processed`` (set by analyze_multimodal only
                #     after the full processing loop succeeds) → genuine
                #     completion; stamp ``analyzing_end_time``.
                #   - Neither flag → analyze_multimodal soft-swallowed an
                #     exception (generic ``except Exception``) or hit a
                #     malformed/empty sidecar early return. Failure is not a
                #     skip AND not a completion, so write neither field.
                # The skipped/end_time pair is mutually exclusive. The two
                # outcome-bearing branches persist immediately below (before
                # the doc enters q_process) so analyzing_end_time /
                # analyzing_stage_skipped reflect the actual end of analysis
                # rather than only landing at the PROCESSING transition.
                if not isinstance(status_doc_w.metadata, dict):
                    status_doc_w.metadata = {}
                analyze_outcome_recorded = False
                if analyzed.pop("analyzing_stage_skipped", False):
                    status_doc_w.metadata["analyzing_stage_skipped"] = True
                    analyze_outcome_recorded = True
                elif analyzed.get("multimodal_processed"):
                    status_doc_w.metadata["analyzing_end_time"] = int(time.time())
                    analyze_outcome_recorded = True
                # Soft-failed attempts (neither flag) write nothing new, so skip
                # the extra upsert; PROCESSING will be their next doc_status write.
                if analyze_outcome_recorded:
                    await self._upsert_doc_status_transition(
                        ctx=ctx,
                        doc_id=doc_id_w,
                        status=DocStatus.ANALYZING,
                        status_doc=status_doc_w,
                        file_path=file_path_w,
                    )
                await ctx.q_process.put((doc_id_w, status_doc_w, analyzed))
            except PipelineCancelledException:
                # In-flight cancellation surfaced from analyze_multimodal
                # (poll loop detected cancellation_requested mid-VLM).
                # Route through the friendly message path so error_msg and
                # history_messages match the boundary-check branch.
                await self._mark_doc_cancelled_in_stage(
                    ctx=ctx,
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                    stage_label="analyze",
                    pipeline_status=ctx.pipeline_status,
                    pipeline_status_lock=ctx.pipeline_status_lock,
                )
            except Exception as e:
                # Mirror _parse_worker: failures here must transition the
                # document to FAILED with a diagnostic ``error_msg``, otherwise
                # MultimodalAnalysisError (raised by analyze_multimodal under
                # the new hard-failure contract) would leave the doc stuck in
                # ANALYZING forever.
                logger.error(f"Analyze worker failed: {e}")
                try:
                    await self._upsert_doc_status_transition(
                        ctx=ctx,
                        doc_id=doc_id_w,
                        status=DocStatus.FAILED,
                        status_doc=status_doc_w,
                        file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                        extra_fields={"error_msg": str(e)},
                    )
                except Exception as upsert_err:
                    # Mirror _parse_worker: log instead of swallowing so a
                    # storage write failure leaves the doc stuck in ANALYZING
                    # with a diagnosable trail rather than silently.
                    logger.error(
                        f"Failed to record FAILED status for {doc_id_w}: {upsert_err}"
                    )
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
            except Exception as e:
                # process_single_document handles its own per-doc failures; an
                # escape here means even the FAILED-status write failed (e.g.
                # the doc_status backend is down). Do NOT let the worker die —
                # that strands the remaining queued items and hangs
                # q_process.join() forever, wedging the pipeline busy. Route it
                # to the batch-abort path (same flag as IndexFlushError) and
                # keep draining so the batch winds down cleanly. CancelledError
                # is a BaseException, not caught here, so a normal worker
                # cancellation at batch end still propagates.
                logger.error(f"Unhandled error in process worker; aborting batch: {e}")
                logger.error(traceback.format_exc())
                async with ctx.pipeline_status_lock:
                    ctx.pipeline_status.update(
                        {
                            "cancellation_requested": True,
                            "cancellation_reason": "internal_error",
                            "cancellation_detail": (
                                f"process worker unhandled error: {e}"
                            ),
                        }
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
        from lightrag.parser.routing import parse_process_options

        file_path = resolve_doc_file_path(status_doc=status_doc)
        current_file_number = 0
        file_extraction_stage_ok = False
        process_start_time = int(time.time())
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

                    extraction_message = (
                        f"Extracting stage {current_file_number}/"
                        f"{ctx.total_files}: {file_path}"
                    )
                    logger.info(extraction_message)
                    processing_message = f"Processing d-id: {doc_id}"
                    logger.info(processing_message)
                    ctx.pipeline_status.update(
                        {
                            "cur_batch": ctx.processed_count,
                            "latest_message": processing_message,
                        }
                    )
                    # One call: the pair belongs together in the log, and the
                    # ring capacity that used to be enforced right here now
                    # lives in the funnel — this loop was never the only writer
                    # (deletes, scans and manual resets log plenty without ever
                    # reaching it), so trimming here bounded nothing.
                    append_pipeline_history(
                        ctx.pipeline_status, extraction_message, processing_message
                    )

                # The parsed body is no longer carried through q_analyze /
                # q_process (it would pin large documents in memory). Re-read it
                # from full_docs (already fetched into content_data above) and
                # strip the lightrag marker according to the stored parse_format
                # — parse persisted the body for every engine before enqueue.
                content = strip_lightrag_doc_prefix(
                    (content_data or {}).get("content"),
                    (content_data or {}).get("parse_format"),
                )

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
                    # fresh build from current addon_params['chunker'],
                    # scoped to the per-doc strategy decoded above so
                    # the slim shape stays consistent with newly
                    # enqueued rows.  F-strategy split args fall back
                    # to whatever lives in
                    # ``addon_params['chunker']['fixed_token']``;
                    # runtime overrides are an ainsert-time concern and
                    # don't apply at process time for legacy rows.
                    from lightrag.parser.routing import resolve_chunk_options

                    chunk_opts = resolve_chunk_options(
                        self.addon_params, process_options=doc_process_opts
                    )
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
                        # P carries its own ``chunk_token_size`` (CHUNK_P_SIZE
                        # env or ``addon_params['chunker']['paragraph_semantic']``);
                        # pop it out of the kwargs so we don't pass it
                        # both positionally and via ``**`` splat (which
                        # would TypeError).  Unlike R/V, ``default_chunker_config``
                        # always populates this slot — falling back to
                        # ``resolved_chunk_size`` (global CHUNK_SIZE) here is
                        # only a safety net for snapshots predating that
                        # change; new docs always carry ``DEFAULT_CHUNK_P_SIZE``.
                        p_opts = dict(chunk_opts.get("paragraph_semantic") or {})
                        p_chunk_size = int(
                            p_opts.pop("chunk_token_size", resolved_chunk_size)
                        )
                        p_blocks_path = (
                            str(parsed_data.get("blocks_path") or "").strip() or None
                        )
                        chunk_opts_str = _format_chunking_params(p_chunk_size, p_opts)
                        logger.info(f"Chunking P: {chunk_opts_str}, doc_id: {doc_id}")
                        chunking_result = chunking_by_paragraph_semantic(
                            self.tokenizer,
                            content,
                            p_chunk_size,
                            blocks_path=p_blocks_path,
                            doc_id=doc_id,
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
                        logger.info(f"Chunking R: {chunk_opts_str}, doc_id: {doc_id}")
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
                        logger.info(f"Chunking V: {chunk_opts_str}, doc_id: {doc_id}")
                        chunking_result = await chunking_by_semantic_vector(
                            self.tokenizer,
                            content,
                            v_chunk_size,
                            embedding_func=self.embedding_func,
                            **v_opts,
                        )
                    else:  # "F"
                        # F honors its own ``chunk_token_size`` override
                        # (``addon_params['chunker']['fixed_token']`` or a
                        # caller-supplied ``chunk_options``) exactly like
                        # R/V/P: pop it out of the kwargs so we don't pass it
                        # both positionally and via ``**`` splat (which would
                        # TypeError), falling back to the shared top-level
                        # resolved size when unset.
                        f_opts = dict(chunk_opts.get("fixed_token") or {})
                        f_chunk_size = int(
                            f_opts.pop("chunk_token_size", resolved_chunk_size)
                        )
                        chunk_opts_str = _format_chunking_params(f_chunk_size, f_opts)
                        logger.info(f"Chunking F: {chunk_opts_str}, doc_id: {doc_id}")
                        chunking_result = chunking_by_fixed_token(
                            self.tokenizer,
                            content,
                            f_chunk_size,
                            _emit_source_span=True,
                            **f_opts,
                        )
                else:
                    f_opts = chunk_opts.get("fixed_token") or {}
                    # Honor the F-strategy ``chunk_token_size`` override (from
                    # ``CHUNK_F_SIZE`` env or an explicit
                    # ``addon_params['chunker']['fixed_token']`` / per-doc
                    # ``chunk_options``) on this legacy path too, falling back
                    # to the shared top-level resolved size when unset.  This
                    # keeps ``LightRAG.ainsert`` — which intentionally does NOT
                    # pass a ``process_options`` selector (so the user's
                    # ``chunking_func`` still runs) — consistent with the
                    # explicit-F branch instead of silently ignoring
                    # ``fixed_token.chunk_token_size``.  ``f_opts`` is read
                    # field-by-field here (not splatted), so there is no
                    # positional/kwarg collision.
                    legacy_chunk_size = int(
                        f_opts.get("chunk_token_size", resolved_chunk_size)
                    )
                    chunk_opts_str = _format_chunking_params(
                        legacy_chunk_size,
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
                        f"Chunking F(legacy): {chunk_opts_str}, doc_id: {doc_id}"
                    )
                    from lightrag.chunker import chunking_by_token_size

                    # Only the unmodified default fixed-token chunker understands the
                    # private ``_emit_source_span`` kwarg; a user-supplied
                    # ``chunking_func`` must not receive it.
                    legacy_kwargs = {}
                    if self.chunking_func is chunking_by_token_size:
                        legacy_kwargs["_emit_source_span"] = True
                    chunking_result = self.chunking_func(
                        self.tokenizer,
                        content,
                        f_opts.get("split_by_character"),
                        f_opts.get("split_by_character_only", False),
                        f_opts.get(
                            "chunk_overlap_token_size",
                            self.chunk_overlap_token_size,
                        ),
                        legacy_chunk_size,
                        **legacy_kwargs,
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
                # parse_format=raw, which silently mislabelled lightrag docs;
                # _build_mm_chunks_from_sidecars below gates on the persisted
                # format via the sidecar presence check, so the tag must
                # reflect what was actually stored.
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
                    # Shared resolver with the parse stage (_parse_worker), so a
                    # field already stamped at PARSING re-writes to the same
                    # value here — no value jump across the transition.
                    "parse_engine": resolve_doc_status_parse_engine(
                        persisted_format, persisted_engine
                    ),
                    "chunk_method": (
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

                blocks_path = str(parsed_data.get("blocks_path") or "").strip()
                if blocks_path:
                    max_order = -1
                    for ch in chunking_result:
                        if isinstance(ch, dict) and isinstance(
                            ch.get("chunk_order_index"), int
                        ):
                            max_order = max(max_order, int(ch["chunk_order_index"]))
                    # Default to "" (no modalities) when full_docs has no
                    # ``process_options`` key for this doc: a reinsert that
                    # omits i/t/e must NOT re-index stale successful sidecars
                    # left over from an earlier multimodal run. The builder's
                    # None branch is reserved for ad-hoc callers (unit tests)
                    # that intentionally want every modality considered.
                    mm_chunks = self._build_mm_chunks_from_sidecars(
                        doc_id=doc_id,
                        file_path=file_path,
                        blocks_path=blocks_path,
                        base_order_index=max_order + 1,
                        process_options=(content_data or {}).get("process_options")
                        or "",
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

                # Backfill block provenance for F/R/V chunks (P already carries
                # sidecars; multimodal chunks too). Runs on the final, post-split
                # chunk list so each slice maps precisely to the block(s) its
                # content covers. Raises ChunkBlockMatchError -> doc FAILED when a
                # chunk cannot be located in blocks.jsonl.
                #
                # Gated to the built-in F/R/V strategies — or the legacy path only
                # when ``chunking_func`` is still the unmodified default fixed-token
                # chunker. A user-supplied ``chunking_func`` may emit summaries /
                # rewritten text that cannot be located in blocks.jsonl, which would
                # wrongly FAIL the document.
                if doc_process_opts.chunking_explicit:
                    sidecar_backfill_eligible = doc_process_opts.chunking in {
                        "F",
                        "R",
                        "V",
                    }
                else:
                    from lightrag.chunker import chunking_by_token_size

                    sidecar_backfill_eligible = (
                        self.chunking_func is chunking_by_token_size
                    )

                if blocks_path and sidecar_backfill_eligible:
                    from lightrag.sidecar import backfill_chunk_sidecars

                    backfill_chunk_sidecars(chunking_result, blocks_path)

                chunks = build_chunks_dict_from_chunking_result(
                    chunking_result, doc_id=doc_id, file_path=file_path
                )

                if not chunks:
                    logger.warning("No document chunks to process")

                process_start_time = int(time.time())

                await self._raise_if_cancelled(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                )

                # Stage 1: persist doc_status PROCESSING + chunks in parallel.
                doc_status_task = asyncio.create_task(
                    self._upsert_doc_status_transition(
                        ctx=ctx,
                        doc_id=doc_id,
                        status=DocStatus.PROCESSING,
                        status_doc=status_doc,
                        file_path=file_path,
                        extra_fields={
                            "chunks_count": len(chunks),
                            "chunks_list": list(chunks.keys()),
                        },
                        metadata_extra={
                            "process_start_time": process_start_time,
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
                    ctx=ctx,
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
                        "process_start_time": process_start_time,
                        "process_end_time": int(time.time()),
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

                    # If another in-flight document already triggered an abort
                    # (e.g. a storage flush error set cancellation_requested),
                    # do not mark PROCESSED or re-run _insert_done here: the
                    # shared flush buffer is being torn down, so re-flushing
                    # would just re-raise the same error. Bail out as cancelled
                    # so this document is FAILED and retried on the next run.
                    await self._raise_if_cancelled(
                        ctx.pipeline_status, ctx.pipeline_status_lock
                    )

                    # Flush ALL derived stores BEFORE the durable PROCESSED
                    # write. doc_status backends persist immediately on
                    # upsert, so writing PROCESSED first opens a crash window
                    # where the status is durable but the graph/vector/chunk
                    # data is not — a false PROCESSED that recovery can never
                    # detect (issue #3400: status is the commit record).
                    await self._insert_done()

                    # A sibling document's flush error may have aborted the
                    # batch while our flush ran; do not mark PROCESSED during
                    # teardown — bail out so this document is FAILED and
                    # retried on the next run.
                    await self._raise_if_cancelled(
                        ctx.pipeline_status, ctx.pipeline_status_lock
                    )

                    process_end_time = int(time.time())
                    await self._upsert_doc_status_transition(
                        ctx=ctx,
                        doc_id=doc_id,
                        status=DocStatus.PROCESSED,
                        status_doc=status_doc,
                        file_path=file_path,
                        extra_fields={
                            "chunks_count": len(chunks),
                            "chunks_list": list(chunks.keys()),
                        },
                        metadata_extra={
                            "process_start_time": process_start_time,
                            "process_end_time": process_end_time,
                            **extraction_meta,
                        },
                    )

                    async with ctx.pipeline_status_lock:
                        log_message = (
                            f"Completed processing file "
                            f"{current_file_number}/{ctx.total_files}: "
                            f"{file_path}"
                        )
                        logger.info(log_message)
                        ctx.pipeline_status["latest_message"] = log_message
                        append_pipeline_history(ctx.pipeline_status, log_message)

                except Exception as e:
                    # A storage flush failure (raised by _insert_done) is not
                    # attributable to this document: index_done_callback flushes
                    # a buffer shared across concurrently-processed files. We
                    # cannot tell whose record failed, so continuing risks
                    # marking other in-flight files PROCESSED with missing data.
                    # Abort the whole batch via the cooperative cancellation
                    # flag, tagging it as an internal error with the driver name
                    # and root cause so it is distinguishable from a user cancel.
                    if isinstance(e, IndexFlushError):
                        async with ctx.pipeline_status_lock:
                            ctx.pipeline_status.update(
                                {
                                    "cancellation_requested": True,
                                    "cancellation_reason": "internal_error",
                                    "cancellation_detail": (
                                        f"{e.storage_name}[{e.namespace}]: "
                                        f"{e.__cause__}"
                                    ),
                                }
                            )
                        logger.error(
                            f"Aborting pipeline batch due to storage flush error: {e}"
                        )
                    await self._finalize_doc_failure(
                        ctx=ctx,
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
                            "process_start_time": process_start_time,
                            "process_end_time": int(time.time()),
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
                and content_data.get("sidecar_location")
            )
            or (
                content_data.get("parse_format") == FULL_DOCS_FORMAT_RAW
                and (content_data.get("content") or "").strip()
            )
        )
        if not content_already_extracted:
            return

        from lightrag.parser.routing import normalize_parser_engine

        intended_engine, _ = resolve_file_parser_directives(file_path)
        # ``parse_engine`` may carry encoded params; compare bare engine names.
        stored_engine = normalize_parser_engine(content_data.get("parse_engine"))
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
                append_pipeline_history(pipeline_status, log_message)

        # Order-preserving dedup; keep a list so it satisfies the storage delete
        # contract (``delete(ids: list[str])``) when passed down to purge.
        stored_chunk_ids = list(
            dict.fromkeys(
                chunk_id
                for chunk_id in (status_doc.chunks_list or [])
                if isinstance(chunk_id, str) and chunk_id
            )
        )
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
            append_pipeline_history(pipeline_status, log_message)
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
        ctx: "_BatchRunContext",
        extra_fields: dict[str, Any] | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        """Single source of truth for doc_status state-transition upserts.

        Mirrors the field set used at every PARSING / ANALYZING / PROCESSING /
        PROCESSED / FAILED transition.  ``extra_fields`` carries
        ``chunks_count`` / ``chunks_list`` / ``error_msg``; ``metadata_extra``
        is forwarded to ``doc_status_transition_metadata`` so carry-over
        fields (e.g. ``process_options``) survive every state change.

        Owner-checked (LR2 §7.7 items 3/4/7): every worker status write verifies
        its run still owns ``busy`` immediately before writing, and a run whose
        ownership was reclaimed DISCARDS the result with a warning instead of
        writing. Without this, a storage/LLM response arriving late in a run that
        was already declared dead could stamp a final status over the new owner's
        work — including re-FAILING a document the manual exclusive reset had just
        moved to PENDING, which would ACK the retry with the document still
        failed. ``ctx`` is required so a new call site has to state which run the
        write belongs to.
        """
        if not await self._still_run_owner(ctx):
            logger.warning(
                f"[stale-writer] Discarded {getattr(status, 'value', status)} "
                f"status write for {doc_id} ({file_path}): this run no longer "
                f"owns the pipeline (ownership was reclaimed), so the result is "
                f"late and must not overwrite the current owner's state"
            )
            return
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

    @staticmethod
    def _cancellation_label(pipeline_status: dict) -> str:
        """Human-readable cancel cause: internal error (with detail) vs user.

        Callers building cancellation messages use this so an internal abort
        (e.g. a storage flush failure) is not mislabeled as a user cancel.
        """
        if pipeline_status.get("cancellation_reason") == "internal_error":
            detail = pipeline_status.get("cancellation_detail") or "unknown"
            return f"Cancelled by internal error: {detail}"
        return "User cancelled"

    @staticmethod
    def _internal_halt_message(label: str) -> str:
        """Actionable halt message for an internal-error abort.

        Shared by the loop-top cancellation handler and the finally cleanup so
        the same wording surfaces whichever exit path the batch takes.
        """
        return (
            f"Pipeline halted on internal storage error ({label}). Resolve the "
            f"storage issue and restart processing; affected documents remain "
            f"queued (PENDING/FAILED)."
        )

    async def _cancellation_requested(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """Read-only cancellation check.

        Use this when a worker wants to branch on the flag (e.g. drain a queue
        item) instead of raising. Callers that prefer the exception style
        should use :meth:`_raise_if_cancelled` instead.
        """
        async with pipeline_status_lock:
            return bool(pipeline_status.get("cancellation_requested", False))

    async def _persist_llm_response_cache_best_effort(
        self,
        *,
        stage_label: str,
        doc_id: str,
    ) -> None:
        """Commit pending LLM cache entries without failing document work.

        Cache writes are valuable for cost and repeatability, but they are not
        a prerequisite for a parser or multimodal result that has otherwise
        succeeded. Stage-boundary callers use this narrow commit instead of
        ``_insert_done()``, which would flush every KG/vector storage too.
        """
        if self.llm_response_cache is None:
            return
        try:
            await self.llm_response_cache.index_done_callback()
        except Exception as persist_error:
            logger.error(
                "Failed to persist LLM cache after %s for d-id %s: %s",
                stage_label,
                doc_id,
                persist_error,
            )

    async def _mark_doc_cancelled_in_stage(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        stage_label: str,
        ctx: "_BatchRunContext",
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Mark a queued document FAILED with a 'User cancelled' message.

        Used by the PARSE and ANALYZE workers, which do not have the
        chunks-snapshot / pending-tasks bookkeeping that
        :meth:`_finalize_doc_failure` carries for the PROCESS stage. Also
        flushes the LLM response cache so any cache_ids written by completed
        sibling tasks (e.g. successful multimodal items inside a doc that is
        being cancelled) survive a server restart.
        """
        status_snapshot = pipeline_status.copy()
        error_msg = (
            f"{self._cancellation_label(status_snapshot)} during "
            f"{stage_label}: {file_path}"
        )
        logger.warning(error_msg)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = error_msg
            append_pipeline_history(pipeline_status, error_msg)
        await self._persist_llm_response_cache_best_effort(
            stage_label=f"{stage_label} cancellation",
            doc_id=doc_id,
        )
        try:
            await self._upsert_doc_status_transition(
                ctx=ctx,
                doc_id=doc_id,
                status=DocStatus.FAILED,
                status_doc=status_doc,
                file_path=file_path,
                extra_fields={"error_msg": error_msg},
            )
        except Exception as exc:
            logger.error(f"Failed to mark cancelled doc {doc_id} as FAILED: {exc}")

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
        ctx: "_BatchRunContext",
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Common epilogue for an extract / merge stage failure.

        Logs the error (or cancellation), cancels any pending stage tasks,
        flushes the LLM response cache, and writes a FAILED status row that
        preserves the failed chunks snapshot and processing-time metadata.
        """
        if isinstance(error, PipelineCancelledException):
            cancel_label = self._cancellation_label(pipeline_status.copy())
            # The cancel exceptions raised by the merge/summary stages hardcode a
            # generic "User cancelled during <stage>" message. When the batch was
            # actually aborted by an internal error (e.g. a storage outage), that
            # mislabels the cause. Swap the generic prefix for the reason-aware
            # label so doc_status records "Cancelled by internal error: <detail>
            # during <stage>" rather than "User cancelled during <stage>".
            raw = str(error)
            if raw.startswith("User cancelled"):
                doc_error_msg = f"{cancel_label}{raw[len('User cancelled') :]}"
            elif raw:
                doc_error_msg = f"{cancel_label}: {raw}"
            else:
                doc_error_msg = cancel_label
            if stage_label == "merge":
                error_msg = (
                    f"{cancel_label} during merge {current_file_number}/"
                    f"{total_files}: {file_path}"
                )
            else:
                error_msg = (
                    f"{cancel_label} {current_file_number}/{total_files}: {file_path}"
                )
            logger.warning(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                append_pipeline_history(pipeline_status, error_msg)
        else:
            doc_error_msg = str(error)
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
                append_pipeline_history(pipeline_status, traceback.format_exc())
                append_pipeline_history(pipeline_status, error_msg)

        for task in pending_tasks:
            if task and not task.done():
                task.cancel()

        await self._persist_llm_response_cache_best_effort(
            stage_label=f"{stage_label} failure",
            doc_id=doc_id,
        )

        failed_chunks_list, failed_chunks_count = failed_chunks_snapshot
        await self._upsert_doc_status_transition(
            ctx=ctx,
            doc_id=doc_id,
            status=DocStatus.FAILED,
            status_doc=status_doc,
            file_path=file_path,
            extra_fields={
                "error_msg": doc_error_msg,
                "chunks_count": failed_chunks_count,
                "chunks_list": failed_chunks_list,
            },
            metadata_extra=metadata_extra,
        )

    # ============================================================
    # Parser internals
    # ============================================================

    async def _persist_parsed_full_docs(
        self,
        doc_id: str,
        record: dict[str, Any],
    ) -> str | None:
        """Write a parse-result record to ``full_docs`` and sync ``content_hash``.

        Computes ``content_hash`` from the actual extracted body so subsequent
        ``get_doc_by_content_hash`` lookups can dedupe across pending_parse
        records that did not have a hash at enqueue time. Also patches the
        existing ``doc_status`` row so both storages stay aligned on
        ``content_hash``.

        The original ``pending_parse`` record carries metadata seeded at
        enqueue time (``process_options`` etc.) that downstream stages still
        need after parsing. ``full_docs`` upserts overwrite the entire row,
        so we merge the existing record with the new ``record`` payload
        before upserting: fresh fields from ``record`` (``content`` /
        ``parse_format`` / ``sidecar_location`` / ``parse_engine`` /
        ``update_time``) take precedence, while pre-existing fields are
        preserved.
        """
        # Strip C0 control/separator chars (incl. \x1c-\x1f FS/GS/RS/US) from the
        # parsed body before it lands in full_docs — this is the single
        # convergence point for every parser engine's persist. For RAW (legacy)
        # the full_docs content IS the chunk source, so this guarantees clean
        # chunks; for sidecar engines it is an idempotent backstop (the sidecar
        # writer already cleaned the same text). Done before content_hash so the
        # dedup hash is computed on the sanitized body. No-op for clean input.
        record_content = record.get("content")
        if isinstance(record_content, str):
            record = {**record, "content": strip_control_characters(record_content)}

        fmt = record.get("parse_format")
        content_hash: str | None = None
        # Hash the bare merged text (after stripping the ``{{LRdoc}}`` marker
        # for lightrag-format) so cross-filename dedup fires regardless of
        # whether the same body was ingested as raw text or via a sidecar.
        # ``strip_lightrag_doc_prefix`` is a no-op for non-lightrag formats.
        if fmt in (FULL_DOCS_FORMAT_RAW, FULL_DOCS_FORMAT_LIGHTRAG):
            content_hash = compute_text_content_hash(
                strip_lightrag_doc_prefix(record.get("content") or "", fmt)
            )

        # Owner-checked like every other write a worker makes (LR2 §7.7 items
        # 3/4/7). This one is reached from inside a parser via ``ctx.rag``, so it
        # cannot take the batch context as an argument without leaving every
        # third-party parser unguarded; ``_run_pipeline_batch`` publishes the
        # context on the instance instead (see there). A parse result that comes
        # back after this run's ownership was reclaimed must write NEITHER store:
        # the full_docs body would overwrite content the new owner re-parsed, and
        # the doc_status patch below would stamp a row the new owner now owns.
        run_ctx = getattr(self, "_active_run_ctx", None)
        if run_ctx is not None and not await self._still_run_owner(run_ctx):
            logger.warning(
                f"[stale-writer] Discarded parsed content for {doc_id}: this run "
                "no longer owns the pipeline"
            )
            return None

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
            # Targeted field update (LR2 §5.6), not a read-modify-write upsert:
            # only two scalars change, and round-tripping the whole row would drag
            # this document's ``chunks_list`` through memory for them.
            # ``missing_ok`` keeps the previous behaviour of silently skipping a
            # row that is already gone.
            await self.doc_status.update_doc_status_fields(
                doc_id,
                {
                    "content_hash": content_hash,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                missing_ok=True,
            )
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
        ctx: "_BatchRunContext | None" = None,
    ) -> bool:
        """Mark post-parse content duplicates and stop further processing.

        Owner-checked like every other worker status write (LR2 §7.7 items 3/4):
        this one does not go through ``_upsert_doc_status_transition``, so it
        verifies ownership itself before the FAILED-duplicate upsert.

        Marking this document a duplicate REMOVES it from the primary candidates
        of its canonical source, which makes this one of the writers LR2 §5.5
        requires a source-conflict commit to be mutually exclusive with. So the
        decision and the write happen together under
        :func:`source_candidate_set_lock` — the same keyed lock the repair holds —
        and the match is re-read inside it. That is what linearizes the two
        operations instead of merely narrowing the window between them:

        * marking wins the lock → the repair then finds this document is no longer
          a candidate and refuses BEFORE demoting anything (409), rather than
          committing demotions and reporting a 503 about storage that was fine;
        * the repair wins → it commits, verifies the key is Unique and releases;
          this marking then re-reads and, because the row it matched now points at
          this document, the lookup skips it (see ``get_doc_by_content_hash``'s
          exclusion contract) and the marking stands down. Should the content
          genuinely belong to a THIRD document, the marking proceeds and the key
          becomes Absent — an ordinary content-dedup state transition AFTER a
          completed operation, not a half-applied one.

        The probe outside the lock keeps the common case (not a duplicate) exactly
        as cheap as before: no keyed lock is acquired for a document that has no
        content twin.
        """
        if not content_hash:
            return False

        match = await get_duplicate_doc_by_content_hash(
            self.doc_status, content_hash, doc_id
        )
        if not match:
            return False

        async with source_candidate_set_lock(self.workspace, file_path):
            match = await get_duplicate_doc_by_content_hash(
                self.doc_status, content_hash, doc_id
            )
            if not match:
                logger.info(
                    f"Duplicate marking for {doc_id} ({file_path}) stood down: the "
                    "content's other holder is gone or now points at this document "
                    "(a source-conflict repair kept it as the primary)"
                )
                return False

            if ctx is not None and not await self._still_run_owner(ctx):
                logger.warning(
                    f"[stale-writer] Discarded duplicate marking for {doc_id} "
                    f"({file_path}): this run no longer owns the pipeline"
                )
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
                            f"[DUPLICATE:content_hash] Original document: "
                            f"{original_doc_id}"
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

        source_path = _call_source_file_resolver(
            self,
            file_path,
            source_file=_read_source_file(content_data),
        )
        archived = await archive_source_after_full_docs_sync(source_path)
        archive_msg = f"; archived to {archived}" if archived else ""
        warning = f"Duplicate content skipped after parsing: {file_path}{archive_msg}"
        logger.warning(warning)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = warning
                append_pipeline_history(pipeline_status, warning)
        return True

    def _resolve_source_file_for_parser(
        self,
        file_path: str,
        *,
        source_file: str | None = None,
        parser_engine: str | None = None,
    ) -> str:
        """Resolve a readable source file path for parser upload.

        ``file_path`` is the canonical stored basename. Pending-parse records
        may also carry ``source_file`` with the real uploaded/scanned
        basename, including parser hints.
        """
        candidates: list[Path] = []
        roots: list[Path] = []

        def _add_candidate(path_value: Any) -> None:
            raw = str(path_value or "").strip()
            if not raw:
                return
            path = Path(raw)
            candidates.append(path)
            if path.parent != Path("."):
                roots.append(path.parent)
                roots.append(path.parent / PARSED_DIR_NAME)
                candidates.append(path.parent / PARSED_DIR_NAME / path.name)

        _add_candidate(file_path)

        p = Path(file_path)
        name = p.name
        source_name = Path(str(source_file or "").strip()).name
        input_path = input_dir_path()
        # API ``DocumentManager`` scopes its input dir to
        # ``<base_input_dir>/<workspace>/`` (see DocumentManager.__init__);
        # check that location first so files uploaded into a workspace
        # subdirectory resolve correctly. ``self.workspace`` is empty when
        # no workspace is configured, in which case these candidates
        # collapse to the base candidates that follow.
        workspace = getattr(self, "workspace", "") or ""
        if workspace:
            candidates.append(input_path / workspace / name)
            candidates.append(input_path / workspace / PARSED_DIR_NAME / name)
            roots.append(input_path / workspace)
            roots.append(input_path / workspace / PARSED_DIR_NAME)
        candidates.append(input_path / name)
        candidates.append(input_path / PARSED_DIR_NAME / name)
        roots.append(input_path)
        roots.append(input_path / PARSED_DIR_NAME)

        # Common local defaults used by API server.
        cwd = Path.cwd()
        if workspace:
            candidates.append(cwd / "inputs" / workspace / name)
            candidates.append(cwd / "inputs" / workspace / PARSED_DIR_NAME / name)
            roots.append(cwd / "inputs" / workspace)
            roots.append(cwd / "inputs" / workspace / PARSED_DIR_NAME)
        candidates.extend(
            [
                cwd / "inputs" / name,
                cwd / "inputs" / PARSED_DIR_NAME / name,
                cwd / PARSED_DIR_NAME / name,
            ]
        )
        roots.extend(
            [
                cwd / "inputs",
                cwd / "inputs" / PARSED_DIR_NAME,
                cwd / PARSED_DIR_NAME,
            ]
        )

        if source_name:
            candidates = [root / source_name for root in roots] + candidates

        seen_candidates: set[Path] = set()
        for candidate in candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            if candidate.exists() and candidate.is_file():
                return str(candidate)

        canonical_name = normalize_document_file_path(file_path)
        if has_known_document_source(canonical_name):
            matches: list[Path] = []
            seen_roots: set[Path] = set()
            for root in roots:
                if root in seen_roots:
                    continue
                seen_roots.add(root)
                if not root.exists() or not root.is_dir():
                    continue
                for candidate in sorted(root.iterdir(), key=lambda item: item.name):
                    if (
                        candidate.is_file()
                        and normalize_document_file_path(candidate.name)
                        == canonical_name
                    ):
                        matches.append(candidate)

            if source_name:
                for candidate in matches:
                    if candidate.name == source_name:
                        return str(candidate)
            if parser_engine:
                from lightrag.parser.routing import filename_parser_directives

                for candidate in matches:
                    hinted_engine, _ = filename_parser_directives(candidate.name)
                    if hinted_engine == parser_engine:
                        return str(candidate)
            if matches:
                return str(matches[0])
        return file_path

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
        pipeline_status: dict | None = None,
        pipeline_status_lock: Any | None = None,
    ) -> dict[str, Any]:
        """Phase 2: Multimodal analysis (VLM). Writes llm_analyze_result to LightRAG Document.

        Per-document ``i`` / ``t`` / ``e`` flags from
        ``full_docs.process_options`` decide which modalities are sent to the
        VLM.  Sidecars are always written by the parser regardless of these
        flags so toggling options later does not require re-parsing — only
        the ``llm_analyze_result`` payload is gated here.

        Per-item ``llm_analyze_result`` is recomputed and overwritten on each
        run for enabled modalities.  This lets operators fix VLM/EXTRACT
        configuration or prompt limits and retry without manually clearing
        prior failure markers from the sidecar.

        Args:
            process_options: Optional override that bypasses the
                ``full_docs.process_options`` lookup; primarily used by unit
                tests that exercise the VLM analysis path without going
                through the enqueue pipeline.
        """
        from lightrag.parser.routing import parse_process_options

        blocks_path = parsed_data.get("blocks_path")
        if not blocks_path:
            parsed_data["analyzing_stage_skipped"] = True
            return parsed_data

        block_file = Path(blocks_path)
        if not block_file.exists():
            parsed_data["analyzing_stage_skipped"] = True
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
            parsed_data["analyzing_stage_skipped"] = True
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
                f"[analyze_multimodal] {','.join(opt_in_missing)} sidecar empty: {doc_id}"
            )

        # Backfill sidecar `surrounding` for the enabled modalities just
        # before VLM consumption.  Universal coverage: native, MinerU,
        # Docling, and pre-existing LightRAG documents reused from disk
        # all go through this single entrypoint.  Idempotent: re-runs
        # overwrite with stable output given unchanged block content.
        enabled_modalities = {
            mod
            for mod, on in (
                ("drawings", process_opts.images),
                ("tables", process_opts.tables),
                ("equations", process_opts.equations),
            )
            if on
        }
        tokenizer = getattr(self, "tokenizer", None)
        if enabled_modalities and tokenizer is not None:
            try:
                from lightrag.multimodal_context import (
                    enrich_sidecars_with_surrounding,
                )

                enrich_counts = enrich_sidecars_with_surrounding(
                    blocks_path=str(block_file),
                    enabled_modalities=enabled_modalities,
                    tokenizer=tokenizer,
                )
                if any(enrich_counts.values()):
                    logger.info(
                        "[analyze_multimodal] "
                        + ", ".join(f"{k}={v}" for k, v in enrich_counts.items() if v)
                        + f" surrounding backfilled: {doc_id}"
                    )
            except Exception as enrich_err:
                logger.warning(
                    f"[analyze_multimodal] surrounding enrichment failed for "
                    f"d-id: {doc_id}, file: {file_path}: {enrich_err}"
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
                table_content_format_label,
            )
            from lightrag.constants import (
                DEFAULT_MM_ANALYSIS_PRIORITY,
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
            max_image_bytes, min_image_pixel = _vlm_image_budget_limits()
            # Multimodal analysis shares the entity-extraction cache flag
            # (both run with mode="default" — see handle_cache short-circuit
            # in lightrag.utils).  When the flag is off we must NOT save the
            # response either, otherwise stale cache entries would still
            # accumulate while reads are blocked.  cache_id attachment to
            # the sidecar item.llm_cache_list is likewise gated so a
            # disabled cache does not seed cache-cleanup metadata that
            # corresponds to entries that were never persisted.
            analysis_cache_enabled = bool(
                global_config.get("enable_llm_cache_for_entity_extract")
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
                """Tolerant JSON object recovery via :func:`tolerant_load_json_dict`.

                String values are passed through ``repair_vlm_json_escape_damage``,
                the single choke point both fresh responses and cache hits flow
                through: models writing LaTeX inside JSON strings routinely
                under-escape backslashes (``"\\frac"`` is valid JSON meaning form
                feed + ``rac``).
                """
                obj = tolerant_load_json_dict(text)
                if not obj:
                    return {}
                return repair_vlm_json_escape_damage_nested(obj)

            class _MMJSONConformanceError(Exception):
                """Raised only when an LLM/VLM response violates MM JSON schema."""

            def _required_json_string(
                parsed: dict[str, Any], prefix: str, field: str
            ) -> str:
                value = parsed.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise _MMJSONConformanceError(
                        f"{prefix}: missing or invalid field '{field}'"
                    )
                return value.strip()

            def _validate_drawing_analysis(
                item_id: str, parsed: dict[str, Any]
            ) -> dict[str, str]:
                prefix = f"drawings/{item_id}"
                name = _required_json_string(parsed, prefix, "name")
                description = _required_json_string(parsed, prefix, "description")
                type_value = _required_json_string(parsed, prefix, "type")
                if type_value not in _IMAGE_TYPE_VALUES:
                    type_value = IMAGE_TYPE_FALLBACK
                return {
                    "name": name,
                    "type": type_value,
                    "description": description,
                }

            def _validate_text_analysis(
                kind: str, item_id: str, parsed: dict[str, Any]
            ) -> dict[str, str]:
                prefix = f"{kind}/{item_id}"
                result_obj = {
                    "name": _required_json_string(parsed, prefix, "name"),
                    "description": _required_json_string(parsed, prefix, "description"),
                }
                if kind == "equation":
                    result_obj["equation"] = _required_json_string(
                        parsed, prefix, "equation"
                    )
                return result_obj

            async def _run_json_conformance_retry(
                prefix: str,
                cached: tuple[str, int] | None,
                call_model_once,
                validate_result,
            ) -> tuple[dict[str, str], str, bool]:
                """Retry once only for JSON/schema conformance failures.

                The first attempt may use the analysis cache.  If that cached
                response is malformed, bypass the cache on the retry so a good
                fresh response can overwrite the same cache key after success.
                """

                def _attempt(raw: Any, fresh: bool) -> tuple[dict[str, str], str, bool]:
                    # Keep str input as-is: str(raw) would rebuild a plain str
                    # and drop the TruncatedResponse marker the cache guards
                    # below need to observe.
                    text = raw if isinstance(raw, str) else str(raw)
                    return validate_result(_json_extract(text)), text, fresh

                use_cached_response = cached is not None
                first_text = (
                    cached[0] if use_cached_response else await call_model_once()
                )
                try:
                    return _attempt(first_text, fresh=not use_cached_response)
                except _MMJSONConformanceError as exc:
                    source = "cache" if use_cached_response else "model"
                    logger.warning(
                        f"[analyze_multimodal] {prefix}: invalid JSON schema "
                        f"from {source}; retrying once: {exc} "
                        f"(response snippet: {str(first_text)[:200]!r})"
                    )

                try:
                    return _attempt(await call_model_once(), fresh=True)
                except _MMJSONConformanceError as exc:
                    raise MultimodalAnalysisError(str(exc)) from exc

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
                            f"image width or height is smaller than {min_image_pixel}px"
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

                async def _call_vlm_once() -> str:
                    try:
                        return await use_vlm_func(
                            prompt,
                            stream=False,
                            image_inputs=[img_payload],
                            response_format={"type": "json_object"},
                            _priority=DEFAULT_MM_ANALYSIS_PRIORITY,
                        )
                    except PipelineCancelledException:
                        raise
                    except Exception as exc:
                        raise MultimodalAnalysisError(
                            f"drawings/{item_id}: VLM call failed: {exc}"
                        ) from exc

                analysis_fields, result_text, fresh = await _run_json_conformance_retry(
                    f"drawings/{item_id}",
                    cached,
                    _call_vlm_once,
                    lambda parsed: _validate_drawing_analysis(item_id, parsed),
                )
                cache_id_to_attach: str | None = None
                if fresh and analysis_cache_enabled:
                    if is_truncated_response(result_text):
                        # A token-limit-truncated VLM response that still
                        # parsed must not be persisted: the cache would replay
                        # the partial analysis on every later run, even once a
                        # larger token budget would have completed it.
                        logger.warning(
                            f"[analyze_multimodal] drawings/{item_id}: skipping "
                            "analysis cache write for token-limit-truncated "
                            "VLM response"
                        )
                    else:
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
                        cache_id_to_attach = cache_id
                elif not fresh:
                    # Cache hit: the entry exists, so attaching its id is
                    # safe (and necessary for document-delete cleanup).
                    cache_id_to_attach = cache_id
                return (
                    {
                        "name": analysis_fields["name"],
                        "type": analysis_fields["type"],
                        "description": analysis_fields["description"],
                        "analyze_time": int(time.time()),
                        "status": "success",
                        "message": "",
                    },
                    cache_id_to_attach,
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
                    if kind == "table":
                        # Defensive fallback for sidecars that still carry
                        # empty-bodied table items (e.g. produced by an older
                        # parser run, or by a parser that doesn't filter
                        # MinerU-style misidentified blanks). Don't abort the
                        # whole worker — record the skip and move on.
                        logger.warning(
                            f"[analyze_multimodal] table/{item_id}: missing "
                            f"table content; skipping analysis ({file_path})"
                        )
                        return (
                            _skipped_result("missing table content"),
                            None,
                        )
                    raise MultimodalAnalysisError(
                        f"{kind}/{item_id}: missing {kind} content"
                    )
                template = MULTIMODAL_PROMPTS[f"{kind}_analysis"]

                # A table item written by the sidecar writer ALWAYS carries a
                # valid ``format``; a missing/unknown one means a corrupt or
                # incompatible sidecar — fail loudly rather than guess.
                content_format = ""
                if kind == "table":
                    fmt = (item.get("format") or "").strip().lower()
                    if fmt not in ("html", "json"):
                        raise MultimodalAnalysisError(
                            f"table/{item_id}: missing or invalid table format "
                            f"{item.get('format')!r} ({file_path})"
                        )
                    content_format = table_content_format_label(fmt)

                def _render(content_value: str) -> str:
                    return template.format(
                        language=language,
                        content=content_value,
                        content_format=content_format,
                        captions=_captions_value(item),
                        footnotes=_footnotes_value(item),
                        leading=_surrounding_value(item, "leading"),
                        trailing=_surrounding_value(item, "trailing"),
                        item_id=item_id,
                        file_path=file_path,
                    )

                prompt = _render(content_text)

                # Cap the EXTRACT prompt at MAX_EXTRACT_INPUT_TOKENS by
                # trimming the (typically huge) sidecar `content` field — the
                # other slots (surrounding/captions/footnotes) already have
                # their own per-field caps upstream.  The cap is resolved
                # from the env var (falling back to
                # DEFAULT_MAX_EXTRACT_INPUT_TOKENS) so deployments can tune
                # it for their model's context window.
                tokenizer = getattr(self, "tokenizer", None)
                if tokenizer is not None:
                    from lightrag.constants import (
                        DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
                        DEFAULT_MM_EXTRACT_CONTENT_MIN_TOKENS,
                    )
                    from lightrag.multimodal_context import (
                        _resolve_surrounding_budget,
                        trim_content_to_budget,
                    )

                    SAFETY_BUFFER = 256
                    max_extract_tokens = get_env_value(
                        "MAX_EXTRACT_INPUT_TOKENS",
                        DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
                        int,
                    )
                    content_min_tokens = get_env_value(
                        "MM_EXTRACT_CONTENT_MIN_TOKENS",
                        DEFAULT_MM_EXTRACT_CONTENT_MIN_TOKENS,
                        int,
                    )
                    # One-time operator heads-up when the configured caps alone
                    # (before the per-item template frame) already starve
                    # content below its floor.  The per-item guard below is the
                    # real enforcement; this only surfaces gross misconfig once
                    # instead of as a burst of per-item failures.  Skip it when
                    # the cap is disabled (max_extract_tokens <= 0): enforcement
                    # is bypassed too, so no item can fail and the warning would
                    # be false.
                    if max_extract_tokens > 0:
                        leading_cap, trailing_cap = _resolve_surrounding_budget(
                            None, None
                        )
                        _warn_content_budget_structurally_starved(
                            max_extract=max_extract_tokens,
                            leading_cap=leading_cap,
                            trailing_cap=trailing_cap,
                            frame_reserve=SAFETY_BUFFER,
                            content_min=content_min_tokens,
                        )
                    total_tokens = len(tokenizer.encode(prompt))
                    if max_extract_tokens > 0 and total_tokens > max_extract_tokens:
                        frame_tokens = len(tokenizer.encode(_render("")))
                        content_budget = (
                            max_extract_tokens - frame_tokens - SAFETY_BUFFER
                        )
                        if content_budget <= 0:
                            # The prompt template alone (with empty content)
                            # already exceeds the cap — no content trim can
                            # bring the request under the limit.  Fail this
                            # item rather than handing the LLM a payload we
                            # know will trigger ``context_length_exceeded``.
                            # Operators must raise MAX_EXTRACT_INPUT_TOKENS
                            # above the template frame for analysis to
                            # succeed; the document is reprocessable
                            # idempotently once the cap is widened.
                            raise MultimodalAnalysisError(
                                f"{kind}/{item_id}: prompt frame "
                                f"({frame_tokens} tokens) exceeds "
                                f"MAX_EXTRACT_INPUT_TOKENS "
                                f"({max_extract_tokens}); raise the cap"
                            )
                        if content_budget < content_min_tokens:
                            # Budget is positive but too thin to ground a useful
                            # analysis: trimming here would hand the LLM a
                            # near-empty stub (a few chars of a table body /
                            # equation), wasting the call and polluting the
                            # graph with a hallucinated description.  Fail the
                            # item loudly instead — it is reprocessable
                            # idempotently once the budget is widened.  Input
                            # mirror of the output-side floor
                            # DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS.
                            raise MultimodalAnalysisError(
                                f"{kind}/{item_id}: content budget "
                                f"({content_budget} tokens) is below the "
                                f"minimum ({content_min_tokens}); the prompt "
                                f"frame ({frame_tokens} tokens: template + "
                                f"surrounding + captions + footnotes) consumed "
                                f"most of MAX_EXTRACT_INPUT_TOKENS "
                                f"({max_extract_tokens}), leaving too little "
                                f"room for content. Raise "
                                f"MAX_EXTRACT_INPUT_TOKENS or lower "
                                f"SURROUNDING_LEADING_MAX_TOKENS / "
                                f"SURROUNDING_TRAILING_MAX_TOKENS / "
                                f"MM_EXTRACT_CONTENT_MIN_TOKENS"
                            )
                        trimmed, was_trimmed = trim_content_to_budget(
                            content_text,
                            kind=f"{kind}s",
                            max_tokens=content_budget,
                            tokenizer=tokenizer,
                        )
                        if was_trimmed:
                            prompt = _render(trimmed)
                            logger.warning(
                                f"[analyze_multimodal] {kind}/{item_id} "
                                f"content trimmed (prompt {total_tokens} "
                                f"→ fit {max_extract_tokens}, "
                                f"content_budget={content_budget})"
                            )
                        # Post-trim hard guard: ``trim_content_to_budget``
                        # is constrained by ``content_budget`` so the final
                        # prompt should fit within ``max_extract_tokens``;
                        # defend against tokenizer rounding / future template
                        # changes that could push it over.  Refuse the call
                        # rather than send an over-cap prompt to the LLM.
                        final_tokens = len(tokenizer.encode(prompt))
                        if final_tokens > max_extract_tokens:
                            raise MultimodalAnalysisError(
                                f"{kind}/{item_id}: trimmed prompt "
                                f"({final_tokens} tokens) still exceeds "
                                f"MAX_EXTRACT_INPUT_TOKENS "
                                f"({max_extract_tokens})"
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

                async def _call_extract_once() -> str:
                    try:
                        return await use_extract_func(
                            prompt,
                            stream=False,
                            response_format={"type": "json_object"},
                            _priority=DEFAULT_MM_ANALYSIS_PRIORITY,
                        )
                    except PipelineCancelledException:
                        raise
                    except Exception as exc:
                        raise MultimodalAnalysisError(
                            f"{kind}/{item_id}: EXTRACT call failed: {exc}"
                        ) from exc

                analysis_fields, result_text, fresh = await _run_json_conformance_retry(
                    f"{kind}/{item_id}",
                    cached,
                    _call_extract_once,
                    lambda parsed: _validate_text_analysis(kind, item_id, parsed),
                )
                result_obj: dict[str, Any] = {
                    "name": analysis_fields["name"],
                    "description": analysis_fields["description"],
                    "analyze_time": int(time.time()),
                    "status": "success",
                    "message": "",
                }
                if kind == "equation":
                    result_obj["equation"] = analysis_fields["equation"]
                cache_id_to_attach: str | None = None
                if fresh and analysis_cache_enabled:
                    if is_truncated_response(result_text):
                        logger.warning(
                            f"[analyze_multimodal] {kind}/{item_id}: skipping "
                            "analysis cache write for token-limit-truncated "
                            "EXTRACT response"
                        )
                    else:
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
                        cache_id_to_attach = cache_id
                elif not fresh:
                    # Cache hit path (handle_cache already gated by flag):
                    # safe to surface the existing cache_id for cleanup.
                    cache_id_to_attach = cache_id
                return (result_obj, cache_id_to_attach)

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

            async def _run_with_progress_log(coro, kind: str, item_id: str):
                """Append per-item completion log to pipeline_status the moment
                this single ``_analyze_*`` task finishes — not after the whole
                ``asyncio.gather`` batch returns — so the UI sees each
                drawing/table/equation result land in real time.

                Skipped items are demoted to debug-only logs and do NOT write
                pipeline_status — benign skips (image too small / wrong format
                / missing table body) otherwise flood the UI history for docs
                with many items. The per-item ``llm_analyze_result.message``
                still records why the item was skipped."""
                try:
                    result = await coro
                except Exception:
                    log_message = f"Analyzing {kind}/{item_id}: failed"
                    logger.warning(log_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = log_message
                            append_pipeline_history(pipeline_status, log_message)
                    raise
                result_obj = result[0] if isinstance(result, tuple) else {}
                is_success = (
                    isinstance(result_obj, dict)
                    and result_obj.get("status") == "success"
                )
                if is_success:
                    log_message = f"Analyzing  {kind}/{item_id}: ok"
                    logger.info(log_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = log_message
                            append_pipeline_history(pipeline_status, log_message)
                else:
                    logger.debug(f"Analyzing  {kind}/{item_id}: skipped")
                return result

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
            start_logged = False
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

                if (
                    items
                    and not start_logged
                    and pipeline_status is not None
                    and pipeline_status_lock is not None
                ):
                    async with pipeline_status_lock:
                        log_message = f"Analyzing multimodal: {doc_id}"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        append_pipeline_history(pipeline_status, log_message)
                    start_logged = True

                # Pre-schedule cancellation check: if the user cancelled
                # between _analyze_worker's boundary check and the moment
                # we are about to spawn VLM tasks for this sidecar, raise
                # here so no item task ever runs. Without this we'd briefly
                # create tasks and then cancel them on the very first poll
                # iteration — wasteful and harder to reason about.
                if pipeline_status is not None and pipeline_status_lock is not None:
                    await self._raise_if_cancelled(
                        pipeline_status, pipeline_status_lock
                    )

                task_meta: dict[asyncio.Task, tuple[str, dict]] = {}
                for item_id, item in items.items():
                    if not isinstance(item, dict):
                        continue
                    if kind == "drawing":
                        inner_coro = _analyze_drawing(
                            item_id, item, sidecar_path.parent
                        )
                    else:
                        inner_coro = _analyze_text_modality(kind, item_id, item)
                    task = asyncio.create_task(
                        _run_with_progress_log(inner_coro, kind, item_id)
                    )
                    task_meta[task] = (item_id, item)

                if not task_meta:
                    # No valid items in this sidecar — asyncio.wait([]) would
                    # ValueError, so skip the wait loop entirely.
                    continue

                # Fail-fast polling loop. Three trigger paths:
                #   1. an item task raises (e.g. MultimodalAnalysisError) →
                #      asyncio.wait returns early via FIRST_EXCEPTION;
                #   2. an item task raises PipelineCancelledException →
                #      same path, preserving the exception type;
                #   3. user clicks /cancel_pipeline mid-VLM → the
                #      cancellation_requested check at the top of the next
                #      poll iteration (≤ POLL_INTERVAL_SECONDS) fabricates
                #      a PipelineCancelledException.
                #
                # Do NOT add a watcher coroutine to the wait set: it would be
                # an infinite loop that stays pending when all items succeed,
                # preventing FIRST_EXCEPTION from ever returning.
                pending: set[asyncio.Task] = set(task_meta.keys())
                fail_fast_exc: BaseException | None = None
                POLL_INTERVAL_SECONDS = 0.5
                while pending:
                    if (
                        pipeline_status is not None
                        and pipeline_status_lock is not None
                        and await self._cancellation_requested(
                            pipeline_status, pipeline_status_lock
                        )
                    ):
                        fail_fast_exc = PipelineCancelledException(
                            "User cancelled during analyze"
                        )
                        break

                    done_now, pending = await asyncio.wait(
                        pending,
                        timeout=POLL_INTERVAL_SECONDS,
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for t in done_now:
                        if t.cancelled():
                            continue
                        texc = t.exception()
                        if texc is not None:
                            # Preserve original exception type so the
                            # _analyze_worker except dispatch can distinguish
                            # PipelineCancelledException from
                            # MultimodalAnalysisError.
                            fail_fast_exc = texc
                            break
                    if fail_fast_exc is not None:
                        break

                # If we broke early, cancel the still-running tasks.
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                # Collect results — preserve completed successes so reprocess
                # can hit the LLM cache instead of re-running the VLM.
                for t, (item_id, item) in task_meta.items():
                    if t.cancelled():
                        item["llm_analyze_result"] = _failure_result("cancelled")
                        continue
                    texc = t.exception()
                    if texc is None:
                        result_obj, cache_id = t.result()
                        item["llm_analyze_result"] = result_obj
                        _attach_cache_id(item, cache_id)
                    elif isinstance(texc, PipelineCancelledException):
                        item["llm_analyze_result"] = _failure_result("cancelled")
                    elif isinstance(texc, MultimodalAnalysisError):
                        item["llm_analyze_result"] = _failure_result(str(texc))
                    else:
                        item["llm_analyze_result"] = _failure_result(
                            f"unexpected error: {texc}"
                        )

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

                if fail_fast_exc is not None:
                    # Best-effort cache flush so any cache_ids written by
                    # already-completed sibling tasks survive a restart —
                    # otherwise the sidecar references cache rows that
                    # haven't been persisted yet. Mirrors
                    # _finalize_doc_failure's PROCESS-stage behaviour.
                    await self._persist_llm_response_cache_best_effort(
                        stage_label="multimodal analyze fail-fast",
                        doc_id=doc_id,
                    )
                    raise fail_fast_exc

            # Successful multimodal cache rows used to wait until the much
            # later PROCESS-stage _insert_done. Commit after the analysis
            # sidecar has been written so a restart can reuse the result as
            # soon as this stage has completed.
            await self._persist_llm_response_cache_best_effort(
                stage_label="multimodal analyze",
                doc_id=doc_id,
            )
            parsed_data["multimodal_processed"] = True
            logger.info(f"[analyze_multimodal] completed for d-id: {doc_id}")
        except PipelineCancelledException:
            # Must re-raise BEFORE the generic Exception handler below,
            # otherwise the doc would be returned as if analyze succeeded
            # and would advance to PROCESS instead of being marked FAILED.
            raise
        except MultimodalAnalysisError:
            raise
        except Exception as e:
            logger.warning(f"[analyze_multimodal] failed for d-id: {doc_id}: {e}")
        return parsed_data

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
        from lightrag.parser.routing import parse_process_options

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
                cleaned = (sanitize_text_for_encoding(str(x)) for x in v)
                return [s for s in cleaned if s]
            s = sanitize_text_for_encoding(str(v))
            return [s] if s else []

        def _norm_parent_headings(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            cleaned = (sanitize_text_for_encoding(str(p or "")) for p in value)
            return [p for p in cleaned if p]

        def _build_heading_dict(item: dict[str, Any]) -> dict[str, Any] | None:
            heading_raw = item.get("heading")
            if isinstance(heading_raw, dict):
                heading_text = sanitize_text_for_encoding(
                    str(heading_raw.get("heading") or "")
                )
                parents = _norm_parent_headings(heading_raw.get("parent_headings"))
                try:
                    level = int(heading_raw.get("level") or 0)
                except (TypeError, ValueError):
                    level = 0
            else:
                heading_text = sanitize_text_for_encoding(str(heading_raw or ""))
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
            image_type: str,
            description: str,
            footnotes_joined: str,
            equation_body: str,
        ) -> str:
            # NOTE: the `[Image Name]` / `[Table Name]` / `[Equation Name]`
            # leading labels below are a contract consumed by
            # ``lightrag.operate._parse_mm_display_name`` (regex
            # ``_MM_DISPLAY_NAME_PATTERN``). If you rename or restructure
            # these labels, update that regex too, or relation descriptions
            # will silently fall back to sidecar ids. The
            # ``test_parse_mm_display_name_on_real_builder_output``
            # regression pins this contract end-to-end.
            if kind == "drawing":
                head = f"[Image Name]{name}\n[Image Type]{image_type}"
                footnote_label = "Image Footnotes"
            elif kind == "table":
                head = f"[Table Name]{name}"
                footnote_label = "Table Footnotes"
            else:  # equation
                head = f"{equation_body}\n[Equation Name]{name}"
                footnote_label = "Equation Footnotes"

            sections = [head, description]
            if footnotes_joined:
                sections.append(f"[{footnote_label}]{footnotes_joined}")
            return "\n\n".join(s for s in sections if s).strip()

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

                # Sanitize every VLM-produced field: analysis results are
                # parsed from LLM JSON where unescaped LaTeX (e.g. "\frac")
                # decodes into control characters ("\f" -> \x0c). These
                # strings feed text_chunks, vector stores and — via the
                # multimodal entity injection in operate.extract_entities —
                # graph node/edge attributes, where XML-illegal characters
                # crash the GraphML flush.
                name = sanitize_text_for_encoding(str(analysis.get("name") or ""))
                description = sanitize_text_for_encoding(
                    str(analysis.get("description") or "")
                )
                equation_body = sanitize_text_for_encoding(
                    str(analysis.get("equation") or "")
                )
                image_type = sanitize_text_for_encoding(str(analysis.get("type") or ""))
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
                    return _render(
                        kind=kind,
                        name=name,
                        image_type=image_type,
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
