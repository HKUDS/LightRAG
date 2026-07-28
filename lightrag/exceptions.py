from __future__ import annotations

import httpx
from typing import Any, Literal


class APIStatusError(Exception):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(
        self, message: str, *, response: httpx.Response, body: object | None
    ) -> None:
        super().__init__(message)
        self.request = response.request
        self.body = body
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(Exception):
    def __init__(
        self, *, message: str = "Connection error.", request: httpx.Request | None
    ) -> None:
        super().__init__(message)
        self.request = request


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400  # pyright: ignore[reportIncompatibleVariableOverride]


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401  # pyright: ignore[reportIncompatibleVariableOverride]


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403  # pyright: ignore[reportIncompatibleVariableOverride]


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404  # pyright: ignore[reportIncompatibleVariableOverride]


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409  # pyright: ignore[reportIncompatibleVariableOverride]


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422  # pyright: ignore[reportIncompatibleVariableOverride]


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request | None) -> None:
        super().__init__(message="Request timed out.", request=request)


class StorageNotInitializedError(RuntimeError):
    """Raised when storage operations are attempted before initialization."""

    def __init__(self, storage_type: str = "Storage"):
        super().__init__(
            f"{storage_type} not initialized. Please ensure proper initialization:\n"
            f"\n"
            f"  rag = LightRAG(...)\n"
            f"  await rag.initialize_storages()  # Required - auto-initializes pipeline_status\n"
            f"\n"
            f"See: https://github.com/HKUDS/LightRAG#important-initialization-requirements"
        )


class StorageCapabilityError(RuntimeError):
    """Raised when a storage backend lacks a capability the caller requires.

    Callers that depend on a hard guarantee (strict counting, strict point
    reads, strict source resolution, ...) must fail closed on this error —
    never silently substitute a weaker code path.
    """


class StorageControlPlaneError(RuntimeError):
    """A storage control-plane read failed (e.g. an index that must exist is
    unexpectedly absent, or a not-yet-ready index during rebuild/recovery).

    Distinct from a data-plane miss: the caller cannot know the true state and
    MUST fail closed (retry with backoff / surface 503 / keep sticky work
    unacknowledged). Degrading to a full-materialization or destructive
    fallback on this error is forbidden — that is exactly the OOM/corruption
    window the control plane exists to fence off.
    """


class PipelineBackpressureError(RuntimeError):
    """Admission refused: the pipeline already holds its capacity of documents.

    Raised when ``MAX_PENDING_DOCUMENTS > 0`` and

        strict active count + other in-flight reservation weights + requested

    would exceed the capacity. Active means PENDING / PARSING / ANALYZING /
    PROCESSING — work the pipeline still has to do.

    The fields are structured on purpose (LR2 §9.1): an SDK caller must be able
    to tell the client how much room there is rather than parse a string, and
    the API maps ONLY this error to 429. A mutual-exclusion refusal
    (manual freeze / scanning_exclusive / destructive_busy) is a
    ``PipelineReservationConflict`` → 409, and a storage / control-plane failure
    is 503 — never disguised as "no capacity".
    """

    def __init__(
        self,
        *,
        current: int,
        requested: int,
        capacity: int,
        reason: str = "",
    ) -> None:
        self.current = current
        self.requested = requested
        self.capacity = capacity
        self.reason = reason or "pipeline document capacity reached"
        super().__init__(
            f"{self.reason}: {current} document(s) already active or reserved "
            f"+ {requested} requested exceeds capacity {capacity}"
        )


class PipelineReservationConflictError(RuntimeError):
    """A pipeline mutual-exclusion fence refused this caller.

    The structured counterpart of the reservation helpers'
    ``PipelineReservationResult`` for the paths that have to RAISE rather than
    return it — the SDK / direct ``apipeline_enqueue_documents`` entry points,
    which have no HTTP response to shape.

    ``conflict`` is the ``PipelineReservationConflict`` member (kept as a plain
    string-valued enum member so importing this module never pulls in the
    shared-storage layer) and ``fence`` is the ``pipeline_status`` flag that
    refused, when one specific flag did. LR2 §9.1 requires the distinction to
    survive as data: ``recovery_required`` is a fenced workspace (→ 503) while
    a manual freeze / ``scanning_exclusive`` / ``destructive_busy`` is a bounded
    window the caller should retry (→ 409), and an SDK caller cannot tell those
    apart from a message string.
    """

    def __init__(
        self,
        message: str,
        *,
        conflict: Any = None,
        fence: str | None = None,
    ) -> None:
        self.conflict = conflict
        self.fence = fence
        super().__init__(message)

    @property
    def recovery_required(self) -> bool:
        """True when the workspace is fenced pending recovery (→ 503), as
        opposed to a bounded mutual-exclusion window (→ 409)."""
        return getattr(self.conflict, "value", self.conflict) == "recovery_required"


class PipelineRecoveryRequiredError(RuntimeError):
    """The pipeline fenced its own workspace with ``recovery_required``.

    Raised when a manual retry's DRAIN_TO_IDLE cannot make forward progress:
    the same active ``doc_status`` rows blocked the drain for several
    consecutive rounds without any of them changing state, so re-sweeping can
    only spin (LR2 §7.2 "DRAIN_TO_IDLE 的前进性" / §13.2 case 16). The workspace
    is fenced instead, every mutation is refused with 503, and
    ``blocked_doc_ids`` carries the BOUNDED sample an operator needs to find the
    offending rows — never the whole set.

    Two causes reach this, distinguished by the fence record's ``kind``:
    ``manual_drain_stalled`` (rows that look routable but never change state) and
    ``manual_drain_blocked`` (rows the drain can never advance at all — an
    unfinished custom-chunk operation). Both are cleared the same way:
    ``POST /documents/recovery/force_reset``, which also cancels the queued manual
    intents, since a sticky request is itself what makes ``/documents/scan``
    refuse (``refuse_when_manual_pending``) and ``/scan`` is the remedy for the
    blocked case.
    """

    def __init__(self, message: str, *, blocked_doc_ids: tuple[str, ...] = ()) -> None:
        self.blocked_doc_ids = blocked_doc_ids
        super().__init__(message)


class SourceConflictRepairCASError(StorageControlPlaneError):
    """A source-conflict repair commit lost its compare-and-set check.

    The candidate set for the canonical source key changed between the
    operator's dry-run and the commit (count / fingerprint mismatch), so
    ``repair_source_conflict`` refused instead of overwriting the concurrent
    change.

    Unlike its parent — "the true state is unknown, fail closed" — this error
    reports a *known* state that simply no longer matches what the operator
    echoed back: re-running the dry-run yields a fresh token and the repair can
    proceed. Callers therefore surface it as a conflict (HTTP 409), never as an
    unavailable control plane (503). It subclasses
    ``StorageControlPlaneError`` so a caller that only knows the coarse
    fail-closed contract still behaves safely.
    """


class SourceConflictPrimaryUnusableError(ValueError):
    """The document an operator chose to keep cannot own the canonical source.

    Two distinct reasons, and a caller that renders one message for both tells
    the operator something false about their data — so the reason travels with
    the exception as :attr:`reason` rather than only inside the message text
    (which callers deliberately do not forward: it is not guaranteed to be
    client-safe):

    * ``REASON_NO_CONTENT`` — a CONFIRMED absence of ``full_docs`` content. Such
      a row is an unprocessable stub, so the source key would end up owned by a
      document that can never exist, and scan classification deletes exactly
      those rows (``STALE_STUB``) — which would remove the primary and leave the
      demoted documents pointing at an id that no longer exists;
    * ``REASON_CONTENT_ELSEWHERE`` — the content already belongs to another
      document (:attr:`holder_doc_id`), under a different canonical source. The
      processing stage marks such a document FAILED-duplicate and deletes its
      body, so the key it was just given would end up with no primary.

    Either way a repair only demotes, so a key left with no candidate has no
    conflict left to settle — which is why both are refused BEFORE the
    demotions rather than reported after them.

    Distinct from the plain ``ValueError`` that ``repair_source_conflict`` raises
    for "not a current primary candidate": those are all 409s, but conflating
    them makes the API report "not a candidate" for a row that IS one, which
    sends the operator to re-list a conflict that has not changed. It subclasses
    ``ValueError`` so a caller that only handles the coarse "bad primary" case
    still refuses safely.
    """

    REASON_NO_CONTENT = "no_full_docs_content"
    REASON_CONTENT_ELSEWHERE = "content_belongs_to_another_document"

    def __init__(self, message: str, *, reason: str, holder_doc_id: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        # Set for REASON_CONTENT_ELSEWHERE: the document the content belongs to,
        # which is the one thing the operator needs in order to act.
        self.holder_doc_id = holder_doc_id


class StorageRecordNotFoundError(KeyError):
    """A targeted doc_status field update referenced a non-existent record.

    Raised by ``update_doc_status_fields(..., missing_ok=False)`` — the
    default — so callers cannot silently patch a record that a concurrent
    delete already removed.
    """


class PipelineNotInitializedError(KeyError):
    """Raised when pipeline status is accessed before initialization."""

    def __init__(self, namespace: str = ""):
        msg = (
            f"Pipeline namespace '{namespace}' not found.\n"
            f"\n"
            f"Pipeline status should be auto-initialized by initialize_storages().\n"
            f"If you see this error, please ensure:\n"
            f"\n"
            f"  1. You called await rag.initialize_storages()\n"
            f"  2. For multi-workspace setups, each LightRAG instance was properly initialized\n"
            f"\n"
            f"Standard initialization:\n"
            f"  rag = LightRAG(workspace='your_workspace')\n"
            f"  await rag.initialize_storages()  # Auto-initializes pipeline_status\n"
            f"\n"
            f"If you need manual control (advanced):\n"
            f"  from lightrag.kg.shared_storage import initialize_pipeline_status\n"
            f"  await initialize_pipeline_status(workspace='your_workspace')"
        )
        super().__init__(msg)


class PipelineCancelledException(Exception):
    """Raised when pipeline processing is cancelled by user request."""

    def __init__(self, message: str = "User cancelled"):
        super().__init__(message)
        self.message = message


class IndexFlushError(Exception):
    """Raised when a storage backend fails to flush buffered index ops.

    Carries the storage driver name and namespace so the pipeline can abort
    the batch with an actionable reason. The underlying error is preserved as
    the exception ``__cause__`` (set via ``raise ... from cause``).
    """

    def __init__(self, storage_name: str, namespace: str, cause: BaseException):
        self.storage_name = storage_name
        self.namespace = namespace
        super().__init__(f"{storage_name}[{namespace}] index flush failed: {cause}")


class ChunkTokenLimitExceededError(ValueError):
    """Raised when a chunk exceeds the configured token limit."""

    def __init__(
        self,
        chunk_tokens: int,
        chunk_token_limit: int,
        chunk_preview: str | None = None,
    ) -> None:
        preview = chunk_preview.strip() if chunk_preview else None
        truncated_preview = preview[:80] if preview else None
        preview_note = f" Preview: '{truncated_preview}'" if truncated_preview else ""
        message = (
            f"Chunk token length {chunk_tokens} exceeds chunk_token_size {chunk_token_limit}."
            f"{preview_note}"
        )
        super().__init__(message)
        self.chunk_tokens = chunk_tokens
        self.chunk_token_limit = chunk_token_limit
        self.chunk_preview = truncated_preview


class ChunkBlockMatchError(ValueError):
    """Raised when a chunk cannot be located in the document's blocks.jsonl.

    Sidecar backfill (``lightrag.sidecar.backfill``) maps F/R/V chunks back to
    their source block(s) by matching chunk content against the parse-time
    ``*.blocks.jsonl`` merged text. When a sidecar-less chunk cannot be located,
    this is raised so the pipeline marks the document FAILED rather than
    persisting chunks with missing/incorrect provenance.
    """

    def __init__(
        self,
        chunk_order_index: int,
        chunk_preview: str | None = None,
        blocks_path: str | None = None,
    ) -> None:
        preview = chunk_preview.strip() if chunk_preview else None
        truncated_preview = preview[:80] if preview else None
        preview_note = f" Preview: '{truncated_preview}'" if truncated_preview else ""
        path_note = f" (blocks: {blocks_path})" if blocks_path else ""
        message = (
            f"Chunk #{chunk_order_index} could not be located in the document "
            f"blocks during sidecar backfill.{preview_note}{path_note}"
        )
        super().__init__(message)
        self.chunk_order_index = chunk_order_index
        self.chunk_preview = truncated_preview
        self.blocks_path = blocks_path


class DataMigrationError(Exception):
    """Raised when data migration from legacy collection/table fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MultimodalAnalysisError(RuntimeError):
    """Raised when multimodal analysis must fail the current document.

    Hard failures (missing required field, schema mismatch, model not
    available, sidecar already carries ``status="failure"``) bubble this
    exception so the pipeline marks the document failed instead of writing
    an unusable analyze result. Callers persist a ``status="failure"``
    sidecar entry alongside the raise so a re-run sees the failure.
    """
