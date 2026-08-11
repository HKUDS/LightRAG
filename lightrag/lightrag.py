from __future__ import annotations

import traceback
import asyncio
import os
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None
from dataclasses import InitVar, asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    NoReturn,
    TypeVar,
    cast,
    final,
    Literal,
    Mapping,
    Optional,
    List,
    Dict,
    Union,
)
from lightrag.prompt import (
    PROMPTS,
    get_default_entity_extraction_prompt_profile,
    resolve_entity_extraction_prompt_profile,
    validate_entity_extraction_prompt_profile_for_mode,
)
from lightrag.constants import (
    DEFAULT_CHUNK_P_SIZE,
    DEFAULT_MAX_GLEANING,
    DEFAULT_MAX_EXTRACTION_RECORDS,
    DEFAULT_MAX_EXTRACTION_ENTITIES,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    ROLLBACK_REPORT_SAMPLE_CAP,
    DEFAULT_SUMMARY_PRIORITY,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_RERANK_TIMEOUT,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_MAX_PARALLEL_ANALYZE,
    DEFAULT_MAX_PARALLEL_PARSE_NATIVE,
    DEFAULT_MAX_PARALLEL_PARSE_MINERU,
    DEFAULT_MAX_PARALLEL_PARSE_DOCLING,
    DEFAULT_QUEUE_SIZE_PARSE,
    DEFAULT_QUEUE_SIZE_ANALYZE,
    DEFAULT_QUEUE_SIZE_INSERT,
    DEFAULT_PIPELINE_SCHEDULING_PAGE_SIZE,
    DEFAULT_PIPELINE_REQUIRE_STRICT_STORAGE_READS,
    DEFAULT_MAX_PENDING_DOCUMENTS,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    KG_PURGE_METADATA_KEY,
    KG_PURGE_PHASE_ANCHORS_PENDING,
    KG_PURGE_PHASE_COMPLETED,
    KG_PURGE_PHASE_DERIVED_COMMITTED,
    KG_PURGE_PHASE_PREPARED,
    KG_PURGE_SCHEMA_VERSION,
    KG_WRITE_STATE_PRE_GRAPH,
)
from lightrag.utils import get_env_value
from lightrag.parser.routing import _chunk_env_int

from lightrag.kg import (
    verify_storage_implementation,
)


from lightrag.kg.shared_storage import (
    PipelineReservationConflict,
    acquire_reservation,
    append_pipeline_history,
    check_pipeline_status_mutation,
    get_namespace_data,
    get_default_workspace,
    set_default_workspace,
    get_namespace_lock,
    get_pipeline_ingress,
    get_storage_keyed_lock,
    with_reservation_lock,
)

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    CursorPosition,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    DeletionResult,
    OllamaServerInfos,
    QueryResult,
)
from lightrag.namespace import NameSpace
from lightrag.chunker import chunking_by_token_size
from lightrag.operate import (
    KGRebuildReport,
    _truncate_vdb_content,
    collect_kg_merge_candidates,
    extract_entities,
    kg_query,
    merge_nodes_and_edges,
    naive_query,
    rebuild_knowledge_from_chunks,
)
from lightrag.utils_pipeline import (
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    KG_RECOVERY_WARNINGS_METADATA_KEY,
    compute_text_content_hash,
    doc_status_custom_chunk_patch,
    doc_status_field,
    doc_status_kg_purge_journal,
    doc_status_kg_write_state,
    enforce_strict_storage_capabilities,
    make_custom_chunk_id,
    make_custom_chunk_operation_id,
    make_kg_purge_operation_id,
    normalize_document_file_path,
    require_doc_status_record,
)
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.exceptions import (
    IndexFlushError,
    KGPurgeOperationConflictError,
    RecoveryAnchorMissingError,
)
from lightrag.utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    priority_limit_async_func_call,
    sanitize_text_for_encoding,
    check_storage_env_vars,
    generate_track_id,
    get_content_summary,
    convert_to_user_format,
    logger,
    make_relation_vdb_ids,
    subtract_source_ids,
    make_relation_chunk_key,
    normalize_entity_name,
    normalize_source_ids_limit_method,
    normalize_string_list,
    run_in_chunking_executor,
    TokenLimitTruncationTally,
    LLM_TRUNCATION_METADATA_KEY,
    merge_truncation_metadata,
)
from lightrag.types import KnowledgeGraph
from dotenv import load_dotenv
from lightrag.pipeline import _PipelineMixin
from lightrag.kg.factory import get_storage_class
from lightrag.addon_params import (
    ObservableAddonParams,
    normalize_addon_params,
)
from lightrag.llm_roles import (
    ROLE_NAMES,
    ROLES,
    RoleLLMConfig,
    RoleSpec,  # noqa: F401  # re-exported via lightrag/__init__.py
    _optional_env_int,
    _RoleLLMMixin,
    _RoleLLMState,
)
from lightrag.storage_migrations import _StorageMigrationMixin

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

_SyncResultT = TypeVar("_SyncResultT")

# Ordered purge journal phases (issue #3400). Index = how much destructive work
# is already persisted, so a resumed purge can skip exactly that much:
#   prepared          — proof verified, journal durable, NOTHING deleted yet
#   derived_committed — graph/vector/tracking contributions repaired or removed
#   anchors_pending   — chunks deleted too; only the anchor rows remain
#   completed         — anchors deleted; the caller's finalization is all that's left
# Only phases at/after ``derived_committed`` are proof in their own right: they
# are the ones that may legitimately have removed the anchors.
_KG_PURGE_PHASE_ORDER: tuple[str, ...] = (
    KG_PURGE_PHASE_PREPARED,
    KG_PURGE_PHASE_DERIVED_COMMITTED,
    KG_PURGE_PHASE_ANCHORS_PENDING,
    KG_PURGE_PHASE_COMPLETED,
)
_KG_PURGE_RESUMABLE_PHASES: frozenset[str] = frozenset(_KG_PURGE_PHASE_ORDER)


class _PurgeStageError(Exception):
    """A purge step failed, tagged with which step it was.

    ``adelete_by_doc_id`` records the failing step in
    ``doc_status.metadata.deletion_failure_stage`` so an operator can see how
    far a failed deletion got. Now that the delete path delegates its KG
    cleanup to the purge primitive, the stage has to travel with the exception
    instead of being tracked by the caller — carried as an attribute rather
    than parsed back out of the message, which callers must not depend on.

    Subclasses ``Exception`` and keeps the historical ``Failed to ...`` message
    text, so existing callers and tests that match on either are unaffected.
    """

    def __init__(self, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.purge_stage = stage


@dataclass(frozen=True)
class _PurgeRecoveryProof:
    """Why a whole-document purge is (or is not) allowed to delete anything.

    See :py:meth:`LightRAG._resolve_purge_recovery_proof`. ``proof_kind`` is
    ``None`` exactly when no proof applies, in which case ``missing_reason``
    carries the :class:`~lightrag.exceptions.RecoveryAnchorMissingError` reason
    to raise with.
    """

    proof_kind: Literal["anchors", "pre_graph", "journal", "empty_scope"] | None
    operation_id: str
    journal_phase: str | None = None
    write_state: str | None = None
    missing_namespaces: tuple[str, ...] = ()
    missing_reason: str | None = None
    # Candidates anchored ONLY in an unfinished custom-chunk patch journal.
    # Patch-mode merges deliberately skip the write-ahead anchor prewrite (the
    # journal is that operation's anchor) and union into the base document's
    # anchor rows only at commit — so until then these graph objects exist
    # while no anchor row names them. Whole-document purge must union them in,
    # or deleting the document removes the staged chunks and leaves those
    # objects orphaned.
    patch_candidate_entities: tuple[str, ...] = ()
    patch_candidate_relations: tuple[tuple[str, str], ...] = ()

    def phase_at_least(self, phase: str) -> bool:
        """True when the journal records ``phase`` or a later one."""
        if self.journal_phase is None:
            return False
        return _KG_PURGE_PHASE_ORDER.index(
            self.journal_phase
        ) >= _KG_PURGE_PHASE_ORDER.index(phase)


def _run_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, _SyncResultT]],
    *,
    sync_name: str,
    async_name: str,
    owning_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> _SyncResultT:
    """Drive an async coroutine to completion from a synchronous wrapper.

    The synchronous wrappers (``insert``, ``query``, ``delete_by_entity``,
    …) all share the same shape: acquire an event loop and call
    ``loop.run_until_complete()``.  That call is only valid when the current
    thread has **no** running loop *and* the loop it ends up driving is the
    same one the instance's storages were initialized on.  Two misuse modes
    break this, and both have the same fix — use the ``a*`` coroutine from
    async code:

    * **Same thread, inside a running loop** (e.g. directly inside an
      ``async def`` / FastAPI handler): ``run_until_complete()`` would raise
      ``RuntimeError: This event loop is already running``.  Detected up front
      via :func:`asyncio.get_running_loop`.
    * **A different — but still alive — loop than the storages bound to**
      (e.g. ``loop.run_in_executor(None, rag.insert, …)`` runs the wrapper on a
      pool thread that spins up a fresh loop while the app's loop keeps
      running, or an asyncio loop runs on another thread): the shared
      ``asyncio.Lock`` objects in ``lightrag.kg.shared_storage`` are bound to
      ``owning_loop``, so acquiring them from a second loop raises
      ``RuntimeError: <Lock> is bound to a different event loop`` or stalls on
      a callback scheduled on an idle loop.  Detected by comparing the loop we
      are about to drive against ``owning_loop``.

    The cross-loop check only fires while ``owning_loop`` is still **open**.  A
    *closed* ``owning_loop`` means the loop the storages were initialized on is
    gone — e.g. the common ``rag = asyncio.run(initialize_rag())`` pattern,
    where ``asyncio.run`` closes the loop after ``initialize_storages()`` — so
    there is no live loop to clash with.  We let those calls through to run on a
    fresh loop, preserving the long-standing behavior (any lock still bound to
    the closed loop is handled by ``asyncio.Lock`` itself, exactly as before).

    These synchronous wrappers are compatibility conveniences for simple,
    single-threaded scripts. SDK integrations, async applications, and
    concurrent workloads should prefer the ``a*`` coroutine APIs. The wrappers
    are not cross-thread-safe entry points for concurrent calls against the same
    ``LightRAG`` instance; callers that must invoke them from multiple threads
    need to serialize access externally or route work through one event loop.

    The coroutine is created lazily via ``coro_factory`` so neither guard ever
    leaves an un-awaited coroutine behind when it raises.

    Args:
        coro_factory: Zero-arg callable returning the coroutine to run.
        sync_name: Name of the sync wrapper, used in the error message.
        async_name: Name of the async method to recommend instead.
        owning_loop: The loop the instance's storages were initialized on
            (``LightRAG._owning_loop``). ``None`` when storages have not been
            initialized yet, or closed, in which case the cross-loop check is
            skipped.

    Returns:
        The result of awaiting the coroutine.

    Raises:
        RuntimeError: if called from within a running asyncio event loop, or
            from a different loop while the loop the storages were bound to is
            still open.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # No running loop on this thread — safe to drive our own.
    else:
        raise RuntimeError(
            f"{sync_name}() cannot be called from within a running asyncio "
            f"event loop. Synchronous wrappers internally call "
            f"loop.run_until_complete(), which Python forbids while a loop is "
            f"already running on this thread. "
            f"Use `await {async_name}(...)` from async code instead."
        )
    loop = always_get_an_event_loop()
    if (
        owning_loop is not None
        and not owning_loop.is_closed()
        and loop is not owning_loop
    ):
        raise RuntimeError(
            f"{sync_name}() must run on the same event loop this LightRAG "
            f"instance was initialized on, but it is being driven from a "
            f"different loop — typically `loop.run_in_executor(None, "
            f"rag.{sync_name}, ...)`, or an asyncio loop running on another "
            f"thread. LightRAG's shared storage locks are bound to the original "
            f"loop, so running here would raise '<Lock> is bound to a different "
            f"event loop' or stall. "
            f"Use `await {async_name}(...)` on the original loop instead."
        )
    return loop.run_until_complete(coro_factory())


@final
@dataclass
class LightRAG(_RoleLLMMixin, _StorageMigrationMixin, _PipelineMixin):
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    # Directory
    # ---

    working_dir: str = field(default="./rag_storage")
    """Directory where cache and temporary files are stored."""

    # Storage
    # ---

    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data."""

    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""

    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Workspace
    # ---

    workspace: str = field(default_factory=lambda: os.getenv("WORKSPACE", ""))
    """Workspace for data isolation. Defaults to empty string if WORKSPACE environment variable is not set."""

    # ---
    # TODO: Deprecated, use setup_logger in utils.py instead
    log_level: int | None = field(default=None)
    log_file_path: str | None = field(default=None)

    # Query parameters
    # ---

    top_k: int = field(default=get_env_value("TOP_K", DEFAULT_TOP_K, int))
    """Number of entities/relations to retrieve for each query."""

    chunk_top_k: int = field(
        default=get_env_value("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int)
    )
    """Maximum number of chunks in context."""

    max_entity_tokens: int = field(
        default=get_env_value("MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int)
    )
    """Maximum number of tokens for entity in context."""

    max_relation_tokens: int = field(
        default=get_env_value("MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int)
    )
    """Maximum number of tokens for relation in context."""

    max_total_tokens: int = field(
        default=get_env_value("MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int)
    )
    """Maximum total tokens in context (including system prompt, entities, relations and chunks)."""

    related_chunk_number: int = field(
        default=get_env_value("RELATED_CHUNK_NUMBER", DEFAULT_RELATED_CHUNK_NUMBER, int)
    )
    """Number of related chunks to grab from single entity or relation."""

    kg_chunk_pick_method: str = field(
        default=get_env_value("KG_CHUNK_PICK_METHOD", DEFAULT_KG_CHUNK_PICK_METHOD, str)
    )
    """Method for selecting text chunks: 'WEIGHT' for weight-based selection, 'VECTOR' for embedding similarity-based selection."""

    enable_content_headings: bool = field(
        default_factory=lambda: get_env_value("ENABLE_CONTENT_HEADINGS", True, bool)
    )
    """Append each chunk's parent heading path as a `content_headings` field in the chunk JSON sent to the LLM."""

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(
        default=get_env_value("MAX_GLEANING", DEFAULT_MAX_GLEANING, int)
    )
    """Maximum number of entity extraction attempts for ambiguous content."""

    entity_extract_max_records: int = field(
        default=get_env_value(
            "MAX_EXTRACTION_RECORDS", DEFAULT_MAX_EXTRACTION_RECORDS, int
        )
    )
    """Per-response cap on total entity+relationship rows/records."""

    entity_extract_max_entities: int = field(
        default=get_env_value(
            "MAX_EXTRACTION_ENTITIES", DEFAULT_MAX_EXTRACTION_ENTITIES, int
        )
    )
    """Per-response cap on entity rows/objects."""

    force_llm_summary_on_merge: int = field(
        default=get_env_value(
            "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
        )
    )

    # Text chunking
    # ---

    chunk_token_size: int | None = field(default=None)
    """Maximum number of tokens per text chunk when splitting documents.

    ``None`` means "use ``addon_params['chunker']['chunk_token_size']``"
    (env-driven via ``CHUNK_SIZE``).  When the constructor is given a
    non-None value it overlays onto ``addon_params['chunker']`` in
    ``__post_init__`` so the per-document ``chunk_options`` snapshot
    actually picks it up.  Always an ``int`` after construction —
    back-filled from the resolved chunker config so legacy readers
    (``self.chunk_token_size``) keep working."""

    chunk_overlap_token_size: int | None = field(default=None)
    """Number of overlapping tokens between consecutive text chunks (F-strategy semantics).

    ``None`` means "use the per-strategy default in
    ``addon_params['chunker']``" (env-driven via
    ``CHUNK_F_OVERLAP_SIZE`` / ``CHUNK_R_OVERLAP_SIZE`` falling back to
    ``CHUNK_OVERLAP_SIZE``).  When non-None at construction time, the
    value overlays onto every strategy sub-dict that natively takes
    ``chunk_overlap_token_size`` (``fixed_token``, ``recursive_character``)
    so the per-doc snapshot reflects the constructor choice.  Per-strategy
    chunker parameters (R / V separators, thresholds, overlap overrides,
    etc.) live in ``addon_params['chunker']`` and are documented in
    :func:`lightrag.parser.routing.default_chunker_config`.  Per-doc
    snapshots are persisted to ``full_docs[doc_id]['chunk_options']``
    at enqueue time."""

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.

    Injection contract: token counting is CPU-bound and runs in worker threads so
    it does not block the event loop, so an injected tokenizer MUST be safe to
    call concurrently from multiple threads, and MUST survive ``copy.deepcopy``
    (``asdict`` in ``_build_global_config`` copies non-dataclass fields; an
    implementation guarding itself with a ``threading.Lock`` should declare
    ``__deepcopy__`` returning ``self``). LightRAG deliberately does not serialize
    it — a lock owned by LightRAG would be waited on by the event loop and would
    recreate the freeze the offload removes. The built-in ``TiktokenTokenizer``
    satisfies both. See :class:`lightrag.utils.Tokenizer`.
    """

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text with tiktoken. Defaults to `gpt-4o-mini`."""

    chunking_func: Callable[
        [
            Tokenizer,
            str,
            Optional[str],
            bool,
            int,
            int,
        ],
        Union[List[Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Legacy chunking-function customization point. Synchronous or async.

    **Where it runs.** A custom ``chunking_func`` is called on the event loop,
    exactly as before. The built-in default is dispatched to a worker thread
    instead, because chunking is CPU-bound and holding the loop for its duration
    stops the server answering anything (GHSA-26pm-px5v-8c4w). The extension
    point is deliberately excluded from that: "synchronous or async" is part of
    its contract, so an implementation that touches the running loop when called
    — a synchronous factory returning a Task, say — is supported and would fail
    outright in a worker thread, while an ``async def`` would gain nothing from
    the hop since its body runs on the loop regardless. A CPU-bound custom
    chunker should therefore do its own ``asyncio.to_thread``.

    **When this function is actually invoked.** The chunker dispatch in
    ``_PipelineMixin.process_single_document`` is driven by the
    document's ``process_options``:

      - If ``process_options`` explicitly contains a chunking selector
        char (``F``/``R``/``V``/``P``), the dispatcher routes to a
        chunker that follows the new file-chunker contract — see
        :mod:`lightrag.chunker` (``chunking_by_fixed_token`` for ``F``,
        ``chunking_by_paragraph_semantic`` for ``P``; ``R``/``V`` are
        not yet implemented and fall back to ``F``). **This
        ``chunking_func`` is NOT called in that case** — it is a
        legacy escape hatch and is intentionally bypassed when the user
        opted into a specific strategy.

      - If ``process_options`` does **not** name a chunking strategy
        (empty string, or only non-chunking flags such as ``i`` / ``t``
        / ``e`` / ``!``), the dispatcher invokes this ``chunking_func``
        with the legacy 6-arg signature below. This is the path taken
        by direct ``ainsert(text)`` calls and by any document whose
        ``process_options`` simply does not select a chunker.

    The presence/absence of the selector is exposed by
    :attr:`lightrag.parser.routing.ProcessOptions.chunking_explicit`.

    **Signature** — preserved unchanged from earlier LightRAG releases
    so externally-supplied chunkers continue to drop in without edits:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If
          None, the text is split into chunks of `chunk_token_size`
          tokens.
        - `split_by_character_only`: If True, the text is split only on
          the specified character.
        - `chunk_overlap_token_size`: The number of overlapping tokens
          between consecutive chunks.
        - `chunk_token_size`: The maximum number of tokens per chunk.

    The function should return a list of dictionaries (or an awaitable
    that resolves to one), each containing:

        - `tokens` (int): The number of tokens in the chunk.
        - `content` (str): The text content of the chunk.
        - `chunk_order_index` (int): Zero-based index indicating the
          chunk's order in the document.

    Defaults to :func:`lightrag.chunker.chunking_by_token_size`.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_token_limit: int | None = field(default=None, init=False)
    """Token limit for embedding model. Set automatically from embedding_func.max_token_size in __post_init__."""

    embedding_chunk_overlap_token_size: int = field(
        default=get_env_value(
            "EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE",
            DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE,
            int,
        )
    )
    """Overlap (in tokens) the embedding hard fallback borrows from the tail of
    the previous window when a chunk still exceeds ``embedding_token_limit``
    after chunking and has to be token-window-split.

    Independent from the chunker's own ``chunk_overlap_token_size``: that field
    is a semantic-chunking concern (paragraph/heading-aware splitting), while
    this one only applies to the mechanical last-resort split that runs after
    chunking, on chunks the chunker already produced. Some strategies (e.g. V)
    deliberately zero out ``chunk_overlap_token_size`` for reasons unrelated to
    this fallback, so this field must never read or fall back to that one.

    ``0`` disables the fallback's overlap. Negative values raise ``ValueError``
    in ``__post_init__``. Effective overlap is clamped at runtime to
    ``previous_content_token_count - 1`` and retreats further (exponentially)
    if that still cannot make forward progress.
    """

    embedding_batch_num: int = field(
        default=get_env_value("EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM, int)
    )
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(
        default=get_env_value(
            "EMBEDDING_FUNC_MAX_ASYNC", DEFAULT_EMBEDDING_FUNC_MAX_ASYNC, int
        )
    )
    """Maximum number of concurrent embedding function calls."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """

    default_embedding_timeout: int = field(
        default=get_env_value("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int)
    )

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""

    role_llm_configs: dict[str, RoleLLMConfig | dict[str, Any]] | None = field(
        default=None
    )
    """Per-role LLM overrides keyed by role name (see :data:`ROLES`).

    Each entry is a :class:`RoleLLMConfig` (or a plain dict with the same
    keys ``func`` / ``kwargs`` / ``max_async`` / ``timeout``). Any field left
    as ``None`` falls back to the corresponding base LLM setting. Roles not
    present in the dict are wrapped from the base ``llm_model_func`` and
    pick up ``{ROLE_PREFIX}_MAX_ASYNC_LLM`` env defaults."""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    summary_max_tokens: int = field(
        default=get_env_value("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS, int)
    )
    """Maximum tokens allowed for entity/relation description."""

    summary_context_size: int = field(
        default=get_env_value("SUMMARY_CONTEXT_SIZE", DEFAULT_SUMMARY_CONTEXT_SIZE, int)
    )
    """Maximum number of tokens allowed per LLM response."""

    summary_length_recommended: int = field(
        default=get_env_value(
            "SUMMARY_LENGTH_RECOMMENDED", DEFAULT_SUMMARY_LENGTH_RECOMMENDED, int
        )
    )
    """Recommended length of LLM summary output."""

    llm_model_max_async: int = field(
        default=get_env_value(
            "MAX_ASYNC_LLM",
            get_env_value("MAX_ASYNC", DEFAULT_MAX_ASYNC, int),
            int,
        )
    )
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    default_llm_timeout: int = field(
        default=get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)
    )

    entity_extraction_use_json: bool = field(
        default=os.getenv("ENTITY_EXTRACTION_USE_JSON", "false").lower() == "true"
    )
    """When True, entity extraction uses JSON structured output instead of delimiter-based text.
    JSON mode is slower but significantly improves extraction quality and compatibility with smaller models.
    Providers with native structured output support (OpenAI, Ollama, Gemini) will use their
    native capabilities. Other providers rely on JSON-formatted prompts with json_repair parsing.
    Default: False. Set ENTITY_EXTRACTION_USE_JSON=true in .env to enable."""

    # Rerank Configuration
    # ---

    rerank_model_func: Callable[..., object] | None = field(default=None)
    """Function for reranking retrieved documents. All rerank configurations (model name, API keys, top_k, etc.) should be included in this function. Optional."""

    rerank_model_max_async: int = field(
        default=get_env_value(
            "MAX_ASYNC_RERANK",
            get_env_value(
                "MAX_ASYNC_LLM",
                get_env_value("MAX_ASYNC", DEFAULT_MAX_ASYNC, int),
                int,
            ),
            int,
        )
    )
    """Maximum number of concurrent rerank calls.
    Falls back to MAX_ASYNC_LLM when MAX_ASYNC_RERANK is unset
    (MAX_ASYNC is still accepted as a deprecated alias)."""

    default_rerank_timeout: int = field(
        default=get_env_value("RERANK_TIMEOUT", DEFAULT_RERANK_TIMEOUT, int)
    )
    """Rerank request timeout in seconds.
    Independent from LLM_TIMEOUT since reranker calls are much shorter
    than full LLM generation."""

    min_rerank_score: float = field(
        default=get_env_value("MIN_RERANK_SCORE", DEFAULT_MIN_RERANK_SCORE, float)
    )
    """Minimum rerank score threshold for filtering chunks after reranking."""

    # Storage
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    vlm_process_enable: bool = field(
        default_factory=lambda: get_env_value("VLM_PROCESS_ENABLE", False, bool)
    )
    """Master switch for VLM multimodal analysis (i/t/e items).

    When False, the pipeline emits a warning and skips every multimodal item
    without invoking the VLM. When True, the configured VLM binding must
    support image inputs.
    """

    # Extensions
    # ---

    max_parallel_insert: int = field(
        default=get_env_value("MAX_PARALLEL_INSERT", DEFAULT_MAX_PARALLEL_INSERT, int)
    )
    """Maximum number of parallel insert operations."""

    max_parallel_parse_native: int = field(
        default=get_env_value(
            "MAX_PARALLEL_PARSE_NATIVE", DEFAULT_MAX_PARALLEL_PARSE_NATIVE, int
        )
    )
    max_parallel_parse_mineru: int = field(
        default=get_env_value(
            "MAX_PARALLEL_PARSE_MINERU", DEFAULT_MAX_PARALLEL_PARSE_MINERU, int
        )
    )
    max_parallel_parse_docling: int = field(
        default=get_env_value(
            "MAX_PARALLEL_PARSE_DOCLING", DEFAULT_MAX_PARALLEL_PARSE_DOCLING, int
        )
    )
    max_parallel_analyze: int = field(
        default=get_env_value("MAX_PARALLEL_ANALYZE", DEFAULT_MAX_PARALLEL_ANALYZE, int)
    )
    queue_size_parse: int = field(
        default=get_env_value("QUEUE_SIZE_PARSE", DEFAULT_QUEUE_SIZE_PARSE, int)
    )
    queue_size_analyze: int = field(
        default=get_env_value("QUEUE_SIZE_ANALYZE", DEFAULT_QUEUE_SIZE_ANALYZE, int)
    )
    queue_size_insert: int = field(
        default=get_env_value("QUEUE_SIZE_INSERT", DEFAULT_QUEUE_SIZE_INSERT, int)
    )

    pipeline_scheduling_page_size: int = field(
        default=get_env_value(
            "PIPELINE_SCHEDULING_PAGE_SIZE",
            DEFAULT_PIPELINE_SCHEDULING_PAGE_SIZE,
            int,
        )
    )
    """Bounded scheduling page size (LR2 Phase 2).

    The scheduler sweeps the doc_status backlog through keyset pages of this
    many records so memory stays O(page_size + inflight) instead of O(backlog).
    ``0`` disables paging: one page holds the whole result set, exactly
    reproducing the legacy single-scan behaviour. Negative values are rejected
    at initialization.
    """

    pipeline_require_strict_storage_reads: bool = field(
        default=get_env_value(
            "PIPELINE_REQUIRE_STRICT_STORAGE_READS",
            DEFAULT_PIPELINE_REQUIRE_STRICT_STORAGE_READS,
            bool,
        )
    )
    """Refuse to start when doc_status lacks a strict capability (LR2 §11).

    ``False`` (the default) logs a warning naming each gap and its operator-facing
    consequence, and ``/health`` reports the same under ``capabilities``. ``True``
    turns those gaps into a startup failure, for a deployment that would rather
    not run at all than serve 503s. Checked once, after storages initialize.

    There is deliberately no equivalent for bounded PAGING: the paging and typed
    source-resolution methods are ``@abstractmethod``, so a backend that lacks
    them cannot be instantiated in the first place.
    """

    max_pending_documents: int = field(
        default=get_env_value(
            "MAX_PENDING_DOCUMENTS",
            DEFAULT_MAX_PENDING_DOCUMENTS,
            int,
        )
    )
    """Admission capacity for ordinary ingestion (LR2 §9.1).

    A new upload / text insert is refused with
    :class:`~lightrag.exceptions.PipelineBackpressureError` (HTTP 429) when

        strict active count + other in-flight reservation weights + requested

    would exceed this value. Active means PENDING / PARSING / ANALYZING /
    PROCESSING. ``0`` (the default) disables admission control; negative values
    are rejected at initialization. Enabling it requires a doc_status backend
    with strict counting — every first-party backend has one.
    """

    max_graph_nodes: int = field(
        default=get_env_value("MAX_GRAPH_NODES", DEFAULT_MAX_GRAPH_NODES, int)
    )
    """Maximum number of graph nodes to return in knowledge graph queries."""

    max_source_ids_per_entity: int = field(
        default=get_env_value(
            "MAX_SOURCE_IDS_PER_ENTITY", DEFAULT_MAX_SOURCE_IDS_PER_ENTITY, int
        )
    )
    """Maximum number of source (chunk) ids in entity Grpah + VDB."""

    max_source_ids_per_relation: int = field(
        default=get_env_value(
            "MAX_SOURCE_IDS_PER_RELATION",
            DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
            int,
        )
    )
    """Maximum number of source (chunk) ids in relation Graph + VDB."""

    source_ids_limit_method: str = field(
        default_factory=lambda: normalize_source_ids_limit_method(
            get_env_value(
                "SOURCE_IDS_LIMIT_METHOD",
                DEFAULT_SOURCE_IDS_LIMIT_METHOD,
                str,
            )
        )
    )
    """Strategy for enforcing source_id limits: IGNORE_NEW or FIFO."""

    max_file_paths: int = field(
        default=get_env_value("MAX_FILE_PATHS", DEFAULT_MAX_FILE_PATHS, int)
    )
    """Maximum number of file paths to store in entity/relation file_path field."""

    file_path_more_placeholder: str = field(default=DEFAULT_FILE_PATH_MORE_PLACEHOLDER)
    """Placeholder text when file paths exceed max_file_paths limit."""

    addon_params: InitVar[dict[str, Any] | None] = None
    _addon_params: ObservableAddonParams = field(
        default_factory=ObservableAddonParams,
        init=False,
        repr=False,
    )
    _addon_params_dirty: bool = field(default=True, init=False, repr=False)
    _entity_extraction_prompt_profile: dict[str, Any] = field(
        default_factory=get_default_entity_extraction_prompt_profile,
        init=False,
        repr=False,
    )
    _cached_entity_extraction_use_json: bool | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _resolved_summary_language: str = field(
        default=DEFAULT_SUMMARY_LANGUAGE,
        init=False,
        repr=False,
    )

    # Storages Management
    # ---

    # TODO: Deprecated (LightRAG will never initialize storage automatically on creation，and finalize should be call before destroying)
    auto_manage_storages_states: bool = field(default=False)
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    cosine_better_than_threshold: float = field(
        default=get_env_value("COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, float)
    )

    ollama_server_infos: Optional[OllamaServerInfos] = field(default=None)
    """Configuration for Ollama server information."""

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    def _mark_addon_params_dirty(self) -> None:
        self._addon_params_dirty = True

    def _on_addon_params_changed(self) -> None:
        """Normalize a replacement chunker config before marking caches dirty.

        ``ObservableAddonParams`` observes top-level assignments, so replacing
        ``rag.addon_params['chunker']`` is corrected and reported immediately.
        Nested in-place mutation remains supported for compatibility; its first
        subsequent enqueue is normalized and cached by
        ``resolve_chunk_options``.

        The correction is applied in place so the caller's own nested
        ``recursive_character`` dict stays the object that is read later, and so
        it cannot recursively re-enter this callback through
        ``ObservableAddonParams.__setitem__``.
        """
        chunker_config = self._addon_params.get("chunker")
        if isinstance(chunker_config, Mapping):
            from lightrag.parser.routing import normalize_chunker_r_separators

            normalized, corrected = normalize_chunker_r_separators(
                chunker_config, context="addon_params['chunker']", in_place=True
            )
            if corrected and normalized is not chunker_config:
                # ``in_place`` could not apply (an immutable mapping was
                # supplied). Store the corrected copy without re-entering this
                # callback; the dirty mark below already covers it.
                dict.__setitem__(self._addon_params, "chunker", dict(normalized))
        self._mark_addon_params_dirty()

    def _replace_addon_params(
        self, addon_params: Mapping[str, Any] | None, *, mark_dirty: bool
    ) -> None:
        wrapped = ObservableAddonParams(
            normalize_addon_params(addon_params),
            on_change=self._on_addon_params_changed,
        )
        self._addon_params = wrapped
        if mark_dirty:
            self._mark_addon_params_dirty()

    def _get_addon_params(self) -> ObservableAddonParams:
        """Return the live addon_params store.

        Mutations on the returned instance trigger a cache refresh on the next
        _build_global_config() call. If the whole mapping is replaced via the
        setter, previously captured references point at the old instance and
        will no longer propagate changes — always re-read `rag.addon_params`
        after replacement rather than caching references.
        """
        return self._addon_params

    def _set_runtime_addon_params(self, addon_params: Mapping[str, Any] | None) -> None:
        self._replace_addon_params(addon_params, mark_dirty=True)
        self._apply_chunk_size_overlay()

    def _apply_chunk_size_overlay(self) -> None:
        """Reconcile chunk-size config across all four configuration tiers.

        Specificity-ordered precedence (high → low) per slot:

        1. ``addon_params['chunker']`` explicit (user-supplied dict that
           already carries the key).
        2. Strategy-specific env (``CHUNK_F_SIZE`` / ``CHUNK_R_SIZE`` /
           ``CHUNK_V_SIZE`` for per-strategy ``chunk_token_size``;
           ``CHUNK_F_OVERLAP_SIZE`` / ``CHUNK_R_OVERLAP_SIZE`` /
           ``CHUNK_P_OVERLAP_SIZE`` for overlap).  These are pre-filled into
           the strategy sub-dict by
           :func:`lightrag.parser.routing.default_chunker_config` when it
           builds the dict from scratch; for a *partial*
           ``addon_params['chunker']`` (which skips that builder) this overlay
           mirrors the size-env reads below so the env still applies.  Either
           way the slot is filled *only* when the env var is set.  No strategy
           env feeds the *top-level* ``chunk_token_size`` slot; that chain
           stays addon_params > legacy ctor > ``CHUNK_SIZE``.
        3. Legacy constructor field
           (``LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)``).
           Strategy-agnostic; only fills slots that were not already set
           by tiers 1–2.
        4. Legacy env (``CHUNK_SIZE`` / ``CHUNK_OVERLAP_SIZE``) — final
           fallback.

        After this runs, ``self._addon_params['chunker']`` carries fully
        resolved values for every slot the pipeline needs, and the
        legacy ``self.chunk_token_size`` / ``self.chunk_overlap_token_size``
        instance fields are back-filled to ``int`` so downstream readers
        (e.g. ``process_single_document``'s
        ``chunk_opts.get("chunk_token_size") or self.chunk_token_size``
        fallback) keep working.
        """
        chunker_cfg = self._addon_params.get("chunker")
        if not isinstance(chunker_cfg, dict):
            chunker_cfg = {}
            self._addon_params["chunker"] = chunker_cfg

        # Top-level chunk_token_size — no strategy-specific env exists,
        # so the chain is: addon_params > legacy ctor > CHUNK_SIZE env.
        if "chunk_token_size" not in chunker_cfg:
            if self.chunk_token_size is not None:
                chunker_cfg["chunk_token_size"] = self.chunk_token_size
            else:
                chunker_cfg["chunk_token_size"] = _chunk_env_int("CHUNK_SIZE", 1200)
        # Per-strategy chunk_overlap_token_size — strategy env (if set)
        # already lives in the sub-dict.  Slots still missing fall back
        # to the legacy ctor field, then CHUNK_OVERLAP_SIZE env.
        if self.chunk_overlap_token_size is not None:
            legacy_overlap_default = self.chunk_overlap_token_size
        else:
            legacy_overlap_default = _chunk_env_int("CHUNK_OVERLAP_SIZE", 100)
        for strategy_key in (
            "fixed_token",
            "recursive_character",
            "paragraph_semantic",
        ):
            sub = chunker_cfg.get(strategy_key)
            if not isinstance(sub, dict):
                sub = {}
                chunker_cfg[strategy_key] = sub
            if "chunk_overlap_token_size" not in sub:
                sub["chunk_overlap_token_size"] = legacy_overlap_default

        # P-specific chunk_token_size backfill — P does NOT inherit the
        # top-level chunk_token_size (CHUNK_SIZE / legacy ctor) when
        # nothing more specific was set; paragraph-semantic merging
        # needs more headroom than the global default to keep related
        # paragraphs together.  ``default_chunker_config`` already
        # pre-fills this slot for the default-built chunker dict, but
        # when the caller hands us a partial ``addon_params['chunker']``
        # that lacks the slot (e.g. ``{"paragraph_semantic": {}}``)
        # ``normalize_addon_params`` does not re-run the defaults
        # builder — so this overlay is the last guard that ensures the
        # slot is always populated.  Precedence (high → low):
        # explicit ``addon_params`` > ``CHUNK_P_SIZE`` env >
        # ``DEFAULT_CHUNK_P_SIZE``.  ``setdefault`` preserves any
        # explicit value the caller did provide; the env read here
        # mirrors ``default_chunker_config`` so partial-addon-params
        # callers still pick up env overrides.
        chunker_cfg["paragraph_semantic"].setdefault(
            "chunk_token_size",
            _chunk_env_int("CHUNK_P_SIZE", DEFAULT_CHUNK_P_SIZE),
        )
        # Per-strategy F/R/V chunk_token_size from strategy env
        # (CHUNK_F_SIZE / CHUNK_R_SIZE / CHUNK_V_SIZE).  Same rationale as the
        # P backfill above: ``default_chunker_config`` seeds these when it
        # builds the chunker dict from scratch, but a partial
        # ``addon_params['chunker']`` skips that builder
        # (``normalize_addon_params`` only defaults the whole ``chunker`` key
        # when it is absent), so this overlay is the last guard.  Unlike P,
        # the slot is filled ONLY when the env is actually set — leaving it
        # absent otherwise so F/R/V inherit the top-level ``chunk_token_size``
        # at consumption time.  ``setdefault`` preserves an explicit
        # caller-supplied value (tier 1 wins over the env tier 2).
        for strategy_key, size_env in (
            ("fixed_token", "CHUNK_F_SIZE"),
            ("recursive_character", "CHUNK_R_SIZE"),
            ("semantic_vector", "CHUNK_V_SIZE"),
        ):
            size_val = _chunk_env_int(size_env, None)
            if size_val is None:
                continue
            sub = chunker_cfg.get(strategy_key)
            if not isinstance(sub, dict):
                sub = {}
                chunker_cfg[strategy_key] = sub
            sub.setdefault("chunk_token_size", size_val)

        # Back-fill legacy instance fields → always int afterwards.
        # Overlap mirrors the F-strategy resolved value, matching the
        # F-flavoured legacy ``self.chunk_overlap_token_size`` semantics
        # used by the legacy 6-arg ``chunking_func`` path.
        self.chunk_token_size = chunker_cfg["chunk_token_size"]
        self.chunk_overlap_token_size = chunker_cfg["fixed_token"][
            "chunk_overlap_token_size"
        ]

    def _refresh_addon_params_cache(self) -> None:
        summary_language = self._addon_params.get("language", DEFAULT_SUMMARY_LANGUAGE)
        if not isinstance(summary_language, str) or not summary_language.strip():
            summary_language = DEFAULT_SUMMARY_LANGUAGE
        self._resolved_summary_language = summary_language

        resolved_prompt_profile = resolve_entity_extraction_prompt_profile(
            self._addon_params,
            self.entity_extraction_use_json,
        )
        self._entity_extraction_prompt_profile = (
            validate_entity_extraction_prompt_profile_for_mode(
                resolved_prompt_profile,
                self.entity_extraction_use_json,
                self._addon_params.get("entity_type_prompt_file"),
            )
        )
        self._cached_entity_extraction_use_json = self.entity_extraction_use_json
        self._addon_params_dirty = False

    def _ensure_addon_params_cache(self) -> None:
        if (
            not self._addon_params_dirty
            and self._cached_entity_extraction_use_json
            == self.entity_extraction_use_json
        ):
            return
        self._refresh_addon_params_cache()

    def _build_global_config(self) -> dict[str, Any]:
        self._ensure_addon_params_cache()
        global_config = asdict(self)
        # Restore the tokenizer object itself, the same way __post_init__ restores
        # embedding_func. This is about IDENTITY, not cost: asdict has already
        # deep-copied every non-dataclass field by the time we get here, so the
        # copy is made and discarded, and an injected tokenizer must still survive
        # copy.deepcopy (a bare threading.Lock raises TypeError — see Tokenizer).
        #
        # What it buys is one tokenizer object instead of a per-operation copy
        # nobody can trace back. That copy was never the isolation it looked like:
        # tiktoken caches encodings in a process-wide registry and
        # Encoding.__setstate__ rebinds a copy's __dict__ to the registered
        # instance's, so every "independent" copy already shared one CoreBPE. Now
        # that the injection contract is thread safety, sharing is what it asks for.
        global_config["tokenizer"] = self.tokenizer
        global_config.pop("_addon_params", None)
        global_config.pop("_addon_params_dirty", None)
        global_config.pop("_cached_entity_extraction_use_json", None)
        global_config["addon_params"] = dict(self._addon_params)
        # Inject runtime per-role wrapped LLM funcs (callable; not part of asdict
        # because they live in the private _role_llm_states map). The first
        # _build_global_config() call from __post_init__ runs before the role
        # state is built, so fall back to an empty dict in that case.
        states = getattr(self, "_role_llm_states", None) or {}
        global_config["role_llm_funcs"] = {
            spec.name: states[spec.name].wrapped if spec.name in states else None
            for spec in ROLES
        }
        global_config["llm_cache_identities"] = {
            spec.name: self._build_role_llm_cache_identity(
                spec.name, states.get(spec.name)
            )
            for spec in ROLES
        }
        return global_config

    def _build_role_llm_cache_identity(
        self, role: str, state: _RoleLLMState | None
    ) -> dict[str, Any]:
        # `state` is None during the first _build_global_config() call from
        # __post_init__ — role builders have not run yet, so metadata is empty
        # and we fall back to self.llm_model_name. Once roles are initialized
        # or aupdate_llm_role_config() runs, metadata always carries `model`.
        metadata = state.metadata if state is not None else {}
        return {
            "role": role,
            "binding": metadata.get("binding"),
            "model": metadata.get("model") or self.llm_model_name,
            "host": metadata.get("host"),
        }

    def __post_init__(self, addon_params: dict[str, Any] | None):
        from lightrag.kg.shared_storage import (
            initialize_share_data,
        )

        # Fail fast if deprecated ENTITY_TYPES env var is set
        if os.getenv("ENTITY_TYPES") is not None:
            raise SystemExit(
                "ERROR: ENTITY_TYPES environment variable is no longer supported. "
                "Please customize entity type guidance through the prompt template instead. "
                "Set addon_params={'entity_types_guidance': '...'} or replace the prompt template."
            )

        self._replace_addon_params(addon_params, mark_dirty=False)
        self._apply_chunk_size_overlay()
        self._refresh_addon_params_cache()

        # Bounded scheduling page size: 0 disables paging (single-scan legacy
        # behaviour); a negative value is a misconfiguration, fail fast.
        if self.pipeline_scheduling_page_size < 0:
            raise ValueError(
                "PIPELINE_SCHEDULING_PAGE_SIZE must be >= 0 (0 disables "
                f"paging); got {self.pipeline_scheduling_page_size}"
            )

        # Admission capacity: 0 disables admission control; a negative capacity
        # would refuse every request, which is never what an operator meant.
        if self.max_pending_documents < 0:
            raise ValueError(
                "MAX_PENDING_DOCUMENTS must be >= 0 (0 disables admission "
                f"control); got {self.max_pending_documents}"
            )

        # Embedding hard-fallback overlap: 0 disables it; negative is a
        # misconfiguration (there is no such thing as negative overlap).
        if self.embedding_chunk_overlap_token_size < 0:
            raise ValueError(
                "EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE must be >= 0 (0 disables "
                f"the embedding hard fallback's overlap); got "
                f"{self.embedding_chunk_overlap_token_size}"
            )

        # Handle deprecated parameters
        if self.log_level is not None:
            warnings.warn(
                "WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )
        if self.log_file_path is not None:
            warnings.warn(
                "WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )

        # Remove these attributes to prevent their use
        if hasattr(self, "log_level"):
            delattr(self, "log_level")
        if hasattr(self, "log_file_path"):
            delattr(self, "log_file_path")

        initialize_share_data()

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Init Tokenizer
        # Post-initialization hook to handle backward compatabile tokenizer initialization based on provided parameters
        if self.tokenizer is None:
            if self.tiktoken_model_name:
                self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
            else:
                self.tokenizer = TiktokenTokenizer()

        # Initialize ollama_server_infos if not provided
        if self.ollama_server_infos is None:
            self.ollama_server_infos = OllamaServerInfos()

        # Validate config
        if self.force_llm_summary_on_merge < 3:
            logger.warning(
                f"force_llm_summary_on_merge should be at least 3, got {self.force_llm_summary_on_merge}"
            )
        if self.summary_context_size > self.max_total_tokens:
            logger.warning(
                f"summary_context_size({self.summary_context_size}) should not be greater than max_total_tokens({self.max_total_tokens})"
            )
        if self.summary_length_recommended > self.summary_max_tokens:
            logger.warning(
                f"summary_max_tokens({self.summary_max_tokens}) should be greater than summary_length_recommended({self.summary_length_recommended})"
            )

        if self.rerank_model_func is not None:
            self.rerank_model_func = priority_limit_async_func_call(
                self.rerank_model_max_async,
                llm_timeout=self.default_rerank_timeout,
                queue_name="Rerank func",
                concurrency_group="rerank",
            )(self.rerank_model_func)

        # Init Embedding
        # Step 1: Capture embedding_func and max_token_size before applying rate_limit decorator
        original_embedding_func = self.embedding_func
        embedding_max_token_size = None
        if self.embedding_func and hasattr(self.embedding_func, "max_token_size"):
            embedding_max_token_size = self.embedding_func.max_token_size
            logger.debug(
                f"Captured embedding max_token_size: {embedding_max_token_size}"
            )
        self.embedding_token_limit = embedding_max_token_size

        # Fix global_config now
        global_config = self._build_global_config()
        # Restore original EmbeddingFunc object (asdict converts it to dict)
        global_config["embedding_func"] = original_embedding_func

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Step 2: Apply priority wrapper decorator to EmbeddingFunc's inner func
        # Create a NEW EmbeddingFunc instance with the wrapped func to avoid mutating the caller's object
        # This ensures _generate_collection_suffix can still access attributes (model_name, embedding_dim)
        # while preventing side effects when the same EmbeddingFunc is reused across multiple LightRAG instances
        if self.embedding_func is not None:
            wrapped_func = priority_limit_async_func_call(
                self.embedding_func_max_async,
                llm_timeout=self.default_embedding_timeout,
                queue_name="Embedding func",
                concurrency_group="embedding",
            )(self.embedding_func.func)
            # Use dataclasses.replace() to create a new instance, leaving the original unchanged
            self.embedding_func = replace(self.embedding_func, func=wrapped_func)

        # Initialize all storages
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = get_storage_class(
            self.kv_storage
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage
        self.doc_status_storage_cls = get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=self.embedding_func,
        )

        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_entities: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_relations: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_RELATIONS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entity_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.relation_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_RELATION_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=NameSpace.DOC_STATUS,
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=None,
        )

        # Per-role isolated LLM wrappers (independent queues per role).
        # The base ``self.llm_model_func`` is intentionally NOT queue-wrapped:
        # every code path that calls an LLM goes through one of the role
        # wrappers built below, so concurrency is enforced at the role layer.
        base_llm_func = self.llm_model_func
        if base_llm_func is None:
            raise ValueError("llm_model_func must be provided")

        self._llm_role_builder = None
        self._retired_llm_queue_cleanup_tasks: set[asyncio.Task] = set()

        # The event loop this instance's storages bind to (set in
        # initialize_storages). Kept off the dataclass fields so asdict() in
        # _build_global_config() never tries to (deep)copy a live loop — that
        # raises TypeError on Python 3.14. _run_sync uses it only as a reference
        # for the cross-loop guard.
        self._owning_loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-instance native-parser thread pool (lazily built; see
        # _get_parse_native_executor). Kept off the dataclass fields — a live
        # executor/Event must never be deep-copied by asdict(). The shutdown
        # event is polled by parser LLM bridges so an extract waiting on an
        # LLM exits within one poll interval of finalize_storages().
        self._parser_executor: Optional[ThreadPoolExecutor] = None
        self._parser_shutdown_event: threading.Event = threading.Event()

        user_role_configs = self.role_llm_configs or {}
        if not isinstance(user_role_configs, Mapping):
            raise TypeError(
                "role_llm_configs must be a Mapping or None, got "
                f"{type(user_role_configs).__name__}"
            )
        unknown_roles = [role for role in user_role_configs if role not in ROLE_NAMES]
        if unknown_roles:
            valid_roles = ", ".join(sorted(ROLE_NAMES))
            unknown = ", ".join(repr(role) for role in unknown_roles)
            raise ValueError(
                f"Unknown role_llm_configs key(s): {unknown}. "
                f"Valid roles are: {valid_roles}"
            )

        self._role_llm_states: dict[str, _RoleLLMState] = {}
        for spec in ROLES:
            override = user_role_configs.get(spec.name)
            if override is None:
                cfg = RoleLLMConfig()
            elif isinstance(override, RoleLLMConfig):
                cfg = override
            elif isinstance(override, Mapping):
                cfg = RoleLLMConfig(**dict(override))
            else:
                raise TypeError(
                    f"role_llm_configs[{spec.name!r}] must be RoleLLMConfig or "
                    f"a dict, got {type(override).__name__}"
                )

            max_async = cfg.max_async
            if max_async is None:
                max_async = _optional_env_int(f"{spec.env_prefix}_MAX_ASYNC_LLM")

            metadata = {}
            if cfg.metadata is not None:
                if not isinstance(cfg.metadata, Mapping):
                    raise TypeError(
                        f"role_llm_configs[{spec.name!r}].metadata must be a "
                        f"Mapping or None, got {type(cfg.metadata).__name__}"
                    )
                metadata = deepcopy(dict(cfg.metadata))

            self._role_llm_states[spec.name] = _RoleLLMState(
                raw_func=cfg.func or base_llm_func,
                kwargs=cfg.kwargs,
                max_async=max_async,
                timeout=cfg.timeout,
                metadata=metadata,
            )

        self._rebuild_role_llm_funcs()
        self._log_llm_role_config("initialized")

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Storage initialization must be called one by one to prevent deadlock"""
        if self._storages_status == StoragesStatus.CREATED:
            # Record the loop the storages (and their shared_storage locks) bind
            # to, so the synchronous wrappers can fail fast if later driven from a
            # different loop (run_in_executor / a loop on another thread).
            self._owning_loop = asyncio.get_running_loop()

            # Set the first initialized workspace will set the default workspace
            # Allows namespace operation without specifying workspace for backward compatibility
            default_workspace = get_default_workspace()
            if default_workspace is None:
                set_default_workspace(self.workspace)
            elif default_workspace != self.workspace:
                logger.info(
                    f"Creating LightRAG instance with workspace='{self.workspace}' "
                    f"while default workspace is set to '{default_workspace}'"
                )

            # Auto-initialize pipeline_status for this workspace
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status(workspace=self.workspace)

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    # logger.debug(f"Initializing storage: {storage}")
                    await storage.initialize()

            # After initialize(), so a backend that derives capabilities during
            # startup (Redis builds its status/source indexes there) is judged on
            # what it can actually do. Raises only under require=True.
            enforce_strict_storage_capabilities(
                self.doc_status,
                require=bool(self.pipeline_require_strict_storage_reads),
            )

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("All storage types initialized")

    def _get_parse_native_executor(self) -> ThreadPoolExecutor:
        """Lazily build the per-instance native-parser thread pool.

        Owned by the rag instance (NOT the parser): parser objects are
        registry-cached singletons shared across instances, the pool size
        tracks this instance's ``max_parallel_parse_native``, and only the
        instance has a shutdown point (``finalize_storages``). Pool size
        equals the parse worker count, so at most one extract per worker —
        the pool itself never queues.
        """
        if self._parser_executor is None:
            self._parser_executor = ThreadPoolExecutor(
                max_workers=max(1, int(self.max_parallel_parse_native)),
                thread_name_prefix="parse-native",
            )
        return self._parser_executor

    def _shutdown_parser_executor(self) -> None:
        """Tear down the parser pool (idempotent; called by finalize).

        Order matters: set the CURRENT shutdown event first so any bridge
        blocked on an LLM wait exits within one poll interval, then shut the
        executor down without waiting (``cancel_futures`` only cancels
        not-yet-started work; running extracts exit via the event). A fresh
        event replaces the old one so a later re-initialize + parse does not
        observe a stale set event — in-flight bridges keep their reference
        to the old (set) event.
        """
        self._parser_shutdown_event.set()
        executor = self._parser_executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._parser_executor = None
        self._parser_shutdown_event = threading.Event()

    async def finalize_storages(self):
        """Asynchronously finalize the storages with improved error handling"""
        self._shutdown_parser_executor()
        if self._storages_status == StoragesStatus.INITIALIZED:
            storages = [
                ("full_docs", self.full_docs),
                ("text_chunks", self.text_chunks),
                ("full_entities", self.full_entities),
                ("full_relations", self.full_relations),
                ("entity_chunks", self.entity_chunks),
                ("relation_chunks", self.relation_chunks),
                ("entities_vdb", self.entities_vdb),
                ("relationships_vdb", self.relationships_vdb),
                ("chunks_vdb", self.chunks_vdb),
                ("chunk_entity_relation_graph", self.chunk_entity_relation_graph),
                ("llm_response_cache", self.llm_response_cache),
                ("doc_status", self.doc_status),
            ]

            # Finalize each storage individually to ensure one failure doesn't prevent others from closing
            successful_finalizations = []
            failed_finalizations = []

            for storage_name, storage in storages:
                if storage:
                    try:
                        await storage.finalize()
                        successful_finalizations.append(storage_name)
                        logger.debug(f"Successfully finalized {storage_name}")
                    except Exception as e:
                        error_msg = f"Failed to finalize {storage_name}: {e}"
                        logger.error(error_msg)
                        failed_finalizations.append(storage_name)

            # Log summary of finalization results
            if successful_finalizations:
                logger.info(
                    f"Successfully finalized {len(successful_finalizations)} storages"
                )

            if failed_finalizations:
                logger.error(
                    f"Failed to finalize {len(failed_finalizations)} storages: {', '.join(failed_finalizations)}"
                )
            else:
                logger.debug("All storages finalized successfully")

            self._storages_status = StoragesStatus.FINALIZED

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to self.max_graph_nodes.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """
        # Use self.max_graph_nodes as default if max_nodes is None
        if max_nodes is None:
            max_nodes = self.max_graph_nodes
        else:
            # Limit max_nodes to not exceed self.max_graph_nodes
            max_nodes = min(max_nodes, self.max_graph_nodes)

        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label, max_depth, max_nodes
        )

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated

        Returns:
            str: tracking ID for monitoring processing status
        """
        return _run_sync(
            lambda: self.ainsert(
                input,
                split_by_character,
                split_by_character_only,
                ids,
                file_paths,
                track_id,
            ),
            sync_name="insert",
            async_name="ainsert",
            owning_loop=self._owning_loop,
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """Async insert documents with checkpoint support (fixed-token chunking only).

        SDK convenience entry point. It **always** chunks with the fixed-token
        (F) strategy: ``process_options`` is intentionally not passed, so the
        document runs the F chunker. ``split_by_character`` /
        ``split_by_character_only`` are F-strategy runtime args; the rest of
        the F config (``chunk_token_size`` / ``chunk_overlap_token_size``,
        seeded from ``CHUNK_F_SIZE`` / ``CHUNK_SIZE`` etc.) comes from
        ``addon_params['chunker']['fixed_token']``. ``ainsert`` cannot select
        the recursive-character (R), semantic-vector (V), or paragraph-semantic
        (P) strategies.

        The LightRAG **server / REST API does not call this method** — it
        ingests via :meth:`apipeline_enqueue_documents` +
        :meth:`apipeline_process_enqueue_documents` with a per-document
        ``process_options`` selector, which is how F/R/V/P are chosen there.
        To use R/V/P (or pass an explicit per-document ``chunk_options``) from
        the SDK, call those two methods directly with ``process_options=…``
        instead of ``ainsert``.

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated

        Returns:
            str: tracking ID for monitoring processing status
        """
        # Generate track_id if not provided
        if track_id is None:
            track_id = generate_track_id("insert")

        # Capture the F-strategy runtime args into a chunk_options
        # snapshot before enqueue so they become a per-document
        # setting.  ``apipeline_enqueue_documents`` itself doesn't take
        # split args — chunk_options is the canonical chunker-config
        # carrier; runtime split args are an ainsert-only concern.
        from lightrag.parser.routing import resolve_chunk_options

        chunk_opts = resolve_chunk_options(
            self.addon_params,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
        )
        await self.apipeline_enqueue_documents(
            input,
            ids,
            file_paths,
            track_id,
            chunk_options=chunk_opts,
        )
        await self.apipeline_process_enqueue_documents()

        return track_id

    # TODO: deprecated, use insert instead
    def insert_custom_chunks(
        self,
        full_text: str,
        text_chunks: list[str],
        doc_id: str | list[str] | None = None,
    ) -> None:
        _run_sync(
            lambda: self.ainsert_custom_chunks(full_text, text_chunks, doc_id),
            sync_name="insert_custom_chunks",
            async_name="ainsert_custom_chunks",
            owning_loop=self._owning_loop,
        )

    # TODO: deprecated, use ainsert instead
    async def ainsert_custom_chunks(
        self, full_text: str, text_chunks: list[str], doc_id: str | None = None
    ) -> None:
        """Insert caller-chunked content as a journaled, recoverable operation.

        Issue #3400 Phase 3 semantics — one invocation is one incremental
        operation with a durable journal in ``doc_status.metadata``:

        - Target document absent → **create** mode: the document is created
          with full doc_status bookkeeping (the legacy behavior of this
          deprecated API, now crash-discoverable).
        - Target document ``PROCESSED`` → **patch** mode: the chunks are added
          to the document; recovery anchors and ``chunks_list`` are unioned at
          commit, never overwritten.
        - Target document carrying a journal for the SAME operation (same
          doc + same chunk set) → resume/roll-forward; for a DIFFERENT
          operation → rejected until the original is retried or rolled back
          by scan.
        - Target document in any other state → rejected.

        Identity is deterministic and document-scoped: chunk ids hash
        ``(doc_id, chunk_content)`` and the operation id hashes the ordered
        chunk-id set, so repeating the same logical input is idempotent
        (committed → no-op; failed → resume).

        On failure/cancellation after the journal is durable the document is
        marked ``FAILED`` with the journal (and any staged data) retained —
        the same call can be retried by the SDK caller, and ``/documents/scan``
        can roll the operation back. Partial work is never blind-flushed.
        """
        # Owner token for the busy reservation, generated before any await so the
        # finally always holds it and can release the slot by owner (even if the
        # acquire is cancelled at the lock exit).
        token = uuid.uuid4().hex
        busy_acquired = False
        pipeline_status = None
        pipeline_status_lock = None
        # Set when the operation fails after acquiring busy: the finally then
        # discards partial buffers instead of flushing them (issue #3400:
        # failure finalization must not persist partial work).
        op_failed = False
        try:
            # Clean input texts
            full_text = sanitize_text_for_encoding(full_text)
            text_chunks = [sanitize_text_for_encoding(chunk) for chunk in text_chunks]
            file_path = normalize_document_file_path("")

            # Process cleaned texts
            if doc_id is None:
                doc_key = compute_mdhash_id(full_text, prefix="doc-")
            else:
                doc_key = doc_id

            # Deterministic, document-scoped identity: chunk ids hash
            # (doc_key, content) — length-prefixed, see make_custom_chunk_id —
            # so identical text in two documents never shares a chunk row (a
            # shared row's full_doc_id would be clobbered, and rollback could
            # delete another document's chunk). The operation id hashes the
            # ordered chunk-id set so the same logical input is the same
            # operation across retries.
            chunk_entries: list[tuple[str, str, str]] = []
            seen_chunk_ids: set[str] = set()
            for chunk_text in text_chunks:
                if not chunk_text:
                    continue
                chunk_id = make_custom_chunk_id(doc_key, chunk_text)
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunk_entries.append(
                    (chunk_id, chunk_text, compute_text_content_hash(chunk_text))
                )
            if not chunk_entries:
                logger.warning("No non-empty custom chunks to insert.")
                return
            operation_id = make_custom_chunk_operation_id(
                doc_key, [cid for cid, _, _ in chunk_entries]
            )

            # pipeline_status/lock are already initialized by
            # initialize_storages() (which is idempotent), so fetch the existing
            # shared state rather than re-initializing it here.
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=self.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=self.workspace
            )

            # Reserve the pipeline busy slot before doing any extraction/merge.
            # This path now writes the KG (Stage 3), so it must be mutually
            # exclusive with the processing loop, destructive ops, and another
            # custom-chunks insert — the single-writer contract the file
            # pipeline enforces via ``busy``. Entity-keyed locks already keep
            # the graph data consistent under concurrency, but without this a
            # second merger would still run: LLM concurrency doubles past
            # ``llm_model_max_async``, ``pipeline_status`` misreports idle, and
            # the shared ``cancellation_requested`` flag cross-cancels. Reject
            # (rather than queue) when the slot is taken, mirroring
            # ``_acquire_destructive_busy``; ``destructive_busy`` implies
            # ``busy`` so a running clear/delete is covered too.
            # Pre-arm owner-checked cleanup before acquire so cancellation at the
            # helper's lock exit cannot leak the slot.
            busy_acquired = True
            reservation = await acquire_reservation(
                pipeline_status,
                pipeline_status_lock,
                owner_key="busy_owner",
                owner=token,
                owner_kind="custom_chunks",
                flags={
                    "busy": True,
                    "operation_record": {
                        "kind": "custom_chunks",
                        "doc_id": doc_key,
                    },
                    "job_name": "Custom chunks insert",
                    "job_start": datetime.now(timezone.utc).isoformat(),
                    "latest_message": "Inserting custom chunks",
                    "cancellation_requested": False,
                    "cancellation_reason": None,
                    "cancellation_detail": None,
                },
                reject_when=(
                    (
                        "busy",
                        "Pipeline is busy with another operation; "
                        "ainsert_custom_chunks cannot run concurrently. Wait for "
                        "it to finish and retry.",
                    ),
                    (
                        "scanning",
                        "A document scan is in progress; ainsert_custom_chunks "
                        "cannot run concurrently. Wait for the scan to finish and "
                        "retry.",
                    ),
                ),
            )
            if not reservation.acquired:
                busy_acquired = False
                raise RuntimeError(reservation.message)

            # Per-document keyed lock: only one custom-chunk operation may be
            # active for a document (issue #3400 §2.5). The busy reservation
            # already serializes writers globally; the keyed lock preserves
            # correctness if that global policy is ever relaxed.
            doc_lock_namespace = (
                f"{self.workspace}:DocPatch" if self.workspace else "DocPatch"
            )
            async with get_storage_keyed_lock(
                [doc_key], namespace=doc_lock_namespace, enable_logging=False
            ):
                # ---- Resolve operation mode under the reservation ----
                existing_row = await self.doc_status.get_by_id(doc_key)
                existing_journal = doc_status_custom_chunk_patch(existing_row)

                def _row_status_text(row: dict | None) -> str:
                    raw = (row or {}).get("status")
                    return raw.value if isinstance(raw, DocStatus) else str(raw or "")

                resume = False
                if existing_journal is not None:
                    if existing_journal.get("operation_id") != operation_id:
                        raise RuntimeError(
                            f"Document {doc_key} has an unfinished custom-chunk "
                            f"operation {existing_journal.get('operation_id')}; "
                            "retry that operation with its original input, or "
                            "run /documents/scan to roll it back, before "
                            "submitting a different one."
                        )
                    mode = existing_journal.get("mode") or "patch"
                    resume = True
                elif existing_row is None:
                    mode = "create"
                elif _row_status_text(existing_row) == DocStatus.PROCESSED.value:
                    mode = "patch"
                else:
                    raise RuntimeError(
                        f"Document {doc_key} is in status "
                        f"'{_row_status_text(existing_row)}'; only PROCESSED "
                        "documents (or the document's own unfinished "
                        "operation) can receive a custom-chunk insert."
                    )

                if mode == "create" and not resume:
                    # Legacy dedup: a document created by an earlier call (or
                    # an old build without doc_status bookkeeping) is a
                    # committed no-op, mirroring the historical behavior.
                    missing_keys = await self.full_docs.filter_keys({doc_key})
                    if doc_key not in missing_keys:
                        logger.warning("This document is already in the storage.")
                        return

                if mode == "patch":
                    # Committed-content dedup: drop chunks whose content the
                    # base document already owns, so repeating an
                    # already-committed patch (or re-adding base content) is
                    # idempotent instead of double-merging contributions.
                    committed_ids = [
                        cid
                        for cid in ((existing_row or {}).get("chunks_list") or [])
                        if isinstance(cid, str) and cid
                    ]
                    committed_hashes: set[str] = set()
                    if committed_ids:
                        committed_rows = await self.text_chunks.get_by_ids(
                            committed_ids
                        )
                        for row in committed_rows:
                            if isinstance(row, dict) and row.get("content"):
                                committed_hashes.add(
                                    compute_text_content_hash(row["content"])
                                )
                    chunk_entries = [
                        entry
                        for entry in chunk_entries
                        if entry[2] not in committed_hashes
                    ]
                    if not chunk_entries and not resume:
                        logger.warning(
                            "All custom chunks are already committed to this "
                            "document; nothing to patch."
                        )
                        return

                logger.info(
                    f"Custom chunks {mode} for {doc_key}: "
                    f"{len(chunk_entries)} chunk(s), operation {operation_id}"
                    f"{' (resume)' if resume else ''}"
                )

                # ---- Journal first: durable PROCESSING + operation record
                # BEFORE any data store is touched (discoverability before
                # mutation). doc_status backends persist on upsert.
                now_iso = datetime.now(timezone.utc).isoformat()
                # Pre-operation llm_truncation snapshot (None = key absent).
                # Captured on the FIRST attempt, while ``existing_row`` is
                # still the untouched base document; a resume MUST reuse the
                # journaled value instead of re-reading the row, because the
                # failed attempt's terminal write already stamped its own
                # tally there. This snapshot is what the terminal writes merge
                # the operation tally INTO, and what a rollback restores.
                if existing_journal and "prior_llm_truncation" in existing_journal:
                    prior_truncation = existing_journal["prior_llm_truncation"]
                elif mode == "patch":
                    prior_truncation = ((existing_row or {}).get("metadata") or {}).get(
                        LLM_TRUNCATION_METADATA_KEY
                    )
                else:
                    # create: the document is born with this operation, so
                    # there is nothing prior to preserve or restore.
                    prior_truncation = None
                # Truncation accumulated by PREVIOUS attempts of this
                # operation. Truncated extraction responses are never cached,
                # so a resume re-runs those calls and can get a clean sample —
                # but the failed attempt's partial graph mutations stay (the
                # resume is additive, it never purges them). Without this
                # carry, a clean resume's terminal write would merge only
                # snapshot + its own empty tally and silently drop the record
                # of the truncated output still living in the graph. Captured
                # ONCE from the journal as loaded — the applying/FAILED writes
                # below fold the current attempt into the journal copy, and
                # recomputing from that copy would double-count (the merge
                # sums event counts exactly).
                carried_operation_truncation = (existing_journal or {}).get(
                    "operation_llm_truncation"
                )
                journal: dict[str, Any] = {
                    "schema_version": 1,
                    "operation_id": operation_id,
                    "mode": mode,
                    "chunk_ids": [cid for cid, _, _ in chunk_entries],
                    "content_hashes": [h for _, _, h in chunk_entries],
                    "phase": "prepared",
                    "entity_names": (existing_journal or {}).get("entity_names", []),
                    "relation_pairs": (existing_journal or {}).get(
                        "relation_pairs", []
                    ),
                    "prior_llm_truncation": prior_truncation,
                    "operation_llm_truncation": carried_operation_truncation,
                    "created_at": (existing_journal or {}).get("created_at") or now_iso,
                    "updated_at": now_iso,
                }
                status_row = await self._upsert_custom_chunk_status(
                    doc_key,
                    DocStatus.PROCESSING,
                    base_row=existing_row,
                    journal=journal,
                    full_text=full_text,
                    file_path=file_path,
                )
                # Operation-scoped truncation record, fed by both KG stages
                # below and stamped onto whichever terminal transition this
                # operation reaches — the same durable evidence the normal
                # pipeline writes, for the path that does not go through it.
                truncation_tally = TokenLimitTruncationTally()

                def _operation_truncation_record() -> dict[str, Any] | None:
                    """Previous attempts' accumulated payload + this attempt.

                    Always recomputed from ``carried_operation_truncation``
                    (previous attempts ONLY), never from the journal copy the
                    applying write already folded this attempt into — the
                    merge sums event counts exactly, so folding twice would
                    double-count.
                    """
                    return merge_truncation_metadata(
                        carried_operation_truncation,
                        truncation_tally.as_metadata(),
                    )

                def _terminal_truncation_kwargs() -> dict[str, Any]:
                    """Set-or-drop arguments for a terminal status write.

                    The operation-accumulated record (every attempt of this
                    operation, current one included) is merged into the
                    journal's pre-operation snapshot — never into the live
                    row, whose value after a failed attempt is that attempt's
                    own stamp and would double-count across resumes. Replace
                    semantics with an explicit drop: when neither the base run
                    nor any attempt of this operation truncated, the key must
                    come OFF the row (the live row may still carry a dead
                    attempt's stamp).
                    """
                    merged = merge_truncation_metadata(
                        journal.get("prior_llm_truncation"),
                        _operation_truncation_record(),
                    )
                    if merged is None:
                        return {"metadata_drop": (LLM_TRUNCATION_METADATA_KEY,)}
                    return {"metadata_extra": {LLM_TRUNCATION_METADATA_KEY: merged}}

                truncation_persist_lock = asyncio.Lock()

                async def _journal_truncation_write_ahead(
                    pending: TokenLimitTruncationTally,
                ) -> None:
                    """Journal a truncation event BEFORE the mutation it warns about.

                    Awaited by the merge's summary path between recording a
                    truncated summary and upserting the object that carries
                    it, so even a hard crash (no FAILED write) cannot leave
                    truncated output in the graph with no journaled evidence.
                    ``pending`` is the merge-local summary tally: its events
                    reach the operation tally only when the merge publishes,
                    which happens after every hook call has completed (a
                    failing phase drains its sibling tasks first) — so the two
                    sides of this merge are disjoint, and the later FAILED or
                    terminal recompute overwrites this snapshot with a
                    superset. A hook failure fails the recording entity or
                    relation itself — fail-closed into the normal FAILED
                    path. Fires once per truncation event (rare); serialized
                    because sibling merges can record concurrently.
                    """
                    nonlocal status_row
                    async with truncation_persist_lock:
                        status_row = await self._upsert_custom_chunk_status(
                            doc_key,
                            DocStatus.PROCESSING,
                            base_row=status_row,
                            journal={
                                **journal,
                                "operation_llm_truncation": merge_truncation_metadata(
                                    _operation_truncation_record(),
                                    pending.as_metadata(),
                                ),
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                try:
                    # Stage 1 (barrier): persist chunks BEFORE extraction.
                    # Extraction records per-chunk LLM cache references
                    # (update_chunk_cache_list -> text_chunks) keyed by chunk id
                    # and reads chunks back via get_by_id; running it
                    # concurrently with these upserts can observe a
                    # not-yet-persisted chunk and silently drop the cache
                    # reference.
                    # Off the loop: this shares ``self.tokenizer`` with the
                    # chunking executor, and this entry point is gated by
                    # neither max_parallel_insert nor the pipeline busy flag, so
                    # nothing else keeps it from encoding concurrently with a
                    # document being chunked.
                    def _build_inserting_chunks() -> dict[str, Any]:
                        return {
                            chunk_id: {
                                "content": content,
                                "full_doc_id": doc_key,
                                "tokens": len(self.tokenizer.encode(content)),
                                "chunk_order_index": index,
                                "file_path": file_path,
                            }
                            for index, (chunk_id, content, _) in enumerate(
                                chunk_entries
                            )
                        }

                    inserting_chunks: dict[str, Any] = await run_in_chunking_executor(
                        _build_inserting_chunks
                    )
                    stage1_writes = [
                        self.chunks_vdb.upsert(inserting_chunks),
                        self.text_chunks.upsert(inserting_chunks),
                    ]
                    if mode == "create":
                        stage1_writes.append(
                            self.full_docs.upsert(
                                {
                                    doc_key: {
                                        "content": full_text,
                                        "file_path": file_path,
                                    }
                                }
                            )
                        )
                    await asyncio.gather(*stage1_writes)

                    # Stage 2: extract now that chunks are persisted, then make
                    # the staged chunks AND the extraction cache durable before
                    # any graph mutation — a resume must reuse the cached
                    # extraction, never call the LLM again and merge a
                    # different result into a partially applied operation.
                    chunk_results = await self._process_extract_entities(
                        inserting_chunks,
                        pipeline_status,
                        pipeline_status_lock,
                        truncation_tally=truncation_tally,
                    )
                    staging_flush = [
                        self.text_chunks,
                        self.chunks_vdb,
                        self.llm_response_cache,
                    ]
                    if mode == "create":
                        staging_flush.append(self.full_docs)
                    await self._flush_storages(staging_flush)

                    # Persist the complete candidate set in the journal BEFORE
                    # merging — the write-ahead anchor for this operation.
                    #
                    # UNIONED with the previous attempt's candidates (carried
                    # into ``journal`` at the prepared write), never replaced:
                    # truncated extraction responses are deliberately excluded
                    # from the LLM cache, so a resume re-runs those calls and
                    # can extract a DIFFERENT sample. A previous attempt whose
                    # Stage 3 merge partially applied has already written ITS
                    # candidates into the graph; dropping them here would strand
                    # exactly those objects — the journal is this operation's
                    # only recovery anchor and must keep naming the complete
                    # superset across every attempt. Union is safe on the
                    # consuming side by contract: candidates are a superset,
                    # and purge/commit treat absent objects as no-ops.
                    candidate_entities, candidate_relations = (
                        collect_kg_merge_candidates(chunk_results or [])
                    )
                    candidate_entities |= {
                        name
                        for name in (journal.get("entity_names") or [])
                        if isinstance(name, str) and name
                    }
                    candidate_relations |= {
                        (pair[0], pair[1])
                        for pair in (journal.get("relation_pairs") or [])
                        if isinstance(pair, (list, tuple)) and len(pair) == 2
                    }
                    journal = {
                        **journal,
                        "phase": "applying",
                        "entity_names": sorted(candidate_entities),
                        "relation_pairs": [
                            list(pair) for pair in sorted(candidate_relations)
                        ],
                        # Write-ahead for the truncation record too. The
                        # invariant: any truncation event whose subject's
                        # graph mutation has landed is already journaled —
                        # extraction-stage events by this write (which
                        # precedes ALL merge mutation), summary-stage events
                        # by _journal_truncation_write_ahead (awaited before
                        # the object carrying the truncated summary is
                        # upserted).
                        "operation_llm_truncation": _operation_truncation_record(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    status_row = await self._upsert_custom_chunk_status(
                        doc_key,
                        DocStatus.PROCESSING,
                        base_row=status_row,
                        journal=journal,
                    )

                    # Stage 3: merge into the knowledge graph. In patch mode
                    # the merge must NOT touch the base document's
                    # full_entities/full_relations rows (passing None skips the
                    # merge's own anchor prewrite — the journal is this
                    # operation's anchor); candidates are UNIONED into the base
                    # rows only at commit. Create mode uses the merge's normal
                    # write-ahead anchor prewrite.
                    if chunk_results:
                        await merge_nodes_and_edges(
                            chunk_results=chunk_results,
                            knowledge_graph_inst=self.chunk_entity_relation_graph,
                            entity_vdb=self.entities_vdb,
                            relationships_vdb=self.relationships_vdb,
                            global_config=self._build_global_config(),
                            full_entities_storage=(
                                self.full_entities if mode == "create" else None
                            ),
                            full_relations_storage=(
                                self.full_relations if mode == "create" else None
                            ),
                            doc_id=doc_key,
                            pipeline_status=pipeline_status,
                            pipeline_status_lock=pipeline_status_lock,
                            llm_response_cache=self.llm_response_cache,
                            entity_chunks_storage=self.entity_chunks,
                            relation_chunks_storage=self.relation_chunks,
                            file_path=file_path,
                            truncation_tally=truncation_tally,
                            truncation_write_ahead=_journal_truncation_write_ahead,
                        )

                    # ---- Commit: flush ALL derived stores, union the
                    # accumulated candidates into the document's anchors, then
                    # write PROCESSED (the commit record) last with the
                    # journal cleared.
                    await self._insert_done()

                    # Patch mode AND resumed creates, not patch alone. Patch
                    # merges pass None for the anchor storages, so this union
                    # is their only anchor write. Create merges DO write
                    # anchors in Phase 0 — but from the current attempt's
                    # chunk_results only (replace semantics, correct for the
                    # pipeline's whole-document reprocess). A resumed create
                    # can extract a DIFFERENT sample (truncated responses are
                    # never cached), and the PROCESSED write below clears the
                    # journal — without this union, first-attempt-only graph
                    # objects would be named nowhere durable and stranded
                    # from later document purges. A first-attempt create
                    # skips it: Phase 0 already wrote exactly these
                    # candidates.
                    if mode == "patch" or resume:
                        await self._union_doc_recovery_anchors(
                            doc_key, candidate_entities, candidate_relations
                        )
                    if mode == "patch":
                        base_chunks = [
                            cid
                            for cid in ((status_row or {}).get("chunks_list") or [])
                            if isinstance(cid, str) and cid
                        ]
                        final_chunks_list = list(
                            dict.fromkeys(base_chunks + list(inserting_chunks.keys()))
                        )
                    else:
                        final_chunks_list = list(inserting_chunks.keys())

                    await self._upsert_custom_chunk_status(
                        doc_key,
                        DocStatus.PROCESSED,
                        base_row=status_row,
                        journal=None,
                        chunks_list=final_chunks_list,
                        **_terminal_truncation_kwargs(),
                    )
                except BaseException as op_exc:
                    # Journal is durable: record FAILED, keep the journal and
                    # every staged artifact for SDK resume or scan rollback,
                    # then re-raise. Merge siblings are already drained by
                    # wait_tasks_with_drain inside merge_nodes_and_edges.
                    op_failed = True
                    try:
                        failed_journal = {
                            **journal,
                            "phase": "failed",
                            # Recompute — overwrites the applying write's and
                            # the write-ahead hook's copies (which already
                            # folded this attempt in) with the same
                            # carried-plus-current merge; the tally has
                            # absorbed the merge's summary events by now, so
                            # this is always a superset of those snapshots.
                            "operation_llm_truncation": (
                                _operation_truncation_record()
                            ),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        await self._upsert_custom_chunk_status(
                            doc_key,
                            DocStatus.FAILED,
                            base_row=status_row,
                            journal=failed_journal,
                            error_msg=str(op_exc)[:500],
                            **_terminal_truncation_kwargs(),
                        )
                    except Exception as bookkeeping_error:
                        logger.error(
                            "Failed to record FAILED custom-chunk journal for "
                            f"{doc_key}: {bookkeeping_error}"
                        )
                    raise

        finally:
            # Everything here is gated on busy_acquired. If we rejected
            # (busy/scanning) or returned before acquiring, we neither own the
            # slot nor wrote any data, so we must NOT flush/discard the SHARED
            # pending buffers — doing so could commit or tear down another
            # running job's in-flight index ops.
            if busy_acquired:
                # The handoff decision consults the ingress mailbox: work that
                # arrived while we held busy lives only in the mailbox (a
                # document message, the auto-rescan flag, or a sticky manual
                # retry request that was refused a drive because WE were the
                # busy holder).  A resolve/probe failure fails TOWARD handoff:
                # the driven run re-probes the mailbox itself, so a transient
                # flake self-heals, and a persistent failure raises inside
                # that run whose finally releases busy — never a wedge, while
                # releasing here instead would silently defer a committed
                # manual retry or an enqueued document to the next unrelated
                # trigger.
                #
                # This await precedes the cancellation-resistant release —
                # safe only because get_pipeline_ingress is suspension-free
                # by contract (see its docstring): a pending re-cancellation
                # cannot be delivered here and skip the releases below.
                try:
                    handoff_ingress = await get_pipeline_ingress(self.workspace)
                except Exception as ingress_error:
                    handoff_ingress = None
                    logger.warning(
                        "custom-chunks exit: pipeline ingress unavailable; "
                        "failing toward handoff so mailbox-only work is not "
                        f"silently deferred: {ingress_error}"
                    )

                def _exit_action(status):
                    # Clear cancellation (mirroring the processing loop) and
                    # decide, atomically: if a doc was enqueued while we held
                    # busy, hand the slot to a processing run WITHOUT releasing
                    # busy (releasing first would open a busy=False window —
                    # with work resident only in the mailbox — in which a
                    # clear/delete/scan reservation could start and drop the
                    # accepted document); otherwise release the slot.
                    updates = {
                        # custom_chunks work is complete here; clear its
                        # operation_record so a later dead-owner reclaim does not
                        # read a stale target.
                        "operation_record": None,
                        "cancellation_requested": False,
                        "cancellation_reason": None,
                        "cancellation_detail": None,
                    }
                    if handoff_ingress is None:
                        ingress_has_work = True  # fail toward handoff
                    else:
                        try:
                            ingress_has_work = handoff_ingress.has_work()
                        except Exception as probe_error:
                            ingress_has_work = True  # fail toward handoff
                            logger.warning(
                                "custom-chunks exit: ingress has_work probe "
                                "failed; handing off so a committed manual "
                                "retry or enqueued document is not silently "
                                f"deferred: {probe_error}"
                            )
                    if ingress_has_work:
                        status.update(updates)
                        return "handoff"  # keep busy (owner stays ours)
                    updates.update({"busy": False, "busy_owner": None})
                    status.update(updates)
                    return "released"

                def _terminal_release(status):
                    # Owner-checked backstop: release the slot we still hold.
                    status.update(
                        {
                            "busy": False,
                            "busy_owner": None,
                            "operation_record": None,
                            "cancellation_requested": False,
                            "cancellation_reason": None,
                            "cancellation_detail": None,
                        }
                    )

                try:
                    # Flush first (while still busy so no other writer starts
                    # mid-flush). BOTH the release/handoff decision AND the
                    # handoff itself live in the inner finally so a flush error
                    # cannot skip them: _exit_action keeps busy=True for a
                    # handoff, so if the queued docs were not drained here the
                    # outer terminal release would clear busy and strand them
                    # in the mailbox with no active processor.
                    try:
                        if op_failed:
                            # Failure path: do NOT blind-flush partial work
                            # (issue #3400 root cause 6). Persist what is
                            # valuable (the LLM cache — handled inside the
                            # discard helper) and drop the partial buffers;
                            # the journal anchors whatever immediate-write
                            # backends already committed, and the same SDK
                            # call resumes from the staged data.
                            await self._discard_pending_index_ops(
                                skip_enqueue_owned=False
                            )
                        else:
                            await self._insert_done_with_cleanup()
                    finally:
                        decision = await with_reservation_lock(
                            pipeline_status,
                            pipeline_status_lock,
                            owner_key="busy_owner",
                            token=token,
                            action=_exit_action,
                        )
                        if decision == "handoff":
                            # busy is still True (ours); the handoff run takes
                            # over the slot and releases it at its own exit. Pass
                            # our token so it owns and releases the SAME
                            # reservation. Runs even when the flush raised — its
                            # error still propagates after this — so the queued
                            # docs are not stranded. A pending cancellation makes
                            # this await re-raise immediately, deferring the
                            # drain to the next trigger (#3400).
                            try:
                                await self.apipeline_process_enqueue_documents(
                                    _holding_busy=True, token=token
                                )
                            except Exception as drain_error:
                                logger.warning(
                                    "Failed to process queued documents handed "
                                    "off from a custom-chunks insert: "
                                    f"{drain_error}"
                                )
                finally:
                    # Guarantee release even if the flush / decision / handoff was
                    # cancelled or errored. Owner-checked + cancellation-resistant;
                    # a no-op if the handoff run already released the slot (or a
                    # later holder took over).
                    await with_reservation_lock(
                        pipeline_status,
                        pipeline_status_lock,
                        owner_key="busy_owner",
                        token=token,
                        action=_terminal_release,
                    )

    async def _upsert_custom_chunk_status(
        self,
        doc_key: str,
        status: DocStatus,
        *,
        base_row: dict[str, Any] | None,
        journal: dict[str, Any] | None,
        full_text: str | None = None,
        file_path: str | None = None,
        chunks_list: list[str] | None = None,
        error_msg: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
        metadata_drop: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Upsert the doc_status row for a custom-chunk operation.

        doc_status backends replace the whole record (and its ``metadata``
        blob) on upsert, so the full field set is rebuilt from ``base_row``
        (the previously fetched/returned record) — base-document metadata is
        carried verbatim with only the ``custom_chunk_patch`` key added
        (``journal``) or removed (``journal=None``, at commit). Returns the
        upserted payload so consecutive transitions can pass it back in as
        ``base_row`` without refetching.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        if base_row:
            metadata = dict(base_row.get("metadata") or {})
            payload: dict[str, Any] = {
                "status": status,
                "content_summary": base_row.get("content_summary") or "",
                "content_length": base_row.get("content_length") or 0,
                "created_at": base_row.get("created_at") or now_iso,
                "updated_at": now_iso,
                "file_path": base_row.get("file_path")
                or file_path
                or normalize_document_file_path(""),
                "track_id": base_row.get("track_id"),
                "content_hash": base_row.get("content_hash"),
                "metadata": metadata,
            }
            existing_chunks = [
                cid
                for cid in (base_row.get("chunks_list") or [])
                if isinstance(cid, str) and cid
            ]
        else:
            payload = {
                "status": status,
                "content_summary": get_content_summary(full_text or ""),
                "content_length": len(full_text or ""),
                "created_at": now_iso,
                "updated_at": now_iso,
                "file_path": file_path or normalize_document_file_path(""),
                "track_id": generate_track_id("insert"),
                "content_hash": compute_text_content_hash(full_text or ""),
                "metadata": {},
            }
            existing_chunks = []

        effective_chunks = chunks_list if chunks_list is not None else existing_chunks
        payload["chunks_list"] = list(effective_chunks)
        payload["chunks_count"] = len(effective_chunks)

        if journal is None:
            payload["metadata"].pop(CUSTOM_CHUNK_PATCH_METADATA_KEY, None)
        else:
            payload["metadata"][CUSTOM_CHUNK_PATCH_METADATA_KEY] = journal

        # ``base_row``'s metadata is copied verbatim above, so keys the caller
        # does not name are left standing — the opposite default from the
        # pipeline's whitelist-rebuilt transition metadata, and the right one
        # here because a patch ADDS to a base document whose earlier records
        # still describe live graph objects. A caller that must retire a key
        # (rollback restoring "no truncation ever happened") therefore needs
        # ``metadata_drop``: as with doc_status_transition_metadata, passing
        # None/empty via ``metadata_extra`` would persist that value rather
        # than remove the key. ``metadata_drop`` wins over ``metadata_extra``.
        if metadata_extra:
            payload["metadata"].update(metadata_extra)
        for key in metadata_drop:
            payload["metadata"].pop(key, None)

        if error_msg is not None:
            payload["error_msg"] = error_msg
        elif status == DocStatus.PROCESSED:
            payload["error_msg"] = ""

        await self.doc_status.upsert({doc_key: payload})
        return payload

    async def _union_doc_recovery_anchors(
        self,
        doc_id: str,
        entity_names,
        relation_pairs,
    ) -> None:
        """Union a patch's candidates into the document's recovery anchors.

        A custom-chunk patch must never overwrite the base document's
        ``full_entities`` / ``full_relations`` rows — the base document keeps
        owning its previously committed contributions — so the patch
        candidates are unioned in and both rows flushed via the narrow
        barrier (issue #3400 Phase 3, commit step).
        """
        entity_row = await self.full_entities.get_by_id(doc_id) or {}
        union_names = set(entity_row.get("entity_names") or []) | set(entity_names)
        await self.full_entities.upsert(
            {
                doc_id: {
                    "entity_names": sorted(union_names),
                    "count": len(union_names),
                }
            }
        )

        relation_row = await self.full_relations.get_by_id(doc_id) or {}
        union_pairs = {
            tuple(pair) for pair in (relation_row.get("relation_pairs") or [])
        } | {tuple(pair) for pair in relation_pairs}
        await self.full_relations.upsert(
            {
                doc_id: {
                    "relation_pairs": [list(pair) for pair in sorted(union_pairs)],
                    "count": len(union_pairs),
                }
            }
        )

        await self._flush_storages([self.full_entities, self.full_relations])

    async def arollback_failed_custom_chunk_patches(
        self,
        *,
        pipeline_status: dict | None = None,
        pipeline_status_lock: Any | None = None,
    ) -> dict[str, Any]:
        """Roll back every failed/stale custom-chunk operation (issue #3400 P4).

        The administrative escape hatch invoked at the start of
        ``/documents/scan``: while the SDK caller owns roll-forward (repeating
        the same call resumes it), scan rolls incomplete operations BACK to
        the previously committed document state — repeated roll-forward may
        keep failing, and the journal row would otherwise stay dirty forever.

        Discovers candidates from ``doc_status`` directly (FAILED or stale
        PROCESSING rows carrying a ``custom_chunk_patch`` journal — SDK
        operations may have no scan-visible input file). A PROCESSING row is
        necessarily stale here: a live operation holds the pipeline ``busy``
        slot, and the scan reservation refuses to start while ``busy``.

        Per document, under the same per-document keyed lock the operations
        use: staged contributions are removed/rebuilt via the candidate-driven
        purge primitive (missing recovery cache triggers structural provenance
        repair plus a persistent warning; operational write failures retain the
        journal), operation-scoped LLM cache is deleted, anchor unions are
        pruned where the document no longer owns a contribution, and only then
        is the journal cleared:

        - ``patch`` mode: the base document is restored to ``PROCESSED``.
        - ``create`` mode: the document did not exist before the operation,
          so its remaining rows (``full_docs``, ``doc_status``) are removed.

        A rollback failure keeps the journal and the FAILED status — the next
        scan retries it. Absent objects are idempotent no-ops, but success is
        never reported merely because something was already missing.

        Returns ``{"rolled_back_count": int, "failed_count": int,
        "rolled_back_sample": [...], "failed_sample": [...]}``. Counts are
        exact; the id lists are bounded to ``ROLLBACK_REPORT_SAMPLE_CAP``,
        because a report that grows one entry per journaled document would
        reintroduce the O(N) accumulation this sweep exists to remove.
        """
        if pipeline_status is None or pipeline_status_lock is None:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=self.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=self.workspace
            )

        rolled_back_count = 0
        failed_count = 0
        rolled_back_sample: list[str] = []
        failed_sample: list[str] = []
        doc_lock_namespace = (
            f"{self.workspace}:DocPatch" if self.workspace else "DocPatch"
        )

        async def _roll_back_batch(doc_ids: list[str]) -> None:
            """Roll back one page's journaled docs, accumulating only counts."""
            nonlocal rolled_back_count, failed_count
            for doc_id in doc_ids:
                async with get_storage_keyed_lock(
                    [doc_id], namespace=doc_lock_namespace, enable_logging=False
                ):
                    # Re-read under the lock: the SDK may have resumed (and
                    # committed) the operation between discovery and here.
                    row = await self.doc_status.get_by_id(doc_id)
                    journal = doc_status_custom_chunk_patch(row)
                    if journal is None:
                        continue
                    try:
                        await self._rollback_one_custom_chunk_patch(
                            doc_id,
                            row,
                            journal,
                            pipeline_status=pipeline_status,
                            pipeline_status_lock=pipeline_status_lock,
                        )
                        rolled_back_count += 1
                        if len(rolled_back_sample) < ROLLBACK_REPORT_SAMPLE_CAP:
                            rolled_back_sample.append(doc_id)
                        log_message = (
                            f"[rollback] Rolled back custom-chunk operation "
                            f"{journal.get('operation_id')} on {doc_id} "
                            f"(mode: {journal.get('mode', 'patch')})"
                        )
                        logger.info(log_message)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = log_message
                            append_pipeline_history(pipeline_status, log_message)
                    except Exception as rollback_error:
                        # Journal and FAILED status remain; the next scan retries.
                        failed_count += 1
                        if len(failed_sample) < ROLLBACK_REPORT_SAMPLE_CAP:
                            failed_sample.append(doc_id)
                        log_message = (
                            f"[rollback] Failed to roll back custom-chunk "
                            f"operation on {doc_id}: {rollback_error}"
                        )
                        logger.error(log_message)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = log_message
                            append_pipeline_history(pipeline_status, log_message)

        # Scheduling-control-plane discovery, memory-bounded (LR2 Phase 2):
        # page the FAILED/PROCESSING keyset instead of materializing every such
        # row, taking journaled ids straight from the lightweight page's
        # has_custom_chunk_journal flag (no per-row hydration).
        #
        # Each page is rolled back BEFORE the next is fetched, so nothing
        # proportional to the journaled-document count is ever held — collecting
        # every id first would have kept the sweep O(N) on the assumption that
        # journaled docs are rare, which is a workload guess, not a bound.
        # Interleaving is safe against the live-view keyset: the sort key
        # ``(created_at, id)`` is immutable (``update_doc_status_fields``
        # refuses ``created_at``), and a rollback only ever leaves the
        # FAILED/PROCESSING set — it flips a row to PROCESSED or deletes it, and
        # never creates one — so it can only touch rows the sweep has already
        # consumed and passed.
        #
        # strict so a mid-pagination or per-record backend failure raises rather
        # than silently ending the sweep early and reporting the scan done. Docs
        # rolled back before that raise stay rolled back: each one is an
        # independent, idempotent unit committed under its own lock, and its
        # journal is gone, so the next scan simply skips it and picks up the
        # rest.
        page_size = getattr(self, "pipeline_scheduling_page_size", 0)
        if page_size > 0:
            position: CursorPosition = CURSOR_START
            while True:
                page = await self.doc_status.get_docs_by_statuses_page(
                    [DocStatus.FAILED, DocStatus.PROCESSING],
                    limit=page_size,
                    position=position,
                    strict=True,
                )
                await _roll_back_batch(
                    [
                        doc_id
                        for doc_id, record in page.docs.items()
                        if record.has_custom_chunk_journal
                    ]
                )
                if page.next_position is CURSOR_END:
                    break
                position = page.next_position
        else:
            # Paging disabled (PIPELINE_SCHEDULING_PAGE_SIZE=0): legacy scan.
            candidates = await self.doc_status.get_docs_by_statuses(
                [DocStatus.FAILED, DocStatus.PROCESSING], strict=True
            )
            await _roll_back_batch(
                [
                    doc_id
                    for doc_id, status_doc in candidates.items()
                    if doc_status_custom_chunk_patch(status_doc) is not None
                ]
            )

        return {
            "rolled_back_count": rolled_back_count,
            "failed_count": failed_count,
            "rolled_back_sample": rolled_back_sample,
            "failed_sample": failed_sample,
        }

    async def _rollback_one_custom_chunk_patch(
        self,
        doc_id: str,
        row: dict[str, Any],
        journal: dict[str, Any],
        *,
        pipeline_status: dict,
        pipeline_status_lock: Any,
    ) -> None:
        """Roll a single journaled custom-chunk operation back (see caller)."""
        mode = journal.get("mode") or "patch"
        staged_chunk_ids = [
            cid
            for cid in (journal.get("chunk_ids") or [])
            if isinstance(cid, str) and cid
        ]
        candidate_entities = [
            name
            for name in (journal.get("entity_names") or [])
            if isinstance(name, str) and name
        ]
        candidate_relations = [
            (pair[0], pair[1])
            for pair in (journal.get("relation_pairs") or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]

        # Operation-scoped LLM cache ids must be collected BEFORE the purge
        # deletes the staged chunk rows that reference them.
        cache_ids: list[str] = []
        if staged_chunk_ids:
            staged_rows = await self.text_chunks.get_by_ids(staged_chunk_ids)
            for staged_row in staged_rows:
                if isinstance(staged_row, dict):
                    cache_ids.extend(
                        cid
                        for cid in (staged_row.get("llm_cache_list") or [])
                        if isinstance(cid, str) and cid
                    )
            cache_ids = list(dict.fromkeys(cache_ids))

        rebuild_report = KGRebuildReport()
        if staged_chunk_ids:
            rebuild_report = await self._purge_kg_contributions(
                doc_id,
                staged_chunk_ids,
                candidate_entities=candidate_entities,
                candidate_relations=candidate_relations,
                patch_only=(mode == "patch"),
                rebuild_policy="rollback",
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )

        if cache_ids and self.llm_response_cache:
            await self.llm_response_cache.delete(cache_ids)
            await self._flush_storages([self.llm_response_cache])

        warning_rows = await self._persist_custom_chunk_recovery_warning(
            doc_id,
            journal,
            rebuild_report,
            mode=mode,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
        )
        row = warning_rows.get(doc_id, row)

        if mode == "create":
            # The document did not exist before this operation: remove its
            # body and status row entirely. The purge above (whole-document
            # mode) already dropped the anchors. The doc_status delete MUST
            # be flushed here: unlike upsert (immediate-persist), delete is
            # deferred-commit on the JSON backend, and reporting a rollback
            # as successful while the FAILED journal row can reappear after
            # a crash would be a false success.
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])
            await self._flush_storages([self.full_docs, self.doc_status])
            return

        # Patch mode: prune anchor unions this document no longer owns (a
        # previous attempt may have failed after the commit union), then
        # restore the base document to PROCESSED with the journal cleared and
        # any staged ids stripped from chunks_list.
        staged_set = set(staged_chunk_ids)
        base_chunk_ids = {
            cid
            for cid in ((row or {}).get("chunks_list") or [])
            if isinstance(cid, str) and cid and cid not in staged_set
        }
        await self._prune_doc_recovery_anchors(
            doc_id,
            candidate_entities,
            candidate_relations,
            owned_chunk_ids=base_chunk_ids,
        )
        cleaned_chunks = [
            cid
            for cid in ((row or {}).get("chunks_list") or [])
            if isinstance(cid, str) and cid and cid not in staged_set
        ]
        # The FAILED row this restore copies from carries the failed attempt's
        # llm_truncation stamp, describing extractions the purge above just
        # removed — restoring it verbatim would leave a clean base document
        # permanently advertising truncation from rolled-back content. The
        # journal's pre-operation snapshot is the authority: reinstate it, or
        # drop the key when the base never truncated. A journal written before
        # the snapshot existed has no key and changes nothing (its attempts
        # never stamped a tally either).
        truncation_restore: dict[str, Any] = {}
        if "prior_llm_truncation" in journal:
            prior_truncation = journal["prior_llm_truncation"]
            if prior_truncation is None:
                truncation_restore["metadata_drop"] = (LLM_TRUNCATION_METADATA_KEY,)
            else:
                truncation_restore["metadata_extra"] = {
                    LLM_TRUNCATION_METADATA_KEY: prior_truncation
                }
        await self._upsert_custom_chunk_status(
            doc_id,
            DocStatus.PROCESSED,
            base_row=row,
            journal=None,
            chunks_list=cleaned_chunks,
            error_msg="",
            **truncation_restore,
        )

    async def _persist_custom_chunk_recovery_warning(
        self,
        doc_id: str,
        journal: dict[str, Any],
        report: KGRebuildReport,
        *,
        mode: str,
        pipeline_status: dict,
        pipeline_status_lock: Any,
    ) -> dict[str, dict[str, Any]]:
        """Persist bounded degraded-recovery warnings on surviving owners.

        Patch rollback always includes the target document. Create rollback
        removes that target entirely, so its warning is attached to documents
        owning the surviving chunks that contributed to degraded aggregates.
        """
        if not report.has_warnings:
            return {}

        owner_doc_ids: set[str] = set()
        surviving_chunk_ids = sorted(report.surviving_chunk_ids)
        if surviving_chunk_ids:
            chunk_rows = await self.text_chunks.get_by_ids(surviving_chunk_ids)
            for chunk_row in chunk_rows:
                if not isinstance(chunk_row, dict):
                    continue
                owner_doc_id = chunk_row.get("full_doc_id")
                if isinstance(owner_doc_id, str) and owner_doc_id:
                    owner_doc_ids.add(owner_doc_id)

        if mode == "patch":
            owner_doc_ids.add(doc_id)
        else:
            owner_doc_ids.discard(doc_id)

        if not owner_doc_ids:
            raise RuntimeError(
                "Cannot persist degraded KG recovery warning: no surviving "
                "owner document was found"
            )

        def _sample(values) -> dict[str, Any]:
            ordered = sorted(set(values))
            return {"count": len(ordered), "sample": ordered[:20]}

        relationship_labels = [
            f"{src}~{tgt}" for src, tgt in report.degraded_relationships
        ]
        event = {
            "code": "degraded_custom_chunk_rollback",
            "message": (
                "Custom-chunk rollback completed with structural provenance "
                "repair, but some KG semantic aggregates could not be fully "
                "rebuilt from extraction cache."
            ),
            "operation_id": journal.get("operation_id"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "missing_cache_chunks": _sample(report.missing_cache_chunk_ids),
            "unusable_cache_chunks": _sample(report.unusable_cache_chunk_ids),
            "degraded_entities": _sample(report.degraded_entities),
            "degraded_relationships": _sample(relationship_labels),
            "surviving_chunks": _sample(surviving_chunk_ids),
        }

        updated_rows: dict[str, dict[str, Any]] = {}
        missing_owner_doc_ids: list[str] = []
        for owner_doc_id in sorted(owner_doc_ids):
            owner_row = await self.doc_status.get_by_id(owner_doc_id)
            if not isinstance(owner_row, dict):
                logger.warning(
                    "Cannot persist degraded KG recovery warning: document "
                    "status `%s` was not found",
                    owner_doc_id,
                )
                missing_owner_doc_ids.append(owner_doc_id)
                continue

            metadata = dict(owner_row.get("metadata") or {})
            raw_warnings = metadata.get(KG_RECOVERY_WARNINGS_METADATA_KEY)
            warnings_list = (
                [item for item in raw_warnings if isinstance(item, dict)]
                if isinstance(raw_warnings, list)
                else []
            )
            operation_id = event.get("operation_id")
            warnings_list = [
                item
                for item in warnings_list
                if not (
                    item.get("code") == event["code"]
                    and item.get("operation_id") == operation_id
                )
            ]
            warnings_list.append(event)
            metadata[KG_RECOVERY_WARNINGS_METADATA_KEY] = warnings_list[-10:]
            updated_row = {
                **owner_row,
                "metadata": metadata,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            updated_rows[owner_doc_id] = updated_row

        if missing_owner_doc_ids:
            raise RuntimeError(
                "Cannot persist degraded KG recovery warning: document status "
                f"missing for {', '.join(missing_owner_doc_ids)}"
            )

        if updated_rows:
            await self.doc_status.upsert(updated_rows)
            await self._flush_storages([self.doc_status])

        log_message = (
            "[rollback] Degraded KG recovery warning for custom-chunk operation "
            f"{journal.get('operation_id')} persisted on "
            f"{len(updated_rows)} surviving document(s)"
        )
        logger.warning(log_message)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            append_pipeline_history(pipeline_status, log_message)

        return updated_rows

    async def _prune_doc_recovery_anchors(
        self,
        doc_id: str,
        entity_names,
        relation_pairs,
        *,
        owned_chunk_ids: set[str],
    ) -> None:
        """Prune rolled-back candidates from the document's recovery anchors.

        A journal candidate stays in ``full_entities`` / ``full_relations``
        only if the document still owns a contribution to it through one of
        ``owned_chunk_ids`` (its committed base chunks, per the tracking
        stores) — e.g. a patch that added a source to a base-document entity.
        Candidates the document no longer contributes to are removed so the
        anchors stay a faithful recovery superset. No-op for anchors/keys
        that do not exist.
        """
        entity_candidates = set(entity_names or [])
        if entity_candidates:
            entity_row = await self.full_entities.get_by_id(doc_id)
            if entity_row and entity_row.get("entity_names"):
                keep: list[str] = []
                for name in entity_row["entity_names"]:
                    if name not in entity_candidates:
                        keep.append(name)
                        continue
                    tracking = (
                        await self.entity_chunks.get_by_id(name)
                        if self.entity_chunks
                        else None
                    )
                    sources = set((tracking or {}).get("chunk_ids") or [])
                    if sources & owned_chunk_ids:
                        keep.append(name)
                await self.full_entities.upsert(
                    {doc_id: {"entity_names": sorted(keep), "count": len(keep)}}
                )

        relation_candidates = {tuple(pair) for pair in (relation_pairs or [])}
        if relation_candidates:
            relation_row = await self.full_relations.get_by_id(doc_id)
            if relation_row and relation_row.get("relation_pairs"):
                keep_pairs: list[list[str]] = []
                for pair in relation_row["relation_pairs"]:
                    pair_tuple = tuple(pair)
                    if pair_tuple not in relation_candidates:
                        keep_pairs.append(list(pair))
                        continue
                    tracking = (
                        await self.relation_chunks.get_by_id(
                            make_relation_chunk_key(*pair_tuple)
                        )
                        if self.relation_chunks
                        else None
                    )
                    sources = set((tracking or {}).get("chunk_ids") or [])
                    if sources & owned_chunk_ids:
                        keep_pairs.append(list(pair))
                await self.full_relations.upsert(
                    {
                        doc_id: {
                            "relation_pairs": sorted(keep_pairs),
                            "count": len(keep_pairs),
                        }
                    }
                )

        await self._flush_storages([self.full_entities, self.full_relations])

    async def _process_extract_entities(
        self,
        chunk: dict[str, Any],
        pipeline_status=None,
        pipeline_status_lock=None,
        truncation_tally: TokenLimitTruncationTally | None = None,
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                global_config=self._build_global_config(),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
                truncation_tally=truncation_tally,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            # Guard the status write: this method may be called with no
            # pipeline_status/lock (both default to None). Without the guard,
            # ``async with None`` raises TypeError here and masks the real
            # extraction error ``e`` that we want to surface.
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    append_pipeline_history(pipeline_status, error_msg)
            raise e

    def _index_storages(self) -> list:
        """All storages flushed together by index_done_callback / abort."""
        return [
            storage_inst
            for storage_inst in [
                self.full_docs,
                self.doc_status,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]

    async def _discard_pending_index_ops(
        self, *, skip_enqueue_owned: bool = True
    ) -> None:
        """Drop not-yet-flushed buffers on an aborting batch.

        Called when a batch aborts on an internal storage error. Each
        still-buffered KG/vector record belongs to a document that will be
        marked FAILED and fully reprocessed, so dropping the shared cross-file
        buffers is safe and stops the poisoned/stale records from being
        re-flushed by remaining in-flight documents or carried into the next
        batch.

        ``skip_enqueue_owned`` controls whether ``full_docs`` / ``doc_status``
        are cleared:

        * ``True`` (the file pipeline) — skip them. They are written by the
          concurrent ``apipeline_enqueue_documents`` path (under
          ``enqueue_serialize_lock``, which this cleanup does not hold), as
          ``full_docs.upsert -> index_done_callback -> doc_status.upsert``.
          Clearing ``full_docs``'s buffer in the window between an in-flight
          upload's upsert and its flush would drop the document body while the
          PENDING ``doc_status`` row still gets written, leaving an accepted
          document with no content. Those writes self-flush immediately, so
          skipping them discards nothing processing-owned.
        * ``False`` (direct, non-pipeline callers like ``ainsert_custom_chunks``
          via ``_insert_done_with_cleanup``) — clear them too. There is no
          concurrent-enqueue contract for these callers, and a permanent
          ``full_docs`` bulk failure (e.g. OpenSearch KV) must be cleared or it
          stays buffered and every later ``_insert_done()`` replays the same
          poisoned record. ``doc_status`` is immediate-write (no buffered
          backend overrides ``drop_pending_index_ops``), so dropping it is a
          no-op; only ``full_docs`` is meaningfully cleared. (Edge: a direct
          insert racing a concurrent enqueue mid-window could still drop that
          enqueue's in-flight body, but per-item backends only retain the
          failed item and the enqueue race is a pipeline-only concern.)

        The LLM response cache gets a final flush *before* its buffer is
        dropped, because — unlike regenerable KG data — re-running LLM calls
        is expensive, so cached results must be persisted maximally:

        * When the abort was NOT caused by the cache, the cache backend is
          healthy and this flush commits every still-buffered entry, leaving
          the buffer empty so the subsequent drop discards nothing persistable.
        * When a poisoned cache item is itself the abort cause (OpenSearch now
          raises on non-retryable bulk failures), the flush persists the
          writable entries (per-item backends pop successes) while the
          un-writable item stays buffered and the drop then clears it — so a
          bad cache entry cannot re-flush and re-abort every subsequent batch
          and wedge the pipeline.

        Backends that materialize writes in memory and only persist on a
        later save (FAISS / Nano) discard just the pending buffer here and do
        NOT roll back already-materialized-but-unsaved writes: the FAILED
        documents are reprocessed idempotently, so the rollback would be
        non-load-bearing and inconsistent with the server-backed backends
        (see those backends' ``drop_pending_index_ops`` docstrings).

        Best-effort throughout: a flush/clear failure is logged, not raised,
        so cleanup never masks the original abort cause.
        """
        for storage_inst in self._index_storages():
            if skip_enqueue_owned and (
                storage_inst is self.full_docs or storage_inst is self.doc_status
            ):
                # enqueue-owned (see docstring): skipped for the file pipeline
                # to avoid racing a concurrent enqueue; direct callers pass
                # skip_enqueue_owned=False so a poisoned full_docs op is cleared.
                continue
            if storage_inst is self.llm_response_cache:
                # Persist what can still be written, then fall through to drop
                # whatever could not (a poisoned item) so it cannot wedge the
                # next batch.
                try:
                    await cast(StorageNameSpace, storage_inst).index_done_callback()
                except Exception as e:
                    logger.error(f"Failed to persist LLM cache on abort: {e}")
            try:
                await cast(StorageNameSpace, storage_inst).drop_pending_index_ops()
            except Exception as e:
                logger.error(
                    f"Failed to discard pending ops on "
                    f"{type(storage_inst).__name__}: {e}"
                )

    async def _flush_storages(self, storages: list) -> None:
        """Flush the given storage instances — a narrow, named flush barrier.

        Failure paths and write-ahead barriers (issue #3400) must be able to
        flush a specific subset of storages (e.g. only the two recovery
        indexes) instead of the unconditional all-storage ``_insert_done``,
        which on a partially-failed operation would blindly flush partial
        work into every store. ``_insert_done`` delegates here with the full
        ``_index_storages()`` list.

        Raises ``IndexFlushError`` (carrying driver name + namespace) for the
        first failed flush after every flush has run to completion;
        ``CancelledError`` propagates as-is.
        """

        async def _flush_one(storage_inst) -> None:
            # Wrap each flush so a failure carries the driver name + namespace.
            # The pipeline uses this to abort the batch with an actionable
            # reason instead of misattributing a shared-buffer flush error to
            # whichever document happened to trigger index_done_callback.
            try:
                await cast(StorageNameSpace, storage_inst).index_done_callback()
            except Exception as e:
                namespace = getattr(storage_inst, "final_namespace", None) or getattr(
                    storage_inst, "namespace", ""
                )
                raise IndexFlushError(type(storage_inst).__name__, namespace, e) from e

        # Await every flush to completion (return_exceptions=True) before
        # raising. With the default gather, the first IndexFlushError is
        # propagated while sibling flush coroutines keep running detached —
        # they could commit records or race _discard_pending_index_ops after
        # the abort decision, and a second failing sibling would surface as a
        # "Task exception was never retrieved" warning. Collecting all results
        # first makes teardown deterministic and lets us report every failure.
        results = await asyncio.gather(
            *[_flush_one(inst) for inst in storages if inst is not None],
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # A cooperative cancellation must propagate as-is, not be reported
            # as a storage flush failure (_flush_one's `except Exception` does
            # not catch CancelledError, so it lands here as a result).
            for exc in errors:
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
            for extra in errors[1:]:
                logger.error(f"Additional index flush failure: {extra}")
            raise errors[0]

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        await self._flush_storages(self._index_storages())

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                append_pipeline_history(pipeline_status, log_message)

    async def _insert_done_with_cleanup(self) -> None:
        """``_insert_done`` for UPSERT-oriented direct (non-pipeline) callers,
        discarding the pending buffers on a flush failure.

        The file pipeline aborts and calls ``_discard_pending_index_ops()``
        centrally, but direct insert callers (custom KG / chunks insert) have
        no such cleanup. Without it, a permanent flush failure leaves the
        poisoned op buffered — OpenSearch keeps a non-retryable bulk item;
        milvus/qdrant/postgres/mongo keep the whole buffer — and every later
        ``_insert_done()`` replays it, even after the caller submits otherwise
        valid work. Discard pending on ``IndexFlushError`` so the buffer is
        clean for the next attempt, then re-raise so the failure still
        surfaces to the caller.

        WARNING: do NOT use this on deletion paths. ``_discard_pending_index_ops``
        drops pending DELETES too, but deletes are not regenerable by
        reprocessing (the document is being removed, nothing re-issues them).
        Dropping them — while a deletion may still report success — would leave
        stale vectors/KV searchable. Deletion paths must use plain
        ``_insert_done`` so failed deletes stay buffered for a later retry.
        """
        try:
            await self._insert_done()
        except IndexFlushError:
            # Direct callers have no concurrent-enqueue contract, so clear
            # full_docs too (skip_enqueue_owned=False) — otherwise a permanent
            # full_docs bulk failure stays buffered and replays on every later
            # _insert_done().
            await self._discard_pending_index_ops(skip_enqueue_owned=False)
            raise

    def insert_custom_kg(
        self, custom_kg: dict[str, Any], full_doc_id: str = None
    ) -> None:
        _run_sync(
            lambda: self.ainsert_custom_kg(custom_kg, full_doc_id),
            sync_name="insert_custom_kg",
            async_name="ainsert_custom_kg",
            owning_loop=self._owning_loop,
        )

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str = None,
    ) -> None:
        """Insert caller-constructed KG objects directly into the stores.

        Entity names and relationship endpoints are normalized with the same
        contract used by document extraction before any storage write begins.

        .. warning:: (issue #3400 Phase 5 — direct-writer audit)
           This path is OUTSIDE the document-level recovery guarantee. It has
           no durable operation journal and does not prewrite
           ``full_entities`` / ``full_relations`` recovery anchors, so a
           crash or partial failure mid-call can leave graph/vector/tracking
           contributions that per-document purge, retry, and scan rollback
           cannot discover. Use the journaled ingestion paths (``ainsert`` /
           ``ainsert_custom_chunks``) when crash recovery matters; the
           offline ``lightrag.tools.kg_integrity_repair`` audit can
           reconstruct anchors for data written here after the fact.
        """
        # Direct KG write path — refuse on a fenced workspace (recovery_required)
        # like the other SDK mutations, before touching any storage.
        await self._raise_if_recovery_required()

        update_storage = False
        try:

            def _normalize_custom_kg_entity_name(value: Any, *, field: str) -> str:
                if not isinstance(value, str):
                    raise ValueError(f"Custom KG {field} must be a string")
                normalized_value = normalize_entity_name(value)
                if not normalized_value:
                    raise ValueError(
                        f"Custom KG {field} cannot be empty after normalization"
                    )
                return normalized_value

            # Validate and canonicalize the complete identifier set before
            # chunks or graph objects are written. Copies keep the caller's
            # custom_kg payload unchanged.
            normalized_entities: list[dict[str, Any]] = []
            for index, entity_data in enumerate(custom_kg.get("entities", [])):
                normalized_entity_data = dict(entity_data)
                normalized_entity_data["entity_name"] = (
                    _normalize_custom_kg_entity_name(
                        entity_data["entity_name"],
                        field=f"entities[{index}].entity_name",
                    )
                )
                normalized_entities.append(normalized_entity_data)

            normalized_relationships: list[dict[str, Any]] = []
            for index, relationship_data in enumerate(
                custom_kg.get("relationships", [])
            ):
                normalized_relationship_data = dict(relationship_data)
                normalized_relationship_data["src_id"] = (
                    _normalize_custom_kg_entity_name(
                        relationship_data["src_id"],
                        field=f"relationships[{index}].src_id",
                    )
                )
                normalized_relationship_data["tgt_id"] = (
                    _normalize_custom_kg_entity_name(
                        relationship_data["tgt_id"],
                        field=f"relationships[{index}].tgt_id",
                    )
                )
                # Compared after normalization, because the normalized values
                # are the ones written as the graph edge below: two spellings
                # that canonicalize to the same name are the same self-loop.
                # Extraction drops self-loops (operate.py) and amerge_entities
                # refuses to form one, so accepting them here would make this
                # the only path that puts src == tgt into the graph.
                if (
                    normalized_relationship_data["src_id"]
                    == normalized_relationship_data["tgt_id"]
                ):
                    raise ValueError(
                        f"Custom KG relationships[{index}] is a self-loop on "
                        f"'{normalized_relationship_data['src_id']}': src_id and "
                        "tgt_id must be different entities"
                    )
                normalized_relationships.append(normalized_relationship_data)

            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}

            def _build_custom_kg_chunks() -> None:
                """Encode and assemble every custom-KG chunk in one hop.

                ``self.tokenizer`` is shared with the chunking executor, and
                this entry point is gated by neither max_parallel_insert nor the
                pipeline busy flag, so leaving the loop here would let it encode
                concurrently with a document being chunked.
                """
                nonlocal update_storage
                for chunk_data in custom_kg.get("chunks", []):
                    chunk_content = sanitize_text_for_encoding(chunk_data["content"])
                    source_id = chunk_data["source_id"]
                    file_path = normalize_document_file_path(
                        chunk_data.get("file_path", "custom_kg")
                    )
                    tokens = len(self.tokenizer.encode(chunk_content))
                    chunk_order_index = (
                        0
                        if "chunk_order_index" not in chunk_data.keys()
                        else chunk_data["chunk_order_index"]
                    )
                    chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

                    chunk_entry = {
                        "content": chunk_content,
                        "source_id": source_id,
                        "tokens": tokens,
                        "chunk_order_index": chunk_order_index,
                        "full_doc_id": full_doc_id
                        if full_doc_id is not None
                        else source_id,
                        "file_path": file_path,
                        "status": DocStatus.PROCESSED,
                    }
                    all_chunks_data[chunk_id] = chunk_entry
                    chunk_to_source_map[source_id] = chunk_id
                    update_storage = True

            await run_in_chunking_executor(_build_custom_kg_chunks)

            if all_chunks_data:
                await asyncio.gather(
                    self.chunks_vdb.upsert(all_chunks_data),
                    self.text_chunks.upsert(all_chunks_data),
                )

            # Keep the last declaration for each entity_name so batch backends
            # preserve the old serial upsert semantics deterministically.
            deduped_entities: dict[str, dict[str, Any]] = {}
            for entity_data in normalized_entities:
                entity_name = entity_data["entity_name"]
                deduped_entities.pop(entity_name, None)
                deduped_entities[entity_name] = entity_data

            # Insert entities into knowledge graph (batch for performance)
            all_entities_data: list[dict[str, str]] = []
            entity_nodes: list[tuple[str, dict[str, str]]] = []
            for entity_data in deduped_entities.values():
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = normalize_document_file_path(
                    entity_data.get("file_path", "custom_kg")
                )

                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                node_data: dict[str, str] = {
                    "entity_id": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                entity_nodes.append((entity_name, node_data))
                node_data_copy = dict(node_data)
                node_data_copy["entity_name"] = entity_name
                all_entities_data.append(node_data_copy)
                update_storage = True

            # Relationship storage is undirected, so keep only the last update
            # for each endpoint pair regardless of order.
            deduped_relationships: dict[tuple[str, str], dict[str, Any]] = {}
            for relationship_data in normalized_relationships:
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                relation_key = tuple(sorted((src_id, tgt_id)))
                deduped_relationships.pop(relation_key, None)
                deduped_relationships[relation_key] = relationship_data

            # Coarse-grained keyed lock covering every entity name and every
            # relationship endpoint this batch will write. Keys collide with
            # the per-entity and sorted([src, tgt]) edge locks held by the
            # doc-ingest pipeline (operate.py:_locked_process_entity_name and
            # _locked_process_edges) in the same namespace, so a concurrent
            # insert_custom_kg waits behind an in-flight document ingest
            # rather than racing it. Two concurrent custom-KG inserts that
            # touch overlapping entities likewise mutually exclude here.
            # An empty batch skips the lock entirely — nothing to serialise on.
            lock_key_set: set[str] = {entity_name for entity_name, _ in entity_nodes}
            for relationship_data in deduped_relationships.values():
                lock_key_set.add(relationship_data["src_id"])
                lock_key_set.add(relationship_data["tgt_id"])

            workspace = self.workspace or ""
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"

            async def _do_graph_and_vdb_writes() -> None:
                # Construct and verify the entity VDB payload BEFORE the
                # first graph mutation below (entity_nodes batch upsert): if
                # truncation fails (a deterministic, non-retryable
                # content-shape problem), nothing has been written yet. The
                # actual VDB upsert I/O still happens after all graph writes,
                # at the end of this function. Skipped entirely when there is
                # nothing to insert (e.g. a chunks-only custom_kg) —
                # _build_global_config is real work callers with no
                # entities/relationships should not pay for. Shared with the
                # relationship VDB payload built further below in this same
                # function, so it is built at most once per call.
                global_config: dict[str, Any] | None = None
                data_for_entities_vdb: dict[str, Any] = {}
                if all_entities_data or deduped_relationships:
                    global_config = self._build_global_config()
                if all_entities_data:
                    data_for_entities_vdb = {
                        compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                            "content": _truncate_vdb_content(
                                dp["entity_name"] + "\n" + dp["description"],
                                global_config,
                                f"entity:{dp['entity_name']}",
                            ),
                            "entity_name": dp["entity_name"],
                            "source_id": dp["source_id"],
                            "description": dp["description"],
                            "entity_type": dp["entity_type"],
                            "file_path": dp.get("file_path", "custom_kg"),
                        }
                        for dp in all_entities_data
                    }

                # Batch insert entities (reduces N serial awaits to 1).
                # Writing entity_nodes here (before the relationship-endpoint
                # discovery below) means has_nodes_batch naturally sees them,
                # so a relationship endpoint that is also one of this batch's
                # own explicit entities is never mistaken for a missing node.
                if entity_nodes:
                    await self.chunk_entity_relation_graph.upsert_nodes_batch(
                        entity_nodes
                    )

                # Insert relationships into knowledge graph (batch for performance)
                all_relationships_data: list[dict[str, str]] = []
                edge_list: list[tuple[str, str, dict[str, str]]] = []

                # Batch check which relationship endpoints exist (1 await instead of 2M)
                needed_node_ids: set[str] = set()
                for relationship_data in deduped_relationships.values():
                    needed_node_ids.add(relationship_data["src_id"])
                    needed_node_ids.add(relationship_data["tgt_id"])

                existing_nodes = await self.chunk_entity_relation_graph.has_nodes_batch(
                    list(needed_node_ids)
                )

                # Create missing nodes in batch
                missing_nodes: list[tuple[str, dict[str, str]]] = []
                for relationship_data in deduped_relationships.values():
                    src_id = relationship_data["src_id"]
                    tgt_id = relationship_data["tgt_id"]
                    source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                    source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                    file_path = normalize_document_file_path(
                        relationship_data.get("file_path", "custom_kg")
                    )

                    if source_id == "UNKNOWN":
                        logger.warning(
                            f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                        )

                    for need_insert_id in [src_id, tgt_id]:
                        if need_insert_id not in existing_nodes:
                            missing_nodes.append(
                                (
                                    need_insert_id,
                                    {
                                        "entity_id": need_insert_id,
                                        "source_id": source_id,
                                        "description": "UNKNOWN",
                                        "entity_type": "UNKNOWN",
                                        "file_path": file_path,
                                        "created_at": int(time.time()),
                                    },
                                )
                            )
                            existing_nodes.add(need_insert_id)

                    normalized_src_id, normalized_tgt_id = sorted((src_id, tgt_id))

                    edge_data = {
                        "weight": relationship_data.get("weight", 1.0),
                        "description": relationship_data["description"],
                        "keywords": relationship_data["keywords"],
                        "source_id": source_id,
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    }
                    edge_list.append((src_id, tgt_id, edge_data))

                    all_relationships_data.append(
                        {
                            "src_id": normalized_src_id,
                            "tgt_id": normalized_tgt_id,
                            "description": relationship_data["description"],
                            "keywords": relationship_data["keywords"],
                            "source_id": source_id,
                            "weight": relationship_data.get("weight", 1.0),
                            "file_path": file_path,
                            "created_at": int(time.time()),
                        }
                    )

                # Construct and verify the relationship VDB payload BEFORE
                # the graph mutations below (missing-node + edge batch
                # upserts): if truncation fails, nothing has been written
                # yet. The actual VDB upsert I/O still happens after all
                # graph writes, at the end of this function. Reuses the
                # entity-side global_config built above via closure — that
                # guard (`all_entities_data or deduped_relationships`)
                # already covers this branch, since non-empty
                # all_relationships_data implies non-empty
                # deduped_relationships.
                data_for_rels_vdb: dict[str, Any] = {}
                if all_relationships_data:
                    data_for_rels_vdb = {
                        compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                            "src_id": dp["src_id"],
                            "tgt_id": dp["tgt_id"],
                            "source_id": dp["source_id"],
                            "content": _truncate_vdb_content(
                                f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
                                global_config,
                                f"relation:{dp['src_id']}-{dp['tgt_id']}",
                            ),
                            "keywords": dp["keywords"],
                            "description": dp["description"],
                            "weight": dp["weight"],
                            "file_path": dp.get("file_path", "custom_kg"),
                        }
                        for dp in all_relationships_data
                    }

                # Batch insert missing placeholder nodes
                if missing_nodes:
                    await self.chunk_entity_relation_graph.upsert_nodes_batch(
                        missing_nodes
                    )

                # Batch insert edges
                if edge_list:
                    await self.chunk_entity_relation_graph.upsert_edges_batch(edge_list)

                legacy_rel_ids_to_delete = sorted(
                    {
                        rel_id
                        for dp in all_relationships_data
                        for rel_id in make_relation_vdb_ids(dp["src_id"], dp["tgt_id"])[
                            1:
                        ]
                    }
                )

                # Parallel VDB upserts (was serial in original)
                await asyncio.gather(
                    self.entities_vdb.upsert(data_for_entities_vdb),
                    self.relationships_vdb.upsert(data_for_rels_vdb),
                )

                if legacy_rel_ids_to_delete:
                    await self.relationships_vdb.delete(legacy_rel_ids_to_delete)

            if lock_key_set:
                if entity_nodes or deduped_relationships:
                    update_storage = True
                async with get_storage_keyed_lock(
                    sorted(lock_key_set),
                    namespace=namespace,
                    enable_logging=False,
                ):
                    await _do_graph_and_vdb_writes()
            else:
                # No entities, no relationships — nothing to serialise on.
                await _do_graph_and_vdb_writes()

        except Exception as e:
            logger.error(f"Error in ainsert_custom_kg: {e}")
            raise
        finally:
            if update_storage:
                await self._insert_done_with_cleanup()

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        return _run_sync(  # type: ignore
            lambda: self.aquery(query, param, system_prompt),
            sync_name="query",
            async_name="aquery",
            owning_loop=self._owning_loop,
        )

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query (backward compatibility wrapper).

        This function is now a wrapper around aquery_llm that maintains backward compatibility
        by returning only the LLM response content in the original format.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            system_prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str | AsyncIterator[str]: The LLM response content.
                - Non-streaming: Returns str
                - Streaming: Returns AsyncIterator[str]
        """
        # Call the new aquery_llm function to get complete results
        result = await self.aquery_llm(query, param, system_prompt)

        # Extract and return only the LLM response for backward compatibility
        llm_response = result.get("llm_response", {})

        if llm_response.get("is_streaming"):
            return llm_response.get("response_iterator")
        else:
            return llm_response.get("content", "")

    def query_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """
        Synchronous data retrieval API: returns structured retrieval results without LLM generation.

        This function is the synchronous version of aquery_data, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval.
            param: Query parameters controlling retrieval behavior (same as aquery).

        Returns:
            dict[str, Any]: Same structured data result as aquery_data.
        """
        return _run_sync(
            lambda: self.aquery_data(query, param),
            sync_name="query_data",
            async_name="aquery_data",
            owning_loop=self._owning_loop,
        )

    async def aquery_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """
        Asynchronous data retrieval API: returns structured retrieval results without LLM generation.

        This function reuses the same logic as aquery but stops before LLM generation,
        returning the final processed entities, relationships, and chunks data that would be sent to LLM.

        Args:
            query: Query text for retrieval.
            param: Query parameters controlling retrieval behavior (same as aquery).

        Returns:
            dict[str, Any]: Structured data result in the following format:

            **Success Response:**
            ```python
            {
                "status": "success",
                "message": "Query executed successfully",
                "data": {
                    "entities": [
                        {
                            "entity_name": str,      # Entity identifier
                            "entity_type": str,      # Entity category/type
                            "description": str,      # Entity description
                            "source_id": str,        # Source chunk references
                            "file_path": str,        # Origin file path
                            "created_at": str,       # Creation timestamp
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "relationships": [
                        {
                            "src_id": str,           # Source entity name
                            "tgt_id": str,           # Target entity name
                            "description": str,      # Relationship description
                            "keywords": str,         # Relationship keywords
                            "weight": float,         # Relationship strength
                            "source_id": str,        # Source chunk references
                            "file_path": str,        # Origin file path
                            "created_at": str,       # Creation timestamp
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "chunks": [
                        {
                            "content": str,          # Document chunk content
                            "file_path": str,        # Origin file path
                            "chunk_id": str,         # Unique chunk identifier
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "references": [
                        {
                            "reference_id": str,     # Reference identifier
                            "file_path": str         # Corresponding file path
                        }
                    ]
                },
                "metadata": {
                    "query_mode": str,           # Query mode used ("local", "global", "hybrid", "mix", "naive", "bypass")
                    "keywords": {
                        "high_level": List[str], # High-level keywords extracted
                        "low_level": List[str]   # Low-level keywords extracted
                    },
                    "processing_info": {
                        "total_entities_found": int,        # Total entities before truncation
                        "total_relations_found": int,       # Total relations before truncation
                        "entities_after_truncation": int,   # Entities after token truncation
                        "relations_after_truncation": int,  # Relations after token truncation
                        "merged_chunks_count": int,          # Chunks before final processing
                        "final_chunks_count": int            # Final chunks in result
                    }
                }
            }
            ```

            **Query Mode Differences:**
            - **local**: Focuses on entities and their related chunks based on low-level keywords
            - **global**: Focuses on relationships and their connected entities based on high-level keywords
            - **hybrid**: Combines local and global results using round-robin merging
            - **mix**: Includes knowledge graph data plus vector-retrieved document chunks
            - **naive**: Only vector-retrieved chunks, entities and relationships arrays are empty
            - **bypass**: All data arrays are empty, used for direct LLM queries

            ** processing_info is optional and may not be present in all responses, especially when query result is empty**

            **Failure Response:**
            ```python
            {
                "status": "failure",
                "message": str,  # Error description
                "data": {}       # Empty data object
            }
            ```

            **Common Failure Cases:**
            - Empty query string
            - Both high-level and low-level keywords are empty
            - Query returns empty dataset
            - Missing tokenizer or system configuration errors

        Note:
            The function adapts to the new data format from convert_to_user_format where
            actual data is nested under the 'data' field, with 'status' and 'message'
            fields at the top level.
        """
        global_config = self._build_global_config()

        # Create a copy of param to avoid modifying the original
        data_param = QueryParam(
            mode=param.mode,
            only_need_context=True,  # Skip LLM generation, only get context and data
            only_need_prompt=False,
            response_type=param.response_type,
            stream=False,  # Data retrieval doesn't need streaming
            top_k=param.top_k,
            chunk_top_k=param.chunk_top_k,
            max_entity_tokens=param.max_entity_tokens,
            max_relation_tokens=param.max_relation_tokens,
            max_total_tokens=param.max_total_tokens,
            hl_keywords=param.hl_keywords,
            ll_keywords=param.ll_keywords,
            conversation_history=param.conversation_history,
            user_prompt=param.user_prompt,
            enable_rerank=param.enable_rerank,
        )

        query_result = None

        if data_param.mode in ["local", "global", "hybrid", "mix"]:
            logger.debug(f"[aquery_data] Using kg_query for mode: {data_param.mode}")
            query_result = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                data_param,  # Use data_param with only_need_context=True
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                chunks_vdb=self.chunks_vdb,
            )
        elif data_param.mode == "naive":
            logger.debug(f"[aquery_data] Using naive_query for mode: {data_param.mode}")
            query_result = await naive_query(
                query.strip(),
                self.chunks_vdb,
                data_param,  # Use data_param with only_need_context=True
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                text_chunks_db=self.text_chunks,
            )
        elif data_param.mode == "bypass":
            logger.debug("[aquery_data] Using bypass mode")
            # bypass mode returns empty data using convert_to_user_format
            empty_raw_data = convert_to_user_format(
                [],  # no entities
                [],  # no relationships
                [],  # no chunks
                [],  # no references
                "bypass",
            )
            query_result = QueryResult(content="", raw_data=empty_raw_data)
        else:
            raise ValueError(f"Unknown mode {data_param.mode}")

        if query_result is None:
            no_result_message = "Query returned no results"
            if data_param.mode == "naive":
                no_result_message = "No relevant document chunks found."
            final_data: dict[str, Any] = {
                "status": "failure",
                "message": no_result_message,
                "data": {},
                "metadata": {
                    "failure_reason": "no_results",
                    "mode": data_param.mode,
                },
            }
            logger.info("[aquery_data] Query returned no results.")
        else:
            # Extract raw_data from QueryResult
            final_data = query_result.raw_data or {}

            # Log final result counts - adapt to new data format from convert_to_user_format
            if final_data and "data" in final_data:
                data_section = final_data["data"]
                entities_count = len(data_section.get("entities", []))
                relationships_count = len(data_section.get("relationships", []))
                chunks_count = len(data_section.get("chunks", []))
                logger.debug(
                    f"[aquery_data] Final result: {entities_count} entities, {relationships_count} relationships, {chunks_count} chunks"
                )
            else:
                logger.warning("[aquery_data] No data section found in query result")

        await self._query_done()
        return final_data

    async def aquery_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Asynchronous complete query API: returns structured retrieval results with LLM generation.

        This function performs a single query operation and returns both structured data and LLM response,
        based on the original aquery logic to avoid duplicate calls.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            dict[str, Any]: Complete response with structured data and LLM response.
        """
        logger.debug(f"[aquery_llm] Query param: {param}")

        global_config = self._build_global_config()

        try:
            query_result = None

            if param.mode in ["local", "global", "hybrid", "mix"]:
                query_result = await kg_query(
                    query.strip(),
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    global_config,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                    chunks_vdb=self.chunks_vdb,
                    progress_callback=progress_callback,
                )
            elif param.mode == "naive":
                query_result = await naive_query(
                    query.strip(),
                    self.chunks_vdb,
                    param,
                    global_config,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                    text_chunks_db=self.text_chunks,
                    progress_callback=progress_callback,
                )
            elif param.mode == "bypass":
                # Bypass mode: directly use LLM without knowledge retrieval
                # Apply higher priority to entity/relation summary tasks
                use_llm_func = partial(
                    global_config["role_llm_funcs"]["query"],
                    _priority=DEFAULT_SUMMARY_PRIORITY,
                )

                param.stream = True if param.stream is None else param.stream
                response = await use_llm_func(
                    query.strip(),
                    system_prompt=system_prompt,
                    history_messages=param.conversation_history,
                    enable_cot=True,
                    stream=param.stream,
                )
                # isinstance, not exact type: a truncated non-streaming
                # response arrives as TruncatedResponse (a str subclass) and
                # must not be misclassified as a streaming iterator.
                if isinstance(response, str):
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM non streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": response,
                            "response_iterator": None,
                            "is_streaming": False,
                        },
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": None,
                            "response_iterator": response,
                            "is_streaming": True,
                        },
                    }
            else:
                raise ValueError(f"Unknown mode {param.mode}")

            await self._query_done()

            # Check if query_result is None
            if query_result is None:
                return {
                    "status": "failure",
                    "message": "Query returned no results",
                    "data": {},
                    "metadata": {
                        "failure_reason": "no_results",
                        "mode": param.mode,
                    },
                    "llm_response": {
                        "content": PROMPTS["fail_response"],
                        "response_iterator": None,
                        "is_streaming": False,
                    },
                }

            # Extract structured data from query result
            raw_data = query_result.raw_data or {}
            raw_data["llm_response"] = {
                "content": query_result.content
                if not query_result.is_streaming
                else None,
                "response_iterator": query_result.response_iterator
                if query_result.is_streaming
                else None,
                "is_streaming": query_result.is_streaming,
            }

            return raw_data

        except Exception as e:
            logger.error(f"Query failed: {e}")
            # Return error response
            return {
                "status": "failure",
                "message": f"Query failed: {str(e)}",
                "data": {},
                "metadata": {},
                "llm_response": {
                    "content": None,
                    "response_iterator": None,
                    "is_streaming": False,
                },
            }

    def query_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Synchronous complete query API: returns structured retrieval results with LLM generation.

        This function is the synchronous version of aquery_llm, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            dict[str, Any]: Same complete response format as aquery_llm.
        """
        return _run_sync(
            lambda: self.aquery_llm(query, param, system_prompt),
            sync_name="query_llm",
            async_name="aquery_llm",
            owning_loop=self._owning_loop,
        )

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    async def _update_delete_retry_state(
        self,
        doc_id: str,
        doc_status_data: dict[str, Any],
        *,
        deletion_stage: str,
        doc_llm_cache_ids: list[str],
        error_message: str | None = None,
        failed: bool,
    ) -> dict[str, Any]:
        """Persist deletion retry metadata and return the updated status record.

        Re-reads the record first, and writes only the fields it changes. The
        caller's ``doc_status_data`` is a snapshot taken before deletion began,
        and the purge that runs in between writes its journal into
        ``metadata`` through a targeted update — so upserting the whole stale
        snapshot would silently clobber that journal, which is precisely what
        keeps the retry from being refused for missing recovery anchors
        (issue #3400).

        For the same reason the re-read must never silently fall back to that
        snapshot. Strict where the backend supports it, so a read failure
        raises (every caller wraps this in its own try/except and settles for
        logging). A ``None`` — confirmed absent under strict reads, ambiguous
        otherwise — skips the write entirely: rebuilding ``metadata`` from the
        pre-deletion snapshot when the row is actually alive (a masked read
        failure) would erase the journal, and an unrecorded retry state is
        recoverable while a clobbered journal is a permanent 409. Retry-state
        metadata is diagnostics; the journal is load-bearing.
        """
        strict_reads = getattr(self.doc_status, "supports_strict_point_reads", False)
        current = await (
            self.doc_status.get_by_id_strict(doc_id)
            if strict_reads
            else self.doc_status.get_by_id(doc_id)
        )
        if not isinstance(current, dict):
            logger.warning(
                "Skipping deletion retry-state write for document %s: its "
                "doc_status row is %s",
                doc_id,
                "confirmed absent" if strict_reads else "absent or unreadable",
            )
            return doc_status_data
        base_record = current
        metadata = base_record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        backup_cache_ids = normalize_string_list(
            metadata.get("deletion_llm_cache_ids", []),
            context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
        )
        retry_cache_ids = doc_llm_cache_ids or backup_cache_ids

        updated_metadata = dict(metadata)
        if retry_cache_ids:
            updated_metadata["deletion_llm_cache_ids"] = retry_cache_ids
        updated_metadata["last_deletion_attempt_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        if failed:
            updated_metadata["deletion_failed"] = True
            updated_metadata["deletion_failure_stage"] = deletion_stage
        else:
            updated_metadata.pop("deletion_failed", None)
            updated_metadata.pop("deletion_failure_stage", None)

        changed_fields = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": updated_metadata,
            "error_msg": error_message if failed else "",
        }

        # Targeted update, and ``missing_ok`` so a record a concurrent delete
        # already removed is not resurrected as a zombie by this write.
        await self.doc_status.update_doc_status_fields(
            doc_id, changed_fields, missing_ok=True
        )
        return {**base_record, **changed_fields}

    async def _get_existing_llm_cache_ids(self, cache_ids: list[str]) -> list[str]:
        """Return cache IDs that still exist in cache storage.

        Some KV storage backends only log delete failures and return without
        raising, so callers must verify which records still exist after delete.

        Returns an empty list immediately if cache storage is unavailable.
        Callers must check storage availability independently before treating
        an empty result as a confirmed deletion.
        """
        if not self.llm_response_cache or not cache_ids:
            return []

        try:
            existing_records = await self.llm_response_cache.get_by_ids(cache_ids)
        except Exception as verification_error:
            raise Exception(
                f"Failed to verify LLM cache deletion "
                f"(delete may have succeeded): {verification_error}"
            ) from verification_error
        return [
            cache_id
            for cache_id, record in zip(cache_ids, existing_records)
            if record is not None
        ]

    async def aclear_cache(self) -> None:
        """Clear all cache data from the LLM response cache storage.

        This method clears all cached LLM responses regardless of mode.

        Example:
            # Clear all cache
            await rag.aclear_cache()
        """
        if not self.llm_response_cache:
            logger.warning("No cache storage configured")
            return

        try:
            # Clear all cache using drop method
            success = await self.llm_response_cache.drop()
            if success:
                logger.info("Cleared all cache")
            else:
                logger.warning("Failed to clear all cache")

            await self.llm_response_cache.index_done_callback()

        except Exception as e:
            logger.error(f"Error while clearing cache: {e}")

    def clear_cache(self) -> None:
        """Synchronous version of aclear_cache."""
        return _run_sync(
            lambda: self.aclear_cache(),
            sync_name="clear_cache",
            async_name="aclear_cache",
            owning_loop=self._owning_loop,
        )

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def aget_docs_by_ids(
        self, ids: str | list[str]
    ) -> dict[str, DocProcessingStatus]:
        """Retrieves the processing status for one or more documents by their IDs.

        Args:
            ids: A single document ID (string) or a list of document IDs (list of strings).

        Returns:
            A dictionary where keys are the document IDs for which a status was found,
            and values are the corresponding DocProcessingStatus objects. IDs that
            are not found in the storage will be omitted from the result dictionary.
        """
        if isinstance(ids, str):
            # Ensure input is always a list of IDs for uniform processing
            id_list = [ids]
        elif (
            ids is None
        ):  # Handle potential None input gracefully, although type hint suggests str/list
            logger.warning(
                "aget_docs_by_ids called with None input, returning empty dict."
            )
            return {}
        else:
            # Assume input is already a list if not a string
            id_list = ids

        # Return early if the final list of IDs is empty
        if not id_list:
            logger.debug("aget_docs_by_ids called with an empty list of IDs.")
            return {}

        # Create tasks to fetch document statuses concurrently using the doc_status storage
        tasks = [self.doc_status.get_by_id(doc_id) for doc_id in id_list]
        # Execute tasks concurrently and gather the results. Results maintain order.
        # Type hint indicates results can be DocProcessingStatus or None if not found.
        results_list: list[Optional[DocProcessingStatus]] = await asyncio.gather(*tasks)

        # Build the result dictionary, mapping found IDs to their statuses
        found_statuses: dict[str, DocProcessingStatus] = {}
        # Keep track of IDs for which no status was found (for logging purposes)
        not_found_ids: list[str] = []

        # Iterate through the results, correlating them back to the original IDs
        for i, status_obj in enumerate(results_list):
            doc_id = id_list[
                i
            ]  # Get the original ID corresponding to this result index
            if status_obj:
                # If a status object was returned (not None), add it to the result dict
                found_statuses[doc_id] = status_obj
            else:
                # If status_obj is None, the document ID was not found in storage
                not_found_ids.append(doc_id)

        # Log a warning if any of the requested document IDs were not found
        if not_found_ids:
            logger.warning(
                f"Document statuses not found for the following IDs: {not_found_ids}"
            )

        # Return the dictionary containing statuses only for the found document IDs
        return found_statuses

    async def _purge_doc_chunks_and_kg(
        self,
        doc_id: str,
        chunk_ids: list[str],
        *,
        pipeline_status: dict,
        pipeline_status_lock: Any,
    ) -> None:
        """Remove a document's chunks and clean up its knowledge-graph contributions.

        Thin wrapper over :py:meth:`_purge_kg_contributions` in whole-document
        mode: candidates are discovered from the per-doc ``full_entities`` /
        ``full_relations`` recovery rows and those rows are deleted at the
        end. See the primitive for the detailed contract.

        Used by:
            - The pipeline resume branch in ``process_document`` when a
              document whose content is already extracted is re-processed
              under different ``process_options``: chunks must be wiped and
              entities/relations rebuilt fresh.
            - Future deletion paths that want a focused "purge KG only"
              operation without the LLM-cache / doc_status / full_docs
              cleanup that ``adelete_by_doc_id`` also performs.
        """
        await self._purge_kg_contributions(
            doc_id,
            chunk_ids,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
        )

    @staticmethod
    def _patch_journal_candidates(
        status_doc: Any,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Candidates anchored only in an in-flight custom-chunk patch journal.

        A patch-mode merge passes no anchor storages — the journal IS that
        operation's recovery anchor — and its candidates are unioned into the
        base document's ``full_entities`` / ``full_relations`` rows only at
        commit. So between a failed patch and its rollback, these graph objects
        exist while no anchor row names them, and a whole-document purge that
        consulted the anchors alone would delete the staged chunks and leave
        the objects behind.
        """
        journal = doc_status_custom_chunk_patch(status_doc)
        if journal is None:
            return (), ()
        entities = tuple(
            name
            for name in (journal.get("entity_names") or [])
            if isinstance(name, str) and name
        )
        relations = tuple(
            (pair[0], pair[1])
            for pair in (journal.get("relation_pairs") or [])
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(end, str) and end for end in pair)
        )
        return entities, relations

    async def _resolve_purge_recovery_proof(
        self,
        doc_id: str,
        chunk_ids: list[str],
    ) -> _PurgeRecoveryProof:
        """Decide whether a whole-document purge is allowed to delete anything.

        Fail-closed precondition for issue #3400. A purge discovers what a
        document contributed to the shared graph from its write-ahead anchors;
        without them the reverse lookup (graph ``source_id`` → ``text_chunks``
        → ``full_doc_id``) is impossible once the chunks are gone. So rather
        than treating absent anchors as "no contributions" — which silently
        skipped graph cleanup and stranded unattributable entities — this
        resolves one of three PROOFS, or reports that none applies:

        * ``anchors`` — both anchor rows are present and structurally usable.
          Presence is the test, never list truthiness: a row holding an empty
          list is a valid proof (a document that extracted no entities), and
          conflating the two is the original bug.
        * ``pre_graph`` — persisted ``kg_write_state`` says the document never
          reached its first graph mutation, so there is provably nothing in the
          graph to find. The caller may clean up staged chunks with an
          explicitly empty candidate set.
        * ``journal`` — a ``kg_purge`` journal from a previous attempt records a
          phase past the point where the anchors were legitimately deleted.
          Without this, purge's own last step (deleting the anchors) would make
          every subsequent retry refuse forever.

        Never raises for a missing proof — returns it as
        :attr:`_PurgeRecoveryProof.missing_reason` so the caller raises after
        it has assembled a full, actionable message. A ``doc_status`` read
        failure DOES propagate: an unknown state must not be read as an absent
        journal (that would re-run a purge whose anchors are already gone).
        """
        # Strict-where-supported, honoring the docstring's "a read failure
        # DOES propagate": a plain get_by_id may present backend trouble as
        # None, which would silently discard an in-flight journal — the one
        # record distinguishing "anchors legitimately deleted" from "never
        # written". A None from the strict read is CONFIRMED absence, which
        # stays a legal input: every derived proof reads as unknown and the
        # resolution below fails closed with the actionable 409.
        strict_status_reads = getattr(
            self.doc_status, "supports_strict_point_reads", False
        )
        status_doc = await (
            self.doc_status.get_by_id_strict(doc_id)
            if strict_status_reads
            else self.doc_status.get_by_id(doc_id)
        )
        journal = doc_status_kg_purge_journal(status_doc)
        write_state = doc_status_kg_write_state(status_doc)
        operation_id = make_kg_purge_operation_id(doc_id, chunk_ids)
        patch_entities, patch_relations = self._patch_journal_candidates(status_doc)

        journal_phase: str | None = None
        if journal is not None:
            journal_operation_id = journal.get("operation_id")
            mismatched = (
                not isinstance(journal_operation_id, str)
                or journal_operation_id != operation_id
            )
            if mismatched and journal.get("phase") == KG_PURGE_PHASE_COMPLETED:
                # A ``completed`` journal describes a FINISHED purge, so it
                # holds no resume information worth protecting — and its
                # retirement is a separate write that a crash can skip. Treat a
                # stale one as ignorable rather than a conflict, or a document
                # re-purged with a different chunk set would be refused forever
                # over bookkeeping that no longer means anything.
                logger.info(
                    f"[purge] {doc_id}: ignoring a stale completed purge journal "
                    f"for operation {journal_operation_id!r} (now purging "
                    f"{operation_id!r})"
                )
                journal = None
            elif mismatched:
                raise KGPurgeOperationConflictError(
                    f"Document {doc_id} carries a KG purge journal for a different "
                    f"operation (journal={journal_operation_id!r}, "
                    f"requested={operation_id!r}); the document's chunk set changed "
                    f"since that purge started. Retry the operation that wrote the "
                    f"journal (its chunk set is unchanged, so its purge resumes), "
                    f"or resolve with audit_kg_integrity(..., apply=True) before "
                    f"retrying this one.",
                    doc_id=doc_id,
                    journal_operation_id=(
                        journal_operation_id
                        if isinstance(journal_operation_id, str)
                        else ""
                    ),
                    requested_operation_id=operation_id,
                )
            if journal is not None:
                phase = journal.get("phase")
                if phase in _KG_PURGE_RESUMABLE_PHASES:
                    journal_phase = phase

        # A journal past ``prepared`` is itself the proof: the anchors may
        # already be gone precisely BECAUSE a previous attempt got that far.
        if journal_phase is not None and journal_phase != KG_PURGE_PHASE_PREPARED:
            return _PurgeRecoveryProof(
                proof_kind="journal",
                operation_id=operation_id,
                journal_phase=journal_phase,
                write_state=write_state,
                patch_candidate_entities=patch_entities,
                patch_candidate_relations=patch_relations,
            )

        missing_namespaces: list[str] = []
        entities_row = await self.full_entities.get_by_id(doc_id)
        if not isinstance(entities_row, dict) or not isinstance(
            entities_row.get("entity_names"), list
        ):
            missing_namespaces.append("full_entities")
        relations_row = await self.full_relations.get_by_id(doc_id)
        if not isinstance(relations_row, dict) or not isinstance(
            relations_row.get("relation_pairs"), list
        ):
            missing_namespaces.append("full_relations")

        if not missing_namespaces:
            # Anchors name KG objects but the document owns no chunks to
            # attribute them to. Classification subtracts this document's
            # chunk ids from each object's sources, so an empty chunk set
            # classifies every object as "keep" and then the anchors are
            # deleted anyway — the same orphan outcome fail-closed exists to
            # prevent. Empty anchors with no chunks is fine: nothing to strand.
            if not chunk_ids and (
                entities_row["entity_names"] or relations_row["relation_pairs"]
            ):
                return _PurgeRecoveryProof(
                    proof_kind=None,
                    operation_id=operation_id,
                    journal_phase=journal_phase,
                    write_state=write_state,
                    missing_reason=RecoveryAnchorMissingError.REASON_CHUNKLESS_CONTRIBUTIONS,
                    patch_candidate_entities=patch_entities,
                    patch_candidate_relations=patch_relations,
                )
            return _PurgeRecoveryProof(
                proof_kind="anchors",
                operation_id=operation_id,
                journal_phase=journal_phase,
                write_state=write_state,
                patch_candidate_entities=patch_entities,
                patch_candidate_relations=patch_relations,
            )

        if write_state == KG_WRITE_STATE_PRE_GRAPH:
            return _PurgeRecoveryProof(
                proof_kind="pre_graph",
                operation_id=operation_id,
                journal_phase=journal_phase,
                write_state=write_state,
                missing_namespaces=tuple(missing_namespaces),
                patch_candidate_entities=patch_entities,
                patch_candidate_relations=patch_relations,
            )

        # Nothing that CARRIES attribution would be deleted, so there is no
        # damage for a proof to guard against. Fail-closed exists to stop a
        # purge from destroying the chunk rows and anchor rows that are the
        # only record of what a document contributed; an operation that
        # removes neither cannot strand anything, whatever the document's
        # history. Reached only when at least one anchor row is absent (the
        # both-present case returned above), so this covers exactly the
        # documents no proof can be produced for: a legacy row enqueued before
        # ``kg_write_state`` existed, still holding no chunks.
        #
        # Deliberately NOT generalised into a persisted marker. A stored
        # ``pre_graph`` says "this document never touched the graph", which
        # licenses deleting its chunks while skipping the graph — so inferring
        # one from observable state would turn a transient inconsistency (a
        # chunks_list that is momentarily empty) into a durable licence to
        # reproduce issue #3400 the moment the chunks reappear. This decision
        # is re-evaluated against live state on every call and grants nothing
        # beyond the call, which is why it is safe where a backfill is not.
        if (
            not chunk_ids
            and not (
                isinstance(entities_row, dict) and entities_row.get("entity_names")
            )
            and not (
                isinstance(relations_row, dict) and relations_row.get("relation_pairs")
            )
        ):
            return _PurgeRecoveryProof(
                proof_kind="empty_scope",
                operation_id=operation_id,
                journal_phase=journal_phase,
                write_state=write_state,
                missing_namespaces=tuple(missing_namespaces),
                patch_candidate_entities=patch_entities,
                patch_candidate_relations=patch_relations,
            )

        return _PurgeRecoveryProof(
            proof_kind=None,
            operation_id=operation_id,
            journal_phase=journal_phase,
            write_state=write_state,
            missing_namespaces=tuple(missing_namespaces),
            missing_reason=RecoveryAnchorMissingError.REASON_MISSING_ANCHOR_ROWS,
            patch_candidate_entities=patch_entities,
            patch_candidate_relations=patch_relations,
        )

    @staticmethod
    def _raise_missing_recovery_proof(
        doc_id: str, proof: _PurgeRecoveryProof
    ) -> NoReturn:
        """Turn an unproven purge into an actionable, client-safe refusal.

        Split out so the purge primitive and the delete path raise identical
        messages, and so the remedy (the offline audit tool) is named exactly
        once.
        """
        remedy = (
            "Nothing was deleted. Rebuild the recovery anchors with "
            "audit_kg_integrity(..., apply=True) "
            "(python -m lightrag.tools.kg_integrity_repair), then retry."
        )
        if (
            proof.missing_reason
            == RecoveryAnchorMissingError.REASON_CHUNKLESS_CONTRIBUTIONS
        ):
            message = (
                f"Refusing to purge document {doc_id}: its recovery anchors name "
                f"knowledge-graph objects, but the document owns no chunks to "
                f"attribute them to, so those objects cannot be classified and "
                f"would be orphaned. {remedy}"
            )
        else:
            missing = (
                ", ".join(proof.missing_namespaces) or "full_entities, full_relations"
            )
            message = (
                f"Refusing to purge document {doc_id}: recovery anchor row(s) "
                f"missing or unusable ({missing}) and the document may already "
                f"have written to the knowledge graph "
                f"(kg_write_state={proof.write_state or 'unknown'}). Purging now "
                f"would delete its chunks while leaving unattributable graph "
                f"objects behind. {remedy}"
            )
        raise RecoveryAnchorMissingError(
            message,
            doc_id=doc_id,
            reason=proof.missing_reason
            or RecoveryAnchorMissingError.REASON_MISSING_ANCHOR_ROWS,
            missing_namespaces=proof.missing_namespaces,
        )

    async def _write_kg_purge_phase(
        self,
        doc_id: str,
        phase: str,
        *,
        operation_id: str,
        chunk_count: int,
    ) -> None:
        """Persist (and flush) the purge journal's phase for ``doc_id``.

        ``metadata`` is an opaque replace-on-write blob in every backend, so
        this is a read-modify-write of that one field via the targeted
        ``update_doc_status_fields`` primitive — never a whole-record upsert,
        which would drag ``chunks_list`` through memory.

        Flushed before returning: a journal that is only in memory proves
        nothing about what is already deleted on disk.

        Raises when the row cannot be read (strict where the backend supports
        it) or has vanished. Every call site runs while the row is guaranteed
        to exist — ``adelete_by_doc_id`` read it before starting, and the
        caller's finalization (which deletes it) comes only after the purge —
        so an unreadable row must abort the purge rather than skip the
        journal: a destructive purge that runs unjournaled loses the only
        record that lets a retry survive its own anchor deletion (step 8), and
        the failure mode it creates — anchors gone, journal absent — is a
        permanent fail-closed refusal.
        """
        status_doc = await require_doc_status_record(
            self.doc_status, doc_id, purpose=f"journal purge phase '{phase}'"
        )
        metadata = doc_status_field(status_doc, "metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata[KG_PURGE_METADATA_KEY] = {
            "schema_version": KG_PURGE_SCHEMA_VERSION,
            "operation_id": operation_id,
            "phase": phase,
            "chunk_count": chunk_count,
            "updated_at": int(time.time()),
        }
        # missing_ok=False: the row vanishing between the read above and this
        # write is the same unjournaled-purge hazard, not a tolerable race.
        await self.doc_status.update_doc_status_fields(doc_id, {"metadata": metadata})
        await self._flush_storages([self.doc_status])

    async def _clear_kg_purge_journal(
        self,
        doc_id: str,
        *,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Retire a completed purge journal.

        Used by callers that KEEP the ``doc_status`` row after a purge (the
        pipeline's stale-extraction reset). A caller that deletes the row
        instead — ``adelete_by_doc_id`` — must NOT call this: the ``completed``
        journal is what keeps its own post-purge finalization retryable, and it
        disappears with the row.

        ``extra_fields`` rides in the same targeted write so the reset of
        ``chunks_list`` / ``chunks_count`` and the journal retirement land
        atomically — a crash between them would leave the document pointing at
        chunks this purge already deleted.

        Deliberately does NOT touch ``kg_write_state``. That marker is
        monotonic, and ``_mark_graph_mutation_started`` is its only writer, so
        neither direction is safe to guess here: forcing
        ``graph_mutation_started`` would make a document that provably never
        merged start demanding anchors it will never have (a false refusal that
        leaves it neither reprocessable nor deletable), while clearing it would
        discard the proof a still-pre-graph document depends on.

        Raises when the row cannot be read or has vanished, same as
        :py:meth:`_write_kg_purge_phase`: the one caller runs mid-pipeline
        with the row guaranteed present, and skipping silently would leave
        the stored ``chunks_list`` pointing at chunks the purge just deleted.
        """
        status_doc = await require_doc_status_record(
            self.doc_status, doc_id, purpose="retire the purge journal"
        )
        metadata = doc_status_field(status_doc, "metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata.pop(KG_PURGE_METADATA_KEY, None)
        fields: dict[str, Any] = {"metadata": metadata}
        if extra_fields:
            fields.update(extra_fields)
        await self.doc_status.update_doc_status_fields(doc_id, fields)
        await self._flush_storages([self.doc_status])

    async def _purge_kg_contributions(
        self,
        doc_id: str,
        chunk_ids: list[str],
        *,
        candidate_entities: list[str] | None = None,
        candidate_relations: list[tuple[str, str]] | None = None,
        patch_only: bool = False,
        rebuild_policy: Literal["best_effort", "rollback"] = "best_effort",
        pipeline_status: dict,
        pipeline_status_lock: Any,
    ) -> KGRebuildReport:
        """Candidate-driven purge/rebuild primitive (issue #3400).

        Shared by whole-document purge (normal deletion / retry) and, in later
        phases, custom-chunk patch rollback. Candidates are a recovery
        SUPERSET — a candidate that never reached the graph is an idempotent
        no-op; ownership is verified per candidate from the union of
        chunk-tracking entries and graph ``source_id`` lists before anything
        is deleted or rebuilt.

        What this method does:
            1. Resolves the candidate entity/relation set: the explicit
               ``candidate_entities`` / ``candidate_relations`` arguments if
               given, otherwise the per-doc ``full_entities`` /
               ``full_relations`` recovery rows.
            2. For each affected entity / relation, intersects the doc's
               ``chunk_ids`` with the union of chunk-tracking entries
               (``entity_chunks`` / ``relation_chunks``) and graph
               ``source_id`` lists, then classifies it as either
               *delete-outright* (no remaining sources) or *rebuild*
               (still references chunks from other documents).
            3. For *delete-outright* entries: removes the relationship /
               entity from the graph storage, vector storage, and chunk
               tracking, then persists via :py:meth:`_insert_done`.
            4. Calls :func:`rebuild_knowledge_from_chunks` to rebuild any
               *rebuild* entries from their remaining chunks (so other
               documents that also contributed to the same entity /
               relation keep their data intact), then flushes the rebuilt
               state. Rollback policy repairs provenance structurally when
               extraction data is missing, but still raises on operational
               graph/vector/tracking failures.
            5. Only after every derived contribution is repaired/removed
               and flushed: deletes the chunks themselves from
               ``chunks_vdb`` and ``text_chunks`` and flushes them (safe
               destructive ordering — graph objects never point at deleted
               chunks).
            6. Unless ``patch_only``: deletes the per-doc ``full_entities``
               / ``full_relations`` index rows LAST and flushes, so every
               intermediate failure keeps the recovery anchors and stays
               retryable. A patch rollback must NOT drop the base
               document's recovery rows — the document keeps owning its
               previously committed contributions.

        Does NOT touch:
            - ``doc_status`` / ``full_docs`` records — caller manages those.
            - ``llm_response_cache`` — orthogonal to KG cleanup.
            - Pipeline busy-flag — assumes the caller already holds the
              pipeline (i.e. this runs inside a pipeline run).

        Idempotent and RESUMABLE. In whole-document (anchor-driven) mode this
        additionally journals its progress to
        ``doc_status.metadata.kg_purge`` and refuses to start without a
        recovery proof — see :py:meth:`_resolve_purge_recovery_proof`. The
        journal is what makes fail-closed safe to combine with step 8: once
        the anchors are gone, only the journal can tell a retry that they were
        deleted legitimately rather than never written. A resumed purge skips
        exactly the phases already persisted, so it never re-runs the
        LLM-cache-backed rebuild. Repeating a partially-failed purge converges
        (absent objects are skipped, remaining ones are cleaned).

        Explicit-candidate mode (custom-chunk patch rollback) is NOT journaled
        and skips the proof: its caller's own operation journal already names
        the complete candidate superset, and an empty ``chunk_ids`` returns
        immediately without touching storage.
        """
        # Anchor-driven whole-document purge: the only mode that both needs a
        # recovery proof (it discovers candidates FROM the anchors) and deletes
        # the anchors at the end (so it needs a journal to stay retryable).
        journaled = (
            candidate_entities is None
            and candidate_relations is None
            and not patch_only
        )

        proof: _PurgeRecoveryProof | None = None
        if journaled:
            proof = await self._resolve_purge_recovery_proof(doc_id, chunk_ids)
            if proof.phase_at_least(KG_PURGE_PHASE_COMPLETED):
                logger.info(
                    f"[purge] {doc_id}: journal reports this purge already "
                    f"completed; nothing left to delete"
                )
                return KGRebuildReport()
            if proof.proof_kind is None:
                self._raise_missing_recovery_proof(doc_id, proof)
            if proof.proof_kind == "pre_graph":
                # The document's OWN merge provably never ran, so force an
                # EXPLICIT empty candidate set rather than inferring "no
                # candidates" from anchors that could not be read (the original
                # bug). The patch-journal extras still apply below: a patch-mode
                # merge mutates the graph without writing anchors, so those are
                # the one kind of contribution a pre-graph document can have.
                logger.warning(
                    f"[purge] {doc_id}: recovery anchor row(s) missing "
                    f"({', '.join(proof.missing_namespaces)}), but kg_write_state "
                    f"proves the document's own merge never started; cleaning up "
                    f"staged chunks"
                    + (
                        f" and {len(proof.patch_candidate_entities)} journal-anchored "
                        f"candidate(s)"
                        if proof.patch_candidate_entities
                        else " only"
                    )
                )
                candidate_entities = []
                candidate_relations = []
            elif proof.proof_kind == "empty_scope":
                # Nothing attribution-bearing is being deleted (no chunks, no
                # anchor row that names anything), so there is nothing for the
                # derived pass to do and no reason to read the graph.
                logger.info(
                    f"[purge] {doc_id}: no chunks and no populated recovery "
                    f"anchors, so nothing that carries attribution would be "
                    f"deleted; removing the empty record"
                )
                candidate_entities = []
                candidate_relations = []
            if proof.journal_phase is None:
                # Journal BEFORE the first destructive write. A crash between
                # here and step 8 is then distinguishable from a document that
                # never had anchors at all.
                await self._write_kg_purge_phase(
                    doc_id,
                    KG_PURGE_PHASE_PREPARED,
                    operation_id=proof.operation_id,
                    chunk_count=len(chunk_ids),
                )
        elif not chunk_ids:
            return KGRebuildReport()

        rebuild_report = KGRebuildReport()

        # ---- Steps 1-6: repair or remove every derived contribution ----
        if proof is None or not proof.phase_at_least(KG_PURGE_PHASE_DERIVED_COMMITTED):
            rebuild_report = await self._purge_derived_kg_contributions(
                doc_id,
                chunk_ids,
                candidate_entities=candidate_entities,
                candidate_relations=candidate_relations,
                extra_candidate_entities=(
                    proof.patch_candidate_entities if proof is not None else ()
                ),
                extra_candidate_relations=(
                    proof.patch_candidate_relations if proof is not None else ()
                ),
                rebuild_policy=rebuild_policy,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )
            if journaled and proof is not None:
                await self._write_kg_purge_phase(
                    doc_id,
                    KG_PURGE_PHASE_DERIVED_COMMITTED,
                    operation_id=proof.operation_id,
                    chunk_count=len(chunk_ids),
                )
        else:
            logger.info(
                f"[purge] {doc_id}: resuming at phase "
                f"'{proof.journal_phase}'; derived contributions already clean"
            )

        # ---- 7. Delete chunks themselves ----
        if chunk_ids and (
            proof is None or not proof.phase_at_least(KG_PURGE_PHASE_ANCHORS_PENDING)
        ):
            # Deleted only AFTER every derived graph/vector/tracking contribution
            # has been repaired or removed AND flushed (issue #3400: unsafe
            # destructive ordering). A failure before this point leaves the
            # chunks in place, so graph objects never reference deleted chunks.
            try:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)
                await self._flush_storages([self.chunks_vdb, self.text_chunks])
                async with pipeline_status_lock:
                    log_message = f"[purge] {doc_id}: deleted {len(chunk_ids)} chunk(s) from storage"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    append_pipeline_history(pipeline_status, log_message)
            except Exception as e:
                logger.error(f"[purge] Failed to delete chunks for {doc_id}: {e}")
                raise _PurgeStageError(
                    f"Failed to delete document chunks: {e}", stage="delete_chunks"
                ) from e

        # Chunk deletion is confirmed durable: the anchors may now go. This
        # phase is the one that keeps step 8 retryable — without it, a failure
        # after the first anchor delete would make every retry see "anchors
        # missing" and refuse forever.
        if (
            journaled
            and proof is not None
            and not proof.phase_at_least(KG_PURGE_PHASE_ANCHORS_PENDING)
        ):
            await self._write_kg_purge_phase(
                doc_id,
                KG_PURGE_PHASE_ANCHORS_PENDING,
                operation_id=proof.operation_id,
                chunk_count=len(chunk_ids),
            )

        # ---- 8. Delete per-doc full_entities / full_relations index rows ----
        # LAST, so every intermediate purge failure keeps the recovery
        # anchors and stays retryable. Patch-only rollback keeps the base
        # document's recovery rows: the document still owns its previously
        # committed contributions.
        if not patch_only:
            try:
                await self.full_entities.delete([doc_id])
                await self.full_relations.delete([doc_id])
                await self._flush_storages([self.full_entities, self.full_relations])
            except Exception as e:
                logger.error(
                    f"[purge] Failed to delete full_entities/full_relations rows for {doc_id}: {e}"
                )
                raise _PurgeStageError(
                    f"Failed to delete from full_entities/full_relations: {e}",
                    stage="delete_doc_graph_metadata",
                ) from e

        if journaled and proof is not None:
            # The caller finalizes from here (doc_status / full_docs / LLM
            # cache). The journal stays until the caller either retires it
            # (_clear_kg_purge_journal) or deletes the doc_status row, so a
            # finalization failure retries as a no-op purge instead of
            # tripping the missing-anchor refusal.
            await self._write_kg_purge_phase(
                doc_id,
                KG_PURGE_PHASE_COMPLETED,
                operation_id=proof.operation_id,
                chunk_count=len(chunk_ids),
            )

        return rebuild_report

    async def _purge_derived_kg_contributions(
        self,
        doc_id: str,
        chunk_ids: list[str],
        *,
        candidate_entities: list[str] | None,
        candidate_relations: list[tuple[str, str]] | None,
        extra_candidate_entities: tuple[str, ...] = (),
        extra_candidate_relations: tuple[tuple[str, str], ...] = (),
        rebuild_policy: Literal["best_effort", "rollback"],
        pipeline_status: dict,
        pipeline_status_lock: Any,
    ) -> KGRebuildReport:
        """Steps 1-6 of :py:meth:`_purge_kg_contributions`.

        Repairs or removes every DERIVED contribution of ``chunk_ids`` —
        graph nodes/edges, entity/relation vectors, chunk tracking — and
        flushes, leaving the chunks themselves and the recovery anchors in
        place. Split out so a resumed purge whose journal already records
        ``derived_committed`` can skip it wholesale: it is the expensive part
        (candidate re-analysis plus an LLM-cache-backed rebuild).

        Candidates are a recovery SUPERSET: an explicit ``[]`` means "provably
        nothing to clean" (the caller established that), while ``None`` means
        "resolve from this document's anchor rows". The caller is responsible
        for having proven that the anchors are readable — passing ``None`` with
        absent anchors is exactly the silent-skip bug of issue #3400.

        ``extra_candidate_*`` are unioned in after that resolution, for objects
        the anchor rows cannot name yet (an in-flight custom-chunk patch
        journal). They are additive only, so they can never mask a missing
        anchor: an empty union with absent anchors still resolves to nothing.
        """
        rebuild_report = KGRebuildReport()

        # Set view for membership/intersection checks below (chunk_ids stays a list
        # so it satisfies the storage delete contract: ``delete(ids: list[str])``).
        chunk_ids_set = set(chunk_ids)
        # ---- 1. Analyze affected entities/relations from the candidate set ----
        entities_to_delete: set[str] = set()
        entities_to_rebuild: dict[str, list[str]] = {}
        relationships_to_delete: set[tuple[str, str]] = set()
        relationships_to_rebuild: dict[tuple[str, str], list[str]] = {}
        entity_chunk_updates: dict[str, list[str]] = {}
        relation_chunk_updates: dict[tuple[str, str], list[str]] = {}

        try:
            if candidate_entities is None:
                doc_entities_data = await self.full_entities.get_by_id(doc_id)
                candidate_entities = (
                    doc_entities_data["entity_names"]
                    if doc_entities_data and "entity_names" in doc_entities_data
                    else []
                )
            if candidate_relations is None:
                doc_relations_data = await self.full_relations.get_by_id(doc_id)
                candidate_relations = (
                    doc_relations_data["relation_pairs"]
                    if doc_relations_data and "relation_pairs" in doc_relations_data
                    else []
                )

            # Union the anchor-independent extras (order-preserving dedup).
            if extra_candidate_entities:
                merged_entities = list(candidate_entities)
                seen_names = set(merged_entities)
                for name in extra_candidate_entities:
                    if name not in seen_names:
                        seen_names.add(name)
                        merged_entities.append(name)
                candidate_entities = merged_entities
            if extra_candidate_relations:
                merged_relations = list(candidate_relations)
                seen_pairs = {tuple(pair) for pair in merged_relations}
                for pair in extra_candidate_relations:
                    if tuple(pair) not in seen_pairs:
                        seen_pairs.add(tuple(pair))
                        merged_relations.append(pair)
                candidate_relations = merged_relations

            affected_nodes: list[dict[str, Any]] = []
            affected_edges: list[dict[str, Any]] = []

            if candidate_entities:
                nodes_dict = await self.chunk_entity_relation_graph.get_nodes_batch(
                    list(candidate_entities)
                )
                for entity_name in candidate_entities:
                    node_data = nodes_dict.get(entity_name)
                    # Absent graph node: the candidate was prewritten but the
                    # mutation never landed (or a previous purge already
                    # removed it) — an idempotent no-op, not an error.
                    if node_data:
                        if "id" not in node_data:
                            node_data["id"] = entity_name
                        affected_nodes.append(node_data)

            if candidate_relations:
                edge_pairs_dicts = [
                    {"src": pair[0], "tgt": pair[1]} for pair in candidate_relations
                ]
                edges_dict = await self.chunk_entity_relation_graph.get_edges_batch(
                    edge_pairs_dicts
                )
                for pair in candidate_relations:
                    src, tgt = pair[0], pair[1]
                    edge_data = edges_dict.get((src, tgt))
                    if edge_data:
                        if "source" not in edge_data:
                            edge_data["source"] = src
                        if "target" not in edge_data:
                            edge_data["target"] = tgt
                        affected_edges.append(edge_data)
        except Exception as e:
            logger.error(
                f"[purge] Failed to analyze affected graph elements for {doc_id}: {e}"
            )
            raise _PurgeStageError(
                f"Failed to analyze graph dependencies: {e}",
                stage="analyze_graph_dependencies",
            ) from e

        # ---- 2. Classify entities/relations into delete vs rebuild ----
        try:
            for node_data in affected_nodes:
                node_label = node_data.get("entity_id")
                if not node_label:
                    continue

                existing_sources: list[str] = []
                graph_sources: list[str] = []
                if self.entity_chunks:
                    stored_chunks = await self.entity_chunks.get_by_id(node_label)
                    if stored_chunks and isinstance(stored_chunks, dict):
                        existing_sources = [
                            chunk_id
                            for chunk_id in stored_chunks.get("chunk_ids", [])
                            if chunk_id
                        ]

                if node_data.get("source_id"):
                    graph_sources = [
                        chunk_id
                        for chunk_id in node_data["source_id"].split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]

                if not existing_sources:
                    existing_sources = graph_sources

                if not existing_sources:
                    entities_to_delete.add(node_label)
                    entity_chunk_updates[node_label] = []
                    continue

                remaining_sources = subtract_source_ids(existing_sources, chunk_ids)
                graph_references_deleted_chunks = bool(
                    graph_sources and set(graph_sources) & chunk_ids_set
                )

                if not remaining_sources:
                    entities_to_delete.add(node_label)
                    entity_chunk_updates[node_label] = []
                elif (
                    remaining_sources != existing_sources
                    or graph_references_deleted_chunks
                    or rebuild_policy == "rollback"
                ):
                    entities_to_rebuild[node_label] = remaining_sources
                    entity_chunk_updates[node_label] = remaining_sources

            async with pipeline_status_lock:
                log_message = (
                    f"[purge] {doc_id}: {len(entities_to_rebuild)} entity(ies) "
                    f"to rebuild, {len(entities_to_delete)} to delete"
                )
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                append_pipeline_history(pipeline_status, log_message)

            for edge_data in affected_edges:
                src = edge_data.get("source")
                tgt = edge_data.get("target")
                if not src or not tgt or "source_id" not in edge_data:
                    continue

                edge_tuple = tuple(sorted((src, tgt)))
                if (
                    edge_tuple in relationships_to_delete
                    or edge_tuple in relationships_to_rebuild
                ):
                    continue

                existing_sources = []
                graph_sources = []
                if self.relation_chunks:
                    storage_key = make_relation_chunk_key(src, tgt)
                    stored_chunks = await self.relation_chunks.get_by_id(storage_key)
                    if stored_chunks and isinstance(stored_chunks, dict):
                        existing_sources = [
                            chunk_id
                            for chunk_id in stored_chunks.get("chunk_ids", [])
                            if chunk_id
                        ]

                if edge_data.get("source_id"):
                    graph_sources = [
                        chunk_id
                        for chunk_id in edge_data["source_id"].split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]

                if not existing_sources:
                    existing_sources = graph_sources

                if not existing_sources:
                    relationships_to_delete.add(edge_tuple)
                    relation_chunk_updates[edge_tuple] = []
                    continue

                remaining_sources = subtract_source_ids(existing_sources, chunk_ids)
                graph_references_deleted_chunks = bool(
                    graph_sources and set(graph_sources) & chunk_ids_set
                )

                if not remaining_sources:
                    relationships_to_delete.add(edge_tuple)
                    relation_chunk_updates[edge_tuple] = []
                elif (
                    remaining_sources != existing_sources
                    or graph_references_deleted_chunks
                    or rebuild_policy == "rollback"
                ):
                    relationships_to_rebuild[edge_tuple] = remaining_sources
                    relation_chunk_updates[edge_tuple] = remaining_sources

            async with pipeline_status_lock:
                log_message = (
                    f"[purge] {doc_id}: {len(relationships_to_rebuild)} relation(s) "
                    f"to rebuild, {len(relationships_to_delete)} to delete"
                )
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                append_pipeline_history(pipeline_status, log_message)

            # Update entity/relation chunk-tracking with the remaining sources.
            current_time = int(time.time())
            if entity_chunk_updates and self.entity_chunks:
                entity_upsert_payload = {}
                for entity_name, remaining in entity_chunk_updates.items():
                    if not remaining:
                        continue
                    entity_upsert_payload[entity_name] = {
                        "chunk_ids": remaining,
                        "count": len(remaining),
                        "updated_at": current_time,
                    }
                if entity_upsert_payload:
                    await self.entity_chunks.upsert(entity_upsert_payload)

            if relation_chunk_updates and self.relation_chunks:
                relation_upsert_payload = {}
                for edge_tuple, remaining in relation_chunk_updates.items():
                    if not remaining:
                        continue
                    storage_key = make_relation_chunk_key(*edge_tuple)
                    relation_upsert_payload[storage_key] = {
                        "chunk_ids": remaining,
                        "count": len(remaining),
                        "updated_at": current_time,
                    }
                if relation_upsert_payload:
                    await self.relation_chunks.upsert(relation_upsert_payload)
        except Exception as e:
            logger.error(
                f"[purge] Failed to process graph analysis results for {doc_id}: {e}"
            )
            raise _PurgeStageError(
                f"Failed to process graph dependencies: {e}",
                stage="update_chunk_tracking",
            ) from e

        # ---- 3. Delete relationships with no remaining sources ----
        if relationships_to_delete:
            try:
                rel_ids_to_delete = []
                for src, tgt in relationships_to_delete:
                    rel_ids_to_delete.extend(
                        [
                            compute_mdhash_id(src + tgt, prefix="rel-"),
                            compute_mdhash_id(tgt + src, prefix="rel-"),
                        ]
                    )
                await self.relationships_vdb.delete(rel_ids_to_delete)
                await self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                if self.relation_chunks:
                    relation_storage_keys = [
                        make_relation_chunk_key(src, tgt)
                        for src, tgt in relationships_to_delete
                    ]
                    await self.relation_chunks.delete(relation_storage_keys)
                async with pipeline_status_lock:
                    log_message = (
                        f"[purge] {doc_id}: deleted "
                        f"{len(relationships_to_delete)} relation(s)"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    append_pipeline_history(pipeline_status, log_message)
            except Exception as e:
                logger.error(
                    f"[purge] Failed to delete relationships for {doc_id}: {e}"
                )
                raise _PurgeStageError(
                    f"Failed to delete relationships: {e}",
                    stage="delete_relationships",
                ) from e

        # ---- 4. Delete entities with no remaining sources ----
        if entities_to_delete:
            try:
                nodes_edges_dict = (
                    await self.chunk_entity_relation_graph.get_nodes_edges_batch(
                        list(entities_to_delete)
                    )
                )

                edges_to_delete: set[tuple[str, str]] = set()
                for entity, edges in nodes_edges_dict.items():
                    if edges:
                        for src, tgt in edges:
                            edges_to_delete.add(tuple(sorted((src, tgt))))

                if edges_to_delete:
                    rel_ids_to_delete = []
                    for src, tgt in edges_to_delete:
                        rel_ids_to_delete.extend(
                            [
                                compute_mdhash_id(src + tgt, prefix="rel-"),
                                compute_mdhash_id(tgt + src, prefix="rel-"),
                            ]
                        )
                    await self.relationships_vdb.delete(rel_ids_to_delete)
                    if self.relation_chunks:
                        relation_storage_keys = [
                            make_relation_chunk_key(src, tgt)
                            for src, tgt in edges_to_delete
                        ]
                        await self.relation_chunks.delete(relation_storage_keys)
                    logger.info(
                        f"[purge] {doc_id}: cleaned {len(edges_to_delete)} residual "
                        f"edge(s) from VDB and chunk-tracking storage"
                    )

                await self.chunk_entity_relation_graph.remove_nodes(
                    list(entities_to_delete)
                )

                entity_vdb_ids = [
                    compute_mdhash_id(entity, prefix="ent-")
                    for entity in entities_to_delete
                ]
                await self.entities_vdb.delete(entity_vdb_ids)

                if self.entity_chunks:
                    await self.entity_chunks.delete(list(entities_to_delete))

                async with pipeline_status_lock:
                    log_message = (
                        f"[purge] {doc_id}: deleted "
                        f"{len(entities_to_delete)} entity(ies)"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    append_pipeline_history(pipeline_status, log_message)
            except Exception as e:
                logger.error(f"[purge] Failed to delete entities for {doc_id}: {e}")
                raise _PurgeStageError(
                    f"Failed to delete entities: {e}", stage="delete_entities"
                ) from e

        # ---- 5. Persist pre-rebuild changes ----
        # Use plain _insert_done (no discard-on-failure): the pending buffer
        # here holds DELETES, which are not regenerable by reprocessing. On a
        # flush failure they must stay buffered for a later retry, not be
        # discarded (see _insert_done_with_cleanup docstring).
        try:
            await self._insert_done()
        except Exception as e:
            logger.error(f"[purge] Failed to persist pre-rebuild changes: {e}")
            raise _PurgeStageError(
                f"Failed to persist pre-rebuild changes: {e}",
                stage="persist_pre_rebuild_changes",
            ) from e

        # ---- 6. Rebuild entities/relations that still have remaining sources ----
        if entities_to_rebuild or relationships_to_rebuild:
            try:
                rebuild_report = await rebuild_knowledge_from_chunks(
                    entities_to_rebuild=entities_to_rebuild,
                    relationships_to_rebuild=relationships_to_rebuild,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entities_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    text_chunks_storage=self.text_chunks,
                    llm_response_cache=self.llm_response_cache,
                    global_config=self._build_global_config(),
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    entity_chunks_storage=self.entity_chunks,
                    relation_chunks_storage=self.relation_chunks,
                    rebuild_policy=rebuild_policy,
                )
            except Exception as e:
                logger.error(f"[purge] Failed to rebuild knowledge from chunks: {e}")
                raise _PurgeStageError(
                    f"Failed to rebuild knowledge graph: {e}",
                    stage="rebuild_knowledge_graph",
                ) from e

            # Persist the rebuilt graph/vector/tracking state before the
            # source chunks disappear — a crash after this flush leaves a
            # fully repaired graph plus still-present chunks (retryable),
            # never repaired-in-memory-only state.
            try:
                await self._insert_done()
            except Exception as e:
                logger.error(f"[purge] Failed to persist rebuilt knowledge: {e}")
                raise _PurgeStageError(
                    f"Failed to persist rebuilt knowledge: {e}",
                    stage="persist_rebuilt_knowledge",
                ) from e

        return rebuild_report

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> DeletionResult:
        """Delete a document and all its related data, including chunks, graph elements.

        This method orchestrates a comprehensive deletion process for a given document ID.
        It ensures that not only the document itself but also all its derived and associated
        data across different storage layers are removed or rebuiled. If entities or relationships
        are partially affected, they will be rebuilded using LLM cached from remaining documents.

        **Concurrency Control Design:**

        This function implements a pipeline-based concurrency control to prevent data corruption:

        1. **Single Document Deletion** (when WE acquire pipeline):
           - Sets job_name to "Single document deletion" (NOT starting with "deleting")
           - Prevents other adelete_by_doc_id calls from running concurrently
           - Ensures exclusive access to graph operations for this deletion

        2. **Batch Document Deletion** (when background_delete_documents acquires pipeline):
           - Sets job_name to "Deleting {N} Documents" (starts with "deleting")
           - Allows multiple adelete_by_doc_id calls to join the deletion queue
           - Each call validates the job name to ensure it's part of a deletion operation

        The validation logic `if not job_name.startswith("deleting") or "document" not in job_name`
        ensures that:
        - adelete_by_doc_id can only run when pipeline is idle OR during batch deletion
        - Prevents concurrent single deletions that could cause race conditions
        - Rejects operations when pipeline is busy with non-deletion tasks

        Args:
            doc_id (str): The unique identifier of the document to be deleted.
            delete_llm_cache (bool): Whether to delete cached LLM extraction results
                associated with the document. Defaults to False.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", "not_allowed", or "fail".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 403, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
        """
        # Get pipeline status shared data and lock for validation
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Track whether WE acquired the pipeline. ``token`` owns the busy slot so
        # the finally releases it by owner (never clobbering a later holder).
        we_acquired_pipeline = False
        token = uuid.uuid4().hex

        # Initialized BEFORE the try so the finally can always read them, even
        # when the acquire itself (or the post-acquire status logging) raises.
        deletion_operations_started = False
        deletion_fully_completed = False
        in_final_delete_stage = False
        original_exception = None
        doc_llm_cache_ids: list[str] = []
        deletion_stage = "initializing"
        doc_status_data: dict[str, Any] | None = None
        file_path: str | None = None

        # Acquire (or join a batch delete) INSIDE the same handoff-aware
        # try/finally as the deletion itself: any escape after the slot is
        # stamped — a cancellation or RPC failure at the acquire/logging lock
        # exit included — reaches the same owner-checked exit decision as the
        # main flow (probe ``has_work()`` → handoff or release).  A concurrent
        # enqueue can commit mailbox work the instant the reservation lock is
        # released, so an unconditional early release in that window would
        # strand it (PR #3467 review round 2).
        try:
            # Pre-arm ownership before acquire so cancellation after the atomic
            # update cannot leak this token's reservation (the finally is
            # owner-checked; a stamp that never happened makes it a no-op).
            we_acquired_pipeline = True
            reservation = await acquire_reservation(
                pipeline_status,
                pipeline_status_lock,
                owner_key="busy_owner",
                owner=token,
                owner_kind="delete",
                flags={
                    "busy": True,
                    "operation_record": {"kind": "delete", "doc_id": doc_id},
                    "job_name": "Single document deletion",
                    "job_start": datetime.now(timezone.utc).isoformat(),
                    "docs": 1,
                    "batchs": 1,
                    "cur_batch": 0,
                    "cancellation_requested": False,
                    "latest_message": f"Starting deletion for document: {doc_id}",
                },
                reject_when=(("busy", "Pipeline is busy with another operation."),),
            )
            if reservation.acquired:
                history = (reservation.snapshot or {}).get("history_messages")
                if history is not None:
                    history[:] = [f"Starting deletion for document: {doc_id}"]
            else:
                we_acquired_pipeline = False
                if (
                    reservation.conflict
                    is PipelineReservationConflict.RECOVERY_REQUIRED
                ):
                    return DeletionResult(
                        status="not_allowed",
                        doc_id=doc_id,
                        message=reservation.message or "Pipeline recovery is required.",
                        status_code=503,
                        file_path=None,
                    )
                # Pipeline already busy - verify it's a batch deletion job.
                snapshot = reservation.snapshot or {}
                job_name = str(snapshot.get("job_name", "")).lower()
                if not job_name.startswith("deleting") or "document" not in job_name:
                    return DeletionResult(
                        status="not_allowed",
                        doc_id=doc_id,
                        message=(
                            "Deletion not allowed: current job "
                            f"'{snapshot.get('job_name')}' is not a document deletion job"
                        ),
                        status_code=403,
                        file_path=None,
                    )
                # Pipeline is busy with batch deletion - join without acquiring.

            async with pipeline_status_lock:
                log_message = f"Starting deletion process for document {doc_id}"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                append_pipeline_history(pipeline_status, log_message)

            # 1. Get the document status and related data
            doc_status_data = await self.doc_status.get_by_id(doc_id)
            if not doc_status_data:
                logger.warning(f"Document {doc_id} not found")
                return DeletionResult(
                    status="not_found",
                    doc_id=doc_id,
                    message=f"Document {doc_id} not found.",
                    status_code=404,
                    file_path="",
                )
            file_path = doc_status_data.get("file_path")

            # Check document status and log warning for non-completed documents
            raw_status = doc_status_data.get("status")
            try:
                doc_status = DocStatus(raw_status)
            except ValueError:
                doc_status = raw_status

            if doc_status != DocStatus.PROCESSED:
                if doc_status == DocStatus.PENDING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                    )
                elif doc_status == DocStatus.PROCESSING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                    )
                elif doc_status == DocStatus.PREPROCESSED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PREPROCESSED)"
                    )
                elif doc_status == DocStatus.FAILED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                    )
                else:
                    status_text = (
                        doc_status.value
                        if isinstance(doc_status, DocStatus)
                        else str(doc_status)
                    )
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: {status_text})"
                    )
                logger.info(warning_msg)
                # Update pipeline status for monitoring
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = warning_msg
                    append_pipeline_history(pipeline_status, warning_msg)

            # 2. Get chunk IDs from document status
            metadata = doc_status_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata_cache_ids = normalize_string_list(
                metadata.get("deletion_llm_cache_ids", []),
                context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
            )
            # Parse-stage LLM cache keys (docx smart_heading) ride their own
            # metadata key — no chunk llm_cache_list exists at parse time.
            # Merge them into the same deletion pool (order-preserving dedup);
            # a failed delete then persists the union back into
            # deletion_llm_cache_ids for the retry path.
            metadata_cache_ids = list(
                dict.fromkeys(
                    metadata_cache_ids
                    + normalize_string_list(
                        metadata.get("smartheading_llm_cache_ids", []),
                        context=f"doc {doc_id} metadata.smartheading_llm_cache_ids",
                    )
                )
            )
            # Order-preserving dedup so chunk_ids stays a list and satisfies the
            # storage delete contract (``delete(ids: list[str])``); a set view is
            # built below for membership/intersection checks. Staged chunks of
            # an unfinished custom-chunk operation (issue #3400 Phase 3) live
            # only in the journal until commit unions them into chunks_list —
            # include them so deleting the document also cleans the staging.
            journal = metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY)
            journal_chunk_ids = (
                normalize_string_list(
                    journal.get("chunk_ids", []),
                    context=f"doc {doc_id} custom_chunk_patch.chunk_ids",
                )
                if isinstance(journal, dict)
                else []
            )
            chunk_ids = list(
                dict.fromkeys(
                    normalize_string_list(
                        doc_status_data.get("chunks_list", []),
                        context=f"doc {doc_id} chunks_list",
                    )
                    + journal_chunk_ids
                )
            )

            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")

                # Fail closed before touching anything (issue #3400). This
                # branch used to delete doc_status + full_docs and report
                # success WITHOUT looking at the graph at all — so a document
                # whose anchors still named live entities was reported deleted
                # while those entities stayed behind, now unattributable
                # (removing doc_status is what breaks the provenance chain).
                #
                # Run the same primitive as the chunk-backed path: with an
                # empty chunk set it refuses exactly that case, and otherwise
                # cleans up a genuinely empty stub — including its two empty
                # anchor rows, which this branch used to leave behind keyed to
                # a document that no longer exists.
                try:
                    deletion_stage = "validate_recovery_anchors"
                    await self._purge_kg_contributions(
                        doc_id,
                        chunk_ids,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                except _PurgeStageError as purge_error:
                    deletion_stage = purge_error.purge_stage
                    raise

                # Mark that deletion operations have started
                deletion_operations_started = True

                # A prior failed deletion may have collected LLM cache IDs before the
                # chunks were removed. If delete_llm_cache is requested and persisted IDs
                # exist, clean them up now before removing the doc/status entries.
                if delete_llm_cache and metadata_cache_ids:
                    if not self.llm_response_cache:
                        no_cache_msg = (
                            f"Cannot delete LLM cache for document {doc_id}: "
                            "cache storage is unavailable"
                        )
                        logger.error(no_cache_msg)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = no_cache_msg
                            append_pipeline_history(pipeline_status, no_cache_msg)
                        raise Exception(no_cache_msg)
                    try:
                        deletion_stage = "delete_llm_cache"
                        await self.llm_response_cache.delete(metadata_cache_ids)
                        remaining_cache_ids = await self._get_existing_llm_cache_ids(
                            metadata_cache_ids
                        )
                        if remaining_cache_ids:
                            raise Exception(
                                f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                            )
                        logger.info(
                            "Cleaned up %d LLM cache entries from prior attempt for document %s",
                            len(metadata_cache_ids),
                            doc_id,
                        )
                    except Exception as cache_err:
                        raise Exception(
                            f"Failed to delete LLM cache for document {doc_id}: {cache_err}"
                        ) from cache_err

                try:
                    # Still need to delete the doc status and full doc.
                    # Delete doc_status first: if full_docs.delete fails on retry, the
                    # doc_status record is already gone so the retry finds no record and
                    # treats the document as already deleted rather than creating a zombie.
                    deletion_stage = "delete_doc_entries"
                    await self.doc_status.delete([doc_id])
                    await self.full_docs.delete([doc_id])
                except Exception as e:
                    logger.error(
                        f"Failed to delete document {doc_id} with no chunks: {e}"
                    )
                    raise Exception(f"Failed to delete document entry: {e}") from e

                async with pipeline_status_lock:
                    log_message = (
                        f"Document deleted without associated chunks: {doc_id}"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    append_pipeline_history(pipeline_status, log_message)

                deletion_fully_completed = True
                return DeletionResult(
                    status="success",
                    doc_id=doc_id,
                    message=log_message,
                    status_code=200,
                    file_path=file_path,
                )

            # Mark that deletion operations have started
            deletion_operations_started = True

            if chunk_ids:
                # Always collect/persist cache IDs for chunk-backed documents, even when
                # this call does not request cache deletion. If a delete fails after the
                # chunks/graph have already been removed, a later retry may turn on
                # delete_llm_cache=True, and doc_status metadata is then the only durable
                # place left to recover the cache keys for cleanup.
                deletion_stage = "collect_llm_cache"
                doc_llm_cache_ids = list(metadata_cache_ids)
                if not self.text_chunks:
                    logger.info(
                        "Skipping LLM cache id collection for document %s because text chunk storage is unavailable",
                        doc_id,
                    )
                else:
                    try:
                        chunk_data_list = await self.text_chunks.get_by_ids(chunk_ids)
                        seen_cache_ids: set[str] = set(doc_llm_cache_ids)
                        for chunk_data in chunk_data_list:
                            if not chunk_data or not isinstance(chunk_data, dict):
                                continue
                            cache_ids = chunk_data.get("llm_cache_list", [])
                            if not isinstance(cache_ids, list):
                                continue
                            for cache_id in cache_ids:
                                if (
                                    isinstance(cache_id, str)
                                    and cache_id
                                    and cache_id not in seen_cache_ids
                                ):
                                    doc_llm_cache_ids.append(cache_id)
                                    seen_cache_ids.add(cache_id)
                    except Exception as cache_collect_error:
                        logger.error(
                            "Failed to collect LLM cache ids for document %s: %s",
                            doc_id,
                            cache_collect_error,
                        )
                        raise Exception(
                            f"Failed to collect LLM cache ids for document {doc_id}: {cache_collect_error}"
                        ) from cache_collect_error

                if doc_llm_cache_ids:
                    try:
                        doc_status_data = await self._update_delete_retry_state(
                            doc_id,
                            doc_status_data,
                            deletion_stage=deletion_stage,
                            doc_llm_cache_ids=doc_llm_cache_ids,
                            failed=False,
                        )
                    except Exception as status_write_error:
                        logger.error(
                            "Failed to persist LLM cache IDs for document %s to retry state: %s",
                            doc_id,
                            status_write_error,
                        )
                        # Describe whether this is a fresh attempt or a retry so
                        # operators can tell whether prior partial deletions exist.
                        attempt_context = (
                            "retry — prior partial deletions may exist"
                            if metadata_cache_ids
                            else "deletion not yet started"
                        )
                        raise Exception(
                            f"Failed to persist LLM cache IDs for document {doc_id} "
                            f"({attempt_context}): {status_write_error}"
                        ) from status_write_error
                    logger.info(
                        "Collected %d LLM cache entries for document %s",
                        len(doc_llm_cache_ids),
                        doc_id,
                    )
                else:
                    logger.info("No LLM cache entries found for document %s", doc_id)

            # 4. Purge every KG contribution of this document, then its chunks.
            #
            # Delegated to the shared primitive rather than reimplemented here
            # (issue #3400). Two things follow from that:
            #
            #  * The primitive refuses to start without a recovery proof, so a
            #    document whose anchors were lost fails closed with a 409
            #    instead of silently skipping graph cleanup and orphaning its
            #    entities. Candidates anchored ONLY in an unfinished
            #    custom-chunk patch journal are covered too — the primitive
            #    reads that journal itself and unions them in, since a
            #    patch-mode merge writes no anchor rows of its own — and this
            #    path already merged the journal's staged chunk ids into
            #    ``chunk_ids`` above.
            #  * Destructive ordering is the primitive's (correct) one. The
            #    inline implementation this replaces deleted the chunks BEFORE
            #    repairing the graph, which is the window where a mid-delete
            #    failure left graph objects pointing at chunks that no longer
            #    existed.
            #
            # The failing step travels on the exception as ``purge_stage`` so
            # ``deletion_failure_stage`` keeps reporting the same granularity
            # it always did.
            try:
                deletion_stage = "validate_recovery_anchors"
                await self._purge_kg_contributions(
                    doc_id,
                    chunk_ids,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                )
            except _PurgeStageError as purge_error:
                deletion_stage = purge_error.purge_stage
                raise

            # 9. Delete LLM cache while the document status still exists so a failure
            # remains retryable via the same doc_id.
            log_message = f"Document {doc_id} successfully deleted"
            if delete_llm_cache and doc_llm_cache_ids:
                if not self.llm_response_cache:
                    log_message = (
                        f"Cannot delete LLM cache for document {doc_id}: "
                        "cache storage is unavailable"
                    )
                    logger.error(log_message)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        append_pipeline_history(pipeline_status, log_message)
                    raise Exception(log_message)
                try:
                    deletion_stage = "delete_llm_cache"
                    await self.llm_response_cache.delete(doc_llm_cache_ids)
                    # Some storage implementations do not raise on delete errors and
                    # instead only log internally, so confirm the cache entries are
                    # actually gone before deleting the document/status records.
                    remaining_cache_ids = await self._get_existing_llm_cache_ids(
                        doc_llm_cache_ids
                    )
                    if remaining_cache_ids:
                        doc_llm_cache_ids = remaining_cache_ids
                        raise Exception(
                            f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                        )
                    cache_log_message = f"Successfully deleted {len(doc_llm_cache_ids)} LLM cache entries for document {doc_id}"
                    logger.info(cache_log_message)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = cache_log_message
                        append_pipeline_history(pipeline_status, cache_log_message)
                    log_message = cache_log_message
                except Exception as cache_delete_error:
                    log_message = (
                        f"Failed to delete LLM cache for document {doc_id}: "
                        f"{cache_delete_error}"
                    )
                    logger.error(log_message)
                    logger.error(traceback.format_exc())
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        append_pipeline_history(pipeline_status, log_message)
                    raise Exception(log_message) from cache_delete_error

            # 10. (The recovery anchor rows were deleted by the purge above —
            # LAST within that operation, and only once its journal recorded
            # that the chunks were gone. Deleting them again here would be
            # redundant, and doing it BEFORE the LLM cache step, as this
            # function used to, removed the retry's recovery proof while work
            # remained.)

            # 11. Delete original document and status.
            # doc_status is deleted first so that if full_docs.delete fails, a retry
            # finds no doc_status record and treats the document as already gone,
            # rather than finding a doc_status that points to a missing full_docs entry.
            try:
                deletion_stage = "delete_doc_entries"
                in_final_delete_stage = True
                await self.doc_status.delete([doc_id])
                await self.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

            deletion_fully_completed = True
            return DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )

        except (RecoveryAnchorMissingError, KGPurgeOperationConflictError) as e:
            # A refused PRECONDITION, not a failed deletion: the purge raised
            # before its first write, so nothing was deleted and the document
            # is exactly as it was. Surfaced as 409 rather than 500 because
            # retrying unchanged will refuse again — the operator has to run
            # the integrity audit first. The message is written to be
            # client-safe and names that remedy.
            original_exception = e
            error_message = str(e)
            logger.error(f"Refusing to delete document {doc_id}: {e}")
            try:
                if doc_status_data is not None:
                    doc_status_data = await self._update_delete_retry_state(
                        doc_id,
                        doc_status_data,
                        deletion_stage="validate_recovery_anchors",
                        doc_llm_cache_ids=doc_llm_cache_ids,
                        error_message=error_message,
                        failed=True,
                    )
            except Exception as status_update_error:
                logger.error(
                    "Failed to record recovery-anchor refusal for document %s: %s",
                    doc_id,
                    status_update_error,
                )
            return DeletionResult(
                status="fail",
                doc_id=doc_id,
                message=error_message,
                status_code=409,
                file_path=file_path,
            )

        except Exception as e:
            original_exception = e
            error_message = f"Error while deleting document {doc_id}: {e}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            try:
                # Do not attempt to write retry state if doc_status was already deleted:
                # upsert would re-create the record as a zombie. All earlier stages still
                # have doc_status intact and can safely update it, even if some chunk/graph
                # data has already been removed.
                if doc_status_data is not None and not in_final_delete_stage:
                    doc_status_data = await self._update_delete_retry_state(
                        doc_id,
                        doc_status_data,
                        deletion_stage=deletion_stage,
                        doc_llm_cache_ids=doc_llm_cache_ids,
                        error_message=error_message,
                        failed=True,
                    )
            except Exception as status_update_error:
                logger.error(
                    "Failed to update deletion retry state for document %s: %s",
                    doc_id,
                    status_update_error,
                )
                logger.error(traceback.format_exc())
                error_message = (
                    f"{error_message}. Additionally, failed to persist retry state: "
                    f"{status_update_error}. Manual cleanup may be required."
                )
            return DeletionResult(
                status="fail",
                doc_id=doc_id,
                message=error_message,
                status_code=500,
                file_path=file_path,
            )

        finally:
            # A direct SDK delete holds ``busy`` WITHOUT ``destructive_busy``,
            # so concurrent enqueues are permitted the whole time: an
            # ``ainsert`` lands its PENDING row + document message, and its
            # busy-refused process call arms the auto-rescan flag.  Those
            # signals live only in the mailbox — releasing without a handoff
            # would strand them until an unrelated trigger.  Mirror the
            # custom-chunks exit: probe ``has_work()`` in the owner-checked
            # critical section and hand the slot to a processing run instead
            # of releasing.  A resolve/probe failure fails TOWARD handoff (the
            # driven run re-probes and its finally releases — never a wedge).
            #
            # This await precedes the cancellation-resistant release — safe
            # only because get_pipeline_ingress is suspension-free by contract
            # (see its docstring).  Only resolved when WE own the slot: a
            # joined batch delete leaves the handoff to the batch's own exit.
            handoff_ingress = None
            if we_acquired_pipeline:
                try:
                    handoff_ingress = await get_pipeline_ingress(self.workspace)
                except Exception as ingress_error:
                    logger.warning(
                        "single-delete exit: pipeline ingress unavailable; "
                        "failing toward handoff so mailbox-only work is not "
                        f"silently deferred: {ingress_error}"
                    )

            def _release_action(status):
                # Owner-checked exit decision (+ cancellation reset +
                # completion log), applied atomically. Runs via
                # with_reservation_lock so it is cancellation-resistant and
                # cannot clobber a later holder.
                completion_msg = f"Deletion process completed for document: {doc_id}"
                updates = {
                    "operation_record": None,
                    "cancellation_requested": False,
                    "latest_message": completion_msg,
                }
                if handoff_ingress is None:
                    ingress_has_work = True  # fail toward handoff
                else:
                    try:
                        ingress_has_work = handoff_ingress.has_work()
                    except Exception as probe_error:
                        ingress_has_work = True  # fail toward handoff
                        logger.warning(
                            "single-delete exit: ingress has_work probe "
                            "failed; handing off so a committed manual retry "
                            "or enqueued document is not silently deferred: "
                            f"{probe_error}"
                        )
                append_pipeline_history(status, completion_msg)
                logger.info(completion_msg)
                if ingress_has_work:
                    status.update(updates)
                    return "handoff"  # keep busy (owner stays ours)
                updates.update({"busy": False, "busy_owner": None})
                status.update(updates)
                return "released"

            def _terminal_release(status):
                # Owner-checked backstop: release the slot we still hold (a
                # no-op when the handoff run — or a later holder — already
                # released it).
                status.update(
                    {
                        "busy": False,
                        "busy_owner": None,
                        "operation_record": None,
                        "cancellation_requested": False,
                    }
                )

            try:
                # ALWAYS ensure persistence if any deletion operations were started
                if deletion_operations_started:
                    # Plain _insert_done: this finally reports the deletion as
                    # successful after logging a persistence error, so discarding
                    # pending DELETES here would drop them with no retry and leave
                    # stale vectors/KV behind. Keep them buffered for a later flush.
                    try:
                        await self._insert_done()
                    except Exception as persistence_error:
                        persistence_error_msg = f"Failed to persist data after deletion attempt for {doc_id}: {persistence_error}"
                        logger.error(persistence_error_msg)
                        logger.error(traceback.format_exc())

                        if deletion_fully_completed:
                            # All deletion stages succeeded; the flush error is a post-cleanup
                            # concern. Do not override the success result already returned.
                            logger.error(
                                "Post-deletion persistence flush failed for %s, "
                                "but deletion completed successfully: %s",
                                doc_id,
                                persistence_error,
                            )
                        elif original_exception is None:
                            # Deletion stages were in-flight but the try-block return was never
                            # reached; treat the persistence failure as the primary error.
                            return DeletionResult(
                                status="fail",
                                doc_id=doc_id,
                                message=f"Deletion completed but failed to persist changes: {persistence_error}",
                                status_code=500,
                                file_path=file_path,
                            )
                        # If there was an original exception, log the persistence error but
                        # don't override it — the original error result was already returned.
                else:
                    logger.debug(
                        f"No deletion operations were started for document {doc_id}, skipping persistence"
                    )
            finally:
                # Exit the slot even if the flush above was cancelled or
                # returned — owner-checked + cancellation-resistant, so a cancel
                # during _insert_done can never leave the pipeline wedged.
                if we_acquired_pipeline:
                    try:
                        decision = await with_reservation_lock(
                            pipeline_status,
                            pipeline_status_lock,
                            owner_key="busy_owner",
                            token=token,
                            action=_release_action,
                        )
                        if decision == "handoff":
                            # busy is still True (ours); the handoff run takes
                            # over the slot and releases it at its own exit.
                            # Pass our token so it owns and releases the SAME
                            # reservation. A pending cancellation makes this
                            # await re-raise immediately, deferring the drain
                            # to the next trigger — the backstop below still
                            # releases the slot.
                            try:
                                await self.apipeline_process_enqueue_documents(
                                    _holding_busy=True, token=token
                                )
                            except Exception as drain_error:
                                logger.warning(
                                    "Failed to process queued documents handed "
                                    "off from a single-document delete: "
                                    f"{drain_error}"
                                )
                    finally:
                        await with_reservation_lock(
                            pipeline_status,
                            pipeline_status_lock,
                            owner_key="busy_owner",
                            token=token,
                            action=_terminal_release,
                        )

    async def _raise_if_recovery_required(self) -> None:
        """Refuse a graph/data mutation while the workspace is fenced for
        recovery (a worker died mid custom_chunks/delete/clear, possibly leaving
        storage partially committed).

        The REST graph-edit routes go through ``check_pipeline_busy_or_raise``,
        which already refuses a fenced workspace, but a DIRECT SDK caller of the
        ``a{create,edit,delete,merge}_*`` methods bypasses that. This is the
        SDK-level fence so "refuse all mutations while fenced" holds on both
        paths. It intentionally does NOT check ``busy`` (SDK graph edits may run
        concurrently with the pipeline, guarded by per-entity keyed locks). No-op
        when pipeline_status was never initialised (test rigs).
        """
        from lightrag.exceptions import PipelineNotInitializedError

        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=self.workspace
            )
        except PipelineNotInitializedError:
            return
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )
        result = await check_pipeline_status_mutation(
            pipeline_status, pipeline_status_lock
        )
        if not result.acquired:
            raise RuntimeError(result.message)

    async def adelete_by_entity(self, entity_name: str) -> DeletionResult:
        """Asynchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import adelete_by_entity

        return await adelete_by_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
        )

    def delete_by_entity(self, entity_name: str) -> DeletionResult:
        """Synchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        return _run_sync(
            lambda: self.adelete_by_entity(entity_name),
            sync_name="delete_by_entity",
            async_name="adelete_by_entity",
            owning_loop=self._owning_loop,
        )

    async def adelete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """Asynchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import adelete_by_relation

        return await adelete_by_relation(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            source_entity,
            target_entity,
        )

    def delete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """Synchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        return _run_sync(
            lambda: self.adelete_by_relation(source_entity, target_entity),
            sync_name="delete_by_relation",
            async_name="adelete_by_relation",
            owning_loop=self._owning_loop,
        )

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def aget_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by track_id

        Args:
            track_id: The tracking ID to search for

        Returns:
            Dict with document id as keys and document status as values
        """
        return await self.doc_status.get_docs_by_track_id(track_id)

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity.

        Args:
            entity_name: Name of the entity to look up.
            include_vector_data: DEPRECATED. Attaches a ``vector_data`` field
                read from the entity vector store. The vector store no longer
                returns the embedding, so this payload is derived from the
                graph node (the authoritative source) and duplicates
                ``graph_data``; the only signal it adds is whether a VDB record
                exists at all. No LightRAG code path sets this — it is kept for
                backward compatibility only and may be removed in a future
                release. For graph/VDB consistency, use the offline
                ``lightrag-rebuild-vdb`` check instead.

        Returns:
            ``{"entity_name", "source_id", "graph_data"}`` (plus a redundant
            ``"vector_data"`` when ``include_vector_data`` is True).
        """
        from lightrag.utils_graph import get_entity_info

        return await get_entity_info(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            entity_name,
            include_vector_data,
        )

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship.

        Args:
            src_entity: Source entity name.
            tgt_entity: Target entity name.
            include_vector_data: DEPRECATED. Attaches a ``vector_data`` field
                read from the relationship vector store. The vector store no
                longer returns the embedding, so this payload is derived from
                the graph edge (the authoritative source) and duplicates
                ``graph_data``; the only signal it adds is whether a VDB record
                exists at all. No LightRAG code path sets this — it is kept for
                backward compatibility only and may be removed in a future
                release. For graph/VDB consistency, use the offline
                ``lightrag-rebuild-vdb`` check instead.

        Returns:
            ``{"src_entity", "tgt_entity", "source_id", "graph_data"}`` (plus a
            redundant ``"vector_data"`` when ``include_vector_data`` is True).
        """
        from lightrag.utils_graph import get_relation_info

        return await get_relation_info(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            src_entity,
            tgt_entity,
            include_vector_data,
        )

    async def aedit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, str],
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.
        Also synchronizes entity_chunks_storage and relation_chunks_storage to track chunk references.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True
            allow_merge: Whether to merge into an existing entity when renaming to an existing name

        Returns:
            Dictionary containing updated entity information
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import aedit_entity

        return await aedit_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            updated_data,
            allow_rename,
            allow_merge,
            self.entity_chunks,
            self.relation_chunks,
        )

    def edit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, str],
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        return _run_sync(
            lambda: self.aedit_entity(
                entity_name, updated_data, allow_rename, allow_merge
            ),
            sync_name="edit_entity",
            async_name="aedit_entity",
            owning_loop=self._owning_loop,
        )

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.
        Also synchronizes the relation_chunks_storage to track which chunks reference this relation.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import aedit_relation

        return await aedit_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            updated_data,
            self.relation_chunks,
        )

    def edit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        return _run_sync(
            lambda: self.aedit_relation(source_entity, target_entity, updated_data),
            sync_name="edit_relation",
            async_name="aedit_relation",
            owning_loop=self._owning_loop,
        )

    async def acreate_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new entity.

        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import acreate_entity

        return await acreate_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            entity_data,
        )

    def create_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        return _run_sync(
            lambda: self.acreate_entity(entity_name, entity_data),
            sync_name="create_entity",
            async_name="acreate_entity",
            owning_loop=self._owning_loop,
        )

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new relation between entities.

        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import acreate_relation

        return await acreate_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            relation_data,
        )

    def create_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        return _run_sync(
            lambda: self.acreate_relation(source_entity, target_entity, relation_data),
            sync_name="create_relation",
            async_name="acreate_relation",
            owning_loop=self._owning_loop,
        )

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        await self._raise_if_recovery_required()

        from lightrag.utils_graph import amerge_entities

        return await amerge_entities(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entities,
            target_entity,
            merge_strategy,
            target_entity_data,
            self.entity_chunks,
            self.relation_chunks,
        )

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        return _run_sync(
            lambda: self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            ),
            sync_name="merge_entities",
            async_name="amerge_entities",
            owning_loop=self._owning_loop,
        )

    async def aexport_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Asynchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        from lightrag.utils import aexport_data as utils_aexport_data

        await utils_aexport_data(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            output_path,
            file_format,
            include_vector_data,
        )

    def export_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Synchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        _run_sync(
            lambda: self.aexport_data(output_path, file_format, include_vector_data),
            sync_name="export_data",
            async_name="aexport_data",
            owning_loop=self._owning_loop,
        )


# `addon_params` is declared as an InitVar on the dataclass so it can still be
# passed through LightRAG(addon_params=...). InitVars are not stored as
# instance attributes, which frees the name to be installed here as a property
# that routes reads/writes through the observable `_addon_params` store.
# Declaring it as both a dataclass field and a property is not supported by
# @dataclass, so the property is attached after class creation.
LightRAG.addon_params = property(  # type: ignore[attr-defined]
    LightRAG._get_addon_params,
    LightRAG._set_runtime_addon_params,
)
