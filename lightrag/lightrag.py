from __future__ import annotations

import traceback
import asyncio
import os
import time
import warnings
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
    Iterator,
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
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    verify_storage_implementation,
)


from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_default_workspace,
    set_default_workspace,
    get_namespace_lock,
    get_storage_keyed_lock,
)

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
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
    extract_entities,
    kg_query,
    naive_query,
    rebuild_knowledge_from_chunks,
)
from lightrag.utils_pipeline import normalize_document_file_path
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.exceptions import IndexFlushError
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
    convert_to_user_format,
    logger,
    make_relation_vdb_ids,
    subtract_source_ids,
    make_relation_chunk_key,
    normalize_source_ids_limit_method,
    normalize_string_list,
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

    cosine_threshold: int = field(
        default=get_env_value("COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, int)
    )
    """Cosine threshold of vector DB retrieval for entities, relations and chunks."""

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

    embedding_batch_num: int = field(default=int(os.getenv("EMBEDDING_BATCH_NUM", 10)))
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 8))
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
        default=int(os.getenv("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT))
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
        default=int(os.getenv("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS))
    )
    """Maximum tokens allowed for entity/relation description."""

    summary_context_size: int = field(
        default=int(os.getenv("SUMMARY_CONTEXT_SIZE", DEFAULT_SUMMARY_CONTEXT_SIZE))
    )
    """Maximum number of tokens allowed per LLM response."""

    summary_length_recommended: int = field(
        default=int(
            os.getenv("SUMMARY_LENGTH_RECOMMENDED", DEFAULT_SUMMARY_LENGTH_RECOMMENDED)
        )
    )
    """Recommended length of LLM summary output."""

    llm_model_max_async: int = field(
        default=int(
            os.getenv("MAX_ASYNC_LLM", os.getenv("MAX_ASYNC", DEFAULT_MAX_ASYNC))
        )
    )
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    default_llm_timeout: int = field(
        default=int(os.getenv("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT))
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
        default=int(
            os.getenv(
                "MAX_ASYNC_RERANK",
                os.getenv("MAX_ASYNC_LLM", os.getenv("MAX_ASYNC", DEFAULT_MAX_ASYNC)),
            )
        )
    )
    """Maximum number of concurrent rerank calls.
    Falls back to MAX_ASYNC_LLM when MAX_ASYNC_RERANK is unset
    (MAX_ASYNC is still accepted as a deprecated alias)."""

    default_rerank_timeout: int = field(
        default=int(os.getenv("RERANK_TIMEOUT", DEFAULT_RERANK_TIMEOUT))
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
        default=int(os.getenv("MAX_PARALLEL_INSERT", DEFAULT_MAX_PARALLEL_INSERT))
    )
    """Maximum number of parallel insert operations."""

    max_parallel_parse_native: int = field(
        default=int(
            os.getenv(
                "MAX_PARALLEL_PARSE_NATIVE", str(DEFAULT_MAX_PARALLEL_PARSE_NATIVE)
            )
        )
    )
    max_parallel_parse_mineru: int = field(
        default=int(
            os.getenv(
                "MAX_PARALLEL_PARSE_MINERU", str(DEFAULT_MAX_PARALLEL_PARSE_MINERU)
            )
        )
    )
    max_parallel_parse_docling: int = field(
        default=int(
            os.getenv(
                "MAX_PARALLEL_PARSE_DOCLING", str(DEFAULT_MAX_PARALLEL_PARSE_DOCLING)
            )
        )
    )
    max_parallel_analyze: int = field(
        default=int(
            os.getenv("MAX_PARALLEL_ANALYZE", str(DEFAULT_MAX_PARALLEL_ANALYZE))
        )
    )
    queue_size_parse: int = field(
        default=int(os.getenv("QUEUE_SIZE_PARSE", str(DEFAULT_QUEUE_SIZE_PARSE)))
    )
    queue_size_analyze: int = field(
        default=int(os.getenv("QUEUE_SIZE_ANALYZE", str(DEFAULT_QUEUE_SIZE_ANALYZE)))
    )
    queue_size_insert: int = field(
        default=int(os.getenv("QUEUE_SIZE_INSERT", str(DEFAULT_QUEUE_SIZE_INSERT)))
    )

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
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )

    ollama_server_infos: Optional[OllamaServerInfos] = field(default=None)
    """Configuration for Ollama server information."""

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    def _mark_addon_params_dirty(self) -> None:
        self._addon_params_dirty = True

    def _replace_addon_params(
        self, addon_params: Mapping[str, Any] | None, *, mark_dirty: bool
    ) -> None:
        wrapped = ObservableAddonParams(
            normalize_addon_params(addon_params),
            on_change=self._mark_addon_params_dirty,
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
                chunker_cfg["chunk_token_size"] = int(os.getenv("CHUNK_SIZE", 1200))

        # Per-strategy chunk_overlap_token_size — strategy env (if set)
        # already lives in the sub-dict.  Slots still missing fall back
        # to the legacy ctor field, then CHUNK_OVERLAP_SIZE env.
        if self.chunk_overlap_token_size is not None:
            legacy_overlap_default = self.chunk_overlap_token_size
        else:
            legacy_overlap_default = int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
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
        p_size_raw = os.getenv("CHUNK_P_SIZE")
        chunker_cfg["paragraph_semantic"].setdefault(
            "chunk_token_size",
            int(p_size_raw) if p_size_raw is not None else DEFAULT_CHUNK_P_SIZE,
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
            size_raw = os.getenv(size_env)
            if size_raw is None:
                continue
            sub = chunker_cfg.get(strategy_key)
            if not isinstance(sub, dict):
                sub = {}
                chunker_cfg[strategy_key] = sub
            sub.setdefault("chunk_token_size", int(size_raw))

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
                f"summary_context_size({self.summary_context_size}) should no greater than max_total_tokens({self.max_total_tokens})"
            )
        if self.summary_length_recommended > self.summary_max_tokens:
            logger.warning(
                f"max_total_tokens({self.summary_max_tokens}) should greater than summary_length_recommended({self.summary_length_recommended})"
            )

        if self.rerank_model_func is not None:
            self.rerank_model_func = priority_limit_async_func_call(
                self.rerank_model_max_async,
                llm_timeout=self.default_rerank_timeout,
                queue_name="Rerank func",
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

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("All storage types initialized")

    async def finalize_storages(self):
        """Asynchronously finalize the storages with improved error handling"""
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert(
                input,
                split_by_character,
                split_by_character_only,
                ids,
                file_paths,
                track_id,
            )
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
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert_custom_chunks(full_text, text_chunks, doc_id)
        )

    # TODO: deprecated, use ainsert instead
    async def ainsert_custom_chunks(
        self, full_text: str, text_chunks: list[str], doc_id: str | None = None
    ) -> None:
        update_storage = False
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
            new_docs = {doc_key: {"content": full_text, "file_path": file_path}}

            _add_doc_keys = await self.full_docs.filter_keys({doc_key})
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("This document is already in the storage.")
                return

            update_storage = True
            logger.info(f"Inserting {len(new_docs)} docs")

            inserting_chunks: dict[str, Any] = {}
            for index, chunk_text in enumerate(text_chunks):
                chunk_key = compute_mdhash_id(chunk_text, prefix="chunk-")
                tokens = len(self.tokenizer.encode(chunk_text))
                inserting_chunks[chunk_key] = {
                    "content": chunk_text,
                    "full_doc_id": doc_key,
                    "tokens": tokens,
                    "chunk_order_index": index,
                    "file_path": file_path,
                }

            doc_ids = set(inserting_chunks.keys())
            add_chunk_keys = await self.text_chunks.filter_keys(doc_ids)
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return

            tasks = [
                self.chunks_vdb.upsert(inserting_chunks),
                self._process_extract_entities(inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done_with_cleanup()

    async def _process_extract_entities(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                global_config=self._build_global_config(),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
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

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        storages = self._index_storages()

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
            *[_flush_one(inst) for inst in storages], return_exceptions=True
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

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

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
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_kg(custom_kg, full_doc_id))

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str = None,
    ) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
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

            if all_chunks_data:
                await asyncio.gather(
                    self.chunks_vdb.upsert(all_chunks_data),
                    self.text_chunks.upsert(all_chunks_data),
                )

            # Keep the last declaration for each entity_name so batch backends
            # preserve the old serial upsert semantics deterministically.
            deduped_entities: dict[str, dict[str, Any]] = {}
            for entity_data in custom_kg.get("entities", []):
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
            for relationship_data in custom_kg.get("relationships", []):
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
                # Batch insert entities (reduces N serial awaits to 1)
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

                # Batch insert missing placeholder nodes
                if missing_nodes:
                    await self.chunk_entity_relation_graph.upsert_nodes_batch(
                        missing_nodes
                    )

                # Batch insert edges
                if edge_list:
                    await self.chunk_entity_relation_graph.upsert_edges_batch(edge_list)

                # Insert entities and relationships into vector storage (parallel)
                data_for_entities_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + "\n" + dp["description"],
                        "entity_name": dp["entity_name"],
                        "source_id": dp["source_id"],
                        "description": dp["description"],
                        "entity_type": dp["entity_type"],
                        "file_path": dp.get("file_path", "custom_kg"),
                    }
                    for dp in all_entities_data
                }

                data_for_rels_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "source_id": dp["source_id"],
                        "content": f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
                        "keywords": dp["keywords"],
                        "description": dp["description"],
                        "weight": dp["weight"],
                        "file_path": dp.get("file_path", "custom_kg"),
                    }
                    for dp in all_relationships_data
                }

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
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_data(query, param))

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
                if type(response) is str:
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_llm(query, param, system_prompt))

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
        """Persist deletion retry metadata and return the updated status record."""
        metadata = doc_status_data.get("metadata", {})
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

        updated_status_data = {
            **doc_status_data,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": updated_metadata,
            "error_msg": error_message if failed else "",
        }

        await self.doc_status.upsert({doc_id: updated_status_data})
        return updated_status_data

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
        return always_get_an_event_loop().run_until_complete(self.aclear_cache())

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
        chunk_ids: set[str],
        *,
        pipeline_status: dict,
        pipeline_status_lock: Any,
    ) -> None:
        """Remove a document's chunks and clean up its knowledge-graph contributions.

        Used by:
            - The pipeline resume branch in ``process_document`` when a
              document whose content is already extracted is re-processed
              under different ``process_options``: chunks must be wiped and
              entities/relations rebuilt fresh.
            - Future deletion paths that want a focused "purge KG only"
              operation without the LLM-cache / doc_status / full_docs
              cleanup that ``adelete_by_doc_id`` also performs.

        What this method does:
            1. Reads ``full_entities`` / ``full_relations`` to identify which
               graph nodes / edges this document contributed to.
            2. For each affected entity / relation, intersects the doc's
               ``chunk_ids`` with the union of chunk-tracking entries
               (``entity_chunks`` / ``relation_chunks``) and graph
               ``source_id`` lists, then classifies it as either
               *delete-outright* (no remaining sources) or *rebuild*
               (still references chunks from other documents).
            3. Deletes the chunks themselves from ``chunks_vdb`` and
               ``text_chunks``.
            4. For *delete-outright* entries: removes the relationship /
               entity from the graph storage, vector storage, and chunk
               tracking.
            5. Calls :py:meth:`_insert_done` to persist graph changes
               before rebuilding (so the rebuild step sees a consistent
               state).
            6. Calls :func:`rebuild_knowledge_from_chunks` to rebuild any
               *rebuild* entries from their remaining chunks (so other
               documents that also contributed to the same entity /
               relation keep their data intact).
            7. Deletes the per-doc ``full_entities`` / ``full_relations``
               index rows so subsequent re-extraction starts fresh.

        Does NOT touch:
            - ``doc_status`` / ``full_docs`` records — caller manages those.
            - ``llm_response_cache`` — orthogonal to KG cleanup.
            - Pipeline busy-flag — assumes the caller already holds the
              pipeline (i.e. this runs inside a pipeline run).

        Idempotent: passing an empty ``chunk_ids`` returns immediately
        without touching storage.
        """
        if not chunk_ids:
            return

        # ---- 1. Analyze affected entities/relations from full_entities/full_relations ----
        entities_to_delete: set[str] = set()
        entities_to_rebuild: dict[str, list[str]] = {}
        relationships_to_delete: set[tuple[str, str]] = set()
        relationships_to_rebuild: dict[tuple[str, str], list[str]] = {}
        entity_chunk_updates: dict[str, list[str]] = {}
        relation_chunk_updates: dict[tuple[str, str], list[str]] = {}

        try:
            doc_entities_data = await self.full_entities.get_by_id(doc_id)
            doc_relations_data = await self.full_relations.get_by_id(doc_id)

            affected_nodes: list[dict[str, Any]] = []
            affected_edges: list[dict[str, Any]] = []

            if doc_entities_data and "entity_names" in doc_entities_data:
                entity_names = doc_entities_data["entity_names"]
                nodes_dict = await self.chunk_entity_relation_graph.get_nodes_batch(
                    entity_names
                )
                for entity_name in entity_names:
                    node_data = nodes_dict.get(entity_name)
                    if node_data:
                        if "id" not in node_data:
                            node_data["id"] = entity_name
                        affected_nodes.append(node_data)

            if doc_relations_data and "relation_pairs" in doc_relations_data:
                relation_pairs = doc_relations_data["relation_pairs"]
                edge_pairs_dicts = [
                    {"src": pair[0], "tgt": pair[1]} for pair in relation_pairs
                ]
                edges_dict = await self.chunk_entity_relation_graph.get_edges_batch(
                    edge_pairs_dicts
                )
                for pair in relation_pairs:
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
            raise Exception(f"Failed to analyze graph dependencies: {e}") from e

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
                    graph_sources and set(graph_sources) & chunk_ids
                )

                if not remaining_sources:
                    entities_to_delete.add(node_label)
                    entity_chunk_updates[node_label] = []
                elif (
                    remaining_sources != existing_sources
                    or graph_references_deleted_chunks
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
                pipeline_status["history_messages"].append(log_message)

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
                    graph_sources and set(graph_sources) & chunk_ids
                )

                if not remaining_sources:
                    relationships_to_delete.add(edge_tuple)
                    relation_chunk_updates[edge_tuple] = []
                elif (
                    remaining_sources != existing_sources
                    or graph_references_deleted_chunks
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
                pipeline_status["history_messages"].append(log_message)

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
            raise Exception(f"Failed to process graph dependencies: {e}") from e

        # ---- 3. Delete chunks themselves ----
        try:
            await self.chunks_vdb.delete(chunk_ids)
            await self.text_chunks.delete(chunk_ids)
            async with pipeline_status_lock:
                log_message = (
                    f"[purge] {doc_id}: deleted {len(chunk_ids)} chunk(s) from storage"
                )
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        except Exception as e:
            logger.error(f"[purge] Failed to delete chunks for {doc_id}: {e}")
            raise Exception(f"Failed to delete document chunks: {e}") from e

        # ---- 4. Delete relationships with no remaining sources ----
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
                    pipeline_status["history_messages"].append(log_message)
            except Exception as e:
                logger.error(
                    f"[purge] Failed to delete relationships for {doc_id}: {e}"
                )
                raise Exception(f"Failed to delete relationships: {e}") from e

        # ---- 5. Delete entities with no remaining sources ----
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
                    pipeline_status["history_messages"].append(log_message)
            except Exception as e:
                logger.error(f"[purge] Failed to delete entities for {doc_id}: {e}")
                raise Exception(f"Failed to delete entities: {e}") from e

        # ---- 6. Persist pre-rebuild changes ----
        # Use plain _insert_done (no discard-on-failure): the pending buffer
        # here holds DELETES, which are not regenerable by reprocessing. On a
        # flush failure they must stay buffered for a later retry, not be
        # discarded (see _insert_done_with_cleanup docstring).
        try:
            await self._insert_done()
        except Exception as e:
            logger.error(f"[purge] Failed to persist pre-rebuild changes: {e}")
            raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

        # ---- 7. Rebuild entities/relations that still have remaining sources ----
        if entities_to_rebuild or relationships_to_rebuild:
            try:
                await rebuild_knowledge_from_chunks(
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
                )
            except Exception as e:
                logger.error(f"[purge] Failed to rebuild knowledge from chunks: {e}")
                raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

        # ---- 8. Delete per-doc full_entities / full_relations index rows ----
        try:
            await self.full_entities.delete([doc_id])
            await self.full_relations.delete([doc_id])
        except Exception as e:
            logger.error(
                f"[purge] Failed to delete full_entities/full_relations rows for {doc_id}: {e}"
            )
            raise Exception(
                f"Failed to delete from full_entities/full_relations: {e}"
            ) from e

    async def adelete_by_doc_id(
        self,
        doc_id: str,
        delete_llm_cache: bool = False,
        skip_rebuild: bool = False,
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
            skip_rebuild (bool): When True, skip the per-document KG rebuild step.
                The caller is responsible for performing a single deferred rebuild
                using the entities/relationships returned in the DeletionResult.
                Used by batch deletion to avoid N redundant rebuilds. Defaults to False.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", "not_allowed", or "fail".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 403, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
                - `entities_to_rebuild` (dict | None): Populated when skip_rebuild=True.
                - `relationships_to_rebuild` (dict | None): Populated when skip_rebuild=True.
        """
        # Get pipeline status shared data and lock for validation
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Track whether WE acquired the pipeline
        we_acquired_pipeline = False

        # Check and acquire pipeline if needed
        async with pipeline_status_lock:
            if not pipeline_status.get("busy", False):
                # Pipeline is idle - WE acquire it for this deletion
                we_acquired_pipeline = True
                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Single document deletion",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 1,
                        "batchs": 1,
                        "cur_batch": 0,
                        "request_pending": False,
                        "cancellation_requested": False,
                        "latest_message": f"Starting deletion for document: {doc_id}",
                    }
                )
                # Initialize history messages
                pipeline_status["history_messages"][:] = [
                    f"Starting deletion for document: {doc_id}"
                ]
            else:
                # Pipeline already busy - verify it's a deletion job
                job_name = pipeline_status.get("job_name", "").lower()
                if not job_name.startswith("deleting") or "document" not in job_name:
                    return DeletionResult(
                        status="not_allowed",
                        doc_id=doc_id,
                        message=f"Deletion not allowed: current job '{pipeline_status.get('job_name')}' is not a document deletion job",
                        status_code=403,
                        file_path=None,
                    )
                # Pipeline is busy with deletion - proceed without acquiring

        deletion_operations_started = False
        deletion_fully_completed = False
        in_final_delete_stage = False
        original_exception = None
        doc_llm_cache_ids: list[str] = []
        deletion_stage = "initializing"
        doc_status_data: dict[str, Any] | None = None
        file_path: str | None = None

        async with pipeline_status_lock:
            log_message = f"Starting deletion process for document {doc_id}"
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

        try:
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
                    pipeline_status["history_messages"].append(warning_msg)

            # 2. Get chunk IDs from document status
            metadata = doc_status_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata_cache_ids = normalize_string_list(
                metadata.get("deletion_llm_cache_ids", []),
                context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
            )
            chunk_ids = set(
                normalize_string_list(
                    doc_status_data.get("chunks_list", []),
                    context=f"doc {doc_id} chunks_list",
                )
            )

            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
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
                            pipeline_status["history_messages"].append(no_cache_msg)
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
                    pipeline_status["history_messages"].append(log_message)

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
                        chunk_data_list = await self.text_chunks.get_by_ids(
                            list(chunk_ids)
                        )
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

            # 4. Analyze entities and relationships that will be affected
            entities_to_delete = set()
            entities_to_rebuild = {}  # entity_name -> remaining chunk id list
            relationships_to_delete = set()
            relationships_to_rebuild = {}  # (src, tgt) -> remaining chunk id list
            entity_chunk_updates: dict[str, list[str]] = {}
            relation_chunk_updates: dict[tuple[str, str], list[str]] = {}

            try:
                deletion_stage = "analyze_graph_dependencies"
                # Get affected entities and relations from full_entities and full_relations storage
                doc_entities_data = await self.full_entities.get_by_id(doc_id)
                doc_relations_data = await self.full_relations.get_by_id(doc_id)

                affected_nodes = []
                affected_edges = []

                # Get entity data from graph storage using entity names from full_entities
                if doc_entities_data and "entity_names" in doc_entities_data:
                    entity_names = doc_entities_data["entity_names"]
                    # get_nodes_batch returns dict[str, dict], need to convert to list[dict]
                    nodes_dict = await self.chunk_entity_relation_graph.get_nodes_batch(
                        entity_names
                    )
                    for entity_name in entity_names:
                        node_data = nodes_dict.get(entity_name)
                        if node_data:
                            # Ensure compatibility with existing logic that expects "id" field
                            if "id" not in node_data:
                                node_data["id"] = entity_name
                            affected_nodes.append(node_data)

                # Get relation data from graph storage using relation pairs from full_relations
                if doc_relations_data and "relation_pairs" in doc_relations_data:
                    relation_pairs = doc_relations_data["relation_pairs"]
                    edge_pairs_dicts = [
                        {"src": pair[0], "tgt": pair[1]} for pair in relation_pairs
                    ]
                    # get_edges_batch returns dict[tuple[str, str], dict], need to convert to list[dict]
                    edges_dict = await self.chunk_entity_relation_graph.get_edges_batch(
                        edge_pairs_dicts
                    )

                    for pair in relation_pairs:
                        src, tgt = pair[0], pair[1]
                        edge_key = (src, tgt)
                        edge_data = edges_dict.get(edge_key)
                        if edge_data:
                            # Ensure compatibility with existing logic that expects "source" and "target" fields
                            if "source" not in edge_data:
                                edge_data["source"] = src
                            if "target" not in edge_data:
                                edge_data["target"] = tgt
                            affected_edges.append(edge_data)

            except Exception as e:
                logger.error(f"Failed to analyze affected graph elements: {e}")
                raise Exception(f"Failed to analyze graph dependencies: {e}") from e

            try:
                # Process entities
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
                            for chunk_id in node_data["source_id"].split(
                                GRAPH_FIELD_SEP
                            )
                            if chunk_id
                        ]

                    if not existing_sources:
                        existing_sources = graph_sources

                    if not existing_sources:
                        # No chunk references means this entity should be deleted
                        entities_to_delete.add(node_label)
                        entity_chunk_updates[node_label] = []
                        continue

                    remaining_sources = subtract_source_ids(existing_sources, chunk_ids)
                    # `existing_sources` comes from chunk-tracking storage when available, but
                    # graph `source_id` can still be stale after a failed prior delete. If the
                    # graph still references any chunk being deleted in this attempt, force a
                    # rebuild/delete so the graph metadata gets synchronized instead of being
                    # left untouched with orphaned source references.
                    graph_references_deleted_chunks = bool(
                        graph_sources and set(graph_sources) & chunk_ids
                    )

                    if not remaining_sources:
                        entities_to_delete.add(node_label)
                        entity_chunk_updates[node_label] = []
                    elif (
                        remaining_sources != existing_sources
                        or graph_references_deleted_chunks
                    ):
                        entities_to_rebuild[node_label] = remaining_sources
                        entity_chunk_updates[node_label] = remaining_sources
                    else:
                        logger.info(f"Untouch entity: {node_label}")

                async with pipeline_status_lock:
                    log_message = f"Found {len(entities_to_rebuild)} affected entities"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                # Process relationships
                for edge_data in affected_edges:
                    # source target is not in normalize order in graph db property
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

                    existing_sources: list[str] = []
                    graph_sources: list[str] = []
                    if self.relation_chunks:
                        storage_key = make_relation_chunk_key(src, tgt)
                        stored_chunks = await self.relation_chunks.get_by_id(
                            storage_key
                        )
                        if stored_chunks and isinstance(stored_chunks, dict):
                            existing_sources = [
                                chunk_id
                                for chunk_id in stored_chunks.get("chunk_ids", [])
                                if chunk_id
                            ]

                    if edge_data.get("source_id"):
                        graph_sources = [
                            chunk_id
                            for chunk_id in edge_data["source_id"].split(
                                GRAPH_FIELD_SEP
                            )
                            if chunk_id
                        ]

                    if not existing_sources:
                        existing_sources = graph_sources

                    if not existing_sources:
                        # No chunk references means this relationship should be deleted
                        relationships_to_delete.add(edge_tuple)
                        relation_chunk_updates[edge_tuple] = []
                        continue

                    remaining_sources = subtract_source_ids(existing_sources, chunk_ids)
                    # Same as the entity path above: even when relation chunk-tracking is already
                    # correct, the graph edge may still carry a stale `source_id` that mentions a
                    # chunk deleted in this attempt. Treat that as an affected relation so retry
                    # deletion can repair the graph metadata rather than skipping it as "untouched".
                    graph_references_deleted_chunks = bool(
                        graph_sources and set(graph_sources) & chunk_ids
                    )

                    if not remaining_sources:
                        relationships_to_delete.add(edge_tuple)
                        relation_chunk_updates[edge_tuple] = []
                    elif (
                        remaining_sources != existing_sources
                        or graph_references_deleted_chunks
                    ):
                        relationships_to_rebuild[edge_tuple] = remaining_sources
                        relation_chunk_updates[edge_tuple] = remaining_sources
                    else:
                        logger.info(f"Untouch relation: {edge_tuple}")

                async with pipeline_status_lock:
                    log_message = (
                        f"Found {len(relationships_to_rebuild)} affected relations"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                current_time = int(time.time())
                deletion_stage = "update_chunk_tracking"

                if entity_chunk_updates and self.entity_chunks:
                    entity_upsert_payload = {}
                    for entity_name, remaining in entity_chunk_updates.items():
                        if not remaining:
                            # Empty entities are deleted alongside graph nodes later
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
                            # Empty relations are deleted alongside graph edges later
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
                logger.error(f"Failed to process graph analysis results: {e}")
                raise Exception(f"Failed to process graph dependencies: {e}") from e

            # Data integrity is ensured by allowing only one process to hold pipeline at a time（no graph db lock is needed anymore)

            # 5. Delete chunks from storage
            if chunk_ids:
                try:
                    deletion_stage = "delete_chunks"
                    await self.chunks_vdb.delete(chunk_ids)
                    await self.text_chunks.delete(chunk_ids)

                    async with pipeline_status_lock:
                        log_message = (
                            f"Successfully deleted {len(chunk_ids)} chunks from storage"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete chunks: {e}")
                    raise Exception(f"Failed to delete document chunks: {e}") from e

            # 6. Delete relationships that have no remaining sources
            if relationships_to_delete:
                try:
                    deletion_stage = "delete_relationships"
                    # Delete from relation vdb
                    rel_ids_to_delete = []
                    for src, tgt in relationships_to_delete:
                        rel_ids_to_delete.extend(
                            [
                                compute_mdhash_id(src + tgt, prefix="rel-"),
                                compute_mdhash_id(tgt + src, prefix="rel-"),
                            ]
                        )
                    await self.relationships_vdb.delete(rel_ids_to_delete)

                    # Delete from graph
                    await self.chunk_entity_relation_graph.remove_edges(
                        list(relationships_to_delete)
                    )

                    # Delete from relation_chunks storage
                    if self.relation_chunks:
                        relation_storage_keys = [
                            make_relation_chunk_key(src, tgt)
                            for src, tgt in relationships_to_delete
                        ]
                        await self.relation_chunks.delete(relation_storage_keys)

                    async with pipeline_status_lock:
                        log_message = f"Successfully deleted {len(relationships_to_delete)} relations"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete relationships: {e}")
                    raise Exception(f"Failed to delete relationships: {e}") from e

            # 7. Delete entities that have no remaining sources
            if entities_to_delete:
                try:
                    deletion_stage = "delete_entities"
                    # Batch get all edges for entities to avoid N+1 query problem
                    nodes_edges_dict = (
                        await self.chunk_entity_relation_graph.get_nodes_edges_batch(
                            list(entities_to_delete)
                        )
                    )

                    # Debug: Check and log all edges before deleting nodes
                    edges_to_delete = set()
                    edges_still_exist = 0

                    for entity, edges in nodes_edges_dict.items():
                        if edges:
                            for src, tgt in edges:
                                # Normalize edge representation (sorted for consistency)
                                edge_tuple = tuple(sorted((src, tgt)))
                                edges_to_delete.add(edge_tuple)

                                if (
                                    src in entities_to_delete
                                    and tgt in entities_to_delete
                                ):
                                    logger.warning(
                                        f"Edge still exists: {src} <-> {tgt}"
                                    )
                                elif src in entities_to_delete:
                                    logger.warning(
                                        f"Edge still exists: {src} --> {tgt}"
                                    )
                                else:
                                    logger.warning(
                                        f"Edge still exists: {src} <-- {tgt}"
                                    )
                            edges_still_exist += 1

                    if edges_still_exist:
                        logger.warning(
                            f"⚠️ {edges_still_exist} entities still has edges before deletion"
                        )

                    # Clean residual edges from VDB and storage before deleting nodes
                    if edges_to_delete:
                        # Delete from relationships_vdb
                        rel_ids_to_delete = []
                        for src, tgt in edges_to_delete:
                            rel_ids_to_delete.extend(
                                [
                                    compute_mdhash_id(src + tgt, prefix="rel-"),
                                    compute_mdhash_id(tgt + src, prefix="rel-"),
                                ]
                            )
                        await self.relationships_vdb.delete(rel_ids_to_delete)

                        # Delete from relation_chunks storage
                        if self.relation_chunks:
                            relation_storage_keys = [
                                make_relation_chunk_key(src, tgt)
                                for src, tgt in edges_to_delete
                            ]
                            await self.relation_chunks.delete(relation_storage_keys)

                        logger.info(
                            f"Cleaned {len(edges_to_delete)} residual edges from VDB and chunk-tracking storage"
                        )

                    # Delete from graph (edges will be auto-deleted with nodes)
                    await self.chunk_entity_relation_graph.remove_nodes(
                        list(entities_to_delete)
                    )

                    # Delete from vector vdb
                    entity_vdb_ids = [
                        compute_mdhash_id(entity, prefix="ent-")
                        for entity in entities_to_delete
                    ]
                    await self.entities_vdb.delete(entity_vdb_ids)

                    # Delete from entity_chunks storage
                    if self.entity_chunks:
                        await self.entity_chunks.delete(list(entities_to_delete))

                    async with pipeline_status_lock:
                        log_message = (
                            f"Successfully deleted {len(entities_to_delete)} entities"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete entities: {e}")
                    raise Exception(f"Failed to delete entities: {e}") from e

            # Persist changes to graph database before entity and relationship rebuild
            # Plain _insert_done: pending DELETES must be retained for retry on
            # failure, not discarded (see _insert_done_with_cleanup docstring).
            try:
                deletion_stage = "persist_pre_rebuild_changes"
                await self._insert_done()
            except Exception as e:
                logger.error(f"Failed to persist pre-rebuild changes: {e}")
                raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

            # 8. Rebuild entities and relationships from remaining chunks
            #    When skip_rebuild is set (batch deletion), we hand the targets back
            #    to the caller so it can do one combined rebuild at the end.
            if entities_to_rebuild or relationships_to_rebuild:
                if skip_rebuild:
                    logger.info(
                        "Skipping per-doc rebuild (skip_rebuild=True), "
                        "%d entities / %d relations deferred",
                        len(entities_to_rebuild),
                        len(relationships_to_rebuild),
                    )
                else:
                    try:
                        deletion_stage = "rebuild_knowledge_graph"
                        await rebuild_knowledge_from_chunks(
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
                        )

                    except Exception as e:
                        logger.error(f"Failed to rebuild knowledge from chunks: {e}")
                        raise Exception(
                            f"Failed to rebuild knowledge graph: {e}"
                        ) from e

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
                        pipeline_status["history_messages"].append(log_message)
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
                        pipeline_status["history_messages"].append(cache_log_message)
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
                        pipeline_status["history_messages"].append(log_message)
                    raise Exception(log_message) from cache_delete_error

            # 10. Delete from full_entities and full_relations storage
            try:
                deletion_stage = "delete_doc_graph_metadata"
                await self.full_entities.delete([doc_id])
                await self.full_relations.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete from full_entities/full_relations: {e}")
                raise Exception(
                    f"Failed to delete from full_entities/full_relations: {e}"
                ) from e

            # 11. Delete original document and status.
            #     Normal mode: doc_status is deleted first so that if full_docs.delete
            #     fails, a retry finds no doc_status record and treats the doc as gone.
            #     Batch mode (skip_rebuild): doc_status stays alive because the caller
            #     still needs to do a deferred rebuild that might fail. If it does fail,
            #     having doc_status around means the user can just re-trigger deletion.
            try:
                deletion_stage = "delete_doc_entries"
                in_final_delete_stage = True
                if not skip_rebuild:
                    await self.doc_status.delete([doc_id])
                await self.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

            deletion_fully_completed = True

            result = DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )
            if skip_rebuild:
                result.entities_to_rebuild = entities_to_rebuild
                result.relationships_to_rebuild = relationships_to_rebuild
            return result

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

            # Release pipeline only if WE acquired it
            if we_acquired_pipeline:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["cancellation_requested"] = False
                    completion_msg = (
                        f"Deletion process completed for document: {doc_id}"
                    )
                    pipeline_status["latest_message"] = completion_msg
                    pipeline_status["history_messages"].append(completion_msg)
                    logger.info(completion_msg)

    async def adelete_by_entity(self, entity_name: str) -> DeletionResult:
        """Asynchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.adelete_by_relation(source_entity, target_entity)
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
        """Get detailed information of an entity"""
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
        """Get detailed information of a relationship"""
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename, allow_merge)
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_relation(source_entity, target_entity, updated_data)
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data))

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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.acreate_relation(source_entity, target_entity, relation_data)
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
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
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
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.aexport_data(output_path, file_format, include_vector_data)
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
