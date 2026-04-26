from __future__ import annotations

import traceback
import asyncio
import inspect
import os
import json
import re
import fnmatch
import hashlib
import base64
import mimetypes
import sys
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
from pathlib import Path
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
from lightrag.exceptions import PipelineCancelledException
from lightrag.constants import (
    DEFAULT_MAX_GLEANING,
    DEFAULT_MAX_EXTRACTION_RECORDS,
    DEFAULT_MAX_EXTRACTION_ENTITIES,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
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
    FULL_DOCS_FORMAT_RAW,
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
    DEFAULT_MAX_PARALLEL_ANALYZE,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)


from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_data_init_lock,
    get_default_workspace,
    set_default_workspace,
    get_namespace_lock,
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
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    rebuild_knowledge_from_chunks,
    _warn_deprecated_query_model_func,
)
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    lazy_external_import,
    priority_limit_async_func_call,
    get_content_summary,
    sanitize_text_for_encoding,
    check_storage_env_vars,
    generate_track_id,
    convert_to_user_format,
    logger,
    make_relation_vdb_ids,
    subtract_source_ids,
    make_relation_chunk_key,
    normalize_source_ids_limit_method,
)
from lightrag.types import KnowledgeGraph
from dotenv import load_dotenv
from lightrag.extraction.interchange import parse_interchange_jsonl

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def _chunk_fields_from_status_doc(
    status_doc: "DocProcessingStatus",
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


def _resolve_doc_file_path(
    status_doc: "DocProcessingStatus" | None = None,
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


def _normalize_string_list(raw_values: Any, context: str = "") -> list[str]:
    """Return a list of non-empty strings from raw_values.

    Non-string elements are dropped and logged as warnings. If raw_values is
    not a list, an empty list is returned.
    """
    if not isinstance(raw_values, list):
        return []
    result = []
    for i, value in enumerate(raw_values):
        if isinstance(value, str) and value:
            result.append(value)
        else:
            logger.warning(
                "Non-string element dropped from list%s at index %d: %r",
                f" ({context})" if context else "",
                i,
                value,
            )
    return result


def _split_text_units_for_hard_fallback(text: str) -> list[str]:
    """Split text into sentence/paragraph-like units for fallback chunking."""
    if not text:
        return []
    units: list[str] = []
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        for sentence in re.split(r"(?<=[。！？；.!?])", p):
            s = sentence.strip()
            if s:
                units.append(s)
    return units if units else [text]


def _split_text_by_token_limit(
    text: str, tokenizer: Tokenizer, max_tokens: int
) -> list[str]:
    """Split text by token limit with sentence-first, token-window fallback."""
    if not text:
        return []

    try:
        total_tokens = len(tokenizer.encode(text))
    except Exception:
        total_tokens = 0

    if total_tokens > 0 and total_tokens <= max_tokens:
        return [text]

    units = _split_text_units_for_hard_fallback(text)
    out: list[str] = []
    cur_parts: list[str] = []
    cur_tokens = 0

    for unit in units:
        try:
            unit_tokens = len(tokenizer.encode(unit))
        except Exception:
            unit_tokens = 0

        # Sentence itself is oversize: token-window split directly.
        if unit_tokens > max_tokens:
            if cur_parts:
                out.append("\n\n".join(cur_parts))
                cur_parts = []
                cur_tokens = 0

            token_ids = tokenizer.encode(unit)
            for start in range(0, len(token_ids), max_tokens):
                piece = tokenizer.decode(token_ids[start : start + max_tokens]).strip()
                if piece:
                    out.append(piece)
            continue

        if cur_parts and cur_tokens + unit_tokens > max_tokens:
            out.append("\n\n".join(cur_parts))
            cur_parts = [unit]
            cur_tokens = unit_tokens
        else:
            cur_parts.append(unit)
            cur_tokens += unit_tokens

    if cur_parts:
        out.append("\n\n".join(cur_parts))

    return [x for x in out if x.strip()]


def _enforce_chunk_token_limit_before_embedding(
    chunking_result: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Hard fallback split before embedding while preserving heading hierarchy."""
    if max_tokens <= 0:
        return list(chunking_result)

    normalized: list[dict[str, Any]] = []

    for dp in chunking_result:
        if not isinstance(dp, dict):
            continue

        content = dp.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        try:
            token_count = len(tokenizer.encode(content))
        except Exception:
            token_count = (
                dp.get("tokens", 0) if isinstance(dp.get("tokens"), int) else 0
            )

        if token_count <= max_tokens:
            ndp = dict(dp)
            ndp["tokens"] = token_count if token_count > 0 else ndp.get("tokens", 0)
            normalized.append(ndp)
            continue

        pieces = _split_text_by_token_limit(content, tokenizer, max_tokens)
        if not pieces:
            ndp = dict(dp)
            ndp["tokens"] = token_count
            normalized.append(ndp)
            continue

        base_chunk_id = dp.get("chunk_id")
        total_parts = len(pieces)
        for i, piece in enumerate(pieces, 1):
            new_dp = dict(dp)
            new_dp["content"] = piece
            try:
                new_dp["tokens"] = len(tokenizer.encode(piece))
            except Exception:
                new_dp["tokens"] = max(1, int(len(piece) * 0.5))

            # Keep heading/parent_headings/level unchanged; only split payload.
            if isinstance(base_chunk_id, str) and base_chunk_id.strip():
                new_dp["chunk_id"] = f"{base_chunk_id}-s{i:02d}"

            new_dp["split_type"] = "hard_fallback"
            new_dp["split_part"] = i
            new_dp["split_total"] = total_parts
            normalized.append(new_dp)

    # Rebuild order index to keep continuity after splitting.
    for idx, item in enumerate(normalized):
        item["chunk_order_index"] = idx
    return normalized


def _default_addon_params() -> dict[str, Any]:
    return {
        "language": get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE, str),
        "entity_type_prompt_file": get_env_value("ENTITY_TYPE_PROMPT_FILE", "", str),
    }


def _optional_env_int(env_key: str) -> int | None:
    return get_env_value(env_key, None, int, special_none=True)


@dataclass(frozen=True)
class RoleSpec:
    """Static descriptor for a known LLM role.

    Adding a new role anywhere in LightRAG is a single-line edit: append a
    ``RoleSpec`` to :data:`ROLES`. Every other component (env var loop in
    ``api/config.py``, queue observability, role config update flow) iterates
    this registry rather than hard-coding role names.
    """

    name: str
    """Canonical lowercase role key (used in ``role_llm_configs`` dict and CLI/log output)."""

    env_prefix: str
    """Uppercase prefix used by the API env-var layer, e.g. ``"EXTRACT"`` for
    ``EXTRACT_LLM_BINDING`` / ``MAX_ASYNC_EXTRACT_LLM`` / ``LLM_TIMEOUT_EXTRACT_LLM``."""

    queue_name: str
    """Display name passed to ``priority_limit_async_func_call`` for log lines."""


ROLES: tuple[RoleSpec, ...] = (
    RoleSpec("extract", "EXTRACT", "extract LLM func"),
    RoleSpec("keyword", "KEYWORD", "keyword LLM func"),
    RoleSpec("query", "QUERY", "query LLM func"),
    RoleSpec("vlm", "VLM", "vlm LLM func"),
)
ROLE_NAMES: frozenset[str] = frozenset(spec.name for spec in ROLES)
ROLES_BY_NAME: dict[str, RoleSpec] = {spec.name: spec for spec in ROLES}


@dataclass
class RoleLLMConfig:
    """Per-role LLM override accepted at :class:`LightRAG` init time.

    Any field left as ``None`` falls back to the corresponding base LLM
    setting (``llm_model_func`` / ``llm_model_kwargs`` / ``llm_model_max_async``
    / ``default_llm_timeout``). When ``max_async`` is None at init and the
    user did not pass a ``role_llm_configs`` entry for the role, the value is
    additionally seeded from ``MAX_ASYNC_{ROLE_PREFIX}_LLM``. ``metadata`` seeds
    runtime observability and role-builder context.
    """

    func: Callable[..., object] | None = None
    kwargs: dict[str, Any] | None = None
    max_async: int | None = None
    timeout: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class _RoleLLMState:
    """Runtime state for one role. Internal — not part of the public API."""

    raw_func: Callable[..., object]
    kwargs: dict[str, Any] | None
    max_async: int | None
    timeout: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    wrapped: Callable[..., object] | None = None


class ObservableAddonParams(dict[str, Any]):
    def __init__(
        self,
        *args: Any,
        on_change: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_change = on_change

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._changed()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._changed()

    def clear(self) -> None:
        if self:
            super().clear()
            self._changed()

    def pop(self, key: str, default: Any = ...):
        existed = key in self
        if default is ...:
            value = super().pop(key)
            self._changed()
        else:
            value = super().pop(key, default)
            if existed:
                self._changed()
        return value

    def popitem(self) -> tuple[str, Any]:
        item = super().popitem()
        self._changed()
        return item

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        value = super().setdefault(key, default)
        self._changed()
        return value

    def update(self, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            return
        super().update(*args, **kwargs)
        self._changed()

    def __ior__(self, other: Mapping[str, Any]):
        super().__ior__(other)
        self._changed()
        return self


@final
@dataclass
class LightRAG:
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

    chunk_token_size: int = field(default=int(os.getenv("CHUNK_SIZE", 1200)))
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = field(
        default=int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

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
    Custom chunking function for splitting text into chunks before processing.

    The function can be either synchronous or asynchronous.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.
        - `chunk_token_size`: The maximum number of tokens per chunk.


    The function should return a list of dictionaries (or an awaitable that resolves to a list),
    where each dictionary contains the following keys:
        - `tokens` (int): The number of tokens in the chunk.
        - `content` (str): The text content of the chunk.
        - `chunk_order_index` (int): Zero-based index indicating the chunk's order in the document.

    Defaults to `chunking_by_token_size` if not specified.
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
    pick up ``MAX_ASYNC_{ROLE_PREFIX}_LLM`` env defaults."""

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
        default=int(os.getenv("MAX_ASYNC", DEFAULT_MAX_ASYNC))
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
                "MAX_ASYNC_RERANK_LLM",
                os.getenv("MAX_ASYNC", DEFAULT_MAX_ASYNC),
            )
        )
    )
    """Maximum number of concurrent rerank calls.
    Falls back to MAX_ASYNC when MAX_ASYNC_RERANK_LLM is unset."""

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

    # Extensions
    # ---

    max_parallel_insert: int = field(
        default=int(os.getenv("MAX_PARALLEL_INSERT", DEFAULT_MAX_PARALLEL_INSERT))
    )
    """Maximum number of parallel insert operations."""

    max_parallel_parse_native: int = field(
        default=int(os.getenv("MAX_PARALLEL_PARSE_NATIVE", "5"))
    )
    max_parallel_parse_mineru: int = field(
        default=int(os.getenv("MAX_PARALLEL_PARSE_MINERU", "3"))
    )
    max_parallel_parse_docling: int = field(
        default=int(os.getenv("MAX_PARALLEL_PARSE_DOCLING", "3"))
    )
    max_parallel_analyze: int = field(
        default=int(
            os.getenv("MAX_PARALLEL_ANALYZE", str(DEFAULT_MAX_PARALLEL_ANALYZE))
        )
    )
    queue_size_default: int = field(default=int(os.getenv("QUEUE_SIZE_DEFAULT", "100")))
    queue_size_insert: int = field(default=int(os.getenv("QUEUE_SIZE_INSERT", "4")))

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

    @staticmethod
    def _normalize_llm_role(role: str) -> str:
        normalized = role.strip().lower()
        if normalized not in ROLE_NAMES:
            raise ValueError(f"Invalid LLM role: {role}")
        return normalized

    def register_role_llm_builder(
        self,
        builder: Callable[
            [str, dict[str, Any]], tuple[Callable[..., object], dict[str, Any] | None]
        ],
    ) -> None:
        """Register a runtime builder used by update_llm_role_config for binding/model updates."""
        self._llm_role_builder = builder

    def set_role_llm_metadata(self, role: str, **metadata: Any) -> None:
        """Store role metadata used when rebuilding a role-specific LLM function."""
        role = self._normalize_llm_role(role)
        state = self._role_llm_states[role]
        for key, value in metadata.items():
            if value is None:
                continue
            state.metadata[key] = value

    @property
    def role_llm_funcs(self) -> Mapping[str, Callable[..., object]]:
        """Read-only mapping of role name → wrapped (queue-managed) LLM func."""
        return {
            name: state.wrapped
            for name, state in self._role_llm_states.items()
            if state.wrapped is not None
        }

    @property
    def role_llm_kwargs(self) -> Mapping[str, dict[str, Any] | None]:
        """Read-only mapping of role name → effective LLM kwargs (None means inherit base)."""
        return {name: state.kwargs for name, state in self._role_llm_states.items()}

    def _get_effective_role_llm_kwargs(self, role: str) -> dict[str, Any]:
        state = self._role_llm_states[self._normalize_llm_role(role)]
        if state.kwargs is not None:
            return state.kwargs
        if state.metadata.get("is_cross_provider"):
            return {}
        return self.llm_model_kwargs

    def _get_effective_role_llm_timeout(self, role: str) -> int:
        state = self._role_llm_states[self._normalize_llm_role(role)]
        return state.timeout if state.timeout is not None else self.default_llm_timeout

    def _get_effective_role_llm_max_async(self, role: str) -> int:
        state = self._role_llm_states[self._normalize_llm_role(role)]
        return (
            state.max_async if state.max_async is not None else self.llm_model_max_async
        )

    def _wrap_llm_role_func(
        self,
        role_name: str,
        raw_func: Callable[..., object],
        max_async: int,
        timeout: int,
        model_kwargs: dict[str, Any],
    ) -> Callable[..., object]:
        spec = ROLES_BY_NAME[role_name]
        return priority_limit_async_func_call(
            max_async,
            llm_timeout=timeout,
            queue_name=spec.queue_name,
        )(
            partial(
                raw_func,
                hashing_kv=self.llm_response_cache,
                **model_kwargs,
            )
        )

    def _rebuild_role_llm_funcs(self) -> None:
        """Wrap each role's raw_func with its own priority queue.

        Base ``llm_model_func`` is intentionally NOT wrapped — concurrency
        for the base function is enforced at the role layer (every code path
        that calls an LLM goes through a role wrapper).
        """
        for spec in ROLES:
            self._rebuild_single_role_llm_func(spec.name)

    def _rebuild_single_role_llm_func(self, role: str) -> None:
        role = self._normalize_llm_role(role)
        state = self._role_llm_states[role]
        state.wrapped = self._wrap_llm_role_func(
            role,
            state.raw_func,
            self._get_effective_role_llm_max_async(role),
            self._get_effective_role_llm_timeout(role),
            self._get_effective_role_llm_kwargs(role),
        )

    async def _shutdown_llm_wrapper(self, wrapped_func: Callable[..., object]) -> None:
        shutdown = getattr(wrapped_func, "shutdown", None)
        if callable(shutdown):
            await shutdown(graceful=True)

    def _schedule_retired_llm_queue_cleanup(
        self, wrapped_func: Callable[..., object] | None
    ) -> None:
        if wrapped_func is None or not callable(
            getattr(wrapped_func, "shutdown", None)
        ):
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # The retired wrapper's queue and worker tasks are tied to the
            # event loop that first used them. Spinning up a fresh loop via
            # asyncio.run would either hang on queue.join() or touch
            # primitives bound to a closed loop. Skip cleanup with a warning
            # — call aupdate_llm_role_config() from an async context for
            # deterministic shutdown.
            logger.warning(
                "update_llm_role_config: skipping retired LLM queue cleanup "
                "because no event loop is running; call aupdate_llm_role_config() "
                "from an async context for deterministic shutdown"
            )
            return

        task = loop.create_task(self._shutdown_llm_wrapper(wrapped_func))
        self._retired_llm_queue_cleanup_tasks.add(task)
        task.add_done_callback(self._finalize_retired_llm_queue_cleanup)

    def _finalize_retired_llm_queue_cleanup(self, task: asyncio.Task) -> None:
        self._retired_llm_queue_cleanup_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Retired LLM queue cleanup failed: {e}")

    async def wait_for_retired_llm_queues(self) -> None:
        """Wait until all retired role LLM queues have drained and shut down.

        Cleanup failures are logged by ``_finalize_retired_llm_queue_cleanup``
        and intentionally swallowed here so callers can rely on this method
        always returning once every retired wrapper has finished.
        """
        while self._retired_llm_queue_cleanup_tasks:
            tasks = list(self._retired_llm_queue_cleanup_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _apply_llm_role_config_update(
        self,
        role: str,
        *,
        model_func: Callable[..., object] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_async: int | None = None,
        timeout: int | None = None,
        binding: str | None = None,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> Callable[..., object] | None:
        role = self._normalize_llm_role(role)
        state = self._role_llm_states[role]
        old_wrapped = state.wrapped

        snapshot = _RoleLLMState(
            raw_func=state.raw_func,
            kwargs=deepcopy(state.kwargs),
            max_async=state.max_async,
            timeout=state.timeout,
            metadata=deepcopy(state.metadata),
            wrapped=state.wrapped,
        )

        try:
            if model_func is not None and not callable(model_func):
                raise TypeError("model_func must be callable")

            if model_kwargs is not None:
                state.kwargs = model_kwargs
            if max_async is not None:
                state.max_async = max_async
            if timeout is not None:
                state.timeout = timeout
            if model_func is not None:
                state.raw_func = model_func

            metadata_updated = any(
                value is not None
                for value in (binding, model, host, api_key, provider_options)
            )
            if binding is not None:
                state.metadata["binding"] = binding
            if model is not None:
                state.metadata["model"] = model
            if host is not None:
                state.metadata["host"] = host
            if api_key is not None:
                state.metadata["api_key"] = api_key
            if provider_options is not None:
                state.metadata["provider_options"] = provider_options
            if "base_binding" in state.metadata and "binding" in state.metadata:
                state.metadata["is_cross_provider"] = (
                    state.metadata["binding"] != state.metadata["base_binding"]
                )

            if metadata_updated:
                builder = getattr(self, "_llm_role_builder", None)
                if builder is None and model_func is None:
                    raise ValueError(
                        "Runtime role builder is not configured; provide model_func or register_role_llm_builder() first"
                    )
                if builder is not None:
                    built_func, built_kwargs = builder(role, state.metadata)
                    state.raw_func = built_func
                    if model_kwargs is None and built_kwargs is not None:
                        state.kwargs = built_kwargs

            self._rebuild_single_role_llm_func(role)
        except Exception:
            state.raw_func = snapshot.raw_func
            state.kwargs = snapshot.kwargs
            state.max_async = snapshot.max_async
            state.timeout = snapshot.timeout
            state.metadata = snapshot.metadata
            state.wrapped = snapshot.wrapped
            raise

        self._log_llm_role_config("updated", role=role)
        return old_wrapped

    def update_llm_role_config(
        self,
        role: str,
        *,
        model_func: Callable[..., object] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_async: int | None = None,
        timeout: int | None = None,
        binding: str | None = None,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Update a role-specific LLM configuration at runtime.

        Supports lightweight updates (kwargs/max_async/timeout/model_func) directly.
        For binding/model/host/api_key/provider_options updates, a role builder must
        be registered via register_role_llm_builder().
        """
        old_wrapped = self._apply_llm_role_config_update(
            role,
            model_func=model_func,
            model_kwargs=model_kwargs,
            max_async=max_async,
            timeout=timeout,
            binding=binding,
            model=model,
            host=host,
            api_key=api_key,
            provider_options=provider_options,
        )
        self._schedule_retired_llm_queue_cleanup(old_wrapped)

    async def aupdate_llm_role_config(
        self,
        role: str,
        *,
        model_func: Callable[..., object] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_async: int | None = None,
        timeout: int | None = None,
        binding: str | None = None,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        """Async variant of update_llm_role_config that waits for queue cleanup.

        Blocking behavior:
            This coroutine awaits a graceful shutdown of the retired role
            wrapper's priority queue. The shutdown blocks on
            ``queue.join()`` until every already-queued LLM call has been
            executed (workers always call ``task_done()`` in ``finally``,
            so in-flight requests are not cut off).

            The wait is bounded by ``max_task_duration`` of the retired
            queue, which is computed as ``llm_timeout * 2 + 15`` seconds
            (default ``180 * 2 + 15 = 375`` seconds, ~6 min 15 s). When
            this bound is reached, the drain times out and the shutdown
            falls through to forced cancellation: pending futures are
            cancelled, the queue is cleared, workers are stopped. So this
            method **never blocks indefinitely**, but with a deep backlog
            of slow LLM calls it can take up to that bound to return, and
            in-flight calls past the bound will be cancelled.

            If you need a non-blocking switch, use the sync
            ``update_llm_role_config()`` (which schedules cleanup as a
            background task) and await ``wait_for_retired_llm_queues()``
            separately when you want to confirm the old queue is gone.
        """
        old_wrapped = self._apply_llm_role_config_update(
            role,
            model_func=model_func,
            model_kwargs=model_kwargs,
            max_async=max_async,
            timeout=timeout,
            binding=binding,
            model=model,
            host=host,
            api_key=api_key,
            provider_options=provider_options,
        )
        if old_wrapped is not None:
            await self._shutdown_llm_wrapper(old_wrapped)

    _SECRET_MARKERS = (
        "api_key",
        "api-key",
        "apikey",
        "access_key",
        "access-key",
        "secret",
        "token",
        "credential",
        "password",
        "passphrase",
        "pwd",
        "auth",
        "session",
    )

    @classmethod
    def _is_secret_key(cls, key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in cls._SECRET_MARKERS)

    def _scrubbed_llm_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a deep copy of ``metadata`` with auth-bearing fields removed.

        Auth-bearing fields are stripped entirely — not masked — because a
        masked ``"***"`` carries no information for an external consumer
        (operators already see ``binding`` / ``host`` to confirm a role is
        configured). Stripping makes the invariant simple: anything that
        appears in this output is safe to log, cache, ship over the wire.

        Components that legitimately need the raw secret (the role builder,
        provider clients) read it directly off the private
        ``_role_llm_states[role].metadata`` dict.
        """

        def scrub_value(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: scrub_value(inner_value)
                    for key, inner_value in value.items()
                    if not self._is_secret_key(str(key))
                }
            if isinstance(value, list):
                return [scrub_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(scrub_value(item) for item in value)
            return deepcopy(value)

        return scrub_value(metadata)

    def get_llm_role_config(self, role: str | None = None) -> dict[str, Any]:
        """Return effective role LLM runtime configuration (observability snapshot).

        Each role entry exposes ``binding`` / ``model`` / ``host`` at the top
        level for convenience and again inside ``metadata`` as part of the
        full runtime snapshot (which may contain extra builder-specific
        keys). Auth-bearing fields (``api_key``, ``aws_secret_access_key``,
        ``password``, …) are **stripped entirely** from ``metadata`` — this
        method is intended for ``/health`` / WebUI / audit output and must
        never leak credentials. There is no escape hatch; runtime components
        that legitimately need the raw value read it from
        ``_role_llm_states[role].metadata`` directly.
        """

        def role_config(role_name: str) -> dict[str, Any]:
            state = self._role_llm_states[role_name]
            metadata = self._scrubbed_llm_metadata(state.metadata)
            return {
                "binding": metadata.get("binding"),
                "model": metadata.get("model"),
                "host": metadata.get("host"),
                "is_cross_provider": metadata.get("is_cross_provider", False),
                "max_async": self._get_effective_role_llm_max_async(role_name),
                "timeout": self._get_effective_role_llm_timeout(role_name),
                "has_model_kwargs": state.kwargs is not None,
                "metadata": metadata,
            }

        if role is not None:
            return role_config(self._normalize_llm_role(role))

        return {spec.name: role_config(spec.name) for spec in ROLES}

    def _log_llm_role_config(self, reason: str, role: str | None = None) -> None:
        """Log the sanitized role LLM runtime configuration."""
        if role is None:
            configs = self.get_llm_role_config()
            role_names = [spec.name for spec in ROLES]
            logger.info(f"Role LLM Configuration ({reason}):")
        else:
            normalized_role = self._normalize_llm_role(role)
            configs = {normalized_role: self.get_llm_role_config(normalized_role)}
            role_names = [normalized_role]
            logger.info(f"Role LLM Configuration ({reason}: {normalized_role}):")

        for role_name in role_names:
            cfg = configs[role_name]
            logger.info(
                " - %s: %s/%s, host=%s, max_async=%s, timeout=%s",
                role_name,
                cfg["binding"],
                cfg["model"],
                cfg["host"],
                cfg["max_async"],
                cfg["timeout"],
            )

    async def _queue_status_for_func(
        self, func: Callable[..., object] | None
    ) -> dict[str, Any]:
        if func is None:
            return {"available": False}
        get_stats = getattr(func, "get_queue_stats", None)
        if not callable(get_stats):
            return {"available": False}
        stats = get_stats()
        if inspect.isawaitable(stats):
            stats = await stats
        stats["available"] = True
        return stats

    async def get_llm_queue_status(self, include_base: bool = True) -> dict[str, Any]:
        """Return queue status for each role's wrapped LLM func.

        The base ``llm_model_func`` is no longer queue-wrapped, so it is not
        reported here. ``include_base`` is kept for signature compatibility
        but has no effect.
        """
        del include_base  # base is unwrapped — see docstring

        result: dict[str, Any] = {}
        for spec in ROLES:
            state = self._role_llm_states.get(spec.name)
            result[spec.name] = await self._queue_status_for_func(
                state.wrapped if state else None
            )
        return result

    async def get_embedding_queue_status(self) -> dict[str, Any]:
        """Return queue status for the wrapped embedding function."""
        return await self._queue_status_for_func(
            self.embedding_func.func if self.embedding_func is not None else None
        )

    async def get_rerank_queue_status(self) -> dict[str, Any]:
        """Return queue status for the wrapped rerank function."""
        return await self._queue_status_for_func(self.rerank_model_func)

    def _mark_addon_params_dirty(self) -> None:
        self._addon_params_dirty = True

    def _normalize_addon_params(
        self, addon_params: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if addon_params is None:
            normalized = _default_addon_params()
        elif isinstance(addon_params, Mapping):
            normalized = dict(addon_params)
        else:
            raise TypeError(
                "addon_params must be a Mapping or None, got "
                f"{type(addon_params).__name__}"
            )

        # When the caller supplies addon_params explicitly, the dataclass
        # default_factory is skipped — fall back to environment variables so
        # ENTITY_TYPE_PROMPT_FILE / SUMMARY_LANGUAGE still apply.
        normalized.setdefault(
            "language", get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE, str)
        )
        normalized.setdefault(
            "entity_type_prompt_file",
            get_env_value("ENTITY_TYPE_PROMPT_FILE", "", str),
        )
        return normalized

    def _replace_addon_params(
        self, addon_params: Mapping[str, Any] | None, *, mark_dirty: bool
    ) -> None:
        wrapped = ObservableAddonParams(
            self._normalize_addon_params(addon_params),
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
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
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
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

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
                max_async = _optional_env_int(f"MAX_ASYNC_{spec.env_prefix}_LLM")

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

    async def check_and_migrate_data(self):
        """Check if data migration is needed and perform migration if necessary"""
        async with get_data_init_lock():
            try:
                # Check if migration is needed:
                # 1. chunk_entity_relation_graph has entities and relations (count > 0)
                # 2. full_entities and full_relations are empty

                # Get all entity labels from graph
                all_entity_labels = (
                    await self.chunk_entity_relation_graph.get_all_labels()
                )

                if not all_entity_labels:
                    logger.debug("No entities found in graph, skipping migration check")
                    return

                try:
                    # Initialize chunk tracking storage after migration
                    await self._migrate_chunk_tracking_storage()
                except Exception as e:
                    logger.error(f"Error during chunk_tracking migration: {e}")
                    raise e

                # Check if full_entities and full_relations are empty
                # Get all processed documents to check their entity/relation data
                try:
                    processed_docs = await self.doc_status.get_docs_by_status(
                        DocStatus.PROCESSED
                    )

                    if not processed_docs:
                        logger.debug("No processed documents found, skipping migration")
                        return

                    # Check first few documents to see if they have full_entities/full_relations data
                    migration_needed = True
                    checked_count = 0
                    max_check = min(5, len(processed_docs))  # Check up to 5 documents

                    for doc_id in list(processed_docs.keys())[:max_check]:
                        checked_count += 1
                        entity_data = await self.full_entities.get_by_id(doc_id)
                        relation_data = await self.full_relations.get_by_id(doc_id)

                        if entity_data or relation_data:
                            migration_needed = False
                            break

                    if not migration_needed:
                        logger.debug(
                            "Full entities/relations data already exists, no migration needed"
                        )
                        return

                    logger.info(
                        f"Data migration needed: found {len(all_entity_labels)} entities in graph but no full_entities/full_relations data"
                    )

                    # Perform migration
                    await self._migrate_entity_relation_data(processed_docs)

                except Exception as e:
                    logger.error(f"Error during migration check: {e}")
                    raise e

            except Exception as e:
                logger.error(f"Error in data migration check: {e}")
                raise e

    async def _migrate_entity_relation_data(self, processed_docs: dict):
        """Migrate existing entity and relation data to full_entities and full_relations storage"""
        logger.info(f"Starting data migration for {len(processed_docs)} documents")

        # Create mapping from chunk_id to doc_id
        chunk_to_doc = {}
        for doc_id, doc_status in processed_docs.items():
            chunk_ids = (
                doc_status.chunks_list
                if hasattr(doc_status, "chunks_list") and doc_status.chunks_list
                else []
            )
            for chunk_id in chunk_ids:
                chunk_to_doc[chunk_id] = doc_id

        # Initialize document entity and relation mappings
        doc_entities = {}  # doc_id -> set of entity_names
        doc_relations = {}  # doc_id -> set of relation_pairs (as tuples)

        # Get all nodes and edges from graph
        all_nodes = await self.chunk_entity_relation_graph.get_all_nodes()
        all_edges = await self.chunk_entity_relation_graph.get_all_edges()

        # Process all nodes once
        for node in all_nodes:
            if "source_id" in node:
                entity_id = node.get("entity_id") or node.get("id")
                if not entity_id:
                    continue

                # Get chunk IDs from source_id
                source_ids = node["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this entity belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_entities:
                            doc_entities[doc_id] = set()
                        doc_entities[doc_id].add(entity_id)

        # Process all edges once
        for edge in all_edges:
            if "source_id" in edge:
                src = edge.get("source")
                tgt = edge.get("target")
                if not src or not tgt:
                    continue

                # Get chunk IDs from source_id
                source_ids = edge["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this relation belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_relations:
                            doc_relations[doc_id] = set()
                        # Use tuple for set operations, convert to list later
                        doc_relations[doc_id].add(tuple(sorted((src, tgt))))

        # Store the results in full_entities and full_relations
        migration_count = 0

        # Store entities
        if doc_entities:
            entities_data = {}
            for doc_id, entity_set in doc_entities.items():
                entities_data[doc_id] = {
                    "entity_names": list(entity_set),
                    "count": len(entity_set),
                }
            await self.full_entities.upsert(entities_data)

        # Store relations
        if doc_relations:
            relations_data = {}
            for doc_id, relation_set in doc_relations.items():
                # Convert tuples back to lists
                relations_data[doc_id] = {
                    "relation_pairs": [list(pair) for pair in relation_set],
                    "count": len(relation_set),
                }
            await self.full_relations.upsert(relations_data)

        migration_count = len(
            set(list(doc_entities.keys()) + list(doc_relations.keys()))
        )

        # Persist the migrated data
        await self.full_entities.index_done_callback()
        await self.full_relations.index_done_callback()

        logger.info(
            f"Data migration completed: migrated {migration_count} documents with entities/relations"
        )

    async def _migrate_chunk_tracking_storage(self) -> None:
        """Ensure entity/relation chunk tracking KV stores exist and are seeded."""

        if not self.entity_chunks or not self.relation_chunks:
            return

        need_entity_migration = False
        need_relation_migration = False

        try:
            need_entity_migration = await self.entity_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check entity chunks storage: {exc}")
            raise exc

        try:
            need_relation_migration = await self.relation_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check relation chunks storage: {exc}")
            raise exc

        if not need_entity_migration and not need_relation_migration:
            return

        BATCH_SIZE = 500  # Process 500 records per batch

        if need_entity_migration:
            try:
                nodes = await self.chunk_entity_relation_graph.get_all_nodes()
            except Exception as exc:
                logger.error(f"Failed to fetch nodes for chunk migration: {exc}")
                nodes = []

            logger.info(f"Starting chunk_tracking data migration: {len(nodes)} nodes")

            # Process nodes in batches
            total_nodes = len(nodes)
            total_batches = (total_nodes + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_nodes)
                batch_nodes = nodes[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for node in batch_nodes:
                    entity_id = node.get("entity_id") or node.get("id")
                    if not entity_id:
                        continue

                    raw_source = node.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    upsert_payload[entity_id] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.entity_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed entity batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_nodes})"
                    )

            if total_migrated > 0:
                # Persist entity_chunks data to disk
                await self.entity_chunks.index_done_callback()
                logger.info(
                    f"Entity chunk_tracking migration completed: {total_migrated} records persisted"
                )

        if need_relation_migration:
            try:
                edges = await self.chunk_entity_relation_graph.get_all_edges()
            except Exception as exc:
                logger.error(f"Failed to fetch edges for chunk migration: {exc}")
                edges = []

            logger.info(f"Starting chunk_tracking data migration: {len(edges)} edges")

            # Process edges in batches
            total_edges = len(edges)
            total_batches = (total_edges + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_edges)
                batch_edges = edges[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for edge in batch_edges:
                    src = edge.get("source") or edge.get("src_id") or edge.get("src")
                    tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
                    if not src or not tgt:
                        continue

                    raw_source = edge.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    storage_key = make_relation_chunk_key(src, tgt)
                    upsert_payload[storage_key] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.relation_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed relation batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_edges})"
                    )

            if total_migrated > 0:
                # Persist relation_chunks data to disk
                await self.relation_chunks.index_done_callback()
                logger.info(
                    f"Relation chunk_tracking migration completed: {total_migrated} records persisted"
                )

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

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        # Direct imports for default storage implementations
        if storage_name == "JsonKVStorage":
            from lightrag.kg.json_kv_impl import JsonKVStorage

            return JsonKVStorage
        elif storage_name == "NanoVectorDBStorage":
            from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage

            return NanoVectorDBStorage
        elif storage_name == "NetworkXStorage":
            from lightrag.kg.networkx_impl import NetworkXStorage

            return NetworkXStorage
        elif storage_name == "JsonDocStatusStorage":
            from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage

            return JsonDocStatusStorage
        else:
            # Fallback to dynamic import for other storage implementations
            import_path = STORAGES[storage_name]
            storage_class = lazy_external_import(import_path, storage_name)
            return storage_class

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
        """Async Insert documents with checkpoint support

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

        await self.apipeline_enqueue_documents(input, ids, file_paths, track_id)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

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
            file_path = ""

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
                await self._insert_done()

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
        docs_format: str = FULL_DOCS_FORMAT_RAW,
        lightrag_document_paths: str | list[str] | None = None,
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

        Returns:
            str: tracking ID for monitoring processing status
        """
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

        # 1. Validate ids and build contents (when lightrag: no content dedup, content may be empty)
        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

        if is_lightrag_format:
            # LightRAG Document: no content hash dedup; content may be empty
            contents = {}
            for i in range(len(file_paths)):
                path = file_paths[i]
                doc_id = (
                    ids[i]
                    if ids is not None
                    else compute_mdhash_id(path, prefix="doc-")
                )
                lightrag_path = (
                    lightrag_document_paths[i] if lightrag_document_paths else ""
                ) or path
                content_str = (input[i] or "").strip() if i < len(input) else ""
                contents[doc_id] = {
                    "content": content_str,
                    "file_path": path,
                    "format": FULL_DOCS_FORMAT_LIGHTRAG,
                    "lightrag_document_path": lightrag_path,
                }
        elif ids is not None:
            # Generate contents dict and remove duplicates in one pass
            unique_contents = {}
            for id_, doc, path in zip(ids, input, file_paths):
                cleaned_content = sanitize_text_for_encoding(doc)
                if cleaned_content not in unique_contents:
                    unique_contents[cleaned_content] = (id_, path)

            contents = {
                id_: {
                    "content": content,
                    "file_path": file_path,
                    "format": FULL_DOCS_FORMAT_RAW,
                }
                for content, (id_, file_path) in unique_contents.items()
            }
        elif docs_format == FULL_DOCS_FORMAT_PENDING_PARSE:
            contents = {}
            for i, (doc, path) in enumerate(zip(input, file_paths)):
                doc_id = (
                    ids[i]
                    if ids is not None
                    else compute_mdhash_id(path, prefix="doc-")
                )
                contents[doc_id] = {
                    "content": doc or "",
                    "file_path": path,
                    "format": FULL_DOCS_FORMAT_PENDING_PARSE,
                }
        else:
            # Clean input text and remove duplicates in one pass
            unique_content_with_paths = {}
            for doc, path in zip(input, file_paths):
                cleaned_content = sanitize_text_for_encoding(doc)
                if cleaned_content not in unique_content_with_paths:
                    unique_content_with_paths[cleaned_content] = path

            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                    "format": FULL_DOCS_FORMAT_RAW,
                }
                for content, path in unique_content_with_paths.items()
            }

        # 2. Generate document initial status (without content)
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(content_data.get("content", "")),
                "content_length": len(content_data.get("content", "")),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data["file_path"],
                "track_id": track_id,
            }
            for id_, content_data in contents.items()
        }

        # 3. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already enqueued
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Handle duplicate documents - create trackable records with current track_id
        ignored_ids = list(all_new_doc_ids - unique_new_doc_ids)
        if ignored_ids:
            duplicate_docs: dict[str, Any] = {}
            for doc_id in ignored_ids:
                file_path = (
                    new_docs.get(doc_id, {}).get("file_path") or "unknown_source"
                )
                logger.warning(f"Duplicate document detected: {doc_id} ({file_path})")

                # Get existing document info for reference
                existing_doc = await self.doc_status.get_by_id(doc_id)
                existing_status = (
                    existing_doc.get("status", "unknown") if existing_doc else "unknown"
                )
                existing_track_id = (
                    existing_doc.get("track_id", "") if existing_doc else ""
                )

                # Create a new record with unique ID for this duplicate attempt
                dup_record_id = compute_mdhash_id(f"{doc_id}-{track_id}", prefix="dup-")
                duplicate_docs[dup_record_id] = {
                    "status": DocStatus.FAILED,
                    "content_summary": f"[DUPLICATE] Original document: {doc_id}",
                    "content_length": new_docs.get(doc_id, {}).get("content_length", 0),
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": file_path,
                    "track_id": track_id,  # Use current track_id for tracking
                    "error_msg": f"Content already exists. Original doc_id: {doc_id}, Status: {existing_status}",
                    "metadata": {
                        "is_duplicate": True,
                        "original_doc_id": doc_id,
                        "original_track_id": existing_track_id,
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
                "format": contents[doc_id].get("format", FULL_DOCS_FORMAT_RAW),
            }
            for doc_id in new_docs.keys()
        }
        for doc_id in new_docs.keys():
            if contents[doc_id].get("lightrag_document_path"):
                full_docs_data[doc_id]["lightrag_document_path"] = contents[doc_id][
                    "lightrag_document_path"
                ]
        await self.full_docs.upsert(full_docs_data)
        # Persist data to disk immediately
        await self.full_docs.index_done_callback()

        # Store document status (without content)
        await self.doc_status.upsert(new_docs)
        logger.debug(f"Stored {len(new_docs)} new unique documents")

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
                    file_path = _resolve_doc_file_path(status_doc=status_doc)

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
                        _chunk_fields_from_status_doc(status_doc)
                    )
                    resolved_file_path = _resolve_doc_file_path(
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
                        # Clear any error messages and processing metadata
                        "error_msg": "",
                        "metadata": {},
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
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                to_process_docs: dict[
                    str, DocProcessingStatus
                ] = await self.doc_status.get_docs_by_statuses(
                    [
                        DocStatus.PROCESSING,
                        DocStatus.FAILED,
                        DocStatus.PENDING,
                        DocStatus.PARSING,
                        DocStatus.ANALYZING,
                    ]
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
                    break

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
                    break

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                # Update pipeline_status, batchs now represents the total number of files to be processed
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

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
                    file_path = _resolve_doc_file_path(status_doc=status_doc)
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
                        return _chunk_fields_from_status_doc(status_doc)

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
                                file_path = _resolve_doc_file_path(
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
                                        }
                                    }
                                )

                                if not content_data:
                                    raise Exception(
                                        f"Document content not found in full_docs for doc_id: {doc_id}"
                                    )

                                parse_engine = self._resolve_parser_engine(
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

                                # ---- Phase 2: ANALYZING ----
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.ANALYZING,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": len(content),
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,
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

                            # Try to parse as interchange JSONL (smart extraction output)
                            parsed_interchange = parse_interchange_jsonl(
                                content, self.tokenizer
                            )
                            if parsed_interchange is not None:
                                interchange_meta, interchange_chunks = (
                                    parsed_interchange
                                )
                                logger.info(
                                    f"Detected interchange JSONL for d-id: {doc_id}, {len(interchange_chunks)} chunks"
                                )
                                chunking_result = interchange_chunks
                                extraction_meta = {
                                    "extraction_format": "interchange_jsonl",
                                    "format_version": interchange_meta.get(
                                        "format_version"
                                    ),
                                    "engine": interchange_meta.get("engine"),
                                    "engine_capabilities": interchange_meta.get(
                                        "engine_capabilities", []
                                    ),
                                    "chunking_method": interchange_meta.get(
                                        "chunking_method"
                                    ),
                                }
                            else:
                                # Call chunking function, supporting both sync and async implementations
                                chunking_result = self.chunking_func(
                                    self.tokenizer,
                                    content,
                                    split_by_character,
                                    split_by_character_only,
                                    self.chunk_overlap_token_size,
                                    self.chunk_token_size,
                                )

                                # If result is awaitable, await to get actual result
                                if inspect.isawaitable(chunking_result):
                                    chunking_result = await chunking_result

                                # Validate return type
                                if not isinstance(chunking_result, (list, tuple)):
                                    raise TypeError(
                                        f"chunking_func must return a list or tuple of dicts, "
                                        f"got {type(chunking_result)}"
                                    )
                                extraction_meta = {
                                    "extraction_format": "plain_text_chunking",
                                    "engine": "legacy",
                                }

                            # Multimodal post-process hook entrypoint:
                            # runs after interchange parsing and before entity extraction.
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
                                    _enforce_chunk_token_limit_before_embedding(
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
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                **extraction_meta,
                                            },
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

                            # Stage 2: Process entity relation graph (after text_chunks are saved)
                            entity_relation_task = asyncio.create_task(
                                self._process_extract_entities(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            chunk_results = await entity_relation_task
                            chunk_results = (
                                self._augment_chunk_results_with_mm_entities(
                                    chunk_results=chunk_results,
                                    mm_specs=mm_specs,
                                    file_path=file_path,
                                )
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
                                        "metadata": {
                                            "processing_start_time": processing_start_time,
                                            "processing_end_time": processing_end_time,
                                        },
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

                                # Use chunk_results from entity_relation_task
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
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                                **extraction_meta,
                                            },
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
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                                **extraction_meta,
                                            },
                                        }
                                    }
                                )

                # Three-stage worker queues:
                # parse_native/mineru/docling -> analyze_multimodal -> process_document
                q_native: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size_default)
                q_mineru: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size_default)
                q_docling: asyncio.Queue = asyncio.Queue(
                    maxsize=self.queue_size_default
                )
                q_analyze: asyncio.Queue = asyncio.Queue(
                    maxsize=self.queue_size_default
                )
                q_process: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size_insert)

                workers: list[asyncio.Task] = []

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
                            await self.doc_status.upsert(
                                {
                                    doc_id_w: {
                                        "status": DocStatus.ANALYZING,
                                        "content_summary": status_doc_w.content_summary,
                                        "content_length": len(
                                            parsed_data_w.get("content", "")
                                        ),
                                        "created_at": status_doc_w.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path_w,
                                        "track_id": status_doc_w.track_id,
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

                for doc_id, status_doc in to_process_docs.items():
                    content_data = await self.full_docs.get_by_id(doc_id) or {}
                    engine = self._resolve_parser_engine(
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

                # Check if there's a pending request to process more documents (with lock)
                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        # Clear the request flag before checking for more documents
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                to_process_docs = await self.doc_status.get_docs_by_statuses(
                    [
                        DocStatus.PROCESSING,
                        DocStatus.FAILED,
                        DocStatus.PENDING,
                        DocStatus.PARSING,
                        DocStatus.ANALYZING,
                    ]
                )

        finally:
            log_message = "Enqueued document processing pipeline stopped"
            logger.info(log_message)
            # Always reset busy status and cancellation flag when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["cancellation_requested"] = (
                    False  # Always reset cancellation flag
                )
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

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

    async def analyze_multimodal(
        self,
        doc_id: str,
        file_path: str,
        parsed_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Phase 2: Multimodal analysis (VLM). Writes llm_analyze_result and analyze_time to LightRAG Document.
        Uses vlm_llm_model_func (VLM role). When Ray-Anything merges, bind VLM model here.
        Default: no-op, returns parsed_data unchanged.
        """
        blocks_path = parsed_data.get("blocks_path")
        if not blocks_path:
            return parsed_data

        block_file = Path(blocks_path)
        if not block_file.exists():
            return parsed_data

        try:
            lines = block_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                return parsed_data
            meta = json.loads(lines[0])
            if not isinstance(meta, dict) or meta.get("type") != "meta":
                return parsed_data
            if meta.get("analyze_time"):
                return parsed_data

            now_iso = datetime.now(timezone.utc).isoformat()
            meta["analyze_time"] = now_iso
            lines[0] = json.dumps(meta, ensure_ascii=False)
            block_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
                (Path(base_name + ".drawings.json"), "drawings"),
                (Path(base_name + ".tables.json"), "tables"),
                (Path(base_name + ".equations.json"), "equations"),
            ]
            for sidecar_path, root_key in sidecars:
                if not sidecar_path.exists():
                    continue
                try:
                    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    items = payload.get(root_key, {})
                    if isinstance(items, dict):
                        analyze_tasks = []
                        valid_keys = []
                        for item_id, item in items.items():
                            if isinstance(item, dict):
                                valid_keys.append(item_id)
                                analyze_tasks.append(
                                    _analyze_item(root_key, item_id, item)
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

    async def _load_lightrag_document_content(
        self, lightrag_document_path: str
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

    def _resolve_parser_engine(
        self, file_path: str, content_data: dict[str, Any]
    ) -> str:
        explicit_engine = str(content_data.get("parsed_engine") or "").strip().lower()
        if explicit_engine in {"native", "mineru", "docling"}:
            return explicit_engine

        file_name = Path(file_path).name
        m = re.search(r"\.\[([^\]]+)\]\.[^.]+$", file_name)
        if m:
            hint = m.group(1).split("-")[0].strip().lower()
            if hint in {"native", "mineru", "docling"}:
                return hint

        parser_rules = os.getenv("LIGHTRAG_PARSER", "").strip()
        if parser_rules:
            suffix = Path(file_name).suffix.lower().lstrip(".")
            for item in [x.strip() for x in parser_rules.split(",") if x.strip()]:
                if ":" not in item:
                    continue
                pattern, engine_hint = item.split(":", 1)
                pattern = pattern.strip().lower()
                engine = engine_hint.strip().split("-")[0].lower()
                if engine not in {"native", "mineru", "docling"}:
                    continue
                if fnmatch.fnmatch(suffix, pattern):
                    return engine

        # Auto routing for normal user uploads (without filename hints):
        # prefer multimodal-capable engines when corresponding service endpoint is configured.
        suffix = Path(file_name).suffix.lower().lstrip(".")
        mineru_endpoint = os.getenv("MINERU_ENDPOINT", "").strip()
        docling_endpoint = os.getenv("DOCLING_ENDPOINT", "").strip()

        # Keep this mapping conservative and practical:
        # - PDF defaults to MinerU (layout-heavy parsing)
        # - Office-like docs default to Docling
        mineru_suffixes = {"pdf"}
        docling_suffixes = {
            "doc",
            "docx",
            "ppt",
            "pptx",
            "xls",
            "xlsx",
            "md",
            "markdown",
            "html",
            "htm",
        }

        if suffix in mineru_suffixes and mineru_endpoint:
            return "mineru"
        if suffix in docling_suffixes and docling_endpoint:
            return "docling"

        # Fallback cross-coverage when only one service is available.
        if suffix in (mineru_suffixes | docling_suffixes):
            if docling_endpoint:
                return "docling"
            if mineru_endpoint:
                return "mineru"
        return "native"

    def _get_by_path(self, payload: Any, path: str) -> Any:
        if not path:
            return None
        cur = payload
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

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
            task_id = self._get_by_path(upload_payload, id_field)
            if not task_id:
                direct_content = self._get_by_path(upload_payload, content_field)
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
                status_raw = self._get_by_path(poll_payload, status_field)
                status_val = str(status_raw).lower() if status_raw is not None else ""

                if status_val in success_values:
                    result_url = self._get_by_path(poll_payload, result_url_field)
                    if result_url:
                        dl = await client.get(str(result_url))
                        dl.raise_for_status()
                        return dl.text
                    content_val = self._get_by_path(poll_payload, content_field)
                    return str(content_val) if content_val else None
                if status_val in failed_values:
                    raise RuntimeError(
                        f"Parse service failed for task {task_id}: {poll_payload}"
                    )
        raise TimeoutError(f"Parse service polling timeout for task: {task_id}")

    def _extract_content_list_from_payload(
        self, payload: Any
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
                extracted = self._extract_content_list_from_payload(value)
                if extracted is not None:
                    return extracted
            elif isinstance(value, dict):
                extracted = self._extract_content_list_from_payload(value)
                if extracted is not None:
                    return extracted

        # Deep search as fallback
        for value in payload.values():
            extracted = self._extract_content_list_from_payload(value)
            if extracted is not None:
                return extracted
        return None

    def _normalize_parser_result_to_content_list(
        self, parser_result: str | list[dict[str, Any]] | dict[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        """Normalize parser result to structured content list if possible."""
        if parser_result is None:
            return None
        if isinstance(parser_result, list):
            return self._extract_content_list_from_payload(parser_result)
        if isinstance(parser_result, dict):
            return self._extract_content_list_from_payload(parser_result)
        text = str(parser_result).strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            return self._extract_content_list_from_payload(payload)
        except Exception:
            return None

    def _resolve_local_raganything_root(self) -> Path | None:
        """Resolve local RAG-Anything source root if present."""
        env_root = os.getenv("RAGANYTHING_ROOT", "").strip()
        candidates: list[Path] = []
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend(
            [
                Path("/root/autodl-tmp/RAG-Anything"),
                Path.cwd().parent / "RAG-Anything",
            ]
        )
        for c in candidates:
            if (c / "raganything" / "__init__.py").exists():
                return c
        return None

    def _ensure_local_raganything_importable(self) -> bool:
        root = self._resolve_local_raganything_root()
        if root is None:
            return False
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return True

    async def _parse_via_local_raganything(
        self, engine: str, file_path: str
    ) -> list[dict[str, Any]] | None:
        """Use local RAG-Anything parser classes directly."""
        if engine not in {"mineru", "docling"}:
            return None
        if not self._ensure_local_raganything_importable():
            return None
        try:
            from raganything.parser import MineruParser, DoclingParser  # type: ignore[import-untyped]
        except Exception as e:
            logger.info(f"Local RAG-Anything import unavailable: {e}")
            return None

        parser = DoclingParser() if engine == "docling" else MineruParser()
        try:
            if (
                hasattr(parser, "check_installation")
                and not parser.check_installation()
            ):
                logger.info(
                    f"Local RAG-Anything {engine} parser not installed/configured"
                )
                return None
        except Exception:
            return None

        source_file_path = self._resolve_source_file_for_parser(file_path)
        try:
            content_list = await asyncio.to_thread(
                parser.parse_document, source_file_path, "auto", None
            )
            if isinstance(content_list, list):
                return content_list
        except Exception as e:
            logger.warning(f"Local RAG-Anything {engine} parse failed: {e}")
        return None

    def _resolve_source_file_for_parser(self, file_path: str) -> str:
        """Resolve a readable source file path for parser upload."""
        p = Path(file_path)
        if p.exists() and p.is_file():
            return str(p)

        name = p.name
        candidates: list[Path] = []
        input_dir = os.getenv("INPUT_DIR", "").strip()
        if input_dir:
            input_path = Path(input_dir)
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
    ) -> dict[str, Any]:
        """Convert parser content list to LightRAG Document files and return parsed_data."""
        parsed_dir = Path(self.working_dir) / PARSED_DIR_NAME
        parsed_dir.mkdir(parents=True, exist_ok=True)

        source_name = Path(file_path).name or f"{doc_id}.bin"
        base_name = f"{doc_id}.{source_name}"
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
        parsed_time = datetime.now(timezone.utc).isoformat()
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
            "parsed_engine": engine,
            "parsed_time": parsed_time,
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
        await self.full_docs.upsert(
            {
                doc_id: {
                    "content": merged_text,
                    "file_path": file_path,
                    "format": FULL_DOCS_FORMAT_LIGHTRAG,
                    "lightrag_document_path": str(blocks_path),
                    "parsed_engine": engine,
                    "update_time": int(time.time()),
                }
            }
        )
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "format": FULL_DOCS_FORMAT_LIGHTRAG,
            "content": merged_text,
            "blocks_path": str(blocks_path),
        }

    def _content_list_to_plain_text(self, content_list: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            tp = item.get("type")
            if tp in {"text", "equation"}:
                text = item.get("text") or ""
                if text:
                    parts.append(str(text))
            elif tp == "table":
                caption = item.get("table_caption") or []
                body = item.get("table_body") or ""
                parts.append(f"[TABLE] {caption} {body}")
            elif tp == "image":
                caption = item.get("image_caption") or []
                parts.append(f"[IMAGE] {caption}")
        return "\n\n".join([x for x in parts if str(x).strip()])

    async def parse_native(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Phase 1 parse for native/raw, lightrag and pending_parse formats."""
        doc_format = content_data.get("format", FULL_DOCS_FORMAT_RAW)
        if doc_format == FULL_DOCS_FORMAT_LIGHTRAG:
            doc_path = content_data.get("lightrag_document_path") or file_path
            merged_text, blocks_path = await self._load_lightrag_document_content(
                str(doc_path)
            )
            return {
                "doc_id": doc_id,
                "file_path": file_path,
                "format": doc_format,
                "content": merged_text,
                "blocks_path": blocks_path,
            }

        if doc_format == FULL_DOCS_FORMAT_PENDING_PARSE:
            source_path = self._resolve_source_file_for_parser(file_path)
            p = Path(source_path)
            if p.exists() and p.is_file() and p.suffix.lower() == ".docx":
                try:
                    from lightrag.extraction.parse_document import (
                        parse_docx_to_interchange_jsonl,
                    )

                    file_bytes = await asyncio.to_thread(p.read_bytes)
                    parsed_dir = Path(self.working_dir) / PARSED_DIR_NAME
                    parsed_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(parsed_dir)
                    interchange_text = await asyncio.to_thread(
                        parse_docx_to_interchange_jsonl,
                        file_bytes,
                        p.name,
                        doc_id,
                        output_dir,
                    )
                    if interchange_text and interchange_text.strip():
                        await self.full_docs.upsert(
                            {
                                doc_id: {
                                    "content": interchange_text,
                                    "file_path": file_path,
                                    "format": FULL_DOCS_FORMAT_RAW,
                                    "update_time": int(time.time()),
                                }
                            }
                        )
                        logger.info(
                            f"[parse_native] pending_parse completed for {file_path} via interchange JSONL"
                        )
                        return {
                            "doc_id": doc_id,
                            "file_path": file_path,
                            "format": FULL_DOCS_FORMAT_RAW,
                            "content": interchange_text,
                            "blocks_path": "",
                        }
                except Exception as e:
                    logger.warning(
                        f"[parse_native] pending_parse interchange failed for {file_path}: {e}, fallback to basic extraction"
                    )
            if p.exists() and p.is_file():
                try:
                    file_bytes = await asyncio.to_thread(p.read_bytes)
                    from lightrag.api.routers.document_routes import _extract_docx

                    content = await asyncio.to_thread(_extract_docx, file_bytes)
                    await self.full_docs.upsert(
                        {
                            doc_id: {
                                "content": content,
                                "file_path": file_path,
                                "format": FULL_DOCS_FORMAT_RAW,
                                "update_time": int(time.time()),
                            }
                        }
                    )
                    return {
                        "doc_id": doc_id,
                        "file_path": file_path,
                        "format": FULL_DOCS_FORMAT_RAW,
                        "content": content,
                        "blocks_path": "",
                    }
                except Exception as fallback_err:
                    logger.warning(
                        f"[parse_native] pending_parse fallback also failed for {file_path}: {fallback_err}"
                    )
            return {
                "doc_id": doc_id,
                "file_path": file_path,
                "format": FULL_DOCS_FORMAT_RAW,
                "content": content_data.get("content", ""),
                "blocks_path": "",
            }

        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "format": FULL_DOCS_FORMAT_RAW,
            "content": content_data.get("content", ""),
            "blocks_path": "",
        }

    async def parse_mineru(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        endpoint = os.getenv(
            "MINERU_ENDPOINT", "https://mineru.net/api/v4/extract/task"
        ).strip()
        try:
            if endpoint:
                protocol = {
                    "upload_url": endpoint,
                    "poll_url_template": os.getenv(
                        "MINERU_POLL_ENDPOINT",
                        endpoint + "/{trace_id}",
                    ),
                    "poll_method": os.getenv("MINERU_POLL_METHOD", "GET"),
                    "id_field": os.getenv("MINERU_ID_FIELD", "trace_id"),
                    "status_field": os.getenv("MINERU_STATUS_FIELD", "status"),
                    "result_url_field": os.getenv(
                        "MINERU_RESULT_URL_FIELD", "result_url"
                    ),
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
                source_file_path = self._resolve_source_file_for_parser(file_path)
                result_text = await self._call_protocol_parse_service(
                    protocol=protocol,
                    file_path=source_file_path,
                )
                content_list = self._normalize_parser_result_to_content_list(
                    result_text
                )
                if content_list:
                    return await self._write_lightrag_document_from_content_list(
                        doc_id=doc_id,
                        file_path=file_path,
                        content_list=content_list,
                        engine="mineru",
                    )
                if result_text:
                    return {
                        "doc_id": doc_id,
                        "file_path": file_path,
                        "format": FULL_DOCS_FORMAT_RAW,
                        "content": str(result_text),
                        "blocks_path": "",
                    }
        except Exception as e:
            logger.warning(f"MinerU async service failed, fallback local/native: {e}")

        raganything_content_list = await self._parse_via_local_raganything(
            engine="mineru",
            file_path=file_path,
        )
        if raganything_content_list:
            return await self._write_lightrag_document_from_content_list(
                doc_id=doc_id,
                file_path=file_path,
                content_list=raganything_content_list,
                engine="mineru",
            )

        return await self.parse_native(doc_id, file_path, content_data)

    async def parse_docling(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        endpoint = os.getenv(
            "DOCLING_ENDPOINT", "http://localhost:8081/v1/convert/file/async"
        ).strip()
        try:
            if endpoint:
                protocol = {
                    "upload_url": endpoint,
                    "poll_url_template": os.getenv(
                        "DOCLING_POLL_ENDPOINT",
                        endpoint + "/{task_id}",
                    ),
                    "poll_method": os.getenv("DOCLING_POLL_METHOD", "GET"),
                    "id_field": os.getenv("DOCLING_ID_FIELD", "task_id"),
                    "status_field": os.getenv("DOCLING_STATUS_FIELD", "status"),
                    "result_url_field": os.getenv(
                        "DOCLING_RESULT_URL_FIELD", "result_url"
                    ),
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
                source_file_path = self._resolve_source_file_for_parser(file_path)
                result_text = await self._call_protocol_parse_service(
                    protocol=protocol,
                    file_path=source_file_path,
                )
                content_list = self._normalize_parser_result_to_content_list(
                    result_text
                )
                if content_list:
                    return await self._write_lightrag_document_from_content_list(
                        doc_id=doc_id,
                        file_path=file_path,
                        content_list=content_list,
                        engine="docling",
                    )
                if result_text:
                    return {
                        "doc_id": doc_id,
                        "file_path": file_path,
                        "format": FULL_DOCS_FORMAT_RAW,
                        "content": str(result_text),
                        "blocks_path": "",
                    }
        except Exception as e:
            logger.warning(f"Docling async service failed, fallback local/native: {e}")

        raganything_content_list = await self._parse_via_local_raganything(
            engine="docling",
            file_path=file_path,
        )
        if raganything_content_list:
            return await self._write_lightrag_document_from_content_list(
                doc_id=doc_id,
                file_path=file_path,
                content_list=raganything_content_list,
                engine="docling",
            )

        return await self.parse_native(doc_id, file_path, content_data)

    async def _run_multimodal_postprocess_hook(
        self,
        doc_id: str,
        file_path: str,
        chunking_result: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        extraction_meta: dict[str, Any],
    ) -> list[dict[str, Any]] | tuple[dict[str, Any], ...]:
        """Multimodal post-process entrypoint.

        Placement:
            interchange parsing -> [this hook] -> entity extraction

        Default behavior is no-op. This method defines a stable extension point
        for integrating RAG-Anything multimodal processors.
        """
        addon_params = self.addon_params
        if not addon_params.get("enable_multimodal_pipeline", False):
            return chunking_result

        extraction_format = extraction_meta.get("extraction_format")
        capabilities = set(extraction_meta.get("engine_capabilities", []) or [])
        if extraction_format != "interchange_jsonl":
            return chunking_result
        if not capabilities.intersection({"i", "e", "t"}):
            return chunking_result

        logger.info(
            f"[multimodal-hook] enabled for d-id: {doc_id}, file: {file_path}, "
            f"engine={extraction_meta.get('engine')}, caps={sorted(capabilities)}"
        )

        # TODO(RAG-Anything integration):
        # 1) convert interchange chunks -> RAG-Anything content_list
        # 2) call modal processors (image/table/equation) using vlm_llm_model_func (VLM role)
        # 3) merge multimodal outputs back into chunk dicts
        # 4) keep chunk_order_index continuity and chunk_id stability
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

                analysis = (
                    item.get("llm_analyze_result")
                    if isinstance(item.get("llm_analyze_result"), dict)
                    else {}
                )
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

    def _augment_chunk_results_with_mm_entities(
        self,
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
                    extracted_by_chunk.setdefault(source_id, set()).add(
                        str(entity_name)
                    )

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

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
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
        await asyncio.gather(*tasks)

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

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
                file_path = chunk_data.get("file_path", "custom_kg")
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
                file_path = entity_data.get("file_path", "custom_kg")

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

            # Batch insert entities (reduces N serial awaits to 1)
            if entity_nodes:
                await self.chunk_entity_relation_graph.upsert_nodes_batch(entity_nodes)

            # Relationship storage is undirected, so keep only the last update
            # for each endpoint pair regardless of order.
            deduped_relationships: dict[tuple[str, str], dict[str, Any]] = {}
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                relation_key = tuple(sorted((src_id, tgt_id)))
                deduped_relationships.pop(relation_key, None)
                deduped_relationships[relation_key] = relationship_data

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
                file_path = relationship_data.get("file_path", "custom_kg")

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
                update_storage = True

            # Batch insert missing placeholder nodes
            if missing_nodes:
                await self.chunk_entity_relation_graph.upsert_nodes_batch(missing_nodes)

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
                    for rel_id in make_relation_vdb_ids(dp["src_id"], dp["tgt_id"])[1:]
                }
            )

            # Parallel VDB upserts (was serial in original)
            await asyncio.gather(
                self.entities_vdb.upsert(data_for_entities_vdb),
                self.relationships_vdb.upsert(data_for_rels_vdb),
            )

            if legacy_rel_ids_to_delete:
                await self.relationships_vdb.delete(legacy_rel_ids_to_delete)

        except Exception as e:
            logger.error(f"Error in ainsert_custom_kg: {e}")
            raise
        finally:
            if update_storage:
                await self._insert_done()

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
                If param.model_func is provided, it will be used instead of the global model.
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
            history_turns=param.history_turns,
            model_func=param.model_func,
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
                )
            elif param.mode == "bypass":
                # Bypass mode: directly use LLM without knowledge retrieval
                if param.model_func:
                    _warn_deprecated_query_model_func("bypass query generation")
                use_llm_func = (
                    param.model_func or global_config["role_llm_funcs"]["query"]
                )
                # Apply higher priority (8) to entity/relation summary tasks
                use_llm_func = partial(use_llm_func, _priority=8)

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

        backup_cache_ids = _normalize_string_list(
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
            file_path = doc_status_data.get("file_path") if doc_status_data else None
            if not doc_status_data:
                logger.warning(f"Document {doc_id} not found")
                return DeletionResult(
                    status="not_found",
                    doc_id=doc_id,
                    message=f"Document {doc_id} not found.",
                    status_code=404,
                    file_path="",
                )

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
            metadata_cache_ids = _normalize_string_list(
                metadata.get("deletion_llm_cache_ids", []),
                context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
            )
            chunk_ids = set(
                _normalize_string_list(
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
            try:
                deletion_stage = "persist_pre_rebuild_changes"
                await self._insert_done()
            except Exception as e:
                logger.error(f"Failed to persist pre-rebuild changes: {e}")
                raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

            # 8. Rebuild entities and relationships from remaining chunks
            if entities_to_rebuild or relationships_to_rebuild:
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
                    raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

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
