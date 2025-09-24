from __future__ import annotations

import traceback
import asyncio
import configparser
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
)
from lightrag.constants import (
    DEFAULT_MAX_GLEANING,
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
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)


from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    get_graph_db_lock,
    get_data_init_lock,
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
)
from lightrag.namespace import NameSpace
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    _rebuild_knowledge_from_chunks,
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
)
from lightrag.types import KnowledgeGraph
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


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

    # Logging (Deprecated, use setup_logger in utils.py instead)
    # ---
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
        List[Dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_token_size`: The maximum number of tokens per chunk.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_token_size` if not specified.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

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

    # Rerank Configuration
    # ---

    rerank_model_func: Callable[..., object] | None = field(default=None)
    """Function for reranking retrieved documents. All rerank configurations (model name, API keys, top_k, etc.) should be included in this function. Optional."""

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

    max_graph_nodes: int = field(
        default=get_env_value("MAX_GRAPH_NODES", DEFAULT_MAX_GRAPH_NODES, int)
    )
    """Maximum number of graph nodes to return in knowledge graph queries."""

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {
            "language": get_env_value(
                "SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE, str
            ),
            "entity_types": get_env_value("ENTITY_TYPES", DEFAULT_ENTITY_TYPES, list),
        }
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

    def __post_init__(self):
        from lightrag.kg.shared_storage import (
            initialize_share_data,
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
                f"summary_context_size({self.summary_context_size}) should no greater than max_total_tokens({self.max_total_tokens})"
            )
        if self.summary_length_recommended > self.summary_max_tokens:
            logger.warning(
                f"max_total_tokens({self.summary_max_tokens}) should greater than summary_length_recommended({self.summary_length_recommended})"
            )

        # Fix global_config now
        global_config = asdict(self)

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init Embedding
        self.embedding_func = priority_limit_async_func_call(
            self.embedding_func_max_async,
            llm_timeout=self.default_embedding_timeout,
            queue_name="Embedding func",
        )(self.embedding_func)

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

        # Directly use llm_response_cache, don't create a new object
        hashing_kv = self.llm_response_cache

        # Get timeout from LLM model kwargs for dynamic timeout calculation
        self.llm_model_func = priority_limit_async_func_call(
            self.llm_model_max_async,
            llm_timeout=self.default_llm_timeout,
            queue_name="LLM func",
        )(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Storage initialization must be called one by one to prevent deadlock"""
        if self._storages_status == StoragesStatus.CREATED:
            for storage in (
                self.full_docs,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
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
        async with get_data_init_lock(enable_logging=True):
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
                    # Don't raise the error, just log it to avoid breaking initialization

            except Exception as e:
                logger.error(f"Error in data migration check: {e}")
                # Don't raise the error to avoid breaking initialization

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
            new_docs = {doc_key: {"content": full_text}}

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
    ) -> str:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        2. Generate document initial status
        3. Filter out already processed documents
        4. Enqueue document in status

        Args:
            input: Single document string or list of document strings
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated with "enqueue" prefix

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

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
        else:
            # If no file paths provided, use placeholder
            file_paths = ["unknown_source"] * len(input)

        # 1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict and remove duplicates in one pass
            unique_contents = {}
            for id_, doc, path in zip(ids, input, file_paths):
                cleaned_content = sanitize_text_for_encoding(doc)
                if cleaned_content not in unique_contents:
                    unique_contents[cleaned_content] = (id_, path)

            # Reconstruct contents with unique content
            contents = {
                id_: {"content": content, "file_path": file_path}
                for content, (id_, file_path) in unique_contents.items()
            }
        else:
            # Clean input text and remove duplicates in one pass
            unique_content_with_paths = {}
            for doc, path in zip(input, file_paths):
                cleaned_content = sanitize_text_for_encoding(doc)
                if cleaned_content not in unique_content_with_paths:
                    unique_content_with_paths[cleaned_content] = path

            # Generate contents dict of MD5 hash IDs and documents with paths
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                }
                for content, path in unique_content_with_paths.items()
            }

        # 2. Generate document initial status (without content)
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(content_data["content"]),
                "content_length": len(content_data["content"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
                "track_id": track_id,  # Store track_id in document status
            }
            for id_, content_data in contents.items()
        }

        # 3. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already enqueued
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs (documents that were filtered out because they already exist)
        ignored_ids = list(all_new_doc_ids - unique_new_doc_ids)
        if ignored_ids:
            for doc_id in ignored_ids:
                file_path = new_docs.get(doc_id, {}).get("file_path", "unknown_source")
                logger.warning(
                    f"Ignoring document ID (already exists): {doc_id} ({file_path})"
                )
            if len(ignored_ids) > 3:
                logger.warning(
                    f"Total Ignoring {len(ignored_ids)} document IDs that already exist in storage"
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
        #    Store full document content separately
        full_docs_data = {
            doc_id: {"content": contents[doc_id]["content"]}
            for doc_id in new_docs.keys()
        }
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
                    file_path = getattr(status_doc, "file_path", "unknown_source")

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

        # Reset PROCESSING and FAILED documents that pass consistency checks to PENDING status
        docs_to_reset = {}
        reset_count = 0

        for doc_id, status_doc in to_process_docs.items():
            # Check if document has corresponding content in full_docs (consistency check)
            content_data = await self.full_docs.get_by_id(doc_id)
            if content_data:  # Document passes consistency check
                # Check if document is in PROCESSING or FAILED status
                if hasattr(status_doc, "status") and status_doc.status in [
                    DocStatus.PROCESSING,
                    DocStatus.FAILED,
                ]:
                    # Prepare document for status reset to PENDING
                    docs_to_reset[doc_id] = {
                        "status": DocStatus.PENDING,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": getattr(status_doc, "file_path", "unknown_source"),
                        "track_id": getattr(status_doc, "track_id", ""),
                        # Clear any error messages and processing metadata
                        "error_msg": "",
                        "metadata": {},
                    }

                    # Update the status in to_process_docs as well
                    status_doc.status = DocStatus.PENDING
                    reset_count += 1

        # Update doc_status storage if there are documents to reset
        if docs_to_reset:
            await self.doc_status.upsert(docs_to_reset)

            async with pipeline_status_lock:
                reset_message = f"Reset {reset_count} documents from PROCESSING/FAILED to PENDING status"
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
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

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
                ) -> None:
                    """Process single document"""
                    file_extraction_stage_ok = False
                    async with semaphore:
                        nonlocal processed_count
                        current_file_number = 0
                        # Initialize to prevent UnboundLocalError in error handling
                        first_stage_tasks = []
                        entity_relation_task = None
                        try:
                            # Get file path from status document
                            file_path = getattr(
                                status_doc, "file_path", "unknown_source"
                            )

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
                                    pipeline_status["history_messages"] = (
                                        pipeline_status["history_messages"][-5000:]
                                    )

                            # Get document content from full_docs
                            content_data = await self.full_docs.get_by_id(doc_id)
                            if not content_data:
                                raise Exception(
                                    f"Document content not found in full_docs for doc_id: {doc_id}"
                                )
                            content = content_data["content"]

                            # Generate chunks from document
                            chunks: dict[str, Any] = {
                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                    **dp,
                                    "full_doc_id": doc_id,
                                    "file_path": file_path,  # Add file path to each chunk
                                    "llm_cache_list": [],  # Initialize empty LLM cache list for each chunk
                                }
                                for dp in self.chunking_func(
                                    self.tokenizer,
                                    content,
                                    split_by_character,
                                    split_by_character_only,
                                    self.chunk_overlap_token_size,
                                    self.chunk_token_size,
                                )
                            }

                            if not chunks:
                                logger.warning("No document chunks to process")

                            # Record processing start time
                            processing_start_time = int(time.time())

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
                                                "processing_start_time": processing_start_time
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
                            await entity_relation_task
                            file_extraction_stage_ok = True

                        except Exception as e:
                            # Log error and update pipeline status
                            logger.error(traceback.format_exc())
                            error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(
                                    traceback.format_exc()
                                )
                                pipeline_status["history_messages"].append(error_msg)

                            # Cancel tasks that are not yet completed
                            all_tasks = first_stage_tasks + (
                                [entity_relation_task] if entity_relation_task else []
                            )
                            for task in all_tasks:
                                if task and not task.done():
                                    task.cancel()

                            # Persistent llm cache
                            if self.llm_response_cache:
                                await self.llm_response_cache.index_done_callback()

                            # Record processing end time for failed case
                            processing_end_time = int(time.time())

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error_msg": str(e),
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
                                # Get chunk_results from entity_relation_task
                                chunk_results = await entity_relation_task
                                await merge_nodes_and_edges(
                                    chunk_results=chunk_results,  # result collected from entity_relation_task
                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                    entity_vdb=self.entities_vdb,
                                    relationships_vdb=self.relationships_vdb,
                                    global_config=asdict(self),
                                    full_entities_storage=self.full_entities,
                                    full_relations_storage=self.full_relations,
                                    doc_id=doc_id,
                                    pipeline_status=pipeline_status,
                                    pipeline_status_lock=pipeline_status_lock,
                                    llm_response_cache=self.llm_response_cache,
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
                                # Log error and update pipeline status
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

                                # Persistent llm cache
                                if self.llm_response_cache:
                                    await self.llm_response_cache.index_done_callback()

                                # Record processing end time for failed case
                                processing_end_time = int(time.time())

                                # Update document status to failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error_msg": str(e),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now().isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        }
                                    }
                                )

                # Create processing tasks for all documents
                doc_tasks = []
                for doc_id, status_doc in to_process_docs.items():
                    doc_tasks.append(
                        process_document(
                            doc_id,
                            status_doc,
                            split_by_character,
                            split_by_character_only,
                            pipeline_status,
                            pipeline_status_lock,
                            semaphore,
                        )
                    )

                # Wait for all document processing to complete
                await asyncio.gather(*doc_tasks)

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
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Enqueued document processing pipeline stoped"
            logger.info(log_message)
            # Always reset busy status when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    async def _process_extract_entities(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                global_config=asdict(self),
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

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = entity_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_id": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = relationship_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "entity_id": need_insert_id,
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                                "file_path": file_path,
                                "created_at": int(time.time()),
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    },
                )

                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                    "weight": weight,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage with consistent format
            data_for_vdb = {
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
            await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage with consistent format
            data_for_vdb = {
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
            await self.relationships_vdb.upsert(data_for_vdb)

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
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        # If a custom model is provided in param, temporarily update global config
        global_config = asdict(self)

        if param.mode in ["local", "global", "hybrid", "mix"]:
            response = await kg_query(
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
            response = await naive_query(
                query.strip(),
                self.chunks_vdb,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "bypass":
            # Bypass mode: directly use LLM without knowledge retrieval
            use_llm_func = param.model_func or global_config["llm_model_func"]
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
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

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
        global_config = asdict(self)

        if param.mode in ["local", "global", "hybrid", "mix"]:
            logger.debug(f"[aquery_data] Using kg_query for mode: {param.mode}")
            final_data = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                chunks_vdb=self.chunks_vdb,
                return_raw_data=True,  # Get final processed data
            )
        elif param.mode == "naive":
            logger.debug(f"[aquery_data] Using naive_query for mode: {param.mode}")
            final_data = await naive_query(
                query.strip(),
                self.chunks_vdb,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                return_raw_data=True,  # Get final processed data
            )
        elif param.mode == "bypass":
            logger.debug("[aquery_data] Using bypass mode")
            # bypass mode returns empty data using convert_to_user_format
            final_data = convert_to_user_format(
                [],  # no entities
                [],  # no relationships
                [],  # no chunks
                [],  # no references
                "bypass",
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        # Log final result counts - adapt to new data format from convert_to_user_format
        if isinstance(final_data, dict) and "data" in final_data:
            # New format: data is nested under 'data' field
            data_section = final_data["data"]
            entities_count = len(data_section.get("entities", []))
            relationships_count = len(data_section.get("relationships", []))
            chunks_count = len(data_section.get("chunks", []))
        else:
            # Fallback for other formats
            entities_count = len(final_data.get("entities", []))
            relationships_count = len(final_data.get("relationships", []))
            chunks_count = len(final_data.get("chunks", []))

        logger.debug(
            f"[aquery_data] Final result: {entities_count} entities, {relationships_count} relationships, {chunks_count} chunks"
        )

        await self._query_done()
        return final_data

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

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

    async def adelete_by_doc_id(self, doc_id: str) -> DeletionResult:
        """Delete a document and all its related data, including chunks, graph elements, and cached entries.

        This method orchestrates a comprehensive deletion process for a given document ID.
        It ensures that not only the document itself but also all its derived and associated
        data across different storage layers are removed. If entities or relationships are partially affected, it triggers.

        Args:
            doc_id (str): The unique identifier of the document to be deleted.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", or "failure".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
        """
        deletion_operations_started = False
        original_exception = None

        # Get pipeline status shared data and lock for status updates
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

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
            doc_status = doc_status_data.get("status")
            if doc_status != DocStatus.PROCESSED:
                if doc_status == DocStatus.PENDING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                    )
                elif doc_status == DocStatus.PROCESSING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                    )
                elif doc_status == DocStatus.FAILED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                    )
                else:
                    warning_msg = f"Deleting {doc_id} {file_path}(previous status: {doc_status.value})"
                logger.info(warning_msg)
                # Update pipeline status for monitoring
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = warning_msg
                    pipeline_status["history_messages"].append(warning_msg)

            # 2. Get chunk IDs from document status
            chunk_ids = set(doc_status_data.get("chunks_list", []))

            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
                # Mark that deletion operations have started
                deletion_operations_started = True
                try:
                    # Still need to delete the doc status and full doc
                    await self.full_docs.delete([doc_id])
                    await self.doc_status.delete([doc_id])
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

                return DeletionResult(
                    status="success",
                    doc_id=doc_id,
                    message=log_message,
                    status_code=200,
                    file_path=file_path,
                )

            # Mark that deletion operations have started
            deletion_operations_started = True

            # 4. Analyze entities and relationships that will be affected
            entities_to_delete = set()
            entities_to_rebuild = {}  # entity_name -> remaining_chunk_ids
            relationships_to_delete = set()
            relationships_to_rebuild = {}  # (src, tgt) -> remaining_chunk_ids

            try:
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
                    if node_label and "source_id" in node_data:
                        sources = set(node_data["source_id"].split(GRAPH_FIELD_SEP))
                        remaining_sources = sources - chunk_ids

                        if not remaining_sources:
                            entities_to_delete.add(node_label)
                        elif remaining_sources != sources:
                            entities_to_rebuild[node_label] = remaining_sources

                async with pipeline_status_lock:
                    log_message = f"Found {len(entities_to_rebuild)} affected entities"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                # Process relationships
                for edge_data in affected_edges:
                    src = edge_data.get("source")
                    tgt = edge_data.get("target")

                    if src and tgt and "source_id" in edge_data:
                        edge_tuple = tuple(sorted((src, tgt)))
                        if (
                            edge_tuple in relationships_to_delete
                            or edge_tuple in relationships_to_rebuild
                        ):
                            continue

                        sources = set(edge_data["source_id"].split(GRAPH_FIELD_SEP))
                        remaining_sources = sources - chunk_ids

                        if not remaining_sources:
                            relationships_to_delete.add(edge_tuple)
                        elif remaining_sources != sources:
                            relationships_to_rebuild[edge_tuple] = remaining_sources

                async with pipeline_status_lock:
                    log_message = (
                        f"Found {len(relationships_to_rebuild)} affected relations"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

            except Exception as e:
                logger.error(f"Failed to process graph analysis results: {e}")
                raise Exception(f"Failed to process graph dependencies: {e}") from e

            # Use graph database lock to prevent dirty read
            graph_db_lock = get_graph_db_lock(enable_logging=False)
            async with graph_db_lock:
                # 5. Delete chunks from storage
                if chunk_ids:
                    try:
                        await self.chunks_vdb.delete(chunk_ids)
                        await self.text_chunks.delete(chunk_ids)

                        async with pipeline_status_lock:
                            log_message = f"Successfully deleted {len(chunk_ids)} chunks from storage"
                            logger.info(log_message)
                            pipeline_status["latest_message"] = log_message
                            pipeline_status["history_messages"].append(log_message)

                    except Exception as e:
                        logger.error(f"Failed to delete chunks: {e}")
                        raise Exception(f"Failed to delete document chunks: {e}") from e

                # 6. Delete entities that have no remaining sources
                if entities_to_delete:
                    try:
                        # Delete from vector database
                        entity_vdb_ids = [
                            compute_mdhash_id(entity, prefix="ent-")
                            for entity in entities_to_delete
                        ]
                        await self.entities_vdb.delete(entity_vdb_ids)

                        # Delete from graph
                        await self.chunk_entity_relation_graph.remove_nodes(
                            list(entities_to_delete)
                        )

                        async with pipeline_status_lock:
                            log_message = f"Successfully deleted {len(entities_to_delete)} entities"
                            logger.info(log_message)
                            pipeline_status["latest_message"] = log_message
                            pipeline_status["history_messages"].append(log_message)

                    except Exception as e:
                        logger.error(f"Failed to delete entities: {e}")
                        raise Exception(f"Failed to delete entities: {e}") from e

                # 7. Delete relationships that have no remaining sources
                if relationships_to_delete:
                    try:
                        # Delete from vector database
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

                        async with pipeline_status_lock:
                            log_message = f"Successfully deleted {len(relationships_to_delete)} relations"
                            logger.info(log_message)
                            pipeline_status["latest_message"] = log_message
                            pipeline_status["history_messages"].append(log_message)

                    except Exception as e:
                        logger.error(f"Failed to delete relationships: {e}")
                        raise Exception(f"Failed to delete relationships: {e}") from e

                # Persist changes to graph database before releasing graph database lock
                await self._insert_done()

            # 8. Rebuild entities and relationships from remaining chunks
            if entities_to_rebuild or relationships_to_rebuild:
                try:
                    await _rebuild_knowledge_from_chunks(
                        entities_to_rebuild=entities_to_rebuild,
                        relationships_to_rebuild=relationships_to_rebuild,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entities_vdb=self.entities_vdb,
                        relationships_vdb=self.relationships_vdb,
                        text_chunks_storage=self.text_chunks,
                        llm_response_cache=self.llm_response_cache,
                        global_config=asdict(self),
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )

                except Exception as e:
                    logger.error(f"Failed to rebuild knowledge from chunks: {e}")
                    raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

            # 9. Delete from full_entities and full_relations storage
            try:
                await self.full_entities.delete([doc_id])
                await self.full_relations.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete from full_entities/full_relations: {e}")
                raise Exception(
                    f"Failed to delete from full_entities/full_relations: {e}"
                ) from e

            # 10. Delete original document and status
            try:
                await self.full_docs.delete([doc_id])
                await self.doc_status.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

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

                    # If there was no original exception, this persistence error becomes the main error
                    if original_exception is None:
                        return DeletionResult(
                            status="fail",
                            doc_id=doc_id,
                            message=f"Deletion completed but failed to persist changes: {persistence_error}",
                            status_code=500,
                            file_path=file_path,
                        )
                    # If there was an original exception, log the persistence error but don't override the original error
                    # The original error result was already returned in the except block
            else:
                logger.debug(
                    f"No deletion operations were started for document {doc_id}, skipping persistence"
                )

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
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        """Asynchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True

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
        )

    def edit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename)
        )

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

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
