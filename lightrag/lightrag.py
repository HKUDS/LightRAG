from __future__ import annotations

import asyncio
import configparser
import os
import csv
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, cast, final, Literal
import pandas as pd


from lightrag.kg import (
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
)
from .namespace import NameSpace, make_namespace
from .operate import (
    chunking_by_token_size,
    extract_entities,
    kg_query,
    mix_kg_vector_query,
    naive_query,
    query_with_keywords,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .utils import (
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    encode_string_by_tiktoken,
    lazy_external_import,
    limit_async_func_call,
    get_content_summary,
    clean_text,
    check_storage_env_vars,
    logger,
)
from .types import KnowledgeGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    # Directory
    # ---

    working_dir: str = field(
        default=f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
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

    # Logging (Deprecated, use setup_logger in utils.py instead)
    # ---
    log_level: int | None = field(default=None)
    log_file_path: str | None = field(default=None)

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(default=1)
    """Maximum number of entity extraction attempts for ambiguous content."""

    entity_summary_to_max_tokens: int = field(
        default=int(os.getenv("MAX_TOKEN_SUMMARY", 500))
    )

    # Text chunking
    # ---

    chunk_token_size: int = field(default=int(os.getenv("CHUNK_SIZE", 1200)))
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = field(
        default=int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text."""

    """Maximum number of tokens used for summarizing extracted entities."""

    chunking_func: Callable[
        [
            str,
            str | None,
            bool,
            int,
            int,
            str,
        ],
        list[dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:

        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_token_size`: The maximum number of tokens per chunk.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.
        - `tiktoken_model_name`: The name of the tiktoken model to use for tokenization.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_token_size` if not specified.
    """

    # Node embedding
    # ---

    node_embedding_algorithm: str = field(default="node2vec")
    """Algorithm used for node embedding in knowledge graphs."""

    node2vec_params: dict[str, int] = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    """Configuration for the node2vec embedding algorithm:
    - dimensions: Number of dimensions for embeddings.
    - num_walks: Number of random walks per node.
    - walk_length: Number of steps per random walk.
    - window_size: Context window size for training.
    - iterations: Number of iterations for training.
    - random_seed: Seed value for reproducibility.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = field(default=32)
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(default=16)
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

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = field(default=int(os.getenv("MAX_TOKENS", 32768)))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 16)))
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    namespace_prefix: str = field(default="")
    """Prefix for namespacing stored data across different environments."""

    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    # ---

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 20)))
    """Maximum number of parallel insert operations."""

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {
            "language": os.getenv("SUMMARY_LANGUAGE", PROMPTS["DEFAULT_LANGUAGE"])
        }
    )

    # Storages Management
    # ---

    auto_manage_storages_states: bool = field(default=True)
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    # Storages Management
    # ---

    convert_response_to_json_func: Callable[[str], dict[str, Any]] = field(
        default_factory=lambda: convert_response_to_json
    )
    """
    Custom function for converting LLM responses to JSON format.

    The default function is :func:`.utils.convert_response_to_json`.
    """

    cosine_better_than_threshold: float = field(
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )

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

        # Show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init LLM
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
            self.embedding_func
        )

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
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            global_config=asdict(
                self
            ),  # Add global_config to ensure cache works properly
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

        # Directly use llm_response_cache, don't create a new object
        hashing_kv = self.llm_response_cache

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self._storages_status = StoragesStatus.CREATED

        if self.auto_manage_storages_states:
            self._run_async_safely(self.initialize_storages, "Storage Initialization")

    def __del__(self):
        if self.auto_manage_storages_states:
            self._run_async_safely(self.finalize_storages, "Storage Finalization")

    def _run_async_safely(self, async_func, action_name=""):
        """Safely execute an async function, avoiding event loop conflicts."""
        try:
            loop = always_get_an_event_loop()
            if loop.is_running():
                task = loop.create_task(async_func())
                task.add_done_callback(
                    lambda t: logger.info(f"{action_name} completed!")
                )
            else:
                loop.run_until_complete(async_func())
        except RuntimeError:
            logger.warning(
                f"No running event loop, creating a new loop for {action_name}."
            )
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_func())
            loop.close()

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        min_degree: int = 0,
        inclusive: bool = False,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            min_degree (int, optional): Minimum degree of nodes to include. Defaults to 0.
            inclusive (bool, optional): Whether to use inclusive search mode. Defaults to False.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """
        # get params supported by get_knowledge_graph of specified storage
        import inspect

        storage_params = inspect.signature(
            self.chunk_entity_relation_graph.get_knowledge_graph
        ).parameters

        kwargs = {"node_label": node_label, "max_depth": max_depth}

        if "min_degree" in storage_params and min_degree > 0:
            kwargs["min_degree"] = min_degree

        if "inclusive" in storage_params:
            kwargs["inclusive"] = inclusive

        return await self.chunk_entity_relation_graph.get_knowledge_graph(**kwargs)

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(input, split_by_character, split_by_character_only, ids)
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
        """
        await self.apipeline_enqueue_documents(input, ids)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

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
            full_text = clean_text(full_text)
            text_chunks = [clean_text(chunk) for chunk in text_chunks]

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
            for chunk_text in text_chunks:
                chunk_key = compute_mdhash_id(chunk_text, prefix="chunk-")

                inserting_chunks[chunk_key] = {
                    "content": chunk_text,
                    "full_doc_id": doc_key,
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
                self._process_entity_relation_graph(inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done()

    async def apipeline_enqueue_documents(
        self, input: str | list[str], ids: list[str] | None = None
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]

        # 1. Validate ids if provided or generate MD5 hash IDs
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict of IDs provided by user and documents
            contents = {id_: doc for id_, doc in zip(ids, input)}
        else:
            # Clean input text and remove duplicates
            input = list(set(clean_text(doc) for doc in input))
            # Generate contents dict of MD5 hash IDs and documents
            contents = {compute_mdhash_id(doc, prefix="doc-"): doc for doc in input}

        # 2. Remove duplicate contents
        unique_contents = {
            id_: content
            for content, id_ in {
                content: id_ for id_, content in contents.items()
            }.items()
        }

        # 3. Generate document initial status
        new_docs: dict[str, Any] = {
            id_: {
                "content": content,
                "content_summary": get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for id_, content in unique_contents.items()
        }

        # 4. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs
        ignored_ids = [
            doc_id for doc_id in unique_new_doc_ids if doc_id not in new_docs
        ]
        if ignored_ids:
            logger.warning(
                f"Ignoring {len(ignored_ids)} document IDs not found in new_docs"
            )
            for doc_id in ignored_ids:
                logger.warning(f"Ignored document ID: {doc_id}")

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 5. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

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
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                # 先检查是否有需要处理的文档
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                # 如果没有需要处理的文档，直接返回，保留 pipeline_status 中的内容不变
                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                # 有文档需要处理，更新 pipeline_status
                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "indexing files",
                        "job_start": datetime.now().isoformat(),
                        "docs": 0,
                        "batchs": 0,
                        "cur_batch": 0,
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
                    log_message = "All documents have been processed or are duplicates"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                # 2. split docs into chunks, insert chunks, update doc status
                docs_batches = [
                    list(to_process_docs.items())[i : i + self.max_parallel_insert]
                    for i in range(0, len(to_process_docs), self.max_parallel_insert)
                ]

                log_message = f"Number of batches to process: {len(docs_batches)}."
                logger.info(log_message)

                # Update pipeline status with current batch information
                pipeline_status["docs"] += len(to_process_docs)
                pipeline_status["batchs"] += len(docs_batches)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                batches: list[Any] = []
                # 3. iterate over batches
                for batch_idx, docs_batch in enumerate(docs_batches):
                    # Update current batch in pipeline status (directly, as it's atomic)
                    pipeline_status["cur_batch"] += 1

                    async def batch(
                        batch_idx: int,
                        docs_batch: list[tuple[str, DocProcessingStatus]],
                        size_batch: int,
                    ) -> None:
                        log_message = (
                            f"Start processing batch {batch_idx + 1} of {size_batch}."
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)
                        # 4. iterate over batch
                        for doc_id_processing_status in docs_batch:
                            doc_id, status_doc = doc_id_processing_status
                            # Generate chunks from document
                            chunks: dict[str, Any] = {
                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                    **dp,
                                    "full_doc_id": doc_id,
                                }
                                for dp in self.chunking_func(
                                    status_doc.content,
                                    split_by_character,
                                    split_by_character_only,
                                    self.chunk_overlap_token_size,
                                    self.chunk_token_size,
                                    self.tiktoken_model_name,
                                )
                            }
                            # Process document (text chunks and full docs) in parallel
                            # Create tasks with references for potential cancellation
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSING,
                                            "updated_at": datetime.now().isoformat(),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                        }
                                    }
                                )
                            )
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks)
                            )
                            entity_relation_task = asyncio.create_task(
                                self._process_entity_relation_graph(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            full_docs_task = asyncio.create_task(
                                self.full_docs.upsert(
                                    {doc_id: {"content": status_doc.content}}
                                )
                            )
                            text_chunks_task = asyncio.create_task(
                                self.text_chunks.upsert(chunks)
                            )
                            tasks = [
                                doc_status_task,
                                chunks_vdb_task,
                                entity_relation_task,
                                full_docs_task,
                                text_chunks_task,
                            ]
                            try:
                                await asyncio.gather(*tasks)
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSED,
                                            "chunks_count": len(chunks),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now().isoformat(),
                                        }
                                    }
                                )
                            except Exception as e:
                                # Log error and update pipeline status
                                error_msg = (
                                    f"Failed to process document {doc_id}: {str(e)}"
                                )
                                logger.error(error_msg)
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)

                                # Cancel other tasks as they are no longer meaningful
                                for task in [
                                    chunks_vdb_task,
                                    entity_relation_task,
                                    full_docs_task,
                                    text_chunks_task,
                                ]:
                                    if not task.done():
                                        task.cancel()

                                # Update document status to failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error": str(e),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now().isoformat(),
                                        }
                                    }
                                )
                                continue
                        log_message = (
                            f"Completed batch {batch_idx + 1} of {len(docs_batches)}."
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                    batches.append(batch(batch_idx, docs_batch, len(docs_batches)))

                await asyncio.gather(*batches)
                await self._insert_done()

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
            log_message = "Document processing pipeline completed"
            logger.info(log_message)
            # Always reset busy status when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        try:
            await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
        except Exception as e:
            logger.error("Failed to extract entities and relationships")
            raise e

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        log_message = "All Insert done"
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
        self, custom_kg: dict[str, Any], full_doc_id: str = None
    ) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = clean_text(chunk_data["content"])
                source_id = chunk_data["source_id"]
                tokens = len(
                    encode_string_by_tiktoken(
                        chunk_content, model_name=self.tiktoken_model_name
                    )
                )
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
                    },
                )
                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                    "weight": weight,
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
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,  # Directly use llm_response_cache
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query.strip(),
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,  # Directly use llm_response_cache
                system_prompt=system_prompt,
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,  # Directly use llm_response_cache
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    def query_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        Query with separate keyword extraction step.

        This method extracts keywords from the query first, then uses them for the query.

        Args:
            query: User query
            prompt: Additional prompt for the query
            param: Query parameters

        Returns:
            Query response
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(query, prompt, param)
        )

    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> str | AsyncIterator[str]:
        """
        Async version of query_with_separate_keyword_extraction.

        Args:
            query: User query
            prompt: Additional prompt for the query
            param: Query parameters

        Returns:
            Query response or async iterator
        """
        response = await query_with_keywords(
            query=query,
            prompt=prompt,
            param=param,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            chunks_vdb=self.chunks_vdb,
            text_chunks_db=self.text_chunks,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache,
        )

        await self._query_done()
        return response

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str) -> None:
        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_entity_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    def delete_by_relation(self, source_entity: str, target_entity: str) -> None:
        """Synchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.adelete_by_relation(source_entity, target_entity)
        )

    async def adelete_by_relation(self, source_entity: str, target_entity: str) -> None:
        """Asynchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
        """
        try:
            # Check if the relation exists
            edge_exists = await self.chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                logger.warning(
                    f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                )
                return

            # Delete relation from vector database
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            await self.relationships_vdb.delete([relation_id])

            # Delete relation from knowledge graph
            await self.chunk_entity_relation_graph.remove_edges(
                [(source_entity, target_entity)]
            )

            logger.info(
                f"Successfully deleted relation from '{source_entity}' to '{target_entity}'"
            )
            await self._delete_relation_done()
        except Exception as e:
            logger.error(
                f"Error while deleting relation from '{source_entity}' to '{target_entity}': {e}"
            )

    async def _delete_relation_done(self) -> None:
        """Callback after relation deletion is complete"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get_by_id(doc_id)
            if not doc_status:
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all chunks related to this document
            # Find all chunks where full_doc_id equals the current doc_id
            all_chunks = await self.text_chunks.get_all()
            related_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in all_chunks.items()
                if isinstance(chunk_data, dict)
                and chunk_data.get("full_doc_id") == doc_id
            }

            if not related_chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return

            # Get all related chunk IDs
            chunk_ids = set(related_chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities_storage = await self.entities_vdb.client_storage
                entities = [
                    dp
                    for dp in entities_storage["data"]
                    if chunk_id in dp.get("source_id")
                ]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relationships_storage = await self.relationships_vdb.client_storage
                relations = [
                    dp
                    for dp in relationships_storage["data"]
                    if chunk_id in dp.get("source_id")
                ]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes and edges from the graph storage using storage-agnostic methods
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities - use storage-agnostic methods
            all_labels = await self.chunk_entity_relation_graph.get_all_labels()
            for node_label in all_labels:
                node_data = await self.chunk_entity_relation_graph.get_node(node_label)
                if node_data and "source_id" in node_data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(node_data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node_label)
                        logger.debug(
                            f"Entity {node_label} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node_label] = new_source_id
                        logger.debug(
                            f"Entity {node_label} will be updated with new source_id: {new_source_id}"
                        )

            # Process relationships
            for node_label in all_labels:
                node_edges = await self.chunk_entity_relation_graph.get_node_edges(
                    node_label
                )
                if node_edges:
                    for src, tgt in node_edges:
                        edge_data = await self.chunk_entity_relation_graph.get_edge(
                            src, tgt
                        )
                        if edge_data and "source_id" in edge_data:
                            # Split source_id using GRAPH_FIELD_SEP
                            sources = set(edge_data["source_id"].split(GRAPH_FIELD_SEP))
                            sources.difference_update(chunk_ids)
                            if not sources:
                                relationships_to_delete.add((src, tgt))
                                logger.debug(
                                    f"Relationship {src}-{tgt} marked for deletion - no remaining sources"
                                )
                            else:
                                new_source_id = GRAPH_FIELD_SEP.join(sources)
                                relationships_to_update[(src, tgt)] = new_source_id
                                logger.debug(
                                    f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                                )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logger.debug(f"Deleted entity {entity} from vector DB")
                await self.chunk_entity_relation_graph.remove_nodes(
                    list(entities_to_delete)
                )
                logger.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = await self.chunk_entity_relation_graph.get_node(entity)
                if node_data:
                    node_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_node(
                        entity, node_data
                    )
                    logger.debug(
                        f"Updated entity {entity} with new source_id: {new_source_id}"
                    )

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logger.debug(f"Deleted relationship {src}-{tgt} from vector DB")
                await self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logger.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = await self.chunk_entity_relation_graph.get_edge(src, tgt)
                if edge_data:
                    edge_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_edge(
                        src, tgt, edge_data
                    )
                    logger.debug(
                        f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}"
                    )

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            async def process_data(data_type, vdb, chunk_id):
                # Check data (entities or relationships)
                storage = await vdb.client_storage
                data_with_chunk = [
                    dp
                    for dp in storage["data"]
                    if chunk_id in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                ]

                data_for_vdb = {}
                if data_with_chunk:
                    logger.warning(
                        f"found {len(data_with_chunk)} {data_type} still referencing chunk {chunk_id}"
                    )

                    for item in data_with_chunk:
                        old_sources = item["source_id"].split(GRAPH_FIELD_SEP)
                        new_sources = [src for src in old_sources if src != chunk_id]

                        if not new_sources:
                            logger.info(
                                f"{data_type} {item.get('entity_name', 'N/A')} is deleted because source_id is not exists"
                            )
                            await vdb.delete_entity(item)
                        else:
                            item["source_id"] = GRAPH_FIELD_SEP.join(new_sources)
                            item_id = item["__id__"]
                            data_for_vdb[item_id] = item.copy()
                            if data_type == "entities":
                                data_for_vdb[item_id]["content"] = data_for_vdb[
                                    item_id
                                ].get("content") or (
                                    item.get("entity_name", "")
                                    + (item.get("description") or "")
                                )
                            else:  # relationships
                                data_for_vdb[item_id]["content"] = data_for_vdb[
                                    item_id
                                ].get("content") or (
                                    (item.get("keywords") or "")
                                    + (item.get("src_id") or "")
                                    + (item.get("tgt_id") or "")
                                    + (item.get("description") or "")
                                )

                    if data_for_vdb:
                        await vdb.upsert(data_for_vdb)
                        logger.info(f"Successfully updated {data_type} in vector DB")

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.warning(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                all_remaining_chunks = await self.text_chunks.get_all()
                remaining_related_chunks = {
                    chunk_id: chunk_data
                    for chunk_id, chunk_data in all_remaining_chunks.items()
                    if isinstance(chunk_data, dict)
                    and chunk_data.get("full_doc_id") == doc_id
                }

                if remaining_related_chunks:
                    logger.warning(
                        f"Found {len(remaining_related_chunks)} remaining chunks"
                    )

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    await process_data("entities", self.entities_vdb, chunk_id)
                    await process_data(
                        "relationships", self.relationships_vdb, chunk_id
                    )

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity"""

        # Get information from the graph
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = await self.entities_vdb.get_by_id(entity_id)
            result["vector_data"] = vector_data

        return result

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship"""

        # Get information from the graph
        edge_data = await self.chunk_entity_relation_graph.get_edge(
            src_entity, tgt_entity
        )
        source_id = edge_data.get("source_id") if edge_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "src_entity": src_entity,
            "tgt_entity": tgt_entity,
            "source_id": source_id,
            "graph_data": edge_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            vector_data = await self.relationships_vdb.get_by_id(rel_id)
            result["vector_data"] = vector_data

        return result

    def check_storage_env_vars(self, storage_name: str) -> None:
        """Check if all required environment variables for storage implementation exist

        Args:
            storage_name: Storage implementation name

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Storage implementation '{storage_name}' requires the following "
                f"environment variables: {', '.join(missing_vars)}"
            )

    async def aclear_cache(self, modes: list[str] | None = None) -> None:
        """Clear cache data from the LLM response cache storage.

        Args:
            modes (list[str] | None): Modes of cache to clear. Options: ["default", "naive", "local", "global", "hybrid", "mix"].
                             "default" represents extraction cache.
                             If None, clears all cache.

        Example:
            # Clear all cache
            await rag.aclear_cache()

            # Clear local mode cache
            await rag.aclear_cache(modes=["local"])

            # Clear extraction cache
            await rag.aclear_cache(modes=["default"])
        """
        if not self.llm_response_cache:
            logger.warning("No cache storage configured")
            return

        valid_modes = ["default", "naive", "local", "global", "hybrid", "mix"]

        # Validate input
        if modes and not all(mode in valid_modes for mode in modes):
            raise ValueError(f"Invalid mode. Valid modes are: {valid_modes}")

        try:
            # Reset the cache storage for specified mode
            if modes:
                await self.llm_response_cache.delete(modes)
                logger.info(f"Cleared cache for modes: {modes}")
            else:
                # Clear all modes
                await self.llm_response_cache.delete(valid_modes)
                logger.info("Cleared all cache")

            await self.llm_response_cache.index_done_callback()

        except Exception as e:
            logger.error(f"Error while clearing cache: {e}")

    def clear_cache(self, modes: list[str] | None = None) -> None:
        """Synchronous version of aclear_cache."""
        return always_get_an_event_loop().run_until_complete(self.aclear_cache(modes))

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
        try:
            # 1. Get current entity information
            node_exists = await self.chunk_entity_relation_graph.has_node(entity_name)
            if not node_exists:
                raise ValueError(f"Entity '{entity_name}' does not exist")
            node_data = await self.chunk_entity_relation_graph.get_node(entity_name)

            # Check if entity is being renamed
            new_entity_name = updated_data.get("entity_name", entity_name)
            is_renaming = new_entity_name != entity_name

            # If renaming, check if new name already exists
            if is_renaming:
                if not allow_rename:
                    raise ValueError(
                        "Entity renaming is not allowed. Set allow_rename=True to enable this feature"
                    )

                existing_node = await self.chunk_entity_relation_graph.has_node(
                    new_entity_name
                )
                if existing_node:
                    raise ValueError(
                        f"Entity name '{new_entity_name}' already exists, cannot rename"
                    )

            # 2. Update entity information in the graph
            new_node_data = {**node_data, **updated_data}
            if "entity_name" in new_node_data:
                del new_node_data[
                    "entity_name"
                ]  # Node data should not contain entity_name field

            # If renaming entity
            if is_renaming:
                logger.info(f"Renaming entity '{entity_name}' to '{new_entity_name}'")

                # Create new entity
                await self.chunk_entity_relation_graph.upsert_node(
                    new_entity_name, new_node_data
                )

                # Store relationships that need to be updated
                relations_to_update = []

                # Get all edges related to the original entity
                edges = await self.chunk_entity_relation_graph.get_node_edges(
                    entity_name
                )
                if edges:
                    # Recreate edges for the new entity
                    for source, target in edges:
                        edge_data = await self.chunk_entity_relation_graph.get_edge(
                            source, target
                        )
                        if edge_data:
                            if source == entity_name:
                                await self.chunk_entity_relation_graph.upsert_edge(
                                    new_entity_name, target, edge_data
                                )
                                relations_to_update.append(
                                    (new_entity_name, target, edge_data)
                                )
                            else:  # target == entity_name
                                await self.chunk_entity_relation_graph.upsert_edge(
                                    source, new_entity_name, edge_data
                                )
                                relations_to_update.append(
                                    (source, new_entity_name, edge_data)
                                )

                # Delete old entity
                await self.chunk_entity_relation_graph.delete_node(entity_name)

                # Delete old entity record from vector database
                old_entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.delete([old_entity_id])
                logger.info(
                    f"Deleted old entity '{entity_name}' and its vector embedding from database"
                )

                # Update relationship vector representations
                for src, tgt, edge_data in relations_to_update:
                    description = edge_data.get("description", "")
                    keywords = edge_data.get("keywords", "")
                    source_id = edge_data.get("source_id", "")
                    weight = float(edge_data.get("weight", 1.0))

                    # Create new content for embedding
                    content = f"{src}\t{tgt}\n{keywords}\n{description}"

                    # Calculate relationship ID
                    relation_id = compute_mdhash_id(src + tgt, prefix="rel-")

                    # Prepare data for vector database update
                    relation_data = {
                        relation_id: {
                            "content": content,
                            "src_id": src,
                            "tgt_id": tgt,
                            "source_id": source_id,
                            "description": description,
                            "keywords": keywords,
                            "weight": weight,
                        }
                    }

                    # Update vector database
                    await self.relationships_vdb.upsert(relation_data)

                # Update working entity name to new name
                entity_name = new_entity_name
            else:
                # If not renaming, directly update node data
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, new_node_data
                )

            # 3. Recalculate entity's vector representation and update vector database
            description = new_node_data.get("description", "")
            source_id = new_node_data.get("source_id", "")
            entity_type = new_node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            # Calculate entity ID
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Prepare data for vector database update
            entity_data = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            # Update vector database
            await self.entities_vdb.upsert(entity_data)

            # 4. Save changes
            await self._edit_entity_done()

            logger.info(f"Entity '{entity_name}' successfully updated")
            return await self.get_entity_info(entity_name, include_vector_data=True)
        except Exception as e:
            logger.error(f"Error while editing entity '{entity_name}': {e}")
            raise

    def edit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        """Synchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True

        Returns:
            Dictionary containing updated entity information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename)
        )

    async def _edit_entity_done(self) -> None:
        """Callback after entity editing is complete, ensures updates are persisted"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
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
        try:
            # 1. Get current relation information
            edge_exists = await self.chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                raise ValueError(
                    f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                )
            edge_data = await self.chunk_entity_relation_graph.get_edge(
                source_entity, target_entity
            )
            # Important: First delete the old relation record from the vector database
            old_relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            await self.relationships_vdb.delete([old_relation_id])
            logger.info(
                f"Deleted old relation record from vector database for relation {source_entity} -> {target_entity}"
            )

            # 2. Update relation information in the graph
            new_edge_data = {**edge_data, **updated_data}
            await self.chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, new_edge_data
            )

            # 3. Recalculate relation's vector representation and update vector database
            description = new_edge_data.get("description", "")
            keywords = new_edge_data.get("keywords", "")
            source_id = new_edge_data.get("source_id", "")
            weight = float(new_edge_data.get("weight", 1.0))

            # Create content for embedding
            content = f"{source_entity}\t{target_entity}\n{keywords}\n{description}"

            # Calculate relation ID
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )

            # Prepare data for vector database update
            relation_data = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                }
            }

            # Update vector database
            await self.relationships_vdb.upsert(relation_data)

            # 4. Save changes
            await self._edit_relation_done()

            logger.info(
                f"Relation from '{source_entity}' to '{target_entity}' successfully updated"
            )
            return await self.get_relation_info(
                source_entity, target_entity, include_vector_data=True
            )
        except Exception as e:
            logger.error(
                f"Error while editing relation from '{source_entity}' to '{target_entity}': {e}"
            )
            raise

    def edit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_relation(source_entity, target_entity, updated_data)
        )

    async def _edit_relation_done(self) -> None:
        """Callback after relation editing is complete, ensures updates are persisted"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
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
        try:
            # Check if entity already exists
            existing_node = await self.chunk_entity_relation_graph.has_node(entity_name)
            if existing_node:
                raise ValueError(f"Entity '{entity_name}' already exists")

            # Prepare node data with defaults if missing
            node_data = {
                "entity_id": entity_name,
                "entity_type": entity_data.get("entity_type", "UNKNOWN"),
                "description": entity_data.get("description", ""),
                "source_id": entity_data.get("source_id", "manual"),
            }

            # Add entity to knowledge graph
            await self.chunk_entity_relation_graph.upsert_node(entity_name, node_data)

            # Prepare content for entity
            description = node_data.get("description", "")
            source_id = node_data.get("source_id", "")
            entity_type = node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            # Calculate entity ID
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Prepare data for vector database update
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            # Update vector database
            await self.entities_vdb.upsert(entity_data_for_vdb)

            # Save changes
            await self._edit_entity_done()

            logger.info(f"Entity '{entity_name}' successfully created")
            return await self.get_entity_info(entity_name, include_vector_data=True)
        except Exception as e:
            logger.error(f"Error while creating entity '{entity_name}': {e}")
            raise

    def create_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronously create a new entity.

        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
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
        try:
            # Check if both entities exist
            source_exists = await self.chunk_entity_relation_graph.has_node(
                source_entity
            )
            target_exists = await self.chunk_entity_relation_graph.has_node(
                target_entity
            )

            if not source_exists:
                raise ValueError(f"Source entity '{source_entity}' does not exist")
            if not target_exists:
                raise ValueError(f"Target entity '{target_entity}' does not exist")

            # Check if relation already exists
            existing_edge = await self.chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if existing_edge:
                raise ValueError(
                    f"Relation from '{source_entity}' to '{target_entity}' already exists"
                )

            # Prepare edge data with defaults if missing
            edge_data = {
                "description": relation_data.get("description", ""),
                "keywords": relation_data.get("keywords", ""),
                "source_id": relation_data.get("source_id", "manual"),
                "weight": float(relation_data.get("weight", 1.0)),
            }

            # Add relation to knowledge graph
            await self.chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, edge_data
            )

            # Prepare content for embedding
            description = edge_data.get("description", "")
            keywords = edge_data.get("keywords", "")
            source_id = edge_data.get("source_id", "")
            weight = edge_data.get("weight", 1.0)

            # Create content for embedding
            content = f"{keywords}\t{source_entity}\n{target_entity}\n{description}"

            # Calculate relation ID
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )

            # Prepare data for vector database update
            relation_data_for_vdb = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                }
            }

            # Update vector database
            await self.relationships_vdb.upsert(relation_data_for_vdb)

            # Save changes
            await self._edit_relation_done()

            logger.info(
                f"Relation from '{source_entity}' to '{target_entity}' successfully created"
            )
            return await self.get_relation_info(
                source_entity, target_entity, include_vector_data=True
            )
        except Exception as e:
            logger.error(
                f"Error while creating relation from '{source_entity}' to '{target_entity}': {e}"
            )
            raise

    def create_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronously create a new relation between entities.

        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
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
        try:
            # Default merge strategy
            default_strategy = {
                "description": "concatenate",
                "entity_type": "keep_first",
                "source_id": "join_unique",
            }

            merge_strategy = (
                default_strategy
                if merge_strategy is None
                else {**default_strategy, **merge_strategy}
            )
            target_entity_data = (
                {} if target_entity_data is None else target_entity_data
            )

            # 1. Check if all source entities exist
            source_entities_data = {}
            for entity_name in source_entities:
                node_exists = await self.chunk_entity_relation_graph.has_node(
                    entity_name
                )
                if not node_exists:
                    raise ValueError(f"Source entity '{entity_name}' does not exist")
                node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
                source_entities_data[entity_name] = node_data

            # 2. Check if target entity exists and get its data if it does
            target_exists = await self.chunk_entity_relation_graph.has_node(
                target_entity
            )
            existing_target_entity_data = {}
            if target_exists:
                existing_target_entity_data = (
                    await self.chunk_entity_relation_graph.get_node(target_entity)
                )
                logger.info(
                    f"Target entity '{target_entity}' already exists, will merge data"
                )

            # 3. Merge entity data
            merged_entity_data = self._merge_entity_attributes(
                list(source_entities_data.values())
                + ([existing_target_entity_data] if target_exists else []),
                merge_strategy,
            )

            # Apply any explicitly provided target entity data (overrides merged data)
            for key, value in target_entity_data.items():
                merged_entity_data[key] = value

            # 4. Get all relationships of the source entities
            all_relations = []
            for entity_name in source_entities:
                # Get all relationships where this entity is the source
                outgoing_edges = await self.chunk_entity_relation_graph.get_node_edges(
                    entity_name
                )
                if outgoing_edges:
                    for src, tgt in outgoing_edges:
                        # Ensure src is the current entity
                        if src == entity_name:
                            edge_data = await self.chunk_entity_relation_graph.get_edge(
                                src, tgt
                            )
                            all_relations.append(("outgoing", src, tgt, edge_data))

                # Get all relationships where this entity is the target
                incoming_edges = []
                all_labels = await self.chunk_entity_relation_graph.get_all_labels()
                for label in all_labels:
                    if label == entity_name:
                        continue
                    node_edges = await self.chunk_entity_relation_graph.get_node_edges(
                        label
                    )
                    for src, tgt in node_edges or []:
                        if tgt == entity_name:
                            incoming_edges.append((src, tgt))

                for src, tgt in incoming_edges:
                    edge_data = await self.chunk_entity_relation_graph.get_edge(
                        src, tgt
                    )
                    all_relations.append(("incoming", src, tgt, edge_data))

            # 5. Create or update the target entity
            if not target_exists:
                await self.chunk_entity_relation_graph.upsert_node(
                    target_entity, merged_entity_data
                )
                logger.info(f"Created new target entity '{target_entity}'")
            else:
                await self.chunk_entity_relation_graph.upsert_node(
                    target_entity, merged_entity_data
                )
                logger.info(f"Updated existing target entity '{target_entity}'")

            # 6. Recreate all relationships, pointing to the target entity
            relation_updates = {}  # Track relationships that need to be merged

            for rel_type, src, tgt, edge_data in all_relations:
                new_src = target_entity if src in source_entities else src
                new_tgt = target_entity if tgt in source_entities else tgt

                # Skip relationships between source entities to avoid self-loops
                if new_src == new_tgt:
                    logger.info(
                        f"Skipping relationship between source entities: {src} -> {tgt} to avoid self-loop"
                    )
                    continue

                # Check if the same relationship already exists
                relation_key = f"{new_src}|{new_tgt}"
                if relation_key in relation_updates:
                    # Merge relationship data
                    existing_data = relation_updates[relation_key]["data"]
                    merged_relation = self._merge_relation_attributes(
                        [existing_data, edge_data],
                        {
                            "description": "concatenate",
                            "keywords": "join_unique",
                            "source_id": "join_unique",
                            "weight": "max",
                        },
                    )
                    relation_updates[relation_key]["data"] = merged_relation
                    logger.info(
                        f"Merged duplicate relationship: {new_src} -> {new_tgt}"
                    )
                else:
                    relation_updates[relation_key] = {
                        "src": new_src,
                        "tgt": new_tgt,
                        "data": edge_data.copy(),
                    }

            # Apply relationship updates
            for rel_data in relation_updates.values():
                await self.chunk_entity_relation_graph.upsert_edge(
                    rel_data["src"], rel_data["tgt"], rel_data["data"]
                )
                logger.info(
                    f"Created or updated relationship: {rel_data['src']} -> {rel_data['tgt']}"
                )

            # 7. Update entity vector representation
            description = merged_entity_data.get("description", "")
            source_id = merged_entity_data.get("source_id", "")
            entity_type = merged_entity_data.get("entity_type", "")
            content = target_entity + "\n" + description

            entity_id = compute_mdhash_id(target_entity, prefix="ent-")
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            await self.entities_vdb.upsert(entity_data_for_vdb)

            # 8. Update relationship vector representations
            for rel_data in relation_updates.values():
                src = rel_data["src"]
                tgt = rel_data["tgt"]
                edge_data = rel_data["data"]

                description = edge_data.get("description", "")
                keywords = edge_data.get("keywords", "")
                source_id = edge_data.get("source_id", "")
                weight = float(edge_data.get("weight", 1.0))

                content = f"{keywords}\t{src}\n{tgt}\n{description}"
                relation_id = compute_mdhash_id(src + tgt, prefix="rel-")

                relation_data_for_vdb = {
                    relation_id: {
                        "content": content,
                        "src_id": src,
                        "tgt_id": tgt,
                        "source_id": source_id,
                        "description": description,
                        "keywords": keywords,
                        "weight": weight,
                    }
                }

                await self.relationships_vdb.upsert(relation_data_for_vdb)

            # 9. Delete source entities
            for entity_name in source_entities:
                if entity_name == target_entity:
                    logger.info(
                        f"Skipping deletion of '{entity_name}' as it's also the target entity"
                    )
                    continue

                # Delete entity node from knowledge graph
                await self.chunk_entity_relation_graph.delete_node(entity_name)

                # Delete entity record from vector database
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.delete([entity_id])

                # Also ensure any relationships specific to this entity are deleted from vector DB
                # This is a safety check, as these should have been transformed to the target entity already
                entity_relation_prefix = compute_mdhash_id(entity_name, prefix="rel-")
                relations_with_entity = await self.relationships_vdb.search_by_prefix(
                    entity_relation_prefix
                )
                if relations_with_entity:
                    relation_ids = [r["id"] for r in relations_with_entity]
                    await self.relationships_vdb.delete(relation_ids)
                    logger.info(
                        f"Deleted {len(relation_ids)} relation records for entity '{entity_name}' from vector database"
                    )

                logger.info(
                    f"Deleted source entity '{entity_name}' and its vector embedding from database"
                )

            # 10. Save changes
            await self._merge_entities_done()

            logger.info(
                f"Successfully merged {len(source_entities)} entities into '{target_entity}'"
            )
            return await self.get_entity_info(target_entity, include_vector_data=True)

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            raise

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
        # Collect data
        entities_data = []
        relations_data = []
        relationships_data = []

        # --- Entities ---
        all_entities = await self.chunk_entity_relation_graph.get_all_labels()
        for entity_name in all_entities:
            entity_info = await self.get_entity_info(
                entity_name, include_vector_data=include_vector_data
            )
            entity_row = {
                "entity_name": entity_name,
                "source_id": entity_info["source_id"],
                "graph_data": str(
                    entity_info["graph_data"]
                ),  # Convert to string to ensure compatibility
            }
            if include_vector_data and "vector_data" in entity_info:
                entity_row["vector_data"] = str(entity_info["vector_data"])
            entities_data.append(entity_row)

        # --- Relations ---
        for src_entity in all_entities:
            for tgt_entity in all_entities:
                if src_entity == tgt_entity:
                    continue

                edge_exists = await self.chunk_entity_relation_graph.has_edge(
                    src_entity, tgt_entity
                )
                if edge_exists:
                    relation_info = await self.get_relation_info(
                        src_entity, tgt_entity, include_vector_data=include_vector_data
                    )
                    relation_row = {
                        "src_entity": src_entity,
                        "tgt_entity": tgt_entity,
                        "source_id": relation_info["source_id"],
                        "graph_data": str(
                            relation_info["graph_data"]
                        ),  # Convert to string
                    }
                    if include_vector_data and "vector_data" in relation_info:
                        relation_row["vector_data"] = str(relation_info["vector_data"])
                    relations_data.append(relation_row)

        # --- Relationships (from VectorDB) ---
        all_relationships = await self.relationships_vdb.client_storage
        for rel in all_relationships["data"]:
            relationships_data.append(
                {
                    "relationship_id": rel["__id__"],
                    "data": str(rel),  # Convert to string for compatibility
                }
            )

        # Export based on format
        if file_format == "csv":
            # CSV export
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                # Entities
                if entities_data:
                    csvfile.write("# ENTITIES\n")
                    writer = csv.DictWriter(csvfile, fieldnames=entities_data[0].keys())
                    writer.writeheader()
                    writer.writerows(entities_data)
                    csvfile.write("\n\n")

                # Relations
                if relations_data:
                    csvfile.write("# RELATIONS\n")
                    writer = csv.DictWriter(
                        csvfile, fieldnames=relations_data[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(relations_data)
                    csvfile.write("\n\n")

                # Relationships
                if relationships_data:
                    csvfile.write("# RELATIONSHIPS\n")
                    writer = csv.DictWriter(
                        csvfile, fieldnames=relationships_data[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(relationships_data)

        elif file_format == "excel":
            # Excel export
            entities_df = (
                pd.DataFrame(entities_data) if entities_data else pd.DataFrame()
            )
            relations_df = (
                pd.DataFrame(relations_data) if relations_data else pd.DataFrame()
            )
            relationships_df = (
                pd.DataFrame(relationships_data)
                if relationships_data
                else pd.DataFrame()
            )

            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                if not entities_df.empty:
                    entities_df.to_excel(writer, sheet_name="Entities", index=False)
                if not relations_df.empty:
                    relations_df.to_excel(writer, sheet_name="Relations", index=False)
                if not relationships_df.empty:
                    relationships_df.to_excel(
                        writer, sheet_name="Relationships", index=False
                    )

        elif file_format == "md":
            # Markdown export
            with open(output_path, "w", encoding="utf-8") as mdfile:
                mdfile.write("# LightRAG Data Export\n\n")

                # Entities
                mdfile.write("## Entities\n\n")
                if entities_data:
                    # Write header
                    mdfile.write("| " + " | ".join(entities_data[0].keys()) + " |\n")
                    mdfile.write(
                        "| "
                        + " | ".join(["---"] * len(entities_data[0].keys()))
                        + " |\n"
                    )

                    # Write rows
                    for entity in entities_data:
                        mdfile.write(
                            "| " + " | ".join(str(v) for v in entity.values()) + " |\n"
                        )
                    mdfile.write("\n\n")
                else:
                    mdfile.write("*No entity data available*\n\n")

                # Relations
                mdfile.write("## Relations\n\n")
                if relations_data:
                    # Write header
                    mdfile.write("| " + " | ".join(relations_data[0].keys()) + " |\n")
                    mdfile.write(
                        "| "
                        + " | ".join(["---"] * len(relations_data[0].keys()))
                        + " |\n"
                    )

                    # Write rows
                    for relation in relations_data:
                        mdfile.write(
                            "| "
                            + " | ".join(str(v) for v in relation.values())
                            + " |\n"
                        )
                    mdfile.write("\n\n")
                else:
                    mdfile.write("*No relation data available*\n\n")

                # Relationships
                mdfile.write("## Relationships\n\n")
                if relationships_data:
                    # Write header
                    mdfile.write(
                        "| " + " | ".join(relationships_data[0].keys()) + " |\n"
                    )
                    mdfile.write(
                        "| "
                        + " | ".join(["---"] * len(relationships_data[0].keys()))
                        + " |\n"
                    )

                    # Write rows
                    for relationship in relationships_data:
                        mdfile.write(
                            "| "
                            + " | ".join(str(v) for v in relationship.values())
                            + " |\n"
                        )
                else:
                    mdfile.write("*No relationship data available*\n\n")

        elif file_format == "txt":
            # Plain text export
            with open(output_path, "w", encoding="utf-8") as txtfile:
                txtfile.write("LIGHTRAG DATA EXPORT\n")
                txtfile.write("=" * 80 + "\n\n")

                # Entities
                txtfile.write("ENTITIES\n")
                txtfile.write("-" * 80 + "\n")
                if entities_data:
                    # Create fixed width columns
                    col_widths = {
                        k: max(len(k), max(len(str(e[k])) for e in entities_data))
                        for k in entities_data[0]
                    }
                    header = "  ".join(k.ljust(col_widths[k]) for k in entities_data[0])
                    txtfile.write(header + "\n")
                    txtfile.write("-" * len(header) + "\n")

                    # Write rows
                    for entity in entities_data:
                        row = "  ".join(
                            str(v).ljust(col_widths[k]) for k, v in entity.items()
                        )
                        txtfile.write(row + "\n")
                    txtfile.write("\n\n")
                else:
                    txtfile.write("No entity data available\n\n")

                # Relations
                txtfile.write("RELATIONS\n")
                txtfile.write("-" * 80 + "\n")
                if relations_data:
                    # Create fixed width columns
                    col_widths = {
                        k: max(len(k), max(len(str(r[k])) for r in relations_data))
                        for k in relations_data[0]
                    }
                    header = "  ".join(
                        k.ljust(col_widths[k]) for k in relations_data[0]
                    )
                    txtfile.write(header + "\n")
                    txtfile.write("-" * len(header) + "\n")

                    # Write rows
                    for relation in relations_data:
                        row = "  ".join(
                            str(v).ljust(col_widths[k]) for k, v in relation.items()
                        )
                        txtfile.write(row + "\n")
                    txtfile.write("\n\n")
                else:
                    txtfile.write("No relation data available\n\n")

                # Relationships
                txtfile.write("RELATIONSHIPS\n")
                txtfile.write("-" * 80 + "\n")
                if relationships_data:
                    # Create fixed width columns
                    col_widths = {
                        k: max(len(k), max(len(str(r[k])) for r in relationships_data))
                        for k in relationships_data[0]
                    }
                    header = "  ".join(
                        k.ljust(col_widths[k]) for k in relationships_data[0]
                    )
                    txtfile.write(header + "\n")
                    txtfile.write("-" * len(header) + "\n")

                    # Write rows
                    for relationship in relationships_data:
                        row = "  ".join(
                            str(v).ljust(col_widths[k]) for k, v in relationship.items()
                        )
                        txtfile.write(row + "\n")
                else:
                    txtfile.write("No relationship data available\n\n")

        else:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Choose from: csv, excel, md, txt"
            )
        if file_format is not None:
            print(f"Data exported to: {output_path} with format: {file_format}")
        else:
            print("Data displayed as table format")

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

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Synchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
        )

    def _merge_entity_attributes(
        self, entity_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
    ) -> dict[str, Any]:
        """Merge attributes from multiple entities.

        Args:
            entity_data_list: List of dictionaries containing entity data
            merge_strategy: Merge strategy for each field

        Returns:
            Dictionary containing merged entity data
        """
        merged_data = {}

        # Collect all possible keys
        all_keys = set()
        for data in entity_data_list:
            all_keys.update(data.keys())

        # Merge values for each key
        for key in all_keys:
            # Get all values for this key
            values = [data.get(key) for data in entity_data_list if data.get(key)]

            if not values:
                continue

            # Merge values according to strategy
            strategy = merge_strategy.get(key, "keep_first")

            if strategy == "concatenate":
                merged_data[key] = "\n\n".join(values)
            elif strategy == "keep_first":
                merged_data[key] = values[0]
            elif strategy == "keep_last":
                merged_data[key] = values[-1]
            elif strategy == "join_unique":
                # Handle fields separated by GRAPH_FIELD_SEP
                unique_items = set()
                for value in values:
                    items = value.split(GRAPH_FIELD_SEP)
                    unique_items.update(items)
                merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
            else:
                # Default strategy
                merged_data[key] = values[0]

        return merged_data

    def _merge_relation_attributes(
        self, relation_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
    ) -> dict[str, Any]:
        """Merge attributes from multiple relationships.

        Args:
            relation_data_list: List of dictionaries containing relationship data
            merge_strategy: Merge strategy for each field

        Returns:
            Dictionary containing merged relationship data
        """
        merged_data = {}

        # Collect all possible keys
        all_keys = set()
        for data in relation_data_list:
            all_keys.update(data.keys())

        # Merge values for each key
        for key in all_keys:
            # Get all values for this key
            values = [
                data.get(key)
                for data in relation_data_list
                if data.get(key) is not None
            ]

            if not values:
                continue

            # Merge values according to strategy
            strategy = merge_strategy.get(key, "keep_first")

            if strategy == "concatenate":
                merged_data[key] = "\n\n".join(str(v) for v in values)
            elif strategy == "keep_first":
                merged_data[key] = values[0]
            elif strategy == "keep_last":
                merged_data[key] = values[-1]
            elif strategy == "join_unique":
                # Handle fields separated by GRAPH_FIELD_SEP
                unique_items = set()
                for value in values:
                    items = str(value).split(GRAPH_FIELD_SEP)
                    unique_items.update(items)
                merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
            elif strategy == "max":
                # For numeric fields like weight
                try:
                    merged_data[key] = max(float(v) for v in values)
                except (ValueError, TypeError):
                    merged_data[key] = values[0]
            else:
                # Default strategy
                merged_data[key] = values[0]

        return merged_data

    async def _merge_entities_done(self) -> None:
        """Callback after entity merging is complete, ensures updates are persisted"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )
