import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Callable, Coroutine, Optional, Type, Union, cast
from .operate import (
    chunking_by_token_size,
    extract_entities,
    extract_keywords_only,
    kg_query,
    kg_query_with_keywords,
    mix_kg_vector_query,
    naive_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
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
)

from .namespace import NameSpace, make_namespace

from .prompt import GRAPH_FIELD_SEP

STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.jsondocstatus_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorge": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "FaissVectorDBStorage": ".kg.faiss_impl",
}


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class LightRAG:
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    # logging
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)
    log_dir: str = field(default=os.getcwd())

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = None  # This must be set (we do want to separate llm from the corte, so no more default initialization)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = None  # This must be set (we do want to separate llm from the corte, so no more default initialization)
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  # 'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = int(os.getenv("MAX_TOKENS", "32768"))
    llm_model_max_async: int = int(os.getenv("MAX_ASYNC", "16"))
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    namespace_prefix: str = field(default="")

    enable_llm_cache: bool = True
    # Sometimes there are some reason the LLM failed at Extracting Entities, and we want to continue without LLM cost, we can use this flag
    enable_llm_cache_for_entity_extract: bool = True

    # extension
    addon_params: dict[str, Any] = field(default_factory=dict)
    convert_response_to_json_func: Callable[[str], dict[str, Any]] = (
        convert_response_to_json
    )

    # Add new field for document status storage type
    doc_status_storage: str = field(default="JsonDocStatusStorage")

    # Custom Chunking Function
    chunking_func: Callable[
        [
            str,
            Optional[str],
            bool,
            int,
            int,
            str,
        ],
        list[dict[str, Any]],
    ] = chunking_by_token_size

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "lightrag.log")
        set_logger(log_file)

        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init LLM
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        # Initialize all storages
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )

        self.key_string_value_json_storage_cls = partial(
            self.key_string_value_json_storage_cls, global_config=global_config
        )

        self.vector_db_storage_cls = partial(
            self.vector_db_storage_cls, global_config=global_config
        )

        self.graph_storage_cls = partial(
            self.graph_storage_cls, global_config=global_config
        )

        self.json_doc_status_storage = self.key_string_value_json_storage_cls(
            namespace=self.namespace_prefix + "json_doc_status_storage",
            embedding_func=None,
        )

        self.llm_response_cache = self.key_string_value_json_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            embedding_func=self.embedding_func,
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_vdb = self.vector_db_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )

        if self.llm_response_cache and hasattr(
            self.llm_response_cache, "global_config"
        ):
            hashing_kv = self.llm_response_cache
        else:
            hashing_kv = self.key_string_value_json_storage_cls(
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                embedding_func=self.embedding_func,
            )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_graps(self, nodel_label: str, max_depth: int):
        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label=nodel_label, max_depth=max_depth
        )

    def _get_storage_class(self, storage_name: str) -> dict:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def set_storage_client(self, db_client):
        # Now only tested on Oracle Database
        for storage in [
            self.vector_db_storage_cls,
            self.graph_storage_cls,
            self.doc_status,
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.key_string_value_json_storage_cls,
            self.chunks_vdb,
            self.relationships_vdb,
            self.entities_vdb,
            self.graph_storage_cls,
            self.chunk_entity_relation_graph,
            self.llm_response_cache,
        ]:
            # set client
            storage.db = db_client

    def insert(
        self,
        string_or_strings: Union[str, list[str]],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ):
        """Sync Insert documents with checkpoint support

        Args:
            string_or_strings: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_size, split the sub chunk by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert(string_or_strings, split_by_character, split_by_character_only)
        )

    async def ainsert(
        self,
        string_or_strings: Union[str, list[str]],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ):
        """Async Insert documents with checkpoint support

        Args:
            string_or_strings: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_size, split the sub chunk by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        await self.apipeline_enqueue_documents(string_or_strings)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

    def insert_custom_chunks(self, full_text: str, text_chunks: list[str]):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert_custom_chunks(full_text, text_chunks)
        )

    async def ainsert_custom_chunks(self, full_text: str, text_chunks: list[str]):
        update_storage = False
        try:
            doc_key = compute_mdhash_id(full_text.strip(), prefix="doc-")
            new_docs = {doc_key: {"content": full_text.strip()}}

            _add_doc_keys = await self.full_docs.filter_keys([doc_key])
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("This document is already in the storage.")
                return

            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for chunk_text in text_chunks:
                chunk_text_stripped = chunk_text.strip()
                chunk_key = compute_mdhash_id(chunk_text_stripped, prefix="chunk-")

                inserting_chunks[chunk_key] = {
                    "content": chunk_text_stripped,
                    "full_doc_id": doc_key,
                }

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return

            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )

            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            else:
                self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)

        finally:
            if update_storage:
                await self._insert_done()

    async def apipeline_enqueue_documents(self, string_or_strings: str | list[str]):
        """
        Pipeline for Processing Documents

        1. Remove duplicate contents from the list
        2. Generate document IDs and initial status
        3. Filter out already processed documents
        4. Enqueue document in status
        """
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]

        # 1. Remove duplicate contents from the list
        unique_contents = list(set(doc.strip() for doc in string_or_strings))

        # 2. Generate document IDs and initial status
        new_docs: dict[str, Any] = {
            compute_mdhash_id(content, prefix="doc-"): {
                "content": content,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for content in unique_contents
        }

        # 3. Filter out already processed documents
        add_doc_keys: set[str] = set()
        # Get docs ids
        in_process_keys = list(new_docs.keys())
        # Get in progress docs ids
        excluded_ids = await self.doc_status.get_by_ids(in_process_keys)
        # Exclude already in process
        add_doc_keys = new_docs.keys() - excluded_ids
        # Filter
        new_docs = {k: v for k, v in new_docs.items() if k in add_doc_keys}

        if not new_docs:
            logger.info("All documents have been processed or are duplicates")
            return

        # 4. Store status document
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

        1. Get all pending and failed documents
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """
        # 1. get all pending and failed documents
        to_process_docs: dict[str, DocProcessingStatus] = {}

        # Fetch failed documents
        failed_docs = await self.doc_status.get_failed_docs()
        to_process_docs.update(failed_docs)

        pending_docs = await self.doc_status.get_pending_docs()
        to_process_docs.update(pending_docs)

        if not to_process_docs:
            logger.info("All documents have been processed or are duplicates")
            return

        to_process_docs_ids = list(to_process_docs.keys())

        # Get allready processed documents (text chunks and full docs)
        text_chunks_processed_doc_ids = await self.text_chunks.filter_keys(
            to_process_docs_ids
        )
        full_docs_processed_doc_ids = await self.full_docs.filter_keys(
            to_process_docs_ids
        )

        # 2. split docs into chunks, insert chunks, update doc status
        batch_size = self.addon_params.get("insert_batch_size", 10)
        batch_docs_list = [
            list(to_process_docs.items())[i : i + batch_size]
            for i in range(0, len(to_process_docs), batch_size)
        ]

        # 3. iterate over batches
        tasks: dict[str, list[Coroutine[Any, Any, None]]] = {}
        for batch_idx, ids_doc_processing_status in tqdm_async(
            enumerate(batch_docs_list),
            desc="Process Batches",
        ):
            # 4. iterate over batch
            for id_doc_processing_status in tqdm_async(
                ids_doc_processing_status,
                desc=f"Process Batch {batch_idx}",
            ):
                id_doc, status_doc = id_doc_processing_status
                # Update status in processing
                await self.doc_status.upsert(
                    {
                        id_doc: {
                            "status": DocStatus.PROCESSING,
                            "updated_at": datetime.now().isoformat(),
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                        }
                    }
                )
                # Generate chunks from document
                chunks: dict[str, Any] = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": id_doc_processing_status,
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

                # Ensure chunk insertion and graph processing happen sequentially, not in parallel
                await self.chunks_vdb.upsert(chunks)
                await self._process_entity_relation_graph(chunks)

                tasks[id_doc] = []
                # Check if document already processed the doc
                if id_doc not in full_docs_processed_doc_ids:
                    tasks[id_doc].append(
                        self.full_docs.upsert({id_doc: {"content": status_doc.content}})
                    )

                # Check if chunks already processed  the doc
                if id_doc not in text_chunks_processed_doc_ids:
                    tasks[id_doc].append(self.text_chunks.upsert(chunks))

                # Process document (text chunks and full docs) in parallel
                for id_doc_processing_status, task in tasks.items():
                    try:
                        await asyncio.gather(*task)
                        await self.doc_status.upsert(
                            {
                                id_doc_processing_status: {
                                    "status": DocStatus.PROCESSED,
                                    "chunks_count": len(chunks),
                                    "updated_at": datetime.now().isoformat(),
                                }
                            }
                        )
                        await self._insert_done()

                    except Exception as e:
                        logger.error(
                            f"Failed to process document {id_doc_processing_status}: {str(e)}"
                        )
                        await self.doc_status.upsert(
                            {
                                id_doc_processing_status: {
                                    "status": DocStatus.FAILED,
                                    "error": str(e),
                                    "updated_at": datetime.now().isoformat(),
                                }
                            }
                        )
                        continue

    async def _process_entity_relation_graph(self, chunk: dict[str, Any]) -> None:
        try:
            new_kg = await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_response_cache=self.llm_response_cache,
                global_config=asdict(self),
            )
            if new_kg is None:
                logger.info("No entities or relationships extracted!")
            else:
                self.chunk_entity_relation_graph = new_kg

        except Exception as e:
            logger.error("Failed to extract entities and relationships")
            raise e

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
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
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
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
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, prompt: str = "", param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, prompt, param))

    async def aquery(
        self, query: str, prompt: str = "", param: QueryParam = QueryParam()
    ):
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                prompt=prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    def query_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        1. Extract keywords from the 'query' using new function in operate.py.
        2. Then run the standard aquery() flow with the final prompt (formatted_question).
        """

        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(query, prompt, param)
        )

    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        1. Calls extract_keywords_only to get HL/LL keywords from 'query'.
        2. Then calls kg_query(...) or naive_query(...), etc. as the main query, while also injecting the newly extracted keywords if needed.
        """

        # ---------------------
        # STEP 1: Keyword Extraction
        # ---------------------
        # We'll assume 'extract_keywords_only(...)' returns (hl_keywords, ll_keywords).
        hl_keywords, ll_keywords = await extract_keywords_only(
            text=query,
            param=param,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache
            or self.key_string_value_json_storage_cls(
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            ),
        )

        param.hl_keywords = (hl_keywords,)
        param.ll_keywords = (ll_keywords,)

        # ---------------------
        # STEP 2: Final Query Logic
        # ---------------------

        # Create a new string with the prompt and the keywords
        ll_keywords_str = ", ".join(ll_keywords)
        hl_keywords_str = ", ".join(hl_keywords)
        formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query_with_keywords(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_funcne,
                ),
            )
        elif param.mode == "naive":
            response = await naive_query(
                formatted_question,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

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

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def _get_content_summary(self, content: str, max_length: int = 100) -> str:
        """Get summary of document content

        Args:
            content: Original document content
            max_length: Maximum length of summary

        Returns:
            Truncated content with ellipsis if needed
        """
        content = content.strip()
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def adelete_by_doc_id(self, doc_id: str):
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get(doc_id)
            if not doc_status:
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all related chunks
            chunks = await self.text_chunks.filter(
                lambda x: x.get("full_doc_id") == doc_id
            )
            chunk_ids = list(chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities = [
                    dp
                    for dp in self.entities_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relations = [
                    dp
                    for dp in self.relationships_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes in the graph
            nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
            edges = self.chunk_entity_relation_graph._graph.edges(data=True)

            # Track which entities and relationships need to be deleted or updated
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities
            for node, data in nodes:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node)
                        logger.debug(
                            f"Entity {node} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node] = new_source_id
                        logger.debug(
                            f"Entity {node} will be updated with new source_id: {new_source_id}"
                        )

            # Process relationships
            for src, tgt, data in edges:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
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
                self.chunk_entity_relation_graph.remove_nodes(list(entities_to_delete))
                logger.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = self.chunk_entity_relation_graph._graph.nodes[entity]
                node_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_node(entity, node_data)
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
                self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logger.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = self.chunk_entity_relation_graph._graph.edges[src, tgt]
                edge_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_edge(src, tgt, edge_data)
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

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.error(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                remaining_chunks = await self.text_chunks.filter(
                    lambda x: x.get("full_doc_id") == doc_id
                )
                if remaining_chunks:
                    logger.error(f"Found {len(remaining_chunks)} remaining chunks")

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    # Check entities
                    entities_with_chunk = [
                        dp
                        for dp in self.entities_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if entities_with_chunk:
                        logger.error(
                            f"Found {len(entities_with_chunk)} entities still referencing chunk {chunk_id}"
                        )

                    # Check relationships
                    relations_with_chunk = [
                        dp
                        for dp in self.relationships_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if relations_with_chunk:
                        logger.error(
                            f"Found {len(relations_with_chunk)} relations still referencing chunk {chunk_id}"
                        )

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    def delete_by_doc_id(self, doc_id: str):
        """Synchronous version of adelete"""
        return asyncio.run(self.adelete_by_doc_id(doc_id))

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ):
        """Get detailed information of an entity

        Args:
            entity_name: Entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing entity information, including:
                - entity_name: Entity name
                - source_id: Source document ID
                - graph_data: Complete node data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        entity_name = f'"{entity_name.upper()}"'

        # Get information from the graph
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = self.entities_vdb._client.get([entity_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    def get_entity_info_sync(self, entity_name: str, include_vector_data: bool = False):
        """Synchronous version of getting entity information

        Args:
            entity_name: Entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database
        """
        try:
            import tracemalloc

            tracemalloc.start()
            return asyncio.run(self.get_entity_info(entity_name, include_vector_data))
        finally:
            tracemalloc.stop()

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ):
        """Get detailed information of a relationship

        Args:
            src_entity: Source entity name (no need for quotes)
            tgt_entity: Target entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing relationship information, including:
                - src_entity: Source entity name
                - tgt_entity: Target entity name
                - source_id: Source document ID
                - graph_data: Complete edge data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        src_entity = f'"{src_entity.upper()}"'
        tgt_entity = f'"{tgt_entity.upper()}"'

        # Get information from the graph
        edge_data = await self.chunk_entity_relation_graph.get_edge(
            src_entity, tgt_entity
        )
        source_id = edge_data.get("source_id") if edge_data else None

        result = {
            "src_entity": src_entity,
            "tgt_entity": tgt_entity,
            "source_id": source_id,
            "graph_data": edge_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            vector_data = self.relationships_vdb._client.get([rel_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    def get_relation_info_sync(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ):
        """Synchronous version of getting relationship information

        Args:
            src_entity: Source entity name (no need for quotes)
            tgt_entity: Target entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database
        """
        try:
            import tracemalloc

            tracemalloc.start()
            return asyncio.run(
                self.get_relation_info(src_entity, tgt_entity, include_vector_data)
            )
        finally:
            tracemalloc.stop()
