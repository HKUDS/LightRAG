import os
import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Union, final, Dict, List, Set
from ..base import (
    BaseKVStorage,
    DocStatusStorage,
    DocProcessingStatus,
    DocStatus,
    BaseVectorStorage,
)

from ..utils import logger, compute_mdhash_id
import pipmaster as pm

if not pm.is_installed("elasticsearch"):
    pm.install('"elasticsearch>=8.0.0,<9.0.0"')

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk, BulkIndexError
from elasticsearch.exceptions import ConnectionError, TransportError, RequestError


class ESClientManager:
    """
    Manages singleton instance of AsyncElasticsearch client with thread-safe operations.
    Handles client initialization, release, index name sanitization, and index creation.
    """

    _client: AsyncElasticsearch | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        """
        Get a singleton instance of AsyncElasticsearch client.
        Creates a new client if it doesn't exist, using environment variables for authentication.

        Returns:
            AsyncElasticsearch: An instance of the Elasticsearch async client.
        """
        async with cls._lock:
            if cls._client is None:
                es_user = os.environ.get("ES_USERNAME")
                es_pass = os.environ.get("ES_PASSWORD")
                auth = (es_user, es_pass) if es_user and es_pass else None
                cls._client = AsyncElasticsearch(
                    hosts=os.environ.get("ES_HOST", "http://localhost:9200"),
                    basic_auth=auth,
                )
            return cls._client

    @classmethod
    async def release_client(cls):
        """
        Release the Elasticsearch client by closing the connection and resetting the singleton instance.
        Uses a lock to ensure thread-safe operation.
        """
        async with cls._lock:
            if cls._client:
                await cls._client.close()
                cls._client = None

    @classmethod
    def _sanitize_index_name(cls, name: str) -> str:
        """
        Sanitize index name to comply with Elasticsearch naming restrictions.
        Replaces invalid characters with underscores and converts to lowercase.

        Args:
            name: Original index name to sanitize.

        Returns:
            Sanitized index name suitable for Elasticsearch.
        """
        sanitized = name.lower()
        for char in ["/", "\\", "*", "?", '"', "<", ">", "|", " ", ","]:
            sanitized = sanitized.replace(char, "_")
        return sanitized

    @classmethod
    async def create_index_if_not_exist(
        cls, index_name: str, mapping: Dict[str, Any]
    ) -> None:
        """
        Asynchronously create an Elasticsearch index if it doesn't exist.

        Args:
            index_name: Name of the index to create.
            mapping: Dictionary defining the index mapping (schema).
        """
        safe_index_name = cls._sanitize_index_name(index_name)

        client = await cls.get_client()

        # Check if the index exists asynchronously
        exists = await client.indices.exists(index=safe_index_name)
        if not exists:
            # Create the index asynchronously if it does not exist
            await client.indices.create(index=safe_index_name, body=mapping)
            logger.info(f"Created index: {index_name}")


@final
@dataclass
class ESKVStorage(BaseKVStorage):
    """
    Elasticsearch-based implementation of the BaseKVStorage interface.
    Provides key-value storage functionality using Elasticsearch indices.
    """

    es_client: AsyncElasticsearch = field(default=None)
    index_name: str = field(default=None)

    def __post_init__(self):
        """
        Post-initialization setup. Constructs the final namespace with workspace prefix if provided,
        and sets the index name based on the namespace.
        """
        es_workspace = os.environ.get("ES_WORKSPACE")
        if es_workspace and es_workspace.strip():
            effective_workspace = es_workspace.strip()
        else:
            effective_workspace = self.workspace

        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        self.index_name = self.namespace

    async def initialize(self):
        """
        Initialize the KV storage. Retrieves the Elasticsearch client and creates the index
        with appropriate mapping if it doesn't exist.
        """
        if self.es_client is None:
            self.es_client = await ESClientManager.get_client()
            kv_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(self.index_name, kv_mapping)

    async def finalize(self):
        """
        Clean up resources by releasing the Elasticsearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    def _flatten_es_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten an Elasticsearch document response into a simplified dictionary.
        Extracts document ID and metadata fields from the source.

        Args:
            doc: Elasticsearch document response (including '_id' and '_source').

        Returns:
            Flattened dictionary containing 'id', timestamps, and metadata fields.
        """
        source = doc["_source"]
        return {
            "id": doc["_id"],
            "create_time": source.get("create_time", 0),
            "update_time": source.get("update_time", 0),
            **source.get("meta", {}),
        }

    async def get_by_id(self, id: str) -> Union[Dict[str, Any], None]:
        """
        Retrieve a document by its ID from the KV storage.

        Args:
            id: Document ID to retrieve.

        Returns:
            Flattened document data if found; None if the document does not exist.
        """
        try:
            doc = await self.es_client.get(index=self.index_name, id=id)
            return self._flatten_es_doc(doc)
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple documents by their IDs from the KV storage.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of flattened document data for found IDs (excludes non-existent IDs).
        """
        if not ids:
            return []

        body = {"ids": ids}
        response = await self.es_client.mget(index=self.index_name, body=body)
        docs = []
        for hit in response["docs"]:
            if hit["found"]:
                doc = self._flatten_es_doc(hit)
                docs.append(doc)
        return docs

    async def filter_keys(self, keys: Set[str]) -> Set[str]:
        """
        Filter a set of keys to identify those that do NOT exist in the storage.

        Args:
            keys: Set of keys to check for existence.

        Returns:
            Subset of keys that are not found in the storage.
        """
        if not keys:
            return set()

        body = {"ids": list(keys)}
        res = await self.es_client.mget(index=self.index_name, body=body)
        found_ids = {doc["_id"] for doc in res["docs"] if doc.get("found")}
        return keys - found_ids

    async def get_all(self) -> dict[str, Any]:
        """
        Retrieve all documents from the KV storage using scroll API for large result sets.

        Returns:
            Dictionary mapping document IDs to their flattened data.
        """
        result = {}
        scroll = "2m"  # Maintain search context for 2 minutes
        response = await self.es_client.search(
            index=self.index_name,
            body={"query": {"match_all": {}}},
            scroll=scroll,
            size=1000,
        )

        scroll_id = response.get("_scroll_id")
        while scroll_id:
            for hit in response["hits"]["hits"]:
                doc_id = hit["_id"]
                doc = self._flatten_es_doc(hit)
                result[doc_id] = doc

            response = await self.es_client.scroll(scroll_id=scroll_id, scroll=scroll)
            scroll_id = response.get("_scroll_id")

        # Clear the scroll context to free resources
        if scroll_id:
            await self.es_client.clear_scroll(scroll_id=scroll_id)

        return result

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Insert or update multiple documents in bulk. Handles both new documents (insert)
        and existing documents (update) with timestamp tracking.

        Args:
            data: Dictionary where keys are document IDs and values are metadata to store.
        """
        if not data:
            return

        current_time = int(time.time())
        actions = []

        for k, v in data.items():
            # Ensure 'llm_cache_list' exists for text_chunks namespace
            if self.namespace.endswith("text_chunks"):
                if "llm_cache_list" not in v:
                    v["llm_cache_list"] = []

            # Extract metadata (exclude reserved fields like 'id')
            meta_data = {
                key: value
                for key, value in v.items()
                if key not in ["id", "create_time", "update_time"]
            }

            # Prepare bulk action: update if exists, insert (upsert) if not
            action = {
                "_op_type": "update",
                "_index": self.index_name,
                "_id": k,
                "doc": {
                    "update_time": current_time,
                    "meta": meta_data,
                },
                "upsert": {
                    "id": k,
                    "create_time": current_time,
                    "update_time": current_time,
                    "meta": meta_data,
                },
            }

            actions.append(action)

        # Execute bulk operation
        try:
            await async_bulk(self.es_client, actions, refresh="wait_for")
        except Exception as e:
            logger.error(f"Unexpected error during bulk upsert: {e}")

    async def index_done_callback(self):
        """
        Callback invoked after indexing completes. No specific operation implemented.
        """
        pass

    async def delete(self, ids: List[str]) -> None:
        """
        Delete multiple documents by their IDs from the KV storage.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        # Prepare bulk delete actions
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]

        try:
            results = await async_bulk(self.es_client, actions, refresh="wait_for")
            logger.info(f"Deleted {results[0]} documents from {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting documents from {self.index_name}: {e}")

    async def drop_cache_by_modes(self, modes: List[str] = None) -> bool:
        """
        Delete documents associated with specific modes (for LLM response cache).
        Matches documents using regex pattern on document IDs.

        Args:
            modes: List of modes to filter documents for deletion.

        Returns:
            True if deletion is successful; False if modes are not provided.
        """
        if not modes:
            return False

        try:
            # Regex pattern: match IDs starting with any mode in the list
            pattern = f"({'|'.join(modes)}):.*"
            response = await self.es_client.delete_by_query(
                index=self.index_name, body={"query": {"regexp": {"_id": pattern}}}
            )
            logger.info(f"Deleted {response['deleted']} documents by modes: {modes}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache by modes {modes}: {e}")
            return False

    async def drop(self) -> Dict[str, str]:
        """
        Delete all documents in the KV storage index.

        Returns:
            Dictionary with 'status' (success/error) and 'message' describing the result.
        """
        try:
            await self.es_client.delete_by_query(
                index=self.index_name, 
                body={"query": {"match_all": {}}},
                wait_for_completion=True
            )
            return {"status": "success", "message": "All documents deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Define the Elasticsearch index mapping for the KV storage.
        Enforces strict dynamic mapping and defines core fields.

        Returns:
            Dictionary specifying the index mapping.
        """
        return {
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "id": {"type": "keyword"},
                    "create_time": {"type": "long"},
                    "update_time": {"type": "long"},
                    "meta": {"type": "object", "dynamic": True},
                },
            }
        }


@final
@dataclass
class ESDocStatusStorage(DocStatusStorage):
    """
    Elasticsearch-based implementation of the DocStatusStorage interface.
    Tracks document processing status (e.g., indexing state, chunk counts) using Elasticsearch.
    """

    es_client: AsyncElasticsearch = field(default=None)
    index_name: str = field(default=None)

    def __post_init__(self):
        """
        Post-initialization setup. Constructs the final namespace with workspace prefix if provided,
        and sets the index name based on the namespace.
        """
        mongodb_workspace = os.environ.get("ES_WORKSPACE")
        if mongodb_workspace and mongodb_workspace.strip():
            effective_workspace = mongodb_workspace.strip()
        else:
            effective_workspace = self.workspace

        # Apply workspace prefix to namespace
        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        # Set index name
        self.index_name = self.namespace

    async def initialize(self):
        """
        Initialize the document status storage. Retrieves the Elasticsearch client and creates
        the index with appropriate mapping if it doesn't exist.
        """
        if self.es_client is None:
            self.es_client = await ESClientManager.get_client()
            doc_status_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(
                self.index_name, doc_status_mapping
            )

    async def finalize(self):
        """
        Clean up resources by releasing the Elasticsearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    async def get_by_id(self, id: str) -> Union[Dict[str, Any], None]:
        """
        Retrieve a document's status by its ID.

        Args:
            id: Document ID to retrieve status for.

        Returns:
            Status data if found; None if the document status does not exist.
        """
        try:
            res = await self.es_client.get(index=self.index_name, id=id)
            doc = res["_source"]
            return doc
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve status data for multiple documents by their IDs.

        Args:
            ids: List of document IDs.

        Returns:
            List of status data dictionaries for found documents.
        """
        if not ids:
            return []

        body = {"ids": ids}
        res = await self.es_client.mget(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["docs"] if hit["found"]]

    async def filter_keys(self, data: Set[str]) -> Set[str]:
        """
        Filter a set of keys to identify those that do NOT exist in the storage.

        Args:
            data: Set of keys to check for existence.

        Returns:
            Subset of keys that are not found in the storage.
        """
        if not data:
            return set()

        response = await self.es_client.mget(
            index=self.index_name, ids=list(data), _source=False
        )
        existing_ids = {hit["_id"] for hit in response["docs"] if hit["found"]}
        return data - existing_ids

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Insert or update document status data in bulk. Ensures 'chunks_list' is a list of strings.

        Args:
            data: Dictionary where keys are document IDs and values are status metadata.
        """
        if not data:
            return

        actions = []
        for doc_id, doc_data in data.items():
            # Ensure 'chunks_list' is a list (normalize input)
            if "chunks_list" not in doc_data or doc_data["chunks_list"] is None:
                doc_data["chunks_list"] = []
            elif not isinstance(doc_data["chunks_list"], list):
                doc_data["chunks_list"] = [doc_data["chunks_list"]]

            logger.info(f"Upserting doc {doc_id}: {doc_data}")

            # Prepare bulk action: update if exists, insert if not
            actions.append(
                {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": doc_id,
                    "doc": doc_data,
                    "doc_as_upsert": True,  # Insert as new document if not exists
                }
            )

        # Execute bulk operation
        try:
            await async_bulk(self.es_client, actions, refresh="wait_for")
        except BulkIndexError as e:
            logger.error(
                f"BulkIndexError: {len(e.errors)} document(s) failed to index."
            )
            for err in e.errors:
                logger.error(f"Indexing error detail: {err}")
            raise
        except (ConnectionError, TransportError, RequestError) as e:
            logger.error(f"Elasticsearch error: {e}")
            raise
        except Exception:
            logger.exception("Unexpected exception during Elasticsearch bulk upsert.")
            raise

    async def get_status_counts(self) -> Dict[str, int]:
        """
        Get the count of documents grouped by their processing status.

        Returns:
            Dictionary with status values as keys and their respective counts as values.
        """
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "size": 0,  # Do not return actual documents
                "aggs": {
                    "status_counts": {
                        "terms": {
                            "field": "status.keyword",  # Use keyword sub-field for exact matches
                            "size": 100,  # Support up to 100 distinct statuses
                        }
                    }
                },
            },
        )

        counts = {}
        for bucket in response["aggregations"]["status_counts"]["buckets"]:
            counts[bucket["key"]] = bucket["doc_count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> Dict[str, DocProcessingStatus]:
        """
        Retrieve documents with a specific processing status.

        Args:
            status: Target document status to filter by (from DocStatus enum).

        Returns:
            Dictionary mapping document IDs to DocProcessingStatus objects.
        """
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {"term": {"status": status.value}},  # Match status enum value
                "size": 1000,  # Adjust based on expected result size
            },
        )

        result = {}
        for hit in response["hits"]["hits"]:
            doc_id = hit["_id"]
            doc_data = hit["_source"]

            result[doc_id] = DocProcessingStatus(
                content=doc_data.get("content", ""),
                content_summary=doc_data.get("content_summary"),
                content_length=doc_data.get("content_length", 0),
                status=doc_data.get("status", status.value),
                created_at=doc_data.get("created_at"),
                updated_at=doc_data.get("updated_at"),
                chunks_count=doc_data.get("chunks_count", -1),
                file_path=doc_data.get("file_path", doc_id),
                chunks_list=doc_data.get("chunks_list", []),
            )
        return result

    async def index_done_callback(self):
        """
        Callback invoked after indexing completes. No specific operation implemented.
        """
        pass

    async def drop(self) -> Dict[str, str]:
        """
        Delete all documents in the status storage index.

        Returns:
            Dictionary with 'status' (success/error) and 'message' describing the result.
        """
        try:
            await self.es_client.delete_by_query(
                index=self.index_name, 
                body={"query": {"match_all": {}}},
                wait_for_completion=True
            )
            return {"status": "success", "message": "All documents deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete(self, ids: list[str]) -> None:
        """
        Delete status records for multiple documents by their IDs.

        Args:
            ids: List of document IDs to delete status records for.
        """
        if not ids:
            return

        # Prepare bulk delete actions
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]

        try:
            await async_bulk(self.es_client, actions, refresh="wait_for", raise_on_error=False)
            logger.debug(f"Deleted {len(ids)} doc statuses from {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting doc statuses: {e}")

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Define the Elasticsearch index mapping for document status storage.
        Specifies field types for status tracking (e.g., dates, counts, lists).

        Returns:
            Dictionary specifying the index mapping.
        """
        return {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "status": {
                        "type": "keyword"  # Exact matches for status filtering
                    },
                    "content": {
                        "type": "text"  # Full-text searchable content
                    },
                    "content_summary": {
                        "type": "text"  # Summary of content
                    },
                    "content_length": {
                        "type": "integer"  # Length of content
                    },
                    "created_at": {
                        "type": "date"  # Timestamp of creation
                    },
                    "updated_at": {
                        "type": "date"  # Timestamp of last update
                    },
                    "chunks_count": {
                        "type": "integer"  # Number of chunks in the document
                    },
                    "chunks_list": {
                        "type": "keyword",  # List of chunk IDs (as keywords)
                    },
                    "file_path": {
                        "type": "keyword"  # Path to source file (exact matches)
                    },
                }
            }
        }


@final
@dataclass
class ESVectorDBStorage(BaseVectorStorage):
    """
    Elasticsearch-based implementation of the BaseVectorStorage interface.
    Stores and queries vector embeddings using Elasticsearch's dense vector support,
    enabling similarity search for embeddings (e.g., text embeddings).
    """

    es_client: AsyncElasticsearch = field(default=None)
    index_name: str = field(default="", init=False)
    embedding_dim: int = field(default=0, init=False)

    def __post_init__(self):
        """
        Post-initialization setup. Configures workspace, index name, embedding dimension,
        and similarity threshold from environment variables and global config.
        """
        # Handle workspace prefix for namespace
        es_workspace = os.environ.get("ES_WORKSPACE")
        if es_workspace and es_workspace.strip():
            effective_workspace = es_workspace.strip()
        else:
            effective_workspace = self.workspace

        # Apply workspace prefix to namespace
        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        # Set index name for vector storage
        self.index_name = f"vector_{self.namespace}"

        # Get embedding dimension from the embedding function
        self.embedding_dim = self.embedding_func.embedding_dim

        # Get cosine similarity threshold from global config
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in global config"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Set batch size for embedding generation
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        """
        Initialize the vector storage. Retrieves the Elasticsearch client and creates
        the vector index with dense vector mapping if it doesn't exist.
        """
        if self.es_client is None:
            self.es_client = await ESClientManager.get_client()
            vector_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(
                self.index_name, vector_mapping
            )

    async def finalize(self):
        """
        Clean up resources by releasing the Elasticsearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Insert or update vector documents in bulk. Generates embeddings for content using
        the configured embedding function and stores them with metadata.

        Args:
            data: Dictionary where keys are document IDs and values contain 'content' and metadata.
        """
        logger.info(f"Inserting {len(data)} documents to {self.index_name}")
        if not data:
            return

        current_time = int(time.time())

        # Extract content for embedding generation
        contents = [v["content"] for v in data.values()]
        # Split into batches to avoid overwhelming the embedding function
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        # Generate embeddings for all batches (async)
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        # Concatenate batch embeddings into a single array
        embeddings = np.concatenate(embeddings_list)

        # Prepare bulk index actions
        actions = []
        for i, (doc_id, doc_data) in enumerate(data.items()):
            # Construct document with vector, timestamps, and allowed metadata
            doc = {
                "id": doc_id,
                "vector": embeddings[i].tolist(),  # Convert numpy array to list
                "created_at": current_time,
                "meta": {k: v for k, v in doc_data.items() if k in self.meta_fields},
            }
            actions.append(
                {
                    "_op_type": "index",  # Overwrite if exists (idempotent)
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": doc,
                }
            )

        # Execute bulk insertion with refresh to make data immediately searchable
        await async_bulk(self.es_client, actions, refresh="wait_for")

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform a vector similarity search using a query text. Generates a query embedding,
        then finds the top-k most similar vectors in the storage.

        Args:
            query: Input text to generate a query vector from.
            top_k: Number of top matching results to return.
            ids: Optional list of document IDs to filter the search (only return matches from this list).

        Returns:
            List of matching documents with metadata, IDs, distances, and timestamps,
            filtered by the cosine similarity threshold.
        """
        # Generate embedding for the query text
        embedding = await self.embedding_func([query], _priority=5)
        query_vector = embedding[0].tolist()

        # Configure k-nearest neighbor (KNN) query
        knn_query = {
            "field": "vector",  # Field containing the dense vector
            "k": top_k,  # Number of results to return
            "num_candidates": top_k * 2,  # Candidates to consider (improves recall)
            "query_vector": query_vector,  # Embedding of the query
        }

        # Optional filter: only include documents with IDs in the provided list
        filter_clause = {"terms": {"id": ids}} if ids else {"match_all": {}}

        # Combine KNN with filter in the Elasticsearch query
        es_query = {"knn": knn_query, "query": filter_clause}

        # Execute the search
        response = await self.es_client.search(index=self.index_name, body=es_query)
        hits = response["hits"]["hits"]

        # Format results, filtering by similarity threshold
        return [
            {
                "id": hit["_id"],
                "distance": hit.get("_score"),  # Cosine similarity score
                "created_at": hit["_source"].get("created_at"),
                **hit["_source"]["meta"],  # Include metadata fields
            }
            for hit in hits
            if hit.get("_score") > self.cosine_better_than_threshold  # Apply threshold
        ]

    async def index_done_callback(self) -> None:
        """
        Callback after indexing completes. No specific operation required for Elasticsearch.
        """
        pass

    async def delete(self, ids: list[str]) -> None:
        """
        Delete multiple vector documents by their IDs from the storage.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        max_batch_size = 100
        ids = list(ids)

        batches = [
            ids[i : i + max_batch_size] for i in range(0, len(ids), max_batch_size)
        ]

        for batch_index, batch_ids in enumerate(batches, start=1):
            # Prepare bulk delete actions
            actions = [
                {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
                for doc_id in batch_ids
            ]

            try:
                success, failed = await async_bulk(
                    self.es_client, actions, refresh="wait_for", raise_on_error=False
                )

                if failed:
                    for item in failed:
                        # Ignore 404 errors (document not found)
                        if (
                            "result" in item.get("delete", {})
                            and item["delete"]["result"] == "not_found"
                        ):
                            continue
                            # logger.info(f"Document {item['delete']['_id']} not found, skipping deletion.")
                        else:
                            logger.error(f"Failure details: {item}")
                else:
                    logger.info(
                        f"Successfully deleted {success} documents in batch {batch_index}."
                    )

            except Exception as e:
                logger.error(f"Batch delete failed for batch {batch_index}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """
        Delete a vector document associated with a specific entity name.
        The entity ID is generated using a hash of the entity name.

        Args:
            entity_name: Name of the entity to delete (e.g., a named entity from text).
        """
        try:
            # Generate entity ID using consistent hashing
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            # Delete the document, ignoring 404 (not found) errors
            response = await self.es_client.delete(
                index=self.index_name, id=entity_id, ignore=[404]
            )
            if response["result"] == "deleted":
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Delete all vector documents representing relations (edges) involving a specific entity.
        Matches documents where the entity is either the source or target in the metadata.

        Args:
            entity_name: Name of the entity whose relations to delete.
        """
        try:
            # Query to match relations where entity is source or target
            query = {
                "query": {
                    "bool": {
                        "should": [  # Logical OR
                            {
                                "term": {"meta.src_id.keyword": entity_name}
                            },  # Entity is source
                            {
                                "term": {"meta.tgt_id.keyword": entity_name}
                            },  # Entity is target
                        ]
                    }
                }
            }
            # Delete all matching documents
            await self.es_client.delete_by_query(
                index=self.index_name, 
                body=query, 
                refresh=True,
                wait_for_completion=True
            )
            logger.debug(f"Deleted relations for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """
        Retrieve a vector document by its ID (excluding the raw vector to save bandwidth).

        Args:
            id: Document ID to retrieve.

        Returns:
            Document data with metadata, ID, and timestamps if found; None otherwise.
        """
        try:
            # Exclude 'vector' field to reduce payload size
            result = await self.es_client.get(
                index=self.index_name, id=id, _source_excludes=["vector"]
            )
            if not result.get("found"):
                return None

            return {
                "id": result["_id"],
                "created_at": result["_source"].get("created_at"),
                **result["_source"]["meta"],  # Include metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving document {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve multiple vector documents by their IDs (excluding raw vectors).

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of document data for found IDs (excludes non-existent IDs).
        """
        if not ids:
            return []

        try:
            # Exclude 'vector' field to reduce payload size
            res = await self.es_client.mget(
                index=self.index_name, body={"ids": ids}, _source_excludes=["vector"]
            )
            docs = res["docs"]
            results = []
            for doc in docs:
                if doc.get("found"):
                    results.append(
                        {
                            "id": doc["_id"],
                            "created_at": doc["_source"].get("created_at"),
                            **doc["_source"]["meta"],
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Error retrieving multiple docs: {e}")
            return []

    async def drop(self) -> dict[str, str]:
        """
        Delete the entire vector index and recreate it with the same mapping.
        Useful for resetting the vector storage.

        Returns:
            Dictionary with 'status' (success/error) and 'message' describing the result.
        """
        try:
            # Check if the index exists
            exists = await self.es_client.indices.exists(index=self.index_name)
            if exists:
                # Delete the index if it exists
                await self.es_client.indices.delete(index=self.index_name)
                logger.info(f"Dropped index {self.index_name}")

            # Recreate the index with the correct mapping
            vector_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(
                self.index_name, vector_mapping
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping index {self.index_name}: {e}")
            return {"status": "error", "message": str(e)}

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Define the Elasticsearch index mapping for vector storage.
        Includes a dense vector field with cosine similarity and metadata fields.

        Returns:
            Dictionary specifying the index mapping.
        """
        return {
            "mappings": {
                "dynamic": "strict",  # Prevent dynamic addition of new fields
                "properties": {
                    "id": {"type": "keyword"},  # Document ID (exact matches)
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,  # Dimension of the vector
                        "index": True,  # Enable indexing for KNN search
                        "similarity": "cosine",  # Use cosine similarity
                    },
                    "created_at": {"type": "date"},  # Timestamp of creation
                    "meta": {
                        "type": "object",
                        "dynamic": True,
                    },  # Metadata (dynamic fields)
                },
            }
        }
