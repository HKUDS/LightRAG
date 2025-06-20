import os
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Union, final, Dict, List, Set
from enum import Enum
from ..base import (
    BaseKVStorage,
    DocStatusStorage,
    DocProcessingStatus,
    DocStatus,
    BaseVectorStorage,
)

from ..namespace import NameSpace, is_namespace
from ..utils import logger, compute_mdhash_id
import pipmaster as pm
import threading

if not pm.is_installed("elasticsearch"):
    pm.install('"elasticsearch>=8.0.0,<9.0.0"')

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk


class ESClientManager:
    _client: AsyncElasticsearch | None = None
    _lock = threading.Lock()
    _ref_count = 0

    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        with cls._lock:
            if cls._client is None:
                cls._client = AsyncElasticsearch(
                    hosts=os.environ.get("ES_HOST", "http://localhost:9200"),
                    basic_auth=(
                        os.environ.get("ES_USERNAME", ""),
                        os.environ.get("ES_PASSWORD", ""),
                    ),
                )
            cls._ref_count += 1
            return cls._client

    @classmethod
    async def release_client(cls):
        with cls._lock:
            cls._ref_count -= 1
            if cls._ref_count <= 0 and cls._client:
                cls._client.close()
                cls._client = None
                cls._ref_count = 0

    @staticmethod
    async def create_index_if_not_exist_kv_docstatus(
        client: AsyncElasticsearch, index_name: str
    ):
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "meta": {"type": "object", "dynamic": True},
                }
            }
        }
        exists = await client.indices.exists(index=index_name)
        if not exists:
            await client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created KV/DocStatus index: {index_name}")

    @staticmethod
    async def create_index_if_not_exist_vector(
        client: AsyncElasticsearch, index_name: str, dims: int = 1024
    ):
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "meta": {"type": "object", "dynamic": True},
                }
            }
        }
        exists = await client.indices.exists(index=index_name)
        if not exists:
            await client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created Vector index: {index_name} with dims={dims}")


@final
@dataclass
class ESKVStorage(BaseKVStorage):
    _client: AsyncElasticsearch = field(default=None)

    def __post_init__(self):
        self._index = self.namespace

    async def initialize(self):
        if self._client is None:
            self._client = await ESClientManager.get_client()
            await ESClientManager.create_index_if_not_exist_kv_docstatus(
                self._client, self._index
            )

    async def finalize(self):
        if self._client is not None:
            await ESClientManager.release_client()
            self._client = None

    def _flatten_es_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        source = doc["_source"]
        return {"id": doc["_id"], **source.get("meta", {})}

    async def get_by_id(self, id: str) -> Union[Dict[str, Any], None]:
        try:
            doc = await self._client.get(index=self._index, id=id)
            return self._flatten_es_doc(doc)
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        body = {"ids": ids}
        res = await self._client.mget(index=self._index, body=body)
        docs = res["docs"]
        return [self._flatten_es_doc(doc) for doc in docs if doc.get("found")]

    async def filter_keys(self, keys: Set[str]) -> Set[str]:
        body = {"ids": list(keys)}
        res = await self._client.mget(index=self._index, body=body)
        docs = res["docs"]
        found_ids = {doc["_id"] for doc in docs if doc.get("found")}
        return keys - found_ids

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return

        actions = []
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for mode, items in data.items():
                for k, v in items.items():
                    key = f"{mode}_{k}"
                    doc = {"id": key, "meta": v}
                    actions.append(
                        {
                            "_op_type": "index",
                            "_index": self._index,
                            "_id": key,
                            "_source": doc,
                        }
                    )
        else:
            for k, v in data.items():
                doc = {"id": k, "meta": v}
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": self._index,
                        "_id": k,
                        "_source": doc,
                    }
                )

        if actions:
            try:
                await async_bulk(self._client, actions, refresh="wait_for")
            except Exception as e:
                print(f"Bulk upsert failed: {e}")

    async def get_by_mode_and_id(
        self, mode: str, id: str
    ) -> Union[Dict[str, Any], None]:
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            key = f"{mode}_{id}"
            try:
                doc = await self._client.get(index=self._index, id=key)
                return self._flatten_es_doc(doc)
            except NotFoundError:
                return None
        return None

    async def index_done_callback(self):
        pass

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        actions = [{"delete": {"_index": self._index, "id": id}} for id in ids]
        await async_bulk(self._client, actions, refresh="wait_for")

    async def drop_cache_by_modes(self, modes: List[str] = None) -> bool:
        if not modes:
            return False
        query = {
            "bool": {
                "should": [{"prefix": {"id": f"{mode}_"}} for mode in modes],
                "minimum_should_match": 1,
            }
        }
        await self._client.delete_by_query(index=self._index, body={"query": query})
        return True

    async def drop(self) -> Dict[str, str]:
        try:
            await self._client.delete_by_query(
                index=self._index, body={"query": {"match_all": {}}}
            )
            return {"status": "success", "message": "All documents deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class ESDocStatusStorage(DocStatusStorage):
    _client: AsyncElasticsearch = field(default=None)

    def __post_init__(self):
        self._index = self.namespace

    async def initialize(self):
        if self._client is None:
            self._client = await ESClientManager.get_client()
            await ESClientManager.create_index_if_not_exist_kv_docstatus(
                self._client, self._index
            )

    async def finalize(self):
        if self._client is not None:
            await ESClientManager.release_client()
            self._client = None

    async def get_by_id(self, id: str) -> Union[Dict[str, Any], None]:
        try:
            res = await self._client.get(index=self._index, id=id)
            doc = res["_source"]
            return {"id": doc["id"], **doc.get("meta", {})}
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        body = {"ids": ids}
        res = await self._client.mget(index=self._index, body=body)
        docs = res["docs"]
        return [
            {"id": doc["_source"]["id"], **doc["_source"].get("meta", {})}
            for doc in docs
            if doc.get("found")
        ]

    async def filter_keys(self, data: Set[str]) -> Set[str]:
        body = {"ids": list(data)}
        res = await self._client.mget(index=self._index, body=body)
        docs = res["docs"]
        found_ids = {doc["_id"] for doc in docs if doc.get("found")}
        return data - found_ids

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return

        actions = []
        for k, v in data.items():
            doc = {"id": k, "meta": v}
            actions.append(
                {"_op_type": "index", "_index": self._index, "_id": k, "_source": doc}
            )

        try:
            await async_bulk(self._client, actions, refresh="wait_for")
        except Exception as e:
            print(f"Bulk upsert failed: {e}")

    async def get_status_counts(self) -> Dict[str, int]:
        aggs = {"status_counts": {"terms": {"field": "meta.status", "size": 100}}}
        res = await self._client.search(
            index=self._index, body={"size": 0, "aggs": aggs}
        )
        return {
            bucket["key"]: bucket["doc_count"]
            for bucket in res["aggregations"]["status_counts"]["buckets"]
        }

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> Dict[str, DocProcessingStatus]:
        query = {"term": {"meta.status": status.value}}
        res = await self._client.search(
            index=self._index, body={"query": query, "size": 10000}
        )

        result = {}
        for doc in res["hits"]["hits"]:
            source = doc["_source"]
            meta = source.get("meta", {})
            result[doc["_id"]] = DocProcessingStatus(
                content=meta.get("content"),
                content_summary=meta.get("content_summary"),
                content_length=meta.get("content_length"),
                status=meta.get("status"),
                created_at=meta.get("created_at"),
                updated_at=meta.get("updated_at"),
                chunks_count=meta.get("chunks_count", -1),
                file_path=meta.get("file_path", doc["_id"]),
            )

        return result

    async def index_done_callback(self):
        pass

    async def drop(self) -> Dict[str, str]:
        try:
            await self._client.delete_by_query(
                index=self._index, body={"query": {"match_all": {}}}
            )
            return {"status": "success", "message": "All documents deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        actions = [{"delete": {"_index": self._index, "_id": doc_id}} for doc_id in ids]
        try:
            await async_bulk(self._client, actions, refresh="wait_for")
        except Exception as e:
            print(f"Error deleting documents {ids}: {e}")

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        if not modes:
            return False

        try:
            query = {"terms": {"meta.cache_mode": modes}}
            await self._client.delete_by_query(index=self._index, body={"query": query})
            return True
        except Exception as e:
            print(f"Error dropping cache by modes {modes}: {e}")
            return False


@final
@dataclass
class ESVectorDBStorage(BaseVectorStorage):
    _client: AsyncElasticsearch = field(default=None)

    def __post_init__(self):
        """Initialize the AsyncElasticsearch client and index."""
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError("cosine_better_than_threshold must be specified")
        self.cosine_better_than_threshold = cosine_threshold

        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._index = self.namespace

    async def initialize(self):
        if self._client is None:
            self._client = await ESClientManager.get_client()
            await ESClientManager.create_index_if_not_exist_vector(
                self._client, self._index, dims=self.embedding_func.embedding_dim
            )

    async def finalize(self):
        if self._client is not None:
            await ESClientManager.release_client()
            self._client = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vector documents in AsyncElasticsearch.

        Args:
            data: Dictionary where key is document ID and value is metadata including content.
        """
        logger.info(f"Inserting {len(data)} documents to {self._index}")
        if not data:
            return

        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        actions = []
        for i, (id_, item) in enumerate(data.items()):
            doc = {
                "id": id_,
                "vector": embeddings[i].tolist(),
                "meta": {k: v for k, v in item.items() if k in self.meta_fields},
            }
            actions.append(
                {"_op_type": "index", "_index": self._index, "_id": id_, "_source": doc}
            )

        await async_bulk(self._client, actions, refresh="wait_for")

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query: The query text
            top_k: Number of top results to return
            ids: Optional list of document IDs to filter search

        Returns:
            List of matched documents with metadata and similarity scores
        """
        embedding = await self.embedding_func([query], _priority=5)
        query_vector = embedding[0].tolist()

        knn_query = {
            "field": "vector",
            "k": top_k,
            "num_candidates": top_k * 2,
            "query_vector": query_vector,
        }

        filter_clause = {"terms": {"id": ids}} if ids else {"match_all": {}}

        es_query = {"knn": knn_query, "query": filter_clause}

        response = await self._client.search(index=self._index, body=es_query)
        hits = response["hits"]["hits"]
        return [
            {
                "id": hit["_id"],
                "distance": hit.get("_score"),
                "created_at": hit["_source"].get("meta", {}).get("created_at"),
                **hit["_source"]["meta"],
            }
            for hit in hits
        ]

    async def index_done_callback(self) -> None:
        """Callback after indexing completes. No operation required for AsyncElasticsearch."""
        pass

    async def delete_entity(self, entity_name: str) -> None:
        """Delete a document based on entity name.

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            await self._client.delete(index=self._index, id=entity_id, ignore=[404])
            logger.debug(f"Deleted entity {entity_name} (ID: {entity_id})")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations (edges) for a given entity.

        Args:
            entity_name: Name of the entity whose relations should be deleted
        """
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"meta.src_id.keyword": entity_id}},
                            {"term": {"meta.tgt_id.keyword": entity_id}},
                        ]
                    }
                }
            }
            await self._client.delete_by_query(
                index=self._index, body=query, refresh=True
            )
            logger.debug(f"Deleted relations for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            result = await self._client.get(index=self._index, id=id)
            if not result.get("found"):
                return None
            return {
                "id": result["_id"],
                "created_at": result["_source"].get("meta", {}).get("created_at"),
                **result["_source"]["meta"],
            }
        except Exception as e:
            logger.error(f"Error retrieving document {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple documents by their IDs

        Args:
            ids: List of document IDs

        Returns:
            List of found documents
        """
        if not ids:
            return []
        try:
            res = await self._client.mget(index=self._index, body={"ids": ids})
            docs = res["docs"]
            results = []
            for doc in docs:
                if doc.get("found"):
                    results.append(
                        {
                            "id": doc["_id"],
                            "created_at": doc["_source"]
                            .get("meta", {})
                            .get("created_at"),
                            **doc["_source"]["meta"],
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Error retrieving multiple docs: {e}")
            return []

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs

        Args:
            ids: List of document IDs to delete
        """
        for id_ in ids:
            try:
                await self._client.delete(index=self._index, id=id_, ignore=[404])
            except Exception as e:
                logger.error(f"Error deleting {id_}: {e}")

    async def drop(self) -> dict[str, str]:
        """Drop the entire index and recreate it

        Returns:
            Status message indicating success or failure
        """
        try:
            exists = await self._client.indices.exists(index=self._index)
            if exists:
                await self._client.indices.delete(index=self._index)
                logger.info(f"Dropped index {self._index}")
            await ESClientManager.create_index_if_not_exist_vector(
                self._client, self._index, dims=self.embedding_func.embedding_dim
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping index {self._index}: {e}")
            return {"status": "error", "message": str(e)}
