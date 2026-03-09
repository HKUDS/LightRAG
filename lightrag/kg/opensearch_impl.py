"""
OpenSearch Storage Implementation for LightRAG

This module provides OpenSearch-based storage backends for LightRAG,
including KV storage, document status storage, graph storage, and vector storage.

Requirements:
    - opensearch-py >= 3.0.0
    - OpenSearch 3.x or higher with k-NN plugin enabled
"""

import os
import re
import ssl as ssl_module
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np
import configparser

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..utils import logger, compute_mdhash_id
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock

import pipmaster as pm

if not pm.is_installed("opensearch-py"):
    pm.install("opensearch-py")

from opensearchpy import AsyncOpenSearch, helpers
from opensearchpy.exceptions import OpenSearchException, NotFoundError, RequestError

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


def _get_opensearch_env(key, fallback):
    cfg_key = key.replace("OPENSEARCH_", "").lower()
    return os.environ.get(key, config.get("opensearch", cfg_key, fallback=fallback))


def _sanitize_index_name(name: str) -> str:
    """Sanitize a string to be a valid OpenSearch index name."""
    sanitized = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    if sanitized and sanitized[0] in "-_+":
        sanitized = "x" + sanitized
    return sanitized


class ClientManager:
    """Singleton manager for OpenSearch client connections."""

    _instances = {"client": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncOpenSearch:
        """Get or create a shared AsyncOpenSearch client with reference counting."""
        async with cls._lock:
            if cls._instances["client"] is None:
                hosts_str = _get_opensearch_env("OPENSEARCH_HOSTS", "localhost:9200")
                hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]
                username = _get_opensearch_env("OPENSEARCH_USER", "admin")
                password = _get_opensearch_env("OPENSEARCH_PASSWORD", "admin")
                use_ssl = _get_opensearch_env("OPENSEARCH_USE_SSL", "true").lower() in (
                    "true",
                    "1",
                    "yes",
                )
                verify_certs = _get_opensearch_env(
                    "OPENSEARCH_VERIFY_CERTS", "false"
                ).lower() in ("true", "1", "yes")
                timeout = int(_get_opensearch_env("OPENSEARCH_TIMEOUT", "30"))
                max_retries = int(_get_opensearch_env("OPENSEARCH_MAX_RETRIES", "3"))

                ssl_context = None
                if use_ssl and not verify_certs:
                    ssl_context = ssl_module.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl_module.CERT_NONE

                client = AsyncOpenSearch(
                    hosts=hosts,
                    http_auth=(username, password) if username else None,
                    use_ssl=use_ssl,
                    verify_certs=verify_certs,
                    ssl_context=ssl_context,
                    ssl_show_warn=False,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_on_timeout=True,
                )
                cls._instances["client"] = client
                cls._instances["ref_count"] = 0
                logger.info(f"OpenSearch client connected to {hosts}")

            cls._instances["ref_count"] += 1
            return cls._instances["client"]

    @classmethod
    async def release_client(cls, client: AsyncOpenSearch):
        """Release a client reference. Closes the connection when ref count reaches 0."""
        async with cls._lock:
            if client is not None and client is cls._instances["client"]:
                cls._instances["ref_count"] -= 1
                if cls._instances["ref_count"] <= 0:
                    try:
                        await cls._instances["client"].close()
                    except Exception:
                        pass
                    cls._instances["client"] = None
                    cls._instances["ref_count"] = 0
                    logger.info("OpenSearch client connection closed")


def _resolve_workspace(workspace: str, namespace: str):
    """Resolve effective workspace from env or parameter."""
    opensearch_workspace = os.environ.get("OPENSEARCH_WORKSPACE")
    if opensearch_workspace and opensearch_workspace.strip():
        effective = opensearch_workspace.strip()
        logger.info(
            f"Using OPENSEARCH_WORKSPACE: '{effective}' (overriding '{workspace}/{namespace}')"
        )
        return effective
    return workspace


def _build_index_name(workspace: str, namespace: str) -> tuple[str, str, str]:
    """Build index name and return (effective_workspace, final_namespace, index_name)."""
    effective = _resolve_workspace(workspace, namespace)
    if effective:
        final_ns = f"{effective}_{namespace}"
    else:
        final_ns = namespace
        effective = ""
    index_name = _sanitize_index_name(final_ns)
    return effective, final_ns, index_name


@final
@dataclass
class OpenSearchKVStorage(BaseKVStorage):
    """Key-Value storage using OpenSearch. Uses dynamic mapping to support varied schemas."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )

    async def initialize(self):
        """Initialize client connection and create index if needed."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            logger.debug(
                f"[{self.workspace}] OpenSearch KV storage initialized: {self._index_name}"
            )

    async def _create_index_if_not_exists(self):
        try:
            if not await self.client.indices.exists(index=self._index_name):
                # Use dynamic mapping so any namespace schema works
                body = {
                    "mappings": {"dynamic": True},
                    "settings": {
                        "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    },
                }
                await self.client.indices.create(index=self._index_name, body=body)
                logger.info(f"[{self.workspace}] Created index: {self._index_name}")
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating index: {e}")
            raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a document by its ID, or None if not found."""
        try:
            response = await self.client.get(index=self._index_name, id=id)
            doc = response["_source"]
            doc["_id"] = response["_id"]
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
            return doc
        except NotFoundError:
            return None
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting document {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple documents by IDs, preserving input order."""
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    data.setdefault("create_time", 0)
                    data.setdefault("update_time", 0)
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting documents: {e}")
            return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of keys that do not exist in storage."""
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": list(keys)}, _source=False
            )
            existing_ids = {doc["_id"] for doc in response["docs"] if doc.get("found")}
            return keys - existing_ids
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error filtering keys: {e}")
            return keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update documents with automatic timestamping."""
        if not data:
            return
        logger.debug(
            f"[{self.workspace}] Upserting {len(data)} documents to {self.namespace}"
        )
        current_time = int(time.time())
        actions = []
        for doc_id, doc_data in data.items():
            doc_data["update_time"] = current_time
            doc_data.setdefault("create_time", current_time)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": doc_id,
                    "_source": {k: v for k, v in doc_data.items() if k != "_id"},
                }
            )
        try:
            success, failed = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
            if failed:
                logger.warning(
                    f"[{self.workspace}] {len(failed)} documents failed to upsert"
                )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting documents: {e}")
            raise

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed documents searchable."""
        try:
            await self.client.indices.refresh(index=self._index_name)
        except Exception:
            pass

    async def is_empty(self) -> bool:
        """Return True if the index contains no documents."""
        try:
            response = await self.client.count(index=self._index_name)
            return response["count"] == 0
        except OpenSearchException:
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            success, _ = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
            logger.info(
                f"[{self.workspace}] Deleted {success} documents from {self.namespace}"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error deleting documents: {e}")

    async def drop(self) -> dict[str, str]:
        """Delete the entire index."""
        try:
            await self.client.indices.delete(index=self._index_name)
            logger.info(f"[{self.workspace}] Dropped index: {self._index_name}")
            return {"status": "success", "message": f"Index {self._index_name} dropped"}
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error dropping index: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchDocStatusStorage(DocStatusStorage):
    """Document status storage using OpenSearch."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )

    def _prepare_doc_status_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw OpenSearch document to DocProcessingStatus-compatible dict."""
        data = doc.copy()
        data.pop("_id", None)
        if "file_path" not in data:
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        if "error" in data:
            if not data.get("error_msg"):
                data["error_msg"] = data.pop("error")
            else:
                data.pop("error", None)
        return data

    async def initialize(self):
        """Initialize client connection and create doc status index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            logger.debug(
                f"[{self.workspace}] OpenSearch DocStatus storage initialized: {self._index_name}"
            )

    async def _create_index_if_not_exists(self):
        try:
            if not await self.client.indices.exists(index=self._index_name):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "status": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "track_id": {"type": "keyword"},
                            "created_at": {"type": "long"},
                            "updated_at": {"type": "long"},
                        },
                    },
                    "settings": {
                        "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    },
                }
                await self.client.indices.create(index=self._index_name, body=body)
                logger.info(
                    f"[{self.workspace}] Created doc status index: {self._index_name}"
                )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating doc status index: {e}")
            raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        """Get a document status record by ID."""
        try:
            response = await self.client.get(index=self._index_name, id=id)
            doc = response["_source"]
            doc["_id"] = response["_id"]
            return doc
        except NotFoundError:
            return None
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting doc status {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple document status records by IDs."""
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting doc statuses: {e}")
            return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of keys that do not exist in storage."""
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": list(keys)}, _source=False
            )
            existing_ids = {doc["_id"] for doc in response["docs"] if doc.get("found")}
            return keys - existing_ids
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error filtering keys: {e}")
            return keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update document status records."""
        if not data:
            return
        logger.debug(f"[{self.workspace}] Upserting {len(data)} doc statuses")
        actions = []
        for k, v in data.items():
            v.setdefault("chunks_list", [])
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": k,
                    "_source": {fk: fv for fk, fv in v.items() if fk != "_id"},
                }
            )
        try:
            await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting doc statuses: {e}")

    async def get_status_counts(self) -> dict[str, int]:
        """Get document counts grouped by status."""
        try:
            body = {
                "size": 0,
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
            response = await self.client.search(index=self._index_name, body=body)
            return {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["status_counts"]["buckets"]
            }
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting status counts: {e}")
            return {}

    async def _search_all_docs(self, query: dict) -> dict[str, DocProcessingStatus]:
        """Fetch all documents matching a query using PIT + search_after."""
        result = {}
        batch_size = 10000
        try:
            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "size": batch_size,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        try:
                            data = self._prepare_doc_status_data(hit["_source"])
                            result[hit["_id"]] = DocProcessingStatus(**data)
                        except (KeyError, TypeError) as e:
                            logger.error(
                                f"[{self.workspace}] Error parsing doc {hit['_id']}: {e}"
                            )
                    search_after = hits[-1]["sort"]
                    if len(hits) < batch_size:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error fetching docs: {e}")
        return result

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching a specific processing status."""
        return await self._search_all_docs({"term": {"status": status.value}})

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching a specific track ID."""
        return await self._search_all_docs({"term": {"track_id": track_id}})

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination using PIT + search_after."""
        page = max(1, page)
        page_size = max(10, min(200, page_size))
        if sort_field == "id":
            sort_field = "_id"
        if sort_field not in ("created_at", "updated_at", "_id", "file_path"):
            sort_field = "updated_at"
        sort_order = "asc" if sort_direction.lower() == "asc" else "desc"

        query = {"match_all": {}}
        if status_filter is not None:
            query = {"term": {"status": status_filter.value}}

        skip_count = (page - 1) * page_size

        try:
            count_resp = await self.client.count(
                index=self._index_name, body={"query": query}
            )
            total_count = count_resp.get("count", 0)
            if total_count == 0 or skip_count >= total_count:
                return [], total_count

            sort_clause = [{sort_field: {"order": sort_order}}, {"_shard_doc": "asc"}]

            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                skipped = 0
                while skipped < skip_count:
                    batch = min(page_size, skip_count - skipped)
                    body = {
                        "query": query,
                        "sort": sort_clause,
                        "size": batch,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                    }
                    if search_after:
                        body["search_after"] = search_after
                    resp = await self.client.search(body=body)
                    hits = resp["hits"]["hits"]
                    if not hits:
                        return [], total_count
                    search_after = hits[-1]["sort"]
                    skipped += len(hits)

                body = {
                    "query": query,
                    "sort": sort_clause,
                    "size": page_size,
                    "pit": {"id": pit_id, "keep_alive": "1m"},
                }
                if search_after:
                    body["search_after"] = search_after
                response = await self.client.search(body=body)
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass

            documents = []
            for hit in response["hits"]["hits"]:
                try:
                    data = self._prepare_doc_status_data(hit["_source"])
                    documents.append((hit["_id"], DocProcessingStatus(**data)))
                except (KeyError, TypeError) as e:
                    logger.error(
                        f"[{self.workspace}] Error parsing doc {hit['_id']}: {e}"
                    )
            return documents, total_count
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error in paginated query: {e}")
            return [], 0

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get document counts for all statuses including an 'all' total."""
        try:
            body = {
                "size": 0,
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
            response = await self.client.search(index=self._index_name, body=body)
            counts = {}
            total = 0
            for bucket in response["aggregations"]["status_counts"]["buckets"]:
                counts[bucket["key"]] = bucket["doc_count"]
                total += bucket["doc_count"]
            counts["all"] = total
            return counts
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting all status counts: {e}")
            return {}

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Find a document status record by its file_path field."""
        try:
            body = {"query": {"term": {"file_path": file_path}}, "size": 1}
            response = await self.client.search(index=self._index_name, body=body)
            hits = response["hits"]["hits"]
            if hits:
                doc = hits[0]["_source"]
                doc["_id"] = hits[0]["_id"]
                return doc
            return None
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting doc by file_path: {e}")
            return None

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed documents searchable."""
        try:
            await self.client.indices.refresh(index=self._index_name)
        except Exception:
            pass

    async def is_empty(self) -> bool:
        """Return True if the index contains no documents."""
        try:
            response = await self.client.count(index=self._index_name)
            return response["count"] == 0
        except OpenSearchException:
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete document status records by IDs."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error deleting doc statuses: {e}")

    async def drop(self) -> dict[str, str]:
        """Delete the entire doc status index."""
        try:
            await self.client.indices.delete(index=self._index_name)
            logger.info(
                f"[{self.workspace}] Dropped doc status index: {self._index_name}"
            )
            return {"status": "success", "message": f"Index {self._index_name} dropped"}
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error dropping doc status index: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchGraphStorage(BaseGraphStorage):
    """Graph storage using OpenSearch with separate nodes and edges indices.

    Supports two BFS traversal strategies:
    - PPL graphlookup (server-side BFS, requires OpenSearch SQL plugin with Calcite engine)
    - Application-level batched BFS (fallback, works on any OpenSearch 3.x+)

    The strategy is auto-detected during initialize() and can be overridden via
    the OPENSEARCH_USE_PPL_GRAPHLOOKUP environment variable (true/false).
    """

    client: AsyncOpenSearch = field(default=None)
    _nodes_index: str = field(default="", init=False)
    _edges_index: str = field(default="", init=False)
    _ppl_graphlookup_available: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, base_name = _build_index_name(
            self.workspace, self.namespace
        )
        self._nodes_index = f"{base_name}-nodes"
        self._edges_index = f"{base_name}-edges"

    async def initialize(self):
        """Initialize client, create indices, and detect PPL graphlookup support."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_indices_if_not_exist()
            await self._detect_ppl_graphlookup()
            logger.debug(
                f"[{self.workspace}] OpenSearch Graph storage initialized: "
                f"{self._nodes_index}, {self._edges_index} "
                f"(PPL graphlookup: {self._ppl_graphlookup_available})"
            )

    async def _detect_ppl_graphlookup(self):
        """Detect whether PPL graphlookup command is available on this cluster."""
        env_override = os.environ.get("OPENSEARCH_USE_PPL_GRAPHLOOKUP", "").lower()
        if env_override == "true":
            self._ppl_graphlookup_available = True
            return
        if env_override == "false":
            self._ppl_graphlookup_available = False
            return
        # Auto-detect by sending a minimal PPL query
        try:
            await self.client.transport.perform_request(
                "POST",
                "/_plugins/_ppl",
                body={"query": f"source = {self._edges_index} | head 0"},
            )
            # PPL endpoint works; now test graphlookup syntax with a no-op query
            await self.client.transport.perform_request(
                "POST",
                "/_plugins/_ppl",
                body={
                    "query": (
                        f"source = {self._edges_index} | head 1 "
                        f"| graphLookup {self._edges_index} "
                        f"startField=source_node_id fromField=target_node_id "
                        f"toField=source_node_id maxDepth=0 as _gl_probe"
                    )
                },
            )
            self._ppl_graphlookup_available = True
            logger.info(
                f"[{self.workspace}] PPL graphlookup is available, using server-side BFS"
            )
        except Exception:
            self._ppl_graphlookup_available = False
            logger.info(
                f"[{self.workspace}] PPL graphlookup not available, using client-side BFS"
            )

    async def _create_indices_if_not_exist(self):
        try:
            if not await self.client.indices.exists(index=self._nodes_index):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "entity_id": {"type": "keyword"},
                            "entity_type": {"type": "keyword"},
                            "description": {"type": "text"},
                            "source_id": {"type": "text"},
                            "source_ids": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "created_at": {"type": "long"},
                        },
                    },
                    "settings": {
                        "index": {"number_of_shards": 1, "number_of_replicas": 0}
                    },
                }
                await self.client.indices.create(index=self._nodes_index, body=body)
                logger.info(
                    f"[{self.workspace}] Created nodes index: {self._nodes_index}"
                )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise

        try:
            if not await self.client.indices.exists(index=self._edges_index):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "source_node_id": {"type": "keyword"},
                            "target_node_id": {"type": "keyword"},
                            "relationship": {"type": "keyword"},
                            "description": {"type": "text"},
                            "weight": {"type": "float"},
                            "keywords": {"type": "text"},
                            "source_id": {"type": "text"},
                            "source_ids": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "created_at": {"type": "long"},
                        },
                    },
                    "settings": {
                        "index": {"number_of_shards": 1, "number_of_replicas": 0}
                    },
                }
                await self.client.indices.create(index=self._edges_index, body=body)
                logger.info(
                    f"[{self.workspace}] Created edges index: {self._edges_index}"
                )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    # --- Basic queries ---

    async def has_node(self, node_id: str) -> bool:
        """Check whether a node exists in the graph."""
        try:
            return await self.client.exists(index=self._nodes_index, id=node_id)
        except OpenSearchException:
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check whether an edge exists between two nodes (bidirectional)."""
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "bool": {
                                    "must": [
                                        {"term": {"source_node_id": source_node_id}},
                                        {"term": {"target_node_id": target_node_id}},
                                    ]
                                }
                            },
                            {
                                "bool": {
                                    "must": [
                                        {"term": {"source_node_id": target_node_id}},
                                        {"term": {"target_node_id": source_node_id}},
                                    ]
                                }
                            },
                        ]
                    }
                },
                "size": 0,
            }
            response = await self.client.search(index=self._edges_index, body=body)
            return response["hits"]["total"]["value"] > 0
        except OpenSearchException:
            return False

    async def node_degree(self, node_id: str) -> int:
        """Count the number of edges connected to a node."""
        try:
            response = await self.client.count(
                index=self._edges_index,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"source_node_id": node_id}},
                                {"term": {"target_node_id": node_id}},
                            ]
                        }
                    }
                },
            )
            return response.get("count", 0)
        except OpenSearchException:
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Sum of degrees of both endpoint nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get a node document by ID, or None if not found."""
        try:
            response = await self.client.get(index=self._nodes_index, id=node_id)
            doc = response["_source"]
            doc["_id"] = response["_id"]
            return doc
        except NotFoundError:
            return None
        except OpenSearchException:
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get an edge between two nodes (bidirectional), or None."""
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "bool": {
                                    "must": [
                                        {"term": {"source_node_id": source_node_id}},
                                        {"term": {"target_node_id": target_node_id}},
                                    ]
                                }
                            },
                            {
                                "bool": {
                                    "must": [
                                        {"term": {"source_node_id": target_node_id}},
                                        {"term": {"target_node_id": source_node_id}},
                                    ]
                                }
                            },
                        ]
                    }
                },
                "size": 1,
            }
            response = await self.client.search(index=self._edges_index, body=body)
            hits = response["hits"]["hits"]
            if hits:
                doc = hits[0]["_source"]
                doc["_id"] = hits[0]["_id"]
                return doc
            return None
        except OpenSearchException:
            return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all (source, target) edge tuples connected to a node."""
        try:
            query = {
                "bool": {
                    "should": [
                        {"term": {"source_node_id": source_node_id}},
                        {"term": {"target_node_id": source_node_id}},
                    ]
                }
            }
            edges = []
            pit = await self.client.create_pit(
                index=self._edges_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "_source": ["source_node_id", "target_node_id"],
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        edges.append(
                            (
                                hit["_source"]["source_node_id"],
                                hit["_source"]["target_node_id"],
                            )
                        )
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            return edges
        except OpenSearchException:
            return None

    # --- Batch operations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch multiple nodes by ID."""
        try:
            response = await self.client.mget(
                index=self._nodes_index, body={"ids": node_ids}
            )
            result = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    result[doc["_id"]] = data
            return result
        except OpenSearchException:
            return {}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Batch-fetch edge counts for multiple nodes using aggregations."""
        if not node_ids:
            return {}
        try:
            # Use a single query with aggregations for both source and target
            body = {
                "size": 0,
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_node_id": node_ids}},
                            {"terms": {"target_node_id": node_ids}},
                        ]
                    }
                },
                "aggs": {
                    "source_degrees": {
                        "terms": {
                            "field": "source_node_id",
                            "size": len(node_ids) * 2,
                        }
                    },
                    "target_degrees": {
                        "terms": {
                            "field": "target_node_id",
                            "size": len(node_ids) * 2,
                        }
                    },
                },
            }
            response = await self.client.search(index=self._edges_index, body=body)
            result = {}
            for bucket in response["aggregations"]["source_degrees"]["buckets"]:
                if bucket["key"] in node_ids:
                    result[bucket["key"]] = (
                        result.get(bucket["key"], 0) + bucket["doc_count"]
                    )
            for bucket in response["aggregations"]["target_degrees"]["buckets"]:
                if bucket["key"] in node_ids:
                    result[bucket["key"]] = (
                        result.get(bucket["key"], 0) + bucket["doc_count"]
                    )
            return result
        except OpenSearchException:
            return {}

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch-fetch edge tuples for multiple nodes."""
        result = {nid: [] for nid in node_ids}
        try:
            query = {
                "bool": {
                    "should": [
                        {"terms": {"source_node_id": node_ids}},
                        {"terms": {"target_node_id": node_ids}},
                    ]
                }
            }
            pit = await self.client.create_pit(
                index=self._edges_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "_source": ["source_node_id", "target_node_id"],
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        src = hit["_source"]["source_node_id"]
                        tgt = hit["_source"]["target_node_id"]
                        if src in result:
                            result[src].append((src, tgt))
                        if tgt in result:
                            result[tgt].append((src, tgt))
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
        except OpenSearchException:
            pass
        return result

    # --- Upsert operations ---

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update a node. Adds entity_id for PPL compatibility."""
        try:
            doc = {k: v for k, v in node_data.items() if k != "_id"}
            doc["entity_id"] = node_id
            if node_data.get("source_id", ""):
                doc["source_ids"] = node_data["source_id"].split(GRAPH_FIELD_SEP)
            await self.client.index(
                index=self._nodes_index, id=node_id, body=doc, refresh="wait_for"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting node {node_id}: {e}")

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert or update an edge with deterministic ID for bidirectional handling."""
        try:
            # Ensure source node exists (don't overwrite if it already has data)
            if not await self.has_node(source_node_id):
                await self.upsert_node(source_node_id, {})

            doc = {k: v for k, v in edge_data.items() if k != "_id"}
            doc["source_node_id"] = source_node_id
            doc["target_node_id"] = target_node_id
            if edge_data.get("source_id", ""):
                doc["source_ids"] = edge_data["source_id"].split(GRAPH_FIELD_SEP)

            # Use a deterministic ID for the edge so upserts work
            edge_id = compute_mdhash_id(
                f"{source_node_id}-{target_node_id}", prefix="edge-"
            )

            # Check if reverse edge exists
            reverse_id = compute_mdhash_id(
                f"{target_node_id}-{source_node_id}", prefix="edge-"
            )
            try:
                if await self.client.exists(index=self._edges_index, id=reverse_id):
                    edge_id = reverse_id
            except OpenSearchException:
                pass

            await self.client.index(
                index=self._edges_index, id=edge_id, body=doc, refresh="wait_for"
            )
        except OpenSearchException as e:
            logger.error(
                f"[{self.workspace}] Error upserting edge {source_node_id}->{target_node_id}: {e}"
            )

    # --- Delete operations ---

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its connected edges."""
        try:
            # Delete all edges referencing this node
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source_node_id": node_id}},
                            {"term": {"target_node_id": node_id}},
                        ]
                    }
                }
            }
            await self.client.delete_by_query(
                index=self._edges_index, body=body, refresh=True
            )
            # Delete the node
            try:
                await self.client.delete(
                    index=self._nodes_index, id=node_id, refresh="wait_for"
                )
            except NotFoundError:
                pass
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error deleting node {node_id}: {e}")

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Batch-delete multiple nodes and their connected edges."""
        if not nodes:
            return
        logger.info(f"[{self.workspace}] Deleting {len(nodes)} nodes")
        try:
            # Delete edges
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_node_id": nodes}},
                            {"terms": {"target_node_id": nodes}},
                        ]
                    }
                }
            }
            await self.client.delete_by_query(
                index=self._edges_index, body=body, refresh=True
            )
            # Delete nodes
            actions = [
                {"_op_type": "delete", "_index": self._nodes_index, "_id": nid}
                for nid in nodes
            ]
            await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error removing nodes: {e}")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Batch-delete multiple edges (bidirectional matching)."""
        if not edges:
            return
        logger.info(f"[{self.workspace}] Deleting {len(edges)} edges")
        try:
            should_clauses = []
            for src, tgt in edges:
                should_clauses.append(
                    {
                        "bool": {
                            "must": [
                                {"term": {"source_node_id": src}},
                                {"term": {"target_node_id": tgt}},
                            ]
                        }
                    }
                )
                should_clauses.append(
                    {
                        "bool": {
                            "must": [
                                {"term": {"source_node_id": tgt}},
                                {"term": {"target_node_id": src}},
                            ]
                        }
                    }
                )
            body = {"query": {"bool": {"should": should_clauses}}}
            await self.client.delete_by_query(
                index=self._edges_index, body=body, refresh=True
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error removing edges: {e}")

    # --- Query operations ---

    async def get_all_labels(self) -> list[str]:
        """Get all node IDs (entity names) sorted alphabetically."""
        try:
            labels = []
            pit = await self.client.create_pit(
                index=self._nodes_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "_source": False,
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        labels.append(hit["_id"])
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            labels.sort()
            return labels
        except OpenSearchException:
            return []

    def _construct_graph_node(self, node_id, node_data: dict) -> KnowledgeGraphNode:
        return KnowledgeGraphNode(
            id=node_id,
            labels=[node_id],
            properties={
                k: v
                for k, v in node_data.items()
                if k
                not in (
                    "_id",
                    "entity_id",
                    "source_ids",
                    "connected_edges",
                    "edge_count",
                )
            },
        )

    def _construct_graph_edge(self, edge_id: str, edge: dict) -> KnowledgeGraphEdge:
        return KnowledgeGraphEdge(
            id=edge_id,
            type=edge.get("relationship", ""),
            source=edge["source_node_id"],
            target=edge["target_node_id"],
            properties={
                k: v
                for k, v in edge.items()
                if k
                not in (
                    "_id",
                    "source_node_id",
                    "target_node_id",
                    "relationship",
                    "source_ids",
                )
            },
        )

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Retrieve a subgraph via PPL graphlookup (if available) or client-side BFS."""
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        start = time.perf_counter()

        try:
            if node_label == "*":
                result = await self._get_knowledge_graph_all(max_nodes)
            elif self._ppl_graphlookup_available:
                result = await self._bfs_subgraph_ppl(node_label, max_depth, max_nodes)
            else:
                result = await self._bfs_subgraph(node_label, max_depth, max_nodes)

            duration = time.perf_counter() - start
            logger.info(
                f"[{self.workspace}] Subgraph query in {duration:.4f}s | "
                f"Nodes: {len(result.nodes)} | Edges: {len(result.edges)} | Truncated: {result.is_truncated}"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Graph query failed: {e}")

        return result

    async def _get_knowledge_graph_all(self, max_nodes: int) -> KnowledgeGraph:
        """Get all nodes (up to max_nodes, ranked by degree) and their interconnecting edges."""
        result = KnowledgeGraph()
        try:
            total = (await self.client.count(index=self._nodes_index))["count"]
            result.is_truncated = total > max_nodes

            if result.is_truncated:
                # Get top nodes by degree
                body = {
                    "size": 0,
                    "aggs": {
                        "src": {
                            "terms": {
                                "field": "source_node_id",
                                "size": max_nodes,
                            }
                        },
                        "tgt": {
                            "terms": {
                                "field": "target_node_id",
                                "size": max_nodes,
                            }
                        },
                    },
                }
                resp = await self.client.search(index=self._edges_index, body=body)
                degree_map = {}
                for bucket in resp["aggregations"]["src"]["buckets"]:
                    degree_map[bucket["key"]] = (
                        degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                    )
                for bucket in resp["aggregations"]["tgt"]["buckets"]:
                    degree_map[bucket["key"]] = (
                        degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                    )
                top_ids = sorted(degree_map, key=degree_map.get, reverse=True)[
                    :max_nodes
                ]
            else:
                # Get all node IDs
                body = {"query": {"match_all": {}}, "_source": False, "size": max_nodes}
                resp = await self.client.search(index=self._nodes_index, body=body)
                top_ids = [hit["_id"] for hit in resp["hits"]["hits"]]

            # Fetch node data
            if top_ids:
                node_resp = await self.client.mget(
                    index=self._nodes_index, body={"ids": top_ids}
                )
                for doc in node_resp["docs"]:
                    if doc.get("found"):
                        result.nodes.append(
                            self._construct_graph_node(doc["_id"], doc["_source"])
                        )

                # Fetch edges between these nodes
                edge_body = {
                    "query": {
                        "bool": {
                            "must": [
                                {"terms": {"source_node_id": top_ids}},
                                {"terms": {"target_node_id": top_ids}},
                            ]
                        }
                    },
                    "size": 10000,
                }
                edge_resp = await self.client.search(
                    index=self._edges_index, body=edge_body
                )
                seen_edges = set()
                for hit in edge_resp["hits"]["hits"]:
                    e = hit["_source"]
                    eid = f"{e['source_node_id']}-{e['target_node_id']}"
                    if eid not in seen_edges:
                        seen_edges.add(eid)
                        result.edges.append(self._construct_graph_edge(eid, e))
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error in get_knowledge_graph_all: {e}")
        return result

    async def _bfs_subgraph_ppl(
        self, start_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Server-side BFS using PPL graphlookup command.

        Queries the nodes index for the start node, then uses graphLookup to traverse
        the edges index with bidirectional BFS. Uses `flatten` to unnest results and
        `depthField` for depth-based sorting. Falls back to client-side BFS on failure.
        """
        result = KnowledgeGraph()

        # Verify start node exists
        start_node = await self.get_node(start_label)
        if not start_node:
            return result

        seen_nodes = {start_label}
        result.nodes.append(self._construct_graph_node(start_label, start_node))

        if max_depth == 0:
            return result

        # PPL maxDepth=0 means 1 hop (direct match), so max_depth-1
        ppl_depth = max(0, max_depth - 1)
        escaped = self._escape_ppl(start_label)
        ppl_query = (
            f"source = {self._nodes_index}"
            f" | where entity_id = '{escaped}'"
            f" | graphLookup {self._edges_index}"
            f" startField=entity_id"
            f" fromField=target_node_id"
            f" toField=source_node_id"
            f" maxDepth={ppl_depth}"
            f" depthField=_depth"
            f" direction=bi"
            f" as connected_edges"
        )

        try:
            resp = await self.client.transport.perform_request(
                "POST",
                "/_plugins/_ppl",
                body={"query": ppl_query},
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] PPL graphlookup failed, falling back to client BFS: {e}"
            )
            return await self._bfs_subgraph(start_label, max_depth, max_nodes)

        # Parse PPL response — schema-driven to avoid fragile positional access
        try:
            datarows = resp.get("datarows", [])
            schema = [col["name"] for col in resp.get("schema", [])]
            ce_idx = (
                schema.index("connected_edges") if "connected_edges" in schema else -1
            )

            # Collect all edge rows from connected_edges arrays
            all_edge_rows = []
            for row in datarows:
                edges_arr = row[ce_idx] if ce_idx >= 0 else []
                if isinstance(edges_arr, list):
                    all_edge_rows.extend(edges_arr)

            if not all_edge_rows:
                return result

            # Build field index map from the first edge row if it's a dict,
            # otherwise fall back to known edge schema order
            if isinstance(all_edge_rows[0], dict):
                # Dict-based response (ideal)
                for edge_row in all_edge_rows:
                    src = edge_row.get("source_node_id")
                    tgt = edge_row.get("target_node_id")
                    if src:
                        seen_nodes.add(src)
                    if tgt:
                        seen_nodes.add(tgt)
            else:
                # Positional array — column positions are unknown, fall back to client BFS
                logger.warning(
                    f"[{self.workspace}] PPL returned positional arrays, falling back to client BFS"
                )
                return await self._bfs_subgraph(start_label, max_depth, max_nodes)

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning(
                f"[{self.workspace}] Error parsing PPL response, falling back: {e}"
            )
            return await self._bfs_subgraph(start_label, max_depth, max_nodes)

        # Limit to max_nodes
        node_ids = list(seen_nodes)[:max_nodes]
        result.is_truncated = len(seen_nodes) > max_nodes

        # Batch fetch node data (start node already added)
        new_node_ids = [nid for nid in node_ids if nid != start_label]
        if new_node_ids:
            node_resp = await self.client.mget(
                index=self._nodes_index, body={"ids": new_node_ids}
            )
            for doc in node_resp["docs"]:
                if doc.get("found"):
                    result.nodes.append(
                        self._construct_graph_node(doc["_id"], doc["_source"])
                    )

        # Re-fetch full edge data between collected nodes for complete properties
        if node_ids:
            edge_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"source_node_id": node_ids}},
                            {"terms": {"target_node_id": node_ids}},
                        ]
                    }
                },
                "size": 10000,
            }
            edge_resp = await self.client.search(
                index=self._edges_index, body=edge_body
            )
            seen_edges = set()
            for hit in edge_resp["hits"]["hits"]:
                e = hit["_source"]
                eid = f"{e['source_node_id']}-{e['target_node_id']}"
                if eid not in seen_edges:
                    seen_edges.add(eid)
                    result.edges.append(self._construct_graph_edge(eid, e))

        return result

    @staticmethod
    def _escape_ppl(value: str) -> str:
        """Escape a string for safe inclusion in a PPL single-quoted literal."""
        return value.replace("\\", "\\\\").replace("'", "\\'")

    async def _bfs_subgraph(
        self, start_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """BFS traversal from a starting node, batching neighbor lookups per level."""
        result = KnowledgeGraph()
        seen_nodes = set()

        # Verify start node exists
        start_node = await self.get_node(start_label)
        if not start_node:
            return result

        seen_nodes.add(start_label)
        result.nodes.append(self._construct_graph_node(start_label, start_node))

        current_level = [start_label]
        for _ in range(max_depth):
            if not current_level or len(seen_nodes) >= max_nodes:
                break

            # Batch fetch all edges for current level
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_node_id": current_level}},
                            {"terms": {"target_node_id": current_level}},
                        ]
                    }
                },
                "_source": ["source_node_id", "target_node_id"],
                "size": 10000,
            }
            try:
                resp = await self.client.search(index=self._edges_index, body=body)
            except OpenSearchException:
                break

            next_level = set()
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]["source_node_id"]
                tgt = hit["_source"]["target_node_id"]
                if src not in seen_nodes:
                    next_level.add(src)
                if tgt not in seen_nodes:
                    next_level.add(tgt)

            # Limit to max_nodes
            new_ids = []
            for nid in next_level:
                if len(seen_nodes) + len(new_ids) >= max_nodes:
                    break
                new_ids.append(nid)

            if new_ids:
                # Batch fetch node data
                node_resp = await self.client.mget(
                    index=self._nodes_index, body={"ids": new_ids}
                )
                for doc in node_resp["docs"]:
                    if doc.get("found"):
                        seen_nodes.add(doc["_id"])
                        result.nodes.append(
                            self._construct_graph_node(doc["_id"], doc["_source"])
                        )

            current_level = new_ids

        # Fetch all edges between seen nodes
        all_ids = list(seen_nodes)
        if all_ids:
            edge_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"source_node_id": all_ids}},
                            {"terms": {"target_node_id": all_ids}},
                        ]
                    }
                },
                "size": 10000,
            }
            try:
                edge_resp = await self.client.search(
                    index=self._edges_index, body=edge_body
                )
                seen_edges = set()
                for hit in edge_resp["hits"]["hits"]:
                    e = hit["_source"]
                    eid = f"{e['source_node_id']}-{e['target_node_id']}"
                    if eid not in seen_edges:
                        seen_edges.add(eid)
                        result.edges.append(self._construct_graph_edge(eid, e))
            except OpenSearchException:
                pass

        result.is_truncated = len(seen_nodes) >= max_nodes
        return result

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes with their properties."""
        try:
            nodes = []
            pit = await self.client.create_pit(
                index=self._nodes_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        node = hit["_source"]
                        node["id"] = hit["_id"]
                        nodes.append(node)
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            return nodes
        except OpenSearchException:
            return []

    async def get_all_edges(self) -> list[dict]:
        """Get all edges with source/target fields added."""
        try:
            edges = []
            pit = await self.client.create_pit(
                index=self._edges_index, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": {"match_all": {}},
                        "size": 10000,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        edge = hit["_source"]
                        edge["source"] = edge.get("source_node_id")
                        edge["target"] = edge.get("target_node_id")
                        edges.append(edge)
                    search_after = hits[-1]["sort"]
                    if len(hits) < 10000:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
            return edges
        except OpenSearchException:
            return []

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get node labels ranked by edge degree (most connected first)."""
        try:
            body = {
                "size": 0,
                "aggs": {
                    "src": {"terms": {"field": "source_node_id", "size": limit * 2}},
                    "tgt": {"terms": {"field": "target_node_id", "size": limit * 2}},
                },
            }
            response = await self.client.search(index=self._edges_index, body=body)
            degree_map = {}
            for bucket in response["aggregations"]["src"]["buckets"]:
                degree_map[bucket["key"]] = (
                    degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                )
            for bucket in response["aggregations"]["tgt"]["buckets"]:
                degree_map[bucket["key"]] = (
                    degree_map.get(bucket["key"], 0) + bucket["doc_count"]
                )
            sorted_labels = sorted(degree_map, key=degree_map.get, reverse=True)[:limit]
            return sorted_labels
        except OpenSearchException:
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search node labels with wildcard and prefix matching."""
        query = query.strip()
        if not query:
            return []
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"entity_id": {"value": query, "boost": 10}}},
                            {
                                "prefix": {
                                    "entity_id": {"value": query.lower(), "boost": 5}
                                }
                            },
                            {
                                "wildcard": {
                                    "entity_id": {
                                        "value": f"*{query.lower()}*",
                                        "case_insensitive": True,
                                        "boost": 2,
                                    }
                                }
                            },
                        ]
                    }
                },
                "_source": False,
                "size": limit,
            }
            response = await self.client.search(index=self._nodes_index, body=body)
            return [hit["_id"] for hit in response["hits"]["hits"]]
        except OpenSearchException:
            return []

    async def index_done_callback(self) -> None:
        """Refresh both node and edge indices."""
        try:
            await self.client.indices.refresh(index=self._nodes_index)
            await self.client.indices.refresh(index=self._edges_index)
        except Exception:
            pass

    async def drop(self) -> dict[str, str]:
        """Delete both node and edge indices."""
        try:
            for idx in (self._nodes_index, self._edges_index):
                try:
                    await self.client.indices.delete(index=idx)
                except NotFoundError:
                    pass
            logger.info(f"[{self.workspace}] Dropped graph indices")
            return {"status": "success", "message": "Graph indices dropped"}
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error dropping graph indices: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchVectorDBStorage(BaseVectorStorage):
    """Vector storage using OpenSearch k-NN plugin with corrected cosine score handling."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)

    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    def __post_init__(self):
        self._validate_embedding_func()
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        """Initialize client and create k-NN vector index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_knn_index_if_not_exists()
            logger.debug(
                f"[{self.workspace}] OpenSearch Vector storage initialized: {self._index_name}"
            )

    async def _create_knn_index_if_not_exists(self):
        try:
            if await self.client.indices.exists(index=self._index_name):
                # Validate existing index dimension
                try:
                    mapping = await self.client.indices.get_mapping(
                        index=self._index_name
                    )
                    existing_dim = (
                        mapping[self._index_name]["mappings"]["properties"]
                        .get("vector", {})
                        .get("dimension")
                    )
                    expected_dim = self.embedding_func.embedding_dim
                    if existing_dim is not None and existing_dim != expected_dim:
                        raise ValueError(
                            f"Vector dimension mismatch! Index '{self._index_name}' has "
                            f"dimension {existing_dim}, but current embedding model expects "
                            f"dimension {expected_dim}. Please drop the existing index or "
                            f"use an embedding model with matching dimensions."
                        )
                except (KeyError, TypeError):
                    logger.warning(
                        f"[{self.workspace}] Could not read vector mapping for index "
                        f"'{self._index_name}'; skipping dimension validation"
                    )
                return

            ef_construction = int(
                _get_opensearch_env("OPENSEARCH_KNN_EF_CONSTRUCTION", "200")
            )
            m = int(_get_opensearch_env("OPENSEARCH_KNN_M", "16"))
            ef_search = int(_get_opensearch_env("OPENSEARCH_KNN_EF_SEARCH", "100"))

            body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": ef_search,
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": self.embedding_func.embedding_dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": ef_construction,
                                    "m": m,
                                },
                            },
                        },
                        "content": {"type": "text"},
                        "entity_name": {"type": "keyword"},
                        "src_id": {"type": "keyword"},
                        "tgt_id": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "created_at": {"type": "long"},
                    },
                    "dynamic": True,
                },
            }
            await self.client.indices.create(index=self._index_name, body=body)
            logger.info(
                f"[{self.workspace}] Created k-NN index: {self._index_name} "
                f"(dim={self.embedding_func.embedding_dim})"
            )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                logger.error(f"[{self.workspace}] Error creating k-NN index: {e}")
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating k-NN index: {e}")
            raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Generate embeddings and upsert vectors in batches."""
        if not data:
            return
        logger.debug(
            f"[{self.workspace}] Upserting {len(data)} vectors to {self.namespace}"
        )
        current_time = int(time.time())

        list_data = [
            {
                "_id": k,
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        for i, doc in enumerate(list_data):
            doc["vector"] = embeddings[i].tolist()

        actions = [
            {
                "_op_type": "index",
                "_index": self._index_name,
                "_id": doc["_id"],
                "_source": {k: v for k, v in doc.items() if k != "_id"},
            }
            for doc in list_data
        ]
        try:
            success, failed = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
            if failed:
                logger.warning(
                    f"[{self.workspace}] {len(failed)} vectors failed to upsert"
                )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting vectors: {e}")
            raise

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """k-NN similarity search with cosine score conversion for lucene engine."""
        if query_embedding is not None:
            query_vector = (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            )
        else:
            embedding = await self.embedding_func([query], _priority=5)
            query_vector = embedding[0].tolist()

        search_body = {
            "size": top_k,
            "query": {"knn": {"vector": {"vector": query_vector, "k": top_k}}},
            "_source": {"excludes": ["vector"]},
        }
        try:
            response = await self.client.search(
                index=self._index_name, body=search_body
            )
            results = []
            for hit in response["hits"]["hits"]:
                # OpenSearch k-NN with lucene engine and cosinesimil space type
                # returns scores that can be used directly as similarity measure.
                score = hit["_score"]

                if score >= self.cosine_better_than_threshold:
                    doc = hit["_source"]
                    doc["id"] = hit["_id"]
                    doc["distance"] = score
                    results.append(doc)
            logger.info(
                f"[{self.workspace}] Vector query on {self._index_name}: "
                f"top_k={top_k}, threshold={self.cosine_better_than_threshold}, "
                f"total_hits={len(response['hits']['hits'])}, "
                f"passed_filter={len(results)}, "
                f"score_range=[{min((h['_score'] for h in response['hits']['hits']), default=0):.4f}, "
                f"{max((h['_score'] for h in response['hits']['hits']), default=0):.4f}]"
            )
            return results
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error querying vectors: {e}")
            return []

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed vectors searchable."""
        try:
            await self.client.indices.refresh(index=self._index_name)
        except Exception:
            pass

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a vector document by ID."""
        try:
            response = await self.client.get(index=self._index_name, id=id)
            doc = response["_source"]
            doc["id"] = response["_id"]
            return doc
        except NotFoundError:
            return None
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting vector {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector documents by IDs, preserving order."""
        if not ids:
            return []
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["id"] = doc["_id"]
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting vectors by ids: {e}")
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get only the vector embeddings for given IDs."""
        if not ids:
            return {}
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": ids}, _source_includes=["vector"]
            )
            result = {}
            for doc in response["docs"]:
                if doc.get("found") and "vector" in doc.get("_source", {}):
                    result[doc["_id"]] = doc["_source"]["vector"]
            return result
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error getting vectors: {e}")
            return {}

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by their IDs."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            result = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
            logger.debug(
                f"[{self.workspace}] Deleted {result[0]} vectors from {self.namespace}"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error deleting vectors: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity vector by computing its hash ID."""
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            try:
                await self.client.delete(
                    index=self._index_name, id=entity_id, refresh="wait_for"
                )
                logger.debug(f"[{self.workspace}] Deleted entity {entity_name}")
            except NotFoundError:
                logger.debug(f"[{self.workspace}] Entity {entity_name} not found")
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt."""
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"src_id": entity_name}},
                            {"term": {"tgt_id": entity_name}},
                        ]
                    }
                }
            }
            await self.client.delete_by_query(
                index=self._index_name, body=body, refresh=True
            )
            logger.debug(
                f"[{self.workspace}] Deleted relations for entity {entity_name}"
            )
        except OpenSearchException as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        """Delete and recreate the vector index."""
        try:
            await self.client.indices.delete(index=self._index_name)
            # Recreate the index
            await self._create_knn_index_if_not_exists()
            logger.info(
                f"[{self.workspace}] Dropped and recreated vector index: {self._index_name}"
            )
            return {
                "status": "success",
                "message": f"Vector index {self._index_name} dropped and recreated",
            }
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error dropping vector index: {e}")
            return {"status": "error", "message": str(e)}
