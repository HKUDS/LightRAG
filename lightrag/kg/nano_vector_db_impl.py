import asyncio
import base64
import os
import zlib
from typing import Any, final
from dataclasses import dataclass
import numpy as np
import time

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)

from lightrag.base import BaseVectorStorage
from nano_vectordb import NanoVectorDB
from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
)


@final
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        # Initialize basic attributes
        self._client = None
        self._storage_lock = None
        self.storage_updated = None

        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
            self.final_namespace = f"{self.workspace}_{self.namespace}"
        else:
            # Default behavior when workspace is empty
            self.final_namespace = self.namespace
            self.workspace = "_"
            workspace_dir = working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        self._client_file_name = os.path.join(
            workspace_dir, f"vdb_{self.namespace}.json"
        )

        self._max_batch_size = self.global_config["embedding_batch_num"]

        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim,
            storage_file=self._client_file_name,
        )

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(self.final_namespace)
        # Get the storage lock for use in other methods
        self._storage_lock = get_storage_lock(enable_logging=False)

    async def _get_client(self):
        """Check if the storage should be reloaded"""
        # Acquire lock to prevent concurrent read and write
        async with self._storage_lock:
            # Check if data needs to be reloaded
            if self.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} reloading {self.namespace} due to update by another process"
                )
                # Reload data
                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )
                # Reset update flag
                self.storage_updated.value = False

            return self._client

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        # logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        current_time = int(time.time())
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        # Execute embedding outside of lock to avoid long lock times
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                # Compress vector using Float16 + zlib + Base64 for storage optimization
                vector_f16 = embeddings[i].astype(np.float16)
                compressed_vector = zlib.compress(vector_f16.tobytes())
                encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
                d["vector"] = encoded_vector
                d["__vector__"] = embeddings[i]
            client = await self._get_client()
            results = client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"[{self.workspace}] embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = query_embedding
        else:
            # Execute embedding outside of lock to avoid improve cocurrent
            embedding = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query
            embedding = embedding[0]

        client = await self._get_client()
        results = client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {
                **{k: v for k, v in dp.items() if k != "vector"},
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results

    @property
    async def client_storage(self):
        client = await self._get_client()
        return getattr(client, "_NanoVectorDB__storage")

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            client = await self._get_client()
            # Record count before deletion
            before_count = len(client)

            client.delete(ids)

            # Calculate actual deleted count
            after_count = len(client)
            deleted_count = before_count - after_count

            logger.debug(
                f"[{self.workspace}] Successfully deleted {deleted_count} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Check if the entity exists
            client = await self._get_client()
            if client.get([entity_id]):
                client.delete([entity_id])
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

        try:
            client = await self._get_client()
            storage = getattr(client, "_NanoVectorDB__storage")
            relations = [
                dp
                for dp in storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            logger.debug(
                f"[{self.workspace}] Found {len(relations)} relations for entity {entity_name}"
            )
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                client = await self._get_client()
                client.delete(ids_to_delete)
                logger.debug(
                    f"[{self.workspace}] Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def index_done_callback(self) -> bool:
        """Save data to disk"""
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.warning(
                    f"[{self.workspace}] Storage for {self.namespace} was updated by another process, reloading..."
                )
                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )
                # Reset update flag
                self.storage_updated.value = False
                return False  # Return error

        # Acquire lock and perform persistence
        async with self._storage_lock:
            try:
                # Save data to disk
                self._client.save()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.final_namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error saving data for {self.namespace}: {e}"
                )
                return False  # Return error

        return True  # Return success

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        client = await self._get_client()
        result = client.get([id])
        if result:
            dp = result[0]
            return {
                **{k: v for k, v in dp.items() if k != "vector"},
                "id": dp.get("__id__"),
                "created_at": dp.get("__created_at__"),
            }
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        client = await self._get_client()
        results = client.get(ids)
        result_map: dict[str, dict[str, Any]] = {}

        for dp in results:
            if not dp:
                continue
            record = {
                **{k: v for k, v in dp.items() if k != "vector"},
                "id": dp.get("__id__"),
                "created_at": dp.get("__created_at__"),
            }
            key = record.get("id")
            if key is not None:
                result_map[str(key)] = record

        ordered_results: list[dict[str, Any] | None] = []
        for requested_id in ids:
            ordered_results.append(result_map.get(str(requested_id)))

        return ordered_results

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        client = await self._get_client()
        results = client.get(ids)

        vectors_dict = {}
        for result in results:
            if result and "vector" in result and "__id__" in result:
                # Decompress vector data (Base64 + zlib + Float16 compressed)
                decoded = base64.b64decode(result["vector"])
                decompressed = zlib.decompress(decoded)
                vector_f16 = np.frombuffer(decompressed, dtype=np.float16)
                vector_f32 = vector_f16.astype(np.float32).tolist()
                vectors_dict[result["__id__"]] = vector_f32

        return vectors_dict

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will:
        1. Remove the vector database storage file if it exists
        2. Reinitialize the vector database client
        3. Update flags to notify other processes
        4. Changes is persisted to disk immediately

        This method is intended for use in scenarios where all data needs to be removed,

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                # delete _client_file_name
                if os.path.exists(self._client_file_name):
                    os.remove(self._client_file_name)

                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )

                # Notify other processes that data has been updated
                await set_all_update_flags(self.final_namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False

                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}(file:{self._client_file_name})"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
