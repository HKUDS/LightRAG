import os
import time
import asyncio
from typing import Any, final
import json
import numpy as np
from dataclasses import dataclass

from lightrag.utils import logger, compute_mdhash_id
from lightrag.base import BaseVectorStorage

from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)

# You must manually install faiss-cpu or faiss-gpu before using FAISS vector db
import faiss  # type: ignore


@final
@dataclass
class FaissVectorDBStorage(BaseVectorStorage):
    """
    A Faiss-based Vector DB Storage for LightRAG.
    Uses cosine similarity by storing normalized vectors in a Faiss index with inner product search.
    """

    def __post_init__(self):
        # Grab config values if available
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Where to save index file if you want persistent storage
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)

        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._faiss_index_file = os.path.join(
            workspace_dir, f"faiss_index_{self.namespace}.index"
        )
        self._meta_file = self._faiss_index_file + ".meta.json"

        self._max_batch_size = self.global_config["embedding_batch_num"]
        # Embedding dimension (e.g. 768) must match your embedding function
        self._dim = self.embedding_func.embedding_dim

        # Create an empty Faiss index for inner product (useful for normalized vectors = cosine similarity).
        # If you have a large number of vectors, you might want IVF or other indexes.
        # For demonstration, we use a simple IndexFlatIP.
        self._index = faiss.IndexFlatIP(self._dim)
        # Keep a local store for metadata, IDs, etc.
        # Maps <int faiss_id> → metadata (including your original ID).
        self._id_to_meta = {}

        self._load_faiss_index()

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        # Get the storage lock for use in other methods
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )

    async def _get_index(self):
        """Check if the shtorage should be reloaded"""
        # Acquire lock to prevent concurrent read and write
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} FAISS reloading {self.namespace} due to update by another process"
                )
                # Reload data
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_to_meta = {}
                self._load_faiss_index()
                self.storage_updated.value = False
            return self._index

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Insert or update vectors in the Faiss index.

        data: {
           "custom_id_1": {
               "content": <text>,
               ...metadata...
           },
           "custom_id_2": {
               "content": <text>,
               ...metadata...
           },
           ...
        }
        """
        logger.debug(
            f"[{self.workspace}] FAISS: Inserting {len(data)} to {self.namespace}"
        )
        if not data:
            return

        current_time = int(time.time())

        # Prepare data for embedding
        list_data = []
        contents = []
        for k, v in data.items():
            # Store only known meta fields if needed
            meta = {mf: v[mf] for mf in self.meta_fields if mf in v}
            meta["__id__"] = k
            meta["__created_at__"] = current_time
            list_data.append(meta)
            contents.append(v["content"])

        # Split into batches for embedding if needed
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        # Flatten the list of arrays
        embeddings = np.concatenate(embeddings_list, axis=0)
        if len(embeddings) != len(list_data):
            logger.error(
                f"[{self.workspace}] Embedding size mismatch. Embeddings: {len(embeddings)}, Data: {len(list_data)}"
            )
            return []

        # Convert to float32 and normalize embeddings for cosine similarity (in-place)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        # Upsert logic:
        # 1. Identify which vectors to remove if they exist
        # 2. Remove them
        # 3. Add the new vectors
        existing_ids_to_remove = []
        for meta, emb in zip(list_data, embeddings):
            faiss_internal_id = self._find_faiss_id_by_custom_id(meta["__id__"])
            if faiss_internal_id is not None:
                existing_ids_to_remove.append(faiss_internal_id)

        if existing_ids_to_remove:
            await self._remove_faiss_ids(existing_ids_to_remove)

        # Step 2: Add new vectors
        index = await self._get_index()
        start_idx = index.ntotal
        index.add(embeddings)

        # Step 3: Store metadata + vector for each new ID
        for i, meta in enumerate(list_data):
            fid = start_idx + i
            # Store the raw vector so we can rebuild if something is removed
            meta["__vector__"] = embeddings[i].tolist()
            self._id_to_meta.update({fid: meta})

        logger.debug(
            f"[{self.workspace}] Upserted {len(list_data)} vectors into Faiss index."
        )
        return [m["__id__"] for m in list_data]

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """
        Search by a textual query; returns top_k results with their metadata + similarity distance.
        """
        if query_embedding is not None:
            embedding = np.array([query_embedding], dtype=np.float32)
        else:
            embedding = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query
            # embedding is shape (1, dim)
            embedding = np.array(embedding, dtype=np.float32)

        faiss.normalize_L2(embedding)  # we do in-place normalization

        # Perform the similarity search
        index = await self._get_index()
        distances, indices = index.search(embedding, top_k)

        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                # Faiss returns -1 if no neighbor
                continue

            # Cosine similarity threshold
            if dist < self.cosine_better_than_threshold:
                continue

            meta = self._id_to_meta.get(idx, {})
            # Filter out __vector__ from query results to avoid returning large vector data
            filtered_meta = {k: v for k, v in meta.items() if k != "__vector__"}
            results.append(
                {
                    **filtered_meta,
                    "id": meta.get("__id__"),
                    "distance": float(dist),
                    "created_at": meta.get("__created_at__"),
                }
            )

        return results

    @property
    def client_storage(self):
        # Return whatever structure LightRAG might need for debugging
        return {"data": list(self._id_to_meta.values())}

    async def delete(self, ids: list[str]):
        """
        Delete vectors for the provided custom IDs.

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        logger.debug(
            f"[{self.workspace}] Deleting {len(ids)} vectors from {self.namespace}"
        )
        to_remove = []
        for cid in ids:
            fid = self._find_faiss_id_by_custom_id(cid)
            if fid is not None:
                to_remove.append(fid)

        if to_remove:
            await self._remove_faiss_ids(to_remove)
        logger.debug(
            f"[{self.workspace}] Successfully deleted {len(to_remove)} vectors from {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        logger.debug(
            f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
        )
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        logger.debug(f"[{self.workspace}] Searching relations for entity {entity_name}")
        relations = []
        for fid, meta in self._id_to_meta.items():
            if meta.get("src_id") == entity_name or meta.get("tgt_id") == entity_name:
                relations.append(fid)

        logger.debug(
            f"[{self.workspace}] Found {len(relations)} relations for {entity_name}"
        )
        if relations:
            await self._remove_faiss_ids(relations)
            logger.debug(
                f"[{self.workspace}] Deleted {len(relations)} relations for {entity_name}"
            )

    # --------------------------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------------------------

    def _find_faiss_id_by_custom_id(self, custom_id: str):
        """
        Return the Faiss internal ID for a given custom ID, or None if not found.
        """
        for fid, meta in self._id_to_meta.items():
            if meta.get("__id__") == custom_id:
                return fid
        return None

    async def _remove_faiss_ids(self, fid_list):
        """
        Remove a list of internal Faiss IDs from the index.
        Because IndexFlatIP doesn't support 'removals',
        we rebuild the index excluding those vectors.
        """
        keep_fids = [fid for fid in self._id_to_meta if fid not in fid_list]

        # Rebuild the index
        vectors_to_keep = []
        new_id_to_meta = {}
        for new_fid, old_fid in enumerate(keep_fids):
            vec_meta = self._id_to_meta[old_fid]
            vectors_to_keep.append(vec_meta["__vector__"])  # stored as list
            new_id_to_meta[new_fid] = vec_meta

        async with self._storage_lock:
            # Re-init index
            self._index = faiss.IndexFlatIP(self._dim)
            if vectors_to_keep:
                arr = np.array(vectors_to_keep, dtype=np.float32)
                self._index.add(arr)

            self._id_to_meta = new_id_to_meta

    def _save_faiss_index(self):
        """
        Save the current Faiss index + metadata to disk so it can persist across runs.
        """
        faiss.write_index(self._index, self._faiss_index_file)

        # Save metadata dict to JSON. Convert all keys to strings for JSON storage.
        # _id_to_meta is { int: { '__id__': doc_id, '__vector__': [float,...], ... } }
        # We'll keep the int -> dict, but JSON requires string keys.
        serializable_dict = {}
        for fid, meta in self._id_to_meta.items():
            serializable_dict[str(fid)] = meta

        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f)

    def _load_faiss_index(self):
        """
        Load the Faiss index + metadata from disk if it exists,
        and rebuild in-memory structures so we can query.
        """
        if not os.path.exists(self._faiss_index_file):
            logger.warning(
                f"[{self.workspace}] No existing Faiss index file found for {self.namespace}"
            )
            return

        try:
            # Load the Faiss index
            self._index = faiss.read_index(self._faiss_index_file)
            # Load metadata
            with open(self._meta_file, "r", encoding="utf-8") as f:
                stored_dict = json.load(f)

            # Convert string keys back to int
            self._id_to_meta = {}
            for fid_str, meta in stored_dict.items():
                fid = int(fid_str)
                self._id_to_meta[fid] = meta

            logger.info(
                f"[{self.workspace}] Faiss index loaded with {self._index.ntotal} vectors from {self._faiss_index_file}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to load Faiss index or metadata: {e}"
            )
            logger.warning(f"[{self.workspace}] Starting with an empty Faiss index.")
            self._index = faiss.IndexFlatIP(self._dim)
            self._id_to_meta = {}

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.warning(
                    f"[{self.workspace}] Storage for FAISS {self.namespace} was updated by another process, reloading..."
                )
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_to_meta = {}
                self._load_faiss_index()
                self.storage_updated.value = False
                return False  # Return error

        # Acquire lock and perform persistence
        async with self._storage_lock:
            try:
                # Save data to disk
                self._save_faiss_index()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error saving FAISS index for {self.namespace}: {e}"
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
        # Find the Faiss internal ID for the custom ID
        fid = self._find_faiss_id_by_custom_id(id)
        if fid is None:
            return None

        # Get the metadata for the found ID
        metadata = self._id_to_meta.get(fid, {})
        if not metadata:
            return None

        # Filter out __vector__ from metadata to avoid returning large vector data
        filtered_metadata = {k: v for k, v in metadata.items() if k != "__vector__"}
        return {
            **filtered_metadata,
            "id": metadata.get("__id__"),
            "created_at": metadata.get("__created_at__"),
        }

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        results: list[dict[str, Any] | None] = []
        for id in ids:
            record = None
            fid = self._find_faiss_id_by_custom_id(id)
            if fid is not None:
                metadata = self._id_to_meta.get(fid)
                if metadata:
                    # Filter out __vector__ from metadata to avoid returning large vector data
                    filtered_metadata = {
                        k: v for k, v in metadata.items() if k != "__vector__"
                    }
                    record = {
                        **filtered_metadata,
                        "id": metadata.get("__id__"),
                        "created_at": metadata.get("__created_at__"),
                    }
            results.append(record)

        return results

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

        vectors_dict = {}
        for id in ids:
            # Find the Faiss internal ID for the custom ID
            fid = self._find_faiss_id_by_custom_id(id)
            if fid is not None and fid in self._id_to_meta:
                metadata = self._id_to_meta[fid]
                # Get the stored vector from metadata
                if "__vector__" in metadata:
                    vectors_dict[id] = metadata["__vector__"]

        return vectors_dict

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will:
        1. Remove the vector database storage file if it exists
        2. Reinitialize the vector database client
        3. Update flags to notify other processes
        4. Changes is persisted to disk immediately

        This method will remove all vectors from the Faiss index and delete the storage files.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                # Reset the index
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_to_meta = {}

                # Remove storage files if they exist
                if os.path.exists(self._faiss_index_file):
                    os.remove(self._faiss_index_file)
                if os.path.exists(self._meta_file):
                    os.remove(self._meta_file)

                self._id_to_meta = {}
                self._load_faiss_index()

                # Notify other processes
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                self.storage_updated.value = False

                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop FAISS index {self.namespace}"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping FAISS index {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
