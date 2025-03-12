import os
import time
import asyncio
from typing import Any, final
import json
import numpy as np

from dataclasses import dataclass
import pipmaster as pm

from lightrag.utils import logger, compute_mdhash_id
from lightrag.base import BaseVectorStorage

if not pm.is_installed("faiss"):
    pm.install("faiss")

import faiss  # type: ignore
from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
    is_multiprocess,
)


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
        self._faiss_index_file = os.path.join(
            self.global_config["working_dir"], f"faiss_index_{self.namespace}.index"
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
        self.storage_updated = await get_update_flag(self.namespace)
        # Get the storage lock for use in other methods
        self._storage_lock = get_storage_lock()

    async def _get_index(self):
        """Check if the shtorage should be reloaded"""
        # Acquire lock to prevent concurrent read and write
        async with self._storage_lock:
            # Check if storage was updated by another process
            if (is_multiprocess and self.storage_updated.value) or (
                not is_multiprocess and self.storage_updated
            ):
                logger.info(
                    f"Process {os.getpid()} FAISS reloading {self.namespace} due to update by another process"
                )
                # Reload data
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_to_meta = {}
                self._load_faiss_index()
                if is_multiprocess:
                    self.storage_updated.value = False
                else:
                    self.storage_updated = False
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
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        current_time = time.time()

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
                f"Embedding size mismatch. Embeddings: {len(embeddings)}, Data: {len(list_data)}"
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

        logger.info(f"Upserted {len(list_data)} vectors into Faiss index.")
        return [m["__id__"] for m in list_data]

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Search by a textual query; returns top_k results with their metadata + similarity distance.
        """
        embedding = await self.embedding_func([query])
        # embedding is shape (1, dim)
        embedding = np.array(embedding, dtype=np.float32)
        faiss.normalize_L2(embedding)  # we do in-place normalization

        logger.info(
            f"Query: {query}, top_k: {top_k}, threshold: {self.cosine_better_than_threshold}"
        )

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
            results.append(
                {
                    **meta,
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
        """
        logger.info(f"Deleting {len(ids)} vectors from {self.namespace}")
        to_remove = []
        for cid in ids:
            fid = self._find_faiss_id_by_custom_id(cid)
            if fid is not None:
                to_remove.append(fid)

        if to_remove:
            await self._remove_faiss_ids(to_remove)
        logger.debug(
            f"Successfully deleted {len(to_remove)} vectors from {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        logger.debug(f"Attempting to delete entity {entity_name} with ID {entity_id}")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Delete relations for a given entity by scanning metadata.
        """
        logger.debug(f"Searching relations for entity {entity_name}")
        relations = []
        for fid, meta in self._id_to_meta.items():
            if meta.get("src_id") == entity_name or meta.get("tgt_id") == entity_name:
                relations.append(fid)

        logger.debug(f"Found {len(relations)} relations for {entity_name}")
        if relations:
            await self._remove_faiss_ids(relations)
            logger.debug(f"Deleted {len(relations)} relations for {entity_name}")

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
            logger.warning("No existing Faiss index file found. Starting fresh.")
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
                f"Faiss index loaded with {self._index.ntotal} vectors from {self._faiss_index_file}"
            )
        except Exception as e:
            logger.error(f"Failed to load Faiss index or metadata: {e}")
            logger.warning("Starting with an empty Faiss index.")
            self._index = faiss.IndexFlatIP(self._dim)
            self._id_to_meta = {}

    async def index_done_callback(self) -> None:
        # Check if storage was updated by another process
        if is_multiprocess and self.storage_updated.value:
            # Storage was updated by another process, reload data instead of saving
            logger.warning(
                f"Storage for FAISS {self.namespace} was updated by another process, reloading..."
            )
            async with self._storage_lock:
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
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                if is_multiprocess:
                    self.storage_updated.value = False
                else:
                    self.storage_updated = False
            except Exception as e:
                logger.error(f"Error saving FAISS index for {self.namespace}: {e}")
                return False  # Return error

        return True  # Return success

    async def search_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Search for records with IDs starting with a specific prefix.

        Args:
            prefix: The prefix to search for in record IDs

        Returns:
            List of records with matching ID prefixes
        """
        matching_records = []

        # Search for records with IDs starting with the prefix
        for faiss_id, meta in self._id_to_meta.items():
            if "__id__" in meta and meta["__id__"].startswith(prefix):
                # Create a copy of all metadata and add "id" field
                record = {**meta, "id": meta["__id__"]}
                matching_records.append(record)

        logger.debug(f"Found {len(matching_records)} records with prefix '{prefix}'")
        return matching_records

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

        return {**metadata, "id": metadata.get("__id__")}

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        results = []
        for id in ids:
            fid = self._find_faiss_id_by_custom_id(id)
            if fid is not None:
                metadata = self._id_to_meta.get(fid, {})
                if metadata:
                    results.append({**metadata, "id": metadata.get("__id__")})

        return results
