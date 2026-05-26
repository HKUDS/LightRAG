import glob
import os
import time
import asyncio
from typing import Any, final
import json
import numpy as np
from dataclasses import dataclass

from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
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
    """Faiss-backed vector storage for LightRAG.

    Uses cosine similarity by storing L2-normalized vectors in an
    ``IndexFlatIP`` (inner-product search on normalized vectors == cosine).

    Storage model:
        Two on-disk files per ``(workspace, namespace)``:
            * ``working_dir/[workspace/]faiss_index_<namespace>.index`` —
              the Faiss index (binary, written by ``faiss.write_index``).
            * ``…<namespace>.index.meta.json`` — the ``_id_to_meta`` dict
              serialized as JSON, **without** the ``__vector__`` field
              (vectors are reconstructed from the Faiss index on load).
        In memory the storage is split across two fields:
            * ``self._index`` — the Faiss index.
            * ``self._id_to_meta`` — ``dict[int_faiss_id, metadata]``.
        Both files are the **only** cross-process synchronization surface
        — there is no shared memory between processes. Cross-process
        visibility is mediated by (a) per-file atomic writes and (b) a
        per-namespace ``storage_updated`` flag distributed through
        ``lightrag.kg.shared_storage``.

        **Cross-file atomicity is not guaranteed**: the two ``atomic_write``
        renames in ``_save_faiss_index`` are independent, so a crash
        between them can leave ``.index`` and ``.meta.json`` referring to
        different snapshots. ``_load_faiss_index`` tolerates this by
        skipping metadata rows whose ``fid`` exceeds ``self._index.ntotal``.

    Concurrency invariants (the code here is correct *only* while all
    three hold):
        1. **Single writer per workspace.** The document pipeline's
           ``busy`` / ``destructive_busy`` flags (see ``AGENTS.md``
           *Pipeline concurrency contract*) guarantee at most one process
           performs ``upsert`` / ``delete`` / ``index_done_callback`` at
           any time. Every other process is read-only.
        2. **Eventual consistency is sufficient.** Read-only processes
           only need to observe the writer's data *after* the writer's
           ``index_done_callback`` completes. Reads in the gap between a
           writer's in-memory mutation and its commit may legitimately
           return the pre-update snapshot.
        3. **Faiss + dict mutations are synchronous.** Under a
           single-threaded asyncio event loop, ``index.add`` /
           ``index.search`` / ``self._id_to_meta`` mutations cannot be
           preempted by another coroutine, which gives them implicit
           mutual exclusion. This is why most methods don't hold
           ``_storage_lock`` while touching ``self._index`` /
           ``self._id_to_meta``.

    Cross-process sync protocol:
        Writer side (``index_done_callback``):
            1. ``_save_faiss_index`` writes both files atomically (per
               file; cross-file atomicity is best-effort, see above).
            2. ``set_all_update_flags`` flips every process's
               ``storage_updated`` flag (including the writer's own).
            3. Reset the writer's own flag to ``False`` so the next
               ``_get_index`` does not trigger a self-reload of what we
               just wrote.
        Reader side (any method that goes through ``_get_index``):
            1. Inside ``_storage_lock``, observe
               ``storage_updated.value is True``.
            2. **Fully reload**: re-init ``self._index`` from
               ``IndexFlatIP``, clear ``self._id_to_meta``, then call
               ``_load_faiss_index`` to re-parse both files. Faiss has no
               incremental sync API.
            3. Reset the reader's own flag.

    Lock scope:
        ``_storage_lock`` is a per-``(namespace, workspace)`` keyed lock
        spanning both intra-process coroutines and inter-process workers.
        It wraps:
            * ``_get_index`` reload checks.
            * The two critical sections in ``index_done_callback``.
            * The rebuild inside ``_remove_faiss_ids`` (which mutates
              ``self._index`` and ``self._id_to_meta`` together).
            * The entire ``drop`` body.
        It does **not** wrap routine ``index.search`` /
        ``self._id_to_meta`` reads or the ``index.add`` /
        ``self._id_to_meta.update`` mutations in ``upsert`` — those rely
        on invariant (3) above. If either premise is broken (e.g.
        Faiss calls moved to a thread pool), the lock scope must be
        widened.

    Caveat — methods that read ``_id_to_meta`` *without* going through
    ``_get_index``:
        ``client_storage`` (synchronous property), ``delete``,
        ``delete_entity_relation``, ``get_by_id``, ``get_by_ids``,
        ``get_vectors_by_ids``, and ``_find_faiss_id_by_custom_id``
        directly read ``self._id_to_meta`` without first calling
        ``_get_index``. In a reader process that has not yet observed a
        commit (no recent ``_get_index`` call), these can return data
        from before a writer's most recent ``index_done_callback``. This
        is consistent with invariant (2) but is a stricter staleness
        bound than NanoVectorDB's equivalents, which always funnel
        through ``_get_client`` first.

    Non-pipeline write paths:
        The pipeline ``busy`` gate serializes ``upsert`` / ``delete`` /
        ``index_done_callback`` called from document ingestion and purge.
        The following entry points are **not** serialized by the pipeline
        and must be guarded externally:
            * ``drop`` — gated by the API layer (``/documents/clear``
              takes the pipeline busy reservation before invoking it).
            * ``delete_entity`` / ``delete_entity_relation`` — currently
              not exposed in the WebUI. Any future caller must arrange
              single-writer serialization the same way the pipeline does.
    """

    def __post_init__(self):
        self._validate_embedding_func()
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

        # Sweep orphan tmp siblings left behind by hard kills mid-save.
        # The meta file also needs an extra pattern: legacy versions of this
        # storage wrote a fixed "<meta>.tmp" suffix without further dot-segments,
        # which the default ".tmp.*" pattern does not match.
        reap_orphan_tmp_files(self._faiss_index_file, self.workspace or "_")
        reap_orphan_tmp_files(
            self._meta_file,
            self.workspace or "_",
            extra_patterns=(glob.escape(self._meta_file) + ".tmp",),
        )

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
        """Return the live Faiss index, reloading from disk if needed.

        This is the entry point through which ``upsert`` and ``query``
        fetch ``self._index``. When another process has committed (via
        ``index_done_callback``) and flipped this process's
        ``storage_updated`` flag, the next call here rebuilds *both*
        ``self._index`` and ``self._id_to_meta`` by re-reading the
        ``.index`` + ``.meta.json`` pair from disk. Faiss has no
        incremental sync API — the reload is unconditionally a full
        file reload.

        Under the *Single writer* invariant (see class docstring), the
        reload branch never fires in the writer process: the writer
        resets its own flag at the end of every ``index_done_callback``.
        The branch exists for readers.

        Note that several methods (``client_storage``, ``delete``,
        ``delete_entity_relation``, ``get_by_*``) read
        ``self._id_to_meta`` directly without funnelling through this
        method — see class docstring *Caveat* for the staleness
        implications.

        ``_storage_lock`` is held during the check-and-reload to (a)
        serialize concurrent reload attempts by sibling coroutines and
        (b) interlock with ``index_done_callback`` so a reader cannot
        observe a partially-saved file pair.
        """
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
        """Insert or update vectors in the Faiss index; persistence is deferred.

        ``data`` shape::

            {
                "custom_id_1": {"content": <text>, ...metadata...},
                "custom_id_2": {"content": <text>, ...metadata...},
                ...
            }

        Persistence:
            Changes live only in this process's memory until the next
            ``index_done_callback``. Cross-process readers will not see
            them until that commit fires (see class docstring,
            *Cross-process sync protocol*).

        Concurrency:
            The embedding step runs **outside** ``_storage_lock`` on
            purpose — it can issue network / GPU calls and we don't want
            to hold the per-namespace lock for that latency. The
            existing-id lookup (``_find_faiss_id_by_custom_id``) and
            the rebuild path (``_remove_faiss_ids``) run *before*
            ``_get_index`` is called; ``_remove_faiss_ids`` takes the
            lock itself for its rebuild. The final ``index.add`` and
            ``self._id_to_meta.update`` mutations are unlocked and rely
            on the class docstring *Lock scope* invariant (synchronous
            Faiss ops + single-writer pipeline gate).
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

        embedding_tasks = [
            self.embedding_func(batch, context="document") for batch in batches
        ]
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
                [query], context="query", _priority=5
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
        """Return a snapshot view of the metadata dict for debugging.

        The outer list is a fresh shallow copy taken at access time, but
        each element is still a **live reference** into
        ``self._id_to_meta``; callers must not mutate them and must not
        retain them across operations that may rebuild the index
        (``upsert`` / ``delete`` / ``_remove_faiss_ids`` /
        ``_get_index`` reload), since a rebuild swaps ``self._index``
        and replaces ``self._id_to_meta`` with a new dict.

        This property is **synchronous and does not call** ``_get_index``,
        so in a reader process it can return data older than the latest
        committed snapshot until some other method triggers a reload —
        see class docstring *Caveat*.
        """
        return {"data": list(self._id_to_meta.values())}

    async def delete(self, ids: list[str]):
        """Delete vectors for the provided custom IDs.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. In ``lightrag.py`` this
            is handled by ``_insert_done()`` at the end of the document
            batch. Callers outside the pipeline must persist explicitly.

        Note: the id-resolution step ``_find_faiss_id_by_custom_id`` reads
        ``self._id_to_meta`` directly without going through
        ``_get_index``; the actual rebuild happens inside
        ``_remove_faiss_ids`` under ``_storage_lock``. See class
        docstring *Caveat*.
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
        """Delete the vector associated with a single entity name.

        Thin wrapper over ``delete([entity_id])`` where ``entity_id`` is
        ``compute_mdhash_id(entity_name, prefix="ent-")``.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        **Not pipeline-gated** — see class docstring
        *Non-pipeline write paths*. The caller is responsible for
        ensuring single-writer serialization.
        """
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        logger.debug(
            f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
        )
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete every relation vector incident to ``entity_name``.

        Scans ``self._id_to_meta`` for entries whose ``src_id`` or
        ``tgt_id`` matches and rebuilds the index via
        ``_remove_faiss_ids``.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Note: the scan reads ``self._id_to_meta`` directly without going
        through ``_get_index``. In a reader process this can miss
        relations added by a writer whose commit has not yet been
        observed locally; the actual rebuild happens inside
        ``_remove_faiss_ids`` under ``_storage_lock``. See class
        docstring *Caveat*.

        **Not pipeline-gated** — see class docstring
        *Non-pipeline write paths*. The caller is responsible for
        ensuring single-writer serialization.
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
        for old_fid in keep_fids:
            vec_meta = self._id_to_meta[old_fid]
            if "__vector__" in vec_meta:
                vec = vec_meta["__vector__"]
            elif old_fid < self._index.ntotal:
                vec = self._index.reconstruct(old_fid).tolist()
                vec_meta["__vector__"] = vec
            else:
                logger.warning(
                    f"[{self.workspace}] Skipping fid={old_fid} during rebuild: "
                    f"no vector and fid exceeds index size ({self._index.ntotal})"
                )
                continue
            new_fid = len(vectors_to_keep)
            vectors_to_keep.append(vec)
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

        Each file lands via a per-writer tmp + os.replace so a crash mid-write
        leaves the prior snapshot intact. Cross-file consistency between the
        .index and .meta.json (the two renames are not joint) is intentionally
        out of scope here.
        """
        atomic_write(
            self._faiss_index_file,
            lambda tmp: faiss.write_index(self._index, tmp),
            self.workspace or "_",
        )

        # Save metadata dict to JSON, excluding __vector__ since vectors are
        # already stored in the Faiss index file and can be reconstructed on load.
        serializable_dict = {}
        for fid, meta in self._id_to_meta.items():
            filtered_meta = {k: v for k, v in meta.items() if k != "__vector__"}
            serializable_dict[str(fid)] = filtered_meta

        def _write_meta(tmp: str) -> None:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(serializable_dict, f)

        atomic_write(self._meta_file, _write_meta, self.workspace or "_")

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

        dim_mismatch = False
        try:
            # Load the Faiss index
            self._index = faiss.read_index(self._faiss_index_file)

            # Verify dimension consistency between loaded index and embedding function
            if self._index.d != self._dim:
                error_msg = (
                    f"Dimension mismatch: loaded Faiss index has dimension {self._index.d}, "
                    f"but embedding function expects dimension {self._dim}. "
                    f"Please ensure the embedding model matches the stored index or rebuild the index."
                )
                logger.error(error_msg)
                dim_mismatch = True
                raise ValueError(error_msg)

            # Load metadata
            with open(self._meta_file, "r", encoding="utf-8") as f:
                stored_dict = json.load(f)

            # Convert string keys back to int and reconstruct vectors from index
            self._id_to_meta = {}
            for fid_str, meta in stored_dict.items():
                fid = int(fid_str)
                if fid >= self._index.ntotal:
                    logger.warning(
                        f"[{self.workspace}] Skipping metadata row fid={fid}: "
                        f"exceeds index size ({self._index.ntotal})"
                    )
                    continue
                if "__vector__" not in meta:
                    meta["__vector__"] = self._index.reconstruct(fid).tolist()
                self._id_to_meta[fid] = meta

            logger.info(
                f"[{self.workspace}] Faiss index loaded with {self._index.ntotal} vectors from {self._faiss_index_file}"
            )
        except Exception as e:
            if dim_mismatch:
                raise
            logger.error(
                f"[{self.workspace}] Failed to load Faiss index or metadata: {e}"
            )
            logger.warning(f"[{self.workspace}] Starting with an empty Faiss index.")
            self._index = faiss.IndexFlatIP(self._dim)
            self._id_to_meta = {}

    async def index_done_callback(self) -> None:
        """Commit in-memory state to disk and notify other processes.

        This is the writer's **commit point** in the cross-process sync
        protocol (see class docstring). Two effects, in order:
            1. ``_save_faiss_index`` atomically writes ``.index`` and
               ``.meta.json`` (per file; cross-file atomicity is *not*
               guaranteed — see ``_save_faiss_index`` and class docstring
               *Storage model*).
            2. ``set_all_update_flags`` flips every registered process's
               ``storage_updated`` flag, then we immediately reset our
               own flag to ``False`` so the writer does not self-reload
               on the next call to ``_get_index``.

        Two-block structure (intentional, do not collapse):
            * **First ``async with``** — early-return path for a
              hypothetical second writer. Under the current single-writer
              pipeline contract (class docstring, invariant 1) the
              ``storage_updated.value`` check is permanently ``False`` in
              the writer, so this branch is **dead code in production**.
              It is kept as defensive scaffolding for any future
              relaxation of the single-writer invariant; removing it
              would silently re-enable lost-write bugs the moment a
              second writer is introduced.
            * **Second ``async with``** — the actual save + notify.
        """
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
        """Drop all vector data from storage and reinitialize the index.

        This method will:
            1. Reset ``self._index`` to a fresh ``IndexFlatIP`` and clear
               ``self._id_to_meta``.
            2. Remove both on-disk files (``.index`` and ``.meta.json``)
               if they exist.
            3. Notify other processes via ``set_all_update_flags`` and
               reset the writer's own flag.

        Caller contract:
            ``drop`` is destructive and **not** serialized by this storage
            class. The caller must hold the pipeline ``busy`` reservation
            (the ``/documents/clear`` endpoint does this) before invoking
            it — running ``drop`` concurrently with an active document
            pipeline will tear down storage out from under the writer and
            silently lose data. See class docstring,
            *Non-pipeline write paths*.

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
