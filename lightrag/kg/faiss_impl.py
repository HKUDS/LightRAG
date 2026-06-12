import glob
import os
import time
import asyncio
from typing import Any, final
import json
import numpy as np
from dataclasses import dataclass

from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
from lightrag.utils import logger, compute_mdhash_id, validate_workspace
from lightrag.base import BaseVectorStorage
from lightrag.constants import DEFAULT_QUERY_PRIORITY

from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)

# You must manually install faiss-cpu or faiss-gpu before using FAISS vector db
import faiss  # type: ignore


@dataclass
class _PendingFaissDoc:
    """A buffered upsert waiting for deferred embedding and materialization.

    ``record`` holds ``__id__`` / ``__created_at__`` plus the ``meta_fields``
    (which always include ``content`` for the entity/relation/chunk vdbs), so
    the content needed for deferred embedding lives in the record itself — no
    separate copy is kept. ``vector`` starts as ``None`` and is filled either
    during the lock-held flush or by a lazy ``get_vectors_by_ids`` embedding;
    once set it is always an **already-L2-normalized float32 1D ndarray**, so
    the next flush can ``vstack`` and ``index.add`` without re-normalizing.
    ``__vector__`` is materialized into the metadata dict only at flush time,
    right before ``self._index.add``.
    """

    record: dict[str, Any]
    vector: np.ndarray | None = None


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
        different snapshots. ``_load_faiss_index`` tolerates both
        directions on load: ``meta > index`` rows are dropped silently;
        ``index > meta`` (the more dangerous case) is logged as a warning
        but **not** auto-repaired — orphan vectors remain in the loaded
        index but are unreachable via custom-id lookups. Repair semantics
        (truncate index vs rebuild meta) are deliberately left to a
        follow-up PR.

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
            * Pending-buffer mutations in ``upsert`` and pending-buffer
              reads in ``get_by_id`` / ``get_by_ids`` /
              ``get_vectors_by_ids`` (read-your-writes).
            * The single critical section in ``index_done_callback`` and
              ``finalize`` (reload → flush → save → notify).
            * The pending-cancel + rebuild critical sections in
              ``delete`` / ``delete_entity_relation``.
            * The entire ``drop`` body.
        The lock is **non-reentrant**, so ``_flush_pending_locked`` /
        ``_remove_faiss_ids_locked`` / ``_save_faiss_index`` /
        ``_reload_index_from_disk_locked`` all require the caller to
        already hold it and never re-enter via ``_get_index``. Routine
        ``index.search`` outside ``_get_index`` and the synchronous
        ``client_storage`` read rely on invariant (3) above — if either
        premise is broken (e.g. Faiss calls moved to a thread pool),
        the lock scope must be widened.

    Caveat — synchronous ``client_storage`` reads:
        ``client_storage`` is a synchronous property and does **not** go
        through ``_get_index``, so in a reader process it can return data
        older than the latest committed snapshot until some other method
        triggers a reload. The async read methods (``get_by_id`` /
        ``get_by_ids`` / ``get_vectors_by_ids``) now funnel through
        ``_get_index`` after checking the pending buffer, so they observe
        the latest on-disk snapshot.

    Deferred-embedding protocol:
        ``upsert`` does **not** call the embedding model. It only buffers
        a ``_PendingFaissDoc`` (content-bearing record + ``vector=None``)
        in the minimal ``self._pending_upserts`` area, overwriting any
        prior pending doc for the same id (which also clears a temp
        vector a previous ``get_vectors_by_ids`` may have cached). The
        model is called once per id at flush time
        (``_flush_pending_locked``), so repeated upserts of the same id —
        and many small upsert calls — embed only once. See issue #2785
        and the ``NanoVectorDBStorage`` / ``OpenSearchVectorDBStorage``
        equivalents.

        Embedding runs **inside ``_storage_lock``** during the flush (not
        in ``upsert``): under the single-writer invariant this keeps the
        content used for embedding consistent with the rows written to
        disk and prevents a destructive op from interleaving between
        embed and write. The lock is non-reentrant, so
        ``_flush_pending_locked`` requires the caller to already hold it
        and operates on ``self._index`` / ``self._id_to_meta`` directly
        (never through ``_get_index``).

        Vector storage invariant: once a ``_PendingFaissDoc.vector`` is
        set it is an **already-L2-normalized float32 1D ndarray** — both
        flush and lazy ``get_vectors_by_ids`` normalize the entire batch
        with ``faiss.normalize_L2`` before caching back, so a later flush
        can ``vstack`` and ``index.add`` without re-normalizing.

        Reads are read-your-writes: ``get_by_id`` / ``get_by_ids`` /
        ``get_vectors_by_ids`` consult ``_pending_upserts`` first, then
        funnel through ``_get_index`` for the materialized fallback.
        ``get_vectors_by_ids`` lazily embeds a pending doc on demand and
        caches the (normalized) vector back for the next flush.
        ``query`` and ``client_storage`` see only data already
        materialized into ``self._index`` / ``self._id_to_meta`` —
        unflushed pending data is intentionally not queryable.

        A flush failure (embedding error, count mismatch, or save IO
        error) raises through ``index_done_callback``; the pending buffer
        is preserved on flush failure, and if only the save failed
        ``_index_dirty`` stays ``True`` so a subsequent ``finalize``
        retries the save without re-embedding.

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
        # Reject path traversal before using workspace in a file path
        validate_workspace(self.workspace)
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

        # Minimal pending area for deferred embedding: custom-id -> _PendingFaissDoc.
        # Holds only records not yet embedded+materialized into self._index;
        # it never duplicates rows already added to the Faiss index. Flushed
        # under _storage_lock by _flush_pending_locked().
        self._pending_upserts: dict[str, _PendingFaissDoc] = {}
        # True when self._index / self._id_to_meta have materialized changes
        # that have not been successfully saved to disk yet. This lets
        # finalize retry a save even after a previous flush cleared the
        # pending buffer (see _flush_pending_locked / index_done_callback).
        self._index_dirty = False

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

    def _reload_index_from_disk_locked(self, *, for_write: bool = False) -> bool:
        """Reload ``self._index`` + ``self._id_to_meta`` if another process committed newer data.

        Precondition: the caller must already hold ``_storage_lock``. This is
        used by write paths as well as reads because deferred upserts mean a
        stale writer must merge its pending buffer into the latest on-disk
        snapshot, not save over it or return without flushing.

        Returns True if a reload happened, False if the local snapshot was
        already current.
        """
        if not self.storage_updated.value:
            return False

        log_message = (
            f"[{self.workspace}] Process {os.getpid()} FAISS reloading {self.namespace} "
            "due to update by another process"
        )
        if for_write:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        self._index = faiss.IndexFlatIP(self._dim)
        self._id_to_meta = {}
        self._load_faiss_index()
        self.storage_updated.value = False
        return True

    async def _get_index(self):
        """Return the live Faiss index, reloading from disk if needed.

        Read paths (``query`` / ``get_by_id`` / ``get_by_ids`` /
        ``get_vectors_by_ids``) funnel through this method so that a stale
        reader picks up any commit made by another process before reading
        ``self._index`` / ``self._id_to_meta``. Faiss has no incremental
        sync API — the reload is unconditionally a full reload of both
        files via ``_reload_index_from_disk_locked``.

        Under the *Single writer* invariant (see class docstring), the
        reload branch never fires in the writer process: the writer
        resets its own flag at the end of every ``index_done_callback``.
        The branch exists for readers.

        ``_storage_lock`` is held during the check-and-reload to (a)
        serialize concurrent reload attempts by sibling coroutines and
        (b) interlock with ``index_done_callback`` so a reader cannot
        observe a partially-saved file pair.
        """
        async with self._storage_lock:
            self._reload_index_from_disk_locked()
            return self._index

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vectors for deferred embedding; persistence is deferred too.

        ``data`` shape::

            {
                "custom_id_1": {"content": <text>, ...metadata...},
                "custom_id_2": {"content": <text>, ...metadata...},
                ...
            }

        Embedding is **not** performed here. Each record is buffered in
        ``self._pending_upserts`` with ``vector=None`` and the embedding
        model is called once per id at flush time (``_flush_pending_locked``
        during ``index_done_callback`` / ``finalize``). This coalesces
        repeated upserts of the same id and many small upsert calls into a
        single embedding pass (see class docstring,
        *Deferred-embedding protocol*, and issue #2785).

        Persistence:
            Changes live only in this process's memory until the next
            ``index_done_callback``. Cross-process readers will not see
            them until that commit fires (see class docstring,
            *Cross-process sync protocol*). Until the flush, an upserted
            id is observable only through the read-your-writes read paths,
            not through ``query``.
        """
        if not data:
            return

        current_time = int(time.time())
        pending = [
            (
                k,
                {
                    "__id__": k,
                    "__created_at__": current_time,
                    **{mf: v[mf] for mf in self.meta_fields if mf in v},
                },
            )
            for k, v in data.items()
        ]

        # Buffer under the lock to interlock with the lock-held flush. A new
        # _PendingFaissDoc(vector=None) overwrites any prior pending doc for
        # the same id, discarding a temp vector a previous get_vectors_by_ids
        # may have cached (content-version change -> must re-embed new content).
        async with self._storage_lock:
            for doc_id, record in pending:
                self._pending_upserts[doc_id] = _PendingFaissDoc(record=record)

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Similarity search over data already materialized into ``self._index``.

        Buffered (unflushed) upserts are intentionally **not** searchable —
        only rows that a prior ``index_done_callback`` / ``finalize``
        flushed are considered. Use the read-your-writes paths
        (``get_by_id`` / ``get_by_ids`` / ``get_vectors_by_ids``) to observe
        pending data before a flush.

        Returns top_k results with their metadata + similarity distance.
        """
        if query_embedding is not None:
            embedding = np.array([query_embedding], dtype=np.float32)
        else:
            embedding = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
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

            meta = self._id_to_meta.get(idx)
            if not meta:
                # Orphan vector: a row lives at this fid in self._index but
                # has no metadata in self._id_to_meta. This happens after an
                # index > meta skew on reload (see _load_faiss_index). The
                # vector is reachable via faiss search but not via custom id;
                # surfacing it as {"id": None, ...} would leak a ghost row to
                # callers, so we silently skip — the skew was already warned
                # about at load time.
                continue
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
        """Return a snapshot view of the materialized metadata dict for debugging.

        **Buffered (unflushed) upserts are intentionally not visible here**
        — only rows that a prior ``index_done_callback`` / ``finalize``
        flushed into ``self._id_to_meta`` are returned. Use the
        read-your-writes paths (``get_by_id`` / ``get_by_ids`` /
        ``get_vectors_by_ids``) to observe pending data before a flush.

        The outer list is a fresh shallow copy taken at access time, but
        each element is still a **live reference** into
        ``self._id_to_meta``; callers must not mutate them and must not
        retain them across operations that may rebuild the index
        (``upsert`` flush, ``delete``, ``_remove_faiss_ids_locked``,
        ``_get_index`` reload), since a rebuild swaps ``self._index`` and
        replaces ``self._id_to_meta`` with a new dict.

        This property is **synchronous and does not call** ``_get_index``,
        so in a reader process it can return data older than the latest
        committed snapshot until some other method triggers a reload.
        """
        return {"data": list(self._id_to_meta.values())}

    async def delete(self, ids: list[str]):
        """Delete vectors for the provided custom IDs.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. In ``lightrag.py`` this
            is handled by ``_insert_done()`` at the end of the document
            batch. Callers outside the pipeline must persist explicitly.

        Errors propagate to the caller — Faiss delete is destructive enough
        that document deletion / status updates must not proceed if the
        vectors were not actually removed. (This intentionally diverges
        from Nano, whose delete swallows + logs.)

        Args:
            ids: List of custom IDs to be deleted.
        """
        # Hold the lock so the pending-cancel and the rebuild are a single
        # critical section against a concurrent flush. Operate on
        # self._index / self._id_to_meta directly (the lock is
        # non-reentrant; no _get_index).
        async with self._storage_lock:
            self._reload_index_from_disk_locked(for_write=True)

            for doc_id in ids:
                self._pending_upserts.pop(doc_id, None)

            # Use the find-all variant so legacy/corrupt stores with
            # duplicate __id__ rows still get fully cleaned.
            to_remove: list[int] = []
            for cid in ids:
                to_remove.extend(self._find_faiss_ids_by_custom_id(cid))
            if to_remove:
                self._remove_faiss_ids_locked(to_remove)
                self._index_dirty = True

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

        Scans both ``self._pending_upserts`` (so buffered relation upserts
        get cancelled) and ``self._id_to_meta`` (the materialized rows) for
        entries whose ``src_id`` or ``tgt_id`` matches, then rebuilds the
        index without them.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Errors propagate (same rationale as ``delete``).

        Buffer semantics — post-prune with caller short-circuit contract:
            The materialized index rebuild runs first; matching pending
            upserts are pruned **only after** it succeeds. If the
            rebuild raises, the pending buffer stays intact so the
            caller (``adelete_by_entity`` in ``utils_graph.py``) can
            short-circuit before ``_persist_graph_updates`` flushes a
            half-cleaned buffer.

        **Not pipeline-gated** — see class docstring
        *Non-pipeline write paths*. The caller is responsible for
        ensuring single-writer serialization.
        """
        async with self._storage_lock:
            self._reload_index_from_disk_locked(for_write=True)

            # Materialized side first so a failure leaves the pending
            # buffer intact for the caller's retry path. .get() so rows
            # from foreign namespaces (no src_id / tgt_id) silently
            # don't match.
            relations = [
                fid
                for fid, meta in self._id_to_meta.items()
                if meta.get("src_id") == entity_name
                or meta.get("tgt_id") == entity_name
            ]
            if relations:
                self._remove_faiss_ids_locked(relations)
                self._index_dirty = True

            # Materialized rebuild succeeded — safe to prune matching
            # buffered upserts (their records carry src_id / tgt_id from
            # the relationships vdb meta_fields).
            pending_ids = [
                doc_id
                for doc_id, pdoc in self._pending_upserts.items()
                if pdoc.record.get("src_id") == entity_name
                or pdoc.record.get("tgt_id") == entity_name
            ]
            for doc_id in pending_ids:
                del self._pending_upserts[doc_id]

        total = len(pending_ids) + len(relations)
        if total:
            logger.debug(
                f"[{self.workspace}] Deleted {total} relations for {entity_name}"
            )
        else:
            logger.debug(
                f"[{self.workspace}] No relations found for entity {entity_name}"
            )

    # --------------------------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------------------------

    def _find_faiss_id_by_custom_id(self, custom_id: str):
        """Return the first Faiss internal ID matching ``custom_id``, or ``None``.

        Adequate for read paths (any of N duplicate rows would carry the same
        ``__id__`` so returning one is fine semantically). Write paths that
        need to remove **all** duplicates — flush overwrite, ``delete`` —
        must use :py:meth:`_find_faiss_ids_by_custom_id` (plural) instead.
        """
        for fid, meta in self._id_to_meta.items():
            if meta.get("__id__") == custom_id:
                return fid
        return None

    def _find_faiss_ids_by_custom_id(self, custom_id: str) -> list[int]:
        """Return **every** Faiss internal ID whose metadata's ``__id__`` matches.

        In a healthy store every custom id maps to at most one fid (each flush
        rebuilds the index without the prior fid before adding the new one).
        This plural variant exists to defend against legacy / externally
        corrupted stores where multiple fids share a ``__id__`` — a re-upsert
        or ``delete`` using only the first match would leave stale duplicates
        behind. Used by ``_flush_pending_locked`` and ``delete``.
        """
        return [
            fid
            for fid, meta in self._id_to_meta.items()
            if meta.get("__id__") == custom_id
        ]

    def _remove_faiss_ids_locked(self, fid_list) -> None:
        """Remove a list of internal Faiss IDs by rebuilding the index.

        Precondition: the caller must already hold ``_storage_lock``. This
        is synchronous (no ``await``) because every step — dict scan,
        ``IndexFlatIP`` re-init, ``index.add`` — is synchronous, and the
        single critical section guarantees ``self._index`` and
        ``self._id_to_meta`` flip together. Because ``IndexFlatIP`` has no
        in-place removal API, we collect the kept vectors and rebuild.

        Callers that mutate via this helper are responsible for setting
        ``self._index_dirty = True`` themselves (skipped here so a no-op
        call — empty intersection between ``fid_list`` and current ids —
        does not falsely mark the storage dirty).
        """
        if not fid_list:
            return

        fid_set = set(fid_list)
        keep_fids = [fid for fid in self._id_to_meta if fid not in fid_set]

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

        self._index = faiss.IndexFlatIP(self._dim)
        if vectors_to_keep:
            arr = np.array(vectors_to_keep, dtype=np.float32)
            self._index.add(arr)
        self._id_to_meta = new_id_to_meta

    async def _flush_pending_locked(self) -> None:
        """Embed pending docs and materialize them into ``self._index`` + ``self._id_to_meta``.

        Precondition: the caller **must already hold** ``_storage_lock``. The
        lock is non-reentrant, so this helper never calls ``_get_index`` and
        operates on ``self._index`` / ``self._id_to_meta`` directly. Embedding
        runs inside the lock on purpose (see class docstring,
        *Deferred-embedding protocol*).

        Invariant: once ``_PendingFaissDoc.vector`` is set it is an **already
        L2-normalized float32 1D ndarray**. The flush honours this — vectors
        cached by a prior ``get_vectors_by_ids`` are not re-normalized; only
        newly embedded vectors go through ``faiss.normalize_L2``.

        Failure handling:
            * Embedding error / count mismatch → raises before any mutation
              to ``self._index`` / ``self._id_to_meta``; ``_pending_upserts``
              is left intact and ``self._index_dirty`` is not touched.
            * Rebuild / ``index.add`` failure → raises mid-write. The
              materialized state may already be partially mutated (e.g.
              ``_remove_faiss_ids_locked`` ran and dropped the prior fids
              for re-upserted ids), but ``_index_dirty`` is **not** set
              because we deliberately treat ``_pending_upserts`` as the
              source of truth on this path: pending stays intact, and the
              next ``finalize`` call re-enters ``_flush_pending_locked``,
              which will rebuild the affected rows from the cached vectors
              and re-add them — self-healing without re-embedding. The
              dirty flag is reserved for "materialized but unsaved",
              which is only true after ``index.add`` completes.
        """
        if not self._pending_upserts:
            return

        # Snapshot for stable ordering between the embed list and the write.
        pending_items = list(self._pending_upserts.items())
        to_embed = [
            (doc_id, pdoc) for doc_id, pdoc in pending_items if pdoc.vector is None
        ]

        if to_embed:
            contents = [pdoc.record["content"] for _, pdoc in to_embed]
            batches = [
                contents[i : i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]
            logger.info(
                f"[{self.workspace}] {self.namespace} flush: embedding "
                f"{len(to_embed)} vectors in {len(batches)} batch(es) "
                f"(batch_num={self._max_batch_size})"
            )
            try:
                embeddings_list = await asyncio.gather(
                    *[
                        self.embedding_func(batch, context="document")
                        for batch in batches
                    ]
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error embedding pending vector ops "
                    f"(upserts={len(to_embed)}): {e}"
                )
                raise
            arr = np.concatenate(embeddings_list, axis=0).astype(np.float32)
            if len(arr) != len(to_embed):
                # Explicit raise (not a log): a mismatch would mis-pair vectors
                # with records. Keep pending intact so the next flush retries.
                raise RuntimeError(
                    f"[{self.workspace}] embedding is not 1-1 with pending data, "
                    f"{len(arr)} != {len(to_embed)}"
                )
            # Batch in-place normalize once (faiss.normalize_L2 requires 2D).
            faiss.normalize_L2(arr)
            for i, (_, pdoc) in enumerate(to_embed):
                pdoc.vector = arr[i].copy()

        # All pending vectors are now non-None and already-normalized float32.
        # Remove every existing fid in self._id_to_meta whose custom id is
        # being re-upserted (find-all so duplicate __id__ rows from a legacy /
        # corrupt store still get fully cleaned), then add the new vectors in
        # a single batch.
        existing_fids: list[int] = []
        for doc_id, _ in pending_items:
            existing_fids.extend(self._find_faiss_ids_by_custom_id(doc_id))
        self._remove_faiss_ids_locked(existing_fids)

        matrix = np.vstack([pdoc.vector for _, pdoc in pending_items]).astype(
            np.float32
        )
        start_idx = self._index.ntotal
        self._index.add(matrix)
        for i, (_, pdoc) in enumerate(pending_items):
            fid = start_idx + i
            record = pdoc.record
            record["__vector__"] = matrix[i].tolist()
            self._id_to_meta[fid] = record

        self._index_dirty = True

        # Clear only the entries we just flushed. Today the non-reentrant
        # _storage_lock locks out concurrent upserts for the entire flush
        # (including the asyncio.gather await), so the `is pdoc` identity
        # check is always True — it's kept as defensive scaffolding so that
        # if the lock scope is ever relaxed (e.g. embedding moved outside the
        # lock), a concurrent upsert that re-set vector=None would not be
        # silently dropped here.
        for doc_id, pdoc in pending_items:
            if self._pending_upserts.get(doc_id) is pdoc:
                del self._pending_upserts[doc_id]

    def _save_faiss_index(self):
        """Atomically persist ``self._index`` + ``self._id_to_meta`` to disk.

        Precondition: the caller must already hold ``_storage_lock`` (this is
        the symmetric counterpart of ``_flush_pending_locked`` — see Nano's
        ``_save_to_disk_locked``).

        Each file lands via a per-writer tmp + os.replace so a crash mid-write
        leaves the prior snapshot intact. **Cross-file consistency between
        the .index and .meta.json is not guaranteed**: the two renames are
        independent, so a crash between them can produce
        ``ntotal(.index) > rows(.meta)`` skew. ``_load_faiss_index`` tolerates
        skew on load by skipping unbacked rows and logs a warning if the
        index has more vectors than the meta describes. The
        ``index < meta`` direction is covered by
        ``test_faiss_meta_inconsistency``; the ``index > meta`` direction is
        a known gap (logged on reload, not auto-repaired) — see class
        docstring *Storage model*.
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

            # Cross-file skew detection (index > meta direction): a crash
            # between the two atomic_writes in _save_faiss_index can leave
            # the index with more vectors than the meta describes. We log
            # but do not auto-repair — repair semantics (truncate index vs
            # rebuild meta) are out of scope here. See class docstring.
            if self._index.ntotal > len(self._id_to_meta):
                logger.warning(
                    f"[{self.workspace}] FAISS index has {self._index.ntotal} vectors "
                    f"but only {len(self._id_to_meta)} metadata rows — index > meta "
                    f"skew from a prior crash between the .index and .meta.json "
                    f"writes. Not auto-repairing; orphan vectors remain in the index "
                    f"but unreachable via custom-id lookups."
                )

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

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts on an aborting batch.

        Only the pending buffer is dropped; vectors already materialized into
        ``self._index`` by a prior ``_flush_pending_locked`` whose save step
        then failed (``_index_dirty=True``) are intentionally NOT rolled back.

        The pipeline treats each file as an atomic unit: an abort marks the
        affected documents FAILED and the whole file is reprocessed on the
        next run. Because upserts are keyed by deterministic ids (entity-name
        / relation / chunk hashes), reprocessing overwrites those vectors
        idempotently, so the final state is identical whether or not we roll
        back here. This matches the server-backed backends (Milvus / OpenSearch
        / Postgres / Mongo / Qdrant), which likewise keep a sibling flush's
        already-committed partial data on abort rather than rolling it back;
        and if the process crashes before the next save, these in-memory
        writes are dropped anyway. Rolling back only FAISS/Nano would add an
        inconsistent, non-load-bearing "FAILED == clean" guarantee, so it is
        deliberately omitted.
        """
        if self._storage_lock is None:
            self._pending_upserts.clear()
            return
        async with self._storage_lock:
            self._pending_upserts.clear()

    async def index_done_callback(self) -> bool:
        """Flush deferred embeddings, commit to disk, and notify other processes.

        This is the writer's **commit point** in the cross-process sync
        protocol (see class docstring). Effects, in order:
            1. If another process committed first, reload the latest on-disk
               snapshot while preserving this process's pending buffer.
            2. ``_flush_pending_locked`` embeds every buffered upsert (once
               per id) and materializes it into ``self._index`` +
               ``self._id_to_meta``. A failure here **raises** — pending is
               kept, ``_index_dirty`` is not touched, nothing is written to
               the index.
            3. ``_save_faiss_index`` atomically writes ``.index`` and
               ``.meta.json``. A failure here **also raises**;
               ``_pending_upserts`` is already empty (flush succeeded) and
               ``_index_dirty`` stays ``True`` so a later ``finalize``
               retries the save without re-embedding.
            4. ``set_all_update_flags`` flips every registered process's
               ``storage_updated`` flag, then we immediately reset our own
               flag to ``False`` so the writer does not self-reload on the
               next call to ``_get_index``.

        Either failure surfaces loudly through ``_insert_done`` so the
        caller can abort the document batch instead of silently losing
        vectors. The bool return is kept for legacy callers but is
        effectively always ``True`` on the success path.
        """
        async with self._storage_lock:
            self._reload_index_from_disk_locked(for_write=True)

            # Flush + save both raise on failure (embedding mismatch / save IO
            # error). The exception propagates out of the lock so _insert_done
            # aborts the batch; pending stays intact and _index_dirty stays
            # True (if only the save failed) for a later retry.
            await self._flush_pending_locked()
            self._save_faiss_index()
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            self._index_dirty = False
            return True

    @staticmethod
    def _format_record(dp: dict[str, Any]) -> dict[str, Any]:
        """Shape a stored/pending record into the public read result."""
        return {
            **{k: v for k, v in dp.items() if k != "__vector__"},
            "id": dp.get("__id__"),
            "created_at": dp.get("__created_at__"),
        }

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID (read-your-writes against the pending buffer).

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        # Read-your-writes: a buffered upsert is visible before its flush.
        async with self._storage_lock:
            pending = self._pending_upserts.get(id)
            if pending is not None:
                return self._format_record(pending.record)

        await self._get_index()  # reload-if-stale
        fid = self._find_faiss_id_by_custom_id(id)
        if fid is None:
            return None
        metadata = self._id_to_meta.get(fid)
        return self._format_record(metadata) if metadata else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs (read-your-writes), preserving order.

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects (or ``None`` placeholders) in the
            same order as ``ids``.
        """
        if not ids:
            return []

        # Read-your-writes: serve buffered upserts from the pending area and
        # only query the materialized index for the remaining ids.
        result_map: dict[str, dict[str, Any]] = {}
        remaining: list[str] = []
        async with self._storage_lock:
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is not None:
                    result_map[str(requested_id)] = self._format_record(pending.record)
                else:
                    remaining.append(requested_id)

        if remaining:
            await self._get_index()  # reload-if-stale
            for cid in remaining:
                fid = self._find_faiss_id_by_custom_id(cid)
                if fid is None:
                    continue
                metadata = self._id_to_meta.get(fid)
                if metadata:
                    result_map[str(cid)] = self._format_record(metadata)

        return [result_map.get(str(requested_id)) for requested_id in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs (read-your-writes), returning only ID and vector.

        For buffered upserts the vector is computed lazily (and cached back
        onto the pending doc so the next flush reuses it instead of
        re-embedding); for materialized rows the stored normalized vector is
        returned directly.

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        vectors_dict: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._storage_lock:
            to_embed: list[tuple[str, _PendingFaissDoc]] = []
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is None:
                    remaining.append(requested_id)
                elif pending.vector is not None:
                    vectors_dict[requested_id] = pending.vector.astype(
                        np.float32
                    ).tolist()
                else:
                    to_embed.append((requested_id, pending))

            if to_embed:
                contents = [pdoc.record["content"] for _, pdoc in to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                embeddings_list = await asyncio.gather(
                    *[
                        self.embedding_func(batch, context="document")
                        for batch in batches
                    ]
                )
                arr = np.concatenate(embeddings_list, axis=0).astype(np.float32)
                if len(arr) != len(to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] embedding is not 1-1 with pending data, "
                        f"{len(arr)} != {len(to_embed)}"
                    )
                # Batch normalize once; shared invariant with _flush_pending_locked.
                faiss.normalize_L2(arr)
                for i, (requested_id, pdoc) in enumerate(to_embed):
                    # Cache the normalized vector back so the next flush reuses it.
                    pdoc.vector = arr[i].copy()
                    vectors_dict[requested_id] = arr[i].tolist()

        if remaining:
            await self._get_index()  # reload-if-stale
            for cid in remaining:
                fid = self._find_faiss_id_by_custom_id(cid)
                if fid is None or fid not in self._id_to_meta:
                    continue
                metadata = self._id_to_meta[fid]
                if "__vector__" in metadata:
                    vectors_dict[cid] = metadata["__vector__"]

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
                # Discard buffered (unflushed) upserts along with the data.
                self._pending_upserts.clear()

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
                self._index_dirty = False

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

    async def finalize(self):
        """Flush any buffered upserts and persist before shutdown (safety net).

        Normally ``index_done_callback`` has already drained the pending
        buffer and synced to disk, but two paths land here with work to do:

        - **Pending upserts only** (no prior ``index_done_callback``): flush
          and save. We reload first so a stale process picks up other
          writers' commits before merging its pending buffer in.
        - **Unsaved materialized changes** (``_index_dirty=True``): an
          earlier ``index_done_callback`` flushed pending into ``self._index``
          but its save raised. Skip the reload — reloading would drop those
          materialized-but-unsaved rows — and just retry the save.

        Flush / save failures propagate (same contract as
        ``index_done_callback``); a partially flushed buffer is preserved
        for a future retry.
        """
        async with self._storage_lock:
            if not self._pending_upserts and not self._index_dirty:
                return
            if self._pending_upserts:
                # Only reload when we have nothing un-persisted in self._index.
                # A dirty index carries successfully-flushed-but-unsaved rows
                # from a prior index_done_callback; reloading would silently
                # drop them.
                if not self._index_dirty:
                    self._reload_index_from_disk_locked(for_write=True)
                await self._flush_pending_locked()
            self._save_faiss_index()
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            self._index_dirty = False
