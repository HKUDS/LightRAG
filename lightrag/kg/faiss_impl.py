import glob
import os
import time
import asyncio
from hashlib import md5
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
from .write_seq import WRITE_SEQ_FIELD, next_write_seq, row_is_strictly_newer

# You must manually install faiss-cpu or faiss-gpu before using FAISS vector db
import faiss  # type: ignore


@dataclass
class _PendingFaissDoc:
    """A buffered upsert waiting for deferred embedding and materialization.

    ``record`` holds ``__id__`` / ``__created_at__`` / ``__write_seq__`` plus
    the ``meta_fields`` (which always include ``content`` for the
    entity/relation/chunk vdbs), so the content needed for deferred embedding
    lives in the record itself — no separate copy is kept. ``vector`` starts as
    ``None`` and is filled either during the lock-held flush or by a lazy
    ``get_vectors_by_ids`` embedding; once set it is always an
    **already-L2-normalized float32 1D ndarray**, so the next flush can
    ``vstack`` and ``index.add`` without re-normalizing.
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
        is preserved on flush failure. If only the save failed, the
        flushed docs have moved into the ``_unsaved_upserts`` redo log
        (record + cached normalized vector) and ``_index_dirty`` stays
        ``True``: a later commit or ``finalize`` reloads whatever another
        writer committed meanwhile and replays the logged rows on top —
        without re-embedding (issue #3688). A logged row is *not* replayed
        over a row the other writer committed strictly later — by the
        ``__write_seq__`` token, or by whole-second ``__created_at__`` when a
        row predates it (see ``write_seq``): that writer legitimately
        superseded us (a reprocess of the same document under the same
        content-hash id), and the redo entry is dropped instead. The read
        paths apply the same ordering rule over the same find-all row set
        (``_resolve_resident_rows``), so read-your-writes never reports a row
        the replay is about to decline to restore — not even in a corrupt
        store where one id carries several rows. The log is cleared only once
        a save lands.

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
        # Custom ids queued for removal, applied in one batched index rebuild
        # by _flush_pending_locked (see *Deferred-delete protocol* in delete's
        # docstring). An eager delete rebuilt the whole IndexFlatIP per call
        # and the merge stage deletes once per relation (#3681).
        self._pending_deletes: set[str] = set()
        # Redo log: removals already applied to self._index / self._id_to_meta
        # but not yet on disk, kept as custom id -> {_row_fingerprint(removed
        # row), ...}. If a save fails and a reload replaces the in-memory state
        # with the on-disk snapshot, the removed rows come back; replaying this
        # log after the reload removes them again. The fingerprint (not the
        # bare id) ensures a replay only removes the row versions it removed
        # before, never a newer row another writer published under the same
        # content-hash id. A *set* per id because a legacy / corrupt store can
        # hold several rows under one id (see _find_faiss_ids_by_custom_id):
        # the delete removes all of them, so the replay has to name all of
        # them. Cleared once a save lands. See delete's docstring.
        self._unsaved_deletes: dict[str, set[str]] = {}
        # True when self._index / self._id_to_meta have materialized changes
        # that have not been successfully saved to disk yet. This lets
        # finalize retry a save even after a previous flush drained the
        # pending buffer (see _flush_pending_locked / index_done_callback).
        self._index_dirty = False
        # Rows materialized into self._index whose save has not landed yet,
        # kept as id -> the flushed _PendingFaissDoc (record + cached
        # normalized vector). The upsert mirror of _unsaved_deletes: a reload
        # after a failed save rebuilds the in-memory state from the on-disk
        # snapshot, and the next flush replays these rows on top without
        # re-embedding (issue #3688). Cleared only once a save lands.
        self._unsaved_upserts: dict[str, _PendingFaissDoc] = {}

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

        # One timestamp and one write-sequence token for the whole batch: both
        # order *writes*, and the redo-log comparison only ever puts rows from
        # two different writes side by side. The token breaks the whole-second
        # ties __created_at__ leaves open (see write_seq).
        current_time = int(time.time())
        write_seq = next_write_seq()
        pending = [
            (
                k,
                {
                    "__id__": k,
                    "__created_at__": current_time,
                    WRITE_SEQ_FIELD: write_seq,
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
                # A fresh upsert supersedes a queued delete for the same id:
                # the flush materializes the new row after applying deletes,
                # so applying the delete would be redundant work. Mirrors the
                # Nano / PG / Qdrant buffers (#3681).
                self._pending_deletes.discard(doc_id)
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
            # Filter out __vector__ from query results to avoid returning large
            # vector data; __write_seq__ is storage bookkeeping, not payload.
            filtered_meta = {
                k: v
                for k, v in meta.items()
                if k not in ("__vector__", WRITE_SEQ_FIELD)
            }
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

        Deletes are **deferred**: each id is queued in
        ``self._pending_deletes`` (cancelling any pending upsert for it) and
        every queued id is applied in one batched index rebuild by
        ``_flush_pending_locked``. The entity/relation merge stage deletes the
        stale forward/reverse rows once per relation, and an eager delete
        rebuilt the whole ``IndexFlatIP`` per call — deferring turns that into
        one rebuild per flush (#3681; the same change #3680 made on Nano, and
        the contract the Qdrant / PostgreSQL / Milvus / MongoDB / OpenSearch
        buffers already document).

        Deferring also keeps a delete from being lost across writers: the
        removal used to be applied to ``self._index`` while still unsaved,
        and because ``index_done_callback`` reloads from disk unconditionally
        when another process has committed — a reload that *replaces*
        ``self._index`` / ``self._id_to_meta`` — the row could silently
        reappear. A queued id is applied *after* that reload, so it survives.

        Two buffers, because the flush and the save can fail independently
        (the same protocol Nano documents, #3680):

            * ``_pending_deletes`` — queued, not applied to the index yet.
            * ``_unsaved_deletes`` — applied to the index but not yet on
              disk, kept as ``id -> {fingerprints of the removed rows}``.
              This is a redo log, not a pending buffer.

        The redo log exists because a removal that reached ``self._index``
        can still be undone: if the save fails, the next
        ``index_done_callback``'s unconditional reload replaces the in-memory
        state with the on-disk snapshot and the row returns. Replaying the
        log after that reload removes it again, so the reload stays lossless
        for deletes. (Upserts carry the mirror log: the flush moves its docs
        from ``_pending_upserts`` into the ``_unsaved_upserts`` redo log
        rather than dropping them, so a materialized-but-unsaved upsert is
        replayed after the same reload; see the deferred-embedding protocol
        above and issue #3688. A removal request evicts the id's redo entry
        — ``delete`` / ``delete_entity_relation`` — or the replay would
        resurrect the row the removal just took out.)

        A replay matches on the row, not on the id alone. Ids are content
        hashes, so another writer can publish a *new* row under an id we
        removed, and deleting by id would destroy it. The log therefore
        stores the ``_row_fingerprint`` of every row it removed — a set per
        id, because a legacy / corrupt store can hold several rows under one
        id and ``delete`` removes all of them (see
        ``_find_faiss_ids_by_custom_id``) — and a replay removes only rows
        that still match one of them. The two buffers are scoped
        differently on purpose: ``_pending_deletes`` holds a *request* — the
        flush removes whatever row carries that id, the by-id contract purge
        relies on — while ``_unsaved_deletes`` holds a *record* of a removal
        that already happened, so replaying it must not remove a row that
        has since taken the id's place.

        **Boundary of that guarantee.** Successor preservation is only as
        sharp as content-based identity, and the case it has to resolve is a
        successor *identical* in content to the row removed.
        ``__created_at__`` does not break that tie — ``upsert`` stamps
        ``int(time.time())``, so a rewrite inside the same second carries the
        same timestamp — but ``__write_seq__`` does: every ``upsert`` stamps
        its own token into the record (see ``write_seq``), so a successor
        written by another writer, or by us, fingerprints differently and the
        replay preserves it. What remains is a row written before the token
        existed, which no content-based identity can tell from the row we
        deleted, so the replay removes it — a case that already presupposes
        the *Single writer* invariant above being violated (see class
        docstring, *Concurrency invariants*).

        The read-your-writes paths mirror both rules: a queued id reads as
        absent, while a logged one hides only the row the entry names — a
        replacement the replay would preserve stays readable. An ``upsert``
        cancels a *queued* delete for the same id but never touches the redo
        log: the buffered row may be discarded by an aborting batch before
        it materializes, and dropping the entry then would leave the reload
        nothing to replay. Ids that matched no row are not logged at all.
        The log is cleared once a save lands; an aborting batch keeps it
        (``drop_pending_index_ops`` discards buffered work, not removals
        that already reached the index). ``delete_entity_relation`` stays
        eager — it is off the merge hot path — but its removals are
        applied-and-unsaved just the same, so they are recorded in the log
        too. Both buffers are in-memory only: they are dropped by ``drop``
        and lost on a crash before the flush.

        Persistence:
            Queued ids are in-memory only: they land on the index at the next
            ``index_done_callback`` / ``finalize`` flush and are lost on a
            crash before it. Cross-process visibility requires the flush.

        Errors propagate to the caller at flush time — Faiss delete is
        destructive enough that document deletion / status updates must not
        proceed if the vectors were not actually removed. (This intentionally
        diverges from Nano, whose delete swallows + logs.)

        Args:
            ids: List of custom IDs to be deleted.
        """
        if not ids:
            return

        async with self._storage_lock:
            for doc_id in ids:
                # Cancel a buffered upsert for the same id; the row is going
                # away, so embedding it at flush time would be wasted work.
                self._pending_upserts.pop(doc_id, None)
                # Evict the upsert redo entry too: this request removes
                # whatever row the id has, and a surviving entry would
                # replay — resurrect — that row at the next flush.
                self._unsaved_upserts.pop(doc_id, None)
                self._pending_deletes.add(doc_id)

        logger.debug(
            f"[{self.workspace}] {self.namespace}: queued {len(ids)} vector deletes "
            f"({len(self._pending_deletes)} pending)"
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
                # Applied-but-unsaved like a flushed delete: record the
                # removals in the redo log (queued requests for the same
                # ids are now redundant), or a failed save plus a reload
                # resurrects the rows.
                removed: dict[str, set[str]] = {}
                for fid in relations:
                    meta = self._id_to_meta[fid]
                    doc_id = meta.get("__id__")
                    if doc_id is not None:
                        # Accumulate: a corrupt store can carry several rows
                        # under one id and all of them are being removed.
                        removed.setdefault(doc_id, set()).add(
                            self._row_fingerprint(meta)
                        )
                self._remove_faiss_ids_locked(relations)
                self._pending_deletes.difference_update(removed)
                for doc_id, fingerprints in removed.items():
                    self._unsaved_deletes.setdefault(doc_id, set()).update(fingerprints)
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
            # Same predicate over the upsert redo log, and unconditional for
            # the same reason as the pending prune: after a foreign reload
            # the unsaved relation rows are not in the scan above, yet
            # surviving entries would replay them at the next flush.
            redo_ids = [
                doc_id
                for doc_id, pdoc in self._unsaved_upserts.items()
                if pdoc.record.get("src_id") == entity_name
                or pdoc.record.get("tgt_id") == entity_name
            ]
            for doc_id in redo_ids:
                del self._unsaved_upserts[doc_id]

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

    def _resolve_resident_rows(
        self,
        custom_ids: set[str],
        logged: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Resolve the materialized row each requested id reads as.

        Find-all rather than first-match, in **one** scan of
        ``self._id_to_meta`` for the whole requested set. A legacy /
        corrupted store can hold several rows under one id (see
        ``_find_faiss_ids_by_custom_id``), and the read paths do not just
        pick a row any more — they *compare* it, against an upsert-redo
        entry's record and against the fingerprints an unsaved delete
        removed. Answering from an arbitrary duplicate would disagree with
        the next flush, which applies both rules across **every** fid the id
        maps to: it declines a replay when *any* resident row is newer, and
        removes exactly the row versions a delete entry names.

        So a row an unsaved delete removed is skipped (``logged``, per id —
        the replay will take it out, and a row matching none of the
        fingerprints took the id's place and must stay visible), and of the
        rest the newest wins. In a healthy store this is the id's only row.
        """
        if not custom_ids:
            return {}
        resolved: dict[str, dict[str, Any]] = {}
        for meta in self._id_to_meta.values():
            doc_id = meta.get("__id__")
            if doc_id not in custom_ids:
                continue
            if self._matches_redo_entry(
                meta, logged.get(doc_id) if logged is not None else None
            ):
                continue
            current = resolved.get(doc_id)
            if current is None or row_is_strictly_newer(meta, current):
                resolved[doc_id] = meta
        return resolved

    def _resolve_resident_row(
        self, custom_id: str, logged: set[str] | None = None
    ) -> dict[str, Any] | None:
        """Single-id form of :py:meth:`_resolve_resident_rows`."""
        return self._resolve_resident_rows(
            {custom_id}, {custom_id: logged} if logged is not None else None
        ).get(custom_id)

    def _find_faiss_ids_by_custom_id(self, custom_id: str) -> list[int]:
        """Return **every** Faiss internal ID whose metadata's ``__id__`` matches.

        In a healthy store every custom id maps to at most one fid (each flush
        rebuilds the index without the prior fid before adding the new one).
        This variant exists to defend against legacy / externally corrupted
        stores where multiple fids share a ``__id__`` — a re-upsert or
        ``delete`` touching only the first match would leave stale duplicates
        behind, and a read *comparing* only the first would disagree with what
        the flush then does. Used by ``_flush_pending_locked``, ``delete``, and
        the read paths' ``_resolve_resident_rows``; there is deliberately no
        first-match variant left.
        """
        return [
            fid
            for fid, meta in self._id_to_meta.items()
            if meta.get("__id__") == custom_id
        ]

    def _rebuild_index_locked(
        self, drop_fids, add_records: list[dict[str, Any]] = ()
    ) -> None:
        """Rebuild the index without ``drop_fids`` and with ``add_records``.

        Precondition: the caller must already hold ``_storage_lock``. This
        is synchronous (no ``await``) because every step — dict scan,
        ``IndexFlatIP`` re-init, ``index.add`` — is synchronous, and the
        single critical section guarantees ``self._index`` and
        ``self._id_to_meta`` flip together. Because ``IndexFlatIP`` has no
        in-place update API, a change to any existing row means rebuilding.

        Both halves in one call, deliberately. An overwrite is "drop the row
        the id had, write the row it has now", and doing that as two index
        operations means a failure between them lands one half without the
        other: remove-then-add can leave the id with no row at all, and
        add-then-remove can leave it with two rows of different vintages,
        where the read paths return the stale one. Neither state is
        reachable through ``NanoVectorDBStorage``, whose client updates an
        existing id in place, and neither should be reachable here.

        So the replacement is assembled locally and swapped in only once its
        ``add`` has succeeded: either the whole change lands or nothing does,
        leaving both structures exactly as they were for the caller's retry
        or reload. ``add_records`` must already carry ``__vector__``.

        Callers that mutate via this helper are responsible for setting
        ``self._index_dirty = True`` themselves (skipped here so a no-op
        call does not falsely mark the storage dirty).
        """
        if not drop_fids and not add_records:
            return

        drop_set = set(drop_fids)
        vectors: list[list[float]] = []
        new_id_to_meta: dict[int, dict[str, Any]] = {}

        for old_fid, vec_meta in self._id_to_meta.items():
            if old_fid in drop_set:
                continue
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
            new_id_to_meta[len(vectors)] = vec_meta
            vectors.append(vec)

        # Appended after the survivors, which is where a plain index.add
        # would have put them — so fid ordering is unchanged.
        for record in add_records:
            new_id_to_meta[len(vectors)] = record
            vectors.append(record["__vector__"])

        new_index = faiss.IndexFlatIP(self._dim)
        if vectors:
            new_index.add(np.array(vectors, dtype=np.float32))
        self._index = new_index
        self._id_to_meta = new_id_to_meta

    def _remove_faiss_ids_locked(self, fid_list) -> None:
        """Remove a list of internal Faiss IDs by rebuilding the index.

        Thin wrapper over :py:meth:`_rebuild_index_locked` for the delete
        paths, which drop rows without writing any. Same preconditions and
        the same all-or-nothing guarantee.
        """
        if not fid_list:
            return
        self._rebuild_index_locked(fid_list)

    @staticmethod
    def _row_fingerprint(meta: dict[str, Any]) -> str:
        """Identify one *version* of a stored row.

        The redo log has to name the exact row it removed so a replay after
        a reload cannot hit a newer row another writer published under the
        same (content-hash) id. The digest covers the stored metadata record
        — id, timestamp, ``__write_seq__``, every meta field — **excluding**
        ``__vector__``:
        ``_save_faiss_index`` strips the vector from ``.meta.json`` and
        ``_load_faiss_index`` / ``_remove_faiss_ids_locked`` re-attach it
        lazily (``index.reconstruct``), so its presence on a row is not
        stable across a save/reload round-trip and including it would make
        the same row stop matching its own log entry. Everything else
        round-trips through ``json.dump``/``json.load`` unchanged; sorted
        keys make key order irrelevant and ``default=str`` keeps the dump
        total. The vector adds no identity anyway — it is a deterministic
        function of the content already covered by the digest.
        """
        canonical = json.dumps(
            {k: v for k, v in meta.items() if k != "__vector__"},
            sort_keys=True,
            default=str,
        )
        return md5(canonical.encode("utf-8")).hexdigest()

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
            * Queued deletes are applied first; one that lands on
              ``self._index`` moves into the ``_unsaved_deletes`` redo log
              (replayed after any reload until a save persists it), while
              one whose rebuild raises stays queued for the retry.
            * Embedding error / count mismatch → raises before any mutation
              to ``self._index`` / ``self._id_to_meta``; ``_pending_upserts``
              is left intact and ``self._index_dirty`` is not touched.
            * Materialization failure (``index.add``, or the rebuild an
              overwrite goes through) → raises with ``self._index`` and
              ``self._id_to_meta`` exactly as they were: the upsert phase
              mutates them through a single all-or-nothing operation, so
              there is no half-applied overwrite to persist or repair.
              ``_index_dirty`` is set and the flushed docs move into the
              ``_unsaved_upserts`` redo log only after it succeeds, and
              ``_pending_upserts`` stays intact, so the next ``finalize``
              re-enters here and re-applies from the cached vectors —
              self-healing without re-embedding.
        """
        if (
            not self._pending_upserts
            and not self._pending_deletes
            and not self._unsaved_deletes
            and not self._unsaved_upserts
        ):
            return

        # Apply every deferred delete in ONE pass + ONE index rebuild —
        # before materializing upserts, so an id that was deleted and
        # re-upserted in the same batch ends up with exactly the new row
        # (upsert also cancels the queued delete, but a delete queued after
        # the upsert must still win over the *materialized* stale row).
        # One scan of ``_id_to_meta`` for the whole queued set instead of
        # one full scan per deleted id (#3681). A queued id removes whatever
        # row carries it; a replayed one (redo log, after a reload undid an
        # unsaved removal) only removes the row version it removed before,
        # so anything written under that id since — by another writer or by
        # us — survives.
        if self._pending_deletes or self._unsaved_deletes:
            queued = len(self._pending_deletes)
            matched: dict[str, set[str]] = {}
            to_remove: list[int] = []
            for fid, meta in self._id_to_meta.items():
                doc_id = meta.get("__id__")
                if doc_id in self._pending_deletes:
                    # Find-all: every row carrying the id goes, so every
                    # removed version has to be logged — a single fingerprint
                    # per id would let the unlogged duplicates of a corrupt
                    # store survive the replay.
                    to_remove.append(fid)
                    matched.setdefault(doc_id, set()).add(self._row_fingerprint(meta))
                elif doc_id in self._unsaved_deletes:
                    fingerprint = self._row_fingerprint(meta)
                    if fingerprint in self._unsaved_deletes[doc_id]:
                        to_remove.append(fid)
                        matched.setdefault(doc_id, set()).add(fingerprint)
            # Consume the queue only after the rebuild succeeds: if
            # ``_remove_faiss_ids_locked`` raises (FAISS rebuild failure),
            # the tombstones must survive so the next
            # ``index_done_callback`` retry re-applies them instead of
            # persisting the stale rows as if the deletes had happened.
            if to_remove:
                self._remove_faiss_ids_locked(to_remove)
                self._index_dirty = True
            # Removals that landed move into the redo log until a save
            # persists them; ids that matched no row have nothing to
            # persist and are not logged at all. Union rather than replace:
            # an id already logged may have versions this pass did not see
            # (a row another writer has not published back yet).
            for doc_id, fingerprints in matched.items():
                self._unsaved_deletes.setdefault(doc_id, set()).update(fingerprints)
            self._pending_deletes = set()
            logger.info(
                f"[{self.workspace}] {self.namespace} flush: applied "
                f"{len(to_remove)} deferred vector deletes ({queued} queued, "
                f"{len(self._unsaved_deletes)} awaiting save)"
            )

        # Replay materialized-but-unsaved upserts (redo log). A prior flush
        # materialized these rows and the save then failed; a reload since
        # (foreign commit) rebuilt the in-memory state from a snapshot that
        # lacks them. Rebuilding from the logged records + cached vectors
        # puts them back on top of whatever the other writer committed,
        # without re-embedding (issue #3688). An id that is pending again is
        # skipped — the newer buffered doc materializes below and supersedes
        # the logged row; one whose stored row already fingerprints equal to
        # the logged record (no reload happened, or an earlier retry
        # replayed it) has nothing to redo.
        #
        # A row that fingerprints *differently* is not automatically stale:
        # under the pipeline's dead-process recovery, the process that
        # materialized this redo entry can be declared dead, superseded by a
        # new writer that reprocesses the same document and commits a
        # genuinely newer row under the same (content-hash) id, and then
        # turn out not to have been dead after all and reach this replay.
        # Overwriting that newer row with our stale one would revert a
        # completed reprocess. Ordering runs through
        # `_resident_supersedes_redo`, which the read paths use too:
        # `__write_seq__` decides whenever both rows carry it, whole seconds
        # only when one predates the token, so a resident row written after
        # ours is left alone even when both landed inside the same second.
        if self._unsaved_upserts:
            drop_fids: list[int] = []
            replay_records: list[dict[str, Any]] = []
            superseded: list[str] = []
            for doc_id, pdoc in self._unsaved_upserts.items():
                if doc_id in self._pending_upserts:
                    continue
                fids = self._find_faiss_ids_by_custom_id(doc_id)
                fingerprint = self._row_fingerprint(pdoc.record)
                if len(fids) == 1 and (
                    self._row_fingerprint(self._id_to_meta[fids[0]]) == fingerprint
                ):
                    continue
                # Find-all: any resident row under this id being strictly
                # newer blocks the replay, since the rebuild below would drop
                # every one of them.
                if any(
                    self._resident_supersedes_redo(self._id_to_meta[fid], pdoc.record)
                    for fid in fids
                ):
                    superseded.append(doc_id)
                    continue
                # Replace whatever rows the id has now (find-all, same as
                # the pending materialization below) with our logged row.
                drop_fids.extend(fids)
                record = pdoc.record
                record["__vector__"] = pdoc.vector.tolist()
                replay_records.append(record)
            # Evicted after the loop, never inside it (mutating the dict
            # under iteration raises). A superseded entry must leave the log
            # rather than linger: it would be rescanned by every later flush
            # and, worse, keep poisoning the read paths, which serve the
            # logged record for any id still present here.
            for doc_id in superseded:
                del self._unsaved_upserts[doc_id]
            if replay_records:
                self._rebuild_index_locked(drop_fids, replay_records)
                self._index_dirty = True
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: replayed "
                    f"{len(replay_records)} unsaved upserts after a reload"
                )
            if superseded:
                # No dirty flag: dropping a redo entry touches the index not
                # at all, so there is nothing new to persist.
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: {len(superseded)} "
                    "unsaved upserts superseded by a strictly newer row "
                    "committed under the same id, redo entries dropped"
                )

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
        # Note which existing fids the re-upserted ids occupy (find-all so
        # duplicate __id__ rows from a legacy / corrupt store still get fully
        # cleaned).
        existing_fids: list[int] = []
        for doc_id, _ in pending_items:
            existing_fids.extend(self._find_faiss_ids_by_custom_id(doc_id))

        matrix = np.vstack([pdoc.vector for _, pdoc in pending_items]).astype(
            np.float32
        )
        records = []
        for i, (_, pdoc) in enumerate(pending_items):
            pdoc.record["__vector__"] = matrix[i].tolist()
            records.append(pdoc.record)

        if existing_fids:
            # An overwrite: drop the rows those ids had and write the ones
            # they have now, in a single rebuild. Splitting it into an add
            # and a removal would let a failure land one half — a missing
            # row one way round, two rows of different vintages the other —
            # and neither is reachable through the Nano backend, whose
            # client updates an existing id in place. This is also one pass
            # cheaper than adding and then rebuilding to clean up.
            self._rebuild_index_locked(existing_fids, records)
        else:
            # Nothing is being superseded, so appending cannot leave a
            # half-applied overwrite behind — and it avoids an O(rows)
            # rebuild on the path that does not need one.
            start_idx = self._index.ntotal
            self._index.add(matrix)
            for i, record in enumerate(records):
                self._id_to_meta[start_idx + i] = record

        self._index_dirty = True

        # The flushed entries move from the pending buffer into the redo log
        # rather than being dropped: the rows are materialized but not
        # durable yet, and clearing the only replayable copy here is exactly
        # the loss window of issue #3688. The caller clears the log once its
        # save lands. Only entries we just flushed leave the pending buffer —
        # the `is pdoc` identity check is defensive scaffolding: today the
        # non-reentrant _storage_lock locks out concurrent upserts for the
        # entire flush (including the asyncio.gather await), but if the lock
        # scope is ever relaxed, a concurrent upsert that re-set vector=None
        # must stay buffered (its redo entry is superseded next flush).
        for doc_id, pdoc in pending_items:
            self._unsaved_upserts[doc_id] = pdoc
            if self._pending_upserts.get(doc_id) is pdoc:
                del self._pending_upserts[doc_id]

        # No reconciliation pass against `_unsaved_deletes` follows. A row we
        # just wrote used to be able to fingerprint equal to one we removed
        # (same id, same content, same whole second), which would have made the
        # replay delete it — so the entry was dropped here. `__write_seq__`
        # rules that out: every upsert stamps its own token, so a rewrite is a
        # distinct row version and the entry names only the version it removed.
        # Keeping it removes and hides nothing that exists, and it is still
        # needed — a reload resurrecting the removed version is taken out again
        # by the replay at the top of the next flush, rewrite preserved.

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

        Deletes split along that same line. ``_pending_deletes`` is buffered
        work and is discarded; ``_unsaved_deletes`` is the redo log of
        removals that already reached ``self._index``, so it is **kept** —
        dropping it would neither restore the rows nor keep them gone, it
        would only make the removal fragile again, which is the exact state
        this log exists to end.

        ``_unsaved_upserts`` is kept for the same reason as its delete twin:
        it names rows that already reached ``self._index`` (the class this
        method intentionally does not roll back), and dropping it would only
        reopen the reload-loses-them window of issue #3688 for rows whose
        fate is already sealed either way — reprocessing overwrites them
        idempotently whether or not the replay preserved them.
        """
        if self._storage_lock is None:
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            return
        async with self._storage_lock:
            self._pending_upserts.clear()
            self._pending_deletes.clear()

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
               ``.meta.json``. A failure here **also raises**; the flushed
               docs sit in the ``_unsaved_upserts`` redo log (flush moved
               them there) and ``_index_dirty`` stays ``True``, so a later
               commit or ``finalize`` can reload a foreign snapshot and
               replay them on top — without re-embedding (issue #3688).
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
            self._unsaved_deletes.clear()  # the removals are durable now
            self._unsaved_upserts.clear()  # the rows are durable now
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            self._index_dirty = False
            return True

    @staticmethod
    def _format_record(dp: dict[str, Any]) -> dict[str, Any]:
        """Shape a stored/pending record into the public read result."""
        return {
            **{k: v for k, v in dp.items() if k not in ("__vector__", WRITE_SEQ_FIELD)},
            "id": dp.get("__id__"),
            "created_at": dp.get("__created_at__"),
        }

    def _matches_redo_entry(
        self, meta: dict[str, Any], fingerprints: set[str] | None
    ) -> bool:
        """True when a materialized row is one a redo entry removed.

        ``fingerprints`` is the set snapshotted from ``_unsaved_deletes``
        under the lock (``None`` when the id is not logged) — a set because
        a corrupt store can hold several rows under one id and the delete
        removed all of them. A row matching none of them took the id's place
        after the removal, and the replay deliberately preserves it — so the
        read paths must show it.
        """
        return fingerprints is not None and self._row_fingerprint(meta) in fingerprints

    @staticmethod
    def _resident_supersedes_redo(
        resident: dict[str, Any] | None, redo_record: dict[str, Any]
    ) -> bool:
        """True when a materialized row wins over a pending upsert-redo entry.

        The read paths must agree with what the next flush's replay will
        actually do, or read-your-writes reports a row the replay is about to
        decline to restore, so both go through this one rule: a resident row is
        preferred only when it was written **strictly** after the logged one —
        by the ``__write_seq__`` token when both rows carry one, falling back
        to whole-second ``__created_at__`` when either predates it (see
        ``write_seq``). An absent row, an older one, or a tie no token can
        break — either side written before the token existed, or two processes
        stamping inside one clock tick, since the bump that keeps tokens
        distinct is process-local — all leave the logged record as the answer.
        """
        return row_is_strictly_newer(resident, redo_record)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID (read-your-writes against the pending buffer).

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        # Read-your-writes: a buffered upsert is visible before its flush,
        # a queued delete hides the materialized row, and a redo-log entry
        # hides only the row version it removed (see _matches_redo_entry).
        async with self._storage_lock:
            pending = self._pending_upserts.get(id)
            if pending is not None:
                return self._format_record(pending.record)
            if id in self._pending_deletes:
                return None
            # Applied-but-unsaved row. A reload may have dropped it from the
            # index, in which case the next flush replays it and it is the row
            # this id resolves to — but another writer may also have committed
            # a strictly newer row under the same (content-hash) id, which the
            # replay deliberately preserves. So this cannot answer on its own:
            # consult the index and let the same ordering rule the replay uses
            # pick the winner.
            redo = self._unsaved_upserts.get(id)
            snapshot = self._unsaved_deletes.get(id)
            logged = set(snapshot) if snapshot is not None else None

        await self._get_index()  # reload-if-stale
        if redo is not None:
            resident = self._resolve_resident_row(id)
            if self._resident_supersedes_redo(resident, redo.record):
                return self._format_record(resident)
            return self._format_record(redo.record)
        metadata = self._resolve_resident_row(id, logged)
        if metadata is None:
            return None
        return self._format_record(metadata)

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

        # Read-your-writes: serve buffered upserts from the pending area,
        # hide ids with a queued delete (a redo-log entry hides only the row
        # version it removed), and only query the materialized index for the
        # remaining ids.
        result_map: dict[str, dict[str, Any]] = {}
        remaining: list[str] = []
        logged: dict[str, set[str]] = {}
        redo_docs: dict[str, _PendingFaissDoc] = {}
        async with self._storage_lock:
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is not None:
                    result_map[str(requested_id)] = self._format_record(pending.record)
                elif requested_id in self._pending_deletes:
                    continue
                else:
                    redo = self._unsaved_upserts.get(requested_id)
                    if redo is not None:
                        # Still queried against the index: the replay only
                        # restores the logged row when no strictly newer row
                        # was committed under the id meanwhile, so the index
                        # is needed to apply the same ordering rule below.
                        redo_docs[requested_id] = redo
                        remaining.append(requested_id)
                        continue
                    if requested_id in self._unsaved_deletes:
                        logged[requested_id] = set(self._unsaved_deletes[requested_id])
                    remaining.append(requested_id)

        if remaining:
            await self._get_index()  # reload-if-stale
            # One scan for the whole batch; ids with a redo entry are never in
            # ``logged`` (a removal evicts the entry), so both rules can be
            # resolved in the same pass.
            resolved = self._resolve_resident_rows(set(remaining), logged)
            for cid in remaining:
                metadata = resolved.get(cid)
                redo = redo_docs.get(cid)
                if redo is not None:
                    if self._resident_supersedes_redo(metadata, redo.record):
                        result_map[str(cid)] = self._format_record(metadata)
                    else:
                        result_map[str(cid)] = self._format_record(redo.record)
                    continue
                if metadata is not None:
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
        logged: dict[str, set[str]] = {}
        redo_docs: dict[str, _PendingFaissDoc] = {}
        async with self._storage_lock:
            to_embed: list[tuple[str, _PendingFaissDoc]] = []
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is None:
                    if requested_id in self._pending_deletes:
                        # Queued delete: the materialized row (if any) is
                        # about to go away — report it as absent.
                        continue
                    redo = self._unsaved_upserts.get(requested_id)
                    if redo is not None:
                        # Still queried against the index: the replay only
                        # restores the logged row when no strictly newer row
                        # was committed under the id meanwhile, so the index
                        # is needed to apply the same ordering rule below.
                        # The cached vector is always set once a doc reaches
                        # the log, so the redo branch can always answer.
                        redo_docs[requested_id] = redo
                        remaining.append(requested_id)
                        continue
                    if requested_id in self._unsaved_deletes:
                        logged[requested_id] = set(self._unsaved_deletes[requested_id])
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
            # One scan for the whole batch. A row an unsaved delete removed is
            # already filtered out here — it is the version the replay takes
            # out, so it reads as absent.
            resolved = self._resolve_resident_rows(set(remaining), logged)
            for cid in remaining:
                metadata = resolved.get(cid)
                redo = redo_docs.get(cid)
                if redo is not None:
                    if (
                        self._resident_supersedes_redo(metadata, redo.record)
                        and "__vector__" in metadata
                    ):
                        vectors_dict[cid] = metadata["__vector__"]
                    else:
                        vectors_dict[cid] = redo.vector.astype(np.float32).tolist()
                    continue
                if metadata is None:
                    continue
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
                # Discard buffered (unflushed) upserts, queued deletes and
                # both redo logs along with the data.
                self._pending_upserts.clear()
                self._pending_deletes.clear()
                self._unsaved_deletes.clear()
                self._unsaved_upserts.clear()

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
        """Flush buffered upserts/deletes and persist before shutdown (safety net).

        Normally ``index_done_callback`` has already drained the pending
        buffers and synced to disk, but two paths land here with work to do:

        - **Pending upserts/deletes only** (no prior ``index_done_callback``): flush
          and save. We reload first so a stale process picks up other
          writers' commits before merging its pending buffers in.
        - **Unsaved materialized changes** (``_index_dirty=True``): an
          earlier ``index_done_callback`` flushed into ``self._index`` but
          its save raised. Reloading is safe *and* necessary here: the redo
          logs replay both the removals (``_unsaved_deletes``) and the rows
          (``_unsaved_upserts``) on top of the snapshot, whereas skipping
          the reload would write our pre-commit snapshot over another
          writer's durable rows (issue #3688).

        ``_index_dirty`` can be ``True`` while all four buffers are empty —
        e.g. a materialized-but-unsaved upsert whose id is then ``delete()``-d
        (evicting its redo entry) inside a batch that itself aborts
        (``drop_pending_index_ops`` discards the now-queued pending delete,
        by design; see its docstring). Nothing here names that row for replay
        any more, but that is exactly why the reload below must still run:
        without it, this branch would save the stale in-memory snapshot
        straight over whatever another writer committed in the meantime.

        Flush / save failures propagate (same contract as
        ``index_done_callback``); a partially flushed buffer is preserved
        for a future retry.
        """
        async with self._storage_lock:
            if (
                not self._pending_upserts
                and not self._pending_deletes
                and not self._unsaved_deletes
                and not self._unsaved_upserts
                and not self._index_dirty
            ):
                return
            if (
                self._pending_upserts
                or self._pending_deletes
                or self._unsaved_deletes
                or self._unsaved_upserts
                or self._index_dirty
            ):
                # Reload so a stale process picks up other writers' commits
                # before the flush merges the pending buffers in and replays
                # the redo logs on top. Unsaved materialized changes are no
                # longer a reason to skip this: both logs replay after the
                # reload, whereas skipping used to save our pre-commit
                # snapshot over the other writer's durable rows. This must
                # fire even when all four buffers are empty (see the
                # docstring note above) — a bare ``_index_dirty`` still means
                # self._index may be stale relative to disk.
                if self._reload_index_from_disk_locked(for_write=True):
                    # The reload just replaced self._index with the fresh
                    # on-disk snapshot, so any prior materialized change is
                    # gone with it; nothing is dirty relative to this new
                    # baseline until the flush below applies something on
                    # top of it.
                    self._index_dirty = False
                await self._flush_pending_locked()
            if not self._index_dirty:
                # The flush changed nothing (e.g. every queued delete
                # targeted an id that is not in the index). Saving here
                # would rewrite both files and flag every other process
                # for a full reload for no reason.
                return
            self._save_faiss_index()
            self._unsaved_deletes.clear()  # the removals are durable now
            self._unsaved_upserts.clear()  # the rows are durable now
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            self._index_dirty = False
