import asyncio
import base64
import json
import os
import zlib
from hashlib import md5
from typing import Any, final
from dataclasses import dataclass
import numpy as np
import time

from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    validate_workspace,
)

from lightrag.base import BaseVectorStorage
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from nano_vectordb import NanoVectorDB
from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)
from .write_seq import WRITE_SEQ_FIELD, next_write_seq, row_is_strictly_newer


@dataclass
class _PendingNanoDoc:
    """A buffered upsert waiting for deferred embedding and materialization.

    ``record`` holds ``__id__`` / ``__created_at__`` / ``__write_seq__`` plus
    the ``meta_fields`` (which always include ``content`` for the
    entity/relation/chunk vdbs), so the content needed for deferred embedding
    lives in the record itself — no separate copy is kept. ``vector`` starts as
    ``None`` and is filled either during the lock-held flush or by a lazy
    ``get_vectors_by_ids`` embedding; once set it is reused by the next flush
    instead of re-calling the model. The compressed ``vector`` / raw
    ``__vector__`` keys are added to ``record`` only at flush time, right
    before ``client.upsert``.
    """

    record: dict[str, Any]
    vector: np.ndarray | None = None


@final
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    """File-backed vector storage built on the in-memory ``NanoVectorDB``.

    Storage model:
        A single ``NanoVectorDB`` instance lives in process memory; its full
        state is serialized to one JSON file at
        ``working_dir/[workspace/]vdb_<namespace>.json``. That JSON file is
        the **only** cross-process synchronization surface — there is no
        shared memory, no message bus, and no network channel between
        processes. All cross-process visibility is therefore mediated by
        (a) an atomic file write at commit time and (b) a per-namespace
        ``storage_updated`` flag distributed through
        ``lightrag.kg.shared_storage``.

    Concurrency invariants (the code in this file is correct *only* while
    all three hold):
        1. **Single writer per workspace.** The document pipeline's
           ``busy`` / ``destructive_busy`` flags (see ``AGENTS.md``
           *Pipeline concurrency contract*) guarantee that at most one
           process performs ``upsert`` / ``delete`` /
           ``index_done_callback`` at any time. Every other process is
           read-only with respect to this storage.
        2. **Eventual consistency is sufficient.** Read-only processes
           only need to observe the writer's data *after* the writer's
           ``index_done_callback`` completes. Reads that land in the gap
           between a writer's in-memory mutation and its commit may
           legitimately return the pre-update snapshot.
        3. **NanoVectorDB operations are fully synchronous.** Under a
           single-threaded asyncio event loop, ``client.upsert`` /
           ``client.query`` / ``client.delete`` cannot be preempted by
           another coroutine, which gives them implicit mutual exclusion
           over ``self._client.__storage``. This is why the methods below
           don't have to hold ``_storage_lock`` while calling into
           ``client``.

    Cross-process sync protocol:
        Writer side (``index_done_callback``):
            1. Atomically write the in-memory state to disk
               (``atomic_write`` swaps a tmp file into place).
            2. Call ``set_all_update_flags`` to flip every process's
               ``storage_updated`` flag (including the writer's own).
            3. Immediately reset the writer's own flag to ``False`` so
               the next call to ``_get_client`` does not trigger a
               self-reload of the data this process just wrote.
        Reader side (any method that goes through ``_get_client``):
            1. Inside ``_storage_lock``, observe
               ``storage_updated.value is True``.
            2. **Fully reload** ``self._client`` from disk — NanoVectorDB
               has no incremental sync API, so the entire JSON file is
               re-parsed and a fresh in-memory matrix is rebuilt.
            3. Reset the reader's own flag to ``False`` so concurrent
               coroutines in the same process don't double-reload.

    Lock scope:
        ``_storage_lock`` is a per-``(namespace, workspace)`` keyed lock
        spanning both intra-process coroutines and inter-process workers.
        It only wraps the *reload* and *commit* critical sections, not
        every ``client.xxx`` call. Operating on ``client`` outside the
        lock is safe today *because of invariant (3)* — if either premise
        is ever broken (e.g. ``client.xxx`` is moved to a thread pool, or
        NanoVectorDB is swapped for an async vector library), the lock
        scope must be widened to cover the mutation/read itself.

    Non-pipeline write paths:
        The pipeline's ``busy`` gate serializes ``upsert`` / ``delete`` /
        ``index_done_callback`` called from the document ingestion and
        purge flows. The following entry points are **not** serialized by
        the pipeline gate and must be guarded externally:
            * ``drop`` — currently gated by the API layer (the
              ``/documents/clear`` endpoint takes the pipeline busy
              reservation before invoking it).
            * ``delete_entity`` / ``delete_entity_relation`` — currently
              not exposed in the WebUI. If you wire them up to a new
              caller, that caller must arrange single-writer
              serialization the same way the pipeline does.

    Deferred-embedding protocol:
        ``upsert`` does **not** call the embedding model. It only buffers a
        ``_PendingNanoDoc`` (content-bearing record + ``vector=None``) in the
        minimal ``self._pending_upserts`` area, overwriting any prior pending
        doc for the same id (which also clears a temp vector a previous
        ``get_vectors_by_ids`` may have cached). The model is called once per
        id at flush time (``_flush_pending_locked``), so repeated upserts of
        the same id — and many small upsert calls — embed only once. See
        issue #2785 and the ``OpenSearchVectorDBStorage`` equivalent.

        Embedding runs **inside ``_storage_lock``** during the flush (not in
        ``upsert``): under the single-writer invariant this keeps the content
        used for embedding consistent with the record written to disk and
        prevents a destructive op from interleaving between embed and write.
        The lock is non-reentrant, so ``_flush_pending_locked`` requires the
        caller to already hold it and operates on ``self._client`` directly
        (never through ``_get_client``).

        Reads are read-your-writes: ``get_by_id`` / ``get_by_ids`` /
        ``get_vectors_by_ids`` consult ``_pending_upserts`` first.
        ``get_vectors_by_ids`` lazily embeds a pending doc on demand and
        caches the vector back for the next flush. ``query`` and
        ``client_storage`` see only data already materialized into
        ``self._client`` — unflushed pending data is intentionally not
        queryable. A flush failure (embedding error, count mismatch, or save
        IO error) raises through ``index_done_callback``; the pending buffer
        is preserved on flush failure. If only the save failed, the flushed
        docs have moved into the ``_unsaved_upserts`` redo log (record +
        cached vector) and ``_client_dirty`` stays ``True``: a later commit
        or ``finalize`` reloads whatever another writer committed meanwhile
        and replays the logged rows on top — without re-embedding (issue
        #3688). The log is cleared only once a save lands.

    Deferred-delete protocol:
        ``delete`` mirrors the buffering above: it cancels any pending upsert
        for the id and queues the id in ``self._pending_deletes`` instead of
        removing the row from ``self._client``. ``_flush_pending_locked``
        applies every queued id in **one** ``client.delete`` call, and does so
        **before** materializing pending upserts, so an id that was deleted
        and re-upserted in the same batch ends up with exactly the new row.

        The motivation is cost: ``NanoVectorDB.delete`` rebuilds the entire
        matrix (``np.delete``) on every call, and the entity/relation merge
        stage deletes the stale forward/reverse rows once per relation — so an
        eager delete copied the whole matrix per merged relation.

        Two buffers, because the flush and the save can fail independently:
            * ``_pending_deletes`` — queued, not applied to ``self._client``.
            * ``_unsaved_deletes`` — applied to ``self._client`` but not yet
              on disk, kept as ``id -> a fingerprint of the removed row``.
              This is a redo log, not a pending buffer.

        The redo log exists because a removal that reached ``self._client``
        can still be undone: if the save fails, ``index_done_callback``'s
        unconditional reload replaces ``self._client`` with the on-disk
        snapshot and the row returns. Replaying the log after that reload
        removes it again, so the reload stays lossless for deletes. (Upserts
        carry the mirror log: a flush moves its docs from ``_pending_upserts``
        into the ``_unsaved_upserts`` redo log rather than dropping them, so
        a materialized-but-unsaved upsert is replayed after the same reload;
        see the deferred-embedding protocol above and issue #3688.)

        A replay matches on the row, not on the id alone. Ids are content
        hashes, so another writer can publish a *new* row under an id we
        removed, and deleting by id would destroy it. The log therefore stores
        ``_row_fingerprint`` of the row it removed — a digest of the whole
        stored record, stable across a save/reload — and a replay removes only
        a row that still matches it. A whole-second ``__created_at__`` is not
        enough on its own: ``upsert`` stamps ``int(time.time())``, so a rewrite
        inside the same second as the removed row carries the same timestamp.

        A rewrite *identical* in content to the row removed is still a
        distinct row version: every write also stamps its own
        ``__write_seq__`` token (see ``write_seq``), so it fingerprints
        differently and the replay preserves it — whether we wrote it or
        another writer did. Only a row written before the token existed is
        indistinguishable from the row we removed, and the replay removes it.

        Being replayable is also what lets ``finalize`` reload before it
        retries a save: without the logs it had to choose between skipping
        the reload (saving a pre-commit snapshot over another writer's rows)
        and reloading (dropping its own unsaved changes). With both logs —
        ``_unsaved_deletes`` for removals, ``_unsaved_upserts`` for rows —
        every retry path reloads first and replays on top. The reload is
        unconditional: ``finalize`` runs it even when both logs are empty and
        only the dirty flag is set, since a bare dirty flag still means the
        in-memory state may be stale relative to disk. What a replay
        deliberately does *not* restore is a row another writer has since
        superseded with a strictly newer one.

        The upsert replay is scoped by the redo entry itself: an id whose
        stored row already fingerprints equal to the logged record has
        nothing to redo. Any *other* row under that id is ordered against
        ours by the ``__write_seq__`` token, or by whole-second
        ``__created_at__`` when one of them predates it: a strictly newer one
        is left alone and our redo entry is dropped (another writer
        legitimately superseded us — reverting it would undo a completed
        reprocess), while an older one is overwritten by ours. Only a tie no
        token can break — one of the two rows written before the token
        existed — still falls back to being overwritten by the replay. A removal request
        evicts the id's redo entry (``delete`` / ``delete_entity`` /
        ``delete_entity_relation``), or the replay would resurrect the row
        the removal just took out.

        The read paths apply that same ordering rule rather than answering
        from the log alone, so read-your-writes never reports a row the
        replay is about to decline to restore.

        Ids that matched no row are not logged at all — there is nothing to
        persist for them, and replaying them could only hit a row they were
        never meant to touch. The log is cleared once a save lands, and an
        aborting batch keeps it: ``drop_pending_index_ops`` discards buffered
        work, not removals that already reached ``self._client``.

        ``delete_entity`` / ``delete_entity_relation`` stay eager — they are
        off the merge hot path — but their removals are applied-and-unsaved
        just the same, so they are recorded in the log too.

        The two buffers are scoped differently, and deliberately so.
        ``_pending_deletes`` holds a *request*: the caller named an id, so the
        flush removes whatever row carries that id — the by-id contract every
        server-backed backend implements, and the one purge relies on to leave
        nothing behind. Version-scoping a request would silently skip a delete
        the caller asked for, and pinning the version at ``delete`` time would
        also put back the per-call ``O(rows)`` lookup this protocol exists to
        remove. ``_unsaved_deletes`` holds a *record* of a removal that already
        happened, so it is version-scoped: replaying it must not remove a row
        that has since taken the id's place.

        The read-your-writes paths mirror both rules. ``get_by_id`` /
        ``get_by_ids`` / ``get_vectors_by_ids`` consult the buffers after
        ``_pending_upserts``: a queued id reads as absent, while a logged one
        hides only the row the entry names — a replacement the replay would
        preserve stays readable. An ``upsert`` cancels a *queued* delete for
        the same id (``client.upsert`` overwrites the row in place, so
        applying it first would be redundant work) but never the redo log: the
        buffered row may be discarded by an aborting batch before it
        materializes, and dropping the entry then would leave the reload
        nothing to replay.
        ``query`` and ``client_storage`` are unchanged: they read the
        materialized index, so a queued delete still surfaces there until the
        flush — the same contract the Qdrant and PostgreSQL buffers document.

        Both buffers are in-memory only: they are dropped by ``drop`` and lost
        on a crash before the flush.
    """

    def __post_init__(self):
        # Reject path traversal before using workspace in a file path
        validate_workspace(self.workspace)
        self._validate_embedding_func()
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
            self.workspace = ""
            workspace_dir = working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        self._client_file_name = os.path.join(
            workspace_dir, f"vdb_{self.namespace}.json"
        )

        self._max_batch_size = self.global_config["embedding_batch_num"]

        # Sweep orphan tmp siblings left behind by hard kills mid-save before
        # NanoVectorDB opens the target file.
        reap_orphan_tmp_files(self._client_file_name, self.workspace or "_")

        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim,
            storage_file=self._client_file_name,
        )

        # Minimal pending area for deferred embedding: id -> _PendingNanoDoc.
        # Holds only records not yet embedded+materialized into self._client;
        # it never duplicates rows already written to the client. Flushed
        # under _storage_lock by _flush_pending_locked().
        self._pending_upserts: dict[str, _PendingNanoDoc] = {}
        # Ids queued for removal, applied in one batched client.delete() by
        # _flush_pending_locked (see *Deferred-delete protocol* in the class
        # docstring). NanoVectorDB.delete() rebuilds the whole matrix per
        # call, and the merge stage deletes once per relation, so deferring
        # turns O(relations) full-matrix copies into one per flush.
        self._pending_deletes: set[str] = set()
        # Removals already applied to self._client but not yet on disk, kept
        # as id -> _row_fingerprint(removed row) so a replay after a reload
        # only removes that row again, never a newer one written meanwhile.
        self._unsaved_deletes: dict[str, str] = {}
        # True when self._client has materialized changes that have not been
        # successfully saved to disk yet. This lets finalize retry a save even
        # after a previous flush drained the pending buffer.
        self._client_dirty = False
        # Rows materialized into self._client whose save has not landed yet,
        # kept as id -> the flushed _PendingNanoDoc (record + cached vector).
        # The upsert mirror of _unsaved_deletes: a reload after a failed save
        # replaces self._client with the on-disk snapshot, and the next flush
        # replays these rows on top without re-embedding (issue #3688).
        # Cleared only once a save lands.
        self._unsaved_upserts: dict[str, _PendingNanoDoc] = {}

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

    def _reload_client_from_disk_locked(self, *, for_write: bool = False) -> bool:
        """Reload ``self._client`` if another process committed newer data.

        Precondition: the caller must already hold ``_storage_lock``. This is
        used by write paths as well as reads because deferred upserts mean a
        stale writer must merge its pending buffer into the latest on-disk
        snapshot, not save over it or return without flushing.
        """
        if not self.storage_updated.value:
            return False

        log_message = (
            f"[{self.workspace}] Process {os.getpid()} reloading {self.namespace} "
            "due to update by another process"
        )
        if for_write:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim,
            storage_file=self._client_file_name,
        )
        self.storage_updated.value = False
        return True

    async def _get_client(self):
        """Return the live ``NanoVectorDB`` instance, reloading from disk if needed.

        This is the **single entry point** every public method funnels
        through to obtain ``self._client``. It is also the **only place
        readers transition to a fresher on-disk snapshot**: when another
        process has committed (via ``index_done_callback``) and flipped
        this process's ``storage_updated`` flag, the next call here
        rebuilds ``self._client`` by re-parsing the entire JSON file.
        NanoVectorDB has no incremental sync API — the reload is
        unconditionally a full file reload.

        Under the *Single writer* invariant (see class docstring), the
        reload branch never fires in the writer process: the writer
        resets its own flag at the end of every ``index_done_callback``.
        The branch exists for readers.

        ``_storage_lock`` is held during the check-and-reload to (a)
        serialize concurrent reload attempts by sibling coroutines in
        the same process and (b) interlock with ``index_done_callback``
        so a reader cannot observe a partially-saved file.
        """
        async with self._storage_lock:
            self._reload_client_from_disk_locked()
            return self._client

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vectors for deferred embedding; persistence is deferred too.

        Embedding is **not** performed here. Each record is buffered in
        ``self._pending_upserts`` with ``vector=None`` and the embedding model
        is called once per id at flush time (``_flush_pending_locked`` during
        ``index_done_callback`` / ``finalize``). This coalesces repeated
        upserts of the same id and many small upsert calls into a single
        embedding pass (see class docstring, *Deferred-embedding protocol*,
        and issue #2785).

        Persistence:
            Changes live only in this process's memory until the next
            ``index_done_callback``. Cross-process readers will not see
            them until that commit fires (see class docstring,
            *Cross-process sync protocol*). Until the flush, an upserted id
            is observable only through the read-your-writes read paths, not
            through ``query``.
        """
        # logger.debug(f"[{self.workspace}] Buffering {len(data)} to {self.namespace}")
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
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                },
            )
            for k, v in data.items()
        ]

        # Buffer under the lock to interlock with the lock-held flush. A new
        # _PendingNanoDoc(vector=None) overwrites any prior pending doc for the
        # same id, discarding a temp vector a previous get_vectors_by_ids may
        # have cached (content-version change -> must re-embed new content).
        async with self._storage_lock:
            for doc_id, record in pending:
                # A fresh upsert supersedes a queued delete for the same id:
                # client.upsert overwrites the row in place, so applying the
                # delete first would be redundant work. Mirrors the
                # PG/Qdrant buffers. The redo log is deliberately NOT touched
                # here: this row may never materialize (an aborting batch
                # drops it), and the fingerprint already keeps a replay off a
                # row written under the same id.
                self._pending_deletes.discard(doc_id)
                self._pending_upserts[doc_id] = _PendingNanoDoc(record=record)

    @staticmethod
    def _row_fingerprint(dp: dict[str, Any]) -> str:
        """Identify one *version* of a stored row.

        The redo log has to name the exact row it removed so a replay after a
        reload cannot hit a newer row another writer published under the same
        (content-hash) id. The digest covers the whole stored record — id,
        timestamp, ``__write_seq__``, content, the base64 vector, every meta
        field — and is stable across a save/reload round-trip: sorted keys
        make key order irrelevant, tuples and lists both render as JSON
        arrays, and the default ``ensure_ascii`` keeps the output encodable
        whatever the record holds.

        ``__vector__`` never reaches here — ``NanoVectorDB.upsert`` strips it
        into the matrix before the record is stored.
        """
        canonical = json.dumps(dp, sort_keys=True, default=str)
        return md5(canonical.encode("utf-8")).hexdigest()

    async def _flush_pending_locked(self) -> None:
        """Embed pending docs and materialize them into ``self._client``.

        Precondition: the caller **must already hold** ``_storage_lock``. The
        lock is non-reentrant, so this helper never calls ``_get_client`` and
        operates on ``self._client`` directly. Embedding runs inside the lock
        on purpose (see class docstring, *Deferred-embedding protocol*).

        Failure handling: if embedding raises or the returned count does not
        match, the exception propagates and both buffers are left intact so
        the next flush retries; no upsert is written to ``self._client``.
        """
        if (
            not self._pending_upserts
            and not self._pending_deletes
            and not self._unsaved_deletes
            and not self._unsaved_upserts
        ):
            return

        # One batched delete before the upserts materialize: one matrix copy
        # per flush instead of one per relation, and a deleted-then-reinserted
        # id keeps only the new row. The queue survives until a save persists
        # the removal, so a failed save can replay it (see the class
        # docstring); ids that matched nothing have nothing to persist.
        if self._pending_deletes or self._unsaved_deletes:
            queued = len(self._pending_deletes)
            storage = getattr(self._client, "_NanoVectorDB__storage")
            # A queued id removes whatever row is there; a replayed one only
            # removes the row version it removed before, so anything written
            # under that id since — by another writer or by us — survives.
            matched: dict[str, str] = {}
            for dp in storage["data"]:
                doc_id = dp["__id__"]
                if doc_id in self._pending_deletes:
                    matched[doc_id] = self._row_fingerprint(dp)
                elif doc_id in self._unsaved_deletes:
                    fingerprint = self._row_fingerprint(dp)
                    if fingerprint == self._unsaved_deletes[doc_id]:
                        matched[doc_id] = fingerprint
            if matched:
                self._client.delete(list(matched))
                self._client_dirty = True
            self._unsaved_deletes.update(matched)
            self._pending_deletes.clear()
            logger.info(
                f"[{self.workspace}] {self.namespace} flush: applied "
                f"{len(matched)} deferred deletes ({queued} queued, "
                f"{len(self._unsaved_deletes)} awaiting save)"
            )

        # Replay materialized-but-unsaved upserts (redo log). A prior flush
        # moved these rows into self._client and the save then failed; a
        # reload since (foreign commit) replaced self._client with a snapshot
        # that lacks them. Re-upserting from the logged record + cached
        # vector puts them back on top of whatever the other writer
        # committed, without re-embedding (issue #3688). An id that is
        # pending again is skipped — the newer buffered doc materializes
        # below and supersedes the logged row; one whose stored row already
        # fingerprints equal to the logged record (no reload happened, or an
        # earlier retry replayed it) has nothing to redo.
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
            storage = getattr(self._client, "_NanoVectorDB__storage")
            present = {
                dp["__id__"]: dp
                for dp in storage["data"]
                if dp["__id__"] in self._unsaved_upserts
            }
            replay_data = []
            superseded: list[str] = []
            for doc_id, pdoc in self._unsaved_upserts.items():
                if doc_id in self._pending_upserts:
                    continue
                resident = present.get(doc_id)
                if resident is not None:
                    # Fingerprint BEFORE re-attaching __vector__: the digest
                    # covers every record key, and the stored row never
                    # carries __vector__ (client.upsert strips it into the
                    # matrix).
                    if self._row_fingerprint(resident) == self._row_fingerprint(
                        pdoc.record
                    ):
                        continue
                    if self._resident_supersedes_redo(resident, pdoc.record):
                        superseded.append(doc_id)
                        continue
                record = pdoc.record
                record["__vector__"] = pdoc.vector
                replay_data.append(record)
            # Evicted after the loop, never inside it (mutating the dict
            # under iteration raises). A superseded entry must leave the log
            # rather than linger: it would be rescanned by every later flush
            # and, worse, keep poisoning the read paths, which serve the
            # logged record for any id still present here.
            for doc_id in superseded:
                del self._unsaved_upserts[doc_id]
            if replay_data:
                self._client.upsert(datas=replay_data)
                self._client_dirty = True
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: replayed "
                    f"{len(replay_data)} unsaved upserts after a reload"
                )
            if superseded:
                # No dirty flag: dropping a redo entry touches self._client
                # not at all, so there is nothing new to persist.
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
            embeddings = np.concatenate(embeddings_list)
            if len(embeddings) != len(to_embed):
                # Explicit raise (not a log): a mismatch would mis-pair vectors
                # with records. Keep pending intact so the next flush retries.
                raise RuntimeError(
                    f"[{self.workspace}] embedding is not 1-1 with pending data, "
                    f"{len(embeddings)} != {len(to_embed)}"
                )
            for (_, pdoc), embedding in zip(to_embed, embeddings):
                pdoc.vector = embedding

        list_data = []
        for _, pdoc in pending_items:
            vector = pdoc.vector
            # Compress vector using Float16 + zlib + Base64 for storage optimization
            vector_f16 = vector.astype(np.float16)
            compressed_vector = zlib.compress(vector_f16.tobytes())
            encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
            record = pdoc.record
            record["vector"] = encoded_vector
            record["__vector__"] = vector
            list_data.append(record)

        self._client.upsert(datas=list_data)
        self._client_dirty = True

        # The flushed entries move from the pending buffer into the redo log
        # rather than being dropped: the rows are materialized but not durable
        # yet, and clearing the only replayable copy here is exactly the loss
        # window of issue #3688. The caller clears the log once its save
        # lands. Only entries we just flushed leave the pending buffer (an
        # upsert that arrived after the snapshot would have re-set vector=None
        # and must stay buffered; its redo entry is superseded next flush).
        for doc_id, pdoc in pending_items:
            self._unsaved_upserts[doc_id] = pdoc
            if self._pending_upserts.get(doc_id) is pdoc:
                del self._pending_upserts[doc_id]

        # No reconciliation pass against `_unsaved_deletes` follows. A row we
        # just wrote used to be able to fingerprint equal to one we removed
        # (same id, same bytes, same whole second), which would have made the
        # replay delete it — so the entry was dropped here. `__write_seq__`
        # rules that out: every upsert stamps its own token, so a rewrite is a
        # distinct row version and the entry names only the version it removed.
        # Keeping it removes and hides nothing that exists, and it is still
        # needed — a reload resurrecting the removed version is taken out again
        # by the replay at the top of the next flush, rewrite preserved.

    def _save_to_disk_locked(self) -> None:
        """Atomically persist ``self._client`` and notify other processes.

        Precondition: the caller must already hold ``_storage_lock``. Factored
        out of ``index_done_callback`` so ``finalize`` reuses the exact same
        save+notify sequence. ``NanoVectorDB.save()`` always writes to whatever
        path is on the instance, so we temporarily redirect ``storage_file`` to
        the per-writer tmp and let ``atomic_write`` own the rename; the original
        path is restored on every path (success and exception).
        """

        def _save_atomic(tmp: str) -> None:
            original = self._client.storage_file
            self._client.storage_file = tmp
            try:
                self._client.save()
            finally:
                self._client.storage_file = original

        atomic_write(self._client_file_name, _save_atomic, self.workspace or "_")

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Similarity search over data already materialized into ``self._client``.

        Buffered upserts **and** buffered deletes are not visible here — only
        rows that a prior ``index_done_callback`` / ``finalize`` flushed are
        considered, so a queued delete still surfaces until the flush applies
        it. Use the read-your-writes paths (``get_by_id`` / ``get_by_ids`` /
        ``get_vectors_by_ids``, which consult both buffers) or flush first.
        Matches the contract documented for the other buffering backends.
        """
        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = query_embedding
        else:
            # Execute embedding outside of lock to avoid improve cocurrent
            embedding = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
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
                **{k: v for k, v in dp.items() if k not in ("vector", WRITE_SEQ_FIELD)},
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results

    @property
    async def client_storage(self):
        """Return a **live reference** to ``NanoVectorDB.__storage``.

        The returned dict is the same object NanoVectorDB mutates in
        place during ``upsert`` / ``delete``. Reading it outside
        ``_storage_lock`` is safe today only because NanoVectorDB
        mutations are fully synchronous (see class docstring,
        *Lock scope*). Callers must not retain this reference across an
        ``await`` that might cross into ``_get_client`` again: a reload
        will swap ``self._client`` for a fresh instance and leave the
        held reference pointing at the old (now-stale) storage.
        """
        client = await self._get_client()
        return getattr(client, "_NanoVectorDB__storage")

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs.

        Buffer semantics — deferred delete (see *Deferred-delete protocol*):
            The ids are queued, not removed from ``self._client`` yet. The
            batched removal runs at flush time, before pending upserts
            materialize. Reads already report a queued id as absent, so the
            deferral is invisible to callers.

            The request is scoped to the **id**, not to the row version
            present when it was made: the flush removes whatever row carries
            the id, exactly as the eager call and the server-backed backends
            do. If another writer rewrites the row inside the deferral window,
            that rewrite is what gets removed. Skipping it instead would leave
            purge with vectors it was told to delete — worse than the race it
            would avoid, and only reachable by breaking the single-writer
            invariant above. The redo log is version-scoped precisely because
            it is not a request; see the protocol section.

        Persistence:
            Changes are in-memory only; cross-process visibility requires a
            subsequent ``index_done_callback``. In ``lightrag.py`` this is
            handled by ``_insert_done()`` at the end of the document batch.
            Callers outside the pipeline must persist explicitly.

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            # Cancelling the pending upsert and queueing the id are one
            # critical section against a concurrent flush.
            async with self._storage_lock:
                for doc_id in ids:
                    self._pending_upserts.pop(doc_id, None)
                    # Evict the upsert redo entry too: this request removes
                    # whatever row the id has, and a surviving entry would
                    # replay — resurrect — that row at the next flush.
                    self._unsaved_upserts.pop(doc_id, None)
                    self._pending_deletes.add(doc_id)

            logger.debug(
                f"[{self.workspace}] Queued {len(ids)} deferred vector deletes for {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete the vector associated with a single entity name.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Buffer semantics — post-prune with caller short-circuit contract:
            The materialized client delete runs first; the matching
            pending upsert (if any) is popped **only after** it
            succeeds. If the materialized delete raises, the pending
            buffer stays intact and the exception is re-raised so the
            caller can short-circuit before ``index_done_callback``
            flushes a half-cleaned buffer.

        **Not pipeline-gated** — see class docstring
        *Non-pipeline write paths*. The caller is responsible for
        ensuring single-writer serialization.
        """
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            async with self._storage_lock:
                self._reload_client_from_disk_locked(for_write=True)

                # Materialized side first so a failure leaves the
                # pending buffer intact for the caller's retry path.
                existing = self._client.get([entity_id])
                if existing:
                    self._client.delete([entity_id])
                    # This removal is applied-but-unsaved like a flushed one,
                    # so it goes in the redo log too (a queued request for the
                    # same id is now redundant). Without it the same failed
                    # save + reload resurrects the row.
                    self._pending_deletes.discard(entity_id)
                    self._unsaved_deletes[entity_id] = self._row_fingerprint(
                        existing[0]
                    )
                    self._client_dirty = True
                    deleted = True
                else:
                    deleted = False

                # Materialized delete succeeded — safe to cancel any
                # buffered upsert for this entity.
                pending_cancelled = (
                    self._pending_upserts.pop(entity_id, None) is not None
                )
                # Evict the upsert redo entry unconditionally (like the
                # pending cancel): after a foreign reload the unsaved row is
                # not in self._client — `existing` is empty — yet a surviving
                # entry would replay it at the next flush.
                self._unsaved_upserts.pop(entity_id, None)

            if deleted or pending_cancelled:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")
            raise

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete every relation vector incident to ``entity_name``.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Buffer semantics — post-prune with caller short-circuit contract:
            The materialized client delete runs first; matching pending
            upserts are pruned **only after** it succeeds. If the
            materialized delete raises, the pending buffer stays intact
            and the exception is re-raised so the caller (e.g.
            ``adelete_by_entity``) can short-circuit before
            ``_persist_graph_updates`` triggers ``index_done_callback``
            on a half-cleaned buffer.

            Previously the buffer was pre-pruned and the outer
            ``except`` swallowed exceptions into ``logger.error`` — that
            combination silently dropped both buffered relation vectors
            and the failure signal.

        **Not pipeline-gated** — see class docstring
        *Non-pipeline write paths*. The caller is responsible for
        ensuring single-writer serialization.
        """
        try:
            async with self._storage_lock:
                self._reload_client_from_disk_locked(for_write=True)

                # Materialized side first so a failure leaves the
                # pending buffer intact for the caller's retry path.
                # Use .get() for src_id / tgt_id so rows from foreign
                # namespaces without those keys silently don't match.
                storage = getattr(self._client, "_NanoVectorDB__storage")
                rows_to_delete = {
                    dp["__id__"]: self._row_fingerprint(dp)
                    for dp in storage["data"]
                    if dp.get("src_id") == entity_name
                    or dp.get("tgt_id") == entity_name
                }
                ids_to_delete = list(rows_to_delete)
                if ids_to_delete:
                    self._client.delete(ids_to_delete)
                    # Applied-but-unsaved: record the removals in the redo log
                    # (queued requests for the same ids are now redundant), or
                    # a failed save plus a reload resurrects the rows.
                    self._pending_deletes.difference_update(ids_to_delete)
                    self._unsaved_deletes.update(rows_to_delete)
                    self._client_dirty = True

                # Materialized delete succeeded — safe to prune matching
                # buffered upserts so a subsequent flush won't re-upsert
                # the just-deleted relations.
                pending_ids = [
                    doc_id
                    for doc_id, pdoc in self._pending_upserts.items()
                    if pdoc.record.get("src_id") == entity_name
                    or pdoc.record.get("tgt_id") == entity_name
                ]
                for doc_id in pending_ids:
                    del self._pending_upserts[doc_id]
                # Same predicate over the upsert redo log, and unconditional
                # for the same reason: after a foreign reload the unsaved
                # relation rows are not in the scan above, yet surviving
                # entries would replay them at the next flush.
                redo_ids = [
                    doc_id
                    for doc_id, pdoc in self._unsaved_upserts.items()
                    if pdoc.record.get("src_id") == entity_name
                    or pdoc.record.get("tgt_id") == entity_name
                ]
                for doc_id in redo_ids:
                    del self._unsaved_upserts[doc_id]

            total = len(pending_ids) + len(ids_to_delete)
            if total:
                logger.debug(
                    f"[{self.workspace}] Deleted {total} relations for {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )
            raise

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts on an aborting batch.

        Only the pending buffer is dropped; records already materialized into
        ``self._client`` by a prior ``_flush_pending_locked`` whose save step
        then failed (``_client_dirty=True``) are intentionally NOT rolled back.

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
        removals that already reached ``self._client``, so it is **kept** —
        dropping it would neither restore the rows nor keep them gone, it
        would only make the removal fragile again, which is the exact state
        this log exists to end. The idempotent-reprocessing argument above is
        also weaker here: an upsert is re-issued by any reprocess of the
        document, while a delete is re-issued only if the reprocess reaches
        the same merge.

        ``_unsaved_upserts`` is kept for the same reason as its delete twin:
        it names rows that already reached ``self._client`` (the class this
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
               per id) and materializes it into ``self._client``. A failure
               here **raises** — pending is kept, nothing is written.
            3. ``_save_to_disk_locked`` (``atomic_write``) lays a tmp file
               beside the target and renames it into place — readers either
               see the previous file in full or the new file in full, never a
               torn write. A failure here **also raises**; ``_client_dirty``
               stays ``True`` and the redo logs (``_unsaved_upserts`` /
               ``_unsaved_deletes``) keep the flushed ops, so a later commit
               or ``finalize`` can reload a foreign snapshot and replay them
               on top (issue #3688).
            4. ``set_all_update_flags`` flips every registered process's
               ``storage_updated`` flag, then we immediately reset our own
               flag to ``False`` so the writer does not self-reload on the
               next call to ``_get_client``.

        Either failure surfaces loudly through ``_insert_done`` so the caller
        can abort the document batch instead of silently losing vectors. The
        bool return is kept for legacy callers but is effectively always
        ``True`` on the success path.
        """
        async with self._storage_lock:
            self._reload_client_from_disk_locked(for_write=True)

            # Flush + save both raise on failure (embedding mismatch / save IO
            # error). The exception propagates out of the lock so _insert_done
            # aborts the batch; pending stays intact and _client_dirty stays
            # True (if only the save failed) for a later retry.
            await self._flush_pending_locked()
            self._save_to_disk_locked()
            self._unsaved_deletes.clear()  # the removals are durable now
            self._unsaved_upserts.clear()  # the rows are durable now
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            self._client_dirty = False
            return True

    @staticmethod
    def _format_record(dp: dict[str, Any]) -> dict[str, Any]:
        """Shape a stored/pending record into the public read result."""
        return {
            **{
                k: v
                for k, v in dp.items()
                if k not in ("vector", "__vector__", WRITE_SEQ_FIELD)
            },
            "id": dp.get("__id__"),
            "created_at": dp.get("__created_at__"),
        }

    def _matches_redo_entry(self, dp: dict[str, Any], fingerprint: str | None) -> bool:
        """True when a materialized row is the one a redo entry removed.

        ``fingerprint`` is the value snapshotted from ``_unsaved_deletes``
        under the lock (``None`` when the id is not logged). A row that no
        longer matches took the id's place after the removal, and the replay
        deliberately preserves it — so the read paths must show it.
        """
        return fingerprint is not None and fingerprint == self._row_fingerprint(dp)

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
        # Read-your-writes: a buffered upsert is visible before its flush; a
        # queued delete makes the id read as absent; a logged one hides only
        # the row version it removed (see *Deferred-delete protocol*).
        async with self._storage_lock:
            pending = self._pending_upserts.get(id)
            if pending is not None:
                return self._format_record(pending.record)
            if id in self._pending_deletes:
                return None
            # Applied-but-unsaved row. A reload may have dropped it from
            # self._client, in which case the next flush replays it and it is
            # the row this id resolves to — but another writer may also have
            # committed a strictly newer row under the same (content-hash) id,
            # which the replay deliberately preserves. So this cannot answer
            # on its own: consult the client and let the same ordering rule
            # the replay uses pick the winner.
            redo = self._unsaved_upserts.get(id)
            logged = self._unsaved_deletes.get(id)

        client = await self._get_client()
        result = client.get([id])
        resident = result[0] if result else None
        if redo is not None:
            if self._resident_supersedes_redo(resident, redo.record):
                return self._format_record(resident)
            return self._format_record(redo.record)
        if resident is None or self._matches_redo_entry(resident, logged):
            return None
        return self._format_record(resident)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs (read-your-writes), preserving order.

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        # Read-your-writes: serve buffered upserts from the pending area and
        # only query the materialized client for the remaining ids.
        result_map: dict[str, dict[str, Any]] = {}
        remaining: list[str] = []
        logged: dict[str, str] = {}
        redo_docs: dict[str, _PendingNanoDoc] = {}
        async with self._storage_lock:
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is not None:
                    result_map[str(requested_id)] = self._format_record(pending.record)
                elif requested_id in self._pending_deletes:
                    continue  # queued delete, no newer upsert -> absent
                else:
                    redo = self._unsaved_upserts.get(requested_id)
                    if redo is not None:
                        # Still queried against the client: the replay only
                        # restores the logged row when no strictly newer row
                        # was committed under the id meanwhile, so the client
                        # is needed to apply the same ordering rule below.
                        redo_docs[requested_id] = redo
                        remaining.append(requested_id)
                        continue
                    # A logged id still goes to the client: only the removed
                    # row version is hidden, not one written in its place.
                    if requested_id in self._unsaved_deletes:
                        logged[requested_id] = self._unsaved_deletes[requested_id]
                    remaining.append(requested_id)

        if remaining:
            client = await self._get_client()
            resident_map: dict[str, dict[str, Any]] = {}
            for dp in client.get(remaining):
                if not dp:
                    continue
                doc_id = dp.get("__id__")
                if doc_id is not None:
                    resident_map[str(doc_id)] = dp
                if doc_id in redo_docs:
                    continue  # resolved against the redo entry below
                if self._matches_redo_entry(dp, logged.get(doc_id)):
                    continue
                record = self._format_record(dp)
                key = record.get("id")
                if key is not None:
                    result_map[str(key)] = record
            for requested_id, redo in redo_docs.items():
                resident = resident_map.get(str(requested_id))
                if self._resident_supersedes_redo(resident, redo.record):
                    result_map[str(requested_id)] = self._format_record(resident)
                else:
                    result_map[str(requested_id)] = self._format_record(redo.record)

        return [result_map.get(str(requested_id)) for requested_id in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs (read-your-writes), returning only ID and vector.

        For buffered upserts the vector is computed lazily (and cached back onto
        the pending doc so the next flush reuses it instead of re-embedding);
        for materialized rows the stored compressed vector is decoded.

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
        logged: dict[str, str] = {}
        redo_docs: dict[str, _PendingNanoDoc] = {}
        async with self._storage_lock:
            to_embed: list[tuple[str, _PendingNanoDoc]] = []
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is None:
                    if requested_id in self._pending_deletes:
                        continue  # queued delete, no newer upsert -> absent
                    redo = self._unsaved_upserts.get(requested_id)
                    if redo is not None:
                        # Still queried against the client: the replay only
                        # restores the logged row when no strictly newer row
                        # was committed under the id meanwhile, so the client
                        # is needed to apply the same ordering rule below.
                        # The cached vector is always set once a doc reaches
                        # the log, so the redo branch can always answer.
                        redo_docs[requested_id] = redo
                        remaining.append(requested_id)
                        continue
                    # A logged id still goes to the client: only the removed
                    # row version is hidden, not one written in its place.
                    if requested_id in self._unsaved_deletes:
                        logged[requested_id] = self._unsaved_deletes[requested_id]
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
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] embedding is not 1-1 with pending data, "
                        f"{len(embeddings)} != {len(to_embed)}"
                    )
                for (requested_id, pdoc), embedding in zip(to_embed, embeddings):
                    # Cache the vector back so the next flush reuses it.
                    pdoc.vector = embedding
                    vectors_dict[requested_id] = embedding.astype(np.float32).tolist()

        if remaining:
            client = await self._get_client()
            resident_map: dict[str, dict[str, Any]] = {}
            for result in client.get(remaining):
                if not result:
                    continue
                doc_id = result.get("__id__")
                if doc_id is not None:
                    resident_map[str(doc_id)] = result
                if doc_id in redo_docs:
                    continue  # resolved against the redo entry below
                if self._matches_redo_entry(result, logged.get(doc_id)):
                    continue
                if "vector" in result and "__id__" in result:
                    # Decompress vector data (Base64 + zlib + Float16 compressed)
                    decoded = base64.b64decode(result["vector"])
                    decompressed = zlib.decompress(decoded)
                    vector_f16 = np.frombuffer(decompressed, dtype=np.float16)
                    vector_f32 = vector_f16.astype(np.float32).tolist()
                    vectors_dict[result["__id__"]] = vector_f32
            for requested_id, redo in redo_docs.items():
                resident = resident_map.get(str(requested_id))
                if (
                    self._resident_supersedes_redo(resident, redo.record)
                    and "vector" in resident
                ):
                    decoded = base64.b64decode(resident["vector"])
                    decompressed = zlib.decompress(decoded)
                    vector_f16 = np.frombuffer(decompressed, dtype=np.float16)
                    vectors_dict[requested_id] = vector_f16.astype(np.float32).tolist()
                else:
                    vectors_dict[requested_id] = redo.vector.astype(np.float32).tolist()

        return vectors_dict

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and reinitialize the client.

        This method will:
        1. Remove the vector database storage file if it exists
        2. Reinitialize the vector database client
        3. Update flags to notify other processes
        4. Changes are persisted to disk immediately

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
                # Discard buffered (unflushed) upserts and queued deletes
                # along with the data — and both redo logs: there is nothing
                # left to replay onto.
                self._pending_upserts.clear()
                self._pending_deletes.clear()
                self._unsaved_deletes.clear()
                self._unsaved_upserts.clear()

                # delete _client_file_name
                if os.path.exists(self._client_file_name):
                    os.remove(self._client_file_name)

                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )
                self._client_dirty = False

                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False

                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}(file:{self._client_file_name})"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def finalize(self):
        """Flush any buffered upserts and persist before shutdown (safety net).

        Normally ``index_done_callback`` has already drained the pending buffer
        and synced to disk, but two paths land here with work to do:

        - **Pending upserts only** (no prior ``index_done_callback``): flush
          and save. We reload first so a stale process picks up other writers'
          commits before merging its pending buffer in.
        - **Unsaved materialized changes** (``_client_dirty=True``): an
          earlier ``index_done_callback`` flushed into ``self._client`` but
          its save raised. Reloading is safe *and* necessary here: the redo
          logs replay both the removals (``_unsaved_deletes``) and the rows
          (``_unsaved_upserts``) on top of the snapshot, whereas skipping the
          reload would write our pre-commit snapshot over another writer's
          durable rows (issue #3688).

        ``_client_dirty`` can be ``True`` while all four buffers are empty —
        e.g. a materialized-but-unsaved upsert whose id is then ``delete()``-d
        (evicting its redo entry) inside a batch that itself aborts
        (``drop_pending_index_ops`` discards the now-queued pending delete,
        by design; see its docstring). Nothing here names that row for replay
        any more, but that is exactly why the reload below must still run:
        without it, this branch would save the stale in-memory snapshot
        straight over whatever another writer committed in the meantime.

        Flush / save failures propagate (same contract as
        ``index_done_callback``); a partially flushed buffer is preserved for
        a future retry.
        """
        async with self._storage_lock:
            if (
                not self._pending_upserts
                and not self._pending_deletes
                and not self._unsaved_deletes
                and not self._unsaved_upserts
                and not self._client_dirty
            ):
                return
            if (
                self._pending_upserts
                or self._pending_deletes
                or self._unsaved_deletes
                or self._unsaved_upserts
                or self._client_dirty
            ):
                # Reload so a stale process picks up other writers' commits
                # before the flush merges the pending buffer in and replays
                # the redo logs on top. Unsaved materialized changes are no
                # longer a reason to skip this: both logs replay after the
                # reload, whereas skipping used to save our pre-commit
                # snapshot over the other writer's durable rows. This must
                # fire even when all four buffers are empty (see the
                # docstring note above) — a bare ``_client_dirty`` still
                # means self._client may be stale relative to disk.
                if self._reload_client_from_disk_locked(for_write=True):
                    # The reload just replaced self._client with the fresh
                    # on-disk snapshot, so any prior materialized change is
                    # gone with it; nothing is dirty relative to this new
                    # baseline until the flush below applies something on
                    # top of it.
                    self._client_dirty = False
                await self._flush_pending_locked()
            if not self._client_dirty:
                # The flush changed nothing (e.g. every queued delete targeted
                # an id that is not in the index). Saving here would rewrite
                # the whole file and flag every other process for a full
                # reload for no reason.
                return
            self._save_to_disk_locked()
            self._unsaved_deletes.clear()  # the removals are durable now
            self._unsaved_upserts.clear()  # the rows are durable now
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            self._client_dirty = False
