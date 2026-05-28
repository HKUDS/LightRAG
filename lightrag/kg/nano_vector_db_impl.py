import asyncio
import base64
import os
import zlib
from typing import Any, final
from dataclasses import dataclass
import numpy as np
import time

from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
from lightrag.utils import (
    logger,
    compute_mdhash_id,
)

from lightrag.base import BaseVectorStorage
from nano_vectordb import NanoVectorDB
from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)


@dataclass
class _PendingNanoDoc:
    """A buffered upsert waiting for deferred embedding and materialization.

    ``record`` holds ``__id__`` / ``__created_at__`` plus the ``meta_fields``
    (which always include ``content`` for the entity/relation/chunk vdbs), so
    the content needed for deferred embedding lives in the record itself — no
    separate copy is kept. ``vector`` starts as ``None`` and is filled either
    during the lock-held flush or by a lazy ``get_vectors_by_ids`` embedding;
    once set it is reused by the next flush instead of re-calling the model.
    The compressed ``vector`` / raw ``__vector__`` keys are added to ``record``
    only at flush time, right before ``client.upsert``.
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
        queryable. A flush failure (embedding error / count mismatch) raises,
        leaves the pending buffer intact, and skips the disk write so no data
        is silently lost.
    """

    def __post_init__(self):
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

        current_time = int(time.time())
        pending = [
            (
                k,
                {
                    "__id__": k,
                    "__created_at__": current_time,
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
                self._pending_upserts[doc_id] = _PendingNanoDoc(record=record)

    async def _flush_pending_locked(self) -> None:
        """Embed pending docs and materialize them into ``self._client``.

        Precondition: the caller **must already hold** ``_storage_lock``. The
        lock is non-reentrant, so this helper never calls ``_get_client`` and
        operates on ``self._client`` directly. Embedding runs inside the lock
        on purpose (see class docstring, *Deferred-embedding protocol*).

        Failure handling: if embedding raises or the returned count does not
        match, the exception propagates and ``_pending_upserts`` is left intact
        so the next flush retries; nothing is written to ``self._client``.
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
            embeddings_list = await asyncio.gather(
                *[
                    self.embedding_func(batch, context="document")
                    for batch in batches
                ]
            )
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

        # Clear only the entries we just flushed (an upsert that arrived after
        # the snapshot would have re-set vector=None and must not be dropped).
        for doc_id, pdoc in pending_items:
            if self._pending_upserts.get(doc_id) is pdoc:
                del self._pending_upserts[doc_id]

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

        Buffered (unflushed) upserts are **not** searchable — only rows that a
        prior ``index_done_callback`` / ``finalize`` flushed are considered.
        Use the read-your-writes paths (``get_by_id`` / ``get_by_ids`` /
        ``get_vectors_by_ids``) to observe pending data before a flush.
        """
        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = query_embedding
        else:
            # Execute embedding outside of lock to avoid improve cocurrent
            embedding = await self.embedding_func(
                [query], context="query", _priority=5
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

        Persistence:
            Changes are in-memory only; cross-process visibility requires a
            subsequent ``index_done_callback``. In ``lightrag.py`` this is
            handled by ``_insert_done()`` at the end of the document batch.
            Callers outside the pipeline must persist explicitly.

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            # Hold the lock so the pending-cancel and the client delete are a
            # single critical section against a concurrent flush. Operate on
            # self._client directly (the lock is non-reentrant; no _get_client).
            async with self._storage_lock:
                for doc_id in ids:
                    self._pending_upserts.pop(doc_id, None)

                # Record count before deletion
                before_count = len(self._client)

                self._client.delete(ids)

                # Calculate actual deleted count
                after_count = len(self._client)
                deleted_count = before_count - after_count

            logger.debug(
                f"[{self.workspace}] Successfully deleted {deleted_count} vectors from {self.namespace}"
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
                # Cancel a buffered upsert for this entity, then delete from the
                # materialized client (lock non-reentrant; no _get_client).
                pending_cancelled = self._pending_upserts.pop(entity_id, None) is not None
                if self._client.get([entity_id]):
                    self._client.delete([entity_id])
                    deleted = True
                else:
                    deleted = False

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

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete every relation vector incident to ``entity_name``.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        **Not pipeline-gated** — see class docstring
        *Non-pipeline write paths*. The caller is responsible for
        ensuring single-writer serialization.
        """

        try:
            async with self._storage_lock:
                # Prune matching buffered upserts (their records carry src_id /
                # tgt_id from the relationships vdb meta_fields)...
                pending_ids = [
                    doc_id
                    for doc_id, pdoc in self._pending_upserts.items()
                    if pdoc.record.get("src_id") == entity_name
                    or pdoc.record.get("tgt_id") == entity_name
                ]
                for doc_id in pending_ids:
                    del self._pending_upserts[doc_id]

                # ...then scan the materialized client and delete matches.
                storage = getattr(self._client, "_NanoVectorDB__storage")
                ids_to_delete = [
                    dp["__id__"]
                    for dp in storage["data"]
                    if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
                ]
                if ids_to_delete:
                    self._client.delete(ids_to_delete)

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

    async def index_done_callback(self) -> bool:
        """Flush deferred embeddings, commit to disk, and notify other processes.

        This is the writer's **commit point** in the cross-process sync
        protocol (see class docstring). Effects, in order:
            1. ``_flush_pending_locked`` embeds every buffered upsert (once
               per id) and materializes it into ``self._client``. A failure
               here **raises** — pending is kept, nothing is written — so the
               loss surfaces through ``_insert_done`` instead of being silent.
            2. ``_save_to_disk_locked`` (``atomic_write``) lays a tmp file
               beside the target and renames it into place — readers either
               see the previous file in full or the new file in full, never a
               torn write.
            3. ``set_all_update_flags`` flips every registered process's
               ``storage_updated`` flag, then we immediately reset our own
               flag to ``False`` so the writer does not self-reload on the
               next call to ``_get_client``.

        Two-block structure (intentional, do not collapse):
            * **First ``async with``** — early-return path for a
              hypothetical second writer. Under the current single-writer
              pipeline contract (class docstring, invariant 1) the
              ``storage_updated.value`` check is permanently ``False`` in
              the writer, so this branch is **dead code in production**.
              It is kept as defensive scaffolding for any future relaxation
              of the single-writer invariant; removing it would silently
              re-enable lost-write bugs the moment a second writer is
              introduced. The pending buffer is left intact here so the next
              callback retries the flush.
            * **Second ``async with``** — flush + save + notify.
        """
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

        # Acquire lock and perform flush + persistence
        async with self._storage_lock:
            # Flush deferred embeddings first. On embedding error / count
            # mismatch this raises, leaving pending intact and skipping the
            # disk write (no silent data loss); the exception propagates.
            await self._flush_pending_locked()
            try:
                self._save_to_disk_locked()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error saving data for {self.namespace}: {e}"
                )
                return False  # Return error

        return True  # Return success

    @staticmethod
    def _format_record(dp: dict[str, Any]) -> dict[str, Any]:
        """Shape a stored/pending record into the public read result."""
        return {
            **{k: v for k, v in dp.items() if k not in ("vector", "__vector__")},
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

        client = await self._get_client()
        result = client.get([id])
        if result:
            return self._format_record(result[0])
        return None

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
        async with self._storage_lock:
            for requested_id in ids:
                pending = self._pending_upserts.get(requested_id)
                if pending is not None:
                    result_map[str(requested_id)] = self._format_record(pending.record)
                else:
                    remaining.append(requested_id)

        if remaining:
            client = await self._get_client()
            for dp in client.get(remaining):
                if not dp:
                    continue
                record = self._format_record(dp)
                key = record.get("id")
                if key is not None:
                    result_map[str(key)] = record

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
        async with self._storage_lock:
            to_embed: list[tuple[str, _PendingNanoDoc]] = []
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
            for result in client.get(remaining):
                if result and "vector" in result and "__id__" in result:
                    # Decompress vector data (Base64 + zlib + Float16 compressed)
                    decoded = base64.b64decode(result["vector"])
                    decompressed = zlib.decompress(decoded)
                    vector_f16 = np.frombuffer(decompressed, dtype=np.float16)
                    vector_f32 = vector_f16.astype(np.float32).tolist()
                    vectors_dict[result["__id__"]] = vector_f32

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
                # Discard buffered (unflushed) upserts along with the data.
                self._pending_upserts.clear()

                # delete _client_file_name
                if os.path.exists(self._client_file_name):
                    os.remove(self._client_file_name)

                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )

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

        Normally ``index_done_callback`` has already drained the pending buffer,
        but a flow that upserts without a trailing callback would otherwise lose
        those vectors silently. Flush + save here, and if anything remains
        buffered afterward raise ``RuntimeError`` naming the count so the loss is
        recorded (``finalize_storages`` logs it as an error).
        """
        async with self._storage_lock:
            if not self._pending_upserts:
                return
            await self._flush_pending_locked()
            self._save_to_disk_locked()
            await set_all_update_flags(self.namespace, workspace=self.workspace)
            self.storage_updated.value = False
            leftover = len(self._pending_upserts)

        if leftover:
            raise RuntimeError(
                f"[{self.workspace}] NanoVectorDBStorage.finalize() left {leftover} "
                f"pending upserts buffered after the final flush for {self.namespace}"
            )
