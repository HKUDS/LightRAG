import copy
import os
from dataclasses import dataclass
from typing import Any, ClassVar, final

from lightrag.base import (
    normalize_kv_create_time,
    BaseKVStorage,
)
from lightrag.file_atomic import reap_orphan_tmp_files
from lightrag.utils import (
    _cooperative_yield,
    load_json,
    log_without_raising,
    logger,
    validate_workspace,
    commit_in_storage_io,
    write_json,
)
from lightrag.exceptions import CommitBookkeepingError, StorageNotInitializedError
from .shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)


@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    """JSON-file-backed KV storage with **shared in-memory state across processes**.

    A *fundamentally different* cross-process model from
    ``NanoVectorDBStorage`` / ``FaissVectorDBStorage`` / ``NetworkXStorage``,
    which keep one in-memory copy per process and reconcile via file reloads.
    Compare carefully before changing either side.

    **Full contract: ``docs/design/FileBackedSnapshotContract.md``** (see its
    ``JsonKVStorage`` section) -- the reversed flag semantics, lock scope,
    commit granularity and what it means for chunk tracking, and the caveats.

    ``self._data`` is NOT a per-process dict: it is a reference into
    ``shared_storage._shared_dicts``, a ``Manager().dict()`` proxy every worker
    sees the same instance of (a plain dict in single-process mode). A mutation
    in any process is immediately visible in every other -- there is no reload,
    and adding a ``_get_*`` entry method would be wrong. The on-disk file is
    for durability only: read once in ``initialize``, written by
    ``index_done_callback``, and not on the steady-state path.

    ``storage_updated`` means the OPPOSITE of what it means in the file-backed
    classes: ``True`` is "there is dirty data still to flush", never "there is
    fresher data on disk to reload".

    ``_storage_lock`` is held over EVERY ``self._data`` access, read or write,
    because the Manager proxy is not free-threaded across processes.

    A commit rewrites the whole namespace, so any writer's flush durably
    publishes every other writer's pending mutation here. That matters most for
    the chunk-tracking namespaces, whose rows carry purge attribution.

    Supported for small-scale testing and validation only.
    """

    supports_strict_point_reads: ClassVar[bool] = True

    def __post_init__(self):
        # Reject path traversal before using workspace in a file path
        validate_workspace(self.workspace)
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

        reap_orphan_tmp_files(self._file_name, self.workspace or "_")

    async def initialize(self):
        """Bind to the shared namespace dict and load from disk on first init.

        ``try_initialize_namespace`` is a global init lock that returns
        ``True`` for exactly one process per ``(namespace, workspace)``;
        that process reads the JSON file and populates the shared
        ``self._data`` under ``_storage_lock``. Subsequent processes
        skip the file read — they will see the same shared dict via
        ``get_namespace_data``.

        For ``*_cache`` namespaces an extra
        ``_migrate_legacy_cache_structure`` pass runs against the loaded
        data and may rewrite the on-disk file if a migration was applied.
        """
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(
                self.namespace, workspace=self.workspace
            )
            self._data = await get_namespace_data(
                self.namespace, workspace=self.workspace
            )
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    # Migrate legacy cache structure if needed
                    if self.namespace.endswith("_cache"):
                        loaded_data = await self._migrate_legacy_cache_structure(
                            loaded_data
                        )

                    self._data.update(loaded_data)
                    data_count = len(loaded_data)

                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )

    async def index_done_callback(self) -> None:
        """Flush dirty in-memory state to disk and clear all dirty flags.

        Commit point in the shared-memory protocol (see the contract doc,
        *Reversed flag semantics*). Steps:
            1. Under ``_storage_lock``, check this process's
               ``storage_updated.value``. If ``False``, nothing to do —
               return.
            2. Snapshot ``self._data`` (converting from ``Manager.dict``
               proxy to a plain ``dict`` so the JSON encoder doesn't trip
               over the proxy) and write it via ``write_json``, which runs
               in the storage-IO pool rather than on the event loop. The
               snapshot is taken here, on the loop, so the worker thread
               only ever reads a private dict.
            3. If ``write_json`` reports sanitization was applied, the
               on-disk file no longer matches what was in memory — reload
               the cleaned data back into ``self._data`` under the same
               lock so the shared view stays consistent.
            4. ``clear_all_update_flags`` — wipe every process's
               ``storage_updated`` flag back to ``False``, signaling
               that the dirty data has been persisted.

        Note the **semantic difference** from the file-backed classes'
        commit: there is no ``set_all_update_flags`` here. The shared
        dict is already consistent across processes; the only thing
        ``index_done_callback`` does globally is *clear* the dirty
        flags.
        """
        async with self._storage_lock:
            if self.storage_updated.value:
                # DictProxy.copy() is a single Manager RPC that marshals the
                # whole mapping server-side; dict(proxy) would walk the mapping
                # protocol and fetch every value with its own RPC. Plain dicts
                # (single-process mode) copy cheaply and identically — write_json
                # only reads its argument, so the shallow copy is safe there too.
                data_dict = self._data.copy()

                # Calculate data count - all data is now flattened
                data_count = len(data_dict)

                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} KV writting {data_count} records to {self.namespace}"
                )

                # Off the event loop: this rewrites the whole file, which on a
                # large corpus takes seconds during which the single worker
                # serving HTTP would otherwise be blocked. Only the write moves
                # -- `data_dict` is snapshotted above, on the loop, under the
                # lock, so the worker thread touches nothing shared.
                write_outcome: dict[str, bool] = {}
                reconcile_failure: list[Exception] = []

                def _write() -> None:
                    write_outcome["needs_reload"] = write_json(
                        data_dict, self._file_name
                    )

                async def _committed() -> None:
                    # Reconciliation belongs INSIDE the write's uncancellable
                    # region. The sanitized file is already published at this
                    # point, so a cancellation landing between the two would
                    # leave the shared dict holding the rows that failed to
                    # encode — diverging from disk until some later flush
                    # happens to sanitize again. Before the offload the write
                    # and this branch were one unpreemptable synchronous block,
                    # which is the guarantee being restored here.
                    #
                    # Still not offloaded itself: it writes back into the shared
                    # dict, and it only runs for payloads that fail to encode.
                    if write_outcome.get("needs_reload"):
                        logger.info(
                            f"[{self.workspace}] Reloading sanitized data into shared memory for {self.namespace}"
                        )
                        cleaned_data = load_json(self._file_name)
                        if cleaned_data is not None:
                            try:
                                self._data.clear()
                                self._data.update(cleaned_data)
                            except Exception as exc:
                                # NOT publication, and not absorbable. On a
                                # shared ``Manager().dict()`` these are two
                                # separate RPCs, so a failure between them
                                # leaves the shared dict EMPTY while the file on
                                # disk holds the correct sanitized snapshot —
                                # and the dirty flags are still set, so the next
                                # flush would write that empty dict straight
                                # over it, losing every row in the namespace.
                                # Recorded so the handler below re-raises
                                # instead of reporting a healthy deferred
                                # publication.
                                reconcile_failure.append(exc)
                                raise

                    await clear_all_update_flags(
                        self.namespace, workspace=self.workspace
                    )

                try:
                    await commit_in_storage_io(_write, _committed)
                except CommitBookkeepingError as e:
                    if reconcile_failure:
                        # Fail loud. What is unreliable now is the shared
                        # in-memory view, and no later flush heals it — a later
                        # flush is what would PUBLISH it. The file on disk is
                        # the correct snapshot, so the recovery is to stop
                        # writing to this workspace and restart the workers,
                        # which reload it. Re-raised as the original failure
                        # rather than as CommitBookkeepingError: callers read
                        # that type as "committed, only publication deferred"
                        # and some deliberately absorb it.
                        raise reconcile_failure[0]
                    # Past the guard above, the only thing that can have
                    # failed is the dirty-flag clear — the file is published and
                    # the shared dict matches it. That heals on the next flush:
                    # the flags stay set, so the next index_done_callback
                    # rewrites this same snapshot and retries the clear.
                    #
                    # Re-raising instead would report a durable write as one that
                    # never happened, and every caller inherits that: _insert_done
                    # marks a document FAILED whose rows are on disk, and
                    # utils_graph's deletion paths turn a chunk-tracking cleanup
                    # they have already completed into fail/500.
                    log_without_raising(
                        logger.error,
                        f"[{self.workspace}] KV data for {self.namespace} was "
                        f"written to {self._file_name}, but its post-write "
                        f"bookkeeping failed: {e.__cause__}. The dirty flags stay "
                        "set, so the next commit rewrites this snapshot and "
                        "retries them.",
                    )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            result = self._data.get(id)
            if result:
                # Deep-copy so nested mutable fields (e.g. text_chunks'
                # llm_cache_list) don't alias the live storage row — a
                # shallow copy here only protects top-level keys, letting a
                # caller that mutates a nested list/dict in place (see
                # update_chunk_cache_list in utils.py) corrupt persisted
                # state without going through upsert.
                result = copy.deepcopy(result)
                # Ensure time fields are present, provide default values for old data
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                # Ensure _id field contains the clean ID
                result["_id"] = id
            return result

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        """Strict point read (base contract): the in-memory shared dict has
        no transport failure surface — a miss is a confirmed absence, and an
        uninitialized storage raises instead of returning one."""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")
        return await self.get_by_id(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            results = []
            for id in ids:
                data = self._data.get(id, None)
                if data:
                    # Deep-copy — see get_by_id for why a shallow copy isn't
                    # enough to protect nested mutable fields.
                    result = copy.deepcopy(data)
                    # Ensure time fields are present, provide default values for old data
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    # Ensure _id field contains the clean ID
                    result["_id"] = id
                    results.append(result)
                else:
                    results.append(None)
            return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update KV records in shared memory; mark all processes dirty.

        Two side effects under ``_storage_lock``:
            1. Stamp ``create_time`` / ``update_time`` / ``_id`` on each
               value, then ``self._data.update(data)``. Timestamping
               follows the ``BaseKVStorage.upsert`` contract: a new key
               gets both stamps, an existing key keeps its stored
               ``create_time`` (``0`` when the row never had one) and only
               advances ``update_time``, and a caller-supplied
               ``create_time`` is ignored. No I/O is needed for that --
               unlike the remote backends, the previous value is already in
               shared memory. Because
               ``self._data`` is the shared ``Manager.dict()`` proxy, the
               update is visible to all processes immediately — no
               reload needed.
            2. ``set_all_update_flags`` — flip every process's
               ``storage_updated.value`` to ``True``. Here ``True``
               means *"there is dirty data that still needs to be
               flushed to disk"*, **not** *"there is fresher data on
               disk"* as in the file-backed classes (see the contract doc
               for the contrast).

        Persistence is deferred to the next ``index_done_callback`` (the
        pipeline calls this via ``_insert_done()`` after each batch).

        Note: the per-key prep loop calls ``_cooperative_yield`` inside
        the lock. That is safe because ``NamespaceLock`` is non-
        reentrant — siblings waiting on this lock stay blocked across
        the yield; only unrelated coroutines benefit from the yield.
        """
        if not data:
            return

        import time

        current_time = int(time.time())  # Get current Unix timestamp

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")
        async with self._storage_lock:
            # Add timestamps to data based on whether key exists.
            # The loop reads self._data (k in self._data) so it must stay inside
            # the lock. _cooperative_yield is safe here: NamespaceLock is
            # non-reentrant, so other coroutines waiting on this lock will block
            # until we release it; the yield only benefits unrelated coroutines.
            for i, (k, v) in enumerate(data.items(), start=1):
                # For text_chunks namespace, ensure llm_cache_list field exists
                if self.namespace.endswith("text_chunks"):
                    if "llm_cache_list" not in v:
                        v["llm_cache_list"] = []

                # Timestamps per the BaseKVStorage.upsert contract. A single
                # ``get`` -- not ``__contains__`` + ``__getitem__`` -- because
                # on a multi-worker deployment ``self._data`` is a
                # ``Manager().dict()`` proxy and each subscript is a separate
                # RPC; ``get`` is what this file's read paths already use.
                # Values are always dicts, so ``None`` means absent.
                existing = self._data.get(k)
                if existing is not None:
                    # Update: the business value is replaced wholesale, but the
                    # storage-managed create_time survives it. A legacy row
                    # without the field records 0 (unknown) -- never a
                    # fabricated original timestamp.
                    v["update_time"] = current_time
                    v["create_time"] = normalize_kv_create_time(
                        existing.get("create_time")
                        if isinstance(existing, dict)
                        else None
                    )
                else:  # New key, set both create_time and update_time
                    v["create_time"] = current_time
                    v["update_time"] = current_time

                v["_id"] = k
                await _cooperative_yield(i)

            self._data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete(self, ids: list[str]) -> None:
        """Remove records from shared memory; mark all processes dirty if any deleted.

        Under ``_storage_lock``: ``self._data.pop(doc_id, None)`` for
        each id. Only calls ``set_all_update_flags`` if at least one key
        was actually present (avoids creating spurious dirty state for
        no-op deletes).

        See the contract doc for the shared-memory + dirty-flag protocol
        and the semantic contrast vs file-backed classes.

        Args:
            ids: List of document IDs to be deleted from storage
        """
        async with self._storage_lock:
            any_deleted = False
            for doc_id in ids:
                result = self._data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage contains no data, False otherwise
        """
        async with self._storage_lock:
            return len(self._data) == 0

    async def drop(self) -> dict[str, str]:
        """Clear shared memory and immediately persist the empty state.

        This method will:
            1. Clear the shared ``self._data`` dict under
               ``_storage_lock`` (visible to all processes immediately).
            2. ``set_all_update_flags`` so every process knows there is
               dirty state pending persistence.
            3. Call ``index_done_callback`` synchronously to flush the
               empty state to disk and clear the dirty flags.

        Caller contract:
            ``drop`` is destructive and **not** serialized by this
            storage class. The caller must hold the pipeline ``busy``
            reservation (the ``/documents/clear`` endpoint does this)
            before invoking it — running ``drop`` concurrently with an
            active document pipeline will wipe out in-flight work and
            silently lose data. See the contract doc,
            *Non-pipeline write paths*.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self, data: dict) -> dict:
        """Migrate legacy nested cache structure to flattened structure

        Args:
            data: Original data dictionary that may contain legacy structure

        Returns:
            Migrated data dictionary with flattened cache keys (sanitized if needed)
        """
        from lightrag.utils import generate_cache_key

        # Early return if data is empty
        if not data:
            return data

        # Check first entry to see if it's already in new format
        first_key = next(iter(data.keys()))
        if ":" in first_key and len(first_key.split(":")) == 3:
            # Already in flattened format, return as-is
            return data

        migrated_data = {}
        migration_count = 0

        for key, value in data.items():
            # Check if this is a legacy nested cache structure
            if isinstance(value, dict) and all(
                isinstance(v, dict) and "return" in v for v in value.values()
            ):
                # This looks like a legacy cache mode with nested structure
                mode = key
                for cache_hash, cache_entry in value.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    migrated_data[flattened_key] = cache_entry
                    migration_count += 1
            else:
                # Keep non-cache data or already flattened cache data as-is
                migrated_data[key] = value

        if migration_count > 0:
            logger.info(
                f"[{self.workspace}] Migrated {migration_count} legacy cache entries to flattened structure"
            )
            # Persist migrated data immediately and check if sanitization was applied
            needs_reload = write_json(migrated_data, self._file_name)

            # If data was sanitized during write, reload cleaned data
            if needs_reload:
                logger.info(
                    f"[{self.workspace}] Reloading sanitized migration data for {self.namespace}"
                )
                cleaned_data = load_json(self._file_name)
                if cleaned_data is not None:
                    return cleaned_data  # Return cleaned data to update shared memory

        return migrated_data

    async def finalize(self):
        """On shutdown, flush ``*_cache`` namespaces to disk.

        Cache namespaces are routinely written to during query/extract
        without triggering an immediate ``index_done_callback`` (caches
        churn fast and the pipeline doesn't always end at a natural
        commit point). This hook ensures whatever dirty cache state is
        in shared memory at process exit gets persisted, so the next
        run can pick it up.

        Non-cache namespaces don't need this — their writes already
        flow through pipeline-driven ``_insert_done()`` commits.
        """
        if self.namespace.endswith("_cache"):
            await self.index_done_callback()
