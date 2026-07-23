import os
import logging
import asyncio
import time
import uuid
from typing import Any, ClassVar, final, Union
from dataclasses import dataclass
import pipmaster as pm
import configparser
from contextlib import asynccontextmanager
import threading

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis, ConnectionPool  # type: ignore
from redis.exceptions import (  # type: ignore
    RedisError,
    ConnectionError,
    TimeoutError,
    WatchError,
)
from lightrag.utils import (
    logger,
    get_pinyin_sort_key,
    _cooperative_yield,
    validate_workspace,
)

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    BaseKVStorage,
    DocSchedulingRecord,
    DocStatusPage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
    FailureGenerationMode,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import (
    StorageControlPlaneError,
    StorageMigrationInProgressError,
    StorageRecordNotFoundError,
)
from ..kg.shared_storage import get_data_init_lock
import json

# Import tenacity for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Constants for Redis connection pool with environment variable support
MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "200"))
SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "30.0"))
SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "10.0"))
RETRY_ATTEMPTS = int(os.getenv("REDIS_RETRY_ATTEMPTS", "3"))

# Tenacity retry decorator for Redis operations
redis_retry = retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=(
        retry_if_exception_type(ConnectionError)
        | retry_if_exception_type(TimeoutError)
        | retry_if_exception_type(RedisError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class RedisConnectionManager:
    """Shared Redis connection pool manager to avoid creating multiple pools for the same Redis URI"""

    _pools = {}
    _pool_refs = {}  # Track reference count for each pool
    _lock = threading.Lock()
    # Strong references to background pool-close tasks scheduled from sync
    # contexts (init-error path). Held until each task completes so the loop
    # cannot GC an in-flight task, and so a failing aclose() does not surface
    # as "Task exception was never retrieved".
    _cleanup_tasks: set = set()

    @classmethod
    def get_pool(cls, redis_url: str) -> ConnectionPool:
        """Get or create a connection pool for the given Redis URL"""
        with cls._lock:
            if redis_url not in cls._pools:
                cls._pools[redis_url] = ConnectionPool.from_url(
                    redis_url,
                    max_connections=MAX_CONNECTIONS,
                    decode_responses=True,
                    socket_timeout=SOCKET_TIMEOUT,
                    socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
                )
                cls._pool_refs[redis_url] = 0
                logger.info(f"Created shared Redis connection pool for {redis_url}")

            # Increment reference count
            cls._pool_refs[redis_url] += 1
            logger.debug(
                f"Redis pool {redis_url} reference count: {cls._pool_refs[redis_url]}"
            )

        return cls._pools[redis_url]

    @classmethod
    def release_pool_ref(cls, redis_url: str):
        """Release a reference to the connection pool (synchronous).

        Decrements the reference count and, if it reaches zero, pops the pool
        from the registry and **returns it** so the caller can await its
        async ``aclose()``. The disconnect is intentionally NOT performed here
        because ``ConnectionPool.aclose()`` / ``disconnect()`` are coroutines
        and this method is called from both sync (init-error) and async
        (close/finalize) contexts.

        Returns the pool to disconnect, or ``None`` if the reference count is
        still positive (or the URL is unknown).
        """
        with cls._lock:
            if redis_url in cls._pool_refs:
                cls._pool_refs[redis_url] -= 1
                logger.debug(
                    f"Redis pool {redis_url} reference count: {cls._pool_refs[redis_url]}"
                )

                # If no more references, remove from registry and return for disconnect
                if cls._pool_refs[redis_url] <= 0:
                    pool = cls._pools.pop(redis_url)
                    del cls._pool_refs[redis_url]
                    return pool
            return None

    @classmethod
    async def release_pool(cls, redis_url: str):
        """Release a reference to the connection pool (async).

        Decrements the reference count and, when it reaches zero, awaits the
        pool's async ``aclose()`` so sockets are actually closed before
        ``finalize_storages()`` reports success.
        """
        pool = cls.release_pool_ref(redis_url)
        if pool is not None:
            try:
                await pool.aclose()
                logger.info(
                    f"Closed Redis connection pool for {redis_url} (no more references)"
                )
            except Exception as e:
                logger.error(f"Error closing Redis pool for {redis_url}: {e}")

    @classmethod
    async def close_all_pools(cls):
        """Close all connection pools (for cleanup)"""
        with cls._lock:
            pools = dict(cls._pools)
            cls._pools.clear()
            cls._pool_refs.clear()
        for url, pool in pools.items():
            try:
                await pool.aclose()
                logger.info(f"Closed Redis connection pool for {url}")
            except Exception as e:
                logger.error(f"Error closing Redis pool for {url}: {e}")

    @classmethod
    async def _close_pool_safely(cls, pool: ConnectionPool, redis_url: str) -> None:
        """Await ``pool.aclose()``, swallowing errors as best-effort cleanup."""
        try:
            await pool.aclose()
            logger.info(
                f"Closed Redis connection pool for {redis_url} (no more references)"
            )
        except Exception as e:
            logger.error(f"Error closing Redis pool for {redis_url}: {e}")

    @classmethod
    def schedule_pool_close(cls, pool: ConnectionPool, redis_url: str) -> None:
        """Schedule a background pool disconnect from a *synchronous* context.

        Used by the ``__post_init__`` init-error path, which cannot ``await``.
        If an event loop is running, the disconnect runs as a managed task
        (strong-referenced until done, errors swallowed by
        ``_close_pool_safely``). If no loop is running, this is a no-op: the
        coroutine is never created, so no ``RuntimeError`` masks the original
        init exception and no unawaited-coroutine warning is emitted. Skipping
        is safe — the pool was just created (no I/O yet) if this instance is the
        sole owner, and if it is shared the refcount is still positive so
        ``release_pool_ref`` would not have returned the pool.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(cls._close_pool_safely(pool, redis_url))
        cls._cleanup_tasks.add(task)
        task.add_done_callback(cls._cleanup_tasks.discard)


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    supports_strict_point_reads: ClassVar[bool] = True

    def __post_init__(self):
        validate_workspace(self.workspace)
        # Check for REDIS_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Redis storage instances
        redis_workspace = os.environ.get("REDIS_WORKSPACE")
        if redis_workspace and redis_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = redis_workspace.strip()
            logger.info(
                f"Using REDIS_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = ""
            logger.debug(f"Final namespace (no workspace): '{self.final_namespace}'")

        self._redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        self._pool = None
        self._redis = None
        self._initialized = False

        try:
            # Use shared connection pool
            self._pool = RedisConnectionManager.get_pool(self._redis_url)
            self._redis = Redis(connection_pool=self._pool)
            logger.info(
                f"[{self.workspace}] Initialized Redis KV storage for {self.namespace} using shared connection pool"
            )
        except Exception as e:
            # Clean up on initialization failure
            if self._redis_url:
                pool = RedisConnectionManager.release_pool_ref(self._redis_url)
                if pool is not None:
                    RedisConnectionManager.schedule_pool_close(pool, self._redis_url)
            logger.error(
                f"[{self.workspace}] Failed to initialize Redis KV storage: {e}"
            )
            raise

    async def initialize(self):
        """Initialize Redis connection and migrate legacy cache structure if needed"""
        async with get_data_init_lock():
            if self._initialized:
                return

            # Test connection
            try:
                async with self._get_redis_connection() as redis:
                    await redis.ping()
                    logger.info(
                        f"[{self.workspace}] Connected to Redis for namespace {self.namespace}"
                    )
                    self._initialized = True
            except Exception as e:
                logger.error(f"[{self.workspace}] Failed to connect to Redis: {e}")
                # Clean up on connection failure
                await self.close()
                raise

            # Migrate legacy cache structure if this is a cache namespace
            if self.namespace.endswith("_cache"):
                try:
                    await self._migrate_legacy_cache_structure()
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Failed to migrate legacy cache structure: {e}"
                    )
                    # Don't fail initialization for migration errors, just log them

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        if not self._redis:
            raise ConnectionError("Redis connection not initialized")

        try:
            # Use the existing Redis instance with shared pool
            yield self._redis
        except ConnectionError as e:
            logger.error(
                f"[{self.workspace}] Redis connection error in {self.namespace}: {e}"
            )
            raise
        except RedisError as e:
            logger.error(
                f"[{self.workspace}] Redis operation error in {self.namespace}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Unexpected error in Redis operation for {self.namespace}: {e}"
            )
            raise

    async def close(self):
        """Close the Redis connection and release pool reference to prevent resource leaks."""
        # Detach all instance state BEFORE the first await, so a concurrent /
        # re-entrant close() (or finalize-after-__aexit__) is a complete no-op
        # from the start and cannot release the shared pool a second time.
        redis = getattr(self, "_redis", None)
        redis_url = getattr(self, "_redis_url", None)
        self._redis = None
        self._redis_url = None
        self._pool = None

        # Close the client first, then release the pool reference. The pool
        # release lives in ``finally`` so that even if ``aclose()`` is cancelled
        # (CancelledError is a BaseException and escapes ``except Exception``),
        # the reference is still released — otherwise redis_url is already gone
        # and the refcount would leak permanently.
        try:
            if redis is not None:
                try:
                    await redis.aclose()
                    logger.debug(
                        f"[{self.workspace}] Closed Redis connection for {self.namespace}"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error closing Redis connection: {e}"
                    )
        finally:
            if redis_url:
                await RedisConnectionManager.release_pool(redis_url)
                logger.debug(
                    f"[{self.workspace}] Released Redis connection pool reference for {self.namespace}"
                )

    async def finalize(self):
        """Release the Redis client and shared-pool reference on shutdown.

        The base ``finalize`` is a no-op, so without this override the
        standard shutdown path (``LightRAG.finalize_storages``) never calls
        ``close()``: the shared pool's reference count only ever grows and
        the pool is never disconnected until process exit. Mirrors the
        release-on-finalize contract of the other KV backends (e.g.
        ``MongoKVStorage`` releases its client via ``ClientManager``).
        ``close()`` is best-effort and idempotent, so double-finalize and
        finalize-after-error are both safe.
        """
        await self.close()

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self.close()

    @redis_retry
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{self.final_namespace}:{id}")
                if data:
                    result = json.loads(data)
                    # Ensure time fields are present, provide default values for old data
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    return result
                return None
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error for id {id}: {e}")
                raise

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        """Strict point read (base contract): ``get_by_id`` already
        propagates every transport/decode failure (no swallow-and-None
        path), so a ``None`` from a healthy GET is a confirmed absence."""
        return await self.get_by_id(id)

    @redis_retry
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{self.final_namespace}:{id}")
                results = await pipe.execute()

                processed_results = []
                for result in results:
                    if result:
                        data = json.loads(result)
                        # Ensure time fields are present for all documents
                        data.setdefault("create_time", 0)
                        data.setdefault("update_time", 0)
                        processed_results.append(data)
                    else:
                        processed_results.append(None)

                return processed_results
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error in batch get: {e}")
                raise

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)  # Convert set to list for indexing
            for key in keys_list:
                pipe.exists(f"{self.final_namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    @redis_retry
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return


        current_time = int(time.time())  # Get current Unix timestamp

        async with self._get_redis_connection() as redis:
            try:
                # Check which keys already exist to determine create vs update
                pipe = redis.pipeline()
                for i, k in enumerate(data.keys(), start=1):
                    pipe.exists(f"{self.final_namespace}:{k}")
                    await _cooperative_yield(i)
                exists_results = await pipe.execute()

                # Add timestamps to data
                for i, (k, v) in enumerate(data.items(), start=1):
                    # For text_chunks namespace, ensure llm_cache_list field exists
                    if self.namespace.endswith("text_chunks"):
                        if "llm_cache_list" not in v:
                            v["llm_cache_list"] = []

                    # Add timestamps based on whether key exists
                    if exists_results[i - 1]:  # Key exists, only update update_time
                        v["update_time"] = current_time
                    else:  # New key, set both create_time and update_time
                        v["create_time"] = current_time
                        v["update_time"] = current_time

                    v["_id"] = k
                    await _cooperative_yield(i)

                # Store the data
                pipe = redis.pipeline()
                for i, (k, v) in enumerate(data.items(), start=1):
                    pipe.set(f"{self.final_namespace}:{k}", json.dumps(v))
                    await _cooperative_yield(i)
                await pipe.execute()

            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error during upsert: {e}")
                raise

    async def index_done_callback(self) -> None:
        # Redis handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        pattern = f"{self.final_namespace}:*"
        try:
            async with self._get_redis_connection() as redis:
                # Use scan to check if any keys exist
                async for key in redis.scan_iter(match=pattern, count=1):
                    return False  # Found at least one key
                return True  # No keys found
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs"""
        if not ids:
            return

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for id in ids:
                pipe.delete(f"{self.final_namespace}:{id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"[{self.workspace}] Deleted {deleted_count} of {len(ids)} entries from {self.namespace}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all keys under the current namespace.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to find all keys with the namespace prefix
                pattern = f"{self.final_namespace}:*"
                cursor = 0
                deleted_count = 0

                while True:
                    cursor, keys = await redis.scan(cursor, match=pattern, count=1000)
                    if keys:
                        # Delete keys in batches
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.delete(key)
                        results = await pipe.execute()
                        deleted_count += sum(results)

                    if cursor == 0:
                        break

                logger.info(
                    f"[{self.workspace}] Dropped {deleted_count} keys from {self.namespace}"
                )
                return {
                    "status": "success",
                    "message": f"{deleted_count} keys dropped",
                }

            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error dropping keys from {self.namespace}: {e}"
                )
                return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self):
        """Migrate legacy nested cache structure to flattened structure for Redis

        Redis already stores data in a flattened way, but we need to check for
        legacy keys that might contain nested JSON structures and migrate them.

        Early exit if any flattened key is found (indicating migration already done).
        """
        from lightrag.utils import generate_cache_key

        async with self._get_redis_connection() as redis:
            # Get all keys for this namespace
            keys = await redis.keys(f"{self.final_namespace}:*")

            if not keys:
                return

            # Check if we have any flattened keys already - if so, skip migration
            has_flattened_keys = False
            keys_to_migrate = []

            for key in keys:
                # Extract the ID part (after namespace:)
                key_id = key.split(":", 1)[1]

                # Check if already in flattened format (contains exactly 2 colons for mode:cache_type:hash)
                if ":" in key_id and len(key_id.split(":")) == 3:
                    has_flattened_keys = True
                    break  # Early exit - migration already done

                # Get the data to check if it's a legacy nested structure
                data = await redis.get(key)
                if data:
                    try:
                        parsed_data = json.loads(data)
                        # Check if this looks like a legacy cache mode with nested structure
                        if isinstance(parsed_data, dict) and all(
                            isinstance(v, dict) and "return" in v
                            for v in parsed_data.values()
                        ):
                            keys_to_migrate.append((key, key_id, parsed_data))
                    except json.JSONDecodeError:
                        continue

            # If we found any flattened keys, assume migration is already done
            if has_flattened_keys:
                logger.debug(
                    f"[{self.workspace}] Found flattened cache keys in {self.namespace}, skipping migration"
                )
                return

            if not keys_to_migrate:
                return

            # Perform migration
            pipe = redis.pipeline()
            migration_count = 0

            for old_key, mode, nested_data in keys_to_migrate:
                # Delete the old key
                pipe.delete(old_key)

                # Create new flattened keys
                for cache_hash, cache_entry in nested_data.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    full_key = f"{self.final_namespace}:{flattened_key}"
                    pipe.set(full_key, json.dumps(cache_entry))
                    migration_count += 1

            await pipe.execute()

            if migration_count > 0:
                logger.info(
                    f"[{self.workspace}] Migrated {migration_count} legacy cache entries to flattened structure in Redis"
                )


@final
@dataclass
class RedisDocStatusStorage(DocStatusStorage):
    """Redis implementation of document status storage.

    Memory-bounding scheduling sidecar (Phase 1)
    --------------------------------------------
    Alongside the primary ``{final_namespace}:{doc_id}`` JSON records this
    class maintains, ATOMICALLY with every write (``WATCH``→``MULTI/EXEC``
    with conflict retry), a scheduling sidecar under the
    ``{final_namespace}__sched:`` prefix — deliberately NOT matched by the
    ``{final_namespace}:*`` patterns the legacy SCAN readers and ``drop``
    use, so sidecar keys never leak into full-scan reads:

    * ``…__sched:status:{status}`` — one ZSET per status, score 0,
      member ``"{created_at}|{doc_id}"``; ISO-8601 strings make
      lexicographic member order the ``(created_at, id)`` keyset order, so
      ``ZRANGEBYLEX`` pages the sweep and ``ZCARD`` gives O(1) counts.
    * ``…__sched:basename:{canonical}`` — canonical basename → PRIMARY
      doc_id (rows with ``metadata.is_duplicate != true`` and a real
      file_path only). ``file_path`` is one-to-many across rows (duplicate
      markers keep the canonical name) so ONLY the primary row is indexed;
      maintenance is an eligibility state machine keyed on the (old, new)
      row values — a primary that turns into a content-duplicate in place
      deletes its own mapping (CAS: only if it still points to itself).
    * ``…__sched:ctrl`` — HASH {schema_version, mode,
      failure_generation_counter}. Counter and rows live in the same Redis
      persistence unit, so restore rolls them back together — no per-init
      recalibration is needed (unlike the JSON sidecar file); calibration
      happens once at migration.

    Existing deployments lack the sidecar: ``initialize`` runs a one-time
    streaming migration (SCAN in bounded batches → ZADD/SET, then publishes
    the ctrl marker LAST) guarded by a short-lease migration lock; ctrl
    presence with a foreign schema_version reads as MIGRATING — never
    LEGACY. NOTE: the migration lock only prevents duplicate migrations
    among new workers; isolating OLD writers is a deployment prerequisite
    (coordinated stop-write upgrade), not something this class can enforce.
    """

    supports_bounded_scheduling_pages: ClassVar[bool] = True
    supports_failure_generation: ClassVar[bool] = True
    supports_strict_doc_identity_lookup: ClassVar[bool] = True
    supports_strict_point_reads: ClassVar[bool] = True

    _CTRL_SCHEMA_VERSION: ClassVar[str] = "1"
    # Bounded retry budget for WATCH/MULTI conflict loops. High contention
    # on a single doc key is not expected (per-doc writes are serialized by
    # the pipeline); the cap turns a pathological livelock into an error.
    _WATCH_RETRY_LIMIT: ClassVar[int] = 50

    def __post_init__(self):
        validate_workspace(self.workspace)
        # Check for REDIS_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Redis storage instances
        redis_workspace = os.environ.get("REDIS_WORKSPACE")
        if redis_workspace and redis_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = redis_workspace.strip()
            logger.info(
                f"Using REDIS_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(
                f"[{self.workspace}] Final namespace with workspace prefix: '{self.namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = "_"
            logger.debug(
                f"[{self.workspace}] Final namespace (no workspace): '{self.namespace}'"
            )

        self._redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        self._pool = None
        self._redis = None
        self._initialized = False

        try:
            # Use shared connection pool
            self._pool = RedisConnectionManager.get_pool(self._redis_url)
            self._redis = Redis(connection_pool=self._pool)
            logger.info(
                f"[{self.workspace}] Initialized Redis doc status storage for {self.namespace} using shared connection pool"
            )
        except Exception as e:
            # Clean up on initialization failure
            if self._redis_url:
                pool = RedisConnectionManager.release_pool_ref(self._redis_url)
                if pool is not None:
                    RedisConnectionManager.schedule_pool_close(pool, self._redis_url)
            logger.error(
                f"[{self.workspace}] Failed to initialize Redis doc status storage: {e}"
            )
            raise

    async def initialize(self):
        """Initialize Redis connection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                async with self._get_redis_connection() as redis:
                    await redis.ping()
                    logger.info(
                        f"[{self.workspace}] Connected to Redis for doc status namespace {self.namespace}"
                    )
                # One-time scheduling sidecar bootstrap (no-op once the ctrl
                # marker exists) — must complete before serving: the paged
                # sweep, ZCARD counts and basename index all depend on it.
                await self._migrate_scheduling_sidecar()
                self._initialized = True
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to connect to Redis for doc status: {e}"
                )
                # Clean up on connection failure
                await self.close()
                raise

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        if not self._redis:
            raise ConnectionError("Redis connection not initialized")

        try:
            # Use the existing Redis instance with shared pool
            yield self._redis
        except ConnectionError as e:
            logger.error(
                f"[{self.workspace}] Redis connection error in doc status {self.namespace}: {e}"
            )
            raise
        except RedisError as e:
            logger.error(
                f"[{self.workspace}] Redis operation error in doc status {self.namespace}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Unexpected error in Redis doc status operation for {self.namespace}: {e}"
            )
            raise

    async def close(self):
        """Close the Redis connection and release pool reference to prevent resource leaks."""
        # Detach all instance state BEFORE the first await, so a concurrent /
        # re-entrant close() (or finalize-after-__aexit__) is a complete no-op
        # from the start and cannot release the shared pool a second time.
        redis = getattr(self, "_redis", None)
        redis_url = getattr(self, "_redis_url", None)
        self._redis = None
        self._redis_url = None
        self._pool = None

        # Close the client first, then release the pool reference. The pool
        # release lives in ``finally`` so that even if ``aclose()`` is cancelled
        # (CancelledError is a BaseException and escapes ``except Exception``),
        # the reference is still released — otherwise redis_url is already gone
        # and the refcount would leak permanently.
        try:
            if redis is not None:
                try:
                    await redis.aclose()
                    logger.debug(
                        f"[{self.workspace}] Closed Redis connection for doc status {self.namespace}"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error closing Redis connection: {e}"
                    )
        finally:
            if redis_url:
                await RedisConnectionManager.release_pool(redis_url)
                logger.debug(
                    f"[{self.workspace}] Released Redis connection pool reference for doc status {self.namespace}"
                )

    async def finalize(self):
        """Release the Redis client and shared-pool reference on shutdown.

        The base ``finalize`` is a no-op, so without this override the
        standard shutdown path (``LightRAG.finalize_storages``) never calls
        ``close()``: the shared pool's reference count only ever grows and
        the pool is never disconnected until process exit. Mirrors the
        release-on-finalize contract of the other DocStatus backends.
        ``close()`` is best-effort and idempotent, so double-finalize and
        finalize-after-error are both safe.
        """
        await self.close()

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self.close()

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)
            for key in keys_list:
                pipe.exists(f"{self.final_namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        ordered_results: list[dict[str, Any] | None] = []
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{self.final_namespace}:{id}")
                results = await pipe.execute()

                for result_data in results:
                    if result_data:
                        try:
                            ordered_results.append(json.loads(result_data))
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"[{self.workspace}] JSON decode error in get_by_ids: {e}"
                            )
                            raise
                    else:
                        ordered_results.append(None)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in get_by_ids: {e}")
                raise
        return ordered_results

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to iterate through all keys in the namespace
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        # Get all values in batch
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        # Count statuses
                        for value in values:
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    status = doc_data.get("status")
                                    if status in counts:
                                        counts[status] += 1
                                except json.JSONDecodeError:
                                    continue

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting status counts: {e}")

        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses in a single SCAN pass.

        Redis has no server-side multi-value filter, so documents must be fetched
        and filtered in Python.  This override performs a single SCAN + pipeline
        GET over the keyspace, filtering against a set of status values.  The
        previous pattern of N separate get_docs_by_status() calls would do N full
        SCANs (one per status), so this reduces keyspace traversal from N passes to one.
        Transport errors always propagate (SCAN interruption re-raises below);
        ``strict=True`` additionally raises on any record that cannot be parsed
        (complete-or-raise scheduling contract, see base class).
        """
        if not statuses:
            return {}
        status_values = {s.value for s in statuses}
        result = {}
        async with self._get_redis_connection() as redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        for key, value in zip(keys, values):
                            if not value:
                                continue
                            try:
                                doc_data = json.loads(value)
                                if doc_data.get("status") not in status_values:
                                    continue
                                doc_id = key.split(":", 1)[1]
                                data = doc_data.copy()
                                data.pop("content", None)
                                if "file_path" not in data:
                                    data["file_path"] = "no-file-path"
                                if "metadata" not in data:
                                    data["metadata"] = {}
                                if "error_msg" not in data:
                                    data["error_msg"] = None
                                result[doc_id] = DocProcessingStatus(**data)
                            except (json.JSONDecodeError, KeyError, TypeError) as e:
                                # TypeError is what DocProcessingStatus(**data)
                                # actually raises on missing required fields —
                                # without it the relaxed skip-and-log contract
                                # would crash the whole call instead.
                                logger.error(
                                    f"[{self.workspace}] Error processing document {key}: {e}"
                                )
                                if strict:
                                    raise
                                continue

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] SCAN interrupted while fetching docs by statuses "
                    f"— result is incomplete ({len(result)} documents collected): {e!r}"
                )
                raise

        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        result = {}
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to iterate through all keys in the namespace
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        # Get all values in batch
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        # Filter by track_id and create DocProcessingStatus objects
                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    if doc_data.get("track_id") == track_id:
                                        # Extract document ID from key
                                        doc_id = key.split(":", 1)[1]

                                        # Make a copy of the data to avoid modifying the original
                                        data = doc_data.copy()
                                        # Remove deprecated content field if it exists
                                        data.pop("content", None)
                                        # If file_path is not in data, use document id as file path
                                        if "file_path" not in data:
                                            data["file_path"] = "no-file-path"
                                        # Ensure new fields exist with default values
                                        if "metadata" not in data:
                                            data["metadata"] = {}
                                        if "error_msg" not in data:
                                            data["error_msg"] = None

                                        result[doc_id] = DocProcessingStatus(**data)
                                except (json.JSONDecodeError, KeyError) as e:
                                    logger.error(
                                        f"[{self.workspace}] Error processing document {key}: {e}"
                                    )
                                    continue

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting docs by track_id: {e}")

        return result

    async def index_done_callback(self) -> None:
        """Redis handles persistence automatically"""
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        pattern = f"{self.final_namespace}:*"
        try:
            async with self._get_redis_connection() as redis:
                # Use scan to check if any keys exist
                async for key in redis.scan_iter(match=pattern, count=1):
                    return False  # Found at least one key
                return True  # No keys found
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    @redis_retry
    # ------------------------------------------------------------------
    # Scheduling sidecar helpers (Phase 1) — see class docstring.
    # ------------------------------------------------------------------

    @property
    def _sched_prefix(self) -> str:
        return f"{self.final_namespace}__sched"

    def _zset_key(self, status_value: str) -> str:
        return f"{self._sched_prefix}:status:{status_value}"

    def _basename_key(self, basename: str) -> str:
        return f"{self._sched_prefix}:basename:{basename}"

    @property
    def _ctrl_key(self) -> str:
        return f"{self._sched_prefix}:ctrl"

    @staticmethod
    def _zset_member(row: dict[str, Any], doc_id: str) -> str:
        """``"{created_at}|{doc_id}"`` — ISO-8601 lexicographic order makes
        member order the (created_at, id) keyset order. A missing/non-str
        created_at sorts deterministically first as ""; '|' cannot occur in
        ISO timestamps so the split is unambiguous."""
        created = row.get("created_at")
        return f"{created if isinstance(created, str) else ''}|{doc_id}"

    @staticmethod
    def _split_member(member: str) -> tuple[str, str]:
        created, _, doc_id = member.partition("|")
        return created, doc_id

    @staticmethod
    def _is_duplicate_row(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        metadata = row.get("metadata")
        return bool(isinstance(metadata, dict) and metadata.get("is_duplicate"))

    @classmethod
    def _basename_of(cls, row: Any) -> str | None:
        """Canonical basename this row would claim in the primary index,
        or None when the row is index-ineligible (duplicate marker /
        placeholder file_path)."""
        if not isinstance(row, dict) or cls._is_duplicate_row(row):
            return None
        file_path = row.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return None
        if file_path in ("unknown_source", "no-file-path"):
            return None
        return file_path

    def _queue_index_ops(
        self,
        pipe,
        doc_id: str,
        old_row: dict[str, Any] | None,
        new_row: dict[str, Any] | None,
        *,
        old_basename_owner: str | None,
    ) -> None:
        """Queue sidecar maintenance into an open MULTI for one doc write.

        Status ZSET: remove the old member, add the new one (created_at is
        immutable for existing rows, but removing by the OLD row's member is
        still the correct general form). Basename index: the eligibility
        state machine on (old, new) — the caller supplies the CURRENT index
        owner (read under WATCH) so deletes are CAS-like: only a mapping
        that points to this doc is ever touched.
        """
        if old_row is not None:
            old_status = str(old_row.get("status") or "")
            if old_status:
                pipe.zrem(
                    self._zset_key(old_status), self._zset_member(old_row, doc_id)
                )
        if new_row is not None:
            new_status = str(new_row.get("status") or "")
            if new_status:
                pipe.zadd(
                    self._zset_key(new_status),
                    {self._zset_member(new_row, doc_id): 0},
                )

        old_basename = self._basename_of(old_row)
        new_basename = self._basename_of(new_row)
        if old_basename == new_basename:
            if new_basename is not None:
                # eligible → eligible, same name: assert ownership
                # (idempotent repair of a missing mapping).
                pipe.set(self._basename_key(new_basename), doc_id)
            return
        if old_basename is not None and old_basename_owner == doc_id:
            # eligible → ineligible (e.g. a primary rewritten in place as a
            # post-parse content duplicate) or file_path moved: release only
            # a mapping that still points to us.
            pipe.delete(self._basename_key(old_basename))
        if new_basename is not None:
            # ineligible → eligible, or file_path moved: claim the new name.
            pipe.set(self._basename_key(new_basename), doc_id)

    async def _atomic_doc_write(
        self,
        redis,
        doc_id: str,
        mutate,
    ) -> dict[str, Any] | None:
        """WATCH→MULTI/EXEC skeleton for one doc: read the old row and the
        current basename-index owners under WATCH, let ``mutate(old_row)``
        produce the new row (or ``None`` to abort without writing), then
        commit row + sidecar in one transaction; retry on conflict.

        Returns the committed new row (or None when aborted).
        """
        main_key = f"{self.final_namespace}:{doc_id}"
        for _ in range(self._WATCH_RETRY_LIMIT):
            async with redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(main_key)
                    old_raw = await pipe.get(main_key)
                    old_row = json.loads(old_raw) if old_raw else None
                    new_row = mutate(old_row)
                    if new_row is None:
                        await pipe.unwatch()
                        return None
                    old_basename = self._basename_of(old_row)
                    old_owner = None
                    if old_basename is not None:
                        basename_key = self._basename_key(old_basename)
                        await pipe.watch(basename_key)
                        old_owner = await pipe.get(basename_key)
                    pipe.multi()
                    pipe.set(main_key, json.dumps(new_row))
                    self._queue_index_ops(
                        pipe,
                        doc_id,
                        old_row,
                        new_row,
                        old_basename_owner=old_owner,
                    )
                    await pipe.execute()
                    return new_row
                except WatchError:
                    continue
        raise StorageControlPlaneError(
            f"[{self.workspace}] doc_status write for {doc_id} exceeded the "
            f"WATCH retry budget ({self._WATCH_RETRY_LIMIT})"
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update document status data.

        Each record commits atomically with its scheduling sidecar (status
        ZSET member + basename primary index) via a per-doc WATCH/MULTI
        transaction — a few extra RPCs per doc, bounded by the enqueue
        batch cap; correctness of the sidecar is the priority."""
        if not data:
            return

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        async with self._get_redis_connection() as redis:
            for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
                if "chunks_list" not in doc_data:
                    doc_data["chunks_list"] = []

                def _replace(_old, _new=doc_data):
                    return _new

                await self._atomic_doc_write(redis, doc_id, _replace)
                await _cooperative_yield(i)

    @redis_retry
    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{self.final_namespace}:{id}")
                return json.loads(data) if data else None
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error for id {id}: {e}")
                raise

    async def delete(self, doc_ids: list[str]) -> None:
        """Delete records atomically with their scheduling sidecar entries.

        Per-doc WATCH/MULTI: remove the primary key, its status-ZSET member
        and — only when the basename index still points to THIS doc — the
        basename mapping (deleting a duplicate marker or a non-owning row
        must not strip another row's mapping)."""
        if not doc_ids:
            return

        deleted_count = 0
        async with self._get_redis_connection() as redis:
            for doc_id in doc_ids:
                main_key = f"{self.final_namespace}:{doc_id}"
                for _ in range(self._WATCH_RETRY_LIMIT):
                    async with redis.pipeline(transaction=True) as pipe:
                        try:
                            await pipe.watch(main_key)
                            old_raw = await pipe.get(main_key)
                            if not old_raw:
                                await pipe.unwatch()
                                break
                            old_row = json.loads(old_raw)
                            old_basename = self._basename_of(old_row)
                            old_owner = None
                            if old_basename is not None:
                                basename_key = self._basename_key(old_basename)
                                await pipe.watch(basename_key)
                                old_owner = await pipe.get(basename_key)
                            pipe.multi()
                            pipe.delete(main_key)
                            old_status = str(old_row.get("status") or "")
                            if old_status:
                                pipe.zrem(
                                    self._zset_key(old_status),
                                    self._zset_member(old_row, doc_id),
                                )
                            if old_basename is not None and old_owner == doc_id:
                                pipe.delete(self._basename_key(old_basename))
                            await pipe.execute()
                            deleted_count += 1
                            break
                        except WatchError:
                            continue
                else:
                    raise StorageControlPlaneError(
                        f"[{self.workspace}] doc_status delete for {doc_id} "
                        f"exceeded the WATCH retry budget"
                    )
            logger.info(
                f"[{self.workspace}] Deleted {deleted_count} of {len(doc_ids)} doc status entries from {self.namespace}"
            )

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """
        status_filter_values = self.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )

        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # For Redis, we need to load all data and sort/filter in memory
        all_docs = []
        total_count = 0

        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to iterate through all keys in the namespace
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        # Get all values in batch
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        # Process documents
                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    doc_data = json.loads(value)

                                    # Apply status filter
                                    if (
                                        status_filter_values is not None
                                        and doc_data.get("status")
                                        not in status_filter_values
                                    ):
                                        continue

                                    # Extract document ID from key
                                    doc_id = key.split(":", 1)[1]

                                    # Prepare document data
                                    data = doc_data.copy()
                                    data.pop("content", None)
                                    if "file_path" not in data:
                                        data["file_path"] = "no-file-path"
                                    if "metadata" not in data:
                                        data["metadata"] = {}
                                    if "error_msg" not in data:
                                        data["error_msg"] = None

                                    # Calculate sort key for sorting (but don't add to data)
                                    if sort_field == "id":
                                        sort_key = doc_id
                                    elif sort_field == "file_path":
                                        # Use pinyin sorting for file_path field to support Chinese characters
                                        file_path_value = data.get(sort_field, "")
                                        sort_key = get_pinyin_sort_key(file_path_value)
                                    else:
                                        sort_key = data.get(sort_field, "")

                                    doc_status = DocProcessingStatus(**data)
                                    all_docs.append((doc_id, doc_status, sort_key))

                                except (json.JSONDecodeError, KeyError) as e:
                                    logger.error(
                                        f"[{self.workspace}] Error processing document {key}: {e}"
                                    )
                                    continue

                    if cursor == 0:
                        break

            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting paginated docs: {e}")
                return [], 0

        # Sort documents using the separate sort key
        reverse_sort = sort_direction.lower() == "desc"
        all_docs.sort(key=lambda x: x[2], reverse=reverse_sort)

        # Remove sort key from tuples and keep only (doc_id, doc_status)
        all_docs = [(doc_id, doc_status) for doc_id, doc_status, _ in all_docs]

        total_count = len(all_docs)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = all_docs[start_idx:end_idx]

        return paginated_docs, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        counts = await self.get_status_counts()

        # Add 'all' field with total count
        total_count = sum(counts.values())
        counts["all"] = total_count

        return counts

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_id method
        """
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to iterate through all keys in the namespace
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        # Get all values in batch
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        # Check each document for matching file_path
                        for value in values:
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    if doc_data.get("file_path") == file_path:
                                        return doc_data
                                except json.JSONDecodeError as e:
                                    logger.error(
                                        f"[{self.workspace}] JSON decode error in get_doc_by_file_path: {e}"
                                    )
                                    continue

                    if cursor == 0:
                        break

                return None
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in get_doc_by_file_path: {e}")
                return None

    async def _basename_lookup(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Primary-row basename lookup via the sidecar index (O(1)).

        The index only ever holds PRIMARY rows (``is_duplicate != true``),
        maintained atomically with every write — so a hit is the document,
        never a duplicate marker, and covers custom-ID rows that a
        deterministic ``md5(canonical)`` key lookup would miss. A mapping
        whose primary row is gone is a broken invariant: raise (the strict
        caller must not treat it as confirmed absence).
        """
        if not basename:
            return None
        if basename == "unknown_source":
            return None
        async with self._get_redis_connection() as redis:
            doc_id = await redis.get(self._basename_key(basename))
            if not doc_id:
                return None
            raw = await redis.get(f"{self.final_namespace}:{doc_id}")
            if not raw:
                raise StorageControlPlaneError(
                    f"[{self.workspace}] basename index points at missing doc "
                    f"{doc_id} for '{basename}' — sidecar invariant broken"
                )
            return doc_id, json.loads(raw)

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find the PRIMARY record whose canonical basename matches.

        The caller is responsible for passing an already-canonical basename.
        Stored ``file_path`` values are canonicalized by the business layer, so
        this lookup intentionally performs an exact match only. Duplicate
        marker rows (``metadata.is_duplicate``) are never returned — see the
        class docstring's basename-index contract.

        Legacy error semantics preserved: failures log and return ``None``
        (best-effort miss). Identity-critical callers use
        :meth:`get_doc_by_file_basename_strict` instead.
        """
        try:
            return await self._basename_lookup(basename)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_doc_by_file_basename: {e}")
            return None

    async def get_doc_by_file_basename_strict(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Fail-closed variant: ``None`` only after a healthy index miss;
        transport errors and broken index invariants propagate (base
        contract — a swallowed failure would mint duplicate rows)."""
        return await self._basename_lookup(basename)

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose content_hash field matches."""
        if not content_hash:
            return None

        async with self._get_redis_connection() as redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        for key, value in zip(keys, values):
                            if not value:
                                continue
                            try:
                                doc_data = json.loads(value)
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"[{self.workspace}] JSON decode error in get_doc_by_content_hash: {e}"
                                )
                                continue
                            if doc_data.get("content_hash") == content_hash:
                                doc_id = key.split(":", 1)[1]
                                return doc_id, doc_data

                    if cursor == 0:
                        break

                return None
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error in get_doc_by_content_hash: {e}"
                )
                return None

    # ------------------------------------------------------------------
    # Memory-bounding scheduling API (Phase 1)
    # ------------------------------------------------------------------

    async def get_by_id_strict(self, id: str) -> Union[dict[str, Any], None]:
        """Strict point read: ``get_by_id`` already propagates transport and
        decode failures, so a ``None`` is a confirmed absence."""
        return await self.get_by_id(id)

    @staticmethod
    def _encode_cursor(positions: dict[str, str]) -> str:
        return json.dumps(positions, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _decode_cursor(opaque: str) -> dict[str, str]:
        try:
            decoded = json.loads(opaque)
            if not isinstance(decoded, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in decoded.items()
            ):
                raise ValueError("cursor must map status -> last member")
        except (ValueError, TypeError) as e:
            raise StorageControlPlaneError(
                f"Malformed scheduling cursor for RedisDocStatusStorage: {e}"
            ) from e
        return decoded

    def _scheduling_record_from_row(
        self, doc_id: str, row: dict[str, Any], *, strict: bool
    ) -> DocSchedulingRecord | None:
        try:
            status = DocStatus(str(row["status"]))
            created_at = row["created_at"]
            updated_at = row.get("updated_at", created_at)
            if not isinstance(created_at, str) or not isinstance(updated_at, str):
                raise TypeError("created_at/updated_at must be strings")
            metadata = row.get("metadata")
            return DocSchedulingRecord(
                id=doc_id,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                file_path=row.get("file_path") or "no-file-path",
                track_id=row.get("track_id"),
                has_custom_chunk_journal=isinstance(metadata, dict)
                and isinstance(metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"[{self.workspace}] Unusable scheduling row {doc_id}: {e}")
            if strict:
                raise
            return None

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        max_failure_generation: int | None = None,
        strict: bool = False,
    ) -> DocStatusPage:
        """Bounded k-way merge over the per-status ZSETs.

        Each status stream is read with ``ZRANGEBYLEX (last_member +`` —
        lexicographic member order IS the (created_at, id) keyset order.
        The composite cursor records each status's own last CONSUMED member:
        a candidate merged into the page window is consumed (returned,
        dropped by the generation predicate, or skipped as unusable in
        relaxed mode) and advances its stream; a prefetched head that did
        not fit stays unconsumed and is re-read next page. A stream is
        exhausted when its prefetch came back short AND fully consumed;
        the sweep ends when every stream is exhausted.
        """
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        if isinstance(position, CursorAfter):
            positions = self._decode_cursor(position.opaque)
        else:
            positions = {}
        status_values = [s.value for s in statuses]
        failed_value = DocStatus.FAILED.value

        async with self._get_redis_connection() as redis:
            # Prefetch up to `limit` members per stream, strictly after each
            # stream's own consumed position.
            pipe = redis.pipeline()
            for value in status_values:
                last = positions.get(value)
                lex_min = f"({last}" if last else "-"
                pipe.zrangebylex(self._zset_key(value), lex_min, "+", 0, limit)
            prefetched: list[list[str]] = await pipe.execute()

            streams: dict[str, list[str]] = dict(zip(status_values, prefetched))
            short_streams = {
                value for value, members in streams.items() if len(members) < limit
            }

            # Merge candidates globally by member (== (created_at, id) key),
            # keep the first `limit` as this page's consumption window.
            merged: list[tuple[str, str]] = []  # (member, status_value)
            for value, members in streams.items():
                merged.extend((member, value) for member in members)
            merged.sort(key=lambda t: t[0])
            window = merged[:limit]

            # Hydrate the window's primary rows in one pipeline.
            doc_ids = [self._split_member(member)[1] for member, _ in window]
            rows: list[str | None] = []
            if doc_ids:
                pipe = redis.pipeline()
                for doc_id in doc_ids:
                    pipe.get(f"{self.final_namespace}:{doc_id}")
                rows = await pipe.execute()

        docs: dict[str, DocSchedulingRecord] = {}
        consumed_counts: dict[str, int] = {value: 0 for value in status_values}
        new_positions = dict(positions)
        for (member, value), raw in zip(window, rows):
            _, doc_id = self._split_member(member)
            consumed_counts[value] += 1
            new_positions[value] = member
            if raw is None:
                message = (
                    f"[{self.workspace}] status ZSET member {member!r} has no "
                    f"primary row — sidecar invariant broken"
                )
                if strict:
                    raise StorageControlPlaneError(message)
                logger.error(message)
                continue  # consumed; the stale member self-heals on rewrite
            row = json.loads(raw)
            if max_failure_generation is not None and value == failed_value:
                try:
                    generation = int(row.get("failure_generation") or 0)
                except (TypeError, ValueError):
                    generation = 0
                if generation > max_failure_generation:
                    continue  # consumed by the cohort predicate
            record = self._scheduling_record_from_row(doc_id, row, strict=strict)
            if record is None:
                continue  # relaxed skip is still consumed
            docs[doc_id] = record

        exhausted = all(
            value in short_streams and consumed_counts[value] == len(streams[value])
            for value in status_values
        )
        if exhausted:
            return DocStatusPage(docs=docs, next_position=CURSOR_END)
        return DocStatusPage(
            docs=docs,
            next_position=CursorAfter(self._encode_cursor(new_positions)),
        )

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        """O(1) fail-closed count: sum of per-status ZCARDs (errors
        propagate — admission control must never read a failure as zero)."""
        if not statuses:
            return 0
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for status in statuses:
                pipe.zcard(self._zset_key(status.value))
            cards = await pipe.execute()
        return int(sum(cards))

    async def update_doc_status_fields(
        self,
        doc_id: str,
        fields: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        """Targeted field update, atomic with the sidecar (WATCH/MULTI)."""
        if "created_at" in fields:
            raise ValueError(
                "created_at is an immutable scheduling sort key and cannot "
                "be changed via update_doc_status_fields"
            )
        missing = False

        def _merge(old_row):
            nonlocal missing
            if old_row is None:
                missing = True
                return None
            return {**old_row, **fields}

        async with self._get_redis_connection() as redis:
            await self._atomic_doc_write(redis, doc_id, _merge)
        if missing and not missing_ok:
            raise StorageRecordNotFoundError(doc_id)

    # ------------------------------------------------------------------
    # failure_generation write side (Phase 1)
    # ------------------------------------------------------------------

    async def get_failure_generation_mode(self) -> FailureGenerationMode:
        """Read the per-workspace marker; transport errors propagate and a
        missing/foreign marker reads MIGRATING — never LEGACY."""
        async with self._get_redis_connection() as redis:
            ctrl = await redis.hgetall(self._ctrl_key)
        if not ctrl:
            return FailureGenerationMode.MIGRATING
        if ctrl.get("schema_version") != self._CTRL_SCHEMA_VERSION:
            return FailureGenerationMode.MIGRATING
        try:
            return FailureGenerationMode(str(ctrl.get("mode")))
        except ValueError:
            return FailureGenerationMode.MIGRATING

    async def reserve_failure_generation(self) -> int:
        """Atomic counter reservation (WATCH on the ctrl hash so the marker
        validation and the increment commit together)."""
        async with self._get_redis_connection() as redis:
            for _ in range(self._WATCH_RETRY_LIMIT):
                async with redis.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(self._ctrl_key)
                        ctrl = await pipe.hgetall(self._ctrl_key)
                        self._validate_ctrl(ctrl)
                        counter = int(ctrl.get("failure_generation_counter") or 0) + 1
                        pipe.multi()
                        pipe.hset(self._ctrl_key, "failure_generation_counter", counter)
                        await pipe.execute()
                        return counter
                    except WatchError:
                        continue
        raise StorageControlPlaneError(
            f"[{self.workspace}] failure-generation reservation exceeded the "
            f"WATCH retry budget"
        )

    def _validate_ctrl(self, ctrl: dict[str, str]) -> None:
        if not ctrl or ctrl.get("schema_version") != self._CTRL_SCHEMA_VERSION:
            raise StorageControlPlaneError(
                f"[{self.workspace}] failure-generation marker missing or "
                f"version-mismatched for {self.namespace}; refusing (never "
                "degrades to LEGACY full-snapshot behaviour)"
            )

    async def mark_doc_failed(self, doc_id: str, fields: dict[str, Any]) -> int | None:
        """FAILED transition funnel: reserve + publish in ONE transaction.

        WATCHes the doc key AND the ctrl hash, reads the counter, then
        commits ``counter+1`` and the FAILED row together — a concurrent
        reservation bumps the ctrl hash and retries this transaction, so the
        reserve→publish window is zero. Idempotent per attempt; existing
        created_at preserved; missing rows conditionally created.
        """
        main_key = f"{self.final_namespace}:{doc_id}"
        async with self._get_redis_connection() as redis:
            for _ in range(self._WATCH_RETRY_LIMIT):
                async with redis.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(main_key, self._ctrl_key)
                        old_raw = await pipe.get(main_key)
                        old_row = json.loads(old_raw) if old_raw else None
                        if isinstance(old_row, dict):
                            current_attempt = old_row.get(
                                "processing_attempt_id"
                            ) or fields.get("processing_attempt_id")
                        else:
                            current_attempt = fields.get("processing_attempt_id")
                        if (
                            isinstance(old_row, dict)
                            and str(old_row.get("status")) == DocStatus.FAILED.value
                            and current_attempt
                            and old_row.get("failure_attempt_id") == current_attempt
                        ):
                            await pipe.unwatch()
                            try:
                                return int(old_row.get("failure_generation") or 0)
                            except (TypeError, ValueError):
                                return 0
                        ctrl = await pipe.hgetall(self._ctrl_key)
                        self._validate_ctrl(ctrl)
                        generation = (
                            int(ctrl.get("failure_generation_counter") or 0) + 1
                        )
                        new_row = (
                            {**old_row, **fields}
                            if isinstance(old_row, dict)
                            else dict(fields)
                        )
                        if isinstance(old_row, dict) and "created_at" in old_row:
                            new_row["created_at"] = old_row["created_at"]
                        new_row["status"] = DocStatus.FAILED.value
                        new_row["failure_generation"] = generation
                        if current_attempt:
                            new_row["failure_attempt_id"] = current_attempt
                            new_row.setdefault("processing_attempt_id", current_attempt)
                        if "chunks_list" not in new_row:
                            new_row["chunks_list"] = []
                        old_basename = self._basename_of(old_row)
                        old_owner = None
                        if old_basename is not None:
                            basename_key = self._basename_key(old_basename)
                            await pipe.watch(basename_key)
                            old_owner = await pipe.get(basename_key)
                        pipe.multi()
                        pipe.hset(
                            self._ctrl_key, "failure_generation_counter", generation
                        )
                        pipe.set(main_key, json.dumps(new_row))
                        self._queue_index_ops(
                            pipe,
                            doc_id,
                            old_row,
                            new_row,
                            old_basename_owner=old_owner,
                        )
                        await pipe.execute()
                        return generation
                    except WatchError:
                        continue
        raise StorageControlPlaneError(
            f"[{self.workspace}] mark_doc_failed for {doc_id} exceeded the "
            f"WATCH retry budget"
        )

    async def ensure_processing_attempt_id(self, doc_id: str) -> str:
        """Atomic mint-or-reuse of the row's attempt id (WATCH/MULTI)."""
        result: dict[str, str | None] = {"attempt": None}

        def _ensure(old_row):
            if old_row is None:
                return None
            attempt = old_row.get("processing_attempt_id")
            if attempt:
                result["attempt"] = str(attempt)
                return None  # nothing to write
            result["attempt"] = uuid.uuid4().hex
            return {**old_row, "processing_attempt_id": result["attempt"]}

        async with self._get_redis_connection() as redis:
            await self._atomic_doc_write(redis, doc_id, _ensure)
        if result["attempt"] is None:
            raise StorageRecordNotFoundError(doc_id)
        return result["attempt"]

    # ------------------------------------------------------------------
    # Sidecar migration (Phase 1)
    # ------------------------------------------------------------------

    async def _migrate_scheduling_sidecar(self) -> None:
        """One-time streaming sidecar build for pre-existing deployments.

        SCAN the primary rows in bounded batches, populate the status ZSETs
        and the basename primary index, calibrate the counter to
        ``max(persisted failure_generation)``, then publish the ctrl marker
        LAST (atomic ENFORCED activation). A short-lease migration lock
        prevents duplicate migrations among concurrently starting workers;
        losers wait (bounded) for the marker. Old writers are NOT isolated
        by this lock — the coordinated stop-write upgrade is a deployment
        prerequisite (class docstring).
        """
        lock_key = f"{self._sched_prefix}:migrate_lock"
        async with self._get_redis_connection() as redis:
            ctrl = await redis.hgetall(self._ctrl_key)
            if ctrl:
                return  # marker present (any version): nothing to bootstrap
            got_lock = await redis.set(lock_key, "1", nx=True, ex=300)
            if not got_lock:
                # Another worker is migrating: wait (bounded) for the marker.
                for _ in range(60):
                    await asyncio.sleep(1)
                    if await redis.hgetall(self._ctrl_key):
                        return
                raise StorageMigrationInProgressError(
                    f"[{self.workspace}] scheduling sidecar migration by "
                    f"another worker did not complete in time"
                )
            try:
                # Clear any half-built sidecar from a crashed prior attempt
                # (marker absent ⇒ nothing published yet, rebuild is safe).
                for pattern in (
                    f"{self._sched_prefix}:status:*",
                    f"{self._sched_prefix}:basename:*",
                ):
                    cursor = 0
                    while True:
                        cursor, keys = await redis.scan(
                            cursor, match=pattern, count=1000
                        )
                        if keys:
                            await redis.delete(*keys)
                        if cursor == 0:
                            break

                max_generation = 0
                migrated = 0
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()
                        write_pipe = redis.pipeline()
                        for key, raw in zip(keys, values):
                            if not raw:
                                continue
                            try:
                                row = json.loads(raw)
                            except json.JSONDecodeError:
                                logger.error(
                                    f"[{self.workspace}] skipping undecodable "
                                    f"row {key} during sidecar migration"
                                )
                                continue
                            if not isinstance(row, dict):
                                continue
                            doc_id = key.split(":", 1)[1]
                            status = str(row.get("status") or "")
                            if status:
                                write_pipe.zadd(
                                    self._zset_key(status),
                                    {self._zset_member(row, doc_id): 0},
                                )
                            basename = self._basename_of(row)
                            if basename is not None:
                                write_pipe.set(self._basename_key(basename), doc_id)
                            try:
                                generation = int(row.get("failure_generation") or 0)
                            except (TypeError, ValueError):
                                generation = 0
                            max_generation = max(max_generation, generation)
                            migrated += 1
                        await write_pipe.execute()
                    if cursor == 0:
                        break

                # Publish the marker LAST: readers treat its absence as
                # MIGRATING, so a crash before this line re-runs cleanly.
                await redis.hset(
                    self._ctrl_key,
                    mapping={
                        "schema_version": self._CTRL_SCHEMA_VERSION,
                        "mode": FailureGenerationMode.ENFORCED.value,
                        "failure_generation_counter": max_generation,
                    },
                )
                logger.info(
                    f"[{self.workspace}] scheduling sidecar migrated "
                    f"({migrated} rows, counter={max_generation}) for "
                    f"{self.namespace}"
                )
            finally:
                await redis.delete(lock_key)

    async def drop(self) -> dict[str, str]:
        """Drop all document status data from storage and clean up resources.

        Also clears the scheduling sidecar's status ZSETs and basename index
        (they mirror the dropped rows). The ctrl hash is KEPT: the
        failure-generation counter must stay monotonic across a workspace
        clear (holes are allowed, reuse is not)."""
        try:
            async with self._get_redis_connection() as redis:
                deleted_count = 0
                for pattern in (
                    f"{self.final_namespace}:*",
                    f"{self._sched_prefix}:status:*",
                    f"{self._sched_prefix}:basename:*",
                ):
                    cursor = 0
                    while True:
                        cursor, keys = await redis.scan(
                            cursor, match=pattern, count=1000
                        )
                        if keys:
                            # Delete keys in batches
                            pipe = redis.pipeline()
                            for key in keys:
                                pipe.delete(key)
                            results = await pipe.execute()
                            deleted_count += sum(results)

                        if cursor == 0:
                            break

                logger.info(
                    f"[{self.workspace}] Dropped {deleted_count} doc status keys from {self.namespace}"
                )
                return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping doc status {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
