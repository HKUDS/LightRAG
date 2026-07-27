import os
import logging
import asyncio
import time
import hashlib
from typing import Any, ClassVar, final, Sequence, Union
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
    ResponseError,
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
    SourceAbsent,
    SourceConflict,
    SourceConflictPage,
    SourceConflictRepairResult,
    SourceConflictSummary,
    SourceResolution,
    SourceUnique,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
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

# One eviction-policy check per Redis URL per process: the answer is a server
# property, not a per-storage one, and both storage classes share the pool.
_eviction_checked: set[str] = set()


async def _ensure_no_eviction_policy(redis, redis_url: str, workspace: str) -> None:
    """Refuse to use an evicting Redis as a system of record.

    ``doc_status`` rows, ``full_docs`` entries and the derived scheduling sidecar
    (status ZSETs + basename SETs) are all plain keys with NO TTL. Under an
    ``allkeys-*`` maxmemory policy Redis may evict any of them, and that loss is
    silent AND undetectable: an evicted status ZSET drops every document in that
    status out of the sweep, an evicted primary row loses the document outright,
    and the sidecar's structural probe only asks whether ANY status key exists —
    so a partially evicted index looks perfectly healthy and gets no rebuild.
    There is no cheap invariant that would catch it either (verifying the sidecar
    against the rows is an O(total_docs) scan), which is exactly why this is a
    startup precondition rather than a runtime repair.

    Safe configurations, allowed silently:

    * ``maxmemory=0`` — no limit, so nothing is ever evicted;
    * ``volatile-*`` / ``noeviction`` — only keys carrying a TTL can be evicted
      and none of ours do (just the short-lived rebuild lease), so a full
      instance fails writes LOUDLY instead of quietly dropping data.

    Managed Redis frequently blocks ``CONFIG GET``. An unreadable config warns
    rather than fails: refusing to start over a policy we cannot even observe
    would break working deployments. ``REDIS_ALLOW_EVICTION_POLICY=true`` is the
    explicit override for an operator who accepts the data loss.
    """
    if redis_url in _eviction_checked:
        return
    try:
        settings = await redis.config_get("maxmemory*")
    except Exception as config_error:
        # Unknown command, NOPERM, or a proxy that does not implement CONFIG.
        _eviction_checked.add(redis_url)
        logger.warning(
            f"[{workspace}] Could not read the Redis maxmemory policy "
            f"({config_error}); cannot verify that this instance will not EVICT "
            f"doc_status/full_docs keys. Ensure maxmemory-policy is 'noeviction' "
            f"or a 'volatile-*' policy — an 'allkeys-*' policy silently drops "
            f"documents and scheduling state."
        )
        return
    _eviction_checked.add(redis_url)
    policy = str((settings or {}).get("maxmemory-policy") or "").strip().lower()
    raw_maxmemory = str((settings or {}).get("maxmemory") or "0").strip()
    try:
        maxmemory = int(raw_maxmemory)
    except ValueError:
        maxmemory = 0
    if maxmemory <= 0 or not policy.startswith("allkeys"):
        return
    if os.getenv("REDIS_ALLOW_EVICTION_POLICY", "false").strip().lower() == "true":
        logger.warning(
            f"[{workspace}] Redis is configured to EVICT (maxmemory-policy="
            f"'{policy}', maxmemory={maxmemory}) and REDIS_ALLOW_EVICTION_POLICY "
            f"is set: doc_status rows, full_docs entries and the scheduling "
            f"sidecar may be dropped silently, losing documents."
        )
        return
    raise StorageControlPlaneError(
        f"[{workspace}] Redis at {redis_url} is configured as an evicting cache "
        f"(maxmemory-policy='{policy}', maxmemory={maxmemory}), but LightRAG "
        f"stores doc_status, full_docs and the derived scheduling index there as "
        f"a system of record — none of those keys carry a TTL, so an "
        f"'allkeys-*' policy can drop documents and scheduling state with no "
        f"error and no way to detect it. Set maxmemory-policy to 'noeviction' "
        f"(or a 'volatile-*' policy), give LightRAG its own Redis instance/db, "
        f"or set REDIS_ALLOW_EVICTION_POLICY=true to accept the data loss."
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
                    await _ensure_no_eviction_policy(
                        redis, self._redis_url, self.workspace
                    )
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
    * ``…__sched:basename:{canonical}`` — canonical basename → SET of
      eligible PRIMARY doc IDs (rows with ``metadata.is_duplicate != true``
      and a real file_path). ``file_path`` is NOT assumed globally unique
      (custom-id inserts, legacy ids and historical basename collisions can
      all share one basename), so the index is a multimap: each write does a
      member-level ``SADD``/``SREM`` of its OWN doc_id keyed on the (old,
      new) eligibility of that row — never a single-value CAS. A primary
      rewritten in place as a content-duplicate removes only itself; a
      genuine collision leaves ≥2 members and is surfaced by the typed
      resolver as a ``SourceConflict``.

    The sidecar is a DERIVED structure, not a migration source of truth and
    NOT gated by any persistent semantic marker. ``initialize`` rebuilds it
    from the actual doc_status rows when it is absent (fresh bootstrap of a
    pre-existing deployment, or recovery after the backend was replaced /
    the derived keys were lost): it streams the primary rows into a
    TEMPORARY keyspace (``…__sched_rebuild:*``) and then switches it into
    place by ``RENAME``-ing the temp keys over the official ones in bounded
    batches (see :meth:`_publish_rebuilt_index` — not one atomic
    ``MULTI/EXEC``, deliberately). A crash before the switch leaves the old
    official index untouched (still serving) and the unpublished temp
    keyspace safe to discard on the next attempt; a crash during the switch
    leaves a mix of renamed and un-renamed keys, detected on the next
    startup by the same leftover temp keyspace (see
    :meth:`_rebuild_scheduling_sidecar`). Once present, the index is kept
    consistent by every write and never rebuilt, so a running system's
    restarts do not clobber a live index. Exclusivity across workers comes
    from :func:`get_data_init_lock` (cross-process, server-instance-wide,
    held for all of ``initialize()``) rather than a Redis-side lock — see
    :meth:`_rebuild_scheduling_sidecar` for why that is sufficient as long
    as LightRAG does not support multiple server instances sharing one
    Redis. NOTE: an online rebuild over a workspace under concurrent write
    traffic still requires a maintenance freeze (deployment prerequisite) —
    ``get_data_init_lock`` only serializes ``initialize()`` calls, it does
    not isolate unrelated live writers already past their own initialize.
    """

    supports_strict_point_reads: ClassVar[bool] = True

    # Bounded upper limit on the sample of conflicting doc IDs surfaced by the
    # source-conflict listing/repair APIs — never materialize the whole set.
    _CONFLICT_SAMPLE_CAP: ClassVar[int] = 32
    # SCAN batch for the conflict listing: the unit of work a page never
    # abandons half-examined, so it also caps how far a page may overshoot
    # ``limit`` (see list_source_conflicts_page).
    _CONFLICT_SCAN_BATCH: ClassVar[int] = 64
    # Bounded retry budget for WATCH/MULTI conflict loops. High contention
    # on a single doc key is not expected (per-doc writes are serialized by
    # the pipeline); the cap turns a pathological livelock into an error.
    _WATCH_RETRY_LIMIT: ClassVar[int] = 50
    # Key names per batch when the rebuilt sidecar is switched into place. This
    # is the ONLY thing bounding that switch: it caps both the Python list of key
    # names and the commands queued in one round trip, so a rebuild costs
    # O(batch) instead of O(total_docs) (see _publish_rebuilt_index).
    _PUBLISH_BATCH: ClassVar[int] = 512

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
                    await _ensure_no_eviction_policy(
                        redis, self._redis_url, self.workspace
                    )
                    logger.info(
                        f"[{self.workspace}] Connected to Redis for doc status namespace {self.namespace}"
                    )
                # Rebuild the derived scheduling sidecar from the actual
                # doc_status rows when it is absent (no persistent marker) —
                # must complete before serving: the paged sweep, ZCARD counts
                # and the basename source index all depend on it.
                await self._rebuild_scheduling_sidecar()
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

    @staticmethod
    def _redis_doc_processing_status_from_data(
        data: dict[str, Any],
    ) -> DocProcessingStatus:
        """Normalise a decoded doc_status row into a FULL DocProcessingStatus.

        Single source of the decoded-JSON -> status construction shared by
        ``get_docs_by_statuses`` and the ``get_full_docs_by_ids`` hydration
        path. Raises ``KeyError``/``TypeError`` on a malformed row (``TypeError``
        is what ``DocProcessingStatus(**data)`` raises on missing required
        fields); the caller decides strict (raise) vs relaxed (skip).
        """
        data = data.copy()
        data.pop("content", None)
        if "file_path" not in data:
            data["file_path"] = "no-file-path"
        if "metadata" not in data:
            data["metadata"] = {}
        if "error_msg" not in data:
            data["error_msg"] = None
        return DocProcessingStatus(**data)

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
                                result[doc_id] = (
                                    self._redis_doc_processing_status_from_data(
                                        doc_data
                                    )
                                )
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

    def _require_sidecar_ready(self) -> None:
        """Guard for strict page/count/resolver reads: refuse to serve while
        the derived index is not yet published, so a not-ready index is never
        misread as "empty" (which scan/enqueue would treat as confirmed-new
        and mint duplicates)."""
        if not self._initialized:
            raise StorageControlPlaneError(
                f"[{self.workspace}] scheduling sidecar not ready for "
                f"{self.namespace}; refusing to read a rebuilding / "
                "uninitialized index as empty"
            )

    @staticmethod
    def _zset_member(row: dict[str, Any], doc_id: str) -> str:
        """``"{created_at}|{doc_id}"`` — ISO-8601 lexicographic order makes
        member order the (created_at, id) keyset order. ``'|'`` cannot occur in
        ISO timestamps so the split is unambiguous.

        A missing/non-str created_at degrades to ``""``, which puts the member
        LAST, not first: ``'|'`` (0x7C) sorts after every digit an ISO year can
        start with. That ordering is not load-bearing — such a row cannot build a
        ``DocSchedulingRecord`` at all, so a strict page raises on it and a
        relaxed page consumes-and-skips it (pinned for all five backends in
        tests/kg/test_doc_status_scheduling_contract.py). Do not "fix" the
        position by changing this encoding: the ZSET is only rebuilt when absent,
        so a live index would end up holding both encodings and a rewrite would
        fail to ZREM the stale member, listing the row twice."""
        created = row.get("created_at")
        return f"{created if isinstance(created, str) else ''}|{doc_id}"

    @staticmethod
    def _split_member(member: str) -> tuple[str, str]:
        created, _, doc_id = member.partition("|")
        return created, doc_id

    @staticmethod
    def _status_value(value: Any) -> str:
        """Plain status text for a sidecar ZSET key, from either a
        persisted row (plain ``str``) or a caller-supplied ``DocStatus``.

        ``DocStatus`` mixes in ``str``, so ``json.dumps`` happily encodes it
        as its bare value ("pending") — but ``str(DocStatus.PENDING)`` goes
        through ``Enum.__str__`` and yields ``"DocStatus.PENDING"`` instead.
        Calling plain ``str()`` on a live enum member (as opposed to a value
        already round-tripped through JSON) silently files the sidecar entry
        under the wrong key, so every write after that member is added is
        invisible to ``get_docs_by_statuses_page`` — do not inline this."""
        if isinstance(value, DocStatus):
            return value.value
        return str(value) if value is not None else ""

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
    ) -> None:
        """Queue sidecar maintenance into an open MULTI for one doc write.

        Status ZSET: remove the old member, add the new one (created_at is
        immutable for existing rows, but removing by the OLD row's member is
        still the correct general form). Basename source multimap: the
        eligibility state machine on (old, new) is applied at MEMBER level —
        this doc only ever ``SADD``s / ``SREM``s its OWN doc_id, so it never
        touches another doc's membership and needs no single-value CAS. This
        makes the whole write atomic under a WATCH on the doc key alone.
        """
        if old_row is not None:
            old_status = self._status_value(old_row.get("status"))
            if old_status:
                pipe.zrem(
                    self._zset_key(old_status), self._zset_member(old_row, doc_id)
                )
        if new_row is not None:
            new_status = self._status_value(new_row.get("status"))
            if new_status:
                pipe.zadd(
                    self._zset_key(new_status),
                    {self._zset_member(new_row, doc_id): 0},
                )

        old_basename = self._basename_of(old_row)
        new_basename = self._basename_of(new_row)
        if old_basename == new_basename:
            if new_basename is not None:
                # eligible → eligible, same name: assert own membership
                # (idempotent repair of a missing member).
                pipe.sadd(self._basename_key(new_basename), doc_id)
            return
        if old_basename is not None:
            # eligible → ineligible (e.g. a primary rewritten in place as a
            # post-parse content duplicate) or file_path moved: drop only our
            # own membership from the old source set.
            pipe.srem(self._basename_key(old_basename), doc_id)
        if new_basename is not None:
            # ineligible → eligible, or file_path moved: join the new source set.
            pipe.sadd(self._basename_key(new_basename), doc_id)

    async def _atomic_doc_write(
        self,
        redis,
        doc_id: str,
        mutate,
    ) -> dict[str, Any] | None:
        """WATCH→MULTI/EXEC skeleton for one doc: read the old row under a
        WATCH on the doc key alone, let ``mutate(old_row)`` produce the new
        row (or ``None`` to abort without writing), then commit row + sidecar
        in one transaction; retry on conflict.

        The basename source multimap needs no WATCH: ``_queue_index_ops`` only
        SADD/SREMs this doc's OWN membership, which commutes with concurrent
        writes to other docs sharing the same source set.

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
                    pipe.multi()
                    pipe.set(main_key, json.dumps(new_row))
                    self._queue_index_ops(pipe, doc_id, old_row, new_row)
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

        Per-doc WATCH/MULTI on the doc key: remove the primary key, its
        status-ZSET member and — member-level — this doc's own entry from its
        source set (``SREM`` never strips another doc's membership, so
        deleting a duplicate marker or a non-owning row is safe)."""
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
                            pipe.multi()
                            pipe.delete(main_key)
                            old_status = self._status_value(old_row.get("status"))
                            if old_status:
                                pipe.zrem(
                                    self._zset_key(old_status),
                                    self._zset_member(old_row, doc_id),
                                )
                            if old_basename is not None:
                                pipe.srem(self._basename_key(old_basename), doc_id)
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

    async def _scan_source_candidates(
        self, redis, canonical: str, *, limit: int | None
    ) -> list[tuple[str, dict[str, Any]]]:
        """Incrementally SSCAN the source multimap for ``canonical`` and return
        up to ``limit`` validated PRIMARY candidates (all when ``limit`` is
        ``None``), each as ``(doc_id, raw_row)``.

        Every member is hydrated from its primary key and re-validated as a
        current primary (``_basename_of(row) == canonical``); a member whose
        row is gone or no longer eligible is stale and self-healed with a
        member-level ``SREM``. Transport / decode errors propagate — a strict
        caller must never read a failure as absence. Reading incrementally
        (not ``SMEMBERS``) with an early ``limit`` lets the resolver decide
        Conflict after two valid candidates without materializing a large set.
        """
        set_key = self._basename_key(canonical)
        candidates: list[tuple[str, dict[str, Any]]] = []
        cursor = 0
        while True:
            cursor, members = await redis.sscan(set_key, cursor, count=256)
            for doc_id in members:
                raw = await redis.get(f"{self.final_namespace}:{doc_id}")
                if not raw:
                    # stale member: primary row gone — self-heal.
                    await redis.srem(set_key, doc_id)
                    continue
                row = json.loads(raw)
                if self._basename_of(row) != canonical:
                    # no longer an eligible primary for this key (demoted to
                    # duplicate / file_path moved): self-heal.
                    await redis.srem(set_key, doc_id)
                    continue
                candidates.append((doc_id, row))
                if limit is not None and len(candidates) >= limit:
                    return candidates
            if cursor == 0:
                break
        return candidates

    async def _valid_primary_ids(self, redis, canonical: str) -> list[str]:
        """Full sorted list of validated primary doc IDs for ``canonical``,
        self-healing stale members. Used by the operator conflict listing /
        repair (one source key at a time, bounded by that key's cardinality)."""
        candidates = await self._scan_source_candidates(redis, canonical, limit=None)
        return sorted({doc_id for doc_id, _ in candidates})

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Legacy best-effort primary-row lookup by canonical basename.

        The caller passes an already-canonical basename; stored ``file_path``
        values are canonicalized by the business layer, so this is an exact
        match. Duplicate marker rows are never returned. ``file_path`` is not
        globally unique, so on a historical collision this returns SOME primary
        (first validated member) — identity-critical callers must use
        :meth:`resolve_doc_source_strict` to detect the conflict instead.

        Legacy error semantics preserved: failures log and return ``None``
        (best-effort miss)."""
        if not basename or basename == "unknown_source":
            return None
        try:
            async with self._get_redis_connection() as redis:
                candidates = await self._scan_source_candidates(
                    redis, basename, limit=1
                )
            return candidates[0] if candidates else None
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_doc_by_file_basename: {e}")
            return None

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:
        """Typed, conflict-aware source resolution (see base contract).

        Reads the source multimap incrementally, stopping once two valid
        primary candidates are found: 0 → :class:`SourceAbsent`, 1 →
        :class:`SourceUnique`, ≥2 → :class:`SourceConflict`. Transport errors,
        broken invariants and a not-ready index all raise a control-plane
        error rather than returning ``SourceAbsent`` (which scan/enqueue would
        treat as confirmed-new). ``candidate_count`` is ``None`` — the resolver
        only proves "at least two" without enumerating the whole set."""
        self._require_sidecar_ready()
        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()
        async with self._get_redis_connection() as redis:
            candidates = await self._scan_source_candidates(
                redis, canonical_source_key, limit=2
            )
        if not candidates:
            return SourceAbsent()
        if len(candidates) == 1:
            doc_id, row = candidates[0]
            return SourceUnique(
                doc_id=doc_id,
                doc=self._scheduling_record_from_row(doc_id, row, strict=True),
            )
        return SourceConflict(
            candidate_count=None,
            sample_doc_ids=tuple(sorted(doc_id for doc_id, _ in candidates)),
        )

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> Union[tuple[str, dict[str, Any]], None]:
        """Find an existing record whose content_hash field matches.

        ``exclude_doc_id`` skips that row AND any row that merely points at it
        (``is_duplicate`` naming it as ``original_doc_id``), so the duplicate
        check can neither be answered with the row being processed nor with a
        record of that row (see base contract). Redis has no content_hash index,
        so this SCANs the keyspace either way — both exclusions are predicates
        in the same pass, which is also why skipping a pointer row cannot
        truncate the search: the scan keeps looking for the earliest remaining
        holder.

        Fail-closed: transport and decode failures PROPAGATE. ``None`` means
        "confirmed no other holder", and the dedup callers act on it
        destructively (enqueue the document, ingest its content), so reporting
        a failure as "no duplicate" would mint duplicate rows and duplicate
        graph contributions. A row that cannot be decoded is not skipped
        either: it might be the holder, so its content_hash is unknown, not
        absent.

        Returns the EARLIEST match by ``(created_at, id)`` rather than the
        first one SCAN happens to reach — SCAN order is arbitrary, so returning
        on first hit would make the ``original_doc_id`` recorded on a duplicate
        vary between runs over the same data (base contract). That costs the
        lucky early exit, not an extra pass: the scan was already unindexed and
        whole-keyspace in the miss case, and only one candidate is ever held.
        """
        if not content_hash:
            return None

        best: tuple[tuple[str, str], str, dict[str, Any]] | None = None
        async with self._get_redis_connection() as redis:
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
                        doc_id = key.split(":", 1)[1]
                        if doc_id == exclude_doc_id:
                            continue
                        try:
                            doc_data = json.loads(value)
                        except json.JSONDecodeError as e:
                            raise StorageControlPlaneError(
                                f"[{self.workspace}] undecodable doc_status row "
                                f"{key} while checking content_hash "
                                f"'{content_hash}': cannot confirm whether it "
                                f"holds the hash"
                            ) from e
                        if doc_data.get("content_hash") != content_hash:
                            continue
                        if self._row_points_at_as_duplicate(doc_data, exclude_doc_id):
                            continue
                        created = doc_data.get("created_at")
                        sort_key = (
                            created if isinstance(created, str) else "",
                            doc_id,
                        )
                        if best is None or sort_key < best[0]:
                            best = (sort_key, doc_id, doc_data)

                if cursor == 0:
                    break

        if best is None:
            return None
        return best[1], best[2]

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
        strict: bool = False,
    ) -> DocStatusPage:
        """Bounded k-way merge over the per-status ZSETs.

        Each status stream is read with ``ZRANGEBYLEX (last_member +`` —
        lexicographic member order IS the (created_at, id) keyset order.
        The composite cursor records each status's own last CONSUMED member:
        a candidate merged into the page window is consumed (returned, or
        skipped as unusable in relaxed mode) and advances its stream; a
        prefetched head that did not fit stays unconsumed and is re-read next
        page. A stream is exhausted when its prefetch came back short AND
        fully consumed; the sweep ends when every stream is exhausted.
        """
        self._require_sidecar_ready()
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        if isinstance(position, CursorAfter):
            positions = self._decode_cursor(position.opaque)
        else:
            positions = {}
        status_values = [s.value for s in statuses]

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
        self._require_sidecar_ready()
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
    # Strict batch read
    # ------------------------------------------------------------------

    async def get_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocSchedulingRecord]:
        """Batch strict read of scheduling records by doc id (see base).

        A single pipelined ``GET`` per id straight off the primary rows (the
        source of truth, no sidecar dependency): a ``None`` is a confirmed
        absence and is omitted. The pipeline is all-or-nothing at transport,
        so any server/transport error fails the WHOLE call — the feeder never
        receives a partial mapping. ``strict=True`` additionally raises on a
        malformed row; results use the lightweight projection."""
        if not doc_ids:
            return {}
        ids = list(doc_ids)
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for doc_id in ids:
                pipe.get(f"{self.final_namespace}:{doc_id}")
            raws = await pipe.execute()
        result: dict[str, DocSchedulingRecord] = {}
        for doc_id, raw in zip(ids, raws):
            if raw is None:
                continue  # confirmed absent
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(
                    f"[{self.workspace}] undecodable row for {doc_id} in "
                    f"get_docs_by_ids"
                )
                if strict:
                    raise
                continue
            if not isinstance(row, dict):
                if strict:
                    raise TypeError(f"doc_status record {doc_id} is not a mapping")
                continue
            record = self._scheduling_record_from_row(doc_id, row, strict=strict)
            if record is not None:
                result[doc_id] = record
        return result

    async def get_full_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocProcessingStatus]:
        """Batch hydration to FULL DocProcessingStatus by doc id (see base).

        A single pipelined ``GET`` per id straight off the primary rows (same
        by-id read mechanism as ``get_docs_by_ids``): a ``None`` is a confirmed
        absence and is omitted. The pipeline is all-or-nothing at transport, so
        any server/transport error fails the WHOLE call — the scheduler never
        receives a partial mapping. ``strict=True`` additionally raises on an
        undecodable/unconvertible row; ``strict=False`` logs and skips it.
        Results are FULL records reusing the same raw -> DocProcessingStatus
        normalisation as ``get_docs_by_statuses``."""
        if not doc_ids:
            return {}
        ids = list(doc_ids)
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for doc_id in ids:
                pipe.get(f"{self.final_namespace}:{doc_id}")
            raws = await pipe.execute()
        result: dict[str, DocProcessingStatus] = {}
        for doc_id, raw in zip(ids, raws):
            if raw is None:
                continue  # confirmed absent
            try:
                data = json.loads(raw)
                result[doc_id] = self._redis_doc_processing_status_from_data(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(
                    f"[{self.workspace}] Unusable doc_status row hydrating "
                    f"{doc_id} in get_full_docs_by_ids: {e}"
                )
                if strict:
                    raise
                continue
        return result

    # ------------------------------------------------------------------
    # Source-conflict listing and explicit CAS repair
    # ------------------------------------------------------------------

    @staticmethod
    def _conflict_fingerprint(sorted_doc_ids: list[str]) -> str:
        """Deterministic digest over candidate doc IDs in stable sort order."""
        digest = hashlib.sha256()
        for doc_id in sorted_doc_ids:
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    @staticmethod
    def _decode_conflict_cursor(opaque: str) -> int:
        """Decode the opaque conflict cursor: a raw Redis ``SCAN`` cursor.

        Redis cannot enumerate the source keyspace in key order without
        materializing (and sorting) all of it, so the conflict listing pages on
        the SCAN cursor itself — stateless, resumable and bounded. See
        :meth:`list_source_conflicts_page` for the duplicate-visit caveat this
        inherits from SCAN.
        """
        try:
            cursor = json.loads(opaque)
            if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
                raise ValueError("conflict cursor must be a non-negative SCAN cursor")
        except (ValueError, TypeError) as e:
            raise StorageControlPlaneError(
                f"Malformed source-conflict cursor for RedisDocStatusStorage: {e}"
            ) from e
        return cursor

    async def list_source_conflicts_page(
        self,
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
    ) -> SourceConflictPage:
        """Page canonical source keys with >1 validated primary candidate.

        Streams the source-set keyspace one bounded ``SCAN`` batch at a time,
        pipelines a cheap ``SCARD >= 2`` prefilter over the batch, then
        validates only the surviving keys exactly (hydrating members, self-
        healing stale ones) so a set inflated purely by stale members is not
        reported as a conflict. Stops as soon as ``limit`` conflicts are in
        hand.

        Bounded in memory (one SCAN batch + the page, never the whole canonical
        keyspace) and in round-trips per page (one pipelined SCARD per batch,
        one validation per surviving key — not one per canonical key in the
        workspace). Paging rides the raw SCAN cursor because Redis cannot
        enumerate the keyspace in key order without materializing all of it.

        Two consequences of that, both acceptable for operator tooling and
        neither able to hide a conflict: the page is not ordered by canonical
        key, and SCAN may revisit a key, so the same conflict can appear on two
        pages (repair is idempotent and CAS-guarded, and each conflict needs a
        human decision anyway). ``conflicts`` may overshoot ``limit`` by up to
        one SCAN batch, because a batch is never abandoned half-examined —
        doing so would advance the cursor past keys nobody looked at.
        """
        self._require_sidecar_ready()
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if position is CURSOR_END:
            return SourceConflictPage(conflicts=(), next_position=CURSOR_END)
        cursor = 0
        if isinstance(position, CursorAfter):
            cursor = self._decode_conflict_cursor(position.opaque)
        prefix = f"{self._sched_prefix}:basename:"
        conflicts: list[SourceConflictSummary] = []
        async with self._get_redis_connection() as redis:
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match=f"{prefix}*",
                    count=self._CONFLICT_SCAN_BATCH,
                )
                if keys:
                    card_pipe = redis.pipeline()
                    for key in keys:
                        card_pipe.scard(key)
                    cards = await card_pipe.execute()
                    for key, card in zip(keys, cards):
                        if not card or card < 2:
                            continue
                        canonical = key[len(prefix) :]
                        ids = await self._valid_primary_ids(redis, canonical)
                        if len(ids) >= 2:
                            conflicts.append(
                                SourceConflictSummary(
                                    canonical_source_key=canonical,
                                    candidate_count=len(ids),
                                    sample_doc_ids=tuple(
                                        ids[: self._CONFLICT_SAMPLE_CAP]
                                    ),
                                )
                            )
                # cursor == 0 means the iteration completed: nothing is left to
                # resume from, so the page is terminal even if it filled up.
                if cursor == 0 or len(conflicts) >= limit:
                    break
        next_position: CursorPosition = (
            CURSOR_END if cursor == 0 else CursorAfter(json.dumps(cursor))
        )
        return SourceConflictPage(
            conflicts=tuple(conflicts), next_position=next_position
        )

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> SourceConflictRepairResult:
        """Demote all-but-one primary to duplicate, CAS-guarded (see base).

        Recomputes the validated candidate set (self-healing stale members)
        and its deterministic fingerprint; dry-run reports without mutating.
        On commit, a mismatch with the operator-echoed ``expected_*`` fails as
        a CAS conflict rather than overwriting a concurrent change; otherwise
        each losing candidate is demoted via :meth:`_atomic_doc_write`, so its
        ``metadata.is_duplicate``/``original_doc_id`` write and its source-set
        + status-ZSET maintenance commit in one per-doc transaction. A new
        primary racing in mid-repair simply leaves the resolver in Conflict,
        so a re-run (fresh dry-run) continues without any repair marker."""
        self._require_sidecar_ready()
        async with self._get_redis_connection() as redis:
            candidates = await self._valid_primary_ids(redis, canonical_source_key)
            count = len(candidates)
            fingerprint = self._conflict_fingerprint(candidates)
            if primary_doc_id not in candidates:
                raise ValueError(
                    f"primary_doc_id {primary_doc_id!r} is not a current primary "
                    f"candidate for {canonical_source_key!r}"
                )
            demoted = [d for d in candidates if d != primary_doc_id]
            if dry_run:
                return SourceConflictRepairResult(
                    canonical_source_key=canonical_source_key,
                    primary_doc_id=primary_doc_id,
                    candidate_count=count,
                    fingerprint=fingerprint,
                    demoted_sample_doc_ids=tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
                    committed=False,
                )
            if (
                count != expected_candidate_count
                or fingerprint != expected_candidate_fingerprint
            ):
                raise SourceConflictRepairCASError(
                    f"[{self.workspace}] source-conflict repair CAS failed for "
                    f"{canonical_source_key!r}: candidate set changed "
                    f"(count {count} vs {expected_candidate_count})"
                )

            def _demote(old_row, _primary=primary_doc_id):
                if old_row is None:
                    return None
                row = dict(old_row)
                metadata = dict(row.get("metadata") or {})
                metadata["is_duplicate"] = True
                metadata["original_doc_id"] = _primary
                row["metadata"] = metadata
                return row

            for doc_id in demoted:
                await self._atomic_doc_write(redis, doc_id, _demote)
        return SourceConflictRepairResult(
            canonical_source_key=canonical_source_key,
            primary_doc_id=primary_doc_id,
            candidate_count=count,
            fingerprint=fingerprint,
            demoted_sample_doc_ids=tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
            committed=True,
        )

    # ------------------------------------------------------------------
    # Derived sidecar rebuild (Phase 1)
    # ------------------------------------------------------------------

    async def _scan_keys(self, redis, patterns: Sequence[str]) -> list[str]:
        """SCAN every ``pattern`` and collect the matching keys."""
        found: list[str] = []
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=1000)
                found.extend(keys)
                if cursor == 0:
                    break
        return found

    async def _delete_by_patterns(self, redis, patterns: Sequence[str]) -> None:
        """Delete every key matching any of ``patterns`` in bounded batches."""
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=1000)
                if keys:
                    await redis.delete(*keys)
                if cursor == 0:
                    break

    async def _any_key(self, redis, pattern: str) -> bool:
        """True if any key matches ``pattern`` (bounded SCAN)."""
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=100)
            if keys:
                return True
            if cursor == 0:
                return False

    async def _rebuild_scheduling_sidecar(self) -> None:
        """Rebuild the derived scheduling index from the actual doc_status rows
        when it is absent — no persistent semantic marker gates this.

        Structural check (not a marker): if any ``…__sched:status:*`` key
        exists the index is already built and is kept consistent by every
        write, so leave it untouched (never clobber a live index); if none
        exists but primary rows do, it is a fresh pre-existing deployment or a
        recovery and must be rebuilt.

        Callable only while holding :func:`get_data_init_lock` — the caller,
        ``initialize()``, holds it for its whole body. That lock is
        cross-process and scoped to the ENTIRE server instance, so at most one
        worker is ever inside this method at a time; LightRAG does not support
        multiple server instances sharing one Redis. There is therefore no
        second rebuilder to elect against or wait for here — unlike an
        ordinary WRITE from an already-initialized peer worker, which is a
        separate, still-real hazard handled by :meth:`_publish_rebuilt_index`
        (it refuses to publish over a live index instead of overwriting it).
        If that premise ever changes (multiple server instances against one
        Redis), this method needs a distributed rebuild lock again.

        Deliberate limitation: "any status key exists" cannot detect a PARTIAL
        index (some status keys or members gone), so a partial loss is NOT
        repaired here — it looks healthy. Verifying the index against the rows
        would be an O(total_docs) scan on every startup, and a persisted
        counter would be the second source of truth this design exists to
        avoid. Partial loss is therefore prevented at its only real cause
        instead: ``_ensure_no_eviction_policy`` refuses at startup to run on a
        Redis configured to evict TTL-less keys. An operator who suspects a
        damaged index deletes the ``…__sched:*`` keys and restarts, which takes
        the branch below.

        One exception to "any status key means built": a LEFTOVER temp
        keyspace. The publish is streamed in bounded batches instead of
        switched in one transaction (see :meth:`_publish_rebuilt_index`), so a
        rebuild that died mid-publish leaves some official keys switched and
        the rest absent — which the structural check alone would read as
        healthy. The temp keyspace is what distinguishes that: a completed
        rebuild leaves none, and with no concurrent rebuilder possible, any
        keyspace found here is leftover from a prior, interrupted attempt."""
        status_pattern = f"{self._sched_prefix}:status:*"
        primary_pattern = f"{self.final_namespace}:*"
        async with self._get_redis_connection() as redis:
            interrupted = await self._temp_keyspace_present(redis)
            if interrupted:
                logger.warning(
                    f"[{self.workspace}] a previous scheduling sidecar rebuild for "
                    f"{self.namespace} left its temp keyspace behind; the index "
                    f"may be half-published, so it is rebuilt instead of trusted"
                )
            if not interrupted and await self._any_key(redis, status_pattern):
                return  # already built + maintained live
            if not await self._any_key(redis, primary_pattern):
                return  # nothing to build (empty / freshly dropped workspace)
            await self._do_rebuild(redis, recover_stale_official=interrupted)

    async def _temp_keyspace_present(self, redis) -> bool:
        """Does the rebuild temp keyspace hold any key? (bounded probe)"""
        temp_prefix = f"{self._sched_prefix}_rebuild"
        for pattern in (f"{temp_prefix}:status:*", f"{temp_prefix}:basename:*"):
            if await self._any_key(redis, pattern):
                return True
        return False

    async def _do_rebuild(self, redis, *, recover_stale_official: bool = False) -> None:
        """Stream the primary rows into a temporary keyspace, then switch it in.

        The temp keyspace (``…__sched_rebuild:*``) is disjoint from both the
        official ``…__sched:*`` patterns and the primary ``{ns}:*`` pattern, so
        neither the SCAN of primary rows nor the official index touches it. The
        switch RENAMEs each temp key over its official name in bounded batches
        (:meth:`_publish_rebuilt_index`); a crash before the switch leaves the
        temp keyspace safe to discard on the next attempt, and a crash DURING it
        is detected by that same leftover keyspace (see
        :meth:`_rebuild_scheduling_sidecar`).

        Stale official ``basename:*`` keys are deleted UP FRONT rather than
        during the switch. Two reasons: it needs no O(total_docs) record of which
        names the switch recreated, and the window in which a basename key is
        missing is exactly the window in which the status keys are missing too —
        the state that forces a rebuild on the next startup. Deleting them
        between the renames could instead drop a name the switch had already
        published, and a missing source-multimap entry does NOT self-heal: it
        reads as "no primary for this source" and lets an enqueue admit a
        duplicate.

        ``recover_stale_official`` additionally clears the official STATUS keys:
        set only when the caller detected a half-published index, where those
        keys are this rebuild's own leftovers rather than a live writer's work.
        Clearing them here (instead of teaching the publish to tolerate them)
        keeps the publish probe's meaning intact — a status key present at
        publish time still means a concurrent writer, and still refuses.
        """
        temp_prefix = f"{self._sched_prefix}_rebuild"
        stale_patterns = [
            f"{temp_prefix}:status:*",
            f"{temp_prefix}:basename:*",
            f"{self._sched_prefix}:basename:*",
        ]
        if recover_stale_official:
            stale_patterns.append(f"{self._sched_prefix}:status:*")
        # Clear the leftover temp keyspace from a crashed prior attempt, plus the
        # stale official keys this rebuild is about to republish.
        await self._delete_by_patterns(redis, stale_patterns)

        rebuilt = 0
        statuses_seen: set[str] = set()
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
                            f"[{self.workspace}] skipping undecodable row {key} "
                            f"during sidecar rebuild"
                        )
                        continue
                    if not isinstance(row, dict):
                        continue
                    doc_id = key.split(":", 1)[1]
                    status = self._status_value(row.get("status"))
                    if status:
                        statuses_seen.add(status)
                        write_pipe.zadd(
                            f"{temp_prefix}:status:{status}",
                            {self._zset_member(row, doc_id): 0},
                        )
                    basename = self._basename_of(row)
                    if basename is not None:
                        write_pipe.sadd(f"{temp_prefix}:basename:{basename}", doc_id)
                    rebuilt += 1
                await write_pipe.execute()
            if cursor == 0:
                break

        await self._publish_rebuilt_index(redis, temp_prefix, statuses_seen)
        logger.info(
            f"[{self.workspace}] scheduling sidecar rebuilt "
            f"({rebuilt} rows) for {self.namespace}"
        )

    @staticmethod
    def _is_missing_source_rename_error(exc: Exception) -> bool:
        """True for the specific, benign ``RENAME`` failure a duplicate SCAN
        return produces: the source key is already gone because an earlier
        batch in the same publish pass already renamed it. Redis's own error
        text for this case is exactly ``"no such key"`` (see ``t_string.c`` /
        ``renameGenericCommand``) — matched narrowly so any other
        :class:`ResponseError` (auth, readonly replica, wrong type, OOM, …)
        still aborts the publish instead of being silently absorbed."""
        return isinstance(exc, ResponseError) and "no such key" in str(exc).lower()

    async def _publish_rebuilt_index(
        self, redis, temp_prefix: str, statuses_seen: set[str]
    ) -> None:
        """RENAME the temp keyspace into place in BOUNDED batches.

        Cost is O(batch), not O(total_docs). This previously listed every
        official and every temp key into Python lists and queued one
        DELETE/RENAME per key into a single ``MULTI``, so a million-document
        workspace paid an O(N) client allocation AND an O(N) server-side
        transaction buffer — during the very initialization this phase exists to
        bound. Nothing here holds more than ``_PUBLISH_BATCH`` key names.

        The switch is consequently **not atomic**, which is a scoped trade rather
        than an oversight. LightRAG runs a single service, so the only writers
        that could interleave are that service's own workers, and:

        * a write that landed BEFORE the switch is caught by the probe below —
          the rebuild only starts when no official status key exists, so one
          present now was created by a concurrent writer, and publishing our
          older snapshot over it would silently revert a correctly-maintained
          index (``_queue_index_ops`` commits row + sidecar in one transaction),
          leaving the doc in the wrong status ZSET so the sweep either re-runs it
          or never sees it. We refuse instead;
        * a write that lands DURING the switch is no longer excluded — the
          ``WATCH``/``MULTI`` made that impossible, and giving it up is what buys
          the bound. It needs an ordinary doc_status write to a workspace whose
          sidecar is simultaneously being built from scratch by an elected
          rebuilder, which single-service initialization does not produce.

        Renames go through ``transaction=False``: the batch needs no
        all-or-nothing semantics now, and ``RENAME`` already replaces its
        destination atomically per key — the granularity that actually matters,
        since no reader ever sees a half-written status ZSET.

        SCAN's OWN contract, separately from the concurrent-writer question
        above: "a full iteration retrieves all elements present throughout,
        but an element may be returned more than once" (Redis docs). This sweep
        hands SCAN a moving target on purpose (each pass's renames delete from
        the very pattern being scanned), which is exactly the shape that
        provokes a duplicate return — a key renamed by an earlier batch in this
        SAME pass can be handed to us again by a later one. A repeat ``RENAME``
        on that key fails with "no such key", not a fresh state to react to
        (the first rename already moved it where it belongs), so that specific
        error is swallowed per-command rather than left to abort the whole
        batch — see ``_is_missing_source_rename_error``.
        """
        status_pattern = f"{self._sched_prefix}:status:*"
        if await self._any_key(redis, status_pattern):
            raise StorageControlPlaneError(
                f"[{self.workspace}] another writer maintained the scheduling "
                f"sidecar for {self.namespace} while it was being rebuilt; "
                f"refusing to publish a stale snapshot over live writes. Retry "
                f"with writers quiesced."
            )

        published = 0
        for pattern in (f"{temp_prefix}:status:*", f"{temp_prefix}:basename:*"):
            # Each rename REMOVES a key from the pattern being scanned, and no
            # SCAN cursor survives that intact — a positional cursor skips the
            # entries that shifted under it, and a real hash cursor is only
            # promised not to miss keys that stay present for the whole
            # iteration. So sweep in passes: every pass is cursor-based (no
            # keyspace re-walk per batch) and whatever one pass skipped the next
            # one picks up. This terminates because a pass that renames nothing
            # ends it, and any other pass strictly shrinks the pattern.
            while True:
                renamed_in_pass = 0
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=pattern, count=self._PUBLISH_BATCH
                    )
                    if keys:
                        pipe = redis.pipeline(transaction=False)
                        for temp_key in keys:
                            official = self._sched_prefix + temp_key[len(temp_prefix) :]
                            pipe.rename(temp_key, official)
                        # raise_on_error=False: a duplicate SCAN return makes one
                        # of these a legitimate "no such key" (an earlier batch in
                        # THIS pass already renamed it away), not a real failure —
                        # see the docstring. Any other error still aborts loudly.
                        results = await pipe.execute(raise_on_error=False)
                        for temp_key, result in zip(keys, results):
                            if isinstance(result, Exception):
                                if not self._is_missing_source_rename_error(result):
                                    raise result
                                continue
                            renamed_in_pass += 1
                    if cursor == 0:
                        break
                published += renamed_in_pass
                if not renamed_in_pass:
                    break

        # Post-condition, not decoration: a leftover temp keyspace is exactly what
        # ``_rebuild_scheduling_sidecar`` reads as "half-published", so returning
        # success while leaving one behind would arm a rebuild on every startup.
        if await self._temp_keyspace_present(redis):
            raise StorageControlPlaneError(
                f"[{self.workspace}] scheduling sidecar publish for "
                f"{self.namespace} left temp keys behind after renaming "
                f"{published}; the index is half-published and will be rebuilt "
                f"on the next startup"
            )

    async def drop(self) -> dict[str, str]:
        """Drop all document status data from storage and clean up resources.

        Also clears the derived scheduling sidecar — status ZSETs, source
        multimap sets, and any leftover rebuild temp keyspace — since they
        mirror the dropped rows. Nothing persists across the clear: the index
        is purely derived and would be rebuilt from the (now empty) rows."""
        try:
            async with self._get_redis_connection() as redis:
                deleted_count = 0
                for pattern in (
                    f"{self.final_namespace}:*",
                    f"{self._sched_prefix}:status:*",
                    f"{self._sched_prefix}:basename:*",
                    f"{self._sched_prefix}_rebuild:status:*",
                    f"{self._sched_prefix}_rebuild:basename:*",
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
