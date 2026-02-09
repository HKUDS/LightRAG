import os
import logging
from typing import Any, final, Union
from dataclasses import dataclass
import pipmaster as pm
import configparser
from contextlib import asynccontextmanager
import threading

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis, ConnectionPool  # type: ignore
from redis.exceptions import RedisError, ConnectionError, TimeoutError  # type: ignore
from lightrag.utils import logger, get_pinyin_sort_key

from lightrag.base import (
    BaseKVStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
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
    def release_pool(cls, redis_url: str):
        """Release a reference to the connection pool"""
        with cls._lock:
            if redis_url in cls._pool_refs:
                cls._pool_refs[redis_url] -= 1
                logger.debug(
                    f"Redis pool {redis_url} reference count: {cls._pool_refs[redis_url]}"
                )

                # If no more references, close the pool
                if cls._pool_refs[redis_url] <= 0:
                    try:
                        cls._pools[redis_url].disconnect()
                        logger.info(
                            f"Closed Redis connection pool for {redis_url} (no more references)"
                        )
                    except Exception as e:
                        logger.error(f"Error closing Redis pool for {redis_url}: {e}")
                    finally:
                        del cls._pools[redis_url]
                        del cls._pool_refs[redis_url]

    @classmethod
    def close_all_pools(cls):
        """Close all connection pools (for cleanup)"""
        with cls._lock:
            for url, pool in cls._pools.items():
                try:
                    pool.disconnect()
                    logger.info(f"Closed Redis connection pool for {url}")
                except Exception as e:
                    logger.error(f"Error closing Redis pool for {url}: {e}")
            cls._pools.clear()
            cls._pool_refs.clear()


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
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
                RedisConnectionManager.release_pool(self._redis_url)
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
        if hasattr(self, "_redis") and self._redis:
            try:
                await self._redis.close()
                logger.debug(
                    f"[{self.workspace}] Closed Redis connection for {self.namespace}"
                )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error closing Redis connection: {e}")
            finally:
                self._redis = None

        # Release the pool reference (will auto-close pool if no more references)
        if hasattr(self, "_redis_url") and self._redis_url:
            RedisConnectionManager.release_pool(self._redis_url)
            self._pool = None
            logger.debug(
                f"[{self.workspace}] Released Redis connection pool reference for {self.namespace}"
            )

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
                return None

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
                return [None] * len(ids)

    @redis_retry
    async def get_ids_by_doc_id(self, doc_id: str) -> list[str]:
        pattern = f"{self.final_namespace}:*"
        matched_ids: list[str] = []

        async with self._get_redis_connection() as redis:
            batch_keys: list[str] = []
            async for key in redis.scan_iter(match=pattern, count=1000):
                batch_keys.append(key)
                if len(batch_keys) < 200:
                    continue

                values = await redis.mget(batch_keys)
                for raw_key, raw_value in zip(batch_keys, values):
                    if not raw_value:
                        continue
                    try:
                        payload = json.loads(raw_value)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(payload, dict)
                        and payload.get("full_doc_id") == doc_id
                    ):
                        matched_ids.append(raw_key[len(self.final_namespace) + 1 :])
                batch_keys = []

            if batch_keys:
                values = await redis.mget(batch_keys)
                for raw_key, raw_value in zip(batch_keys, values):
                    if not raw_value:
                        continue
                    try:
                        payload = json.loads(raw_value)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(payload, dict)
                        and payload.get("full_doc_id") == doc_id
                    ):
                        matched_ids.append(raw_key[len(self.final_namespace) + 1 :])

        return matched_ids

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

        import time

        current_time = int(time.time())  # Get current Unix timestamp

        async with self._get_redis_connection() as redis:
            try:
                # Check which keys already exist to determine create vs update
                pipe = redis.pipeline()
                for k in data.keys():
                    pipe.exists(f"{self.final_namespace}:{k}")
                exists_results = await pipe.execute()

                # Add timestamps to data
                for i, (k, v) in enumerate(data.items()):
                    # For text_chunks namespace, ensure llm_cache_list field exists
                    if self.namespace.endswith("text_chunks"):
                        if "llm_cache_list" not in v:
                            v["llm_cache_list"] = []

                    # Add timestamps based on whether key exists
                    if exists_results[i]:  # Key exists, only update update_time
                        v["update_time"] = current_time
                    else:  # New key, set both create_time and update_time
                        v["create_time"] = current_time
                        v["update_time"] = current_time

                    v["_id"] = k

                # Store the data
                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{self.final_namespace}:{k}", json.dumps(v))
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
    """Redis implementation of document status storage"""

    def __post_init__(self):
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
                RedisConnectionManager.release_pool(self._redis_url)
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
        if hasattr(self, "_redis") and self._redis:
            try:
                await self._redis.close()
                logger.debug(
                    f"[{self.workspace}] Closed Redis connection for doc status {self.namespace}"
                )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error closing Redis connection: {e}")
            finally:
                self._redis = None

        # Release the pool reference (will auto-close pool if no more references)
        if hasattr(self, "_redis_url") and self._redis_url:
            RedisConnectionManager.release_pool(self._redis_url)
            self._pool = None
            logger.debug(
                f"[{self.workspace}] Released Redis connection pool reference for doc status {self.namespace}"
            )

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
                            ordered_results.append(None)
                    else:
                        ordered_results.append(None)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in get_by_ids: {e}")
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

                        # Filter by status and create DocProcessingStatus objects
                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    if doc_data.get("status") == status.value:
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
                logger.error(f"[{self.workspace}] Error getting docs by status: {e}")

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
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update document status data"""
        if not data:
            return

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        async with self._get_redis_connection() as redis:
            try:
                # Ensure chunks_list field exists for new documents
                for doc_id, doc_data in data.items():
                    if "chunks_list" not in doc_data:
                        doc_data["chunks_list"] = []

                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{self.final_namespace}:{k}", json.dumps(v))
                await pipe.execute()
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error during upsert: {e}")
                raise

    @redis_retry
    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{self.final_namespace}:{id}")
                return json.loads(data) if data else None
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error for id {id}: {e}")
                return None

    async def delete(self, doc_ids: list[str]) -> None:
        """Delete specific records from storage by their IDs"""
        if not doc_ids:
            return

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for doc_id in doc_ids:
                pipe.delete(f"{self.final_namespace}:{doc_id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"[{self.workspace}] Deleted {deleted_count} of {len(doc_ids)} doc status entries from {self.namespace}"
            )

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
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
                                        status_filter is not None
                                        and doc_data.get("status")
                                        != status_filter.value
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

    async def drop(self) -> dict[str, str]:
        """Drop all document status data from storage and clean up resources"""
        try:
            async with self._get_redis_connection() as redis:
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
                    f"[{self.workspace}] Dropped {deleted_count} doc status keys from {self.namespace}"
                )
                return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping doc status {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
