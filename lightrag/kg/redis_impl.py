import os
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
from redis.exceptions import RedisError, ConnectionError  # type: ignore
from lightrag.utils import logger

from lightrag.base import (
    BaseKVStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
)
import json


config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Constants for Redis connection pool
MAX_CONNECTIONS = 50
SOCKET_TIMEOUT = 5.0
SOCKET_CONNECT_TIMEOUT = 3.0


class RedisConnectionManager:
    """Shared Redis connection pool manager to avoid creating multiple pools for the same Redis URI"""

    _pools = {}
    _lock = threading.Lock()

    @classmethod
    def get_pool(cls, redis_url: str) -> ConnectionPool:
        """Get or create a connection pool for the given Redis URL"""
        if redis_url not in cls._pools:
            with cls._lock:
                if redis_url not in cls._pools:
                    cls._pools[redis_url] = ConnectionPool.from_url(
                        redis_url,
                        max_connections=MAX_CONNECTIONS,
                        decode_responses=True,
                        socket_timeout=SOCKET_TIMEOUT,
                        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
                    )
                    logger.info(f"Created shared Redis connection pool for {redis_url}")
        return cls._pools[redis_url]

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


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        # Use shared connection pool
        self._pool = RedisConnectionManager.get_pool(redis_url)
        self._redis = Redis(connection_pool=self._pool)
        logger.info(
            f"Initialized Redis KV storage for {self.namespace} using shared connection pool"
        )

    async def initialize(self):
        """Initialize Redis connection and migrate legacy cache structure if needed"""
        # Test connection
        try:
            async with self._get_redis_connection() as redis:
                await redis.ping()
                logger.info(f"Connected to Redis for namespace {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Migrate legacy cache structure if this is a cache namespace
        if self.namespace.endswith("_cache"):
            await self._migrate_legacy_cache_structure()

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        try:
            yield self._redis
        except ConnectionError as e:
            logger.error(f"Redis connection error in {self.namespace}: {e}")
            raise
        except RedisError as e:
            logger.error(f"Redis operation error in {self.namespace}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in Redis operation for {self.namespace}: {e}"
            )
            raise

    async def close(self):
        """Close the Redis connection pool to prevent resource leaks."""
        if hasattr(self, "_redis") and self._redis:
            await self._redis.close()
            await self._pool.disconnect()
            logger.debug(f"Closed Redis connection pool for {self.namespace}")

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self.close()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{self.namespace}:{id}")
                if data:
                    result = json.loads(data)
                    # Ensure time fields are present, provide default values for old data
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    return result
                return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for id {id}: {e}")
                return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{self.namespace}:{id}")
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
                logger.error(f"JSON decode error in batch get: {e}")
                return [None] * len(ids)

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """
        async with self._get_redis_connection() as redis:
            try:
                # Get all keys for this namespace
                keys = await redis.keys(f"{self.namespace}:*")

                if not keys:
                    return {}

                # Get all values in batch
                pipe = redis.pipeline()
                for key in keys:
                    pipe.get(key)
                values = await pipe.execute()

                # Build result dictionary
                result = {}
                for key, value in zip(keys, values):
                    if value:
                        # Extract the ID part (after namespace:)
                        key_id = key.split(":", 1)[1]
                        try:
                            data = json.loads(value)
                            # Ensure time fields are present for all documents
                            data.setdefault("create_time", 0)
                            data.setdefault("update_time", 0)
                            result[key_id] = data
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error for key {key}: {e}")
                            continue

                return result
            except Exception as e:
                logger.error(f"Error getting all data from Redis: {e}")
                return {}

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)  # Convert set to list for indexing
            for key in keys_list:
                pipe.exists(f"{self.namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

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
                    pipe.exists(f"{self.namespace}:{k}")
                exists_results = await pipe.execute()

                # Add timestamps to data
                for i, (k, v) in enumerate(data.items()):
                    # For text_chunks namespace, ensure llm_cache_list field exists
                    if "text_chunks" in self.namespace:
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
                    pipe.set(f"{self.namespace}:{k}", json.dumps(v))
                await pipe.execute()

            except json.JSONEncodeError as e:
                logger.error(f"JSON encode error during upsert: {e}")
                raise

    async def index_done_callback(self) -> None:
        # Redis handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete entries with specified IDs"""
        if not ids:
            return

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for id in ids:
                pipe.delete(f"{self.namespace}:{id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"Deleted {deleted_count} of {len(ids)} entries from {self.namespace}"
            )

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Delete specific records from storage by cache mode

        Importance notes for Redis storage:
        1. This will immediately delete the specified cache modes from Redis

        Args:
            modes (list[str]): List of cache modes to be dropped from storage

        Returns:
             True: if the cache drop successfully
             False: if the cache drop failed
        """
        if not modes:
            return False

        try:
            async with self._get_redis_connection() as redis:
                keys_to_delete = []

                # Find matching keys for each mode using SCAN
                for mode in modes:
                    # Use correct pattern to match flattened cache key format {namespace}:{mode}:{cache_type}:{hash}
                    pattern = f"{self.namespace}:{mode}:*"
                    cursor = 0
                    mode_keys = []

                    while True:
                        cursor, keys = await redis.scan(
                            cursor, match=pattern, count=1000
                        )
                        if keys:
                            mode_keys.extend(keys)

                        if cursor == 0:
                            break

                    keys_to_delete.extend(mode_keys)
                    logger.info(
                        f"Found {len(mode_keys)} keys for mode '{mode}' with pattern '{pattern}'"
                    )

                if keys_to_delete:
                    # Batch delete
                    pipe = redis.pipeline()
                    for key in keys_to_delete:
                        pipe.delete(key)
                    results = await pipe.execute()
                    deleted_count = sum(results)
                    logger.info(
                        f"Dropped {deleted_count} cache entries for modes: {modes}"
                    )
                else:
                    logger.warning(f"No cache entries found for modes: {modes}")

            return True
        except Exception as e:
            logger.error(f"Error dropping cache by modes in Redis: {e}")
            return False

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all keys under the current namespace.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to find all keys with the namespace prefix
                pattern = f"{self.namespace}:*"
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

                logger.info(f"Dropped {deleted_count} keys from {self.namespace}")
                return {
                    "status": "success",
                    "message": f"{deleted_count} keys dropped",
                }

            except Exception as e:
                logger.error(f"Error dropping keys from {self.namespace}: {e}")
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
            keys = await redis.keys(f"{self.namespace}:*")

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
                    f"Found flattened cache keys in {self.namespace}, skipping migration"
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
                    full_key = f"{self.namespace}:{flattened_key}"
                    pipe.set(full_key, json.dumps(cache_entry))
                    migration_count += 1

            await pipe.execute()

            if migration_count > 0:
                logger.info(
                    f"Migrated {migration_count} legacy cache entries to flattened structure in Redis"
                )


@final
@dataclass
class RedisDocStatusStorage(DocStatusStorage):
    """Redis implementation of document status storage"""

    def __post_init__(self):
        redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        # Use shared connection pool
        self._pool = RedisConnectionManager.get_pool(redis_url)
        self._redis = Redis(connection_pool=self._pool)
        logger.info(
            f"Initialized Redis doc status storage for {self.namespace} using shared connection pool"
        )

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            async with self._get_redis_connection() as redis:
                await redis.ping()
                logger.info(
                    f"Connected to Redis for doc status namespace {self.namespace}"
                )
        except Exception as e:
            logger.error(f"Failed to connect to Redis for doc status: {e}")
            raise

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        try:
            yield self._redis
        except ConnectionError as e:
            logger.error(f"Redis connection error in doc status {self.namespace}: {e}")
            raise
        except RedisError as e:
            logger.error(f"Redis operation error in doc status {self.namespace}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in Redis doc status operation for {self.namespace}: {e}"
            )
            raise

    async def close(self):
        """Close the Redis connection."""
        if hasattr(self, "_redis") and self._redis:
            await self._redis.close()
            logger.debug(f"Closed Redis connection for doc status {self.namespace}")

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
                pipe.exists(f"{self.namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{self.namespace}:{id}")
                results = await pipe.execute()

                for result_data in results:
                    if result_data:
                        try:
                            result.append(json.loads(result_data))
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in get_by_ids: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error in get_by_ids: {e}")
        return result

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to iterate through all keys in the namespace
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.namespace}:*", count=1000
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
                logger.error(f"Error getting status counts: {e}")

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
                        cursor, match=f"{self.namespace}:*", count=1000
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
                                        # If content is missing, use content_summary as content
                                        if (
                                            "content" not in data
                                            and "content_summary" in data
                                        ):
                                            data["content"] = data["content_summary"]
                                        # If file_path is not in data, use document id as file path
                                        if "file_path" not in data:
                                            data["file_path"] = "no-file-path"

                                        result[doc_id] = DocProcessingStatus(**data)
                                except (json.JSONDecodeError, KeyError) as e:
                                    logger.error(
                                        f"Error processing document {key}: {e}"
                                    )
                                    continue

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Error getting docs by status: {e}")

        return result

    async def index_done_callback(self) -> None:
        """Redis handles persistence automatically"""
        pass

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update document status data"""
        if not data:
            return

        logger.debug(f"Inserting {len(data)} records to {self.namespace}")
        async with self._get_redis_connection() as redis:
            try:
                # Ensure chunks_list field exists for new documents
                for doc_id, doc_data in data.items():
                    if "chunks_list" not in doc_data:
                        doc_data["chunks_list"] = []

                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{self.namespace}:{k}", json.dumps(v))
                await pipe.execute()
            except json.JSONEncodeError as e:
                logger.error(f"JSON encode error during upsert: {e}")
                raise

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{self.namespace}:{id}")
                return json.loads(data) if data else None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for id {id}: {e}")
                return None

    async def delete(self, doc_ids: list[str]) -> None:
        """Delete specific records from storage by their IDs"""
        if not doc_ids:
            return

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for doc_id in doc_ids:
                pipe.delete(f"{self.namespace}:{doc_id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"Deleted {deleted_count} of {len(doc_ids)} doc status entries from {self.namespace}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop all document status data from storage and clean up resources"""
        try:
            async with self._get_redis_connection() as redis:
                # Use SCAN to find all keys with the namespace prefix
                pattern = f"{self.namespace}:*"
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
                    f"Dropped {deleted_count} doc status keys from {self.namespace}"
                )
                return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping doc status {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
