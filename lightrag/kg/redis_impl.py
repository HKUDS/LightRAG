import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import configparser
from contextlib import asynccontextmanager

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis, ConnectionPool  # type: ignore
from redis.exceptions import RedisError, ConnectionError  # type: ignore
from lightrag.utils import logger

from lightrag.base import BaseKVStorage
import json


config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Constants for Redis connection pool
MAX_CONNECTIONS = 50
SOCKET_TIMEOUT = 5.0
SOCKET_CONNECT_TIMEOUT = 3.0


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        # Create a connection pool with limits
        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=MAX_CONNECTIONS,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT,
            socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        )
        self._redis = Redis(connection_pool=self._pool)
        logger.info(
            f"Initialized Redis connection pool for {self.namespace} with max {MAX_CONNECTIONS} connections"
        )

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
                return json.loads(data) if data else None
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
                return [json.loads(result) if result else None for result in results]
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in batch get: {e}")
                return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for key in keys:
                pipe.exists(f"{self.namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        logger.info(f"Inserting {len(data)} items to {self.namespace}")
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{self.namespace}:{k}", json.dumps(v))
                await pipe.execute()

                for k in data:
                    data[k]["_id"] = k
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
        """Delete specific records from storage by by cache mode

        Importance notes for Redis storage:
        1. This will immediately delete the specified cache modes from Redis

        Args:
            modes (list[str]): List of cache mode to be drop from storage

        Returns:
             True: if the cache drop successfully
             False: if the cache drop failed
        """
        if not modes:
            return False

        try:
            await self.delete(modes)
            return True
        except Exception:
            return False

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all keys under the current namespace.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        async with self._get_redis_connection() as redis:
            try:
                keys = await redis.keys(f"{self.namespace}:*")

                if keys:
                    pipe = redis.pipeline()
                    for key in keys:
                        pipe.delete(key)
                    results = await pipe.execute()
                    deleted_count = sum(results)

                    logger.info(f"Dropped {deleted_count} keys from {self.namespace}")
                    return {
                        "status": "success",
                        "message": f"{deleted_count} keys dropped",
                    }
                else:
                    logger.info(f"No keys found to drop in {self.namespace}")
                    return {"status": "success", "message": "no keys to drop"}

            except Exception as e:
                logger.error(f"Error dropping keys from {self.namespace}: {e}")
                return {"status": "error", "message": str(e)}
