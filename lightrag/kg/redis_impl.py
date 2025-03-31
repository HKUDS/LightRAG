import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import configparser

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis # type: ignore
from lightrag.utils import logger
from lightrag.base import BaseKVStorage
import json


config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Use Redis as KV {self.namespace}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        data = await self._redis.get(f"{self.namespace}:{id}")
        return json.loads(data) if data else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        pipe = self._redis.pipeline()
        for id in ids:
            pipe.get(f"{self.namespace}:{id}")
        results = await pipe.execute()
        return [json.loads(result) if result else None for result in results]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        pipe = self._redis.pipeline()
        for key in keys:
            pipe.exists(f"{self.namespace}:{key}")
        results = await pipe.execute()

        existing_ids = {keys[i] for i, exists in enumerate(results) if exists}
        return set(keys) - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        pipe = self._redis.pipeline()

        for k, v in data.items():
            pipe.set(f"{self.namespace}:{k}", json.dumps(v))
        await pipe.execute()

        for k in data:
            data[k]["_id"] = k

    async def index_done_callback(self) -> None:
        # Redis handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete entries with specified IDs

        Args:
            ids: List of entry IDs to be deleted
        """
        if not ids:
            return

        pipe = self._redis.pipeline()
        for id in ids:
            pipe.delete(f"{self.namespace}:{id}")

        results = await pipe.execute()
        deleted_count = sum(results)
        logger.info(
            f"Deleted {deleted_count} of {len(ids)} entries from {self.namespace}"
        )
        
    async def drop_cache_by_modes(self, modes: list[str] | None = None) ->  bool:
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
        try:
            keys = await self._redis.keys(f"{self.namespace}:*")
            
            if keys:
                pipe = self._redis.pipeline()
                for key in keys:
                    pipe.delete(key)
                results = await pipe.execute()
                deleted_count = sum(results)
                
                logger.info(f"Dropped {deleted_count} keys from {self.namespace}")
                return {"status": "success", "message": f"{deleted_count} keys dropped"}
            else:
                logger.info(f"No keys found to drop in {self.namespace}")
                return {"status": "success", "message": "no keys to drop"}
                
        except Exception as e:
            logger.error(f"Error dropping keys from {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
