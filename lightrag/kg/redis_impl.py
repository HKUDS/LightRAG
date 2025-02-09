import os
from typing import Any, Union
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import pipmaster as pm

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis
from lightrag.utils import logger
from lightrag.base import BaseKVStorage
import json


@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get("REDIS_URI", "redis://localhost:6379")
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Use Redis as KV {self.namespace}")

    async def all_keys(self) -> list[str]:
        keys = await self._redis.keys(f"{self.namespace}:*")
        return [key.split(":", 1)[-1] for key in keys]

    async def get_by_id(self, id):
        data = await self._redis.get(f"{self.namespace}:{id}")
        return json.loads(data) if data else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        pipe = self._redis.pipeline()
        for id in ids:
            pipe.get(f"{self.namespace}:{id}")
        results = await pipe.execute()
        return [json.loads(result) if result else None for result in results]

    async def filter_keys(self, data: list[str]) -> set[str]:
        pipe = self._redis.pipeline()
        for key in data:
            pipe.exists(f"{self.namespace}:{key}")
        results = await pipe.execute()

        existing_ids = {data[i] for i, exists in enumerate(results) if exists}
        return set(data) - existing_ids

    async def upsert(self, data: dict[str, Any]) -> None:
        pipe = self._redis.pipeline()
        for k, v in tqdm_async(data.items(), desc="Upserting"):
            pipe.set(f"{self.namespace}:{k}", json.dumps(v))
        await pipe.execute()

        for k in data:
            data[k]["_id"] = k

    async def drop(self) -> None:
        keys = await self._redis.keys(f"{self.namespace}:*")
        if keys:
            await self._redis.delete(*keys)

    async def get_by_status(
        self, status: str
    ) -> Union[list[dict[str, Any]], None]:
        pipe = self._redis.pipeline()
        for key in await self._redis.keys(f"{self.namespace}:*"):
            pipe.hgetall(key)
        results = await pipe.execute()
        return [data for data in results if data.get("status") == status] or None
