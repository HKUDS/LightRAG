import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import configparser

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis
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
