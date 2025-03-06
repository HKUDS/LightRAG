import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import configparser

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis
from lightrag.utils import logger, compute_mdhash_id
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

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by name

        Args:
            entity_name: Name of the entity to delete
        """

        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Delete the entity
            result = await self._redis.delete(f"{self.namespace}:{entity_id}")

            if result:
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: Name of the entity whose relations should be deleted
        """
        try:
            # Get all keys in this namespace
            cursor = 0
            relation_keys = []
            pattern = f"{self.namespace}:*"

            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern)

                # For each key, get the value and check if it's related to entity_name
                for key in keys:
                    value = await self._redis.get(key)
                    if value:
                        data = json.loads(value)
                        # Check if this is a relation involving the entity
                        if (
                            data.get("src_id") == entity_name
                            or data.get("tgt_id") == entity_name
                        ):
                            relation_keys.append(key)

                # Exit loop when cursor returns to 0
                if cursor == 0:
                    break

            # Delete the relation keys
            if relation_keys:
                deleted = await self._redis.delete(*relation_keys)
                logger.debug(f"Deleted {deleted} relations for {entity_name}")
            else:
                logger.debug(f"No relations found for entity {entity_name}")

        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")
