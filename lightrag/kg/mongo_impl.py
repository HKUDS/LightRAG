import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from pymongo import MongoClient
from typing import Union
from lightrag.utils import logger

from lightrag.base import BaseKVStorage


@dataclass
class MongoKVStorage(BaseKVStorage):
    def __post_init__(self):
        client = MongoClient(
            os.environ.get("MONGO_URI", "mongodb://root:root@localhost:27017/")
        )
        database = client.get_database(os.environ.get("MONGO_DATABASE", "LightRAG"))
        self._data = database.get_collection(self.namespace)
        logger.info(f"Use MongoDB as KV {self.namespace}")

    async def all_keys(self) -> list[str]:
        return [x["_id"] for x in self._data.find({}, {"_id": 1})]

    async def get_by_id(self, id):
        return self._data.find_one({"_id": id})

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return list(self._data.find({"_id": {"$in": ids}}))
        return list(
            self._data.find(
                {"_id": {"$in": ids}},
                {field: 1 for field in fields},
            )
        )

    async def filter_keys(self, data: list[str]) -> set[str]:
        existing_ids = [
            str(x["_id"]) for x in self._data.find({"_id": {"$in": data}}, {"_id": 1})
        ]
        return set([s for s in data if s not in existing_ids])

    async def upsert(self, data: dict[str, dict]):
        if self.namespace == "llm_response_cache":
            for mode, items in data.items():
                for k, v in tqdm_async(items.items(), desc="Upserting"):
                    key = f"{mode}_{k}"
                    result = self._data.update_one(
                        {"_id": key}, {"$setOnInsert": v}, upsert=True
                    )
                    if result.upserted_id:
                        logger.debug(f"\nInserted new document with key: {key}")
                    data[mode][k]["_id"] = key
        else:
            for k, v in tqdm_async(data.items(), desc="Upserting"):
                self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
                data[k]["_id"] = k
        return data

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        if "llm_response_cache" == self.namespace:
            res = {}
            v = self._data.find_one({"_id": mode + "_" + id})
            if v:
                res[id] = v
                logger.debug(f"llm_response_cache find one by:{id}")
                return res
            else:
                return None
        else:
            return None

    async def drop(self):
        """ """
        pass
