import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from pymongo import MongoClient

from lightrag.utils import logger

from lightrag.kg.mongo_impl import MongoKVStorage as BaseKVStorage


@dataclass
class NewMongoKVStorage(BaseKVStorage):
    def __post_init__(self):
        logger.info("Initializing New MongoKVStorage")
        client = MongoClient(
            os.environ.get("MONGO_URI", "mongodb://root:root@localhost:27017/"),
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=30000,
            connect=False,
        )
        database = client.get_database(os.environ.get("MONGO_DATABASE", "LightRAG"))
        self._data = database.get_collection(self.namespace)
        logger.info(f"Use MongoDB as KV {self.namespace}")