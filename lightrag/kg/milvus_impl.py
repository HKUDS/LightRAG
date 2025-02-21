import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np
from lightrag.utils import logger
from ..base import BaseVectorStorage
import pipmaster as pm


if not pm.is_installed("configparser"):
    pm.install("configparser")

if not pm.is_installed("pymilvus"):
    pm.install("pymilvus")

import configparser
from pymilvus import MilvusClient

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: MilvusClient, collection_name: str, **kwargs
    ):
        if client.has_collection(collection_name):
            return
        client.create_collection(
            collection_name, max_length=64, id_type="string", **kwargs
        )

    def __post_init__(self):
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        self._client = MilvusClient(
            uri=os.environ.get(
                "MILVUS_URI",
                config.get(
                    "milvus",
                    "uri",
                    fallback=os.path.join(
                        self.global_config["working_dir"], "milvus_lite.db"
                    ),
                ),
            ),
            user=os.environ.get(
                "MILVUS_USER", config.get("milvus", "user", fallback=None)
            ),
            password=os.environ.get(
                "MILVUS_PASSWORD", config.get("milvus", "password", fallback=None)
            ),
            token=os.environ.get(
                "MILVUS_TOKEN", config.get("milvus", "token", fallback=None)
            ),
            db_name=os.environ.get(
                "MILVUS_DB_NAME", config.get("milvus", "db_name", fallback=None)
            ),
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        MilvusVectorDBStorage.create_collection_if_not_exist(
            self._client,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        list_data: list[dict[str, Any]] = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(collection_name=self.namespace, data=list_data)
        return results

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={
                "metric_type": "COSINE",
                "params": {"radius": self.cosine_better_than_threshold},
            },
        )
        print(results)
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]

    async def index_done_callback(self) -> None:
        # Milvus handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError
