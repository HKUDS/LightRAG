import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np
import hashlib
import uuid
from ..utils import logger
from ..base import BaseVectorStorage
import configparser


config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

import pipmaster as pm

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, models


def compute_mdhash_id_for_qdrant(
    content: str, prefix: str = "", style: str = "simple"
) -> str:
    """
    Generate a UUID based on the content and support multiple formats.

    :param content: The content used to generate the UUID.
    :param style: The format of the UUID, optional values are "simple", "hyphenated", "urn".
    :return: A UUID that meets the requirements of Qdrant.
    """
    if not content:
        raise ValueError("Content must not be empty.")

    # Use the hash value of the content to create a UUID.
    hashed_content = hashlib.sha256((prefix + content).encode("utf-8")).digest()
    generated_uuid = uuid.UUID(bytes=hashed_content[:16], version=4)

    # Return the UUID according to the specified format.
    if style == "simple":
        return generated_uuid.hex
    elif style == "hyphenated":
        return str(generated_uuid)
    elif style == "urn":
        return f"urn:uuid:{generated_uuid}"
    else:
        raise ValueError("Invalid style. Choose from 'simple', 'hyphenated', or 'urn'.")


@final
@dataclass
class QdrantVectorDBStorage(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: QdrantClient, collection_name: str, **kwargs
    ):
        if client.collection_exists(collection_name):
            return
        client.create_collection(collection_name, **kwargs)

    def __post_init__(self):
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        self._client = QdrantClient(
            url=os.environ.get(
                "QDRANT_URL", config.get("qdrant", "uri", fallback=None)
            ),
            api_key=os.environ.get(
                "QDRANT_API_KEY", config.get("qdrant", "apikey", fallback=None)
            ),
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        QdrantVectorDBStorage.create_collection_if_not_exist(
            self._client,
            self.namespace,
            vectors_config=models.VectorParams(
                size=self.embedding_func.embedding_dim, distance=models.Distance.COSINE
            ),
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        list_data = [
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

        list_points = []
        for i, d in enumerate(list_data):
            list_points.append(
                models.PointStruct(
                    id=compute_mdhash_id_for_qdrant(d["id"]),
                    vector=embeddings[i],
                    payload=d,
                )
            )

        results = self._client.upsert(
            collection_name=self.namespace, points=list_points, wait=True
        )
        return results

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            query_vector=embedding[0],
            limit=top_k,
            with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
        )

        logger.debug(f"query result: {results}")

        return [{**dp.payload, "id": dp.id, "distance": dp.score} for dp in results]

    async def index_done_callback(self) -> None:
        # Qdrant handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError
