import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import numpy as np
from lightrag.utils import logger
from lightrag.utils import compute_mdhash_id_for_qdrant
from ..base import BaseVectorStorage

from qdrant_client import QdrantClient,models


@dataclass
class QdrantVectorDBStorage(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: QdrantClient, collection_name: str, **kwargs
    ):
        if client.collection_exists(collection_name):
            return
        client.create_collection(
            collection_name, **kwargs
        )

    def __post_init__(self):
        self._client = QdrantClient(
            url=os.environ.get(
                "QDRANT_URL"
            ),
            port=os.environ.get("QDRANT_PORT", 6333),
            grpc_port=os.environ.get("QDRANT_GRPC_PORT", 6334),
            https=False,
            api_key=os.environ.get("QDRANT_API_KEY", None),
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        QdrantVectorDBStorage.create_collection_if_not_exist(
            self._client,
            self.namespace,
            vectors_config=models.VectorParams(size=self.embedding_func.embedding_dim, distance=models.Distance.DOT)
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
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

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)

        list_points = []
        for i, d in enumerate(list_data):
            list_points.append(models.PointStruct(
                id=compute_mdhash_id_for_qdrant(d["id"]),
                vector=embeddings[i],
                payload=d
            ))
        
        results = self._client.upsert(collection_name=self.namespace, points=list_points, wait=True)
        return results

    async def query(self, query, top_k=5):
        query = f"{self.query_instruction}{query}"
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            query_vector=embedding[0],
            limit=top_k,
            with_payload=True,
        )
        logger.debug(f"query result: {results}")
        return [
            {**dp.payload, "id": dp.id, "distance": dp.score}
            for dp in results
        ]
