import asyncio
from dataclasses import dataclass
from typing import Any, final
import numpy as np

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger
import pipmaster as pm

if not pm.is_installed("chromadb"):
    pm.install("chromadb")

from chromadb import HttpClient, PersistentClient
from chromadb.config import Settings


@final
@dataclass
class ChromaVectorDBStorage(BaseVectorStorage):
    """ChromaDB vector storage implementation."""

    def __post_init__(self):
        try:
            config = self.global_config.get("vector_db_storage_cls_kwargs", {})
            cosine_threshold = config.get("cosine_better_than_threshold")
            if cosine_threshold is None:
                raise ValueError(
                    "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
                )
            self.cosine_better_than_threshold = cosine_threshold

            user_collection_settings = config.get("collection_settings", {})
            # Default HNSW index settings for ChromaDB
            default_collection_settings = {
                # Distance metric used for similarity search (cosine similarity)
                "hnsw:space": "cosine",
                # Number of nearest neighbors to explore during index construction
                # Higher values = better recall but slower indexing
                "hnsw:construction_ef": 128,
                # Number of nearest neighbors to explore during search
                # Higher values = better recall but slower search
                "hnsw:search_ef": 128,
                # Number of connections per node in the HNSW graph
                # Higher values = better recall but more memory usage
                "hnsw:M": 16,
                # Number of vectors to process in one batch during indexing
                "hnsw:batch_size": 100,
                # Number of updates before forcing index synchronization
                # Lower values = more frequent syncs but slower indexing
                "hnsw:sync_threshold": 1000,
            }
            collection_settings = {
                **default_collection_settings,
                **user_collection_settings,
            }

            local_path = config.get("local_path", None)
            if local_path:
                self._client = PersistentClient(
                    path=local_path,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                    ),
                )
            else:
                auth_provider = config.get(
                    "auth_provider", "chromadb.auth.token_authn.TokenAuthClientProvider"
                )
                auth_credentials = config.get("auth_token", "secret-token")
                headers = {}

                if "token_authn" in auth_provider:
                    headers = {
                        config.get(
                            "auth_header_name", "X-Chroma-Token"
                        ): auth_credentials
                    }
                elif "basic_authn" in auth_provider:
                    auth_credentials = config.get("auth_credentials", "admin:admin")

                self._client = HttpClient(
                    host=config.get("host", "localhost"),
                    port=config.get("port", 8000),
                    headers=headers,
                    settings=Settings(
                        chroma_api_impl="rest",
                        chroma_client_auth_provider=auth_provider,
                        chroma_client_auth_credentials=auth_credentials,
                        allow_reset=True,
                        anonymized_telemetry=False,
                    ),
                )

            self._collection = self._client.get_or_create_collection(
                name=self.namespace,
                metadata={
                    **collection_settings,
                    "dimension": self.embedding_func.embedding_dim,
                },
            )
            # Use batch size from collection settings if specified
            self._max_batch_size = self.global_config.get(
                "embedding_batch_num", collection_settings.get("hnsw:batch_size", 32)
            )
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        try:
            ids = list(data.keys())
            documents = [v["content"] for v in data.values()]
            metadatas = [
                {k: v for k, v in item.items() if k in self.meta_fields}
                or {"_default": "true"}
                for item in data.values()
            ]

            # Process in batches
            batches = [
                documents[i : i + self._max_batch_size]
                for i in range(0, len(documents), self._max_batch_size)
            ]

            embedding_tasks = [self.embedding_func(batch) for batch in batches]
            embeddings_list = []

            # Pre-allocate embeddings_list with known size
            embeddings_list = [None] * len(embedding_tasks)

            # Use asyncio.gather instead of as_completed if order doesn't matter
            embeddings_results = await asyncio.gather(*embedding_tasks)
            embeddings_list = list(embeddings_results)

            embeddings = np.concatenate(embeddings_list)

            # Upsert in batches
            for i in range(0, len(ids), self._max_batch_size):
                batch_slice = slice(i, i + self._max_batch_size)

                self._collection.upsert(
                    ids=ids[batch_slice],
                    embeddings=embeddings[batch_slice].tolist(),
                    documents=documents[batch_slice],
                    metadatas=metadatas[batch_slice],
                )

            return ids

        except Exception as e:
            logger.error(f"Error during ChromaDB upsert: {str(e)}")
            raise

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            embedding = await self.embedding_func([query])

            results = self._collection.query(
                query_embeddings=embedding.tolist()
                if not isinstance(embedding, list)
                else embedding,
                n_results=top_k * 2,  # Request more results to allow for filtering
                include=["metadatas", "distances", "documents"],
            )

            # Filter results by cosine similarity threshold and take top k
            # We request 2x results initially to have enough after filtering
            # ChromaDB returns cosine similarity (1 = identical, 0 = orthogonal)
            # We convert to distance (0 = identical, 1 = orthogonal) via (1 - similarity)
            # Only keep results with distance below threshold, then take top k
            return [
                {
                    "id": results["ids"][0][i],
                    "distance": 1 - results["distances"][0][i],
                    "content": results["documents"][0][i],
                    **results["metadatas"][0][i],
                }
                for i in range(len(results["ids"][0]))
                if (1 - results["distances"][0][i]) >= self.cosine_better_than_threshold
            ][:top_k]

        except Exception as e:
            logger.error(f"Error during ChromaDB query: {str(e)}")
            raise

    async def index_done_callback(self) -> None:
        # ChromaDB handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError
