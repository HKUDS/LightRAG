import asyncio
import os
from typing import Any, final, List
from dataclasses import dataclass
import numpy as np
import hashlib
import uuid
from ..utils import logger
from ..base import BaseVectorStorage
import configparser
import pipmaster as pm

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, models  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


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

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        embedding = await self.embedding_func(
            [query], _priority=5
        )  # higher priority for query
        results = self._client.search(
            collection_name=self.namespace,
            query_vector=embedding[0],
            limit=top_k,
            with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
        )

        logger.debug(f"query result: {results}")

        return [{**dp.payload, "distance": dp.score} for dp in results]

    async def index_done_callback(self) -> None:
        # Qdrant handles persistence automatically
        pass

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            # Convert regular ids to Qdrant compatible ids
            qdrant_ids = [compute_mdhash_id_for_qdrant(id) for id in ids]
            # Delete points from the collection
            self._client.delete(
                collection_name=self.namespace,
                points_selector=models.PointIdsList(
                    points=qdrant_ids,
                ),
                wait=True,
            )
            logger.debug(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by name

        Args:
            entity_name: Name of the entity to delete
        """
        try:
            # Generate the entity ID
            entity_id = compute_mdhash_id_for_qdrant(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Delete the entity point from the collection
            self._client.delete(
                collection_name=self.namespace,
                points_selector=models.PointIdsList(
                    points=[entity_id],
                ),
                wait=True,
            )
            logger.debug(f"Successfully deleted entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: Name of the entity whose relations should be deleted
        """
        try:
            # Find relations where the entity is either source or target
            results = self._client.scroll(
                collection_name=self.namespace,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="src_id", match=models.MatchValue(value=entity_name)
                        ),
                        models.FieldCondition(
                            key="tgt_id", match=models.MatchValue(value=entity_name)
                        ),
                    ]
                ),
                with_payload=True,
                limit=1000,  # Adjust as needed for your use case
            )

            # Extract points that need to be deleted
            relation_points = results[0]
            ids_to_delete = [point.id for point in relation_points]

            if ids_to_delete:
                # Delete the relations
                self._client.delete(
                    collection_name=self.namespace,
                    points_selector=models.PointIdsList(
                        points=ids_to_delete,
                    ),
                    wait=True,
                )
                logger.debug(
                    f"Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(f"No relations found for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def search_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Search for records with IDs starting with a specific prefix.

        Args:
            prefix: The prefix to search for in record IDs

        Returns:
            List of records with matching ID prefixes
        """
        try:
            # Use scroll method to find records with IDs starting with the prefix
            results = self._client.scroll(
                collection_name=self.namespace,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id", match=models.MatchText(text=prefix, prefix=True)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=1000,  # Adjust as needed for your use case
            )

            # Extract matching points
            matching_records = results[0]

            # Format the results to match expected return format
            formatted_results = [{**point.payload} for point in matching_records]

            logger.debug(
                f"Found {len(formatted_results)} records with prefix '{prefix}'"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching for prefix '{prefix}': {e}")
            return []

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Convert to Qdrant compatible ID
            qdrant_id = compute_mdhash_id_for_qdrant(id)

            # Retrieve the point by ID
            result = self._client.retrieve(
                collection_name=self.namespace,
                ids=[qdrant_id],
                with_payload=True,
            )

            if not result:
                return None

            return result[0].payload
        except Exception as e:
            logger.error(f"Error retrieving vector data for ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Convert to Qdrant compatible IDs
            qdrant_ids = [compute_mdhash_id_for_qdrant(id) for id in ids]

            # Retrieve the points by IDs
            results = self._client.retrieve(
                collection_name=self.namespace,
                ids=qdrant_ids,
                with_payload=True,
            )

            return [point.payload for point in results]
        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all data from the Qdrant collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            # Delete the collection and recreate it
            if self._client.collection_exists(self.namespace):
                self._client.delete_collection(self.namespace)

            # Recreate the collection
            QdrantVectorDBStorage.create_collection_if_not_exist(
                self._client,
                self.namespace,
                vectors_config=models.VectorParams(
                    size=self.embedding_func.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )

            logger.info(
                f"Process {os.getpid()} drop Qdrant collection {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping Qdrant collection {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
