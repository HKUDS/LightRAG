import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np
from lightrag.utils import logger, compute_mdhash_id
from ..base import BaseVectorStorage
import pipmaster as pm


if not pm.is_installed("configparser"):
    pm.install("configparser")

if not pm.is_installed("pymilvus"):
    pm.install("pymilvus")

import configparser
from pymilvus import MilvusClient  # type: ignore

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

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        embedding = await self.embedding_func(
            [query], _priority=5
        )  # higher priority for query
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
        """Delete an entity from the vector database

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Compute entity ID from name
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Delete the entity from Milvus collection
            result = self._client.delete(
                collection_name=self.namespace, pks=[entity_id]
            )

            if result and result.get("delete_count", 0) > 0:
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")

        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Search for relations where entity is either source or target
            expr = f'src_id == "{entity_name}" or tgt_id == "{entity_name}"'

            # Find all relations involving this entity
            results = self._client.query(
                collection_name=self.namespace, filter=expr, output_fields=["id"]
            )

            if not results or len(results) == 0:
                logger.debug(f"No relations found for entity {entity_name}")
                return

            # Extract IDs of relations to delete
            relation_ids = [item["id"] for item in results]
            logger.debug(
                f"Found {len(relation_ids)} relations for entity {entity_name}"
            )

            # Delete the relations
            if relation_ids:
                delete_result = self._client.delete(
                    collection_name=self.namespace, pks=relation_ids
                )

                logger.debug(
                    f"Deleted {delete_result.get('delete_count', 0)} relations for {entity_name}"
                )

        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            # Delete vectors by IDs
            result = self._client.delete(collection_name=self.namespace, pks=ids)

            if result and result.get("delete_count", 0) > 0:
                logger.debug(
                    f"Successfully deleted {result.get('delete_count', 0)} vectors from {self.namespace}"
                )
            else:
                logger.debug(f"No vectors were deleted from {self.namespace}")

        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def search_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Search for records with IDs starting with a specific prefix.

        Args:
            prefix: The prefix to search for in record IDs

        Returns:
            List of records with matching ID prefixes
        """
        try:
            # Use Milvus query with expression to find IDs with the given prefix
            expression = f'id like "{prefix}%"'
            results = self._client.query(
                collection_name=self.namespace,
                filter=expression,
                output_fields=list(self.meta_fields) + ["id"],
            )

            logger.debug(f"Found {len(results)} records with prefix '{prefix}'")
            return results

        except Exception as e:
            logger.error(f"Error searching for records with prefix '{prefix}': {e}")
            return []

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Query Milvus for a specific ID
            result = self._client.query(
                collection_name=self.namespace,
                filter=f'id == "{id}"',
                output_fields=list(self.meta_fields) + ["id"],
            )

            if not result or len(result) == 0:
                return None

            return result[0]
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
            # Prepare the ID filter expression
            id_list = '", "'.join(ids)
            filter_expr = f'id in ["{id_list}"]'

            # Query Milvus with the filter
            result = self._client.query(
                collection_name=self.namespace,
                filter=filter_expr,
                output_fields=list(self.meta_fields) + ["id"],
            )

            return result or []
        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all data from the Milvus collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            # Drop the collection and recreate it
            if self._client.has_collection(self.namespace):
                self._client.drop_collection(self.namespace)

            # Recreate the collection
            MilvusVectorDBStorage.create_collection_if_not_exist(
                self._client,
                self.namespace,
                dimension=self.embedding_func.embedding_dim,
            )

            logger.info(
                f"Process {os.getpid()} drop Milvus collection {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping Milvus collection {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
