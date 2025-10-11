import asyncio
import os
from typing import Any, final, List
from dataclasses import dataclass
import numpy as np
import hashlib
import uuid
from ..utils import logger
from ..base import BaseVectorStorage
from ..kg.shared_storage import get_data_init_lock, get_storage_lock
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
    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    @staticmethod
    def create_collection_if_not_exist(
        client: QdrantClient, collection_name: str, **kwargs
    ):
        exists = False
        if hasattr(client, "collection_exists"):
            try:
                exists = client.collection_exists(collection_name)
            except Exception:
                exists = False
        else:
            try:
                client.get_collection(collection_name)
                exists = True
            except Exception:
                exists = False

        if not exists:
            client.create_collection(collection_name, **kwargs)

    def __post_init__(self):
        # Check for QDRANT_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Qdrant storage instances
        qdrant_workspace = os.environ.get("QDRANT_WORKSPACE")
        if qdrant_workspace and qdrant_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            self.workspace = qdrant_workspace.strip()
            logger.info(
                f"Using QDRANT_WORKSPACE environment variable: '{self.workspace}' (overriding passed workspace)"
            )
        else:
            # Use the workspace parameter passed during initialization
            if self.workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{self.workspace}'"
                )

        # Get composite workspace (supports multi-tenant isolation)
        composite_workspace = self._get_composite_workspace()
        
        # Sanitize for Qdrant (replace colons with underscores)
        safe_composite_workspace = composite_workspace.replace(":", "_")

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if safe_composite_workspace and safe_composite_workspace != "_":
            self.final_namespace = f"{safe_composite_workspace}_{self.namespace}"
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            self.workspace = "_"
            logger.debug(f"Final namespace (no workspace): '{self.final_namespace}'")

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Initialize client as None - will be created in initialize() method
        self._client = None
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._initialized = False

    async def initialize(self):
        """Initialize Qdrant collection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                # Create QdrantClient if not already created
                if self._client is None:
                    self._client = QdrantClient(
                        url=os.environ.get(
                            "QDRANT_URL", config.get("qdrant", "uri", fallback=None)
                        ),
                        api_key=os.environ.get(
                            "QDRANT_API_KEY",
                            config.get("qdrant", "apikey", fallback=None),
                        ),
                    )
                    logger.debug(
                        f"[{self.workspace}] QdrantClient created successfully"
                    )

                # Create collection if not exists
                QdrantVectorDBStorage.create_collection_if_not_exist(
                    self._client,
                    self.final_namespace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                self._initialized = True
                logger.info(
                    f"[{self.workspace}] Qdrant collection '{self.namespace}' initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize Qdrant collection '{self.namespace}': {e}"
                )
                raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        import time

        current_time = int(time.time())

        list_data = [
            {
                "id": k,
                "created_at": current_time,
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
            collection_name=self.final_namespace, points=list_points, wait=True
        )
        return results

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding_result = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query
            embedding = embedding_result[0]

        results = self._client.search(
            collection_name=self.final_namespace,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
        )

        # logger.debug(f"[{self.workspace}] query result: {results}")

        return [
            {
                **dp.payload,
                "distance": dp.score,
                "created_at": dp.payload.get("created_at"),
            }
            for dp in results
        ]

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
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(
                    points=qdrant_ids,
                ),
                wait=True,
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by name

        Args:
            entity_name: Name of the entity to delete
        """
        try:
            # Generate the entity ID
            entity_id = compute_mdhash_id_for_qdrant(entity_name, prefix="ent-")
            # logger.debug(
            #     f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            # )

            # Delete the entity point from the collection
            self._client.delete(
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(
                    points=[entity_id],
                ),
                wait=True,
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted entity {entity_name}"
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: Name of the entity whose relations should be deleted
        """
        try:
            # Find relations where the entity is either source or target
            results = self._client.scroll(
                collection_name=self.final_namespace,
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
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(
                        points=ids_to_delete,
                    ),
                    wait=True,
                )
                logger.debug(
                    f"[{self.workspace}] Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

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
                collection_name=self.final_namespace,
                ids=[qdrant_id],
                with_payload=True,
            )

            if not result:
                return None

            # Ensure the result contains created_at field
            payload = result[0].payload
            if "created_at" not in payload:
                payload["created_at"] = None

            return payload
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
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
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_payload=True,
            )

            # Ensure each result contains created_at field and preserve caller ordering
            payload_by_original_id: dict[str, dict[str, Any]] = {}
            payload_by_qdrant_id: dict[str, dict[str, Any]] = {}

            for point in results:
                payload = dict(point.payload or {})
                if "created_at" not in payload:
                    payload["created_at"] = None

                qdrant_point_id = str(point.id) if point.id is not None else ""
                if qdrant_point_id:
                    payload_by_qdrant_id[qdrant_point_id] = payload

                original_id = payload.get("id")
                if original_id is not None:
                    payload_by_original_id[str(original_id)] = payload

            ordered_payloads: list[dict[str, Any] | None] = []
            for requested_id, qdrant_id in zip(ids, qdrant_ids):
                payload = payload_by_original_id.get(str(requested_id))
                if payload is None:
                    payload = payload_by_qdrant_id.get(str(qdrant_id))
                ordered_payloads.append(payload)

            return ordered_payloads
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        try:
            # Convert to Qdrant compatible IDs
            qdrant_ids = [compute_mdhash_id_for_qdrant(id) for id in ids]

            # Retrieve the points by IDs with vectors
            results = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_vectors=True,  # Important: request vectors
                with_payload=True,
            )

            vectors_dict = {}
            for point in results:
                if point and point.vector is not None and point.payload:
                    # Get original ID from payload
                    original_id = point.payload.get("id")
                    if original_id:
                        # Convert numpy array to list if needed
                        vector_data = point.vector
                        if isinstance(vector_data, np.ndarray):
                            vector_data = vector_data.tolist()
                        vectors_dict[original_id] = vector_data

            return vectors_dict
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all data from the Qdrant collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        async with get_storage_lock():
            try:
                # Delete the collection and recreate it
                exists = False
                if hasattr(self._client, "collection_exists"):
                    try:
                        exists = self._client.collection_exists(self.final_namespace)
                    except Exception:
                        exists = False
                else:
                    try:
                        self._client.get_collection(self.final_namespace)
                        exists = True
                    except Exception:
                        exists = False

                if exists:
                    self._client.delete_collection(self.final_namespace)

                # Recreate the collection
                QdrantVectorDBStorage.create_collection_if_not_exist(
                    self._client,
                    self.final_namespace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )

                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop Qdrant collection {self.namespace}"
                )
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error dropping Qdrant collection {self.namespace}: {e}"
                )
                return {"status": "error", "message": str(e)}
