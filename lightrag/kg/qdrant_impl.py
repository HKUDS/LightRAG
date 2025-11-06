import asyncio
import configparser
import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Any, List, final

import numpy as np
import pipmaster as pm

from ..base import BaseVectorStorage
from ..exceptions import QdrantMigrationError
from ..kg.shared_storage import get_data_init_lock, get_storage_lock
from ..utils import compute_mdhash_id, logger

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, models  # type: ignore

DEFAULT_WORKSPACE = "_"
WORKSPACE_ID_FIELD = "workspace_id"
ENTITY_PREFIX = "ent-"
CREATED_AT_FIELD = "created_at"
ID_FIELD = "id"

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


def workspace_filter_condition(workspace: str) -> models.FieldCondition:
    """
    Create a workspace filter condition for Qdrant queries.
    """
    return models.FieldCondition(
        key=WORKSPACE_ID_FIELD, match=models.MatchValue(value=workspace)
    )


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
    def setup_collection(
        client: QdrantClient,
        collection_name: str,
        legacy_namespace: str = None,
        workspace: str = None,
        **kwargs,
    ):
        """
        Setup Qdrant collection with migration support from legacy collections.

        Args:
            client: QdrantClient instance
            collection_name: Name of the new collection
            legacy_namespace: Name of the legacy collection (if exists)
            workspace: Workspace identifier for data isolation
            **kwargs: Additional arguments for collection creation (vectors_config, hnsw_config, etc.)
        """
        new_collection_exists = client.collection_exists(collection_name)
        legacy_exists = legacy_namespace and client.collection_exists(legacy_namespace)

        # Case 1: Both new and legacy collections exist - Warning only (no migration)
        if new_collection_exists and legacy_exists:
            logger.warning(
                f"Qdrant: Legacy collection '{legacy_namespace}' still exist. Remove it if migration is complete."
            )
            return

        # Case 2: Only new collection exists - Ensure index exists
        if new_collection_exists:
            # Check if workspace index exists, create if missing
            try:
                collection_info = client.get_collection(collection_name)
                if WORKSPACE_ID_FIELD not in collection_info.payload_schema:
                    logger.info(
                        f"Qdrant: Creating missing workspace index for '{collection_name}'"
                    )
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=WORKSPACE_ID_FIELD,
                        field_schema=models.KeywordIndexParams(
                            type=models.KeywordIndexType.KEYWORD,
                            is_tenant=True,
                        ),
                    )
            except Exception as e:
                logger.warning(
                    f"Qdrant: Could not verify/create workspace index for '{collection_name}': {e}"
                )
            return

        # Case 3: Neither exists - Create new collection
        if not legacy_exists:
            logger.info(f"Qdrant: Creating new collection '{collection_name}'")
            client.create_collection(collection_name, **kwargs)
            client.create_payload_index(
                collection_name=collection_name,
                field_name=WORKSPACE_ID_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            logger.info(f"Qdrant: Collection '{collection_name}' created successfully")
            return

        # Case 4: Only legacy exists - Migrate data
        logger.info(
            f"Qdrant: Migrating data from legacy collection '{legacy_namespace}'"
        )

        try:
            # Get legacy collection count
            legacy_count = client.count(
                collection_name=legacy_namespace, exact=True
            ).count
            logger.info(f"Qdrant: Found {legacy_count} records in legacy collection")

            if legacy_count == 0:
                logger.info("Qdrant: Legacy collection is empty, skipping migration")
                # Create new empty collection
                client.create_collection(collection_name, **kwargs)
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=WORKSPACE_ID_FIELD,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                        is_tenant=True,
                    ),
                )
                return

            # Create new collection first
            logger.info(f"Qdrant: Creating new collection '{collection_name}'")
            client.create_collection(collection_name, **kwargs)

            # Batch migration (500 records per batch)
            migrated_count = 0
            offset = None
            batch_size = 500

            while True:
                # Scroll through legacy data
                result = client.scroll(
                    collection_name=legacy_namespace,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                points, next_offset = result

                if not points:
                    break

                # Transform points for new collection
                new_points = []
                for point in points:
                    # Add workspace_id to payload
                    new_payload = dict(point.payload or {})
                    new_payload[WORKSPACE_ID_FIELD] = workspace or DEFAULT_WORKSPACE

                    # Create new point with workspace-prefixed ID
                    original_id = new_payload.get(ID_FIELD)
                    if original_id:
                        new_point_id = compute_mdhash_id_for_qdrant(
                            original_id, prefix=workspace or DEFAULT_WORKSPACE
                        )
                    else:
                        # Fallback: use original point ID
                        new_point_id = str(point.id)

                    new_points.append(
                        models.PointStruct(
                            id=new_point_id,
                            vector=point.vector,
                            payload=new_payload,
                        )
                    )

                # Upsert to new collection
                client.upsert(
                    collection_name=collection_name, points=new_points, wait=True
                )

                migrated_count += len(points)
                logger.info(f"Qdrant: {migrated_count}/{legacy_count} records migrated")

                # Check if we've reached the end
                if next_offset is None:
                    break
                offset = next_offset

            # Verify migration by comparing counts
            logger.info("Verifying migration...")
            new_count = client.count(collection_name=collection_name, exact=True).count

            if new_count != legacy_count:
                error_msg = f"Qdrant: Migration verification failed, expected {legacy_count} records, got {new_count} in new collection"
                logger.error(error_msg)
                raise QdrantMigrationError(error_msg)

            logger.info(
                f"Qdrant: Migration completed successfully: {migrated_count} records migrated"
            )

            # Create payload index after successful migration
            logger.info("Qdrant: Creating workspace payload index...")
            client.create_payload_index(
                collection_name=collection_name,
                field_name=WORKSPACE_ID_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            logger.info(
                f"Qdrant: Migration from '{legacy_namespace}' to '{collection_name}' completed successfully"
            )

        except QdrantMigrationError:
            # Re-raise migration errors without wrapping
            raise
        except Exception as e:
            error_msg = f"Qdrant: Migration failed with error: {e}"
            logger.error(error_msg)
            raise QdrantMigrationError(error_msg) from e

    def __post_init__(self):
        # Check for QDRANT_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Qdrant storage instances
        qdrant_workspace = os.environ.get("QDRANT_WORKSPACE")
        if qdrant_workspace and qdrant_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = qdrant_workspace.strip()
            logger.info(
                f"Using QDRANT_WORKSPACE environment variable: '{effective_workspace}' (overriding passed workspace: '{self.workspace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Get legacy namespace for data migration from old version
        if effective_workspace:
            self.legacy_namespace = f"{effective_workspace}_{self.namespace}"
        else:
            self.legacy_namespace = self.namespace

        self.effective_workspace = effective_workspace or DEFAULT_WORKSPACE

        # Use a shared collection with payload-based partitioning (Qdrant's recommended approach)
        # Ref: https://qdrant.tech/documentation/guides/multiple-partitions/
        self.final_namespace = f"lightrag_vdb_{self.namespace}"
        logger.debug(
            f"Using shared collection '{self.final_namespace}' with workspace '{self.effective_workspace}' for payload-based partitioning"
        )

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

                # Setup collection (create if not exists and configure indexes)
                # Pass legacy_namespace and workspace for migration support
                QdrantVectorDBStorage.setup_collection(
                    self._client,
                    self.final_namespace,
                    legacy_namespace=self.legacy_namespace,
                    workspace=self.effective_workspace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=0,
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
                ID_FIELD: k,
                WORKSPACE_ID_FIELD: self.effective_workspace,
                CREATED_AT_FIELD: current_time,
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
                    id=compute_mdhash_id_for_qdrant(
                        d[ID_FIELD], prefix=self.effective_workspace
                    ),
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

        results = self._client.query_points(
            collection_name=self.final_namespace,
            query=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
            query_filter=models.Filter(
                must=[workspace_filter_condition(self.effective_workspace)]
            ),
        ).points

        return [
            {
                **dp.payload,
                "distance": dp.score,
                CREATED_AT_FIELD: dp.payload.get(CREATED_AT_FIELD),
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
            if not ids:
                return

            # Convert regular ids to Qdrant compatible ids
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in ids
            ]
            # Delete points from the collection with workspace filtering
            self._client.delete(
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(points=qdrant_ids),
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
            # Generate the entity ID using the same function as used for storage
            entity_id = compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)
            qdrant_entity_id = compute_mdhash_id_for_qdrant(
                entity_id, prefix=self.effective_workspace
            )

            # Delete the entity point by its Qdrant ID directly
            self._client.delete(
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(points=[qdrant_entity_id]),
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
            # Find relations where the entity is either source or target, with workspace filtering
            results = self._client.scroll(
                collection_name=self.final_namespace,
                scroll_filter=models.Filter(
                    must=[workspace_filter_condition(self.effective_workspace)],
                    should=[
                        models.FieldCondition(
                            key="src_id", match=models.MatchValue(value=entity_name)
                        ),
                        models.FieldCondition(
                            key="tgt_id", match=models.MatchValue(value=entity_name)
                        ),
                    ],
                ),
                with_payload=True,
                limit=1000,  # Adjust as needed for your use case
            )

            # Extract points that need to be deleted
            relation_points = results[0]
            ids_to_delete = [point.id for point in relation_points]

            if ids_to_delete:
                # Delete the relations with workspace filtering
                assert isinstance(self._client, QdrantClient)
                self._client.delete(
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(points=ids_to_delete),
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
            qdrant_id = compute_mdhash_id_for_qdrant(
                id, prefix=self.effective_workspace
            )

            # Retrieve the point by ID with workspace filtering
            result = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=[qdrant_id],
                with_payload=True,
            )

            if not result:
                return None

            payload = result[0].payload
            if CREATED_AT_FIELD not in payload:
                payload[CREATED_AT_FIELD] = None

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
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in ids
            ]

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
                if CREATED_AT_FIELD not in payload:
                    payload[CREATED_AT_FIELD] = None

                qdrant_point_id = str(point.id) if point.id is not None else ""
                if qdrant_point_id:
                    payload_by_qdrant_id[qdrant_point_id] = payload

                original_id = payload.get(ID_FIELD)
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
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in ids
            ]

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
                    original_id = point.payload.get(ID_FIELD)
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

        This method will delete all data for the current workspace from the Qdrant collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        async with get_storage_lock():
            try:
                # Delete all points for the current workspace
                self._client.delete(
                    collection_name=self.final_namespace,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[workspace_filter_condition(self.effective_workspace)]
                        )
                    ),
                    wait=True,
                )

                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} dropped workspace data from Qdrant collection {self.namespace}"
                )
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error dropping workspace data from Qdrant collection {self.namespace}: {e}"
                )
                return {"status": "error", "message": str(e)}
