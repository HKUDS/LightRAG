import asyncio
import configparser
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, List, final

import numpy as np
import pipmaster as pm

from ..base import BaseVectorStorage
from ..exceptions import DataMigrationError
from ..kg.shared_storage import get_data_init_lock, get_namespace_lock
from ..utils import _cooperative_yield, compute_mdhash_id, logger

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, models  # type: ignore


@dataclass
class _PendingVectorDoc:
    """Buffered vector upsert waiting for embedding and/or bulk flush."""

    source: dict[str, Any]
    content: str
    vector: list[float] | None = None


DEFAULT_WORKSPACE = "_"
WORKSPACE_ID_FIELD = "workspace_id"
ENTITY_PREFIX = "ent-"
CREATED_AT_FIELD = "created_at"
ID_FIELD = "id"
DEFAULT_QDRANT_UPSERT_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16MB
DEFAULT_QDRANT_UPSERT_MAX_POINTS_PER_BATCH = 128
DEFAULT_QDRANT_DELETE_MAX_POINTS_PER_BATCH = 1000

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


def _find_legacy_collection(
    client: QdrantClient,
    namespace: str,
    workspace: str = None,
    model_suffix: str = None,
) -> str | None:
    """
    Find legacy collection with backward compatibility support.

    This function tries multiple naming patterns to locate legacy collections
    created by older versions of LightRAG:

    1. lightrag_vdb_{namespace} - if model_suffix is provided (HIGHEST PRIORITY)
    2. {workspace}_{namespace} or {namespace} - no matter if model_suffix is provided or not
    3. lightrag_vdb_{namespace} - fall back value no matter if model_suffix is provided or not (LOWEST PRIORITY)

    Args:
        client: QdrantClient instance
        namespace: Base namespace (e.g., "chunks", "entities")
        workspace: Optional workspace identifier
        model_suffix: Optional model suffix for new collection

    Returns:
        Collection name if found, None otherwise
    """
    # Try multiple naming patterns for backward compatibility
    # More specific names (with workspace) have higher priority
    candidates = [
        f"lightrag_vdb_{namespace}" if model_suffix else None,
        f"{workspace}_{namespace}" if workspace else None,
        f"lightrag_vdb_{namespace}",
        namespace,
    ]

    for candidate in candidates:
        if candidate and client.collection_exists(candidate):
            logger.info(
                f"Qdrant: Found legacy collection '{candidate}' "
                f"(namespace={namespace}, workspace={workspace or 'none'})"
            )
            return candidate

    return None


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
        namespace: str,
        workspace: str,
        vectors_config: models.VectorParams,
        hnsw_config: models.HnswConfigDiff,
        model_suffix: str,
    ):
        """
        Setup Qdrant collection with migration support from legacy collections.

        Ensure final collection has workspace isolation index.
        Check vector dimension compatibility before new collection creation.
        Drop legacy collection if it exists and is empty.
        Only migrate data from legacy collection to new collection when new collection first created and legacy collection is not empty.

        Args:
            client: QdrantClient instance
            collection_name: Name of the final collection
            namespace: Base namespace (e.g., "chunks", "entities")
            workspace: Workspace identifier for data isolation
            vectors_config: Vector configuration parameters for the collection
            hnsw_config: HNSW index configuration diff for the collection
        """
        if not namespace or not workspace:
            raise ValueError("namespace and workspace must be provided")

        workspace_count_filter = models.Filter(
            must=[workspace_filter_condition(workspace)]
        )

        new_collection_exists = client.collection_exists(collection_name)
        legacy_collection = _find_legacy_collection(
            client, namespace, workspace, model_suffix
        )

        # Case 1: Only new collection exists or  new collection is the same as legacy collection
        #         No data migration needed,  and ensuring index is created then return
        if (new_collection_exists and not legacy_collection) or (
            collection_name == legacy_collection
        ):
            # create_payload_index return without error if index already exists
            client.create_payload_index(
                collection_name=collection_name,
                field_name=WORKSPACE_ID_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            new_workspace_count = client.count(
                collection_name=collection_name,
                count_filter=workspace_count_filter,
                exact=True,
            ).count

            # Skip data migration if new collection already has workspace data
            if new_workspace_count == 0 and not (collection_name == legacy_collection):
                logger.warning(
                    f"Qdrant: workspace data in collection '{collection_name}' is empty. "
                    f"Ensure it is caused by new workspace setup and not an unexpected embedding model change."
                )

            return

        legacy_count = None
        if not new_collection_exists:
            # Check vector dimension compatibility before creating new collection
            if legacy_collection:
                legacy_count = client.count(
                    collection_name=legacy_collection, exact=True
                ).count
                if legacy_count > 0:
                    legacy_info = client.get_collection(legacy_collection)
                    legacy_dim = legacy_info.config.params.vectors.size

                    if vectors_config.size and legacy_dim != vectors_config.size:
                        logger.error(
                            f"Qdrant: Dimension mismatch detected! "
                            f"Legacy collection '{legacy_collection}' has {legacy_dim}d vectors, "
                            f"but new embedding model expects {vectors_config.size}d."
                        )

                        raise DataMigrationError(
                            f"Dimension mismatch between legacy collection '{legacy_collection}' "
                            f"and new collection. Expected {vectors_config.size}d but got {legacy_dim}d."
                        )

            client.create_collection(
                collection_name, vectors_config=vectors_config, hnsw_config=hnsw_config
            )
            logger.info(f"Qdrant: Collection '{collection_name}' created successfully")
            if not legacy_collection:
                logger.warning(
                    "Qdrant: Ensure this new collection creation is caused by new workspace setup and not an unexpected embedding model change."
                )

        # create_payload_index return without error if index already exists
        client.create_payload_index(
            collection_name=collection_name,
            field_name=WORKSPACE_ID_FIELD,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )

        # Case 2: Legacy collection exist
        if legacy_collection:
            # Only drop legacy collection if it's empty
            if legacy_count is None:
                legacy_count = client.count(
                    collection_name=legacy_collection, exact=True
                ).count
            if legacy_count == 0:
                client.delete_collection(collection_name=legacy_collection)
                logger.info(
                    f"Qdrant: Empty legacy collection '{legacy_collection}' deleted successfully"
                )
                return

            new_workspace_count = client.count(
                collection_name=collection_name,
                count_filter=workspace_count_filter,
                exact=True,
            ).count

            # Skip data migration if new collection already has workspace data
            if new_workspace_count > 0:
                logger.warning(
                    f"Qdrant: Both new and legacy collection have data. "
                    f"{legacy_count} records in {legacy_collection} require manual deletion after migration verification."
                )
                return

            # Case 3: Only legacy exists - migrate data from legacy collection to new collection
            # Check if legacy collection has workspace_id to determine migration strategy
            # Note: payload_schema only reflects INDEXED fields, so we also sample
            # actual payloads to detect unindexed workspace_id fields
            legacy_info = client.get_collection(legacy_collection)
            has_workspace_index = WORKSPACE_ID_FIELD in (
                legacy_info.payload_schema or {}
            )

            # Detect workspace_id field presence by sampling payloads if not indexed
            # This prevents cross-workspace data leakage when workspace_id exists but isn't indexed
            has_workspace_field = has_workspace_index
            if not has_workspace_index:
                # Sample a small batch of points to check for workspace_id in payloads
                # All points must have workspace_id if any point has it
                sample_result = client.scroll(
                    collection_name=legacy_collection,
                    limit=10,  # Small sample is sufficient for detection
                    with_payload=True,
                    with_vectors=False,
                )
                sample_points, _ = sample_result
                for point in sample_points:
                    if point.payload and WORKSPACE_ID_FIELD in point.payload:
                        has_workspace_field = True
                        logger.info(
                            f"Qdrant: Detected unindexed {WORKSPACE_ID_FIELD} field "
                            f"in legacy collection '{legacy_collection}' via payload sampling"
                        )
                        break

            # Build workspace filter if legacy collection has workspace support
            # This prevents cross-workspace data leakage during migration
            legacy_scroll_filter = None
            if has_workspace_field:
                legacy_scroll_filter = models.Filter(
                    must=[workspace_filter_condition(workspace)]
                )
                # Recount with workspace filter for accurate migration tracking
                legacy_count = client.count(
                    collection_name=legacy_collection,
                    count_filter=legacy_scroll_filter,
                    exact=True,
                ).count
                logger.info(
                    f"Qdrant: Legacy collection has workspace support, "
                    f"filtering to {legacy_count} records for workspace '{workspace}'"
                )

            logger.info(
                f"Qdrant: Found legacy collection '{legacy_collection}' with {legacy_count} records to migrate."
            )
            logger.info(
                f"Qdrant: Migrating data from legacy collection '{legacy_collection}' to new collection '{collection_name}'"
            )

            try:
                # Batch migration (500 records per batch)
                migrated_count = 0
                offset = None
                batch_size = 500

                while True:
                    # Scroll through legacy data with optional workspace filter
                    result = client.scroll(
                        collection_name=legacy_collection,
                        scroll_filter=legacy_scroll_filter,
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
                        # Set workspace_id in payload
                        new_payload = dict(point.payload or {})
                        new_payload[WORKSPACE_ID_FIELD] = workspace

                        # Create new point with workspace-prefixed ID
                        original_id = new_payload.get(ID_FIELD)
                        if original_id:
                            new_point_id = compute_mdhash_id_for_qdrant(
                                original_id, prefix=workspace
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
                    logger.info(
                        f"Qdrant: {migrated_count}/{legacy_count} records migrated"
                    )

                    # Check if we've reached the end
                    if next_offset is None:
                        break
                    offset = next_offset

                new_count_after = client.count(
                    collection_name=collection_name,
                    count_filter=workspace_count_filter,
                    exact=True,
                ).count
                inserted_count = new_count_after - new_workspace_count
                if inserted_count != legacy_count:
                    error_msg = (
                        "Qdrant: Migration verification failed, expected "
                        f"{legacy_count} inserted records, got {inserted_count}."
                    )
                    logger.error(error_msg)
                    raise DataMigrationError(error_msg)

            except DataMigrationError:
                # Re-raise DataMigrationError as-is to preserve specific error messages
                raise
            except Exception as e:
                logger.error(
                    f"Qdrant: Failed to migrate data from legacy collection '{legacy_collection}' to new collection '{collection_name}': {e}"
                )
                raise DataMigrationError(
                    f"Failed to migrate data from legacy collection '{legacy_collection}' to new collection '{collection_name}'"
                ) from e

            logger.info(
                f"Qdrant: Migration from '{legacy_collection}' to '{collection_name}' completed successfully"
            )
            logger.warning(
                "Qdrant: Manual deletion is required after data migration verification."
            )

    def __post_init__(self):
        self._validate_embedding_func()
        # Check for QDRANT_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Qdrant storage instances
        qdrant_workspace = os.environ.get("QDRANT_WORKSPACE")
        if qdrant_workspace and qdrant_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = qdrant_workspace.strip()
            logger.info(
                f"Using QDRANT_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        self.effective_workspace = effective_workspace or DEFAULT_WORKSPACE

        # Generate model suffix
        self.model_suffix = self._generate_collection_suffix()

        # New naming scheme with model isolation
        # Example: "lightrag_vdb_chunks_text_embedding_ada_002_1536d"
        # Ensure model_suffix is not empty before appending
        if self.model_suffix:
            self.final_namespace = f"lightrag_vdb_{self.namespace}_{self.model_suffix}"
            logger.info(f"Qdrant collection: {self.final_namespace}")
        else:
            # Fallback: use legacy namespace if model_suffix is unavailable
            self.final_namespace = f"lightrag_vdb_{self.namespace}"
            logger.warning(
                f"Qdrant collection: {self.final_namespace} missing suffix. Pls add model_name to embedding_func for proper workspace data isolation."
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
        self._max_upsert_payload_bytes = int(
            os.getenv(
                "QDRANT_UPSERT_MAX_PAYLOAD_BYTES",
                str(DEFAULT_QDRANT_UPSERT_MAX_PAYLOAD_BYTES),
            )
        )
        self._max_upsert_points_per_batch = int(
            os.getenv(
                "QDRANT_UPSERT_MAX_POINTS_PER_BATCH",
                str(DEFAULT_QDRANT_UPSERT_MAX_POINTS_PER_BATCH),
            )
        )
        self._max_delete_points_per_batch = int(
            os.getenv(
                "QDRANT_DELETE_MAX_POINTS_PER_BATCH",
                str(DEFAULT_QDRANT_DELETE_MAX_POINTS_PER_BATCH),
            )
        )
        if self._max_upsert_payload_bytes <= 0:
            logger.warning(
                f"QDRANT_UPSERT_MAX_PAYLOAD_BYTES={self._max_upsert_payload_bytes} is non-positive, disable payload-size splitting"
            )
        if self._max_upsert_points_per_batch <= 0:
            logger.warning(
                f"QDRANT_UPSERT_MAX_POINTS_PER_BATCH={self._max_upsert_points_per_batch} is non-positive, disable point-count splitting"
            )
        if self._max_delete_points_per_batch <= 0:
            logger.warning(
                f"QDRANT_DELETE_MAX_POINTS_PER_BATCH={self._max_delete_points_per_batch} is non-positive, disable delete point-count splitting"
            )
        self._initialized = False

        # Deferred-embedding buffers and the per-namespace flush lock.
        # Qdrant partitions a single physical collection across workspaces
        # via the workspace_id payload field, so the lock must include the
        # effective workspace (not just final_namespace) to avoid letting
        # two effectively-different writers race on the same collection.
        self._pending_vector_docs: dict[str, _PendingVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        self._flush_lock = None

    @staticmethod
    def _to_json_serializable(value: Any) -> Any:
        """Convert nested values to JSON-serializable types for payload size estimation."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {
                str(k): QdrantVectorDBStorage._to_json_serializable(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [QdrantVectorDBStorage._to_json_serializable(v) for v in value]
        return value

    @staticmethod
    def _estimate_point_payload_bytes(point: models.PointStruct) -> int:
        """Estimate serialized JSON byte size of a single Qdrant point."""
        point_obj = {
            "id": point.id,
            "vector": QdrantVectorDBStorage._to_json_serializable(point.vector),
            "payload": QdrantVectorDBStorage._to_json_serializable(point.payload or {}),
        }
        return len(
            json.dumps(
                point_obj,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    @staticmethod
    def _build_upsert_batches(
        points: list[models.PointStruct],
        max_payload_bytes: int,
        max_points_per_batch: int,
    ) -> list[tuple[list[models.PointStruct], int]]:
        """Split points into batches using payload size and point count limits.

        The byte budget is the primary limiter; the point count is a secondary
        guard. A single point larger than the byte budget is emitted as its own
        single-point batch rather than raising: the JSON estimate is
        conservative (and the default budget sits well below the real
        server/gateway limit), so the request may still be accepted. Leaving the
        server as the final arbiter avoids failing the entire flush over one
        oversized point, which would also block every healthy point buffered
        alongside it from ever committing.
        """
        if not points:
            return []

        payload_limit = max_payload_bytes if max_payload_bytes > 0 else float("inf")
        points_limit = (
            max_points_per_batch if max_points_per_batch > 0 else float("inf")
        )

        batches: list[tuple[list[models.PointStruct], int]] = []
        current_batch: list[models.PointStruct] = []
        # JSON array overhead ("[]")
        current_estimated_bytes = 2

        for point in points:
            point_size = QdrantVectorDBStorage._estimate_point_payload_bytes(point)

            # If current batch not empty, a comma is needed before next element.
            separator_overhead = 1 if current_batch else 0
            next_batch_size = current_estimated_bytes + separator_overhead + point_size

            if current_batch and (
                len(current_batch) >= points_limit or next_batch_size > payload_limit
            ):
                batches.append((current_batch, current_estimated_bytes))
                current_batch = []
                current_estimated_bytes = 2
                next_batch_size = current_estimated_bytes + point_size

            current_batch.append(point)
            current_estimated_bytes = next_batch_size

        if current_batch:
            batches.append((current_batch, current_estimated_bytes))

        return batches

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
                # Pass namespace and workspace for backward-compatible migration support
                QdrantVectorDBStorage.setup_collection(
                    self._client,
                    self.final_namespace,
                    namespace=self.namespace,
                    workspace=self.effective_workspace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=0,
                    ),
                    model_suffix=self.model_suffix,
                )

                # Removed duplicate max batch size initialization

                self._initialized = True
                logger.info(
                    f"[{self.workspace}] Qdrant collection '{self.namespace}' initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize Qdrant collection '{self.namespace}': {e}"
                )
                raise

        if self._flush_lock is None:
            self._flush_lock = get_namespace_lock(
                namespace=self.final_namespace,
                workspace=self.effective_workspace,
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush.

        Embedding deliberately does NOT happen here: repeated upserts of
        the same id, or many small batches, collapse into a single
        flush-time embedding pass. The buffer is keyed by the caller's
        original doc id; the Qdrant UUID conversion runs at flush time.
        """
        if not data:
            return

        import time

        current_time = int(time.time())

        pending_docs: list[tuple[str, _PendingVectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            source = {
                ID_FIELD: k,
                WORKSPACE_ID_FIELD: self.effective_workspace,
                CREATED_AT_FIELD: current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            pending_docs.append(
                (
                    k,
                    _PendingVectorDoc(source=source, content=v["content"]),
                )
            )
            await _cooperative_yield(i)

        # An upsert overrides any pending delete on the same id; installing
        # a fresh _PendingVectorDoc invalidates any vector cached by a
        # prior get_vectors_by_ids() call on a stale revision.
        async with self._flush_lock:
            for doc_id, pdoc in pending_docs:
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pdoc

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Query the vector database via Qdrant ``query_points``.

        Reads from the server-side index only; buffered upserts and deletes
        are NOT visible until ``index_done_callback`` / ``finalize`` flushes
        them. Callers that need read-your-writes for a freshly upserted id
        should use ``get_by_id`` / ``get_by_ids`` (which consult the buffer)
        or flush first. Matches the deferred-embedding contract used by the
        other lazy-embedding backends (Mongo / OpenSearch / FAISS / Nano).
        """
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding_result = await self.embedding_func(
                [query], context="query", _priority=5
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
        """Flush buffered vector ops; Qdrant persists automatically once written."""
        await self._flush_pending_vector_ops()

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes via batched client calls.

        Embedding runs *inside* this lock (not in `upsert` or lock-free):
        it makes deferred embedding and the upsert atomic against
        concurrent upserts and destructive mutations. Reuses
        ``_build_upsert_batches`` to respect Qdrant's payload size limit.
        Any failure (embed or server write) raises and leaves both
        buffers intact; the next ``index_done_callback`` retries.

        Concurrency invariant: ``_flush_lock`` is a non-reentrant asyncio
        lock. Callers MUST NOT hold it when invoking this method --
        re-entry would deadlock. The only in-tree callers are
        ``index_done_callback`` and ``finalize``, both lock-free.
        """
        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self._client is None:
                return

            pending_docs = self._pending_vector_docs
            pending_deletes = self._pending_vector_deletes

            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = [
                (doc_id, pdoc)
                for doc_id, pdoc in pending_docs.items()
                if pdoc.vector is None
            ]

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es) "
                    f"(batch_num={self._max_batch_size})"
                )
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise

                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((_, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    # Cache the raw numpy row so a second flush after a
                    # server-side error doesn't re-embed.
                    pdoc.vector = np.array(embedding, dtype=np.float32).tolist()
                    await _cooperative_yield(i)

            # Build PointStruct list, converting caller-supplied ids to
            # Qdrant UUIDs only now (the buffer keeps caller ids so
            # read-your-writes works against the same key).
            list_points: list[models.PointStruct] = []
            committed_ids: list[str] = []
            for doc_id, pdoc in pending_docs.items():
                if pdoc.vector is None:
                    continue
                committed_ids.append(doc_id)
                list_points.append(
                    models.PointStruct(
                        id=compute_mdhash_id_for_qdrant(
                            doc_id, prefix=self.effective_workspace
                        ),
                        vector=pdoc.vector,
                        payload=dict(pdoc.source),
                    )
                )

            try:
                if list_points:
                    point_batches = self._build_upsert_batches(
                        list_points,
                        max_payload_bytes=self._max_upsert_payload_bytes,
                        max_points_per_batch=self._max_upsert_points_per_batch,
                    )

                    if len(point_batches) > 1:
                        logger.info(
                            f"[{self.workspace}] Qdrant upsert split into {len(point_batches)} batches "
                            f"for {len(list_points)} records (max_payload={self._max_upsert_payload_bytes}, "
                            f"batch={self._max_upsert_points_per_batch})"
                        )

                    for batch_index, (points_batch, estimated_bytes) in enumerate(
                        point_batches, 1
                    ):
                        if (
                            len(points_batch) == 1
                            and self._max_upsert_payload_bytes > 0
                            and estimated_bytes > self._max_upsert_payload_bytes
                        ):
                            logger.warning(
                                f"[{self.workspace}] {self.namespace} flush: single point "
                                f"id={points_batch[0].id} estimated {estimated_bytes} bytes "
                                f"exceeds QDRANT_UPSERT_MAX_PAYLOAD_BYTES="
                                f"{self._max_upsert_payload_bytes}; sending as its own batch"
                            )
                        logger.debug(
                            f"[{self.workspace}] Qdrant upsert batch {batch_index}/{len(point_batches)}: "
                            f"points={len(points_batch)}, estimated_payload_bytes={estimated_bytes}"
                        )
                        # Fail-fast: any batch failure raises immediately
                        # and stops subsequent batches; the full buffer is
                        # retained so the next flush retries.
                        self._client.upsert(
                            collection_name=self.final_namespace,
                            points=points_batch,
                            wait=True,
                        )

                if pending_deletes:
                    qdrant_delete_ids = [
                        compute_mdhash_id_for_qdrant(
                            doc_id, prefix=self.effective_workspace
                        )
                        for doc_id in pending_deletes
                    ]
                    # Chunk deletes by point count; ids are short so a count cap
                    # is enough to keep each request under the server limit.
                    delete_chunk = (
                        self._max_delete_points_per_batch
                        if self._max_delete_points_per_batch > 0
                        else len(qdrant_delete_ids)
                    )
                    for i in range(0, len(qdrant_delete_ids), delete_chunk):
                        self._client.delete(
                            collection_name=self.final_namespace,
                            points_selector=models.PointIdsList(
                                points=qdrant_delete_ids[i : i + delete_chunk]
                            ),
                            wait=True,
                        )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error flushing vector ops "
                    f"(upserts={len(pending_docs)}, "
                    f"deletes={len(pending_deletes)}): {e}"
                )
                raise

            for doc_id in committed_ids:
                pending_docs.pop(doc_id, None)
            pending_deletes.clear()

    async def delete(self, ids: List[str]) -> None:
        """Buffer vector deletes for batched flush."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} vectors in {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        """Buffer an entity vector delete by computing its hash ID."""
        entity_id = compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)
        async with self._flush_lock:
            self._pending_vector_docs.pop(entity_id, None)
            self._pending_vector_deletes.add(entity_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for entity {entity_name} (id={entity_id})"
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt.

        The whole method runs under ``_flush_lock`` so the server-side
        scroll + delete cannot interleave with an in-flight bulk upsert.
        Server-side failures are re-raised (no log-and-swallow): the
        caller decides whether to retry.

        Buffer semantics — post-prune with caller short-circuit contract:
            Matching pending upserts in ``_pending_vector_docs`` are
            pruned **only after** the server-side scroll+delete loop
            completes fully. If any iteration raises, the pending buffer
            is left intact so a higher-level failure does not silently
            drop buffered relation vectors that the user never told us
            to discard. The trade-off is that partial server-side
            deletes plus preserved pending upserts can re-insert deleted
            relations on the next flush — correctness therefore relies
            on the caller short-circuiting before ``index_done_callback``
            can run. The single in-tree caller ``adelete_by_entity``
            in ``utils_graph.py`` honors this: its ``except`` clause
            skips both ``delete_node`` and ``_persist_graph_updates``,
            so on failure the graph and the pending buffer stay
            consistent with the "delete never happened" state and the
            operation converges on the next retry.
        """
        async with self._flush_lock:
            if self._client is None:
                # pre-init / post-finalize: only buffer state remains, so
                # apply the delete intent there.
                for doc_id in [
                    k
                    for k, v in self._pending_vector_docs.items()
                    if v.source.get("src_id") == entity_name
                    or v.source.get("tgt_id") == entity_name
                ]:
                    self._pending_vector_docs.pop(doc_id, None)
                return

            relation_filter = models.Filter(
                must=[workspace_filter_condition(self.effective_workspace)],
                should=[
                    models.FieldCondition(
                        key="src_id", match=models.MatchValue(value=entity_name)
                    ),
                    models.FieldCondition(
                        key="tgt_id", match=models.MatchValue(value=entity_name)
                    ),
                ],
            )

            total_deleted = 0
            offset = None
            batch_size = 1000

            while True:
                results = self._client.scroll(
                    collection_name=self.final_namespace,
                    scroll_filter=relation_filter,
                    with_payload=False,
                    with_vectors=False,
                    limit=batch_size,
                    offset=offset,
                )

                points, next_offset = results
                if not points:
                    break

                ids_to_delete = [point.id for point in points]
                self._client.delete(
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                    wait=True,
                )
                total_deleted += len(ids_to_delete)

                if next_offset is None:
                    break
                offset = next_offset

            # Server-side scroll+delete fully succeeded — safe to prune
            # matching pending relation upserts so the next flush won't
            # re-upsert the just-deleted relations. If the loop above
            # raised, this prune is skipped and the buffer state stays
            # available for the caller's retry path.
            for doc_id in [
                k
                for k, v in self._pending_vector_docs.items()
                if v.source.get("src_id") == entity_name
                or v.source.get("tgt_id") == entity_name
            ]:
                self._pending_vector_docs.pop(doc_id, None)

            if total_deleted > 0:
                logger.debug(
                    f"[{self.workspace}] Deleted {total_deleted} relations for {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID, with read-your-writes against the buffer."""
        async with self._flush_lock:
            if id in self._pending_vector_deletes:
                return None
            pending = self._pending_vector_docs.get(id)
            if pending is not None:
                # Buffer hits return the source payload (no vector); the
                # Qdrant fallback path also returns just the payload.
                payload = dict(pending.source)
                payload.setdefault(CREATED_AT_FIELD, None)
                return payload

        try:
            qdrant_id = compute_mdhash_id_for_qdrant(
                id, prefix=self.effective_workspace
            )

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
        """Get multiple vector data by their IDs (read-your-writes), preserving order."""
        if not ids:
            return []

        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    payload = dict(pending.source)
                    payload.setdefault(CREATED_AT_FIELD, None)
                    buffered[doc_id] = payload
                    continue
                remaining.append(doc_id)

        payload_by_original_id: dict[str, dict[str, Any]] = {}
        payload_by_qdrant_id: dict[str, dict[str, Any]] = {}

        if remaining:
            try:
                qdrant_ids = [
                    compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                    for id in remaining
                ]
                results = self._client.retrieve(
                    collection_name=self.final_namespace,
                    ids=qdrant_ids,
                    with_payload=True,
                )

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
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error retrieving vector data for IDs {remaining}: {e}"
                )
                return []

        ordered_payloads: list[dict[str, Any] | None] = []
        for doc_id in ids:
            if doc_id in buffered:
                ordered_payloads.append(buffered[doc_id])
                continue
            payload = payload_by_original_id.get(str(doc_id))
            if payload is None:
                payload = payload_by_qdrant_id.get(
                    compute_mdhash_id_for_qdrant(
                        doc_id, prefix=self.effective_workspace
                    )
                )
            ordered_payloads.append(payload)
        return ordered_payloads

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings for given IDs, with read-your-writes.

        Pending docs whose vector hasn't been embedded yet are embedded
        lazily inside the lock; the resulting vector is cached on the
        buffered ``_PendingVectorDoc`` so the next flush won't re-embed.

        Visibility caveat for ids not in the buffer: the server-side
        ``retrieve`` fallback runs *outside* ``_flush_lock``. A concurrent
        ``delete()`` that lands between lock release and the server read
        only buffers the delete -- the old vector is still on disk
        until the next flush, so this method may return a stale vector
        for an id that has been buffered for deletion. This is
        best-effort read-after-uncommitted-delete and matches the
        ``query()`` contract: callers needing strict consistency must
        ``index_done_callback()`` first.
        """
        if not ids:
            return {}

        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = []
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    if pending.vector is None:
                        docs_to_embed.append((doc_id, pending))
                    else:
                        result[doc_id] = pending.vector
                    continue
                remaining.append(doc_id)

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error lazily embedding pending vectors "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((doc_id, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pdoc.vector = np.array(embedding, dtype=np.float32).tolist()
                    result[doc_id] = pdoc.vector
                    await _cooperative_yield(i)

        if not remaining:
            return result

        try:
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in remaining
            ]
            results = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_vectors=True,
                with_payload=True,
            )

            for point in results:
                if point and point.vector is not None and point.payload:
                    original_id = point.payload.get(ID_FIELD)
                    if original_id:
                        vector_data = point.vector
                        if isinstance(vector_data, np.ndarray):
                            vector_data = vector_data.tolist()
                        result[original_id] = vector_data

            return result
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting vectors: {e}")
            return result

    async def finalize(self):
        """Flush pending vector ops; surface unflushed data as RuntimeError.

        Qdrant has no client connection that needs explicit release here
        (the QdrantClient is held by the storage instance and torn down
        on GC), but we still need to fail loudly when a transient bulk
        error left writes buffered. ``_flush_pending_vector_ops`` is
        all-or-nothing: it either clears both buffers or raises with
        them intact, but we still defensively check both buffers after a
        successful flush in case a future refactor breaks that invariant.
        """
        flush_error: Exception | None = None
        try:
            await self._flush_pending_vector_ops()
        except Exception as e:
            flush_error = e

        async with self._flush_lock:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)

        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] QdrantVectorDBStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] QdrantVectorDBStorage.finalize() left "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes buffered after final flush attempt (these writes have been lost)"
            )

    async def drop(self) -> dict[str, str]:
        """Drop all vector data for the current workspace. Destructive.

        Deletes every point matching ``effective_workspace`` from the
        shared Qdrant collection ``final_namespace`` (Qdrant partitions a
        single physical collection across workspaces via the
        ``workspace_id`` payload field, so sibling workspaces on the same
        collection are untouched). The collection itself and its vector
        index are NOT recreated — they were provisioned at
        ``initialize()`` and remain in place.

        MUST only be called when ``pipeline_status`` is idle (see the
        Pipeline concurrency contract in ``AGENTS.md``); the only
        in-tree caller ``clear_documents`` enforces this.

        Pending-write buffers are cleared *before* the server-side delete
        is issued so a concurrent flush on this instance cannot resurrect
        the dropped data. As a consequence, if the server-side delete
        fails, the buffered writes are also lost — the caller cannot
        recover them by retrying ``drop()``. This matches ``drop()``'s
        contract ("discard everything for this workspace") and the other
        lazy-embedding backends.

        Caveat — only this instance's buffers are cleared. Other
        ``QdrantVectorDBStorage`` instances aliased onto the same
        ``(final_namespace, effective_workspace)`` (multi-worker
        processes, or distinct workspaces collapsed by
        ``QDRANT_WORKSPACE``) keep their own buffers; a sibling whose
        prior flush failed and left buffers intact will, on its next
        flush, upsert those stale points back into the freshly emptied
        workspace. Direct callers bypassing the idle precondition MUST
        flush every aliased instance first.

        Returns:
            dict[str, str]: ``{"status": "success"|"error", "message": str}``
        """
        try:
            async with self._flush_lock:
                # Discard buffered writes before the workspace is wiped;
                # a concurrent flush would otherwise resurrect them.
                self._pending_vector_docs.clear()
                self._pending_vector_deletes.clear()

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
