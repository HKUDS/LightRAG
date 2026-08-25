"""DocumentDB storage backends using its MongoDB-compatible wire protocol."""

from __future__ import annotations

import asyncio
import os
from typing import ClassVar, final

from pymongo.errors import PyMongoError

from ..utils import logger
from .mongo_impl import (
    ClientManager,
    _MongoDocStatusStorageBase,
    _MongoGraphStorageBase,
    _MongoKVStorageBase,
    _MongoVectorDBStorageBase,
)


class DocumentDBClientManager(ClientManager):
    """Keep DocumentDB connections isolated from MongoDB connections."""

    _instances: ClassVar[dict] = {"client": None, "db": None, "ref_count": 0}
    _lock = asyncio.Lock()
    uri_env_var = "DOCUMENTDB_URI"
    database_env_var = "DOCUMENTDB_DATABASE"
    config_section = "documentdb"
    default_uri = "mongodb://localhost:10260/"
    default_database = "LightRAG"
    driver_name = "LightRAG-DocumentDB"


class _DocumentDBStorageMixin:
    client_manager: ClassVar[type[ClientManager]] = DocumentDBClientManager
    workspace_env_var: ClassVar[str] = "DOCUMENTDB_WORKSPACE"


@final
class DocumentDBKVStorage(_DocumentDBStorageMixin, _MongoKVStorageBase):
    """DocumentDB-backed key-value storage."""


@final
class DocumentDBDocStatusStorage(
    _DocumentDBStorageMixin, _MongoDocStatusStorageBase
):
    """DocumentDB-backed document status storage."""

    supports_collation_indexes: ClassVar[bool] = False
    supports_query_collation: ClassVar[bool] = False


@final
class DocumentDBGraphStorage(_DocumentDBStorageMixin, _MongoGraphStorageBase):
    """DocumentDB-backed graph storage using native ``$graphLookup``."""

    async def create_search_index_if_not_exists(self) -> None:
        # DocumentDB does not implement Atlas createSearchIndexes. Entity lookup
        # already falls back to the Mongo protocol regex path when no index exists.
        logger.debug(
            f"[{self.workspace}] DocumentDB graph search uses the regex fallback"
        )


@final
class DocumentDBVectorDBStorage(
    _DocumentDBStorageMixin, _MongoVectorDBStorageBase
):
    """DocumentDB vector storage using native HNSW ``cosmosSearch`` indexes."""

    vector_query_uses_index_name: ClassVar[bool] = False
    vector_query_uses_cosmos_search: ClassVar[bool] = True

    async def create_vector_index_if_not_exists(self) -> None:
        expected_dimensions = self.embedding_func.embedding_dim

        try:
            indexes_cursor = await self._data.list_indexes()
            indexes = await indexes_cursor.to_list(length=None)
            for index in indexes:
                if index.get("name") != self._index_name:
                    continue

                options = index.get("cosmosSearchOptions", {})
                existing_dimensions = options.get("dimensions")
                if (
                    existing_dimensions is not None
                    and existing_dimensions != expected_dimensions
                ):
                    raise ValueError(
                        f"Vector dimension mismatch! Index '{self._index_name}' has "
                        f"dimension {existing_dimensions}, but current embedding model "
                        f"expects dimension {expected_dimensions}. Drop the existing "
                        "index or use an embedding model with matching dimensions."
                    )
                return

            hnsw_m = int(os.environ.get("DOCUMENTDB_HNSW_M", "16"))
            hnsw_ef_construction = int(
                os.environ.get("DOCUMENTDB_HNSW_EF_CONSTRUCTION", "64")
            )
            command = {
                "createIndexes": self._collection_name,
                "indexes": [
                    {
                        "key": {"vector": "cosmosSearch"},
                        "name": self._index_name,
                        "cosmosSearchOptions": {
                            "kind": "vector-hnsw",
                            "m": hnsw_m,
                            "efConstruction": hnsw_ef_construction,
                            "similarity": "COS",
                            "dimensions": expected_dimensions,
                        },
                    }
                ],
            }
            await self.db.command(command)
            logger.info(
                f"[{self.workspace}] DocumentDB vector index "
                f"{self._index_name} created successfully"
            )
        except PyMongoError as exc:
            error_msg = (
                f"[{self.workspace}] Error creating DocumentDB vector index "
                f"{self._index_name}: {exc}"
            )
            logger.error(error_msg)
            raise SystemExit(
                "Failed to create DocumentDB vector index. "
                f"Program cannot continue. {error_msg}"
            ) from exc