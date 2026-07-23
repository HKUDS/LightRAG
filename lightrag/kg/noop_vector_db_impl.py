"""No-op vector storage for graph-only ingestion workflows."""

from dataclasses import dataclass
from typing import Any, ClassVar, final

from lightrag.base import BaseVectorStorage


@final
@dataclass
class NoopVectorDBStorage(BaseVectorStorage):
    """Accept vector storage mutations without embedding or persistence.

    Use this backend when ingestion should build only the graph and KV stores.
    Configure a persistent vector backend and run ``lightrag-rebuild-vdb``
    before using retrieval modes that query vector indexes.
    """

    supports_vector_queries: ClassVar[bool] = False

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        raise RuntimeError(
            "Vector retrieval is disabled by NoopVectorDBStorage. "
            "Configure a persistent vector storage and run "
            "`lightrag-rebuild-vdb` before querying."
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        return None

    async def delete(self, ids: list[str]) -> None:
        return None

    async def delete_entity(self, entity_name: str) -> None:
        return None

    async def delete_entity_relation(self, entity_name: str) -> None:
        return None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return [None] * len(ids)

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        return {}

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        return {
            "status": "success",
            "message": "Noop vector storage contains no data",
        }
