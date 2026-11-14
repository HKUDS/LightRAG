"""No-op vector storage for graph-only ingestion workflows."""

from dataclasses import dataclass
from typing import Any, ClassVar, final

from lightrag.base import BaseVectorStorage
from lightrag.exceptions import StorageCapabilityError


@final
@dataclass
class NoopVectorDBStorage(BaseVectorStorage):
    """Accept vector storage mutations without embedding or persistence.

    Use this backend when ingestion should build only the graph and KV stores.
    Configure a persistent vector backend and run ``lightrag-rebuild-vdb``
    before using retrieval modes that query vector indexes.
    """

    requires_embedding_func: ClassVar[bool] = False
    persists_vectors: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self._validate_embedding_func()

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        raise StorageCapabilityError(
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

    async def is_empty(self) -> bool:
        """Refuse the question rather than answer the literal truth.

        This container really does hold nothing, so ``True`` would be
        accurate -- and it is the one answer that lets the startup gate refuse
        a deployment. Graph-only ingestion is a supported configuration whose
        vector stores are empty by design, so an accurate answer here is a
        false outage waiting for the day a caller forgets to check
        ``persists_vectors`` first. The gate does check it; this makes the
        mistake impossible rather than merely absent.
        """
        raise StorageCapabilityError(
            "NoopVectorDBStorage holds no vectors by design, so its emptiness "
            "is not evidence about any embedding space. Check "
            "`persists_vectors` before asking."
        )

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        return {
            "status": "success",
            "message": "Noop vector storage contains no data",
        }
