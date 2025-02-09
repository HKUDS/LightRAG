import os
from dataclasses import dataclass, field
from typing import (
    TypedDict,
    Union,
    Literal,
    TypeVar,
    Any,
)


from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    # Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode.
    top_k: int = int(os.getenv("TOP_K", "60"))
    # Number of document chunks to retrieve.
    # top_n: int = 10
    # Number of tokens for the original chunks.
    max_token_for_text_unit: int = 4000
    # Number of tokens for the relationship descriptions
    max_token_for_global_context: int = 4000
    # Number of tokens for the entity descriptions
    max_token_for_local_context: int = 4000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    # Conversation history support
    conversation_history: list[dict] = field(
        default_factory=list
    )  # Format: [{"role": "user/assistant", "content": "message"}]
    history_turns: int = (
        3  # Number of complete conversation turns (user-assistant pairs) to consider
    )


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict[str, Any]

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> dict[str, Any]:
        raise NotImplementedError

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, Any]) -> None:
        raise NotImplementedError

    async def drop(self) -> None:
        raise NotImplementedError

    async def get_by_status(
        self, status: str
    ) -> Union[list[dict[str, Any]], None]:
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")

    async def get_all_labels(self) -> List[str]:
        raise NotImplementedError

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> Dict[str, List[Dict]]:
        raise NotImplementedError


class DocStatus(str, Enum):
    """Document processing status enum"""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content_summary: str  # First 100 chars of document content
    content_length: int  # Total length of document
    status: DocStatus  # Current processing status
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    chunks_count: Optional[int] = None  # Number of chunks after splitting
    error: Optional[str] = None  # Error message if failed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class DocStatusStorage(BaseKVStorage):
    """Base class for document status storage"""

    async def get_status_counts(self) -> Dict[str, int]:
        """Get counts of documents in each status"""
        raise NotImplementedError

    async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        raise NotImplementedError

    async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        raise NotImplementedError
