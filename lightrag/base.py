import os
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np

from .utils import EmbeddingFunc
from .types import KnowledgeGraph


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


T = TypeVar("T")


@dataclass
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval.
    """

    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""

    only_need_prompt: bool = False
    """If True, only returns the generated prompt without producing a response."""

    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    stream: bool = False
    """If True, enables streaming output for real-time responses."""

    top_k: int = int(os.getenv("TOP_K", "60"))
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    max_token_for_text_unit: int = 4000
    """Maximum number of tokens allowed for each retrieved text chunk."""

    max_token_for_global_context: int = 4000
    """Maximum number of tokens allocated for relationship descriptions in global retrieval."""

    max_token_for_local_context: int = 4000
    """Maximum number of tokens allocated for entity descriptions in local retrieval."""

    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    history_turns: int = 3
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict[str, Any]

    async def index_done_callback(self):
        """Commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """Commit the storage operations after querying"""
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

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        raise NotImplementedError

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def filter_keys(self, data: set[str]) -> set[str]:
        """Return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, Any]) -> None:
        raise NotImplementedError

    async def drop(self) -> None:
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

    async def get_all_labels(self) -> list[str]:
        raise NotImplementedError

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
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

    content: str
    """Original content of the document"""
    content_summary: str
    """First 100 chars of document content, used for preview"""
    content_length: int
    """Total length of document"""
    status: DocStatus
    """Current processing status"""
    created_at: str
    """ISO format timestamp when document was created"""
    updated_at: str
    """ISO format timestamp when document was last updated"""
    chunks_count: Optional[int] = None
    """Number of chunks after splitting, used for processing"""
    error: Optional[str] = None
    """Error message if failed"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


class DocStatusStorage(BaseKVStorage):
    """Base class for document status storage"""

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        raise NotImplementedError

    async def get_failed_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        raise NotImplementedError

    async def get_pending_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        raise NotImplementedError

    async def get_processing_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all processing documents"""
        raise NotImplementedError

    async def get_processed_docs(self) -> dict[str, DocProcessingStatus]:
        """Get all procesed documents"""
        raise NotImplementedError

    async def update_doc_status(self, data: dict[str, Any]) -> None:
        """Updates the status of a document. By default, it calls upsert."""
        await self.upsert(data)
