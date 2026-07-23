from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import os
import uuid
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Literal,
    TypedDict,
    TypeVar,
    Optional,
    Dict,
    List,
    AsyncIterator,
)
from .utils import EmbeddingFunc, get_env_value, logger
from .types import KnowledgeGraph
from .exceptions import (
    StorageCapabilityError,
    StorageRecordNotFoundError,
)
from .constants import (
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_OLLAMA_MODEL_SIZE,
    DEFAULT_OLLAMA_CREATED_AT,
    DEFAULT_OLLAMA_DIGEST,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class OllamaServerInfos:
    def __init__(self, name=None, tag=None):
        self._lightrag_name = name or os.getenv(
            "OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME
        )
        self._lightrag_tag = tag or os.getenv(
            "OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG
        )
        self.LIGHTRAG_SIZE = DEFAULT_OLLAMA_MODEL_SIZE
        self.LIGHTRAG_CREATED_AT = DEFAULT_OLLAMA_CREATED_AT
        self.LIGHTRAG_DIGEST = DEFAULT_OLLAMA_DIGEST

    @property
    def LIGHTRAG_NAME(self):
        return self._lightrag_name

    @LIGHTRAG_NAME.setter
    def LIGHTRAG_NAME(self, value):
        self._lightrag_name = value

    @property
    def LIGHTRAG_TAG(self):
        return self._lightrag_tag

    @LIGHTRAG_TAG.setter
    def LIGHTRAG_TAG(self, value):
        self._lightrag_tag = value

    @property
    def LIGHTRAG_MODEL(self):
        return f"{self._lightrag_name}:{self._lightrag_tag}"


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


T = TypeVar("T")


@dataclass
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
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

    top_k: int = get_env_value("TOP_K", DEFAULT_TOP_K, int)
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    chunk_top_k: int = get_env_value("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int)
    """Number of text chunks to retrieve initially from vector search and keep after reranking.
    If None, defaults to top_k value.
    """

    max_entity_tokens: int = get_env_value(
        "MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int
    )
    """Maximum number of tokens allocated for entity context in unified token control system."""

    max_relation_tokens: int = get_env_value(
        "MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int
    )
    """Maximum number of tokens allocated for relationship context in unified token control system."""

    max_total_tokens: int = get_env_value(
        "MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int
    )
    """Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt)."""

    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    # History mesages is only send to LLM for context, not used for retrieval
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    user_prompt: str | None = None
    """User-provided prompt for the query.
    Addition instructions for LLM. If provided, this will be inject into the prompt template.
    It's purpose is the let user customize the way LLM generate the response.
    """

    enable_rerank: bool = os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"
    """Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued.
    Default is True to enable reranking when rerank model is available.
    """

    include_references: bool = False
    """If True, includes reference list in the response for supported endpoints.
    This parameter controls whether the API response includes a references field
    containing citation information for the retrieved content.
    """


@dataclass
class StorageNameSpace(ABC):
    namespace: str
    workspace: str
    global_config: dict[str, Any]

    async def initialize(self):
        """Initialize the storage"""
        pass

    async def finalize(self):
        """Finalize the storage"""
        pass

    @abstractmethod
    async def index_done_callback(self) -> None:
        """Commit the storage operations after indexing"""

    async def drop_pending_index_ops(self) -> None:
        """Discard any not-yet-flushed buffered index ops.

        Backends that defer writes to ``index_done_callback`` (via an
        in-memory ``_pending_*`` buffer) override this to clear that buffer.
        The pipeline calls it when a batch is aborting on an internal error:
        every still-buffered record belongs to a document that is being
        marked FAILED and fully reprocessed on the next run, so dropping the
        buffer is safe and prevents the poisoned/stale records from being
        re-flushed by the remaining in-flight documents or carried over to
        the next batch. Immediate-write backends keep the default no-op.
        """
        return None

    @abstractmethod
    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources

        This abstract method defines the contract for dropping all data from a storage implementation.
        Each storage type must implement this method to:
        1. Clear all data from memory and/or external storage
        2. Remove any associated storage files if applicable
        3. Reset the storage to its initial state
        4. Handle cleanup of any resources
        5. Notify other processes if necessary
        6. This action should persistent the data to disk immediately.

        Returns:
            dict[str, str]: Operation status and message with the following format:
                {
                    "status": str,  # "success" or "error"
                    "message": str  # "data dropped" on success, error details on failure
                }

        Implementation specific:
        - On success: return {"status": "success", "message": "data dropped"}
        - On failure: return {"status": "error", "message": "<error details>"}
        - If not supported: return {"status": "error", "message": "unsupported"}
        """


@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc
    cosine_better_than_threshold: float = field(default=0.2)
    meta_fields: set[str] = field(default_factory=set)

    def _validate_embedding_func(self):
        """Validate that embedding_func is provided.

        This method should be called at the beginning of __post_init__
        in all vector storage implementations.

        Raises:
            ValueError: If embedding_func is None
        """
        if self.embedding_func is None:
            raise ValueError(
                "embedding_func is required for vector storage. "
                "Please provide a valid EmbeddingFunc instance."
            )

    def _generate_collection_suffix(self) -> str | None:
        """Generates collection/table suffix from embedding_func.

        Return suffix if model_name exists in embedding_func, otherwise return None.
        Note: embedding_func is guaranteed to exist (validated in __post_init__).

        Returns:
            str | None: Suffix string e.g. "text_embedding_3_large_3072d", or None if model_name not available
        """
        import re

        # Check if model_name exists (model_name is optional in EmbeddingFunc)
        model_name = getattr(self.embedding_func, "model_name", None)
        if not isinstance(model_name, str):
            return None

        model_name = model_name.strip()
        if not model_name:
            return None

        # embedding_dim is required in EmbeddingFunc
        embedding_dim = self.embedding_func.embedding_dim

        # Generate suffix: clean model name and append dimension
        safe_model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.lower())
        return f"{safe_model_name}_{embedding_dim}d"

    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage and retrieve top_k results.

        Args:
            query: The query string to search for
            top_k: Number of top results to return
            query_embedding: Optional pre-computed embedding for the query.
                           If provided, skips embedding computation for better performance.
        """

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vectors in the storage.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Multi-worker note:
            Backends that buffer writes in process memory (e.g.
            OpenSearchVectorDBStorage as of #3043) keep the buffer
            process-local. In a multi-worker deployment (e.g.
            lightrag-gunicorn) other workers will not observe these writes
            until the writing worker has called index_done_callback().
            Callers that depend on cross-worker read-after-write visibility
            must explicitly await index_done_callback() before relying on
            reads from another worker.
        """

    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None:
        """Delete a single entity by its name.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Multi-worker note: see ``upsert`` -- buffered tombstones are
        process-local until index_done_callback() runs.
        """

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Multi-worker note: see ``upsert`` -- backends may prune their
        in-process buffer in addition to issuing a server-side delete,
        so cross-worker visibility still follows the index_done_callback
        contract.
        """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Multi-worker note: see ``upsert`` -- buffered tombstones are
        process-local until index_done_callback() runs.

        Args:
            ids: List of vector IDs to be deleted
        """

    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        pass


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc

    supports_strict_point_reads: ClassVar[bool] = False
    """Class-level capability flag: the backend implements
    :meth:`get_by_id_strict` with complete-or-raise semantics.

    ``False`` (the base default, and any third-party backend that has not
    opted in) means strict point reads are unavailable — callers that would
    take a destructive action on "confirmed absent" (e.g. deleting a FAILED
    doc_status stub because its full_docs entry is gone) MUST NOT act and
    should fall back to a safe path with a warning instead.
    """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get value by id"""

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        """Point read with complete-or-raise semantics.

        Contract (implemented by backends that declare
        ``supports_strict_point_reads = True``):

        * ``None`` means **confirmed absent** — the backend positively
          determined that no record with this id exists.
        * Any transport/server error, an index/collection that is not ready,
          or a state where absence cannot be positively confirmed (e.g. an
          OpenSearch index that is unexpectedly missing after a restore)
          MUST raise instead of returning ``None``.

        This differs from :meth:`get_by_id`, whose implementations may treat
        failures as a best-effort miss. The base implementation raises
        :class:`~lightrag.exceptions.StorageCapabilityError`; callers must
        gate on :attr:`supports_strict_point_reads` before calling.
        """
        raise StorageCapabilityError(
            f"{type(self).__name__} does not support strict point reads "
            "(supports_strict_point_reads=False); the caller must fall back "
            "to a non-destructive path instead of trusting a miss."
        )

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get values by ids"""

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return un-exist keys"""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Multi-worker note:
            Backends that buffer writes in process memory (e.g.
            OpenSearchKVStorage as of the KV-batching change derived from
            #2822) keep the buffer process-local. In a multi-worker
            deployment (e.g. lightrag-gunicorn) other workers will not
            observe these writes until the writing worker has called
            index_done_callback(). Callers that depend on cross-worker
            read-after-write visibility must explicitly await
            index_done_callback() before relying on reads from another
            worker.
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Multi-worker note: see ``upsert`` -- buffered tombstones are
        process-local until index_done_callback() runs.

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """

    @abstractmethod
    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage contains no data, False otherwise
        """


@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    """All operations related to edges in graph should be undirected."""

    embedding_func: EmbeddingFunc

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node exists, False otherwise
        """

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            True if the edge exists, False otherwise
        """

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node.

        Args:
            node_id: The ID of the node

        Returns:
            The number of edges connected to the node
        """

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes).

        Args:
            src_id: The ID of the source node
            tgt_id: The ID of the target node

        Returns:
            The sum of the degrees of the source and target nodes
        """

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            A dictionary of edge properties if found, None otherwise
        """

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node.

        Args:
            source_node_id: The ID of the node to get edges for

        Returns:
            A list of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get nodes as a batch using UNWIND

        Default implementation fetches nodes one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node is not None:
                result[node_id] = node
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Node degrees as a batch using UNWIND

        Default implementation fetches node degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            degree = await self.node_degree(node_id)
            result[node_id] = degree
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Edge degrees as a batch using UNWIND also uses node_degrees_batch

        Default implementation calculates edge degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for src_id, tgt_id in edge_pairs:
            degree = await self.edge_degree(src_id, tgt_id)
            result[(src_id, tgt_id)] = degree
        return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Get edges as a batch using UNWIND

        Default implementation fetches edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for pair in pairs:
            src_id = pair["src"]
            tgt_id = pair["tgt"]
            edge = await self.get_edge(src_id, tgt_id)
            if edge is not None:
                result[(src_id, tgt_id)] = edge
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Get nodes edges as a batch using UNWIND

        Default implementation fetches node edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to insert or update
            node_data: A dictionary of node properties
        """

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Insert or update multiple nodes in a single batch call.

        Default implementation falls back to calling upsert_node() serially.
        Override in storage backends that support native batch operations for
        better performance when importing large knowledge graphs.

        Args:
            nodes: List of (node_id, node_data) tuples.
        """
        for node_id, node_data in nodes:
            await self.upsert_node(node_id, node_data=node_data)

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Check existence of multiple nodes in a single batch call.

        Default implementation falls back to calling has_node() serially.
        Override in storage backends that support native batch operations for
        better performance when importing large knowledge graphs.

        Args:
            node_ids: List of node IDs to check.

        Returns:
            Set of node_ids that exist in the graph.
        """
        existing: set[str] = set()
        for node_id in node_ids:
            if await self.has_node(node_id):
                existing.add(node_id)
        return existing

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Insert or update multiple edges in a single batch call.

        Default implementation falls back to calling upsert_edge() serially.
        Override in storage backends that support native batch operations for
        better performance when importing large knowledge graphs.

        Args:
            edges: List of (source_node_id, target_node_id, edge_data) tuples.
        """
        for source_node_id, target_node_id, edge_data in edges:
            await self.upsert_edge(source_node_id, target_node_id, edge_data=edge_data)

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to delete
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

    @abstractmethod
    async def get_all_labels(self) -> list[str]:
        """Get all labels(entity names) in the graph.
        Do not use this method for large graph, use get_popular_labels or search_labels instead.

        Returns:
            A list of all node labels in the graph, sorted alphabetically
        """

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label(entity name) of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000（BFS if possible)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """

    @abstractmethod
    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
            (Edge is bidirectional for some storage implementation; deduplication must be handled by the caller)
        """

    @abstractmethod
    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """

    @abstractmethod
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels(entity names) by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """

    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels(entity names) with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """


class DocStatus(str, Enum):
    """Document processing status.
    Pipeline order: PENDING -> PARSING -> ANALYZING (optional) -> PROCESSING -> PROCESSED | FAILED.
    PREPROCESSED is deprecated, kept for backward compatibility.
    """

    PENDING = "pending"
    PARSING = "parsing"  # Phase 1: content extraction (parse_native/mineru/docling)
    ANALYZING = "analyzing"  # Phase 2: multimodal analysis (VLM)
    PROCESSING = "processing"  # Phase 3: entity/relation extraction
    PREPROCESSED = "preprocessed"  # Deprecated: use ANALYZING in new pipeline
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content_summary: str
    """First 100 chars of document content, used for preview"""
    content_length: int
    """Total length of document"""
    file_path: str
    """Canonical basename of the document.

    Always a hint-stripped basename (e.g. ``abc.docx``) or the literal
    ``"unknown_source"`` sentinel; never carries directory components or
    parser ``[hint]`` segments. UI display, filename-based dedup, and
    citation paths all share this value.
    """
    status: DocStatus
    """Current processing status"""
    created_at: str
    """ISO format timestamp when document was created"""
    updated_at: str
    """ISO format timestamp when document was last updated"""
    track_id: str | None = None
    """Tracking ID for monitoring progress"""
    chunks_count: int | None = None
    """Number of chunks after splitting, used for processing"""
    chunks_list: list[str] | None = field(default_factory=list)
    """List of chunk IDs associated with this document, used for deletion"""
    error_msg: str | None = None
    """Error message if failed"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""
    multimodal_processed: bool | None = field(default=None, repr=False)
    content_hash: str | None = None
    """MD5 hash of the underlying document content (raw text or source file).

    Used together with file_path basename for duplicate detection. Empty for
    pending_parse records whose content has not been extracted yet.
    """
    failure_generation: int | None = None
    """Monotonic failure-cohort generation assigned when the row transitioned
    to FAILED via ``mark_doc_failed`` (memory-bounding Phase 1).

    ``None`` on legacy rows and rows that never failed — query predicates
    treat missing as logical 0, so all history belongs to the first manual
    cutoff. Never reset on retry; a later failure assigns a fresh, larger
    generation.
    """
    processing_attempt_id: str | None = None
    """Identity of the current processing attempt.

    Minted before a brand-new attempt starts, kept across PENDING→PARSING→
    PROCESSING and across crash recovery of the same interrupted attempt; a
    manual retry grant mints a new one. ``mark_doc_failed`` stamps it into
    ``failure_attempt_id`` so concurrent failure writes of the same attempt
    collapse idempotently.
    """
    failure_attempt_id: str | None = None
    """The ``processing_attempt_id`` whose failure produced the current
    FAILED state (None when never failed / legacy)."""
    """Internal field: indicates if multimodal processing is complete. Not shown in repr() but accessible for debugging."""

    def __post_init__(self):
        """
        Handle status conversion based on multimodal_processed field.

        Business rules:
        - If multimodal_processed is False and status is PROCESSED,
          then change status to PREPROCESSED
        - The multimodal_processed field is kept (with repr=False) for internal use and debugging
        """
        # Apply status conversion logic
        if self.multimodal_processed is not None:
            if (
                self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED
            ):
                self.status = DocStatus.PREPROCESSED


class FailureGenerationMode(str, Enum):
    """Per-workspace activation state of the failure-generation machinery.

    Read through :meth:`DocStatusStorage.get_failure_generation_mode`. The
    manual-retry scheduler dispatches EXHAUSTIVELY on this enum — never with
    a default-else — because mapping an unknown/MIGRATING state to LEGACY
    would silently reopen the full-materialization (OOM) window:

    * ``LEGACY`` — workspace data/writers not migrated: manual retry uses the
      old single-snapshot cohort; no generation predicate.
    * ``MIGRATING`` — migration in flight or persisted version markers
      disagree: scheduling features that depend on migrated state must raise
      :class:`~lightrag.exceptions.StorageMigrationInProgressError`.
    * ``DUAL_WRITE_READY`` — writers assign generations but the read side is
      not yet authoritative: manual retry still uses the legacy snapshot.
    * ``ENFORCED`` — migration complete (schema version + counter epoch +
      migration version all consistent): manual retry uses paged
      generation-cutoff cohorts.
    """

    LEGACY = "legacy"
    MIGRATING = "migrating"
    DUAL_WRITE_READY = "dual_write_ready"
    ENFORCED = "enforced"


class CursorPosition:
    """Sealed three-state cursor for stable keyset sweeps.

    Exactly three shapes exist: the :data:`CURSOR_START` singleton, the
    :data:`CURSOR_END` singleton, and :class:`CursorAfter` (an opaque,
    backend-defined continuation token). ``page.next_position is CURSOR_END``
    is the ONLY termination signal — an empty ``docs`` dict is not (a page
    may be fully filtered yet not exhausted).
    """

    __slots__ = ()


class _CursorSentinel(CursorPosition):
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self._name


CURSOR_START = _CursorSentinel("CURSOR_START")
"""Begin a sweep from the smallest ``(created_at, id)`` key."""

CURSOR_END = _CursorSentinel("CURSOR_END")
"""The sweep is exhausted; no further page exists."""


@dataclass(frozen=True)
class CursorAfter(CursorPosition):
    """Opaque continuation token: resume strictly after the last CONSUMED
    underlying record (see the consumed-position contract on
    :meth:`DocStatusStorage.get_docs_by_statuses_page`)."""

    opaque: str


@dataclass(frozen=True)
class DocSchedulingRecord:
    """Lightweight scheduling projection of a doc_status row.

    Deliberately excludes ``chunks_list``, large ``metadata`` blobs and full
    error text so a page of records stays O(page_size × small constant) —
    the pipeline hydrates full records per-document only when it actually
    routes one.
    """

    id: str
    status: DocStatus
    created_at: str
    updated_at: str
    file_path: str
    track_id: str | None
    has_custom_chunk_journal: bool
    """True when doc_status.metadata carries the custom-chunk patch journal —
    such rows belong to scan/custom-chunk recovery, not ordinary routing."""


@dataclass(frozen=True)
class DocStatusPage:
    """One page of a stable keyset sweep.

    ``next_position is CURSOR_END`` terminates the sweep; any other value
    (including with an empty ``docs``) means "call again".
    """

    docs: dict[str, DocSchedulingRecord]
    next_position: CursorPosition


# One-time-per-class warning registry for base-default page reads that must
# ignore ``max_failure_generation`` (the class lacks failure-generation
# support). Module-level so dataclass instances stay stateless.
_PAGE_GENERATION_FILTER_WARNED: set[str] = set()


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage"""

    supports_bounded_scheduling_pages: ClassVar[bool] = False
    """The backend implements :meth:`get_docs_by_statuses_page` as a true
    bounded keyset sweep. ``False`` (base default / third-party backends)
    means the default single-page implementation is used, which materializes
    ALL matching rows — correctness holds but there is NO memory-boundedness
    guarantee. Operators can require boundedness via
    ``PIPELINE_REQUIRE_BOUNDED_SCHEDULING`` (init fail-fast) or accept a
    strong startup warning."""

    supports_failure_generation: ClassVar[bool] = False
    """The backend implements the failure-generation write side
    (:meth:`reserve_failure_generation` + CAS :meth:`mark_doc_failed`).
    Class-level implementation capability only — whether a given WORKSPACE
    actually enforces generation cohorts is the per-workspace marker read via
    :meth:`get_failure_generation_mode`."""

    supports_strict_doc_identity_lookup: ClassVar[bool] = False
    """The backend implements :meth:`get_doc_by_file_basename_strict` with
    fail-closed semantics (None == confirmed absent). ``False`` means the
    strict variant merely delegates to the legacy lookup: a ``None`` is only
    a best-effort miss and callers must not present it as confirmation."""

    @staticmethod
    def resolve_status_filter_values(
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
    ) -> set[str] | None:
        """Normalize single- and multi-status filters into comparable values.

        `status_filters` takes precedence over `status_filter`. Empty multi-status
        filters are treated as no filter for backward-compatible request handling.
        """
        if status_filters:
            return {status.value for status in status_filters}
        if status_filter is not None:
            return {status_filter.value}
        return None

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""

    @abstractmethod
    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""

    @abstractmethod
    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses.

        ``strict`` selects the completeness contract:

        * ``strict=False`` (default) — best-effort read for UI/listing paths:
          a record that fails to deserialize is logged and skipped, and a
          backend MAY return what it collected before a transport error.
        * ``strict=True`` — scheduling control-plane contract: the result is
          COMPLETE or the call raises.  Implementations must propagate any
          transport/pagination failure, any incomplete response structure and
          any record that cannot be converted to
          :class:`DocProcessingStatus` — a partial result silently consumed
          by the pipeline supervisor would strand the missed documents (the
          caller has already consumed its wake-up signal).  A legitimately
          empty result (nothing matches, index/collection not created yet)
          is complete and returns ``{}``.
        """

    @abstractmethod
    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""

    @abstractmethod
    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Legacy single-status filter, ignored when status_filters is set
            status_filters: Filter by multiple document statuses, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """

    @abstractmethod
    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts
        """

    @abstractmethod
    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            dict[str, Any] | None: Document data if found, None otherwise
            Returns the same format as get_by_ids method
        """

    @abstractmethod
    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Get document by canonical file basename.

        Used for filename-based deduplication. Callers must pass the canonical
        basename; storage implementations only compare against the canonical
        ``file_path`` persisted by the business layer.

        Args:
            basename: The filename basename to search for (e.g. "report.pdf").

        Returns:
            (doc_id, doc_data) when a matching record exists, otherwise None.
        """

    @abstractmethod
    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Get document by content_hash field.

        Used for content-hash deduplication of full documents.

        Args:
            content_hash: The content hash value to search for.

        Returns:
            (doc_id, doc_data) when a matching record exists, otherwise None.
        """

    # ------------------------------------------------------------------
    # Memory-bounding scheduling API (Phase 1). Every method below ships a
    # concrete base default so existing third-party subclasses keep
    # instantiating and working; built-in backends override for bounded /
    # fail-closed behaviour and declare the matching capability flags.
    # ------------------------------------------------------------------

    @staticmethod
    def _scheduling_record_from_status(
        doc_id: str, status_doc: DocProcessingStatus
    ) -> DocSchedulingRecord:
        """Project a full status row into the lightweight scheduling record."""
        metadata = status_doc.metadata if isinstance(status_doc.metadata, dict) else {}
        return DocSchedulingRecord(
            id=doc_id,
            status=status_doc.status,
            created_at=status_doc.created_at,
            updated_at=status_doc.updated_at,
            file_path=status_doc.file_path,
            track_id=status_doc.track_id,
            has_custom_chunk_journal=isinstance(
                metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict
            ),
        )

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        max_failure_generation: int | None = None,
        strict: bool = False,
    ) -> DocStatusPage:
        """Read one page of a stable keyset sweep over the given statuses.

        Contract for bounded implementations
        (``supports_bounded_scheduling_pages = True``):

        * **Sort (MUST)**: ``(created_at ASC, id ASC)`` keyset order across
          ALL requested statuses combined (per-status keysets merged with a
          bounded k-way merge where the backend cannot sort natively).
          Live-view: the sweep observes concurrent writes; no snapshot
          isolation is claimed. Termination is ``next_position is
          CURSOR_END`` — never an empty ``docs``.
        * **Immutable sort key**: ``created_at`` is written once at record
          creation and preserved by every later transition
          (``update_doc_status_fields`` refuses it; ``mark_doc_failed``
          ignores a caller-supplied value for existing rows) so a record can
          never move underneath a sweep.
        * **Consumed-position advance**: ``next_position`` advances past the
          last CONSUMED underlying record — records returned to the caller
          plus records read and dropped by filtering (e.g. the
          ``max_failure_generation`` predicate) are consumed; a record
          prefetched for merging but NOT returned because the page filled is
          NOT consumed (or must be encoded into the opaque cursor and
          returned first on the next page). A fully-filtered page therefore
          advances the cursor without terminating, never re-reads in place,
          and never skips a prefetched head. With multiple statuses the
          opaque cursor tracks each status stream's own consumed position.
        * ``max_failure_generation`` (optional): return FAILED rows only when
          ``failure_generation <= max_failure_generation`` (missing field ==
          logical 0). Non-FAILED rows are unaffected. This is the manual
          cohort-freeze predicate; AUTO sweeps pass ``None``.
        * ``strict=True``: the page is complete or the call raises — on an
          internal partial failure the implementation must raise WITHOUT
          returning partial docs or a new cursor. All scheduling/control-plane
          callers pass ``strict=True`` explicitly.

        Base default (third-party compatibility, NOT memory-bounded):
        ``CURSOR_START`` → delegate to ``get_docs_by_statuses(strict=True)``
        and return everything as a single page ending the sweep (``limit`` is
        ignored); any other position → empty terminal page. A non-``None``
        ``max_failure_generation`` is ignored with a one-time warning because
        the class lacks failure-generation support (missing == logical 0
        keeps every legacy row eligible, which is the safe direction).
        """
        if max_failure_generation is not None and not self.supports_failure_generation:
            cls_name = type(self).__name__
            if cls_name not in _PAGE_GENERATION_FILTER_WARNED:
                _PAGE_GENERATION_FILTER_WARNED.add(cls_name)
                logger.warning(
                    f"{cls_name} does not support failure_generation; "
                    "max_failure_generation is ignored by the base "
                    "single-page implementation (all FAILED rows stay "
                    "eligible for this sweep)."
                )
        if position is not CURSOR_START:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        snapshot = await self.get_docs_by_statuses(statuses, strict=True)
        docs = {
            doc_id: self._scheduling_record_from_status(doc_id, status_doc)
            for doc_id, status_doc in snapshot.items()
        }
        return DocStatusPage(docs=docs, next_position=CURSOR_END)

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        """Count documents in the given statuses, fail-closed.

        Unlike ``get_status_counts()`` implementations that swallow errors
        and report zeros, this method MUST either return an accurate count or
        raise — admission control treats an error as "refuse", never as
        "capacity available". The base implementation raises
        :class:`~lightrag.exceptions.StorageCapabilityError`; deployments
        that enable ``MAX_PENDING_DOCUMENTS`` validate support at
        initialization.
        """
        raise StorageCapabilityError(
            f"{type(self).__name__} does not implement strict "
            "count_docs_by_statuses; admission control cannot run on this "
            "backend."
        )

    async def update_doc_status_fields(
        self,
        doc_id: str,
        fields: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        """Update only the given fields of one doc_status record.

        Contract:

        * Fields not present in ``fields`` are left untouched — this is the
          targeted alternative to read-modify-write upserts that would drag
          a huge ``chunks_list`` through memory.
        * Implementations MUST atomically maintain every secondary index
          affected by the updated fields (e.g. Redis per-status ZSETs and the
          basename→primary-doc index) in the same transaction as the write.
        * ``created_at`` is an immutable sort key: passing it raises
          ``ValueError`` (see the keyset-sweep contract).
        * Unknown ``doc_id`` raises
          :class:`~lightrag.exceptions.StorageRecordNotFoundError` unless
          ``missing_ok=True`` (best-effort callers only).

        The base default performs a read-modify-write via ``get_by_id`` +
        ``upsert`` — correct but not memory-optimal; built-in backends
        override with native partial updates.
        """
        if "created_at" in fields:
            raise ValueError(
                "created_at is an immutable scheduling sort key and cannot "
                "be changed via update_doc_status_fields"
            )
        existing = await self.get_by_id(doc_id)
        if existing is None:
            if missing_ok:
                return
            raise StorageRecordNotFoundError(doc_id)
        merged = {**existing, **fields}
        merged.pop("_id", None)
        await self.upsert({doc_id: merged})

    async def get_failure_generation_mode(self) -> FailureGenerationMode:
        """Read the per-workspace failure-generation activation marker.

        Built-in backends read a persisted per-workspace marker (never a
        local config override) and MUST propagate read failures as
        control-plane errors — a marker read failure never degrades to
        ``LEGACY`` because LEGACY reopens the full-snapshot manual path.

        The base default returns ``LEGACY``: third-party backends keep the
        old single-snapshot manual behaviour with plain FAILED upserts.
        """
        return FailureGenerationMode.LEGACY

    async def reserve_failure_generation(self) -> int:
        """Atomically reserve the next failure-generation number.

        Reserve-before-publish: the reservation is the linearization point of
        a failure event; a reservation whose FAILED write later fails becomes
        a permanent hole (never reused, never rolled back). Counter
        durability must be no weaker than the FAILED status write itself.

        Only called on ``ENFORCED`` workspaces, so the base default (no
        generation support) raises
        :class:`~lightrag.exceptions.StorageCapabilityError`.
        """
        raise StorageCapabilityError(
            f"{type(self).__name__} does not support failure_generation "
            "reservation (supports_failure_generation=False)."
        )

    async def mark_doc_failed(self, doc_id: str, fields: dict[str, Any]) -> int | None:
        """Transition one document to FAILED.

        This is the SINGLE funnel for FAILED writes (Phase 1 routes every
        call site through it). Contract for generation-capable backends
        (``supports_failure_generation = True``):

        * Reserve a generation (:meth:`reserve_failure_generation`) and CAS
          it into the row together with ``failure_attempt_id`` = the row's
          current ``processing_attempt_id``; two concurrent failures of the
          same attempt land exactly one effective generation (idempotent:
          already-FAILED with the same attempt returns the existing
          generation).
        * For an EXISTING row a caller-supplied ``created_at`` is IGNORED —
          the scheduling sort key is immutable.
        * A missing row is conditionally created as FAILED (parse/enqueue
          errors can fail before the PENDING row landed).
        * Returns the effective failure generation.

        The base default implements the LEGACY write side: a plain merge +
        upsert with ``status=FAILED`` (no generation isolation), returning
        ``None``.
        """
        existing = await self.get_by_id(doc_id)
        if existing is None:
            row = dict(fields)
        else:
            row = {**existing, **fields}
            # Immutable sort key: never let a failure rewrite move the row.
            if "created_at" in existing:
                row["created_at"] = existing["created_at"]
        row["status"] = DocStatus.FAILED
        row.pop("_id", None)
        await self.upsert({doc_id: row})
        return None

    async def ensure_processing_attempt_id(self, doc_id: str) -> str:
        """Return the row's ``processing_attempt_id``, minting one if absent.

        Idempotent per attempt: an interrupted call retried after a lost
        response reuses the persisted id instead of minting a second one.
        Used to bootstrap legacy active rows (PENDING/PARSING/ANALYZING/
        PROCESSING without an attempt id) before recovery processing, and by
        new documents before any step that can fail.

        The base default is read-then-write (not atomic across processes);
        built-in backends may override with a CAS variant.
        """
        existing = await self.get_by_id(doc_id)
        if existing is None:
            raise StorageRecordNotFoundError(doc_id)
        attempt_id = existing.get("processing_attempt_id")
        if attempt_id:
            return str(attempt_id)
        attempt_id = uuid.uuid4().hex
        await self.update_doc_status_fields(
            doc_id, {"processing_attempt_id": attempt_id}
        )
        return attempt_id

    async def get_doc_by_file_basename_strict(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Fail-closed variant of :meth:`get_doc_by_file_basename`.

        Contract for backends declaring
        ``supports_strict_doc_identity_lookup = True``:

        * Returns the PRIMARY (non-duplicate, ``metadata.is_duplicate !=
          true``) document row for the canonical basename, or ``None`` when
          absence was positively confirmed.
        * Query failures, a not-ready index, or a schema/migration version
          mismatch raise a control-plane error instead of returning ``None``
          — scan classification and enqueue dedup treat ``None`` as
          "confirmed new", so a swallowed failure would mint duplicate rows.

        The base default DELEGATES to the legacy
        :meth:`get_doc_by_file_basename` (old signature untouched — never
        call the legacy method with new kwargs, third-party overrides would
        raise ``TypeError``). Under delegation a ``None`` is only a
        best-effort miss with the backend's own error semantics; deployments
        see a startup warning and status-endpoint exposure for this
        degradation.
        """
        return await self.get_doc_by_file_basename(basename)


class StoragesStatus(str, Enum):
    """Storages status"""

    NOT_CREATED = "not_created"
    CREATED = "created"
    INITIALIZED = "initialized"
    FINALIZED = "finalized"


@dataclass
class DeletionResult:
    """Represents the result of a deletion operation."""

    status: Literal["success", "not_found", "not_allowed", "fail"]
    doc_id: str
    message: str
    status_code: int = 200
    file_path: str | None = None


# Unified Query Result Data Structures for Reference List Support


@dataclass
class QueryResult:
    """
    Unified query result data structure for all query modes.

    Attributes:
        content: Text content for non-streaming responses
        response_iterator: Streaming response iterator for streaming responses
        raw_data: Complete structured data including references and metadata
        is_streaming: Whether this is a streaming result
    """

    content: Optional[str] = None
    response_iterator: Optional[AsyncIterator[str]] = None
    raw_data: Optional[Dict[str, Any]] = None
    is_streaming: bool = False

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """
        Convenient property to extract reference list from raw_data.

        Returns:
            List[Dict[str, str]]: Reference list in format:
            [{"reference_id": "1", "file_path": "/path/to/file.pdf"}, ...]
        """
        if self.raw_data:
            return self.raw_data.get("data", {}).get("references", [])
        return []

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Convenient property to extract metadata from raw_data.

        Returns:
            Dict[str, Any]: Query metadata including query_mode, keywords, etc.
        """
        if self.raw_data:
            return self.raw_data.get("metadata", {})
        return {}


@dataclass
class QueryContextResult:
    """
    Unified query context result data structure.

    Attributes:
        context: LLM context string
        raw_data: Complete structured data including reference_list
    """

    context: str
    raw_data: Dict[str, Any]

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """Convenient property to extract reference list from raw_data."""
        return self.raw_data.get("data", {}).get("references", [])
