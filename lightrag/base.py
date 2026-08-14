from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    ClassVar,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    Optional,
    Dict,
    List,
    AsyncIterator,
)
from .utils import EmbeddingFunc, get_env_value
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

    # History messages are only sent to LLM for context, not used for retrieval
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    user_prompt: str | None = None
    """User-provided prompt for the query.
    Additional instructions for LLM. If provided, this will be injected into the prompt template.
    Its purpose is to let the user customize the way LLM generates the response.
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

        Important notes for in-memory storage:
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

        Important notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Multi-worker note: see ``upsert`` -- buffered tombstones are
        process-local until index_done_callback() runs.
        """

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity.

        Important notes for in-memory storage:
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

        Important notes for in-memory storage:
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

    Unlike the doc_status scheduling API (mandatory ``@abstractmethod`` — see
    :class:`DocStatusStorage`), strict point reads are an OPTIONAL KV
    capability: ``BaseKVStorage`` serves many namespaces (``full_docs``,
    ``text_chunks``, ``llm_response_cache``, ...) and only the ``full_docs``
    stale-stub decision needs it. A backend that does not provide it degrades
    to the conservative path (keep the FAILED stub, never delete on an
    unconfirmed miss). Callers MUST gate on this flag before calling
    :meth:`get_by_id_strict`.
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
        failures as a best-effort miss. The base default raises
        :class:`~lightrag.exceptions.StorageCapabilityError`; callers must gate
        on :attr:`supports_strict_point_reads` before calling.
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
        """Return keys that do not exist in storage."""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data

        Important notes for in-memory storage:
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

        Important notes for in-memory storage:
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

        Three outcomes, three answers — implementations must not merge them:

        * ``[]``   — the node exists and has no relations.
        * ``None`` — the node is **confirmed absent**.
        * raise    — the backend could not answer. A transport/server error is
          neither of the above; reporting it as ``[]`` or ``None`` turns an
          unknown into a fact that callers (entity merge/dedup, graph edits)
          act on. Same rule as :meth:`BaseKVStorage.get_by_id_strict`.

        ``get_nodes_edges_batch`` cannot express the middle case (it returns a
        dict of lists) and flattens ``None`` to ``[]`` by design; callers that
        need the distinction must use this single-node form.
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

        Important notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Attribute contract (applies to ``upsert_edge`` and the batch variants
        too):
            Every value must be a storable scalar -- ``str``
            (XML-compatible), ``int``, finite ``float``, or ``bool``. Nothing
            else: no nested containers, and no ``None``. Attribute names must
            not contain ``"."`` or start with ``"$"``.

            This is the intersection of what the registered backends can carry,
            and callers are responsible for it because the backends disagree on
            what happens when it is violated. The same non-scalar is refused by
            the Neo4j driver, stored verbatim by MongoDB and by
            PostgreSQL's ``jsonb`` column, and fatal to GraphML serialization;
            ``None`` deletes the property on the Cypher backends but is a hard
            error on NetworkX. ``lightrag.utils.validate_graph_attributes``
            is the shared enforcement point -- prefer it over re-deriving the
            rule per backend.

            This is a **caller** contract, enforced where input enters the
            system. An implementation should reject only what *it* cannot store,
            which may be less: ``NetworkXStorage`` accepts ``NaN`` and integers
            past int64 because GraphML round-trips them, even though the Neo4j
            driver cannot pack either.

            That asymmetry is deliberate, and the reason is the same for names
            and values. Every rewrite path (entity edit, rename, merge,
            extraction rebuild) spreads a fetched object's stored attributes back
            into the upsert payload, and a workspace can already hold values or
            names that predate this contract -- the manual edit API accepted
            anything before its field allowlist landed. An implementation that
            enforced the full contract on a rewrite would make those objects
            permanently unmodifiable, gaining nothing it could not already store.
            So: ``lightrag.utils.graph_attribute_value_rejection`` at the
            ingress, ``xml_attribute_value_rejection`` (or the equivalent for
            that store) inside it.

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

        Important notes for in-memory storage:
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

        Important notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to delete
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Important notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Important notes:
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
            max_nodes: Maximum nodes to return, Defaults to 1000 (BFS if possible)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit

        Ranks by node degree, descending, and **ties break on the label,
        ascending** — the same tie-break :meth:`get_popular_labels` documents,
        for the same reason. Degree alone does not order a real graph: leaf
        entities of degree 1 or 2 outnumber everything else, so the ``max_nodes``
        cutoff almost always lands inside a band of equal-degree entities, and
        whatever orders that band decides which nodes the caller ever sees.
        Left to the backend's natural order that is node INSERTION order (a
        stable sort in Python, an unconstrained plan order in SQL/Cypher), so
        re-ingesting the same corpus with the documents in a different order
        returned a different graph at the same ``max_nodes`` — the defect fixed
        for ``get_popular_labels`` first, then here.

        Order the labels by code point (SQL ``COLLATE "C"``, not a locale
        collation) so every backend agrees with Python's ``str`` comparison.

        **Scope: the ``*`` whole-graph ranking on every backend.** The rule
        should govern the non-wildcard path too -- a BFS level that overflows
        ``max_nodes`` faces the same tie, and it decides both which neighbours
        survive and which get expanded next -- but today only
        :class:`~lightrag.kg.networkx_impl.NetworkXStorage` and
        :class:`~lightrag.kg.pgtable_impl.PGTableGraphStorage` order their BFS
        levels that way. Neo4j, Memgraph, Mongo and OpenSearch admit same-depth
        nodes in traversal order, so their non-wildcard cutoff is still
        ingestion-order dependent. Tracked in issue #3612; a new backend should
        implement both paths rather than match that gap.

        **Known deviation -- PGGraphStorage (Apache AGE)** ranks the ``*`` view
        on ``degree DESC, v.id ASC``, the internal vertex id, not the label.
        Selecting only ``v.id`` lets the vertex scan be index-only; the label
        lives in the vertex ``properties``, so ordering on it forces a full heap
        read (~1.5x buffers, ~25% wall clock on a 200k-vertex/600k-edge graph),
        and no index removes that -- the ORDER BY leads with an aggregate
        computed from the edge table. The id is an insertion counter, so that
        backend's view is stable for a given database but still varies with
        ingestion order across databases holding the same graph, and its
        :meth:`get_popular_labels` (which DOES order by label) can disagree with
        its graph view at the same cutoff. Every other backend orders its ``*``
        ranking on the label; do not copy the deviation into a new one.

        **Known approximation -- OpenSearchGraphStorage** applies the rule, but
        only to the candidates its degree aggregations surfaced, and that set is
        approximate: the two endpoint aggregations are each capped at
        ``max_nodes``, so an entity whose in- and out-degree both fall outside
        their respective top-N never reaches the ranking however high its
        undirected degree is, and terms aggregations are count-approximate
        across shards. Tracked in issue #3613; it needs a storage-shape change,
        not an ordering one.

        This constrains WHICH nodes survive truncation, not the order of
        :class:`KnowledgeGraph.nodes` in the response — implementations
        materialize that list from a dict or a subgraph view, and callers that
        need a specific presentation order must sort it themselves.
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

        Ranks the WHOLE node set: an isolated (degree-0) entity ranks last, but
        it still ranks. Implementations that derive degrees from their edge
        store must therefore not let a node absent from that store fall out of
        the result. Ties break on the label, ascending.

        The expected shape is two phases, because the second one almost never
        runs. Rank the entities that HAVE edges first — the cheap path, and any
        graph with more than ``limit`` connected entities fills every slot there
        — then top the result up from the isolated entities only when that comes
        up short. The top-up is bounded by the shortfall (never more than
        ``limit`` rows), which is what keeps it affordable even where it means a
        sequential scan of the node store. Coming up short also means the
        connected set is now known in full, so every remaining entity is
        isolated by definition.

        Driving the whole ranking off the node store instead is simpler to
        write and gives the same answer, but it pays a full pass over the nodes
        on every call — on a large graph, to produce a result the first phase
        already had.

        The result is best-effort about the reverse direction: a backend that
        derives degrees from its edge store MAY also surface an id that has
        edges but no node document. Only a data-quality defect produces one —
        the write paths materialize both endpoints of every edge — and
        confirming every ranked id against the node store would cost a join or
        an extra round trip on a hot, interactive endpoint. Callers must
        therefore tolerate a returned label whose :meth:`get_node` comes back
        empty, rather than assume every label resolves.

        An empty list means the graph holds no entities. A backend error MUST
        raise instead — ``/graph/label/popular`` renders both, and the two are
        indistinguishable to the user once the error is swallowed.
        """

    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels(entity names) with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance

        As with :meth:`get_popular_labels`, an empty list means "nothing
        matched" — a backend error MUST raise rather than return it. A blank
        query is a real empty result, not an error.
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

    @classmethod
    def from_stored(cls, data: dict[str, Any]) -> "DocProcessingStatus":
        """Construct from a stored doc_status row, ignoring undeclared fields.

        Producers evolve independently of this schema: RAG-Anything writes
        ``scheme_name``/``multimodal_content``, other LightRAG versions write
        fields an installed release does not declare yet (or no longer does).
        ``DocProcessingStatus(**row)`` raises ``TypeError`` on any such field,
        which makes every read path treat the row as malformed and drop it
        from listings — the document silently disappears from the WebUI and
        the API even though its record is intact (HKUDS/RAG-Anything#73).

        Extra fields are ignored for construction only; the stored row is
        not modified. A *missing required* field still raises ``TypeError``
        — tolerance is strictly for extras, not for malformed rows.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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
    file_path: str | None
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


@dataclass(frozen=True)
class SourceAbsent:
    """No primary candidate exists for the canonical source key."""


@dataclass(frozen=True)
class SourceUnique:
    """Exactly one primary (non-duplicate) candidate for the canonical source
    key; ``doc_id`` is the identity every subsequent operation must use."""

    doc_id: str
    doc: DocSchedulingRecord


@dataclass(frozen=True)
class SourceConflict:
    """Two or more primary candidates share the canonical source key.

    Scan must not enqueue, delete, or archive — the job surfaces a repair
    request instead. ``candidate_count`` is ``None`` when the backend only
    proved "at least two" (via two samples) without a cheap exact count.
    ``sample_doc_ids`` uses a fixed upper bound — never materialize the whole
    conflicting set into memory.
    """

    candidate_count: int | None
    sample_doc_ids: tuple[str, ...]


# doc_id is the sole identity for records and downstream processing; the
# canonical source key (a physical basename) only locates candidate rows and
# is NOT assumed globally unique (custom-id inserts, legacy ids and historical
# basename collisions can all share one basename).
SourceResolution = SourceAbsent | SourceUnique | SourceConflict


@dataclass(frozen=True)
class SourceConflictSummary:
    """One conflicting canonical source key with a bounded candidate sample."""

    canonical_source_key: str
    candidate_count: int | None
    sample_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceConflictPage:
    """One page of the source-conflict listing (bounded projection)."""

    conflicts: tuple[SourceConflictSummary, ...]
    next_position: CursorPosition


@dataclass(frozen=True)
class SourceConflictRepairResult:
    """Outcome of a source-conflict repair, dry-run or committed.

    ``fingerprint`` is a deterministic digest over the candidate doc IDs in
    stable sort order — the CAS token an operator echoes back on commit so a
    concurrent change to the candidate set fails the repair instead of being
    silently overwritten. ``demoted_sample_doc_ids`` is a bounded sample of
    the rows that would be (dry-run) or were (commit) marked
    ``metadata.is_duplicate=true``; content is never deleted.
    """

    canonical_source_key: str
    primary_doc_id: str
    candidate_count: int
    fingerprint: str
    demoted_sample_doc_ids: tuple[str, ...]
    committed: bool


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage.

    All backends are first-party and MUST implement the bounded scheduling +
    strict-read API (``get_docs_by_statuses_page`` / ``get_docs_by_ids`` /
    ``resolve_doc_source_strict``) — these are ``@abstractmethod`` so a backend
    missing any of them cannot instantiate. There is no degraded fallback and
    no capability flag: instantiability IS the capability guarantee.
    """

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
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """Get a document by its content_hash field.

        Used for content-hash deduplication of full documents.

        **Fail-closed (MUST).** ``None`` means "CONFIRMED no such row". Query,
        transport and decode failures, and a backend that cannot answer (an
        index not ready or vanished), MUST raise — never return ``None``. The
        callers act on ``None`` destructively: enqueue treats it as "not a
        duplicate" and admits the document, and the post-parse check lets a full
        ingestion proceed, so a swallowed failure mints duplicate rows and
        duplicate graph contributions.

        **Deterministic (MUST).** With several matching rows, return the
        EARLIEST by ``(created_at, id)``. The winner is persisted as
        ``original_doc_id`` on the duplicate's row, so an arbitrary "first row
        the index/scan reached" would make that attribution vary between runs
        over identical data. Backends without a content_hash index scan
        regardless and must track the minimum rather than return on first hit.

        Args:
            content_hash: The content hash value to search for.
            exclude_doc_id: When set, TWO kinds of row are not a match and the
                lookup returns the earliest remaining holder (or None):

                1. the row with this id — which lets the duplicate check
                   exclude the very row being processed (whose own content_hash
                   is already persisted) in a single bounded/indexed query,
                   instead of a full-store scan fallback;
                2. a row that merely POINTS at it: ``metadata.is_duplicate``
                   true with ``metadata.original_doc_id == exclude_doc_id``.
                   Such a row is not an independent holder of the content — it
                   is a record that this content belongs to ``exclude_doc_id``
                   — so returning it makes the asking document a duplicate of a
                   record of ITSELF. Both rows then carry
                   ``is_duplicate=true`` naming each other and the canonical
                   source they share is left with no primary at all, which the
                   repair endpoint cannot settle (it only accepts a current
                   primary candidate) and which a re-upload cannot undo (the
                   doc id is derived from the file name, so it asks about the
                   very id the pointer names). Reachable with no race through
                   a source-conflict repair: it demotes losing candidates with
                   ``original_doc_id=<the kept primary>``, and a kept primary
                   that was not parsed yet reaches its post-parse duplicate
                   check afterwards with the hash the demoted row still holds.

                Skipping such a row must not stop the search: the point is the
                EARLIEST holder that is not one of these two, so an
                implementation that filtered one result after the fact would
                hide a genuine third holder behind a pointer row and re-admit
                the duplicate it was asked about. Every backend MUST honour
                both exclusions where it selects (in-query, or inside the
                implementation over a bounded ordered window) so the exclusion
                never widens the scan and never truncates the search.

                This keyword was added to an already-abstract method, so a
                subclass carrying the older ``(self, content_hash)`` signature
                raises ``TypeError`` when the dedup path passes it. That is
                subsumed by the scheduling API (``get_docs_by_statuses_page`` /
                ``get_docs_by_ids`` / ``resolve_doc_source_strict``) being
                abstract with no fallback in the same release: such a subclass
                cannot be instantiated at all, and fails at construction naming
                the methods it is missing.

        Returns:
            (doc_id, doc_data) when a matching record exists, otherwise None.
        """

    @staticmethod
    def _row_points_at_as_duplicate(row: Any, doc_id: str | None) -> bool:
        """Is ``row`` merely a record that its content belongs to ``doc_id``?

        The second half of ``get_doc_by_content_hash``'s ``exclude_doc_id``
        predicate (see its contract), shared so the backends that filter rows in
        Python read the same ``metadata`` the ones that filter in-query do.
        """
        if not doc_id or not isinstance(row, dict):
            return False
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not metadata.get("is_duplicate"):
            return False
        return metadata.get("original_doc_id") == doc_id

    # ------------------------------------------------------------------
    # Memory-bounding scheduling API (Phase 1). The paging / batch / source
    # methods are ``@abstractmethod`` — every (first-party) backend MUST
    # implement them with bounded / fail-closed behaviour; there is no
    # degraded base fallback. ``count`` / ``update`` / conflict-repair keep a
    # concrete base default.
    # ------------------------------------------------------------------

    @staticmethod
    def _scheduling_record_from_raw(
        doc_id: str, raw: dict[str, Any]
    ) -> DocSchedulingRecord:
        """Project a raw doc_status dict into the lightweight scheduling record.

        Used by base-default paths that only have the persisted dict (e.g. the
        legacy-delegating source resolver). ``status`` may already be a
        :class:`DocStatus` or its string value.
        """
        status_val = raw.get("status")
        status = (
            status_val if isinstance(status_val, DocStatus) else DocStatus(status_val)
        )
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        return DocSchedulingRecord(
            id=doc_id,
            status=status,
            created_at=raw.get("created_at") or "",
            updated_at=raw.get("updated_at") or "",
            file_path=raw.get("file_path"),
            track_id=raw.get("track_id"),
            has_custom_chunk_journal=isinstance(
                metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict
            ),
        )

    @abstractmethod
    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        """Read one page of a stable keyset sweep over the given statuses.

        Contract (every backend MUST implement this as a true bounded sweep):

        * **Sort (MUST)**: ``(created_at ASC, id ASC)`` keyset order across
          ALL requested statuses combined (per-status keysets merged with a
          bounded k-way merge where the backend cannot sort natively).
          Live-view: the sweep observes concurrent writes; no snapshot
          isolation is claimed. Termination is ``next_position is
          CURSOR_END`` — never an empty ``docs``.
        * **Immutable sort key**: ``created_at`` is written once at record
          creation and preserved by every later transition
          (``update_doc_status_fields`` refuses it) so a record can never move
          underneath a sweep. Where the backend indexes a missing/NULL
          ``created_at`` at all, that bucket must stay reachable across page
          boundaries via bucket-aware resume rather than becoming permanently
          unreachable behind a real-timestamp cursor. Note that such a row can
          never be RETURNED: ``DocSchedulingRecord`` requires ``created_at``, so
          a strict page raises on it (complete-or-raise) and a relaxed page
          consumes-and-skips it so the cursor still advances. Only legacy rows
          and external edits reach that state — every write path stamps the
          field. Pinned for all five backends in
          ``tests/kg/test_doc_status_scheduling_contract.py``.
        * **Consumed-position advance**: ``next_position`` advances past the
          last CONSUMED underlying record — records returned to the caller
          plus records read and dropped by filtering are consumed; a record
          prefetched for merging but NOT returned because the page filled is
          NOT consumed (or must be encoded into the opaque cursor and returned
          first on the next page). A fully-filtered page therefore advances
          the cursor without terminating, never re-reads in place, and never
          skips a prefetched head. With multiple statuses the opaque cursor
          tracks each status stream's own consumed position.
        * ``strict=True``: the page is complete or the call raises — on an
          internal partial failure the implementation must raise WITHOUT
          returning partial docs or a new cursor. All scheduling/control-plane
          callers pass ``strict=True`` explicitly.
        """

    @abstractmethod
    async def get_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocSchedulingRecord]:
        """Batch strict read of scheduling records by doc id.

        Contract (every backend MUST implement it as a true batch strict read):

        * The returned mapping contains ONLY ids confirmed to exist; a missing
          id may be omitted, but the backend must have positively confirmed
          its absence.
        * With ``strict=True`` any transport/server/chunking error fails the
          WHOLE call — the implementation must not return a partial mapping
          and let the feeder treat un-returned ids as stale.
        * Results use the lightweight projection (no chunks / full content).
        """

    @abstractmethod
    async def get_full_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocProcessingStatus]:
        """Batch hydration of FULL doc_status records by doc id.

        The scheduling page (:meth:`get_docs_by_statuses_page`) yields the
        lightweight :class:`DocSchedulingRecord` projection — only enough to
        order, bound and filter a sweep. The scheduler hydrates the survivors
        of ONE page into full :class:`DocProcessingStatus` rows through this
        method right before routing them, so memory stays O(page_size) while
        the consistency validator and the parse/analyze/process workers still
        see every field they need (``content_summary`` / ``content_length`` /
        ``chunks_list`` / ``metadata`` / ...).

        Contract (every backend MUST implement it as a true batch read reusing
        the SAME raw → :class:`DocProcessingStatus` normalisation as
        :meth:`get_docs_by_statuses`):

        * The returned mapping contains ONLY ids confirmed to exist; an id
          the backend positively confirms absent (deleted, or its row no
          longer decodes) may be omitted — a benign race, since a doc that
          vanished between the page and its hydration is simply no longer
          routable.
        * With ``strict=True`` any transport/server/decode error fails the
          WHOLE call — the implementation must not return a partial mapping
          (the scheduler cannot distinguish "gone" from "unread" on a
          partial, and a silently short page would strand the missing docs in
          PENDING). All scheduling/control-plane callers pass ``strict=True``.
        * Results are FULL :class:`DocProcessingStatus`, NOT the lightweight
          projection.
        """

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        """Count documents in the given statuses, fail-closed.

        Unlike ``get_status_counts()`` implementations that swallow errors and
        report zeros, this method MUST either return an accurate count or
        raise — admission control treats an error as "refuse", never as
        "capacity available". The base implementation raises
        :class:`~lightrag.exceptions.StorageCapabilityError`; deployments that
        enable ``MAX_PENDING_DOCUMENTS`` validate support at initialization.
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
          targeted alternative to read-modify-write upserts that would drag a
          huge ``chunks_list`` through memory.
        * Implementations MUST atomically maintain every secondary index
          affected by the updated fields (e.g. Redis per-status ZSETs and the
          source multimap) in the same transaction as the write.
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

    @abstractmethod
    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:
        """Resolve a canonical source key to a typed source resolution.

        Contract (every backend MUST implement it fail-closed and
        conflict-aware):

        * Only rows with ``metadata.is_duplicate != true`` are primary
          candidates.
        * 0 candidates → :class:`SourceAbsent`; 1 → :class:`SourceUnique`;
          ≥2 → :class:`SourceConflict` (found by locating just two candidates,
          never materializing the whole conflicting set).
        * Query failure, a not-ready index, or an ambiguous state raises a
          control-plane error instead of returning ``SourceAbsent`` — scan and
          enqueue treat Absent as "confirmed new", so a swallowed failure would
          mint duplicate rows.
        """

    async def list_source_conflicts_page(
        self,
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
    ) -> SourceConflictPage:
        """List canonical source keys with more than one primary candidate.

        Operator/admin tooling for the explicit repair flow (see
        :meth:`repair_source_conflict`). The base default raises
        :class:`~lightrag.exceptions.StorageCapabilityError`; only backends
        with a real strict resolver can enumerate conflicts.
        """
        raise StorageCapabilityError(
            f"{type(self).__name__} does not implement "
            "list_source_conflicts_page (no strict source resolution)."
        )

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> SourceConflictRepairResult:
        """Resolve a historical multi-primary conflict, CAS-guarded.

        The system never picks a winner automatically; the operator names the
        ``primary_doc_id`` to keep. Contract for capable backends:

        * ``dry_run=True`` (default) makes no mutation and returns the bounded
          summary plus the deterministic ``fingerprint`` / ``candidate_count``.
        * On commit, the current candidate set is re-read strictly and a
          mismatch with ``expected_candidate_count`` /
          ``expected_candidate_fingerprint`` raises
          :class:`~lightrag.exceptions.SourceConflictRepairCASError` rather
          than overwriting a concurrent change — a *known* stale-token state
          (surface it as 409), distinct from its ``StorageControlPlaneError``
          parent, which means the true state is unknown (503).
        * A ``primary_doc_id`` that is not currently a primary candidate
          raises ``ValueError``; the operator must re-read the candidates.
        * The other candidates are marked ``metadata.is_duplicate=true`` with
          ``original_doc_id=primary_doc_id``; content is never deleted.

        **What this method does NOT serialize.** The storage layer holds no
        canonical-source-key lock, and no backend here can take one that also
        excludes an INSERT: a re-read plus CAS detects changes to the candidate
        rows it saw, but a brand-new primary for the same key is a phantom that
        neither a ``SELECT … FOR UPDATE`` (PostgreSQL), a snapshot transaction
        (MongoDB) nor a non-transactional re-scan (Redis, OpenSearch) can block.
        Serializing repair against enqueue is therefore the CALLER's job:
        ``lightrag.tools.source_conflict_repair.source_conflict_repair_lock``
        holds a ``get_storage_keyed_lock`` on the canonical source key (so
        concurrent repairs of one key serialize, and so does the processing
        stage's duplicate marking, which takes that same lock) plus the
        workspace's
        ``ENQUEUE_SERIALIZE_LOCK_NAMESPACE`` namespace lock, which is the one
        that actually excludes the phantom because the enqueue critical section
        (``filter_keys`` → dedup → ``upsert``) already holds it. Those locks span
        one deployment (its worker processes included), not several deployments
        sharing a database.

        ``committed=True`` accordingly means "the demotions named in this result
        were committed" — NOT "this source key is now conflict-free". A primary
        inserted mid-repair simply leaves the resolver in
        :class:`SourceConflict`, which is fail-closed (scan refuses to classify,
        nothing destructive follows) and which a fresh dry-run + commit
        resolves; repair is idempotent, so re-running is always safe.

        The base default raises
        :class:`~lightrag.exceptions.StorageCapabilityError`.
        """
        raise StorageCapabilityError(
            f"{type(self).__name__} does not implement repair_source_conflict "
            "(no strict source resolution)."
        )


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
