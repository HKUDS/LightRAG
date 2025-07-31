# LightRAG MCP Tools Specification

**Technical Specification for MCP Tool Implementation**

## Overview

This document provides the complete technical specification for all MCP tools that will be implemented in the LightRAG MCP server. Each tool includes detailed parameter schemas, response formats, error handling, and implementation notes.

## Tool Categories

1. [Query Tools](#query-tools) - RAG querying and search
2. [Document Management Tools](#document-management-tools) - Document lifecycle management
3. [Knowledge Graph Tools](#knowledge-graph-tools) - Graph exploration and manipulation
4. [System Management Tools](#system-management-tools) - Health monitoring and administration

---

## Query Tools

### lightrag_query

**Purpose**: Execute RAG queries using LightRAG's multiple retrieval modes

**Schema**:
```python
@mcp.tool()
async def lightrag_query(
    query: str,
    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid",
    top_k: int = 40,
    chunk_top_k: int = 10,
    cosine_threshold: float = 0.2,
    max_tokens: int = 30000,
    enable_rerank: bool = False,
    history_turns: int = 0
) -> QueryResponse:
```

**Parameters**:
- `query` (str, required): The question or search query
- `mode` (str, optional): Query mode determining retrieval strategy
  - `naive`: Basic vector search without graph enhancement
  - `local`: Context-dependent entity-focused retrieval
  - `global`: Global knowledge graph relationship queries
  - `hybrid`: Combines local and global approaches
  - `mix`: Integrates knowledge graph traversal with vector similarity
  - `bypass`: Direct LLM query without retrieval augmentation
- `top_k` (int, optional): Number of entities/relations to retrieve (default: 40)
- `chunk_top_k` (int, optional): Number of document chunks to include (default: 10)
- `cosine_threshold` (float, optional): Similarity threshold for retrieval (default: 0.2)
- `max_tokens` (int, optional): Maximum tokens in final context (default: 30000)
- `enable_rerank` (bool, optional): Enable reranking of results (default: false)
- `history_turns` (int, optional): Number of conversation history turns to include (default: 0)

**Response Schema**:
```python
class QueryResponse(BaseModel):
    response: str                    # Generated answer
    mode: str                       # Query mode used
    metadata: QueryMetadata
    sources: List[SourceAttribution]

class QueryMetadata(BaseModel):
    entities_used: int
    relations_used: int
    chunks_used: int
    processing_time: float
    token_usage: TokenUsage
    cache_hit: bool

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class SourceAttribution(BaseModel):
    document_id: str
    chunk_id: str
    content_preview: str
    relevance_score: float
    start_position: Optional[int]
    end_position: Optional[int]
```

**Error Conditions**:
- `INVALID_QUERY_MODE`: Unknown query mode specified
- `QUERY_TOO_LONG`: Query exceeds maximum length
- `NO_RESULTS_FOUND`: No relevant content found for query
- `PROCESSING_TIMEOUT`: Query processing exceeded timeout
- `INSUFFICIENT_CONTEXT`: Not enough context for meaningful response

**Implementation Notes**:
- Cache responses for identical queries within TTL window
- Implement query validation and sanitization
- Support streaming responses for long-running queries
- Track query performance metrics
- Handle timeout gracefully with partial results

---

### lightrag_stream_query

**Purpose**: Execute streaming RAG queries with real-time response generation

**Schema**:
```python
@mcp.tool()
async def lightrag_stream_query(
    query: str,
    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid",
    top_k: int = 40,
    chunk_top_k: int = 10,
    cosine_threshold: float = 0.2,
    max_tokens: int = 30000
) -> AsyncIterator[StreamChunk]:
```

**Parameters**: Same as `lightrag_query` (excluding history_turns and enable_rerank)

**Response Schema**:
```python
class StreamChunk(BaseModel):
    chunk_type: Literal["content", "metadata", "error", "complete"]
    content: Optional[str]          # Text content (for content chunks)
    metadata: Optional[Dict[str, Any]]  # Metadata (for metadata chunks)
    error: Optional[str]            # Error message (for error chunks)

# Final metadata chunk includes:
class StreamMetadata(BaseModel):
    total_chunks_sent: int
    entities_used: int
    relations_used: int
    chunks_used: int
    processing_time: float
    token_usage: TokenUsage
```

**Implementation Notes**:
- Send content chunks as they become available
- Include progress indicators during processing
- Send metadata as final chunk
- Handle connection drops gracefully
- Implement backpressure control

---

## Document Management Tools

### lightrag_insert_text

**Purpose**: Insert text documents directly into the knowledge base

**Schema**:
```python
@mcp.tool()
async def lightrag_insert_text(
    text: str,
    title: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    processing_options: Optional[ProcessingOptions] = None
) -> DocumentResponse:
```

**Parameters**:
- `text` (str, required): Document content to process
- `title` (str, optional): Document title for identification
- `metadata` (dict, optional): Additional document metadata
  - `source`: Source identifier
  - `author`: Document author
  - `created_date`: Creation date (ISO format)
  - `tags`: List of tags
  - `custom_fields`: Custom metadata fields
- `processing_options` (dict, optional): Processing configuration
  - `chunk_size`: Override default chunk size
  - `chunk_overlap`: Override default chunk overlap
  - `enable_entity_extraction`: Enable/disable entity extraction
  - `processing_priority`: Priority level (low, normal, high)

**Response Schema**:
```python
class DocumentResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    message: str
    processing_info: ProcessingInfo
    estimated_completion: Optional[str]  # ISO timestamp

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"

class ProcessingInfo(BaseModel):
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    processing_time: Optional[float]
    error_details: Optional[str]
```

**Error Conditions**:
- `TEXT_TOO_LARGE`: Text exceeds maximum size limit
- `INVALID_METADATA`: Metadata format is invalid
- `PROCESSING_FAILED`: Document processing encountered errors
- `DUPLICATE_DOCUMENT`: Document with same content already exists
- `QUOTA_EXCEEDED`: Processing quota exceeded

**Implementation Notes**:
- Validate text content and metadata
- Support asynchronous processing with status updates
- Implement deduplication checks
- Handle large documents efficiently
- Provide progress tracking for long processing

---

### lightrag_insert_file

**Purpose**: Process and index files from filesystem or uploaded content

**Schema**:
```python
@mcp.tool()
async def lightrag_insert_file(
    file_path: str,
    processing_options: Optional[ProcessingOptions] = None,
    file_validation: Optional[FileValidation] = None
) -> DocumentResponse:
```

**Parameters**:
- `file_path` (str, required): Path to file or file identifier
- `processing_options` (dict, optional): Same as `lightrag_insert_text`
- `file_validation` (dict, optional): File validation settings
  - `max_size_mb`: Maximum file size in MB
  - `allowed_extensions`: List of allowed file extensions
  - `scan_for_malware`: Enable malware scanning
  - `extract_metadata`: Extract file metadata

**Supported File Types**:
- **Text**: .txt, .md, .rtf
- **Documents**: .pdf, .docx, .doc, .odt
- **Presentations**: .pptx, .ppt, .odp
- **Spreadsheets**: .xlsx, .xls, .ods
- **Web**: .html, .htm

**Response Schema**: Same as `lightrag_insert_text`

**Error Conditions**:
- `FILE_NOT_FOUND`: Specified file does not exist
- `UNSUPPORTED_FILE_TYPE`: File type not supported
- `FILE_TOO_LARGE`: File exceeds size limit
- `FILE_CORRUPTED`: File appears to be corrupted
- `EXTRACTION_FAILED`: Content extraction failed
- `SECURITY_VIOLATION`: File failed security scan

**Implementation Notes**:
- Implement robust file type detection
- Use secure file processing libraries
- Support batch file processing
- Handle file encoding detection
- Implement file quarantine for security

---

### lightrag_list_documents

**Purpose**: List documents with filtering, sorting, and pagination

**Schema**:
```python
@mcp.tool()
async def lightrag_list_documents(
    status_filter: Optional[DocumentStatus] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: Literal["created_date", "title", "status", "processing_time"] = "created_date",
    sort_order: Literal["asc", "desc"] = "desc",
    search_query: Optional[str] = None,
    date_range: Optional[DateRange] = None
) -> DocumentListResponse:
```

**Parameters**:
- `status_filter` (str, optional): Filter by document status
- `limit` (int, optional): Maximum documents to return (default: 50, max: 200)
- `offset` (int, optional): Pagination offset (default: 0)
- `sort_by` (str, optional): Field to sort by
- `sort_order` (str, optional): Sort direction
- `search_query` (str, optional): Search in document titles and metadata
- `date_range` (dict, optional): Date range filter
  - `start_date`: Start date (ISO format)
  - `end_date`: End date (ISO format)

**Response Schema**:
```python
class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
    pagination: PaginationInfo
    statistics: DocumentStatistics

class DocumentSummary(BaseModel):
    id: str
    title: str
    status: DocumentStatus
    created_at: str
    processed_at: Optional[str]
    file_type: Optional[str]
    file_size: Optional[int]
    chunk_count: int
    entity_count: int
    relationship_count: int
    error_message: Optional[str]
    metadata: Dict[str, Any]

class PaginationInfo(BaseModel):
    total_documents: int
    current_page: int
    total_pages: int
    has_next: bool
    has_previous: bool

class DocumentStatistics(BaseModel):
    total_documents: int
    by_status: Dict[str, int]
    by_file_type: Dict[str, int]
    processing_success_rate: float
```

**Implementation Notes**:
- Implement efficient database queries with proper indexing
- Support full-text search across document content
- Cache frequently accessed document lists
- Provide rich filtering capabilities
- Handle large result sets efficiently

---

### lightrag_delete_documents

**Purpose**: Remove documents and associated data from knowledge base

**Schema**:
```python
@mcp.tool()
async def lightrag_delete_documents(
    document_ids: List[str],
    delete_options: Optional[DeleteOptions] = None
) -> DeleteResponse:
```

**Parameters**:
- `document_ids` (list, required): List of document IDs to delete
- `delete_options` (dict, optional): Deletion configuration
  - `cascade_delete`: Delete associated entities/relationships (default: true)
  - `create_backup`: Create backup before deletion (default: false)
  - `force_delete`: Skip safety checks (default: false)
  - `notify_on_completion`: Send notification when complete (default: false)

**Response Schema**:
```python
class DeleteResponse(BaseModel):
    deleted_documents: List[str]
    failed_deletions: List[DeleteFailure]
    cascade_deletions: CascadeDeletions
    backup_info: Optional[BackupInfo]
    processing_time: float

class DeleteFailure(BaseModel):
    document_id: str
    error_code: str
    error_message: str

class CascadeDeletions(BaseModel):
    entities_deleted: int
    relationships_deleted: int
    chunks_deleted: int

class BackupInfo(BaseModel):
    backup_id: str
    backup_location: str
    backup_size: int
```

**Error Conditions**:
- `DOCUMENT_NOT_FOUND`: One or more documents not found
- `DELETION_FORBIDDEN`: Document deletion not permitted
- `CASCADE_FAILURE`: Cascade deletion failed
- `BACKUP_FAILED`: Backup creation failed
- `PARTIAL_DELETION`: Some documents could not be deleted

**Implementation Notes**:
- Implement transactional deletions
- Support soft delete with recovery options
- Handle cascade deletions properly
- Provide detailed deletion reports
- Implement deletion quotas and rate limiting

---

### lightrag_batch_process

**Purpose**: Process multiple documents in batch with progress tracking

**Schema**:
```python
@mcp.tool()
async def lightrag_batch_process(
    items: List[BatchItem],
    batch_options: Optional[BatchOptions] = None
) -> BatchResponse:
```

**Parameters**:
- `items` (list, required): List of items to process
- `batch_options` (dict, optional): Batch processing configuration
  - `max_concurrent`: Maximum concurrent processing (default: 5)
  - `stop_on_error`: Stop batch if error occurs (default: false)
  - `priority`: Batch priority (low, normal, high)
  - `notification_webhook`: URL for progress notifications

**Types**:
```python
class BatchItem(BaseModel):
    item_type: Literal["file", "text", "url"]
    content: str  # File path, text content, or URL
    title: Optional[str]
    metadata: Optional[Dict[str, Any]]

class BatchOptions(BaseModel):
    max_concurrent: int = 5
    stop_on_error: bool = False
    priority: Literal["low", "normal", "high"] = "normal"
    notification_webhook: Optional[str] = None
```

**Response Schema**:
```python
class BatchResponse(BaseModel):
    batch_id: str
    total_items: int
    status: BatchStatus
    progress: BatchProgress
    estimated_completion: Optional[str]
    results: List[BatchItemResult]

class BatchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchProgress(BaseModel):
    completed: int
    failed: int
    remaining: int
    percentage: float

class BatchItemResult(BaseModel):
    item_index: int
    status: str
    document_id: Optional[str]
    error_message: Optional[str]
    processing_time: Optional[float]
```

**Implementation Notes**:
- Implement queue-based batch processing
- Support batch cancellation and resumption
- Provide real-time progress updates
- Handle partial failures gracefully
- Implement batch result persistence

---

## Knowledge Graph Tools

### lightrag_get_graph

**Purpose**: Extract knowledge graph data with filtering and formatting options

**Schema**:
```python
@mcp.tool()
async def lightrag_get_graph(
    label_filter: Optional[str] = None,
    max_nodes: int = 100,
    max_edges: int = 200,
    output_format: Literal["json", "cypher", "graphml", "gexf"] = "json",
    include_properties: bool = True,
    graph_options: Optional[GraphOptions] = None
) -> GraphResponse:
```

**Parameters**:
- `label_filter` (str, optional): Filter nodes/edges by label
- `max_nodes` (int, optional): Maximum nodes to return (default: 100, max: 1000)
- `max_edges` (int, optional): Maximum edges to return (default: 200, max: 2000)
- `output_format` (str, optional): Output format for graph data
- `include_properties` (bool, optional): Include node/edge properties (default: true)
- `graph_options` (dict, optional): Additional graph options
  - `min_degree`: Minimum node degree to include
  - `community_filter`: Filter by community/cluster
  - `centrality_threshold`: Minimum centrality score
  - `time_range`: Time range for temporal filtering

**Response Schema**:
```python
class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    statistics: GraphStatistics
    metadata: GraphMetadata

class GraphNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]
    degree: int
    centrality: Optional[float]
    community: Optional[str]

class GraphEdge(BaseModel):
    id: str
    type: str
    source: str
    target: str
    properties: Dict[str, Any]
    weight: Optional[float]
    confidence: Optional[float]

class GraphStatistics(BaseModel):
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    density: float
    clustering_coefficient: float
    average_path_length: Optional[float]

class GraphMetadata(BaseModel):
    extraction_time: str
    filters_applied: Dict[str, Any]
    format: str
    truncated: bool
```

**Implementation Notes**:
- Support multiple export formats
- Implement efficient graph sampling for large graphs
- Calculate graph metrics on-demand
- Support temporal and spatial filtering
- Optimize for visualization tools

---

### lightrag_search_entities

**Purpose**: Search entities by name, properties, or relationships

**Schema**:
```python
@mcp.tool()
async def lightrag_search_entities(
    query: str,
    search_options: Optional[EntitySearchOptions] = None,
    limit: int = 20,
    offset: int = 0
) -> EntitySearchResponse:
```

**Parameters**:
- `query` (str, required): Search query
- `search_options` (dict, optional): Search configuration
  - `search_type`: "fuzzy", "exact", "semantic", "regex"
  - `search_fields`: Fields to search (name, properties, relationships)
  - `entity_types`: Filter by entity types
  - `min_confidence`: Minimum confidence score
- `limit` (int, optional): Maximum results (default: 20, max: 100)
- `offset` (int, optional): Pagination offset

**Response Schema**:
```python
class EntitySearchResponse(BaseModel):
    entities: List[EntityMatch]
    total_matches: int
    search_metadata: SearchMetadata

class EntityMatch(BaseModel):
    entity: GraphNode
    relevance_score: float
    match_reasons: List[str]
    highlighted: Dict[str, str]  # Highlighted matching text

class SearchMetadata(BaseModel):
    query: str
    search_type: str
    processing_time: float
    index_used: Optional[str]
```

**Implementation Notes**:
- Support fuzzy matching with configurable similarity
- Implement semantic search using embeddings
- Provide match highlighting
- Use appropriate indexes for performance
- Support search result ranking

---

### lightrag_update_entity

**Purpose**: Modify entity properties and relationships

**Schema**:
```python
@mcp.tool()
async def lightrag_update_entity(
    entity_id: str,
    updates: EntityUpdates,
    update_options: Optional[UpdateOptions] = None
) -> UpdateResponse:
```

**Parameters**:
- `entity_id` (str, required): Entity identifier
- `updates` (dict, required): Updates to apply
  - `properties`: Property updates (add/modify/remove)
  - `labels`: Label updates (add/remove)
  - `relationships`: Relationship updates
- `update_options` (dict, optional): Update configuration
  - `merge_mode`: "replace", "merge", "append"
  - `create_if_missing`: Create entity if not found
  - `validate_updates`: Validate before applying
  - `create_backup`: Backup before update

**Types**:
```python
class EntityUpdates(BaseModel):
    properties: Optional[Dict[str, Any]]
    add_labels: Optional[List[str]]
    remove_labels: Optional[List[str]]
    add_relationships: Optional[List[RelationshipSpec]]
    remove_relationships: Optional[List[str]]

class RelationshipSpec(BaseModel):
    target_entity: str
    relationship_type: str
    properties: Optional[Dict[str, Any]]

class UpdateOptions(BaseModel):
    merge_mode: Literal["replace", "merge", "append"] = "merge"
    create_if_missing: bool = False
    validate_updates: bool = True
    create_backup: bool = False
```

**Response Schema**:
```python
class UpdateResponse(BaseModel):
    success: bool
    entity_id: str
    changes_applied: ChangesSummary
    warnings: List[str]
    backup_id: Optional[str]

class ChangesSummary(BaseModel):
    properties_updated: int
    labels_added: int
    labels_removed: int
    relationships_added: int
    relationships_removed: int
```

**Error Conditions**:
- `ENTITY_NOT_FOUND`: Entity does not exist
- `INVALID_UPDATE`: Update format is invalid
- `CONSTRAINT_VIOLATION`: Update violates constraints
- `RELATIONSHIP_TARGET_NOT_FOUND`: Target entity for relationship not found
- `UPDATE_CONFLICT`: Concurrent update conflict

**Implementation Notes**:
- Implement optimistic locking for concurrent updates
- Validate all updates before applying
- Support atomic updates with rollback
- Track update history for auditing
- Handle relationship consistency

---

### lightrag_get_entity_relationships

**Purpose**: Get relationships for specific entities with filtering

**Schema**:
```python
@mcp.tool()
async def lightrag_get_entity_relationships(
    entity_id: str,
    relationship_filters: Optional[RelationshipFilters] = None,
    limit: int = 50,
    offset: int = 0
) -> EntityRelationshipsResponse:
```

**Parameters**:
- `entity_id` (str, required): Entity identifier
- `relationship_filters` (dict, optional): Filtering options
  - `relationship_types`: Filter by relationship types
  - `direction`: "incoming", "outgoing", "both"
  - `target_entity_types`: Filter by target entity types
  - `min_confidence`: Minimum relationship confidence
- `limit` (int, optional): Maximum relationships to return
- `offset` (int, optional): Pagination offset

**Response Schema**:
```python
class EntityRelationshipsResponse(BaseModel):
    entity: GraphNode
    relationships: List[RelationshipDetail]
    relationship_counts: RelationshipCounts
    pagination: PaginationInfo

class RelationshipDetail(BaseModel):
    relationship: GraphEdge
    connected_entity: GraphNode
    direction: Literal["incoming", "outgoing"]
    path_length: int = 1

class RelationshipCounts(BaseModel):
    total: int
    by_type: Dict[str, int]
    by_direction: Dict[str, int]
    by_target_type: Dict[str, int]
```

**Implementation Notes**:
- Support multi-hop relationship traversal
- Implement efficient graph queries
- Provide relationship analytics
- Support relationship strength/weight filtering
- Cache frequently accessed entity relationships

---

## System Management Tools

### lightrag_health_check

**Purpose**: Comprehensive system health and status monitoring

**Schema**:
```python
@mcp.tool()
async def lightrag_health_check(
    include_detailed: bool = False,
    check_dependencies: bool = True
) -> HealthResponse:
```

**Parameters**:
- `include_detailed` (bool, optional): Include detailed diagnostic information
- `check_dependencies` (bool, optional): Check external dependencies

**Response Schema**:
```python
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime: str
    timestamp: str
    components: Dict[str, ComponentHealth]
    dependencies: Optional[Dict[str, DependencyHealth]]
    system_info: SystemInfo

class ComponentHealth(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    message: Optional[str]
    last_check: str
    metrics: Optional[Dict[str, float]]

class DependencyHealth(BaseModel):
    status: Literal["connected", "degraded", "unavailable"]
    response_time: Optional[float]
    version: Optional[str]
    last_check: str

class SystemInfo(BaseModel):
    configuration: ConfigurationSummary
    statistics: SystemStatistics
    performance: PerformanceMetrics

class ConfigurationSummary(BaseModel):
    llm_binding: str
    llm_model: str
    embedding_model: str
    storage_backends: Dict[str, str]
    workspace: Optional[str]

class SystemStatistics(BaseModel):
    total_documents: int
    total_entities: int
    total_relationships: int
    total_chunks: int
    processing_queue_size: int

class PerformanceMetrics(BaseModel):
    average_query_time: float
    average_processing_time: float
    cache_hit_rates: Dict[str, float]
    memory_usage: Dict[str, float]
    storage_usage: Dict[str, int]
```

**Health Check Components**:
- **Database Connectivity**: All storage backends
- **LLM Service**: Language model availability
- **Embedding Service**: Embedding model availability
- **Cache Systems**: Redis/memory cache status
- **File System**: Disk space and permissions
- **Processing Queue**: Queue status and backlogs
- **Memory Usage**: Current memory consumption
- **API Endpoints**: Critical endpoint availability

**Implementation Notes**:
- Implement circuit breakers for external dependencies
- Cache health check results with appropriate TTL
- Provide actionable diagnostic information
- Support health check webhooks/notifications
- Include performance benchmarks

---

### lightrag_clear_cache

**Purpose**: Clear various system caches with granular control

**Schema**:
```python
@mcp.tool()
async def lightrag_clear_cache(
    cache_types: List[CacheType],
    cache_options: Optional[CacheOptions] = None
) -> ClearCacheResponse:
```

**Parameters**:
- `cache_types` (list, required): Types of cache to clear
  - `llm`: LLM response cache
  - `embedding`: Embedding cache
  - `query`: Query result cache
  - `document`: Document processing cache
  - `graph`: Graph query cache
  - `all`: All cache types
- `cache_options` (dict, optional): Cache clearing options
  - `older_than`: Clear caches older than specified time
  - `pattern`: Clear caches matching pattern
  - `force`: Force clear even if in use
  - `preserve_recent`: Preserve recent cache entries

**Response Schema**:
```python
class ClearCacheResponse(BaseModel):
    cleared_caches: List[str]
    cache_sizes_before: Dict[str, CacheSize]
    cache_sizes_after: Dict[str, CacheSize]
    processing_time: float
    warnings: List[str]

class CacheSize(BaseModel):
    entries: int
    size_bytes: int
    oldest_entry: Optional[str]
    newest_entry: Optional[str]
```

**Implementation Notes**:
- Support selective cache clearing by age/pattern
- Provide detailed cache statistics
- Handle cache clearing during active operations
- Support cache warming after clearing
- Implement cache clearing quotas

---

### lightrag_get_system_stats

**Purpose**: Detailed system usage statistics and analytics

**Schema**:
```python
@mcp.tool()
async def lightrag_get_system_stats(
    time_range: str = "24h",
    include_breakdown: bool = True,
    stat_categories: Optional[List[str]] = None
) -> SystemStatsResponse:
```

**Parameters**:
- `time_range` (str, optional): Statistics time range ("1h", "24h", "7d", "30d")
- `include_breakdown` (bool, optional): Include detailed breakdowns
- `stat_categories` (list, optional): Specific categories to include
  - `queries`: Query statistics
  - `documents`: Document processing statistics
  - `resources`: Resource usage statistics
  - `performance`: Performance metrics
  - `errors`: Error statistics

**Response Schema**:
```python
class SystemStatsResponse(BaseModel):
    time_range: str
    timestamp: str
    query_statistics: Optional[QueryStatistics]
    document_statistics: Optional[DocumentStatistics]
    resource_usage: Optional[ResourceUsage]
    performance_metrics: Optional[PerformanceStatistics]
    error_statistics: Optional[ErrorStatistics]

class QueryStatistics(BaseModel):
    total_queries: int
    queries_by_mode: Dict[str, int]
    average_response_time: float
    cache_hit_rate: float
    popular_queries: List[PopularQuery]
    response_time_percentiles: Dict[str, float]

class DocumentStatistics(BaseModel):
    documents_processed: int
    processing_failures: int
    average_processing_time: float
    documents_by_type: Dict[str, int]
    processing_queue_stats: QueueStatistics

class ResourceUsage(BaseModel):
    memory_usage: MemoryStats
    storage_usage: StorageStats
    api_usage: APIUsageStats
    cache_usage: CacheStats

class PerformanceStatistics(BaseModel):
    throughput: Dict[str, float]
    latency_distribution: Dict[str, Dict[str, float]]
    resource_efficiency: Dict[str, float]
    bottlenecks: List[PerformanceBottleneck]
```

**Implementation Notes**:
- Store statistics in time-series database
- Support real-time statistics updates
- Implement efficient aggregation queries
- Provide trend analysis and anomaly detection
- Support statistics export in multiple formats

---

## Common Types and Schemas

### Error Response Schema

```python
class MCPError(BaseModel):
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]]
    suggested_action: Optional[str]
    documentation_url: Optional[str]
    correlation_id: str
```

### Common Error Codes

```python
ERROR_CODES = {
    # General errors
    "INVALID_PARAMETER": "Invalid parameter value provided",
    "MISSING_PARAMETER": "Required parameter not provided",
    "UNAUTHORIZED": "Authentication required or invalid",
    "FORBIDDEN": "Operation not permitted",
    "RATE_LIMITED": "Rate limit exceeded",
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
    "INTERNAL_ERROR": "Internal server error",

    # LightRAG specific errors
    "LIGHTRAG_UNAVAILABLE": "LightRAG service is not available",
    "PROCESSING_TIMEOUT": "Operation exceeded timeout limit",
    "INVALID_QUERY_MODE": "Invalid query mode specified",
    "DOCUMENT_NOT_FOUND": "Document not found",
    "ENTITY_NOT_FOUND": "Entity not found",
    "GRAPH_ACCESS_ERROR": "Knowledge graph access error",
    "STORAGE_ERROR": "Storage backend error",
    "CONFIGURATION_ERROR": "Configuration error",

    # Resource errors
    "RESOURCE_LIMIT_EXCEEDED": "Resource limit exceeded",
    "QUOTA_EXCEEDED": "Usage quota exceeded",
    "FILE_TOO_LARGE": "File exceeds size limit",
    "UNSUPPORTED_FORMAT": "Unsupported file format",
    "PROCESSING_FAILED": "Processing operation failed",
}
```

### Validation Schemas

```python
# Common validation patterns
ENTITY_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
DOCUMENT_ID_PATTERN = r"^doc_[a-zA-Z0-9_-]+$"
QUERY_MAX_LENGTH = 10000
TITLE_MAX_LENGTH = 500
FILENAME_MAX_LENGTH = 255

# File validation
SUPPORTED_FILE_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".json": "application/json"
}

MAX_FILE_SIZE_MB = 100
MAX_BATCH_SIZE = 50
```

## Implementation Guidelines

### Authentication and Authorization

```python
async def validate_request(tool_name: str, parameters: Dict[str, Any]) -> bool:
    """Validate request authentication and authorization."""
    # Check API key
    # Validate permissions for specific operations
    # Check rate limits
    # Log access attempt
    pass

async def require_user_consent(operation: str, data_summary: str) -> bool:
    """Require explicit user consent for sensitive operations."""
    # Present consent dialog
    # Log consent decision
    # Return consent status
    pass
```

### Error Handling Standards

```python
async def handle_tool_error(error: Exception, tool_name: str) -> MCPError:
    """Standardized error handling for all tools."""
    correlation_id = generate_correlation_id()

    # Log error with correlation ID
    logger.error(f"Tool error in {tool_name}: {error}",
                extra={"correlation_id": correlation_id})

    # Map to standard error codes
    error_code = map_exception_to_code(error)

    return MCPError(
        error_code=error_code,
        error_message=str(error),
        correlation_id=correlation_id,
        suggested_action=get_suggested_action(error_code),
        documentation_url=get_documentation_url(error_code)
    )
```

### Performance Requirements

- **Response Time Targets**:
  - Health checks: <500ms
  - Simple queries: <2s
  - Complex queries: <10s
  - Document operations: <30s
  - Graph operations: <5s

- **Throughput Targets**:
  - Concurrent queries: 50+
  - Document processing: 100+ docs/hour
  - Graph operations: 1000+ ops/minute

- **Resource Limits**:
  - Memory usage: <4GB peak
  - Storage usage: Configurable quotas
  - API rate limits: Configurable per client

This specification provides the complete technical foundation for implementing all MCP tools in the LightRAG integration. Each tool includes comprehensive parameter validation, error handling, and response formatting to ensure a consistent and reliable MCP server implementation.
