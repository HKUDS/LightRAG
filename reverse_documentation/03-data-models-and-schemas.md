# Data Models and Schemas: LightRAG Type System

## Table of Contents
1. [Core Data Models](#core-data-models)
2. [Storage Schema Definitions](#storage-schema-definitions)
3. [Query and Response Models](#query-and-response-models)
4. [Configuration Models](#configuration-models)
5. [TypeScript Type Mapping](#typescript-type-mapping)

## Core Data Models

### Text Chunk Schema

Text chunks are the fundamental unit of document processing in LightRAG. Documents are split into overlapping chunks that preserve context while fitting within token limits.

**Python Definition** (`lightrag/base.py:75-79`):
```python
class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int
```

**TypeScript Definition**:
```typescript
interface TextChunkSchema {
  tokens: number;
  content: string;
  full_doc_id: string;
  chunk_order_index: number;
}
```

**Field Descriptions**:
- `tokens`: Number of tokens in the chunk according to the configured tokenizer (e.g., tiktoken for GPT models)
- `content`: The actual text content of the chunk, UTF-8 encoded
- `full_doc_id`: MD5 hash of the complete document, used as a foreign key to link chunks to their source document
- `chunk_order_index`: Zero-based index indicating the chunk's position in the original document sequence

**Storage Pattern**: Chunks are stored in KV storage with keys following the pattern `{full_doc_id}_{chunk_order_index}`. The chunk content is also embedded and stored in Vector storage for similarity search.

**Validation Rules**:
- `tokens` must be > 0 and typically < 2048
- `content` must not be empty
- `full_doc_id` must be a valid MD5 hash (32 hexadecimal characters)
- `chunk_order_index` must be >= 0

### Entity Schema

Entities represent key concepts, people, organizations, locations, and other named entities extracted from documents.

**Python Definition** (Implicit in `lightrag/operate.py`):
```python
entity_data = {
    "entity_name": str,        # Normalized entity name (title case)
    "entity_type": str,        # One of DEFAULT_ENTITY_TYPES
    "description": str,        # Consolidated description
    "source_id": str,          # Chunk IDs joined by GRAPH_FIELD_SEP
    "file_path": str,          # File paths joined by GRAPH_FIELD_SEP
    "created_at": str,         # ISO 8601 timestamp
    "updated_at": str,         # ISO 8601 timestamp
}
```

**TypeScript Definition**:
```typescript
interface EntityData {
  entity_name: string;
  entity_type: EntityType;
  description: string;
  source_id: string;      // Pipe-separated chunk IDs
  file_path: string;      // Pipe-separated file paths
  created_at: string;     // ISO 8601
  updated_at: string;     // ISO 8601
}

type EntityType = 
  | "Person"
  | "Creature"
  | "Organization"
  | "Location"
  | "Event"
  | "Concept"
  | "Method"
  | "Content"
  | "Data"
  | "Artifact"
  | "NaturalObject"
  | "Other";
```

**Field Descriptions**:
- `entity_name`: Normalized name using title case for consistency (e.g., "John Smith", "OpenAI")
- `entity_type`: Classification of the entity using predefined types from `DEFAULT_ENTITY_TYPES`
- `description`: Rich text description of the entity's attributes, activities, and context. May be merged from multiple sources using LLM summarization
- `source_id`: Pipe-separated (`<SEP>`) list of chunk IDs where this entity was mentioned, enabling citation tracking
- `file_path`: Pipe-separated list of source file paths for traceability
- `created_at`: ISO 8601 timestamp when the entity was first created
- `updated_at`: ISO 8601 timestamp when the entity was last modified

**Storage Locations**:
1. Graph Storage: Entity as a node with `entity_name` as the node ID
2. Vector Storage: Entity description embedding with metadata
3. Full Entities KV Storage: Complete entity data for retrieval

**Normalization Rules**:
- Entity names are case-insensitive for matching but stored in title case
- Multiple mentions of the same entity (fuzzy matched) are merged
- Descriptions are consolidated using LLM summarization when they exceed token limits
- `source_id` and `file_path` are deduplicated when merging

### Relationship Schema

Relationships represent connections between entities, forming the edges of the knowledge graph.

**Python Definition** (Implicit in `lightrag/operate.py`):
```python
relationship_data = {
    "src_id": str,             # Source entity name
    "tgt_id": str,             # Target entity name
    "description": str,        # Relationship description
    "keywords": str,           # Comma-separated keywords
    "weight": float,           # Relationship strength (0-1)
    "source_id": str,          # Chunk IDs joined by GRAPH_FIELD_SEP
    "file_path": str,          # File paths joined by GRAPH_FIELD_SEP
    "created_at": str,         # ISO 8601 timestamp
    "updated_at": str,         # ISO 8601 timestamp
}
```

**TypeScript Definition**:
```typescript
interface RelationshipData {
  src_id: string;
  tgt_id: string;
  description: string;
  keywords: string;      // Comma-separated
  weight: number;        // 0.0 to 1.0
  source_id: string;     // Pipe-separated chunk IDs
  file_path: string;     // Pipe-separated file paths
  created_at: string;    // ISO 8601
  updated_at: string;    // ISO 8601
}
```

**Field Descriptions**:
- `src_id`: Name of the source entity (must match an existing entity)
- `tgt_id`: Name of the target entity (must match an existing entity)
- `description`: Explanation of how and why the entities are related
- `keywords`: High-level keywords summarizing the relationship nature (e.g., "collaboration, project, research")
- `weight`: Numeric weight indicating relationship strength, aggregated when merging duplicates
- `source_id`: Chunk IDs where this relationship was mentioned
- `file_path`: Source file paths for citation
- `created_at`: Creation timestamp
- `updated_at`: Last modification timestamp

**Storage Locations**:
1. Graph Storage: Edge between source and target nodes
2. Vector Storage: Relationship description embedding with metadata
3. Full Relations KV Storage: Complete relationship data

**Validation Rules**:
- Relationships are treated as undirected (bidirectional)
- `src_id` and `tgt_id` must reference existing entities
- `weight` must be between 0.0 and 1.0
- Duplicate relationships (same src_id, tgt_id pair) are merged with weights summed

### Document Processing Status

Tracks the processing state of documents through the ingestion pipeline.

**Python Definition** (`lightrag/base.py:679-724`):
```python
@dataclass
class DocProcessingStatus:
    content_summary: str
    content_length: int
    file_path: str
    status: DocStatus
    created_at: str
    updated_at: str
    track_id: str | None = None
    chunks_count: int | None = None
    chunks_list: list[str] | None = field(default_factory=list)
    entities_count: int | None = None
    relations_count: int | None = None
    batch_number: int | None = None
    error_message: str | None = None
```

**TypeScript Definition**:
```typescript
enum DocStatus {
  PENDING = "PENDING",
  CHUNKING = "CHUNKING",
  EXTRACTING = "EXTRACTING",
  MERGING = "MERGING",
  INDEXING = "INDEXING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED"
}

interface DocProcessingStatus {
  content_summary: string;
  content_length: number;
  file_path: string;
  status: DocStatus;
  created_at: string;      // ISO 8601
  updated_at: string;      // ISO 8601
  track_id?: string;
  chunks_count?: number;
  chunks_list?: string[];
  entities_count?: number;
  relations_count?: number;
  batch_number?: number;
  error_message?: string;
}
```

**Field Descriptions**:
- `content_summary`: First 100 characters of document for preview
- `content_length`: Total character length of the document
- `file_path`: Original file path or identifier
- `status`: Current processing stage (see DocStatus enum)
- `created_at`: Document submission timestamp
- `updated_at`: Last status update timestamp
- `track_id`: Optional tracking ID for batch monitoring (shared across multiple documents)
- `chunks_count`: Number of chunks created during splitting
- `chunks_list`: Array of chunk IDs for reference
- `entities_count`: Number of entities extracted
- `relations_count`: Number of relationships extracted
- `batch_number`: Batch identifier for processing order
- `error_message`: Error details if status is FAILED

**State Transitions**:
```
PENDING → CHUNKING → EXTRACTING → MERGING → INDEXING → COMPLETED
                                                           ↓
                                                        FAILED
```

Any stage can transition to FAILED on error, with `error_message` populated with diagnostic information.

## Storage Schema Definitions

### KV Storage Schema

KV storage handles three types of data: LLM response cache, text chunks, and full documents.

#### LLM Cache Entry
**Key Format**: `cache:{hash(prompt+model+params)}`

**Value Schema**:
```typescript
interface LLMCacheEntry {
  return_message: string;
  embedding_dim: number;
  model: string;
  timestamp: string;
}
```

#### Text Chunk Entry
**Key Format**: `{full_doc_id}_{chunk_order_index}`

**Value Schema**:
```typescript
interface ChunkEntry extends TextChunkSchema {
  // Additional metadata can be stored
  file_path?: string;
  created_at?: string;
}
```

#### Full Document Entry
**Key Format**: Document ID (MD5 hash of content)

**Value Schema**:
```typescript
interface FullDocEntry {
  content: string;
  file_path?: string;
  created_at?: string;
  metadata?: Record<string, any>;
}
```

### Vector Storage Schema

Vector storage maintains embeddings for entities, relationships, and chunks.

**Entry Schema**:
```typescript
interface VectorEntry {
  id: string;                    // Unique identifier
  vector: number[];              // Embedding vector (e.g., 1536 dimensions for OpenAI)
  metadata: {
    content: string;             // Original text that was embedded
    type: "entity" | "relation" | "chunk";
    entity_name?: string;        // For entities
    source_id?: string;          // Chunk IDs where this appears
    file_path?: string;          // Source file paths
    [key: string]: any;          // Additional metadata
  };
}
```

**Index Requirements**:
- Cosine similarity search support
- Efficient top-k retrieval (ANN algorithms recommended for large datasets)
- Metadata filtering capabilities
- Batch upsert and deletion support

**Storage Size Estimates**:
- Entity vectors: ~6KB each (1536 floats × 4 bytes)
- Relationship vectors: ~6KB each
- Chunk vectors: ~6KB each
- 10,000 documents ≈ 100,000 chunks ≈ 600MB vector storage

### Graph Storage Schema

Graph storage maintains the entity-relationship graph structure.

#### Node Schema
**Node ID**: Entity name (normalized, title case)

**Node Properties**:
```typescript
interface GraphNode {
  entity_name: string;           // Node ID
  entity_type: string;           // Entity classification
  description: string;           // Entity description
  source_id: string;             // Pipe-separated chunk IDs
  file_path: string;             // Pipe-separated file paths
  created_at: string;            // ISO 8601
  updated_at: string;            // ISO 8601
}
```

#### Edge Schema
**Edge ID**: Combination of source and target node IDs (undirected)

**Edge Properties**:
```typescript
interface GraphEdge {
  src_id: string;                // Source entity name
  tgt_id: string;                // Target entity name
  description: string;           // Relationship description
  keywords: string;              // Comma-separated
  weight: number;                // 0.0 to 1.0
  source_id: string;             // Pipe-separated chunk IDs
  file_path: string;             // Pipe-separated file paths
  created_at: string;            // ISO 8601
  updated_at: string;            // ISO 8601
}
```

**Graph Constraints**:
- Undirected edges: (A, B) and (B, A) represent the same relationship
- No self-loops: src_id ≠ tgt_id
- Unique edge constraint: Only one edge per (src_id, tgt_id) pair
- Node must exist before creating edges

#### Query Capabilities Required
- Node existence check: `has_node(node_id)`
- Edge existence check: `has_edge(src_id, tgt_id)`
- Degree calculation: `node_degree(node_id)`, `edge_degree(src_id, tgt_id)`
- Node retrieval: `get_node(node_id)`, `get_nodes_batch(node_ids[])`
- Edge retrieval: `get_edge(src_id, tgt_id)`, `get_edges_batch(pairs[])`
- Neighborhood query: `get_node_edges(node_id)`
- Graph traversal: `get_knowledge_graph(start_node, max_depth, max_nodes)`
- Label listing: `get_all_labels()`

### Document Status Storage Schema

Document status storage is a specialized KV storage for tracking pipeline state.

**Key Format**: Document ID (MD5 hash)

**Value Schema**: `DocProcessingStatus` (see above)

**Required Capabilities**:
- Get by ID: `get_by_id(doc_id)`
- Get by IDs: `get_by_ids(doc_ids[])`
- Get by status: `get_by_status(status)` → all documents in that state
- Get by track ID: `get_by_track_id(track_id)` → all documents in that batch
- Status counts: `get_status_counts()` → count of documents in each state
- Upsert: `upsert(doc_id, status_data)`
- Delete: `delete(doc_ids[])`

## Query and Response Models

### Query Parameter Model

Comprehensive configuration for query execution.

**Python Definition** (`lightrag/base.py:86-171`):
```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 40
    chunk_top_k: int = 20
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    history_turns: int = 0
    model_func: Callable[..., object] | None = None
    user_prompt: str | None = None
    enable_rerank: bool = True
    include_references: bool = False
```

**TypeScript Definition**:
```typescript
type QueryMode = "local" | "global" | "hybrid" | "naive" | "mix" | "bypass";

interface ConversationMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface QueryParam {
  mode?: QueryMode;
  only_need_context?: boolean;
  only_need_prompt?: boolean;
  response_type?: string;
  stream?: boolean;
  top_k?: number;
  chunk_top_k?: number;
  max_entity_tokens?: number;
  max_relation_tokens?: number;
  max_total_tokens?: number;
  hl_keywords?: string[];
  ll_keywords?: string[];
  conversation_history?: ConversationMessage[];
  history_turns?: number;
  model_func?: (...args: any[]) => Promise<any>;
  user_prompt?: string;
  enable_rerank?: boolean;
  include_references?: boolean;
}
```

**Field Descriptions**:

**Retrieval Configuration**:
- `mode`: Query strategy (see Query Processing documentation)
- `top_k`: Number of entities (local) or relations (global) to retrieve
- `chunk_top_k`: Number of text chunks to keep after reranking

**Token Budget**:
- `max_entity_tokens`: Token budget for entity descriptions in context
- `max_relation_tokens`: Token budget for relationship descriptions
- `max_total_tokens`: Total context budget including system prompt

**Keyword Guidance**:
- `hl_keywords`: High-level keywords for global retrieval (themes, concepts)
- `ll_keywords`: Low-level keywords for local retrieval (specific terms)

**Conversation Context**:
- `conversation_history`: Previous messages for multi-turn dialogue
- `history_turns`: Number of conversation turns to include (deprecated, all history sent)

**Response Configuration**:
- `response_type`: Desired format ("Multiple Paragraphs", "Single Paragraph", "Bullet Points", etc.)
- `stream`: Enable streaming responses via SSE
- `user_prompt`: Additional instructions to inject into the LLM prompt
- `enable_rerank`: Use reranking model for chunk relevance scoring
- `include_references`: Include citation information in response

**Debug Options**:
- `only_need_context`: Return retrieved context without LLM generation
- `only_need_prompt`: Return the constructed prompt without generation

### Query Result Model

Unified response structure for all query types.

**Python Definition** (`lightrag/base.py:778-820`):
```python
@dataclass
class QueryResult:
    content: Optional[str] = None
    response_iterator: Optional[AsyncIterator[str]] = None
    raw_data: Optional[Dict[str, Any]] = None
    is_streaming: bool = False
```

**TypeScript Definition**:
```typescript
interface QueryResult {
  content?: string;
  response_iterator?: AsyncIterableIterator<string>;
  raw_data?: QueryRawData;
  is_streaming: boolean;
}

interface QueryRawData {
  response: string;
  references?: ReferenceEntry[];
  entities?: EntityData[];
  relationships?: RelationshipData[];
  chunks?: ChunkData[];
  processing_info?: ProcessingInfo;
}

interface ReferenceEntry {
  reference_id: string;
  file_path: string;
}

interface ChunkData {
  content: string;
  tokens: number;
  source_id: string;
  file_path: string;
}

interface ProcessingInfo {
  mode: QueryMode;
  keyword_extraction: {
    high_level_keywords: string[];
    low_level_keywords: string[];
  };
  retrieval_stats: {
    entities_retrieved: number;
    relationships_retrieved: number;
    chunks_retrieved: number;
    chunks_after_rerank?: number;
  };
  context_stats: {
    entity_tokens: number;
    relation_tokens: number;
    chunk_tokens: number;
    total_tokens: number;
  };
  token_budget: {
    max_entity_tokens: number;
    max_relation_tokens: number;
    max_total_tokens: number;
    final_entity_tokens: number;
    final_relation_tokens: number;
    final_chunk_tokens: number;
  };
}
```

**Usage Patterns**:

For non-streaming responses:
```typescript
const result = await rag.query("What is AI?", { stream: false });
console.log(result.content);  // Complete response text
console.log(result.raw_data?.references);  // Citation information
```

For streaming responses:
```typescript
const result = await rag.query("What is AI?", { stream: true });
for await (const chunk of result.response_iterator!) {
  process.stdout.write(chunk);  // Stream to output
}
```

For context-only retrieval:
```typescript
const result = await rag.query("What is AI?", { only_need_context: true });
console.log(result.raw_data?.entities);  // Retrieved entities
console.log(result.raw_data?.chunks);    // Retrieved chunks
```

## Configuration Models

### LightRAG Configuration

Complete configuration for a LightRAG instance.

**Python Definition** (`lightrag/lightrag.py:116-384`):
```python
@dataclass
class LightRAG:
    # Storage
    working_dir: str = "./rag_storage"
    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"
    workspace: str = ""
    
    # LLM and Embedding
    llm_model_func: Callable | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_model_max_async: int = 4
    llm_model_timeout: int = 180
    embedding_func: EmbeddingFunc | None = None
    embedding_batch_num: int = 10
    embedding_func_max_async: int = 8
    default_embedding_timeout: int = 30
    
    # Chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tokenizer: Optional[Tokenizer] = None
    tiktoken_model_name: str = "gpt-4o-mini"
    
    # Extraction
    entity_extract_max_gleaning: int = 1
    entity_types: list[str] = field(default_factory=lambda: DEFAULT_ENTITY_TYPES)
    force_llm_summary_on_merge: int = 8
    summary_max_tokens: int = 1200
    summary_language: str = "English"
    
    # Query
    top_k: int = 40
    chunk_top_k: int = 20
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    cosine_threshold: int = 0.2
    related_chunk_number: int = 5
    kg_chunk_pick_method: str = "VECTOR"
    
    # Reranking
    enable_rerank: bool = True
    rerank_model_func: Callable | None = None
    min_rerank_score: float = 0.0
    
    # Concurrency
    max_async: int = 4
    max_parallel_insert: int = 2
    
    # Optional
    addon_params: dict[str, Any] = field(default_factory=dict)
```

**TypeScript Definition**:
```typescript
interface LightRAGConfig {
  // Storage
  working_dir?: string;
  kv_storage?: string;
  vector_storage?: string;
  graph_storage?: string;
  doc_status_storage?: string;
  workspace?: string;
  
  // LLM and Embedding
  llm_model_func?: LLMFunction;
  llm_model_name?: string;
  llm_model_max_async?: number;
  llm_model_timeout?: number;
  embedding_func?: EmbeddingFunction;
  embedding_batch_num?: number;
  embedding_func_max_async?: number;
  default_embedding_timeout?: number;
  
  // Chunking
  chunk_token_size?: number;
  chunk_overlap_token_size?: number;
  tokenizer?: Tokenizer;
  tiktoken_model_name?: string;
  
  // Extraction
  entity_extract_max_gleaning?: number;
  entity_types?: string[];
  force_llm_summary_on_merge?: number;
  summary_max_tokens?: number;
  summary_language?: string;
  
  // Query
  top_k?: number;
  chunk_top_k?: number;
  max_entity_tokens?: number;
  max_relation_tokens?: number;
  max_total_tokens?: number;
  cosine_threshold?: number;
  related_chunk_number?: number;
  kg_chunk_pick_method?: "VECTOR" | "WEIGHT";
  
  // Reranking
  enable_rerank?: boolean;
  rerank_model_func?: RerankFunction;
  min_rerank_score?: number;
  
  // Concurrency
  max_async?: number;
  max_parallel_insert?: number;
  
  // Optional
  addon_params?: Record<string, any>;
}

type LLMFunction = (
  prompt: string,
  system_prompt?: string,
  history_messages?: ConversationMessage[],
  stream?: boolean,
  **kwargs: any
) => Promise<string> | AsyncIterableIterator<string>;

type EmbeddingFunction = (texts: string[]) => Promise<number[][]>;

type RerankFunction = (
  query: string,
  documents: string[]
) => Promise<Array<{ index: number; score: number }>>;

interface Tokenizer {
  encode(text: string): number[];
  decode(tokens: number[]): string;
}
```

## TypeScript Type Mapping

### Python to TypeScript Type Conversion

| Python Type | TypeScript Type | Notes |
|------------|----------------|-------|
| `str` | `string` | Direct mapping |
| `int` | `number` | JavaScript/TypeScript uses `number` for all numerics |
| `float` | `number` | Same as `int` |
| `bool` | `boolean` | Direct mapping |
| `list[T]` | `T[]` or `Array<T>` | Both notations are valid in TypeScript |
| `dict[K, V]` | `Record<K, V>` or `Map<K, V>` | `Record` for simple objects, `Map` for dynamic keys |
| `set[T]` | `Set<T>` | Direct mapping |
| `tuple[T1, T2]` | `[T1, T2]` | TypeScript tuple syntax |
| `Literal["a", "b"]` | `"a" \| "b"` | Union of literal types |
| `Optional[T]` | `T \| undefined` or `T?` | Optional property syntax |
| `Union[T1, T2]` | `T1 \| T2` | Union type |
| `Any` | `any` | Avoid if possible, use `unknown` for type-safe any |
| `TypedDict` | `interface` | TypeScript interface |
| `@dataclass` | `class` or `interface` | Use `class` for behavior, `interface` for pure data |
| `Callable[..., T]` | `(...args: any[]) => T` | Function type |
| `AsyncIterator[T]` | `AsyncIterableIterator<T>` | Async iteration support |

### Python-Specific Features Requiring Special Handling

**Dataclasses with field defaults**:
```python
# Python
@dataclass
class Example:
    name: str
    items: list[str] = field(default_factory=list)
```

```typescript
// TypeScript - Option 1: Class with constructor
class Example {
  name: string;
  items: string[];
  
  constructor(name: string, items: string[] = []) {
    this.name = name;
    this.items = items;
  }
}

// TypeScript - Option 2: Interface with builder
interface Example {
  name: string;
  items: string[];
}

function createExample(name: string, items: string[] = []): Example {
  return { name, items };
}
```

**Multiple inheritance from ABC**:
```python
# Python
@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    pass
```

```typescript
// TypeScript - Use composition over inheritance
abstract class StorageNameSpace {
  abstract initialize(): Promise<void>;
}

abstract class BaseGraphStorage extends StorageNameSpace {
  // Additional abstract methods
}

// Or use interfaces for pure contracts
interface IStorageNameSpace {
  initialize(): Promise<void>;
}

interface IGraphStorage extends IStorageNameSpace {
  // Graph-specific methods
}
```

**Overloaded functions**:
```python
# Python
@overload
def get(id: str) -> dict | None: ...
@overload
def get(ids: list[str]) -> list[dict]: ...
```

```typescript
// TypeScript - Native overload support
function get(id: string): Promise<Record<string, any> | null>;
function get(ids: string[]): Promise<Record<string, any>[]>;
function get(idOrIds: string | string[]): Promise<any> {
  if (typeof idOrIds === 'string') {
    // Single ID logic
  } else {
    // Multiple IDs logic
  }
}
```

### Validation and Serialization

For runtime validation and serialization in TypeScript, consider using:

**Zod for schema validation**:
```typescript
import { z } from 'zod';

const TextChunkSchema = z.object({
  tokens: z.number().positive(),
  content: z.string().min(1),
  full_doc_id: z.string().regex(/^[a-f0-9]{32}$/),
  chunk_order_index: z.number().nonnegative(),
});

type TextChunk = z.infer<typeof TextChunkSchema>;

// Validate at runtime
const chunk = TextChunkSchema.parse(data);
```

**class-transformer for class serialization**:
```typescript
import { plainToClass, classToPlain } from 'class-transformer';
import { IsString, IsNumber } from 'class-validator';

class TextChunk {
  @IsNumber()
  tokens: number;
  
  @IsString()
  content: string;
  
  @IsString()
  full_doc_id: string;
  
  @IsNumber()
  chunk_order_index: number;
}

// Convert plain object to class instance
const chunk = plainToClass(TextChunk, jsonData);

// Convert class instance to plain object
const json = classToPlain(chunk);
```

## Summary

This comprehensive data models documentation provides:

1. **Complete type definitions** for all core data structures in both Python and TypeScript
2. **Storage schemas** detailing how data is persisted in each storage layer
3. **Query and response models** with full field descriptions and usage patterns
4. **Configuration models** for system setup and customization
5. **Type mapping guide** for Python-to-TypeScript conversion
6. **Validation strategies** using TypeScript libraries

These type definitions form the contract layer between all components of the system, ensuring type safety and consistent data structures throughout the implementation. The TypeScript definitions leverage the language's strong type system to provide compile-time safety while maintaining compatibility with the original Python design.

The next documentation sections will use these type definitions extensively when describing storage implementations, API contracts, and LLM integrations.
