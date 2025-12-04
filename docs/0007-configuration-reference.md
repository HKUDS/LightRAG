# LightRAG Configuration Reference

## Configuration Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONFIGURATION LAYERS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. COMMAND LINE ARGUMENTS (highest priority)                          │ │
│  │     python -m lightrag.api.lightrag_server --port 9621 --workers 2     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  2. ENVIRONMENT VARIABLES                                              │ │
│  │     export PORT=9621                                                   │ │
│  │     export LLM_MODEL=gpt-4o-mini                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  3. .env FILE                                                          │ │
│  │     LLM_MODEL=gpt-4o-mini                                             │ │
│  │     EMBEDDING_DIM=1536                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  4. PYTHON DATACLASS DEFAULTS (lowest priority)                        │ │
│  │     LightRAG(working_dir="./rag_storage")                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Server Configuration

### Basic Server Settings

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `HOST` | `--host` | `0.0.0.0` | Server bind address |
| `PORT` | `--port` | `9621` | Server port |
| `WORKERS` | `--workers` | `2` | Number of worker processes |
| `TIMEOUT` | `--timeout` | `300` | Request timeout (seconds) |
| `WORKING_DIR` | `--working-dir` | `./rag_storage` | RAG storage directory |
| `INPUT_DIR` | `--input-dir` | `./inputs` | Document input directory |

### SSL/TLS Configuration

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `SSL` | `--ssl` | `false` | Enable HTTPS |
| `SSL_CERTFILE` | `--ssl-certfile` | `None` | Path to SSL certificate |
| `SSL_KEYFILE` | `--ssl-keyfile` | `None` | Path to SSL private key |

### Logging Configuration

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `LOG_LEVEL` | `--log-level` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `VERBOSE` | `--verbose` | `false` | Enable verbose debug output |
| `LOG_MAX_BYTES` | — | `10485760` | Max log file size (10MB) |
| `LOG_BACKUP_COUNT` | — | `5` | Number of log backups |
| `LOG_FILENAME` | — | `lightrag.log` | Log filename |

---

## 2. LLM Configuration

### Provider Selection

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `LLM_BINDING` | `--llm-binding` | `openai` | LLM provider |
| `LLM_MODEL` | — | `mistral-nemo:latest` | Model name |
| `LLM_BINDING_HOST` | — | Provider-specific | API endpoint URL |
| `LLM_BINDING_API_KEY` | — | `None` | API key |

**Supported LLM Bindings:**
- `openai` - OpenAI API (default)
- `ollama` - Local Ollama
- `lollms` - LoLLMs server
- `azure_openai` - Azure OpenAI Service
- `aws_bedrock` - AWS Bedrock
- `openai-ollama` - OpenAI LLM + Ollama embeddings

### Provider-Specific Host Defaults

| Provider | Default Host |
|----------|-------------|
| `openai` | `https://api.openai.com/v1` |
| `ollama` | `http://localhost:11434` |
| `lollms` | `http://localhost:9600` |
| `azure_openai` | `AZURE_OPENAI_ENDPOINT` env |

### LLM Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ASYNC` | `4` | Max concurrent LLM requests |
| `LLM_TIMEOUT` | `180` | LLM request timeout (seconds) |
| `ENABLE_LLM_CACHE` | `true` | Enable LLM response caching |
| `ENABLE_LLM_CACHE_FOR_EXTRACT` | `true` | Cache extraction responses |
| `TEMPERATURE` | `1.0` | LLM temperature setting |

### Ollama-Specific Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_NUM_CTX` | `32768` | Context window size |
| `OLLAMA_EMULATING_MODEL_NAME` | `lightrag` | Simulated model name |
| `OLLAMA_EMULATING_MODEL_TAG` | `latest` | Simulated model tag |

---

## 3. Embedding Configuration

### Embedding Provider

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `EMBEDDING_BINDING` | `--embedding-binding` | `openai` | Embedding provider |
| `EMBEDDING_MODEL` | — | `bge-m3:latest` | Embedding model name |
| `EMBEDDING_DIM` | — | `1024` | Embedding dimensions |
| `EMBEDDING_BINDING_HOST` | — | Provider-specific | API endpoint |
| `EMBEDDING_BINDING_API_KEY` | — | `""` | API key |

**Supported Embedding Bindings:**
- `openai` - OpenAI embeddings
- `ollama` - Local Ollama embeddings
- `azure_openai` - Azure OpenAI embeddings
- `aws_bedrock` - AWS Bedrock Titan
- `jina` - Jina embeddings

### Embedding Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_FUNC_MAX_ASYNC` | `8` | Max concurrent embedding requests |
| `EMBEDDING_BATCH_NUM` | `10` | Batch size for embeddings |
| `EMBEDDING_TIMEOUT` | `30` | Embedding request timeout (seconds) |

---

## 4. Storage Configuration

### Storage Backend Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_KV_STORAGE` | `JsonKVStorage` | Key-value storage backend |
| `LIGHTRAG_VECTOR_STORAGE` | `NanoVectorDBStorage` | Vector storage backend |
| `LIGHTRAG_GRAPH_STORAGE` | `NetworkXStorage` | Graph storage backend |
| `LIGHTRAG_DOC_STATUS_STORAGE` | `JsonDocStatusStorage` | Document status storage |

### Available Storage Backends

**KV Storage Options:**
| Value | Use Case |
|-------|----------|
| `JsonKVStorage` | Development, file-based |
| `RedisKVStorage` | Production, distributed |
| `PGKVStorage` | Production, PostgreSQL |
| `MongoKVStorage` | Production, MongoDB |

**Vector Storage Options:**
| Value | Use Case |
|-------|----------|
| `NanoVectorDBStorage` | Development, in-memory |
| `PGVectorStorage` | Production, PostgreSQL |
| `MilvusVectorDBStorage` | Production, high-scale |
| `QdrantStorage` | Production, managed |
| `FAISSStorage` | Production, local |
| `RedisVectorStorage` | Production, distributed |
| `MongoDBVectorStorage` | Production, MongoDB |

**Graph Storage Options:**
| Value | Use Case |
|-------|----------|
| `NetworkXStorage` | Development, in-memory |
| `Neo4JStorage` | Production, native graph |
| `PGGraphStorage` | Production, PostgreSQL |
| `AGEStorage` | Production, PostgreSQL AGE |
| `MemgraphStorage` | Production, real-time |
| `GremlinStorage` | Production, Gremlin-compatible |

### PostgreSQL Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | — | Database user |
| `POSTGRES_PASSWORD` | — | Database password |
| `POSTGRES_DATABASE` | — | Database name |

### Neo4j Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |

### Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URI` | `redis://localhost:6379` | Redis connection URI |

### Milvus Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |
| `MILVUS_USER` | — | Milvus user |
| `MILVUS_PASSWORD` | — | Milvus password |
| `MILVUS_DB_NAME` | `default` | Milvus database |

---

## 5. Document Processing Configuration

### Chunking Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1200` | Token size per chunk |
| `CHUNK_OVERLAP_SIZE` | `100` | Overlap between chunks |

### Entity Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARY_LANGUAGE` | `English` | Language for summaries |
| `MAX_GLEANING` | `1` | Entity extraction iterations |
| `ENTITY_TYPES` | See below | Types of entities to extract |

**Default Entity Types:**
```python
ENTITY_TYPES = [
    "Person", "Creature", "Organization", "Location",
    "Event", "Concept", "Method", "Content",
    "Data", "Artifact", "NaturalObject"
]
```

### Summary Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARY_MAX_TOKENS` | `1200` | Max tokens per summary |
| `SUMMARY_LENGTH_RECOMMENDED` | `600` | Recommended summary length |
| `SUMMARY_CONTEXT_SIZE` | `12000` | Context window for summarization |
| `FORCE_LLM_SUMMARY_ON_MERGE` | `8` | Fragment threshold for LLM summary |

### Document Loader

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENT_LOADING_ENGINE` | `DEFAULT` | Document parser (DEFAULT/DOCLING) |

---

## 6. Query Configuration

### QueryParam Class

```python
@dataclass
class QueryParam:
    mode: str = "mix"                    # Query mode
    only_need_context: bool = False      # Return only context
    only_need_prompt: bool = False       # Return only prompt
    response_type: str = "Multiple Paragraphs"
    stream: bool = False                 # Enable streaming
    top_k: int = 40                      # Top results
    chunk_top_k: int = 20                # Top chunks
    max_entity_tokens: int = 6000        # Entity token budget
    max_relation_tokens: int = 8000      # Relation token budget
    max_total_tokens: int = 30000        # Total token budget
    enable_rerank: bool = True           # Enable reranking
    include_references: bool = False     # Include citations
```

### Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `naive` | Basic vector search | Simple lookups |
| `local` | Entity-focused retrieval | Specific entities |
| `global` | High-level summaries | Broad questions |
| `hybrid` | Local + Global combined | Balanced queries |
| `mix` | KG + Vector integration | Complex questions |
| `bypass` | Skip RAG, direct LLM | Testing |

### Query Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `40` | Top-K entities/relations |
| `CHUNK_TOP_K` | `20` | Top-K text chunks |
| `MAX_ENTITY_TOKENS` | `6000` | Entity context budget |
| `MAX_RELATION_TOKENS` | `8000` | Relation context budget |
| `MAX_TOTAL_TOKENS` | `30000` | Total context budget |
| `COSINE_THRESHOLD` | `0.2` | Similarity threshold |
| `RELATED_CHUNK_NUMBER` | `5` | Related chunks to retrieve |
| `HISTORY_TURNS` | `0` | Conversation history turns |

---

## 7. Reranking Configuration

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `RERANK_BINDING` | `--rerank-binding` | `null` | Rerank provider |
| `RERANK_MODEL` | — | `None` | Rerank model name |
| `RERANK_BINDING_HOST` | — | `None` | Rerank API endpoint |
| `RERANK_BINDING_API_KEY` | — | `None` | Rerank API key |
| `MIN_RERANK_SCORE` | — | `0.0` | Minimum rerank score |
| `RERANK_BY_DEFAULT` | — | `true` | Enable reranking by default |

**Supported Rerank Bindings:**
- `null` - Disabled
- `cohere` - Cohere rerank
- `jina` - Jina rerank
- `aliyun` - Aliyun rerank

---

## 8. Authentication & Security

### API Authentication

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| `LIGHTRAG_API_KEY` | `--key` | `None` | Simple API key auth |
| `AUTH_ACCOUNTS` | — | `""` | Format: `user1:pass1,user2:pass2` |
| `AUTH_USER` | — | `""` | Single username (if AUTH_ACCOUNTS not set) |
| `AUTH_PASS` | — | `""` | Single password |

### JWT Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_SECRET` | `lightrag-jwt-default-secret` | JWT signing secret |
| `TOKEN_EXPIRE_HOURS` | `48` | Token expiration (hours) |
| `GUEST_TOKEN_EXPIRE_HOURS` | `24` | Guest token expiration |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |

### CORS & Network

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `WHITELIST_PATHS` | `/health,/api/*` | Public paths |

### Multi-Tenant Security

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_MULTI_TENANT_STRICT` | `true` | Enforce tenant isolation |
| `LIGHTRAG_REQUIRE_USER_AUTH` | `true` | Require user authentication |
| `LIGHTRAG_SUPER_ADMIN_USERS` | `admin` | Comma-separated admin usernames |

---

## 9. Multi-Tenancy Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MULTI_TENANTS` | `false` | Enable multi-tenant mode |
| `WORKSPACE` | `""` | Default workspace name |
| `MAX_GRAPH_NODES` | `1000` | Max graph nodes per query |
| `MAX_PARALLEL_INSERT` | `2` | Max parallel inserts |

---

## 10. Complete .env Example

```bash
# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
HOST=0.0.0.0
PORT=9621
WORKERS=2
TIMEOUT=300
WORKING_DIR=./rag_storage
INPUT_DIR=./inputs
LOG_LEVEL=INFO

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=sk-xxx
MAX_ASYNC=4
ENABLE_LLM_CACHE=true

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIM=1536
EMBEDDING_BINDING_API_KEY=sk-xxx
EMBEDDING_FUNC_MAX_ASYNC=8
EMBEDDING_BATCH_NUM=10

# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage

# PostgreSQL (if using PG storage)
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_USER=lightrag
# POSTGRES_PASSWORD=secret
# POSTGRES_DATABASE=lightrag

# Neo4j (if using Neo4J storage)
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=secret

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
SUMMARY_LANGUAGE=English
SUMMARY_MAX_TOKENS=1200
SUMMARY_CONTEXT_SIZE=12000

# =============================================================================
# QUERY CONFIGURATION
# =============================================================================
TOP_K=40
CHUNK_TOP_K=20
MAX_ENTITY_TOKENS=6000
MAX_RELATION_TOKENS=8000
MAX_TOTAL_TOKENS=30000
COSINE_THRESHOLD=0.2

# =============================================================================
# RERANKING (optional)
# =============================================================================
# RERANK_BINDING=cohere
# RERANK_MODEL=rerank-english-v2.0
# RERANK_BINDING_API_KEY=xxx

# =============================================================================
# AUTHENTICATION
# =============================================================================
AUTH_USER=admin
AUTH_PASS=admin123
TOKEN_SECRET=your-secure-secret-key
TOKEN_EXPIRE_HOURS=48
CORS_ORIGINS=*

# =============================================================================
# MULTI-TENANCY (optional)
# =============================================================================
# ENABLE_MULTI_TENANTS=true
# LIGHTRAG_MULTI_TENANT_STRICT=true
# LIGHTRAG_REQUIRE_USER_AUTH=true
# LIGHTRAG_SUPER_ADMIN_USERS=admin
```

---

## 11. Python LightRAG Dataclass

```python
@dataclass
class LightRAG:
    # Directory configuration
    working_dir: str = "./rag_storage"
    
    # LLM configuration
    llm_model_name: str = "gpt-4o-mini"
    llm_model_func: Callable = openai_complete_if_cache
    llm_model_max_async: int = 4
    llm_model_max_token_size: int = 32768
    llm_model_kwargs: dict = field(default_factory=dict)
    
    # Embedding configuration
    embedding_model_name: str = "text-embedding-ada-002"
    embedding_func: EmbeddingFunc = None
    embedding_batch_num: int = 10
    embedding_func_max_async: int = 8
    
    # Chunking configuration
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    
    # Entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 1200
    
    # Storage backends
    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"
    
    # Advanced options
    enable_llm_cache: bool = True
    enable_llm_cache_for_entity_extract: bool = True
    auto_manage_storages_states: bool = True
    
    # Namespace (for multi-tenancy)
    namespace: str = None
```

---

**Related Documentation:**
- [Architecture Overview](0002-architecture-overview.md)
- [Storage Backends](0004-storage-backends.md)
- [LLM Integration](0005-llm-integration.md)
- [Deployment Guide](0006-deployment-guide.md)
