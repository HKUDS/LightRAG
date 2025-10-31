# Metadata Filtering in LightRAG

## Overview

LightRAG supports metadata filtering during queries to retrieve only relevant chunks based on metadata criteria.

**Important Limitations**:
- Metadata filtering is **only supported for PostgreSQL (PGVectorStorage), with metadata insertion also visible on Neo4j**
- Only **chunk-based queries** support metadata filtering (Mix and Naive modes)
- Metadata is stored in document status and propagated to chunks during extraction

## Metadata Structure

Metadata is stored as a dictionary (`dict[str, Any]`) in:
- Entity nodes (graph storage)
- Relationship edges (graph storage)
- Text chunks (KV storage)
- Vector embeddings (vector storage)

```python
metadata = {
    "author": "John Doe",
    "department": "Engineering",
    "document_type": "technical_spec",
    "version": "1.0"
}
```

## Critical: Metadata Persistence in Document Status

**Metadata is stored in DocProcessingStatus** - This ensures metadata is not lost if the processing queue is stopped or interrupted.

### How It Works

1. **Document Status Storage** (`lightrag/base.py` - `DocProcessingStatus`)
   ```python
   @dataclass
   class DocProcessingStatus:
       # ... other fields
       metadata: dict[str, Any] = field(default_factory=dict)
       """Additional metadata - PERSISTED across queue restarts"""
   ```

2. **Metadata Flow**:
   - Metadata stored in `DocProcessingStatus.metadata` when document is enqueued
   - If queue stops, metadata persists in document status storage
   - When processing resumes, metadata is read from document status
   - Metadata is propagated to chunks during extraction

3. **Why This Matters**:
   - Queue can be stopped/restarted without losing metadata
   - Metadata survives system crashes or interruptions
   - Ensures data consistency across processing pipeline

## Metadata Filtering During Queries

### MetadataFilter Class

```python
from lightrag.types import MetadataFilter

# Simple filter
filter1 = MetadataFilter(
    operator="AND",
    operands=[{"department": "Engineering"}]
)

# Complex filter with OR
filter2 = MetadataFilter(
    operator="OR",
    operands=[
        {"author": "John Doe"},
        {"author": "Jane Smith"}
    ]
)

# Nested filter
filter3 = MetadataFilter(
    operator="AND",
    operands=[
        {"document_type": "technical_spec"},
        MetadataFilter(
            operator="OR",
            operands=[
                {"version": "1.0"},
                {"version": "2.0"}
            ]
        )
    ]
)
```

### Supported Operators

- **AND**: All conditions must be true
- **OR**: At least one condition must be true
- **NOT**: Negates the condition

## Supported Query Modes

### Mix Mode (Recommended)
Filters vector chunks from both KG and direct vector search:
```python
query_param = QueryParam(
    mode="mix",
    metadata_filter=MetadataFilter(
        operator="AND",
        operands=[
            {"department": "Engineering"},
            {"status": "approved"}
        ]
    )
)
```

### Naive Mode
Filters vector chunks directly:
```python
query_param = QueryParam(
    mode="naive",
    metadata_filter=MetadataFilter(
        operator="AND",
        operands=[{"document_type": "manual"}]
    )
)
```

## Implementation Details

### Architecture Flow

1. **API Layer** (`lightrag/api/routers/query_routes.py`)
   - REST endpoint receives `metadata_filter` as JSON dict
   - Converts JSON to `MetadataFilter` object using `MetadataFilter.from_dict()`
   
2. **QueryParam** (`lightrag/base.py`)
   - `MetadataFilter` object is passed into `QueryParam.metadata_filter`
   - QueryParam carries the filter through the query pipeline
   
3. **Query Execution** (`lightrag/operate.py`)
   - Only chunk-based queries use the filter:
     - Line 2749: `chunks_vdb.query(..., metadata_filter=query_param.metadata_filter)` (Mix/Naive modes)
   
4. **Storage Layer** (`lightrag/kg/postgres_impl.py`)
   - PGVectorStorage: Converts filter to SQL WHERE clause with JSONB operators

### Code Locations

Key files implementing metadata support:
- `lightrag/types.py`: `MetadataFilter` class definition
- `lightrag/base.py`: `QueryParam` with `metadata_filter` field, `DocProcessingStatus` with metadata persistence
- `lightrag/api/routers/query_routes.py`: API endpoint that initializes MetadataFilter from JSON
- `lightrag/operate.py`: Query functions that pass filter to storage (Line 2749)
- `lightrag/kg/postgres_impl.py`: PostgreSQL JSONB filter implementation

## Query Examples

### Example 1: Filter by Department (Mix Mode)
```python
from lightrag import QueryParam
from lightrag.types import MetadataFilter

query_param = QueryParam(
    mode="mix",
    metadata_filter=MetadataFilter(
        operator="AND",
        operands=[{"department": "Engineering"}]
    )
)

response = rag.query("What are the key projects?", param=query_param)
```

### Example 2: Multi-tenant Filtering (Naive Mode)
```python
query_param = QueryParam(
    mode="naive",
    metadata_filter=MetadataFilter(
        operator="AND",
        operands=[
            {"tenant_id": "tenant_a"},
            {"access_level": "admin"}
        ]
    )
)

response = rag.query("Show admin resources", param=query_param)
```

### Example 3: Version Filtering (Mix Mode)
```python
query_param = QueryParam(
    mode="mix",
    metadata_filter=MetadataFilter(
        operator="AND",
        operands=[
            {"doc_type": "manual"},
            {"status": "current"}
        ]
    )
)

response = rag.query("How to configure?", param=query_param)
```

## Storage Backend Support

**Important**: Metadata filtering is currently only supported for PostgreSQL vector storage.

### Vector Storage
- **PGVectorStorage**: Full support with JSONB filtering
- **NanoVectorDBStorage**:  Not supported
- **MilvusVectorDBStorage**:  Not supported
- **ChromaVectorDBStorage**:  Not supported
- **FaissVectorDBStorage**: Not supported
- **QdrantVectorDBStorage**:  Not supported
- **MongoVectorDBStorage**:  Not supported

### Recommended Configuration

For metadata filtering support:
```python
rag = LightRAG(
    working_dir="./storage",
    vector_storage="PGVectorStorage",
    # Graph storage can be any type
    # ... other config
)
```

## Server API Examples

### REST API Query with Metadata Filter

#### Simple Filter (Naive Mode)
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features?",
    "mode": "naive",
    "metadata_filter": {
      "operator": "AND",
      "operands": [
        {"department": "Engineering"},
        {"year": 2024}
      ]
    }
  }'
```

#### Complex Nested Filter (Mix Mode)
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me technical documentation",
    "mode": "mix",
    "metadata_filter": {
      "operator": "AND",
      "operands": [
        {"document_type": "technical_spec"},
        {
          "operator": "OR",
          "operands": [
            {"version": "1.0"},
            {"version": "2.0"}
          ]
        }
      ]
    }
  }'
```

#### Multi-tenant Query (Mix Mode)
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List all projects",
    "mode": "mix",
    "metadata_filter": {
      "operator": "AND",
      "operands": [
        {"tenant_id": "tenant_a"},
        {"access_level": "admin"}
      ]
    },
    "top_k": 20
  }'
```

### Python Client with Server

```python
import requests
from lightrag.types import MetadataFilter

# Option 1: Use MetadataFilter class and convert to dict
metadata_filter = MetadataFilter(
    operator="AND",
    operands=[
        {"department": "Engineering"},
        {"status": "approved"}
    ]
)

response = requests.post(
    "http://localhost:9621/query",
    json={
        "query": "What are the approved engineering documents?",
        "mode": "mix",  # Use mix or naive mode
        "metadata_filter": metadata_filter.to_dict(),
        "top_k": 10
    }
)

# Option 2: Send dict directly (API will convert to MetadataFilter)
response = requests.post(
    "http://localhost:9621/query",
    json={
        "query": "What are the approved engineering documents?",
        "mode": "naive",  # Use mix or naive mode
        "metadata_filter": {
            "operator": "AND",
            "operands": [
                {"department": "Engineering"},
                {"status": "approved"}
            ]
        },
        "top_k": 10
    }
)

result = response.json()
print(result["response"])
```

### How the API Processes Metadata Filters

When you send a query to the REST API:

1. **JSON Request** → API receives `metadata_filter` as a dict
2. **API Conversion** → `MetadataFilter.from_dict()` creates MetadataFilter object
3. **QueryParam** → MetadataFilter is set in `QueryParam.metadata_filter`
4. **Query Execution** → QueryParam with filter is passed to `kg_query()` or `naive_query()`
5. **Storage Query** → Filter is passed to vector storage query methods (chunks only)
6. **SQL** → PGVectorStorage converts filter to JSONB WHERE clause

## Best Practices

### 1. Consistent Metadata Schema
```python
# Good - consistent schema
metadata1 = {"author": "John", "dept": "Eng", "year": 2024}
metadata2 = {"author": "Jane", "dept": "Sales", "year": 2024}
```

### 2. Simple Indexable Values
```python
# Good - simple values
metadata = {
    "status": "approved",
    "priority": "high",
    "year": 2024
}
```

### 3. Use Appropriate Query Mode
- **Mix mode**: Best for combining KG context with filtered chunks
- **Naive mode**: Best for pure vector search with metadata filtering

### 4. Performance Considerations
- Keep metadata fields minimal (Should be done automatically by the ORM)
- For PostgreSQL: Create GIN indexes on JSONB metadata columns:
  ```sql
  CREATE INDEX idx_chunks_metadata ON chunks USING GIN (metadata);
  ```
- Avoid overly complex nested filters

## Troubleshooting

### Filter Not Working
1. **Verify storage backend**: Ensure you're using PGVectorStorage
2. **Verify query mode**: Use "mix" or "naive" mode only
3. Verify metadata exists in chunks
4. Check metadata field names match exactly (case-sensitive)
5. Check logs for filter parsing errors
6. Test without filter first to ensure data exists

### Performance Issues
1. Reduce filter complexity
2. Create GIN indexes on JSONB metadata columns in PostgreSQL
3. Profile query execution time
4. Consider caching frequently used filters

### Unsupported Storage Backend
If you're using a storage backend that doesn't support metadata filtering:
1. Migrate to PGVectorStorage
2. Or implement post-filtering in application code
3. Or contribute metadata filtering support for your backend

### Metadata Not Persisting After Queue Restart
- Metadata is stored in `DocProcessingStatus.metadata`
- Check document status storage is properly configured
- Verify metadata is set before document is enqueued

## API Reference

### MetadataFilter
```python
class MetadataFilter(BaseModel):
    operator: str  # "AND", "OR", or "NOT"
    operands: List[Union[Dict[str, Any], 'MetadataFilter']]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataFilter':
        """Create MetadataFilter from dictionary (used by API)"""
        ...
```

### QueryParam
```python
@dataclass
class QueryParam:
    metadata_filter: MetadataFilter | None = None  # Filter passed to chunk queries
    mode: str = "mix"  # Only "mix" and "naive" support metadata filtering
    top_k: int = 60
    # ... other fields
```

### DocProcessingStatus
```python
@dataclass
class DocProcessingStatus:
    # ... other fields
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata - PERSISTED across queue restarts"""
```

### Query Method
```python
# Synchronous
response = rag.query(
    query: str,
    param: QueryParam  # QueryParam contains metadata_filter
)

# Asynchronous
response = await rag.aquery(
    query: str,
    param: QueryParam  # QueryParam contains metadata_filter
)
```

### REST API Query Endpoint
```python
# In lightrag/api/routers/query_routes.py
@router.post("/query")
async def query_endpoint(request: QueryRequest):
    # API receives metadata_filter as dict
    metadata_filter_dict = request.metadata_filter
    
    # Convert dict to MetadataFilter object
    metadata_filter = MetadataFilter.from_dict(metadata_filter_dict) if metadata_filter_dict else None
    
    # Create QueryParam with MetadataFilter
    query_param = QueryParam(
        mode=request.mode,  # Must be "mix" or "naive"
        metadata_filter=metadata_filter,
        top_k=request.top_k
    )
    
    # Execute query with QueryParam
    result = await rag.aquery(request.query, param=query_param)
    return result
```
