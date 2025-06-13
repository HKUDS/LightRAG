# LightRAG Advanced Features Migration Guide

## Overview

This guide helps you migrate from the old monkey-patching approach to the new clean architecture using `AdvancedLightRAG`. The new implementation provides all advanced features while maintaining clean separation of concerns and following best practices.

## What Changed

### Before (Old Approach)
- Used monkey-patching to add features
- Circular imports and complex initialization
- Mixed concerns between modules
- Difficult to maintain and update

### After (New Approach)
- Clean inheritance with `AdvancedLightRAG`
- Proper module separation
- No monkey-patching or circular imports
- Easy to maintain and extend

## Migration Steps

### 1. Basic Migration

**Old Code:**
```python
from lightrag import LightRAG

rag = LightRAG(working_dir="./storage")
response = rag.query("What is machine learning?")
print(response)  # String response
```

**New Code:**
```python
from lightrag.advanced_lightrag import AdvancedLightRAG

rag = AdvancedLightRAG(working_dir="./storage")
response, details = rag.query("What is machine learning?")
print(response)  # Same string response
print(f"Retrieved {details.get('retrieved_entities_count', 0)} entities")
```

### 2. Using the Factory Function

For simpler setup with sensible defaults:

```python
from lightrag.advanced_lightrag import create_advanced_lightrag

rag = create_advanced_lightrag(
    working_dir="./storage",
    enable_all_features=True,
    query_log_level="VERBOSE"
)
```

### 3. Configuring Advanced Features

```python
rag = AdvancedLightRAG(
    working_dir="./storage",
    # Query Logging
    query_log_file_path="./logs/queries.log",
    query_log_level="STANDARD",
    query_log_max_file_size_bytes=10_000_000,  # 10MB
    query_log_backup_count=5,
    # Advanced Features
    enable_mix_mode=True,
    enable_relationship_types=True,
    enable_semantic_weights=True,
    enable_retrieval_details=True,
    # Standard LightRAG parameters
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
)
```

### 4. Mix Mode Queries

Mix mode combines knowledge graph and vector search:

```python
from lightrag.base import QueryParam

param = QueryParam(mode="mix", top_k=10)
response, details = await rag.aquery(
    "What are the applications of machine learning?",
    param=param
)

# Access detailed retrieval information
kg_details = details.get("kg_retrieval_details", {})
vector_details = details.get("vector_retrieval_details", {})
print(f"KG entities: {kg_details.get('retrieved_entities_count', 0)}")
print(f"Vector chunks: {vector_details.get('retrieved_chunks_count', 0)}")
```

### 5. Query Logging

The new system automatically logs all queries with detailed metrics:

```python
# Queries are automatically logged
response, details = await rag.aquery("Your question here")

# Access the query logger directly if needed
query_logger = await rag.get_query_logger_instance()
```

## Feature Mapping

| Old Feature | New Implementation | Status |
|-------------|-------------------|---------|
| Monkey-patched functions | `advanced_operate.py` functions | ✅ Improved |
| Hardcoded relationship types | `RelationshipTypeRegistry` | ✅ Improved |
| Mixed initialization | Clean class inheritance | ✅ Improved |
| Basic error handling | Comprehensive error handling | ✅ Improved |
| No query logging | Full query logging system | ✅ New |
| Limited retrieval info | Detailed retrieval tracking | ✅ New |

## Advanced Usage Examples

### 1. Accessing Retrieval Details

```python
response, details = await rag.aquery("Complex query about AI")

# Timing information
timings = details.get("timings", {})
print(f"Keyword extraction: {timings.get('keyword_extraction_ms', 0):.2f}ms")
print(f"Context building: {timings.get('context_build_ms', 0):.2f}ms")
print(f"LLM call: {timings.get('llm_call_ms', 0):.2f}ms")

# Retrieved content summaries
entities = details.get("retrieved_entities_summary", [])
for entity in entities[:5]:  # Show first 5
    print(f"Entity: {entity['name']} ({entity['type']})")
```

### 2. Error Handling

```python
try:
    response, details = await rag.aquery("Your query")
    if "error" in details:
        print(f"Query had issues: {details['error']}")
    else:
        print("Query successful!")
except Exception as e:
    print(f"Query failed: {e}")
```

### 3. Streaming with Details

```python
param = QueryParam(stream=True)
response, details = await rag.aquery("Tell me about AI", param=param)

# Handle streaming response
if hasattr(response, '__aiter__'):
    async for chunk in response:
        print(chunk, end='', flush=True)
    print()  # New line after streaming
```

## Breaking Changes

1. **Return Format**: All query methods now return `(response, details)` tuples instead of just the response
2. **Import Paths**: Import from `lightrag.advanced_lightrag` instead of patching
3. **Configuration**: Advanced features are configured in the constructor

## Backward Compatibility

For minimal changes, you can extract just the response:

```python
response, _ = rag.query("Your question")  # Ignore details
# or
response = rag.query("Your question")[0]  # Get first element
```

## Benefits of Migration

1. **Clean Architecture**: No more monkey-patching or circular imports
2. **Better Maintainability**: Clear separation of concerns
3. **Enhanced Features**: More detailed tracking and logging
4. **Future-Proof**: Easy to extend and update
5. **Production Ready**: Comprehensive error handling and logging

## Troubleshooting

### Import Errors
If you get import errors for advanced modules:
```python
# The system gracefully falls back to basic functionality
rag = AdvancedLightRAG(
    enable_relationship_types=False,  # Disable if registry unavailable
    enable_semantic_weights=False,   # Disable if utils unavailable
    enable_retrieval_details=False,  # Use basic mode
)
```

### Performance Concerns
The new implementation is optimized but if you need maximum performance:
```python
rag = AdvancedLightRAG(
    enable_retrieval_details=False,  # Disable detailed tracking
    query_log_level="MINIMAL",       # Reduce logging overhead
)
```

## Support

For issues with migration:
1. Check that all required utility modules are available
2. Review the error logs for specific import failures
3. Use the graceful degradation features for partial functionality
4. Refer to the comprehensive examples in the documentation

The migration provides a clean foundation for future enhancements while preserving all existing functionality. 