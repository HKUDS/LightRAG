# KuzuDB Integration Testing

This document describes the KuzuDB integration with the LightRAG testing framework.

## Overview

KuzuDB has been successfully integrated into the LightRAG project as a graph storage backend. The integration provides:

1. **Automatic test environment setup** - Creates temporary databases for testing
2. **Comprehensive graph operations** - Support for all standard graph operations
3. **Seamless integration** - Works with existing test infrastructure
4. **Proper cleanup** - Automatically cleans up temporary files after tests

## Files

### Main Integration Files

- `lightrag/kg/kuzu_impl.py` - KuzuDB storage implementation
- `tests/test_kuzu_impl.py` - KuzuDB-specific tests
- `tests/test_graph_storage.py` - Universal graph storage tests (KuzuDB-aware)
- `tests/test_kuzu_integration.py` - Integration verification tests

### Configuration

KuzuDB integration uses the following environment variables:

- `KUZU_DB_PATH` - Path to the KuzuDB database file
- `KUZU_WORKSPACE` - Workspace name for the KuzuDB instance
- `LIGHTRAG_GRAPH_STORAGE=KuzuDBStorage` - Enable KuzuDB in universal tests

## Running Tests

### 1. KuzuDB-Specific Tests

Run the dedicated KuzuDB tests:

```bash
# Run basic KuzuDB tests
uv run python tests/test_kuzu_impl.py

# Run with pytest (requires asyncio mode)
uv run pytest tests/test_kuzu_impl.py --asyncio-mode=auto -v

# Run specific test
uv run pytest tests/test_kuzu_impl.py::TestKuzuDBStorage::test_initialization --asyncio-mode=auto -v
```

### 2. Universal Graph Storage Tests

Run the universal tests with KuzuDB:

```bash
# Set environment variable and run interactive tests
LIGHTRAG_GRAPH_STORAGE=KuzuDBStorage uv run python tests/test_graph_storage.py

# For non-interactive testing, create a .env file with:
# LIGHTRAG_GRAPH_STORAGE=KuzuDBStorage
```

### 3. Integration Verification

Run the integration verification tests:

```bash
uv run python tests/test_kuzu_integration.py
```

## Features Tested

### Basic Operations
- ✅ Node insertion and retrieval
- ✅ Edge insertion and retrieval
- ✅ Database initialization and cleanup

### Advanced Operations
- ✅ Node degree calculation
- ✅ Edge degree calculation
- ✅ Batch operations
- ✅ Knowledge graph retrieval
- ✅ Chunk-based queries

### Data Integrity
- ✅ Special character handling
- ✅ Undirected graph properties
- ✅ Proper cleanup and finalization

## Environment Setup

The KuzuDB integration automatically handles environment setup:

1. **Temporary Database Creation**: Creates isolated test databases
2. **Workspace Configuration**: Sets up proper workspace labeling
3. **Automatic Cleanup**: Removes temporary files after tests
4. **Error Handling**: Proper exception handling and resource cleanup

## Configuration Options

### Global Configuration

```python
global_config = {
    "max_graph_nodes": 1000,  # Maximum nodes in knowledge graph
    "embedding_batch_num": 10,  # Batch size for embeddings
    "working_dir": "./rag_storage",  # Working directory
}
```

### KuzuDB-Specific Configuration

```python
storage = KuzuDBStorage(
    namespace="test_graph",
    global_config=global_config,
    embedding_func=mock_embedding_func,
    workspace="test_workspace"  # Optional workspace name
)
```

## Error Handling

The integration includes comprehensive error handling:

- **Connection Issues**: Automatic retry and proper error reporting
- **Resource Cleanup**: Guaranteed cleanup even on test failures
- **Environment Variables**: Automatic setup and validation
- **Temporary Files**: Proper cleanup of test databases

## Performance

KuzuDB provides excellent performance for graph operations:

- **Fast Initialization**: Quick database setup and teardown
- **Efficient Queries**: Optimized graph traversal and queries
- **Memory Management**: Proper resource management
- **Batch Operations**: Efficient bulk operations

## Next Steps

1. Add more comprehensive benchmarking tests
2. Implement stress testing for large graphs
3. Add performance comparison with other graph backends
4. Enhance error recovery mechanisms

## Support

For issues related to KuzuDB integration:

1. Check the KuzuDB documentation: https://kuzudb.com/
2. Review the implementation in `lightrag/kg/kuzu_impl.py`
3. Run the integration tests to verify your setup
4. Check environment variable configuration
