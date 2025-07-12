# LightRAG Test Suite

This directory contains comprehensive tests for the LightRAG graph storage implementations, including both modular and monolithic test structures with bilingual support.

## Test Structure

```
tests/
├── README.md                          # This file
├── __init__.py                        # Package initialization
├── conftest.py                        # Pytest configuration
├── test_graph_storage_modular.py      # Pytest integration for modular tests
├── test_kuzu_impl.py                  # KuzuDB implementation tests
├── test_kuzu_integration.py           # KuzuDB integration tests
├── test_lightrag_ollama_chat.py       # Ollama chat integration tests
└── test_graph_storage/               # Modular test suite
    ├── __init__.py                   # Package initialization
    ├── main.py                       # Direct execution entry point
    ├── core/                         # Core utilities
    │   ├── __init__.py               # Package initialization
    │   ├── storage_setup.py          # Storage initialization
    │   └── translation_engine.py     # Translation system
    ├── tests/                        # Individual test modules
    │   ├── __init__.py               # Package initialization
    │   ├── basic.py                  # Basic operations
    │   ├── advanced.py               # Advanced operations
    │   ├── batch.py                  # Batch operations
    │   ├── special_chars.py          # Special character handling
    │   └── undirected.py             # Undirected graph properties
    └── translations/                 # Translation files
        ├── __init__.py               # Package initialization
        ├── common.py                 # Common translations
        ├── utility.py                # Utility translations
        ├── basic_test.py             # Basic test translations
        ├── advanced_test.py          # Advanced test translations
        ├── batch_test.py             # Batch test translations
        ├── special_char_test.py      # Special character test translations
        └── undirected_test.py        # Undirected test translations
```

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.11+ installed
2. **Dependencies**: Install required packages using:

   ```bash
   pip install -r requirements.txt
   # OR using uv
   uv sync
   ```

3. **Environment Configuration** (Optional):

   ```bash
   # Copy example environment file
   cp .env.example .env

   # Edit .env to configure storage backends
   # Default: NetworkXStorage (no additional setup required)
   # For KuzuDB: LIGHTRAG_GRAPH_STORAGE=KuzuDBStorage
   ```

## Running Tests

### 1. Quick Start - All Tests

```bash
# Run all tests with pytest
pytest tests/ -v

# Using uv (recommended)
uv run pytest tests/ -v
```

### 2. Modular Test Suite

The modular test suite provides clean, organized tests with bilingual support:

```bash
# Run all modular tests
uv run pytest tests/test_graph_storage_modular.py -v

# Run specific test categories
uv run pytest tests/test_graph_storage_modular.py::test_basic_graph_operations -v
uv run pytest tests/test_graph_storage_modular.py::test_advanced_graph_operations -v
uv run pytest tests/test_graph_storage_modular.py::test_batch_graph_operations -v
uv run pytest tests/test_graph_storage_modular.py::test_special_characters_handling -v
uv run pytest tests/test_graph_storage_modular.py::test_undirected_graph_properties -v

# Run Chinese language variants
uv run pytest tests/test_graph_storage_modular.py::test_basic_graph_operations_chinese -v
uv run pytest tests/test_graph_storage_modular.py::test_special_characters_handling_chinese -v
```

### 3. Direct Module Execution

You can run the main test suite directly:

```bash
# Run all tests via main entry point (from project root)
python -m tests.test_graph_storage.main

# Note: Individual test modules are designed to be run via pytest
# Use the specific pytest commands shown in section 2 above
```

### 4. Storage Backend Testing

#### NetworkX Storage (Default)

```bash
# No additional setup required
uv run pytest tests/test_graph_storage_modular.py -v
```

#### KuzuDB Storage

```bash
# Test with KuzuDB backend
LIGHTRAG_GRAPH_STORAGE=KuzuDBStorage uv run pytest tests/test_graph_storage_modular.py -v

# KuzuDB specific tests
uv run pytest tests/test_kuzu_impl.py -v
uv run pytest tests/test_kuzu_integration.py -v
```

#### Other Storage Backends

```bash
# Neo4j (requires Neo4j instance)
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage uv run pytest tests/test_graph_storage_modular.py -v

# MongoDB (requires MongoDB instance)
LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage uv run pytest tests/test_graph_storage_modular.py -v
```

### 5. Language Testing

The test suite supports bilingual execution:

```bash
# Test with English translations
TEST_LANGUAGE=english uv run pytest tests/test_graph_storage_modular.py -v

# Test with Chinese translations
TEST_LANGUAGE=chinese uv run pytest tests/test_graph_storage_modular.py -v
```

### 6. Advanced Testing Options

```bash
# Run with detailed output
uv run pytest tests/test_graph_storage_modular.py -v -s

# Run specific test pattern
uv run pytest tests/test_graph_storage_modular.py -k "basic" -v

# Run with coverage
uv run pytest tests/test_graph_storage_modular.py --cov=lightrag --cov-report=html

# Run with parallel execution
uv run pytest tests/test_graph_storage_modular.py -n auto

# Run with specific markers
uv run pytest tests/test_graph_storage_modular.py -m "asyncio" -v
```

## Test Categories

### Basic Operations (`test_basic_graph_operations`)

- Node insertion and retrieval
- Edge creation and properties
- Basic graph traversal
- Property validation

### Advanced Operations (`test_advanced_graph_operations`)

- Complex graph structures
- Multi-hop relationships
- Advanced queries
- Performance validation

### Batch Operations (`test_batch_graph_operations`)

- Bulk node insertion
- Bulk edge creation
- Transaction handling
- Performance optimization

### Special Characters (`test_special_characters_handling`)

- Unicode support
- Special character encoding
- Internationalization
- Edge cases

### Undirected Properties (`test_undirected_graph_properties`)

- Bidirectional relationships
- Undirected graph behavior
- Consistency validation
- Property symmetry

## Troubleshooting

### Common Issues

1. **Storage Initialization Failed**

   ```bash
   # Check if storage backend is properly configured
   # For KuzuDB, ensure write permissions in test directory
   ```

2. **Translation Errors**

   ```bash
   # Ensure TEST_LANGUAGE is set correctly
   export TEST_LANGUAGE=english  # or chinese
   ```

3. **Missing Dependencies**

   ```bash
   # Install missing packages
   uv sync
   pip install -r requirements.txt
   ```

4. **Permission Errors (KuzuDB)**
   ```bash
   # Ensure write permissions for temporary directories
   chmod 755 /tmp
   ```

### Environment Variables

| Variable                 | Description                 | Default           |
| ------------------------ | --------------------------- | ----------------- |
| `LIGHTRAG_GRAPH_STORAGE` | Storage backend to use      | `NetworkXStorage` |
| `TEST_LANGUAGE`          | Language for test output    | `english`         |
| `KUZU_DB_PATH`           | KuzuDB database path        | Auto-generated    |
| `WORKING_DIR`            | Working directory for tests | `./rag_storage`   |

## Test Results

### Expected Output

Successful test runs should show:

- ✅ All tests passing
- 📊 Coverage information (if enabled)
- 🌐 Bilingual output (if language variants are run)
- 🔄 Proper cleanup of temporary resources

### Performance Benchmarks

Typical execution times:

- Basic operations: ~0.1-0.2 seconds
- Advanced operations: ~0.2-0.5 seconds
- Batch operations: ~0.3-0.8 seconds
- Full test suite: ~1-2 seconds

## Contributing

When adding new tests:

1. **Follow the modular structure** - Add new test files to `tests/test_graph_storage/tests/`
2. **Add translations** - Create corresponding translation files in `translations/`
3. **Update pytest integration** - Add new tests to `test_graph_storage_modular.py`
4. **Document changes** - Update this README with new test categories

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review test logs for specific error messages
3. Ensure all dependencies are properly installed
4. Verify environment configuration

---

_This test suite provides comprehensive coverage for LightRAG graph storage implementations with support for multiple backends and languages._
