# LightRAG Test Suite

This directory contains comprehensive tests for the LightRAG graph storage implementations, including both modular test structures with bilingual support and legacy support for direct execution.

## Test Structure

```
tests/
├── README.md                          # This file
├── __init__.py                        # Package initialization
├── conftest.py                        # Pytest configuration
├── test_cli.py                        # Interactive CLI test runner
├── test_graph_storage.py              # Graph storage test suite
├── test_kuzu_impl.py                  # KuzuDB implementation tests
├── test_kuzu_integration.py           # KuzuDB integration tests
├── test_lightrag_ollama_chat.py       # Ollama chat integration tests
└── graph/                             # Modular test suite
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

## Quick Access 🚀

Use the root-level launcher for the easiest testing experience:

```bash
# Interactive mode with bilingual language selection
<python/uv run> run_tests.py

# Or make it executable and run directly
chmod +x run_tests.py && ./run_tests.py
```

## Running Tests

### 1. Quick Start - All Tests

```bash
# Run all tests with pytest
<pytest/uv run pytest> tests/ -v
```

### 2. Interactive CLI Test Runner 🎯 [RECOMMENDED]

**The easiest way to run tests** - Interactive CLI with bilingual support:

```bash
# Interactive mode (starts with bilingual language selection)
<python/uv run> tests/test_cli.py

# Or use the root-level launcher
<python/uv run> run_tests.py
```

**Quick CLI Mode** - Skip interaction by providing all parameters:

```bash
# Run specific test with English interface
<python/uv run> tests/test_cli.py --language english --storage NetworkXStorage --tests basic

# Run multiple tests
<python/uv run> tests/test_cli.py --language english --storage NetworkXStorage --tests basic advanced

# Run all tests with Chinese interface
<python/uv run> tests/test_cli.py --language chinese --storage KuzuDBStorage --tests all
```

**CLI Features:**

- 🌐 **Bilingual Support**: Always starts with bilingual language selection
- 🔧 **Storage Backend Selection**: NetworkX, KuzuDB, Neo4j, or MongoDB
- 🎯 **Test Selection**: Choose specific tests or run all
- 📊 **Rich Output**: Colored output with progress indicators and summaries
- ⚡ **Quick Mode**: Non-interactive execution with command-line parameters
- 🔄 **Repeatable**: Option to run additional tests after completion

### 3. Pytest Integration

For traditional pytest workflows:

```bash
# Run all graph storage tests
<uv run pytest/pytest> tests/test_graph_storage.py -v

# Run specific test categories
<uv run pytest/pytest> tests/test_graph_storage.py::test_basic_graph_operations -v
<uv run pytest/pytest> tests/test_graph_storage.py::test_advanced_graph_operations -v
<uv run pytest/pytest> tests/test_graph_storage.py::test_batch_graph_operations -v
<uv run pytest/pytest> tests/test_graph_storage.py::test_special_characters_handling -v
<uv run pytest/pytest> tests/test_graph_storage.py::test_undirected_graph_properties -v

# Run Chinese language variants
<uv run pytest/pytest> tests/test_graph_storage.py::test_basic_graph_operations_chinese -v
<uv run pytest/pytest> tests/test_graph_storage.py::test_special_characters_handling_chinese -v
```

### 4. Storage Backend Testing

```bash
# NetworkX Storage (default, no additional setup required)
<uv run pytest/pytest> tests/test_graph_storage.py -v

# KuzuDB Storage
LIGHTRAG_GRAPH_STORAGE=KuzuDBStorage <uv run pytest/pytest> tests/test_graph_storage.py -v

# KuzuDB specific tests
<uv run pytest/pytest> tests/test_kuzu_impl.py -v
<uv run pytest/pytest> tests/test_kuzu_integration.py -v

# Neo4j Storage (requires Neo4j instance)
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage <uv run pytest/pytest> tests/test_graph_storage.py -v

# MongoDB Storage (requires MongoDB instance)
LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage <uv run pytest/pytest> tests/test_graph_storage.py -v
```

### 5. Language Testing

```bash
# Test with English translations
TEST_LANGUAGE=english <uv run pytest/pytest> tests/test_graph_storage.py -v

# Test with Chinese translations
TEST_LANGUAGE=chinese <uv run pytest/pytest> tests/test_graph_storage.py -v
```

### 6. Advanced Testing Options

```bash
# Run with detailed output
<uv run pytest/pytest> tests/test_graph_storage.py -v -s

# Run specific test pattern
<uv run pytest/pytest> tests/test_graph_storage.py -k "basic" -v

# Run with coverage
<uv run pytest/pytest> tests/test_graph_storage.py --cov=lightrag --cov-report=html

# Run with parallel execution
<uv run pytest/pytest> tests/test_graph_storage.py -n auto

# Run with specific markers
<uv run pytest/pytest> tests/test_graph_storage.py -m "asyncio" -v
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

   - Check if storage backend is properly configured
   - For KuzuDB, ensure write permissions in test directory

2. **Translation Errors**

   ```bash
   # Ensure TEST_LANGUAGE is set correctly
   export TEST_LANGUAGE=english  # or chinese
   ```

3. **Missing Dependencies**

   ```bash
   # Install missing packages
   <uv sync/pip install> -r requirements.txt
   ```

4. **Permission Errors (KuzuDB)**

   ```bash
   # Ensure write permissions for temporary directories
   chmod 755 /tmp
   ```

5. **Interactive Mode for Debugging**

   ```bash
   # Use interactive CLI for detailed debugging
   <python/uv run> tests/test_cli.py --language english --storage NetworkXStorage --tests basic
   # This provides more detailed output for debugging specific issues
   ```

6. **CLI Mode Advantages**
   - The CLI provides better error reporting and progress tracking
   - Especially useful for beginners or when debugging storage issues

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

1. **Follow the modular structure** - Add new test files to `tests/graph/tests/`
2. **Add translations** - Create corresponding translation files in `translations/`
3. **Update pytest integration** - Add new tests to `test_graph_storage.py`
4. **Document changes** - Update this README with new test categories

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review test logs for specific error messages
3. Ensure all dependencies are properly installed
4. Verify environment configuration

---

_This test suite provides comprehensive coverage for LightRAG graph storage implementations with support for multiple backends and languages._
