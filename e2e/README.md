# E2E Multi-Tenant Isolation Tests

This directory contains end-to-end tests to verify the isolation of data and processing pipelines in the multi-tenant LightRAG implementation.

## Quick Start

```bash
# Run all tests with file backend (default)
./e2e/run_tests.sh

# Interactive mode - guided test selection
./e2e/run_tests.sh -i

# Show help
./e2e/run_tests.sh --help
```

## Test Runner Features

The enhanced test runner (`run_tests.sh`) provides:

- 🎮 **Interactive Mode**: Menu-driven selection of backends and tests
- 💾 **Multiple Backends**: Support for `file`, `postgres`, or `all` backends
- 🧪 **Test Selection**: Run specific tests or all tests
- 🔍 **Dry Run**: Preview configuration without executing
- 📊 **Verbose/Quiet Modes**: Control output verbosity
- 🎨 **Colored Output**: Easy-to-read results with emojis

## Available Tests

| Test | File | Description |
|------|------|-------------|
| `isolation` | `test_multitenant_isolation.py` | Tests data isolation between tenants |
| `deletion` | `test_deletion.py` | Tests document deletion and cleanup |
| `mixed` | `test_mixed_operations.py` | Tests interleaved tenant operations |

## Available Backends

| Backend | Description |
|---------|-------------|
| `file` | File-based storage (JSON, NetworkX, NanoVectorDB) - Default |
| `postgres` | PostgreSQL with pgvector for production storage |
| `all` | Run tests on both backends |

## Usage Examples

### Basic Usage

```bash
# Run all tests with file backend (default)
./e2e/run_tests.sh

# Run all tests with PostgreSQL backend
./e2e/run_tests.sh -b postgres

# Run tests on all backends
./e2e/run_tests.sh -b all

# Run with OpenAI models (requires OPENAI_API_KEY)
./e2e/run_tests.sh --openai

# Reset database/storage before running tests
./e2e/run_tests.sh --reset-db
```

### Test Selection

```bash
# Run only isolation test
./e2e/run_tests.sh -t isolation

# Run isolation and deletion tests
./e2e/run_tests.sh -t isolation,deletion

# Run specific test with postgres backend
./e2e/run_tests.sh -b postgres -t isolation
```

### Interactive Mode

```bash
# Launch interactive menu
./e2e/run_tests.sh -i
```

The interactive mode guides you through:
1. Backend selection
2. Test selection
3. Advanced options (verbose, keep server)
4. LLM model configuration

### Preview Configuration

```bash
# Dry run to see configuration without executing
./e2e/run_tests.sh --dry-run -b postgres -t isolation,deletion
```

### Custom Model Configuration

```bash
# Use OpenAI models (gpt-4o-mini + text-embedding-3-small)
./e2e/run_tests.sh --openai

# Use custom LLM model (Ollama)
./e2e/run_tests.sh -m llama3.1:8b

# Use custom embedding model (Ollama)
./e2e/run_tests.sh -e nomic-embed-text:latest -d 768

# Combined: OpenAI with reset and postgres backend
./e2e/run_tests.sh --openai --reset-db -b postgres
```

### Server Management

```bash
# Keep server running after tests (for debugging)
./e2e/run_tests.sh --keep-server

# Skip server management (use existing server)
./e2e/run_tests.sh --skip-server

# Use custom port
./e2e/run_tests.sh -p 8080
```

### Output Control

```bash
# Verbose output with debug info
./e2e/run_tests.sh -v

# Quiet mode (minimal output)
./e2e/run_tests.sh -q

# Show live server logs during test execution
./e2e/run_tests.sh --logs

# Full example with logs, OpenAI, reset, and postgres
./e2e/run_tests.sh --openai --reset-db -b postgres --logs
```

### OpenAI Configuration

```bash
# Use OpenAI models (requires OPENAI_API_KEY environment variable)
export OPENAI_API_KEY="sk-..."
./e2e/run_tests.sh --openai

# The --openai flag sets:
# - LLM: gpt-5-nano (OpenAI's fast model)
# - Embedding: text-embedding-3-small (1536 dimensions)
```

### Database Reset

```bash
# Reset storage before running tests (clean slate)
./e2e/run_tests.sh --reset-db

# For file backend: clears rag_storage/ directory
# For postgres: Docker volume is removed and recreated

# Combined with OpenAI for a fresh run
./e2e/run_tests.sh --openai --reset-db -b postgres
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --backend` | Storage backend (file, postgres, all) | `file` |
| `-t, --tests` | Comma-separated test list | all |
| `-m, --llm-model` | LLM model name | `gpt-oss:20b` |
| `--llm-binding` | LLM binding type | `ollama` |
| `-e, --embedding-model` | Embedding model | `bge-m3:latest` |
| `--embedding-binding` | Embedding binding type | `ollama` |
| `-d, --dim` | Embedding dimension | `1024` |
| `-p, --port` | Server port | `9621` |
| `-i, --interactive` | Interactive mode | - |
| `-v, --verbose` | Verbose output | - |
| `-q, --quiet` | Quiet mode | - |
| `--dry-run` | Preview without executing | - |
| `--skip-server` | Don't manage server | - |
| `--keep-server` | Keep server running | - |
| `-l, --list` | List tests and backends | - |
| `--openai` | Use OpenAI models (gpt-5-nano + text-embedding-3-small) | - |
| `--reset-db` | Reset database/storage before tests | - |
| `--skip-docker` | Don't manage Docker containers | - |
| `--keep-docker` | Keep Docker containers running | - |
| `--logs` | Show live server logs during tests | - |
| `-h, --help` | Show help | - |
| `--version` | Show version | - |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required for --openai) | - |
| `LIGHTRAG_API_URL` | API URL | `http://localhost:9621` |
| `AUTH_USER` | Admin username | `admin` |
| `AUTH_PASS` | Admin password | `admin123` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_USER` | PostgreSQL user | `lightrag` |
| `POSTGRES_PASSWORD` | PostgreSQL password | - |
| `POSTGRES_DATABASE` | PostgreSQL database | `lightrag_multitenant` |

## Test Details

### Multi-Tenant Isolation Test (`isolation`)

Tests data isolation between tenants:
1. Creates two distinct tenants (Tenant A and Tenant B)
2. Creates a Knowledge Base (KB) for each tenant
3. Ingests a unique "secret" document into each tenant's KB
4. Waits for indexing to complete
5. Verifies that Tenant A can retrieve its secret but **not** Tenant B's secret
6. Verifies that Tenant B can retrieve its secret but **not** Tenant A's secret

### Document Deletion Test (`deletion`)

Tests document deletion functionality:
1. Creates a tenant and knowledge base
2. Ingests two documents
3. Verifies both documents are queryable
4. Deletes one document
5. Verifies deleted document is no longer retrievable
6. Verifies the other document still exists

### Mixed Operations Test (`mixed`)

Tests interleaved operations across tenants:
1. Creates two tenants with separate KBs
2. Performs interleaved ingestion operations
3. Verifies cross-tenant isolation during concurrent operations
4. Tests deletion in one tenant doesn't affect the other
5. Verifies data integrity throughout

## Prerequisites

- Python 3.10+
- `requests` library installed (`pip install requests`)
- LightRAG installed in the environment
- **Ollama** running locally (or configured via environment variables)
- For PostgreSQL tests: PostgreSQL with pgvector extension

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All tests passed |
| `1` | One or more tests failed |
| `2` | Configuration or setup error |

## Troubleshooting

### Server Issues
- Check `server.log` in the project root if the server fails to start
- Use `--verbose` flag to see detailed server startup logs
- Try `--keep-server` to inspect server state after tests

### Timeout Issues
- If indexing takes too long, check if LLM/Embedding service is responsive
- Increase timeout by modifying `SERVER_TIMEOUT` in the script

### PostgreSQL Issues
- Ensure PostgreSQL is running with pgvector extension
- Verify connection settings in environment variables
- Check that the database exists and user has proper permissions

## Legacy Script

The original `run_isolation_test.sh` is still available for backward compatibility but `run_tests.sh` is recommended for new usage.
