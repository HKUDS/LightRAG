# GitHub Copilot Setup Steps for LightRAG Integration Testing

This document describes the steps needed to set up and run the LightRAG integration tests locally or in CI/CD.

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git

## Local Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/netbrah/LightRAG.git
cd LightRAG
```

### 2. Set Up Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -e ".[api]"
pip install pytest pytest-asyncio httpx
```

### 4. Start Docker Services

The integration tests require three services:
- **Redis**: For KV and document status storage
- **Neo4j**: For graph storage
- **Milvus**: For vector storage

```bash
cd tests
docker-compose -f docker-compose.integration.yml up -d
```

### 5. Wait for Services to Be Ready

```bash
# Wait for Redis
until docker exec lightrag-test-redis redis-cli ping | grep -q PONG; do sleep 2; done

# Wait for Neo4j (may take up to 2 minutes)
until docker exec lightrag-test-neo4j cypher-shell -u neo4j -p testpassword123 "RETURN 1" 2>/dev/null | grep -q "1"; do sleep 5; done

# Wait for Milvus (may take up to 3 minutes)
until curl -s http://localhost:9091/healthz | grep -q "OK"; do sleep 5; done
```

### 6. Start Mock OpenAI Server

The mock server simulates OpenAI API responses for testing without requiring actual API keys.

```bash
cd tests
python mock_openai_server.py --host 127.0.0.1 --port 8000 &
MOCK_PID=$!

# Wait for it to be ready
until curl -s http://127.0.0.1:8000/health | grep -q "healthy"; do sleep 1; done
```

### 7. Prepare Test Environment

```bash
cd tests
cp .env.integration .env
mkdir -p test_inputs test_rag_storage
```

### 8. Start LightRAG Server

```bash
cd tests
lightrag-server &
LIGHTRAG_PID=$!

# Wait for it to be ready
until curl -s http://localhost:9621/health | grep -q "status"; do sleep 2; done
```

### 9. Run Integration Tests

```bash
cd tests
python integration_test.py
```

### 10. Cleanup

```bash
# Stop servers
kill $LIGHTRAG_PID
kill $MOCK_PID

# Stop Docker services
docker-compose -f docker-compose.integration.yml down -v

# Remove test artifacts
rm -rf test_inputs test_rag_storage .env
```

## Service Configuration Details

### Redis Configuration
- **Port**: 6379
- **Container**: lightrag-test-redis
- **Purpose**: KV storage and document status tracking

### Neo4j Configuration
- **HTTP Port**: 7474
- **Bolt Port**: 7687
- **Container**: lightrag-test-neo4j
- **Credentials**: neo4j/testpassword123
- **Purpose**: Graph knowledge base storage

### Milvus Configuration
- **API Port**: 19530
- **Health Port**: 9091
- **Container**: lightrag-test-milvus
- **Database**: lightrag_test
- **Purpose**: Vector embeddings storage

### Mock OpenAI Server Configuration
- **Port**: 8000
- **Endpoints**:
  - `/v1/chat/completions` - Mock LLM responses
  - `/v1/embeddings` - Mock embedding generation
  - `/health` - Health check

### LightRAG Server Configuration
- **Port**: 9621
- **Configuration**: tests/.env.integration
- **Storage Backends**:
  - KV: RedisKVStorage
  - Doc Status: RedisDocStatusStorage
  - Vector: MilvusVectorDBStorage
  - Graph: Neo4JStorage

## CI/CD Integration

The integration tests are automatically run on every commit via GitHub Actions. See `.github/workflows/integration-test.yml` for the workflow configuration.

### Workflow Triggers
- Push to branches: main, dev, copilot/**
- Pull requests to: main, dev
- Manual workflow dispatch

### Workflow Steps
1. Checkout code
2. Set up Python environment
3. Install dependencies
4. Start Docker services (Redis, Neo4j, Milvus)
5. Wait for all services to be healthy
6. Start Mock OpenAI server
7. Configure test environment
8. Start LightRAG server
9. Run integration tests
10. Collect logs on failure
11. Cleanup all resources

## Test Coverage

The integration tests validate:

1. **Health Check**: Server availability and basic functionality
2. **Document Indexing**:
   - File upload (C++ source files)
   - Text insertion
   - Multiple file formats
3. **Query Operations**:
   - Naive mode
   - Local mode
   - Global mode
   - Hybrid mode
4. **Structured Data Retrieval**:
   - Entity extraction
   - Relationship mapping
   - Chunk retrieval
5. **Graph Operations**:
   - Graph data retrieval
   - Node and edge counting

## Sample Test Repository

The tests use a sample C++ repository located at `tests/sample_cpp_repo/`:
- **Files**: calculator.h, calculator.cpp, utils.h, utils.cpp, main.cpp
- **Purpose**: Demonstrates code indexing and querying capabilities
- **Content**: Simple calculator implementation with documentation

## Troubleshooting

### Services Not Starting
- Check Docker is running: `docker ps`
- Check port availability: `lsof -i :6379,7687,19530,8000,9621`
- Review Docker logs: `docker-compose -f tests/docker-compose.integration.yml logs`

### Mock Server Issues
- Verify port 8000 is available
- Check mock server logs
- Test health endpoint: `curl http://127.0.0.1:8000/health`

### LightRAG Server Issues
- Check environment file: `tests/.env`
- Review server logs: `cat tests/lightrag.log*`
- Verify storage connections

### Test Failures
- Ensure all services are healthy before running tests
- Check network connectivity between services
- Review test output for specific error messages

## Environment Variables

Key environment variables used in integration tests:

- `LIGHTRAG_API_URL`: LightRAG server URL (default: http://localhost:9621)
- `LLM_BINDING_HOST`: Mock OpenAI server URL (default: http://127.0.0.1:8000)
- `EMBEDDING_BINDING_HOST`: Mock embedding server URL (default: http://127.0.0.1:8000)
- `REDIS_URI`: Redis connection string
- `NEO4J_URI`: Neo4j connection string
- `MILVUS_URI`: Milvus connection string

All configurations are defined in `tests/.env.integration`.
