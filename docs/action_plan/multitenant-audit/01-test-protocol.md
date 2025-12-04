# Multi-Tenant Test Protocol

**Date:** November 29, 2025

---

## Test Environment Setup

### Prerequisites

1. **macOS System Requirements**
   - Python 3.12+ installed
   - Node.js 18+ installed
   - Docker Desktop installed and running

2. **Repository Setup**
   ```bash
   cd /Users/raphaelmansuy/Github/03-working/LightRAG
   git checkout feat/multi-tenannt
   ```

### Step 1: Start Database Services (Docker)

Create a docker-compose file specifically for testing databases:

```yaml
# docker-compose.test-db.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: lightrag-test-postgres
    environment:
      POSTGRES_USER: lightrag
      POSTGRES_PASSWORD: lightrag123
      POSTGRES_DB: lightrag_test
    ports:
      - "5432:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lightrag -d lightrag_test"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: lightrag-test-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_test_data:
```

**Start databases:**
```bash
docker-compose -f docker-compose.test-db.yml up -d
```

### Step 2: Configure Environment

Create `.env.test` file:
```bash
# Database configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=lightrag123
POSTGRES_DATABASE=lightrag_test

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Multi-tenant configuration
MULTITENANT_MODE=demo
INIT_DEMO_TENANTS=true

# LightRAG configuration
LIGHTRAG_WORKING_DIR=./data/rag_storage
LIGHTRAG_INPUT_DIR=./data/inputs

# LLM Configuration (for testing)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key-here

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# API Configuration
LIGHTRAG_API_KEY=test-api-key-123
AUTH_USER=admin
AUTH_PASS=admin123
```

### Step 3: Initialize Python Environment

```bash
# Create virtual environment
python -m venv .venv-test
source .venv-test/bin/activate

# Install dependencies
pip install -e ".[dev,postgres,redis]"
```

### Step 4: Start REST API Server (Local)

```bash
# Set environment variables
export $(cat .env.test | xargs)

# Start the API server
python -m lightrag.api.lightrag_server --host 0.0.0.0 --port 9621
```

### Step 5: Start Web UI (Local)

```bash
cd lightrag_webui

# Install dependencies
npm install

# Create .env.local
cat > .env.local << EOF
VITE_API_URL=http://localhost:9621
VITE_ENABLE_AUTH=true
EOF

# Start development server
npm run dev
```

---

## Test Protocol

### Phase 1: Tenant Isolation Tests

#### Test 1.1: Create Two Tenants via API
```bash
# Create Tenant A
curl -X POST http://localhost:9621/api/v1/tenants \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-123" \
  -d '{"name": "Tenant Alpha", "description": "Test tenant A"}'

# Create Tenant B
curl -X POST http://localhost:9621/api/v1/tenants \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-123" \
  -d '{"name": "Tenant Beta", "description": "Test tenant B"}'
```

**Expected Result:** Two distinct tenant IDs returned

#### Test 1.2: Create KBs per Tenant
```bash
# Create KB for Tenant A
curl -X POST http://localhost:9621/api/v1/knowledge-bases \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_a_id>" \
  -d '{"name": "KB Alpha-1", "description": "Knowledge base for Tenant A"}'

# Create KB for Tenant B
curl -X POST http://localhost:9621/api/v1/knowledge-bases \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_b_id>" \
  -d '{"name": "KB Beta-1", "description": "Knowledge base for Tenant B"}'
```

**Expected Result:** Each KB associated with correct tenant

#### Test 1.3: Verify KB Isolation
```bash
# List KBs for Tenant A (should only see KB Alpha-1)
curl http://localhost:9621/api/v1/knowledge-bases \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_a_id>"

# List KBs for Tenant B (should only see KB Beta-1)
curl http://localhost:9621/api/v1/knowledge-bases \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_b_id>"
```

**Expected Result:** Each tenant only sees their own KBs

### Phase 2: Document Isolation Tests

#### Test 2.1: Upload Document to Tenant A
```bash
# Upload document
curl -X POST http://localhost:9621/documents/upload \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_a_id>" \
  -H "X-KB-ID: <kb_a_id>" \
  -F "file=@test_doc_a.txt"
```

#### Test 2.2: Upload Document to Tenant B
```bash
curl -X POST http://localhost:9621/documents/upload \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_b_id>" \
  -H "X-KB-ID: <kb_b_id>" \
  -F "file=@test_doc_b.txt"
```

#### Test 2.3: Verify Document Isolation
```bash
# List docs for Tenant A (should only see doc A)
curl http://localhost:9621/documents \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_a_id>" \
  -H "X-KB-ID: <kb_a_id>"

# List docs for Tenant B (should only see doc B)
curl http://localhost:9621/documents \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_b_id>" \
  -H "X-KB-ID: <kb_b_id>"
```

### Phase 3: Query Isolation Tests

#### Test 3.1: Query in Tenant A Context
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_a_id>" \
  -H "X-KB-ID: <kb_a_id>" \
  -d '{"query": "What is in the document?", "mode": "hybrid"}'
```

**Expected Result:** Response based only on Tenant A's documents

#### Test 3.2: Query in Tenant B Context
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-123" \
  -H "X-Tenant-ID: <tenant_b_id>" \
  -H "X-KB-ID: <kb_b_id>" \
  -d '{"query": "What is in the document?", "mode": "hybrid"}'
```

**Expected Result:** Response based only on Tenant B's documents

### Phase 4: Web UI Tests

#### Test 4.1: Tenant Selection
1. Open Web UI at http://localhost:5173
2. Login with admin/admin123
3. Verify tenant selector shows available tenants
4. Select Tenant A
5. Verify X-Tenant-ID header is sent with requests

#### Test 4.2: KB Selection
1. After selecting Tenant A, verify KB list shows only Tenant A's KBs
2. Select a KB
3. Verify X-KB-ID header is sent with requests

#### Test 4.3: Document Visibility
1. Select Tenant A / KB Alpha-1
2. Go to Documents tab
3. Verify only documents for this KB are shown
4. Switch to Tenant B / KB Beta-1
5. Verify documents list changes to show only Tenant B's documents

#### Test 4.4: Query Scope
1. Select Tenant A / KB Alpha-1
2. Enter a query in the chat
3. Verify response is based on Tenant A's data
4. Switch to Tenant B
5. Enter same query
6. Verify response is different (based on Tenant B's data)

### Phase 5: Storage Layer Verification

#### Test 5.1: Verify PostgreSQL Data Isolation
```sql
-- Connect to database
psql -h localhost -U lightrag -d lightrag_test

-- Check tenant_id and kb_id in document tables
SELECT tenant_id, kb_id, id, content_summary 
FROM lightrag_doc_status 
ORDER BY tenant_id, kb_id;

-- Verify no cross-tenant data
SELECT COUNT(*) FROM lightrag_doc_status 
WHERE tenant_id = 'tenant_a' AND kb_id IN (
  SELECT kb_id FROM lightrag_doc_status WHERE tenant_id = 'tenant_b'
);
-- Expected: 0
```

#### Test 5.2: Verify Redis Namespace Isolation
```bash
redis-cli

# List keys for tenant A
KEYS "tenant_a:*"

# List keys for tenant B  
KEYS "tenant_b:*"

# Verify no overlap
```

---

## Test Execution Checklist

| Phase | Test | Status | Notes |
|-------|------|--------|-------|
| 1.1 | Create tenants | ⬜ | |
| 1.2 | Create KBs | ⬜ | |
| 1.3 | Verify KB isolation | ⬜ | |
| 2.1 | Upload doc to A | ⬜ | |
| 2.2 | Upload doc to B | ⬜ | |
| 2.3 | Verify doc isolation | ⬜ | |
| 3.1 | Query in A context | ⬜ | |
| 3.2 | Query in B context | ⬜ | |
| 4.1 | Tenant selection UI | ⬜ | |
| 4.2 | KB selection UI | ⬜ | |
| 4.3 | Document visibility | ⬜ | |
| 4.4 | Query scope | ⬜ | |
| 5.1 | PostgreSQL isolation | ⬜ | |
| 5.2 | Redis isolation | ⬜ | |

---

## Failure Criteria

A test fails if:
1. Data from one tenant is visible to another tenant
2. Query results include information from wrong tenant/KB
3. Headers are not propagated correctly
4. Database records show incorrect tenant_id/kb_id associations
5. Redis keys do not include proper namespace prefixes
