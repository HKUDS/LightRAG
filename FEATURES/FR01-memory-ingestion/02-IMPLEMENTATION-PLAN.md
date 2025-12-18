# FR01: Memory API Ingestion - Implementation Plan

## Implementation Phases

### Phase 1: Core Foundation (Week 1)

**Goal**: Build the minimum viable connector that can pull memories and insert into LightRAG

#### Tasks:

**1.1 Project Setup**
- [ ] Create `lightrag/connectors/` directory structure
- [ ] Set up `memory_connector` package for CLI
- [ ] Create `pyproject.toml` or update existing with new dependencies
- [ ] Add dependencies:
  ```toml
  [tool.poetry.dependencies]
  apscheduler = "^3.10.4"
  httpx = "^0.27.0"
  pydantic = "^2.5.0"
  pyyaml = "^6.0.1"
  python-dateutil = "^2.8.2"
  ```

**1.2 Data Models**
- [ ] Create `lightrag/connectors/models.py`
- [ ] Implement Memory API models (Memory, MemoryList)
- [ ] Implement connector config models
- [ ] Implement ingestion report models
- [ ] Add validation logic

**1.3 Memory API Client**
- [ ] Create `lightrag/connectors/memory_api_client.py`
- [ ] Implement `MemoryAPIClient` class
- [ ] Add authentication (API key header)
- [ ] Implement `get_memories()` method
- [ ] Add retry logic with exponential backoff
- [ ] Add timeout handling
- [ ] Write unit tests

**1.4 Data Transformer**
- [ ] Create `lightrag/connectors/memory_transformer.py`
- [ ] Implement `TransformationStrategy` abstract base
- [ ] Implement `StandardTransformationStrategy`
- [ ] Implement `MemoryTransformer` class
- [ ] Add tests for transformation logic

**1.5 LightRAG Client**
- [ ] Create `lightrag/connectors/lightrag_client.py`
- [ ] Implement `LightRAGClient` abstract base
- [ ] Implement `LightRAGAPIClient` (HTTP mode)
- [ ] Implement `LightRAGDirectClient` (library mode)
- [ ] Write integration tests

**1.6 Basic CLI**
- [ ] Create `memory_connector/__main__.py`
- [ ] Implement `sync` command for one-time sync
- [ ] Add argument parsing (argparse or typer)
- [ ] Basic logging setup

**Testing Phase 1**:
- [ ] Manual test: Pull memories from test API
- [ ] Manual test: Transform and insert into test LightRAG instance
- [ ] Verify knowledge graph contains entities from memory transcripts

### Phase 2: State Management & Scheduling (Week 2)

**Goal**: Add persistence, deduplication, and automated scheduling

#### Tasks:

**2.1 State Manager**
- [ ] Create `lightrag/connectors/state_manager.py`
- [ ] Implement JSON backend for state storage
- [ ] Implement `SyncState` model
- [ ] Implement state CRUD operations
- [ ] Add atomic write operations (temp file + rename)
- [ ] Implement `get_unprocessed_ids()` filtering
- [ ] Add state history tracking
- [ ] Write unit tests

**2.2 Configuration Manager**
- [ ] Create `lightrag/connectors/config_manager.py`
- [ ] Implement YAML config loading
- [ ] Add environment variable substitution
- [ ] Implement config validation
- [ ] Support multiple connector definitions
- [ ] Add config reload mechanism
- [ ] Write validation tests

**2.3 Scheduler Service**
- [ ] Create `lightrag/connectors/scheduler_service.py`
- [ ] Integrate APScheduler
- [ ] Implement interval trigger support
- [ ] Implement cron trigger support
- [ ] Add job control methods (pause, resume, trigger)
- [ ] Implement graceful shutdown
- [ ] Add scheduler persistence (SQLite jobstore)
- [ ] Write scheduler tests

**2.4 Ingestion Orchestrator**
- [ ] Create `lightrag/connectors/ingestion_orchestrator.py`
- [ ] Implement main ingestion pipeline
- [ ] Add batch processing logic
- [ ] Implement error handling and retry
- [ ] Generate ingestion reports
- [ ] Add progress callbacks
- [ ] Implement cancellation support
- [ ] Write integration tests

**2.5 Enhanced CLI**
- [ ] Add `serve` command to run scheduler
- [ ] Add `status` command to check sync state
- [ ] Add `list` command to show connectors
- [ ] Add `trigger` command for manual runs
- [ ] Improve logging (structured logging)

**2.6 Configuration File**
- [ ] Create example `config.yaml`
- [ ] Add inline documentation
- [ ] Create config validation script
- [ ] Write config documentation

**Testing Phase 2**:
- [ ] Test incremental sync (no duplicate processing)
- [ ] Test scheduler (interval and cron)
- [ ] Test state persistence across restarts
- [ ] Test concurrent processing
- [ ] Load test with 1000+ memory items

### Phase 3: Management API & Monitoring (Week 3)

**Goal**: Add REST API for connector management and monitoring

#### Tasks:

**3.1 Management API**
- [ ] Create `lightrag/connectors/connector_api.py`
- [ ] Implement FastAPI application
- [ ] Add authentication (API key)
- [ ] Implement connector CRUD endpoints:
  - `GET /connectors` - List all connectors
  - `POST /connectors` - Create new connector
  - `GET /connectors/{id}` - Get connector details
  - `PUT /connectors/{id}` - Update connector
  - `DELETE /connectors/{id}` - Delete connector
- [ ] Implement control endpoints:
  - `POST /connectors/{id}/trigger` - Manual trigger
  - `POST /connectors/{id}/pause` - Pause scheduler
  - `POST /connectors/{id}/resume` - Resume scheduler
- [ ] Implement status endpoints:
  - `GET /connectors/{id}/status` - Current status
  - `GET /connectors/{id}/history` - Sync history
  - `GET /health` - Health check

**3.2 API Models**
- [ ] Add request/response models
- [ ] Add OpenAPI documentation
- [ ] Add example requests/responses

**3.3 API Integration with Scheduler**
- [ ] Connect API to SchedulerService
- [ ] Connect API to StateManager
- [ ] Add real-time status updates
- [ ] Implement WebSocket for progress streaming (optional)

**3.4 Enhanced CLI**
- [ ] Update `serve` command to start both scheduler and API
- [ ] Add `--api-only` flag for API-only mode
- [ ] Add `--scheduler-only` flag for scheduler-only mode

**3.5 Monitoring & Logging**
- [ ] Add structured logging (JSON format)
- [ ] Add metrics collection (counters, timers)
- [ ] Optional: Add Prometheus exporter
- [ ] Add health check logic

**Testing Phase 3**:
- [ ] API endpoint tests (all CRUD operations)
- [ ] Authentication tests
- [ ] Integration test: Create connector via API, verify it runs
- [ ] Load test API with concurrent requests

### Phase 4: Advanced Features (Week 4)

**Goal**: Add production-ready features and enhancements

#### Tasks:

**4.1 Rich Transformation Strategy**
- [ ] Implement `RichTransformationStrategy`
- [ ] Add geocoding support (optional, using external service)
- [ ] Add datetime formatting improvements
- [ ] Add tag extraction from transcripts
- [ ] Make strategy configurable per connector

**4.2 SQLite State Backend**
- [ ] Implement SQLite backend for StateManager
- [ ] Add database migrations
- [ ] Add connection pooling
- [ ] Performance comparison with JSON backend

**4.3 Error Handling Enhancements**
- [ ] Implement dead letter queue for failed items
- [ ] Add retry with exponential backoff per item
- [ ] Add alerting hooks (email, webhook, Slack)
- [ ] Add error reporting dashboard

**4.4 Security Enhancements**
- [ ] Add secrets encryption at rest
- [ ] Implement API key rotation
- [ ] Add audit logging
- [ ] Security review and hardening

**4.5 Documentation**
- [ ] User guide (setup, configuration, usage)
- [ ] API reference (OpenAPI/Swagger)
- [ ] Architecture documentation
- [ ] Troubleshooting guide
- [ ] Performance tuning guide

**4.6 Deployment Artifacts**
- [ ] Create Dockerfile
- [ ] Create docker-compose.yaml (connector + LightRAG)
- [ ] Create systemd service file
- [ ] Create Kubernetes manifests (optional)
- [ ] Create Helm chart (optional)

**Testing Phase 4**:
- [ ] End-to-end production scenario test
- [ ] Security audit
- [ ] Performance benchmarks
- [ ] Failover and recovery tests
- [ ] Documentation review

## Implementation Details

### Priority 1: Minimal Working Implementation

For fastest time-to-value, implement in this order:

1. **Memory API Client** → Can fetch data
2. **Standard Transformer** → Can convert data
3. **LightRAG Direct Client** → Can insert data
4. **Basic CLI sync command** → Can run manually

**Deliverable**: Manual sync working end-to-end

### Priority 2: Automation

5. **JSON State Manager** → Tracks processed items
6. **Config Manager** → Loads YAML config
7. **Scheduler Service** → Automates periodic sync
8. **Orchestrator** → Coordinates pipeline

**Deliverable**: Automated hourly sync running

### Priority 3: Management

9. **Management API** → Control via REST API
10. **Enhanced logging** → Production monitoring

**Deliverable**: Production-ready service

## Testing Strategy

### Unit Tests
- Every class has unit tests
- Mock external dependencies (APIs, file I/O)
- Target: 80%+ code coverage

### Integration Tests
- Test Memory API client with mock server
- Test LightRAG integration with test instance
- Test complete pipeline with mocked components

### End-to-End Tests
- Real Memory API (test account)
- Real LightRAG instance (test workspace)
- Full sync cycle
- Verify knowledge graph contents

### Performance Tests
- Benchmark: 100 memories in <60 seconds
- Benchmark: 1000 memories in <10 minutes
- Memory usage: <500MB for 10,000 items
- Scheduler overhead: <1% CPU when idle

## File Creation Checklist

### Core Implementation Files

```
✓ Created in architecture planning:
- FEATURES/FR01-memory-ingestion/00-OVERVIEW.md
- FEATURES/FR01-memory-ingestion/01-ARCHITECTURE.md
- FEATURES/FR01-memory-ingestion/02-IMPLEMENTATION-PLAN.md

□ To be created in Phase 1:
- lightrag/connectors/__init__.py
- lightrag/connectors/models.py
- lightrag/connectors/memory_api_client.py
- lightrag/connectors/memory_transformer.py
- lightrag/connectors/lightrag_client.py
- memory_connector/__init__.py
- memory_connector/__main__.py
- tests/connectors/test_memory_api_client.py
- tests/connectors/test_transformer.py
- tests/connectors/test_lightrag_client.py

□ To be created in Phase 2:
- lightrag/connectors/state_manager.py
- lightrag/connectors/config_manager.py
- lightrag/connectors/scheduler_service.py
- lightrag/connectors/ingestion_orchestrator.py
- config.example.yaml
- tests/connectors/test_state_manager.py
- tests/connectors/test_orchestrator.py

□ To be created in Phase 3:
- lightrag/connectors/connector_api.py
- tests/connectors/test_api.py

□ To be created in Phase 4:
- Dockerfile
- docker-compose.yaml
- docs/memory-connector-guide.md
- docs/api-reference.md
```

## Dependencies

### Python Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"

# Existing LightRAG dependencies
# ... (all current dependencies)

# New dependencies for Memory Connector
apscheduler = "^3.10.4"       # Job scheduling
httpx = "^0.27.0"             # Async HTTP client
pydantic = "^2.5.0"           # Data validation (already in LightRAG)
pyyaml = "^6.0.1"             # YAML config parsing
python-dateutil = "^2.8.2"    # Date parsing/formatting

# Optional dependencies
sqlalchemy = { version = "^2.0.0", optional = true }  # SQLite backend
alembic = { version = "^1.13.0", optional = true }    # Migrations

[tool.poetry.extras]
sqlite = ["sqlalchemy", "alembic"]
```

### System Dependencies

- Python 3.11+
- LightRAG (latest version)
- Access to Memory API (API key required)
- Network connectivity

## Configuration Example

Create `config.yaml`:

```yaml
# Memory Connector Configuration

lightrag:
  mode: "api"
  api:
    url: "http://localhost:9621"
    api_key: "your-lightrag-api-key"
    workspace: "memories"

memory_api:
  url: "http://127.0.0.1:8080"
  api_key: "your-memory-api-key"
  timeout: 30

connectors:
  - id: "personal-memories"
    enabled: true
    context_id: "CTX123"
    schedule:
      type: "interval"
      interval_hours: 1
    ingestion:
      query_range: "week"
      query_limit: 100
      batch_size: 10
    transformation:
      strategy: "standard"
    retry:
      max_attempts: 3

state:
  backend: "json"
  path: "./memory_sync_state.json"

api:
  host: "0.0.0.0"
  port: 9622
  enable_auth: true
  api_key: "your-connector-api-key"

logging:
  level: "INFO"
  file: "./memory_connector.log"
```

## Deployment Steps

### Development Deployment

```bash
# 1. Install dependencies
poetry install --extras sqlite

# 2. Create config file
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# 3. Test one-time sync
python -m memory_connector sync \
  --config config.yaml \
  --connector-id personal-memories

# 4. Start scheduler
python -m memory_connector serve \
  --config config.yaml
```

### Production Deployment (Docker)

```bash
# 1. Build image
docker build -t memory-connector:latest .

# 2. Run with docker-compose
docker-compose up -d

# 3. Check logs
docker-compose logs -f memory-connector

# 4. Check status
curl http://localhost:9622/health
```

### Production Deployment (systemd)

```bash
# 1. Install package
pip install -e .

# 2. Create systemd service
sudo cp memory-connector.service /etc/systemd/system/
sudo systemctl daemon-reload

# 3. Start service
sudo systemctl start memory-connector
sudo systemctl enable memory-connector

# 4. Check status
sudo systemctl status memory-connector
sudo journalctl -u memory-connector -f
```

## Metrics & Success Criteria

### Functionality Metrics
- ✅ Successfully connects to Memory API
- ✅ Successfully inserts documents into LightRAG
- ✅ No duplicate processing (idempotency)
- ✅ Handles API errors gracefully
- ✅ State persists across restarts

### Performance Metrics
- ⏱️ Process 100 memories in <60 seconds
- ⏱️ Memory usage <500MB for 10K items
- ⏱️ Scheduler latency <5 seconds
- ⏱️ API response time <200ms

### Reliability Metrics
- 🎯 99.9% uptime for scheduler
- 🎯 <0.1% duplicate processing rate
- 🎯 100% state recovery after crash
- 🎯 Auto-retry on transient failures

### Usability Metrics
- 📖 Complete documentation
- 📖 Example configuration
- 📖 Troubleshooting guide
- 📖 API reference (OpenAPI)

## Risk Mitigation

### Risk: Memory API rate limiting
**Mitigation**: Add configurable rate limiting, backoff, and pagination support

### Risk: LightRAG processing delays
**Mitigation**: Queue-based ingestion with status tracking; process asynchronously

### Risk: State corruption
**Mitigation**: Atomic writes, backup strategy, state validation on load

### Risk: Network failures
**Mitigation**: Retry logic, timeout handling, circuit breaker pattern

### Risk: Large memory volumes
**Mitigation**: Batch processing, incremental sync, configurable limits

## Next Steps After Implementation

1. **Beta Testing**
   - Deploy to test environment
   - Run for 1 week with real data
   - Monitor for issues

2. **Production Rollout**
   - Deploy to production
   - Enable for small subset of contexts
   - Gradual rollout

3. **Phase 2: Memory Manager** (See `05-FUTURE-ENHANCEMENTS.md`)
   - Bidirectional sync
   - Export/backup features
   - Collaborative sharing
