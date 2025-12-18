# FR01: Memory API Ingestion - Implementation Plan (Go)

## Implementation Phases

### Phase 1: Core Foundation (Week 1)

**Goal**: Build the minimum viable connector that can pull memories and insert into LightRAG

#### Tasks:

**1.1 Project Setup**
- [ ] Create `EXTENSIONS/memory-ingestion/` directory structure
- [ ] Initialize Go module: `go mod init github.com/your-org/memory-connector`
- [ ] Set up project folders (cmd, pkg, internal, configs, deployments)
- [ ] Create Makefile with build, test, clean targets
- [ ] Set up `.gitignore` for Go projects
- [ ] Initialize Go dependencies:
  ```bash
  go get github.com/gin-gonic/gin
  go get github.com/robfig/cron/v3
  go get github.com/spf13/cobra
  go get github.com/spf13/viper
  go get go.uber.org/zap
  go get gopkg.in/yaml.v3
  ```

**1.2 Data Models**
- [ ] Create `pkg/models/memory.go` - Memory API models
- [ ] Create `pkg/models/connector.go` - Connector configuration models
- [ ] Create `pkg/models/report.go` - Ingestion report models
- [ ] Add JSON/YAML struct tags
- [ ] Add validation tags

**1.3 Memory API Client**
- [ ] Create `pkg/client/memory_client.go`
- [ ] Implement `MemoryClient` struct
- [ ] Add authentication (API key header)
- [ ] Implement `GetMemories()` method with context
- [ ] Add retry logic with exponential backoff
- [ ] Add timeout handling
- [ ] Write unit tests with httptest

**1.4 Data Transformer**
- [ ] Create `pkg/transformer/transformer.go`
- [ ] Implement `Strategy` interface
- [ ] Implement `StandardStrategy` struct
- [ ] Implement `Transform()` method
- [ ] Add date/time formatting
- [ ] Add tests for transformation logic

**1.5 LightRAG Client**
- [ ] Create `pkg/client/lightrag_client.go`
- [ ] Implement `LightRAGClient` interface
- [ ] Implement HTTP client for LightRAG API
- [ ] Add `InsertDocument()` method
- [ ] Write integration tests

**1.6 Basic CLI**
- [ ] Create `cmd/memory-connector/main.go`
- [ ] Set up Cobra for CLI commands
- [ ] Implement `sync` command for one-time sync
- [ ] Add flag parsing (--config, --connector-id, etc.)
- [ ] Basic logging setup with zap

**Testing Phase 1**:
- [ ] Manual test: Pull memories from test API
- [ ] Manual test: Transform and insert into test LightRAG instance
- [ ] Verify knowledge graph contains entities from memory transcripts

### Phase 2: State Management & Scheduling (Week 2)

**Goal**: Add persistence, deduplication, and automated scheduling

#### Tasks:

**2.1 State Manager**
- [ ] Create `pkg/state/state.go` - State manager interface
- [ ] Create `pkg/state/json_store.go` - JSON backend
- [ ] Implement `SyncState` struct
- [ ] Implement state CRUD operations
- [ ] Add atomic file writes (write to temp, then rename)
- [ ] Implement `GetUnprocessedIDs()` filtering
- [ ] Add state history tracking
- [ ] Write unit tests with temp files

**2.2 Configuration Manager**
- [ ] Create `pkg/config/config.go`
- [ ] Integrate Viper for config loading
- [ ] Implement YAML config parsing
- [ ] Add environment variable substitution (os.ExpandEnv)
- [ ] Implement config validation
- [ ] Support multiple connector definitions
- [ ] Add config reload via SIGHUP signal
- [ ] Write validation tests

**2.3 Scheduler Service**
- [ ] Create `pkg/scheduler/scheduler.go`
- [ ] Integrate robfig/cron/v3
- [ ] Implement interval trigger support (@every 1h)
- [ ] Implement cron trigger support (0 */1 * * *)
- [ ] Add job control methods (AddJob, RemoveJob, TriggerNow)
- [ ] Implement graceful shutdown with context
- [ ] Add job persistence (resume after restart)
- [ ] Write scheduler tests

**2.4 Ingestion Orchestrator**
- [ ] Create `pkg/orchestrator/orchestrator.go`
- [ ] Implement main ingestion pipeline
- [ ] Add batch processing logic with goroutines
- [ ] Implement error handling and retry with backoff
- [ ] Generate ingestion reports
- [ ] Add progress callbacks
- [ ] Implement cancellation via context
- [ ] Write integration tests

**2.5 Enhanced CLI**
- [ ] Add `serve` command to run scheduler + API
- [ ] Add `status` command to check sync state
- [ ] Add `list` command to show connectors
- [ ] Add `trigger` command for manual runs
- [ ] Improve logging (structured JSON logging)

**2.6 Configuration File**
- [ ] Create example `configs/config.yaml`
- [ ] Add inline documentation with comments
- [ ] Create `configs/config.schema.json`
- [ ] Write config documentation

**Testing Phase 2**:
- [ ] Test incremental sync (no duplicate processing)
- [ ] Test scheduler (interval and cron)
- [ ] Test state persistence across restarts
- [ ] Test concurrent processing with goroutines
- [ ] Load test with 1000+ memory items

### Phase 3: Management API & Monitoring (Week 3)

**Goal**: Add REST API for connector management and monitoring

#### Tasks:

**3.1 Management API**
- [ ] Create `pkg/api/server.go`
- [ ] Set up Gin router (or net/http if preferred)
- [ ] Implement authentication middleware (API key)
- [ ] Implement connector CRUD endpoints:
  - `GET /api/v1/connectors` - List all connectors
  - `POST /api/v1/connectors` - Create new connector
  - `GET /api/v1/connectors/:id` - Get connector details
  - `PUT /api/v1/connectors/:id` - Update connector
  - `DELETE /api/v1/connectors/:id` - Delete connector
- [ ] Implement control endpoints:
  - `POST /api/v1/connectors/:id/trigger` - Manual trigger
  - `POST /api/v1/connectors/:id/pause` - Pause scheduler
  - `POST /api/v1/connectors/:id/resume` - Resume scheduler
- [ ] Implement status endpoints:
  - `GET /api/v1/connectors/:id/status` - Current status
  - `GET /api/v1/connectors/:id/history` - Sync history
  - `GET /api/v1/health` - Health check

**3.2 API Handlers**
- [ ] Create `pkg/api/handlers.go`
- [ ] Implement request/response models
- [ ] Add request validation
- [ ] Add error handling middleware
- [ ] Generate OpenAPI/Swagger docs

**3.3 API Integration**
- [ ] Connect API to Scheduler Service
- [ ] Connect API to State Manager
- [ ] Add real-time status updates
- [ ] Optional: Add Server-Sent Events for progress streaming

**3.4 Enhanced CLI**
- [ ] Update `serve` command to start both scheduler and API
- [ ] Add `--api-only` flag for API-only mode
- [ ] Add `--scheduler-only` flag for scheduler-only mode

**3.5 Monitoring & Logging**
- [ ] Set up structured logging with zap
- [ ] Add metrics collection (simple counters)
- [ ] Optional: Add Prometheus /metrics endpoint
- [ ] Add health check logic

**Testing Phase 3**:
- [ ] API endpoint tests (all CRUD operations)
- [ ] Authentication tests
- [ ] Integration test: Create connector via API, verify it runs
- [ ] Load test API with concurrent requests

### Phase 4: Production Ready & Deployment (Week 4)

**Goal**: Add production-ready features and deployment artifacts

#### Tasks:

**4.1 SQLite State Backend**
- [ ] Create `pkg/state/sqlite_store.go`
- [ ] Implement SQLite backend for StateManager
- [ ] Add database schema migrations
- [ ] Add connection pooling
- [ ] Performance comparison with JSON backend

**4.2 Error Handling Enhancements**
- [ ] Implement retry with exponential backoff per item
- [ ] Add error categorization (retryable vs permanent)
- [ ] Add alerting hooks (webhook support)
- [ ] Add error reporting in status API

**4.3 Security Enhancements**
- [ ] Add API key validation
- [ ] Implement rate limiting middleware
- [ ] Add CORS support
- [ ] Security review and hardening

**4.4 Build & Deployment Artifacts**
- [ ] Create `deployments/docker/Dockerfile`
- [ ] Create `deployments/docker-compose.yaml`
- [ ] Create `deployments/systemd/memory-connector.service`
- [ ] Create Kubernetes manifests:
  - `deployments/k8s/deployment.yaml`
  - `deployments/k8s/service.yaml`
  - `deployments/k8s/configmap.yaml`
  - `deployments/k8s/secret.yaml`
- [ ] Optional: Create Helm chart

**4.5 Build Scripts**
- [ ] Create `scripts/build.sh` for multi-platform builds
- [ ] Create `scripts/install.sh` for installation
- [ ] Set up cross-compilation (Linux, macOS, Windows)
- [ ] Add version info (git tag + commit hash)

**4.6 Documentation**
- [ ] Write README.md for EXTENSIONS/memory-ingestion/
- [ ] User guide (setup, configuration, usage)
- [ ] API reference (OpenAPI/Swagger)
- [ ] Troubleshooting guide
- [ ] Performance tuning guide

**Testing Phase 4**:
- [ ] End-to-end production scenario test
- [ ] Security audit
- [ ] Performance benchmarks (Go benchmarking)
- [ ] Failover and recovery tests
- [ ] Documentation review

## Go-Specific Implementation Details

### Project Structure

```
EXTENSIONS/memory-ingestion/
├── cmd/
│   └── memory-connector/
│       └── main.go
├── pkg/
│   ├── client/
│   ├── transformer/
│   ├── state/
│   ├── scheduler/
│   ├── orchestrator/
│   ├── config/
│   ├── api/
│   └── models/
├── internal/
│   └── logger/
├── configs/
├── deployments/
├── scripts/
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

### Key Go Packages to Use

1. **HTTP Client**: `net/http` (standard library)
2. **HTTP Server**: `github.com/gin-gonic/gin` or `net/http`
3. **Cron Scheduling**: `github.com/robfig/cron/v3`
4. **CLI**: `github.com/spf13/cobra`
5. **Configuration**: `github.com/spf13/viper`
6. **Logging**: `go.uber.org/zap`
7. **YAML**: `gopkg.in/yaml.v3`
8. **SQLite**: `github.com/mattn/go-sqlite3` (CGO)
9. **Testing**: `testing` (standard library) + `github.com/stretchr/testify`

### Build Commands

```makefile
# Makefile

.PHONY: build test clean install

BINARY_NAME=memory-connector
BUILD_DIR=bin
VERSION=$(shell git describe --tags --always --dirty)
LDFLAGS=-ldflags "-X main.Version=$(VERSION)"

build:
	go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) cmd/memory-connector/main.go

build-all:
	GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 cmd/memory-connector/main.go
	GOOS=darwin GOARCH=amd64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 cmd/memory-connector/main.go
	GOOS=darwin GOARCH=arm64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 cmd/memory-connector/main.go
	GOOS=windows GOARCH=amd64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe cmd/memory-connector/main.go

test:
	go test -v -race ./...

test-coverage:
	go test -v -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

clean:
	rm -rf $(BUILD_DIR)

install:
	go install $(LDFLAGS) ./cmd/memory-connector

docker-build:
	docker build -t memory-connector:$(VERSION) -f deployments/docker/Dockerfile .

run:
	go run cmd/memory-connector/main.go serve --config configs/config.yaml

lint:
	golangci-lint run ./...
```

### Dockerfile

```dockerfile
# deployments/docker/Dockerfile

# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o memory-connector cmd/memory-connector/main.go

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/memory-connector .

# Copy config
COPY configs/config.yaml /app/config.yaml

# Expose API port
EXPOSE 9622

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:9622/api/v1/health || exit 1

# Run
ENTRYPOINT ["./memory-connector"]
CMD ["serve", "--config", "/app/config.yaml"]
```

### Testing Strategy

#### Unit Tests
```go
// Example: pkg/transformer/transformer_test.go

package transformer_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/your-org/memory-connector/pkg/models"
    "github.com/your-org/memory-connector/pkg/transformer"
)

func TestStandardStrategy_Transform(t *testing.T) {
    strategy := &transformer.StandardStrategy{}

    memory := &models.Memory{
        ID: "test123",
        Type: "record",
        Transcript: "Test transcript",
        CreatedAt: "2025-12-18T10:00:00Z",
    }

    result, err := strategy.Transform(memory)

    assert.NoError(t, err)
    assert.Contains(t, result, "test123")
    assert.Contains(t, result, "Test transcript")
}
```

#### Integration Tests
```go
// Example: pkg/orchestrator/orchestrator_test.go

package orchestrator_test

import (
    "context"
    "net/http/httptest"
    "testing"
)

func TestOrchestrator_RunIngestion(t *testing.T) {
    // Set up mock Memory API server
    mockAPI := httptest.NewServer(...)
    defer mockAPI.Close()

    // Set up mock LightRAG server
    mockLightRAG := httptest.NewServer(...)
    defer mockLightRAG.Close()

    // Create orchestrator with mocks
    orch := setupOrchestrator(mockAPI.URL, mockLightRAG.URL)

    // Run ingestion
    report, err := orch.RunIngestion(context.Background(), "test-connector")

    // Assertions
    assert.NoError(t, err)
    assert.Equal(t, "success", report.Status)
}
```

### Deployment Examples

#### Systemd Service

```ini
# deployments/systemd/memory-connector.service

[Unit]
Description=Memory Connector Service
After=network.target

[Service]
Type=simple
User=memory-connector
Group=memory-connector
WorkingDirectory=/opt/memory-connector
Environment="MEMORY_API_KEY=your-key"
ExecStart=/opt/memory-connector/bin/memory-connector serve --config /etc/memory-connector/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Docker Compose

```yaml
# deployments/docker-compose.yaml

version: '3.8'

services:
  memory-connector:
    build:
      context: .
      dockerfile: deployments/docker/Dockerfile
    ports:
      - "9622:9622"
    environment:
      - MEMORY_API_KEY=${MEMORY_API_KEY}
      - LIGHTRAG_API_KEY=${LIGHTRAG_API_KEY}
    volumes:
      - ./configs/config.yaml:/app/config.yaml:ro
      - connector-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:9622/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  connector-data:
```

## Performance Targets

- **Throughput**: Process 100 memories in <30 seconds (Go's concurrency advantage)
- **Memory Usage**: <200MB for 10,000 items
- **Binary Size**: <20MB (static build)
- **Startup Time**: <1 second
- **API Latency**: <50ms for status endpoints

## Success Metrics

### Functionality
- ✅ Successfully connects to Memory API
- ✅ Successfully inserts documents into LightRAG
- ✅ No duplicate processing (idempotency)
- ✅ Handles API errors gracefully
- ✅ State persists across restarts

### Performance
- ⏱️ Process 100 memories in <30 seconds
- ⏱️ Memory usage <200MB
- ⏱️ Binary size <20MB
- ⏱️ API response time <50ms

### Reliability
- 🎯 99.9% uptime for scheduler
- 🎯 <0.1% duplicate processing rate
- 🎯 100% state recovery after crash
- 🎯 Auto-retry on transient failures

---

## CLARIFICATION QUESTIONS

**Please review and answer the following questions before we start implementation:**

### 1. Go Module Naming
- **Question**: What should be the Go module name?
- **Options**:
  - `github.com/kamir/LightRAG/extensions/memory-ingestion`
  - `github.com/kamir/memory-connector`
  - Other?
- **Your Answer**: github.com/kamir/memory-connector

### 2. HTTP Framework Choice
- **Question**: Which HTTP framework should we use for the REST API?
- **Options**:
  - **Gin** (feature-rich, popular, slightly heavier)
  - **net/http** (standard library, lightweight, more boilerplate)
  - **Fiber** (Express-like, very fast)
  - **Chi** (lightweight router on top of net/http)
- **Your Answer**: Gin
### 3. State Backend Priority
- **Question**: Which state backend should we implement first?
- **Options**:
  - **JSON** (simpler, good for single instance)
  - **SQLite** (better for production, allows querying)
  - **Both in parallel**
- **Your Answer**: Both in parallel

### 4. Logging Format
- **Question**: What logging format do you prefer?
- **Options**:
  - **JSON** (structured, machine-readable, better for log aggregation)
  - **Console** (human-readable, colorized, better for development)
  - **Both** (configurable via config)
- **Your Answer**: Both (configurable via config)

### 5. Configuration Hot-Reload
- **Question**: Should configuration support hot-reload (reload on SIGHUP without restart)?
- **Options**:
  - **Yes** (more flexible, but more complex)
  - **No** (simpler, require restart for config changes)
- **Your Answer**: No

### 6. Metrics/Observability
- **Question**: Should we include Prometheus metrics from the start?
- **Options**:
  - **Yes** (better observability, slightly more code)
  - **No** (defer to Phase 2)
  - **Simple counters only** (minimal overhead)
- **Your Answer**: Simple counters only  

### 7. Database for State (if SQLite chosen)
- **Question**: If using SQLite, should we use CGO or pure Go?
- **Options**:
  - **mattn/go-sqlite3** (CGO, full SQLite features, complicates cross-compilation)
  - **modernc.org/sqlite** (Pure Go, easier cross-compilation, slightly slower)
- **Your Answer**: modernc.org/sqlite

### 8. Concurrent Processing
- **Question**: How many memories should we process concurrently?
- **Options**:
  - **Fixed** (e.g., 10 goroutines)
  - **Configurable** (via config file)
  - **Dynamic** (based on system resources)
- **Your Answer**: via config file

### 9. Binary Distribution
- **Question**: How should we distribute the binary?
- **Options**:
  - **GitHub Releases** (attach binaries to releases)
  - **Docker only**
  - **Both**
  - **Also create install script** (download latest binary)
- **Your Answer**: GitHub Releases, Docker, and also create install script

### 10. Development Environment
- **Question**: What Go version should we target?
- **Options**:
  - **Go 1.21** (stable, widely available)
  - **Go 1.22** (latest stable, better performance)
  - **Go 1.23** (bleeding edge)
- **Your Answer**: Go 1.21

### 11. Testing Coverage Target
- **Question**: What code coverage target should we aim for?
- **Options**:
  - **60%** (reasonable)
  - **80%** (comprehensive)
  - **90%+** (very thorough, more effort)
- **Your Answer**: 90%+

### 12. Error Handling for Memory API
- **Question**: If Memory API returns partial results or times out, should we:
- **Options**:
  - **Process what we got** (partial sync)
  - **Abort and retry** (all-or-nothing)
  - **Configurable per connector**
- **Your Answer**: Process what we got and track was was lost and what went wrong, capture the errors like in a DLQ.

### 13. LightRAG Connection Mode
- **Question**: Should we implement both API and "Direct" mode from the start?
- **Options**:
  - **API mode only** (HTTP to LightRAG, easier)
  - **Both modes** (HTTP + direct library calls, but LightRAG is Python)
- **Note**: Since LightRAG is Python, "Direct" mode would still be HTTP calls
- **Your Answer**: API mode only

### 14. CLI Output Format
- **Question**: What output format for CLI commands?
- **Options**:
  - **Human-readable text** (pretty tables, colors)
  - **JSON** (machine-readable)
  - **Both** (with --json flag)
- **Your Answer**: Both (with --json flag)

### 15. Initial Feature Scope
- **Question**: Should we include any Phase 2 features in initial release?
- **Phase 2 features**: Audio/image processing, rich transformation, webhooks
- **Options**:
  - **No, stick to Phase 1** (faster initial delivery)
  - **Yes, include rich transformation** (geocoding, advanced formatting)
  - **Yes, include webhooks** (for notifications)
- **Your Answer**: No, stick to Phase 1

---

**Instructions**: Please edit this file and fill in your answers. Once complete, let me know and we'll proceed with implementation based on your preferences.
