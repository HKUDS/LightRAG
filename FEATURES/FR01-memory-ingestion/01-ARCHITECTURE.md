# FR01: Memory API Ingestion - Detailed Architecture (Go Implementation)

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Memory Ingestion Connector                       │
│                     Location: EXTENSIONS/memory-ingestion/           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Management API (net/http or Gin)         │    │
│  │  /connectors  /status  /history  /trigger  /health         │    │
│  └───────────────────────────┬────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼────────────────────────────────┐    │
│  │              Ingestion Orchestrator                         │    │
│  │  - Job coordination (goroutines)                            │    │
│  │  - Pipeline execution                                       │    │
│  │  - Error handling & retry                                   │    │
│  └─────┬──────────────┬───────────────┬───────────────────────┘    │
│        │              │               │                             │
│  ┌─────▼─────┐  ┌────▼─────┐  ┌──────▼────────┐                   │
│  │ Scheduler  │  │  State   │  │  Config       │                   │
│  │ Service    │  │  Manager │  │  Manager      │                   │
│  │            │  │          │  │               │                   │
│  │robfig/cron │  │ JSON/DB  │  │  YAML/Env     │                   │
│  └─────┬──────┘  └────┬─────┘  └──────┬────────┘                   │
│        │              │               │                             │
│  ┌─────▼──────────────▼───────────────▼─────────────────────┐     │
│  │              Ingestion Pipeline                           │     │
│  │                                                            │     │
│  │  1. Fetch  →  2. Filter  →  3. Transform  →  4. Submit   │     │
│  │     ↓             ↓              ↓               ↓         │     │
│  │  API Client  State Check   Transformer    LightRAG Client │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
           │                                              │
           │ HTTP                                         │ HTTP
           ▼                                              ▼
   ┌───────────────┐                            ┌──────────────────┐
   │  Memory API   │                            │  LightRAG        │
   │               │                            │  Instance        │
   └───────────────┘                            └──────────────────┘
```

## Project Structure

```
EXTENSIONS/memory-ingestion/
├── cmd/
│   └── memory-connector/
│       └── main.go                    # Application entry point
├── pkg/
│   ├── client/
│   │   ├── memory_client.go           # Memory API client
│   │   └── lightrag_client.go         # LightRAG API client
│   ├── transformer/
│   │   ├── transformer.go             # Data transformation
│   │   └── strategies.go              # Transformation strategies
│   ├── state/
│   │   ├── state.go                   # State manager interface
│   │   ├── json_store.go              # JSON-based state storage
│   │   └── sqlite_store.go            # SQLite-based state storage
│   ├── scheduler/
│   │   └── scheduler.go               # Cron-based job scheduler
│   ├── orchestrator/
│   │   └── orchestrator.go            # Pipeline orchestrator
│   ├── config/
│   │   └── config.go                  # Configuration management
│   ├── api/
│   │   ├── server.go                  # HTTP server
│   │   ├── handlers.go                # HTTP handlers
│   │   └── middleware.go              # Middleware (auth, logging)
│   └── models/
│       ├── memory.go                  # Memory API models
│       ├── connector.go               # Connector configuration
│       └── report.go                  # Ingestion report models
├── internal/
│   └── logger/
│       └── logger.go                  # Structured logging setup
├── configs/
│   ├── config.yaml                    # Example configuration
│   └── config.schema.json             # Config JSON schema
├── deployments/
│   ├── docker/
│   │   └── Dockerfile
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── systemd/
│       └── memory-connector.service
├── scripts/
│   ├── build.sh                       # Build script
│   └── install.sh                     # Installation script
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

## Core Components

### 1. Memory API Client

**File**: `pkg/client/memory_client.go`

**Responsibilities**:
- Authenticate with Memory API
- Fetch memory lists with query parameters
- Handle retries and timeouts
- Rate limiting and backoff
- Error handling for network issues

**Key Structures & Methods**:

```go
package client

import (
    "context"
    "net/http"
    "time"
)

type MemoryClient struct {
    apiURL     string
    apiKey     string
    httpClient *http.Client
    maxRetries int
}

type MemoryList struct {
    Memories []Memory `json:"memories"`
}

type Memory struct {
    ID          string  `json:"id"`
    Type        string  `json:"type"`
    Audio       bool    `json:"audio"`
    Image       bool    `json:"image"`
    Transcript  string  `json:"transcript"`
    LocationLat float64 `json:"location_lat"`
    LocationLon float64 `json:"location_lon"`
    CreatedAt   string  `json:"created_at"`
}

func NewMemoryClient(apiURL, apiKey string, timeout time.Duration) *MemoryClient {
    return &MemoryClient{
        apiURL: apiURL,
        apiKey: apiKey,
        httpClient: &http.Client{
            Timeout: timeout,
        },
        maxRetries: 3,
    }
}

func (c *MemoryClient) GetMemories(ctx context.Context, ctxID string, limit int, rangeParam string) (*MemoryList, error) {
    // Implementation with retry logic
}

func (c *MemoryClient) DownloadAudio(ctx context.Context, ctxID, memoryID string) ([]byte, error) {
    // Implementation
}

func (c *MemoryClient) DownloadImage(ctx context.Context, ctxID, memoryID string) ([]byte, error) {
    // Implementation
}

func (c *MemoryClient) CheckConnection(ctx context.Context) error {
    // Test API connectivity
}
```

**Configuration**:
```yaml
memory_api:
  url: "http://127.0.0.1:8080"
  api_key: "${MEMORY_API_KEY}"
  timeout: 30s
  max_retries: 3
  retry_backoff: 2s
```

### 2. Data Transformer

**File**: `pkg/transformer/transformer.go`

**Responsibilities**:
- Convert Memory API schema to LightRAG document format
- Enrich data with context and metadata
- Generate structured text for entity extraction
- Handle missing/optional fields
- Support multiple transformation strategies

**Key Structures & Methods**:

```go
package transformer

import (
    "fmt"
    "time"
    "github.com/your-org/memory-connector/pkg/models"
)

type Strategy interface {
    Transform(memory *models.Memory) (string, error)
}

type StandardStrategy struct{}

func (s *StandardStrategy) Transform(memory *models.Memory) (string, error) {
    createdAt, _ := time.Parse(time.RFC3339, memory.CreatedAt)
    dateStr := createdAt.Format("January 02, 2006 at 03:04 PM UTC")

    locationStr := ""
    if memory.LocationLat != 0.0 || memory.LocationLon != 0.0 {
        locationStr = fmt.Sprintf("\nLocation: (%.6f, %.6f)",
            memory.LocationLat, memory.LocationLon)
    }

    media := []string{}
    if memory.Audio {
        media = append(media, "Audio")
    }
    if memory.Image {
        media = append(media, "Image")
    }
    mediaStr := "None"
    if len(media) > 0 {
        mediaStr = fmt.Sprintf("%v", media)
    }

    transcript := memory.Transcript
    if transcript == "" {
        transcript = "[No transcript available]"
    }

    return fmt.Sprintf(`---
Memory Record
ID: %s
Type: %s
Recorded: %s%s
Media Available: %s

Transcript:
%s
---`, memory.ID, memory.Type, dateStr, locationStr, mediaStr, transcript), nil
}

type RichStrategy struct {
    GeocodingEnabled bool
}

func (s *RichStrategy) Transform(memory *models.Memory) (string, error) {
    // Enhanced transformation with geocoding, tagging, etc.
}

type Transformer struct {
    strategy Strategy
}

func New(strategy Strategy) *Transformer {
    return &Transformer{strategy: strategy}
}

func (t *Transformer) Transform(memory *models.Memory) (string, error) {
    return t.strategy.Transform(memory)
}

func (t *Transformer) TransformBatch(memories []*models.Memory) ([]string, error) {
    results := make([]string, 0, len(memories))
    for _, memory := range memories {
        doc, err := t.Transform(memory)
        if err != nil {
            return nil, err
        }
        results = append(results, doc)
    }
    return results, nil
}
```

### 3. State Manager

**File**: `pkg/state/state.go`

**Responsibilities**:
- Track ingestion state per connector/context
- Prevent duplicate processing
- Store sync history and metrics
- Support state queries and rollback

**Key Structures & Methods**:

```go
package state

import (
    "context"
    "time"
)

type SyncState struct {
    ConnectorID         string            `json:"connector_id"`
    ContextID           string            `json:"context_id"`
    LastSyncTimestamp   time.Time         `json:"last_sync_timestamp"`
    LastSuccessfulSync  time.Time         `json:"last_successful_sync"`
    ProcessedMemoryIDs  map[string]bool   `json:"processed_memory_ids"`
    FailedMemoryIDs     map[string]string `json:"failed_memory_ids"` // memory_id -> error
    TotalProcessed      int               `json:"total_processed"`
    TotalFailed         int               `json:"total_failed"`
    Status              string            `json:"status"` // idle, running, completed, failed
}

type Manager interface {
    GetState(ctx context.Context, connectorID, contextID string) (*SyncState, error)
    UpdateState(ctx context.Context, state *SyncState) error
    MarkProcessed(ctx context.Context, connectorID, contextID, memoryID string) error
    MarkFailed(ctx context.Context, connectorID, contextID, memoryID string, errorMsg string) error
    IsProcessed(ctx context.Context, connectorID, contextID, memoryID string) (bool, error)
    GetUnprocessedIDs(ctx context.Context, connectorID, contextID string, allIDs []string) ([]string, error)
    GetSyncHistory(ctx context.Context, connectorID string, limit int) ([]*SyncState, error)
    ResetState(ctx context.Context, connectorID, contextID string) error
}

// JSONStore implements Manager using JSON files
type JSONStore struct {
    filePath string
}

// SQLiteStore implements Manager using SQLite database
type SQLiteStore struct {
    dbPath string
}
```

**State Storage Format (JSON)**:
```json
{
  "connectors": {
    "memory-connector-1": {
      "contexts": {
        "CTX123": {
          "connector_id": "memory-connector-1",
          "context_id": "CTX123",
          "last_sync_timestamp": "2025-12-18T15:00:00Z",
          "last_successful_sync": "2025-12-18T15:00:00Z",
          "processed_memory_ids": {
            "mem_001": true,
            "mem_002": true,
            "mem_003": true
          },
          "failed_memory_ids": {
            "mem_004": "Network timeout"
          },
          "total_processed": 145,
          "total_failed": 2,
          "status": "completed"
        }
      }
    }
  }
}
```

### 4. Scheduler Service

**File**: `pkg/scheduler/scheduler.go`

**Responsibilities**:
- Manage periodic ingestion jobs using robfig/cron
- Support multiple schedules
- Job control (pause, resume, trigger)
- Monitor job health

**Key Structures & Methods**:

```go
package scheduler

import (
    "context"
    "github.com/robfig/cron/v3"
)

type Scheduler struct {
    cron        *cron.Cron
    jobs        map[string]cron.EntryID
    orchestrator Orchestrator
}

type JobConfig struct {
    ConnectorID string
    Schedule    string // Cron expression or @every interval
}

func New(orchestrator Orchestrator) *Scheduler {
    return &Scheduler{
        cron: cron.New(cron.WithSeconds()),
        jobs: make(map[string]cron.EntryID),
        orchestrator: orchestrator,
    }
}

func (s *Scheduler) Start(ctx context.Context) error {
    s.cron.Start()
    <-ctx.Done()
    s.cron.Stop()
    return nil
}

func (s *Scheduler) AddJob(config JobConfig) error {
    entryID, err := s.cron.AddFunc(config.Schedule, func() {
        s.runJob(config.ConnectorID)
    })
    if err != nil {
        return err
    }
    s.jobs[config.ConnectorID] = entryID
    return nil
}

func (s *Scheduler) RemoveJob(connectorID string) {
    if entryID, exists := s.jobs[connectorID]; exists {
        s.cron.Remove(entryID)
        delete(s.jobs, connectorID)
    }
}

func (s *Scheduler) TriggerNow(connectorID string) error {
    go s.runJob(connectorID)
    return nil
}

func (s *Scheduler) runJob(connectorID string) {
    ctx := context.Background()
    _ = s.orchestrator.RunIngestion(ctx, connectorID)
}
```

### 5. Ingestion Orchestrator

**File**: `pkg/orchestrator/orchestrator.go`

**Responsibilities**:
- Coordinate the complete ingestion pipeline
- Implement ingestion strategies (full, incremental, selective)
- Error handling and retry logic
- Progress reporting

**Key Structures & Methods**:

```go
package orchestrator

import (
    "context"
    "sync"
    "time"
)

type Orchestrator struct {
    configMgr     ConfigManager
    stateMgr      StateManager
    memoryClient  MemoryClient
    lightragClient LightRAGClient
    transformer   Transformer
    logger        Logger
}

type IngestionReport struct {
    ConnectorID      string    `json:"connector_id"`
    StartTime        time.Time `json:"start_time"`
    EndTime          time.Time `json:"end_time"`
    Status           string    `json:"status"`
    TotalFetched     int       `json:"total_fetched"`
    TotalToProcess   int       `json:"total_to_process"`
    SuccessfulCount  int       `json:"successful_count"`
    FailedCount      int       `json:"failed_count"`
    Errors           []string  `json:"errors"`
    Error            string    `json:"error,omitempty"`
}

func (o *Orchestrator) RunIngestion(ctx context.Context, connectorID string) (*IngestionReport, error) {
    // 1. Load connector config
    // 2. Initialize clients
    // 3. Get current state
    // 4. Fetch memories from API
    // 5. Filter unprocessed
    // 6. Process in batches (concurrent using goroutines)
    // 7. Update state
    // 8. Generate report
}

func (o *Orchestrator) processBatch(ctx context.Context, batch []*Memory, config ConnectorConfig) error {
    var wg sync.WaitGroup
    errChan := make(chan error, len(batch))

    for _, memory := range batch {
        wg.Add(1)
        go func(m *Memory) {
            defer wg.Done()
            if err := o.processMemory(ctx, m, config); err != nil {
                errChan <- err
            }
        }(memory)
    }

    wg.Wait()
    close(errChan)

    // Collect errors
    for err := range errChan {
        // Handle errors
    }

    return nil
}
```

### 6. Configuration Manager

**File**: `pkg/config/config.go`

**Configuration Schema**:

```go
package config

type Config struct {
    LightRAG   LightRAGConfig   `yaml:"lightrag"`
    MemoryAPI  MemoryAPIConfig  `yaml:"memory_api"`
    Connectors []ConnectorConfig `yaml:"connectors"`
    State      StateConfig      `yaml:"state"`
    API        APIConfig        `yaml:"api"`
    Logging    LoggingConfig    `yaml:"logging"`
}

type LightRAGConfig struct {
    Mode   string         `yaml:"mode"` // "api" or "direct"
    API    APIConnConfig  `yaml:"api"`
}

type MemoryAPIConfig struct {
    URL           string        `yaml:"url"`
    APIKey        string        `yaml:"api_key"`
    Timeout       time.Duration `yaml:"timeout"`
    MaxRetries    int           `yaml:"max_retries"`
    RetryBackoff  time.Duration `yaml:"retry_backoff"`
}

type ConnectorConfig struct {
    ID              string               `yaml:"id"`
    Enabled         bool                 `yaml:"enabled"`
    ContextID       string               `yaml:"context_id"`
    Schedule        ScheduleConfig       `yaml:"schedule"`
    Ingestion       IngestionConfig      `yaml:"ingestion"`
    Transformation  TransformationConfig `yaml:"transformation"`
    Retry           RetryConfig          `yaml:"retry"`
}

type ScheduleConfig struct {
    Type          string `yaml:"type"` // "interval" or "cron"
    IntervalHours int    `yaml:"interval_hours,omitempty"`
    Cron          string `yaml:"cron,omitempty"`
}

func Load(path string) (*Config, error) {
    // Load YAML config with environment variable substitution
}

func (c *Config) Validate() error {
    // Validate configuration
}
```

### 7. REST API Server

**File**: `pkg/api/server.go`

**Endpoints**:

```go
package api

import (
    "net/http"
    "github.com/gin-gonic/gin" // or use net/http
)

type Server struct {
    router      *gin.Engine
    orchestrator *Orchestrator
    stateMgr    StateManager
    configMgr   ConfigManager
}

func NewServer(orchestrator *Orchestrator, stateMgr StateManager, configMgr ConfigManager) *Server {
    s := &Server{
        router: gin.Default(),
        orchestrator: orchestrator,
        stateMgr: stateMgr,
        configMgr: configMgr,
    }
    s.setupRoutes()
    return s
}

func (s *Server) setupRoutes() {
    api := s.router.Group("/api/v1")
    {
        api.GET("/health", s.handleHealth)
        api.GET("/connectors", s.handleListConnectors)
        api.GET("/connectors/:id/status", s.handleGetStatus)
        api.GET("/connectors/:id/history", s.handleGetHistory)
        api.POST("/connectors/:id/trigger", s.handleTrigger)
    }
}

func (s *Server) handleHealth(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "status": "healthy",
        "version": "1.0.0",
    })
}
```

## Dependencies (go.mod)

```go
module github.com/your-org/memory-connector

go 1.21

require (
    github.com/gin-gonic/gin v1.10.0
    github.com/robfig/cron/v3 v3.0.1
    github.com/mattn/go-sqlite3 v1.14.19
    github.com/spf13/cobra v1.8.0
    github.com/spf13/viper v1.18.2
    go.uber.org/zap v1.26.0
    gopkg.in/yaml.v3 v3.0.1
)
```

## Build & Deployment

### Makefile

```makefile
.PHONY: build test clean install

BINARY_NAME=memory-connector
BUILD_DIR=bin

build:
	go build -o $(BUILD_DIR)/$(BINARY_NAME) cmd/memory-connector/main.go

build-linux:
	GOOS=linux GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-linux cmd/memory-connector/main.go

test:
	go test -v ./...

clean:
	rm -rf $(BUILD_DIR)

install:
	go install ./cmd/memory-connector

docker-build:
	docker build -t memory-connector:latest -f deployments/docker/Dockerfile .

run:
	go run cmd/memory-connector/main.go serve --config configs/config.yaml
```

## Next Steps

See `02-IMPLEMENTATION-PLAN.md` for the detailed implementation roadmap with Go-specific tooling and build instructions.
