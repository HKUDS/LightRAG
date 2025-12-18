# FR01: Memory API Ingestion - Overview

## Executive Summary

This feature adds automated ingestion capability to LightRAG for periodically pulling memory items from a remote Memory API and populating the knowledge graph. The solution includes a connector interface for configuring ingestion schedules, tracking progress, and managing the ingestion lifecycle.

## Problem Statement

The user has a Memory API service that stores audio recordings, images, transcripts, and metadata (location, timestamps, etc.). Currently, there is no automated way to:

1. Pull memory items from the API on a regular schedule
2. Transform memory data into LightRAG knowledge graph format
3. Track ingestion progress and status
4. Configure which data to ingest and when
5. Handle incremental updates (only new/updated items)

## Solution Approach

We will implement a **standalone ingestion service** that:

- Runs independently or alongside LightRAG API server
- Pulls memory items from the Memory API on configurable schedules
- Transforms memory data (transcript, metadata) into documents for LightRAG
- Tracks ingestion state to avoid duplicate processing
- Provides a configuration interface for connector settings
- Reports progress and status via API endpoints

## Key Design Decisions

### 1. Standalone Tool vs. Integrated Module

**DECISION: Standalone Tool with Optional Integration**

**Rationale:**
- **Separation of Concerns**: Memory API integration is domain-specific logic
- **Deployment Flexibility**: Can run as separate service or embedded
- **Resource Isolation**: Scheduler/polling doesn't impact RAG query performance
- **Easier Testing**: Independent testing without full LightRAG stack
- **Reusability**: Can connect to multiple LightRAG instances

**Architecture:**
```
┌─────────────────────┐
│   Memory API        │
│  (Your Service)     │
└──────────┬──────────┘
           │
           │ HTTP/REST
           │
┌──────────▼──────────────────────────────────────┐
│  Memory Ingestion Connector (Standalone)        │
│  ┌────────────────┐  ┌──────────────────┐      │
│  │  API Client    │  │  Scheduler       │      │
│  │  - Fetch       │  │  - Periodic Runs │      │
│  │  - Transform   │  │  - Cron/Interval │      │
│  └────────────────┘  └──────────────────┘      │
│  ┌────────────────┐  ┌──────────────────┐      │
│  │  State Manager │  │  Config Manager  │      │
│  │  - Last Sync   │  │  - API Keys      │      │
│  │  - Track Items │  │  - Schedules     │      │
│  └────────────────┘  └──────────────────┘      │
└──────────┬──────────────────────────────────────┘
           │
           │ LightRAG API or Direct
           │
┌──────────▼──────────┐
│   LightRAG          │
│   Knowledge Graph   │
└─────────────────────┘
```

### 2. Integration Method

**DECISION: Hybrid Approach**

The tool will support two integration modes:

**A. LightRAG API Mode** (Recommended for production)
- Uses LightRAG's existing REST API (`POST /documents/text`)
- Works with remote or local LightRAG instances
- Proper authentication and workspace support
- Better for distributed deployments

**B. Direct Library Mode** (For embedded scenarios)
- Imports LightRAG as Python library
- Calls `rag.ainsert()` directly
- Lower overhead, no network latency
- Better for single-process deployments

### 3. State Management

**DECISION: JSON-based State Store with SQLite Option**

- **Development/Simple**: JSON file (`memory_sync_state.json`)
- **Production/Advanced**: SQLite database (`memory_sync_state.db`)
- Tracks:
  - Last successful sync timestamp per context
  - Memory IDs already processed
  - Failed ingestion attempts
  - Sync statistics and metrics

### 4. Scheduling

**DECISION: APScheduler with Configurable Triggers**

- Use `APScheduler` library for flexible scheduling
- Support multiple trigger types:
  - Interval: Every N hours/minutes
  - Cron: Specific times (e.g., "0 */1 * * *" for hourly)
  - Manual: On-demand trigger via API
- Background scheduler runs in separate thread
- Graceful shutdown on signals

## Component Overview

### 1. Memory API Client (`memory_api_client.py`)
- Authenticates with Memory API (API key)
- Fetches memory lists with filtering (context, date range, limit)
- Retrieves individual memory items
- Downloads associated resources (audio, images) if needed
- Handles pagination and rate limiting

### 2. Data Transformer (`memory_transformer.py`)
- Converts Memory API schema → LightRAG document format
- Enriches transcript with metadata:
  - Location information (lat/lon → place names if available)
  - Temporal context (created_at → human-readable dates)
  - Memory type and characteristics
- Generates structured text for optimal entity extraction
- Handles missing/optional fields gracefully

### 3. Ingestion Orchestrator (`ingestion_orchestrator.py`)
- Coordinates the ingestion pipeline
- Implements ingestion strategies:
  - **Full Sync**: Process all memories (first run)
  - **Incremental Sync**: Only new memories since last sync
  - **Selective Sync**: Filter by date range, type, etc.
- Handles errors and retries
- Tracks progress and generates reports

### 4. State Manager (`state_manager.py`)
- Persists sync state across runs
- Implements idempotency (no duplicate processing)
- Tracks per-context sync status
- Provides state query APIs

### 5. Configuration Manager (`config_manager.py`)
- Loads configuration from:
  - YAML/JSON config files
  - Environment variables
  - CLI arguments
- Validates configuration
- Supports hot-reload for some settings

### 6. Scheduler Service (`scheduler_service.py`)
- Manages periodic ingestion jobs
- Supports multiple schedules (different contexts on different cadences)
- Provides job control (pause, resume, trigger now)
- Logs scheduler events

### 7. REST API Interface (`connector_api.py`)
- FastAPI-based management API
- Endpoints:
  - `GET /connectors` - List configured connectors
  - `POST /connectors` - Add new connector
  - `PUT /connectors/{id}` - Update connector config
  - `DELETE /connectors/{id}` - Remove connector
  - `POST /connectors/{id}/trigger` - Manual trigger
  - `GET /connectors/{id}/status` - Get sync status
  - `GET /connectors/{id}/history` - Sync history

## Data Flow

```
1. Scheduler triggers ingestion job
   ↓
2. Memory API Client fetches memory list
   - Query: /memory/{ctx_id}?range=week&limit=100
   - Response: MemoryList with Memory items
   ↓
3. State Manager filters out already-processed items
   - Check memory IDs against state store
   - Return only new/updated items
   ↓
4. For each new Memory item:
   a. Fetch full details (if needed)
   b. Download transcript (or use from list)
   c. Transform to LightRAG document format:

      Document:
      ---
      Memory Record [ID: {memory_id}]
      Type: {type}
      Recorded: {created_at}
      Location: {location_lat}, {location_lon}

      Transcript:
      {transcript}

      Audio Available: {audio}
      Image Available: {image}
      ---

   d. Submit to LightRAG:
      - API Mode: POST /documents/text
      - Direct Mode: rag.ainsert(document)
   ↓
5. Update state store:
   - Mark memory IDs as processed
   - Update last_sync_timestamp
   - Record statistics (success/failure counts)
   ↓
6. Generate sync report
   - Items processed
   - Errors encountered
   - Next scheduled run
```

## Key Features

### 1. Connector Configuration UI
- Configure multiple Memory API connectors
- Each connector represents:
  - API endpoint and credentials
  - Target context ID(s)
  - Ingestion schedule
  - Transformation rules
  - LightRAG workspace mapping

### 2. Incremental Sync
- Tracks last sync timestamp per context
- Queries API with `range` and date filtering
- Only processes new items
- Configurable lookback window for safety

### 3. Progress Tracking
- Real-time sync status
- Items queued/processed/failed
- Estimated time remaining
- Historical sync reports

### 4. Error Handling
- Automatic retry with exponential backoff
- Dead letter queue for failed items
- Alerts on consecutive failures
- Detailed error logs

### 5. Multi-Context Support
- Single connector can sync multiple contexts
- Per-context scheduling (context A: hourly, context B: daily)
- Isolated state tracking per context

## Technology Stack

- **Python 3.11+**
- **FastAPI** - REST API framework
- **APScheduler** - Job scheduling
- **httpx** - Async HTTP client
- **Pydantic** - Data validation
- **SQLAlchemy** (optional) - SQLite ORM for state
- **PyYAML** - Configuration parsing

## Deployment Modes

### Mode 1: Standalone Service
```bash
python -m memory_connector serve \
  --config config.yaml \
  --host 0.0.0.0 \
  --port 9622
```

### Mode 2: Embedded in LightRAG API
```python
# In lightrag_server.py
from memory_connector import MemoryConnectorApp
app.mount("/memory-connector", MemoryConnectorApp())
```

### Mode 3: CLI Tool
```bash
# One-time manual sync
python -m memory_connector sync \
  --api-url http://127.0.0.1:8080 \
  --api-key YOUR_KEY \
  --context-id CTX123 \
  --lightrag-workspace memories
```

## Future Extensions (Phase 2)

1. **Memory Manager**
   - Bidirectional sync (LightRAG → Memory API)
   - Export knowledge graph to portable format
   - Backup/restore workflows
   - Collaborative sharing (replicate to other instances)

2. **Advanced Features**
   - Audio/image processing (transcription, OCR, vision models)
   - Entity linking across memory items
   - Temporal analysis (memory timelines)
   - Location-based clustering
   - Multi-modal knowledge graph

3. **Enterprise Features**
   - Multi-tenant support
   - RBAC for connector management
   - Audit logs
   - Prometheus metrics
   - Webhook notifications

## Success Metrics

- Successful hourly sync of new memory items
- <1% duplicate processing rate
- <5 second latency per memory item
- 99.9% uptime for connector service
- Full state recovery after crashes

## Next Steps

1. Review and approve this architectural plan
2. Implement Phase 1: Core ingestion connector
3. Test with real Memory API
4. Deploy and monitor
5. Iterate based on feedback
6. Plan Phase 2: Memory Manager features
