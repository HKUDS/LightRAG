# FR01: Memory API Ingestion - Detailed Architecture

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Memory Ingestion Connector                       │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Management API (FastAPI)                 │    │
│  │  /connectors  /status  /history  /trigger  /health         │    │
│  └───────────────────────────┬────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼────────────────────────────────┐    │
│  │              Ingestion Orchestrator                         │    │
│  │  - Job coordination                                         │    │
│  │  - Pipeline execution                                       │    │
│  │  - Error handling & retry                                   │    │
│  └─────┬──────────────┬───────────────┬───────────────────────┘    │
│        │              │               │                             │
│  ┌─────▼─────┐  ┌────▼─────┐  ┌──────▼────────┐                   │
│  │ Scheduler  │  │  State   │  │  Config       │                   │
│  │ Service    │  │  Manager │  │  Manager      │                   │
│  │            │  │          │  │               │                   │
│  │ APScheduler│  │ JSON/DB  │  │  YAML/Env     │                   │
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
           │ HTTP                                         │ HTTP/Direct
           ▼                                              ▼
   ┌───────────────┐                            ┌──────────────────┐
   │  Memory API   │                            │  LightRAG        │
   │               │                            │  Instance        │
   └───────────────┘                            └──────────────────┘
```

## Core Components

### 1. Memory API Client

**File**: `lightrag/connectors/memory_api_client.py`

**Responsibilities**:
- Authenticate with Memory API
- Fetch memory lists with query parameters
- Handle pagination if API supports it
- Rate limiting and backoff
- Error handling for network issues

**Key Classes**:

```python
class MemoryAPIClient:
    """Client for interacting with Memory API"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """Initialize client with connection settings"""

    async def get_memories(
        self,
        ctx_id: str,
        limit: int = 100,
        range: str = "week"
    ) -> MemoryList:
        """Fetch list of memories for a context"""

    async def get_memory(
        self,
        ctx_id: str,
        memory_id: str
    ) -> Memory:
        """Fetch single memory item with full details"""

    async def download_audio(
        self,
        ctx_id: str,
        memory_id: str,
        output_path: Path
    ) -> bool:
        """Download audio file for a memory"""

    async def download_image(
        self,
        ctx_id: str,
        memory_id: str,
        output_path: Path
    ) -> bool:
        """Download image file for a memory"""

    async def check_connection(self) -> bool:
        """Test API connectivity and authentication"""
```

**Configuration**:
```yaml
memory_api:
  url: "http://127.0.0.1:8080"
  api_key: "${MEMORY_API_KEY}"
  timeout: 30
  max_retries: 3
  retry_backoff: 2.0  # Exponential backoff multiplier
```

### 2. Data Transformer

**File**: `lightrag/connectors/memory_transformer.py`

**Responsibilities**:
- Convert Memory API schema to LightRAG document format
- Enrich data with context and metadata
- Generate structured text for entity extraction
- Handle missing/optional fields
- Support multiple transformation strategies

**Key Classes**:

```python
class TransformationStrategy(ABC):
    """Base class for transformation strategies"""

    @abstractmethod
    def transform(self, memory: Memory) -> str:
        """Transform Memory to LightRAG document text"""


class StandardTransformationStrategy(TransformationStrategy):
    """Standard transformation: transcript + metadata as structured text"""

    def transform(self, memory: Memory) -> str:
        """
        Output format:
        ---
        Memory Record
        ID: {id}
        Type: {type}
        Recorded: {created_at}
        Location: ({lat}, {lon})
        Has Audio: {audio}
        Has Image: {image}

        Transcript:
        {transcript}
        ---
        """


class RichTransformationStrategy(TransformationStrategy):
    """
    Rich transformation: includes geocoding, datetime parsing,
    and additional context
    """

    def __init__(self, geocoding_enabled: bool = False):
        self.geocoding = geocoding_enabled

    def transform(self, memory: Memory) -> str:
        """
        Output format includes:
        - Reverse geocoded location names
        - Formatted dates (relative and absolute)
        - Inferred tags/categories
        - Sentiment analysis (optional)
        """


class MemoryTransformer:
    """Main transformer class"""

    def __init__(self, strategy: TransformationStrategy):
        self.strategy = strategy

    async def transform_batch(
        self,
        memories: List[Memory]
    ) -> List[TransformedDocument]:
        """Transform multiple memories in parallel"""

    def set_strategy(self, strategy: TransformationStrategy):
        """Change transformation strategy at runtime"""
```

**Example Transformation**:

Input (Memory API JSON):
```json
{
  "id": "mem_1234567890",
  "type": "record",
  "audio": true,
  "image": false,
  "transcript": "Had a great meeting with the team today. We discussed the new product roadmap and agreed on Q1 priorities.",
  "location_lat": 37.7749,
  "location_lon": -122.4194,
  "created_at": "2025-12-18T14:30:00Z"
}
```

Output (LightRAG document):
```
---
Memory Record from December 18, 2025
ID: mem_1234567890
Type: Voice Recording
Location: San Francisco, CA (37.7749, -122.4194)
Time: 2:30 PM UTC
Media: Audio available

Transcript:
Had a great meeting with the team today. We discussed the new product roadmap and agreed on Q1 priorities.

Tags: #meeting #team #product-roadmap #planning
---
```

### 3. State Manager

**File**: `lightrag/connectors/state_manager.py`

**Responsibilities**:
- Track ingestion state per connector/context
- Prevent duplicate processing
- Store sync history and metrics
- Support state queries and rollback

**Key Classes**:

```python
class SyncState(BaseModel):
    """State for a single sync operation"""
    connector_id: str
    context_id: str
    last_sync_timestamp: datetime
    last_successful_sync: datetime
    processed_memory_ids: Set[str]
    failed_memory_ids: Dict[str, str]  # memory_id -> error_message
    total_processed: int
    total_failed: int
    status: Literal["idle", "running", "completed", "failed"]


class StateManager:
    """Manages ingestion state persistence"""

    def __init__(self, storage_path: Path, backend: str = "json"):
        """
        Args:
            storage_path: Path to state file/directory
            backend: "json" or "sqlite"
        """

    async def get_state(self, connector_id: str, context_id: str) -> SyncState:
        """Retrieve current state for a connector/context"""

    async def update_state(self, state: SyncState):
        """Update state atomically"""

    async def mark_processed(
        self,
        connector_id: str,
        context_id: str,
        memory_id: str
    ):
        """Mark a memory as successfully processed"""

    async def mark_failed(
        self,
        connector_id: str,
        context_id: str,
        memory_id: str,
        error: str
    ):
        """Mark a memory as failed"""

    async def is_processed(
        self,
        connector_id: str,
        context_id: str,
        memory_id: str
    ) -> bool:
        """Check if memory has been processed"""

    async def get_unprocessed_ids(
        self,
        connector_id: str,
        context_id: str,
        all_memory_ids: List[str]
    ) -> List[str]:
        """Filter out already-processed memory IDs"""

    async def get_sync_history(
        self,
        connector_id: str,
        limit: int = 10
    ) -> List[SyncState]:
        """Retrieve sync history"""

    async def reset_state(
        self,
        connector_id: str,
        context_id: str
    ):
        """Reset state for re-sync"""
```

**State Storage Format (JSON)**:
```json
{
  "connectors": {
    "memory-connector-1": {
      "contexts": {
        "CTX123": {
          "last_sync_timestamp": "2025-12-18T15:00:00Z",
          "last_successful_sync": "2025-12-18T15:00:00Z",
          "processed_memory_ids": [
            "mem_001", "mem_002", "mem_003"
          ],
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

### 4. Configuration Manager

**File**: `lightrag/connectors/config_manager.py`

**Responsibilities**:
- Load and validate configuration
- Support multiple config sources (file, env, CLI)
- Hot-reload for non-critical settings
- Schema validation

**Configuration Schema**:

```yaml
# config.yaml

# LightRAG connection settings
lightrag:
  mode: "api"  # "api" or "direct"

  # API mode settings
  api:
    url: "http://localhost:9621"
    api_key: "${LIGHTRAG_API_KEY}"
    workspace: "memories"

  # Direct mode settings (if mode: "direct")
  direct:
    working_dir: "./lightrag_storage"
    # LLM/embedding settings inherited from environment

# Memory API settings
memory_api:
  url: "http://127.0.0.1:8080"
  api_key: "${MEMORY_API_KEY}"
  timeout: 30
  max_retries: 3

# Connector definitions
connectors:
  - id: "personal-memories"
    enabled: true
    context_id: "CTX123"

    # Scheduling
    schedule:
      type: "interval"  # "interval" or "cron"
      interval_hours: 1  # for interval type
      # cron: "0 */1 * * *"  # for cron type

    # Ingestion settings
    ingestion:
      query_range: "week"  # how far back to query
      query_limit: 100     # max items per query
      batch_size: 10       # memories to process in parallel

    # Transformation settings
    transformation:
      strategy: "standard"  # "standard" or "rich"
      include_audio: false  # download and process audio
      include_image: false  # download and process images
      geocoding: false      # reverse geocode locations

    # Retry settings
    retry:
      max_attempts: 3
      backoff_multiplier: 2.0
      max_backoff_seconds: 60

  - id: "work-memories"
    enabled: false
    context_id: "CTX456"
    schedule:
      type: "cron"
      cron: "0 9,17 * * 1-5"  # 9am and 5pm on weekdays

# State management
state:
  backend: "json"  # "json" or "sqlite"
  path: "./memory_sync_state.json"
  # For SQLite:
  # path: "./memory_sync_state.db"

# API server (for management interface)
api:
  host: "0.0.0.0"
  port: 9622
  enable_auth: true
  api_key: "${CONNECTOR_API_KEY}"

# Logging
logging:
  level: "INFO"
  format: "json"  # "json" or "text"
  file: "./memory_connector.log"
```

### 5. Scheduler Service

**File**: `lightrag/connectors/scheduler_service.py`

**Responsibilities**:
- Manage periodic ingestion jobs
- Support multiple schedules
- Job control (pause, resume, trigger)
- Monitor job health

**Key Classes**:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger


class SchedulerService:
    """Manages scheduled ingestion jobs"""

    def __init__(
        self,
        orchestrator: IngestionOrchestrator,
        config_manager: ConfigManager
    ):
        self.scheduler = AsyncIOScheduler()
        self.orchestrator = orchestrator
        self.config = config_manager

    async def start(self):
        """Start scheduler and register jobs"""
        for connector_config in self.config.get_connectors():
            if connector_config.enabled:
                self.add_job(connector_config)

        self.scheduler.start()

    async def stop(self):
        """Gracefully stop all jobs"""
        self.scheduler.shutdown(wait=True)

    def add_job(self, connector_config: ConnectorConfig):
        """Add a scheduled job for a connector"""
        if connector_config.schedule.type == "interval":
            trigger = IntervalTrigger(
                hours=connector_config.schedule.interval_hours
            )
        elif connector_config.schedule.type == "cron":
            trigger = CronTrigger.from_crontab(
                connector_config.schedule.cron
            )

        self.scheduler.add_job(
            func=self._run_ingestion_job,
            trigger=trigger,
            args=[connector_config.id],
            id=f"connector_{connector_config.id}",
            name=f"Ingestion: {connector_config.id}",
            replace_existing=True
        )

    async def _run_ingestion_job(self, connector_id: str):
        """Execute ingestion for a connector"""
        try:
            await self.orchestrator.run_ingestion(connector_id)
        except Exception as e:
            logger.error(
                f"Ingestion job failed for {connector_id}: {e}"
            )

    def trigger_now(self, connector_id: str):
        """Manually trigger a job immediately"""
        self.scheduler.modify_job(
            f"connector_{connector_id}",
            next_run_time=datetime.now()
        )

    def pause_job(self, connector_id: str):
        """Pause a scheduled job"""
        self.scheduler.pause_job(f"connector_{connector_id}")

    def resume_job(self, connector_id: str):
        """Resume a paused job"""
        self.scheduler.resume_job(f"connector_{connector_id}")

    def get_job_status(self, connector_id: str) -> dict:
        """Get job status and next run time"""
        job = self.scheduler.get_job(f"connector_{connector_id}")
        if job:
            return {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time,
                "paused": job.next_run_time is None
            }
        return None
```

### 6. Ingestion Orchestrator

**File**: `lightrag/connectors/ingestion_orchestrator.py`

**Responsibilities**:
- Coordinate the complete ingestion pipeline
- Implement ingestion strategies (full, incremental, selective)
- Error handling and retry logic
- Progress reporting

**Key Classes**:

```python
class IngestionOrchestrator:
    """Orchestrates the complete ingestion pipeline"""

    def __init__(
        self,
        config_manager: ConfigManager,
        state_manager: StateManager,
        memory_client_factory: callable,
        lightrag_client_factory: callable,
        transformer_factory: callable
    ):
        self.config = config_manager
        self.state = state_manager
        self.memory_client_factory = memory_client_factory
        self.lightrag_client_factory = lightrag_client_factory
        self.transformer_factory = transformer_factory

    async def run_ingestion(
        self,
        connector_id: str,
        strategy: str = "incremental"
    ) -> IngestionReport:
        """
        Run ingestion for a connector

        Args:
            connector_id: ID of connector to run
            strategy: "full", "incremental", or "selective"

        Returns:
            IngestionReport with results
        """

        # 1. Load connector config
        config = self.config.get_connector(connector_id)

        # 2. Initialize clients
        memory_client = self.memory_client_factory(config)
        lightrag_client = self.lightrag_client_factory(config)
        transformer = self.transformer_factory(config)

        # 3. Get current state
        state = await self.state.get_state(
            connector_id,
            config.context_id
        )

        # 4. Update state: mark as running
        state.status = "running"
        await self.state.update_state(state)

        report = IngestionReport(
            connector_id=connector_id,
            start_time=datetime.now(timezone.utc)
        )

        try:
            # 5. Fetch memories from API
            memories = await memory_client.get_memories(
                ctx_id=config.context_id,
                limit=config.ingestion.query_limit,
                range=config.ingestion.query_range
            )

            # 6. Filter unprocessed
            all_memory_ids = [m.id for m in memories.memories]
            unprocessed_ids = await self.state.get_unprocessed_ids(
                connector_id,
                config.context_id,
                all_memory_ids
            )

            memories_to_process = [
                m for m in memories.memories if m.id in unprocessed_ids
            ]

            report.total_fetched = len(memories.memories)
            report.total_to_process = len(memories_to_process)

            # 7. Process in batches
            for i in range(0, len(memories_to_process), config.ingestion.batch_size):
                batch = memories_to_process[i:i + config.ingestion.batch_size]

                await self._process_batch(
                    batch=batch,
                    connector_id=connector_id,
                    context_id=config.context_id,
                    transformer=transformer,
                    lightrag_client=lightrag_client,
                    report=report
                )

            # 8. Update state: mark as completed
            state.status = "completed"
            state.last_successful_sync = datetime.now(timezone.utc)
            await self.state.update_state(state)

            report.status = "success"

        except Exception as e:
            # 9. Update state: mark as failed
            state.status = "failed"
            await self.state.update_state(state)

            report.status = "failed"
            report.error = str(e)
            logger.error(f"Ingestion failed: {e}", exc_info=True)

        finally:
            report.end_time = datetime.now(timezone.utc)

        return report

    async def _process_batch(
        self,
        batch: List[Memory],
        connector_id: str,
        context_id: str,
        transformer: MemoryTransformer,
        lightrag_client: LightRAGClient,
        report: IngestionReport
    ):
        """Process a batch of memories in parallel"""

        async def process_single(memory: Memory):
            try:
                # Transform
                document = await transformer.transform(memory)

                # Submit to LightRAG
                await lightrag_client.insert_document(
                    text=document,
                    file_source=f"memory://{context_id}/{memory.id}"
                )

                # Mark as processed
                await self.state.mark_processed(
                    connector_id,
                    context_id,
                    memory.id
                )

                report.successful_count += 1

            except Exception as e:
                # Mark as failed
                await self.state.mark_failed(
                    connector_id,
                    context_id,
                    memory.id,
                    str(e)
                )

                report.failed_count += 1
                report.errors.append({
                    "memory_id": memory.id,
                    "error": str(e)
                })

        # Process batch concurrently
        await asyncio.gather(*[process_single(m) for m in batch])
```

### 7. LightRAG Client

**File**: `lightrag/connectors/lightrag_client.py`

**Responsibilities**:
- Abstract LightRAG integration (API or Direct)
- Handle authentication
- Submit documents
- Track status

**Key Classes**:

```python
class LightRAGClient(ABC):
    """Abstract base for LightRAG integration"""

    @abstractmethod
    async def insert_document(
        self,
        text: str,
        file_source: Optional[str] = None
    ) -> str:
        """Insert document, return document ID"""


class LightRAGAPIClient(LightRAGClient):
    """LightRAG API integration"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        workspace: str = "default"
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.workspace = workspace
        self.client = httpx.AsyncClient()

    async def insert_document(
        self,
        text: str,
        file_source: Optional[str] = None
    ) -> str:
        response = await self.client.post(
            f"{self.api_url}/documents/text",
            json={
                "text": text,
                "file_source": file_source or "memory_api"
            },
            headers={
                "X-API-Key": self.api_key,
                "LIGHTRAG-WORKSPACE": self.workspace
            }
        )
        response.raise_for_status()
        return response.json()["doc_id"]


class LightRAGDirectClient(LightRAGClient):
    """Direct LightRAG library integration"""

    def __init__(
        self,
        working_dir: str,
        **lightrag_kwargs
    ):
        from lightrag import LightRAG
        self.rag = LightRAG(working_dir=working_dir, **lightrag_kwargs)

    async def initialize(self):
        await self.rag.initialize_storages()

    async def insert_document(
        self,
        text: str,
        file_source: Optional[str] = None
    ) -> str:
        doc_id = await self.rag.ainsert(
            input=text,
            file_paths=[file_source] if file_source else None
        )
        return doc_id
```

## File Structure

```
lightrag/
├── connectors/
│   ├── __init__.py
│   ├── memory_api_client.py       # Memory API client
│   ├── memory_transformer.py      # Data transformation
│   ├── state_manager.py           # State persistence
│   ├── config_manager.py          # Configuration
│   ├── scheduler_service.py       # Job scheduling
│   ├── ingestion_orchestrator.py  # Pipeline orchestration
│   ├── lightrag_client.py         # LightRAG integration
│   ├── connector_api.py           # Management REST API
│   ├── models.py                  # Pydantic models
│   └── cli.py                     # CLI interface
├── api/
│   └── (existing LightRAG API files)
└── (existing LightRAG files)

# Executable entry point
memory_connector/
├── __init__.py
└── __main__.py  # CLI entry point
```

## Data Models

**File**: `lightrag/connectors/models.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# Memory API Models (from OpenAPI spec)
class Memory(BaseModel):
    id: str
    type: str = "record"
    audio: bool
    image: bool
    transcript: str = ""
    location_lat: float = 0.0
    location_lon: float = 0.0
    created_at: str


class MemoryList(BaseModel):
    memories: List[Memory]


# Connector Configuration Models
class ScheduleConfig(BaseModel):
    type: Literal["interval", "cron"]
    interval_hours: Optional[int] = None
    cron: Optional[str] = None


class IngestionConfig(BaseModel):
    query_range: str = "week"
    query_limit: int = 100
    batch_size: int = 10


class TransformationConfig(BaseModel):
    strategy: Literal["standard", "rich"] = "standard"
    include_audio: bool = False
    include_image: bool = False
    geocoding: bool = False


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 60


class ConnectorConfig(BaseModel):
    id: str
    enabled: bool = True
    context_id: str
    schedule: ScheduleConfig
    ingestion: IngestionConfig
    transformation: TransformationConfig
    retry: RetryConfig


# Ingestion Report Models
class IngestionReport(BaseModel):
    connector_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: Literal["running", "success", "failed"] = "running"
    total_fetched: int = 0
    total_to_process: int = 0
    successful_count: int = 0
    failed_count: int = 0
    errors: List[dict] = Field(default_factory=list)
    error: Optional[str] = None
```

## Next Steps

See `02-IMPLEMENTATION-PLAN.md` for the detailed implementation roadmap.
