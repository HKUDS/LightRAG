# FR01: Memory API Ingestion - Future Enhancements

## Phase 2: Memory Manager

The current implementation (Phase 1) focuses on **unidirectional ingestion** from Memory API → LightRAG. Phase 2 will expand this into a full **Memory Manager** with bidirectional sync, backup/restore, and collaborative features.

## Vision: Complete Memory Management Platform

```
┌────────────────────────────────────────────────────────────────────┐
│                      Memory Manager                                │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Ingestion   │  │  Export/     │  │  Replication │            │
│  │  Engine      │  │  Backup      │  │  & Sharing   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│         │                  │                  │                    │
└─────────┼──────────────────┼──────────────────┼────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │  Memory API   │  │  Local Files  │  │  Remote RAG   │
  │  (Source)     │  │  (Backup)     │  │  (Replica)    │
  └───────────────┘  └───────────────┘  └───────────────┘
          ▲                  │                  │
          │                  │                  │
          └──────────────────┴──────────────────┘
                  Bidirectional Sync
```

## Enhanced Features Roadmap

### Feature 1: Bidirectional Sync

**Goal**: Sync changes from LightRAG back to Memory API

**Use Cases**:
- User edits transcript in LightRAG UI → update Memory API
- User adds annotations/tags in RAG → sync to Memory API
- User deletes memory from RAG → optionally delete from Memory API

**Implementation**:

```python
class BidirectionalSyncManager:
    """Manages two-way sync between Memory API and LightRAG"""

    async def sync_to_memory_api(
        self,
        doc_id: str,
        updates: dict
    ):
        """
        Sync LightRAG changes back to Memory API

        Args:
            doc_id: LightRAG document ID (mapped to memory_id)
            updates: Changes to sync (transcript, tags, etc.)
        """

        # 1. Look up original memory_id from doc_id
        memory_id = await self.mapping_store.get_memory_id(doc_id)

        # 2. Determine what changed
        if "transcript" in updates:
            # Update transcript via PUT /memory/{ctx_id}/{memory_id}
            await self.memory_client.update_memory(
                ctx_id=self.ctx_id,
                memory_id=memory_id,
                transcript=updates["transcript"]
            )

        # 3. Record sync event
        await self.state.record_bidirectional_sync(
            doc_id=doc_id,
            memory_id=memory_id,
            direction="rag_to_api",
            updates=updates
        )

    async def detect_changes(self):
        """Detect changes in LightRAG documents"""

        # Monitor doc_status storage for updates
        # Compare last_modified timestamp
        # Identify changed documents
        pass
```

**Challenges**:
- Change detection in LightRAG (no built-in change tracking)
- Conflict resolution (simultaneous edits)
- Mapping document IDs ↔ memory IDs

**Solution**:
- Add metadata to LightRAG documents with memory_id
- Implement change log/audit trail
- Use last-write-wins or manual conflict resolution

### Feature 2: Audio/Image Processing

**Goal**: Process multimedia content from memories

**Use Cases**:
- Transcribe audio if transcript is missing
- Extract text from images (OCR)
- Analyze images with vision models
- Generate summaries of audio content

**Implementation**:

```python
class MultimodalProcessor:
    """Processes audio and image content from memories"""

    async def process_audio(
        self,
        ctx_id: str,
        memory_id: str,
        audio_path: Path
    ) -> str:
        """
        Transcribe audio to text

        Uses: OpenAI Whisper, AssemblyAI, or local Whisper model
        """
        if self.transcription_service == "openai":
            with open(audio_path, "rb") as f:
                transcript = await openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            return transcript.text

        elif self.transcription_service == "local":
            # Use local Whisper model
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path))
            return result["text"]

    async def process_image(
        self,
        ctx_id: str,
        memory_id: str,
        image_path: Path
    ) -> dict:
        """
        Analyze image with vision model

        Uses: GPT-4 Vision, Claude Vision, or LLaVA
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Use GPT-4 Vision
        response = await openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Extract any visible text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )

        return {
            "description": response.choices[0].message.content,
            "ocr_text": self.extract_text_from_description(
                response.choices[0].message.content
            )
        }
```

**Configuration**:
```yaml
connectors:
  - id: "personal-memories"
    transformation:
      strategy: "rich"
      include_audio: true
      include_image: true
      audio_processing:
        enabled: true
        service: "openai"  # openai, assemblyai, local
        model: "whisper-1"
      image_processing:
        enabled: true
        service: "openai"  # openai, anthropic, local
        model: "gpt-4-vision-preview"
        extract_ocr: true
```

### Feature 3: Export & Backup

**Goal**: Export knowledge graph and memory data for backup/archival

**Use Cases**:
- Periodic backup of entire knowledge graph
- Export to portable format (JSON, CSV, RDF)
- Archive old memories to cold storage
- Migrate to different LightRAG instance

**Implementation**:

```python
class MemoryExporter:
    """Exports memory data and knowledge graph"""

    async def export_full_backup(
        self,
        output_path: Path,
        format: str = "json"
    ):
        """
        Export complete backup

        Includes:
        - All memory items from Memory API
        - LightRAG knowledge graph
        - State and metadata
        """

        backup_data = {
            "version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "memories": [],
            "knowledge_graph": {},
            "state": {}
        }

        # 1. Export all memories from Memory API
        for ctx_id in self.context_ids:
            memories = await self.memory_client.get_memories(
                ctx_id=ctx_id,
                limit=10000,
                range="all"
            )
            backup_data["memories"].extend([
                {
                    "context_id": ctx_id,
                    **memory.model_dump()
                }
                for memory in memories.memories
            ])

        # 2. Export LightRAG knowledge graph
        # Export from storage backends
        backup_data["knowledge_graph"] = await self.export_kg()

        # 3. Export state
        backup_data["state"] = await self.state_manager.export_all()

        # 4. Write to file
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(backup_data, f, indent=2)

        elif format == "archive":
            # Create tar.gz with all data + media files
            self.create_archive(backup_data, output_path)

    async def export_kg(self) -> dict:
        """Export knowledge graph from LightRAG"""

        # Read from storage backends
        kg_data = {
            "entities": [],
            "relations": [],
            "chunks": []
        }

        # Export entities
        if hasattr(self.rag, "graph_storage"):
            entities = await self.rag.graph_storage.get_all_nodes()
            kg_data["entities"] = entities

        # Export relations
        if hasattr(self.rag, "graph_storage"):
            relations = await self.rag.graph_storage.get_all_edges()
            kg_data["relations"] = relations

        # Export chunks
        if hasattr(self.rag, "kv_storage"):
            chunks = await self.rag.kv_storage.get_all()
            kg_data["chunks"] = chunks

        return kg_data

    async def restore_from_backup(
        self,
        backup_path: Path
    ):
        """Restore from backup file"""

        with open(backup_path) as f:
            backup_data = json.load(f)

        # 1. Restore memories to Memory API (optional)
        if self.restore_to_memory_api:
            for memory_data in backup_data["memories"]:
                await self.memory_client.upload_memory(...)

        # 2. Restore to LightRAG
        for memory_data in backup_data["memories"]:
            memory = Memory(**memory_data)
            document = self.transformer.transform(memory)
            await self.lightrag_client.insert_document(document)

        # 3. Restore state
        await self.state_manager.import_state(backup_data["state"])
```

**CLI Commands**:
```bash
# Export backup
memory-connector export \
  --config config.yaml \
  --output backup_2025-12-18.json \
  --format json

# Restore from backup
memory-connector restore \
  --config config.yaml \
  --input backup_2025-12-18.json \
  --target lightrag  # or "memory_api" or "both"
```

### Feature 4: Replication & Sharing

**Goal**: Replicate knowledge graph to other instances for collaboration

**Use Cases**:
- Share knowledge graph with team members
- Replicate to multiple LightRAG instances
- Collaborative knowledge building
- Multi-region deployment

**Implementation**:

```python
class ReplicationManager:
    """Manages replication to multiple LightRAG instances"""

    def __init__(self, config: ReplicationConfig):
        self.replicas = [
            LightRAGAPIClient(**replica_config)
            for replica_config in config.replicas
        ]

    async def replicate(
        self,
        memory: Memory,
        strategy: str = "all"
    ):
        """
        Replicate memory to all replicas

        Args:
            memory: Memory to replicate
            strategy: "all", "nearest", "selective"
        """

        document = self.transformer.transform(memory)

        if strategy == "all":
            # Replicate to all instances
            tasks = [
                replica.insert_document(document)
                for replica in self.replicas
            ]
            await asyncio.gather(*tasks)

        elif strategy == "nearest":
            # Replicate to geographically nearest instance
            replica = await self.select_nearest_replica(memory.location)
            await replica.insert_document(document)

        elif strategy == "selective":
            # Replicate based on rules (e.g., tags, type)
            replicas = self.select_replicas_by_rules(memory)
            tasks = [
                replica.insert_document(document)
                for replica in replicas
            ]
            await asyncio.gather(*tasks)

    async def sync_replicas(self):
        """Sync all replicas to ensure consistency"""

        # 1. Get state from primary
        primary_state = await self.get_primary_state()

        # 2. Compare with replicas
        for replica in self.replicas:
            replica_state = await self.get_replica_state(replica)

            # 3. Find differences
            missing_docs = primary_state - replica_state

            # 4. Replicate missing docs
            for doc_id in missing_docs:
                document = await self.get_document(doc_id)
                await replica.insert_document(document)
```

**Configuration**:
```yaml
replication:
  enabled: true
  strategy: "all"  # all, nearest, selective

  replicas:
    - id: "team-instance"
      url: "http://team-lightrag:9621"
      api_key: "${TEAM_LIGHTRAG_KEY}"
      workspace: "shared-memories"

    - id: "backup-instance"
      url: "http://backup-lightrag:9621"
      api_key: "${BACKUP_LIGHTRAG_KEY}"
      workspace: "backup"

  sync:
    enabled: true
    interval_hours: 6  # Sync every 6 hours
```

### Feature 5: Memory Timeline & Analytics

**Goal**: Visualize and analyze memory data over time

**Use Cases**:
- View memory timeline
- Analyze patterns (most active times, locations)
- Trend analysis (topic shifts over time)
- Memory heatmap (geographic distribution)

**Implementation**:

```python
class MemoryAnalytics:
    """Analytics and visualization for memory data"""

    async def get_timeline(
        self,
        ctx_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "day"
    ) -> dict:
        """Get memory timeline data"""

        memories = await self.get_memories_in_range(
            ctx_id, start_date, end_date
        )

        # Group by time bucket
        timeline = {}
        for memory in memories:
            bucket = self.get_time_bucket(
                memory.created_at, granularity
            )
            if bucket not in timeline:
                timeline[bucket] = []
            timeline[bucket].append(memory)

        return {
            "granularity": granularity,
            "buckets": [
                {
                    "timestamp": bucket,
                    "count": len(items),
                    "memories": items
                }
                for bucket, items in sorted(timeline.items())
            ]
        }

    async def get_location_heatmap(
        self,
        ctx_id: str
    ) -> dict:
        """Generate location heatmap data"""

        memories = await self.memory_client.get_memories(
            ctx_id=ctx_id,
            limit=10000,
            range="all"
        )

        locations = [
            {
                "lat": m.location_lat,
                "lon": m.location_lon,
                "timestamp": m.created_at,
                "id": m.id
            }
            for m in memories.memories
            if m.location_lat != 0.0 or m.location_lon != 0.0
        ]

        # Cluster nearby locations
        clusters = self.cluster_locations(locations)

        return {
            "locations": locations,
            "clusters": clusters
        }

    async def get_topic_trends(
        self,
        ctx_id: str,
        time_range: str = "month"
    ) -> dict:
        """Analyze topic trends over time"""

        # Query LightRAG knowledge graph for entities
        entities = await self.get_entities_over_time(ctx_id, time_range)

        # Group by time period and count frequency
        trends = self.calculate_trends(entities)

        return {
            "time_range": time_range,
            "topics": trends
        }
```

**API Endpoints**:
```python
@router.get("/analytics/timeline")
async def get_timeline(
    ctx_id: str,
    start_date: datetime,
    end_date: datetime,
    granularity: str = "day"
):
    """Get memory timeline"""
    return await analytics.get_timeline(...)

@router.get("/analytics/heatmap")
async def get_heatmap(ctx_id: str):
    """Get location heatmap"""
    return await analytics.get_location_heatmap(ctx_id)

@router.get("/analytics/trends")
async def get_trends(ctx_id: str, range: str = "month"):
    """Get topic trends"""
    return await analytics.get_topic_trends(ctx_id, range)
```

### Feature 6: Smart Filtering & Search

**Goal**: Advanced filtering and search capabilities

**Features**:
- Filter by location (radius search)
- Filter by date range (custom ranges)
- Filter by keywords/tags
- Filter by memory type
- Full-text search in transcripts
- Semantic search (using embeddings)

**Implementation**:

```python
class MemorySearch:
    """Advanced search and filtering"""

    async def search(
        self,
        ctx_id: str,
        query: SearchQuery
    ) -> List[Memory]:
        """
        Search memories with advanced filters

        Args:
            ctx_id: Context ID
            query: Search query with filters
        """

        # 1. Fetch base set
        memories = await self.memory_client.get_memories(
            ctx_id=ctx_id,
            limit=query.limit or 1000,
            range=query.time_range or "all"
        )

        # 2. Apply filters
        filtered = memories.memories

        if query.keywords:
            filtered = self.filter_by_keywords(filtered, query.keywords)

        if query.location:
            filtered = self.filter_by_location(
                filtered,
                query.location.lat,
                query.location.lon,
                query.location.radius_km
            )

        if query.date_range:
            filtered = self.filter_by_date_range(
                filtered,
                query.date_range.start,
                query.date_range.end
            )

        if query.memory_type:
            filtered = [m for m in filtered if m.type == query.memory_type]

        # 3. Sort results
        if query.sort_by == "relevance":
            filtered = await self.rank_by_relevance(filtered, query.keywords)
        elif query.sort_by == "date":
            filtered = sorted(filtered, key=lambda m: m.created_at, reverse=True)

        return filtered[:query.limit or 100]

    def filter_by_location(
        self,
        memories: List[Memory],
        lat: float,
        lon: float,
        radius_km: float
    ) -> List[Memory]:
        """Filter memories within radius of location"""

        def distance(m: Memory) -> float:
            # Haversine distance
            from math import radians, sin, cos, sqrt, atan2

            lat1, lon1 = radians(m.location_lat), radians(m.location_lon)
            lat2, lon2 = radians(lat), radians(lon)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))

            return 6371 * c  # Earth radius in km

        return [m for m in memories if distance(m) <= radius_km]
```

### Feature 7: Webhooks & Notifications

**Goal**: Real-time notifications for memory events

**Events**:
- New memory ingested
- Ingestion job completed
- Ingestion job failed
- Duplicate detected
- Interesting entity discovered

**Implementation**:

```python
class WebhookManager:
    """Manages webhooks and notifications"""

    async def send_webhook(
        self,
        event_type: str,
        payload: dict
    ):
        """Send webhook notification"""

        for webhook in self.webhooks:
            if event_type in webhook.events:
                await self.http_client.post(
                    webhook.url,
                    json={
                        "event": event_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": payload
                    },
                    headers={
                        "X-Webhook-Secret": webhook.secret
                    }
                )

    async def on_memory_ingested(self, memory: Memory, doc_id: str):
        """Called when memory is successfully ingested"""
        await self.send_webhook(
            "memory.ingested",
            {
                "memory_id": memory.id,
                "doc_id": doc_id,
                "transcript_preview": memory.transcript[:100]
            }
        )

    async def on_job_completed(self, report: IngestionReport):
        """Called when ingestion job completes"""
        await self.send_webhook(
            "job.completed",
            report.model_dump()
        )
```

**Configuration**:
```yaml
webhooks:
  - url: "https://hooks.slack.com/services/..."
    secret: "${WEBHOOK_SECRET}"
    events:
      - memory.ingested
      - job.completed
      - job.failed
```

## Technology Upgrades

### 1. Web UI Dashboard

Build a React-based dashboard for managing connectors:

**Features**:
- Visual connector configuration
- Real-time sync status
- Memory timeline visualization
- Location heatmap
- Search and filter interface
- Settings management

**Tech Stack**:
- React + TypeScript
- TanStack Query (data fetching)
- Recharts (charts/graphs)
- Mapbox (location visualization)
- Tailwind CSS (styling)

### 2. Mobile App Integration

Mobile app for on-the-go memory management:

**Features**:
- View memories
- Trigger sync
- Monitor status
- Receive notifications
- Search memories

**Tech Stack**:
- React Native
- Expo
- Push notifications

### 3. Database Upgrade

Migrate from JSON/SQLite to production database:

**Options**:
- PostgreSQL (relational, JSONB support)
- MongoDB (document store)
- Redis (caching layer)

**Benefits**:
- Better performance at scale
- Advanced querying
- Concurrent access
- Replication support

## Enterprise Features

### 1. Multi-Tenant Support

Support multiple users/organizations:

```python
class TenantManager:
    """Manages multi-tenant isolation"""

    async def create_tenant(self, tenant_id: str, config: dict):
        """Create new tenant"""
        pass

    async def get_connectors(self, tenant_id: str):
        """Get connectors for tenant"""
        pass
```

### 2. Role-Based Access Control (RBAC)

Different permission levels:

- **Admin**: Full access
- **Manager**: Configure connectors
- **Viewer**: Read-only access

### 3. Audit Logging

Track all operations:

- Who did what, when
- Changes to configuration
- Data access logs
- Compliance reporting

### 4. High Availability

- Active-active deployment
- Load balancing
- Automatic failover
- Data replication

## Implementation Priority

**Priority 1 (Next 3 months)**:
1. Bidirectional sync
2. Export/backup functionality
3. Basic analytics (timeline, stats)

**Priority 2 (3-6 months)**:
4. Audio/image processing
5. Web UI dashboard
6. Advanced search

**Priority 3 (6-12 months)**:
7. Replication & sharing
8. Mobile app
9. Enterprise features

## Conclusion

The Memory Manager will evolve from a simple ingestion tool to a comprehensive memory management platform, enabling users to:

- Seamlessly sync memories between systems
- Backup and archive their data
- Collaborate and share knowledge
- Analyze patterns and insights
- Scale to enterprise use cases

This roadmap ensures the tool grows with user needs while maintaining simplicity and reliability.
