# FR01: Memory API Ingestion - API Integration Details

## Memory API Integration

### API Specification Summary

Based on your OpenAPI spec:

**Base URL**: `http://127.0.0.1:8080`
**Authentication**: API Key in header (`X-API-KEY`)

### Relevant Endpoints

#### 1. Get Memory List

**Endpoint**: `GET /memory/{ctx_id}`

**Description**: Retrieve list of memory items from given context

**Parameters**:
- `ctx_id` (path, required): Context ID
- `limit` (query, optional): Limit of memory items (default: 10)
- `range` (query, optional): Range of memory items (default: "week")

**Response** (200):
```json
{
  "memories": [
    {
      "id": "1234567890",
      "type": "record",
      "audio": true,
      "image": true,
      "transcript": "",
      "location_lat": 0.8008282,
      "location_lon": 6.0274563,
      "created_at": "2025-12-18T14:30:00Z"
    }
  ]
}
```

**Error Responses**:
- 401: Unauthorized (invalid API key)
- 403: Forbidden
- 404: Context not found
- 500: Internal error

#### 2. Get Audio for Memory

**Endpoint**: `GET /memory/{ctx_id}/{memory_id}/audio`

**Description**: Retrieve audio resource for a given memory

**Parameters**:
- `ctx_id` (path, required): Context ID
- `memory_id` (path, required): Memory ID

**Response** (200):
- Content-Type: `audio/mpeg`, `audio/ogg`, or `audio/webm`
- Body: Binary audio data

**Use Case**: Optional - can download and transcribe if transcript is empty

#### 3. Get Image for Memory

**Endpoint**: `GET /memory/{ctx_id}/{memory_id}/image`

**Description**: Retrieve image resource for a given memory

**Parameters**:
- `ctx_id` (path, required): Context ID
- `memory_id` (path, required): Memory ID

**Response** (200):
- Content-Type: `image/jpeg` or `image/png`
- Body: Binary image data

**Use Case**: Optional - can download and process with vision models

### Integration Strategy

#### Phase 1: Transcript-Only Ingestion

**Scope**: Ingest only transcript data and metadata

**Implementation**:
```python
# Fetch memories
response = await client.get(
    f"/memory/{ctx_id}",
    params={"limit": 100, "range": "week"},
    headers={"X-API-KEY": api_key}
)

# Process each memory
for memory in response.json()["memories"]:
    if memory["transcript"]:  # Only if transcript exists
        document = transform_memory_to_document(memory)
        await lightrag.insert(document)
```

**Advantages**:
- Simple and fast
- No additional API calls
- Lower bandwidth usage

**Limitations**:
- Misses memories without transcripts
- No audio/image content

#### Phase 2: Enhanced Ingestion (Future)

**Scope**: Download and process audio/images

**Implementation**:
```python
for memory in memories:
    # Get transcript (if missing, transcribe audio)
    if not memory["transcript"] and memory["audio"]:
        audio_data = await client.get(
            f"/memory/{ctx_id}/{memory_id}/audio"
        )
        transcript = await transcribe_audio(audio_data)
        memory["transcript"] = transcript

    # Get image analysis (if available)
    if memory["image"]:
        image_data = await client.get(
            f"/memory/{ctx_id}/{memory_id}/image"
        )
        image_description = await analyze_image(image_data)
        memory["image_description"] = image_description

    document = transform_memory_to_document(memory)
    await lightrag.insert(document)
```

## Memory API Client Implementation

### Core Client Class

```python
import httpx
from typing import Optional
from lightrag.connectors.models import Memory, MemoryList


class MemoryAPIClient:
    """Client for interacting with Memory API"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"X-API-KEY": api_key}
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def get_memories(
        self,
        ctx_id: str,
        limit: int = 100,
        range: str = "week"
    ) -> MemoryList:
        """
        Fetch list of memories for a context

        Args:
            ctx_id: Context ID
            limit: Maximum number of memories to fetch
            range: Time range (e.g., "week", "month", "day")

        Returns:
            MemoryList with memory items

        Raises:
            httpx.HTTPStatusError: On API errors (401, 403, 404, 500)
            httpx.TimeoutException: On timeout
        """
        url = f"{self.api_url}/memory/{ctx_id}"
        params = {"limit": limit, "range": range}

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                return MemoryList(**data)

            except httpx.TimeoutException as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except httpx.HTTPStatusError as e:
                # Don't retry on auth errors or client errors
                if e.response.status_code in [401, 403, 404]:
                    raise
                # Retry on server errors
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def get_memory(
        self,
        ctx_id: str,
        memory_id: str
    ) -> Memory:
        """
        Fetch single memory item

        Note: Current API spec doesn't have a single-item endpoint,
        so we fetch the list and filter. This can be optimized if
        the API adds a GET /memory/{ctx_id}/{memory_id} endpoint.

        Args:
            ctx_id: Context ID
            memory_id: Memory ID

        Returns:
            Memory item

        Raises:
            ValueError: If memory not found
        """
        memories = await self.get_memories(ctx_id, limit=1000)
        for memory in memories.memories:
            if memory.id == memory_id:
                return memory
        raise ValueError(f"Memory {memory_id} not found in context {ctx_id}")

    async def download_audio(
        self,
        ctx_id: str,
        memory_id: str,
        output_path: Path
    ) -> bool:
        """
        Download audio file for a memory

        Args:
            ctx_id: Context ID
            memory_id: Memory ID
            output_path: Path to save audio file

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.api_url}/memory/{ctx_id}/{memory_id}/audio"

        try:
            response = await self.client.get(url)

            if response.status_code == 404:
                return False  # Audio not available

            response.raise_for_status()

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(response.content)

            return True

        except httpx.HTTPStatusError:
            return False

    async def download_image(
        self,
        ctx_id: str,
        memory_id: str,
        output_path: Path
    ) -> bool:
        """
        Download image file for a memory

        Args:
            ctx_id: Context ID
            memory_id: Memory ID
            output_path: Path to save image file

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.api_url}/memory/{ctx_id}/{memory_id}/image"

        try:
            response = await self.client.get(url)

            if response.status_code == 404:
                return False  # Image not available

            response.raise_for_status()

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(response.content)

            return True

        except httpx.HTTPStatusError:
            return False

    async def check_connection(self) -> bool:
        """
        Test API connectivity and authentication

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to fetch empty list (minimal request)
            await self.get_memories(ctx_id="test", limit=1)
            return True
        except httpx.HTTPStatusError as e:
            # 404 is OK (context doesn't exist, but auth worked)
            if e.response.status_code == 404:
                return True
            # 401/403 means auth failed
            return False
        except Exception:
            return False
```

## LightRAG Integration

### LightRAG API Client

```python
import httpx
from typing import Optional


class LightRAGAPIClient:
    """Client for LightRAG REST API"""

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        workspace: str = "default"
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.workspace = workspace

        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),  # 5 min for processing
            headers=headers
        )

    async def close(self):
        await self.client.aclose()

    async def insert_document(
        self,
        text: str,
        file_source: Optional[str] = None
    ) -> str:
        """
        Insert document into LightRAG

        Args:
            text: Document text content
            file_source: Optional source identifier

        Returns:
            Document ID
        """
        url = f"{self.api_url}/documents/text"

        payload = {
            "text": text,
            "file_source": file_source or "memory_api"
        }

        headers = {"LIGHTRAG-WORKSPACE": self.workspace}

        response = await self.client.post(
            url,
            json=payload,
            headers=headers
        )

        response.raise_for_status()
        result = response.json()

        return result.get("doc_id") or result.get("track_id")

    async def get_document_status(self, doc_id: str) -> dict:
        """
        Get document processing status

        Args:
            doc_id: Document ID

        Returns:
            Status information
        """
        url = f"{self.api_url}/documents/{doc_id}/status"

        headers = {"LIGHTRAG-WORKSPACE": self.workspace}

        response = await self.client.get(url, headers=headers)
        response.raise_for_status()

        return response.json()
```

### LightRAG Direct Client

```python
from lightrag import LightRAG
from typing import Optional


class LightRAGDirectClient:
    """Direct LightRAG library integration"""

    def __init__(
        self,
        working_dir: str,
        **lightrag_kwargs
    ):
        """
        Initialize direct LightRAG client

        Args:
            working_dir: LightRAG working directory
            **lightrag_kwargs: Additional LightRAG parameters
                - embedding_func
                - llm_model_func
                - etc.
        """
        from lightrag import LightRAG

        self.rag = LightRAG(working_dir=working_dir, **lightrag_kwargs)
        self._initialized = False

    async def initialize(self):
        """Initialize storage (call once at startup)"""
        if not self._initialized:
            await self.rag.initialize_storages()
            self._initialized = True

    async def close(self):
        """Clean up resources"""
        if self._initialized:
            await self.rag.finalize_storages()

    async def insert_document(
        self,
        text: str,
        file_source: Optional[str] = None
    ) -> str:
        """
        Insert document into LightRAG

        Args:
            text: Document text content
            file_source: Optional source identifier

        Returns:
            Document ID
        """
        if not self._initialized:
            await self.initialize()

        # Use ainsert with file_paths for source tracking
        doc_id = await self.rag.ainsert(
            input=text,
            file_paths=[file_source] if file_source else None
        )

        return doc_id
```

## Data Transformation

### Standard Transformation

```python
from lightrag.connectors.models import Memory
from datetime import datetime
import re


class StandardTransformationStrategy:
    """Standard transformation: transcript + metadata as structured text"""

    def transform(self, memory: Memory) -> str:
        """
        Transform Memory to LightRAG document

        Args:
            memory: Memory object from API

        Returns:
            Formatted document text
        """
        # Parse timestamp
        try:
            created_at = datetime.fromisoformat(
                memory.created_at.replace("Z", "+00:00")
            )
            date_str = created_at.strftime("%B %d, %Y at %I:%M %p UTC")
        except Exception:
            date_str = memory.created_at

        # Format location
        location_str = ""
        if memory.location_lat != 0.0 or memory.location_lon != 0.0:
            location_str = f"\nLocation: ({memory.location_lat}, {memory.location_lon})"

        # Media availability
        media_parts = []
        if memory.audio:
            media_parts.append("Audio")
        if memory.image:
            media_parts.append("Image")
        media_str = ", ".join(media_parts) if media_parts else "None"

        # Build document
        document = f"""---
Memory Record
ID: {memory.id}
Type: {memory.type}
Recorded: {date_str}{location_str}
Media Available: {media_str}

Transcript:
{memory.transcript if memory.transcript else "[No transcript available]"}
---"""

        return document

    def extract_tags(self, text: str) -> list[str]:
        """
        Extract hashtags from text

        Args:
            text: Input text

        Returns:
            List of hashtags
        """
        return re.findall(r'#\w+', text)
```

### Rich Transformation (Future)

```python
class RichTransformationStrategy:
    """
    Rich transformation with geocoding, enhanced metadata, and tagging
    """

    def __init__(
        self,
        geocoding_enabled: bool = False,
        tagging_enabled: bool = True
    ):
        self.geocoding_enabled = geocoding_enabled
        self.tagging_enabled = tagging_enabled

    async def transform(self, memory: Memory) -> str:
        """Transform with enhanced features"""

        # Geocode location (if enabled)
        location_str = await self._format_location(
            memory.location_lat,
            memory.location_lon
        )

        # Extract or generate tags
        tags = []
        if self.tagging_enabled:
            tags = await self._extract_tags(memory.transcript)

        # Format timestamp
        date_str = self._format_datetime(memory.created_at)

        # Build enhanced document
        document = f"""---
Memory Record from {date_str}
ID: {memory.id}
Type: {self._humanize_type(memory.type)}
{location_str}

Transcript:
{memory.transcript}

Tags: {', '.join(tags) if tags else 'None'}
Media: {'Audio, ' if memory.audio else ''}{'Image' if memory.image else ''}
---"""

        return document

    async def _format_location(self, lat: float, lon: float) -> str:
        """Reverse geocode location"""
        if not self.geocoding_enabled or (lat == 0.0 and lon == 0.0):
            return f"Location: ({lat}, {lon})"

        # Use geocoding service (e.g., Nominatim, Google Maps)
        try:
            place_name = await self._reverse_geocode(lat, lon)
            return f"Location: {place_name} ({lat}, {lon})"
        except Exception:
            return f"Location: ({lat}, {lon})"

    async def _reverse_geocode(self, lat: float, lon: float) -> str:
        """Call geocoding service (placeholder)"""
        # Implement with actual geocoding service
        # Example: Nominatim, Google Maps, Mapbox
        return f"{lat}, {lon}"

    async def _extract_tags(self, text: str) -> list[str]:
        """Extract meaningful tags from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        keywords = re.findall(r'\b[A-Z][a-z]+\b', text)
        return [f"#{kw.lower()}" for kw in keywords[:5]]

    def _format_datetime(self, dt_str: str) -> str:
        """Format datetime in human-readable form"""
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except Exception:
            return dt_str

    def _humanize_type(self, type_str: str) -> str:
        """Convert type to human-readable form"""
        type_map = {
            "record": "Voice Recording",
            "note": "Text Note",
            "photo": "Photo Memory"
        }
        return type_map.get(type_str, type_str.title())
```

## Query Parameters and Filtering

### Understanding the `range` Parameter

The Memory API supports a `range` parameter for filtering memories:

**Common values**:
- `"day"` - Memories from the last 24 hours
- `"week"` - Memories from the last 7 days (default)
- `"month"` - Memories from the last 30 days
- `"all"` - All memories (if supported)

### Incremental Sync Strategy

**Challenge**: The API doesn't have a "since timestamp" filter

**Solution**: Use `range` + client-side filtering

```python
async def get_new_memories(
    client: MemoryAPIClient,
    ctx_id: str,
    last_sync_timestamp: datetime
) -> List[Memory]:
    """
    Get only memories created since last sync

    Args:
        client: Memory API client
        ctx_id: Context ID
        last_sync_timestamp: Timestamp of last successful sync

    Returns:
        List of new memories
    """
    # Fetch recent memories (using appropriate range)
    # If last sync was <1 day ago, use "day"
    # If last sync was <7 days ago, use "week"
    # Otherwise use "month" or "all"

    delta = datetime.now(timezone.utc) - last_sync_timestamp

    if delta.days < 1:
        range_param = "day"
    elif delta.days < 7:
        range_param = "week"
    else:
        range_param = "month"

    # Fetch from API
    result = await client.get_memories(
        ctx_id=ctx_id,
        limit=1000,  # Max limit
        range=range_param
    )

    # Filter client-side
    new_memories = []
    for memory in result.memories:
        memory_dt = datetime.fromisoformat(
            memory.created_at.replace("Z", "+00:00")
        )
        if memory_dt > last_sync_timestamp:
            new_memories.append(memory)

    return new_memories
```

### Pagination Handling

**Note**: Current API spec doesn't show pagination support

**Future Enhancement**: If API adds pagination:

```python
async def get_all_memories(
    client: MemoryAPIClient,
    ctx_id: str,
    range: str = "week"
) -> List[Memory]:
    """Fetch all memories with pagination"""

    all_memories = []
    offset = 0
    limit = 100

    while True:
        result = await client.get_memories(
            ctx_id=ctx_id,
            limit=limit,
            range=range,
            # offset=offset  # If API supports it
        )

        all_memories.extend(result.memories)

        if len(result.memories) < limit:
            break  # No more pages

        offset += limit

    return all_memories
```

## Error Handling

### Common Error Scenarios

**1. Authentication Failures (401)**
```python
try:
    memories = await client.get_memories(ctx_id)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        logger.error("Invalid API key")
        # Alert admin, pause connector
```

**2. Context Not Found (404)**
```python
try:
    memories = await client.get_memories(ctx_id)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        logger.warning(f"Context {ctx_id} not found")
        # Disable connector or alert user
```

**3. Rate Limiting (429)** - Not in spec, but good to handle
```python
try:
    memories = await client.get_memories(ctx_id)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        retry_after = int(e.response.headers.get("Retry-After", 60))
        await asyncio.sleep(retry_after)
        # Retry
```

**4. Server Errors (500)**
```python
try:
    memories = await client.get_memories(ctx_id)
except httpx.HTTPStatusError as e:
    if e.response.status_code >= 500:
        logger.error(f"Memory API server error: {e}")
        # Retry with exponential backoff
```

## Testing the Integration

### Manual Testing

```python
# test_memory_api.py

import asyncio
from lightrag.connectors.memory_api_client import MemoryAPIClient

async def test_connection():
    client = MemoryAPIClient(
        api_url="http://127.0.0.1:8080",
        api_key="your-api-key"
    )

    # Test connection
    connected = await client.check_connection()
    print(f"Connected: {connected}")

    # Fetch memories
    memories = await client.get_memories(
        ctx_id="CTX123",
        limit=10,
        range="week"
    )

    print(f"Fetched {len(memories.memories)} memories")

    for memory in memories.memories:
        print(f"- {memory.id}: {memory.transcript[:50]}...")

    await client.close()

asyncio.run(test_connection())
```

### Integration Testing

```python
# test_end_to_end.py

import asyncio
from lightrag.connectors.memory_api_client import MemoryAPIClient
from lightrag.connectors.memory_transformer import StandardTransformationStrategy
from lightrag.connectors.lightrag_client import LightRAGDirectClient

async def test_end_to_end():
    # 1. Fetch from Memory API
    memory_client = MemoryAPIClient(
        api_url="http://127.0.0.1:8080",
        api_key="your-memory-api-key"
    )

    memories = await memory_client.get_memories(
        ctx_id="CTX123",
        limit=5,
        range="day"
    )

    print(f"Fetched {len(memories.memories)} memories")

    # 2. Transform
    transformer = StandardTransformationStrategy()
    documents = [transformer.transform(m) for m in memories.memories]

    # 3. Insert into LightRAG
    lightrag_client = LightRAGDirectClient(
        working_dir="./test_lightrag"
    )

    await lightrag_client.initialize()

    for i, doc in enumerate(documents):
        doc_id = await lightrag_client.insert_document(
            text=doc,
            file_source=f"memory://{memories.memories[i].id}"
        )
        print(f"Inserted: {doc_id}")

    await lightrag_client.close()
    await memory_client.close()

asyncio.run(test_end_to_end())
```

## Next Steps

See `04-CONFIGURATION.md` for deployment and configuration details.
