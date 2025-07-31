# LightRAG MCP Implementation Guide

**Quick Start Guide for Developers**

## Overview

This guide provides step-by-step instructions for implementing the Model Context Protocol (MCP) integration with LightRAG. Follow this guide to build MCP tools that expose LightRAG's capabilities to Claude CLI users.

## Prerequisites

### System Requirements
- Python 3.9+
- LightRAG instance (local or remote)
- Claude CLI with MCP support
- Development environment with async support

### Dependencies Installation
```bash
# Create project directory
mkdir lightrag-mcp && cd lightrag-mcp

# Initialize with uv (recommended)
uv init
uv add "mcp[cli]" httpx pydantic aiofiles typing-extensions

# Alternative with pip
pip install "mcp[cli]" httpx pydantic aiofiles typing-extensions
```

## Quick Start Implementation

### 1. Basic Server Structure

Create `server.py`:
```python
#!/usr/bin/env python3
"""
LightRAG MCP Server - Basic Implementation
Provides MCP tools for accessing LightRAG functionality via Claude CLI.
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP
import httpx
from pydantic import BaseModel

# Configuration
class Config:
    LIGHTRAG_API_URL = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")
    LIGHTRAG_API_KEY = os.getenv("LIGHTRAG_API_KEY")
    MCP_SERVER_NAME = "lightrag-mcp"

config = Config()

# Create MCP server
mcp = FastMCP(name=config.MCP_SERVER_NAME)

# HTTP client for LightRAG API
async def get_http_client() -> httpx.AsyncClient:
    headers = {}
    if config.LIGHTRAG_API_KEY:
        headers["Authorization"] = f"Bearer {config.LIGHTRAG_API_KEY}"

    return httpx.AsyncClient(
        base_url=config.LIGHTRAG_API_URL,
        headers=headers,
        timeout=60.0
    )

# Response models
class QueryResponse(BaseModel):
    response: str
    mode: str
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]] = []

class DocumentResponse(BaseModel):
    document_id: str
    status: str
    message: str
    processing_info: Dict[str, Any] = {}

# Core MCP Tools
@mcp.tool()
async def lightrag_query(
    query: str,
    mode: str = "hybrid",
    top_k: int = 40,
    chunk_top_k: int = 10
) -> QueryResponse:
    """
    Execute a RAG query using LightRAG.

    Args:
        query: The question or search query
        mode: Query mode (naive, local, global, hybrid, mix, bypass)
        top_k: Number of entities/relations to retrieve
        chunk_top_k: Number of chunks to include

    Returns:
        Query response with answer and metadata
    """
    async with get_http_client() as client:
        try:
            response = await client.post("/query", json={
                "query": query,
                "mode": mode,
                "param": {
                    "top_k": top_k,
                    "chunk_top_k": chunk_top_k
                }
            })
            response.raise_for_status()

            data = response.json()
            return QueryResponse(
                response=data.get("response", ""),
                mode=mode,
                metadata=data.get("metadata", {}),
                sources=data.get("sources", [])
            )
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")

@mcp.tool()
async def lightrag_insert_text(
    text: str,
    title: str = "",
    metadata: Dict[str, Any] = None
) -> DocumentResponse:
    """
    Insert text document into LightRAG knowledge base.

    Args:
        text: Document content
        title: Document title (optional)
        metadata: Additional metadata (optional)

    Returns:
        Document processing status
    """
    async with get_http_client() as client:
        try:
            payload = {"text": text}
            if title:
                payload["title"] = title
            if metadata:
                payload["metadata"] = metadata

            response = await client.post("/documents/text", json=payload)
            response.raise_for_status()

            data = response.json()
            return DocumentResponse(
                document_id=data.get("document_id", ""),
                status=data.get("status", "pending"),
                message=data.get("message", "Document inserted successfully"),
                processing_info=data.get("processing_info", {})
            )
        except Exception as e:
            raise Exception(f"Document insertion failed: {str(e)}")

@mcp.tool()
async def lightrag_list_documents(
    status_filter: str = "",
    limit: int = 50
) -> Dict[str, Any]:
    """
    List documents in the knowledge base.

    Args:
        status_filter: Filter by status (pending, processing, processed, failed)
        limit: Maximum number of documents to return

    Returns:
        List of documents with status information
    """
    async with get_http_client() as client:
        try:
            params = {"limit": limit}
            if status_filter:
                params["status"] = status_filter

            response = await client.get("/documents", params=params)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            raise Exception(f"Failed to list documents: {str(e)}")

@mcp.tool()
async def lightrag_health_check() -> Dict[str, Any]:
    """
    Check LightRAG system health and status.

    Returns:
        System health information and configuration
    """
    async with get_http_client() as client:
        try:
            response = await client.get("/health")
            response.raise_for_status()

            return response.json()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Failed to connect to LightRAG API"
            }

# Main server execution
if __name__ == "__main__":
    print(f"Starting {config.MCP_SERVER_NAME}...")
    print(f"LightRAG API URL: {config.LIGHTRAG_API_URL}")
    mcp.run()
```

### 2. Configuration Setup

Create `.env` file:
```bash
# LightRAG Configuration
LIGHTRAG_API_URL=http://localhost:9621
LIGHTRAG_API_KEY=your-api-key-here

# MCP Server Configuration
MCP_SERVER_NAME=lightrag-mcp
MCP_SERVER_VERSION=1.0.0
```

### 3. Testing the Implementation

Create `test_server.py`:
```python
#!/usr/bin/env python3
"""Test script for LightRAG MCP server."""

import asyncio
import json
from server import mcp

async def test_tools():
    """Test MCP tools functionality."""
    print("Testing LightRAG MCP Tools...")

    # Test health check
    print("\n1. Testing health check...")
    try:
        health = await mcp.call_tool("lightrag_health_check", {})
        print(f"Health status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"Health check failed: {e}")

    # Test document listing
    print("\n2. Testing document listing...")
    try:
        docs = await mcp.call_tool("lightrag_list_documents", {"limit": 5})
        print(f"Found {len(docs.get('documents', []))} documents")
    except Exception as e:
        print(f"Document listing failed: {e}")

    # Test text insertion
    print("\n3. Testing text insertion...")
    try:
        result = await mcp.call_tool("lightrag_insert_text", {
            "text": "This is a test document for MCP integration.",
            "title": "MCP Test Document"
        })
        print(f"Document inserted: {result.document_id}")
    except Exception as e:
        print(f"Text insertion failed: {e}")

    # Test querying
    print("\n4. Testing query...")
    try:
        result = await mcp.call_tool("lightrag_query", {
            "query": "What is this document about?",
            "mode": "hybrid"
        })
        print(f"Query response: {result.response[:100]}...")
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tools())
```

### 4. Running the Server

```bash
# Method 1: Direct execution
python server.py

# Method 2: Using MCP CLI
uv run mcp dev server.py

# Method 3: With environment variables
LIGHTRAG_API_URL=http://localhost:9621 python server.py
```

## Advanced Implementation

### Adding Resources

Add to `server.py`:
```python
@mcp.resource("lightrag://system/config")
async def get_system_config() -> str:
    """Get LightRAG system configuration."""
    async with get_http_client() as client:
        try:
            response = await client.get("/health")
            data = response.json()
            return json.dumps(data.get("configuration", {}), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

@mcp.resource("lightrag://documents/{document_id}")
async def get_document(document_id: str) -> str:
    """Get specific document content."""
    async with get_http_client() as client:
        try:
            response = await client.get(f"/documents/{document_id}")
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
```

### Adding Streaming Support

```python
from mcp.server.fastmcp import FastMCP
from typing import AsyncIterator

@mcp.tool()
async def lightrag_stream_query(
    query: str,
    mode: str = "hybrid"
) -> AsyncIterator[str]:
    """
    Execute a streaming RAG query.

    Args:
        query: The question or search query
        mode: Query mode

    Yields:
        Streaming response chunks
    """
    async with get_http_client() as client:
        try:
            async with client.stream("POST", "/query/stream", json={
                "query": query,
                "mode": mode
            }) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"
```

### Error Handling and Logging

```python
import logging
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightrag-mcp")

@asynccontextmanager
async def handle_errors():
    """Context manager for consistent error handling."""
    try:
        yield
    except httpx.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        raise Exception(f"API communication error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise Exception(f"Operation failed: {str(e)}")

# Updated tool with error handling
@mcp.tool()
async def lightrag_query_safe(query: str, mode: str = "hybrid") -> QueryResponse:
    """Safe query with comprehensive error handling."""
    async with handle_errors():
        async with get_http_client() as client:
            response = await client.post("/query", json={
                "query": query,
                "mode": mode
            })
            response.raise_for_status()
            data = response.json()

            logger.info(f"Query executed successfully: {mode} mode")
            return QueryResponse(
                response=data.get("response", ""),
                mode=mode,
                metadata=data.get("metadata", {})
            )
```

## Integration with Claude CLI

### Setting up Claude CLI

```bash
# Install Claude CLI if not already installed
npm install -g @anthropic-ai/claude-cli

# Configure MCP server in Claude CLI
claude config mcp add lightrag-mcp python server.py
```

### Using from Claude CLI

```bash
# Query the knowledge base
claude mcp lightrag_query "What are the main themes in my documents?" --mode hybrid

# Insert a document
claude mcp lightrag_insert_text "This is important research data..." --title "Research Notes"

# Check system health
claude mcp lightrag_health_check

# List documents
claude mcp lightrag_list_documents --limit 10

# Access resources
claude mcp resource "lightrag://system/config"
```

## Deployment Options

### Option 1: Standalone Server

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .
COPY .env .

EXPOSE 8000

CMD ["python", "server.py"]
```

### Option 2: Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  lightrag-mcp:
    build: .
    environment:
      - LIGHTRAG_API_URL=http://lightrag:9621
      - LIGHTRAG_API_KEY=${LIGHTRAG_API_KEY}
    depends_on:
      - lightrag
    ports:
      - "8000:8000"

  lightrag:
    image: lightrag:latest
    ports:
      - "9621:9621"
    volumes:
      - ./data:/app/data
```

### Option 3: Systemd Service

Create `/etc/systemd/system/lightrag-mcp.service`:
```ini
[Unit]
Description=LightRAG MCP Server
After=network.target

[Service]
Type=simple
User=lightrag
WorkingDirectory=/opt/lightrag-mcp
ExecStart=/opt/lightrag-mcp/venv/bin/python server.py
Restart=always
RestartSec=10
Environment=LIGHTRAG_API_URL=http://localhost:9621

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
# Check if LightRAG is running
curl http://localhost:9621/health

# Check configuration
echo $LIGHTRAG_API_URL
```

#### 2. Authentication Errors
```bash
# Verify API key
curl -H "Authorization: Bearer $LIGHTRAG_API_KEY" http://localhost:9621/health
```

#### 3. MCP Tool Not Found
```bash
# List available tools
claude mcp list-tools

# Refresh MCP configuration
claude config mcp refresh
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to server.py
@mcp.tool()
async def debug_connection() -> Dict[str, Any]:
    """Debug connection to LightRAG."""
    async with get_http_client() as client:
        try:
            response = await client.get("/health")
            return {
                "status": "connected",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "url": str(response.url)
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "type": type(e).__name__
            }
```

## Performance Optimization

### Connection Pooling

```python
from contextlib import asynccontextmanager

class LightRAGClient:
    def __init__(self):
        self._client = None

    async def __aenter__(self):
        if self._client is None:
            headers = {}
            if config.LIGHTRAG_API_KEY:
                headers["Authorization"] = f"Bearer {config.LIGHTRAG_API_KEY}"

            self._client = httpx.AsyncClient(
                base_url=config.LIGHTRAG_API_URL,
                headers=headers,
                timeout=60.0,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self._client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

# Global client instance
lightrag_client = LightRAGClient()

# Use in tools
@mcp.tool()
async def optimized_query(query: str, mode: str = "hybrid") -> QueryResponse:
    """Optimized query with connection pooling."""
    async with lightrag_client as client:
        response = await client.post("/query", json={"query": query, "mode": mode})
        response.raise_for_status()
        data = response.json()
        return QueryResponse(**data)
```

### Caching

```python
from functools import lru_cache
import time

# Simple in-memory cache
cache = {}
CACHE_TTL = 300  # 5 minutes

def cached_response(key: str, ttl: int = CACHE_TTL):
    """Simple caching decorator."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"{key}:{hash(str(args) + str(kwargs))}"

            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result

            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

@mcp.tool()
@cached_response("health_check", ttl=60)
async def cached_health_check() -> Dict[str, Any]:
    """Cached health check."""
    async with lightrag_client as client:
        response = await client.get("/health")
        return response.json()
```

## Best Practices

### 1. Configuration Management
- Use environment variables for all configuration
- Provide sensible defaults
- Validate configuration on startup

### 2. Error Handling
- Always provide meaningful error messages
- Include recovery suggestions when possible
- Log errors for debugging

### 3. Performance
- Use connection pooling for HTTP clients
- Implement caching for frequently accessed data
- Set appropriate timeouts

### 4. Security
- Never log sensitive information
- Validate all inputs
- Use HTTPS in production

### 5. Documentation
- Document all tools and resources
- Provide usage examples
- Keep documentation up to date

This implementation guide provides a solid foundation for building production-ready MCP integration with LightRAG. Start with the basic implementation and gradually add advanced features as needed.
