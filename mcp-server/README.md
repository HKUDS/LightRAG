# LightRAG MCP Server

This standalone FastMCP server exposes three tools backed by a running LightRAG
HTTP API:

- `query`: non-streaming RAG answer
- `query_stream`: ordered streaming response events
- `query_data`: structured retrieval data without LLM generation

## Run

From this directory:

```bash
uv sync
uv run lightrag-mcp
```

The server uses `stdio`, which is the default transport for MCP clients.

Set these environment variables before starting it:

```bash
LIGHTRAG_API_URL=http://localhost:9621
LIGHTRAG_API_KEY=your-api-key
LIGHTRAG_API_TOKEN=your-jwt-token
LIGHTRAG_API_TIMEOUT=150
```

`LIGHTRAG_API_KEY` and `LIGHTRAG_API_TOKEN` are optional. Set only the one your
LightRAG API is configured to accept. The URL may include a reverse-proxy path
prefix, for example `https://example.com/lightrag`.