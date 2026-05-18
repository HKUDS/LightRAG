# ADR-0001: MCP Server is a thin adapter; query caching deferred

**Status:** Accepted  
**Date:** 2026-05-07

## Context

The MCP Server (`lightrag/mcp_server.py`) wraps the LightRAG REST API as two
MCP tools (`query`, `retrieve`). After a grilling session on its design, three
deepening options were considered:

1. **Startup warm-up** — verify the LightRAG server is reachable before
   accepting MCP connections.
2. **In-process query cache** — skip the HTTP round-trip for repeated identical
   questions within a session.
3. **Accept it as a thin adapter** — record the decision so future architecture
   reviews don't re-open it without new evidence.

## Decision

The MCP Server is intentionally a **thin adapter** between the MCP protocol and
the LightRAG REST API. It owns:

- HTTP client lifecycle and authentication
- Error translation (transport/5xx → `McpError`; 4xx → string result)
- Reference formatting (`[title](url)` with fallback)
- Startup warm-up (added; see below)

It does **not** own retrieval logic, Knowledge Graph access, or LLM calls —
those stay in the LightRAG server.

**Query caching was considered and deferred.** Reasons:

- Cache invalidation requires knowing when the Knowledge Graph changes (new
  documents ingested). The MCP Server has no visibility into this event.
- In practice, repeated identical queries within a single MCP session are rare
  for the team-server use case this project targets.
- If repeated-query load becomes measurable, caching belongs in the LightRAG
  REST API (`/query` endpoint) where cache invalidation can be wired to document
  ingestion events — not in the MCP adapter.

**Startup warm-up was added.** A single `GET /health` call on startup surfaces
misconfiguration immediately (wrong `LIGHTRAG_API_URL`, server not started)
rather than silently failing the first user query. A warning — not a crash — is
emitted so transient connectivity issues during rolling restarts don't prevent
the MCP server from coming up.

## Consequences

- Adding behaviour to the MCP layer (e.g. per-user context, audit logging)
  requires revisiting this ADR and justifying why it belongs in the adapter
  rather than the REST API.
- If query caching is later added, it should be implemented in the LightRAG
  REST API, not here.
