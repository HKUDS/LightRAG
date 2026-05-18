# LightRAG Domain Glossary

## Knowledge Graph
The shared corpus of entities and relationships extracted from ingested documents.
A single Knowledge Graph is shared across all users in a team deployment.
Distinct from per-user workspaces, which are not used in this project.

## Query
An operation that retrieves context from the Knowledge Graph and synthesises a
natural-language answer using an LLM. The primary MCP tool surface for callers.

## Retrieve
A diagnostic operation that returns raw entities, relations, and chunks from the
Knowledge Graph without LLM synthesis. Intended for developers and power users
inspecting retrieval quality, not for routine AI-agent use.

## MCP Server
The adapter between MCP-compatible clients (Claude Desktop, Cursor, etc.) and the
LightRAG REST API. Exposes `query` and `retrieve` as MCP tools. Read-only: no
document insertion or deletion is exposed through this layer.

## Retrieval Mode
The algorithm used to fetch context from the Knowledge Graph (local, global,
hybrid, naive, mix). Configured server-side via `LIGHTRAG_QUERY_MODE`; not
exposed as a caller-controlled parameter in the MCP layer.

## QueryParam
A pure value object that carries all parameters for a single Knowledge Graph
Query. Has no environment reads at definition time — all numeric defaults are
compile-time constants from `lightrag/constants.py`.

## ConfigResolver
The single place where environment variables are read and converted into a
`QueryParam`. Tests inject a plain `dict`; production code uses `os.environ`
via the `default_query_param()` convenience function. Lives in
`lightrag/query_config.py`.
