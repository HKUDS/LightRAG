"""LightRAG MCP server — wraps the REST API as MCP tools.

Exposes two tools:
- query:    Full LLM-synthesised answer from the knowledge graph.
- retrieve: Raw entities, relations, and chunks without LLM synthesis.
            Diagnostic/power-user tool for inspecting retrieval quality.

Configure via environment variables:
  LIGHTRAG_API_URL      Base URL of a running lightrag-server (default: http://localhost:9621)
  LIGHTRAG_API_KEY      Optional API key for lightrag-server authentication
  LIGHTRAG_QUERY_MODE   Retrieval mode: local, global, hybrid, naive, or mix (default: mix)
  LIGHTRAG_TOP_K        Number of top entities/relations to retrieve (default: 60)
  LIGHTRAG_MCP_TIMEOUT  HTTP timeout in seconds for LightRAG requests (default: 120)

Usage:
  lightrag-mcp                          # stdio transport (Claude Desktop / Cursor)
  lightrag-mcp --transport sse          # SSE transport (remote clients)
  lightrag-mcp --transport sse --port 8001 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import httpx
from mcp import McpError
from mcp.server.fastmcp import FastMCP
from mcp.types import ErrorData

logger = logging.getLogger(__name__)

API_URL: str = os.environ.get("LIGHTRAG_API_URL", "http://localhost:9621").rstrip("/")
API_KEY: str = os.environ.get("LIGHTRAG_API_KEY", "")
QUERY_MODE: str = os.environ.get("LIGHTRAG_QUERY_MODE", "mix")
TOP_K: int = int(os.environ.get("LIGHTRAG_TOP_K", "60"))
TIMEOUT: float = float(os.environ.get("LIGHTRAG_MCP_TIMEOUT", "120"))

mcp = FastMCP("LightRAG")

_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        headers: dict[str, str] = {}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        _http_client = httpx.AsyncClient(
            base_url=API_URL,
            headers=headers,
            timeout=TIMEOUT,
        )
    return _http_client


def _format_references(refs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for r in refs:
        title: str = r.get("title", "")
        url: str = r.get("url", "")
        ref_id: str = str(r.get("reference_id", "?"))
        if title and url:
            lines.append(f"- [{title}]({url})")
        elif url:
            lines.append(f"- [{ref_id}]({url})")
        else:
            lines.append(f"- {title or ref_id}")
    return "\n".join(lines)


def _raise_for_error(resp: httpx.Response) -> None:
    """Raise McpError on 5xx; let 4xx fall through as a string result."""
    if resp.status_code >= 500:
        raise McpError(
            ErrorData(
                code=-32603,
                message=f"LightRAG server error {resp.status_code}: {resp.text[:500]}",
            )
        )


@mcp.tool()
async def query(question: str) -> str:
    """Query the LightRAG knowledge graph and return an LLM-synthesised answer.

    LightRAG understands entities and their relationships, giving richer context
    than plain vector search. Use this when you want a direct answer.

    Args:
        question: The question to answer.
    """
    payload: dict[str, Any] = {
        "query": question,
        "mode": QUERY_MODE,
        "top_k": TOP_K,
        "stream": False,
        "include_references": True,
    }
    try:
        resp = await _client().post("/query", json=payload)
        _raise_for_error(resp)
        if not resp.is_success:
            return f"LightRAG error {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        answer: str = data.get("response", "")
        refs: list[dict[str, Any]] = data.get("references") or []
        if refs:
            return f"{answer}\n\nSources:\n{_format_references(refs)}"
        return answer
    except McpError:
        raise
    except httpx.RequestError as e:
        raise McpError(
            ErrorData(
                code=-32603,
                message=(
                    f"Could not reach LightRAG server at {API_URL}. "
                    f"Is lightrag-server running? Details: {e}"
                ),
            )
        ) from e


@mcp.tool()
async def retrieve(question: str) -> str:
    """Search the LightRAG knowledge graph and return raw entities, relations, and chunks.

    Returns the retrieved context directly with no LLM synthesis, giving full
    visibility into the graph relationships and source text LightRAG found.
    Intended for developers and power users inspecting retrieval quality.

    Args:
        question: The search query.
    """
    payload: dict[str, Any] = {
        "query": question,
        "mode": QUERY_MODE,
        "top_k": TOP_K,
    }
    try:
        resp = await _client().post("/query/data", json=payload)
        _raise_for_error(resp)
        if not resp.is_success:
            return f"LightRAG error {resp.status_code}: {resp.text[:500]}"
        return json.dumps(resp.json(), indent=2, ensure_ascii=False)
    except McpError:
        raise
    except httpx.RequestError as e:
        raise McpError(
            ErrorData(
                code=-32603,
                message=(
                    f"Could not reach LightRAG server at {API_URL}. "
                    f"Is lightrag-server running? Details: {e}"
                ),
            )
        ) from e


async def _warmup() -> None:
    """Verify the LightRAG server is reachable before accepting MCP connections.

    Surfaces misconfiguration immediately at startup rather than on the first
    query, so the operator sees a clear error instead of a silent tool failure.
    A warning (not a crash) is emitted so transient connectivity issues during
    rolling restarts don't prevent the MCP server from starting.
    """
    try:
        resp = await _client().get("/health", timeout=10.0)
        if resp.is_success:
            data = resp.json()
            status = data.get("status", "unknown")
            logger.info(
                "LightRAG server reachable at %s (status: %s)", API_URL, status
            )
        else:
            logger.warning(
                "LightRAG server at %s returned %s during warm-up — "
                "queries may fail until the server is healthy.",
                API_URL,
                resp.status_code,
            )
    except httpx.RequestError as e:
        logger.warning(
            "Could not reach LightRAG server at %s during warm-up: %s\n"
            "Set LIGHTRAG_API_URL to the correct base URL if this persists.",
            API_URL,
            e,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LightRAG MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for SSE transport (default: 8001)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import asyncio

    asyncio.run(_warmup())

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
