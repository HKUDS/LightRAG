"""FastMCP tools for the LightRAG HTTP API."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastmcp import FastMCP


mcp = FastMCP("LightRAG")


class LightRAGAPIClient:
    """Small HTTP client for the LightRAG query endpoints."""

    def __init__(self) -> None:
        base_url = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621").rstrip(
            "/"
        )
        self.base_url = f"{base_url}/"
        self.timeout = float(os.getenv("LIGHTRAG_API_TIMEOUT", "150"))

        headers = {"Accept": "application/json"}
        api_key = os.getenv("LIGHTRAG_API_KEY")
        bearer_token = os.getenv("LIGHTRAG_API_TOKEN")
        if api_key:
            headers["X-API-Key"] = api_key
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self.headers = headers

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
        ) as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return response.json()

    async def stream_post(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        headers = {**self.headers, "Accept": "application/x-ndjson"}
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        ) as client:
            async with client.stream("POST", path, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield json.loads(line)


def _query_payload(
    query: str,
    mode: str,
    response_type: str,
    top_k: int | None,
    chunk_top_k: int | None,
    max_entity_tokens: int | None,
    max_relation_tokens: int | None,
    max_total_tokens: int | None,
    hl_keywords: list[str] | None,
    ll_keywords: list[str] | None,
    conversation_history: list[dict[str, Any]] | None,
    user_prompt: str | None,
    enable_rerank: bool | None,
    include_references: bool,
    include_chunk_content: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "response_type": response_type,
        "include_references": include_references,
        "include_chunk_content": include_chunk_content,
    }
    optional_values = {
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "max_entity_tokens": max_entity_tokens,
        "max_relation_tokens": max_relation_tokens,
        "max_total_tokens": max_total_tokens,
        "hl_keywords": hl_keywords,
        "ll_keywords": ll_keywords,
        "conversation_history": conversation_history,
        "user_prompt": user_prompt,
        "enable_rerank": enable_rerank,
    }
    payload.update(
        {key: value for key, value in optional_values.items() if value is not None}
    )
    return payload


async def _tool_payload(
    query: str,
    mode: str = "mix",
    response_type: str = "Multiple Paragraphs",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    user_prompt: str | None = None,
    enable_rerank: bool | None = None,
    include_references: bool = True,
    include_chunk_content: bool = False,
) -> dict[str, Any]:
    return _query_payload(
        query,
        mode,
        response_type,
        top_k,
        chunk_top_k,
        max_entity_tokens,
        max_relation_tokens,
        max_total_tokens,
        hl_keywords,
        ll_keywords,
        conversation_history,
        user_prompt,
        enable_rerank,
        include_references,
        include_chunk_content,
    )


@mcp.tool()
async def query(
    query: str,
    mode: str = "mix",
    response_type: str = "Multiple Paragraphs",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    user_prompt: str | None = None,
    enable_rerank: bool | None = None,
    include_references: bool = True,
    include_chunk_content: bool = False,
) -> dict[str, Any]:
    """Run a non-streaming LightRAG query and return its API response."""
    payload = await _tool_payload(
        query,
        mode,
        response_type,
        top_k,
        chunk_top_k,
        max_entity_tokens,
        max_relation_tokens,
        max_total_tokens,
        hl_keywords,
        ll_keywords,
        conversation_history,
        user_prompt,
        enable_rerank,
        include_references,
        include_chunk_content,
    )
    return await LightRAGAPIClient().post("query", payload)


@mcp.tool()
async def query_stream(
    query: str,
    mode: str = "mix",
    response_type: str = "Multiple Paragraphs",
    include_references: bool = True,
    include_chunk_content: bool = False,
    include_progress: bool = False,
) -> list[dict[str, Any]]:
    """Run a LightRAG streaming query and return ordered NDJSON events."""
    payload = await _tool_payload(
        query,
        mode=mode,
        response_type=response_type,
        include_references=include_references,
        include_chunk_content=include_chunk_content,
    )
    payload["stream"] = True
    payload["include_progress"] = include_progress
    return [
        event
        async for event in LightRAGAPIClient().stream_post("query/stream", payload)
    ]


@mcp.tool()
async def query_data(
    query: str,
    mode: str = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
    enable_rerank: bool | None = None,
) -> dict[str, Any]:
    """Retrieve structured LightRAG data without LLM answer generation."""
    payload = await _tool_payload(
        query,
        mode=mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        max_entity_tokens=max_entity_tokens,
        max_relation_tokens=max_relation_tokens,
        max_total_tokens=max_total_tokens,
        hl_keywords=hl_keywords,
        ll_keywords=ll_keywords,
        enable_rerank=enable_rerank,
    )
    return await LightRAGAPIClient().post("query/data", payload)


if __name__ == "__main__":
    mcp.run()