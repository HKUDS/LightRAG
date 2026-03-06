"""
Custom rerank function for local vLLM BAAI/bge-reranker-v2-m3 service.
This provides OpenAI-compatible reranking for LightRAG.
"""

import httpx
import asyncio
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


async def local_llm_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "BAAI/bge-reranker-v2-m3",
    base_url: str = "http://localhost:8001",
    title: Optional[str] = None,
    return_documents: bool = True,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using local vLLM BAAI/bge-reranker-v2-m3 service (OpenAI-compatible).

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: API key (not used for local service, kept for compatibility)
        model: rerank model name
        base_url: Base URL for the reranking service
        title: Title for the document (as dict with 'text' and 'multiModal' keys for some APIs)
        return_documents: Whether to return documents in the results
        show_progress: Whether to show progress (has no effect here)

    Returns:
        List of dictionaries containing rerank scores for each document
    """
    if not documents:
        return []

    # Prepare the request to the local vLLM rerank endpoint
    endpoint = f"{base_url}/v1/rerank"

    # vLLM expects documents as plain documents: ["doc1", "doc2", ...]
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }

    if top_n is not None:
        payload["top_n"] = top_n

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Sending rerank request to {endpoint} with {len(documents)} documents")
            response = await client.post(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key or 'sk-local-vllm'}",
                },
            )
            logger.info(f"Rerank response status: {response.status_code}")
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Rerank API error: {error_text}")
                raise Exception(f"Rerank API error {response.status_code}: {error_text}")
            response.raise_for_status()
            result = response.json()

            # Convert vLLM's result format to the format expected by LightRAG
            # vLLM returns: {"id": "...", "model": "...", "usage": {...}, "results": [...]}
            # LightRAG expects: [{"index": int, "relevance_score": float}, ...]
            if "results" in result:
                formatted_results = [
                    {"index": r["index"], "relevance_score": r["relevance_score"]}
                    for r in result["results"]
                ]
                logger.info(f"Ranked {len(formatted_results)} documents")
                # Sort by relevance_score to get top_n results
                formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                if top_n is not None:
                    formatted_results = formatted_results[:top_n]
                logger.info(f"Returning top {len(formatted_results)} documents")
                return formatted_results
            else:
                logger.warning(f"Unexpected rerank response format: {result}")
                # If format is unexpected, return empty list
                return []

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        # Return results in original order if reranking fails
        # Default to equal scores
        return [{"relevance_score": 0.5} for _ in documents]


__all__ = ["local_llm_rerank"]
