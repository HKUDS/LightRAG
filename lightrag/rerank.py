from __future__ import annotations

import os
import aiohttp
from typing import Any, List, Dict, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from .utils import logger

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(aiohttp.ClientResponseError)
    ),
)
async def generic_rerank_api(
    query: str,
    documents: List[str],
    model: str,
    base_url: str,
    api_key: Optional[str],
    top_n: Optional[int] = None,
    return_documents: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    response_format: str = "standard",  # "standard" (Jina/Cohere) or "aliyun"
    request_format: str = "standard",  # "standard" (Jina/Cohere) or "aliyun"
) -> List[Dict[str, Any]]:
    """
    Generic rerank API call for Jina/Cohere/Aliyun models.

    Args:
        query: The search query
        documents: List of strings to rerank
        model: Model name to use
        base_url: API endpoint URL
        api_key: API key for authentication
        top_n: Number of top results to return
        return_documents: Whether to return document text (Jina only)
        extra_body: Additional body parameters
        response_format: Response format type ("standard" for Jina/Cohere, "aliyun" for Aliyun)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if not base_url:
        raise ValueError("Base URL is required")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build request payload based on request format
    if request_format == "aliyun":
        # Aliyun format: nested input/parameters structure
        payload = {
            "model": model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {},
        }

        # Add optional parameters to parameters object
        if top_n is not None:
            payload["parameters"]["top_n"] = top_n

        if return_documents is not None:
            payload["parameters"]["return_documents"] = return_documents

        # Add extra parameters to parameters object
        if extra_body:
            payload["parameters"].update(extra_body)
    else:
        # Standard format for Jina/Cohere
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
        }

        # Add optional parameters
        if top_n is not None:
            payload["top_n"] = top_n

        # Only Jina API supports return_documents parameter
        if return_documents is not None:
            payload["return_documents"] = return_documents

        # Add extra parameters
        if extra_body:
            payload.update(extra_body)

    logger.debug(
        f"Rerank request: {len(documents)} documents, model: {model}, format: {response_format}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                content_type = response.headers.get("content-type", "").lower()
                is_html_error = (
                    error_text.strip().startswith("<!DOCTYPE html>")
                    or "text/html" in content_type
                )
                if is_html_error:
                    if response.status == 502:
                        clean_error = "Bad Gateway (502) - Rerank service temporarily unavailable. Please try again in a few minutes."
                    elif response.status == 503:
                        clean_error = "Service Unavailable (503) - Rerank service is temporarily overloaded. Please try again later."
                    elif response.status == 504:
                        clean_error = "Gateway Timeout (504) - Rerank service request timed out. Please try again."
                    else:
                        clean_error = f"HTTP {response.status} - Rerank service error. Please try again later."
                else:
                    clean_error = error_text
                logger.error(f"Rerank API error {response.status}: {clean_error}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Rerank API error: {clean_error}",
                )

            response_json = await response.json()

            if response_format == "aliyun":
                # Aliyun format: {"output": {"results": [...]}}
                results = response_json.get("output", {}).get("results", [])
                if not isinstance(results, list):
                    logger.warning(
                        f"Expected 'output.results' to be list, got {type(results)}: {results}"
                    )
                    results = []

            elif response_format == "standard":
                # Standard format: {"results": [...]}
                results = response_json.get("results", [])
                if not isinstance(results, list):
                    logger.warning(
                        f"Expected 'results' to be list, got {type(results)}: {results}"
                    )
                    results = []
            else:
                raise ValueError(f"Unsupported response format: {response_format}")
            if not results:
                logger.warning("Rerank API returned empty results")
                return []

            # Standardize return format
            return [
                {"index": result["index"], "relevance_score": result["relevance_score"]}
                for result in results
            ]


async def cohere_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "rerank-v3.5",
    base_url: str = "https://api.cohere.com/v2/rerank",
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere API.

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: API key
        model: rerank model name
        base_url: API endpoint
        extra_body: Additional body for http request(reserved for extra params)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if api_key is None:
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("RERANK_BINDING_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=None,  # Cohere doesn't support this parameter
        extra_body=extra_body,
        response_format="standard",
    )


async def jina_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "jina-reranker-v2-base-multilingual",
    base_url: str = "https://api.jina.ai/v1/rerank",
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina AI API.

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: API key
        model: rerank model name
        base_url: API endpoint
        extra_body: Additional body for http request(reserved for extra params)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY") or os.getenv("RERANK_BINDING_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=False,
        extra_body=extra_body,
        response_format="standard",
    )


async def ali_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "gte-rerank-v2",
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Aliyun DashScope API.

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: Aliyun API key
        model: rerank model name
        base_url: API endpoint
        extra_body: Additional body for http request(reserved for extra params)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("RERANK_BINDING_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=False,  # Aliyun doesn't need this parameter
        extra_body=extra_body,
        response_format="aliyun",
        request_format="aliyun",
    )


async def sentence_transformers_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "BAAI/bge-reranker-v2-m3",
    base_url: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using CrossEncoder from Sentence Transformers.

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: Unused
        model: rerank model name
        base_url: Unused
        extra_body: Unused

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(model)
    rankings = cross_encoder.rank(query=query, documents=documents, top_k=top_n)
    return [
        {"index": result["corpus_id"], "relevance_score": result["score"]}
        for result in rankings
    ]


"""Please run this test as a module:
python -m lightrag.rerank
"""
if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage - documents should be strings, not dictionaries
        docs = [
            "The capital of France is Paris.",
            "Tokyo is the capital of Japan.",
            "London is the capital of England.",
        ]

        query = "What is the capital of France?"

        # Test Jina rerank
        try:
            print("=== Jina Rerank ===")
            result = await jina_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Jina Error: {e}")

        # Test Cohere rerank
        try:
            print("\n=== Cohere Rerank ===")
            result = await cohere_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Cohere Error: {e}")

        # Test Aliyun rerank
        try:
            print("\n=== Aliyun Rerank ===")
            result = await ali_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Aliyun Error: {e}")
        
        # Test Sentence Transformers rerank
        try:
            print("\n=== Sentence Transformers Rerank ===")
            result = await sentence_transformers_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Sentence Transformers Error: {e}")

    asyncio.run(main())
