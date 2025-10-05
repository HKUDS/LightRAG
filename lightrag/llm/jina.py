import os

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("aiohttp"):
    pm.install("aiohttp")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import base64

import aiohttp
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import logger, wrap_embedding_func_with_attrs


async def fetch_data(url, headers, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()

                # Check if the error response is HTML (common for 502, 503, etc.)
                content_type = response.headers.get("content-type", "").lower()
                is_html_error = (
                    error_text.strip().startswith("<!DOCTYPE html>")
                    or "text/html" in content_type
                )

                if is_html_error:
                    # Provide clean, user-friendly error messages for HTML error pages
                    if response.status == 502:
                        clean_error = "Bad Gateway (502) - Jina AI service temporarily unavailable. Please try again in a few minutes."
                    elif response.status == 503:
                        clean_error = "Service Unavailable (503) - Jina AI service is temporarily overloaded. Please try again later."
                    elif response.status == 504:
                        clean_error = "Gateway Timeout (504) - Jina AI service request timed out. Please try again."
                    else:
                        clean_error = f"HTTP {response.status} - Jina AI service error. Please try again later."
                else:
                    # Use original error text if it's not HTML
                    clean_error = error_text

                logger.error(f"Jina API error {response.status}: {clean_error}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Jina API error: {clean_error}",
                )
            response_json = await response.json()
            data_list = response_json.get("data", [])
            return data_list


@wrap_embedding_func_with_attrs(embedding_dim=2048)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(aiohttp.ClientResponseError)
    ),
)
async def jina_embed(
    texts: list[str],
    dimensions: int = 2048,
    late_chunking: bool = False,
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using Jina AI's API.

    Args:
        texts: List of texts to embed.
        dimensions: The embedding dimensions (default: 2048 for jina-embeddings-v4).
        late_chunking: Whether to use late chunking.
        base_url: Optional base URL for the Jina API.
        api_key: Optional Jina API key. If None, uses the JINA_API_KEY environment variable.

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        aiohttp.ClientError: If there is a connection error with the Jina API.
        aiohttp.ClientResponseError: If the Jina API returns an error response.
    """
    if api_key:
        os.environ["JINA_API_KEY"] = api_key

    if "JINA_API_KEY" not in os.environ:
        raise ValueError("JINA_API_KEY environment variable is required")

    url = base_url or "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
    }
    data = {
        "model": "jina-embeddings-v4",
        "task": "text-matching",
        "dimensions": dimensions,
        "embedding_type": "base64",
        "input": texts,
    }

    # Only add optional parameters if they have non-default values
    if late_chunking:
        data["late_chunking"] = late_chunking

    logger.debug(
        f"Jina embedding request: {len(texts)} texts, dimensions: {dimensions}"
    )

    try:
        data_list = await fetch_data(url, headers, data)

        if not data_list:
            logger.error("Jina API returned empty data list")
            raise ValueError("Jina API returned empty data list")

        if len(data_list) != len(texts):
            logger.error(
                f"Jina API returned {len(data_list)} embeddings for {len(texts)} texts"
            )
            raise ValueError(
                f"Jina API returned {len(data_list)} embeddings for {len(texts)} texts"
            )

        embeddings = np.array(
            [
                np.frombuffer(base64.b64decode(dp["embedding"]), dtype=np.float32)
                for dp in data_list
            ]
        )
        logger.debug(f"Jina embeddings generated: shape {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"Jina embedding error: {e}")
        raise
