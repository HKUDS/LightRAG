import os
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("aiohttp"):
    pm.install("aiohttp")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import numpy as np
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs, logger


async def fetch_data(url, headers, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Jina API error {response.status}: {error_text}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Jina API error: {error_text}",
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

        embeddings = np.array([dp["embedding"] for dp in data_list])
        logger.debug(f"Jina embeddings generated: shape {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"Jina embedding error: {e}")
        raise
