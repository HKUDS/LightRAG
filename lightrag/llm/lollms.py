import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

if not pm.is_installed("aiohttp"):
    pm.install("aiohttp")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)

from typing import Union, List
import numpy as np


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def lollms_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url="http://localhost:9600",
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Client implementation for lollms generation."""

    stream = True if kwargs.get("stream") else False
    api_key = kwargs.pop("api_key", None)
    headers = (
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        if api_key
        else {"Content-Type": "application/json"}
    )

    # Extract lollms specific parameters
    request_data = {
        "prompt": prompt,
        "model_name": model,
        "personality": kwargs.get("personality", -1),
        "n_predict": kwargs.get("n_predict", None),
        "stream": stream,
        "temperature": kwargs.get("temperature", 0.1),
        "top_k": kwargs.get("top_k", 50),
        "top_p": kwargs.get("top_p", 0.95),
        "repeat_penalty": kwargs.get("repeat_penalty", 0.8),
        "repeat_last_n": kwargs.get("repeat_last_n", 40),
        "seed": kwargs.get("seed", None),
        "n_threads": kwargs.get("n_threads", 8),
    }

    # Prepare the full prompt including history
    full_prompt = ""
    if system_prompt:
        full_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        full_prompt += f"{msg['role']}: {msg['content']}\n"
    full_prompt += prompt

    request_data["prompt"] = full_prompt
    timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", None))

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        if stream:

            async def inner():
                async with session.post(
                    f"{base_url}/lollms_generate", json=request_data
                ) as response:
                    async for line in response.content:
                        yield line.decode().strip()

            return inner()
        else:
            async with session.post(
                f"{base_url}/lollms_generate", json=request_data
            ) as response:
                return await response.text()


async def lollms_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    """Complete function for lollms model generation."""

    # Extract and remove keyword_extraction from kwargs if present
    keyword_extraction = kwargs.pop("keyword_extraction", None)

    # Get model name from config
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    # If keyword extraction is needed, we might need to modify the prompt
    # or add specific parameters for JSON output (if lollms supports it)
    if keyword_extraction:
        # Note: You might need to adjust this based on how lollms handles structured output
        pass

    return await lollms_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def lollms_embed(
    texts: List[str], embed_model=None, base_url="http://localhost:9600", **kwargs
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using lollms server.

    Args:
        texts: List of strings to embed
        embed_model: Model name (not used directly as lollms uses configured vectorizer)
        base_url: URL of the lollms server
        **kwargs: Additional arguments passed to the request

    Returns:
        np.ndarray: Array of embeddings
    """
    api_key = kwargs.pop("api_key", None)
    headers = (
        {"Content-Type": "application/json", "Authorization": api_key}
        if api_key
        else {"Content-Type": "application/json"}
    )
    async with aiohttp.ClientSession(headers=headers) as session:
        embeddings = []
        for text in texts:
            request_data = {"text": text}

            async with session.post(
                f"{base_url}/lollms_embed",
                json=request_data,
            ) as response:
                result = await response.json()
                embeddings.append(result["vector"])

        return np.array(embeddings)
