import sys
import warnings

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

if not pm.is_installed("aiohttp"):
    pm.install("aiohttp")

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

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)


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
    enable_cot: bool = False,
    base_url="http://localhost:9600",
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Client implementation for lollms generation.

    Structured output note:
    - This adapter does not support OpenAI-style ``response_format`` JSON mode.
    - If callers pass ``response_format``, it is stripped before the request.
    - Deprecated ``keyword_extraction`` and ``entity_extraction`` booleans are
      accepted only as compatibility shims; they emit warnings and are ignored.
    """
    if enable_cot:
        from lightrag.utils import logger

        logger.debug("enable_cot=True is not supported for lollms and will be ignored.")

    # lollms has no JSON mode; drop response_format and warn when legacy
    # boolean shim flags are set.
    if kwargs.pop("keyword_extraction", False):
        warnings.warn(
            "lollms_model_if_cache(keyword_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if kwargs.pop("entity_extraction", False):
        warnings.warn(
            "lollms_model_if_cache(entity_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    kwargs.pop("response_format", None)

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
        "temperature": kwargs.get("temperature", 1.0),
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
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Complete function for lollms model generation."""

    # Forward legacy extraction flags as kwargs so lollms_model_if_cache can
    # emit a single DeprecationWarning with the correct stack frame.
    if keyword_extraction:
        kwargs.setdefault("keyword_extraction", True)
    if entity_extraction:
        kwargs.setdefault("entity_extraction", True)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    return await lollms_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1024, max_token_size=8192, model_name="lollms_embedding_model"
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
