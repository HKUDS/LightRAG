from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# Install OpenAI SDK since xAI is OpenAI-compatible
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Any

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


def create_xai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client configured for xAI Grok API.

    Args:
        api_key: xAI API key. If None, uses the XAI_API_KEY environment variable.
        base_url: Base URL for the xAI API. Defaults to https://api.x.ai/v1
        client_configs: Additional configuration options for the AsyncOpenAI client.

    Returns:
        An AsyncOpenAI client instance configured for xAI.
    """
    if not api_key:
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required")

    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }

    if client_configs is None:
        client_configs = {}

    # Create a merged config dict with precedence: explicit params > client_configs > defaults
    merged_configs = {
        **client_configs,
        "default_headers": default_headers,
        "api_key": api_key,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.environ.get(
            "XAI_API_BASE", "https://api.x.ai/v1"
        )

    return AsyncOpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def xai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> str:
    """Complete text generation using xAI Grok models.

    Args:
        model: The xAI model to use (e.g., "grok-2-1212", "grok-3-mini")
        prompt: The user prompt
        system_prompt: Optional system prompt
        history_messages: List of previous messages in conversation
        base_url: Optional base URL override
        api_key: Optional API key override
        **kwargs: Additional parameters passed to the API (including stream)

    Returns:
        A string response
    """
    if history_messages is None:
        history_messages = []

    # Set logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    xai_async_client = create_xai_async_client(
        api_key=api_key, base_url=base_url, client_configs=kwargs.get("client_configs")
    )

    # Build messages in OpenAI format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add history messages
    messages.extend(history_messages)

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    # Extract OpenAI-compatible parameters
    stream = kwargs.pop("stream", False)  # Remove stream from kwargs and get its value
    openai_kwargs = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # Add common parameters if present
    if "temperature" in kwargs:
        openai_kwargs["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        openai_kwargs["max_tokens"] = kwargs["max_tokens"]
    if "top_p" in kwargs:
        openai_kwargs["top_p"] = kwargs["top_p"]

    verbose_debug(f"xAI API request: {openai_kwargs}")

    try:
        response = await xai_async_client.chat.completions.create(**openai_kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise InvalidResponseError("xAI API returned empty response")

        verbose_debug(f"xAI API response: {content}")
        return content

    except Exception as e:
        logger.error(f"Error calling xAI API: {str(e)}")
        raise


async def _stream_response(client: AsyncOpenAI, kwargs: dict) -> AsyncIterator[str]:
    """Handle streaming response from xAI API."""
    try:
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Error in xAI streaming: {str(e)}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def xai_embed(
    texts: list[str],
    model: str = "text-embedding-ada-002",  # Default embedding model
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Create embeddings using xAI embedding models.

    Note: xAI doesn't have dedicated embedding models yet,
    so this uses OpenAI embeddings as fallback. This can be updated
    when xAI releases embedding models.

    To avoid dimension conflicts, it's recommended to use a consistent
    embedding model throughout your LightRAG workflow.
    """
    if not api_key:
        # Try xAI API key first, then OpenAI as fallback
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "XAI_API_KEY or OPENAI_API_KEY environment variable is required"
            )

    # For now, we'll use OpenAI embeddings as xAI doesn't have dedicated embedding models
    # This can be updated when xAI releases embedding models
    from .openai import openai_embed

    # Use OpenAI API for embeddings since xAI doesn't have embedding models yet
    openai_api_key = os.environ.get("OPENAI_API_KEY", api_key)
    openai_base_url = "https://api.openai.com/v1"

    return await openai_embed(
        texts, model=model, api_key=openai_api_key, base_url=openai_base_url, **kwargs
    )


# Convenience functions for common Grok models
async def grok_3_mini_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """Complete text using Grok 3 Mini model."""
    return await xai_complete_if_cache(
        model="grok-3-mini",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def grok_2_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """Complete text using Grok 2 model."""
    return await xai_complete_if_cache(
        model="grok-2-1212",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Embedding function with proper attributes for LightRAG
# Using OpenAI's text-embedding-ada-002 dimensions as default since xAI doesn't have embedding models yet
xai_embed_with_attrs = wrap_embedding_func_with_attrs(
    embedding_func=xai_embed,
    embedding_dim=1536,  # OpenAI ada-002 dimension
    max_token_size=8192,  # Default max token size
)
