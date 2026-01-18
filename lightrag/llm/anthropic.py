from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging
import numpy as np
from typing import Any, Union, AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

# Install Anthropic SDK if not present
if not pm.is_installed("anthropic"):
    pm.install("anthropic")

# Add Voyage AI import
if not pm.is_installed("voyageai"):
    pm.install("voyageai")
import voyageai

from anthropic import (
    AsyncAnthropic,
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
    safe_unicode_decode,
    logger,
)
from lightrag.api import __api_version__


# Custom exception for retry mechanism
class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


# Core Anthropic completion function with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def anthropic_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    if enable_cot:
        logger.debug(
            "enable_cot=True is not supported for the Anthropic API and will be ignored."
        )
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }

    # Set logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("anthropic").setLevel(logging.INFO)

    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    timeout = kwargs.pop("timeout", None)

    anthropic_async_client = (
        AsyncAnthropic(
            default_headers=default_headers, api_key=api_key, timeout=timeout
        )
        if base_url is None
        else AsyncAnthropic(
            base_url=base_url,
            default_headers=default_headers,
            api_key=api_key,
            timeout=timeout,
        )
    )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Sending Query to Anthropic LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    verbose_debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    try:
        response = await anthropic_async_client.messages.create(
            model=model, messages=messages, stream=True, **kwargs
        )
    except APIConnectionError as e:
        logger.error(f"Anthropic API Connection Error: {e}")
        raise
    except RateLimitError as e:
        logger.error(f"Anthropic API Rate Limit Error: {e}")
        raise
    except APITimeoutError as e:
        logger.error(f"Anthropic API Timeout Error: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Anthropic API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        raise

    async def stream_response():
        try:
            async for event in response:
                content = (
                    event.delta.text
                    if hasattr(event, "delta") and hasattr(event.delta, "text") and event.delta.text
                    else None
                )
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content
        except Exception as e:
            logger.error(f"Error in stream response: {str(e)}")
            raise

    return stream_response()


# Generic Anthropic completion function
async def anthropic_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await anthropic_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


# Claude 3 Opus specific completion
async def claude_3_opus_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-3-opus-20240229",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


# Claude 3 Sonnet specific completion
async def claude_3_sonnet_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-3-sonnet-20240229",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


# Claude 3 Haiku specific completion
async def claude_3_haiku_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-3-haiku-20240307",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


# Embedding function (placeholder, as Anthropic does not provide embeddings)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def anthropic_embed(
    texts: list[str],
    model: str = "voyage-3",  # Default to voyage-3 as a good general-purpose model
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    """
    Generate embeddings using Voyage AI since Anthropic doesn't provide native embedding support.

    Args:
        texts: List of text strings to embed
        model: Voyage AI model name (e.g., "voyage-3", "voyage-3-large", "voyage-code-3")
        base_url: Optional custom base URL (not used for Voyage AI)
        api_key: API key for Voyage AI (defaults to VOYAGE_API_KEY environment variable)

    Returns:
        numpy array of shape (len(texts), embedding_dimension) containing the embeddings
    """
    if not api_key:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            logger.error("VOYAGE_API_KEY environment variable not set")
            raise ValueError(
                "VOYAGE_API_KEY environment variable is required for embeddings"
            )

    try:
        # Initialize Voyage AI client
        voyage_client = voyageai.Client(api_key=api_key)

        # Get embeddings
        result = voyage_client.embed(
            texts,
            model=model,
            input_type="document",  # Assuming document context; could be made configurable
        )

        # Convert list of embeddings to numpy array
        embeddings = np.array(result.embeddings, dtype=np.float32)

        logger.debug(f"Generated embeddings for {len(texts)} texts using {model}")
        verbose_debug(f"Embedding shape: {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"Voyage AI embedding failed: {str(e)}")
        raise


# Optional: a helper function to get available embedding models
def get_available_embedding_models() -> dict[str, dict]:
    """
    Returns a dictionary of available Voyage AI embedding models and their properties.
    """
    return {
        "voyage-3-large": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Best general-purpose and multilingual",
        },
        "voyage-3": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "General-purpose and multilingual",
        },
        "voyage-3-lite": {
            "context_length": 32000,
            "dimension": 512,
            "description": "Optimized for latency and cost",
        },
        "voyage-code-3": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Optimized for code",
        },
        "voyage-finance-2": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Optimized for finance",
        },
        "voyage-law-2": {
            "context_length": 16000,
            "dimension": 1024,
            "description": "Optimized for legal",
        },
        "voyage-multimodal-3": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Multimodal text and images",
        },
    }
