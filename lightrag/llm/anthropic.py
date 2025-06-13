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

# ⚡ IMPORTING FROM: lightrag.py ⚡
from lightrag.utils import (
    safe_unicode_decode,
    logger,
)
from lightrag.api import __api_version__
from typing import Any, Optional, List, Dict  # Added imports
from ..base import BaseLLM  # Import the base class


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
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }

    # Set logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("anthropic").setLevel(logging.INFO)

    anthropic_async_client = (
        AsyncAnthropic(default_headers=default_headers, api_key=api_key)
        if base_url is None
        else AsyncAnthropic(
            base_url=base_url, default_headers=default_headers, api_key=api_key
        )
    )
    kwargs.pop("hashing_kv", None)
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
                    if hasattr(event, "delta") and event.delta.text
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
        **kwargs,
    )


# Claude 3 Opus specific completion
async def claude_3_opus_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-3-opus-20240229",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Claude 3 Sonnet specific completion
async def claude_3_sonnet_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-3-sonnet-20240229",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Claude 3 Haiku specific completion
async def claude_3_haiku_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-3-haiku-20240307",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Embedding function (updated to fully support all Voyage 3 models and features)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def anthropic_embed(
    texts: list[str],
    model: str = "voyage-large-3",
    base_url: str = None,
    api_key: str = None,
    input_type: str = "document",
    truncation: bool = True,
    output_dimension: int = None,
    output_dtype: str = "float",
) -> np.ndarray:
    """
    Generate embeddings using Voyage AI embedding models.

    Args:
        texts: List of text strings to embed
        model: Voyage AI model name. Supported models include:
               - voyage-3-large (best general-purpose, 1024 default dims)
               - voyage-3 (balanced general-purpose, 1024 dims)
               - voyage-3-lite (optimized for latency & cost, 512 dims)
               - voyage-code-3 (optimized for code, 1024 default dims)
               - voyage-finance-2 (optimized for finance, 1024 dims)
               - voyage-law-2 (optimized for legal, 1024 dims)
        base_url: Optional custom base URL (not used by Voyage AI client)
        api_key: API key for Voyage AI (defaults to VOYAGE_API_KEY environment variable)
        input_type: Type of the input. Options: None, "query", "document"
        truncation: Whether to truncate inputs that exceed context length
        output_dimension: Dimension of output embeddings.
                          voyage-3-large and voyage-code-3 support: 256, 512, 1024 (default), 2048
        output_dtype: Data type for embeddings. Options: "float", "int8", "uint8", "binary", "ubinary"
                     (int8, uint8, binary, ubinary only supported by voyage-3-large and voyage-code-3)

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
        logger.debug(f"Creating Voyage AI client for model: {model}")
        voyage_client = voyageai.Client(api_key=api_key)

        # Set up embedding parameters
        embed_params = {
            "model": model,
            "input_type": input_type,
            "truncation": truncation,
        }

        # Add optional parameters if specified
        if output_dimension is not None:
            # Verify output_dimension is valid for the model
            if model in ["voyage-3-large", "voyage-code-3"]:
                valid_dims = [256, 512, 1024, 2048]
                if output_dimension not in valid_dims:
                    logger.warning(
                        f"Invalid output_dimension {output_dimension} for {model}. Using default."
                    )
                else:
                    embed_params["output_dimension"] = output_dimension
            else:
                logger.warning(
                    f"Model {model} doesn't support custom dimensions. Using default."
                )

        # Set output dtype if it's a valid option
        valid_dtypes = ["float"]
        if model in ["voyage-3-large", "voyage-code-3"]:
            valid_dtypes.extend(["int8", "uint8", "binary", "ubinary"])

        if output_dtype in valid_dtypes:
            embed_params["output_dtype"] = output_dtype
        else:
            logger.warning(
                f"Output dtype {output_dtype} not supported for {model}. Using float."
            )

        logger.debug(f"Embedding parameters: {embed_params}")

        # Get embeddings
        result = voyage_client.embed(texts, **embed_params)

        # Convert to numpy array
        embeddings = np.array(result.embeddings, dtype=np.float32)

        logger.debug(f"Generated embeddings for {len(texts)} texts using {model}")
        logger.debug(f"Total tokens processed: {result.total_tokens}")
        verbose_debug(f"Embedding shape: {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"Voyage AI embedding failed: {str(e)}")
        raise


# Updated to include all available Voyage models as of 2025
def get_available_embedding_models() -> dict[str, dict]:
    """
    Returns a dictionary of available Voyage AI embedding models and their properties.
    """
    return {
        "voyage-3-large": {
            "context_length": 32000,
            "dimensions": [256, 512, 1024, 2048],  # 1024 is default
            "output_dtypes": ["float", "int8", "uint8", "binary", "ubinary"],
            "description": "Best general-purpose and multilingual",
        },
        "voyage-3": {
            "context_length": 32000,
            "dimensions": [1024],
            "output_dtypes": ["float"],
            "description": "Optimized for general-purpose and multilingual retrieval",
        },
        "voyage-3-lite": {
            "context_length": 32000,
            "dimensions": [512],
            "output_dtypes": ["float"],
            "description": "Optimized for latency and cost",
        },
        "voyage-code-3": {
            "context_length": 32000,
            "dimensions": [256, 512, 1024, 2048],  # 1024 is default
            "output_dtypes": ["float", "int8", "uint8", "binary", "ubinary"],
            "description": "Optimized for code retrieval",
        },
        "voyage-finance-2": {
            "context_length": 32000,
            "dimensions": [1024],
            "output_dtypes": ["float"],
            "description": "Optimized for finance retrieval and RAG",
        },
        "voyage-law-2": {
            "context_length": 16000,
            "dimensions": [1024],
            "output_dtypes": ["float"],
            "description": "Optimized for legal retrieval and RAG",
        },
    }


# Define the AnthropicLLM class
class AnthropicLLM(BaseLLM):
    """
    Implementation of the BaseLLM interface for Anthropic models.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        base_url: str | None = None,
        token_tracker: Any | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the AnthropicLLM client.

        Args:
            model: The Anthropic model ID to use (e.g., "claude-3-haiku-20240307").
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            base_url: Optional base URL for the Anthropic API.
            token_tracker: Optional token tracking object (Note: Anthropic API doesn't return token counts).
            **kwargs: Additional default arguments for the Anthropic API call.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.token_tracker = token_tracker  # Store but note Anthropic limitations
        self.default_kwargs = kwargs

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Anthropic API without history.

        Args:
            prompt: The prompt text to send to the model.
            **kwargs: Additional arguments to pass to the Anthropic API, overriding defaults.

        Returns:
            The generated text (as a string, handling potential streaming).
        """
        merged_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = await anthropic_complete_if_cache(
            model=self.model,
            prompt=prompt,
            system_prompt=None,
            history_messages=None,
            base_url=self.base_url,
            api_key=self.api_key,
            **merged_kwargs,
        )

        # Consume the async iterator to get the full string
        full_response = ""
        async for chunk in response_stream:
            full_response += chunk
        return full_response

    async def generate_with_history(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using the Anthropic API with conversation history.

        Args:
            prompt: The prompt text to send to the model.
            system_prompt: Optional system prompt to set context.
            history_messages: Optional conversation history.
            **kwargs: Additional arguments to pass to the Anthropic API, overriding defaults.

        Returns:
            The generated text (as a string, handling potential streaming).
        """
        merged_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = await anthropic_complete_if_cache(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=self.base_url,
            api_key=self.api_key,
            **merged_kwargs,
        )

        # Consume the async iterator to get the full string
        full_response = ""
        async for chunk in response_stream:
            full_response += chunk
        return full_response
