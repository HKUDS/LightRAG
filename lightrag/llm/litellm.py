from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# Install LiteLLM if not present
if not pm.is_installed("litellm"):
    pm.install("litellm")

import litellm
from litellm import acompletion, aembedding
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Any, Union


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


async def litellm_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using LiteLLM with caching support.

    LiteLLM provides a unified interface to 100+ LLM providers including:
    - OpenAI, Anthropic, Cohere, Replicate
    - Azure, AWS Bedrock, Google VertexAI
    - HuggingFace, Ollama, Together AI
    - And many more!

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "anthropic/claude-3-sonnet-20240229", "ollama/llama2")
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        enable_cot: Whether to enable Chain of Thought processing (supported by some providers).
        api_base: Optional base URL for the API (e.g., LiteLLM proxy server).
        api_key: Optional API key. If None, LiteLLM will use environment variables.
        token_tracker: Optional token usage tracker for monitoring API usage.
        **kwargs: Additional keyword arguments to pass to LiteLLM.
            Special kwargs:
            - hashing_kv: Will be removed from kwargs before passing to LiteLLM.
            - keyword_extraction: Will be removed from kwargs before passing to LiteLLM.
            - stream: Set to True to enable streaming responses.
            - temperature: Control randomness (0.0 to 2.0).
            - max_tokens: Maximum tokens in response.
            - top_p: Nucleus sampling parameter.

    Returns:
        The completed text or an async iterator of text chunks if streaming.

    Raises:
        InvalidResponseError: If the response from LiteLLM is invalid or empty.
        APIConnectionError: If there is a connection error with the API.
        RateLimitError: If the API rate limit is exceeded.
        APITimeoutError: If the API request times out.
    """
    if history_messages is None:
        history_messages = []

    if enable_cot:
        logger.debug(
            "enable_cot=True may not be supported by all LiteLLM providers and will be passed through."
        )

    # Set LiteLLM logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("litellm").setLevel(logging.INFO)

    # Remove special kwargs that shouldn't be passed to LiteLLM
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # Prepare messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Build LiteLLM kwargs
    litellm_kwargs = {
        "model": model,
        "messages": messages,
    }

    if api_base:
        litellm_kwargs["api_base"] = api_base
    if api_key:
        litellm_kwargs["api_key"] = api_key

    # Merge additional kwargs
    litellm_kwargs.update(kwargs)

    logger.debug("===== Entering func of LLM (LiteLLM) =====")
    logger.debug(f"Model: {model}   API Base: {api_base}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM (LiteLLM) =====")

    try:
        response = await acompletion(**litellm_kwargs)

        # Handle streaming responses
        if hasattr(response, "__aiter__"):

            async def inner():
                try:
                    final_chunk_usage = None

                    async for chunk in response:
                        # Track usage from final chunk
                        if hasattr(chunk, "usage") and chunk.usage:
                            final_chunk_usage = chunk.usage
                            logger.debug(
                                f"Received usage info in streaming chunk: {chunk.usage}"
                            )

                        # Check if choices exists and is not empty
                        if not hasattr(chunk, "choices") or not chunk.choices:
                            logger.warning(f"Received chunk without choices: {chunk}")
                            continue

                        # Check if delta exists
                        if not hasattr(chunk.choices[0], "delta"):
                            continue

                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)

                        if content is not None and content != "":
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                    # Track token usage after streaming is complete
                    if token_tracker and final_chunk_usage:
                        token_counts = {
                            "prompt_tokens": getattr(
                                final_chunk_usage, "prompt_tokens", 0
                            ),
                            "completion_tokens": getattr(
                                final_chunk_usage, "completion_tokens", 0
                            ),
                            "total_tokens": getattr(
                                final_chunk_usage, "total_tokens", 0
                            ),
                        }
                        token_tracker.add_usage(token_counts)
                        logger.debug(
                            f"Streaming token usage (from API): {token_counts}"
                        )
                    elif token_tracker:
                        logger.debug(
                            "No usage information available in streaming response"
                        )

                except Exception as e:
                    logger.error(f"Error in LiteLLM stream response: {str(e)}")
                    raise

            return inner()

        # Non-streaming response
        else:
            if (
                not response
                or not response.choices
                or not hasattr(response.choices[0], "message")
            ):
                logger.error("Invalid response from LiteLLM API")
                raise InvalidResponseError("Invalid response from LiteLLM API")

            message = response.choices[0].message
            content = getattr(message, "content", None)

            if not content or content.strip() == "":
                logger.error("Received empty content from LiteLLM API")
                raise InvalidResponseError("Received empty content from LiteLLM API")

            # Apply Unicode decoding if needed
            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))

            # Track token usage
            if token_tracker and hasattr(response, "usage"):
                token_counts = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                token_tracker.add_usage(token_counts)

            logger.debug(f"Response content len: {len(content)}")
            verbose_debug(f"Response: {response}")

            return content

    except Exception as e:
        # Log and re-raise exceptions
        # LiteLLM already raises appropriate exceptions that will be caught by retry logic
        logger.error(
            f"LiteLLM API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        raise


async def litellm_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Wrapper function that uses model name from global config."""
    if history_messages is None:
        history_messages = []

    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        # Use json_object format for LiteLLM (supports multiple providers)
        kwargs["response_format"] = {"type": "json_object"}

    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await litellm_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536)
async def litellm_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_base: str = None,
    api_key: str = None,
    **kwargs,
) -> np.ndarray:
    """Generate embeddings for a list of texts using LiteLLM.

    LiteLLM supports embedding models from multiple providers:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - Cohere: embed-english-v3.0, embed-multilingual-v3.0
    - Azure: azure/<deployment-name>
    - Bedrock: bedrock/<model-name>
    - And more!

    Args:
        texts: List of texts to embed.
        model: The embedding model to use. Format depends on provider:
            - OpenAI: "text-embedding-3-small"
            - Cohere: "cohere/embed-english-v3.0"
            - Azure: "azure/<deployment-name>"
            - Bedrock: "bedrock/cohere.embed-english-v3"
        api_base: Optional base URL for the API.
        api_key: Optional API key. If None, LiteLLM uses environment variables.
        **kwargs: Additional keyword arguments for the embedding call.

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        APIConnectionError: If there is a connection error with the API.
        RateLimitError: If the API rate limit is exceeded.
        APITimeoutError: If the API request times out.
    """
    try:
        litellm_kwargs = {
            "model": model,
            "input": texts,
        }

        if api_base:
            litellm_kwargs["api_base"] = api_base
        if api_key:
            litellm_kwargs["api_key"] = api_key

        litellm_kwargs.update(kwargs)

        response = await aembedding(**litellm_kwargs)

        # Extract embeddings from response
        embeddings = [item["embedding"] for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    except Exception as e:
        logger.error(f"LiteLLM Embedding failed: {e}")
        raise