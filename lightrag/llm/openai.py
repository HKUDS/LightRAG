from ..utils import verbose_debug, VERBOSE_DEBUG
import os
import logging

from collections.abc import AsyncIterator

import pipmaster as pm

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
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
    safe_unicode_decode,
    logger,
)

from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__

import numpy as np
import base64
from typing import Any, Union

from dotenv import load_dotenv

# Try to import Langfuse for LLM observability (optional)
# Falls back to standard OpenAI client if not available
# Langfuse requires proper configuration to work correctly
LANGFUSE_ENABLED = False
try:
    # Check if required Langfuse environment variables are set
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    # Only enable Langfuse if both keys are configured
    if langfuse_public_key and langfuse_secret_key:
        from langfuse.openai import AsyncOpenAI

        LANGFUSE_ENABLED = True
        logger.info("Langfuse observability enabled for OpenAI client")
    else:
        from openai import AsyncOpenAI

        logger.debug(
            "Langfuse environment variables not configured, using standard OpenAI client"
        )
except ImportError:
    from openai import AsyncOpenAI

    logger.debug("Langfuse not available, using standard OpenAI client")

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        An AsyncOpenAI client instance.
    """
    if not api_key:
        api_key = os.environ["OPENAI_API_KEY"]

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
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )

    return AsyncOpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    **kwargs: Any,
) -> str:
    """Complete a prompt using OpenAI's API with caching support and Chain of Thought (COT) integration.

    This function supports automatic integration of reasoning content from models that provide
    Chain of Thought capabilities. The reasoning content is seamlessly integrated into the response
    using <think>...</think> tags.

    Note on `reasoning_content`: This feature relies on a Deepseek Style `reasoning_content`
    in the API response, which may be provided by OpenAI-compatible endpoints that support
    Chain of Thought.

    COT Integration Rules:
    1. COT content is accepted only when regular content is empty and `reasoning_content` has content.
    2. COT processing stops when regular content becomes available.
    3. If both `content` and `reasoning_content` are present simultaneously, reasoning is ignored.
    4. If both fields have content from the start, COT is never activated.
    5. For streaming: COT content is inserted into the content stream with <think> tags.
    6. For non-streaming: COT content is prepended to regular content with <think> tags.

    Args:
        model: The OpenAI model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        token_tracker: Optional token usage tracker for monitoring API usage.
        enable_cot: Whether to enable Chain of Thought (COT) processing. Default is False.
        stream: Whether to stream the response. Default is False.
        timeout: Request timeout in seconds. Default is None.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
            Special kwargs:
            - openai_client_configs: Dict of configuration options for the AsyncOpenAI client.
                These will be passed to the client constructor but will be overridden by
                explicit parameters (api_key, base_url).
            - keyword_extraction: Will be removed from kwargs before passing to OpenAI.

    Returns:
        The completed text (with integrated COT content if available) or an async iterator
        of text chunks if streaming. COT content is wrapped in <think>...</think> tags.

    Raises:
        InvalidResponseError: If the response from OpenAI is invalid or empty.
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    if history_messages is None:
        history_messages = []

    # Set openai logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    # Remove special kwargs that shouldn't be passed to OpenAI
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # Extract client configuration options
    client_configs = kwargs.pop("openai_client_configs", {})

    # Create the OpenAI client
    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        client_configs=client_configs,
    )

    # Prepare messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Entering func of LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Client Configs: {client_configs}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    messages = kwargs.pop("messages", messages)

    # Add explicit parameters back to kwargs so they're passed to OpenAI API
    if stream is not None:
        kwargs["stream"] = stream
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        # Don't use async with context manager, use client directly
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {e}")
        await openai_async_client.close()  # Ensure client is closed
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Error: {e}")
        await openai_async_client.close()  # Ensure client is closed
        raise
    except APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {e}")
        await openai_async_client.close()  # Ensure client is closed
        raise
    except Exception as e:
        logger.error(
            f"OpenAI API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        await openai_async_client.close()  # Ensure client is closed
        raise

    if hasattr(response, "__aiter__"):

        async def inner():
            # Track if we've started iterating
            iteration_started = False
            final_chunk_usage = None

            # COT (Chain of Thought) state tracking
            cot_active = False
            cot_started = False
            initial_content_seen = False

            try:
                iteration_started = True
                async for chunk in response:
                    # Check if this chunk has usage information (final chunk)
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
                        # This might be the final chunk, continue to check for usage
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    reasoning_content = getattr(delta, "reasoning_content", "")

                    # Handle COT logic for streaming (only if enabled)
                    if enable_cot:
                        if content:
                            # Regular content is present
                            if not initial_content_seen:
                                initial_content_seen = True
                                # If both content and reasoning_content are present initially, don't start COT
                                if reasoning_content:
                                    cot_active = False
                                    cot_started = False

                            # If COT was active, end it
                            if cot_active:
                                yield "</think>"
                                cot_active = False

                            # Process regular content
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                        elif reasoning_content:
                            # Only reasoning content is present
                            if not initial_content_seen and not cot_started:
                                # Start COT if we haven't seen initial content yet
                                if not cot_active:
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True

                            # Process reasoning content if COT is active
                            if cot_active:
                                if r"\u" in reasoning_content:
                                    reasoning_content = safe_unicode_decode(
                                        reasoning_content.encode("utf-8")
                                    )
                                yield reasoning_content
                    else:
                        # COT disabled, only process regular content
                        if content:
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                    # If neither content nor reasoning_content, continue to next chunk
                    if content is None and reasoning_content is None:
                        continue

                # Ensure COT is properly closed if still active after stream ends
                if enable_cot and cot_active:
                    yield "</think>"
                    cot_active = False

                # After streaming is complete, track token usage
                if token_tracker and final_chunk_usage:
                    # Use actual usage from the API
                    token_counts = {
                        "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            final_chunk_usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)
                    logger.debug(f"Streaming token usage (from API): {token_counts}")
                elif token_tracker:
                    logger.debug("No usage information available in streaming response")
            except Exception as e:
                # Ensure COT is properly closed before handling exception
                if enable_cot and cot_active:
                    try:
                        yield "</think>"
                        cot_active = False
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close COT tag during exception handling: {close_error}"
                        )

                logger.error(f"Error in stream response: {str(e)}")
                # Try to clean up resources if possible
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(getattr(response, "aclose", None))
                ):
                    try:
                        await response.aclose()
                        logger.debug("Successfully closed stream response after error")
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close stream response: {close_error}"
                        )
                # Ensure client is closed in case of exception
                await openai_async_client.close()
                raise
            finally:
                # Final safety check for unclosed COT tags
                if enable_cot and cot_active:
                    try:
                        yield "</think>"
                        cot_active = False
                    except Exception as final_close_error:
                        logger.warning(
                            f"Failed to close COT tag in finally block: {final_close_error}"
                        )

                # Ensure resources are released even if no exception occurs
                # Note: Some wrapped clients (e.g., Langfuse) may not implement aclose() properly
                if iteration_started and hasattr(response, "aclose"):
                    aclose_method = getattr(response, "aclose", None)
                    if callable(aclose_method):
                        try:
                            await response.aclose()
                            logger.debug("Successfully closed stream response")
                        except (AttributeError, TypeError) as close_error:
                            # Some wrapper objects may report hasattr(aclose) but fail when called
                            # This is expected behavior for certain client wrappers
                            logger.debug(
                                f"Stream response cleanup not supported by client wrapper: {close_error}"
                            )
                        except Exception as close_error:
                            logger.warning(
                                f"Unexpected error during stream response cleanup: {close_error}"
                            )

                # This prevents resource leaks since the caller doesn't handle closing
                try:
                    await openai_async_client.close()
                    logger.debug(
                        "Successfully closed OpenAI client for streaming response"
                    )
                except Exception as client_close_error:
                    logger.warning(
                        f"Failed to close OpenAI client in streaming finally block: {client_close_error}"
                    )

        return inner()

    else:
        try:
            if (
                not response
                or not response.choices
                or not hasattr(response.choices[0], "message")
            ):
                logger.error("Invalid response from OpenAI API")
                await openai_async_client.close()  # Ensure client is closed
                raise InvalidResponseError("Invalid response from OpenAI API")

            message = response.choices[0].message
            content = getattr(message, "content", None)
            reasoning_content = getattr(message, "reasoning_content", "")

            # Handle COT logic for non-streaming responses (only if enabled)
            final_content = ""

            if enable_cot:
                # Check if we should include reasoning content
                should_include_reasoning = False
                if reasoning_content and reasoning_content.strip():
                    if not content or content.strip() == "":
                        # Case 1: Only reasoning content, should include COT
                        should_include_reasoning = True
                        final_content = (
                            content or ""
                        )  # Use empty string if content is None
                    else:
                        # Case 3: Both content and reasoning_content present, ignore reasoning
                        should_include_reasoning = False
                        final_content = content
                else:
                    # No reasoning content, use regular content
                    final_content = content or ""

                # Apply COT wrapping if needed
                if should_include_reasoning:
                    if r"\u" in reasoning_content:
                        reasoning_content = safe_unicode_decode(
                            reasoning_content.encode("utf-8")
                        )
                    final_content = f"<think>{reasoning_content}</think>{final_content}"
            else:
                # COT disabled, only use regular content
                final_content = content or ""

            # Validate final content
            if not final_content or final_content.strip() == "":
                logger.error("Received empty content from OpenAI API")
                await openai_async_client.close()  # Ensure client is closed
                raise InvalidResponseError("Received empty content from OpenAI API")

            # Apply Unicode decoding to final content if needed
            if r"\u" in final_content:
                final_content = safe_unicode_decode(final_content.encode("utf-8"))

            if token_tracker and hasattr(response, "usage"):
                token_counts = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                token_tracker.add_usage(token_counts)

            logger.debug(f"Response content len: {len(final_content)}")
            verbose_debug(f"Response: {response}")

            return final_content
        finally:
            # Ensure client is closed in all cases for non-streaming responses
            await openai_async_client.close()


async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    client_configs: dict[str, Any] | None = None,
    token_tracker: Any | None = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's API.

    Args:
        texts: List of texts to embed.
        model: The OpenAI embedding model to use.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).
        token_tracker: Optional token usage tracker for monitoring API usage.

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    # Create the OpenAI client
    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, client_configs=client_configs
    )

    async with openai_async_client:
        response = await openai_async_client.embeddings.create(
            model=model, input=texts, encoding_format="base64"
        )

        if token_tracker and hasattr(response, "usage"):
            token_counts = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            token_tracker.add_usage(token_counts)

        return np.array(
            [
                np.array(dp.embedding, dtype=np.float32)
                if isinstance(dp.embedding, list)
                else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
                for dp in response.data
            ]
        )
