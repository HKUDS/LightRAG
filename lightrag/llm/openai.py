from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
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
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__

import numpy as np
from typing import Any, Union


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] = None,
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
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.environ["OPENAI_API_BASE"]

    return AsyncOpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    extract_reasoning: bool = False,
    **kwargs: Any,
) -> str | tuple[str, str]:
    """Complete a prompt using OpenAI's API with caching support.

    Args:
        model: The OpenAI model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        extract_reasoning: Whether to extract and return reasoning content when available.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
            Special kwargs:
            - openai_client_configs: Dict of configuration options for the AsyncOpenAI client.
                These will be passed to the client constructor but will be overridden by
                explicit parameters (api_key, base_url).
            - hashing_kv: Will be removed from kwargs before passing to OpenAI.
            - keyword_extraction: Will be removed from kwargs before passing to OpenAI.

    Returns:
        Either the completed text string, or a tuple of (completed_text, reasoning_content)
        if extract_reasoning is True or the model supports reasoning.

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

    # Extract client configuration options
    client_configs = kwargs.pop("openai_client_configs", {})

    # Create the OpenAI client
    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, client_configs=client_configs
    )

    # Remove special kwargs that shouldn't be passed to OpenAI
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # Prepare messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Entering func of LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    try:
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
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Error: {e}")
        raise
    except APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {e}")
        raise
    except Exception as e:
        logger.error(
            f"OpenAI API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        raise

    if hasattr(response, "__aiter__"):

        async def inner():
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))
                    yield content
            except Exception as e:
                logger.error(f"Error in stream response: {str(e)}")
                raise

        return inner()

    else:
        if (
            not response
            or not response.choices
            or not hasattr(response.choices[0], "message")
            or not hasattr(response.choices[0].message, "content")
        ):
            logger.error("Invalid response from OpenAI API")
            raise InvalidResponseError("Invalid response from OpenAI API")

        content = response.choices[0].message.content

        if not content or content.strip() == "":
            logger.error("Received empty content from OpenAI API")
            raise InvalidResponseError("Received empty content from OpenAI API")

        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))

        if token_tracker and hasattr(response, "usage"):
            token_counts = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            token_tracker.add_usage(token_counts)

        logger.debug(f"Response content len: {len(content)}")
        verbose_debug(f"Response: {response}")

        # Try to extract reasoning_content if requested or if model supports it
        reasoning_content = ""

        # First check: look for reasoning_content in the message object's attributes or _kwargs
        try:
            # Look directly in the message object
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning_content = response.choices[0].message.reasoning_content
                logger.info("Found reasoning_content in message attributes")
        except Exception as e:
            logger.warning(f"Error checking for reasoning_content: {e}")

        # Log the reasoning content if found
        if reasoning_content:
            logger.info("Successfully extracted chain of thought reasoning")
            logger.debug(f"Reasoning content: {reasoning_content}")
            print(f"==========Reasoning content==========: {reasoning_content}")
        elif extract_reasoning:
            logger.info("No reasoning content found, but extraction was requested")

        # Return tuple if reasoning was requested or found
        if extract_reasoning or reasoning_content:
            return content, reasoning_content

        return content


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
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
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
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def deepseek_r1_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> tuple[str, str]:
    """Complete a prompt using DeepSeek Reasoning-1 model.

    This model is specialized for reasoning and analytical tasks, making it
    useful for tasks like node re-ranking in knowledge graphs where reasoning
    about relevance and importance is required.

    Args:
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        keyword_extraction: Whether to extract keywords from the response.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.

    Returns:
        A tuple containing (completed_text, reasoning_content) where reasoning_content
        contains the chain of thought reasoning if available, otherwise empty string.
    """
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)

    # Ensure we have the right configuration for DeepSeek API
    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.environ.get("DEEPSEEK_API_KEY", None)

    # Create the OpenAI client directly to get more control over the response
    print("\n===== CALLING DEEPSEEK REASONING MODEL =====")
    client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        client_configs=kwargs.pop("openai_client_configs", {}),
    )

    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Make direct API call to get full response
    try:
        response = await client.chat.completions.create(
            model="deepseek-reasoner", messages=messages, **kwargs
        )

        # Extract the content from the response
        content = (
            response.choices[0].message.content
            if response.choices and hasattr(response.choices[0].message, "content")
            else ""
        )

        # Try to extract reasoning content
        reasoning_content = ""

        # Print the entire response for debugging
        print("\n===== DEEPSEEK API RESPONSE STRUCTURE =====")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        if hasattr(response, "choices") and response.choices:
            print(f"Message attributes: {dir(response.choices[0].message)}")

        # Try various ways to access reasoning_content
        try:
            # Direct access
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning_content = response.choices[0].message.reasoning_content
                print("\n===== FOUND REASONING CONTENT DIRECTLY =====")

            # Look in _kwargs dictionary
            elif hasattr(response.choices[0].message, "_kwargs"):
                kwargs_dict = response.choices[0].message._kwargs
                if "reasoning_content" in kwargs_dict:
                    reasoning_content = kwargs_dict["reasoning_content"]
                    print("\n===== FOUND REASONING CONTENT IN _KWARGS =====")

            # Check if it's in the model_dump
            elif hasattr(response, "model_dump"):
                dump = response.model_dump()
                print(f"\n===== MODEL DUMP KEYS =====\n{list(dump.keys())}")
                if "choices" in dump and dump["choices"]:
                    choice = dump["choices"][0]
                    if "message" in choice:
                        message = choice["message"]
                        if "reasoning_content" in message:
                            reasoning_content = message["reasoning_content"]
                            print("\n===== FOUND REASONING CONTENT IN MODEL_DUMP =====")

            # If we have reasoning content, print it
            if reasoning_content:
                print("\n===== CHAIN OF THOUGHT REASONING FROM DEEPSEEK =====")
                print(reasoning_content)
            else:
                # Try to extract reasoning from the content itself
                # If the content includes reasoning before JSON, try to separate it
                if not content.startswith("[") and "[" in content:
                    parts = content.split("[", 1)
                    if parts[0].strip():
                        reasoning_content = parts[0].strip()
                        content = "[" + parts[1]
                        print("\n===== EXTRACTED REASONING FROM CONTENT =====")
                        print(reasoning_content)

                if not reasoning_content:
                    print("\n===== NO REASONING CONTENT FOUND =====")
        except Exception as e:
            print(f"\n===== ERROR EXTRACTING REASONING CONTENT =====\n{str(e)}")

        if keyword_extraction and content:
            return locate_json_string_body_from_string(content), reasoning_content

        return content, reasoning_content

    except Exception as e:
        print(f"\n===== ERROR CALLING DEEPSEEK API =====\n{str(e)}")
        logger.error(f"Error calling DeepSeek API: {e}")
        return f"Error: {str(e)}", ""


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client_configs: dict[str, Any] = None,
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

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
