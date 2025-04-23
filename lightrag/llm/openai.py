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

from dotenv import load_dotenv

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
    **kwargs: Any,
) -> str:
    """Complete a prompt using OpenAI's API with caching support.

    Args:
        model: The OpenAI model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
            Special kwargs:
            - openai_client_configs: Dict of configuration options for the AsyncOpenAI client.
                These will be passed to the client constructor but will be overridden by
                explicit parameters (api_key, base_url).
            - hashing_kv: Will be removed from kwargs before passing to OpenAI.
            - keyword_extraction: Will be removed from kwargs before passing to OpenAI.

    Returns:
        The completed text or an async iterator of text chunks if streaming.

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
                    # Check if choices exists and is not empty
                    if not hasattr(chunk, "choices") or not chunk.choices:
                        logger.warning(f"Received chunk without choices: {chunk}")
                        continue

                    # Check if delta exists and has content
                    if not hasattr(chunk.choices[0], "delta") or not hasattr(
                        chunk.choices[0].delta, "content"
                    ):
                        logger.warning(
                            f"Received chunk without delta content: {chunk.choices[0]}"
                        )
                        continue
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
