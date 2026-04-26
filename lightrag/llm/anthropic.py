from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging
from typing import Any, Union, AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

# Install Anthropic SDK if not present
if not pm.is_installed("anthropic"):
    pm.install("anthropic")

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
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Sending Query to Anthropic LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    verbose_debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    try:
        create_params = {"model": model, "messages": messages, "stream": True, **kwargs}
        if system_prompt:
            create_params["system"] = system_prompt
        response = await anthropic_async_client.messages.create(**create_params)

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
        body = getattr(e, "body", None)
        request_id = getattr(e, "request_id", None)
        req = getattr(e, "request", None)
        extra_parts = []
        if body:
            extra_parts.append(f"Response body: {body}")
        if request_id:
            extra_parts.append(f"Request ID: {request_id}")
        if req is not None:
            extra_parts.append(f"Request URL: {req.url}")
        extra = ("\n" + "\n".join(extra_parts)) if extra_parts else ""
        logger.error(
            f"Anthropic API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}{extra}"
        )
        raise

    async def stream_response():
        try:
            async for event in response:
                content = (
                    event.delta.text
                    if hasattr(event, "delta")
                    and hasattr(event.delta, "text")
                    and event.delta.text
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


# Backward-compatibility shim: the previous embedding implementation lived in
# this module under the (misleading) name ``anthropic_embed`` even though it
# called Voyage AI under the hood. The real implementation now lives in
# ``lightrag.llm.voyageai.voyageai_embed``. Keep the old name importable for one
# release cycle so downstream users get a clear deprecation warning instead of
# an ImportError. Remove in a future major version.
def anthropic_embed(*args, **kwargs):
    """Deprecated alias for :func:`lightrag.llm.voyageai.voyageai_embed`.

    This shim accepts the same arguments as the original ``anthropic_embed``
    function (which was always backed by VoyageAI) and forwards them to
    :func:`voyageai_embed`. It will be removed in a future release.
    """
    import warnings

    warnings.warn(
        "lightrag.llm.anthropic.anthropic_embed is deprecated and will be "
        "removed in a future release. Import "
        "lightrag.llm.voyageai.voyageai_embed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from lightrag.llm.voyageai import voyageai_embed

    return voyageai_embed.func(*args, **kwargs)
