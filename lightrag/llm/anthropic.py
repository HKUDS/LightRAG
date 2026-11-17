from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging
import warnings
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
from lightrag.utils import (
    safe_unicode_decode,
    logger,
    TruncatedResponse,
)
from lightrag.api import __api_version__


class InvalidResponseError(Exception):
    """Kept for import compatibility; this binding never raises it."""

    pass


# Retry ownership belongs to the Anthropic SDK, so there is no tenacity layer
# here: stacking one on top of the SDK loop multiplied the attempts (3x3) and
# overrode a server's ``x-should-retry: false``. The SDK loop obeys
# ``x-should-retry`` and ``retry-after`` and retries connection errors,
# timeouts, 408, 409, 429 and >= 500 (incl. 529 overloaded). Its default of
# two retries backs off only 0.5 s + 1 s when no ``retry-after`` arrives (5xx,
# connection errors, Anthropic-compatible endpoints), which is too short for a
# pipeline whose failed extraction call fails the whole document. Four retries
# back off 0.5 + 1 + 2 + 4 s, about the window the OpenAI binding's tenacity
# loop gives (4 s + 4 s). Each attempt gets the full ``timeout``; inside
# LightRAG's LLM worker queue the whole call is still capped at
# ``2 * llm_timeout``, so the larger budget cannot stretch a hung call.
#
# The OpenAI binding resolved the same double loop the other way
# (``max_retries=0``, tenacity owns retries) so that a predicate can fail
# permanent spend-stop 429s fast, which the body-blind SDK loop cannot. This
# binding has no such classification to protect, and the SDK's
# ``x-should-retry`` / ``retry-after`` handling is worth keeping.
_DEFAULT_MAX_RETRIES = 4


async def anthropic_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    image_inputs: list[Any] | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Call Anthropic Messages API with LightRAG-compatible shims.

    Retries:
    - The Anthropic SDK is the only retry layer (see the comment on
      ``_DEFAULT_MAX_RETRIES``). Pass ``max_retries`` to override its budget
      per call; ``0`` disables retries.
    - When the budget is exhausted, the SDK's own exception
      (``RateLimitError``, ``APIStatusError``, ``APIConnectionError``, ...)
      propagates unchanged, never a ``tenacity.RetryError``.

    Structured output note:
    - This adapter does not support OpenAI-style ``response_format`` JSON mode.
    - If callers pass ``response_format``, it is stripped before the request.
    - Deprecated ``keyword_extraction`` and ``entity_extraction`` booleans are
      accepted only as compatibility shims; they emit warnings and are ignored.
    """
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
    # Anthropic Messages API has no JSON mode; drop legacy flags and
    # response_format. Emit DeprecationWarning when the booleans were set.
    if kwargs.pop("keyword_extraction", False):
        warnings.warn(
            "anthropic_complete_if_cache(keyword_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if kwargs.pop("entity_extraction", False):
        warnings.warn(
            "anthropic_complete_if_cache(entity_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    kwargs.pop("response_format", None)
    timeout = kwargs.pop("timeout", None)
    # Client option, not a ``messages.create`` parameter.
    max_retries = kwargs.pop("max_retries", None)
    if max_retries is None:
        max_retries = _DEFAULT_MAX_RETRIES

    # Require max_tokens; the Anthropic SDK errors if it's missing
    kwargs.setdefault("max_tokens", 8192)
    # Pop stream from kwargs so it doesn't leak into create_params;
    # default to False (non-streaming) for consistency with other providers
    stream = kwargs.pop("stream", False)

    anthropic_async_client = (
        AsyncAnthropic(
            default_headers=default_headers,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        if base_url is None
        else AsyncAnthropic(
            base_url=base_url,
            default_headers=default_headers,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
    )

    messages: list[dict[str, Any]] = []
    messages.extend(history_messages)
    if image_inputs:
        from lightrag.llm._vision_utils import normalize_image_inputs

        normalized_images = normalize_image_inputs(image_inputs)
        user_content: list[dict[str, Any]] = []
        for img in normalized_images:
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type,
                        "data": img.base64_str,
                    },
                }
            )
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": prompt})

    logger.debug("===== Sending Query to Anthropic LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    verbose_debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    try:
        create_params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if system_prompt:
            create_params["system"] = system_prompt
        response = await anthropic_async_client.messages.create(**create_params)
    except APITimeoutError as e:
        # APITimeoutError subclasses APIConnectionError, so it must come first
        # or it would be swallowed by the broader handler below.
        logger.error(f"Anthropic API Timeout Error: {e}")
        try:
            await anthropic_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Anthropic client: {close_error}")
        raise
    except APIConnectionError as e:
        logger.error(f"Anthropic API Connection Error: {e}")
        try:
            await anthropic_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Anthropic client: {close_error}")
        raise
    except RateLimitError as e:
        logger.error(f"Anthropic API Rate Limit Error: {e}")
        try:
            await anthropic_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Anthropic client: {close_error}")
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
        try:
            await anthropic_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Anthropic client: {close_error}")
        raise
    except BaseException:
        # CancelledError (and other BaseExceptions) aren't caught above, yet a
        # cancellation while awaiting create() — before we hold the response or
        # stream — would otherwise leak the client. Close it, then re-raise the
        # cancellation untouched (cooperative cancellation).
        try:
            await anthropic_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Anthropic client: {close_error}")
        raise

    if not stream:
        try:
            # Record usage before extracting content text: the API call is
            # already billed and response.usage is already populated at this
            # point, but content[0].text raises if the first content block
            # isn't a text block (or content is empty) -- that must not cost
            # the tracker a real, billed call.
            if token_tracker and getattr(response, "usage", None):
                usage = response.usage
                prompt_tokens = sum(
                    getattr(usage, field, 0) or 0
                    for field in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                )
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                token_tracker.add_usage(
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    }
                )
            content = response.content[0].text
            if getattr(response, "stop_reason", None) == "max_tokens":
                content = TruncatedResponse(content)
            return content
        finally:
            try:
                await anthropic_async_client.close()
            except Exception as close_error:
                logger.warning(f"Failed to close Anthropic client: {close_error}")

    async def stream_response():
        usage_counts: dict[str, int] = {}
        try:
            async for event in response:
                usage = getattr(event, "usage", None)
                if usage is None:
                    usage = getattr(getattr(event, "message", None), "usage", None)
                if usage is not None:
                    for field in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    ):
                        value = getattr(usage, field, None)
                        if value is not None:
                            usage_counts[field] = value
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
            if token_tracker and usage_counts:
                prompt_tokens = sum(
                    usage_counts.get(field, 0)
                    for field in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                )
                output_tokens = usage_counts.get("output_tokens", 0)
                token_tracker.add_usage(
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    }
                )
        except Exception as e:
            logger.error(f"Error in stream response: {str(e)}")
            raise
        finally:
            # Release the streaming response and the client. The generator owns
            # the client lifetime, so the caller never closes it; finally also
            # runs on GeneratorExit (early break) which the except clause above
            # cannot catch. AsyncStream exposes close() (not aclose()).
            #
            # The client close lives in an outer finally so it still runs if
            # response.close() is interrupted by CancelledError (a BaseException,
            # not caught by `except Exception`) — otherwise task cancellation /
            # client disconnect during stream teardown would leak the client.
            try:
                try:
                    await response.close()
                except Exception as close_error:
                    logger.warning(f"Failed to close Anthropic stream: {close_error}")
            finally:
                try:
                    await anthropic_async_client.close()
                except Exception as close_error:
                    logger.warning(f"Failed to close Anthropic client: {close_error}")

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

    warnings.warn(
        "lightrag.llm.anthropic.anthropic_embed is deprecated and will be "
        "removed in a future release. Import "
        "lightrag.llm.voyageai.voyageai_embed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from lightrag.llm.voyageai import voyageai_embed

    return voyageai_embed.func(*args, **kwargs)
