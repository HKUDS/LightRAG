from collections.abc import AsyncIterator
import os
import re
import warnings

import pipmaster as pm

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")

import ollama

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
from lightrag.api import __api_version__

import numpy as np
from typing import Any, Optional, Union
from lightrag.utils import (
    TruncatedResponse,
    empty_length_truncated_hint,
    format_response_diagnostics,
    wrap_embedding_func_with_attrs,
    logger,
)


class InvalidResponseError(Exception):
    """A response that cannot be used, e.g. empty content.

    Declared per binding, as in ``lightrag.llm.openai`` / ``gemini`` /
    ``anthropic``: each provider owns its own so importing one binding never
    drags in another's import-time setup. Deliberately absent from the retry
    predicate below — see the raise site for why re-running is pointless here.
    """

    pass


_OLLAMA_CLOUD_HOST = "https://ollama.com"
_CLOUD_MODEL_SUFFIX_PATTERN = re.compile(r"(?:-cloud|:cloud)$")

# think= (OllamaLLMOptions.think) needs ollama-python>=0.5.3 on its own -- the
# version where ChatRequest.think widened from Optional[bool] to a Union with a
# Literal, so both the booleans and the named reasoning levels
# ("low"/"medium"/"high") serialize. The gate is pinned to the binding's
# declared floor (>=0.5.4, raised by ollama_embed forwarding dimensions=) rather
# than to think's own 0.5.3, so there is one ollama version to reason about
# instead of a per-feature matrix: an install below the declared floor is out of
# contract for this binding as a whole.
# This check is what enforces that floor where the declaration was never applied
# -- the auto-install at the top of this module installs a *missing* ollama but
# never upgrades an outdated one, so an in-place LightRAG upgrade can still be
# sitting on an old version. Installed-package metadata only (no network call),
# consulted solely from ensure_think_supported: an environment on an older
# ollama still imports and works normally for every call that doesn't set think.
_OLLAMA_SUPPORTS_THINK = pm.is_installed("ollama", ">=0.5.4")


def ensure_think_supported(options: Any, *, context: str = "") -> None:
    """Reject a think= option the installed ollama package cannot forward.

    Called from two places on purpose:

    - ``_ollama_model_if_cache``, right before the option is lifted out --
      the only defence library callers (who never go through the API server)
      get.
    - the API server's option-resolution chokepoints, so a misconfigured
      OLLAMA_LLM_THINK fails while the server is still starting up instead of
      mid-pipeline, hours into a document run.

    ``options`` that is not a dict, or carries no ``think`` key at all (e.g.
    the embedding options dict, which has no such field), is left alone: the
    model keeps its own thinking default and an older ollama stays usable.

    Only the installed *package* is checkable here. Whether the Ollama server
    is new enough for reasoning levels, and whether the model supports thinking
    at all, are answerable only by a live request and so stay runtime errors.
    """
    if not isinstance(options, dict) or "think" not in options:
        return
    if _OLLAMA_SUPPORTS_THINK:
        return
    where = f" for {context}" if context else ""
    raise RuntimeError(
        f"OLLAMA_LLM_THINK / {{ROLE}}_OLLAMA_LLM_THINK is set{where} to "
        f"{options['think']!r}, but the installed ollama package does not "
        'support think= (needs ollama>=0.5.4). Run `pip install -U "ollama'
        '>=0.5.4"` (or `uv sync`) to use it, or unset the option to leave '
        "thinking at the model's own default."
    )


def _coerce_host_for_cloud_model(host: Optional[str], model: object) -> Optional[str]:
    if host:
        return host
    try:
        model_name_str = str(model) if model is not None else ""
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to convert model to string: {e}, using empty string")
        model_name_str = ""
    if _CLOUD_MODEL_SUFFIX_PATTERN.search(model_name_str):
        logger.debug(
            f"Detected cloud model '{model_name_str}', using Ollama Cloud host"
        )
        return _OLLAMA_CLOUD_HOST
    return host


def _response_message_field(response: Any, field: str) -> Any:
    """Read ``message.<field>`` across the ollama response shapes.

    ollama<0.4 returns raw dicts, ollama>=0.4 a ChatResponse whose ``message``
    is a SubscriptableBaseModel; both expose ``.get``, and attribute access
    covers anything that does not. Used only for diagnostics, so an unexpected
    shape must degrade to ``None`` rather than raise over the failure the
    caller is in the middle of reporting.
    """
    try:
        message = response["message"]
    except Exception:
        return None
    getter = getattr(message, "get", None)
    if getter is None:
        return getattr(message, field, None)
    try:
        return getter(field)
    except Exception:
        return None


def _normalize_ollama_response_format(kwargs: dict) -> None:
    """Translate OpenAI-style response_format into Ollama's native format field.

    Precedence: an explicit ``format`` value (Ollama's native field) wins over
    ``response_format`` — if ``format`` is already set, ``response_format`` is
    dropped silently. Otherwise, ``{"type": "json_object"}`` maps to
    ``format="json"`` and any other payload is passed through unchanged so
    callers can supply JSON schemas directly.
    """

    response_format = kwargs.pop("response_format", None)
    if kwargs.get("format") is not None or response_format is None:
        return

    if isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            kwargs["format"] = "json"
            return
        if response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, dict):
                kwargs["format"] = json_schema.get("schema", json_schema)
                return

    # Fall back to passing through schema-like payloads for native Ollama support.
    kwargs["format"] = response_format


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    image_inputs: list[Any] | None = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Call Ollama chat API with OpenAI-style structured-output compatibility.

    Structured output note:
    - This adapter accepts OpenAI-style ``response_format`` and translates it
      to Ollama's native ``format`` field.
    - ``response_format={"type": "json_object"}`` maps to ``format="json"``.
    - Deprecated ``keyword_extraction`` and ``entity_extraction`` booleans are
      compatibility shims; when no explicit ``response_format`` is supplied,
      they are mapped to ``{"type": "json_object"}``.
    """
    if enable_cot:
        logger.debug("enable_cot=True is not supported for ollama and will be ignored.")
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # Deprecation shims: map legacy boolean flags to response_format only when
    # an explicit response_format was not supplied by the caller.
    if kwargs.get("response_format") is None:
        if kwargs.pop("entity_extraction", False):
            warnings.warn(
                "_ollama_model_if_cache(entity_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
        elif kwargs.pop("keyword_extraction", False):
            warnings.warn(
                "_ollama_model_if_cache(keyword_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
    else:
        # response_format was supplied explicitly; drop legacy flags silently.
        kwargs.pop("entity_extraction", None)
        kwargs.pop("keyword_extraction", None)

    _normalize_ollama_response_format(kwargs)

    # `think` (OllamaLLMOptions) travels in with the rest of the generation
    # options, but Ollama's chat() takes it as its own top-level argument,
    # not a key inside `options` -- lift it out here rather than at every
    # call site. Absent entirely (e.g. the embedding options dict, which has
    # no `think` field) leaves thinking at the model's own default.
    options = kwargs.get("options")
    ensure_think_supported(options)
    if isinstance(options, dict) and "think" in options:
        # Read without mutating -- options can be the same dict object
        # reused across every call for a role's lifetime (library callers
        # pass it once via llm_model_kwargs), so popping from it here would
        # only lift `think` out on the first call and silently lose it on
        # every call after that.
        kwargs["think"] = options["think"]
        kwargs["options"] = {k: v for k, v in options.items() if k != "think"}

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout == 0:
        timeout = None
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    # fallback to environment variable when not provided explicitly
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = _coerce_host_for_cloud_model(host, model)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_inputs:
            from lightrag.llm._vision_utils import normalize_image_inputs

            normalized_images = normalize_image_inputs(image_inputs)
            user_message["images"] = [img.base64_str for img in normalized_images]
        messages.append(user_message)

        response = await ollama_client.chat(model=model, messages=messages, **kwargs)
        if stream:
            """cannot cache stream response and process reasoning"""

            async def inner():
                try:
                    async for chunk in response:
                        yield chunk["message"]["content"]
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await ollama_client._client.aclose()
                        logger.debug("Successfully closed Ollama client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close Ollama client: {close_error}")

            return inner()
        else:
            model_response = response["message"]["content"]

            """
            If the model also wraps its thoughts in a specific tag,
            this information is not needed for the final
            response and can simply be trimmed.
            """

            # Flag token-limit truncation (done_reason == "length", num_predict
            # exhausted) so the cache layer skips persisting partial output.
            # .get works on both the raw dict (ollama<0.4) and the
            # SubscriptableBaseModel ChatResponse (ollama>=0.4); a missing key
            # returns None and keeps the previous cache-everything behavior.
            if response.get("done_reason") == "length":
                if not model_response or not model_response.strip():
                    # Empty AND cut off: nothing was generated at all, which is
                    # structurally broken rather than merely short. Returning ""
                    # here indexed an empty knowledge graph and still reported
                    # the document PROCESSED (issue #3601 gap 4, seen with
                    # thinking models burning the whole num_predict budget on
                    # the reasoning trace). Raise like the OpenAI binding does,
                    # so the document ends FAILED and stays retryable.
                    #
                    # Deliberately NOT added to the @retry set: unlike the
                    # OpenAI check — which also covers non-deterministic
                    # empty-content modes — this fires only on the token limit,
                    # and re-running the same prompt against the same budget
                    # would just burn three calls before failing anyway.
                    thinking = _response_message_field(response, "thinking") or ""
                    diagnostics = format_response_diagnostics(
                        done_reason="length",
                        eval_count=response.get("eval_count"),
                        prompt_eval_count=response.get("prompt_eval_count"),
                        thinking_len=len(thinking.strip()),
                    )
                    hint = empty_length_truncated_hint(
                        "consider raising OLLAMA_LLM_NUM_PREDICT or disabling "
                        "thinking mode",
                        reasoning_consumed_budget=bool(thinking.strip()),
                    )
                    error_message = (
                        f"Received empty content from Ollama API "
                        f"({diagnostics}): {hint}"
                    )
                    logger.error(error_message)
                    raise InvalidResponseError(error_message)
                logger.warning(
                    "Ollama response truncated by token limit "
                    f"(done_reason=length, content_len={len(model_response)}), returning partial content"
                )
                model_response = TruncatedResponse(model_response)

            return model_response
    except Exception as e:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception: {close_error}"
            )
        raise e
    finally:
        if not stream:
            try:
                await ollama_client._client.aclose()
                logger.debug(
                    "Successfully closed Ollama client for non-streaming response"
                )
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client in finally block: {close_error}"
                )


async def ollama_model_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    # Forward legacy extraction flags as kwargs so _ollama_model_if_cache can
    # emit a single DeprecationWarning with the correct stack frame.
    if keyword_extraction:
        kwargs.setdefault("keyword_extraction", True)
    if entity_extraction:
        kwargs.setdefault("entity_extraction", True)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
    model_name="bge-m3:latest",
    supports_asymmetric=True,
)
async def ollama_embed(
    texts: list[str],
    embed_model: str = "bge-m3:latest",
    max_token_size: int | None = None,
    context: str = "document",
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    embedding_dim: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Generate embeddings using Ollama's API.

    Args:
        texts: List of texts to embed.
        embed_model: The Ollama embedding model to use. Default is "bge-m3:latest".
        max_token_size: Maximum tokens per text. This parameter is automatically
            injected by the EmbeddingFunc wrapper when the underlying function
            signature supports it (via inspect.signature check). Ollama will
            automatically truncate texts exceeding the model's context length
            (num_ctx), so no client-side truncation is needed.
        context: The embedding context - "query" for search queries, "document" for indexed content.
            **IMPORTANT**: This parameter is automatically injected by the EmbeddingFunc wrapper
            when supports_asymmetric=True. Default is "document".
        query_prefix: Optional prefix to prepend to texts when context="query" (e.g., "search_query: ").
        document_prefix: Optional prefix to prepend to texts when context="document" (e.g., "search_document: ").
        embedding_dim: Optional target dimension. When set, forwarded to Ollama's
            embed API so the model actually returns a vector of that size. Kept as
            the last named parameter (before **kwargs) so existing positional
            callers of the pre-existing parameters are unaffected.
        **kwargs: Additional arguments passed to the Ollama client.

    Returns:
        A numpy array of embeddings, one per input text.

    Note:
        - Ollama API automatically truncates texts exceeding the model's context length
        - The max_token_size parameter is received but not used for client-side truncation
    """
    # Apply context-based prefixes if provided
    if context == "query" and query_prefix:
        texts = [query_prefix + text for text in texts]
    elif context == "document" and document_prefix:
        texts = [document_prefix + text for text in texts]

    # Note: max_token_size is received but not used for client-side truncation.
    # Ollama API handles truncation automatically based on the model's num_ctx setting.
    _ = max_token_size  # Acknowledge parameter to avoid unused variable warning
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)

    host = _coerce_host_for_cloud_model(host, embed_model)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    try:
        options = kwargs.pop("options", {})
        embed_kwargs = {"model": embed_model, "input": texts, "options": options}
        # mirrors lightrag/llm/openai.py's api_params["dimensions"] handling
        if embedding_dim is not None:
            embed_kwargs["dimensions"] = embedding_dim
        data = await ollama_client.embed(**embed_kwargs)
        return np.array(data["embeddings"])
    except Exception as e:
        logger.error(f"Error in ollama_embed: {str(e)}")
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception in embed")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception in embed: {close_error}"
            )
        raise e
    finally:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after embed")
        except Exception as close_error:
            logger.warning(f"Failed to close Ollama client after embed: {close_error}")
