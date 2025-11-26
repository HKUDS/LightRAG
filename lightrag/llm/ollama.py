from collections.abc import AsyncIterator

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
from typing import Union
from lightrag.utils import logger


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
    token_tracker=None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if enable_cot:
        logger.debug("enable_cot=True is not supported for ollama and will be ignored.")
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout == 0:
        timeout = None
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await ollama_client.chat(model=model, messages=messages, **kwargs)
        if stream:
            """cannot cache stream response and process reasoning"""

            async def inner():
                accumulated_response = ""
                try:
                    async for chunk in response:
                        chunk_content = chunk["message"]["content"]
                        accumulated_response += chunk_content
                        yield chunk_content
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    # Track token usage for streaming if token tracker is provided
                    if token_tracker:
                        # Estimate prompt tokens: roughly 4 characters per token for English text
                        prompt_text = ""
                        if system_prompt:
                            prompt_text += system_prompt + " "
                        prompt_text += (
                            " ".join(
                                [msg.get("content", "") for msg in history_messages]
                            )
                            + " "
                        )
                        prompt_text += prompt
                        prompt_tokens = len(prompt_text) // 4 + (
                            1 if len(prompt_text) % 4 else 0
                        )

                        # Estimate completion tokens from accumulated response
                        completion_tokens = len(accumulated_response) // 4 + (
                            1 if len(accumulated_response) % 4 else 0
                        )
                        total_tokens = prompt_tokens + completion_tokens

                        token_tracker.add_usage(
                            {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                            }
                        )

                    try:
                        await ollama_client._client.aclose()
                        logger.debug("Successfully closed Ollama client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close Ollama client: {close_error}")

            return inner()
        else:
            model_response = response["message"]["content"]

            # Track token usage if token tracker is provided
            # Note: Ollama doesn't provide token usage in chat responses, so we estimate
            if token_tracker:
                # Estimate prompt tokens: roughly 4 characters per token for English text
                prompt_text = ""
                if system_prompt:
                    prompt_text += system_prompt + " "
                prompt_text += (
                    " ".join([msg.get("content", "") for msg in history_messages]) + " "
                )
                prompt_text += prompt
                prompt_tokens = len(prompt_text) // 4 + (
                    1 if len(prompt_text) % 4 else 0
                )

                # Estimate completion tokens from response
                completion_tokens = len(model_response) // 4 + (
                    1 if len(model_response) % 4 else 0
                )
                total_tokens = prompt_tokens + completion_tokens

                token_tracker.add_usage(
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            """
            If the model also wraps its thoughts in a specific tag,
            this information is not needed for the final
            response and can simply be trimmed.
            """

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
    token_tracker=None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        token_tracker=token_tracker,
        **kwargs,
    )


async def ollama_embed(
    texts: list[str], embed_model, token_tracker=None, **kwargs
) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    try:
        options = kwargs.pop("options", {})
        data = await ollama_client.embed(
            model=embed_model, input=texts, options=options
        )

        # Track token usage if token tracker is provided
        # Note: Ollama doesn't provide token usage in embedding responses, so we estimate
        if token_tracker:
            # Estimate tokens: roughly 4 characters per token for English text
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = total_chars // 4 + (1 if total_chars % 4 else 0)
            token_tracker.add_usage(
                {
                    "prompt_tokens": estimated_tokens,
                    "completion_tokens": 0,
                    "total_tokens": estimated_tokens,
                }
            )

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
