import copy
import json
import logging
import warnings

import pipmaster as pm  # Pipmaster for dynamic library install

if not pm.is_installed("aioboto3"):
    pm.install("aioboto3")
import aioboto3
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from collections.abc import AsyncIterator
from typing import Any, Union

from lightrag.utils import wrap_embedding_func_with_attrs

# Import botocore exceptions for proper exception handling
try:
    from botocore.exceptions import (
        ClientError,
        ConnectionError as BotocoreConnectionError,
        ReadTimeoutError,
    )
except ImportError:
    # If botocore is not installed, define placeholders
    ClientError = Exception
    BotocoreConnectionError = Exception
    ReadTimeoutError = Exception


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


class BedrockRateLimitError(BedrockError):
    """Error for rate limiting and throttling issues"""


class BedrockConnectionError(BedrockError):
    """Error for network and connection issues"""


class BedrockTimeoutError(BedrockError):
    """Error for timeout issues"""


def _normalize_bedrock_endpoint_url(endpoint_url: str | None) -> str | None:
    """Return a usable Bedrock endpoint override or None for SDK defaults."""
    if endpoint_url is None:
        return None

    normalized = endpoint_url.strip()
    if not normalized or normalized == "DEFAULT_BEDROCK_ENDPOINT":
        return None

    return normalized


def _bedrock_client_kwargs(
    region: str | None,
    endpoint_url: str | None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
) -> dict:
    """Build kwargs for aioboto3 ``session.client("bedrock-runtime", ...)``."""
    client_kwargs: dict = {"region_name": region}
    if endpoint_url is not None:
        client_kwargs["endpoint_url"] = endpoint_url
    if aws_access_key_id:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        client_kwargs["aws_session_token"] = aws_session_token
    return client_kwargs


def _handle_bedrock_exception(e: Exception, operation: str = "Bedrock API") -> None:
    """Convert AWS Bedrock exceptions to appropriate custom exceptions.

    Args:
        e: The exception to handle
        operation: Description of the operation for error messages

    Raises:
        BedrockRateLimitError: For rate limiting and throttling issues (retryable)
        BedrockConnectionError: For network and server issues (retryable)
        BedrockTimeoutError: For timeout issues (retryable)
        BedrockError: For validation and other non-retryable errors
    """
    error_message = str(e)

    # Handle botocore ClientError with specific error codes
    if isinstance(e, ClientError):
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", error_message)

        # Rate limiting and throttling errors (retryable)
        if error_code in [
            "ThrottlingException",
            "ProvisionedThroughputExceededException",
        ]:
            logging.error(f"{operation} rate limit error: {error_msg}")
            raise BedrockRateLimitError(f"Rate limit error: {error_msg}")

        # Server errors (retryable)
        elif error_code in ["ServiceUnavailableException", "InternalServerException"]:
            logging.error(f"{operation} connection error: {error_msg}")
            raise BedrockConnectionError(f"Service error: {error_msg}")

        # Check for 5xx HTTP status codes (retryable)
        elif e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) >= 500:
            logging.error(f"{operation} server error: {error_msg}")
            raise BedrockConnectionError(f"Server error: {error_msg}")

        # Validation and other client errors (non-retryable)
        else:
            logging.error(f"{operation} client error: {error_msg}")
            raise BedrockError(f"Client error: {error_msg}")

    # Connection errors (retryable)
    elif isinstance(e, BotocoreConnectionError):
        logging.error(f"{operation} connection error: {error_message}")
        raise BedrockConnectionError(f"Connection error: {error_message}")

    # Timeout errors (retryable)
    elif isinstance(e, (ReadTimeoutError, TimeoutError)):
        logging.error(f"{operation} timeout error: {error_message}")
        raise BedrockTimeoutError(f"Timeout error: {error_message}")

    # Custom Bedrock errors (already properly typed)
    elif isinstance(
        e,
        (
            BedrockRateLimitError,
            BedrockConnectionError,
            BedrockTimeoutError,
            BedrockError,
        ),
    ):
        raise

    # Unknown errors (non-retryable)
    else:
        logging.error(f"{operation} unexpected error: {error_message}")
        raise BedrockError(f"Unexpected error: {error_message}")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(BedrockRateLimitError)
        | retry_if_exception_type(BedrockConnectionError)
        | retry_if_exception_type(BedrockTimeoutError)
    ),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    aws_region: str | None = None,
    api_key: str | None = None,
    endpoint_url: str | None = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Call Amazon Bedrock Converse API with LightRAG-compatible shims.

    Structured output note:
    - This adapter does not support OpenAI-style ``response_format`` JSON mode.
    - If callers pass ``response_format``, it is stripped before the request.
    - Deprecated ``keyword_extraction`` and ``entity_extraction`` booleans are
      accepted only as compatibility shims; they emit warnings and are ignored.

    Authentication note:
    - Bedrock does not use LightRAG's generic ``api_key`` fields.
    - ``LLM_BINDING_API_KEY`` and ``EMBEDDING_BINDING_API_KEY`` are ignored for
      Bedrock.
    - To use Bedrock API key / bearer-token auth, set
      ``AWS_BEARER_TOKEN_BEDROCK`` before starting the process; this is a
      process-level AWS SDK setting.
    - For role-specific Bedrock LLMs, use explicit SigV4 parameters
      (``aws_access_key_id``, ``aws_secret_access_key``, ``aws_session_token``,
      ``aws_region``). Per-role bearer-token overrides are not supported.

    Endpoint note:
    - ``endpoint_url`` overrides the default regional Bedrock endpoint. Pass
      ``None``, an empty string, or the sentinel ``DEFAULT_BEDROCK_ENDPOINT``
      to let the AWS SDK select its default endpoint.
    """
    if enable_cot:
        import logging

        logging.debug(
            "enable_cot=True is not supported for Bedrock and will be ignored."
        )

    # Bedrock Converse API has no JSON mode; drop legacy extraction flags and
    # response_format below and rely on the prompt template plus downstream
    # tolerant JSON parsing.
    keyword_extraction = kwargs.pop("keyword_extraction", False)
    entity_extraction = kwargs.pop("entity_extraction", False)
    if keyword_extraction:
        warnings.warn(
            "bedrock_complete_if_cache(keyword_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if entity_extraction:
        warnings.warn(
            "bedrock_complete_if_cache(entity_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if api_key:
        warnings.warn(
            "bedrock_complete_if_cache(api_key=...) is ignored; use SigV4 "
            "parameters or set AWS_BEARER_TOKEN_BEDROCK before process start.",
            DeprecationWarning,
            stacklevel=2,
        )

    region = aws_region or kwargs.pop("aws_region", None)
    endpoint_url = _normalize_bedrock_endpoint_url(endpoint_url)
    kwargs.pop("hashing_kv", None)
    # Capture stream flag (if provided) and remove from kwargs since it's not a Bedrock API parameter
    # We'll use this to determine whether to call converse_stream or converse
    stream = bool(kwargs.pop("stream", False))
    # Remove unsupported args for Bedrock Converse API
    for k in [
        "response_format",
        "tools",
        "tool_choice",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "n",
        "logprobs",
        "top_logprobs",
        "max_completion_tokens",
    ]:
        kwargs.pop(k, None)
    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    inference_config: dict[str, Any] = {}
    for param in ("max_tokens", "temperature", "top_p", "stop_sequences"):
        if param not in kwargs:
            continue
        value = kwargs.pop(param)
        # Bedrock rejects None; a None default means "inherit provider default"
        if value is None:
            continue
        inference_config[inference_params_map.get(param, param)] = value
    if inference_config:
        args["inferenceConfig"] = inference_config

    # Pass-through for model-specific parameters (e.g. Anthropic reasoning_config,
    # Nova inferenceConfig extensions). Mirrors OpenAI's `extra_body`.
    extra_fields = kwargs.pop("extra_fields", None)
    if extra_fields:
        args["additionalModelRequestFields"] = extra_fields

    # Import logging for error handling
    import logging

    # For streaming responses, we need a different approach to keep the connection open
    if stream:
        # Create a session that will be used throughout the streaming process
        session = aioboto3.Session()
        client = None
        client_kwargs = _bedrock_client_kwargs(
            region,
            endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

        # Define the generator function that will manage the client lifecycle
        async def stream_generator():
            nonlocal client

            # Create the client outside the generator to ensure it stays open
            client = await session.client(
                "bedrock-runtime", **client_kwargs
            ).__aenter__()
            event_stream = None
            iteration_started = False

            try:
                # Make the API call
                response = await client.converse_stream(**args, **kwargs)
                event_stream = response.get("stream")
                iteration_started = True

                # Process the stream
                async for event in event_stream:
                    # Validate event structure
                    if not event or not isinstance(event, dict):
                        continue

                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        text = delta.get("text")
                        if text:
                            yield text
                    # Handle other event types that might indicate stream end
                    elif "messageStop" in event:
                        break

            except Exception as e:
                # Try to clean up resources if possible
                if (
                    iteration_started
                    and event_stream
                    and hasattr(event_stream, "aclose")
                    and callable(getattr(event_stream, "aclose", None))
                ):
                    try:
                        await event_stream.aclose()
                    except Exception as close_error:
                        logging.warning(
                            f"Failed to close Bedrock event stream: {close_error}"
                        )

                # Convert to appropriate exception type
                _handle_bedrock_exception(e, "Bedrock streaming")

            finally:
                # Clean up the event stream
                if (
                    iteration_started
                    and event_stream
                    and hasattr(event_stream, "aclose")
                    and callable(getattr(event_stream, "aclose", None))
                ):
                    try:
                        await event_stream.aclose()
                    except Exception as close_error:
                        logging.warning(
                            f"Failed to close Bedrock event stream in finally block: {close_error}"
                        )

                # Clean up the client
                if client:
                    try:
                        await client.__aexit__(None, None, None)
                    except Exception as client_close_error:
                        logging.warning(
                            f"Failed to close Bedrock client: {client_close_error}"
                        )

        # Return the generator that manages its own lifecycle
        return stream_generator()

    # For non-streaming responses, use the standard async context manager pattern
    session = aioboto3.Session()
    async with session.client(
        "bedrock-runtime",
        **_bedrock_client_kwargs(
            region,
            endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        ),
    ) as bedrock_async_client:
        try:
            # Use converse for non-streaming responses
            response = await bedrock_async_client.converse(**args, **kwargs)

            # Validate response structure
            if (
                not response
                or "output" not in response
                or "message" not in response["output"]
                or "content" not in response["output"]["message"]
                or not response["output"]["message"]["content"]
            ):
                raise BedrockError("Invalid response structure from Bedrock API")

            # When thinking/reasoning is enabled, the first content block is a
            # `reasoningContent` block and the visible text follows in a later
            # block. Pick the first block that carries a text payload.
            content = next(
                (
                    block["text"]
                    for block in response["output"]["message"]["content"]
                    if isinstance(block, dict) and block.get("text")
                ),
                None,
            )

            if not content or content.strip() == "":
                raise BedrockError("Received empty content from Bedrock API")

            return content

        except Exception as e:
            # Convert to appropriate exception type
            _handle_bedrock_exception(e, "Bedrock converse")


# Generic Bedrock completion function
async def bedrock_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    # Bedrock Converse API has no JSON mode; the shim booleans are absorbed
    # and forwarded so bedrock_complete_if_cache can emit DeprecationWarnings
    # with accurate stack frames.
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    result = await bedrock_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(
    embedding_dim=1024, max_token_size=8192, model_name="amazon.titan-embed-text-v2:0"
)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(BedrockRateLimitError)
        | retry_if_exception_type(BedrockConnectionError)
        | retry_if_exception_type(BedrockTimeoutError)
    ),
)
async def bedrock_embed(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    aws_region: str | None = None,
    api_key: str | None = None,
    endpoint_url: str | None = None,
) -> np.ndarray:
    """Generate embeddings with Amazon Bedrock Runtime.

    Authentication note:
    - Bedrock does not use LightRAG's generic ``api_key`` fields.
    - ``LLM_BINDING_API_KEY`` and ``EMBEDDING_BINDING_API_KEY`` are ignored for
      Bedrock.
    - To use Bedrock API key / bearer-token auth, set
      ``AWS_BEARER_TOKEN_BEDROCK`` before starting the process; this is a
      process-level AWS SDK setting.
    - For role-specific Bedrock configuration, use explicit SigV4 parameters
      (``aws_access_key_id``, ``aws_secret_access_key``, ``aws_session_token``,
      ``aws_region``). Per-role bearer-token overrides are not supported.
    """
    if api_key:
        warnings.warn(
            "bedrock_embed(api_key=...) is ignored; use SigV4 parameters or "
            "set AWS_BEARER_TOKEN_BEDROCK before process start.",
            DeprecationWarning,
            stacklevel=2,
        )

    region = aws_region
    endpoint_url = _normalize_bedrock_endpoint_url(endpoint_url)

    session = aioboto3.Session()
    async with session.client(
        "bedrock-runtime",
        **_bedrock_client_kwargs(
            region,
            endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        ),
    ) as bedrock_async_client:
        try:
            if (model_provider := model.split(".")[0]) == "amazon":
                embed_texts = []
                for text in texts:
                    try:
                        if "v2" in model:
                            body = json.dumps(
                                {
                                    "inputText": text,
                                    # 'dimensions': embedding_dim,
                                    "embeddingTypes": ["float"],
                                }
                            )
                        elif "v1" in model:
                            body = json.dumps({"inputText": text})
                        else:
                            raise BedrockError(f"Model {model} is not supported!")

                        response = await bedrock_async_client.invoke_model(
                            modelId=model,
                            body=body,
                            accept="application/json",
                            contentType="application/json",
                        )

                        response_body = await response.get("body").json()

                        # Validate response structure
                        if not response_body or "embedding" not in response_body:
                            raise BedrockError(
                                f"Invalid embedding response structure for text: {text[:50]}..."
                            )

                        embedding = response_body["embedding"]
                        if not embedding:
                            raise BedrockError(
                                f"Received empty embedding for text: {text[:50]}..."
                            )

                        embed_texts.append(embedding)

                    except Exception as e:
                        # Convert to appropriate exception type
                        _handle_bedrock_exception(
                            e, "Bedrock embedding (amazon, text chunk)"
                        )

            elif model_provider == "cohere":
                try:
                    body = json.dumps(
                        {
                            "texts": texts,
                            "input_type": "search_document",
                            "truncate": "NONE",
                        }
                    )

                    response = await bedrock_async_client.invoke_model(
                        model=model,
                        body=body,
                        accept="application/json",
                        contentType="application/json",
                    )

                    response_body = json.loads(response.get("body").read())

                    # Validate response structure
                    if not response_body or "embeddings" not in response_body:
                        raise BedrockError(
                            "Invalid embedding response structure from Cohere"
                        )

                    embeddings = response_body["embeddings"]
                    if not embeddings or len(embeddings) != len(texts):
                        raise BedrockError(
                            f"Invalid embeddings count: expected {len(texts)}, got {len(embeddings) if embeddings else 0}"
                        )

                    embed_texts = embeddings

                except Exception as e:
                    # Convert to appropriate exception type
                    _handle_bedrock_exception(e, "Bedrock embedding (cohere)")

            else:
                raise BedrockError(
                    f"Model provider '{model_provider}' is not supported!"
                )

            # Final validation
            if not embed_texts:
                raise BedrockError("No embeddings generated")

            return np.array(embed_texts)

        except Exception as e:
            # Convert to appropriate exception type
            _handle_bedrock_exception(e, "Bedrock embedding")
