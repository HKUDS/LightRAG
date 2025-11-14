import copy
import os
import json
import logging

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

import sys
from lightrag.utils import wrap_embedding_func_with_attrs

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
from typing import Union

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


def _set_env_if_present(key: str, value):
    """Set environment variable only if a non-empty value is provided."""
    if value is not None and value != "":
        os.environ[key] = value


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
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if enable_cot:
        import logging

        logging.debug(
            "enable_cot=True is not supported for Bedrock and will be ignored."
        )
    # Respect existing env; only set if a non-empty value is available
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or aws_access_key_id
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or aws_secret_access_key
    session_token = os.environ.get("AWS_SESSION_TOKEN") or aws_session_token
    _set_env_if_present("AWS_ACCESS_KEY_ID", access_key)
    _set_env_if_present("AWS_SECRET_ACCESS_KEY", secret_key)
    _set_env_if_present("AWS_SESSION_TOKEN", session_token)
    # Region handling: prefer env, else kwarg (optional)
    region = os.environ.get("AWS_REGION") or kwargs.pop("aws_region", None)
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
        "response_format",
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
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    # Import logging for error handling
    import logging

    # For streaming responses, we need a different approach to keep the connection open
    if stream:
        # Create a session that will be used throughout the streaming process
        session = aioboto3.Session()
        client = None

        # Define the generator function that will manage the client lifecycle
        async def stream_generator():
            nonlocal client

            # Create the client outside the generator to ensure it stays open
            client = await session.client(
                "bedrock-runtime", region_name=region
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
        "bedrock-runtime", region_name=region
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

            content = response["output"]["message"]["content"][0]["text"]

            if not content or content.strip() == "":
                raise BedrockError("Received empty content from Bedrock API")

            return content

        except Exception as e:
            # Convert to appropriate exception type
            _handle_bedrock_exception(e, "Bedrock converse")


# Generic Bedrock completion function
async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    kwargs.pop("keyword_extraction", None)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    result = await bedrock_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
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
) -> np.ndarray:
    # Respect existing env; only set if a non-empty value is available
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or aws_access_key_id
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or aws_secret_access_key
    session_token = os.environ.get("AWS_SESSION_TOKEN") or aws_session_token
    _set_env_if_present("AWS_ACCESS_KEY_ID", access_key)
    _set_env_if_present("AWS_SECRET_ACCESS_KEY", secret_key)
    _set_env_if_present("AWS_SESSION_TOKEN", session_token)

    # Region handling: prefer env
    region = os.environ.get("AWS_REGION")

    session = aioboto3.Session()
    async with session.client(
        "bedrock-runtime", region_name=region
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
