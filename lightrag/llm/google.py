import sys
import os
import logging
import numpy as np
from typing import Any, Union, Optional, List, Dict

if sys.version_info < (3, 9):
    from typing import AsyncIterator as TypingAsyncIterator
else:
    from collections.abc import AsyncIterator as TypingAsyncIterator

import pipmaster as pm

# Install specific modules if not already present
if not pm.is_installed("google-genai"):
    pm.install("google-genai")
if not pm.is_installed("numpy"):
    pm.install("numpy")  # numpy is used for embeddings
if not pm.is_installed("tenacity"):
    pm.install("tenacity")  # tenacity for retries
if not pm.is_installed("pydantic"):  # For response_schema if using Pydantic models
    pm.install("pydantic")

from google.genai import Client as GoogleGenAIClient
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.types import GPTKeywordExtractionFormat

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    # locate_json_string_body_from_string, # May not be needed if response_schema is robust
    safe_unicode_decode,
    logger,
    verbose_debug,
    VERBOSE_DEBUG,
)
from lightrag.api import __api_version__  # For User-Agent

# Load environment variables from.env file if present
# This allows local.env to override global env vars if override=True,
# but LightRAG's typical pattern is override=False (OS takes precedence)
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism on invalid/empty responses."""

    pass


# Define retryable exceptions for Google API calls
RETRYABLE_GOOGLE_EXCEPTIONS = (
    google_api_exceptions.ResourceExhausted,  # HTTP 429
    google_api_exceptions.ServiceUnavailable,  # HTTP 503
    google_api_exceptions.DeadlineExceeded,  # HTTP 504
    google_api_exceptions.InternalServerError,  # HTTP 500
    google_api_exceptions.Unknown,  # HTTP 500 (often)
    google_api_exceptions.Aborted,  # Context-dependent, can be retryable
    InvalidResponseError,  # Custom error for empty/malformed success
    # google_types.generation_types.BrokenResponseError (if applicable and defined)
    # google_types.StopCandidateException (if indicates a retryable state)
)


DEFAULT_GOOGLE_GEMINI_MODEL = "gemini-2.0-flash"  # Default model for Gemini API
# Default embedding model parameters (for text-embedding-005)
DEFAULT_GOOGLE_EMBEDDING_MODEL = "text-embedding-005"
DEFAULT_GOOGLE_EMBEDDING_DIM = 768
DEFAULT_GOOGLE_MAX_TOKEN_SIZE = 8192  # Max tokens per individual text for this model


async def create_google_async_client(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
    client_configs: Optional[Dict[str, Any]] = None,
) -> GoogleGenAIClient:
    """
    Creates an asynchronous Google Generative AI client.
    Prioritizes explicit params, then environment variables.
    """
    effective_use_vertex_ai = use_vertex_ai

    # Determine API key
    effective_api_key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    # Determine Vertex AI params
    effective_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    effective_location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")

    # Infer use_vertex_ai if not explicitly set by env var parsing
    if effective_use_vertex_ai is None:
        if effective_project_id:  # If project_id is available, assume Vertex AI unless API key is also present and dominant
            effective_use_vertex_ai = True
        elif effective_api_key:  # If only API key is available, assume Gemini API
            effective_use_vertex_ai = False
        else:  # Default to Gemini API if no clear indicators, will likely fail if no API key
            effective_use_vertex_ai = False
            logger.warning(
                "Could not determine Google API mode (Vertex AI vs Gemini API key). Defaulting to Gemini API. Ensure GEMINI_API_KEY is set."
            )

    # User-Agent header
    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        # Content-Type is typically handled by the SDK per request
    }
    http_options = {"headers": default_headers}
    if client_configs and "http_options" in client_configs:
        http_options.update(client_configs.pop("http_options"))

    merged_client_args = client_configs.copy() if client_configs else {}
    merged_client_args["http_options"] = google_types.HttpOptions(**http_options)

    if effective_use_vertex_ai:
        logger.info("Initializing Google GenAI Client for Vertex AI.")
        if not effective_project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable or project_id parameter must be set for Vertex AI."
            )
        # Location is recommended, some models might require it.
        if not effective_location:
            logger.warning(
                "GOOGLE_CLOUD_LOCATION or location parameter not set for Vertex AI. This might lead to errors or default region usage."
            )

        final_vertex_args = {
            "vertexai": True,
            "project": effective_project_id,
            "location": effective_location,
        }

        final_vertex_args.update(merged_client_args)
        return GoogleGenAIClient(**final_vertex_args)
    else:
        logger.info("Initializing Google GenAI Client for Gemini API (API Key).")
        if not effective_api_key:
            raise ValueError(
                "GEMINI_API_KEY/GOOGLE_API_KEY environment variable or api_key parameter must be set for Gemini API."
            )

        final_gemini_args = {"api_key": effective_api_key}
        final_gemini_args.update(merged_client_args)
        return GoogleGenAIClient(**final_gemini_args)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RETRYABLE_GOOGLE_EXCEPTIONS),
    reraise=True,
)
async def google_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
    token_tracker: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """
    Core function to complete a prompt using Google's Generative AI API.
    Handles client creation, request formatting, API call, and response processing.
    """
    if not VERBOSE_DEBUG and logging.getLogger("google_genai").level != logging.WARNING:
        logging.getLogger("google_genai").setLevel(
            logging.WARNING
        )  # Reduce verbosity of underlying SDK

    client_call_configs = kwargs.pop("google_client_configs", {})  # For client creation

    google_client = await create_google_async_client(
        api_key=api_key,
        project_id=project_id,
        location=location,
        use_vertex_ai=use_vertex_ai,
        client_configs=client_call_configs,
    )

    # Prepare contents for API call
    api_contents: List[google_types.Content] = []
    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user").lower()
            # Google SDK expects "user" or "model"
            if role not in ["user", "model"]:
                logger.warning(
                    f"Invalid role '{role}' in history_messages, mapping to 'user'. Supported: 'user', 'model'."
                )
                role = "user" if role != "assistant" else "model"  # common mapping
            api_contents.append(
                google_types.Content(
                    role=role,
                    parts=[google_types.Part(text=str(msg.get("content", "")))],
                )
            )

    api_contents.append(
        google_types.Content(role="user", parts=[google_types.Part(text=prompt)])
    )

    # Prepare GenerateContentConfig
    gen_config_params = {}
    if system_prompt:
        # For google-genai, system_instruction is part of GenerateContentConfig or GenerativeModel constructor
        # Here, we pass it via GenerateContentConfig for per-call flexibility.
        gen_config_params["system_instruction"] = google_types.Content(
            role="system", parts=[google_types.Part(text=system_prompt)]
        )

    # Standard generation parameters from kwargs
    for param_name in [
        "temperature",
        "max_output_tokens",
        "top_p",
        "top_k",
        "candidate_count",
        "seed",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
    ]:
        if param_name in kwargs:
            gen_config_params[param_name] = kwargs[param_name]
    if (
        "stop_sequences" in kwargs and kwargs["stop_sequences"]
    ):  # Ensure it's not None or empty
        gen_config_params["stop_sequences"] = kwargs["stop_sequences"]

    # JSON mode parameters (response_mime_type and response_schema)
    response_mime_type = kwargs.get("response_mime_type")
    response_schema = kwargs.get("response_schema")

    if response_mime_type:
        gen_config_params["response_mime_type"] = response_mime_type
    if response_schema:
        gen_config_params["response_schema"] = response_schema
        if not response_mime_type:  # Default to application/json if schema is provided
            gen_config_params["response_mime_type"] = "application/json"
            logger.debug(
                "response_schema provided without response_mime_type, defaulting to application/json."
            )

    # Safety settings (example, can be made configurable via kwargs)
    safety_settings_obj = kwargs.get("safety_settings")
    if safety_settings_obj:
        gen_config_params["safety_settings"] = safety_settings_obj

    generation_config_obj = (
        google_types.GenerateContentConfig(**gen_config_params)
        if gen_config_params
        else None
    )

    logger.debug("===== Entering func of Google LLM =====")
    logger.debug(f"Model: {model}")
    logger.debug(f"GenerateContentConfig effective params: {gen_config_params}")
    logger.debug(
        f"Num of history messages (converted to Content objects): {len(api_contents) - 1}"
    )
    verbose_debug(f"System prompt (via GenerateContentConfig): {system_prompt}")
    verbose_debug(f"User Query (latest): {prompt}")
    logger.debug("===== Sending Query to Google LLM =====")

    is_streaming = kwargs.get("stream", False)

    try:
        if is_streaming:
            response_iter = await google_client.aio.models.generate_content_stream(
                model=model,
                contents=api_contents,
                config=generation_config_obj,
            )

            async def stream_generator():
                full_response_text_for_log = []
                try:
                    async for chunk in response_iter:
                        if not hasattr(chunk, "text") or chunk.text is None:
                            # Sometimes, finish_reason or other metadata might come in a chunk without text
                            if (
                                hasattr(chunk, "candidates")
                                and chunk.candidates
                                and hasattr(chunk.candidates, "finish_reason")
                                and chunk.candidates.finish_reason
                            ):
                                logger.debug(
                                    f"Stream chunk finish_reason: {chunk.candidates.finish_reason.name}"
                                )
                            else:
                                logger.warning(
                                    f"Received stream chunk without text: {chunk}"
                                )
                            continue

                        content_text = chunk.text
                        if r"\u" in content_text:  # Handle unicode escapes if any
                            content_text = safe_unicode_decode(
                                content_text.encode("utf-8")
                            )

                        full_response_text_for_log.append(content_text)
                        yield content_text
                except Exception as e:
                    logger.error(f"Error during Google API stream processing: {e}")
                    raise
                finally:
                    logger.debug(
                        f"Stream ended. Full streamed response length: {len(''.join(full_response_text_for_log))}"
                    )
                    verbose_debug(
                        f"Full streamed response: {''.join(full_response_text_for_log)}"
                    )

            return stream_generator()
        else:  # Non-streaming
            response = await google_client.aio.models.generate_content(
                model=model,
                contents=api_contents,
                config=generation_config_obj,
            )

            if not response or not response.text:  # Check for empty response
                # Check for blocking reasons
                if (
                    response
                    and response.prompt_feedback
                    and response.prompt_feedback.block_reason
                ):
                    err_msg = f"Google API request was blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}"
                    logger.error(err_msg)
                    raise InvalidResponseError(
                        err_msg
                    )  # This could be a non-retryable error depending on policy

                # Check candidates for blocking
                if response and response.candidates:
                    for candidate in response.candidates:
                        if (
                            candidate.finish_reason.name == "SAFETY"
                        ):  # FinishReason.SAFETY
                            err_msg = f"Google API response candidate blocked due to safety. Ratings: {candidate.safety_ratings}"
                            logger.error(err_msg)
                            raise InvalidResponseError(err_msg)  # Likely non-retryable

                logger.error(
                    "Received empty or invalid content from Google API non-streaming response."
                )
                raise InvalidResponseError("Received empty content from Google API.")

            content_to_return = response.text
            if r"\u" in content_to_return:
                content_to_return = safe_unicode_decode(
                    content_to_return.encode("utf-8")
                )

            # Token tracking for non-streaming
            if (
                token_tracker
                and hasattr(response, "usage_metadata")
                and response.usage_metadata
            ):
                usage = response.usage_metadata
                token_counts = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0)
                    or (
                        getattr(usage, "prompt_token_count", 0)
                        + getattr(usage, "candidates_token_count", 0)
                    ),
                }
                token_tracker.add_usage(token_counts)
                logger.debug(f"Google API token usage: {token_counts}")

            logger.debug(
                f"Google API Response content length: {len(content_to_return)}"
            )
            verbose_debug(
                f"Google API Response: {content_to_return[:500]}{'...' if len(content_to_return) > 500 else ''}"
            )  # Log snippet

            # For JSON mode, response.text is the JSON string.
            # If response_schema was used, response.parsed might contain the Pydantic model.
            # LightRAG expects the string representation for now.
            return content_to_return

    except google_api_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error: {e.__class__.__name__} - {e}")
        # Specific handling for common non-retryable errors if needed, though tenacity handles retryable ones
        if isinstance(
            e, google_api_exceptions.InvalidArgument
        ):  # Typically non-retryable
            logger.error(
                f"Google API Invalid Argument: {e}. This is often due to malformed request or invalid model parameters."
            )
        elif isinstance(e, google_api_exceptions.PermissionDenied):
            logger.error(
                f"Google API Permission Denied: {e}. Check credentials and API enablement."
            )
        raise  # Reraise for tenacity or higher-level handling
    except Exception as e:
        logger.error(
            f"Unexpected error during Google API call: {e.__class__.__name__} - {e}"
        )
        raise


async def google_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, TypingAsyncIterator[str]]:
    """
    Simplified wrapper for Google text completion.
    Determines model name and sets up for keyword extraction if requested.
    """
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    if not model_name.contains("gemini"):
        # TODO check against client.models.list()
        logger.warning(
            f"Invalid `llm_model_name` Argument: {model_name}. Set a correct model name - default to {DEFAULT_GOOGLE_GEMINI_MODEL}."
        )
        # Fallback to environment variable or a hardcoded default
        model_name = os.environ.get("LLM_MODEL", DEFAULT_GOOGLE_GEMINI_MODEL)

    # Keyword extraction setup
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    if keyword_extraction:
        kwargs["response_mime_type"] = "application/json"
        # Use the Pydantic model defined earlier or a dict schema
        kwargs["response_schema"] = GPTKeywordExtractionFormat
        logger.debug(
            "Keyword extraction enabled, setting response_mime_type to application/json and providing schema."
        )

    # API key and Vertex params can be passed via kwargs or picked up from env by create_google_async_client
    api_key = kwargs.pop("api_key", None)
    project_id = kwargs.pop("project_id", None)
    location = kwargs.pop("location", None)
    use_vertex_ai = kwargs.pop("use_vertex_ai", None)

    if use_vertex_ai is None:
        # Determine if GOOGLE_GENAI_USE_VERTEXAI was true from environment
        use_vertex_ai_str = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower()
        use_vertex_ai = use_vertex_ai_str == "true"

    result = await google_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        project_id=project_id,
        location=location,
        use_vertex_ai=use_vertex_ai,
        **kwargs,  # Remaining kwargs (token_tracker, temperature, etc.)
    )

    if (
        keyword_extraction
        and isinstance(result, str)
        and kwargs.get("response_mime_type") == "application/json"
    ):
        # If the model still wraps the JSON in text, try to extract it.
        # However, with response_schema, this should ideally not be needed.
        # The locate_json_string_body_from_string is a fallback.
        # json_body = locate_json_string_body_from_string(result)
        # return json_body if json_body else result
        return result  # Assuming response_schema ensures result is a clean JSON string

    return result


# --- Specific Model Wrappers (Examples) ---
async def gemini_2_0_flash_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, dict]:
    # Keyword extraction setup
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    if keyword_extraction:
        kwargs["response_mime_type"] = "application/json"
        # Use the Pydantic model defined earlier or a dict schema
        kwargs["response_schema"] = GPTKeywordExtractionFormat
        logger.debug(
            "Keyword extraction enabled, setting response_mime_type to application/json and providing schema."
        )
    return await google_complete_if_cache(
        "gemini-2.0-flash-001",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=DEFAULT_GOOGLE_EMBEDDING_DIM,
    max_token_size=DEFAULT_GOOGLE_MAX_TOKEN_SIZE,  # Max tokens for a single text input
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1, min=4, max=60
    ),  # Longer max wait for embeddings
    retry=retry_if_exception_type(RETRYABLE_GOOGLE_EXCEPTIONS),
    reraise=True,
)
async def google_embed(
    texts: List[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
    client_configs: Optional[dict] = None,
    task_type: Optional[
        str
    ] = None,  # e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY" (Defaults to "RETRIEVAL_QUERY" if None)
    title: Optional[str] = None,  # For RETRIEVAL_DOCUMENT task_type
    output_dimensionality: Optional[int] = None,  # For reducing embedding dimensions
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Google's embedding models during querying.
    """
    if not texts:
        return np.array()

    if not VERBOSE_DEBUG and logging.getLogger("google_genai").level != logging.WARNING:
        logging.getLogger("google_genai").setLevel(logging.WARNING)

    if use_vertex_ai is None:
        # Determine if GOOGLE_GENAI_USE_VERTEXAI was true from environment
        use_vertex_ai_str = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower()
        use_vertex_ai = use_vertex_ai_str == "true"

    google_client = await create_google_async_client(
        api_key=api_key,
        project_id=project_id,
        location=location,
        use_vertex_ai=use_vertex_ai,
        client_configs=client_configs,
    )

    if not model:
        model = os.environ.get("EMBEDDING_MODEL", DEFAULT_GOOGLE_EMBEDDING_MODEL)

    logger.debug(f"Requesting embeddings for {len(texts)} texts with model {model}.")
    verbose_debug(f"Embedding texts (first 3): {texts[:3]}")

    # Convert string task_type to google_types.TaskType enum if provided
    task_type_enum: Optional = None
    if task_type:
        try:
            task_type_enum = google_types.TaskType[task_type.upper()]
        except KeyError:
            logger.warning(
                f"Invalid task_type '{task_type}'. Ignoring. Valid types are: {', '.join(google_types.TaskType.__members__)}"
            )
            task_type_enum = None  # Defaults to "RETRIEVAL_QUERY"

    try:
        response = await google_client.aio.embed_contents(
            model=model,
            contents=texts,
            config=google_types.EmbedContentConfig(
                output_dimensionality=output_dimensionality,
                task_type=task_type_enum,
            ),
        )

        if (
            not response
            or not hasattr(response, "embeddings")
            or not response.embeddings
        ):
            logger.error("Invalid or empty embedding response from Google API (batch).")
            raise InvalidResponseError(
                "Invalid or empty embedding response from Google API (batch)."
            )

        embeddings_list = [
            embedding_obj.values for embedding_obj in response.embeddings
        ]

        # Ensure all embeddings have the same dimension if output_dimensionality was not set,
        # or match output_dimensionality if it was.
        if embeddings_list and output_dimensionality:
            if any(len(emb) != output_dimensionality for emb in embeddings_list):
                logger.warning(
                    f"Some embeddings have dimension other than requested {output_dimensionality}. Check API behavior."
                )

        return np.array(embeddings_list, dtype=np.float32)

    except google_api_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error during embedding: {e.__class__.__name__} - {e}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during Google embedding: {e.__class__.__name__} - {e}"
        )
        raise


async def google_embed_insert(
    texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", **kwargs: Any
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Google's embedding models during insertion.
    """
    return await google_embed(texts=texts, task_type=task_type, **kwargs)
