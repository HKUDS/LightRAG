#!/usr/bin/env python3
"""
ABOUTME: Shared token estimation utilities for audit scripts
ABOUTME: XML sanitization helpers for document processing
"""

import json
import os
import re

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    types = None
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:  # pragma: no cover - optional dependency
    openai = None
    HAS_OPENAI = False


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for LLM context management.

    Uses a weighted formula based on character types:
    - Chinese characters: ~0.75 tokens per character (subword tokenization)
    - JSON structural characters (brackets, quotes, commas): ~1 tokens per character
    - Other characters (English, numbers, symbols): ~0.4 tokens per character (~3 chars/token)

    Includes 5% buffer and safety offset for special formatting and system prompt overhead.

    Args:
        text: Input text to estimate tokens for

    Returns:
        int: Estimated token count
    """
    if not text:
        return 0

    chinese_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
    json_chars_count = len(re.findall(r'[\[\]",{}]', text))
    other_count = len(text) - chinese_count - json_chars_count

    base_estimate = (chinese_count * 0.75) + (json_chars_count * 1) + (other_count * 0.4)
    final_tokens = int(base_estimate * 1.05) + 2
    return final_tokens


def sanitize_xml_string(text: str) -> str:
    """
    Remove control characters that are illegal in XML 1.0.

    XML 1.0 allows: #x9 (tab), #xA (LF), #xD (CR), and #x20-#xD7FF, #xE000-#xFFFD, #x10000-#x10FFFF
    This function removes all other control characters (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F).

    Args:
        text: Text that may contain control characters

    Returns:
        Sanitized text safe for XML. Returns input unchanged if not a non-empty string.
    """
    if not text or not isinstance(text, str):
        return text
    # Build a translation table to remove illegal control characters
    # Keep: \t (0x09), \n (0x0A), \r (0x0D)
    # Remove: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F
    illegal_chars = ''.join(
        chr(c) for c in range(0x20)
        if c not in (0x09, 0x0A, 0x0D)
    )
    return text.translate(str.maketrans('', '', illegal_chars))


def is_vertex_ai_mode() -> bool:
    """
    Check if Vertex AI mode is enabled via environment variable.

    Returns:
        True if GOOGLE_GENAI_USE_VERTEXAI is set to 'true', False otherwise
    """
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"


def create_gemini_client(use_async: bool = False):
    """
    Create Gemini client for AI Studio or Vertex AI.

    Supports two modes:
    - AI Studio (default): Uses GOOGLE_API_KEY for authentication
    - Vertex AI: Uses ADC (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)

    Environment variables for Vertex AI mode:
    - GOOGLE_GENAI_USE_VERTEXAI: Set to 'true' to enable Vertex AI mode
    - GOOGLE_CLOUD_PROJECT: Required GCP project ID
    - GOOGLE_CLOUD_LOCATION: Optional region (default: us-central1)
    - GOOGLE_VERTEX_BASE_URL: Optional custom API endpoint (for API gateway proxies)
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (or use gcloud auth)

    Args:
        use_async: If True, return the async client (.aio), otherwise return sync client

    Returns:
        Gemini client instance (sync or async based on use_async parameter)

    Raises:
        ValueError: If required environment variables are not set
    """
    use_vertex = is_vertex_ai_mode()

    if use_vertex:
        # Vertex AI mode - uses ADC (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        base_url = os.getenv("GOOGLE_VERTEX_BASE_URL")

        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT is required for Vertex AI mode. "
                "Set GOOGLE_GENAI_USE_VERTEXAI=false to use AI Studio mode instead."
            )

        # Build http_options only if custom base_url is specified
        http_options = None
        if base_url:
            http_options = {"base_url": base_url}

        # Note: ADC handles authentication automatically
        # via GOOGLE_APPLICATION_CREDENTIALS env var or gcloud auth
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=http_options
        )
    else:
        # AI Studio mode - requires API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for AI Studio mode. "
                "Set GOOGLE_GENAI_USE_VERTEXAI=true and configure GCP credentials for Vertex AI mode."
            )

        client = genai.Client(api_key=api_key)

    # Return async or sync client based on parameter
    return client.aio if use_async else client


def get_gemini_provider_name() -> str:
    """
    Get the Gemini provider name based on current mode.

    Returns:
        Provider name string for display purposes
    """
    if is_vertex_ai_mode():
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "unknown")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        return f"Google Gemini (Vertex AI: {project}/{location})"
    return "Google Gemini (AI Studio)"


def create_openai_client(use_async: bool = True):
    """
    Create OpenAI client with optional custom base URL.

    Environment variables:
    - OPENAI_API_KEY: Required API key
    - OPENAI_BASE_URL: Optional custom API endpoint (for proxies, Azure, etc.)

    Args:
        use_async: If True, return AsyncOpenAI, otherwise return OpenAI

    Returns:
        OpenAI client instance (async or sync based on use_async parameter)

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    if not HAS_OPENAI:
        raise ValueError("openai library is not installed.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI mode.")

    base_url = os.getenv("OPENAI_BASE_URL")

    if use_async:
        return openai.AsyncOpenAI(base_url=base_url)
    return openai.OpenAI(base_url=base_url)


def get_openai_provider_name() -> str:
    """
    Get the OpenAI provider name, including custom endpoint if configured.

    Returns:
        Provider name string for display purposes
    """
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return f"OpenAI (Custom: {base_url})"
    return "OpenAI"


def is_openai_reasoning_model(model_name: str) -> bool:
    """
    Check if the OpenAI model supports reasoning_effort parameter.

    Models that support reasoning_effort:
    - o-series: o1, o3, o4 and their variants (o1-mini, o1-2024-12-17, etc.)
    - gpt-5 series: gpt-5, gpt-5.2, gpt-5-turbo, etc.

    Non-reasoning models like gpt-4.1, gpt-4o, etc. will reject this parameter.

    Handles proxy/router prefixes like "openai/o1-mini" or "openrouter/gpt-5.2".

    Args:
        model_name: The OpenAI model name (may include path prefix)

    Returns:
        True if the model supports reasoning_effort, False otherwise
    """
    model_lower = model_name.lower()

    # Handle proxy/router prefixes like "openai/o1-mini", "openrouter/gpt-5.2"
    # Extract the base model name after the last "/"
    if '/' in model_lower:
        model_lower = model_lower.rsplit('/', 1)[-1]

    # Match o-series and gpt-5 series
    return model_lower.startswith(('o1', 'o3', 'o4', 'gpt-5'))


def is_openai_retryable(error: Exception) -> bool:
    """
    Determine if an OpenAI error should be retried.

    Non-retryable errors:
    - AuthenticationError (401): Invalid API key
    - PermissionDeniedError (403): No access to resource
    - BadRequestError (400): Invalid request format
    - NotFoundError (404): Model or resource not found

    Retryable errors:
    - RateLimitError (429): Rate limit exceeded
    - APIConnectionError: Network issues
    - InternalServerError (500): Server errors
    - APIStatusError with 502, 503, 504: Gateway/service errors

    Args:
        error: The exception from OpenAI API call

    Returns:
        True if the error should be retried, False otherwise
    """
    if not HAS_OPENAI:
        return True

    # Authentication error - invalid API key (401)
    if isinstance(error, openai.AuthenticationError):
        return False

    # Permission denied - no access to resource (403)
    if isinstance(error, openai.PermissionDeniedError):
        return False

    # Bad request - invalid request format (400)
    if isinstance(error, openai.BadRequestError):
        return False

    # Not found - model or resource doesn't exist (404)
    if isinstance(error, openai.NotFoundError):
        return False

    # Rate limit exceeded - should retry with backoff (429)
    if isinstance(error, openai.RateLimitError):
        return True

    # API connection error - network issues, should retry
    if isinstance(error, openai.APIConnectionError):
        return True

    # Internal server error - should retry (500)
    if isinstance(error, openai.InternalServerError):
        return True

    # For other APIStatusError, check HTTP status code
    if isinstance(error, openai.APIStatusError):
        # Retryable server-side errors
        return error.status_code in (429, 500, 502, 503, 504)

    # For unknown errors, default to retry (network issues, timeouts, etc.)
    return True


def is_gemini_retryable(error: Exception) -> bool:
    """
    Determine if a Gemini error should be retried.

    Uses string matching on error messages since google-genai may not have
    well-defined exception types for all error cases.

    Non-retryable errors:
    - API key errors
    - Authentication/permission errors
    - Invalid request errors
    - Model not found errors
    - Billing/quota permanently exceeded

    Retryable errors:
    - Rate limit (429)
    - Server errors (500, 502, 503, 504)
    - Timeout/connection errors

    Args:
        error: The exception from Gemini API call

    Returns:
        True if the error should be retried, False otherwise
    """
    error_str = str(error).lower()

    # API key / authentication errors - do not retry
    if 'api_key' in error_str or 'api key' in error_str:
        return False
    if 'authentication' in error_str or 'authenticate' in error_str:
        return False
    if 'invalid_api_key' in error_str or 'invalid api key' in error_str:
        return False

    # Permission / forbidden errors - do not retry
    if 'permission' in error_str and 'denied' in error_str:
        return False
    if 'forbidden' in error_str or '403' in error_str:
        return False

    # Invalid request errors - do not retry
    if 'invalid' in error_str and ('request' in error_str or 'argument' in error_str):
        return False
    if '400' in error_str and 'bad request' in error_str:
        return False

    # Model not found - do not retry
    if 'model' in error_str and ('not found' in error_str or 'not exist' in error_str):
        return False
    if '404' in error_str:
        return False

    # Billing / permanent quota errors - do not retry
    if 'billing' in error_str:
        return False
    if 'quota' in error_str and ('exceeded' in error_str or 'exhausted' in error_str):
        # Check if it mentions billing which indicates permanent quota issue
        if 'billing' in error_str or 'payment' in error_str:
            return False
        # Temporary quota (rate limit) - should retry
        return True

    # Rate limit errors - should retry (429)
    if 'rate' in error_str and 'limit' in error_str:
        return True
    if '429' in error_str or 'resource_exhausted' in error_str:
        return True

    # Server errors - should retry (500, 502, 503, 504)
    if any(code in error_str for code in ['500', '502', '503', '504']):
        return True
    if 'internal' in error_str and ('error' in error_str or 'server' in error_str):
        return True
    if 'service' in error_str and 'unavailable' in error_str:
        return True
    if 'gateway' in error_str:
        return True

    # Timeout / connection errors - should retry
    if 'timeout' in error_str or 'timed out' in error_str:
        return True
    if 'connection' in error_str:
        return True
    if 'network' in error_str:
        return True

    # Unknown errors - default to retry with limited attempts
    return True


# JSON Schema for LLM structured output
AUDIT_RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_violation": {
            "type": "boolean",
            "description": "Whether any violations were found"
        },
        "violations": {
            "type": "array",
            "description": "List of violations found",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "ID of the violated rule (e.g., R001)"
                    },
                    "violation_text": {
                        "type": "string",
                        "description": "The problematic text directly verbatim quote from the source content, and not span multiple cells"
                    },
                    "violation_reason": {
                        "type": "string",
                        "description": "Explanation of why this violates the rule"
                    },
                    "fix_action": {
                        "type": "string",
                        "enum": ["replace", "manual"],
                        "description": "Action type: replace substitutes text (including deletion-via-replace), manual requires human review"
                    },
                    "revised_text": {
                        "type": "string",
                        "description": "For replace: complete replacement text (including deletion-via-replace). For manual: additional guidance for human reviewer"
                    }
                },
                "required": ["rule_id", "violation_text", "violation_reason", "fix_action", "revised_text"]
            }
        }
    },
    "required": ["is_violation", "violations"]
}

# JSON Schema for global extraction output
GLOBAL_EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rule_id": {"type": "string"},
                    "extracted_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "entity": {"type": "string"},
                                "fields": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "name": {"type": "string"},
                                            "value": {"type": "string"},
                                            "evidence": {"type": "string"}
                                        },
                                        "required": ["name", "value", "evidence"]
                                    }
                                }
                            },
                            "required": ["entity", "fields"]
                        }
                    }
                },
                "required": ["rule_id", "extracted_results"]
            }
        }
    },
    "required": ["results"]
}

# JSON Schema for global verification output
GLOBAL_VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rule_id": {"type": "string"},
                    "uuid": {"type": "string"},
                    "uuid_end": {"type": "string"},
                    "violation_text": {"type": "string"},
                    "violation_reason": {"type": "string"},
                    "fix_action": {
                        "type": "string",
                        "enum": ["replace", "manual"]
                    },
                    "revised_text": {"type": "string"}
                },
                "required": [
                    "rule_id",
                    "uuid",
                    "uuid_end",
                    "violation_text",
                    "violation_reason",
                    "fix_action",
                    "revised_text"
                ]
            }
        }
    },
    "required": ["violations"]
}


async def global_extract_gemini_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    client,
    thinking_level: str = None,
    thinking_budget: int = None
) -> dict:
    thinking_config = None
    if thinking_level and thinking_level.upper() in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
        level_map = {
            "MINIMAL": types.ThinkingLevel.MINIMAL,
            "LOW": types.ThinkingLevel.LOW,
            "MEDIUM": types.ThinkingLevel.MEDIUM,
            "HIGH": types.ThinkingLevel.HIGH,
        }
        thinking_config = types.ThinkingConfig(
            thinking_level=level_map[thinking_level.upper()]
        )
    elif thinking_budget is not None:
        thinking_config = types.ThinkingConfig(
            thinking_budget=int(thinking_budget)
        )

    config_params = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": GLOBAL_EXTRACT_SCHEMA
    }
    if thinking_config:
        config_params["thinking_config"] = thinking_config

    response = await client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_params)
    )
    return json.loads(response.text)


async def global_extract_openai_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    client,
    reasoning_effort: str = None
) -> dict:
    request_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "global_extract",
                "strict": True,
                "schema": GLOBAL_EXTRACT_SCHEMA
            }
        }
    }
    if reasoning_effort and reasoning_effort.lower() in ("low", "medium", "high") \
            and is_openai_reasoning_model(model_name):
        request_params["reasoning_effort"] = reasoning_effort.lower()

    response = await client.chat.completions.create(**request_params)
    return json.loads(response.choices[0].message.content)


async def global_verify_gemini_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    client,
    thinking_level: str = None,
    thinking_budget: int = None
) -> dict:
    thinking_config = None
    if thinking_level and thinking_level.upper() in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
        level_map = {
            "MINIMAL": types.ThinkingLevel.MINIMAL,
            "LOW": types.ThinkingLevel.LOW,
            "MEDIUM": types.ThinkingLevel.MEDIUM,
            "HIGH": types.ThinkingLevel.HIGH,
        }
        thinking_config = types.ThinkingConfig(
            thinking_level=level_map[thinking_level.upper()]
        )
    elif thinking_budget is not None:
        thinking_config = types.ThinkingConfig(
            thinking_budget=int(thinking_budget)
        )

    config_params = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": GLOBAL_VERIFY_SCHEMA
    }
    if thinking_config:
        config_params["thinking_config"] = thinking_config

    response = await client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_params)
    )
    return json.loads(response.text)


async def global_verify_openai_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    client,
    reasoning_effort: str = None
) -> dict:
    request_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "global_verify",
                "strict": True,
                "schema": GLOBAL_VERIFY_SCHEMA
            }
        }
    }
    if reasoning_effort and reasoning_effort.lower() in ("low", "medium", "high") \
            and is_openai_reasoning_model(model_name):
        request_params["reasoning_effort"] = reasoning_effort.lower()

    response = await client.chat.completions.create(**request_params)
    return json.loads(response.choices[0].message.content)


async def audit_block_gemini_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    client,
    thinking_level: str = None,
    thinking_budget: int = None
) -> dict:
    """
    Audit a text block using Google Gemini with strict JSON mode (async version).

    Args:
        user_prompt: User prompt to audit
        system_prompt: Cached system prompt with rules and instructions
        model_name: Gemini model to use
        client: Gemini async client instance (client.aio)
        thinking_level: Thinking level for Gemini 3 models (MINIMAL, LOW, MEDIUM, HIGH)
        thinking_budget: Thinking token budget for Gemini 2.5 models (integer)

    Returns:
        Audit result dictionary
    """
    # Build thinking config based on model and parameters
    thinking_config = None

    if thinking_level and thinking_level.upper() in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
        # For Gemini 3 models
        level_map = {
            "MINIMAL": types.ThinkingLevel.MINIMAL,
            "LOW": types.ThinkingLevel.LOW,
            "MEDIUM": types.ThinkingLevel.MEDIUM,
            "HIGH": types.ThinkingLevel.HIGH,
        }
        thinking_config = types.ThinkingConfig(
            thinking_level=level_map[thinking_level.upper()]
        )
    elif thinking_budget is not None:
        # For Gemini 2.5 models
        thinking_config = types.ThinkingConfig(
            thinking_budget=int(thinking_budget)
        )

    config_params = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": AUDIT_RESULT_SCHEMA
    }

    # Only add thinking_config if it's configured
    if thinking_config:
        config_params["thinking_config"] = thinking_config

    response = await client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_params)
    )

    # With structured output, response is guaranteed to be valid JSON
    result = json.loads(response.text)
    return result


async def audit_block_openai_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    client,
    reasoning_effort: str = None
) -> dict:
    """
    Audit a text block using OpenAI with strict JSON mode (async version).

    Args:
        user_prompt: User prompt to audit
        system_prompt: Cached system prompt with rules and instructions
        model_name: OpenAI model to use
        client: AsyncOpenAI client instance
        reasoning_effort: Reasoning effort for o-series models (low, medium, high)

    Returns:
        Audit result dictionary
    """
    request_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "audit_result",
                "strict": True,
                "schema": AUDIT_RESULT_SCHEMA
            }
        }
    }

    # Add reasoning_effort only for o-series models that support it
    if reasoning_effort and reasoning_effort.lower() in ("low", "medium", "high") \
            and is_openai_reasoning_model(model_name):
        request_params["reasoning_effort"] = reasoning_effort.lower()

    response = await client.chat.completions.create(**request_params)

    # With structured output, response is guaranteed to be valid JSON
    result = json.loads(response.choices[0].message.content)
    return result
