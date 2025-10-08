"""
Enhanced Ollama integration with Pydantic structured outputs support.
This module extends the base Ollama integration to support structured outputs
using Pydantic models for reliable entity and relationship extraction.
"""

import asyncio
import json
import logging
import numpy as np
import ollama
from typing import Union, AsyncIterator, Optional, Type, Any, Dict
from pydantic import BaseModel, ValidationError

from lightrag import __version__

logger = logging.getLogger("lightrag")


async def _ollama_model_with_structured_output(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: list = None,
    schema: Optional[Type[BaseModel]] = None,
    enable_cot: bool = False,
    max_retries: int = 3,
    **kwargs
) -> Union[str, BaseModel, AsyncIterator[str]]:
    """
    Call Ollama with optional structured output using Pydantic schema.
    
    Args:
        model_name: Name of the Ollama model
        prompt: User prompt
        system_prompt: System prompt (optional)
        history_messages: Conversation history
        schema: Pydantic model class for structured output
        enable_cot: Enable chain-of-thought reasoning
        max_retries: Maximum number of retry attempts for validation failures
        **kwargs: Additional arguments (stream, host, api_key, etc.)
        
    Returns:
        Structured Pydantic model instance if schema provided, otherwise string
    """
    stream = kwargs.pop("stream", False)
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    api_key = kwargs.pop("api_key", None)
    
    # Set up headers
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Configure request options
    options = kwargs.pop("options", {})
    
    # If schema is provided, request JSON format
    format_param = None
    if schema is not None:
        # Convert Pydantic schema to JSON schema for Ollama
        format_param = schema.model_json_schema()
        if stream:
            logger.warning("Streaming not supported with structured output, forcing stream=False")
            stream = False
    
    # Retry loop for validation failures
    for attempt in range(max_retries):
        try:
            if stream and schema is None:
                # Streaming response without structured output
                response = await ollama_client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    options=options,
                )
                
                async def stream_generator():
                    async for chunk in response:
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                
                return stream_generator()
            else:
                # Non-streaming response or structured output
                response = await ollama_client.chat(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    format=format_param,
                    options=options,
                )
                
                content = response["message"]["content"]
                
                # If schema provided, parse and validate
                if schema is not None:
                    try:
                        # Parse JSON response
                        if isinstance(content, str):
                            json_data = json.loads(content)
                        else:
                            json_data = content
                        
                        # Validate against Pydantic schema
                        structured_output = schema.model_validate(json_data)
                        logger.debug(f"Successfully validated structured output on attempt {attempt + 1}")
                        return structured_output
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            # Add clarification to prompt for retry
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": "Your response was not valid JSON. Please respond with valid JSON matching the required schema."
                            })
                            continue
                        else:
                            raise ValueError(f"Failed to parse JSON after {max_retries} attempts: {content[:200]}")
                    
                    except ValidationError as e:
                        logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            # Add validation errors to prompt for retry
                            messages.append({"role": "assistant", "content": content})
                            error_details = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
                            messages.append({
                                "role": "user",
                                "content": f"Your response had validation errors: {error_details}. Please correct and respond again."
                            })
                            continue
                        else:
                            raise ValueError(f"Validation failed after {max_retries} attempts: {e}")
                else:
                    # No schema, return raw content
                    return content
                    
        except Exception as e:
            logger.error(f"Error in Ollama API call: {e}")
            raise
        finally:
            if not stream or schema is not None:
                try:
                    await ollama_client._client.aclose()
                except Exception as close_error:
                    logger.debug(f"Failed to close Ollama client: {close_error}")
    
    raise ValueError(f"Failed to get valid response after {max_retries} attempts")


async def ollama_model_complete_structured(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: list = None,
    schema: Optional[Type[BaseModel]] = None,
    enable_cot: bool = False,
    keyword_extraction: bool = False,
    entity_extraction: bool = False,
    **kwargs
) -> Union[str, BaseModel, AsyncIterator[str]]:
    """
    Complete wrapper for Ollama with automatic schema selection.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        history_messages: Conversation history
        schema: Explicit Pydantic schema (overrides auto-detection)
        enable_cot: Enable chain-of-thought
        keyword_extraction: Auto-use KeywordExtraction schema
        entity_extraction: Auto-use ExtractionResult schema
        **kwargs: Additional arguments
        
    Returns:
        Structured output or string based on schema
    """
    # Auto-select schema if not explicitly provided
    if schema is None and (keyword_extraction or entity_extraction):
        from lightrag.pydantic_schemas import KeywordExtraction, ExtractionResult
        
        if keyword_extraction:
            schema = KeywordExtraction
        elif entity_extraction:
            schema = ExtractionResult
    
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    
    return await _ollama_model_with_structured_output(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        schema=schema,
        enable_cot=enable_cot,
        **kwargs
    )


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Generate embeddings using Ollama.
    (Unchanged from original implementation)
    """
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__version__}",
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
        return np.array(data["embeddings"])
    except Exception as e:
        logger.error(f"Error in ollama_embed: {str(e)}")
        raise
    finally:
        try:
            await ollama_client._client.aclose()
        except Exception as close_error:
            logger.debug(f"Failed to close Ollama client: {close_error}")