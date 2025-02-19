import pipmaster as pm
from llama_index.core.llms import (
    ChatMessage,
    MessageRole,
    ChatResponse,
)
from typing import List, Optional
from lightrag.utils import logger

# Install required dependencies
if not pm.is_installed("llama-index"):
    pm.install("llama-index")

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
import numpy as np


def configure_llama_index(settings: LlamaIndexSettings = None, **kwargs):
    """
    Configure LlamaIndex settings.

    Args:
        settings: LlamaIndex Settings instance. If None, uses default settings.
        **kwargs: Additional settings to override/configure
    """
    if settings is None:
        settings = LlamaIndexSettings()

    # Update settings with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            logger.warning(f"Unknown LlamaIndex setting: {key}")

    # Set as global settings
    LlamaIndexSettings.set_global(settings)
    return settings


def format_chat_messages(messages):
    """Format chat messages into LlamaIndex format."""
    formatted_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            formatted_messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=content)
            )
        elif role == "assistant":
            formatted_messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=content)
            )
        elif role == "user":
            formatted_messages.append(
                ChatMessage(role=MessageRole.USER, content=content)
            )
        else:
            logger.warning(f"Unknown role {role}, treating as user message")
            formatted_messages.append(
                ChatMessage(role=MessageRole.USER, content=content)
            )

    return formatted_messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def llama_index_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str:
    """Complete the prompt using LlamaIndex."""
    try:
        # Format messages for chat
        formatted_messages = []

        # Add system message if provided
        if system_prompt:
            formatted_messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )

        # Add history messages
        for msg in history_messages:
            formatted_messages.append(
                ChatMessage(
                    role=MessageRole.USER
                    if msg["role"] == "user"
                    else MessageRole.ASSISTANT,
                    content=msg["content"],
                )
            )

        # Add current prompt
        formatted_messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        # Get LLM instance from kwargs
        if "llm_instance" not in kwargs:
            raise ValueError("llm_instance must be provided in kwargs")
        llm = kwargs["llm_instance"]

        # Get response
        response: ChatResponse = await llm.achat(messages=formatted_messages)

        # In newer versions, the response is in message.content
        content = response.message.content
        return content

    except Exception as e:
        logger.error(f"Error in llama_index_complete_if_cache: {str(e)}")
        raise


async def llama_index_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    settings: LlamaIndexSettings = None,
    **kwargs,
) -> str:
    """
    Main completion function for LlamaIndex

    Args:
        prompt: Input prompt
        system_prompt: Optional system prompt
        history_messages: Optional chat history
        keyword_extraction: Whether to extract keywords from response
        settings: Optional LlamaIndex settings
        **kwargs: Additional arguments
    """
    if history_messages is None:
        history_messages = []

    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await llama_index_complete_if_cache(
        kwargs.get("llm_instance"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:
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
async def llama_index_embed(
    texts: list[str],
    embed_model: BaseEmbedding = None,
    settings: LlamaIndexSettings = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate embeddings using LlamaIndex

    Args:
        texts: List of texts to embed
        embed_model: LlamaIndex embedding model
        settings: Optional LlamaIndex settings
        **kwargs: Additional arguments
    """
    if settings:
        configure_llama_index(settings)

    if embed_model is None:
        raise ValueError("embed_model must be provided")

    # Use _get_text_embeddings for batch processing
    embeddings = embed_model._get_text_embeddings(texts)
    return np.array(embeddings)
