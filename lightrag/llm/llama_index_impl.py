from typing import Any

import pipmaster as pm
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)

from lightrag.utils import logger

# Install required dependencies
if not pm.is_installed('llama-index'):
    pm.install('llama-index')

import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.exceptions import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)


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
            logger.warning(f'Unknown LlamaIndex setting: {key}')

    # Set as global settings
    LlamaIndexSettings.set_global(settings)
    return settings


def format_chat_messages(messages):
    """Format chat messages into LlamaIndex format."""
    formatted_messages = []

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            formatted_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=content))
        elif role == 'assistant':
            formatted_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
        elif role == 'user':
            formatted_messages.append(ChatMessage(role=MessageRole.USER, content=content))
        else:
            logger.warning(f'Unknown role {role}, treating as user message')
            formatted_messages.append(ChatMessage(role=MessageRole.USER, content=content))

    return formatted_messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def llama_index_complete_if_cache(
    model: Any,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    enable_cot: bool = False,
    chat_kwargs=None,
) -> str:
    """Complete the prompt using LlamaIndex."""
    if chat_kwargs is None:
        chat_kwargs = {}
    if history_messages is None:
        history_messages = []
    if enable_cot:
        logger.debug('enable_cot=True is not supported for LlamaIndex implementation and will be ignored.')
    try:
        # Format messages for chat
        formatted_messages = []

        # Add system message if provided
        if system_prompt:
            formatted_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        # Add history messages
        for msg in history_messages:
            formatted_messages.append(
                ChatMessage(
                    role=MessageRole.USER if msg['role'] == 'user' else MessageRole.ASSISTANT,
                    content=msg['content'],
                )
            )

        # Add current prompt
        formatted_messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        response: ChatResponse = await model.achat(messages=formatted_messages, **chat_kwargs)

        # In newer versions, the response is in message.content
        content = response.message.content
        return content

    except Exception as e:
        logger.error(f'Error in llama_index_complete_if_cache: {e!s}')
        raise


async def llama_index_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
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

    kwargs.pop('keyword_extraction', None)
    result = await llama_index_complete_if_cache(
        kwargs.get('llm_instance'),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
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
        raise ValueError('embed_model must be provided')

    # Use _get_text_embeddings for batch processing
    embeddings = embed_model._get_text_embeddings(texts)
    return np.array(embeddings)
