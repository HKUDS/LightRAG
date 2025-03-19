import os
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs

# Model cache to prevent reloading for each embedding call
_MODEL_CACHE = {}

async def load_infinity_model(model_name):
    """
    Load Infinity model from Hugging Face.

    Args:
        model_name: Name of the model to load from Hugging Face

    Returns:
        Loaded infinity model
    """
    # Check if model is already in cache
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from infinity import InfinityEmbedding
        model = InfinityEmbedding(model_name)
        # Cache the model for future use
        _MODEL_CACHE[model_name] = model
        return model
    except ImportError as e:
        raise ImportError(
            "Please install infinity-embed package: pip install infinity-embed"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load Infinity model {model_name}: {str(e)}") from e

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, ValueError)),
)
async def infinity_embed(
    texts: list[str],
    model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
    **kwargs
) -> np.ndarray:
    """
    Generate embeddings using a local Infinity model.

    Args:
        texts: List of texts to embed
        model_name: Name of the model to load from Hugging Face
        **kwargs: Additional arguments to pass to the Infinity model

    Returns:
        NumPy array of embeddings
    """
    if not texts:
        return np.array([])

    model = await load_infinity_model(model_name)

    # Get embedding dimension and max tokens from the model if available
    # The wrapped function will use these values if provided
    embed_dim = kwargs.get("embedding_dim", 1024)
    infinity_embed.embedding_dim = embed_dim

    max_tokens = kwargs.get("max_token_size", 8192)
    infinity_embed.max_token_size = max_tokens

    # Check if this is a query embedding - if so add the prefix for Snowflake models
    if model_name.startswith("Snowflake/") and kwargs.get("is_query", False):
        # Add query prefix for Snowflake models
        prefixed_texts = ["query: " + text for text in texts]
        embeddings = model.embed(prefixed_texts)
    else:
        # Generate embeddings
        embeddings = model.embed(texts)

    # Convert to numpy array
    if isinstance(embeddings, list):
        return np.array(embeddings)
    return embeddings