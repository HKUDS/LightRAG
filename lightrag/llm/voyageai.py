import os
import numpy as np
import pipmaster as pm  # Pipmaster for dynamic library install

# Add Voyage AI import
if not pm.is_installed("voyageai"):
    pm.install("voyageai")

from voyageai.error import (
    RateLimitError,
    APIConnectionError,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs, logger


# Custome exceptions for VoyageAI errors
class VoyageAIError(Exception):
    """Generic VoyageAI API error"""

    pass


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=16000)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def voyageai_embed(
    texts: list[str],
    model: str = "voyage-3",
    api_key: str | None = None,
    embedding_dim: int | None = None,
    input_type: str | None = None,
    truncation: bool | None = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using VoyageAI's API.

    Args:
        texts: List of texts to embed.
        model: The VoyageAI embedding model to use. Options include:
            - "voyage-3": General purpose (1024 dims, 32K context)
            - "voyage-3-lite": Lightweight (512 dims, 32K context)
            - "voyage-3-large": Highest accuracy (1024 dims, 32K context)
            - "voyage-code-3": Code optimized (1024 dims, 32K context)
            - "voyage-law-2": Legal documents (1024 dims, 16K context)
            - "voyage-finance-2": Finance (1024 dims, 32K context)
        api_key: Optional VoyageAI API key. If None, uses VOYAGEAI_API_KEY environment variable.
        input_type: Optional input type hint for the model. Options:
            - "query": For search queries
            - "document": For documents to be indexed
            - None: Let the model decide (default)
        truncation: Whether to truncate texts that exceed token limit (default: None).

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        VoyageAIError: If the API call fails or returns invalid data.

    """

    try:
        import voyageai
    except ImportError:
        raise ImportError(
            "voyageai package is required. Install it with: pip install voyageai"
        )

    # Get API key from parameter or environment
    logger.debug(
        "Starting VoyageAI embedding generation. (Ignore api_key, use env variable)"
    )
    if not api_key:
        api_key = os.environ.get("VOYAGEAI_API_KEY")
        if not api_key:
            logger.error("VOYAGEAI_API_KEY environment variable not set")
            raise ValueError(
                "VOYAGEAI_API_KEY environment variable is required or pass api_key parameter"
            )

    try:
        # Create async client
        client = voyageai.AsyncClient(api_key=api_key)

        logger.debug(f"VoyageAI embedding request: {len(texts)} texts, model: {model}")
        # Calculate total characters for debugging
        total_chars = sum(len(t) for t in texts)
        avg_chars = total_chars / len(texts) if texts else 0
        logger.debug(
            f"VoyageAI embedding request: {len(texts)} texts, "
            f"total_chars={total_chars}, avg_chars={avg_chars:.0f}, model={model}"
        )

        # Prepare API call parameters
        embed_params = dict(
            texts=texts,
            model=model,
            # Optional parameters -- if None, voyageai client uses defaults
            output_dimension=embedding_dim,
            truncation=truncation,
            input_type=input_type,
        )
        # Make API call with timing
        result = await client.embed(**embed_params)

        if not result.embeddings:
            err_msg = "VoyageAI API returned empty embeddings"
            logger.error(err_msg)
            raise VoyageAIError(err_msg)

        if len(result.embeddings) != len(texts):
            err_msg = f"VoyageAI API returned {len(result.embeddings)} embeddings for {len(texts)} texts"
            logger.error(err_msg)
            raise VoyageAIError(err_msg)

        # Convert to numpy array with timing
        embeddings = np.array(result.embeddings, dtype=np.float32)
        logger.debug(f"VoyageAI embeddings generated: shape {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"VoyageAI embedding error: {e}")
        raise


# Optional: a helper function to get available embedding models
def get_available_embedding_models() -> dict[str, dict]:
    """
    Returns a dictionary of available Voyage AI embedding models and their properties.
    """
    return {
        "voyage-3-large": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Best general-purpose and multilingual",
        },
        "voyage-3": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "General-purpose and multilingual",
        },
        "voyage-3-lite": {
            "context_length": 32000,
            "dimension": 512,
            "description": "Optimized for latency and cost",
        },
        "voyage-code-3": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Optimized for code",
        },
        "voyage-finance-2": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Optimized for finance",
        },
        "voyage-law-2": {
            "context_length": 16000,
            "dimension": 1024,
            "description": "Optimized for legal",
        },
        "voyage-multimodal-3": {
            "context_length": 32000,
            "dimension": 1024,
            "description": "Multimodal text and images",
        },
    }
