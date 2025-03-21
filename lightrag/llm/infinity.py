import os
import numpy as np
import importlib.util
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs

# Engine array cache to prevent reloading for each embedding call
_ENGINE_ARRAY_CACHE = None
_MODELS_INITIALIZED = set()

def is_available():
    """
    Check if infinity-emb package is available.

    Returns:
        bool: True if infinity-emb is installed, False otherwise
    """
    return importlib.util.find_spec("infinity_emb") is not None


async def load_infinity_model(model_name, engine="optimum", device="cpu"):
    """
    Load Infinity model from Hugging Face using AsyncEngineArray.

    Args:
        model_name: Name of the model to load from Hugging Face
        engine: Engine to use for model loading (e.g., "torch", "optimum")
        device: Device to run the model on (e.g., "cuda", "cpu"). If None, uses default.

    Returns:
        AsyncEmbeddingEngine instance for the model

    Raises:
        ImportError: If infinity-emb package is not installed
        ValueError: If there's an error loading the model
    """
    if not is_available():
        raise ImportError(
            "Please install infinity-emb package: pip install infinity-emb[all]"
        )

    global _ENGINE_ARRAY_CACHE, _MODELS_INITIALIZED

    try:
        from infinity_emb import AsyncEngineArray, EngineArgs

        # Initialize engine array if not already initialized
        if _ENGINE_ARRAY_CACHE is None:
            engine_args = EngineArgs(
                model_name_or_path=model_name,
                engine=engine,
                device=device,
                trust_remote_code=True,
            )
            _ENGINE_ARRAY_CACHE = AsyncEngineArray.from_args([engine_args])

        # Get the engine for this model
        engine = _ENGINE_ARRAY_CACHE[model_name]

        # Start the engine if not already started
        if model_name not in _MODELS_INITIALIZED:
            await engine.astart()
            _MODELS_INITIALIZED.add(model_name)

        return engine
    except ImportError as e:
        raise ImportError(
            "Please install infinity-emb package: pip install infinity-emb[all]"
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
    engine: str = "optimum",
    device: str = "cpu",
    **kwargs,
) -> np.ndarray:
    """
    Generate embeddings using a local Infinity model.

    Args:
        texts: List of texts to embed
        model_name: Name of the model to load from Hugging Face
        engine: Engine to use ("torch" or "optimum" for CPU with OpenVINO)
        device: Device to run on ("cuda" or "cpu"). If None, uses default.
        **kwargs: Additional arguments to pass to the Infinity model

    Returns:
        NumPy array of embeddings

    Raises:
        ImportError: If infinity-emb package is not installed
        ValueError: If there's an error generating embeddings
    """
    if not is_available():
        raise ImportError(
            "Please install infinity-emb package: pip install infinity-emb[all]"
        )

    if not texts:
        return np.array([])

    engine_obj = await load_infinity_model(model_name, engine=engine, device=device)

    # Get embedding dimension and max tokens from the model if available
    # The wrapped function will use these values if provided
    embed_dim = kwargs.get("embedding_dim", 1024)
    infinity_embed.embedding_dim = embed_dim

    max_tokens = kwargs.get("max_token_size", 8192)
    infinity_embed.max_token_size = max_tokens

    embeddings, _ = await engine_obj.embed(sentences=texts)

    # Convert to numpy array
    if isinstance(embeddings, list):
        return np.array(embeddings)
    return embeddings


async def cleanup_infinity_models():
    """
    Cleanup function to stop all infinity engines properly.
    Should be called when the application is shutting down.

    Note: Does nothing if infinity-emb is not installed or no models were initialized.
    """
    if not is_available():
        return

    global _ENGINE_ARRAY_CACHE, _MODELS_INITIALIZED

    if _ENGINE_ARRAY_CACHE is not None:
        for model_name in _MODELS_INITIALIZED:
            try:
                engine = _ENGINE_ARRAY_CACHE[model_name]
                await engine.astop()
            except Exception:
                pass
        _MODELS_INITIALIZED.clear()
        _ENGINE_ARRAY_CACHE = None
