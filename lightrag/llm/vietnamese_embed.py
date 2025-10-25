"""
Vietnamese Embedding Integration for LightRAG
Model: AITeamVN/Vietnamese_Embedding
Base: BAAI/bge-m3
"""

import os
import numpy as np
import torch
from functools import lru_cache

import pipmaster as pm

# Install required packages
if not pm.is_installed("transformers"):
    pm.install("transformers")
if not pm.is_installed("torch"):
    pm.install("torch")
if not pm.is_installed("numpy"):
    pm.install("numpy")

from transformers import AutoTokenizer, AutoModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs, logger
from lightrag.exceptions import APIConnectionError, RateLimitError, APITimeoutError

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@lru_cache(maxsize=1)
def initialize_vietnamese_embedding_model(
    model_name: str = "AITeamVN/Vietnamese_Embedding",
    token: str | None = None,
):
    """
    Initialize the Vietnamese Embedding model with caching.
    
    Args:
        model_name: HuggingFace model identifier
        token: HuggingFace API token for model access
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading Vietnamese Embedding model: {model_name}")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        logger.info("Vietnamese Embedding model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load Vietnamese Embedding model: {e}")
        raise


@wrap_embedding_func_with_attrs(embedding_dim=1024)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def vietnamese_embed(
    texts: list[str],
    model_name: str = "AITeamVN/Vietnamese_Embedding",
    token: str | None = None,
) -> np.ndarray:
    """
    Generate embeddings for Vietnamese texts using AITeamVN/Vietnamese_Embedding model.
    
    This model is based on BGE-M3 and fine-tuned on Vietnamese data with:
    - Maximum sequence length: 2048 tokens
    - Output dimensionality: 1024 dimensions
    - Similarity function: Dot product similarity
    
    Args:
        texts: List of texts to embed (in Vietnamese or other languages)
        model_name: HuggingFace model identifier (default: AITeamVN/Vietnamese_Embedding)
        token: HuggingFace API token for model access
        
    Returns:
        numpy array of embeddings with shape (len(texts), 1024)
        
    Raises:
        APIConnectionError: If there is a connection error
        RateLimitError: If rate limit is exceeded
        APITimeoutError: If request times out
    """
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_vietnamese_embedding_model(model_name, token)
    
    # Detect the appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug("Using CUDA device for embedding")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Using MPS device for embedding")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device for embedding")
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    try:
        # Tokenize texts with max_length matching the model's training
        # Vietnamese_Embedding was trained with max_length=2048
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Use mean pooling on the token embeddings
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings for dot product similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy array
        if embeddings.dtype == torch.bfloat16:
            embeddings_np = embeddings.to(torch.float32).cpu().numpy()
        else:
            embeddings_np = embeddings.cpu().numpy()
        
        logger.debug(f"Generated embeddings for {len(texts)} texts, shape: {embeddings_np.shape}")
        return embeddings_np
        
    except Exception as e:
        logger.error(f"Error generating Vietnamese embeddings: {e}")
        raise APIConnectionError(f"Vietnamese embedding generation failed: {e}")


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings.
    
    Args:
        model_output: Model output containing token embeddings
        attention_mask: Attention mask to exclude padding tokens
        
    Returns:
        Pooled embeddings
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Convenience function for easier integration
@wrap_embedding_func_with_attrs(embedding_dim=1024)
async def vietnamese_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Convenience wrapper for Vietnamese embedding that reads token from environment.
    
    Set HUGGINGFACE_API_KEY or HF_TOKEN environment variable with your HuggingFace token.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        numpy array of embeddings
    """
    return await vietnamese_embed(texts)
