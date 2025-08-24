"""
OpenAI Embeddings implementation.

Using OpenAI's text-embedding models following the 1,536 dimensions best practice
mentioned in the AI News & Strategy Daily video.
"""

import time
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import EmbeddingModel, EmbeddingResult


class OpenAIEmbeddings(EmbeddingModel):
    """
    OpenAI embeddings implementation using text-embedding models.
    
    Supports the latest OpenAI embedding models with 1,536 dimensions
    as recommended in the video for optimal performance.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI model name (text-embedding-3-large, text-embedding-3-small, etc.)
            api_key: OpenAI API key (if not set in environment)
            dimensions: Override default dimensions (use model default if None)
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI embeddings")
        
        # Model dimension mapping (following best practices)
        model_dimensions = {
            "text-embedding-3-large": 3072,  # Can be reduced to 1536
            "text-embedding-3-small": 1536,  # Best practice dimension
            "text-embedding-ada-002": 1536,   # Legacy but still good
        }
        
        if dimensions is None:
            dimensions = model_dimensions.get(model, 1536)
        
        super().__init__(model_name=model, dimensions=dimensions)
        
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Validate model availability
        if not self.is_available():
            print(f"Warning: Model {model} may not be available")
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single text using OpenAI API.
        """
        start_time = time.time()
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text],
                dimensions=self.dimensions if self.model_name.startswith("text-embedding-3") else None
            )
            
            # Extract embedding
            embedding_vector = np.array(response.data[0].embedding, dtype=np.float32)
            
            processing_time = (time.time() - start_time) * 1000
            
            return EmbeddingResult(
                embeddings=embedding_vector,
                model_name=self.model_name,
                dimensions=len(embedding_vector),
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None,
                processing_time_ms=processing_time,
                metadata={
                    "api_version": "openai_v1",
                    "single_text": True,
                }
            )
            
        except Exception as e:
            raise Exception(f"Error generating OpenAI embedding: {e}")
    
    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed multiple texts in batch for efficiency.
        
        OpenAI allows up to 2048 texts per batch for embedding models.
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                model_name=self.model_name,
                dimensions=self.dimensions,
            )
        
        start_time = time.time()
        
        try:
            # Handle large batches by chunking
            batch_size = 100  # Conservative batch size
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    dimensions=self.dimensions if self.model_name.startswith("text-embedding-3") else None
                )
                
                # Extract embeddings
                batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if hasattr(response, 'usage'):
                    total_tokens += response.usage.total_tokens
            
            embeddings_matrix = np.array(all_embeddings)
            processing_time = (time.time() - start_time) * 1000
            
            return EmbeddingResult(
                embeddings=embeddings_matrix,
                model_name=self.model_name,
                dimensions=self.dimensions,
                tokens_used=total_tokens if total_tokens > 0 else None,
                processing_time_ms=processing_time,
                metadata={
                    "api_version": "openai_v1",
                    "batch_size": len(texts),
                    "actual_dimensions": embeddings_matrix.shape[1] if embeddings_matrix.size > 0 else 0,
                }
            )
            
        except Exception as e:
            raise Exception(f"Error generating OpenAI batch embeddings: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available and model exists."""
        try:
            # Test with a simple embedding
            test_response = self.client.embeddings.create(
                model=self.model_name,
                input=["test"],
                dimensions=self.dimensions if self.model_name.startswith("text-embedding-3") else None
            )
            return True
        except Exception as e:
            print(f"OpenAI model {self.model_name} not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the OpenAI model."""
        base_info = super().get_model_info()
        base_info.update({
            "provider": "OpenAI",
            "api_version": "v1",
            "supports_batch": True,
            "max_batch_size": 2048,
            "pricing_per_1k_tokens": self._get_pricing_info(),
        })
        return base_info
    
    def _get_pricing_info(self) -> Optional[float]:
        """Get pricing information for the model (approximate)."""
        pricing = {
            "text-embedding-3-large": 0.00013,   # $0.00013 per 1K tokens
            "text-embedding-3-small": 0.00002,   # $0.00002 per 1K tokens  
            "text-embedding-ada-002": 0.0001,    # $0.0001 per 1K tokens
        }
        return pricing.get(self.model_name)
    
    def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        """
        Estimate the cost of embedding the given texts.
        
        Returns:
            Dictionary with cost estimation details
        """
        # Rough token estimation (4 chars ≈ 1 token for English)
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4
        
        pricing_per_1k = self._get_pricing_info()
        estimated_cost = (estimated_tokens / 1000) * pricing_per_1k if pricing_per_1k else None
        
        return {
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost,
            "pricing_per_1k_tokens": pricing_per_1k,
            "model": self.model_name,
            "text_count": len(texts),
            "total_characters": total_chars,
        }