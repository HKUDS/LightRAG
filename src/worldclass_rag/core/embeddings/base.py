"""
Base embedding model interface.

Defines the contract for all embedding models in WorldClass RAG.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """
    Result of embedding operation with metadata.
    """
    embeddings: np.ndarray
    model_name: str
    dimensions: int
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.
    
    Implements the interface for embedding text following the best practices
    from AI News & Strategy Daily video:
    - Using 1,536 dimensions as current best practice
    - Similar meanings cluster together mathematically
    - Cosine similarity for finding nearest neighbors
    """
    
    def __init__(self, model_name: str, dimensions: int):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name identifier for the model
            dimensions: Embedding vector dimensions
        """
        self.model_name = model_name
        self.dimensions = dimensions
        
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResult with matrix of embeddings
        """
        pass
    
    def cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Implements the similarity measure recommended in the video
        for finding "nearest neighbors" in vector space.
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: Matrix of candidate vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, candidate)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "type": self.__class__.__name__,
        }
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and ready to use."""
        pass