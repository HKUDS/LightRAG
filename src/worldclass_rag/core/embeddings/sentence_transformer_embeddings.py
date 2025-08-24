"""
Sentence Transformers Embeddings implementation.

Open-source alternative to OpenAI embeddings using HuggingFace models.
"""

import time
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import EmbeddingModel, EmbeddingResult


class SentenceTransformerEmbeddings(EmbeddingModel):
    """
    Sentence Transformers embeddings implementation.
    
    Provides open-source embedding models that can run locally,
    offering privacy and cost benefits compared to API-based solutions.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        cache_folder: Optional[str] = None,
    ):
        """
        Initialize Sentence Transformers embeddings.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cpu', 'cuda', or None for auto)
            trust_remote_code: Whether to trust remote code in model
            cache_folder: Custom cache folder for models
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package is required")
        
        # Model dimension mapping for common models
        model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "multi-qa-MiniLM-L6-cos-v1": 384,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "distiluse-base-multilingual-cased": 512,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        
        # Load model to get actual dimensions
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                cache_folder=cache_folder
            )
            
            # Get actual dimensions from model
            dimensions = self.model.get_sentence_embedding_dimension()
            
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {e}")
        
        super().__init__(model_name=model_name, dimensions=dimensions)
        
        self.device = device or self.model.device
        
        print(f"Loaded {model_name} with {dimensions} dimensions on {self.device}")
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single text using Sentence Transformers.
        """
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return EmbeddingResult(
                embeddings=embedding.astype(np.float32),
                model_name=self.model_name,
                dimensions=len(embedding),
                processing_time_ms=processing_time,
                metadata={
                    "provider": "sentence_transformers",
                    "device": str(self.device),
                    "normalized": True,
                    "single_text": True,
                }
            )
            
        except Exception as e:
            raise Exception(f"Error generating Sentence Transformer embedding: {e}")
    
    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed multiple texts in batch for efficiency.
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                model_name=self.model_name,
                dimensions=self.dimensions,
            )
        
        start_time = time.time()
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,  # Show progress for large batches
                batch_size=32  # Reasonable batch size for most GPUs
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return EmbeddingResult(
                embeddings=embeddings.astype(np.float32),
                model_name=self.model_name,
                dimensions=self.dimensions,
                processing_time_ms=processing_time,
                metadata={
                    "provider": "sentence_transformers",
                    "device": str(self.device),
                    "normalized": True,
                    "batch_size": len(texts),
                    "actual_dimensions": embeddings.shape[1] if embeddings.size > 0 else 0,
                }
            )
            
        except Exception as e:
            raise Exception(f"Error generating Sentence Transformer batch embeddings: {e}")
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        try:
            # Test with a simple embedding
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            return test_embedding is not None and len(test_embedding) > 0
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Sentence Transformer model."""
        base_info = super().get_model_info()
        
        # Get model details
        model_info = {
            "provider": "sentence_transformers",
            "device": str(self.device),
            "supports_batch": True,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "is_local": True,
            "cost_per_embedding": 0.0,  # Free to run locally
        }
        
        # Add GPU info if available
        if 'cuda' in str(self.device):
            try:
                gpu_info = {
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
                model_info.update(gpu_info)
            except:
                pass
        
        base_info.update(model_info)
        return base_info
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of recommended Sentence Transformer models.
        
        Returns models suitable for different use cases.
        """
        return [
            {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "description": "Fast and lightweight, good for general purpose",
                "use_case": "General retrieval, fast inference",
                "languages": ["English"],
                "size_mb": 80,
            },
            {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "description": "Higher quality embeddings, slower inference",
                "use_case": "High accuracy retrieval",
                "languages": ["English"],
                "size_mb": 420,
            },
            {
                "name": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": 384,
                "description": "Multilingual support, good for diverse content",
                "use_case": "Multilingual retrieval",
                "languages": ["50+ languages"],
                "size_mb": 420,
            },
            {
                "name": "multi-qa-MiniLM-L6-cos-v1",
                "dimensions": 384,
                "description": "Optimized for question-answering tasks",
                "use_case": "Q&A systems",
                "languages": ["English"],
                "size_mb": 80,
            },
            {
                "name": "distiluse-base-multilingual-cased",
                "dimensions": 512,
                "description": "Multilingual, balanced performance",
                "use_case": "Multilingual general purpose",
                "languages": ["15+ languages"],
                "size_mb": 480,
            }
        ]
    
    def benchmark_performance(self, test_texts: List[str]) -> Dict[str, Any]:
        """
        Benchmark the model performance on given texts.
        
        Useful for comparing different models or configurations.
        """
        if not test_texts:
            test_texts = [
                "This is a test sentence for benchmarking.",
                "Another example text to measure performance.",
                "Sentence transformers provide excellent embeddings for semantic search."
            ]
        
        # Single text benchmark
        start_time = time.time()
        single_result = self.embed_text(test_texts[0])
        single_time = time.time() - start_time
        
        # Batch benchmark
        start_time = time.time()
        batch_result = self.embed_batch(test_texts)
        batch_time = time.time() - start_time
        
        return {
            "model": self.model_name,
            "device": str(self.device),
            "single_embedding_time_ms": single_time * 1000,
            "batch_embedding_time_ms": batch_time * 1000,
            "batch_size": len(test_texts),
            "embeddings_per_second_single": 1.0 / single_time if single_time > 0 else 0,
            "embeddings_per_second_batch": len(test_texts) / batch_time if batch_time > 0 else 0,
            "dimensions": self.dimensions,
            "memory_efficient": batch_time < single_time * len(test_texts) * 0.8,
        }
    
    def similarity_search_demo(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Demonstrate similarity search capabilities.
        
        Useful for testing and debugging the embedding quality.
        """
        # Embed query and documents
        query_result = self.embed_text(query)
        docs_result = self.embed_batch(documents)
        
        # Find most similar documents
        similarities = []
        for i, doc_embedding in enumerate(docs_result.embeddings):
            similarity = self.cosine_similarity(query_result.embeddings, doc_embedding)
            similarities.append({
                "index": i,
                "document": documents[i],
                "similarity": float(similarity),
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]