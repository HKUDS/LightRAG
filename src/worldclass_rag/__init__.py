"""
WorldClass RAG - Un módulo RAG de clase mundial para integración fácil.

Este módulo implementa las mejores prácticas de Retrieval-Augmented Generation
siguiendo los insights de AI News & Strategy Daily y la investigación más reciente.
"""

from .core.rag_engine import RAGEngine
from .core.chunking import (
    ChunkingStrategy,
    SemanticChunker,
    RecursiveChunker,
    SentenceChunker,
    FixedSizeChunker,
)
from .core.embeddings import EmbeddingModel, OpenAIEmbeddings, SentenceTransformerEmbeddings
from .core.retrieval import HybridRetriever, SemanticRetriever, KeywordRetriever
from .core.generation import ResponseGenerator
from .core.evaluation import RAGEvaluator, EvaluationMetrics
from .storage.vector import VectorStore, ChromaVectorStore
from .config.settings import RAGConfig

__version__ = "1.0.0"
__author__ = "WorldClass RAG Team"
__email__ = "team@worldclassrag.dev"

__all__ = [
    # Main engine
    "RAGEngine",
    
    # Chunking
    "ChunkingStrategy",
    "SemanticChunker", 
    "RecursiveChunker",
    "SentenceChunker",
    "FixedSizeChunker",
    
    # Embeddings
    "EmbeddingModel",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    
    # Retrieval
    "HybridRetriever",
    "SemanticRetriever", 
    "KeywordRetriever",
    
    # Generation
    "ResponseGenerator",
    
    # Evaluation
    "RAGEvaluator",
    "EvaluationMetrics",
    
    # Storage
    "VectorStore",
    "ChromaVectorStore",
    
    # Config
    "RAGConfig",
]