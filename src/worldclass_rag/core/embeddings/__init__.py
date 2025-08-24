"""
Embeddings module for WorldClass RAG.

Provides different embedding models following best practices from AI News & Strategy Daily.
"""

from .base import EmbeddingModel
from .openai_embeddings import OpenAIEmbeddings
from .sentence_transformer_embeddings import SentenceTransformerEmbeddings

__all__ = [
    "EmbeddingModel",
    "OpenAIEmbeddings", 
    "SentenceTransformerEmbeddings",
]