"""
Chunking strategies for RAG - Implementación de estrategias inteligentes de segmentación.

Siguiendo las mejores prácticas del video AI News & Strategy Daily:
- Chunking con superposición para evitar pérdida de contexto
- Múltiples estrategias: semántico, recursivo, basado en oraciones
- Preservación de relaciones espaciales para tablas
- Metadatos enriquecidos para mejor recuperación
"""

from .base import ChunkingStrategy, Chunk
from .semantic import SemanticChunker
from .recursive import RecursiveChunker
from .sentence import SentenceChunker
from .fixed_size import FixedSizeChunker

__all__ = [
    "ChunkingStrategy",
    "Chunk", 
    "SemanticChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "FixedSizeChunker",
]