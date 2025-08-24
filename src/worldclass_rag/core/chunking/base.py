"""
Base classes and interfaces for chunking strategies.

Siguiendo las mejores prácticas identificadas en AI News & Strategy Daily:
- La mala segmentación arruina muchos proyectos RAG
- Siempre incluir superposición entre chunks
- Preservar contexto y relaciones semánticas
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

@dataclass
class Chunk:
    """
    Representa un fragmento de texto procesado con metadatos enriquecidos.
    
    Siguiendo las mejores prácticas:
    - Metadatos de fuente, sección y fecha para mejor recuperación
    - Información de superposición para preservar contexto
    - Índices para reconstrucción si es necesario
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Información de posición y contexto
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    chunk_id: Optional[str] = None
    
    # Información de superposición
    has_overlap_before: bool = False
    has_overlap_after: bool = False
    overlap_before_content: str = ""
    overlap_after_content: str = ""
    
    # Metadatos automáticos
    created_at: datetime = field(default_factory=datetime.now)
    token_count: Optional[int] = None
    character_count: int = field(init=False)
    
    def __post_init__(self):
        """Calcula automáticamente estadísticas del chunk."""
        self.character_count = len(self.content)
        
        # Genera ID único si no se proporciona
        if not self.chunk_id:
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            timestamp = int(self.created_at.timestamp())
            self.chunk_id = f"chunk_{timestamp}_{content_hash}"
    
    def get_full_content(self) -> str:
        """
        Retorna el contenido completo incluyendo superposiciones.
        Útil para preservar contexto completo cuando sea necesario.
        """
        full_content = []
        
        if self.has_overlap_before and self.overlap_before_content:
            full_content.append(self.overlap_before_content)
            
        full_content.append(self.content)
        
        if self.has_overlap_after and self.overlap_after_content:
            full_content.append(self.overlap_after_content)
            
        return " ".join(full_content)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Añade metadatos de forma segura."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtiene metadatos de forma segura."""
        return self.metadata.get(key, default)


class ChunkingStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias de chunking.
    
    Implementa el patrón Strategy para permitir diferentes
    algoritmos de segmentación según el caso de uso.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_sentence_boundaries: bool = True,
        add_metadata: bool = True,
    ):
        """
        Inicializa la estrategia de chunking.
        
        Args:
            chunk_size: Tamaño objetivo de cada chunk en caracteres
            chunk_overlap: Superposición entre chunks para preservar contexto
            preserve_sentence_boundaries: Si preservar límites de oraciones
            add_metadata: Si añadir metadatos automáticamente
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_sentence_boundaries = preserve_sentence_boundaries
        self.add_metadata = add_metadata
        
        # Validaciones siguiendo mejores prácticas
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap debe ser menor que chunk_size para evitar duplicación excesiva"
            )
            
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap debe ser >= 0")
            
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser > 0")
    
    @abstractmethod
    def split_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide el texto en chunks según la estrategia específica.
        
        Args:
            text: Texto a dividir
            metadata: Metadatos base para todos los chunks generados
            
        Returns:
            Lista de chunks procesados con superposición y metadatos
        """
        pass
    
    def _add_overlap_info(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Añade información de superposición a los chunks.
        
        Implementa la mejor práctica de superposición para preservar contexto.
        """
        if len(chunks) <= 1:
            return chunks
            
        for i, chunk in enumerate(chunks):
            # Chunk anterior (superposición antes)
            if i > 0:
                prev_chunk = chunks[i-1]
                overlap_start = max(0, len(prev_chunk.content) - self.chunk_overlap)
                chunk.overlap_before_content = prev_chunk.content[overlap_start:]
                chunk.has_overlap_before = True
            
            # Chunk siguiente (superposición después)  
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                overlap_end = min(len(next_chunk.content), self.chunk_overlap)
                chunk.overlap_after_content = next_chunk.content[:overlap_end]
                chunk.has_overlap_after = True
                
        return chunks
    
    def _add_base_metadata(
        self, 
        chunk: Chunk, 
        base_metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> Chunk:
        """
        Añade metadatos base siguiendo mejores prácticas.
        
        Incluye fuente, sección, fecha y posición para mejor recuperación.
        """
        if not self.add_metadata:
            return chunk
            
        if base_metadata:
            chunk.metadata.update(base_metadata)
        
        # Metadatos de chunking
        chunk.add_metadata("chunk_index", chunk_index)
        chunk.add_metadata("total_chunks", total_chunks)
        chunk.add_metadata("chunking_strategy", self.__class__.__name__)
        chunk.add_metadata("chunk_size_config", self.chunk_size)
        chunk.add_metadata("overlap_size_config", self.chunk_overlap)
        
        # Metadatos de calidad
        chunk.add_metadata("character_count", chunk.character_count)
        if chunk.token_count:
            chunk.add_metadata("token_count", chunk.token_count)
            
        # Timestamp para recuperación basada en actualidad
        chunk.add_metadata("processed_at", chunk.created_at.isoformat())
        
        return chunk
    
    def chunk_document(
        self,
        text: str,
        source: Optional[str] = None,
        section: Optional[str] = None,
        document_date: Optional[datetime] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Interfaz de alto nivel para procesar un documento completo.
        
        Implementa las mejores prácticas de metadatos enriquecidos.
        """
        base_metadata = {}
        
        if source:
            base_metadata["source"] = source
            
        if section:  
            base_metadata["section"] = section
            
        if document_date:
            base_metadata["document_date"] = document_date.isoformat()
            base_metadata["document_timestamp"] = document_date.timestamp()
            
        if additional_metadata:
            base_metadata.update(additional_metadata)
            
        return self.split_text(text, base_metadata)
    
    def estimate_chunks(self, text: str) -> int:
        """
        Estima el número de chunks que se generarán.
        Útil para planificación y progreso.
        """
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        return max(1, len(text) // effective_chunk_size)
    
    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Valida la calidad de los chunks generados.
        Implementa verificaciones de calidad recomendadas.
        """
        if not chunks:
            return {"valid": False, "error": "No chunks generated"}
            
        validation_results = {
            "valid": True,
            "total_chunks": len(chunks),
            "average_size": sum(len(c.content) for c in chunks) / len(chunks),
            "size_variance": 0,
            "chunks_with_overlap": sum(1 for c in chunks if c.has_overlap_before or c.has_overlap_after),
            "warnings": []
        }
        
        sizes = [len(c.content) for c in chunks]
        avg_size = validation_results["average_size"]
        validation_results["size_variance"] = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        
        # Validaciones de calidad
        for i, chunk in enumerate(chunks):
            if len(chunk.content) < 10:
                validation_results["warnings"].append(f"Chunk {i} muy pequeño: {len(chunk.content)} caracteres")
                
            if len(chunk.content) > self.chunk_size * 2:
                validation_results["warnings"].append(f"Chunk {i} muy grande: {len(chunk.content)} caracteres")
        
        return validation_results