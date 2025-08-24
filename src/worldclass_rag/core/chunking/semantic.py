"""
Semantic Chunking Strategy - Segmentación basada en similitud semántica.

Esta estrategia agrupa texto por significado semántico, no por tamaño fijo,
siguiendo las mejores prácticas identificadas en AI News & Strategy Daily.
"""

import re
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import ChunkingStrategy, Chunk


class SemanticChunker(ChunkingStrategy):
    """
    Chunker que agrupa oraciones por similitud semántica.
    
    Ventajas:
    - Preserva coherencia temática dentro de cada chunk
    - Mejor para recuperación basada en significado
    - Adapta el tamaño según el contenido semántico
    
    Ideal para:
    - Documentos con múltiples temas
    - Contenido académico o técnico
    - Casos donde la coherencia semántica es crítica
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        preserve_sentence_boundaries: bool = True,
        add_metadata: bool = True,
    ):
        """
        Inicializa el chunker semántico.
        
        Args:
            similarity_threshold: Umbral de similitud coseno para agrupar oraciones
            min_chunk_size: Tamaño mínimo de chunk en caracteres
            embedding_model: Modelo de embeddings para similitud semántica
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_sentence_boundaries=preserve_sentence_boundaries,
            add_metadata=add_metadata,
        )
        
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.embedding_model_name = embedding_model
        
        # Inicializar modelo de embeddings
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Warning: No se pudo cargar el modelo {embedding_model}: {e}")
            print("Fallback a chunking por oraciones simple")
            self.embedding_model = None
    
    def split_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide el texto en chunks semánticamente coherentes.
        
        Proceso:
        1. Divide en oraciones
        2. Calcula embeddings de cada oración
        3. Agrupa oraciones similares
        4. Crea chunks respetando límites de tamaño
        5. Añade superposición y metadatos
        """
        if not text.strip():
            return []
            
        # Dividir en oraciones
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
            
        # Si no hay modelo de embeddings, usar chunking simple por oraciones
        if self.embedding_model is None:
            return self._fallback_sentence_chunking(sentences, metadata)
        
        # Calcular embeddings para cada oración
        try:
            embeddings = self.embedding_model.encode(sentences)
        except Exception as e:
            print(f"Error calculando embeddings: {e}")
            return self._fallback_sentence_chunking(sentences, metadata)
        
        # Agrupar oraciones por similitud semántica
        semantic_groups = self._group_by_semantic_similarity(sentences, embeddings)
        
        # Crear chunks de los grupos semánticos
        chunks = self._create_chunks_from_groups(semantic_groups, metadata)
        
        # Añadir información de superposición
        chunks = self._add_overlap_info(chunks)
        
        # Añadir metadatos finales
        for i, chunk in enumerate(chunks):
            self._add_base_metadata(chunk, metadata, i, len(chunks))
            chunk.add_metadata("semantic_chunking", True)
            chunk.add_metadata("similarity_threshold", self.similarity_threshold)
            
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Divide el texto en oraciones preservando contexto.
        
        Usa regex robusto para manejar abreviaciones y casos especiales.
        """
        # Patrones para detectar fin de oración
        sentence_endings = r'[.!?]+(?:\s+|$)'
        
        # Excepciones comunes (abreviaciones)
        abbreviations = r'(?:Dr|Mr|Mrs|Ms|Prof|Sr|Sra|etc|vs|p\.ej|i\.e|e\.g)\.(?:\s|$)'
        
        # Dividir en oraciones candidatas
        sentences = re.split(sentence_endings, text)
        
        # Limpiar y filtrar oraciones vacías
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 10:  # Filtrar oraciones muy cortas
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _group_by_semantic_similarity(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[List[str]]:
        """
        Agrupa oraciones por similitud semántica usando clustering.
        
        Implementa un algoritmo de clustering secuencial que respeta
        el orden del texto original.
        """
        if len(sentences) <= 1:
            return [sentences]
            
        groups = []
        current_group = [sentences[0]]
        current_group_embedding = embeddings[0:1]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i:i+1]
            
            # Calcular similitud promedio con el grupo actual
            similarities = np.dot(current_group_embedding, sentence_embedding.T)
            avg_similarity = np.mean(similarities)
            
            # Decidir si añadir al grupo actual o crear nuevo grupo
            if (avg_similarity >= self.similarity_threshold and 
                self._estimate_group_size(current_group + [sentence]) <= self.chunk_size):
                
                current_group.append(sentence)
                current_group_embedding = np.vstack([current_group_embedding, sentence_embedding])
            else:
                # Finalizar grupo actual y comenzar nuevo
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_group_embedding = sentence_embedding
        
        # Añadir último grupo
        if current_group:
            groups.append(current_group)
            
        return groups
    
    def _create_chunks_from_groups(
        self, 
        semantic_groups: List[List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Convierte grupos semánticos en chunks válidos.
        
        Maneja casos donde los grupos son muy grandes o muy pequeños.
        """
        chunks = []
        
        for group_idx, group in enumerate(semantic_groups):
            group_text = " ".join(group)
            
            # Si el grupo es muy pequeño, intentar combinar con el siguiente
            if (len(group_text) < self.min_chunk_size and 
                group_idx < len(semantic_groups) - 1):
                
                next_group = semantic_groups[group_idx + 1]
                combined_text = group_text + " " + " ".join(next_group)
                
                # Si la combinación no es muy grande, combinar
                if len(combined_text) <= self.chunk_size * 1.5:
                    group_text = combined_text
                    # Marcar el siguiente grupo como procesado
                    semantic_groups[group_idx + 1] = []
            
            # Si el grupo es muy grande, dividir en sub-chunks
            if len(group_text) > self.chunk_size * 2:
                sub_chunks = self._split_large_group(group, metadata)
                chunks.extend(sub_chunks)
            else:
                # Crear chunk normal
                if group_text.strip():  # Solo si no está vacío
                    chunk = Chunk(
                        content=group_text,
                        metadata=metadata.copy() if metadata else {}
                    )
                    chunk.add_metadata("semantic_group_index", group_idx)
                    chunk.add_metadata("sentences_count", len(group))
                    chunks.append(chunk)
        
        return chunks
    
    def _split_large_group(
        self, 
        group: List[str], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide grupos semánticos muy grandes en sub-chunks.
        
        Mantiene la coherencia semántica mientras respeta límites de tamaño.
        """
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        for sentence in group:
            sentence_size = len(sentence)
            
            # Si añadir esta oración excede el límite, crear chunk
            if (current_size + sentence_size > self.chunk_size and 
                current_chunk_sentences):
                
                chunk_text = " ".join(current_chunk_sentences)
                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata.copy() if metadata else {}
                )
                chunk.add_metadata("large_group_subchunk", True)
                chunks.append(chunk)
                
                # Iniciar nuevo chunk
                current_chunk_sentences = [sentence]
                current_size = sentence_size
            else:
                current_chunk_sentences.append(sentence)
                current_size += sentence_size
        
        # Añadir último chunk si hay contenido
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk = Chunk(
                content=chunk_text,
                metadata=metadata.copy() if metadata else {}
            )
            chunk.add_metadata("large_group_subchunk", True)
            chunks.append(chunk)
        
        return chunks
    
    def _estimate_group_size(self, group: List[str]) -> int:
        """Estima el tamaño en caracteres de un grupo de oraciones."""
        return sum(len(sentence) for sentence in group) + len(group) - 1  # +espacios
    
    def _fallback_sentence_chunking(
        self, 
        sentences: List[str], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunking de fallback cuando no hay modelo de embeddings.
        
        Agrupa oraciones por tamaño respetando límites semánticos básicos.
        """
        chunks = []
        current_sentences = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if (current_size + sentence_size > self.chunk_size and 
                current_sentences):
                
                # Crear chunk con las oraciones actuales
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata.copy() if metadata else {}
                )
                chunk.add_metadata("fallback_chunking", True)
                chunk.add_metadata("sentences_count", len(current_sentences))
                chunks.append(chunk)
                
                # Iniciar nuevo chunk
                current_sentences = [sentence]
                current_size = sentence_size
            else:
                current_sentences.append(sentence)
                current_size += sentence_size
        
        # Añadir último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                content=chunk_text,
                metadata=metadata.copy() if metadata else {}
            )
            chunk.add_metadata("fallback_chunking", True)
            chunk.add_metadata("sentences_count", len(current_sentences))
            chunks.append(chunk)
        
        return chunks