"""
Recursive Chunking Strategy - Segmentación jerárquica recursiva.

Implementa chunking recursivo que respeta la estructura jerárquica del documento,
siguiendo las mejores prácticas del video AI News & Strategy Daily.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import ChunkingStrategy, Chunk


class RecursiveChunker(ChunkingStrategy):
    """
    Chunker que divide texto de forma jerárquica recursiva.
    
    Divide por delimitadores en orden de prioridad:
    1. Párrafos dobles (\n\n)
    2. Párrafos simples (\n)
    3. Oraciones (. ! ?)
    4. Frases (, ; :)
    5. Palabras (espacios)
    
    Ventajas:
    - Preserva estructura natural del documento
    - Respeta jerarquías semánticas
    - Flexible para diferentes tipos de contenido
    
    Ideal para:
    - Documentos bien estructurados
    - Contenido con jerarquías claras
    - Textos largos con múltiples secciones
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        preserve_sentence_boundaries: bool = True,
        add_metadata: bool = True,
        min_chunk_size: int = 50,
        max_chunk_size: Optional[int] = None,
    ):
        """
        Inicializa el chunker recursivo.
        
        Args:
            separators: Lista de separadores en orden de prioridad
            min_chunk_size: Tamaño mínimo de chunk
            max_chunk_size: Tamaño máximo antes de división forzada
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_sentence_boundaries=preserve_sentence_boundaries,
            add_metadata=add_metadata,
        )
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or (chunk_size * 2)
        
        # Separadores jerárquicos por defecto
        if separators is None:
            self.separators = [
                "\n\n",        # Párrafos dobles (máxima prioridad)
                "\n",          # Párrafos simples
                ". ",          # Fin de oración con espacio
                "! ",          # Exclamación con espacio
                "? ",          # Pregunta con espacio
                "; ",          # Punto y coma
                ", ",          # Coma
                " ",           # Espacios (última opción)
            ]
        else:
            self.separators = separators
            
        # Validar configuración
        if self.min_chunk_size >= self.chunk_size:
            raise ValueError("min_chunk_size debe ser menor que chunk_size")
    
    def split_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide el texto usando chunking recursivo jerárquico.
        
        Proceso:
        1. Intenta dividir por el primer separador
        2. Si las partes son muy grandes, recurse con el siguiente separador
        3. Combina partes pequeñas respetando límites
        4. Añade superposición y metadatos
        """
        if not text.strip():
            return []
        
        # Dividir recursivamente 
        text_chunks = self._split_text_recursive(text, self.separators)
        
        # Convertir a objetos Chunk
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip() and len(chunk_text.strip()) >= self.min_chunk_size:
                chunk = Chunk(
                    content=chunk_text.strip(),
                    metadata=metadata.copy() if metadata else {}
                )
                chunk.add_metadata("recursive_chunking", True)
                chunk.add_metadata("chunk_method", "recursive_hierarchical")
                chunks.append(chunk)
        
        # Añadir información de superposición
        chunks = self._add_overlap_info(chunks)
        
        # Añadir metadatos finales
        for i, chunk in enumerate(chunks):
            self._add_base_metadata(chunk, metadata, i, len(chunks))
            
        return chunks
    
    def _split_text_recursive(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """
        Función recursiva principal para dividir texto.
        
        Args:
            text: Texto a dividir
            separators: Lista de separadores a intentar
            
        Returns:
            Lista de fragmentos de texto
        """
        # Caso base: si el texto es suficientemente pequeño, retornarlo
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Caso base: si no hay más separadores, dividir por fuerza
        if not separators:
            return self._force_split(text)
        
        # Intentar dividir con el primer separador
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator not in text:
            # Si el separador no existe, probar el siguiente
            return self._split_text_recursive(text, remaining_separators)
        
        # Dividir por el separador actual
        splits = text.split(separator)
        
        # Reconstruir fragmentos preservando el separador
        fragments = []
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                # Añadir separador excepto al último fragmento
                fragments.append(split + separator)
            else:
                fragments.append(split)
        
        # Procesar cada fragmento
        final_chunks = []
        current_chunk = ""
        
        for fragment in fragments:
            # Si el fragmento actual + el nuevo exceden el límite
            if len(current_chunk + fragment) > self.chunk_size and current_chunk:
                # Procesar el chunk actual si es necesario
                if len(current_chunk) > self.max_chunk_size:
                    # El chunk actual es muy grande, dividir recursivamente
                    sub_chunks = self._split_text_recursive(current_chunk, remaining_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(current_chunk)
                
                # Iniciar nuevo chunk con el fragmento actual
                current_chunk = fragment
            else:
                # Añadir fragmento al chunk actual
                current_chunk += fragment
        
        # Procesar el último chunk
        if current_chunk:
            if len(current_chunk) > self.max_chunk_size:
                sub_chunks = self._split_text_recursive(current_chunk, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(current_chunk)
        
        return final_chunks
    
    def _force_split(self, text: str) -> List[str]:
        """
        División forzada cuando no hay separadores apropiados.
        
        Divide por tamaño respetando límites de palabras cuando sea posible.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Último fragmento
                chunks.append(text[start:])
                break
            
            # Intentar cortar en límite de palabra si preserve_sentence_boundaries
            if self.preserve_sentence_boundaries:
                # Buscar el último espacio dentro del límite
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Calcular inicio para el siguiente chunk con superposición
            next_start = end - self.chunk_overlap
            start = max(start + 1, next_start)  # Evitar loops infinitos
        
        return chunks
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analiza la estructura del texto para optimizar la estrategia.
        
        Returns:
            Diccionario con estadísticas estructurales
        """
        analysis = {
            "total_length": len(text),
            "separator_counts": {},
            "estimated_complexity": "low",
            "recommended_strategy": "recursive"
        }
        
        # Contar occurrencias de cada separador
        for separator in self.separators:
            count = text.count(separator)
            analysis["separator_counts"][repr(separator)] = count
        
        # Determinar complejidad estructural
        paragraph_breaks = text.count('\n\n')
        line_breaks = text.count('\n')
        sentences = len(re.findall(r'[.!?]+', text))
        
        if paragraph_breaks > 10:
            analysis["estimated_complexity"] = "high"
        elif line_breaks > 20 or sentences > 50:
            analysis["estimated_complexity"] = "medium"
        
        # Recomendar ajustes basados en estructura
        if paragraph_breaks < 2 and line_breaks < 5:
            analysis["recommended_strategy"] = "sentence_based"
        elif paragraph_breaks > len(text) / 500:  # Muchos párrafos cortos
            analysis["recommended_strategy"] = "paragraph_focused"
        
        return analysis
    
    def optimize_for_text(self, text: str) -> 'RecursiveChunker':
        """
        Retorna una versión optimizada del chunker para el texto dado.
        
        Analiza la estructura y ajusta parámetros automáticamente.
        """
        analysis = self._analyze_text_structure(text)
        
        # Crear copia optimizada
        optimized_separators = self.separators.copy()
        optimized_chunk_size = self.chunk_size
        optimized_overlap = self.chunk_overlap
        
        # Ajustar según complejidad
        if analysis["estimated_complexity"] == "high":
            # Priorizar párrafos dobles para textos complejos
            if "\n\n" in optimized_separators:
                optimized_separators.remove("\n\n")
                optimized_separators.insert(0, "\n\n")
            optimized_chunk_size = int(self.chunk_size * 1.2)
            
        elif analysis["estimated_complexity"] == "low":
            # Para textos simples, usar chunks más pequeños
            optimized_chunk_size = int(self.chunk_size * 0.8)
            optimized_overlap = int(self.chunk_overlap * 0.7)
        
        # Crear instancia optimizada
        optimized_chunker = RecursiveChunker(
            chunk_size=optimized_chunk_size,
            chunk_overlap=optimized_overlap,
            separators=optimized_separators,
            preserve_sentence_boundaries=self.preserve_sentence_boundaries,
            add_metadata=self.add_metadata,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        )
        
        # Añadir metadatos de optimización
        optimized_chunker._optimization_analysis = analysis
        
        return optimized_chunker
    
    def split_text_with_structure_preservation(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        preserve_headers: bool = True,
        preserve_lists: bool = True,
    ) -> List[Chunk]:
        """
        Versión avanzada que preserva elementos estructurales específicos.
        
        Args:
            preserve_headers: Si preservar títulos y encabezados
            preserve_lists: Si preservar listas y enumeraciones
        """
        if not text.strip():
            return []
        
        # Pre-procesar para identificar elementos estructurales
        text_with_markers = self._add_structure_markers(
            text, preserve_headers, preserve_lists
        )
        
        # Chunking normal con marcadores
        chunks = self.split_text(text_with_markers, metadata)
        
        # Post-procesar para limpiar marcadores y añadir metadatos estructurales
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._process_structured_chunk(chunk)
            if processed_chunk:
                processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _add_structure_markers(
        self, 
        text: str, 
        preserve_headers: bool, 
        preserve_lists: bool
    ) -> str:
        """Añade marcadores temporales para preservar estructura."""
        marked_text = text
        
        if preserve_headers:
            # Marcar encabezados (líneas seguidas de línea vacía o que empiezan con #)
            header_pattern = r'^(#{1,6}\s+.+|.+\n={3,}|.+\n-{3,})$'
            marked_text = re.sub(
                header_pattern, 
                r'[HEADER_START]\1[HEADER_END]', 
                marked_text, 
                flags=re.MULTILINE
            )
        
        if preserve_lists:
            # Marcar elementos de lista
            list_patterns = [
                r'^(\s*[-*+]\s+.+)$',        # Listas con guiones/asteriscos
                r'^(\s*\d+\.\s+.+)$',        # Listas numeradas
                r'^(\s*[a-zA-Z]\.\s+.+)$',   # Listas con letras
            ]
            
            for pattern in list_patterns:
                marked_text = re.sub(
                    pattern,
                    r'[LIST_ITEM_START]\1[LIST_ITEM_END]',
                    marked_text,
                    flags=re.MULTILINE
                )
        
        return marked_text
    
    def _process_structured_chunk(self, chunk: Chunk) -> Optional[Chunk]:
        """Procesa chunk con marcadores estructurales."""
        content = chunk.content
        
        # Extraer información estructural
        has_headers = '[HEADER_START]' in content
        has_lists = '[LIST_ITEM_START]' in content
        
        # Limpiar marcadores
        content = re.sub(r'\[HEADER_START\](.+?)\[HEADER_END\]', r'\1', content)
        content = re.sub(r'\[LIST_ITEM_START\](.+?)\[LIST_ITEM_END\]', r'\1', content)
        
        # Actualizar chunk
        chunk.content = content
        
        # Añadir metadatos estructurales
        if has_headers:
            chunk.add_metadata("contains_headers", True)
            headers = re.findall(r'\[HEADER_START\](.+?)\[HEADER_END\]', chunk.content)
            chunk.add_metadata("header_count", len(headers))
        
        if has_lists:
            chunk.add_metadata("contains_lists", True)
            list_items = re.findall(r'\[LIST_ITEM_START\](.+?)\[LIST_ITEM_END\]', chunk.content)
            chunk.add_metadata("list_item_count", len(list_items))
        
        return chunk if content.strip() else None