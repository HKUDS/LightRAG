"""
Fixed Size Chunking Strategy - Segmentación por tamaño fijo.

Implementa chunking por tamaño fijo con mejoras para evitar los problemas
identificados en el video AI News & Strategy Daily.
"""

import re
from typing import Any, Dict, List, Optional

from .base import ChunkingStrategy, Chunk


class FixedSizeChunker(ChunkingStrategy):
    """
    Chunker que divide texto en fragmentos de tamaño fijo con mejoras.
    
    ADVERTENCIA (del video AI News & Strategy Daily):
    - El chunking de tamaño fijo es peligroso si no se implementa bien
    - Puede cortar a mitad de oración, destruyendo el contexto
    - Esta implementación incluye mejoras para mitigar estos problemas
    
    Mejoras implementadas:
    - Respeta límites de palabras/oraciones cuando sea posible
    - Superposición inteligente para preservar contexto
    - Detección de puntos de corte seguros
    - Metadatos de calidad para identificar cortes problemáticos
    
    Casos de uso apropiados:
    - Procesamiento de texto muy largo donde la velocidad es crítica
    - Textos con estructura muy uniforme
    - Como fallback cuando otras estrategias fallan
    - Integración con sistemas legacy que requieren tamaños fijos
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        prefer_sentence_boundaries: bool = True,
        prefer_word_boundaries: bool = True,
        add_metadata: bool = True,
        max_boundary_search: int = 100,
        quality_threshold: float = 0.7,
    ):
        """
        Inicializa el chunker de tamaño fijo mejorado.
        
        Args:
            prefer_sentence_boundaries: Intentar cortar en fin de oración
            prefer_word_boundaries: Intentar cortar en límites de palabras
            max_boundary_search: Máximo de caracteres a buscar para punto de corte seguro
            quality_threshold: Umbral de calidad para advertencias
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_sentence_boundaries=prefer_sentence_boundaries,
            add_metadata=add_metadata,
        )
        
        self.prefer_sentence_boundaries = prefer_sentence_boundaries
        self.prefer_word_boundaries = prefer_word_boundaries
        self.max_boundary_search = max_boundary_search
        self.quality_threshold = quality_threshold
        
        # Patrones para detectar puntos de corte seguros
        self.sentence_end_pattern = re.compile(r'[.!?]+\s+')
        self.word_boundary_pattern = re.compile(r'\s+')
        
        # Patrones problemáticos a evitar
        self.avoid_patterns = [
            re.compile(r'\w+-$'),      # Palabras con guión al final
            re.compile(r'^\s*[a-z]'),  # Inicio con minúscula (medio de oración)
            re.compile(r'\d+\.$'),     # Números con punto (posible lista)
            re.compile(r'["\'\(]$'),   # Inicio de cita o paréntesis
        ]
    
    def split_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide el texto en chunks de tamaño fijo con mejoras de calidad.
        
        Proceso:
        1. Divide en segmentos de tamaño objetivo
        2. Busca puntos de corte seguros cerca de los límites
        3. Aplica superposición inteligente
        4. Evalúa calidad de cada corte
        5. Añade metadatos de calidad y advertencias
        """
        if not text.strip():
            return []
        
        chunks = []
        position = 0
        chunk_index = 0
        total_text_length = len(text)
        
        while position < total_text_length:
            # Calcular límites del chunk
            end_position = min(position + self.chunk_size, total_text_length)
            
            # Buscar punto de corte seguro
            safe_end_position, cut_quality = self._find_safe_cut_point(
                text, position, end_position
            )
            
            # Extraer contenido del chunk
            chunk_content = text[position:safe_end_position].strip()
            
            if chunk_content:
                # Crear chunk
                chunk = Chunk(
                    content=chunk_content,
                    metadata=metadata.copy() if metadata else {},
                    start_index=position,
                    end_index=safe_end_position,
                )
                
                # Añadir metadatos de calidad
                self._add_quality_metadata(chunk, cut_quality, chunk_index)
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Calcular siguiente posición con superposición
            next_position = safe_end_position - self.chunk_overlap
            position = max(position + 1, next_position)  # Evitar loops infinitos
        
        # Añadir información de superposición
        chunks = self._add_overlap_info(chunks)
        
        # Añadir metadatos finales
        for i, chunk in enumerate(chunks):
            self._add_base_metadata(chunk, metadata, i, len(chunks))
            chunk.add_metadata("fixed_size_chunking", True)
            
        return chunks
    
    def _find_safe_cut_point(
        self, 
        text: str, 
        start: int, 
        target_end: int
    ) -> tuple[int, Dict[str, Any]]:
        """
        Busca el mejor punto de corte cerca de la posición objetivo.
        
        Returns:
            Tupla de (posición_de_corte, información_de_calidad)
        """
        quality_info = {
            "cut_type": "forced",
            "quality_score": 0.0,
            "boundary_respected": False,
            "context_preserved": False,
            "warnings": []
        }
        
        # Si estamos al final del texto, no hay problema
        if target_end >= len(text):
            quality_info.update({
                "cut_type": "end_of_text",
                "quality_score": 1.0,
                "boundary_respected": True,
                "context_preserved": True,
            })
            return target_end, quality_info
        
        # Buscar punto de corte por prioridad
        best_cut_point = target_end
        best_score = 0.0
        
        # 1. Buscar fin de oración (máxima prioridad)
        if self.prefer_sentence_boundaries:
            sentence_cut, sentence_score = self._find_sentence_boundary(
                text, start, target_end
            )
            if sentence_score > best_score:
                best_cut_point = sentence_cut
                best_score = sentence_score
                quality_info["cut_type"] = "sentence_boundary"
        
        # 2. Buscar límite de palabra (prioridad media)
        if self.prefer_word_boundaries and best_score < 0.8:
            word_cut, word_score = self._find_word_boundary(
                text, start, target_end
            )
            if word_score > best_score:
                best_cut_point = word_cut
                best_score = word_score
                quality_info["cut_type"] = "word_boundary"
        
        # 3. Evaluar calidad del corte elegido
        quality_info["quality_score"] = best_score
        quality_info["boundary_respected"] = best_score > 0.5
        quality_info["context_preserved"] = best_score > 0.7
        
        # Añadir advertencias si la calidad es baja
        if best_score < self.quality_threshold:
            quality_info["warnings"].append(
                f"Corte de baja calidad (score: {best_score:.2f})"
            )
            
        # Verificar patrones problemáticos
        self._check_problematic_patterns(
            text, start, best_cut_point, quality_info
        )
        
        return best_cut_point, quality_info
    
    def _find_sentence_boundary(
        self, 
        text: str, 
        start: int, 
        target_end: int
    ) -> tuple[int, float]:
        """
        Busca el mejor límite de oración cerca del punto objetivo.
        
        Returns:
            Tupla de (posición, puntuación_de_calidad)
        """
        search_start = max(start, target_end - self.max_boundary_search)
        search_end = min(len(text), target_end + self.max_boundary_search)
        search_text = text[search_start:search_end]
        
        # Buscar todos los finales de oración en el rango de búsqueda
        sentence_ends = []
        for match in self.sentence_end_pattern.finditer(search_text):
            abs_position = search_start + match.end()
            distance_from_target = abs(abs_position - target_end)
            sentence_ends.append((abs_position, distance_from_target))
        
        if not sentence_ends:
            return target_end, 0.0
        
        # Elegir el más cercano al objetivo
        best_position, distance = min(sentence_ends, key=lambda x: x[1])
        
        # Calcular puntuación basada en distancia
        max_distance = self.max_boundary_search
        score = max(0.0, 1.0 - (distance / max_distance)) * 0.9  # Max 0.9 para sentence
        
        return best_position, score
    
    def _find_word_boundary(
        self, 
        text: str, 
        start: int, 
        target_end: int
    ) -> tuple[int, float]:
        """
        Busca el mejor límite de palabra cerca del punto objetivo.
        
        Returns:
            Tupla de (posición, puntuación_de_calidad)
        """
        search_start = max(start, target_end - self.max_boundary_search)
        search_end = min(len(text), target_end + self.max_boundary_search)
        
        # Buscar espacios (límites de palabra) cerca del objetivo
        word_boundaries = []
        
        # Buscar hacia atrás desde el objetivo
        for i in range(target_end, search_start - 1, -1):
            if i < len(text) and text[i].isspace():
                distance = target_end - i
                word_boundaries.append((i, distance))
        
        # Buscar hacia adelante desde el objetivo
        for i in range(target_end, search_end):
            if i < len(text) and text[i].isspace():
                distance = i - target_end
                word_boundaries.append((i, distance))
        
        if not word_boundaries:
            return target_end, 0.0
        
        # Elegir el más cercano
        best_position, distance = min(word_boundaries, key=lambda x: x[1])
        
        # Calcular puntuación
        max_distance = self.max_boundary_search
        score = max(0.0, 1.0 - (distance / max_distance)) * 0.6  # Max 0.6 for word
        
        return best_position, score
    
    def _check_problematic_patterns(
        self, 
        text: str, 
        start: int, 
        cut_point: int, 
        quality_info: Dict[str, Any]
    ) -> None:
        """
        Verifica patrones problemáticos en el punto de corte.
        
        Añade advertencias específicas al quality_info.
        """
        # Examinar texto alrededor del corte
        context_before = text[max(0, cut_point - 20):cut_point]
        context_after = text[cut_point:min(len(text), cut_point + 20)]
        
        # Verificar patrones problemáticos
        for pattern in self.avoid_patterns:
            if pattern.search(context_before):
                quality_info["warnings"].append(
                    f"Patrón problemático antes del corte: {pattern.pattern}"
                )
                quality_info["quality_score"] *= 0.8
                
            if pattern.search(context_after):
                quality_info["warnings"].append(
                    f"Patrón problemático después del corte: {pattern.pattern}"
                )
                quality_info["quality_score"] *= 0.8
        
        # Verificar si el corte está en medio de una palabra
        if (cut_point > 0 and cut_point < len(text) and 
            not text[cut_point - 1].isspace() and 
            not text[cut_point].isspace()):
            quality_info["warnings"].append("Corte en medio de palabra")
            quality_info["quality_score"] *= 0.5
        
        # Verificar si comienza con minúscula (posible medio de oración)
        if context_after and context_after[0].islower():
            quality_info["warnings"].append("Chunk comienza con minúscula")
            quality_info["quality_score"] *= 0.7
    
    def _add_quality_metadata(
        self, 
        chunk: Chunk, 
        quality_info: Dict[str, Any], 
        chunk_index: int
    ) -> None:
        """
        Añade metadatos de calidad específicos del chunking fijo.
        """
        # Metadatos de calidad del corte
        chunk.add_metadata("cut_quality", quality_info)
        chunk.add_metadata("cut_type", quality_info["cut_type"])
        chunk.add_metadata("quality_score", quality_info["quality_score"])
        
        # Advertencias si existen
        if quality_info["warnings"]:
            chunk.add_metadata("quality_warnings", quality_info["warnings"])
            chunk.add_metadata("needs_review", True)
        
        # Clasificación de calidad
        score = quality_info["quality_score"]
        if score >= 0.8:
            quality_level = "high"
        elif score >= 0.6:
            quality_level = "medium"
        elif score >= 0.4:
            quality_level = "low"
        else:
            quality_level = "very_low"
            
        chunk.add_metadata("quality_level", quality_level)
        
        # Información específica de chunking fijo
        chunk.add_metadata("target_size", self.chunk_size)
        chunk.add_metadata("actual_size", len(chunk.content))
        chunk.add_metadata("size_variance", 
                          abs(len(chunk.content) - self.chunk_size) / self.chunk_size)
    
    def analyze_chunking_quality(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Analiza la calidad general del chunking realizado.
        
        Proporciona estadísticas útiles para evaluar si el chunking fijo
        es apropiado para el texto dado.
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Recopilar métricas de calidad
        quality_scores = [c.get_metadata("quality_score", 0.0) for c in chunks]
        quality_levels = [c.get_metadata("quality_level", "unknown") for c in chunks]
        warnings_count = sum(1 for c in chunks if c.get_metadata("quality_warnings"))
        
        # Estadísticas de tamaño
        sizes = [len(c.content) for c in chunks]
        size_variance = sum((s - self.chunk_size) ** 2 for s in sizes) / len(sizes)
        
        analysis = {
            "total_chunks": len(chunks),
            "quality_metrics": {
                "average_quality_score": sum(quality_scores) / len(quality_scores),
                "min_quality_score": min(quality_scores),
                "max_quality_score": max(quality_scores),
                "chunks_with_warnings": warnings_count,
                "warning_rate": warnings_count / len(chunks),
            },
            "size_metrics": {
                "target_size": self.chunk_size,
                "average_size": sum(sizes) / len(sizes),
                "size_variance": size_variance,
                "size_consistency": 1.0 - (size_variance / (self.chunk_size ** 2)),
            },
            "quality_distribution": {
                level: quality_levels.count(level) for level in set(quality_levels)
            },
            "recommendations": []
        }
        
        # Generar recomendaciones
        avg_quality = analysis["quality_metrics"]["average_quality_score"]
        warning_rate = analysis["quality_metrics"]["warning_rate"]
        
        if avg_quality < 0.5:
            analysis["recommendations"].append(
                "Calidad promedio baja. Considerar usar RecursiveChunker o SemanticChunker."
            )
            
        if warning_rate > 0.3:
            analysis["recommendations"].append(
                "Alto número de advertencias. El texto puede no ser apropiado para chunking fijo."
            )
            
        if size_variance > (self.chunk_size * 0.2) ** 2:
            analysis["recommendations"].append(
                "Alta varianza en tamaños. Ajustar parámetros de búsqueda de límites."
            )
        
        return analysis
    
    def optimize_for_text_sample(self, text_sample: str) -> 'FixedSizeChunker':
        """
        Analiza una muestra de texto y retorna un chunker optimizado.
        
        Ajusta parámetros basándose en las características del texto.
        """
        # Analizar muestra
        sample_chunks = self.split_text(text_sample)
        quality_analysis = self.analyze_chunking_quality(sample_chunks)
        
        # Calcular ajustes
        avg_quality = quality_analysis["quality_metrics"]["average_quality_score"]
        warning_rate = quality_analysis["quality_metrics"]["warning_rate"]
        
        # Ajustar parámetros
        new_max_boundary_search = self.max_boundary_search
        new_chunk_overlap = self.chunk_overlap
        new_prefer_sentences = self.prefer_sentence_boundaries
        
        if avg_quality < 0.6:
            # Buscar más lejos por límites seguros
            new_max_boundary_search = min(200, self.max_boundary_search * 1.5)
            new_chunk_overlap = min(self.chunk_size // 3, self.chunk_overlap * 1.2)
            
        if warning_rate > 0.4:
            # Ser más estricto con límites de oraciones
            new_prefer_sentences = True
            
        # Crear chunker optimizado
        optimized_chunker = FixedSizeChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=int(new_chunk_overlap),
            prefer_sentence_boundaries=new_prefer_sentences,
            prefer_word_boundaries=self.prefer_word_boundaries,
            add_metadata=self.add_metadata,
            max_boundary_search=int(new_max_boundary_search),
            quality_threshold=self.quality_threshold,
        )
        
        # Añadir información de optimización
        optimized_chunker._optimization_applied = True
        optimized_chunker._original_quality = avg_quality
        optimized_chunker._sample_analysis = quality_analysis
        
        return optimized_chunker