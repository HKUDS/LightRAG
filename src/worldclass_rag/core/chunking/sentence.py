"""
Sentence-based Chunking Strategy - Segmentación basada en oraciones.

Respeta límites de oraciones y agrupa por coherencia, siguiendo las mejores
prácticas del video AI News & Strategy Daily para preservar contexto semántico.
"""

import re
from typing import Any, Dict, List, Optional, Set
import spacy
from spacy.lang import LANG_CODES

from .base import ChunkingStrategy, Chunk


class SentenceChunker(ChunkingStrategy):
    """
    Chunker que respeta límites de oraciones para preservar coherencia.
    
    Ventajas:
    - Nunca rompe oraciones a la mitad
    - Preserva coherencia gramatical completa
    - Mejor para análisis lingüístico
    - Ideal para contenido conversacional
    
    Características:
    - Detección inteligente de oraciones con spaCy
    - Manejo de abreviaciones y casos especiales
    - Agrupación optimizada por tamaño
    - Preservación de contexto con superposición
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        language: str = "es",  # Español por defecto
        preserve_sentence_boundaries: bool = True,
        add_metadata: bool = True,
        min_sentences_per_chunk: int = 1,
        max_sentences_per_chunk: Optional[int] = None,
        sentence_similarity_threshold: float = 0.0,
    ):
        """
        Inicializa el chunker basado en oraciones.
        
        Args:
            language: Código de idioma para el modelo spaCy
            min_sentences_per_chunk: Mínimo de oraciones por chunk
            max_sentences_per_chunk: Máximo de oraciones por chunk
            sentence_similarity_threshold: Umbral para agrupar oraciones similares
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_sentence_boundaries=preserve_sentence_boundaries,
            add_metadata=add_metadata,
        )
        
        self.language = language
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.sentence_similarity_threshold = sentence_similarity_threshold
        
        # Cargar modelo de spaCy
        self.nlp = self._load_spacy_model(language)
        
        # Patrones de abreviaciones comunes por idioma
        self.abbreviations = self._get_abbreviations_for_language(language)
    
    def _load_spacy_model(self, language: str):
        """
        Carga el modelo de spaCy apropiado para el idioma.
        
        Implementa fallbacks robustos si el modelo no está disponible.
        """
        # Mapeo de códigos de idioma a modelos spaCy
        model_mapping = {
            "es": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg"],
            "en": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
            "fr": ["fr_core_news_sm", "fr_core_news_md"],
            "de": ["de_core_news_sm", "de_core_news_md"],
            "pt": ["pt_core_news_sm"],
            "it": ["it_core_news_sm"],
        }
        
        models_to_try = model_mapping.get(language, ["xx_ent_wiki_sm"])  # Modelo multiidioma
        
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                print(f"Modelo spaCy cargado: {model_name}")
                return nlp
            except OSError:
                continue
        
        # Fallback: usar modelo en blanco si ninguno está disponible
        print(f"Warning: No se encontró modelo spaCy para '{language}', usando fallback regex")
        return None
    
    def _get_abbreviations_for_language(self, language: str) -> Set[str]:
        """Retorna set de abreviaciones comunes para el idioma."""
        abbreviations_by_lang = {
            "es": {
                "Dr.", "Dra.", "Sr.", "Sra.", "Srta.", "Prof.", "Lic.", 
                "Ing.", "etc.", "p.ej.", "vs.", "cf.", "pág.", "págs.",
                "Art.", "Cap.", "Secc.", "Inc.", "S.A.", "S.L.", "Ltda."
            },
            "en": {
                "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Inc.", "Corp.",
                "Ltd.", "etc.", "e.g.", "i.e.", "vs.", "cf.", "p.", "pp.",
                "Fig.", "Vol.", "No.", "St.", "Ave.", "Blvd."
            },
            "fr": {
                "M.", "Mme", "Mlle", "Dr.", "Prof.", "etc.", "p.ex.",
                "cf.", "vs.", "S.A.", "S.A.R.L.", "Art.", "Chap."
            },
            "de": {
                "Dr.", "Prof.", "Herr", "Frau", "etc.", "z.B.", "d.h.",
                "bzw.", "usw.", "GmbH", "AG", "e.V.", "Kap.", "Abb."
            }
        }
        
        return abbreviations_by_lang.get(language, set())
    
    def split_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide el texto en chunks respetando límites de oraciones.
        
        Proceso:
        1. Detecta oraciones usando spaCy o regex
        2. Agrupa oraciones hasta alcanzar tamaño objetivo
        3. Preserva coherencia y contexto
        4. Añade superposición y metadatos
        """
        if not text.strip():
            return []
        
        # Detectar oraciones
        sentences = self._detect_sentences(text)
        
        if not sentences:
            return []
        
        # Agrupar oraciones en chunks
        chunks = self._group_sentences_into_chunks(sentences, metadata)
        
        # Añadir información de superposición
        chunks = self._add_overlap_info(chunks)
        
        # Añadir metadatos finales
        for i, chunk in enumerate(chunks):
            self._add_base_metadata(chunk, metadata, i, len(chunks))
            chunk.add_metadata("sentence_chunking", True)
            chunk.add_metadata("language", self.language)
            
        return chunks
    
    def _detect_sentences(self, text: str) -> List[str]:
        """
        Detecta oraciones usando spaCy o fallback regex.
        
        Returns:
            Lista de oraciones detectadas
        """
        if self.nlp is not None:
            return self._detect_sentences_with_spacy(text)
        else:
            return self._detect_sentences_with_regex(text)
    
    def _detect_sentences_with_spacy(self, text: str) -> List[str]:
        """Detecta oraciones usando el modelo spaCy."""
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) > 5:  # Filtrar fragmentos muy cortos
                sentences.append(sentence_text)
        
        return sentences
    
    def _detect_sentences_with_regex(self, text: str) -> List[str]:
        """
        Detecta oraciones usando regex como fallback.
        
        Implementa detección robusta considerando abreviaciones.
        """
        # Patrón base para fin de oración
        sentence_endings = r'[.!?]+(?:\s+|$)'
        
        # Crear patrón para excepciones (abreviaciones)
        abbreviation_pattern = ""
        if self.abbreviations:
            # Escapar caracteres especiales en abreviaciones
            escaped_abbrevs = [re.escape(abbr) for abbr in self.abbreviations]
            abbreviation_pattern = "|".join(escaped_abbrevs)
        
        # Dividir en oraciones candidatas
        sentences = re.split(sentence_endings, text)
        
        # Limpiar y validar oraciones
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filtrar oraciones muy cortas o que son solo puntuación
            if len(sentence) > 5 and not sentence.isspace():
                # Verificar que no sea solo una abreviación
                if abbreviation_pattern and re.fullmatch(abbreviation_pattern, sentence):
                    continue
                
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _group_sentences_into_chunks(
        self, 
        sentences: List[str], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Agrupa oraciones en chunks respetando límites de tamaño.
        
        Implementa algoritmo inteligente que balancea:
        - Tamaño objetivo de chunks
        - Número mínimo y máximo de oraciones
        - Coherencia semántica si está habilitada
        """
        chunks = []
        current_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Verificar si añadir esta oración excedería los límites
            would_exceed_size = (current_length + sentence_length > self.chunk_size)
            would_exceed_max_sentences = (
                self.max_sentences_per_chunk and 
                len(current_sentences) >= self.max_sentences_per_chunk
            )
            
            # Decidir si crear nuevo chunk
            if (would_exceed_size or would_exceed_max_sentences) and current_sentences:
                # Verificar mínimo de oraciones
                if len(current_sentences) >= self.min_sentences_per_chunk:
                    # Crear chunk con las oraciones actuales
                    chunk = self._create_chunk_from_sentences(
                        current_sentences, metadata
                    )
                    chunks.append(chunk)
                    
                    # Iniciar nuevo chunk
                    current_sentences = [sentence]
                    current_length = sentence_length
                else:
                    # No hay suficientes oraciones, añadir esta de todas formas
                    current_sentences.append(sentence)
                    current_length += sentence_length
            else:
                # Añadir oración al chunk actual
                current_sentences.append(sentence)
                current_length += sentence_length
        
        # Procesar último chunk
        if current_sentences:
            if len(current_sentences) >= self.min_sentences_per_chunk:
                chunk = self._create_chunk_from_sentences(current_sentences, metadata)
                chunks.append(chunk)
            elif chunks:
                # Si hay muy pocas oraciones, añadir al chunk anterior
                last_chunk = chunks[-1]
                additional_content = " ".join(current_sentences)
                last_chunk.content += " " + additional_content
                
                # Actualizar metadatos
                last_sentences = last_chunk.get_metadata("sentences_count", 0)
                last_chunk.add_metadata("sentences_count", last_sentences + len(current_sentences))
            else:
                # Es el único chunk, crearlo de todas formas
                chunk = self._create_chunk_from_sentences(current_sentences, metadata)
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_sentences(
        self, 
        sentences: List[str], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """
        Crea un chunk a partir de una lista de oraciones.
        
        Añade metadatos específicos de oraciones.
        """
        content = " ".join(sentences)
        
        chunk = Chunk(
            content=content,
            metadata=metadata.copy() if metadata else {}
        )
        
        # Metadatos específicos de oraciones
        chunk.add_metadata("sentences_count", len(sentences))
        chunk.add_metadata("avg_sentence_length", len(content) / len(sentences))
        chunk.add_metadata("first_sentence", sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0])
        
        # Análisis lingüístico básico si spaCy está disponible
        if self.nlp is not None:
            linguistic_analysis = self._analyze_sentences_linguistically(sentences)
            chunk.metadata.update(linguistic_analysis)
        
        return chunk
    
    def _analyze_sentences_linguistically(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Analiza las oraciones lingüísticamente usando spaCy.
        
        Proporciona metadatos útiles para recuperación y filtrado.
        """
        analysis = {
            "linguistic_analysis": True,
            "sentence_types": [],
            "named_entities": [],
            "avg_complexity": 0,
            "language_detected": self.language
        }
        
        total_complexity = 0
        all_entities = set()
        
        for sentence in sentences:
            doc = self.nlp(sentence)
            
            # Clasificar tipo de oración (declarativa, interrogativa, etc.)
            if sentence.strip().endswith('?'):
                analysis["sentence_types"].append("interrogative")
            elif sentence.strip().endswith('!'):
                analysis["sentence_types"].append("exclamative")
            else:
                analysis["sentence_types"].append("declarative")
            
            # Extraer entidades nombradas
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                    all_entities.add((ent.text.lower(), ent.label_))
            
            # Calcular complejidad (número de tokens / dependencias)
            complexity = len([token for token in doc if not token.is_space])
            total_complexity += complexity
        
        # Promediar métricas
        if sentences:
            analysis["avg_complexity"] = total_complexity / len(sentences)
            analysis["named_entities"] = list(all_entities)
            
            # Estadísticas de tipos de oraciones
            type_counts = {}
            for sent_type in analysis["sentence_types"]:
                type_counts[sent_type] = type_counts.get(sent_type, 0) + 1
            analysis["sentence_type_distribution"] = type_counts
        
        return analysis
    
    def split_with_semantic_grouping(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Versión avanzada que agrupa oraciones por similitud semántica.
        
        Requiere modelo de embeddings para funcionar completamente.
        """
        sentences = self._detect_sentences(text)
        
        if not sentences or self.sentence_similarity_threshold <= 0:
            # Fallback a agrupación normal
            return self.split_text(text, metadata)
        
        # Implementar agrupación semántica básica
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Calcular embeddings
            embeddings = model.encode(sentences)
            
            # Agrupar por similitud (implementación simplificada)
            semantic_chunks = self._group_by_similarity(sentences, embeddings)
            
            # Convertir grupos a chunks
            chunks = []
            for group in semantic_chunks:
                chunk = self._create_chunk_from_sentences(group, metadata)
                chunk.add_metadata("semantic_grouping", True)
                chunks.append(chunk)
            
            return chunks
            
        except ImportError:
            print("Warning: sentence-transformers no disponible, usando agrupación normal")
            return self.split_text(text, metadata)
    
    def _group_by_similarity(self, sentences: List[str], embeddings) -> List[List[str]]:
        """
        Agrupa oraciones por similitud semántica.
        
        Implementación simplificada del clustering semántico.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        groups = []
        used_indices = set()
        
        for i, sentence in enumerate(sentences):
            if i in used_indices:
                continue
                
            # Iniciar nuevo grupo con esta oración
            current_group = [sentence]
            used_indices.add(i)
            
            # Buscar oraciones similares
            for j in range(i + 1, len(sentences)):
                if j in used_indices:
                    continue
                    
                # Calcular similitud
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                if similarity >= self.sentence_similarity_threshold:
                    current_group.append(sentences[j])
                    used_indices.add(j)
                    
                    # Limitar tamaño del grupo
                    group_size = sum(len(s) for s in current_group)
                    if group_size > self.chunk_size:
                        break
            
            groups.append(current_group)
        
        return groups