"""
Relevance Evaluator - "Are the right chunks retrieved?"

Implements relevance evaluation as the first key metric from AI News & Strategy Daily.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import numpy as np

from .base import RAGEvaluator, EvaluationResult, EvaluationMetrics
from ..retrieval.base import RetrievalResult


class RelevanceEvaluator(RAGEvaluator):
    """
    Evaluates retrieval relevance - the first key RAG metric.
    
    Methods for evaluating relevance:
    1. Semantic similarity between query and retrieved chunks
    2. Keyword overlap analysis
    3. Ground truth comparison (if available)
    4. Human-like relevance scoring
    """
    
    def __init__(
        self,
        similarity_model: Optional[str] = "all-MiniLM-L6-v2",
        use_semantic_similarity: bool = True,
        use_keyword_overlap: bool = True,
        relevance_threshold: float = 0.3,
    ):
        """
        Initialize relevance evaluator.
        
        Args:
            similarity_model: Sentence transformer model for semantic similarity
            use_semantic_similarity: Whether to use semantic similarity scoring
            use_keyword_overlap: Whether to use keyword overlap scoring
            relevance_threshold: Minimum similarity score to consider relevant
        """
        super().__init__(name="RelevanceEvaluator")
        
        self.use_semantic_similarity = use_semantic_similarity
        self.use_keyword_overlap = use_keyword_overlap
        self.relevance_threshold = relevance_threshold
        
        # Load similarity model if needed
        self.similarity_model = None
        if self.use_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer(similarity_model)
                print(f"Loaded similarity model: {similarity_model}")
            except Exception as e:
                print(f"Warning: Could not load similarity model: {e}")
                self.use_semantic_similarity = False
        elif self.use_semantic_similarity:
            print("Warning: sentence-transformers not available, disabling semantic similarity")
            self.use_semantic_similarity = False
    
    def evaluate(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[RetrievalResult],
        ground_truth: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate relevance of retrieved chunks.
        
        Combines multiple relevance signals to compute final score.
        """
        start_time = datetime.now()
        
        if not retrieved_chunks:
            # No chunks retrieved - zero relevance
            return EvaluationResult(
                query=query,
                response=response,
                retrieved_chunks=retrieved_chunks,
                metrics=EvaluationMetrics(relevance=0.0),
                evaluation_method="relevance_no_chunks",
                relevance_details={"error": "No chunks retrieved"}
            )
        
        relevance_scores = []
        chunk_details = []
        
        # Evaluate each chunk
        for i, chunk in enumerate(retrieved_chunks):
            chunk_score = self._evaluate_chunk_relevance(query, chunk, ground_truth)
            relevance_scores.append(chunk_score["score"])
            chunk_details.append({
                "chunk_index": i,
                "chunk_id": chunk.chunk_id,
                **chunk_score
            })
        
        # Calculate overall relevance
        overall_relevance = self._calculate_overall_relevance(relevance_scores)
        
        # Additional analysis
        analysis = self._analyze_retrieval_quality(query, retrieved_chunks, relevance_scores)
        
        # Evaluation time
        eval_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        metrics = EvaluationMetrics(relevance=overall_relevance)
        
        result = EvaluationResult(
            query=query,
            response=response,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
            evaluation_method="relevance_multi_signal",
            relevance_details={
                "individual_scores": relevance_scores,
                "chunk_details": chunk_details,
                "overall_score": overall_relevance,
                "analysis": analysis,
                "evaluation_time_s": eval_time,
                "methods_used": {
                    "semantic_similarity": self.use_semantic_similarity,
                    "keyword_overlap": self.use_keyword_overlap,
                    "ground_truth": ground_truth is not None,
                }
            }
        )
        
        return result
    
    def _evaluate_chunk_relevance(
        self, 
        query: str, 
        chunk: RetrievalResult, 
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate relevance of a single chunk to the query.
        
        Returns detailed scoring breakdown.
        """
        scores = {}
        
        # 1. Semantic similarity (if available)
        if self.use_semantic_similarity and self.similarity_model:
            semantic_score = self._compute_semantic_similarity(query, chunk.content)
            scores["semantic_similarity"] = semantic_score
        
        # 2. Keyword overlap
        if self.use_keyword_overlap:
            keyword_score = self._compute_keyword_overlap(query, chunk.content)
            scores["keyword_overlap"] = keyword_score
        
        # 3. Ground truth comparison (if available)
        if ground_truth:
            gt_score = self._compute_ground_truth_relevance(chunk, ground_truth)
            scores["ground_truth"] = gt_score
        
        # 4. Content quality indicators
        quality_score = self._compute_content_quality(chunk.content, query)
        scores["content_quality"] = quality_score
        
        # Combine scores
        final_score = self._combine_relevance_scores(scores)
        
        return {
            "score": final_score,
            "component_scores": scores,
            "is_relevant": final_score >= self.relevance_threshold,
        }
    
    def _compute_semantic_similarity(self, query: str, content: str) -> float:
        """Compute semantic similarity using sentence transformers."""
        try:
            # Generate embeddings
            embeddings = self.similarity_model.encode([query, content])
            
            # Compute cosine similarity
            query_emb = embeddings[0]
            content_emb = embeddings[1]
            
            # Normalize vectors
            query_norm = query_emb / np.linalg.norm(query_emb)
            content_norm = content_emb / np.linalg.norm(content_emb)
            
            # Cosine similarity
            similarity = np.dot(query_norm, content_norm)
            
            return float(max(0.0, similarity))  # Ensure non-negative
            
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _compute_keyword_overlap(self, query: str, content: str) -> float:
        """
        Compute keyword overlap between query and content.
        
        Uses multiple keyword matching strategies.
        """
        # Tokenize and normalize
        query_terms = set(self._tokenize(query.lower()))
        content_terms = set(self._tokenize(content.lower()))
        
        if not query_terms:
            return 0.0
        
        # Exact term overlap
        exact_overlap = len(query_terms.intersection(content_terms))
        exact_score = exact_overlap / len(query_terms)
        
        # Partial term matching (for compound words, etc.)
        partial_matches = 0
        for query_term in query_terms:
            for content_term in content_terms:
                if query_term in content_term or content_term in query_term:
                    if len(query_term) > 3 and len(content_term) > 3:  # Avoid short word matches
                        partial_matches += 0.5
                        break
        
        partial_score = min(1.0, partial_matches / len(query_terms))
        
        # Phrase matching
        phrase_score = self._compute_phrase_overlap(query, content)
        
        # Combine keyword scores
        keyword_score = 0.5 * exact_score + 0.3 * partial_score + 0.2 * phrase_score
        
        return min(1.0, keyword_score)
    
    def _compute_phrase_overlap(self, query: str, content: str) -> float:
        """Check for phrase-level matches between query and content."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Check if query appears as substring
        if query_lower in content_lower:
            return 1.0
        
        # Check for significant phrase overlaps
        query_words = query_lower.split()
        if len(query_words) < 2:
            return 0.0
        
        # Look for consecutive word matches
        max_phrase_match = 0
        for i in range(len(query_words) - 1):
            for j in range(i + 2, len(query_words) + 1):
                phrase = " ".join(query_words[i:j])
                if phrase in content_lower:
                    max_phrase_match = max(max_phrase_match, j - i)
        
        return max_phrase_match / len(query_words) if query_words else 0.0
    
    def _compute_ground_truth_relevance(
        self, 
        chunk: RetrievalResult, 
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Compute relevance based on ground truth data.
        
        Ground truth can contain:
        - relevant_chunk_ids: List of IDs that should be retrieved
        - relevant_sources: List of sources that should be retrieved
        - irrelevant_chunks: List of chunks that should NOT be retrieved
        """
        score = 0.0
        
        # Check if chunk is in relevant chunks
        relevant_chunks = ground_truth.get("relevant_chunk_ids", [])
        if chunk.chunk_id in relevant_chunks:
            score += 1.0
        
        # Check if source is relevant
        relevant_sources = ground_truth.get("relevant_sources", [])
        if chunk.source_id in relevant_sources:
            score += 0.5
        
        # Penalize if chunk is marked as irrelevant
        irrelevant_chunks = ground_truth.get("irrelevant_chunks", [])
        if chunk.chunk_id in irrelevant_chunks:
            score -= 1.0
        
        return max(0.0, min(1.0, score))
    
    def _compute_content_quality(self, content: str, query: str) -> float:
        """
        Assess content quality indicators for relevance.
        """
        score = 0.0
        
        # Length appropriateness (not too short, not too long)
        content_length = len(content)
        if 50 <= content_length <= 2000:
            score += 0.2
        elif content_length < 20:
            score -= 0.3  # Too short, likely not informative
        
        # Sentence structure (indicates well-formed content)
        sentences = content.count('.') + content.count('!') + content.count('?')
        if sentences > 0:
            avg_sentence_length = content_length / sentences
            if 10 <= avg_sentence_length <= 200:  # Reasonable sentence length
                score += 0.2
        
        # Query term density (not too high, indicates keyword stuffing)
        query_terms = set(self._tokenize(query.lower()))
        content_terms = self._tokenize(content.lower())
        
        if content_terms:
            query_term_count = sum(1 for term in content_terms if term in query_terms)
            density = query_term_count / len(content_terms)
            
            if 0.05 <= density <= 0.3:  # Sweet spot for term density
                score += 0.3
            elif density > 0.5:  # Too high, suspicious
                score -= 0.2
        
        # Information density (ratio of informative words)
        informative_words = [word for word in content_terms if len(word) > 3]
        if content_terms:
            info_ratio = len(informative_words) / len(content_terms)
            score += 0.3 * info_ratio
        
        return max(0.0, min(1.0, score))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for keyword analysis."""
        return re.findall(r'\b[a-zA-Z0-9]+\b', text)
    
    def _combine_relevance_scores(self, scores: Dict[str, float]) -> float:
        """
        Combine different relevance signals into final score.
        
        Weights are based on reliability of each signal.
        """
        total_score = 0.0
        total_weight = 0.0
        
        # Semantic similarity (most reliable if available)
        if "semantic_similarity" in scores:
            weight = 0.4
            total_score += weight * scores["semantic_similarity"]
            total_weight += weight
        
        # Keyword overlap (always available, good signal)
        if "keyword_overlap" in scores:
            weight = 0.3
            total_score += weight * scores["keyword_overlap"]
            total_weight += weight
        
        # Ground truth (perfect signal if available)
        if "ground_truth" in scores:
            weight = 0.5
            total_score += weight * scores["ground_truth"]
            total_weight += weight
        
        # Content quality (supporting signal)
        if "content_quality" in scores:
            weight = 0.2
            total_score += weight * scores["content_quality"]
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_overall_relevance(self, chunk_scores: List[float]) -> float:
        """
        Calculate overall relevance from individual chunk scores.
        
        Uses weighted average with emphasis on top results.
        """
        if not chunk_scores:
            return 0.0
        
        # Sort scores in descending order
        sorted_scores = sorted(chunk_scores, reverse=True)
        
        # Weighted average with higher weight for top results
        total_score = 0.0
        total_weight = 0.0
        
        for i, score in enumerate(sorted_scores):
            # Decay weight for lower ranked results
            weight = 1.0 / (1 + i * 0.5)
            total_score += weight * score
            total_weight += weight
        
        return total_score / total_weight
    
    def _analyze_retrieval_quality(
        self, 
        query: str, 
        chunks: List[RetrievalResult], 
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze overall retrieval quality and provide insights.
        """
        analysis = {}
        
        if not scores:
            return {"error": "No scores to analyze"}
        
        # Basic statistics
        analysis["score_stats"] = {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
        }
        
        # Relevance distribution
        relevant_count = sum(1 for s in scores if s >= self.relevance_threshold)
        analysis["relevance_distribution"] = {
            "relevant_chunks": relevant_count,
            "irrelevant_chunks": len(scores) - relevant_count,
            "relevance_rate": relevant_count / len(scores)
        }
        
        # Quality insights
        insights = []
        
        if analysis["score_stats"]["mean"] < 0.3:
            insights.append("Low average relevance - consider improving retrieval strategy")
        
        if relevant_count == 0:
            insights.append("No relevant chunks found - query may be too specific or corpus insufficient")
        
        if analysis["score_stats"]["std"] < 0.1:
            insights.append("Low score variance - retrieval may lack discrimination")
        
        if len(chunks) > 0 and chunks[0].score < self.relevance_threshold:
            insights.append("Top result is not relevant according to threshold")
        
        analysis["insights"] = insights
        
        return analysis