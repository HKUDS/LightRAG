"""
Hybrid Retriever combining semantic and keyword search with re-ranking.

Implements the hybrid search approach recommended in the AI News & Strategy Daily video:
"Combines semantic search with keyword search and re-ranking for better precision"
"""

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from .base import Retriever, RetrievalResult
from .semantic_retriever import SemanticRetriever
from .keyword_retriever import KeywordRetriever


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search behavior."""
    semantic_weight: float = 0.7      # Weight for semantic scores
    keyword_weight: float = 0.3       # Weight for keyword scores
    fusion_method: str = "rrf"        # "linear", "rrf" (Reciprocal Rank Fusion), or "weighted_sum"
    rrf_k: int = 60                   # RRF parameter (only used if fusion_method="rrf")
    rerank: bool = True               # Whether to apply re-ranking
    rerank_top_k: int = 20            # How many results to consider for re-ranking
    final_top_k: int = 5              # Final number of results to return


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining semantic and keyword search.
    
    Implements the advanced retrieval strategy described in the video:
    - Level 2 RAG: Hybrid search combining semantic + keyword
    - Re-ranking to "boost accuracy for business purposes significantly"
    - Better precision and potential for improved speed
    """
    
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_retriever: KeywordRetriever,
        config: Optional[HybridSearchConfig] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_retriever: Semantic search component
            keyword_retriever: Keyword search component
            config: Hybrid search configuration
        """
        super().__init__(name="HybridRetriever")
        
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.config = config or HybridSearchConfig()
        
        # Validate configuration
        self._validate_config()
        
        # Performance tracking
        self.last_search_times: Dict[str, float] = {}
        self.search_stats: Dict[str, Any] = {
            "total_searches": 0,
            "avg_semantic_time": 0.0,
            "avg_keyword_time": 0.0,
            "avg_fusion_time": 0.0,
            "avg_total_time": 0.0,
        }
    
    def _validate_config(self) -> None:
        """Validate hybrid search configuration."""
        if not (0 <= self.config.semantic_weight <= 1):
            raise ValueError("semantic_weight must be between 0 and 1")
        if not (0 <= self.config.keyword_weight <= 1):
            raise ValueError("keyword_weight must be between 0 and 1")
        
        total_weight = self.config.semantic_weight + self.config.keyword_weight
        if abs(total_weight - 1.0) > 0.001:
            print(f"Warning: Weights don't sum to 1.0 (sum={total_weight}), normalizing...")
            self.config.semantic_weight /= total_weight
            self.config.keyword_weight /= total_weight
        
        if self.config.fusion_method not in ["linear", "rrf", "weighted_sum"]:
            raise ValueError("fusion_method must be 'linear', 'rrf', or 'weighted_sum'")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to both semantic and keyword indexes.
        """
        if not chunks:
            return
        
        start_time = datetime.now()
        
        # Add to both retrievers
        self.semantic_retriever.add_chunks(chunks)
        self.keyword_retriever.add_chunks(chunks)
        
        end_time = datetime.now()
        indexing_time = (end_time - start_time).total_seconds() * 1000
        
        print(f"Added {len(chunks)} chunks to hybrid index in {indexing_time:.1f}ms")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid search with fusion and optional re-ranking.
        
        Process:
        1. Get results from both semantic and keyword retrievers
        2. Fuse results using configured method
        3. Apply re-ranking if enabled
        4. Return top-k results
        """
        if not query.strip():
            return []
        
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve from both methods
            semantic_start = datetime.now()
            semantic_results = self.semantic_retriever.retrieve(
                query, 
                top_k=self.config.rerank_top_k if self.config.rerank else top_k,
                filters=filters
            )
            semantic_time = (datetime.now() - semantic_start).total_seconds() * 1000
            
            keyword_start = datetime.now()
            keyword_results = self.keyword_retriever.retrieve(
                query,
                top_k=self.config.rerank_top_k if self.config.rerank else top_k,
                filters=filters
            )
            keyword_time = (datetime.now() - keyword_start).total_seconds() * 1000
            
            # Step 2: Fuse results
            fusion_start = datetime.now()
            fused_results = self._fuse_results(semantic_results, keyword_results, query)
            fusion_time = (datetime.now() - fusion_start).total_seconds() * 1000
            
            # Step 3: Apply re-ranking if enabled
            if self.config.rerank and len(fused_results) > 1:
                rerank_start = datetime.now()
                fused_results = self._rerank_results(fused_results, query)
                rerank_time = (datetime.now() - rerank_start).total_seconds() * 1000
            else:
                rerank_time = 0.0
            
            # Step 4: Return top-k
            final_results = fused_results[:top_k]
            
            # Update performance statistics
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_search_stats(semantic_time, keyword_time, fusion_time, total_time)
            
            # Add hybrid-specific metadata to results
            for rank, result in enumerate(final_results):
                result.add_metadata("hybrid_retrieval", True)
                result.add_metadata("fusion_method", self.config.fusion_method)
                result.add_metadata("reranked", self.config.rerank)
                result.add_metadata("final_rank", rank)
                result.add_metadata("search_times", {
                    "semantic_ms": semantic_time,
                    "keyword_ms": keyword_time,
                    "fusion_ms": fusion_time,
                    "rerank_ms": rerank_time,
                    "total_ms": total_time,
                })
            
            return final_results
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return []
    
    def _fuse_results(
        self, 
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        Fuse results from semantic and keyword search.
        
        Implements different fusion strategies for combining results.
        """
        if not semantic_results and not keyword_results:
            return []
        elif not semantic_results:
            return keyword_results
        elif not keyword_results:
            return semantic_results
        
        # Create unified result set
        all_results = {}  # chunk_id -> RetrievalResult
        
        # Add semantic results
        for rank, result in enumerate(semantic_results):
            chunk_id = result.chunk_id or result.source_id
            
            if chunk_id not in all_results:
                # Create new fused result
                fused_result = RetrievalResult(
                    content=result.content,
                    score=0.0,  # Will be calculated
                    source_id=result.source_id,
                    metadata=result.metadata.copy(),
                    retrieval_method="hybrid",
                    chunk_id=result.chunk_id,
                    source_file=result.source_file,
                    source_section=result.source_section,
                    source_page=result.source_page,
                )
                all_results[chunk_id] = fused_result
            
            # Store component scores and ranks
            all_results[chunk_id].add_metadata("semantic_score", result.score)
            all_results[chunk_id].add_metadata("semantic_rank", rank)
        
        # Add keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result.chunk_id or result.source_id
            
            if chunk_id not in all_results:
                # Create new fused result
                fused_result = RetrievalResult(
                    content=result.content,
                    score=0.0,  # Will be calculated
                    source_id=result.source_id,
                    metadata=result.metadata.copy(),
                    retrieval_method="hybrid",
                    chunk_id=result.chunk_id,
                    source_file=result.source_file,
                    source_section=result.source_section,
                    source_page=result.source_page,
                )
                all_results[chunk_id] = fused_result
            
            # Store component scores and ranks
            all_results[chunk_id].add_metadata("keyword_score", result.score)
            all_results[chunk_id].add_metadata("keyword_rank", rank)
        
        # Calculate fused scores
        for result in all_results.values():
            result.score = self._calculate_fused_score(result)
        
        # Sort by fused score
        fused_results = list(all_results.values())
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
    
    def _calculate_fused_score(self, result: RetrievalResult) -> float:
        """
        Calculate fused score based on configuration.
        """
        semantic_score = result.get_metadata("semantic_score", 0.0)
        keyword_score = result.get_metadata("keyword_score", 0.0)
        semantic_rank = result.get_metadata("semantic_rank")
        keyword_rank = result.get_metadata("keyword_rank")
        
        if self.config.fusion_method == "linear":
            # Simple linear combination of scores
            return (self.config.semantic_weight * semantic_score + 
                   self.config.keyword_weight * keyword_score)
        
        elif self.config.fusion_method == "weighted_sum":
            # Weighted sum with normalization
            total_weight = 0.0
            weighted_sum = 0.0
            
            if semantic_score > 0:
                weighted_sum += self.config.semantic_weight * semantic_score
                total_weight += self.config.semantic_weight
            
            if keyword_score > 0:
                weighted_sum += self.config.keyword_weight * keyword_score
                total_weight += self.config.keyword_weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.config.fusion_method == "rrf":
            # Reciprocal Rank Fusion
            rrf_score = 0.0
            
            if semantic_rank is not None:
                rrf_score += self.config.semantic_weight / (self.config.rrf_k + semantic_rank + 1)
            
            if keyword_rank is not None:
                rrf_score += self.config.keyword_weight / (self.config.rrf_k + keyword_rank + 1)
            
            return rrf_score
        
        else:
            # Fallback to linear
            return (self.config.semantic_weight * semantic_score + 
                   self.config.keyword_weight * keyword_score)
    
    def _rerank_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Apply re-ranking to improve result quality.
        
        Implements basic re-ranking strategies that can "boost accuracy
        for business purposes significantly" as mentioned in the video.
        """
        if len(results) <= 1:
            return results
        
        # For now, implement simple re-ranking based on content features
        # In production, this could be replaced with a dedicated re-ranking model
        
        for result in results:
            rerank_score = self._calculate_rerank_score(result, query)
            result.reranked_score = rerank_score
            
            # Combine original score with re-ranking
            result.score = 0.7 * result.score + 0.3 * rerank_score
        
        # Sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _calculate_rerank_score(self, result: RetrievalResult, query: str) -> float:
        """
        Calculate re-ranking score based on content features.
        
        Simple heuristics that could be replaced with ML models.
        """
        content = result.content.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Exact phrase match bonus
        if query_lower in content:
            score += 0.3
        
        # Query term coverage
        query_terms = set(query_lower.split())
        content_terms = set(content.split())
        coverage = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
        score += 0.2 * coverage
        
        # Length penalty (very short or very long content)
        content_length = len(content)
        if 100 <= content_length <= 1000:
            score += 0.1
        elif content_length < 50:
            score -= 0.1
        
        # Source quality bonus (based on metadata)
        source_quality = result.get_metadata("source_quality", 0.5)
        score += 0.1 * source_quality
        
        # Recency bonus
        document_date = result.get_metadata("document_date")
        if document_date:
            # Could implement recency scoring here
            pass
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _update_search_stats(self, semantic_time: float, keyword_time: float, fusion_time: float, total_time: float) -> None:
        """Update search performance statistics."""
        stats = self.search_stats
        count = stats["total_searches"]
        
        # Update running averages
        stats["avg_semantic_time"] = (stats["avg_semantic_time"] * count + semantic_time) / (count + 1)
        stats["avg_keyword_time"] = (stats["avg_keyword_time"] * count + keyword_time) / (count + 1)
        stats["avg_fusion_time"] = (stats["avg_fusion_time"] * count + fusion_time) / (count + 1)
        stats["avg_total_time"] = (stats["avg_total_time"] * count + total_time) / (count + 1)
        
        stats["total_searches"] += 1
        
        # Store latest times
        self.last_search_times = {
            "semantic_ms": semantic_time,
            "keyword_ms": keyword_time,
            "fusion_ms": fusion_time,
            "total_ms": total_time,
        }
    
    def get_chunk_count(self) -> int:
        """Get number of indexed chunks."""
        # Should be the same for both retrievers
        return self.semantic_retriever.get_chunk_count()
    
    def clear_index(self) -> None:
        """Clear both semantic and keyword indexes."""
        self.semantic_retriever.clear_index()
        self.keyword_retriever.clear_index()
        print("Cleared hybrid index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hybrid retriever."""
        base_stats = super().get_stats()
        
        stats = {
            **base_stats,
            "semantic_retriever": self.semantic_retriever.get_stats(),
            "keyword_retriever": self.keyword_retriever.get_stats(),
            "hybrid_config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "fusion_method": self.config.fusion_method,
                "rerank": self.config.rerank,
                "rrf_k": self.config.rrf_k,
            },
            "search_performance": self.search_stats,
            "last_search_times": self.last_search_times,
        }
        
        return stats
    
    def _explain_retrieval(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Explain hybrid retrieval process."""
        explanation = super()._explain_retrieval(query, results)
        
        # Count results by source
        semantic_count = sum(1 for r in results if r.get_metadata("semantic_score") is not None)
        keyword_count = sum(1 for r in results if r.get_metadata("keyword_score") is not None)
        both_count = sum(1 for r in results if (r.get_metadata("semantic_score") is not None and 
                                               r.get_metadata("keyword_score") is not None))
        
        explanation.update({
            "method": "hybrid_search",
            "fusion_method": self.config.fusion_method,
            "weights": {
                "semantic": self.config.semantic_weight,
                "keyword": self.config.keyword_weight,
            },
            "reranking_applied": self.config.rerank,
            "result_sources": {
                "semantic_only": semantic_count - both_count,
                "keyword_only": keyword_count - both_count,
                "both_methods": both_count,
            },
            "performance": self.last_search_times,
        })
        
        return explanation