"""
Base retrieval interface and classes.

Defines the foundation for all retrieval strategies in WorldClass RAG.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class RetrievalResult:
    """
    Result of a retrieval operation with relevance scoring and metadata.
    
    Following the evaluation metrics from AI News & Strategy Daily:
    - Relevance: Are the right chunks retrieved?
    - Source tracking for fidelity verification
    - Metadata for quality assessment
    """
    content: str
    score: float
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Retrieval details
    retrieval_method: str = "unknown"
    chunk_id: Optional[str] = None
    
    # Ranking information  
    original_rank: Optional[int] = None
    reranked_score: Optional[float] = None
    
    # Source information for fidelity
    source_file: Optional[str] = None
    source_section: Optional[str] = None
    source_page: Optional[int] = None
    
    def __post_init__(self):
        """Ensure essential metadata is present."""
        if "retrieval_timestamp" not in self.metadata:
            self.metadata["retrieval_timestamp"] = datetime.now().isoformat()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata safely."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata with default."""
        return self.metadata.get(key, default)


class Retriever(ABC):
    """
    Abstract base class for all retrieval strategies.
    
    Implements the interface for retrieving relevant chunks following
    the best practices from AI News & Strategy Daily:
    - Focus on relevance as primary metric
    - Support for re-ranking to boost accuracy
    - Metadata preservation for fidelity checking
    """
    
    def __init__(self, name: str):
        """
        Initialize the retriever.
        
        Args:
            name: Human-readable name for this retriever
        """
        self.name = name
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            filters: Optional filters to apply (e.g., source, date, section)
            
        Returns:
            List of RetrievalResult ordered by relevance score (desc)
        """
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the retrieval index.
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "indexed_chunks": self.get_chunk_count(),
        }
    
    @abstractmethod
    def get_chunk_count(self) -> int:
        """Get number of indexed chunks."""
        pass
    
    def clear_index(self) -> None:
        """Clear the retrieval index."""
        pass
    
    def update_chunk(self, chunk_id: str, new_content: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update an existing chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            new_content: Updated content
            new_metadata: Updated metadata
            
        Returns:
            True if update successful, False otherwise
        """
        return False  # Default implementation - not all retrievers support updates
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the index.
        
        Args:
            chunk_id: Unique identifier for the chunk to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        return False  # Default implementation
    
    def search_with_explanation(
        self,
        query: str,
        top_k: int = 5,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Perform search with detailed explanation of the retrieval process.
        
        Useful for debugging and understanding why certain chunks were retrieved.
        """
        start_time = datetime.now()
        
        results = self.retrieve(query, top_k)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        explanation = {
            "query": query,
            "retriever": self.name,
            "retriever_type": self.__class__.__name__,
            "results_count": len(results),
            "latency_ms": latency_ms,
            "results": results,
        }
        
        if explain:
            explanation["explanation"] = self._explain_retrieval(query, results)
            
        return explanation
    
    def _explain_retrieval(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Provide explanation for why these results were retrieved.
        
        Can be overridden by specific retriever implementations.
        """
        return {
            "method": "base_retrieval",
            "query_length": len(query),
            "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            "score_range": {
                "min": min(r.score for r in results) if results else 0,
                "max": max(r.score for r in results) if results else 0,
            },
            "unique_sources": len(set(r.source_id for r in results)),
        }
    
    def validate_results(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Validate retrieval results for quality and consistency.
        
        Implements basic quality checks following best practices.
        """
        if not results:
            return {
                "valid": False,
                "issues": ["No results returned"],
                "warnings": []
            }
        
        issues = []
        warnings = []
        
        # Check score consistency
        scores = [r.score for r in results]
        if not all(scores[i] >= scores[i+1] for i in range(len(scores)-1)):
            issues.append("Results not sorted by score descending")
        
        # Check for duplicate content
        contents = [r.content for r in results]
        if len(contents) != len(set(contents)):
            warnings.append("Duplicate content in results")
        
        # Check score range
        if max(scores) - min(scores) < 0.01:
            warnings.append("Very small score variance - may indicate poor discrimination")
        
        # Check for missing essential fields
        for i, result in enumerate(results):
            if not result.content.strip():
                issues.append(f"Result {i} has empty content")
            if not result.source_id:
                issues.append(f"Result {i} missing source_id")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "result_count": len(results),
            "score_stats": {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
                "variance": sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            }
        }