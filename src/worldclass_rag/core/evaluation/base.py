"""
Base evaluation framework for RAG systems.

Implements the 4 key metrics from AI News & Strategy Daily video:
1. Relevance: Are the right chunks retrieved?
2. Fidelity: Is the response based on real sources?
3. Quality: Would a human rate it as correct?
4. Latency: Is it fast enough (couple of seconds)?
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

from ..retrieval.base import RetrievalResult


@dataclass
class EvaluationMetrics:
    """
    Container for the 4 key RAG evaluation metrics.
    
    Following the framework from AI News & Strategy Daily.
    """
    relevance: float = 0.0      # 0.0 - 1.0: Are right chunks retrieved?
    fidelity: float = 0.0       # 0.0 - 1.0: Response based on real sources?
    quality: float = 0.0        # 0.0 - 1.0: Would human rate as correct?
    latency: float = 0.0        # Time in seconds: Fast enough?
    
    # Additional derived metrics
    overall_score: float = 0.0  # Weighted combination of above
    
    def __post_init__(self):
        """Calculate overall score."""
        # Weighted average: Relevance and Quality are most important
        self.overall_score = (
            0.3 * self.relevance +
            0.2 * self.fidelity +
            0.4 * self.quality +
            0.1 * min(1.0, max(0.0, 1.0 - (self.latency - 2.0) / 8.0))  # Penalty after 2s
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "relevance": self.relevance,
            "fidelity": self.fidelity,
            "quality": self.quality,
            "latency": self.latency,
            "overall_score": self.overall_score,
        }
    
    def is_passing(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if metrics pass minimum thresholds.
        
        Default thresholds based on production standards.
        """
        default_thresholds = {
            "relevance": 0.7,
            "fidelity": 0.8,
            "quality": 0.7,
            "latency": 5.0,  # seconds
            "overall_score": 0.7,
        }
        
        thresholds = thresholds or default_thresholds
        
        return (
            self.relevance >= thresholds.get("relevance", 0.7) and
            self.fidelity >= thresholds.get("fidelity", 0.8) and
            self.quality >= thresholds.get("quality", 0.7) and
            self.latency <= thresholds.get("latency", 5.0) and
            self.overall_score >= thresholds.get("overall_score", 0.7)
        )


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a query-response pair.
    """
    query: str
    response: str
    retrieved_chunks: List[RetrievalResult]
    metrics: EvaluationMetrics
    
    # Evaluation context
    timestamp: datetime = field(default_factory=datetime.now)
    evaluator_version: str = "1.0"
    evaluation_method: str = "automatic"
    
    # Detailed breakdown
    relevance_details: Dict[str, Any] = field(default_factory=dict)
    fidelity_details: Dict[str, Any] = field(default_factory=dict)
    quality_details: Dict[str, Any] = field(default_factory=dict)
    latency_details: Dict[str, Any] = field(default_factory=dict)
    
    # Ground truth (if available)
    ground_truth_answer: Optional[str] = None
    ground_truth_chunks: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "response": self.response,
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "evaluator_version": self.evaluator_version,
            "evaluation_method": self.evaluation_method,
            "relevance_details": self.relevance_details,
            "fidelity_details": self.fidelity_details,
            "quality_details": self.quality_details,
            "latency_details": self.latency_details,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_chunks": self.ground_truth_chunks,
        }
    
    def save_to_json(self, file_path: str) -> None:
        """Save evaluation result to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class RAGEvaluator(ABC):
    """
    Abstract base class for RAG evaluation.
    
    Implements the evaluation framework for the 4 key metrics
    identified in the AI News & Strategy Daily video.
    """
    
    def __init__(self, name: str):
        """
        Initialize evaluator.
        
        Args:
            name: Human-readable name for this evaluator
        """
        self.name = name
    
    @abstractmethod
    def evaluate(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[RetrievalResult],
        ground_truth: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a query-response pair.
        
        Args:
            query: Original user query
            response: Generated response
            retrieved_chunks: Chunks retrieved by RAG system
            ground_truth: Optional ground truth data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete evaluation result with all 4 metrics
        """
        pass
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries with keys:
                       'query', 'response', 'retrieved_chunks', 'ground_truth'
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self.evaluate(
                    query=test_case['query'],
                    response=test_case['response'],
                    retrieved_chunks=test_case['retrieved_chunks'],
                    ground_truth=test_case.get('ground_truth'),
                    **kwargs
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating test case {i}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    query=test_case.get('query', ''),
                    response=test_case.get('response', ''),
                    retrieved_chunks=test_case.get('retrieved_chunks', []),
                    metrics=EvaluationMetrics(),  # All zeros
                )
                error_result.evaluation_method = "error"
                results.append(error_result)
        
        return results
    
    def compute_aggregate_metrics(
        self, 
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics across multiple evaluation results.
        
        Provides summary statistics for the evaluation set.
        """
        if not results:
            return {}
        
        # Extract individual metrics
        relevance_scores = [r.metrics.relevance for r in results]
        fidelity_scores = [r.metrics.fidelity for r in results]
        quality_scores = [r.metrics.quality for r in results]
        latency_scores = [r.metrics.latency for r in results]
        overall_scores = [r.metrics.overall_score for r in results]
        
        # Calculate statistics
        def calc_stats(scores: List[float]) -> Dict[str, float]:
            return {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "median": sorted(scores)[len(scores) // 2],
                "std": (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5
            }
        
        aggregate = {
            "total_evaluations": len(results),
            "relevance": calc_stats(relevance_scores),
            "fidelity": calc_stats(fidelity_scores),
            "quality": calc_stats(quality_scores),
            "latency": calc_stats(latency_scores),
            "overall_score": calc_stats(overall_scores),
        }
        
        # Pass/fail statistics
        passing_results = [r for r in results if r.metrics.is_passing()]
        aggregate["pass_rate"] = len(passing_results) / len(results)
        
        # Latency analysis (important for production)
        fast_results = [r for r in results if r.metrics.latency <= 2.0]  # "couple of seconds"
        aggregate["fast_response_rate"] = len(fast_results) / len(results)
        
        # Quality distribution
        high_quality = [r for r in results if r.metrics.quality >= 0.8]
        medium_quality = [r for r in results if 0.6 <= r.metrics.quality < 0.8]
        low_quality = [r for r in results if r.metrics.quality < 0.6]
        
        aggregate["quality_distribution"] = {
            "high": len(high_quality) / len(results),
            "medium": len(medium_quality) / len(results),
            "low": len(low_quality) / len(results),
        }
        
        return aggregate
    
    def create_evaluation_report(
        self, 
        results: List[EvaluationResult],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report.
        
        Includes aggregate metrics, detailed analysis, and recommendations.
        """
        aggregate_metrics = self.compute_aggregate_metrics(results)
        
        # Identify problematic cases
        failing_cases = [r for r in results if not r.metrics.is_passing()]
        slow_cases = [r for r in results if r.metrics.latency > 5.0]
        low_relevance_cases = [r for r in results if r.metrics.relevance < 0.6]
        
        report = {
            "evaluation_summary": {
                "evaluator": self.name,
                "total_cases": len(results),
                "evaluation_date": datetime.now().isoformat(),
            },
            "aggregate_metrics": aggregate_metrics,
            "problem_analysis": {
                "failing_cases": len(failing_cases),
                "slow_responses": len(slow_cases),
                "low_relevance": len(low_relevance_cases),
            },
            "recommendations": self._generate_recommendations(aggregate_metrics, results),
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(
        self, 
        aggregate_metrics: Dict[str, Any], 
        results: List[EvaluationResult]
    ) -> List[str]:
        """
        Generate recommendations based on evaluation results.
        
        Provides actionable insights for improving RAG performance.
        """
        recommendations = []
        
        # Relevance recommendations
        if aggregate_metrics["relevance"]["mean"] < 0.7:
            recommendations.append(
                "Low relevance scores detected. Consider: "
                "1) Improving chunking strategy, "
                "2) Using better embedding model, "
                "3) Implementing hybrid search"
            )
        
        # Fidelity recommendations
        if aggregate_metrics["fidelity"]["mean"] < 0.8:
            recommendations.append(
                "Fidelity issues detected. The response may not be based on retrieved sources. "
                "Consider: 1) Improving prompt engineering, "
                "2) Adding source verification, "
                "3) Using stricter generation constraints"
            )
        
        # Quality recommendations
        if aggregate_metrics["quality"]["mean"] < 0.7:
            recommendations.append(
                "Quality scores are low. Consider: "
                "1) Using a better LLM model, "
                "2) Improving retrieval quality, "
                "3) Better prompt engineering"
            )
        
        # Latency recommendations
        if aggregate_metrics["fast_response_rate"] < 0.8:
            recommendations.append(
                "Latency is above the 'couple of seconds' target. Consider: "
                "1) Optimizing vector database, "
                "2) Reducing chunk size, "
                "3) Using faster embedding model, "
                "4) Implementing response caching"
            )
        
        # Pass rate recommendations
        if aggregate_metrics["pass_rate"] < 0.8:
            recommendations.append(
                "Overall pass rate is low. Focus on the lowest-scoring metric "
                "and implement systematic improvements"
            )
        
        return recommendations
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get information about this evaluator."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "version": "1.0",
            "metrics": ["relevance", "fidelity", "quality", "latency"],
        }