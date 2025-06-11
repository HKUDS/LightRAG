"""
Adaptive Threshold Optimizer for Relationship Quality Filtering

Integrates with existing ThresholdManager to provide adaptive learning capabilities
for relationship type-specific confidence thresholds based on performance feedback.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import statistics

from .threshold_manager import ThresholdManager
from .relationship_filter_metrics import get_filter_metrics
from ...utils import logger


class AdaptiveThresholdOptimizer:
    """
    Optimizes relationship filtering thresholds based on performance feedback and metrics.
    
    Works in conjunction with ThresholdManager to provide adaptive learning capabilities
    while maintaining the existing infrastructure.
    """
    
    def __init__(self, threshold_manager: Optional[ThresholdManager] = None,
                 min_samples_for_adjustment: int = 20,
                 adjustment_rate: float = 0.05,
                 target_precision: float = 0.8,
                 target_recall: float = 0.85):
        """
        Initialize adaptive threshold optimizer.
        
        Args:
            threshold_manager: Existing ThresholdManager instance
            min_samples_for_adjustment: Minimum samples needed before adjusting thresholds
            adjustment_rate: Rate of threshold adjustments (0.01-0.1 recommended)
            target_precision: Target precision for filtering (0.7-0.9 recommended)
            target_recall: Target recall for filtering (0.8-0.95 recommended)
        """
        self.threshold_manager = threshold_manager or ThresholdManager()
        self.min_samples_for_adjustment = min_samples_for_adjustment
        self.adjustment_rate = adjustment_rate
        self.target_precision = target_precision
        self.target_recall = target_recall
        
        # Performance tracking
        self.category_performance = defaultdict(lambda: {
            "samples": [],
            "precision_estimates": [],
            "recall_estimates": [],
            "retention_rates": [],
            "last_adjustment": None,
            "adjustment_count": 0
        })
        
        # Load existing optimization data
        self._load_optimization_history()
    
    def analyze_and_optimize(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Analyze recent filter performance and optimize thresholds if needed.
        
        Args:
            force_update: Force threshold updates regardless of sample count
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        logger.debug("Starting adaptive threshold optimization analysis")
        
        # Get recent metrics from filter metrics collector
        try:
            metrics = get_filter_metrics()
            performance_analysis = metrics.get_performance_analysis()
        except Exception as e:
            logger.warning(f"Could not access filter metrics for optimization: {e}")
            return {"error": "Metrics not available", "recommendations": []}
        
        category_performance = performance_analysis.get("category_performance", {})
        if not category_performance:
            return {"error": "No category performance data available", "recommendations": []}
        
        optimization_results = {
            "optimizations_made": [],
            "recommendations": [],
            "category_analysis": {},
            "threshold_changes": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Analyze each category
        for category, perf_data in category_performance.items():
            avg_retention = perf_data.get("average_retention", 0)
            session_count = perf_data.get("session_count", 0)
            trend = perf_data.get("trend", "stable")
            
            # Record performance data
            self.category_performance[category]["retention_rates"].append(avg_retention)
            self.category_performance[category]["samples"].append(session_count)
            
            # Estimate precision and recall based on retention patterns
            precision_estimate, recall_estimate = self._estimate_precision_recall(category, avg_retention, trend)
            
            self.category_performance[category]["precision_estimates"].append(precision_estimate)
            self.category_performance[category]["recall_estimates"].append(recall_estimate)
            
            # Analyze category performance
            category_analysis = {
                "current_retention": avg_retention,
                "estimated_precision": precision_estimate,
                "estimated_recall": recall_estimate,
                "trend": trend,
                "sample_count": session_count,
                "needs_adjustment": False,
                "recommended_action": "none"
            }
            
            # Check if optimization is needed
            should_optimize, action, reason = self._should_optimize_category(
                category, precision_estimate, recall_estimate, avg_retention, session_count, force_update
            )
            
            if should_optimize:
                # Calculate new threshold
                current_threshold = self._get_current_threshold(category)
                new_threshold = self._calculate_optimized_threshold(
                    category, current_threshold, action, precision_estimate, recall_estimate
                )
                
                # Apply threshold update
                self._update_category_threshold(category, new_threshold)
                
                optimization_results["optimizations_made"].append({
                    "category": category,
                    "old_threshold": current_threshold,
                    "new_threshold": new_threshold,
                    "action": action,
                    "reason": reason,
                    "estimated_impact": self._estimate_impact(action, precision_estimate, recall_estimate)
                })
                
                optimization_results["threshold_changes"][category] = {
                    "from": current_threshold,
                    "to": new_threshold,
                    "change": new_threshold - current_threshold
                }
                
                category_analysis["needs_adjustment"] = True
                category_analysis["recommended_action"] = action
                
                # Record adjustment
                self.category_performance[category]["last_adjustment"] = datetime.now().isoformat()
                self.category_performance[category]["adjustment_count"] += 1
                
                logger.info(f"Optimized threshold for category '{category}': {current_threshold:.3f} → {new_threshold:.3f} ({action})")
            
            optimization_results["category_analysis"][category] = category_analysis
        
        # Generate general recommendations
        optimization_results["recommendations"] = self._generate_optimization_recommendations(
            category_performance, optimization_results["optimizations_made"]
        )
        
        # Save optimization history
        self._save_optimization_history(optimization_results)
        
        return optimization_results
    
    def _estimate_precision_recall(self, category: str, retention_rate: float, trend: str) -> Tuple[float, float]:
        """
        Estimate precision and recall based on retention patterns and category characteristics.
        
        This is a heuristic approach since we don't have ground truth labels.
        """
        # Category-specific baselines based on expected behavior
        category_baselines = {
            "technical_core": {"precision": 0.85, "recall": 0.80},       # High precision expected
            "development_operations": {"precision": 0.80, "recall": 0.75}, # Moderate precision
            "system_interactions": {"precision": 0.75, "recall": 0.85},    # Higher recall tolerance
            "troubleshooting_support": {"precision": 0.70, "recall": 0.85}, # Higher recall tolerance
            "abstract_conceptual": {"precision": 0.60, "recall": 0.90},     # Very high recall tolerance
            "data_flow": {"precision": 0.80, "recall": 0.75}               # Moderate precision
        }
        
        baseline = category_baselines.get(category, {"precision": 0.75, "recall": 0.80})
        base_precision = baseline["precision"]
        base_recall = baseline["recall"]
        
        # Adjust based on retention rate
        # Higher retention suggests either good precision OR low thresholds (high recall, potentially lower precision)
        # Lower retention suggests either poor recall OR high thresholds (high precision, potentially lower recall)
        
        if retention_rate > 0.9:
            # Very high retention - likely high recall, but precision may vary
            estimated_recall = min(0.95, base_recall + 0.1)
            estimated_precision = max(0.4, base_precision - (retention_rate - 0.9) * 2)  # May be trading precision for recall
        elif retention_rate < 0.5:
            # Very low retention - likely high precision, but poor recall
            estimated_precision = min(0.95, base_precision + (0.5 - retention_rate) * 1.5)
            estimated_recall = max(0.3, base_recall - (0.5 - retention_rate) * 2)
        else:
            # Moderate retention - estimate based on trend and category
            retention_factor = (retention_rate - 0.7) * 0.5  # -0.1 to +0.1 range
            estimated_precision = max(0.3, min(0.95, base_precision + retention_factor))
            estimated_recall = max(0.3, min(0.95, base_recall - retention_factor * 0.5))
        
        # Adjust based on trend
        if trend == "declining":
            estimated_recall -= 0.05  # Declining performance suggests recall issues
        elif trend == "improving":
            estimated_precision += 0.05  # Improving performance suggests better precision
        
        return round(estimated_precision, 3), round(estimated_recall, 3)
    
    def _should_optimize_category(self, category: str, precision: float, recall: float,
                                retention_rate: float, sample_count: int, force_update: bool) -> Tuple[bool, str, str]:
        """
        Determine if a category needs threshold optimization.
        
        Returns:
            (should_optimize, action, reason) tuple
        """
        # Check minimum sample requirement
        if not force_update and sample_count < self.min_samples_for_adjustment:
            return False, "none", f"Insufficient samples ({sample_count} < {self.min_samples_for_adjustment})"
        
        # Check if recently adjusted (avoid over-adjustment)
        last_adjustment = self.category_performance[category]["last_adjustment"]
        if last_adjustment:
            try:
                last_adjustment_time = datetime.fromisoformat(last_adjustment)
                if datetime.now() - last_adjustment_time < timedelta(hours=1):
                    return False, "none", "Recently adjusted (within 1 hour)"
            except:
                pass  # Ignore parsing errors
        
        # Precision too low (false positives)
        if precision < self.target_precision - 0.1:
            return True, "increase_threshold", f"Precision too low ({precision:.3f} < {self.target_precision - 0.1:.3f})"
        
        # Recall too low (false negatives)
        if recall < self.target_recall - 0.1:
            return True, "decrease_threshold", f"Recall too low ({recall:.3f} < {self.target_recall - 0.1:.3f})"
        
        # Precision too high (potentially over-filtering)
        if precision > self.target_precision + 0.15 and recall < self.target_recall:
            return True, "decrease_threshold", f"Over-filtering suspected (precision {precision:.3f} > {self.target_precision + 0.15:.3f}, recall {recall:.3f})"
        
        # Extreme retention rates
        if retention_rate < 0.3:
            return True, "decrease_threshold", f"Very low retention rate ({retention_rate:.3f})"
        elif retention_rate > 0.98:
            return True, "increase_threshold", f"Very high retention rate ({retention_rate:.3f})"
        
        return False, "none", "Performance within acceptable range"
    
    def _calculate_optimized_threshold(self, category: str, current_threshold: float,
                                     action: str, precision: float, recall: float) -> float:
        """Calculate optimized threshold based on current performance."""
        if action == "increase_threshold":
            # Increase threshold to improve precision
            adjustment = self.adjustment_rate
            if precision < 0.5:
                adjustment *= 2  # Larger adjustment for very poor precision
        elif action == "decrease_threshold":
            # Decrease threshold to improve recall
            adjustment = -self.adjustment_rate
            if recall < 0.5:
                adjustment *= 2  # Larger adjustment for very poor recall
        else:
            adjustment = 0
        
        new_threshold = current_threshold + adjustment
        
        # Ensure threshold stays within reasonable bounds
        new_threshold = max(0.1, min(0.95, new_threshold))
        
        return round(new_threshold, 3)
    
    def _get_current_threshold(self, category: str) -> float:
        """Get current threshold for a category."""
        # Map our categories to ThresholdManager's relationship types
        category_mapping = {
            "technical_core": "uses",  # Use a representative type
            "development_operations": "creates",
            "system_interactions": "manages",
            "troubleshooting_support": "debugs",
            "abstract_conceptual": "related",
            "data_flow": "reads_from"
        }
        
        rep_type = category_mapping.get(category, "related")
        return self.threshold_manager.get_threshold(rep_type)
    
    def _update_category_threshold(self, category: str, new_threshold: float):
        """Update threshold for a category."""
        # For now, we'll use the threshold manager's update method
        # In a full implementation, we'd update all types in the category
        category_mapping = {
            "technical_core": "uses",
            "development_operations": "creates", 
            "system_interactions": "manages",
            "troubleshooting_support": "debugs",
            "abstract_conceptual": "related",
            "data_flow": "reads_from"
        }
        
        rep_type = category_mapping.get(category, "related")
        self.threshold_manager.update_threshold(rep_type, new_threshold)
    
    def _estimate_impact(self, action: str, current_precision: float, current_recall: float) -> Dict[str, float]:
        """Estimate the impact of a threshold change."""
        if action == "increase_threshold":
            return {
                "precision_change": +0.05,
                "recall_change": -0.02,
                "retention_change": -0.03
            }
        elif action == "decrease_threshold":
            return {
                "precision_change": -0.03,
                "recall_change": +0.05,
                "retention_change": +0.04
            }
        else:
            return {
                "precision_change": 0.0,
                "recall_change": 0.0,
                "retention_change": 0.0
            }
    
    def _generate_optimization_recommendations(self, category_performance: Dict,
                                             optimizations_made: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on optimization analysis."""
        recommendations = []
        
        if not optimizations_made:
            recommendations.append("No threshold optimizations needed at this time.")
        else:
            recommendations.append(f"Applied {len(optimizations_made)} threshold optimizations.")
        
        # Analyze overall patterns
        low_precision_categories = [
            cat for cat, perf in category_performance.items()
            if perf.get("estimated_precision", 1.0) < 0.6
        ]
        
        low_recall_categories = [
            cat for cat, perf in category_performance.items()
            if perf.get("estimated_recall", 1.0) < 0.7
        ]
        
        if low_precision_categories:
            recommendations.append(
                f"Categories with low precision detected: {', '.join(low_precision_categories)}. "
                f"Consider reviewing relationship extraction quality."
            )
        
        if low_recall_categories:
            recommendations.append(
                f"Categories with low recall detected: {', '.join(low_recall_categories)}. "
                f"Consider reviewing filtering rules or lowering thresholds further."
            )
        
        return recommendations[:3]  # Limit to 3 recommendations
    
    def _load_optimization_history(self):
        """Load optimization history from storage."""
        # For now, use a simple approach - could be enhanced to use persistent storage
        pass
    
    def _save_optimization_history(self, optimization_results: Dict[str, Any]):
        """Save optimization history to storage."""
        # For now, just log the results - could be enhanced to use persistent storage
        logger.debug(f"Optimization results: {len(optimization_results.get('optimizations_made', []))} changes made")


# Convenience function for easy integration
def optimize_relationship_thresholds(threshold_manager: Optional[ThresholdManager] = None,
                                   force_update: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run adaptive threshold optimization.
    
    Args:
        threshold_manager: Optional ThresholdManager instance
        force_update: Force optimization regardless of sample count
        
    Returns:
        Optimization results dictionary
    """
    optimizer = AdaptiveThresholdOptimizer(threshold_manager)
    return optimizer.analyze_and_optimize(force_update)


if __name__ == "__main__":
    # Test adaptive threshold optimization
    print("Testing Adaptive Threshold Optimizer...")
    
    # Create test threshold manager
    from .threshold_manager import ThresholdManager
    threshold_manager = ThresholdManager()
    
    # Create optimizer
    optimizer = AdaptiveThresholdOptimizer(threshold_manager)
    
    # Run optimization (this will use real metrics if available)
    results = optimizer.analyze_and_optimize(force_update=True)
    
    print("Optimization Results:")
    print(json.dumps(results, indent=2))
    
    print("✅ Adaptive threshold optimization test completed!")