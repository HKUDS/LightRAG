"""
Relationship Filter Performance Metrics Collection and Analysis

Tracks and analyzes the performance of relationship quality filtering with
type-specific intelligence to enable continuous improvement.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pathlib import Path

from ...utils import logger


class RelationshipFilterMetrics:
    """
    Tracks and analyzes relationship filter performance metrics to enable
    data-driven optimization of filtering thresholds and rules.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize metrics collection.

        Args:
            storage_path: Optional path to store metrics data
        """
        self.storage_path = (
            Path(storage_path) if storage_path else Path("./filter_metrics.json")
        )
        self.session_metrics = defaultdict(list)
        self.aggregate_metrics = defaultdict(lambda: defaultdict(int))
        self.start_time = time.time()

        # Load existing metrics if available
        self._load_existing_metrics()

    def record_filter_session(
        self,
        filter_stats: Dict[str, Any],
        classification_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Record metrics from a filter session.

        Args:
            filter_stats: Basic filter statistics
            classification_stats: Enhanced classification statistics (if available)
        """
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "total_before": filter_stats.get("total_before", 0),
            "total_after": filter_stats.get("total_after", 0),
            "retention_rate": filter_stats.get("total_after", 0)
            / max(filter_stats.get("total_before", 1), 1),
            "filter_breakdown": {
                "abstract_relationships": filter_stats.get("abstract_relationships", 0),
                "synonym_relationships": filter_stats.get("synonym_relationships", 0),
                "abstract_entities": filter_stats.get("abstract_entities", 0),
                "low_confidence": filter_stats.get("low_confidence", 0),
                "low_quality_relationships": filter_stats.get(
                    "low_quality_relationships", 0
                ),
                "type_specific_filtered": filter_stats.get("type_specific_filtered", 0),
            },
        }

        # Add enhanced classification stats if available
        if classification_stats:
            session_data["classification_enabled"] = True
            session_data["category_stats"] = classification_stats

            # Calculate category-specific metrics
            category_metrics = {}
            for category, stats in classification_stats.items():
                if stats.get("total", 0) > 0:
                    category_metrics[category] = {
                        "retention_rate": stats["kept"] / stats["total"],
                        "filter_rate": stats["filtered"] / stats["total"],
                        "total_count": stats["total"],
                    }
            session_data["category_metrics"] = category_metrics
        else:
            session_data["classification_enabled"] = False

        # Record session
        self.session_metrics["filter_sessions"].append(session_data)

        # Update aggregate metrics
        self._update_aggregates(session_data)

        # Periodically save metrics
        if len(self.session_metrics["filter_sessions"]) % 10 == 0:
            self._save_metrics()

    def record_relationship_classification(
        self, relationship_type: str, category: str, confidence: float, was_kept: bool
    ):
        """
        Record individual relationship classification results.

        Args:
            relationship_type: The relationship type that was classified
            category: The category it was assigned to
            confidence: The confidence score
            was_kept: Whether the relationship was kept after filtering
        """
        classification_data = {
            "timestamp": datetime.now().isoformat(),
            "relationship_type": relationship_type,
            "category": category,
            "confidence": confidence,
            "was_kept": was_kept,
        }

        self.session_metrics["classifications"].append(classification_data)

        # Update aggregates
        self.aggregate_metrics["relationship_types"][relationship_type] += 1
        self.aggregate_metrics["categories"][category] += 1

        if was_kept:
            self.aggregate_metrics["kept_by_type"][relationship_type] += 1
            self.aggregate_metrics["kept_by_category"][category] += 1

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session metrics.

        Returns:
            Summary dictionary with key metrics
        """
        sessions = self.session_metrics["filter_sessions"]
        classifications = self.session_metrics["classifications"]

        if not sessions:
            return {"error": "No filter sessions recorded"}

        # Calculate session-level aggregates
        total_relationships_processed = sum(s["total_before"] for s in sessions)
        total_relationships_kept = sum(s["total_after"] for s in sessions)
        average_retention_rate = sum(s["retention_rate"] for s in sessions) / len(
            sessions
        )

        # Filter breakdown
        filter_breakdown = defaultdict(int)
        for session in sessions:
            for filter_type, count in session["filter_breakdown"].items():
                filter_breakdown[filter_type] += count

        summary = {
            "session_count": len(sessions),
            "total_relationships_processed": total_relationships_processed,
            "total_relationships_kept": total_relationships_kept,
            "overall_retention_rate": total_relationships_kept
            / max(total_relationships_processed, 1),
            "average_retention_rate": average_retention_rate,
            "filter_breakdown": dict(filter_breakdown),
            "classification_enabled": any(
                s.get("classification_enabled", False) for s in sessions
            ),
        }

        # Add classification summary if available
        if classifications:
            summary["classification_summary"] = self._summarize_classifications(
                classifications
            )

        return summary

    def get_performance_analysis(self) -> Dict[str, Any]:
        """
        Analyze filter performance and provide recommendations.

        Returns:
            Analysis with performance insights and recommendations
        """
        sessions = self.session_metrics["filter_sessions"]
        if not sessions:
            return {"error": "No data available for analysis"}

        # Performance trends
        recent_sessions = sessions[-10:] if len(sessions) >= 10 else sessions
        retention_trend = [s["retention_rate"] for s in recent_sessions]

        # Category performance (if enhanced filtering is enabled)
        category_analysis = {}
        enhanced_sessions = [
            s for s in sessions if s.get("classification_enabled", False)
        ]

        if enhanced_sessions:
            category_performance = defaultdict(list)
            for session in enhanced_sessions:
                for category, metrics in session.get("category_metrics", {}).items():
                    category_performance[category].append(metrics["retention_rate"])

            for category, rates in category_performance.items():
                avg_rate = sum(rates) / len(rates)
                category_analysis[category] = {
                    "average_retention": avg_rate,
                    "session_count": len(rates),
                    "trend": (
                        "stable" if len(rates) < 3 else self._calculate_trend(rates)
                    ),
                }

        # Generate recommendations
        recommendations = self._generate_recommendations(sessions, category_analysis)

        return {
            "summary_stats": {
                "total_sessions": len(sessions),
                "recent_retention_average": sum(retention_trend) / len(retention_trend),
                "retention_trend": (
                    self._calculate_trend(retention_trend)
                    if len(retention_trend) >= 3
                    else "insufficient_data"
                ),
            },
            "category_performance": category_analysis,
            "recommendations": recommendations,
            "last_updated": datetime.now().isoformat(),
        }

    def _summarize_classifications(
        self, classifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize classification data."""
        if not classifications:
            return {}

        # Aggregate by relationship type
        type_stats = defaultdict(lambda: {"total": 0, "kept": 0, "avg_confidence": 0})
        confidence_scores = defaultdict(list)

        for c in classifications:
            rel_type = c["relationship_type"]
            type_stats[rel_type]["total"] += 1
            confidence_scores[rel_type].append(c["confidence"])

            if c["was_kept"]:
                type_stats[rel_type]["kept"] += 1

        # Calculate averages
        for rel_type, scores in confidence_scores.items():
            type_stats[rel_type]["avg_confidence"] = sum(scores) / len(scores)
            type_stats[rel_type]["retention_rate"] = (
                type_stats[rel_type]["kept"] / type_stats[rel_type]["total"]
            )

        # Find problematic types
        problematic_types = [
            rel_type
            for rel_type, stats in type_stats.items()
            if stats["retention_rate"] < 0.3 and stats["total"] >= 5
        ]

        return {
            "total_classifications": len(classifications),
            "unique_relationship_types": len(type_stats),
            "type_performance": dict(type_stats),
            "problematic_types": problematic_types,
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear trend analysis
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])

        if increases > decreases * 1.5:
            return "improving"
        elif decreases > increases * 1.5:
            return "declining"
        else:
            return "stable"

    def _generate_recommendations(
        self, sessions: List[Dict], category_analysis: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on performance data."""
        recommendations = []

        # Overall retention analysis
        recent_retention = sum(s["retention_rate"] for s in sessions[-5:]) / min(
            len(sessions), 5
        )

        if recent_retention < 0.6:
            recommendations.append(
                "Low retention rate detected (<60%). Consider lowering confidence thresholds "
                "or reviewing relationship extraction quality."
            )
        elif recent_retention > 0.95:
            recommendations.append(
                "Very high retention rate (>95%). Consider raising confidence thresholds "
                "to improve relationship quality."
            )

        # Category-specific recommendations
        for category, analysis in category_analysis.items():
            avg_retention = analysis["average_retention"]

            if avg_retention < 0.4:
                recommendations.append(
                    f"Category '{category}' has low retention ({avg_retention:.1%}). "
                    f"Consider lowering threshold or reviewing classification rules."
                )
            elif avg_retention > 0.98:
                recommendations.append(
                    f"Category '{category}' has very high retention ({avg_retention:.1%}). "
                    f"Consider raising threshold to improve precision."
                )

        # Filter breakdown analysis
        if sessions:
            latest_session = sessions[-1]
            filter_breakdown = latest_session["filter_breakdown"]
            total_filtered = sum(filter_breakdown.values())

            if total_filtered > 0:
                # Find dominant filter reasons
                dominant_filters = [
                    (filter_type, count)
                    for filter_type, count in filter_breakdown.items()
                    if count > total_filtered * 0.3
                ]

                for filter_type, count in dominant_filters:
                    recommendations.append(
                        f"'{filter_type}' is filtering {count}/{total_filtered} relationships "
                        f"({count/total_filtered:.1%}). Review if this is appropriate."
                    )

        return recommendations[:5]  # Limit to top 5 recommendations

    def _update_aggregates(self, session_data: Dict[str, Any]):
        """Update aggregate metrics with new session data."""
        self.aggregate_metrics["total_sessions"] = (
            self.aggregate_metrics.get("total_sessions", 0) + 1
        )
        self.aggregate_metrics["total_relationships_processed"] = (
            self.aggregate_metrics.get("total_relationships_processed", 0)
            + session_data["total_before"]
        )
        self.aggregate_metrics["total_relationships_kept"] = (
            self.aggregate_metrics.get("total_relationships_kept", 0)
            + session_data["total_after"]
        )

        # Update filter reason aggregates
        if "filter_reasons" not in self.aggregate_metrics:
            self.aggregate_metrics["filter_reasons"] = defaultdict(int)

        for filter_type, count in session_data["filter_breakdown"].items():
            self.aggregate_metrics["filter_reasons"][filter_type] += count

    def _load_existing_metrics(self):
        """Load existing metrics from storage if available."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.session_metrics = defaultdict(
                        list, data.get("session_metrics", {})
                    )
                    self.aggregate_metrics = defaultdict(
                        lambda: defaultdict(int), data.get("aggregate_metrics", {})
                    )
                logger.debug(f"Loaded existing filter metrics from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not load existing metrics: {e}")

    def _save_metrics(self):
        """Save current metrics to storage."""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "session_metrics": dict(self.session_metrics),
                "aggregate_metrics": dict(self.aggregate_metrics),
                "last_saved": datetime.now().isoformat(),
                "session_start": self.start_time,
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved filter metrics to {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics: {e}")

    def export_metrics_report(self, output_path: Optional[str] = None) -> str:
        """
        Export comprehensive metrics report.

        Args:
            output_path: Optional path for report output

        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = (
                f"filter_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_count": len(self.session_metrics["filter_sessions"]),
                "classification_count": len(self.session_metrics["classifications"]),
                "time_range": {
                    "start": (
                        self.session_metrics["filter_sessions"][0]["timestamp"]
                        if self.session_metrics["filter_sessions"]
                        else None
                    ),
                    "end": (
                        self.session_metrics["filter_sessions"][-1]["timestamp"]
                        if self.session_metrics["filter_sessions"]
                        else None
                    ),
                },
            },
            "session_summary": self.get_session_summary(),
            "performance_analysis": self.get_performance_analysis(),
            "raw_data": {
                "sessions": self.session_metrics["filter_sessions"][
                    -50:
                ],  # Last 50 sessions
                "classifications": self.session_metrics["classifications"][
                    -1000:
                ],  # Last 1000 classifications
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Filter metrics report exported to {output_path}")
        return output_path


# Global metrics instance for easy access
_global_metrics_instance = None


def get_filter_metrics(storage_path: Optional[str] = None) -> RelationshipFilterMetrics:
    """
    Get or create global filter metrics instance.

    Args:
        storage_path: Optional path for metrics storage

    Returns:
        RelationshipFilterMetrics instance
    """
    global _global_metrics_instance

    if _global_metrics_instance is None:
        _global_metrics_instance = RelationshipFilterMetrics(storage_path)

    return _global_metrics_instance


if __name__ == "__main__":
    # Test metrics collection
    metrics = RelationshipFilterMetrics("./test_metrics.json")

    # Simulate some filter sessions
    test_sessions = [
        {
            "total_before": 100,
            "total_after": 85,
            "abstract_relationships": 5,
            "synonym_relationships": 2,
            "abstract_entities": 3,
            "low_confidence": 3,
            "low_quality_relationships": 2,
            "type_specific_filtered": 0,
        },
        {
            "total_before": 150,
            "total_after": 120,
            "abstract_relationships": 8,
            "synonym_relationships": 5,
            "abstract_entities": 7,
            "low_confidence": 5,
            "low_quality_relationships": 5,
            "type_specific_filtered": 15,
        },
    ]

    for session_stats in test_sessions:
        metrics.record_filter_session(session_stats)

    # Test metrics reporting
    summary = metrics.get_session_summary()
    analysis = metrics.get_performance_analysis()

    print("Session Summary:")
    print(json.dumps(summary, indent=2))
    print("\nPerformance Analysis:")
    print(json.dumps(analysis, indent=2))

    # Export report
    report_path = metrics.export_metrics_report()
    print(f"\nReport exported to: {report_path}")
