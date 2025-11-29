#!/usr/bin/env python3
"""
A/B Test Results Comparator for RAGAS Evaluation

Compares two RAGAS evaluation result files to determine if a change
(e.g., orphan connections) improved or degraded retrieval quality.

Usage:
    python lightrag/evaluation/compare_results.py baseline.json experiment.json
    python lightrag/evaluation/compare_results.py results_a.json results_b.json --output comparison.json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MetricComparison:
    """Comparison of a single metric between two runs."""
    metric_name: str
    baseline_value: float
    experiment_value: float
    absolute_change: float
    relative_change_percent: float
    improved: bool
    significant: bool  # > 5% change


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling NaN."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def compare_metrics(baseline: dict, experiment: dict) -> list[MetricComparison]:
    """
    Compare metrics between baseline and experiment.

    Args:
        baseline: Benchmark stats from baseline run
        experiment: Benchmark stats from experiment run

    Returns:
        List of MetricComparison objects
    """
    comparisons = []

    baseline_avg = baseline.get("average_metrics", {})
    experiment_avg = experiment.get("average_metrics", {})

    metrics_to_compare = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevance", "Answer Relevance"),
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
        ("ragas_score", "RAGAS Score"),
    ]

    for metric_key, metric_name in metrics_to_compare:
        b_val = safe_float(baseline_avg.get(metric_key, 0))
        e_val = safe_float(experiment_avg.get(metric_key, 0))

        abs_change = e_val - b_val
        rel_change = (abs_change / b_val * 100) if b_val > 0 else 0

        comparisons.append(MetricComparison(
            metric_name=metric_name,
            baseline_value=b_val,
            experiment_value=e_val,
            absolute_change=abs_change,
            relative_change_percent=rel_change,
            improved=abs_change > 0,
            significant=abs(rel_change) > 5,  # > 5% is significant
        ))

    return comparisons


def analyze_results(baseline_path: Path, experiment_path: Path) -> dict:
    """
    Perform comprehensive A/B analysis.

    Args:
        baseline_path: Path to baseline results JSON
        experiment_path: Path to experiment results JSON

    Returns:
        Analysis results dictionary
    """
    # Load results
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(experiment_path) as f:
        experiment = json.load(f)

    baseline_stats = baseline.get("benchmark_stats", {})
    experiment_stats = experiment.get("benchmark_stats", {})

    # Compare metrics
    comparisons = compare_metrics(baseline_stats, experiment_stats)

    # Calculate overall verdict
    improvements = sum(1 for c in comparisons if c.improved)
    regressions = sum(1 for c in comparisons if not c.improved and c.absolute_change != 0)
    significant_improvements = sum(1 for c in comparisons if c.improved and c.significant)
    significant_regressions = sum(1 for c in comparisons if not c.improved and c.significant)

    # Determine verdict
    ragas_comparison = next((c for c in comparisons if c.metric_name == "RAGAS Score"), None)

    if ragas_comparison:
        if ragas_comparison.improved and ragas_comparison.significant:
            verdict = "SIGNIFICANT_IMPROVEMENT"
            verdict_description = f"RAGAS Score improved by {ragas_comparison.relative_change_percent:.1f}%"
        elif ragas_comparison.improved:
            verdict = "MINOR_IMPROVEMENT"
            verdict_description = f"RAGAS Score slightly improved by {ragas_comparison.relative_change_percent:.1f}%"
        elif ragas_comparison.significant:
            verdict = "SIGNIFICANT_REGRESSION"
            verdict_description = f"RAGAS Score regressed by {abs(ragas_comparison.relative_change_percent):.1f}%"
        elif ragas_comparison.absolute_change == 0:
            verdict = "NO_CHANGE"
            verdict_description = "No measurable difference between runs"
        else:
            verdict = "MINOR_REGRESSION"
            verdict_description = f"RAGAS Score slightly regressed by {abs(ragas_comparison.relative_change_percent):.1f}%"
    else:
        verdict = "UNKNOWN"
        verdict_description = "Could not determine RAGAS score comparison"

    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "baseline_file": str(baseline_path),
        "experiment_file": str(experiment_path),
        "verdict": verdict,
        "verdict_description": verdict_description,
        "summary": {
            "metrics_improved": improvements,
            "metrics_regressed": regressions,
            "significant_improvements": significant_improvements,
            "significant_regressions": significant_regressions,
        },
        "metrics": [
            {
                "name": c.metric_name,
                "baseline": round(c.baseline_value, 4),
                "experiment": round(c.experiment_value, 4),
                "change": round(c.absolute_change, 4),
                "change_percent": round(c.relative_change_percent, 2),
                "improved": c.improved,
                "significant": c.significant,
            }
            for c in comparisons
        ],
        "baseline_summary": {
            "total_tests": baseline_stats.get("total_tests", 0),
            "successful_tests": baseline_stats.get("successful_tests", 0),
            "success_rate": baseline_stats.get("success_rate", 0),
        },
        "experiment_summary": {
            "total_tests": experiment_stats.get("total_tests", 0),
            "successful_tests": experiment_stats.get("successful_tests", 0),
            "success_rate": experiment_stats.get("success_rate", 0),
        },
    }


def print_comparison_report(analysis: dict):
    """Print a formatted comparison report to stdout."""
    print("=" * 70)
    print("A/B TEST COMPARISON REPORT")
    print("=" * 70)
    print(f"Baseline:    {analysis['baseline_file']}")
    print(f"Experiment:  {analysis['experiment_file']}")
    print("-" * 70)

    # Verdict
    verdict = analysis["verdict"]
    verdict_icon = {
        "SIGNIFICANT_IMPROVEMENT": "PASS",
        "MINOR_IMPROVEMENT": "PASS",
        "NO_CHANGE": "~",
        "MINOR_REGRESSION": "WARN",
        "SIGNIFICANT_REGRESSION": "FAIL",
        "UNKNOWN": "?",
    }.get(verdict, "?")

    print(f"\n[{verdict_icon}] VERDICT: {verdict}")
    print(f"    {analysis['verdict_description']}")

    # Metrics table
    print("\n" + "-" * 70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Experiment':>10} {'Change':>10} {'Status':>10}")
    print("-" * 70)

    for metric in analysis["metrics"]:
        name = metric["name"]
        baseline = f"{metric['baseline']:.4f}"
        experiment = f"{metric['experiment']:.4f}"

        change = metric["change"]
        change_pct = metric["change_percent"]
        if change > 0:
            change_str = f"+{change:.4f}"
            status = f"+{change_pct:.1f}%"
        elif change < 0:
            change_str = f"{change:.4f}"
            status = f"{change_pct:.1f}%"
        else:
            change_str = "0.0000"
            status = "0.0%"

        if metric["significant"]:
            if metric["improved"]:
                status = f"[UP] {status}"
            else:
                status = f"[DOWN] {status}"
        else:
            status = f"      {status}"

        print(f"{name:<20} {baseline:>10} {experiment:>10} {change_str:>10} {status:>10}")

    print("-" * 70)

    # Summary
    summary = analysis["summary"]
    print(f"\nSummary: {summary['metrics_improved']} improved, {summary['metrics_regressed']} regressed")
    print(f"         {summary['significant_improvements']} significant improvements, {summary['significant_regressions']} significant regressions")

    # Test counts
    b_summary = analysis["baseline_summary"]
    e_summary = analysis["experiment_summary"]
    print(f"\nBaseline:    {b_summary['successful_tests']}/{b_summary['total_tests']} tests ({b_summary['success_rate']:.1f}% success)")
    print(f"Experiment:  {e_summary['successful_tests']}/{e_summary['total_tests']} tests ({e_summary['success_rate']:.1f}% success)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAGAS evaluation results from two runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs experiment
  python lightrag/evaluation/compare_results.py baseline.json experiment.json

  # Save comparison to file
  python lightrag/evaluation/compare_results.py baseline.json experiment.json --output comparison.json

  # Compare with/without orphan connections
  python lightrag/evaluation/compare_results.py results_without_orphans.json results_with_orphans.json
        """,
    )

    parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline results JSON file",
    )

    parser.add_argument(
        "experiment",
        type=str,
        help="Path to experiment results JSON file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for comparison JSON (optional)",
    )

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    experiment_path = Path(args.experiment)

    # Validate files exist
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    if not experiment_path.exists():
        print(f"Error: Experiment file not found: {experiment_path}")
        sys.exit(1)

    # Run analysis
    analysis = analyze_results(baseline_path, experiment_path)

    # Print report
    print_comparison_report(analysis)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nComparison saved to: {output_path}")

    # Exit with status based on verdict
    if analysis["verdict"] in ("SIGNIFICANT_REGRESSION",):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
