#!/usr/bin/env python3
"""
LightRAG System Status Check Utility
Part of Phase 4: Utility Logging implementation.

This script provides comprehensive system status checking and monitoring data access.
Use this to check performance metrics, system health, processing statistics, and more.
"""

import json
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any
import argparse

from .monitoring import (
    get_comprehensive_status,
    get_health_monitor,
    start_system_monitoring,
)
from . import utils

utils.setup_logger("lightrag.status_check")
logger = logging.getLogger("lightrag.status_check")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(bytes_value: float) -> str:
    """Format bytes in human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def print_system_health(health_data: Dict[str, Any]):
    """Print system health information"""
    print("\n🏥 SYSTEM HEALTH")
    print("=" * 50)

    if not health_data:
        print("❌ No health data available")
        return

    current = health_data.get("current", {})
    if current:
        cpu = current.get("cpu_percent", 0)
        memory = current.get("memory_percent", 0)
        memory_used = current.get("memory_used_mb", 0)
        memory_available = current.get("memory_available_mb", 0)
        threads = current.get("active_threads", 0)

        # Health status indicators
        cpu_status = "🔴" if cpu > 90 else "🟡" if cpu > 70 else "🟢"
        memory_status = "🔴" if memory > 90 else "🟡" if memory > 70 else "🟢"

        print(f"{cpu_status} CPU Usage: {cpu:.1f}%")
        print(
            f"{memory_status} Memory Usage: {memory:.1f}% ({format_bytes(memory_used * 1024 * 1024)} used)"
        )
        print(f"💾 Memory Available: {format_bytes(memory_available * 1024 * 1024)}")
        print(f"🧵 Active Threads: {threads}")

    # Average metrics
    avg_cpu = health_data.get("avg_cpu")
    avg_memory = health_data.get("avg_memory")
    max_cpu = health_data.get("max_cpu")
    max_memory = health_data.get("max_memory")

    if avg_cpu is not None:
        print("\n📊 Recent Averages:")
        print(f"   CPU: {avg_cpu:.1f}% (max: {max_cpu:.1f}%)")
        print(f"   Memory: {avg_memory:.1f}% (max: {max_memory:.1f}%)")

    sample_count = health_data.get("sample_count", 0)
    print(f"📈 Samples Collected: {sample_count}")


def print_performance_stats(perf_data: Dict[str, Any]):
    """Print performance statistics"""
    print("\n⚡ PERFORMANCE METRICS")
    print("=" * 50)

    if not perf_data:
        print("❌ No performance data available")
        return

    # Sort operations by count (most frequent first)
    sorted_ops = sorted(
        perf_data.items(), key=lambda x: x[1].get("count", 0), reverse=True
    )

    for op_name, stats in sorted_ops:
        count = stats.get("count", 0)
        if count == 0:
            continue

        avg_duration = stats.get("avg_duration", 0)
        min_duration = stats.get("min_duration", 0)
        max_duration = stats.get("max_duration", 0)

        print(f"\n🔧 {op_name}")
        print(f"   Count: {count:,}")
        print(f"   Avg Duration: {format_duration(avg_duration)}")
        print(
            f"   Range: {format_duration(min_duration)} - {format_duration(max_duration)}"
        )

        # Performance rating
        if avg_duration < 0.1:
            rating = "🟢 Excellent"
        elif avg_duration < 1.0:
            rating = "🟡 Good"
        elif avg_duration < 5.0:
            rating = "🟠 Slow"
        else:
            rating = "🔴 Very Slow"
        print(f"   Performance: {rating}")


def print_processing_stats(proc_data: Dict[str, Any]):
    """Print processing statistics"""
    print("\n📊 PROCESSING STATISTICS")
    print("=" * 50)

    if not proc_data:
        print("❌ No processing data available")
        return

    session_id = proc_data.get("session_id")
    duration = proc_data.get("duration_seconds", 0)
    status = proc_data.get("status", {})
    stats = proc_data.get("statistics", {})

    print(f"🆔 Session ID: {session_id}")
    print(f"⏱️ Duration: {format_duration(duration)}")

    # Current status
    current_stage = status.get("current_stage", "unknown")
    current_file = status.get("current_file")
    progress = status.get("progress", 0) * 100
    completed_files = status.get("completed_files", 0)
    total_files = status.get("total_files", 0)

    print("\n📍 Current Status:")
    print(f"   Stage: {current_stage}")
    if current_file:
        print(f"   File: {current_file}")
    if total_files > 0:
        print(f"   Progress: {progress:.1f}% ({completed_files}/{total_files} files)")

    # Processing statistics
    entities = stats.get("entities", {})
    relationships = stats.get("relationships", {})
    chunks = stats.get("chunks", {})
    llm_calls = stats.get("llm_calls", {})
    database = stats.get("database", {})
    validation = stats.get("validation", {})

    print("\n📈 Extraction Results:")
    if entities:
        print(
            f"   🔖 Entities: {entities.get('validated', 0):,} validated "
            f"({entities.get('success_rate', 0):.1f}% success rate)"
        )
    if relationships:
        print(
            f"   🔗 Relationships: {relationships.get('validated', 0):,} validated "
            f"({relationships.get('success_rate', 0):.1f}% success rate)"
        )
    if chunks:
        print(
            f"   📄 Chunks: {chunks.get('processed', 0):,} processed "
            f"({chunks.get('success_rate', 0):.1f}% success rate)"
        )

    print("\n🤖 LLM Statistics:")
    if llm_calls:
        total_llm = llm_calls.get("total", 0)
        successful_llm = llm_calls.get("successful", 0)
        success_rate = llm_calls.get("success_rate", 0)
        print(f"   Total Calls: {total_llm:,}")
        print(f"   Successful: {successful_llm:,} ({success_rate:.1f}%)")

    print("\n💾 Database Operations:")
    if database:
        total_db = database.get("operations", 0)
        failures = database.get("failures", 0)
        success_rate = database.get("success_rate", 0)
        print(f"   Total Operations: {total_db:,}")
        print(f"   Failures: {failures:,} ({success_rate:.1f}% success rate)")

    if validation:
        errors = validation.get("errors", 0)
        warnings = validation.get("warnings", 0)
        print("\n⚠️ Validation Issues:")
        print(f"   Errors: {errors:,}")
        print(f"   Warnings: {warnings:,}")


def print_status_summary():
    """Print a comprehensive status summary"""
    print("🚀 LIGHTRAG SYSTEM STATUS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Get comprehensive status
        status_data = get_comprehensive_status()

        # Print each section
        print_system_health(status_data.get("health", {}))
        print_performance_stats(status_data.get("performance", {}))
        print_processing_stats(status_data.get("processing", {}))

        print("\n✅ Status check completed successfully!")

    except Exception as e:
        print(f"\n❌ Error getting status: {str(e)}")
        logger.error(f"Status check failed: {str(e)}")


def export_status_json(output_file: str):
    """Export status data to JSON file"""
    try:
        status_data = get_comprehensive_status()

        with open(output_file, "w") as f:
            json.dump(status_data, f, indent=2, default=str)

        print(f"✅ Status data exported to: {output_file}")

    except Exception as e:
        print(f"❌ Error exporting status: {str(e)}")
        logger.error(f"Status export failed: {str(e)}")


def monitor_system(duration: int, interval: int = 10):
    """Monitor system for a specified duration"""
    print(f"🔍 Monitoring system for {duration} seconds (interval: {interval}s)")
    print("Press Ctrl+C to stop early")

    # Start monitoring
    start_system_monitoring()

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")

            # Get current health
            health_monitor = get_health_monitor()
            current_health = health_monitor.get_current_health()

            cpu_status = (
                "🔴"
                if current_health.cpu_percent > 90
                else "🟡" if current_health.cpu_percent > 70 else "🟢"
            )
            memory_status = (
                "🔴"
                if current_health.memory_percent > 90
                else "🟡" if current_health.memory_percent > 70 else "🟢"
            )

            print(f"{cpu_status} CPU: {current_health.cpu_percent:.1f}%")
            print(f"{memory_status} Memory: {current_health.memory_percent:.1f}%")
            print(f"🧵 Threads: {current_health.active_threads}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")


def main():
    """Main entry point for status check utility"""
    parser = argparse.ArgumentParser(description="LightRAG System Status Check Utility")
    parser.add_argument("--json", "-j", help="Export status to JSON file")
    parser.add_argument(
        "--monitor", "-m", type=int, help="Monitor system for N seconds"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=10, help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (errors only)"
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logger.setLevel("ERROR")

    try:
        if args.json:
            export_status_json(args.json)
        elif args.monitor:
            monitor_system(args.monitor, args.interval)
        else:
            print_status_summary()

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
