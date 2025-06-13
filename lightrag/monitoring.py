"""
Performance Monitoring and Utility Logging Module for LightRAG
Part of Phase 4: Utility Logging implementation.

This module provides comprehensive monitoring for:
- Performance metrics and timing
- System health monitoring
- Processing statistics and analytics
- Debug logging and audit trails
- Real-time status tracking
"""

import time
import psutil
import threading
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import asyncio
from contextlib import asynccontextmanager, contextmanager
from . import utils

utils.setup_logger("lightrag.monitoring")
logger = logging.getLogger("lightrag.monitoring")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(
        self, success: bool = True, error_message: Optional[str] = None, **metadata
    ):
        """Mark the operation as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
        self.metadata.update(metadata)


@dataclass
class SystemHealth:
    """Container for system health metrics"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: Optional[float] = None
    active_threads: int = 0

    @classmethod
    def capture(cls) -> "SystemHealth":
        """Capture current system health metrics"""
        memory = psutil.virtual_memory()
        return cls(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            active_threads=threading.active_count(),
        )


@dataclass
class ProcessingStats:
    """Container for processing statistics"""

    entities_extracted: int = 0
    entities_validated: int = 0
    entities_failed: int = 0
    relationships_extracted: int = 0
    relationships_validated: int = 0
    relationships_failed: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0
    llm_calls_total: int = 0
    llm_calls_successful: int = 0
    llm_calls_failed: int = 0
    database_operations: int = 0
    database_failures: int = 0
    validation_errors: int = 0
    validation_warnings: int = 0

    def add_entity_extraction(self, extracted: int, validated: int, failed: int):
        """Add entity extraction statistics"""
        self.entities_extracted += extracted
        self.entities_validated += validated
        self.entities_failed += failed

    def add_relationship_extraction(self, extracted: int, validated: int, failed: int):
        """Add relationship extraction statistics"""
        self.relationships_extracted += extracted
        self.relationships_validated += validated
        self.relationships_failed += failed

    def add_chunk_processing(self, processed: int, failed: int):
        """Add chunk processing statistics"""
        self.chunks_processed += processed
        self.chunks_failed += failed

    def add_llm_call(self, success: bool):
        """Add LLM call statistics"""
        self.llm_calls_total += 1
        if success:
            self.llm_calls_successful += 1
        else:
            self.llm_calls_failed += 1

    def add_database_operation(self, success: bool):
        """Add database operation statistics"""
        self.database_operations += 1
        if not success:
            self.database_failures += 1

    def add_validation_results(self, errors: int, warnings: int):
        """Add validation statistics"""
        self.validation_errors += errors
        self.validation_warnings += warnings


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.operation_counts = defaultdict(int)
        self.operation_durations = defaultdict(list)
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()

    def start_operation(
        self, operation_name: str, operation_id: Optional[str] = None
    ) -> str:
        """Start tracking an operation"""
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        metric = PerformanceMetrics(
            operation_name=operation_name, start_time=time.time()
        )

        with self.lock:
            self.active_operations[operation_id] = metric

        return operation_id

    def complete_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        **metadata,
    ) -> Optional[PerformanceMetrics]:
        """Complete an operation and record metrics"""
        with self.lock:
            if operation_id not in self.active_operations:
                logger.warning(
                    f"Operation {operation_id} not found in active operations"
                )
                return None

            metric = self.active_operations.pop(operation_id)
            metric.complete(success, error_message, **metadata)

            # Update statistics
            self.operation_counts[metric.operation_name] += 1
            self.operation_durations[metric.operation_name].append(metric.duration)

            # Keep only recent durations to prevent memory growth
            if len(self.operation_durations[metric.operation_name]) > 100:
                self.operation_durations[metric.operation_name] = (
                    self.operation_durations[metric.operation_name][-50:]
                )

            self.metrics_history.append(metric)

            return metric

    @contextmanager
    def measure(self, operation_name: str, **metadata):
        """Context manager for measuring operation performance"""
        operation_id = self.start_operation(operation_name)
        start_time = time.time()

        try:
            yield operation_id
            self.complete_operation(operation_id, success=True, **metadata)
        except Exception as e:
            self.complete_operation(
                operation_id, success=False, error_message=str(e), **metadata
            )
            raise

    @asynccontextmanager
    async def measure_async(self, operation_name: str, **metadata):
        """Async context manager for measuring operation performance"""
        operation_id = self.start_operation(operation_name)

        try:
            yield operation_id
            self.complete_operation(operation_id, success=True, **metadata)
        except Exception as e:
            self.complete_operation(
                operation_id, success=False, error_message=str(e), **metadata
            )
            raise

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        with self.lock:
            durations = self.operation_durations.get(operation_name, [])
            if not durations:
                return {"count": 0}

            return {
                "count": self.operation_counts[operation_name],
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "recent_durations": durations[-10:],  # Last 10 operations
            }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations"""
        with self.lock:
            return {
                op: self.get_operation_stats(op) for op in self.operation_counts.keys()
            }


class SystemHealthMonitor:
    """Monitor system health and resource usage"""

    def __init__(self, sample_interval: int = 30):
        self.sample_interval = sample_interval
        self.health_history: deque = deque(maxlen=100)  # Keep last 100 samples
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System health monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                health = SystemHealth.capture()
                with self.lock:
                    self.health_history.append(health)

                # Log warnings for high resource usage
                if health.cpu_percent > 90:
                    logger.warning(f"High CPU usage: {health.cpu_percent:.1f}%")
                if health.memory_percent > 90:
                    logger.warning(f"High memory usage: {health.memory_percent:.1f}%")

                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}")
                time.sleep(self.sample_interval)

    def get_current_health(self) -> SystemHealth:
        """Get current system health"""
        return SystemHealth.capture()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health metrics"""
        with self.lock:
            if not self.health_history:
                return {}

            recent_health = list(self.health_history)[-10:]  # Last 10 samples

            return {
                "current": recent_health[-1].__dict__ if recent_health else {},
                "avg_cpu": sum(h.cpu_percent for h in recent_health)
                / len(recent_health),
                "avg_memory": sum(h.memory_percent for h in recent_health)
                / len(recent_health),
                "max_cpu": max(h.cpu_percent for h in recent_health),
                "max_memory": max(h.memory_percent for h in recent_health),
                "sample_count": len(self.health_history),
            }


class ProcessingMonitor:
    """Monitor processing pipeline statistics and status"""

    def __init__(self):
        self.stats = ProcessingStats()
        self.session_start = time.time()
        self.current_session_id = int(time.time())
        self.processing_status = {
            "current_stage": "idle",
            "current_file": None,
            "progress": 0.0,
            "total_files": 0,
            "completed_files": 0,
            "estimated_completion": None,
        }
        self.lock = threading.Lock()

    def start_session(self, session_id: Optional[str] = None):
        """Start a new processing session"""
        with self.lock:
            self.session_start = time.time()
            self.current_session_id = session_id or int(time.time())
            self.stats = ProcessingStats()
            self.processing_status = {
                "current_stage": "starting",
                "current_file": None,
                "progress": 0.0,
                "total_files": 0,
                "completed_files": 0,
                "estimated_completion": None,
            }

        logger.info(f"Started processing session: {self.current_session_id}")

    def update_status(
        self,
        stage: str,
        current_file: Optional[str] = None,
        progress: Optional[float] = None,
        **kwargs,
    ):
        """Update processing status"""
        with self.lock:
            self.processing_status["current_stage"] = stage
            if current_file is not None:
                self.processing_status["current_file"] = current_file
            if progress is not None:
                self.processing_status["progress"] = progress

            # Update any additional status fields
            self.processing_status.update(kwargs)

            # Estimate completion time
            if progress and progress > 0:
                elapsed = time.time() - self.session_start
                estimated_total = elapsed / progress
                estimated_remaining = estimated_total - elapsed
                self.processing_status["estimated_completion"] = (
                    time.time() + estimated_remaining
                )

    def record_extraction_results(
        self,
        entities_extracted: int,
        entities_validated: int,
        entities_failed: int,
        relationships_extracted: int,
        relationships_validated: int,
        relationships_failed: int,
    ):
        """Record extraction results"""
        with self.lock:
            self.stats.add_entity_extraction(
                entities_extracted, entities_validated, entities_failed
            )
            self.stats.add_relationship_extraction(
                relationships_extracted, relationships_validated, relationships_failed
            )

    def record_chunk_processing(self, processed: int, failed: int):
        """Record chunk processing results"""
        with self.lock:
            self.stats.add_chunk_processing(processed, failed)

    def record_llm_call(self, success: bool):
        """Record LLM call result"""
        with self.lock:
            self.stats.add_llm_call(success)

    def record_database_operation(self, success: bool):
        """Record database operation result"""
        with self.lock:
            self.stats.add_database_operation(success)

    def record_validation_results(self, errors: int, warnings: int):
        """Record validation results"""
        with self.lock:
            self.stats.add_validation_results(errors, warnings)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        with self.lock:
            duration = time.time() - self.session_start

            return {
                "session_id": self.current_session_id,
                "duration_seconds": duration,
                "status": dict(self.processing_status),
                "statistics": {
                    "entities": {
                        "extracted": self.stats.entities_extracted,
                        "validated": self.stats.entities_validated,
                        "failed": self.stats.entities_failed,
                        "success_rate": self.stats.entities_validated
                        / max(self.stats.entities_extracted, 1)
                        * 100,
                    },
                    "relationships": {
                        "extracted": self.stats.relationships_extracted,
                        "validated": self.stats.relationships_validated,
                        "failed": self.stats.relationships_failed,
                        "success_rate": self.stats.relationships_validated
                        / max(self.stats.relationships_extracted, 1)
                        * 100,
                    },
                    "chunks": {
                        "processed": self.stats.chunks_processed,
                        "failed": self.stats.chunks_failed,
                        "success_rate": (
                            self.stats.chunks_processed - self.stats.chunks_failed
                        )
                        / max(self.stats.chunks_processed, 1)
                        * 100,
                    },
                    "llm_calls": {
                        "total": self.stats.llm_calls_total,
                        "successful": self.stats.llm_calls_successful,
                        "failed": self.stats.llm_calls_failed,
                        "success_rate": self.stats.llm_calls_successful
                        / max(self.stats.llm_calls_total, 1)
                        * 100,
                    },
                    "database": {
                        "operations": self.stats.database_operations,
                        "failures": self.stats.database_failures,
                        "success_rate": (
                            self.stats.database_operations
                            - self.stats.database_failures
                        )
                        / max(self.stats.database_operations, 1)
                        * 100,
                    },
                    "validation": {
                        "errors": self.stats.validation_errors,
                        "warnings": self.stats.validation_warnings,
                    },
                },
            }


class EnhancedLogger:
    """Enhanced logging with structured data and context"""

    def __init__(self, name: str):
        utils.setup_logger(name)
        self.logger = logging.getLogger(name)
        self.context_stack = []

    def push_context(self, context: Dict[str, Any]):
        """Push a context for subsequent log messages"""
        self.context_stack.append(context)

    def pop_context(self):
        """Pop the most recent context"""
        if self.context_stack:
            self.context_stack.pop()

    @contextmanager
    def context(self, **context_data):
        """Context manager for temporary context"""
        self.push_context(context_data)
        try:
            yield
        finally:
            self.pop_context()

    def _format_message(self, message: str, extra_data: Optional[Dict] = None) -> str:
        """Format message with context and extra data"""
        context_str = ""
        if self.context_stack:
            combined_context = {}
            for context in self.context_stack:
                combined_context.update(context)
            context_str = (
                f" [{', '.join(f'{k}={v}' for k, v in combined_context.items())}]"
            )

        extra_str = ""
        if extra_data:
            extra_str = f" {extra_data}"

        return f"{message}{context_str}{extra_str}"

    def debug(self, message: str, **extra_data):
        """Log debug message with context"""
        self.logger.debug(self._format_message(message, extra_data))

    def info(self, message: str, **extra_data):
        """Log info message with context"""
        self.logger.info(self._format_message(message, extra_data))

    def warning(self, message: str, **extra_data):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message, extra_data))

    def error(self, message: str, **extra_data):
        """Log error message with context"""
        self.logger.error(self._format_message(message, extra_data))


# Global monitoring instances
global_performance_monitor = PerformanceMonitor()
global_health_monitor = SystemHealthMonitor()
global_processing_monitor = ProcessingMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return global_performance_monitor


def get_health_monitor() -> SystemHealthMonitor:
    """Get the global health monitor instance"""
    return global_health_monitor


def get_processing_monitor() -> ProcessingMonitor:
    """Get the global processing monitor instance"""
    return global_processing_monitor


def get_enhanced_logger(name: str) -> EnhancedLogger:
    """Get an enhanced logger instance"""
    return EnhancedLogger(name)


def start_system_monitoring():
    """Start all system monitoring components"""
    global_health_monitor.start_monitoring()
    logger.info("System monitoring started")


def stop_system_monitoring():
    """Stop all system monitoring components"""
    global_health_monitor.stop_monitoring()
    logger.info("System monitoring stopped")


def get_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    return {
        "timestamp": datetime.now().isoformat(),
        "performance": global_performance_monitor.get_all_stats(),
        "health": global_health_monitor.get_health_summary(),
        "processing": global_processing_monitor.get_session_summary(),
    }
