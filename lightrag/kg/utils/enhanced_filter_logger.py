"""
Enhanced Filter Logging System

Provides specialized logging for the Enhanced Relationship Validation System
with session-based log files and detailed operational tracking.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading

# Thread-local storage for logger instances
_thread_local = threading.local()


class EnhancedFilterLogger:
    """
    Specialized logger for Enhanced Relationship Validation System.
    
    Creates session-based log files and provides structured logging
    for classification, filtering, and optimization events.
    """
    
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None, 
                 enable_console_logging: bool = False):
        """
        Initialize enhanced filter logger.
        
        Args:
            log_dir: Directory for log files (relative to project root)
            session_id: Optional session identifier
            enable_console_logging: Whether to log to console
        """
        self.log_dir = Path(log_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_console_logging = enable_console_logging
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific logger
        self.logger = self._setup_logger()
        
        # Track session metrics
        self.session_stats = {
            "start_time": datetime.now(),
            "filter_sessions": 0,
            "classifications": 0,
            "optimizations": 0,
            "errors": 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup session-specific logger with file and console handlers."""
        logger_name = f"enhanced_filter_{self.session_id}"
        logger = logging.getLogger(logger_name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | Enhanced Filter | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler - detailed logging
        log_file = self.log_dir / f"enhanced_filter_session_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Add file handler always
        logger.addHandler(file_handler)
        
        # Console handler - only if explicitly enabled
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Log session start
        logger.info(f"🚀 Enhanced Filter Logger initialized - Session: {self.session_id}")
        logger.info(f"📁 Log file: {log_file}")
        
        return logger
    
    def log_initialization(self, config: Dict[str, Any]):
        """Log system initialization with configuration details."""
        self.logger.info("=" * 80)
        self.logger.info("🧠 ENHANCED RELATIONSHIP VALIDATION SYSTEM INITIALIZATION")
        self.logger.info("=" * 80)
        
        # Log configuration
        enhanced_enabled = config.get("enable_enhanced_relationship_filter", False)
        log_classification = config.get("log_relationship_classification", False)
        track_performance = config.get("relationship_filter_performance_tracking", False)
        
        if enhanced_enabled:
            self.logger.info("✅ Enhanced Relationship Filter: ENABLED")
            self.logger.info(f"   📊 Performance Tracking: {'ENABLED' if track_performance else 'DISABLED'}")
            self.logger.info(f"   🔍 Classification Logging: {'ENABLED' if log_classification else 'DISABLED'}")
            
            # Log detailed configuration
            self.logger.debug("Enhanced Filter Configuration:")
            for key, value in config.items():
                if 'relationship' in key.lower() or 'filter' in key.lower():
                    self.logger.debug(f"   {key}: {value}")
        else:
            self.logger.info("❌ Enhanced Relationship Filter: DISABLED")
            self.logger.info("   Using basic relationship filtering only")
        
        self.logger.info("=" * 80)
    
    def log_filter_session_start(self, total_relationships: int, session_type: str = "standard"):
        """Log the start of a filter session."""
        self.session_stats["filter_sessions"] += 1
        session_num = self.session_stats["filter_sessions"]
        
        self.logger.info(f"🔄 Filter Session #{session_num} Started ({session_type})")
        self.logger.info(f"   📈 Total relationships to process: {total_relationships}")
        self.logger.debug(f"   ⏱️  Session start time: {datetime.now()}")
    
    def log_classification_result(self, relationship_type: str, src_entity: str, 
                                tgt_entity: str, classification: Dict[str, Any]):
        """Log individual relationship classification results."""
        self.session_stats["classifications"] += 1
        
        category = classification.get("category", "unknown")
        confidence = classification.get("confidence", 0.0)
        should_keep = classification.get("should_keep", False)
        threshold = classification.get("threshold", 0.0)
        
        # Different log levels based on result
        if should_keep:
            self.logger.debug(
                f"✅ KEEP: {src_entity} -[{relationship_type}]-> {tgt_entity} "
                f"| Category: {category} | Confidence: {confidence:.3f} | Threshold: {threshold:.3f}"
            )
        else:
            self.logger.info(
                f"❌ FILTER: {src_entity} -[{relationship_type}]-> {tgt_entity} "
                f"| Category: {category} | Confidence: {confidence:.3f} < {threshold:.3f}"
            )
        
        # Log additional classification details if available
        if "registry_match" in classification:
            self.logger.debug(
                f"   🔍 Registry match: '{classification['registry_match']}' "
                f"(confidence: {classification.get('registry_confidence', 0):.3f})"
            )
    
    def log_filter_session_end(self, filter_stats: Dict[str, Any], 
                             enhanced_stats: Optional[Dict[str, Any]] = None):
        """Log the completion of a filter session with detailed statistics."""
        session_num = self.session_stats["filter_sessions"]
        
        total_before = filter_stats.get("total_before", 0)
        total_after = filter_stats.get("total_after", 0)
        removed = total_before - total_after
        retention_rate = total_after / max(total_before, 1)
        
        self.logger.info(f"✅ Filter Session #{session_num} Completed")
        self.logger.info(f"   📊 Processed: {total_before} relationships")
        self.logger.info(f"   ✅ Kept: {total_after} relationships")
        self.logger.info(f"   ❌ Filtered: {removed} relationships")
        self.logger.info(f"   📈 Retention Rate: {retention_rate:.1%}")
        
        # Log enhanced statistics if available
        if enhanced_stats:
            self.logger.info("   🎯 Category Performance:")
            for category, stats in enhanced_stats.items():
                if stats.get("total", 0) > 0:
                    kept = stats.get("kept", 0)
                    total = stats.get("total", 0)
                    category_retention = kept / total
                    self.logger.info(f"      • {category}: {kept}/{total} kept ({category_retention:.1%})")
        
        # Log filter breakdown
        filter_breakdown = filter_stats.get("filter_breakdown", {})
        if any(count > 0 for count in filter_breakdown.values()):
            self.logger.info("   🔍 Filter Breakdown:")
            for filter_type, count in filter_breakdown.items():
                if count > 0:
                    percentage = count / max(removed, 1) * 100
                    self.logger.info(f"      • {filter_type}: {count} ({percentage:.1f}%)")
    
    def log_quality_assessment(self, assessment: str, recommendations: list = None):
        """Log quality assessment and recommendations."""
        self.logger.info(f"🎯 Quality Assessment: {assessment}")
        
        if recommendations:
            self.logger.info("💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):  # Limit to top 3
                self.logger.info(f"   {i}. {rec}")
    
    def log_metrics_collection(self, metrics_summary: Dict[str, Any]):
        """Log metrics collection summary."""
        self.logger.debug("📊 Metrics Collection Update:")
        self.logger.debug(f"   Session count: {metrics_summary.get('session_count', 0)}")
        self.logger.debug(f"   Overall retention: {metrics_summary.get('overall_retention_rate', 0):.1%}")
        
        if "classification_summary" in metrics_summary:
            summary = metrics_summary["classification_summary"]
            self.logger.debug(f"   Classifications: {summary.get('total_classifications', 0)}")
            self.logger.debug(f"   Unique types: {summary.get('unique_relationship_types', 0)}")
    
    def log_adaptive_optimization(self, optimization_results: Dict[str, Any]):
        """Log adaptive threshold optimization results."""
        self.session_stats["optimizations"] += 1
        
        optimizations = optimization_results.get("optimizations_made", [])
        
        if optimizations:
            self.logger.info(f"🔧 Adaptive Optimization Applied - {len(optimizations)} adjustments")
            for opt in optimizations:
                category = opt.get("category", "unknown")
                old_threshold = opt.get("old_threshold", 0)
                new_threshold = opt.get("new_threshold", 0)
                action = opt.get("action", "unknown")
                reason = opt.get("reason", "")
                
                change_indicator = "⬆️" if new_threshold > old_threshold else "⬇️"
                self.logger.info(
                    f"   {change_indicator} {category}: {old_threshold:.3f} → {new_threshold:.3f} ({action})"
                )
                self.logger.debug(f"      Reason: {reason}")
        else:
            self.logger.debug("🔧 Adaptive Optimization: No adjustments needed")
        
        # Log recommendations
        recommendations = optimization_results.get("recommendations", [])
        if recommendations:
            self.logger.info("💡 Optimization Recommendations:")
            for rec in recommendations[:2]:  # Limit to top 2
                self.logger.info(f"   • {rec}")
    
    def log_error(self, component: str, error: Exception, context: str = ""):
        """Log errors with context information."""
        self.session_stats["errors"] += 1
        
        self.logger.error(f"❌ Error in {component}: {str(error)}")
        if context:
            self.logger.error(f"   Context: {context}")
        self.logger.debug(f"   Error type: {type(error).__name__}")
        
        # Log fallback behavior
        self.logger.warning(f"🔄 Falling back to basic filtering for {component}")
    
    def log_session_summary(self):
        """Log session summary statistics."""
        duration = datetime.now() - self.session_stats["start_time"]
        
        self.logger.info("=" * 80)
        self.logger.info("📊 ENHANCED FILTER SESSION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  Session Duration: {duration}")
        self.logger.info(f"🔄 Filter Sessions: {self.session_stats['filter_sessions']}")
        self.logger.info(f"🎯 Classifications: {self.session_stats['classifications']}")
        self.logger.info(f"🔧 Optimizations: {self.session_stats['optimizations']}")
        self.logger.info(f"❌ Errors: {self.session_stats['errors']}")
        self.logger.info("=" * 80)


def get_enhanced_filter_logger(session_id: Optional[str] = None, 
                             enable_console_logging: bool = False) -> EnhancedFilterLogger:
    """
    Get or create thread-local enhanced filter logger.
    
    Args:
        session_id: Optional session identifier
        enable_console_logging: Whether to enable console logging
        
    Returns:
        EnhancedFilterLogger instance
    """
    if not hasattr(_thread_local, 'logger') or _thread_local.logger is None:
        _thread_local.logger = EnhancedFilterLogger(
            session_id=session_id, 
            enable_console_logging=enable_console_logging
        )
    
    return _thread_local.logger


def log_enhanced_filter_initialization(config: Dict[str, Any]):
    """Convenience function for initialization logging."""
    console_logging = config.get("enhanced_filter_console_logging", False)
    logger = get_enhanced_filter_logger(enable_console_logging=console_logging)
    logger.log_initialization(config)


def log_enhanced_filter_operation(operation_type: str, **kwargs):
    """Convenience function for operational logging."""
    logger = get_enhanced_filter_logger()
    
    if operation_type == "filter_session_start":
        logger.log_filter_session_start(kwargs.get("total_relationships", 0))
    elif operation_type == "classification":
        logger.log_classification_result(
            kwargs.get("relationship_type", ""),
            kwargs.get("src_entity", ""),
            kwargs.get("tgt_entity", ""),
            kwargs.get("classification", {})
        )
    elif operation_type == "filter_session_end":
        logger.log_filter_session_end(
            kwargs.get("filter_stats", {}),
            kwargs.get("enhanced_stats", {})
        )
    elif operation_type == "quality_assessment":
        logger.log_quality_assessment(
            kwargs.get("assessment", ""),
            kwargs.get("recommendations", [])
        )
    elif operation_type == "metrics_collection":
        logger.log_metrics_collection(kwargs.get("metrics_summary", {}))
    elif operation_type == "optimization":
        logger.log_adaptive_optimization(kwargs.get("optimization_results", {}))
    elif operation_type == "error":
        logger.log_error(
            kwargs.get("component", ""),
            kwargs.get("error", Exception("Unknown error")),
            kwargs.get("context", "")
        )


if __name__ == "__main__":
    # Test the enhanced filter logger
    test_config = {
        "enable_enhanced_relationship_filter": True,
        "log_relationship_classification": True,
        "relationship_filter_performance_tracking": True
    }
    
    # Test initialization logging
    log_enhanced_filter_initialization(test_config)
    
    # Test operational logging
    log_enhanced_filter_operation("filter_session_start", total_relationships=100)
    
    log_enhanced_filter_operation("classification",
        relationship_type="USES",
        src_entity="application",
        tgt_entity="redis",
        classification={
            "category": "technical_core",
            "confidence": 0.95,
            "threshold": 0.8,
            "should_keep": True
        }
    )
    
    log_enhanced_filter_operation("filter_session_end",
        filter_stats={"total_before": 100, "total_after": 85},
        enhanced_stats={"technical_core": {"total": 40, "kept": 38}}
    )
    
    logger = get_enhanced_filter_logger()
    logger.log_session_summary()
    
    print("✅ Enhanced filter logger test completed!")