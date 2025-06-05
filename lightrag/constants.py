"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for environment variables
DEFAULT_MAX_TOKEN_SUMMARY = 500
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 6
DEFAULT_WOKERS = 2
DEFAULT_TIMEOUT = 150

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename

# Chunk post-processing configuration defaults
DEFAULT_ENABLE_CHUNK_POST_PROCESSING = False  # Disabled by default for safety
DEFAULT_CHUNK_VALIDATION_BATCH_SIZE = 50  # Max relationships per chunk
DEFAULT_CHUNK_VALIDATION_TIMEOUT = 30  # Timeout in seconds per chunk
DEFAULT_LOG_VALIDATION_CHANGES = False  # Disable detailed logging by default

# Enhanced relationship quality filter configuration defaults
DEFAULT_ENABLE_ENHANCED_RELATIONSHIP_FILTER = True  # Enable type-specific filtering
DEFAULT_LOG_RELATIONSHIP_CLASSIFICATION = False  # Log detailed classification results
DEFAULT_RELATIONSHIP_FILTER_PERFORMANCE_TRACKING = True  # Track filter performance metrics
DEFAULT_ENHANCED_FILTER_CONSOLE_LOGGING = False  # Enable console logging for enhanced filter
DEFAULT_ENHANCED_FILTER_MONITORING_MODE = False  # Only classify and log, don't actually filter
