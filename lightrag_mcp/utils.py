"""
Utility functions for LightRAG MCP server.

Provides common utilities for error handling, correlation IDs,
validation, and other shared functionality.
"""

import hashlib
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Pattern
from datetime import datetime, timezone


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())


def generate_cache_key(data: Dict[str, Any]) -> str:
    """Generate MD5 cache key from data dictionary."""
    # Sort keys for consistent hash
    sorted_data = json.dumps(data, sort_keys=True)
    return hashlib.md5(sorted_data.encode()).hexdigest()


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime to ISO string with timezone."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()


def parse_time_range(time_range: str) -> tuple[datetime, datetime]:
    """Parse time range string to datetime tuple."""
    now = datetime.now(timezone.utc)
    
    # Parse patterns like "1h", "24h", "7d", "30d"
    pattern = re.compile(r'^(\d+)([hd])$')
    match = pattern.match(time_range.lower())
    
    if not match:
        raise ValueError(f"Invalid time range format: {time_range}")
    
    value, unit = int(match.group(1)), match.group(2)
    
    if unit == 'h':
        from datetime import timedelta
        delta = timedelta(hours=value)
    elif unit == 'd':
        from datetime import timedelta
        delta = timedelta(days=value)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
    
    start_time = now - delta
    return start_time, now


class MCPError(Exception):
    """Standardized MCP error with error codes and details."""
    
    def __init__(self, error_code: str, message: str, details: Optional[Dict[str, Any]] = None, 
                 suggested_action: Optional[str] = None):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.suggested_action = suggested_action
        self.correlation_id = generate_correlation_id()
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "error_code": self.error_code,
            "error_message": self.message,
            "correlation_id": self.correlation_id
        }
        
        if self.details:
            result["error_details"] = self.details
        
        if self.suggested_action:
            result["suggested_action"] = self.suggested_action
        
        return result


# Standard error code mappings
ERROR_CODE_MESSAGES = {
    # General errors
    "INVALID_PARAMETER": "Invalid parameter value provided",
    "MISSING_PARAMETER": "Required parameter not provided",
    "UNAUTHORIZED": "Authentication required or invalid",
    "FORBIDDEN": "Operation not permitted",
    "RATE_LIMITED": "Rate limit exceeded",
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
    "INTERNAL_ERROR": "Internal server error",
    
    # LightRAG specific errors
    "LIGHTRAG_UNAVAILABLE": "LightRAG service is not available",
    "PROCESSING_TIMEOUT": "Operation exceeded timeout limit",
    "INVALID_QUERY_MODE": "Invalid query mode specified",
    "DOCUMENT_NOT_FOUND": "Document not found",
    "ENTITY_NOT_FOUND": "Entity not found",
    "GRAPH_ACCESS_ERROR": "Knowledge graph access error",
    "STORAGE_ERROR": "Storage backend error",
    "CONFIGURATION_ERROR": "Configuration error",
    
    # Resource errors
    "RESOURCE_LIMIT_EXCEEDED": "Resource limit exceeded",
    "QUOTA_EXCEEDED": "Usage quota exceeded",
    "FILE_TOO_LARGE": "File exceeds size limit",
    "UNSUPPORTED_FORMAT": "Unsupported file format",
    "PROCESSING_FAILED": "Processing operation failed",
    "FILE_NOT_FOUND": "File not found",
    "NOT_IMPLEMENTED": "Feature not implemented",
    "API_ERROR": "API request failed",
    "NOT_FOUND": "Resource not found",
    "QUERY_FAILED": "Query execution failed"
}


def get_error_message(error_code: str) -> str:
    """Get standard error message for error code."""
    return ERROR_CODE_MESSAGES.get(error_code, "Unknown error")


def get_suggested_action(error_code: str) -> Optional[str]:
    """Get suggested action for error code."""
    suggestions = {
        "LIGHTRAG_UNAVAILABLE": "Check if LightRAG server is running and accessible",
        "UNAUTHORIZED": "Verify API key configuration",
        "FILE_NOT_FOUND": "Check file path and permissions",
        "FILE_TOO_LARGE": "Reduce file size or increase limit",
        "UNSUPPORTED_FORMAT": "Use supported file types: .txt, .md, .pdf, .docx, etc.",
        "INVALID_QUERY_MODE": "Use valid query modes: naive, local, global, hybrid, mix, bypass",
        "PROCESSING_TIMEOUT": "Reduce query complexity or increase timeout",
        "RATE_LIMITED": "Reduce request frequency",
        "QUOTA_EXCEEDED": "Check usage limits and quotas"
    }
    
    return suggestions.get(error_code)


class Validator:
    """Input validation utilities."""
    
    # Validation patterns
    ENTITY_ID_PATTERN: Pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
    DOCUMENT_ID_PATTERN: Pattern = re.compile(r"^doc_[a-zA-Z0-9_-]+$")
    
    # Limits
    QUERY_MAX_LENGTH = 10000
    TITLE_MAX_LENGTH = 500
    FILENAME_MAX_LENGTH = 255
    MAX_BATCH_SIZE = 50
    
    # File type mappings
    SUPPORTED_FILE_TYPES = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".html": "text/html",
        ".json": "application/json"
    }
    
    @staticmethod
    def validate_query(query: str) -> None:
        """Validate query string."""
        if not query or not query.strip():
            raise MCPError("INVALID_PARAMETER", "Query cannot be empty")
        
        if len(query) > Validator.QUERY_MAX_LENGTH:
            raise MCPError("INVALID_PARAMETER", 
                         f"Query too long: {len(query)} > {Validator.QUERY_MAX_LENGTH}")
    
    @staticmethod
    def validate_query_mode(mode: str) -> None:
        """Validate query mode."""
        valid_modes = ["naive", "local", "global", "hybrid", "mix", "bypass"]
        if mode not in valid_modes:
            raise MCPError("INVALID_QUERY_MODE", 
                         f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
    
    @staticmethod
    def validate_entity_id(entity_id: str) -> None:
        """Validate entity ID format."""
        if not Validator.ENTITY_ID_PATTERN.match(entity_id):
            raise MCPError("INVALID_PARAMETER", 
                         f"Invalid entity ID format: {entity_id}")
    
    @staticmethod
    def validate_document_id(document_id: str) -> None:
        """Validate document ID format."""
        if not Validator.DOCUMENT_ID_PATTERN.match(document_id):
            raise MCPError("INVALID_PARAMETER", 
                         f"Invalid document ID format: {document_id}")
    
    @staticmethod
    def validate_file_type(file_path: str, allowed_types: List[str]) -> None:
        """Validate file type."""
        from pathlib import Path
        
        path = Path(file_path)
        file_ext = path.suffix.lower()
        
        if file_ext not in allowed_types:
            raise MCPError("UNSUPPORTED_FORMAT", 
                         f"File type {file_ext} not supported. Allowed: {allowed_types}")
    
    @staticmethod
    def validate_limit_offset(limit: int, offset: int, max_limit: int = 200) -> None:
        """Validate pagination parameters."""
        if limit <= 0:
            raise MCPError("INVALID_PARAMETER", "Limit must be positive")
        
        if limit > max_limit:
            raise MCPError("INVALID_PARAMETER", f"Limit {limit} exceeds maximum {max_limit}")
        
        if offset < 0:
            raise MCPError("INVALID_PARAMETER", "Offset must be non-negative")
    
    @staticmethod
    def validate_time_range(time_range: str) -> None:
        """Validate time range format."""
        try:
            parse_time_range(time_range)
        except ValueError as e:
            raise MCPError("INVALID_PARAMETER", str(e))


class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._cache: Dict[str, tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        
        value, expires_at = self._cache[key]
        
        if time.time() > expires_at:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        self._cache[key] = (value, expires_at)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def size(self) -> int:
        """Get number of cache entries."""
        # Clean expired entries
        current_time = time.time()
        expired_keys = [
            key for key, (_, expires_at) in self._cache.items()
            if current_time > expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(self._cache)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove/replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(sanitized) > Validator.FILENAME_MAX_LENGTH:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = Validator.FILENAME_MAX_LENGTH - len(ext) - 1
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')
    
    return sanitized


def format_bytes(bytes_count: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result.setdefault('metadata', {})['execution_time'] = execution_time
        
        return result
    
    return wrapper