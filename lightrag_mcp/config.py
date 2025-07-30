"""
Configuration management for LightRAG MCP Server.

Handles environment variables, configuration validation, 
and default settings for the MCP server.
"""

import os
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightrag-mcp.config")


@dataclass
class LightRAGMCPConfig:
    """Configuration class for LightRAG MCP Server."""
    
    # Connection settings
    lightrag_api_url: str = "http://localhost:9621"
    lightrag_api_key: Optional[str] = None
    lightrag_working_dir: Optional[str] = None
    
    # MCP server settings
    mcp_server_name: str = "lightrag-mcp"
    mcp_server_version: str = "1.0.0"
    mcp_description: str = "LightRAG Model Context Protocol Server"
    
    # Feature flags
    enable_direct_mode: bool = True  # Use library directly vs API
    enable_streaming: bool = True
    enable_graph_modification: bool = True
    enable_document_upload: bool = True
    
    # Security settings
    require_auth: bool = False
    allowed_file_types: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".json"
    ])
    max_file_size_mb: int = 100
    max_documents_per_batch: int = 10
    
    # Performance settings
    default_query_timeout: int = 60
    max_concurrent_queries: int = 5
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # Query defaults
    default_query_mode: str = "hybrid"
    default_top_k: int = 40
    default_chunk_top_k: int = 10
    default_cosine_threshold: float = 0.2
    default_max_tokens: int = 30000
    
    # HTTP client settings
    http_timeout: int = 60
    http_max_connections: int = 10
    http_max_keepalive: int = 5
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_debug_logging: bool = False
    
    @classmethod
    def from_env(cls) -> "LightRAGMCPConfig":
        """Create configuration from environment variables."""
        
        def get_env_bool(key: str, default: bool) -> bool:
            """Get boolean environment variable."""
            value = os.getenv(key, "").lower()
            if value in ("true", "1", "yes", "on"):
                return True
            elif value in ("false", "0", "no", "off"):
                return False
            return default
        
        def get_env_int(key: str, default: int) -> int:
            """Get integer environment variable."""
            try:
                return int(os.getenv(key, str(default)))
            except (ValueError, TypeError):
                logger.warning(f"Invalid integer value for {key}, using default: {default}")
                return default
        
        def get_env_float(key: str, default: float) -> float:
            """Get float environment variable."""
            try:
                return float(os.getenv(key, str(default)))
            except (ValueError, TypeError):
                logger.warning(f"Invalid float value for {key}, using default: {default}")
                return default
        
        def get_env_list(key: str, default: List[str]) -> List[str]:
            """Get list environment variable (comma-separated)."""
            value = os.getenv(key, "")
            if not value:
                return default
            return [item.strip() for item in value.split(",") if item.strip()]
        
        config = cls(
            # Connection settings
            lightrag_api_url=os.getenv("LIGHTRAG_API_URL", cls.lightrag_api_url),
            lightrag_api_key=os.getenv("LIGHTRAG_API_KEY"),
            lightrag_working_dir=os.getenv("LIGHTRAG_WORKING_DIR"),
            
            # MCP server settings
            mcp_server_name=os.getenv("MCP_SERVER_NAME", cls.mcp_server_name),
            mcp_server_version=os.getenv("MCP_SERVER_VERSION", cls.mcp_server_version),
            mcp_description=os.getenv("MCP_DESCRIPTION", cls.mcp_description),
            
            # Feature flags
            enable_direct_mode=get_env_bool("MCP_ENABLE_DIRECT_MODE", cls.enable_direct_mode),
            enable_streaming=get_env_bool("MCP_ENABLE_STREAMING", cls.enable_streaming),
            enable_graph_modification=get_env_bool("MCP_ENABLE_GRAPH_MODIFICATION", cls.enable_graph_modification),
            enable_document_upload=get_env_bool("MCP_ENABLE_DOCUMENT_UPLOAD", cls.enable_document_upload),
            
            # Security settings
            require_auth=get_env_bool("MCP_REQUIRE_AUTH", cls.require_auth),
            allowed_file_types=get_env_list("MCP_ALLOWED_FILE_TYPES", [
                ".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".json"
            ]),
            max_file_size_mb=get_env_int("MCP_MAX_FILE_SIZE_MB", cls.max_file_size_mb),
            max_documents_per_batch=get_env_int("MCP_MAX_DOCUMENTS_PER_BATCH", cls.max_documents_per_batch),
            
            # Performance settings
            default_query_timeout=get_env_int("MCP_DEFAULT_QUERY_TIMEOUT", cls.default_query_timeout),
            max_concurrent_queries=get_env_int("MCP_MAX_CONCURRENT_QUERIES", cls.max_concurrent_queries),
            cache_enabled=get_env_bool("MCP_CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=get_env_int("MCP_CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            
            # Query defaults
            default_query_mode=os.getenv("MCP_DEFAULT_QUERY_MODE", cls.default_query_mode),
            default_top_k=get_env_int("MCP_DEFAULT_TOP_K", cls.default_top_k),
            default_chunk_top_k=get_env_int("MCP_DEFAULT_CHUNK_TOP_K", cls.default_chunk_top_k),
            default_cosine_threshold=get_env_float("MCP_DEFAULT_COSINE_THRESHOLD", cls.default_cosine_threshold),
            default_max_tokens=get_env_int("MCP_DEFAULT_MAX_TOKENS", cls.default_max_tokens),
            
            # HTTP client settings
            http_timeout=get_env_int("MCP_HTTP_TIMEOUT", cls.http_timeout),
            http_max_connections=get_env_int("MCP_HTTP_MAX_CONNECTIONS", cls.http_max_connections),
            http_max_keepalive=get_env_int("MCP_HTTP_MAX_KEEPALIVE", cls.http_max_keepalive),
            
            # Logging settings
            log_level=os.getenv("MCP_LOG_LEVEL", cls.log_level).upper(),
            log_format=os.getenv("MCP_LOG_FORMAT", cls.log_format),
            enable_debug_logging=get_env_bool("MCP_ENABLE_DEBUG_LOGGING", cls.enable_debug_logging)
        )
        
        # Validate configuration
        config.validate()
        
        return config
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate URL format
        if not self.lightrag_api_url.startswith(("http://", "https://")):
            errors.append(f"Invalid API URL format: {self.lightrag_api_url}")
        
        # Validate query mode
        valid_modes = ["naive", "local", "global", "hybrid", "mix", "bypass"]
        if self.default_query_mode not in valid_modes:
            errors.append(f"Invalid default query mode: {self.default_query_mode}")
        
        # Validate file types
        if not all(ft.startswith(".") for ft in self.allowed_file_types):
            errors.append("All file types must start with a dot (.)")
        
        # Validate numeric ranges
        if self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")
        
        if self.default_top_k <= 0:
            errors.append("default_top_k must be positive")
        
        if self.default_chunk_top_k <= 0:
            errors.append("default_chunk_top_k must be positive")
        
        if not 0 <= self.default_cosine_threshold <= 1:
            errors.append("default_cosine_threshold must be between 0 and 1")
        
        if self.default_max_tokens <= 0:
            errors.append("default_max_tokens must be positive")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.log_level}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        numeric_level = getattr(logging, self.log_level, logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format=self.log_format,
            force=True  # Override existing configuration
        )
        
        # Set specific logger levels
        logger.setLevel(numeric_level)
        
        if self.enable_debug_logging:
            logging.getLogger("httpx").setLevel(logging.DEBUG)
            logging.getLogger("lightrag-mcp").setLevel(logging.DEBUG)
        else:
            # Reduce noise from HTTP libraries
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Setup logging first
        self.setup_logging()
        
        # Log configuration summary
        logger.info(f"LightRAG MCP Server Configuration:")
        logger.info(f"  API URL: {self.lightrag_api_url}")
        logger.info(f"  Server Name: {self.mcp_server_name}")
        logger.info(f"  Default Query Mode: {self.default_query_mode}")
        logger.info(f"  Features: streaming={self.enable_streaming}, "
                   f"graph_mod={self.enable_graph_modification}, "
                   f"doc_upload={self.enable_document_upload}")
        logger.info(f"  Security: auth_required={self.require_auth}, "
                   f"max_file_size={self.max_file_size_mb}MB")
        
        if self.lightrag_api_key:
            logger.info("  Authentication: API key configured")
        else:
            logger.info("  Authentication: No API key configured")


# Global configuration instance
config = LightRAGMCPConfig.from_env()


def get_config() -> LightRAGMCPConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> LightRAGMCPConfig:
    """Reload configuration from environment variables."""
    global config
    config = LightRAGMCPConfig.from_env()
    return config