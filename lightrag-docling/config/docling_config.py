"""
Configuration management for the Docling service.
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path


class DoclingServiceSettings(BaseSettings):
    """Docling service configuration settings."""
    
    # Service settings
    host: str = Field(default="0.0.0.0", description="Service bind host")
    port: int = Field(default=8080, description="Service port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Service identification
    service_name: str = Field(default="lightrag-docling", description="Service name")
    service_version: str = Field(default="1.0.0", description="Service version")
    
    # Processing limits
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    max_batch_size: int = Field(default=10, description="Maximum batch size")
    request_timeout_seconds: int = Field(default=300, description="Request timeout")
    
    # Default docling configuration
    default_export_format: str = Field(default="markdown", description="Default export format")
    default_max_workers: int = Field(default=2, description="Default docling workers")
    default_enable_ocr: bool = Field(default=True, description="Default OCR setting")
    default_enable_table_structure: bool = Field(default=True, description="Default table structure")
    default_enable_figures: bool = Field(default=True, description="Default figure extraction")
    default_enable_cache: bool = Field(default=True, description="Default caching")
    
    # Model settings
    default_layout_model: str = Field(default="auto", description="Default layout model")
    default_ocr_model: str = Field(default="auto", description="Default OCR model") 
    default_table_model: str = Field(default="auto", description="Default table model")
    
    # Quality settings
    default_image_dpi: int = Field(default=300, ge=72, le=600, description="Default image DPI")
    default_ocr_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Default OCR confidence")
    default_table_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Default table confidence")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable service-level caching")
    cache_dir: str = Field(default="./cache", description="Cache directory")
    cache_max_size_gb: int = Field(default=5, description="Maximum cache size in GB")
    cache_ttl_hours: int = Field(default=168, description="Cache TTL in hours")  # 7 days
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    access_log: bool = Field(default=True, description="Enable access logging")
    
    # Security
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    cors_origins: str = Field(default="*", description="CORS allowed origins")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    health_check_interval: int = Field(default=30, description="Health check interval")
    
    # Development settings
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")

    @validator('cache_dir')
    def create_cache_dir(cls, v):
        """Ensure cache directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        env_prefix = "DOCLING_"
        case_sensitive = False


def get_service_settings() -> DoclingServiceSettings:
    """Get service configuration settings."""
    return DoclingServiceSettings()


def get_supported_formats() -> list[str]:
    """Get list of supported file formats."""
    return [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"]


def get_feature_flags() -> Dict[str, bool]:
    """Get feature flags."""
    return {
        "batch_processing": True,
        "async_processing": True,
        "caching": True,
        "metrics": True,
        "health_checks": True,
        "api_authentication": False,  # Disabled by default
    }


def get_service_limits() -> Dict[str, Any]:
    """Get service operational limits."""
    settings = get_service_settings()
    return {
        "max_file_size_mb": settings.max_file_size_mb,
        "max_batch_size": settings.max_batch_size,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "max_workers": settings.default_max_workers,
        "cache_max_size_gb": settings.cache_max_size_gb,
    }


def get_default_docling_config() -> Dict[str, Any]:
    """Get default docling configuration."""
    settings = get_service_settings()
    return {
        "export_format": settings.default_export_format,
        "enable_ocr": settings.default_enable_ocr,
        "enable_table_structure": settings.default_enable_table_structure,
        "enable_figures": settings.default_enable_figures,
        "process_images": True,
        "layout_model": settings.default_layout_model,
        "ocr_model": settings.default_ocr_model,
        "table_model": settings.default_table_model,
        "include_page_numbers": True,
        "include_headings": True,
        "extract_metadata": True,
        "image_dpi": settings.default_image_dpi,
        "ocr_confidence": settings.default_ocr_confidence,
        "table_confidence": settings.default_table_confidence,
        "max_workers": settings.default_max_workers,
        "enable_cache": settings.default_enable_cache,
        "cache_ttl_hours": settings.cache_ttl_hours,
    }


# Global settings instance
service_settings = get_service_settings()