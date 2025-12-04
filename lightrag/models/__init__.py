"""Data models for LightRAG multi-tenant architecture."""

from .tenant import (
    Tenant,
    TenantConfig,
    TenantContext,
    KnowledgeBase,
    KBConfig,
    ResourceQuota,
)

__all__ = [
    "Tenant",
    "TenantConfig",
    "TenantContext",
    "KnowledgeBase",
    "KBConfig",
    "ResourceQuota",
]
