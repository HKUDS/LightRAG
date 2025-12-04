"""
Tenant-aware LightRAG instance manager with caching and isolation.

This module manages per-tenant and per-knowledge-base LightRAG instances,
handling initialization, caching, cleanup, and proper isolation between tenants.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
from lightrag import LightRAG
from lightrag.models.tenant import TenantContext, TenantConfig, Tenant
from lightrag.services.tenant_service import TenantService
from lightrag.utils import logger
from lightrag.security import validate_identifier, validate_working_directory
import asyncio
import os


class TenantRAGManager:
    """
    Manages LightRAG instances per tenant/KB combination with caching and isolation.
    
    Features:
    - Automatic instance caching to avoid repeated initialization
    - Per-tenant isolation through separate working directories
    - Configurable max cached instances (LRU eviction)
    - Async-safe initialization with double-check locking
    - Proper resource cleanup on instance removal
    """
    
    def __init__(
        self,
        base_working_dir: str,
        tenant_service: TenantService,
        template_rag: Optional[LightRAG] = None,
        max_cached_instances: int = 100,
    ):
        """
        Initialize the TenantRAGManager.
        
        Args:
            base_working_dir: Base directory for all tenant/KB data storage
            tenant_service: Service for retrieving tenant configuration
            template_rag: Template RAG instance to copy configuration from
            max_cached_instances: Maximum number of LightRAG instances to keep cached
        """
        self.base_working_dir = base_working_dir
        self.tenant_service = tenant_service
        self.template_rag = template_rag
        self.max_cached_instances = max_cached_instances
        self._instances: Dict[Tuple[str, str], LightRAG] = {}
        self._lock = asyncio.Lock()
        self._access_order: list[Tuple[str, str]] = []  # Track access order for LRU
        logger.info(
            f"TenantRAGManager initialized with base_dir={base_working_dir}, "
            f"max_instances={max_cached_instances}, template_rag={template_rag is not None}"
        )
    
    async def get_rag_instance(
        self,
        tenant_id: str,
        kb_id: str,
        user_id: Optional[str] = None,
    ) -> LightRAG:
        """
        Get or create a LightRAG instance for a tenant/KB combination.

        This method implements double-check locking to avoid race conditions
        when multiple requests try to initialize the same instance concurrently.
        Instances are cached and reused across requests for the same tenant/KB.
        
        SECURITY: Validates user has access to requested tenant before returning instance.

        Args:
            tenant_id: The tenant ID (must be valid UUID)
            kb_id: The knowledge base ID (must be valid UUID)
            user_id: User identifier from JWT token (required for security validation)

        Returns:
            LightRAG: A properly initialized LightRAG instance for this tenant/KB

        Raises:
            ValueError: If the tenant does not exist or is inactive
            PermissionError: If user does not have access to the tenant
            HTTPException: If tenant_id or kb_id are invalid identifiers
        """
        # SECURITY: Validate identifier format to prevent injection attacks
        tenant_id = validate_identifier(tenant_id, "tenant_id")
        kb_id = validate_identifier(kb_id, "kb_id")
        
        cache_key = (tenant_id, kb_id)
        
        # First check (fast path - no lock)
        if cache_key in self._instances:
            instance = self._instances[cache_key]
            # Update access order for LRU
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            logger.debug(f"Cache hit for tenant={tenant_id}, kb={kb_id}")
            return instance
        
        # Acquire lock for initialization
        async with self._lock:
            # Second check (double-check locking pattern)
            if cache_key in self._instances:
                instance = self._instances[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                logger.debug(f"Cache hit (after lock) for tenant={tenant_id}, kb={kb_id}")
                return instance
            
            logger.info(f"Creating new RAG instance for tenant={tenant_id}, kb={kb_id}")
            
            # Get tenant configuration
            tenant = await self.tenant_service.get_tenant(tenant_id)
            if not tenant or not tenant.is_active:
                raise ValueError(f"Tenant {tenant_id} not found or inactive")
            
            # SEC-003 FIX: Check if user authentication is required
            try:
                from lightrag.api.config import REQUIRE_USER_AUTH
                require_auth = REQUIRE_USER_AUTH
            except ImportError:
                require_auth = False
            
            # SECURITY: Verify user has access to this tenant
            if user_id:
                has_access = await self.tenant_service.verify_user_access(user_id, tenant_id)
                if not has_access:
                    logger.warning(
                        f"Access denied: user={user_id} attempted to access tenant={tenant_id}"
                    )
                    raise PermissionError(f"Access denied to tenant {tenant_id}")
            elif require_auth:
                logger.error(
                    f"Access denied: user_id required but not provided for tenant={tenant_id}"
                )
                raise PermissionError("User authentication required for tenant access")
            else:
                logger.warning(
                    f"No user_id provided for tenant access - allowing for backward compatibility"
                )
            
            # SECURITY: Create and validate tenant-specific working directory
            # This prevents path traversal attacks
            tenant_working_dir, composite_workspace = validate_working_directory(
                self.base_working_dir,
                tenant_id,
                kb_id
            )
            os.makedirs(tenant_working_dir, exist_ok=True)
            
            try:
                # Create LightRAG instance with tenant-specific configuration
                # Use template RAG configuration if available, otherwise use defaults
                if self.template_rag:
                    # Copy configuration from template RAG
                    instance = LightRAG(
                        working_dir=tenant_working_dir,
                        workspace=composite_workspace,
                        llm_model_func=self.template_rag.llm_model_func,
                        llm_model_name=self.template_rag.llm_model_name,
                        llm_model_max_async=self.template_rag.llm_model_max_async,
                        llm_model_kwargs=self.template_rag.llm_model_kwargs,
                        embedding_func=self.template_rag.embedding_func,
                        default_llm_timeout=self.template_rag.default_llm_timeout,
                        default_embedding_timeout=self.template_rag.default_embedding_timeout,
                        kv_storage=tenant.config.custom_metadata.get("kv_storage") or self.template_rag.kv_storage,
                        vector_storage=tenant.config.custom_metadata.get("vector_storage") or self.template_rag.vector_storage,
                        graph_storage=tenant.config.custom_metadata.get("graph_storage") or self.template_rag.graph_storage,
                        doc_status_storage=self.template_rag.doc_status_storage,
                        vector_db_storage_cls_kwargs=self.template_rag.vector_db_storage_cls_kwargs,
                        enable_llm_cache=self.template_rag.enable_llm_cache,
                        enable_llm_cache_for_entity_extract=self.template_rag.enable_llm_cache_for_entity_extract,
                        rerank_model_func=self.template_rag.rerank_model_func,
                        chunk_token_size=self.template_rag.chunk_token_size,
                        chunk_overlap_token_size=self.template_rag.chunk_overlap_token_size,
                        max_parallel_insert=self.template_rag.max_parallel_insert,
                        max_graph_nodes=self.template_rag.max_graph_nodes,
                        addon_params=self.template_rag.addon_params,
                        ollama_server_infos=getattr(self.template_rag, 'ollama_server_infos', None),
                        # Override with tenant-specific settings
                        top_k=tenant.config.top_k,
                        chunk_top_k=getattr(tenant.config, "chunk_top_k", 40),
                        cosine_threshold=tenant.config.cosine_threshold,
                    )
                else:
                    # Fallback to basic configuration (will likely fail without embedding_func)
                    instance = LightRAG(
                        working_dir=tenant_working_dir,
                        workspace=composite_workspace,
                        kv_storage=tenant.config.custom_metadata.get("kv_storage", "JsonKVStorage"),
                        vector_storage=tenant.config.custom_metadata.get("vector_storage", "NanoVectorDBStorage"),
                        graph_storage=tenant.config.custom_metadata.get("graph_storage", "NetworkXStorage"),
                        top_k=tenant.config.top_k,
                        chunk_top_k=getattr(tenant.config, "chunk_top_k", 40),
                        cosine_threshold=tenant.config.cosine_threshold,
                    )
                
                # Initialize the instance's storages
                await instance.initialize_storages()
                
                # Check if we need to evict oldest instance
                if len(self._instances) >= self.max_cached_instances:
                    # Evict least recently used instance
                    if self._access_order:
                        oldest_key = self._access_order.pop(0)
                        if oldest_key in self._instances:
                            logger.info(f"Evicting LRU instance: tenant={oldest_key[0]}, kb={oldest_key[1]}")
                            try:
                                await self._instances[oldest_key].finalize_storages()
                            except Exception as e:
                                logger.error(f"Error finalizing evicted instance: {e}")
                            del self._instances[oldest_key]
                
                # Cache the instance
                self._instances[cache_key] = instance
                self._access_order.append(cache_key)
                logger.info(f"RAG instance created and cached for tenant={tenant_id}, kb={kb_id}")
                return instance
                
            except Exception as e:
                logger.error(f"Error creating RAG instance for tenant={tenant_id}, kb={kb_id}: {e}")
                raise
    
    async def cleanup_instance(self, tenant_id: str, kb_id: str) -> None:
        """
        Clean up and remove a cached instance.
        
        This method should be called when a knowledge base is deleted or
        a tenant is removed to ensure proper resource cleanup.
        
        Args:
            tenant_id: The tenant ID
            kb_id: The knowledge base ID
        """
        cache_key = (tenant_id, kb_id)
        async with self._lock:
            if cache_key in self._instances:
                logger.info(f"Cleaning up RAG instance for tenant={tenant_id}, kb={kb_id}")
                try:
                    await self._instances[cache_key].finalize_storages()
                except Exception as e:
                    logger.error(f"Error finalizing instance during cleanup: {e}")
                del self._instances[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
    
    async def cleanup_tenant_instances(self, tenant_id: str) -> None:
        """
        Clean up all cached instances for a specific tenant.
        
        This method should be called when a tenant is deleted to ensure
        all its knowledge bases are properly cleaned up.
        
        Args:
            tenant_id: The tenant ID
        """
        async with self._lock:
            keys_to_remove = [k for k in self._instances.keys() if k[0] == tenant_id]
            for key in keys_to_remove:
                logger.info(f"Cleaning up RAG instance for tenant={key[0]}, kb={key[1]}")
                try:
                    await self._instances[key].finalize_storages()
                except Exception as e:
                    logger.error(f"Error finalizing instance during tenant cleanup: {e}")
                del self._instances[key]
                if key in self._access_order:
                    self._access_order.remove(key)
    
    async def cleanup_all(self) -> None:
        """
        Clean up all cached instances.
        
        This should be called during application shutdown to ensure
        all resources are properly released.
        """
        async with self._lock:
            logger.info(f"Cleaning up all {len(self._instances)} cached RAG instances")
            for key, instance in list(self._instances.items()):
                try:
                    await instance.finalize_storages()
                except Exception as e:
                    logger.error(f"Error finalizing instance {key}: {e}")
            self._instances.clear()
            self._access_order.clear()
    
    def get_instance_count(self) -> int:
        """Get the current number of cached instances."""
        return len(self._instances)
    
    def get_cached_keys(self) -> list[Tuple[str, str]]:
        """Get all currently cached tenant/KB combinations."""
        return list(self._instances.keys())
    
    def __repr__(self) -> str:
        """String representation of the manager state."""
        return (
            f"TenantRAGManager(instances={len(self._instances)}, "
            f"max_cached={self.max_cached_instances})"
        )
