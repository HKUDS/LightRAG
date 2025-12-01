"""
Multi-workspace management for LightRAG Server.

This module provides workspace isolation at the API server level by managing
a pool of LightRAG instances, one per workspace. It enables multi-tenant
deployments where each tenant's data is completely isolated.

Key components:
- WorkspaceConfig: Configuration for multi-workspace behavior
- WorkspacePool: Process-local pool of LightRAG instances with LRU eviction
- get_rag: FastAPI dependency for resolving workspace-specific RAG instance
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

# Workspace identifier validation pattern
# - Must start with alphanumeric
# - Can contain alphanumeric, hyphens, underscores
# - Length 1-64 characters
WORKSPACE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass
class WorkspaceConfig:
    """Configuration for multi-workspace behavior."""

    default_workspace: str = ""
    allow_default_workspace: bool = True
    max_workspaces_in_pool: int = 50


@dataclass
class WorkspaceInstance:
    """A running LightRAG instance for a specific workspace."""

    workspace_id: str
    rag_instance: object  # LightRAG instance
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last access time."""
        self.last_accessed_at = time.time()


class WorkspacePool:
    """
    Process-local pool of LightRAG instances keyed by workspace identifier.

    Uses asyncio.Lock for thread-safe access and LRU eviction when the pool
    reaches its maximum size.
    """

    def __init__(
        self,
        config: WorkspaceConfig,
        rag_factory: Callable[[str], Awaitable[object]],
    ):
        """
        Initialize the workspace pool.

        Args:
            config: Multi-workspace configuration
            rag_factory: Async factory function that creates a LightRAG instance
                        for a given workspace identifier
        """
        self._config = config
        self._rag_factory = rag_factory
        self._instances: dict[str, WorkspaceInstance] = {}
        self._lru_order: list[str] = []
        self._lock = asyncio.Lock()
        self._initializing: dict[str, asyncio.Event] = {}

    @property
    def size(self) -> int:
        """Current number of instances in the pool."""
        return len(self._instances)

    @property
    def max_size(self) -> int:
        """Maximum pool size from configuration."""
        return self._config.max_workspaces_in_pool

    async def get(self, workspace_id: str) -> object:
        """
        Get or create a LightRAG instance for the specified workspace.

        Args:
            workspace_id: The workspace identifier

        Returns:
            LightRAG instance for the workspace

        Raises:
            ValueError: If workspace_id is invalid
            RuntimeError: If instance initialization fails
        """
        # Validate workspace identifier
        validate_workspace_id(workspace_id)

        async with self._lock:
            # Check if instance already exists
            if workspace_id in self._instances:
                instance = self._instances[workspace_id]
                instance.touch()
                self._update_lru(workspace_id)
                logger.debug(f"Returning cached instance for workspace: {workspace_id}")
                return instance.rag_instance

            # Check if another request is already initializing this workspace
            if workspace_id in self._initializing:
                event = self._initializing[workspace_id]
                # Release lock while waiting
                self._lock.release()
                try:
                    await event.wait()
                finally:
                    await self._lock.acquire()

                # Instance should now exist
                if workspace_id in self._instances:
                    instance = self._instances[workspace_id]
                    instance.touch()
                    self._update_lru(workspace_id)
                    return instance.rag_instance
                else:
                    raise RuntimeError(
                        f"Workspace initialization failed: {workspace_id}"
                    )

            # Start initialization
            self._initializing[workspace_id] = asyncio.Event()

        # Initialize outside the lock to avoid blocking other workspaces
        try:
            # Evict if at capacity
            await self._evict_if_needed()

            logger.info(f"Initializing workspace: {workspace_id}")
            start_time = time.time()

            rag_instance = await self._rag_factory(workspace_id)

            elapsed = time.time() - start_time
            logger.info(
                f"Workspace initialized in {elapsed:.2f}s: {workspace_id}"
            )

            async with self._lock:
                instance = WorkspaceInstance(
                    workspace_id=workspace_id,
                    rag_instance=rag_instance,
                )
                self._instances[workspace_id] = instance
                self._lru_order.append(workspace_id)

                # Signal waiting requests
                if workspace_id in self._initializing:
                    self._initializing[workspace_id].set()
                    del self._initializing[workspace_id]

                return rag_instance

        except Exception as e:
            async with self._lock:
                # Clean up initialization state
                if workspace_id in self._initializing:
                    self._initializing[workspace_id].set()
                    del self._initializing[workspace_id]
            logger.error(f"Failed to initialize workspace {workspace_id}: {e}")
            raise RuntimeError(f"Failed to initialize workspace: {workspace_id}") from e

    async def _evict_if_needed(self) -> None:
        """Evict LRU instance if pool is at capacity."""
        async with self._lock:
            if len(self._instances) >= self._config.max_workspaces_in_pool:
                if self._lru_order:
                    oldest_id = self._lru_order.pop(0)
                    instance = self._instances.pop(oldest_id, None)
                    if instance:
                        logger.info(f"Evicting workspace from pool: {oldest_id}")
                        # Finalize storage outside the lock
                        rag = instance.rag_instance
                        # Release lock for finalization
                        self._lock.release()
                        try:
                            if hasattr(rag, "finalize_storages"):
                                await rag.finalize_storages()
                        except Exception as e:
                            logger.warning(
                                f"Error finalizing workspace {oldest_id}: {e}"
                            )
                        finally:
                            await self._lock.acquire()

    def _update_lru(self, workspace_id: str) -> None:
        """Move workspace to end of LRU list (most recently used)."""
        if workspace_id in self._lru_order:
            self._lru_order.remove(workspace_id)
        self._lru_order.append(workspace_id)

    async def finalize_all(self) -> None:
        """Finalize all workspace instances for graceful shutdown."""
        async with self._lock:
            workspace_ids = list(self._instances.keys())

        for workspace_id in workspace_ids:
            async with self._lock:
                instance = self._instances.pop(workspace_id, None)
                if workspace_id in self._lru_order:
                    self._lru_order.remove(workspace_id)

            if instance:
                logger.info(f"Finalizing workspace: {workspace_id}")
                try:
                    rag = instance.rag_instance
                    if hasattr(rag, "finalize_storages"):
                        await rag.finalize_storages()
                except Exception as e:
                    logger.warning(f"Error finalizing workspace {workspace_id}: {e}")

        logger.info("All workspace instances finalized")


def validate_workspace_id(workspace_id: str) -> None:
    """
    Validate a workspace identifier.

    Args:
        workspace_id: The workspace identifier to validate

    Raises:
        ValueError: If the workspace identifier is invalid
    """
    if not workspace_id:
        raise ValueError("Workspace identifier cannot be empty")

    if not WORKSPACE_ID_PATTERN.match(workspace_id):
        raise ValueError(
            f"Invalid workspace identifier '{workspace_id}': "
            "must be 1-64 alphanumeric characters "
            "(hyphens and underscores allowed, must start with alphanumeric)"
        )


def get_workspace_from_request(request: Request) -> str | None:
    """
    Extract workspace identifier from HTTP request headers.

    Checks headers in order of priority:
    1. LIGHTRAG-WORKSPACE (primary)
    2. X-Workspace-ID (fallback)

    Args:
        request: FastAPI request object

    Returns:
        Workspace identifier or None if not present
    """
    # Primary header
    workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
    if workspace:
        return workspace

    # Fallback header
    workspace = request.headers.get("X-Workspace-ID", "").strip()
    if workspace:
        return workspace

    return None


# Global pool instance (initialized by create_app)
_workspace_pool: WorkspacePool | None = None
_workspace_config: WorkspaceConfig | None = None


def init_workspace_pool(
    config: WorkspaceConfig,
    rag_factory: Callable[[str], Awaitable[object]],
) -> WorkspacePool:
    """
    Initialize the global workspace pool.

    Args:
        config: Multi-workspace configuration
        rag_factory: Async factory function for creating LightRAG instances

    Returns:
        The initialized WorkspacePool
    """
    global _workspace_pool, _workspace_config
    _workspace_config = config
    _workspace_pool = WorkspacePool(config, rag_factory)
    logger.info(
        f"Workspace pool initialized: max_size={config.max_workspaces_in_pool}, "
        f"default_workspace='{config.default_workspace}', "
        f"allow_default={config.allow_default_workspace}"
    )
    return _workspace_pool


def get_workspace_pool() -> WorkspacePool:
    """Get the global workspace pool instance."""
    if _workspace_pool is None:
        raise RuntimeError("Workspace pool not initialized")
    return _workspace_pool


def get_workspace_config() -> WorkspaceConfig:
    """Get the global workspace configuration."""
    if _workspace_config is None:
        raise RuntimeError("Workspace configuration not initialized")
    return _workspace_config


async def get_rag(request: Request) -> object:
    """
    FastAPI dependency for resolving the workspace-specific LightRAG instance.

    This dependency:
    1. Extracts workspace from request headers
    2. Falls back to default workspace if configured
    3. Returns 400 if workspace is required but missing
    4. Returns the appropriate LightRAG instance from the pool

    Args:
        request: FastAPI request object

    Returns:
        LightRAG instance for the resolved workspace

    Raises:
        HTTPException: 400 if workspace is missing/invalid, 503 if init fails
    """
    config = get_workspace_config()
    pool = get_workspace_pool()

    # Extract workspace from headers
    workspace = get_workspace_from_request(request)

    # Handle missing workspace
    if not workspace:
        if config.allow_default_workspace and config.default_workspace:
            workspace = config.default_workspace
            logger.debug(f"Using default workspace: {workspace}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Missing LIGHTRAG-WORKSPACE header. Workspace identification is required.",
            )

    # Validate workspace identifier
    try:
        validate_workspace_id(workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log workspace access (non-sensitive)
    logger.info(f"Request to workspace: {workspace}")

    # Get or create instance
    try:
        return await pool.get(workspace)
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize workspace '{workspace}': {str(e)}",
        )
