"""Workspace Manager for LightRAG - LRU cache with reference counting."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class WorkspaceCapacityError(Exception):
    """Raised when all workspace slots have in-flight requests."""

    pass


class WorkspaceManager:
    """Manages LightRAG workspace instances with LRU caching and reference counting.

    Features:
    - LRU cache with configurable max instances
    - Reference counting for safe eviction
    - Per-workspace locks for thread-safe creation
    - Global lock for eviction coordination
    - Graceful shutdown with finalization
    """

    def __init__(
        self,
        factory: Callable[[str], Awaitable[Any]],
        max_instances: int = 10,
    ) -> None:
        """Initialize the WorkspaceManager.

        Args:
            factory: Async callable that creates a LightRAG instance for a given
                     workspace name.
            max_instances: Maximum number of workspace instances to keep in cache.
        """
        self._factory = factory
        self._max_instances = max_instances

        # LRU cache: oldest=first, newest=last
        self._cache: OrderedDict[str, Any] = OrderedDict()

        # Per-workspace creation locks
        self._locks: dict[str, asyncio.Lock] = {}

        # Reference counts: number of active users of each workspace
        self._ref_counts: dict[str, int] = {}

        # Global lock for eviction operations
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _normalize_workspace(self, workspace: str | None) -> str:
        """Normalize workspace name (None/empty → default workspace)."""
        return workspace if workspace else ""

    def _get_per_workspace_lock(self, workspace: str) -> asyncio.Lock:
        """Get or create a per-workspace lock."""
        if workspace not in self._locks:
            self._locks[workspace] = asyncio.Lock()
        return self._locks[workspace]

    async def get_or_create(self, workspace: str | None) -> Any:
        """Get a cached workspace instance or create a new one.

        Args:
            workspace: Workspace name (None or empty defaults to "")

        Returns:
            LightRAG instance for the workspace.

        Raises:
            WorkspaceCapacityError: If cache is full and all slots have in-flight requests.
        """
        workspace = self._normalize_workspace(workspace)

        # Fast path: workspace already cached.
        # No lock needed here — safe under asyncio's single-threaded event loop.
        # For multi-threaded scenarios, a lock would be required for ref_count increment.
        if workspace in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(workspace)
            self._ref_counts[workspace] += 1
            self._hits += 1
            return self._cache[workspace]

        # Slow path: need to create new instance
        # Check capacity BEFORE acquiring locks for performance
        if len(self._cache) >= self._max_instances:
            # Check if ALL entries have in-flight requests (ref_count > 0)
            if all(ref > 0 for ref in self._ref_counts.values()):
                raise WorkspaceCapacityError(
                    "All workspace slots have in-flight requests"
                )

        # Get per-workspace lock for thread-safe creation
        per_workspace_lock = self._get_per_workspace_lock(workspace)

        async with per_workspace_lock:
            # Double-check: workspace might have been added while waiting for lock
            if workspace in self._cache:
                self._cache.move_to_end(workspace)
                self._ref_counts[workspace] += 1
                self._hits += 1
                return self._cache[workspace]

            # Cache is full - need to evict one
            if len(self._cache) >= self._max_instances:
                await self._evict_one()

            # Create new instance
            instance = await self._factory(workspace)

            # Store in cache
            self._cache[workspace] = instance
            self._ref_counts[workspace] = 1
            self._misses += 1

            return instance

    def release(self, workspace: str | None) -> None:
        """Release a reference to a workspace.

        Args:
            workspace: Workspace name to release (None or empty defaults to "").
        """
        workspace = self._normalize_workspace(workspace)

        if workspace not in self._ref_counts:
            logger.warning(
                "Attempted to release unknown workspace '%s'",
                workspace,
            )
            return

        self._ref_counts[workspace] -= 1

        if self._ref_counts[workspace] < 0:
            logger.error(
                "Reference count underflow for workspace '%s', resetting to 0",
                workspace,
            )
            self._ref_counts[workspace] = 0

    async def _evict_one(self) -> None:
        """Evict one LRU workspace with ref_count == 0.

        Must be called while holding the per-workspace lock, but will also
        acquire the global eviction lock for thread-safe cache modification.
        """
        async with self._lock:
            # Find first workspace with ref_count == 0 (oldest such workspace)
            for ws_name in self._cache:
                if self._ref_counts.get(ws_name, 0) == 0:
                    # Found evictable workspace
                    instance = self._cache.pop(ws_name)
                    await self._finalize_instance(instance)
                    del self._ref_counts[ws_name]
                    # Clean up per-workspace lock if it exists
                    if ws_name in self._locks:
                        del self._locks[ws_name]
                    self._evictions += 1
                    logger.info(
                        "Evicted workspace '%s' (LRU)",
                        ws_name,
                    )
                    return

            # No evictable workspace found - should not happen if capacity
            # check was done correctly before calling this method
            raise WorkspaceCapacityError(
                "All workspace slots have in-flight requests"
            )

    async def _finalize_instance(self, instance: Any) -> None:
        """Finalize a workspace instance (cleanup resources).

        Args:
            instance: LightRAG instance to finalize.
        """
        try:
            await instance.finalize_storages()
            logger.debug("Finalized workspace instance")
        except Exception as e:
            logger.warning(
                "Failed to finalize workspace instance: %s",
                e,
            )
            # Don't raise - eviction must succeed

    async def shutdown(self) -> None:
        """Shutdown the workspace manager, finalizing all cached instances."""
        async with self._lock:
            for workspace, instance in list(self._cache.items()):
                await self._finalize_instance(instance)

            # Clear all state
            self._cache.clear()
            self._ref_counts.clear()
            self._locks.clear()

        logger.info(
            "WorkspaceManager shutdown: evicted=%d, hits=%d, misses=%d",
            self._evictions,
            self._hits,
            self._misses,
        )

    def get_stats(self) -> dict:
        """Get statistics about the workspace manager.

        Returns:
            Dictionary containing cache statistics and current state.
        """
        return {
            "active_instances": len(self._cache),
            "max_instances": self._max_instances,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "evictions": self._evictions,
            "workspaces": list(self._cache.keys()),
            "ref_counts": dict(self._ref_counts),
        }
