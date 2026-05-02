"""Workspace Registry for LightRAG - Persistent storage of workspace metadata."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lightrag.utils import logger

# Default filename for the workspace registry
WORKSPACE_REGISTRY_FILENAME = "workspace_registry.json"


class WorkspaceRegistry:
    """Persistent registry for tracking workspace usage.

    Stores workspace names along with metadata (first_seen, last_seen timestamps)
    in a JSON file. Thread-safe for concurrent access.

    Attributes:
        registry_path: Path to the JSON file storing workspace data.
        _lock: Lock for thread-safe file access.
        _cache: In-memory cache of the registry data.
    """

    def __init__(self, working_dir: Optional[str] = None) -> None:
        """Initialize the workspace registry.

        Args:
            working_dir: Directory to store the registry file.
                        Defaults to current working directory.
        """
        if working_dir:
            self.registry_path = Path(working_dir) / WORKSPACE_REGISTRY_FILENAME
        else:
            self.registry_path = Path.cwd() / WORKSPACE_REGISTRY_FILENAME

        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}

        # Load existing registry on initialization
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the registry from disk into memory cache.

        Note: This method is called during __init__ before the object is shared,
        so no locking is needed.
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(
                    f"Loaded workspace registry from {self.registry_path} "
                    f"with {len(self._cache)} entries"
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Failed to load workspace registry: {e}. Starting fresh."
                )
                self._cache = {}
        else:
            self._cache = {}

    def _save_registry(self) -> None:
        """Save the in-memory cache to disk atomically.

        Uses write-to-temp-then-rename pattern for atomicity on POSIX systems.
        If save fails, rolls back to the previous in-memory state.
        """
        # Snapshot current state for potential rollback
        old_cache = self._cache.copy()

        try:
            # Ensure parent directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then rename for atomicity
            tmp_path = self.registry_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            tmp_path.replace(self.registry_path)  # atomic on POSIX
            logger.debug(f"Saved workspace registry to {self.registry_path}")
        except IOError as e:
            # Rollback to previous state on failure
            self._cache = old_cache
            logger.error(
                f"Failed to save workspace registry: {e}. Rolled back in-memory state."
            )
            raise

    def register_workspace(self, workspace: str) -> None:
        """Register a workspace or update its last_seen timestamp.

        Args:
            workspace: The workspace name to register.
                      Empty string represents the default workspace.
        """
        if not workspace:
            # Don't register empty workspace (default)
            logger.debug("Skipping empty workspace registration")
            return

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if workspace in self._cache:
                # Update last_seen
                self._cache[workspace]["last_seen"] = now
            else:
                # Register new workspace
                self._cache[workspace] = {
                    "name": workspace,
                    "first_seen": now,
                    "last_seen": now,
                }
                logger.info(f"Registered new workspace: {workspace}")

            # Persist to disk
            self._save_registry()

    def get_workspaces(self) -> list[dict]:
        """Get all registered workspaces.

        Returns:
            List of workspace metadata dictionaries with name, first_seen, and last_seen.
        """
        with self._lock:
            # Sort by last_seen (most recent first)
            sorted_workspaces = sorted(
                self._cache.values(),
                key=lambda w: w.get("last_seen", ""),
                reverse=True,
            )
            return [
                {
                    "name": w["name"],
                    "first_seen": w["first_seen"],
                    "last_seen": w["last_seen"],
                }
                for w in sorted_workspaces
            ]


# Global singleton instance (initialized lazily)
_registry_instance: Optional[WorkspaceRegistry] = None
_registry_lock = threading.Lock()


def get_workspace_registry(working_dir: Optional[str] = None) -> WorkspaceRegistry:
    """Get or create the global workspace registry instance.

    Args:
        working_dir: Optional working directory for the registry file.
                    Only used on first call.

    Returns:
        The global WorkspaceRegistry instance.

    Raises:
        ValueError: If working_dir is specified and differs from the
                   existing instance's working directory.
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            # Double-check after acquiring lock
            if _registry_instance is None:
                _registry_instance = WorkspaceRegistry(working_dir=working_dir)
                logger.info(
                    f"Workspace registry initialized at {_registry_instance.registry_path}"
                )

    # Guard against re-entry with mismatched working_dir
    if working_dir is not None:
        instance_dir = str(_registry_instance.registry_path.parent)
        if working_dir != instance_dir:
            raise ValueError(
                f"Workspace registry already initialized with working_dir='{instance_dir}', "
                f"but requested working_dir='{working_dir}'. "
                "The global registry can only be initialized once with a specific working directory."
            )

    return _registry_instance
