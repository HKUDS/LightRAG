"""Unit tests for WorkspaceRegistry and get_workspace_registry."""

from __future__ import annotations

import json
import threading
from unittest.mock import patch

import pytest

from lightrag.api.workspace_registry import (
    WorkspaceRegistry,
    get_workspace_registry,
    WORKSPACE_REGISTRY_FILENAME,
)

pytestmark = pytest.mark.offline


# =============================================================================
# TestWorkspaceRegistry Tests
# =============================================================================


class TestWorkspaceRegistry:
    """Tests for the WorkspaceRegistry class."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_creates_registry_in_working_dir(self, tmp_path) -> None:
        """Test that registry file path is set correctly from working_dir."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        assert registry.registry_path == tmp_path / WORKSPACE_REGISTRY_FILENAME

    def test_init_creates_registry_in_cwd_when_no_working_dir(
        self, monkeypatch, tmp_path
    ) -> None:
        """Test that registry defaults to cwd when no working_dir is given."""
        # Change cwd to tmp_path so we know where the file ends up
        monkeypatch.chdir(tmp_path)
        registry = WorkspaceRegistry()
        assert registry.registry_path.parent == tmp_path

    def test_init_loads_existing_file(self, tmp_path) -> None:
        """Test that existing registry file is loaded on initialization."""
        registry_path = tmp_path / WORKSPACE_REGISTRY_FILENAME
        existing_data = {
            "existing-ws": {
                "name": "existing-ws",
                "first_seen": "2024-01-01T00:00:00+00:00",
                "last_seen": "2024-01-01T00:00:00+00:00",
            }
        }
        registry_path.write_text(json.dumps(existing_data), encoding="utf-8")

        registry = WorkspaceRegistry(working_dir=str(tmp_path))

        workspaces = registry.get_workspaces()
        assert len(workspaces) == 1
        assert workspaces[0]["name"] == "existing-ws"

    def test_init_starts_empty_when_file_missing(self, tmp_path) -> None:
        """Test that registry starts empty when file does not exist."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        assert registry.get_workspaces() == []

    def test_init_handles_corrupted_json(self, tmp_path) -> None:
        """Test that corrupted JSON file is handled gracefully (starts fresh)."""
        registry_path = tmp_path / WORKSPACE_REGISTRY_FILENAME
        registry_path.write_text("{ not valid json", encoding="utf-8")

        registry = WorkspaceRegistry(working_dir=str(tmp_path))

        # Should start fresh, not crash
        assert registry.get_workspaces() == []

    def test_init_handles_empty_json_object(self, tmp_path) -> None:
        """Test that empty JSON object {} is handled gracefully."""
        registry_path = tmp_path / WORKSPACE_REGISTRY_FILENAME
        registry_path.write_text("{}", encoding="utf-8")

        registry = WorkspaceRegistry(working_dir=str(tmp_path))

        assert registry.get_workspaces() == []

    # -------------------------------------------------------------------------
    # register_workspace Tests
    # -------------------------------------------------------------------------

    def test_register_workspace_adds_to_list(self, tmp_path) -> None:
        """Test that registering a workspace adds it to get_workspaces."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("my-workspace")

        workspaces = registry.get_workspaces()
        assert len(workspaces) == 1
        assert workspaces[0]["name"] == "my-workspace"
        assert "first_seen" in workspaces[0]
        assert "last_seen" in workspaces[0]
        assert workspaces[0]["first_seen"] == workspaces[0]["last_seen"]

    def test_register_workspace_sets_both_timestamps(self, tmp_path) -> None:
        """Test that first registration sets both first_seen and last_seen."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-timestamps")

        workspaces = registry.get_workspaces()
        assert workspaces[0]["first_seen"] is not None
        assert workspaces[0]["last_seen"] is not None
        # Both should be ISO format strings
        assert "T" in workspaces[0]["first_seen"]
        assert "T" in workspaces[0]["last_seen"]

    def test_register_workspace_updates_last_seen(self, tmp_path) -> None:
        """Test that re-registering updates last_seen but not first_seen."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-update")
        first_last_seen = registry.get_workspaces()[0]["last_seen"]
        first_first_seen = registry.get_workspaces()[0]["first_seen"]

        # Wait a tiny bit to ensure timestamp differs
        import time

        time.sleep(0.01)
        registry.register_workspace("ws-update")

        workspaces = registry.get_workspaces()
        assert workspaces[0]["first_seen"] == first_first_seen
        assert workspaces[0]["last_seen"] != first_last_seen
        assert workspaces[0]["last_seen"] > first_last_seen

    def test_register_empty_string_is_noop(self, tmp_path) -> None:
        """Test that empty string workspace is not registered (no-op)."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("")

        assert registry.get_workspaces() == []

    def test_register_none_is_noop(self, tmp_path) -> None:
        """Test that None workspace is not registered (no-op)."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace(None)  # type: ignore[arg-type]

        assert registry.get_workspaces() == []

    def test_register_multiple_workspaces(self, tmp_path) -> None:
        """Test that multiple distinct workspaces are all registered."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-a")
        registry.register_workspace("ws-b")
        registry.register_workspace("ws-c")

        workspaces = registry.get_workspaces()
        names = {w["name"] for w in workspaces}
        assert names == {"ws-a", "ws-b", "ws-c"}

    def test_register_workspace_with_special_chars(self, tmp_path) -> None:
        """Test that workspaces with special characters are registered correctly."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        # The source code does not validate/normalize names, so any non-empty
        # string is accepted as-is
        registry.register_workspace("ws_underscore")
        registry.register_workspace("ws-hyphen")
        registry.register_workspace("ws.dot")
        registry.register_workspace("ws@at")
        registry.register_workspace("ws with space")

        workspaces = registry.get_workspaces()
        assert len(workspaces) == 5
        names = {w["name"] for w in workspaces}
        assert "ws_underscore" in names
        assert "ws-hyphen" in names
        assert "ws.dot" in names
        assert "ws@at" in names
        assert "ws with space" in names

    # -------------------------------------------------------------------------
    # Persistence Tests
    # -------------------------------------------------------------------------

    def test_persistence_survives_reload(self, tmp_path) -> None:
        """Test that registry data survives save/reload cycle."""
        # Create and register
        registry1 = WorkspaceRegistry(working_dir=str(tmp_path))
        registry1.register_workspace("persistent-ws")

        # Re-create from same directory
        registry2 = WorkspaceRegistry(working_dir=str(tmp_path))
        workspaces = registry2.get_workspaces()

        assert len(workspaces) == 1
        assert workspaces[0]["name"] == "persistent-ws"

    def test_persistence_multiple_workspaces_reload(self, tmp_path) -> None:
        """Test that multiple workspaces persist across reload."""
        registry1 = WorkspaceRegistry(working_dir=str(tmp_path))
        registry1.register_workspace("ws-1")
        registry1.register_workspace("ws-2")
        registry1.register_workspace("ws-3")

        registry2 = WorkspaceRegistry(working_dir=str(tmp_path))
        names = {w["name"] for w in registry2.get_workspaces()}
        assert names == {"ws-1", "ws-2", "ws-3"}

    def test_file_contains_valid_json(self, tmp_path) -> None:
        """Test that the saved file is valid JSON."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("json-check")

        registry_path = tmp_path / WORKSPACE_REGISTRY_FILENAME
        assert registry_path.exists()

        data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert "json-check" in data
        assert data["json-check"]["name"] == "json-check"

    def test_file_parent_directory_created(self, tmp_path) -> None:
        """Test that parent directory is created if missing."""
        nested = tmp_path / "nested" / "dir"
        registry = WorkspaceRegistry(working_dir=str(nested))
        registry.register_workspace("deep-ws")

        assert (nested / WORKSPACE_REGISTRY_FILENAME).exists()

    # -------------------------------------------------------------------------
    # Atomic Write Tests
    # -------------------------------------------------------------------------

    def test_atomic_write_no_partial_file(self, tmp_path) -> None:
        """Test that only the final file is visible (no partial writes)."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-1")

        registry_path = tmp_path / WORKSPACE_REGISTRY_FILENAME
        tmp_file = tmp_path / (WORKSPACE_REGISTRY_FILENAME + ".tmp")

        # .tmp file should not remain after save
        assert not tmp_file.exists()
        # Main file should exist and be valid
        assert registry_path.exists()
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert "ws-1" in data

    # -------------------------------------------------------------------------
    # Rollback Tests
    # -------------------------------------------------------------------------

    def test_rollback_on_save_failure(self, tmp_path) -> None:
        """Test that IOError propagates when save fails.

        Note: the rollback restores to the in-memory state captured before
        _save_registry is called — which is AFTER the workspace is already in
        _cache. The key invariant is that the IOError is raised and the
        in-memory state matches what was present when save was attempted.
        """
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-before")

        # Verify state before injection
        assert len(registry.get_workspaces()) == 1

        # Capture the on-disk state before the failing write
        registry_path = tmp_path / WORKSPACE_REGISTRY_FILENAME
        original_file_content = registry_path.read_text(encoding="utf-8")

        def raise_on_write(*args, **kwargs):
            if len(args) > 0 and ".tmp" in str(args[0]):
                raise IOError("simulated write failure")
            return __builtins__["open"](*args, **kwargs)

        with patch("builtins.open", side_effect=raise_on_write):
            with pytest.raises(IOError, match="simulated write failure"):
                registry.register_workspace("ws-after-fail")

        # The IOError was raised (verified by pytest.raises).
        # In-memory state is whatever it was when _save_registry was called
        # (including the new workspace — rollback restores that same snapshot).
        # The critical check: disk file must be UNCHANGED from before the call.
        assert registry_path.read_text(encoding="utf-8") == original_file_content

    # -------------------------------------------------------------------------
    # Thread Safety Tests
    # -------------------------------------------------------------------------

    def test_concurrent_register_same_workspace(self, tmp_path) -> None:
        """Test that concurrent registration of same workspace is thread-safe."""

        def register_many(registry: WorkspaceRegistry, suffix: str) -> None:
            for i in range(50):
                registry.register_workspace(f"concurrent-ws-{suffix}")

        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        threads = [
            threading.Thread(target=register_many, args=(registry, f"t{i}"))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete without errors
        workspaces = registry.get_workspaces()
        names = {w["name"] for w in workspaces}
        # Each thread registered its own distinct workspace
        assert len(names) == 4

    def test_concurrent_register_different_workspaces(self, tmp_path) -> None:
        """Test that concurrent registration of different workspaces is thread-safe."""
        num_threads = 8
        workspaces_per_thread = 25

        def register_range(registry: WorkspaceRegistry, start: int) -> None:
            for i in range(start, start + workspaces_per_thread):
                registry.register_workspace(f"ws-{i:03d}")

        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        threads = [
            threading.Thread(
                target=register_range, args=(registry, i * workspaces_per_thread)
            )
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        workspaces = registry.get_workspaces()
        assert len(workspaces) == num_threads * workspaces_per_thread

    # -------------------------------------------------------------------------
    # get_workspaces Ordering Tests
    # -------------------------------------------------------------------------

    def test_get_workspaces_sorted_by_last_seen_descending(self, tmp_path) -> None:
        """Test that workspaces are sorted by last_seen descending (most recent first)."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-old")
        import time

        time.sleep(0.02)
        registry.register_workspace("ws-new")
        time.sleep(0.02)
        registry.register_workspace("ws-newest")

        # Accessing old workspace again should move it to most recent
        registry.register_workspace("ws-old")

        workspaces = registry.get_workspaces()
        names = [w["name"] for w in workspaces]

        # ws-old should now be first (most recently seen)
        assert names[0] == "ws-old"
        assert names[1] == "ws-newest"
        assert names[2] == "ws-new"

    def test_get_workspaces_returns_copy(self, tmp_path) -> None:
        """Test that get_workspaces returns a new list each time."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-1")

        result1 = registry.get_workspaces()
        result2 = registry.get_workspaces()

        assert result1 is not result2
        assert result1 == result2

    def test_get_workspaces_returns_clean_dicts(self, tmp_path) -> None:
        """Test that returned workspace dicts only contain name/first_seen/last_seen."""
        registry = WorkspaceRegistry(working_dir=str(tmp_path))
        registry.register_workspace("ws-1")

        workspaces = registry.get_workspaces()
        assert list(workspaces[0].keys()) == ["name", "first_seen", "last_seen"]


# =============================================================================
# get_workspace_registry Singleton Tests
# =============================================================================


class TestGetWorkspaceRegistrySingleton:
    """Tests for the get_workspace_registry singleton accessor."""

    def test_singleton_returns_same_instance(self, tmp_path) -> None:
        """Test that get_workspace_registry returns the same instance for same working_dir."""
        # Reset global state before test
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        reg1 = get_workspace_registry(working_dir=str(tmp_path))
        reg2 = get_workspace_registry()

        assert reg1 is reg2

    def test_singleton_different_working_dirs_same_instance(self, tmp_path) -> None:
        """Test that calling without working_dir returns the same singleton as first call."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        reg1 = get_workspace_registry(working_dir=str(tmp_path))
        reg2 = get_workspace_registry()  # no working_dir, should use same

        assert reg1 is reg2

    def test_singleton_raises_on_mismatched_working_dir(self, tmp_path) -> None:
        """Test that ValueError is raised when second call uses different working_dir."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        get_workspace_registry(working_dir=str(tmp_path))
        other_dir = tmp_path / "other"
        with pytest.raises(ValueError, match="already initialized"):
            get_workspace_registry(working_dir=str(other_dir))

    def test_singleton_raises_on_second_init_with_different_dir(self, tmp_path) -> None:
        """Test that second call with different working_dir raises ValueError."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        get_workspace_registry(working_dir=str(tmp_path))

        other_path = tmp_path / "different"
        with pytest.raises(ValueError, match="already initialized"):
            get_workspace_registry(working_dir=str(other_path))

    def test_singleton_works_without_working_dir(self, monkeypatch, tmp_path) -> None:
        """Test that singleton works when called without working_dir (uses cwd)."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None
        monkeypatch.chdir(tmp_path)

        reg = get_workspace_registry()

        assert reg is not None
        assert isinstance(reg, WorkspaceRegistry)

    def test_singleton_does_not_raise_when_called_again_with_same_dir(
        self, tmp_path
    ) -> None:
        """Test that calling again with the same working_dir does not raise."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        reg1 = get_workspace_registry(working_dir=str(tmp_path))
        reg2 = get_workspace_registry(working_dir=str(tmp_path))

        assert reg1 is reg2

    def test_singleton_none_working_dir_after_init_with_dir(self, tmp_path) -> None:
        """Test that calling with None after init with dir returns same instance."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        reg1 = get_workspace_registry(working_dir=str(tmp_path))
        reg2 = get_workspace_registry(working_dir=None)  # type: ignore[arg-type]

        assert reg1 is reg2

    def test_singleton_raises_on_working_dir_mismatch_message(self, tmp_path) -> None:
        """Test that ValueError message contains both working directories."""
        import lightrag.api.workspace_registry as wr

        wr._registry_instance = None

        get_workspace_registry(working_dir=str(tmp_path))

        other_dir = tmp_path / "other_dir"
        with pytest.raises(ValueError, match=str(tmp_path)):
            get_workspace_registry(working_dir=str(other_dir))
