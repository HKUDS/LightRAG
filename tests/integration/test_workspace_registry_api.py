"""
Integration tests for Workspace Registry API.

These tests verify that the FastAPI workspace listing and auto-register mechanism
correctly:
- GET /workspaces returns empty list when no workspaces registered
- GET /workspaces returns list of registered workspaces with correct fields
- Document API calls with LIGHTRAG-WORKSPACE header auto-register the workspace
- Same workspace called again updates last_seen but keeps first_seen
- Multiple different workspaces each registered correctly
- Empty/missing workspace header is handled correctly
- Special characters in workspace names are handled

All external services are mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field, ConfigDict

from lightrag.api.workspace_registry import WorkspaceRegistry, get_workspace_registry


# =============================================================================
# Pydantic Models (duplicated from workspace_routes to avoid config import)
# =============================================================================


class WorkspaceInfo(BaseModel):
    """Workspace information model."""

    name: str = Field(description="The workspace name")
    first_seen: str = Field(
        description="ISO timestamp when the workspace was first seen"
    )
    last_seen: str = Field(
        description="ISO timestamp when the workspace was last accessed"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "my-workspace",
                "first_seen": "2026-04-30T10:00:00+00:00",
                "last_seen": "2026-04-30T11:30:00+00:00",
            }
        }
    )


class WorkspacesResponse(BaseModel):
    """Response model for listing workspaces."""

    workspaces: list[WorkspaceInfo] = Field(description="List of registered workspaces")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workspaces": [
                    {
                        "name": "my-workspace",
                        "first_seen": "2026-04-30T10:00:00+00:00",
                        "last_seen": "2026-04-30T11:30:00+00:00",
                    },
                ]
            }
        }
    )


# =============================================================================
# Test App Factory
# =============================================================================


def create_test_app(
    working_dir: Optional[str] = None,
) -> tuple[FastAPI, WorkspaceRegistry]:
    """
    Create a minimal FastAPI app that includes workspace routes.

    Args:
        working_dir: Optional directory for the workspace registry storage.

    Returns:
        tuple: (app, workspace_registry) - the test app and the registry instance
    """
    # Reset the global registry instance to use the specified working_dir
    import lightrag.api.workspace_registry as wr_module

    wr_module._registry_instance = None
    wr_module._registry_lock = __import__("threading").Lock()

    # Get fresh registry with our working_dir
    registry = get_workspace_registry(working_dir=working_dir)

    app = FastAPI()

    # Add workspace listing endpoint (GET /workspaces)
    @app.get("/workspaces", response_model=WorkspacesResponse)
    async def list_workspaces():
        """List all registered workspaces."""
        workspaces = registry.get_workspaces()
        return WorkspacesResponse(
            workspaces=[
                WorkspaceInfo(
                    name=w["name"],
                    first_seen=w["first_seen"],
                    last_seen=w["last_seen"],
                )
                for w in workspaces
            ]
        )

    # Add a mock document endpoint that simulates auto-register behavior
    @app.post("/test/documents")
    async def mock_document_endpoint(request: Request):
        """Mock document endpoint that simulates workspace auto-registration."""
        workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
        if workspace:
            registry.register_workspace(workspace)
        return JSONResponse(
            content={"status": "ok", "workspace": workspace or "default"}
        )

    @app.post("/test/texts")
    async def mock_insert_texts_endpoint(request: Request):
        """Mock text insertion endpoint that simulates workspace auto-registration."""
        workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
        if workspace:
            registry.register_workspace(workspace)
        return JSONResponse(
            content={"status": "ok", "workspace": workspace or "default"}
        )

    return app, registry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
async def test_app():
    """Create a test app with temporary directory for registry storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        app, registry = create_test_app(working_dir=tmp_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, registry, Path(tmp_dir)


# =============================================================================
# Test 1: GET /workspaces returns empty list when no workspaces registered
# =============================================================================


@pytest.mark.offline
class TestGetWorkspacesEmpty:
    """Tests for GET /workspaces with no registered workspaces."""

    @pytest.mark.asyncio
    async def test_empty_workspaces_returns_empty_list(self, test_app):
        """Test that GET /workspaces returns empty list when no workspaces registered."""
        ac, registry, _ = test_app

        response = await ac.get("/workspaces")

        assert response.status_code == 200
        data = response.json()
        assert "workspaces" in data
        assert data["workspaces"] == []

    @pytest.mark.asyncio
    async def test_empty_workspaces_response_is_valid_json(self, test_app):
        """Test that response is valid JSON with expected structure."""
        ac, registry, _ = test_app

        response = await ac.get("/workspaces")

        assert response.status_code == 200
        # Verify it's valid JSON
        data = response.json()
        assert isinstance(data, dict)
        assert "workspaces" in data
        assert isinstance(data["workspaces"], list)


# =============================================================================
# Test 2: GET /workspaces returns registered workspaces with correct fields
# =============================================================================


@pytest.mark.offline
class TestGetWorkspacesWithData:
    """Tests for GET /workspaces with registered workspaces."""

    @pytest.mark.asyncio
    async def test_workspace_has_required_fields(self, test_app):
        """Test that registered workspace has all required fields."""
        ac, registry, _ = test_app

        # Register a workspace directly
        registry.register_workspace("test-workspace")

        response = await ac.get("/workspaces")

        assert response.status_code == 200
        data = response.json()
        assert len(data["workspaces"]) == 1

        workspace = data["workspaces"][0]
        assert "name" in workspace
        assert "first_seen" in workspace
        assert "last_seen" in workspace
        assert workspace["name"] == "test-workspace"

    @pytest.mark.asyncio
    async def test_workspace_timestamps_are_iso_format(self, test_app):
        """Test that timestamps are in ISO format."""
        ac, registry, _ = test_app

        registry.register_workspace("test-workspace")

        response = await ac.get("/workspaces")

        data = response.json()
        workspace = data["workspaces"][0]

        # Check that timestamps look like ISO format (contain T and timezone)
        assert "T" in workspace["first_seen"]
        assert "T" in workspace["last_seen"]
        # Should contain timezone offset (+00:00) or Z
        assert "+" in workspace["first_seen"] or workspace["first_seen"].endswith("Z")
        assert "+" in workspace["last_seen"] or workspace["last_seen"].endswith("Z")

    @pytest.mark.asyncio
    async def test_multiple_workspaces_all_appear(self, test_app):
        """Test that multiple registered workspaces all appear in response."""
        ac, registry, _ = test_app

        # Register multiple workspaces
        registry.register_workspace("workspace-a")
        registry.register_workspace("workspace-b")
        registry.register_workspace("workspace-c")

        response = await ac.get("/workspaces")

        assert response.status_code == 200
        data = response.json()
        assert len(data["workspaces"]) == 3

        workspace_names = {w["name"] for w in data["workspaces"]}
        assert workspace_names == {"workspace-a", "workspace-b", "workspace-c"}

    @pytest.mark.asyncio
    async def test_workspaces_sorted_by_last_seen_descending(self, test_app):
        """Test that workspaces are sorted by last_seen (most recent first)."""
        ac, registry, _ = test_app

        # Register workspaces in order
        registry.register_workspace("first")
        registry.register_workspace("second")
        registry.register_workspace("third")

        response = await ac.get("/workspaces")

        data = response.json()
        names = [w["name"] for w in data["workspaces"]]
        # Most recently seen (third) should be first
        assert names[0] == "third"
        assert names[1] == "second"
        assert names[2] == "first"


# =============================================================================
# Test 3: Auto-register on document API calls
# =============================================================================


@pytest.mark.offline
class TestAutoRegister:
    """Tests for auto-register mechanism via document API calls."""

    @pytest.mark.asyncio
    async def test_document_call_with_header_registers_workspace(self, test_app):
        """Test that calling document endpoint with header auto-registers workspace."""
        ac, registry, _ = test_app

        # Call mock document endpoint with workspace header
        response = await ac.post(
            "/test/documents", headers={"LIGHTRAG-WORKSPACE": "new-workspace"}
        )

        assert response.status_code == 200

        # Verify workspace was registered
        response = await ac.get("/workspaces")
        data = response.json()
        assert len(data["workspaces"]) == 1
        assert data["workspaces"][0]["name"] == "new-workspace"

    @pytest.mark.asyncio
    async def test_texts_endpoint_also_registers_workspace(self, test_app):
        """Test that texts insertion endpoint also auto-registers workspace."""
        ac, registry, _ = test_app

        # Call mock texts endpoint with workspace header
        response = await ac.post(
            "/test/texts", headers={"LIGHTRAG-WORKSPACE": "text-workspace"}
        )

        assert response.status_code == 200

        # Verify workspace was registered
        response = await ac.get("/workspaces")
        data = response.json()
        assert len(data["workspaces"]) == 1
        assert data["workspaces"][0]["name"] == "text-workspace"

    @pytest.mark.asyncio
    async def test_same_workspace_updates_last_seen(self, test_app):
        """Test that same workspace called again updates last_seen but keeps first_seen."""
        ac, registry, _ = test_app

        # Register workspace first time
        response1 = await ac.post(
            "/test/documents", headers={"LIGHTRAG-WORKSPACE": "my-workspace"}
        )
        assert response1.status_code == 200

        # Get the first_seen and last_seen
        response = await ac.get("/workspaces")
        data = response.json()
        first_seen_before = data["workspaces"][0]["first_seen"]
        last_seen_before = data["workspaces"][0]["last_seen"]

        # Wait a tiny bit to ensure timestamp difference
        import asyncio

        await asyncio.sleep(0.01)

        # Register same workspace again
        response2 = await ac.post(
            "/test/documents", headers={"LIGHTRAG-WORKSPACE": "my-workspace"}
        )
        assert response2.status_code == 200

        # Verify first_seen is preserved but last_seen is updated
        response = await ac.get("/workspaces")
        data = response.json()
        assert len(data["workspaces"]) == 1  # Still only one workspace
        assert data["workspaces"][0]["name"] == "my-workspace"
        assert data["workspaces"][0]["first_seen"] == first_seen_before
        assert data["workspaces"][0]["last_seen"] >= last_seen_before

    @pytest.mark.asyncio
    async def test_multiple_different_workspaces_registered_correctly(self, test_app):
        """Test that multiple different workspaces are each registered correctly."""
        ac, registry, _ = test_app

        workspaces = ["ws-alpha", "ws-beta", "ws-gamma"]

        for ws in workspaces:
            response = await ac.post(
                "/test/documents", headers={"LIGHTRAG-WORKSPACE": ws}
            )
            assert response.status_code == 200

        # Verify all workspaces are registered
        response = await ac.get("/workspaces")
        data = response.json()
        assert len(data["workspaces"]) == 3

        registered_names = {w["name"] for w in data["workspaces"]}
        assert registered_names == set(workspaces)


# =============================================================================
# Test 4: Edge cases
# =============================================================================


@pytest.mark.offline
class TestEdgeCases:
    """Tests for edge cases in workspace registry."""

    @pytest.mark.asyncio
    async def test_missing_workspace_header(self, test_app):
        """Test that missing workspace header doesn't register anything."""
        ac, registry, _ = test_app

        # Call without workspace header
        response = await ac.post("/test/documents")

        assert response.status_code == 200
        assert response.json()["workspace"] == "default"

        # Verify no workspace was registered
        response = await ac.get("/workspaces")
        data = response.json()
        assert data["workspaces"] == []

    @pytest.mark.asyncio
    async def test_empty_workspace_header(self, test_app):
        """Test that empty workspace header value doesn't register anything."""
        ac, registry, _ = test_app

        # Call with empty workspace header
        response = await ac.post("/test/documents", headers={"LIGHTRAG-WORKSPACE": ""})

        assert response.status_code == 200

        # Verify no workspace was registered (empty string is not registered)
        response = await ac.get("/workspaces")
        data = response.json()
        assert data["workspaces"] == []

    @pytest.mark.asyncio
    async def test_special_characters_in_workspace_name(self, test_app):
        """Test workspace names with special characters are handled correctly."""
        ac, registry, _ = test_app

        # Register workspace with underscore and numbers (valid)
        registry.register_workspace("workspace_123")

        response = await ac.get("/workspaces")
        data = response.json()

        assert len(data["workspaces"]) == 1
        assert data["workspaces"][0]["name"] == "workspace_123"

    @pytest.mark.asyncio
    async def test_workspace_name_case_preserved(self, test_app):
        """Test that workspace names preserve their case."""
        ac, registry, _ = test_app

        # Register workspace with mixed case
        registry.register_workspace("MyWorkspace")

        response = await ac.get("/workspaces")
        data = response.json()

        assert len(data["workspaces"]) == 1
        # The name should be preserved as registered
        assert data["workspaces"][0]["name"] == "MyWorkspace"


# =============================================================================
# Test 5: Registry persistence
# =============================================================================


@pytest.mark.offline
class TestRegistryPersistence:
    """Tests for workspace registry persistence."""

    @pytest.mark.asyncio
    async def test_registry_persists_to_disk(self, test_app):
        """Test that registry is persisted to disk."""
        ac, registry, tmp_dir = test_app

        # Register a workspace
        registry.register_workspace("persistent-workspace")

        # Verify the registry file exists
        registry_file = Path(tmp_dir) / "workspace_registry.json"
        assert registry_file.exists()

        # Read and verify content
        with open(registry_file, "r") as f:
            data = json.load(f)

        assert "persistent-workspace" in data
        assert data["persistent-workspace"]["name"] == "persistent-workspace"
        assert "first_seen" in data["persistent-workspace"]
        assert "last_seen" in data["persistent-workspace"]


# =============================================================================
# Test 6: Response format validation
# =============================================================================


@pytest.mark.offline
class TestResponseFormat:
    """Tests for response format validation."""

    @pytest.mark.asyncio
    async def test_response_structure_matches_schema(self, test_app):
        """Test that response structure matches the WorkspacesResponse schema."""
        ac, registry, _ = test_app

        # Register a workspace
        registry.register_workspace("schema-test")

        response = await ac.get("/workspaces")

        assert response.status_code == 200
        data = response.json()

        # Verify top-level structure
        assert isinstance(data, dict)
        assert "workspaces" in data

        # Verify workspaces array
        assert isinstance(data["workspaces"], list)

        # Verify workspace object structure
        if data["workspaces"]:
            workspace = data["workspaces"][0]
            assert isinstance(workspace.get("name"), str)
            assert isinstance(workspace.get("first_seen"), str)
            assert isinstance(workspace.get("last_seen"), str)

    @pytest.mark.asyncio
    async def test_response_is_json_serializable(self, test_app):
        """Test that response is valid JSON and can be serialized."""
        ac, registry, _ = test_app

        registry.register_workspace("json-test")

        response = await ac.get("/workspaces")

        # Get raw content to ensure it's valid JSON
        content = response.content

        # Should be able to parse it as JSON without error
        data = json.loads(content)
        assert isinstance(data, dict)

        # Should be able to serialize it back to JSON
        serialized = json.dumps(data)
        assert serialized is not None
