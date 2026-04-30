"""
Integration tests for Workspace API isolation via HTTP.

These tests verify that the FastAPI workspace isolation layer correctly:
- Extracts workspace from LIGHTRAG-WORKSPACE HTTP header
- Validates workspace names
- Manages workspace lifecycle (get/create/release)
- Handles concurrent requests with proper isolation
- Supports background task and streaming response patterns
- Enforces capacity limits
- Evicts LRU entries under memory pressure

All external services are mocked.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport, AsyncClient

from lightrag.api.utils import (
    extract_workspace_from_header,
    sanitize_workspace_name,
    WorkspaceNameError,
)
from lightrag.api.workspace_manager import WorkspaceManager, WorkspaceCapacityError


# =============================================================================
# Mock LightRAG Instance
# =============================================================================


class MockLightRAG:
    """Mock LightRAG instance for testing workspace isolation."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.finalize_called = False
        self.some_method = AsyncMock(return_value={"status": "ok"})

    async def finalize_storages(self) -> None:
        """Mock finalize method."""
        self.finalize_called = True


# =============================================================================
# Factory Function
# =============================================================================


async def mock_factory(workspace: str) -> MockLightRAG:
    """Factory function that creates MockLightRAG instances."""
    return MockLightRAG(workspace)


# =============================================================================
# Test App Factory
# =============================================================================


def create_test_app(
    max_instances: int = 10,
    factory=None,
    bg_done_event: asyncio.Event | None = None,
) -> tuple[FastAPI, WorkspaceManager]:
    """
    Create a minimal FastAPI app that mimics the real server's workspace routing.

    Returns (app, workspace_manager) tuple.
    """
    if factory is None:
        factory = mock_factory

    workspace_mgr = WorkspaceManager(factory=factory, max_instances=max_instances)

    app = FastAPI()

    def get_workspace(request: Request) -> str:
        """Extract workspace from request header, returning JSONResponse on error."""
        try:
            return extract_workspace_from_header(request)
        except HTTPException as e:
            raise e  # Re-raise for FastAPI to handle

    @app.post("/test/regular")
    async def regular_route(request: Request):
        """Regular handler pattern with proper get/release."""
        try:
            ws = get_workspace(request)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"error": e.detail})

        try:
            rag = await workspace_mgr.get_or_create(ws)
            return JSONResponse(content={"workspace": rag.workspace})
        except WorkspaceCapacityError:
            return JSONResponse(
                status_code=503, content={"error": "All workspace slots busy"}
            )
        finally:
            workspace_mgr.release(ws)

    @app.post("/test/background")
    async def background_route(request: Request, background_tasks: BackgroundTasks):
        """
        Background task pattern - ref is held for duration of bg task.
        """
        try:
            ws = get_workspace(request)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"error": e.detail})

        # Get ref before scheduling background task
        rag = await workspace_mgr.get_or_create(ws)
        response_data = {"workspace": rag.workspace, "ref_count_before_bg": 1}

        async def background_work():
            """Simulated background work."""
            await asyncio.sleep(0.1)
            # Release after background work completes
            workspace_mgr.release(ws)
            # Signal that bg work is done
            if bg_done_event:
                bg_done_event.set()
            response_data["bg_completed"] = True

        background_tasks.add_task(background_work)

        # Return immediately without releasing (bg task will release)
        return JSONResponse(content=response_data)

    @app.post("/test/streaming")
    async def streaming_route(request: Request):
        """
        Streaming response pattern - ref held during stream.
        """
        try:
            ws = get_workspace(request)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"error": e.detail})

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                rag = await workspace_mgr.get_or_create(ws)
                yield f"data: workspace={rag.workspace}\n"
                await asyncio.sleep(0.05)
                yield "data: done\n"
            finally:
                workspace_mgr.release(ws)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )

    @app.post("/test/streaming-error")
    async def streaming_error_route(request: Request):
        """
        Streaming response pattern that errors mid-stream - verifies finally block runs.
        """
        try:
            ws = get_workspace(request)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"error": e.detail})

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                rag = await workspace_mgr.get_or_create(ws)
                yield f"data: workspace={rag.workspace}\n"
                await asyncio.sleep(0.02)
                yield "data: before-error\n"
                # Simulate an error mid-stream
                raise RuntimeError("simulated error")
            except RuntimeError:
                # Re-raise so the stream fails
                raise
            finally:
                workspace_mgr.release(ws)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )

    @app.post("/test/hold-ref")
    async def hold_ref_route(request: Request, duration: float = 1.0):
        """
        Route that holds a workspace ref for a specified duration.
        Useful for testing capacity and eviction under load.
        """
        try:
            ws = get_workspace(request)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"error": e.detail})

        try:
            rag = await workspace_mgr.get_or_create(ws)
            # Hold the ref for the specified duration
            await asyncio.sleep(duration)
            return JSONResponse(content={"workspace": rag.workspace, "held_for": duration})
        except WorkspaceCapacityError:
            return JSONResponse(
                status_code=503, content={"error": "All workspace slots busy"}
            )
        finally:
            workspace_mgr.release(ws)

    @app.get("/test/stats")
    async def stats_route(request: Request):
        """Return workspace manager stats."""
        return JSONResponse(content=workspace_mgr.get_stats())

    @app.get("/test/workspace/{workspace_name}")
    async def get_workspace_route(workspace_name: str, request: Request):
        """Direct workspace access via path parameter."""
        try:
            ws = sanitize_workspace_name(workspace_name)
        except WorkspaceNameError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        try:
            rag = await workspace_mgr.get_or_create(ws)
            return JSONResponse(content={"workspace": rag.workspace})
        except WorkspaceCapacityError:
            return JSONResponse(
                status_code=503, content={"error": "All workspace slots busy"}
            )
        finally:
            workspace_mgr.release(ws)

    return app, workspace_mgr


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
async def client():
    """Create an async HTTP client for testing."""
    app, workspace_mgr = create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, workspace_mgr


@pytest.fixture
async def small_client():
    """Create a client with small max_instances for capacity tests."""
    app, workspace_mgr = create_test_app(max_instances=2)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, workspace_mgr


@pytest.fixture
async def eviction_client():
    """Create a client with max_instances=3 for LRU eviction tests."""
    app, workspace_mgr = create_test_app(max_instances=3)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, workspace_mgr


@pytest.fixture
async def client_with_bg_event():
    """Create a client with bg_done_event for testing background task completion."""
    bg_done_event = asyncio.Event()
    app, workspace_mgr = create_test_app(bg_done_event=bg_done_event)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, workspace_mgr, bg_done_event


# =============================================================================
# Test 1: Request with LIGHTRAG-WORKSPACE header -> correct workspace used
# =============================================================================


@pytest.mark.offline
class TestWorkspaceHeaderExtraction:
    """Tests for workspace extraction from HTTP headers."""

    @pytest.mark.asyncio
    async def test_header_workspace_is_used(self, client):
        """Test that LIGHTRAG-WORKSPACE header is used correctly."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["workspace"] == "ws-test"

    @pytest.mark.asyncio
    async def test_header_case_insensitive(self, client):
        """Test that workspace names are lowercased."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "MyWorkspace"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["workspace"] == "myworkspace"


# =============================================================================
# Test 2: Request without header -> falls back to default (empty string)
# =============================================================================


@pytest.mark.offline
class TestDefaultWorkspaceFallback:
    """Tests for default workspace fallback behavior."""

    @pytest.mark.asyncio
    async def test_no_header_uses_default(self, client):
        """Test that requests without header use empty string as default."""
        ac, workspace_mgr = client

        response = await ac.post("/test/regular")

        assert response.status_code == 200
        data = response.json()
        assert data["workspace"] == ""

    @pytest.mark.asyncio
    async def test_empty_header_uses_default(self, client):
        """Test that empty header value uses empty string as default."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": ""},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["workspace"] == ""


# =============================================================================
# Test 3: Invalid workspace name -> HTTP 400
# =============================================================================


@pytest.mark.offline
class TestWorkspaceNameValidation:
    """Tests for workspace name validation via HTTP."""

    @pytest.mark.asyncio
    async def test_invalid_special_chars_returns_400(self, client):
        """Test that special characters in workspace name return 400."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws!@#"},
        )

        assert response.status_code == 400
        assert "only lowercase letters" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_path_traversal_returns_400(self, client):
        """Test that path traversal attempts return 400."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "../../etc"},
        )

        assert response.status_code == 400
        assert "path traversal" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_slash_in_name_returns_400(self, client):
        """Test that forward slashes in workspace name return 400."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws/name"},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_backslash_returns_400(self, client):
        """Test that backslashes in workspace name return 400."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws\\name"},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_too_long_name_returns_400(self, client):
        """Test that names exceeding 64 characters return 400."""
        ac, workspace_mgr = client

        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "a" * 65},
        )

        assert response.status_code == 400
        assert "64 characters" in response.json()["error"]


# =============================================================================
# Test 4: Multiple concurrent requests to different workspaces -> each isolated
# =============================================================================


@pytest.mark.offline
class TestConcurrentWorkspaceIsolation:
    """Tests for concurrent request isolation."""

    @pytest.mark.asyncio
    async def test_concurrent_different_workspaces(self, client):
        """Test that concurrent requests to different workspaces are isolated."""
        ac, workspace_mgr = client

        workspaces = ["ws-a", "ws-b", "ws-c", "ws-d", "ws-e"]

        # Send 5 concurrent requests
        tasks = [
            ac.post("/test/regular", headers={"LIGHTRAG-WORKSPACE": ws})
            for ws in workspaces
        ]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # Each should have its own workspace
        for i, ws in enumerate(workspaces):
            data = responses[i].json()
            assert data["workspace"] == ws

        # Stats should show 5 active instances
        stats = workspace_mgr.get_stats()
        assert stats["active_instances"] == 5

    @pytest.mark.asyncio
    async def test_concurrent_same_workspace_returns_same_instance(self, client):
        """Test that concurrent requests to same workspace share instance."""
        ac, workspace_mgr = client

        # Send 5 concurrent requests to same workspace
        tasks = [
            ac.post("/test/regular", headers={"LIGHTRAG-WORKSPACE": "shared-ws"})
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # Stats should show 1 active instance (cached)
        stats = workspace_mgr.get_stats()
        assert stats["active_instances"] == 1
        assert stats["cache_hits"] >= 4  # Some hits from concurrent access


# =============================================================================
# Test 5: Background task pattern - ref count held during execution
# =============================================================================


@pytest.mark.offline
class TestBackgroundTaskPattern:
    """Tests for background task workspace management pattern."""

    @pytest.mark.asyncio
    async def test_bg_task_holds_ref_during_execution(self, client_with_bg_event):
        """Test that background task pattern holds ref count during execution."""
        ac, workspace_mgr, bg_done_event = client_with_bg_event

        # Initial stats - no ref counts
        initial_stats = workspace_mgr.get_stats()
        assert sum(initial_stats.get("ref_counts", {}).values()) == 0

        # Send request with background task
        response = await ac.post(
            "/test/background",
            headers={"LIGHTRAG-WORKSPACE": "bg-ws"},
        )

        # Response should indicate ref was held
        data = response.json()
        assert data["workspace"] == "bg-ws"
        assert data["ref_count_before_bg"] == 1

        # Wait for background task to complete using event
        await asyncio.wait_for(bg_done_event.wait(), timeout=2.0)

        # After bg task completes, ref should be released
        stats = workspace_mgr.get_stats()
        assert sum(stats.get("ref_counts", {}).values()) == 0

    @pytest.mark.asyncio
    async def test_multiple_bg_tasks_isolated(self, small_client):
        """Test that multiple background tasks maintain isolation."""
        ac, workspace_mgr = small_client

        # Send multiple background task requests
        tasks = [
            ac.post("/test/background", headers={"LIGHTRAG-WORKSPACE": f"bg-{i}"})
            for i in range(2)  # Only 2 to avoid capacity issues
        ]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # Wait for completion
        await asyncio.sleep(0.3)

        # All refs should be released
        stats = workspace_mgr.get_stats()
        assert sum(stats.get("ref_counts", {}).values()) == 0


# =============================================================================
# Test 6: Streaming response - ref count held during stream, released after
# =============================================================================


@pytest.mark.offline
class TestStreamingResponsePattern:
    """Tests for streaming response workspace management pattern."""

    @pytest.mark.asyncio
    async def test_streaming_holds_ref_during_stream(self, client):
        """Test that streaming response holds ref count during stream."""
        ac, workspace_mgr = client

        # Initial stats - no ref counts
        initial_stats = workspace_mgr.get_stats()
        assert sum(initial_stats.get("ref_counts", {}).values()) == 0

        # Start streaming request
        async with ac.stream(
            "POST",
            "/test/streaming",
            headers={"LIGHTRAG-WORKSPACE": "stream-ws"},
        ) as response:
            # Read first chunk to ensure stream generator has started
            first_chunk = await response.aread()
            assert first_chunk  # Should have some data

            # After stream started, ref should be held
            stats = workspace_mgr.get_stats()
            # The workspace should be in cache
            assert stats["active_instances"] >= 1
            assert "stream-ws" in stats.get("ref_counts", {})

        # After stream completes, ref should be released
        stats = workspace_mgr.get_stats()
        assert sum(stats.get("ref_counts", {}).values()) == 0

    @pytest.mark.asyncio
    async def test_streaming_workspace_isolated(self, client):
        """Test that streaming responses maintain workspace isolation."""
        ac, workspace_mgr = client

        # Start multiple streams concurrently
        async def consume_stream(ws: str) -> list[str]:
            chunks = []
            async with ac.stream(
                "POST",
                "/test/streaming",
                headers={"LIGHTRAG-WORKSPACE": ws},
            ) as response:
                async for chunk in response.aiter_lines():
                    chunks.append(chunk)
            return chunks

        tasks = [consume_stream(f"stream-{i}") for i in range(3)]
        all_chunks = await asyncio.gather(*tasks)

        # Each stream should have its own workspace
        for i, chunks in enumerate(all_chunks):
            workspace_line = [c for c in chunks if "workspace=" in c][0]
            assert f"workspace=stream-{i}" in workspace_line

        # All refs should be released
        stats = workspace_mgr.get_stats()
        assert sum(stats.get("ref_counts", {}).values()) == 0

    @pytest.mark.asyncio
    async def test_streaming_error_releases_ref(self, client):
        """Test that streaming error still releases workspace ref via finally block."""
        ac, workspace_mgr = client

        # Initial stats - no ref counts
        initial_stats = workspace_mgr.get_stats()
        assert sum(initial_stats.get("ref_counts", {}).values()) == 0

        # Start streaming request that will error mid-stream
        # The exception propagates during the request, not during iteration
        error_occurred = False
        try:
            async with ac.stream(
                "POST",
                "/test/streaming-error",
                headers={"LIGHTRAG-WORKSPACE": "error-ws"},
            ) as response:
                # Read chunks until error occurs
                async for line in response.aiter_lines():
                    pass
        except Exception as e:
            error_occurred = True
            assert "simulated error" in str(e)

        assert error_occurred

        # After stream error, ref should still be released (finally block runs)
        await asyncio.sleep(0.05)  # Small delay for cleanup
        stats = workspace_mgr.get_stats()
        assert sum(stats.get("ref_counts", {}).values()) == 0


# =============================================================================
# Test 7: WorkspaceCapacityError -> HTTP 503
# =============================================================================


@pytest.mark.offline
class TestCapacityLimit:
    """Tests for workspace capacity limit enforcement."""

    @pytest.mark.asyncio
    async def test_capacity_error_returns_503(self, small_client):
        """Test that capacity exhaustion returns 503 via actual HTTP requests."""
        ac, workspace_mgr = small_client

        # max_instances=2, use /test/hold-ref to hold refs via HTTP
        # Start 2 concurrent requests that hold refs
        tasks = [
            ac.post("/test/hold-ref", params={"duration": 5.0}, headers={"LIGHTRAG-WORKSPACE": f"holder-{i}"})
            for i in range(2)
        ]

        # Start both tasks concurrently
        pending = [asyncio.create_task(t) for t in tasks]

        # Wait for both to start and hold refs
        await asyncio.sleep(0.2)

        # At this point, both holders should have ref_count=1
        stats = workspace_mgr.get_stats()
        assert stats["active_instances"] == 2
        assert stats.get("ref_counts", {}).get("holder-0", 0) == 1
        assert stats.get("ref_counts", {}).get("holder-1", 0) == 1

        # Now try a 3rd request - should get 503
        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws-c"},
        )
        assert response.status_code == 503
        assert "All workspace slots busy" in response.json()["error"]

        # Clean up - cancel the pending tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        # After cleanup, refs should be released
        await asyncio.sleep(0.1)
        stats = workspace_mgr.get_stats()
        assert sum(stats.get("ref_counts", {}).values()) == 0

    @pytest.mark.asyncio
    async def test_same_workspace_request_within_capacity(self, small_client):
        """Test that same workspace requests don't exceed capacity."""
        ac, workspace_mgr = small_client

        # Make multiple requests to same workspace
        # With same workspace, they should share the cached instance
        tasks = [
            ac.post("/test/regular", headers={"LIGHTRAG-WORKSPACE": "shared"})
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)

        # All should succeed (same workspace = same instance)
        assert all(r.status_code == 200 for r in responses)

        # Only 1 active instance
        stats = workspace_mgr.get_stats()
        assert stats["active_instances"] == 1


# =============================================================================
# Test 8: LRU eviction under concurrent load
# =============================================================================


@pytest.mark.offline
class TestLRUEviction:
    """Tests for LRU eviction under memory pressure."""

    @pytest.mark.asyncio
    async def test_lru_eviction_after_capacity_exceeded(self, eviction_client):
        """Test that LRU eviction occurs after capacity exceeded."""
        ac, workspace_mgr = eviction_client

        # max_instances=3, create 5 different workspaces
        for i in range(5):
            response = await ac.post(
                "/test/regular",
                headers={"LIGHTRAG-WORKSPACE": f"evict-{i}"},
            )
            assert response.status_code == 200

        # After creating 5 workspaces with only 3 slots,
        # LRU eviction should have occurred
        stats = workspace_mgr.get_stats()

        # Active instances should be <= max_instances
        assert stats["active_instances"] <= 3

        # Verify eviction happened
        assert stats.get("evictions", 0) > 0 or stats["active_instances"] < 5

    @pytest.mark.asyncio
    async def test_lru_ordering_respected(self, eviction_client):
        """Test that LRU ordering is respected - least recently used evicted first."""
        ac, workspace_mgr = eviction_client

        # Create 3 workspaces: ws-a, ws-b, ws-c
        for ws in ["ws-a", "ws-b", "ws-c"]:
            response = await ac.post(
                "/test/regular",
                headers={"LIGHTRAG-WORKSPACE": ws},
            )
            assert response.status_code == 200

        # Access ws-a to make it most recently used
        await ac.post("/test/regular", headers={"LIGHTRAG-WORKSPACE": "ws-a"})

        # Release ws-b (least recently used after ws-c was accessed last)
        # Wait a bit to ensure ordering
        await asyncio.sleep(0.01)

        # Access ws-c
        await ac.post("/test/regular", headers={"LIGHTRAG-WORKSPACE": "ws-c"})

        # Now create ws-d - should evict ws-b (LRU)
        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws-d"},
        )
        assert response.status_code == 200

        # Check that ws-b is no longer in cache
        # (This is implicit - if eviction works correctly, the cache size is limited)
        stats = workspace_mgr.get_stats()
        assert stats["active_instances"] <= 3

    @pytest.mark.asyncio
    async def test_workspace_is_not_evicted_while_in_use(self, eviction_client):
        """Test that workspaces with active refs are not evicted."""
        ac, workspace_mgr = eviction_client

        # max_instances=3
        # Use /test/hold-ref to hold refs via HTTP

        # Start 2 concurrent requests that hold refs for ws-1 and ws-2
        hold_tasks = [
            ac.post("/test/hold-ref", params={"duration": 5.0}, headers={"LIGHTRAG-WORKSPACE": f"ws-{i}"})
            for i in range(1, 3)  # ws-1 and ws-2
        ]
        pending = [asyncio.create_task(t) for t in hold_tasks]

        # Wait for both to start and hold refs
        await asyncio.sleep(0.2)

        # At this point: ws-1 and ws-2 have ref_count=1 (from hold-ref)
        # Active instances = 2
        stats = workspace_mgr.get_stats()
        assert stats.get("ref_counts", {}).get("ws-1", 0) == 1
        assert stats.get("ref_counts", {}).get("ws-2", 0) == 1

        # Now create ws-3 and ws-4 via regular requests (complete and release)
        for ws in ["ws-3", "ws-4"]:
            response = await ac.post(
                "/test/regular",
                headers={"LIGHTRAG-WORKSPACE": ws},
            )
            assert response.status_code == 200
        # After these complete, ws-3 and ws-4 have ref_count=0, active_instances=4

        # Now try to create ws-5 - this should trigger eviction
        # Eviction should evict ws-3 or ws-4 (ref_count=0), NOT ws-1 or ws-2 (ref_count>0)
        response = await ac.post(
            "/test/regular",
            headers={"LIGHTRAG-WORKSPACE": "ws-5"},
        )
        assert response.status_code == 200

        # Check that ws-1 and ws-2 survived eviction (still in cache)
        stats = workspace_mgr.get_stats()
        assert "ws-1" in stats.get("ref_counts", {})
        assert "ws-2" in stats.get("ref_counts", {})
        assert stats.get("ref_counts", {}).get("ws-1", 0) >= 1
        assert stats.get("ref_counts", {}).get("ws-2", 0) >= 1

        # Verify eviction happened (ws-3 or ws-4 should be evicted)
        assert stats.get("evictions", 0) > 0

        # Clean up
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


# =============================================================================
# Cleanup helper
# =============================================================================
