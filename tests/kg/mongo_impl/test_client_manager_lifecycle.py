"""Tests for ClientManager connection lifecycle (no MongoDB instance required).

These tests verify that ClientManager properly closes the underlying
AsyncMongoClient when all references are released, and keeps it alive
while references remain.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo ClientManager tests",
)

from lightrag.kg.mongo_impl import ClientManager


class TestClientManagerLifecycle:
    """Verify ClientManager connection open/close behavior."""

    def _reset_manager(self):
        """Reset ClientManager class state between tests."""
        ClientManager._instances = {"client": None, "db": None, "ref_count": 0}

    def teardown_method(self):
        self._reset_manager()

    @pytest.mark.asyncio
    async def test_release_client_closes_connection_when_ref_count_zero(self):
        """When ref_count drops to 0, the MongoClient should be closed and cleared."""
        mock_client = AsyncMock()
        mock_db = MagicMock()

        ClientManager._instances = {
            "client": mock_client,
            "db": mock_db,
            "ref_count": 1,
        }

        await ClientManager.release_client(mock_db)

        mock_client.close.assert_awaited_once()
        assert ClientManager._instances["client"] is None
        assert ClientManager._instances["db"] is None
        assert ClientManager._instances["ref_count"] == 0

    @pytest.mark.asyncio
    async def test_release_client_keeps_connection_with_multiple_refs(self):
        """When other references exist, the MongoClient must NOT be closed."""
        mock_client = AsyncMock()
        mock_db = MagicMock()

        ClientManager._instances = {
            "client": mock_client,
            "db": mock_db,
            "ref_count": 3,
        }

        await ClientManager.release_client(mock_db)

        mock_client.close.assert_not_awaited()
        assert ClientManager._instances["ref_count"] == 2
        assert ClientManager._instances["client"] is mock_client
        assert ClientManager._instances["db"] is mock_db

    @pytest.mark.asyncio
    async def test_release_client_noop_for_wrong_db(self):
        """Releasing a db that is not the tracked instance should do nothing."""
        mock_client = AsyncMock()
        mock_db = MagicMock()
        other_db = MagicMock()

        ClientManager._instances = {
            "client": mock_client,
            "db": mock_db,
            "ref_count": 1,
        }

        await ClientManager.release_client(other_db)

        mock_client.close.assert_not_awaited()
        assert ClientManager._instances["ref_count"] == 1

    @pytest.mark.asyncio
    async def test_release_client_noop_for_none(self):
        """Releasing None should be a safe no-op."""
        mock_client = AsyncMock()
        mock_db = MagicMock()

        ClientManager._instances = {
            "client": mock_client,
            "db": mock_db,
            "ref_count": 1,
        }

        await ClientManager.release_client(None)

        mock_client.close.assert_not_awaited()
        assert ClientManager._instances["ref_count"] == 1
