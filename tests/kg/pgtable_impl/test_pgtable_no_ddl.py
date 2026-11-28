"""Graph No-DDL initialization validates without normalizing or provisioning."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from .test_pgtable_graph_storage import fake_init_lock, make_uninitialized_storage

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("failure", [None, RuntimeError("foreign key missing")])
@pytest.mark.asyncio
async def test_graph_validates_without_ddl_and_releases_failed_hold(
    monkeypatch, failure
):
    from lightrag.kg.postgres_impl import ClientManager

    storage = make_uninitialized_storage()
    db = MagicMock(auto_migrate=False, workspace="resolved_workspace")
    db.execute = AsyncMock()
    get_client = AsyncMock(return_value=db)
    release = AsyncMock()
    validator = AsyncMock(side_effect=failure)
    normalize = AsyncMock()
    monkeypatch.setattr(ClientManager, "get_client", get_client)
    monkeypatch.setattr(ClientManager, "release_client", release)
    monkeypatch.setattr("lightrag.kg.pgtable_impl.get_data_init_lock", fake_init_lock)
    monkeypatch.setattr("lightrag.kg.postgres_schema.validate_graph_tables", validator)
    monkeypatch.setattr(storage, "_normalize_legacy_edges", normalize)
    if failure:
        with pytest.raises(RuntimeError, match="foreign key missing"):
            await storage.initialize()
        release.assert_awaited_once_with(db)
        assert storage.db is None
    else:
        await storage.initialize()
        assert storage.db is db
        release.assert_not_awaited()
    validator.assert_awaited_once_with(db, "resolved_workspace", storage.namespace)
    assert storage.workspace == "resolved_workspace"
    normalize.assert_not_awaited()
    db.execute.assert_not_awaited()
