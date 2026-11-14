"""``lightrag-rebuild-vdb`` and ``lightrag-clear-storage`` verify the
configuration container against the anchor, and never bind.

They take the shared anchor lock, refuse a backend type the anchor does not
bind before anything opens, and refuse an identity mismatch before any data
storage opens -- exactly as a start would. On a database backend with no
anchor they run as before and say nothing was verified; on JSON the workspace
must be a registered member of the group, or they refuse and advise starting
the server once. Neither ever creates or rewrites the anchor or
the identity row. See *Maintenance tools* under *The anchor and the container
identity* in docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.kg import anchor_lock as al
from lightrag.namespace import SERVER_SCOPE
from tests.config_store.test_config_store import FakeConfigKV
from tests.tools import test_clear_storage as clear_helpers
from tests.tools import test_rebuild_vdb as rebuild_helpers

pytestmark = pytest.mark.offline

UUID_A = "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b"
UUID_B = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _identity_rows(storage_uuid=UUID_A):
    return {
        cs.storage_identity_key(): cs.make_config_row(
            scope_workspace=SERVER_SCOPE,
            suffix=cs.STORAGE_IDENTITY_SUFFIX,
            value={"uuid": storage_uuid},
            updated_by="test",
        )
    }


def _anchor(
    working_dir, backend="PGKVStorage", storage_uuid=UUID_A, members=()
) -> bytes:
    """Publish an anchor; a JSON one carries ``members`` (a DB one none)."""
    ca.publish_anchor(
        str(working_dir),
        ca.StorageAnchor(
            backend=backend,
            storage_uuid=storage_uuid,
            members=members if backend == "JsonKVStorage" else None,
        ),
        replace=False,
    )
    return open(ca.anchor_path(str(working_dir)), "rb").read()


class _Config(FakeConfigKV):
    def __init__(self, rows=None):
        super().__init__(rows)
        self.initialize = AsyncMock()
        self.workspace = None


def _writes(config):
    return [c for c in config.calls if c[0] in ("upsert", "flush", "delete")]


# ---------------------------------------------------------------------------
# lightrag-rebuild-vdb
# ---------------------------------------------------------------------------


def _rebuild_tool(config):
    tool = rebuild_helpers._tool_with_storages(
        rebuild_helpers.MockVDB(), rebuild_helpers.MockVDB(), rebuild_helpers.MockVDB()
    )
    tool.configuration_storage = config
    tool.storage_names = {**tool.storage_names, "config": "PGKVStorage"}
    return tool


def _json_shard_rows(workspace="", storage_uuid=UUID_A, *, owner=True):
    rows = cs.json_shard_metadata_rows(workspace, storage_uuid, updated_by="test")
    if not owner:
        rows.pop(cs.json_shard_owner_key())
    return rows


class TestRebuildTool:
    async def test_no_anchor_runs_as_before_and_binds_nothing(
        self, monkeypatch, capsys
    ):
        working_dir = os.environ["WORKING_DIR"]
        config = _Config()
        tool = _rebuild_tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is True
        assert "was not verified" in capsys.readouterr().out
        assert ca.read_anchor(working_dir) is None
        assert _writes(config) == []
        assert al.holds_anchor_lock(working_dir)
        tool.release_anchor_lock()
        assert not al.holds_anchor_lock(working_dir)

    async def test_a_matching_identity_verifies_without_writing(
        self, monkeypatch, capsys
    ):
        working_dir = os.environ["WORKING_DIR"]
        before = _anchor(working_dir)
        config = _Config(_identity_rows())
        tool = _rebuild_tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is True
        assert f"{UUID_A} (matches the anchor)" in capsys.readouterr().out
        assert _writes(config) == []
        assert open(ca.anchor_path(working_dir), "rb").read() == before

    @pytest.mark.parametrize("rows", [{}, _identity_rows(UUID_B)])
    async def test_an_identity_mismatch_refuses_before_any_source_opens(
        self, monkeypatch, capsys, rows
    ):
        working_dir = os.environ["WORKING_DIR"]
        before = _anchor(working_dir)
        config = _Config(rows)
        tool = _rebuild_tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is False
        assert "Refusing to start" in capsys.readouterr().out
        tool.graph.initialize.assert_not_awaited()
        tool.text_chunks.initialize.assert_not_awaited()
        tool.entities_vdb.initialize.assert_not_awaited()
        assert _writes(config) == []
        assert open(ca.anchor_path(working_dir), "rb").read() == before

    async def test_a_backend_type_mismatch_refuses_before_anything_opens(
        self, monkeypatch, capsys
    ):
        working_dir = os.environ["WORKING_DIR"]
        _anchor(working_dir, backend="MongoKVStorage")
        config = _Config(_identity_rows())
        tool = _rebuild_tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is False
        assert "LIGHTRAG_CONFIG_STORAGE=MongoKVStorage" in capsys.readouterr().out
        config.initialize.assert_not_awaited()
        assert not al.holds_anchor_lock(working_dir)


class TestRebuildToolOnJson:
    """JSON: the workspace must be a registered member. A tool never binds or
    registers one, so no anchor, or an anchor that does not list the
    workspace, refuses with the advice to start the server once."""

    def _tool(self, config):
        tool = _rebuild_tool(config)
        tool.storage_names = {**tool.storage_names, "config": "JsonKVStorage"}
        return tool

    @pytest.mark.parametrize("anchored", [False, True], ids=["no_anchor", "other"])
    async def test_an_unregistered_workspace_refuses_before_any_source_opens(
        self, monkeypatch, capsys, anchored
    ):
        working_dir = os.environ["WORKING_DIR"]
        before = (
            _anchor(working_dir, backend="JsonKVStorage", members=("other",))
            if anchored
            else None
        )
        config = _Config()
        tool = self._tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is False
        out = capsys.readouterr().out
        assert "Start the server once to register this workspace" in out
        tool.graph.initialize.assert_not_awaited()
        tool.text_chunks.initialize.assert_not_awaited()
        tool.entities_vdb.initialize.assert_not_awaited()
        assert _writes(config) == []
        if anchored:
            assert open(ca.anchor_path(working_dir), "rb").read() == before
        else:
            assert ca.read_anchor(working_dir) is None

    async def test_a_registered_member_verifies_without_writing(
        self, monkeypatch, capsys
    ):
        working_dir = os.environ["WORKING_DIR"]
        before = _anchor(working_dir, backend="JsonKVStorage", members=("",))
        config = _Config(_json_shard_rows(""))
        tool = self._tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is True
        assert f"{UUID_A} (matches the anchor)" in capsys.readouterr().out
        assert _writes(config) == []
        assert open(ca.anchor_path(working_dir), "rb").read() == before

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param({}, id="snapshot_lost"),
            pytest.param(_json_shard_rows("", owner=False), id="no_owner"),
            pytest.param(_json_shard_rows("", UUID_B), id="foreign_uuid"),
        ],
    )
    async def test_a_damaged_member_refuses_without_the_first_start_advice(
        self, monkeypatch, capsys, rows
    ):
        """A registered but damaged snapshot needs restoring, not a first
        start -- and is never re-created in place."""
        working_dir = os.environ["WORKING_DIR"]
        before = _anchor(working_dir, backend="JsonKVStorage", members=("",))
        config = _Config(rows)
        tool = self._tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is False
        out = capsys.readouterr().out
        assert "Refusing to start" in out
        assert "Start the server once" not in out
        tool.graph.initialize.assert_not_awaited()
        assert _writes(config) == []
        assert open(ca.anchor_path(working_dir), "rb").read() == before

    async def test_the_workspace_is_normalized_as_the_server_normalizes_it(
        self, monkeypatch, capsys
    ):
        """``WORKSPACE=team-a`` is served, and registered, as ``team_a``; the
        tool must check that member and that snapshot, not ``team-a``'s."""
        working_dir = os.environ["WORKING_DIR"]
        _anchor(working_dir, backend="JsonKVStorage", members=("team_a",))
        monkeypatch.setenv("WORKSPACE", "team-a")
        config = _Config(_json_shard_rows("team_a"))
        tool = self._tool(config)
        assert await rebuild_helpers._setup_with(tool, monkeypatch) is True
        assert tool.workspace == "team_a"
        assert tool.config_dir == os.path.join(os.path.abspath(working_dir), "team_a")
        assert "Start the server once" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# lightrag-clear-storage
# ---------------------------------------------------------------------------


def _clear_tool(tmp_path, monkeypatch, config):
    tool = clear_helpers.ClearTool()
    fakes = clear_helpers.openable_fakes()
    names = {**clear_helpers.FILE_BACKED_NAMES, "config": "PGKVStorage"}
    clear_helpers.setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes, names=names)
    monkeypatch.setattr(
        clear_helpers.clear_storage,
        "create_configuration_storage",
        lambda *a, **k: config,
    )
    built = []
    real_build = tool.build_storage

    def _recording(label, embedding_func):
        built.append(label)
        return real_build(label, embedding_func)

    monkeypatch.setattr(tool, "build_storage", _recording)
    return tool, built


class TestClearTool:
    async def test_no_anchor_runs_as_before_and_binds_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        config = _Config()
        tool, built = _clear_tool(tmp_path, monkeypatch, config)
        assert await tool.setup_storages() is True
        assert "was not verified" in capsys.readouterr().out
        assert ca.read_anchor(str(tmp_path)) is None
        assert _writes(config) == []
        assert built
        tool.release_anchor_lock()

    async def test_a_matching_identity_verifies_without_writing(
        self, tmp_path, monkeypatch, capsys
    ):
        before = _anchor(tmp_path)
        config = _Config(_identity_rows())
        tool, _ = _clear_tool(tmp_path, monkeypatch, config)
        assert await tool.setup_storages() is True
        assert f"{UUID_A} (matches the anchor)" in capsys.readouterr().out
        assert _writes(config) == []
        assert open(ca.anchor_path(str(tmp_path)), "rb").read() == before
        tool.release_anchor_lock()

    @pytest.mark.parametrize("rows", [{}, _identity_rows(UUID_B)])
    async def test_an_identity_mismatch_refuses_before_any_data_storage_opens(
        self, tmp_path, monkeypatch, capsys, rows
    ):
        """A clear against a container this deployment is not bound to would
        delete another container's records."""
        _anchor(tmp_path)
        config = _Config(rows)
        tool, built = _clear_tool(tmp_path, monkeypatch, config)
        assert await tool.setup_storages() is False
        assert "Refusing to start" in capsys.readouterr().out
        assert built == []
        assert _writes(config) == []

    async def test_a_backend_type_mismatch_refuses_before_anything_opens(
        self, tmp_path, monkeypatch, capsys
    ):
        _anchor(tmp_path, backend="JsonKVStorage")
        config = _Config(_identity_rows())
        tool, built = _clear_tool(tmp_path, monkeypatch, config)
        assert await tool.setup_storages() is False
        assert "LIGHTRAG_CONFIG_STORAGE=JsonKVStorage" in capsys.readouterr().out
        config.initialize.assert_not_awaited()
        assert built == []
        assert not al.holds_anchor_lock(str(tmp_path))

    async def test_a_running_migration_refuses_the_tool(
        self, tmp_path, monkeypatch, capsys
    ):
        pytest.importorskip("fcntl")
        config = _Config()
        tool, built = _clear_tool(tmp_path, monkeypatch, config)
        lock = al.acquire_anchor_lock_exclusive(str(tmp_path))
        try:
            assert await tool.setup_storages() is False
        finally:
            lock.release()
        assert "migration" in capsys.readouterr().out
        config.initialize.assert_not_awaited()
