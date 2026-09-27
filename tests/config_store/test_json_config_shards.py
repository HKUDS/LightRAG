"""JSON configuration shards: one snapshot per workspace, one shared anchor.

Each workspace keeps its JSON configuration at a fixed path under
``WORKING_DIR``; the anchor directly under ``WORKING_DIR`` binds the group's
UUID and lists the registered members. A start registers its workspace once,
under the bind lock; a registered member's snapshot is never re-created; a
deleted anchor is rebuilt from the surviving consistent snapshots. See
*JSON configuration shards* in docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.config_shards import (
    RESERVED_WORKSPACE_NAMES,
    discover_shards,
    json_config_path,
)
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationIdentityError,
)
from lightrag.kg import anchor_lock as al
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

_DIM = 16
_PROBE = Path(__file__).with_name("_json_register_probe.py")


class _SimpleTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


async def _mock_llm(prompt, **kwargs):  # pragma: no cover - never called here
    return "mock"


async def _embed(texts, **kwargs):
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i][sum(bytearray(text.encode())) % _DIM] = 1.0
    return out


def _rag(working_dir, workspace: str) -> LightRAG:
    return LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        kv_storage="JsonKVStorage",
        config_storage="JsonKVStorage",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=4096,
            func=_embed,
            model_name="bge-m3",
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
    )


async def _start_and_stop(working_dir, workspace: str) -> None:
    rag = _rag(working_dir, workspace)
    await rag.initialize_storages()
    await rag.finalize_storages()


def _snapshot(working_dir, workspace: str) -> dict:
    with open(json_config_path(str(working_dir), workspace), encoding="utf-8") as f:
        return json.load(f)


def _anchor(working_dir) -> ca.StorageAnchor:
    anchor = ca.read_anchor(str(working_dir))
    assert anchor is not None
    return anchor


IDENTITY_KEY = "_lightrag_server/storage_identity"
OWNER_KEY = "_lightrag_server/json_shard"


# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------


class TestPaths:
    async def test_named_and_empty_workspaces_use_their_own_fixed_files(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        await _start_and_stop(tmp_path, "")

        assert (tmp_path / "config_storage_anchor.json").is_file()
        assert (tmp_path / "teamalpha" / "kv_workspace_config.json").is_file()
        assert (tmp_path / "kv_workspace_config.json").is_file()
        # No configuration subdirectory is created any more.
        assert not (tmp_path / "_lightrag_config").exists()

        alpha = _snapshot(tmp_path, "teamalpha")
        root = _snapshot(tmp_path, "")
        assert alpha[OWNER_KEY]["value"] == {"workspace": "teamalpha"}
        assert root[OWNER_KEY]["value"] == {"workspace": ""}
        assert (
            alpha[IDENTITY_KEY]["value"]["uuid"]
            == root[IDENTITY_KEY]["value"]["uuid"]
            == _anchor(tmp_path).storage_uuid
        )
        assert _anchor(tmp_path).members == ("", "teamalpha")

    async def test_lightrag_config_is_an_ordinary_workspace_beside_the_empty_one(
        self, tmp_path
    ):
        await _start_and_stop(tmp_path, "")
        await _start_and_stop(tmp_path, "_lightrag_config")

        assert _anchor(tmp_path).members == ("", "_lightrag_config")
        own = tmp_path / "_lightrag_config" / "kv_workspace_config.json"
        assert own.is_file()
        assert _snapshot(tmp_path, "_lightrag_config")[OWNER_KEY]["value"] == {
            "workspace": "_lightrag_config"
        }
        # Both restart cleanly: the claims (root file vs child file) coexist.
        await _start_and_stop(tmp_path, "")
        await _start_and_stop(tmp_path, "_lightrag_config")

    @pytest.mark.parametrize("name", ["C:", "a:b", "1:", "\u00e9:"])
    def test_a_drive_qualified_name_refuses_before_any_directory_exists(
        self, tmp_path, name
    ):
        """Windows joins a name whose second character is ':' outside
        WORKING_DIR (``ntpath.join(root, "C:") == "C:"``); refused on every
        platform so a deployment stays portable."""
        working_dir = tmp_path / "rag"
        with pytest.raises(ValueError, match="drive-qualified"):
            _rag(working_dir, name)
        assert not working_dir.exists() or not any(working_dir.iterdir())

    @pytest.mark.parametrize("name", [" ", "... ", ". ."])
    def test_a_name_of_only_dots_and_spaces_refuses_before_any_directory_exists(
        self, tmp_path, name
    ):
        """Windows strips trailing dots and spaces from a path component, so
        such a workspace would use WORKING_DIR itself: the empty workspace's
        snapshot."""
        working_dir = tmp_path / "rag"
        with pytest.raises(ValueError, match="only of dots"):
            _rag(working_dir, name)
        assert not working_dir.exists() or not any(working_dir.iterdir())

    @pytest.mark.parametrize(
        "name",
        [
            *sorted(RESERVED_WORKSPACE_NAMES),
            # Aliases a case-insensitive filesystem, or Windows' trailing dot
            # and space stripping, resolves onto a reserved file.
            "CONFIG_STORAGE_ANCHOR.JSON",
            "Kv_Workspace_Config.json",
            ".LIGHTRAG_ANCHOR_BIND.LOCK",
            "config_storage_anchor.json.",
            ".lightrag_storage.lock ",
            # Unicode case folding: KELVIN SIGN, the fi ligature, long s.
            "\u212av_workspace_config.json",
            "kv_workspace_con\ufb01g.json",
            ".lightrag_\u017ftorage.lock",
        ],
    )
    def test_a_reserved_root_name_refuses_before_any_directory_exists(
        self, tmp_path, name
    ):
        working_dir = tmp_path / "rag"
        with pytest.raises(ValueError, match="deployment-wide file"):
            _rag(working_dir, name)
        assert not (working_dir / name).exists()

    def test_the_wizard_fold_table_is_every_non_ascii_char_folding_to_ascii(self):
        """setup.sh maps exactly these escapes before folding ASCII case; if
        Unicode (or ``caseless_alias``) changes, the wizard must follow."""
        from lightrag import config_shards

        folding = {
            chr(cp)
            for cp in range(0x80, 0x110000)
            if config_shards.caseless_alias(chr(cp)).isascii()
        }
        table = config_shards._ASCII_FOLDING_NON_ASCII
        assert set(table) == folding
        assert all(config_shards.caseless_alias(c) == a for c, a in table.items())
        setup = (
            Path(__file__).resolve().parents[2] / "scripts" / "setup" / "setup.sh"
        ).read_text()
        for char, ascii_fold in table.items():
            escape = "\\\\u%04x" % ord(char)
            assert f'folded="${{folded//{escape}/' in setup, char


# ---------------------------------------------------------------------------
# Several workspaces, one process tree
# ---------------------------------------------------------------------------


async def test_two_workspaces_in_one_process_tree_keep_separate_snapshots(tmp_path):
    """Before, both instances bound the one shared-namespace key of the
    configuration container, and the second (another backing file) was
    refused; each now has its own physical file and its own rows."""
    alpha = _rag(tmp_path, "teamalpha")
    beta = _rag(tmp_path, "teambeta")
    await alpha.initialize_storages()
    await beta.initialize_storages()
    try:
        assert alpha.config_dir == str(tmp_path / "teamalpha")
        assert beta.config_dir == str(tmp_path / "teambeta")
        alpha_rows = await cs.read_shard_rows(alpha.configuration_storage)
        beta_rows = await cs.read_shard_rows(beta.configuration_storage)
        assert alpha_rows[OWNER_KEY]["value"] == {"workspace": "teamalpha"}
        assert beta_rows[OWNER_KEY]["value"] == {"workspace": "teambeta"}
        assert all(
            row["workspace"] in ("teamalpha", "_lightrag_server")
            for row in alpha_rows.values()
        )
        assert all(
            row["workspace"] in ("teambeta", "_lightrag_server")
            for row in beta_rows.values()
        )
    finally:
        await beta.finalize_storages()
        await alpha.finalize_storages()
    assert _anchor(tmp_path).members == ("teamalpha", "teambeta")


# ---------------------------------------------------------------------------
# Registration and its resumable residues
# ---------------------------------------------------------------------------


async def _open_config(working_dir, workspace: str) -> JsonKVStorage:
    config = cs.create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": str(working_dir), "workspace": workspace},
        embedding_func=None,
    )
    await config.initialize()
    return config


async def _bind(working_dir, workspace: str) -> cs.IdentityBinding:
    config = await _open_config(working_dir, workspace)
    try:
        return await cs.bind_configuration_identity(
            config,
            working_dir=str(working_dir),
            backend="JsonKVStorage",
            container=f"json:{workspace}",
            workspace=workspace,
        )
    finally:
        await config.finalize()


class TestRegistration:
    async def test_a_first_start_publishes_the_group_before_registering(
        self, tmp_path, monkeypatch
    ):
        published: list[ca.StorageAnchor] = []
        original = ca.publish_anchor

        def _record(working_dir, anchor, *, replace):
            published.append(anchor)
            return original(working_dir, anchor, replace=replace)

        monkeypatch.setattr(cs, "publish_anchor", _record)
        binding = await _bind(tmp_path, "teamalpha")

        assert binding.action == "registered"
        # The group identity is durable before any snapshot is written, then
        # the member is appended.
        assert [a.members for a in published] == [(), ("teamalpha",)]
        assert len({a.storage_uuid for a in published}) == 1

    async def test_a_registered_start_rewrites_neither_anchor_nor_snapshot(
        self, tmp_path
    ):
        await _bind(tmp_path, "teamalpha")
        anchor_bytes = (tmp_path / "config_storage_anchor.json").read_bytes()
        snapshot_bytes = Path(json_config_path(str(tmp_path), "teamalpha")).read_bytes()

        binding = await _bind(tmp_path, "teamalpha")

        assert binding.action == "verified"
        assert (tmp_path / "config_storage_anchor.json").read_bytes() == anchor_bytes
        assert (
            Path(json_config_path(str(tmp_path), "teamalpha")).read_bytes()
            == snapshot_bytes
        )

    async def test_an_empty_member_anchor_is_reused_not_regenerated(self, tmp_path):
        """Scenario: the group anchor was published and the first snapshot
        never written."""
        storage_uuid = ca.new_storage_uuid()
        ca.publish_anchor(
            str(tmp_path),
            ca.StorageAnchor("JsonKVStorage", storage_uuid, members=()),
            replace=False,
        )
        binding = await _bind(tmp_path, "teamalpha")
        assert binding.storage_uuid == storage_uuid
        assert _anchor(tmp_path).members == ("teamalpha",)

    async def test_a_metadata_only_snapshot_resumes_its_registration(self, tmp_path):
        """Scenario: the snapshot is durable, the member append was not."""
        await _bind(tmp_path, "teamalpha")
        anchor = _anchor(tmp_path)
        ca.publish_anchor(
            str(tmp_path),
            ca.StorageAnchor("JsonKVStorage", anchor.storage_uuid, members=()),
            replace=True,
        )
        before = Path(json_config_path(str(tmp_path), "teamalpha")).read_bytes()

        binding = await _bind(tmp_path, "teamalpha")

        assert binding.action == "registered"
        assert _anchor(tmp_path).members == ("teamalpha",)
        assert Path(json_config_path(str(tmp_path), "teamalpha")).read_bytes() == before

    async def test_metadata_only_in_shared_memory_is_flushed_before_registering(
        self, tmp_path
    ):
        """Scenario: another worker put the metadata rows into the shared
        dict and its flush failed or was cancelled. The rows read as an
        interrupted registration, but the anchor must not name the snapshot
        until they are on disk."""
        storage_uuid = ca.new_storage_uuid()
        ca.publish_anchor(
            str(tmp_path),
            ca.StorageAnchor("JsonKVStorage", storage_uuid, members=()),
            replace=False,
        )
        snapshot = Path(json_config_path(str(tmp_path), "teamalpha"))
        config = await _open_config(tmp_path, "teamalpha")
        try:
            await config.upsert(
                cs.json_shard_metadata_rows(
                    "teamalpha", storage_uuid, updated_by="test"
                )
            )
            assert not snapshot.exists()

            binding = await cs.bind_configuration_identity(
                config,
                working_dir=str(tmp_path),
                backend="JsonKVStorage",
                container="json:teamalpha",
                workspace="teamalpha",
            )

            assert binding.action == "registered"
            assert _anchor(tmp_path).members == ("teamalpha",)
            on_disk = json.loads(snapshot.read_text())
            assert on_disk[OWNER_KEY]["value"] == {"workspace": "teamalpha"}
        finally:
            await config.finalize()

    async def test_an_unregistered_snapshot_with_records_is_refused_untouched(
        self, tmp_path
    ):
        await _start_and_stop(tmp_path, "teamalpha")
        anchor = _anchor(tmp_path)
        ca.publish_anchor(
            str(tmp_path),
            ca.StorageAnchor("JsonKVStorage", anchor.storage_uuid, members=()),
            replace=True,
        )
        path = Path(json_config_path(str(tmp_path), "teamalpha"))
        before = path.read_bytes()

        with pytest.raises(ConfigurationIdentityError) as info:
            await _bind(tmp_path, "teamalpha")

        assert info.value.cause == ca.IDENTITY_SHARD_INVALID
        assert "delete" in str(info.value)
        assert path.read_bytes() == before
        assert _anchor(tmp_path).members == ()

    async def test_a_registered_member_whose_snapshot_is_gone_is_refused(
        self, tmp_path
    ):
        await _start_and_stop(tmp_path, "teamalpha")
        path = Path(json_config_path(str(tmp_path), "teamalpha"))
        path.unlink()

        rag = _rag(tmp_path, "teamalpha")
        with pytest.raises(ConfigurationIdentityError) as info:
            await rag.initialize_storages()
        await rag.finalize_storages()

        assert info.value.cause == ca.IDENTITY_MEMBER_MISSING
        # Never re-created in place.
        assert not path.exists()
        assert _anchor(tmp_path).members == ("teamalpha",)

    async def test_a_copied_snapshot_is_refused_by_its_owner(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        (tmp_path / "teamgamma").mkdir()
        Path(json_config_path(str(tmp_path), "teamgamma")).write_bytes(
            Path(json_config_path(str(tmp_path), "teamalpha")).read_bytes()
        )
        with pytest.raises(ConfigurationIdentityError) as info:
            await _bind(tmp_path, "teamgamma")
        assert info.value.cause == ca.IDENTITY_SHARD_INVALID
        assert "owned by workspace 'teamalpha'" in str(info.value)


def _register_in_subprocess(working_dir, workspace: str, start_at: float):
    return subprocess.Popen(
        [sys.executable, str(_PROBE), str(working_dir), workspace, str(start_at)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


@pytest.mark.skipif(os.name == "nt", reason="exercises the POSIX flock path")
@pytest.mark.parametrize("anchored", [False, True], ids=["first-use", "append"])
def test_concurrent_registrations_across_process_trees_keep_both_members(
    tmp_path, anchored
):
    """Two servers registering different workspaces at the same moment: the
    bind lock serializes the group creation and the read-modify-write of the
    member list, so neither overwrites the other."""
    if anchored:
        ca.publish_anchor(
            str(tmp_path),
            ca.StorageAnchor("JsonKVStorage", ca.new_storage_uuid(), members=()),
            replace=False,
        )
    start_at = time.time() + 1.5
    procs = [
        _register_in_subprocess(tmp_path, workspace, start_at)
        for workspace in ("teamalpha", "teambeta")
    ]
    outputs = [proc.communicate(timeout=60) for proc in procs]
    for proc, (out, err) in zip(procs, outputs):
        assert proc.returncode == 0, err
    uuids = {
        line.split("=", 1)[1]
        for out, _ in outputs
        for line in out.splitlines()
        if line.startswith("UUID=")
    }
    assert len(uuids) == 1
    anchor = _anchor(tmp_path)
    assert anchor.members == ("teamalpha", "teambeta")
    assert anchor.storage_uuid in uuids


# ---------------------------------------------------------------------------
# Rebinding after the anchor is deleted
# ---------------------------------------------------------------------------


class TestRebinding:
    async def test_surviving_snapshots_are_re_enrolled_unchanged(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        await _start_and_stop(tmp_path, "teambeta")
        original = _anchor(tmp_path)
        files = {
            ws: Path(json_config_path(str(tmp_path), ws)).read_bytes()
            for ws in ("teamalpha", "teambeta")
        }
        (tmp_path / "config_storage_anchor.json").unlink()

        await _start_and_stop(tmp_path, "teamalpha")

        rebound = _anchor(tmp_path)
        assert rebound.storage_uuid == original.storage_uuid
        assert rebound.members == ("teamalpha", "teambeta")
        for ws, before in files.items():
            assert Path(json_config_path(str(tmp_path), ws)).read_bytes() == before
        # Repeating the rebind is stable.
        (tmp_path / "config_storage_anchor.json").unlink()
        await _start_and_stop(tmp_path, "teambeta")
        assert _anchor(tmp_path) == rebound

    async def test_conflicting_group_identities_are_refused(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        (tmp_path / "config_storage_anchor.json").unlink()
        # A snapshot of another group, placed under this WORKING_DIR.
        other = tmp_path / "other_deployment"
        await _start_and_stop(other, "teambeta")
        (tmp_path / "teambeta").mkdir()
        Path(json_config_path(str(tmp_path), "teambeta")).write_bytes(
            Path(json_config_path(str(other), "teambeta")).read_bytes()
        )

        with pytest.raises(ConfigurationIdentityError, match="different"):
            await _bind(tmp_path, "teamalpha")
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_corrupt_surviving_snapshot_is_refused_not_skipped(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        (tmp_path / "config_storage_anchor.json").unlink()
        (tmp_path / "teambeta").mkdir()
        Path(json_config_path(str(tmp_path), "teambeta")).write_text("{not json")

        with pytest.raises(ConfigurationIdentityError, match="not valid JSON"):
            await _bind(tmp_path, "teamalpha")
        assert ca.read_anchor(str(tmp_path)) is None

    def test_the_scan_is_bounded_and_ignores_directories_without_a_snapshot(
        self, tmp_path
    ):
        (tmp_path / "inputs").mkdir()
        (tmp_path / "deep" / "deeper").mkdir(parents=True)
        (tmp_path / "deep" / "deeper" / "kv_workspace_config.json").write_text("{}")
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "kv_workspace_config.json").write_text("{}")
        (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
        (tmp_path / "dangling").symlink_to(tmp_path / "gone", target_is_directory=True)

        # Only one file per direct child is probed: the link back to the root
        # finds the root snapshot's absence (none here) and never recurses.
        assert [s.workspace for s in discover_shards(str(tmp_path))] == ["real"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
    async def test_a_symlinked_workspace_directory_is_discovered_like_it_runs(
        self, tmp_path
    ):
        """The running storage follows a symlinked workspace directory, so
        the scan does too: the member it registers survives a rebind, and a
        migration does not report it missing."""
        volume = tmp_path / "volume" / "teamalpha"
        volume.mkdir(parents=True)
        working_dir = tmp_path / "rag"
        working_dir.mkdir()
        (working_dir / "teamalpha").symlink_to(volume, target_is_directory=True)
        await _start_and_stop(working_dir, "teamalpha")
        assert (volume / "kv_workspace_config.json").is_file()

        assert [s.workspace for s in discover_shards(str(working_dir))] == ["teamalpha"]
        (working_dir / "config_storage_anchor.json").unlink()
        await _start_and_stop(working_dir, "teamalpha")
        assert _anchor(working_dir).members == ("teamalpha",)

    async def test_a_symlinked_snapshot_file_is_refused_by_the_start_too(
        self, tmp_path
    ):
        """The first atomic save replaces a symlinked snapshot with a regular
        file, so discovery refuses one -- and a start must refuse it too
        rather than serve a file a rebind or migration then cannot read."""
        await _bind(tmp_path, "teamalpha")
        real = tmp_path / "elsewhere.json"
        snapshot = Path(json_config_path(str(tmp_path), "teamalpha"))
        snapshot.rename(real)
        snapshot.symlink_to(real)

        with pytest.raises(ConfigurationIdentityError, match="symlink"):
            await _bind(tmp_path, "teamalpha")
        with pytest.raises(ConfigurationIdentityError, match="symlink"):
            discover_shards(str(tmp_path))
        assert snapshot.is_symlink()

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
    async def test_a_fifo_snapshot_is_refused_before_it_is_opened(self, tmp_path):
        """Opening a FIFO for reading blocks until a writer appears, so the
        probe must run before the storage's load, not at bind time."""
        import threading

        snapshot = Path(json_config_path(str(tmp_path), "teamalpha"))
        snapshot.parent.mkdir()
        os.mkfifo(snapshot)

        def _release_a_blocked_reader():
            # Bounds a regression to a failure instead of a hang.
            time.sleep(5)
            try:
                fd = os.open(snapshot, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                return
            os.close(fd)

        threading.Thread(target=_release_a_blocked_reader, daemon=True).start()
        started = time.monotonic()
        with pytest.raises(ConfigurationIdentityError, match="not a regular file"):
            await _open_config(tmp_path, "teamalpha")
        assert time.monotonic() - started < 5

    def test_a_snapshot_under_an_illegal_name_is_refused_by_path(self, tmp_path):
        bad = tmp_path / "config_storage_anchor.json"
        bad.mkdir()
        (bad / "kv_workspace_config.json").write_text("{}")
        with pytest.raises(ConfigurationIdentityError, match="not a legal workspace"):
            discover_shards(str(tmp_path))


# ---------------------------------------------------------------------------
# Maintenance tools never register
# ---------------------------------------------------------------------------


class TestToolVerification:
    async def _verify(self, working_dir, workspace):
        config = await _open_config(working_dir, workspace)
        try:
            return await cs.verify_configuration_identity(
                config,
                ca.read_anchor(str(working_dir)),
                working_dir=str(working_dir),
                backend="JsonKVStorage",
                container=f"json:{workspace}",
                workspace=workspace,
            )
        finally:
            await config.finalize()

    async def test_an_unregistered_workspace_gets_the_first_start_advice(
        self, tmp_path
    ):
        await _start_and_stop(tmp_path, "teamalpha")
        with pytest.raises(ConfigurationIdentityError) as info:
            await self._verify(tmp_path, "teambeta")
        assert info.value.cause == ca.IDENTITY_MEMBER_UNREGISTERED
        assert "Start the server once to register this workspace" in str(info.value)
        assert _anchor(tmp_path).members == ("teamalpha",)
        assert not Path(json_config_path(str(tmp_path), "teambeta")).exists()

    async def test_no_anchor_is_refused_for_json_and_nothing_is_bound(self, tmp_path):
        with pytest.raises(ConfigurationIdentityError) as info:
            await self._verify(tmp_path, "teamalpha")
        assert info.value.cause == ca.IDENTITY_MEMBER_UNREGISTERED
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_registered_member_verifies(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        binding = await self._verify(tmp_path, "teamalpha")
        assert binding.action == "verified"
        assert binding.storage_uuid == _anchor(tmp_path).storage_uuid

    async def test_a_symlinked_member_snapshot_is_refused_like_a_start(self, tmp_path):
        """clear-storage and rebuild-vdb verify here, then flush: an atomic
        save would replace the link and leave its target stale."""
        await _start_and_stop(tmp_path, "teamalpha")
        real = tmp_path / "elsewhere.json"
        snapshot = Path(json_config_path(str(tmp_path), "teamalpha"))
        snapshot.rename(real)
        snapshot.symlink_to(real)

        with pytest.raises(ConfigurationIdentityError, match="symlink"):
            await self._verify(tmp_path, "teamalpha")
        assert snapshot.is_symlink()


# ---------------------------------------------------------------------------
# The anchor format
# ---------------------------------------------------------------------------


class TestAnchorFormat:
    def _write(self, tmp_path, payload) -> None:
        (tmp_path / "config_storage_anchor.json").write_text(json.dumps(payload))

    def _json(self, **overrides):
        payload = {
            "schema_version": 1,
            "backend": "JsonKVStorage",
            "storage_uuid": "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b",
            "layout": "json_shards",
            "members": ["", "teamalpha"],
        }
        payload.update(overrides)
        return {k: v for k, v in payload.items() if v is not ...}

    def test_a_json_anchor_round_trips(self, tmp_path):
        self._write(tmp_path, self._json())
        assert _anchor(tmp_path).members == ("", "teamalpha")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"members": ...},
            {"layout": ...},
            {"layout": "flat"},
            {"members": "teamalpha"},
            {"members": ["a", "a"]},
            {"members": ["a/b"]},
            {"members": [".."]},
            {"members": ["config_storage_anchor.json"]},
            {"members": [1]},
            {"schema_version": True},
        ],
    )
    def test_a_malformed_json_anchor_is_unreadable(self, tmp_path, overrides):
        self._write(tmp_path, self._json(**overrides))
        with pytest.raises(ConfigurationIdentityError) as info:
            ca.read_anchor(str(tmp_path))
        assert info.value.cause == ca.IDENTITY_ANCHOR_UNREADABLE

    @pytest.mark.parametrize("extra", [{"members": []}, {"layout": "json_shards"}])
    def test_a_database_anchor_with_json_fields_is_unreadable(self, tmp_path, extra):
        self._write(
            tmp_path,
            {
                "schema_version": 1,
                "backend": "PGKVStorage",
                "storage_uuid": "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b",
                **extra,
            },
        )
        with pytest.raises(ConfigurationIdentityError):
            ca.read_anchor(str(tmp_path))


# ---------------------------------------------------------------------------
# The bind lock on a platform without shared locks
# ---------------------------------------------------------------------------


class _FakeMsvcrt:
    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self):
        self.calls = 0

    def locking(self, fd, mode, nbytes):
        self.calls += 1
        raise OSError("msvcrt cannot tell 'held' from 'unsupported'")


async def test_a_windows_bind_lock_failure_is_contention_never_fail_open(
    tmp_path, monkeypatch
):
    """``msvcrt`` reports a held lock and an unsupported filesystem alike;
    the bind lock reads both as contention and times out instead of
    proceeding unprotected."""
    fake = _FakeMsvcrt()
    monkeypatch.setattr(al, "_fcntl", lambda: None)
    monkeypatch.setattr(al, "_msvcrt", lambda: fake)
    entered = False
    with pytest.raises(ConfigurationAnchorLockError, match="bind lock"):
        async with al.anchor_bind_lock(str(tmp_path), timeout=0.2):
            entered = True
    assert not entered
    assert fake.calls > 1
