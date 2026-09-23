"""The no-anchor bind across process trees on one host.

Several servers may share one ``working_dir`` with different workspaces when
the configuration storage is not a local JSON file, and every one binds the
same container-wide identity row. The in-tree keyed lock cannot see another
server, so the anchor bind lock (``lightrag/kg/anchor_lock.py``) serializes
the no-anchor bind; an anchored start never takes it. See *The anchor and the
container identity* in docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import ConfigurationAnchorLockError
from lightrag.kg import anchor_lock as al
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import SERVER_SCOPE
from tests.config_store.test_config_store import FakeConfigKV

pytestmark = pytest.mark.offline

fcntl = pytest.importorskip("fcntl")

UUID_A = "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b"
IDENTITY_KEY = "_lightrag_server/storage_identity"
CONTAINER = "PGKVStorage (_lightrag_config)"
_PROBE = Path(__file__).with_name("_bind_probe.py")


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


def _identity_row(storage_uuid=UUID_A):
    return cs.make_config_row(
        scope_workspace=SERVER_SCOPE,
        suffix=cs.STORAGE_IDENTITY_SUFFIX,
        value={"uuid": storage_uuid},
        updated_by="another server",
    )


def _foreign_bind_lock(tmp_path):
    """The bind lock as ANOTHER server's process holds it: a second open file
    description, which ``flock`` treats exactly as another process."""
    path = al.anchor_bind_lock_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


async def _bind(config, tmp_path):
    return await cs.bind_configuration_identity(
        config, working_dir=str(tmp_path), backend="PGKVStorage", container=CONTAINER
    )


async def test_a_start_waits_for_another_servers_bind_and_then_verifies(tmp_path):
    config = FakeConfigKV(visible_on_upsert=True)
    other = _foreign_bind_lock(tmp_path)
    task = asyncio.create_task(_bind(config, tmp_path))
    await asyncio.sleep(0.2)
    assert not task.done(), "the start must wait for the other server's bind"
    # The other server finishes ITS bind: identity written, anchor published.
    config.visible[IDENTITY_KEY] = _identity_row()
    ca.publish_anchor(
        str(tmp_path),
        ca.StorageAnchor(backend="PGKVStorage", storage_uuid=UUID_A),
        replace=False,
    )
    other.close()
    binding = await asyncio.wait_for(task, timeout=5)
    assert binding == cs.IdentityBinding(storage_uuid=UUID_A, action="verified")
    assert not [c for c in config.calls if c[0] == "upsert"]


async def test_an_anchored_start_never_waits_on_the_bind_lock(tmp_path):
    ca.publish_anchor(
        str(tmp_path),
        ca.StorageAnchor(backend="PGKVStorage", storage_uuid=UUID_A),
        replace=False,
    )
    other = _foreign_bind_lock(tmp_path)
    try:
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        binding = await asyncio.wait_for(_bind(config, tmp_path), timeout=2)
        assert binding.action == "verified"
    finally:
        other.close()


async def test_a_bind_lock_held_too_long_refuses_and_writes_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(al, "DEFAULT_BIND_LOCK_TIMEOUT_SECONDS", 0.2)
    real = al.anchor_bind_lock
    monkeypatch.setattr(
        al,
        "anchor_bind_lock",
        lambda working_dir: real(working_dir, timeout=0.2),
    )
    config = FakeConfigKV()
    other = _foreign_bind_lock(tmp_path)
    try:
        with pytest.raises(ConfigurationAnchorLockError, match="bind lock"):
            await _bind(config, tmp_path)
    finally:
        other.close()
    assert not [c for c in config.calls if c[0] in ("upsert", "flush")]
    assert ca.read_anchor(str(tmp_path)) is None


async def test_without_file_locks_the_bind_proceeds_with_a_warning(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(al, "_fcntl", lambda: None)
    warnings: list[str] = []
    monkeypatch.setattr(al.logger, "warning", warnings.append)
    binding = await _bind(FakeConfigKV(), tmp_path)
    assert binding.action == "created"
    assert any("bind lock" in w for w in warnings)


def test_two_servers_starting_together_bind_exactly_one_identity(tmp_path):
    """Two real process trees, one working directory, one shared store, both
    reading "no identity" at the same moment. Exactly one creates it; the
    other verifies against the anchor the first published -- never a second
    UUID left in the store under an anchor that names the first."""
    store = tmp_path / "store.json"
    start_at = time.time() + 2.0
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    procs = [
        subprocess.Popen(
            [sys.executable, str(_PROBE), str(tmp_path), str(store), str(start_at)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(2)
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        results.append(dict(line.split("=", 1) for line in out.split() if "=" in line))

    assert sorted(r["ACTION"] for r in results) == ["created", "verified"]
    assert results[0]["UUID"] == results[1]["UUID"]
    stored = json.loads(store.read_text())[IDENTITY_KEY]["value"]["uuid"]
    assert stored == results[0]["UUID"]
    assert ca.read_anchor(str(tmp_path)).storage_uuid == stored
