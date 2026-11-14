"""Steps 0a-0c and 1b as they run through ``LightRAG.initialize_storages()``
on the JSON / Nano / NetworkX backends.

0a-0c (the shared anchor lock, a strict anchor read, the backend-type check)
open nothing and are not sticky; 1b (the identity bind) follows the sticky
rules and the rollback of a step-2 failure. See *The anchor and the container
identity* in docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.base import StoragesStatus
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationIdentityError,
)
from lightrag.kg import anchor_lock as al
from lightrag.namespace import CONFIG_JSON_FILE_NAME
from tests.config_store.test_startup_sequence import (  # noqa: F401
    _Spy,
    _shared_storage,
    _workspace,
)
from tests.config_store.test_startup_sequence import _rag as _base_rag

pytestmark = pytest.mark.offline

IDENTITY_KEY = "$meta/storage_identity"


OWNER_KEY = "$meta/json_shard"


def _rag(tmp_path, *, model_name, workspace=None):
    """The sequence tests' instance, optionally on another workspace."""
    base = _base_rag(tmp_path, model_name=model_name)
    if workspace is None:
        return base
    from lightrag import LightRAG

    return LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        llm_model_func=base.llm_model_func,
        embedding_func=base.embedding_func,
        tokenizer=base.tokenizer,
    )


def _config_file(tmp_path, workspace=None):
    """A workspace's JSON snapshot; the sequence tests' workspace by default."""
    name = _workspace(tmp_path) if workspace is None else workspace
    return (tmp_path / name if name else tmp_path) / CONFIG_JSON_FILE_NAME


def _stored(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _anchor_bytes(tmp_path) -> bytes | None:
    path = ca.anchor_path(str(tmp_path))
    return open(path, "rb").read() if os.path.exists(path) else None


def _write_anchor(tmp_path, backend, storage_uuid):
    path = ca.anchor_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"schema_version": 1, "backend": backend, "storage_uuid": storage_uuid},
            f,
        )


async def _start_and_stop(tmp_path, **kwargs):
    rag = _rag(tmp_path, model_name="bge-m3", **kwargs)
    await rag.initialize_storages()
    await rag.finalize_storages()
    return rag


async def test_a_first_start_creates_the_group_and_registers_the_workspace(
    tmp_path, monkeypatch
):
    """No anchor, no snapshot: the group's anchor is published FIRST with no
    members, then the workspace's snapshot gets its identity and owner rows,
    then the workspace is appended to the members."""
    published: list[tuple[tuple[str, ...], bool]] = []
    real_publish = cs.publish_anchor

    def _recording(working_dir, anchor, *, replace):
        published.append((anchor.members, replace))
        if not replace:
            # The create lands before any snapshot is written.
            assert not _config_file(tmp_path).exists() or not _stored(
                _config_file(tmp_path)
            )
        return real_publish(working_dir, anchor, replace=replace)

    monkeypatch.setattr(cs, "publish_anchor", _recording)
    rag = await _start_and_stop(tmp_path)
    workspace = _workspace(tmp_path)
    stored = _stored(_config_file(tmp_path))
    storage_uuid = stored[IDENTITY_KEY]["value"]["uuid"]
    assert stored[OWNER_KEY]["value"] == {"workspace": workspace}
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        backend="JsonKVStorage", storage_uuid=storage_uuid, members=(workspace,)
    )
    assert published == [((), False), ((workspace,), True)]
    assert rag._holds_anchor_lock is False
    assert not al.holds_anchor_lock(str(tmp_path))


async def test_a_later_start_verifies_and_rewrites_nothing(tmp_path):
    await _start_and_stop(tmp_path)
    anchor = _anchor_bytes(tmp_path)
    identity = _stored(_config_file(tmp_path))[IDENTITY_KEY]

    await _start_and_stop(tmp_path)
    assert _anchor_bytes(tmp_path) == anchor
    assert _stored(_config_file(tmp_path))[IDENTITY_KEY] == identity


async def test_a_backend_type_change_is_refused_before_anything_opens(tmp_path):
    """The KV change that moved the candidate: refused at step 0c -- no
    configuration storage initialized, no business storage initialized, no
    baseline written, anchor untouched -- and NOT sticky: the same instance
    starts once the operator resolves it."""
    _write_anchor(tmp_path, "PGKVStorage", ca.new_storage_uuid())
    anchor = _anchor_bytes(tmp_path)

    rag = _rag(tmp_path, model_name="bge-m3")
    config_init = _Spy(rag.configuration_storage, "initialize")
    full_docs_init = _Spy(rag.full_docs, "initialize")
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        await rag.initialize_storages()
    assert excinfo.value.cause == ca.IDENTITY_BACKEND_MISMATCH
    assert "LIGHTRAG_CONFIG_STORAGE=PGKVStorage" in str(excinfo.value)
    assert config_init.calls == 0
    assert full_docs_init.calls == 0
    assert rag._storages_status is StoragesStatus.CREATED
    assert rag._startup_refusal is None
    assert not _config_file(tmp_path).exists()
    assert _anchor_bytes(tmp_path) == anchor
    assert not al.holds_anchor_lock(str(tmp_path))

    # The sanctioned rebind: delete the anchor. Same instance, real retry.
    os.remove(ca.anchor_path(str(tmp_path)))
    await rag.initialize_storages()
    assert ca.read_anchor(str(tmp_path)).backend == "JsonKVStorage"
    await rag.finalize_storages()


async def test_an_unreadable_anchor_is_refused_and_never_absent(tmp_path):
    path = ca.anchor_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write('{"schema_version": 1, "backend": "Json')
    rag = _rag(tmp_path, model_name="bge-m3")
    config_init = _Spy(rag.configuration_storage, "initialize")
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        await rag.initialize_storages()
    assert excinfo.value.cause == ca.IDENTITY_ANCHOR_UNREADABLE
    assert config_init.calls == 0
    assert rag._startup_refusal is None
    # Never overwritten by a start.
    assert open(path).read() == '{"schema_version": 1, "backend": "Json'


async def test_an_emptied_container_is_refused_and_the_anchor_deletion_rebinds(
    tmp_path, monkeypatch
):
    await _start_and_stop(tmp_path)
    old_uuid = ca.read_anchor(str(tmp_path)).storage_uuid
    os.remove(_config_file(tmp_path))  # the container was emptied

    rag = _rag(tmp_path, model_name="bge-m3")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    full_docs_init = _Spy(rag.full_docs, "initialize")
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        await rag.initialize_storages()
    # A registered member whose snapshot is gone is refused, never
    # re-initialized in place.
    assert excinfo.value.cause == ca.IDENTITY_MEMBER_MISSING
    assert ca.anchor_path(str(tmp_path)) in str(excinfo.value)
    # Step 1b is sticky and rolled back like a step-2 failure.
    assert config_finalize.calls == 1
    assert full_docs_init.calls == 0
    assert IDENTITY_KEY not in _stored(_config_file(tmp_path))
    with pytest.raises(ConfigurationIdentityError):
        await rag.initialize_storages()
    await rag.finalize_storages()
    assert not al.holds_anchor_lock(str(tmp_path))

    os.remove(ca.anchor_path(str(tmp_path)))
    warnings: list[str] = []
    monkeypatch.setattr(cs.logger, "warning", warnings.append)
    await _start_and_stop(tmp_path)
    new_anchor = ca.read_anchor(str(tmp_path))
    assert new_anchor.storage_uuid != old_uuid
    assert new_anchor.members == (_workspace(tmp_path),)
    assert any(
        new_anchor.storage_uuid in w and ca.anchor_path(str(tmp_path)) in w
        for w in warnings
    )


async def test_a_snapshot_copied_to_another_workspace_is_refused(tmp_path):
    """The snapshot's path is fixed, so it cannot be "moved" by a setting --
    only copied. A copy under another workspace's name keeps its owner and its
    rows, is refused, and is never relabelled or enrolled."""
    await _start_and_stop(tmp_path)
    anchor = _anchor_bytes(tmp_path)
    copy = _config_file(tmp_path, "copied")
    copy.parent.mkdir()
    shutil.copy(_config_file(tmp_path), copy)
    before = copy.read_bytes()

    rag = _rag(tmp_path, model_name="bge-m3", workspace="copied")
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        await rag.initialize_storages()
    assert excinfo.value.cause == ca.IDENTITY_SHARD_INVALID
    await rag.finalize_storages()
    assert copy.read_bytes() == before
    assert _anchor_bytes(tmp_path) == anchor


async def test_a_new_workspace_registers_into_the_existing_group(tmp_path):
    """An empty snapshot of an unregistered workspace is not a refusal: it is
    registered with the group's identity, and the first member is kept."""
    await _start_and_stop(tmp_path)
    first = ca.read_anchor(str(tmp_path))

    await _start_and_stop(tmp_path, workspace="newcomer")
    after = ca.read_anchor(str(tmp_path))
    assert after.storage_uuid == first.storage_uuid
    assert after.members == tuple(sorted((_workspace(tmp_path), "newcomer")))
    stored = _stored(_config_file(tmp_path, "newcomer"))
    assert stored[IDENTITY_KEY]["value"]["uuid"] == first.storage_uuid
    assert stored[OWNER_KEY]["value"] == {"workspace": "newcomer"}


async def test_a_crash_after_the_identity_write_heals_on_the_next_start(
    tmp_path, monkeypatch
):
    """The group anchor is published and the snapshot's identity and owner
    are durable, but the member append fails: the next start reuses the
    metadata-only snapshot and appends, with the same identity."""
    real_publish = cs.publish_anchor

    def _crash(working_dir, anchor, *, replace):
        if not replace:
            return real_publish(working_dir, anchor, replace=replace)
        raise ConfigurationIdentityError(
            "injected publish failure", cause=ca.IDENTITY_ANCHOR_WRITE_FAILED
        )

    monkeypatch.setattr(cs, "publish_anchor", _crash)
    rag = _rag(tmp_path, model_name="bge-m3")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    with pytest.raises(ConfigurationIdentityError):
        await rag.initialize_storages()
    assert config_finalize.calls == 1
    assert rag._startup_refusal is not None
    assert not al.holds_anchor_lock(str(tmp_path))
    created = _stored(_config_file(tmp_path))[IDENTITY_KEY]["value"]["uuid"]
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        backend="JsonKVStorage", storage_uuid=created, members=()
    )

    monkeypatch.setattr(cs, "publish_anchor", real_publish)
    await _start_and_stop(tmp_path)
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        backend="JsonKVStorage", storage_uuid=created, members=(_workspace(tmp_path),)
    )


async def test_a_cancellation_during_the_bind_is_sticky_and_releases_everything(
    tmp_path, monkeypatch
):
    def _cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(cs, "publish_anchor", _cancel)
    rag = _rag(tmp_path, model_name="bge-m3")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    with pytest.raises(asyncio.CancelledError):
        await rag.initialize_storages()
    assert config_finalize.calls == 1
    assert not al.holds_anchor_lock(str(tmp_path))
    with pytest.raises(RuntimeError, match="interrupted by CancelledError"):
        await rag.initialize_storages()


async def test_repeated_and_concurrent_instances_never_regenerate_the_identity(
    tmp_path, monkeypatch
):
    creates = []
    real_publish = cs.publish_anchor

    def _counting(working_dir, anchor, *, replace):
        if not replace:
            creates.append(anchor)
        return real_publish(working_dir, anchor, replace=replace)

    monkeypatch.setattr(cs, "publish_anchor", _counting)

    def _instance(name):
        return _rag(tmp_path, model_name="bge-m3", workspace=name)

    rags = [_instance(f"ws{i}") for i in range(3)]
    await asyncio.gather(*(r.initialize_storages() for r in rags))
    # One group created; every concurrent workspace registered into it, and
    # none lost its membership to another's append.
    assert len(creates) == 1
    storage_uuid = creates[0].storage_uuid
    assert ca.read_anchor(str(tmp_path)).members == ("ws0", "ws1", "ws2")
    # A later instance in the same process joins the same group.
    extra = _instance("ws-late")
    await extra.initialize_storages()
    for r in [*rags, extra]:
        await r.finalize_storages()
    anchor = _anchor_bytes(tmp_path)
    assert len(creates) == 1
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        backend="JsonKVStorage",
        storage_uuid=storage_uuid,
        members=("ws-late", "ws0", "ws1", "ws2"),
    )
    for name in ("ws0", "ws1", "ws2", "ws-late"):
        stored = _stored(_config_file(tmp_path, name))
        assert stored[IDENTITY_KEY]["value"]["uuid"] == storage_uuid
    # Restarting every workspace rewrites neither the anchor nor an identity.
    identities = {
        name: _stored(_config_file(tmp_path, name))[IDENTITY_KEY]
        for name in ("ws0", "ws1", "ws2", "ws-late")
    }
    for name in identities:
        await _start_and_stop(tmp_path, workspace=name)
    assert _anchor_bytes(tmp_path) == anchor
    assert {
        name: _stored(_config_file(tmp_path, name))[IDENTITY_KEY] for name in identities
    } == identities
    assert not al.holds_anchor_lock(str(tmp_path))


async def test_a_running_instance_excludes_the_migration(tmp_path):
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    try:
        with pytest.raises(ConfigurationAnchorLockError):
            al.acquire_anchor_lock_exclusive(str(tmp_path))
    finally:
        await rag.finalize_storages()
    al.acquire_anchor_lock_exclusive(str(tmp_path)).release()


async def test_a_running_migration_refuses_the_start_without_sticking(tmp_path):
    pytest.importorskip("fcntl")
    lock = al.acquire_anchor_lock_exclusive(str(tmp_path))
    rag = _rag(tmp_path, model_name="bge-m3")
    try:
        with pytest.raises(ConfigurationAnchorLockError):
            await rag.initialize_storages()
        assert rag._startup_refusal is None
        assert rag._holds_anchor_lock is False
    finally:
        lock.release()
    await rag.initialize_storages()
    await rag.finalize_storages()


async def test_a_relative_working_dir_keeps_its_anchor_across_a_cwd_change(
    tmp_path, monkeypatch
):
    """The anchor is pinned where ``working_dir`` resolves, at construction: a CWD
    change before ``initialize_storages()`` must neither bind a second anchor
    beside another directory nor release a lock path it never took."""
    from lightrag import LightRAG

    deployment = tmp_path / "deploy"
    elsewhere = tmp_path / "elsewhere"
    deployment.mkdir()
    elsewhere.mkdir()
    base = _base_rag(deployment, model_name="bge-m3")
    monkeypatch.chdir(tmp_path)
    rag = LightRAG(
        working_dir="deploy",
        workspace=base.workspace,
        llm_model_func=base.llm_model_func,
        embedding_func=base.embedding_func,
        tokenizer=base.tokenizer,
    )
    monkeypatch.chdir(elsewhere)

    await rag.initialize_storages()
    assert al.holds_anchor_lock(str(deployment))
    await rag.finalize_storages()

    storage_uuid = _stored(_config_file(deployment))[IDENTITY_KEY]["value"]["uuid"]
    assert ca.read_anchor(str(deployment)) == ca.StorageAnchor(
        backend="JsonKVStorage", storage_uuid=storage_uuid, members=(base.workspace,)
    )
    assert ca.read_anchor(str(elsewhere / "deploy")) is None
    assert not al.holds_anchor_lock(str(deployment))
