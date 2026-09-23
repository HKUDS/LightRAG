"""Steps 0a-0c and 1b as they run through ``LightRAG.initialize_storages()``
on the JSON / Nano / NetworkX backends.

0a-0c (the shared anchor lock, a strict anchor read, the backend-type check)
open nothing and are not sticky; 1b (the identity bind) follows the sticky
rules and the rollback of a step-2 failure. See *The anchor and the container
identity* in docs/design/ConfigurationStorage.md.
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
from lightrag.namespace import CONFIG_CONTAINER_TAG, CONFIG_JSON_FILE_NAME
from tests.config_store.test_startup_sequence import (  # noqa: F401
    _Spy,
    _shared_storage,
)
from tests.config_store.test_startup_sequence import _rag as _base_rag

pytestmark = pytest.mark.offline

IDENTITY_KEY = "_lightrag_server/storage_identity"


def _rag(tmp_path, *, model_name, workspace=None, config_dir=None):
    """The sequence tests' instance, optionally on another workspace or
    ``config_dir``."""
    base = _base_rag(tmp_path, model_name=model_name)
    if workspace is None and config_dir is None:
        return base
    from lightrag import LightRAG

    return LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace or base.workspace,
        config_dir=config_dir or "",
        llm_model_func=base.llm_model_func,
        embedding_func=base.embedding_func,
        tokenizer=base.tokenizer,
    )


def _config_file(tmp_path, config_dir=None):
    return (
        (tmp_path / CONFIG_CONTAINER_TAG) if config_dir is None else config_dir
    ) / CONFIG_JSON_FILE_NAME


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


async def test_a_first_start_creates_the_identity_then_the_anchor(tmp_path):
    rag = await _start_and_stop(tmp_path)
    storage_uuid = _stored(_config_file(tmp_path))[IDENTITY_KEY]["value"]["uuid"]
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        backend="JsonKVStorage", storage_uuid=storage_uuid
    )
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
    os.makedirs(os.path.dirname(path))
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
    assert excinfo.value.cause == ca.IDENTITY_UUID_MISSING
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
    new_uuid = ca.read_anchor(str(tmp_path)).storage_uuid
    assert new_uuid != old_uuid
    assert any(new_uuid in w and "JsonKVStorage at" in w for w in warnings)


async def test_a_moved_config_dir_that_keeps_its_identity_passes(tmp_path):
    """``LIGHTRAG_CONFIG_DIR`` changed: the FIXED anchor is still read and the
    verdict is the target container's UUID."""
    await _start_and_stop(tmp_path)
    anchor = _anchor_bytes(tmp_path)
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(_config_file(tmp_path), _config_file(tmp_path, moved))

    await _start_and_stop(tmp_path, config_dir=str(moved))
    assert _anchor_bytes(tmp_path) == anchor


async def test_a_config_dir_pointed_at_an_empty_directory_is_refused(tmp_path):
    await _start_and_stop(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    rag = _rag(tmp_path, model_name="bge-m3", config_dir=str(empty))
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        await rag.initialize_storages()
    assert excinfo.value.cause == ca.IDENTITY_UUID_MISSING
    await rag.finalize_storages()
    # Nothing created in the directory it was pointed at.
    assert IDENTITY_KEY not in _stored(_config_file(tmp_path, empty))


async def test_a_crash_after_the_identity_write_heals_on_the_next_start(
    tmp_path, monkeypatch
):
    real_publish = cs.publish_anchor

    def _crash(*args, **kwargs):
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
    assert ca.read_anchor(str(tmp_path)) is None

    monkeypatch.setattr(cs, "publish_anchor", real_publish)
    await _start_and_stop(tmp_path)
    assert ca.read_anchor(str(tmp_path)).storage_uuid == created


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
    publishes = []
    real_publish = cs.publish_anchor

    def _counting(*args, **kwargs):
        publishes.append(args)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(cs, "publish_anchor", _counting)

    def _instance(name):
        return _rag(tmp_path, model_name="bge-m3", workspace=name)

    rags = [_instance(f"ws{i}") for i in range(3)]
    await asyncio.gather(*(r.initialize_storages() for r in rags))
    anchor = _anchor_bytes(tmp_path)
    identity = _stored(_config_file(tmp_path))[IDENTITY_KEY]
    assert len(publishes) == 1
    # A later instance in the same process verifies against the same row.
    extra = _instance("ws-late")
    await extra.initialize_storages()
    for r in [*rags, extra]:
        await r.finalize_storages()
    assert len(publishes) == 1
    assert _anchor_bytes(tmp_path) == anchor
    assert _stored(_config_file(tmp_path))[IDENTITY_KEY] == identity
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
