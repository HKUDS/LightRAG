"""The Gunicorn master runs the anchor checks (steps 0a-0c) before forking.

A backend type that differs from the anchor refuses the MASTER, instead of
surfacing as every worker failing while the master respawns them. The master
also takes the shared anchor lock so its workers inherit it, and gives it back
in ``on_exit`` after the ``config_dir`` claim. See *The anchor and the
container identity* in docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

from lightrag import config_anchor as ca
from lightrag.exceptions import ConfigurationIdentityError

pytestmark = pytest.mark.offline


@pytest.fixture
def gunicorn_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-gunicorn"])
    module = importlib.import_module("lightrag.api.gunicorn_config")
    monkeypatch.setattr(module, "workers", 4, raising=False)
    monkeypatch.setattr(module, "working_dir", None, raising=False)
    return module


@pytest.fixture(autouse=True)
def _clean_locks():
    from lightrag.kg import anchor_lock as al
    from lightrag.kg import working_dir_lock as wdl

    claims, holds = dict(wdl._claims), dict(al._shared_holds)
    wdl._claims.clear()
    al._shared_holds.clear()
    yield
    for claim in wdl._claims.values():
        claim.handle.close()
    for hold in al._shared_holds.values():
        if hold.handle is not None:
            hold.handle.close()
    wdl._claims.clear()
    al._shared_holds.clear()
    wdl._claims.update(claims)
    al._shared_holds.update(holds)


def _env(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("WORKING_DIR", str(tmp_path))
    for name in ("LIGHTRAG_CONFIG_STORAGE", "LIGHTRAG_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def _anchor(tmp_path, backend):
    path = ca.anchor_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "backend": backend,
                "storage_uuid": ca.new_storage_uuid(),
            },
            f,
        )


def test_the_master_refuses_a_type_mismatch_before_forking(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """First upgrade bound PG; later only KV was changed to Mongo with CONFIG
    unset -- the candidate moved, and the master refuses."""
    from lightrag.kg.anchor_lock import holds_anchor_lock
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _anchor(tmp_path, "JsonKVStorage")
    _env(monkeypatch, tmp_path, LIGHTRAG_KV_STORAGE="MongoKVStorage")
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert excinfo.value.cause == ca.IDENTITY_BACKEND_MISMATCH
    assert "forking workers" not in capsys.readouterr().out
    assert holds_anchor_lock(str(tmp_path)) is False
    assert holds_working_dir_lock(str(tmp_path / "_lightrag_config")) is False


def test_an_explicit_selection_of_the_anchored_backend_passes(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The documented fix: keep KV on Mongo, name the anchored configuration
    backend explicitly. No migration needed."""
    from lightrag.kg.anchor_lock import holds_anchor_lock

    _anchor(tmp_path, "JsonKVStorage")
    _env(
        monkeypatch,
        tmp_path,
        LIGHTRAG_KV_STORAGE="MongoKVStorage",
        LIGHTRAG_CONFIG_STORAGE="JsonKVStorage",
    )
    gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is True
    gunicorn_config.on_exit(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is False


def test_no_anchor_takes_the_lock_for_the_workers_and_writes_nothing(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The master only checks: binding is each worker's step 1b."""
    from lightrag.kg.anchor_lock import holds_anchor_lock

    _env(monkeypatch, tmp_path)
    gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is True
    assert ca.read_anchor(str(tmp_path)) is None
    gunicorn_config.on_exit(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is False


def test_an_unreadable_anchor_refuses_the_master(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    path = ca.anchor_path(str(tmp_path))
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("{")
    _env(monkeypatch, tmp_path)
    with pytest.raises(ConfigurationIdentityError):
        gunicorn_config.on_starting(object())
    capsys.readouterr()


def test_the_master_follows_the_parsed_working_dir(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """``--working-dir`` is never written back to the environment, so the
    master must read the anchor where the workers will."""
    parsed = tmp_path / "parsed"
    _anchor(parsed, "PGKVStorage")
    _env(monkeypatch, tmp_path / "env")
    monkeypatch.setattr(gunicorn_config, "working_dir", str(parsed))
    with pytest.raises(ConfigurationIdentityError):
        gunicorn_config.on_starting(object())
    capsys.readouterr()
