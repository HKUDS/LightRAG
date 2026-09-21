"""The Gunicorn master claims the directory its WORKERS will ask for.

The claim is held by the open file description, which ``fork`` shares: the
master takes it in ``on_starting``, before forking, and every worker counts
itself into the inherited claim rather than opening a second descriptor.

That only works while both name the SAME directory. ``--working-dir`` is
parsed into ``global_args`` and handed to each worker's ``LightRAG``, and
nothing writes it back to the environment -- so a master that read
``WORKING_DIR`` claimed a path nobody inherits: the first worker took the real
one for itself and every worker after it was refused at startup with
``WorkingDirectoryInUseError``.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def gunicorn_config(monkeypatch):
    # The module parses argv at import time; pytest's own would fail it.
    monkeypatch.setattr(sys, "argv", ["lightrag-gunicorn"])
    module = importlib.import_module("lightrag.api.gunicorn_config")
    monkeypatch.setattr(module, "workers", 4, raising=False)
    return module


@pytest.fixture(autouse=True)
def _clean_claims():
    from lightrag.kg import working_dir_lock as wdl

    held = dict(wdl._claims)
    wdl._claims.clear()
    yield
    for claim in wdl._claims.values():
        try:
            claim.handle.close()
        except OSError:  # pragma: no cover - defensive
            pass
    wdl._claims.clear()
    wdl._claims.update(held)


def test_the_master_follows_the_parsed_working_dir_over_the_environment(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    parsed = tmp_path / "from-cli"
    parsed.mkdir()
    environment = tmp_path / "from-env"
    environment.mkdir()

    monkeypatch.setenv("WORKING_DIR", str(environment))
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    monkeypatch.setattr(gunicorn_config, "working_dir", str(parsed), raising=False)

    assert gunicorn_config.resolved_working_dir() == str(parsed)

    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(str(parsed)) is True
    assert holds_working_dir_lock(str(environment)) is False


def test_the_environment_is_the_fallback_when_nothing_parsed_a_directory(
    gunicorn_config, monkeypatch, tmp_path
):
    """A master started some other way than ``lightrag-gunicorn`` sets no
    module attribute, and the environment is all there is."""
    monkeypatch.setenv("WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(gunicorn_config, "working_dir", None, raising=False)

    assert gunicorn_config.resolved_working_dir() == str(tmp_path)


def test_on_exit_releases_what_on_starting_claimed(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    parsed = tmp_path / "from-cli"
    parsed.mkdir()
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "from-env"))
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    monkeypatch.setattr(gunicorn_config, "working_dir", str(parsed), raising=False)

    gunicorn_config.on_starting(object())
    assert holds_working_dir_lock(str(parsed)) is True

    gunicorn_config.on_exit(object())
    capsys.readouterr()

    assert holds_working_dir_lock(str(parsed)) is False
