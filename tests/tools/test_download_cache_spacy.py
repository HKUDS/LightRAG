"""Tests for how ``lightrag-download-cache`` installs/downloads spaCy models.

Regression: ``--spacy-install`` hard-coded ``python -m pip``, but the README's
``uv sync`` setup creates a virtual environment without pip, so the documented
command failed for every model. It now falls back to ``uv pip install``
targeted at the running interpreter, and fails up front with an actionable
message when no installer is usable.
"""

import subprocess
import sys

import pytest

import lightrag.tools.download_cache as download_cache

pytestmark = pytest.mark.offline


@pytest.fixture
def recorded_runs(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _set_tools(monkeypatch, *, pip: bool, uv: str | None):
    import importlib.util
    import shutil

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            (object() if pip else None)
            if name == "pip"
            else real_find_spec(name, *a, **k)
        ),
    )
    monkeypatch.setattr(shutil, "which", lambda name: uv if name == "uv" else None)


def test_install_uses_pip_when_available(monkeypatch, recorded_runs):
    _set_tools(monkeypatch, pip=True, uv="/usr/bin/uv")

    success, failed = download_cache.download_spacy_models(install=True)

    assert (success, failed) == (len(download_cache.SPACY_MODEL_WHEELS), [])
    assert all(
        cmd[:4] == [sys.executable, "-m", "pip", "install"] for cmd in recorded_runs
    )


def test_install_falls_back_to_uv_without_pip(monkeypatch, recorded_runs):
    _set_tools(monkeypatch, pip=False, uv="/opt/uv/bin/uv")

    success, failed = download_cache.download_spacy_models(install=True)

    assert (success, failed) == (len(download_cache.SPACY_MODEL_WHEELS), [])
    assert [cmd[-1] for cmd in recorded_runs] == [
        url for _, url in sorted(download_cache.SPACY_MODEL_WHEELS.items())
    ]
    # --python pins the target to this interpreter's environment, not
    # whatever environment uv would discover from the working directory.
    assert all(
        cmd[:5] == ["/opt/uv/bin/uv", "pip", "install", "--python", sys.executable]
        for cmd in recorded_runs
    )


def test_install_without_pip_or_uv_fails_without_running(monkeypatch, recorded_runs):
    _set_tools(monkeypatch, pip=False, uv=None)

    success, failed = download_cache.download_spacy_models(install=True)

    assert success == 0
    assert recorded_runs == []
    assert sorted(name for name, _ in failed) == sorted(
        download_cache.SPACY_MODEL_WHEELS
    )
    assert "requirements-offline-smart-heading.txt" in failed[0][1]


def test_download_without_pip_fails_even_with_uv(monkeypatch, recorded_runs, tmp_path):
    # uv has no `pip download`, so the wheel download cannot use it.
    _set_tools(monkeypatch, pip=False, uv="/opt/uv/bin/uv")

    success, failed = download_cache.download_spacy_models(str(tmp_path), install=False)

    assert success == 0
    assert recorded_runs == []
    assert len(failed) == len(download_cache.SPACY_MODEL_WHEELS)
    assert "uv pip install pip" in failed[0][1]


def test_download_uses_pip_download(monkeypatch, recorded_runs, tmp_path):
    _set_tools(monkeypatch, pip=True, uv=None)

    success, _ = download_cache.download_spacy_models(str(tmp_path), install=False)

    assert success == len(download_cache.SPACY_MODEL_WHEELS)
    assert all(
        cmd[:6] == [sys.executable, "-m", "pip", "download", "--no-deps", "--dest"]
        and cmd[6] == str(tmp_path)
        for cmd in recorded_runs
    )
