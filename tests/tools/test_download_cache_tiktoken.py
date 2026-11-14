"""Tests for where ``lightrag-download-cache`` writes the tiktoken cache.

Regression: with no ``--cache-dir`` and no ``TIKTOKEN_CACHE_DIR`` the command
wrote into tiktoken's ``<tempdir>/data-gym-cache`` although its help text and
the offline guide said ``~/.tiktoken_cache`` — so the guide's
``tar ... ~/.tiktoken_cache`` packed a directory that did not exist.
"""

import os

import pytest

import lightrag.tools.download_cache as download_cache

pytestmark = pytest.mark.offline


@pytest.fixture
def home(monkeypatch, tmp_path):
    for name in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    return fake_home


class _FakeEncoding:
    def encode(self, text):
        return [0]


@pytest.fixture
def seen_cache_dirs(monkeypatch):
    import tiktoken

    seen = []

    def fake_load(name):
        seen.append(os.environ.get("TIKTOKEN_CACHE_DIR"))
        return _FakeEncoding()

    monkeypatch.setattr(tiktoken, "encoding_for_model", fake_load)
    monkeypatch.setattr(tiktoken, "get_encoding", fake_load)
    return seen


def test_default_is_home_cache_not_temp_dir(home, seen_cache_dirs, capsys):
    success, failed = download_cache.download_tiktoken_cache()

    expected = str(home / ".tiktoken_cache")
    assert (success, failed) == (len(seen_cache_dirs), [])
    assert (home / ".tiktoken_cache").is_dir()
    assert set(seen_cache_dirs) == {expected}
    assert "data-gym-cache" not in capsys.readouterr().out


def test_environment_directory_is_honoured(
    home, monkeypatch, tmp_path, seen_cache_dirs
):
    target = tmp_path / "from-env"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(target))

    download_cache.download_tiktoken_cache(models=["cl100k_base"])

    assert seen_cache_dirs == [str(target)]
    assert target.is_dir()
    assert not (home / ".tiktoken_cache").exists()


def test_data_gym_directory_is_honoured(home, monkeypatch, tmp_path, seen_cache_dirs):
    target = tmp_path / "data-gym"
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(target))

    download_cache.download_tiktoken_cache(models=["cl100k_base"])

    assert seen_cache_dirs == [str(target)]
    assert not (home / ".tiktoken_cache").exists()


def test_explicit_directory_is_expanded_and_used(home, seen_cache_dirs):
    download_cache.download_tiktoken_cache("~/offline/tiktoken", ["cl100k_base"])

    expected = home / "offline" / "tiktoken"
    assert seen_cache_dirs == [str(expected)]
    assert expected.is_dir()


def test_unresolvable_home_fails_instead_of_using_temp_dir(
    home, monkeypatch, seen_cache_dirs
):
    monkeypatch.setattr(os.path, "expanduser", lambda path: path)

    with pytest.raises(SystemExit) as excinfo:
        download_cache.download_tiktoken_cache()

    assert excinfo.value.code == 1
    assert seen_cache_dirs == []
