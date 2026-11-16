"""Tests for LightRAG's default tiktoken cache location.

Regression: tiktoken falls back to ``<tempdir>/data-gym-cache`` when no cache
directory is configured, while the offline guide and
``lightrag-download-cache --help`` both promised ``~/.tiktoken_cache``. A cache
copied there was never read, and a cache left in ``/tmp`` vanished on reboot.
LightRAG now adopts ``~/.tiktoken_cache`` when it exists and nothing explicit
is configured — and never creates it at runtime.

Follow-up regression (PR review): an adopted but unwritable directory made a
missing encoding's cache write raise out of ``TiktokenTokenizer()`` on an
online host, where the old temp-dir default ignored write failures. The load
now falls back to tiktoken's default for the rest of the process.
"""

import os

import pytest

import lightrag.tiktoken_cache as tiktoken_cache

pytestmark = pytest.mark.offline


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fake home with no tiktoken variables set.

    setenv-then-delenv makes monkeypatch record the variables, so a value the
    code under test writes into os.environ is undone at teardown.
    """
    for name in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(tiktoken_cache, "_adopted_dir", None)
    monkeypatch.setattr(tiktoken_cache, "_adoption_disabled", False)
    return tmp_path


def test_adopts_existing_home_cache(home):
    (home / ".tiktoken_cache").mkdir()

    adopted = tiktoken_cache.apply_default_tiktoken_cache_dir()

    expected = str(home / ".tiktoken_cache")
    assert adopted == expected
    assert os.environ["TIKTOKEN_CACHE_DIR"] == expected


def test_missing_home_cache_is_neither_adopted_nor_created(home):
    assert tiktoken_cache.apply_default_tiktoken_cache_dir() is None
    assert "TIKTOKEN_CACHE_DIR" not in os.environ
    assert not (home / ".tiktoken_cache").exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TIKTOKEN_CACHE_DIR", "/srv/tiktoken"),
        ("TIKTOKEN_CACHE_DIR", ""),  # tiktoken's "disable caching" switch
        ("DATA_GYM_CACHE_DIR", "/srv/data-gym"),
    ],
)
def test_explicit_setting_wins(home, monkeypatch, name, value):
    (home / ".tiktoken_cache").mkdir()
    monkeypatch.setenv(name, value)

    assert tiktoken_cache.apply_default_tiktoken_cache_dir() is None
    assert os.environ[name] == value
    if name != "TIKTOKEN_CACHE_DIR":
        assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_unresolvable_home_is_left_alone(home, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda path: path)

    assert tiktoken_cache.default_tiktoken_cache_dir() is None
    assert tiktoken_cache.apply_default_tiktoken_cache_dir() is None
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


class _FakeEncoding:
    def encode(self, text):
        return list(range(len(text)))


@pytest.fixture
def seen_cache_dirs(monkeypatch):
    """Record TIKTOKEN_CACHE_DIR as tiktoken sees it when loading an encoding."""
    import tiktoken

    seen = []

    def fake_load(name):
        seen.append(os.environ.get("TIKTOKEN_CACHE_DIR"))
        return _FakeEncoding()

    monkeypatch.setattr(tiktoken, "encoding_for_model", fake_load)
    monkeypatch.setattr(tiktoken, "get_encoding", fake_load)
    return seen


def _load_via_tokenizer():
    from lightrag.utils import TiktokenTokenizer

    TiktokenTokenizer("gpt-4o-mini")


def _load_via_openai(monkeypatch):
    import lightrag.llm.openai as openai_mod

    monkeypatch.setattr(openai_mod, "_TIKTOKEN_ENCODING_CACHE", {})
    openai_mod._get_tiktoken_encoding_for_model("gpt-4o-mini")


def _load_via_smart_heading(monkeypatch):
    import lightrag.parser.docx.smart_heading.title_block as title_block

    monkeypatch.setattr(title_block, "_TIKTOKEN_ENCODER", None)
    title_block._estimate_tokens("heading")


@pytest.mark.parametrize("loader", ["tokenizer", "openai", "smart_heading"])
def test_every_encoding_load_site_applies_the_default(
    home, monkeypatch, seen_cache_dirs, loader
):
    (home / ".tiktoken_cache").mkdir()

    if loader == "tokenizer":
        _load_via_tokenizer()
    elif loader == "openai":
        _load_via_openai(monkeypatch)
    else:
        _load_via_smart_heading(monkeypatch)

    assert seen_cache_dirs == [str(home / ".tiktoken_cache")]


def _flaky_loader(calls, exc):
    """Fail the first call with ``exc``; succeed afterwards, recording the
    cache directory each call saw."""

    def load():
        calls.append(os.environ.get("TIKTOKEN_CACHE_DIR"))
        if len(calls) == 1:
            raise exc
        return "encoding"

    return load


def _writable(monkeypatch, writable):
    real_access = os.access
    monkeypatch.setattr(
        tiktoken_cache.os,
        "access",
        lambda path, mode: (
            writable if "tiktoken_cache" in str(path) else real_access(path, mode)
        ),
    )


def test_unwritable_adopted_dir_falls_back_to_tiktoken_default(
    home, monkeypatch, lightrag_log_records
):
    cache = home / ".tiktoken_cache"
    cache.mkdir()
    _writable(monkeypatch, False)
    calls = []

    result = tiktoken_cache.load_tiktoken_encoding(
        _flaky_loader(calls, PermissionError(13, "Permission denied"))
    )

    assert result == "encoding"
    assert calls == [str(cache), None]
    assert "TIKTOKEN_CACHE_DIR" not in os.environ
    assert any(
        r.levelname == "WARNING" and str(cache) in r.getMessage()
        for r in lightrag_log_records
    )
    # The directory is not re-adopted for later loads in this process.
    assert tiktoken_cache.apply_default_tiktoken_cache_dir() is None
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_unwritable_adopted_dir_still_serves_cached_encodings(home, monkeypatch):
    cache = home / ".tiktoken_cache"
    cache.mkdir()
    _writable(monkeypatch, False)
    calls = []

    def load():
        calls.append(os.environ.get("TIKTOKEN_CACHE_DIR"))
        return "encoding"

    assert tiktoken_cache.load_tiktoken_encoding(load) == "encoding"
    assert calls == [str(cache)]
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(cache)


def test_error_from_writable_adopted_dir_is_reraised(home, monkeypatch):
    """A network failure (requests' errors are OSErrors) must not be masked."""
    (home / ".tiktoken_cache").mkdir()
    _writable(monkeypatch, True)
    calls = []

    with pytest.raises(OSError, match="network down"):
        tiktoken_cache.load_tiktoken_encoding(
            _flaky_loader(calls, OSError("network down"))
        )

    assert len(calls) == 1
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(home / ".tiktoken_cache")


def test_error_from_user_configured_dir_is_reraised(home, monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(configured))
    _writable(monkeypatch, False)
    calls = []

    with pytest.raises(PermissionError):
        tiktoken_cache.load_tiktoken_encoding(
            _flaky_loader(calls, PermissionError(13, "Permission denied"))
        )

    assert len(calls) == 1
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(configured)
