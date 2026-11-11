"""Tests for LightRAG's default tiktoken cache location.

Regression: tiktoken falls back to ``<tempdir>/data-gym-cache`` when no cache
directory is configured, while the offline guide and
``lightrag-download-cache --help`` both promised ``~/.tiktoken_cache``. A cache
copied there was never read, and a cache left in ``/tmp`` vanished on reboot.
LightRAG now adopts ``~/.tiktoken_cache`` when it exists and nothing explicit
is configured — and never creates it at runtime.
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
