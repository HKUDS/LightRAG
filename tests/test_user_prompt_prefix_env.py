"""``USER_PROMPT_PREFIX`` / ``USER_PROMPT_PREFIX_FILE`` config resolution.

The prefix is operator configuration, so it is returned verbatim: no
stripping, no truncation, no sanitization. The trailing newlines an operator
writes ARE the separator between their policy and the caller's own
``user_prompt``, so normalizing them away would break the documented contract.

The loader must also never raise: it runs inside ``LightRAG()`` through a
dataclass ``default_factory``, where an exception surfaces far from its cause.
Every failure mode degrades to "no prefix" plus a log line.
"""

from __future__ import annotations

import pytest

from lightrag.base import QueryParam
from lightrag.prompt import load_user_prompt_prefix_source
from lightrag.utils import logger as lightrag_logger


pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _clear_prefix_env(monkeypatch):
    # load_dotenv(override=False) refills deleted names on the next module
    # import, so pin neutral values rather than using delenv.
    monkeypatch.setenv("USER_PROMPT_PREFIX", "")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "")
    # The lightrag logger sets propagate=False, so caplog sees nothing without
    # this (same approach as tests/parser/markdown/test_raw_cache.py).
    monkeypatch.setattr(lightrag_logger, "propagate", True)


def _prefix_dir(monkeypatch, tmp_path):
    prompt_dir = tmp_path / "prompts"
    (prompt_dir / "user_prompt").mkdir(parents=True)
    monkeypatch.setenv("PROMPT_DIR", str(prompt_dir))
    return prompt_dir / "user_prompt"


# ---------------------------------------------------------------------------
# Inline env var.
# ---------------------------------------------------------------------------


def test_unset_yields_no_prefix():
    assert load_user_prompt_prefix_source() == ""


def test_inline_value_is_returned_verbatim(monkeypatch):
    monkeypatch.setenv("USER_PROMPT_PREFIX", "Answer in Chinese.")
    assert load_user_prompt_prefix_source() == "Answer in Chinese."


def test_inline_trailing_newlines_are_preserved(monkeypatch):
    """The operator's separator must survive: stripping it would break joins."""
    monkeypatch.setenv("USER_PROMPT_PREFIX", "Answer in Chinese.\n\n")
    assert load_user_prompt_prefix_source() == "Answer in Chinese.\n\n"


def test_inline_special_characters_are_preserved(monkeypatch):
    """Quotes and '#' are a .env quoting concern, not a loader concern."""
    value = 'Use "quotes", it\'s fine, # too.\n\n'
    monkeypatch.setenv("USER_PROMPT_PREFIX", value)
    assert load_user_prompt_prefix_source() == value


def test_long_inline_value_is_not_truncated(monkeypatch):
    """No length bound: this is operator config, not client input."""
    value = "policy. " * 5000
    monkeypatch.setenv("USER_PROMPT_PREFIX", value)
    assert load_user_prompt_prefix_source() == value


# ---------------------------------------------------------------------------
# File source.
# ---------------------------------------------------------------------------


def test_file_source_is_read_verbatim(monkeypatch, tmp_path):
    prefix_dir = _prefix_dir(monkeypatch, tmp_path)
    (prefix_dir / "house.md").write_text("House style.\n\n", encoding="utf-8")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.md")

    assert load_user_prompt_prefix_source() == "House style.\n\n"


def test_file_source_wins_over_the_inline_value(monkeypatch, tmp_path, caplog):
    prefix_dir = _prefix_dir(monkeypatch, tmp_path)
    (prefix_dir / "house.md").write_text("From file.\n\n", encoding="utf-8")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.md")
    monkeypatch.setenv("USER_PROMPT_PREFIX", "From inline.")

    with caplog.at_level("WARNING", logger="lightrag"):
        assert load_user_prompt_prefix_source() == "From file.\n\n"
    assert "USER_PROMPT_PREFIX_FILE" in caplog.text


def test_multiline_file_content_is_preserved(monkeypatch, tmp_path):
    """Why the file source exists: .env values must stay on one line."""
    prefix_dir = _prefix_dir(monkeypatch, tmp_path)
    content = "Line one.\nLine two.\n\nParagraph two: ${not_interpolated} #hash\n\n"
    (prefix_dir / "house.md").write_text(content, encoding="utf-8")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.md")

    assert load_user_prompt_prefix_source() == content


def test_txt_suffix_is_accepted(monkeypatch, tmp_path):
    prefix_dir = _prefix_dir(monkeypatch, tmp_path)
    (prefix_dir / "house.txt").write_text("Plain text.\n", encoding="utf-8")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.txt")

    assert load_user_prompt_prefix_source() == "Plain text.\n"


# ---------------------------------------------------------------------------
# Failure modes: log and degrade, never raise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape.md",
        "/etc/passwd.md",
        "sub/dir.md",
        "sub\\dir.md",
        "profile.yml",
        "noextension",
    ],
)
def test_rejected_file_names_degrade_to_no_prefix(
    monkeypatch, tmp_path, caplog, bad_name
):
    _prefix_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", bad_name)

    with caplog.at_level("ERROR", logger="lightrag"):
        assert load_user_prompt_prefix_source() == ""
    assert "USER_PROMPT_PREFIX_FILE" in caplog.text


def test_missing_file_degrades_to_no_prefix(monkeypatch, tmp_path, caplog):
    _prefix_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "absent.md")

    with caplog.at_level("ERROR", logger="lightrag"):
        assert load_user_prompt_prefix_source() == ""
    assert "absent.md" in caplog.text


def test_undecodable_file_degrades_to_no_prefix(monkeypatch, tmp_path, caplog):
    prefix_dir = _prefix_dir(monkeypatch, tmp_path)
    (prefix_dir / "house.md").write_bytes(b"\xff\xfe invalid utf-8")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.md")

    with caplog.at_level("ERROR", logger="lightrag"):
        assert load_user_prompt_prefix_source() == ""


def test_unresolvable_prompt_dir_degrades_to_no_prefix(monkeypatch, caplog):
    """PROMPT_DIR resolution can raise more than ValueError.

    ``Path.expanduser()`` raises RuntimeError when a home directory cannot be
    determined, and ``Path.resolve()`` raises RuntimeError (<=3.12) or OSError
    on a symlink loop. A `~unknownuser` prefix triggers the same except branch
    as a symlink loop while needing no filesystem setup and behaving the same
    on every platform.
    """
    monkeypatch.setenv("PROMPT_DIR", "~nosuchuser12345/prompts")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.md")

    with caplog.at_level("ERROR", logger="lightrag"):
        assert load_user_prompt_prefix_source() == ""
    assert "USER_PROMPT_PREFIX_FILE" in caplog.text


def test_unresolvable_prompt_dir_does_not_abort_lightrag_construction(monkeypatch):
    """The consequence that actually matters, pinned directly.

    The loader runs as a dataclass default_factory, so an escaping exception
    does not merely skip the prefix -- it makes ``LightRAG()`` unconstructible,
    with a traceback far from the misconfigured environment variable that
    caused it.
    """
    from lightrag import LightRAG

    monkeypatch.setenv("PROMPT_DIR", "~nosuchuser12345/prompts")
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "house.md")

    factory = LightRAG.__dataclass_fields__["user_prompt_prefix"].default_factory
    assert factory() == ""


def test_a_rejected_file_does_not_fall_back_to_the_inline_value(
    monkeypatch, tmp_path, caplog
):
    """Silently using a different policy than the one named would be worse.

    The operator asked for a specific file; if it cannot be used, apply no
    prefix rather than quietly substituting the inline value they had already
    been told is ignored.
    """
    _prefix_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("USER_PROMPT_PREFIX_FILE", "absent.md")
    monkeypatch.setenv("USER_PROMPT_PREFIX", "Inline fallback.")

    with caplog.at_level("ERROR", logger="lightrag"):
        assert load_user_prompt_prefix_source() == ""


# ---------------------------------------------------------------------------
# Wiring into the dataclasses.
# ---------------------------------------------------------------------------


def test_lightrag_field_factory_reads_the_env(monkeypatch):
    """The field uses default_factory, so the env is read per instantiation.

    That is what lets this test avoid the subprocess dance
    ``tests/test_queryparam_empty_int_env.py`` needs for QueryParam's
    import-time defaults.
    """
    from lightrag import LightRAG

    factory = LightRAG.__dataclass_fields__["user_prompt_prefix"].default_factory
    assert factory() == ""

    monkeypatch.setenv("USER_PROMPT_PREFIX", "Answer in Chinese.\n\n")
    assert factory() == "Answer in Chinese.\n\n"


def test_queryparam_switch_defaults_to_applying_the_prefix():
    assert QueryParam().disable_user_prompt_prefix is False
