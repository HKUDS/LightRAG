"""get_env_value must warn when a non-blank env value fails conversion."""

import logging

import pytest

from lightrag.utils import get_env_value

pytestmark = pytest.mark.offline

ENV_KEY = "LIGHTRAG_TEST_CONVERT_ENV"


def _warnings(records):
    return [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and ENV_KEY in r.getMessage()
    ]


@pytest.mark.parametrize(
    ("raw", "value_type", "default"),
    [("abc", int, 7), ("1.5", int, 7), ("12x", float, 0.5)],
)
def test_unconvertible_value_falls_back_with_warning(
    raw, value_type, default, monkeypatch, lightrag_log_records
):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, default, value_type) == default
    messages = _warnings(lightrag_log_records)
    assert len(messages) == 1
    assert repr(raw) in messages[0]
    assert value_type.__name__ in messages[0]


@pytest.mark.parametrize("raw", ["", "  ", "\t"])
def test_blank_value_falls_back_silently(raw, monkeypatch, lightrag_log_records):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, 7, int) == 7
    assert _warnings(lightrag_log_records) == []


def test_convertible_value_does_not_warn(monkeypatch, lightrag_log_records):
    monkeypatch.setenv(ENV_KEY, "42")
    assert get_env_value(ENV_KEY, 7, int) == 42
    assert _warnings(lightrag_log_records) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), (" ON ", True), ("off", False), ("0", False), ("", False)],
)
def test_recognized_bool_does_not_warn(
    raw, expected, monkeypatch, lightrag_log_records
):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, not expected, bool) is expected
    assert _warnings(lightrag_log_records) == []


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("raw", ["ture", "enabled", "2"])
def test_unrecognized_bool_falls_back_with_warning(
    raw, default, monkeypatch, lightrag_log_records
):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, default, bool) is default
    messages = _warnings(lightrag_log_records)
    assert len(messages) == 1
    assert repr(raw) in messages[0]
