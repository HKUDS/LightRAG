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
