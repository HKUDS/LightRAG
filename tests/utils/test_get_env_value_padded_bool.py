"""Regression tests for whitespace-padded boolean env values in :func:`lightrag.utils.get_env_value`."""

import pytest

from lightrag.utils import get_env_value

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("raw", [" true ", "  false\n", "\tyes\t", " 1 ", " ON "])
def test_get_env_value_padded_bool(raw, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_TEST_BOOL_ENV", raw)
    expected = raw.strip().lower() in ("true", "1", "yes", "t", "on")
    assert get_env_value("LIGHTRAG_TEST_BOOL_ENV", not expected, bool) is expected
