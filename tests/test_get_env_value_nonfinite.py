"""get_env_value must reject non-finite float env values."""

import math

import pytest

from lightrag.utils import get_env_value

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "+inf"])
def test_get_env_value_rejects_nonfinite_float(raw, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_TEST_FLOAT_ENV", raw)
    assert get_env_value("LIGHTRAG_TEST_FLOAT_ENV", 1.5, float) == 1.5


def test_get_env_value_keeps_finite_float(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_TEST_FLOAT_ENV", "0.42")
    assert get_env_value("LIGHTRAG_TEST_FLOAT_ENV", 1.5, float) == pytest.approx(0.42)


def test_get_env_value_missing_uses_default(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_TEST_FLOAT_ENV", raising=False)
    assert get_env_value("LIGHTRAG_TEST_FLOAT_ENV", 1.5, float) == 1.5
    assert math.isfinite(get_env_value("LIGHTRAG_TEST_FLOAT_ENV", 1.5, float))
