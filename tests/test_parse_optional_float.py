"""Regression tests for :func:`lightrag.utils.parse_optional_float`."""

from __future__ import annotations

import pytest

from lightrag.utils import parse_optional_float

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "+inf", "-inf"])
def test_parse_optional_float_rejects_nonfinite(raw: str):
    with pytest.raises(ValueError, match="finite"):
        parse_optional_float(raw)


def test_parse_optional_float_accepts_finite_and_unset():
    assert parse_optional_float("1.5") == 1.5
    assert parse_optional_float(None) is None
    assert parse_optional_float("") is None
    assert parse_optional_float("None") is None
