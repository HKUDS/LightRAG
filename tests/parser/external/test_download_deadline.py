"""Offline validation of the operator-controlled download deadline."""

import pytest

from lightrag.constants import DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT
from lightrag.parser.external._common import download_deadline_seconds

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "+Infinity", "-inf", "1e999"])
def test_nonfinite_deadline_uses_default(monkeypatch, caplog, raw):
    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", raw)
    assert download_deadline_seconds() == float(
        DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT
    )
    assert "PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", None), ("-1", None), ("0.25", 0.25), (" 2.5 ", 2.5)],
)
def test_finite_deadline_preserves_operator_setting(monkeypatch, raw, expected):
    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", raw)
    assert download_deadline_seconds() == expected
