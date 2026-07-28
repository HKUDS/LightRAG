"""SCAN_ENQUEUE_BATCH_SIZE parsing and fail-fast validation (LR2 §8.2/§11)."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import (
    initialize_config,
    parse_args,
    validate_scan_batch_configuration,
)
from lightrag.constants import DEFAULT_SCAN_ENQUEUE_BATCH_SIZE

pytestmark = pytest.mark.offline


def _parse(monkeypatch, value: str | None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv("SCAN_ENQUEUE_BATCH_SIZE", raising=False)
    else:
        monkeypatch.setenv("SCAN_ENQUEUE_BATCH_SIZE", value)
    return parse_args()


def test_defaults_when_unset(monkeypatch):
    args = _parse(monkeypatch, None)
    assert args.scan_enqueue_batch_size == DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
    validate_scan_batch_configuration(args)  # accepted


def test_env_value_is_honoured(monkeypatch):
    args = _parse(monkeypatch, "25")
    assert args.scan_enqueue_batch_size == 25
    validate_scan_batch_configuration(args)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_batch_size_fails_fast(monkeypatch, value):
    """Unlike PIPELINE_SCHEDULING_PAGE_SIZE there is no "disabled" reading: an
    unbounded scan batch is exactly what streaming discovery removes, so the
    server must refuse to start instead of guessing."""
    args = _parse(monkeypatch, value)
    with pytest.raises(ValueError, match="SCAN_ENQUEUE_BATCH_SIZE"):
        validate_scan_batch_configuration(args)


def test_initialize_config_enforces_the_batch_size(monkeypatch):
    """The validation runs on the real startup path, not just standalone."""
    args = _parse(monkeypatch, "0")
    with pytest.raises(ValueError, match="SCAN_ENQUEUE_BATCH_SIZE"):
        initialize_config(args, force=True)


def test_partial_namespace_is_accepted(monkeypatch):
    """``initialize_config`` documents a programmatic partial-namespace path, so a
    namespace that simply does not carry the knob must not be rejected — the
    scan's reader falls back to the bounded default. Only a PRESENT bad value is
    an operator misconfiguration."""
    from types import SimpleNamespace

    validate_scan_batch_configuration(SimpleNamespace())
    with pytest.raises(ValueError, match="SCAN_ENQUEUE_BATCH_SIZE"):
        validate_scan_batch_configuration(
            SimpleNamespace(scan_enqueue_batch_size=True)  # bool is not a size
        )
