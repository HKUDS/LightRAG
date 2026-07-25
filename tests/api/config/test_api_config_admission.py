"""MAX_PENDING_DOCUMENTS parsing and fail-fast validation (LR2 §9.1/§11)."""

from __future__ import annotations

import sys
from argparse import Namespace

import pytest

from lightrag.api.config import (
    initialize_config,
    parse_args,
    validate_admission_configuration,
)
from lightrag.constants import DEFAULT_MAX_PENDING_DOCUMENTS

pytestmark = pytest.mark.offline


def _parse(monkeypatch, value: str | None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv("MAX_PENDING_DOCUMENTS", raising=False)
    else:
        monkeypatch.setenv("MAX_PENDING_DOCUMENTS", value)
    return parse_args()


def test_disabled_by_default(monkeypatch):
    args = _parse(monkeypatch, None)
    assert args.max_pending_documents == DEFAULT_MAX_PENDING_DOCUMENTS == 0
    validate_admission_configuration(args)  # accepted


def test_env_value_is_honoured(monkeypatch):
    args = _parse(monkeypatch, "250")
    assert args.max_pending_documents == 250
    validate_admission_configuration(args)


def test_explicit_zero_is_accepted(monkeypatch):
    """Unlike the scan batch size, 0 is a meaningful setting here: admission off."""
    args = _parse(monkeypatch, "0")
    assert args.max_pending_documents == 0
    validate_admission_configuration(args)


def test_negative_capacity_fails_fast(monkeypatch):
    args = _parse(monkeypatch, "-5")
    with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
        validate_admission_configuration(args)


@pytest.mark.parametrize("bad", [True, 1.5, "many", None])
def test_non_integer_capacity_fails_fast(bad):
    with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
        validate_admission_configuration(Namespace(max_pending_documents=bad))


def test_partial_namespace_is_not_an_operator_error():
    """The documented programmatic path may pass a namespace without the field;
    that is not a misconfiguration and must not block startup."""
    validate_admission_configuration(Namespace())


def test_initialize_config_rejects_a_negative_capacity(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_PENDING_DOCUMENTS", "-1")
    with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
        initialize_config(force=True)
