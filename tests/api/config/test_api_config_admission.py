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
from lightrag.constants import (
    DEFAULT_MAX_PENDING_DOCUMENTS,
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_TEXTS_PER_REQUEST,
)

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


# --------------------------------------------------------------------------- #
# MAX_REQUEST_BODY_BYTES (LR2 §9.4)
# --------------------------------------------------------------------------- #


def _parse_body_limit(monkeypatch, value: str | None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv("MAX_REQUEST_BODY_BYTES", raising=False)
    else:
        monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", value)
    return parse_args()


def test_body_limit_is_enabled_by_default(monkeypatch):
    """Was 0 (off) and covered three ingestion routes only, which let an operator
    set it and stay exposed on the one route that needs no credentials."""
    args = _parse_body_limit(monkeypatch, None)
    assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
    assert args.max_request_body_bytes > 0
    assert args.max_request_body_bytes_explicit is False
    validate_admission_configuration(args)


def test_body_limit_can_be_explicitly_disabled(monkeypatch):
    args = _parse_body_limit(monkeypatch, "0")
    assert args.max_request_body_bytes == 0
    validate_admission_configuration(args)


def test_body_limit_env_value_is_honoured(monkeypatch):
    args = _parse_body_limit(monkeypatch, "1048576")
    assert args.max_request_body_bytes == 1048576
    validate_admission_configuration(args)


def test_provenance_survives_a_value_equal_to_the_default(monkeypatch):
    """The provenance flag is the whole point: the value alone cannot say this.

    ``MAX_REQUEST_BODY_BYTES=1048576`` is exactly the default, so anything that
    reconstructs "was it configured?" from the number gets this wrong and hands
    /documents/text(s) the 50 MiB tier instead of the configured 1 MiB.
    """
    args = _parse_body_limit(monkeypatch, str(DEFAULT_MAX_REQUEST_BODY_BYTES))
    assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
    assert args.max_request_body_bytes_explicit is True


def test_an_unparseable_value_falls_back_to_the_default_tiering(monkeypatch):
    """Falling back to the default value must also fall back to the default
    tiers, not leave the ingestion routes pinned to a value nobody chose."""
    args = _parse_body_limit(monkeypatch, "not-a-number")
    assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
    assert args.max_request_body_bytes_explicit is False


def test_negative_body_limit_fails_fast(monkeypatch):
    args = _parse_body_limit(monkeypatch, "-1")
    with pytest.raises(ValueError, match="MAX_REQUEST_BODY_BYTES"):
        validate_admission_configuration(args)


def test_body_limit_absent_from_a_partial_namespace_is_fine():
    """Validated independently of the capacity: a namespace carrying only one of
    the knobs must still pass."""
    validate_admission_configuration(Namespace(max_pending_documents=5))


# --------------------------------------------------------------------------- #
# MAX_TEXTS_PER_REQUEST (LR2 §11)
# --------------------------------------------------------------------------- #


def _parse_texts_limit(monkeypatch, value: str | None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv("MAX_TEXTS_PER_REQUEST", raising=False)
    else:
        monkeypatch.setenv("MAX_TEXTS_PER_REQUEST", value)
    return parse_args()


def test_texts_limit_disabled_by_default(monkeypatch):
    args = _parse_texts_limit(monkeypatch, None)
    assert args.max_texts_per_request == DEFAULT_MAX_TEXTS_PER_REQUEST == 0
    validate_admission_configuration(args)


def test_texts_limit_env_value_is_honoured(monkeypatch):
    args = _parse_texts_limit(monkeypatch, "50")
    assert args.max_texts_per_request == 50
    validate_admission_configuration(args)


def test_negative_texts_limit_fails_fast(monkeypatch):
    args = _parse_texts_limit(monkeypatch, "-1")
    with pytest.raises(ValueError, match="MAX_TEXTS_PER_REQUEST"):
        validate_admission_configuration(args)


# --------------------------------------------------------------------------- #
# PIPELINE_REQUIRE_STRICT_STORAGE_READS (LR2 §11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_strict_reads_gate_parses_as_a_bool(monkeypatch, value, expected):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv("PIPELINE_REQUIRE_STRICT_STORAGE_READS", raising=False)
    else:
        monkeypatch.setenv("PIPELINE_REQUIRE_STRICT_STORAGE_READS", value)

    args = parse_args()

    assert args.pipeline_require_strict_storage_reads is expected
