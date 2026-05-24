"""Offline tests for VLM_PROCESS_ENABLE and renamed role timeout vars."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


def _reset_vlm_env(monkeypatch):
    for key in (
        "VLM_PROCESS_ENABLE",
        "VLM_LLM_BINDING",
        "VLM_LLM_MODEL",
        "VLM_LLM_BINDING_HOST",
        "VLM_LLM_BINDING_API_KEY",
        "VLM_LLM_TIMEOUT",
        "EXTRACT_LLM_TIMEOUT",
        "KEYWORD_LLM_TIMEOUT",
        "QUERY_LLM_TIMEOUT",
        "LLM_TIMEOUT_VLM_LLM",
        "LLM_TIMEOUT_EXTRACT_LLM",
        "LLM_TIMEOUT_KEYWORD_LLM",
        "LLM_TIMEOUT_QUERY_LLM",
    ):
        monkeypatch.delenv(key, raising=False)


def test_vlm_process_enable_defaults_to_false(monkeypatch):
    _reset_vlm_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")

    args = parse_args()

    assert args.vlm_process_enable is False


def test_vlm_process_enable_true_with_openai_passes(monkeypatch):
    _reset_vlm_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")

    args = parse_args()

    assert args.vlm_process_enable is True


def test_vlm_process_enable_rejects_lollms_base_binding(monkeypatch):
    _reset_vlm_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "lollms")
    monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")

    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert "lollms" in str(exc.value).lower()


def test_vlm_process_enable_rejects_lollms_role_binding(monkeypatch):
    _reset_vlm_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")
    monkeypatch.setenv("VLM_LLM_BINDING", "lollms")
    # Cross-provider validation needs model + key; fill them so the lollms
    # branch is the only failure path.
    monkeypatch.setenv("VLM_LLM_MODEL", "anything")
    monkeypatch.setenv("VLM_LLM_BINDING_HOST", "http://localhost:9600")
    monkeypatch.setenv("VLM_LLM_BINDING_API_KEY", "placeholder")

    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert "lollms" in str(exc.value).lower()


def test_role_timeout_uses_new_variable_names(monkeypatch):
    _reset_vlm_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EXTRACT_LLM_TIMEOUT", "240")
    monkeypatch.setenv("KEYWORD_LLM_TIMEOUT", "120")
    monkeypatch.setenv("QUERY_LLM_TIMEOUT", "60")
    monkeypatch.setenv("VLM_LLM_TIMEOUT", "300")

    args = parse_args()

    assert args.extract_llm_timeout == 240
    assert args.keyword_llm_timeout == 120
    assert args.query_llm_timeout == 60
    assert args.vlm_llm_timeout == 300


def test_role_timeout_legacy_variables_no_longer_have_effect(monkeypatch):
    """The breaking-change migration: legacy LLM_TIMEOUT_{ROLE}_LLM is silently ignored."""
    _reset_vlm_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_TIMEOUT_EXTRACT_LLM", "999")
    monkeypatch.setenv("LLM_TIMEOUT_VLM_LLM", "888")

    args = parse_args()

    assert args.extract_llm_timeout is None
    assert args.vlm_llm_timeout is None
