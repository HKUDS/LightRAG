"""Offline tests for the Atlas Cloud LLM binding."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import (
    DEFAULT_ATLASCLOUD_HOST,
    DEFAULT_ATLASCLOUD_MODEL,
    parse_args,
)


pytestmark = pytest.mark.offline


def _reset_atlascloud_env(monkeypatch):
    for key in (
        "LLM_BINDING",
        "LLM_BINDING_HOST",
        "LLM_BINDING_API_KEY",
        "LLM_MODEL",
        "ATLASCLOUD_API_KEY",
        "VLM_PROCESS_ENABLE",
        "VLM_LLM_BINDING",
        "VLM_LLM_MODEL",
        "VLM_LLM_BINDING_HOST",
        "VLM_LLM_BINDING_API_KEY",
        "VLM_LLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_atlascloud_binding_uses_atlas_defaults(monkeypatch):
    _reset_atlascloud_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "atlascloud")
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-secret")

    args = parse_args()

    assert args.llm_binding == "atlascloud"
    assert args.llm_binding_host == DEFAULT_ATLASCLOUD_HOST
    assert args.llm_binding_api_key == "atlas-secret"
    assert args.llm_model == DEFAULT_ATLASCLOUD_MODEL


def test_atlascloud_binding_respects_explicit_openai_compatible_settings(monkeypatch):
    _reset_atlascloud_env(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--llm-binding",
            "atlascloud",
            "--atlascloud-llm-temperature",
            "0.2",
        ],
    )
    monkeypatch.setenv("LLM_BINDING_HOST", "https://proxy.example/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "binding-secret")
    monkeypatch.setenv("LLM_MODEL", "deepseek-ai/deepseek-v4-pro")

    args = parse_args()

    assert args.llm_binding == "atlascloud"
    assert args.llm_binding_host == "https://proxy.example/v1"
    assert args.llm_binding_api_key == "binding-secret"
    assert args.llm_model == "deepseek-ai/deepseek-v4-pro"
    assert args.atlascloud_llm_temperature == 0.2


def test_vlm_process_enable_true_with_atlascloud_passes(monkeypatch):
    _reset_atlascloud_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "atlascloud")
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-secret")
    monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")

    args = parse_args()

    assert args.vlm_process_enable is True
    assert args.llm_binding == "atlascloud"


def test_cross_provider_role_can_use_atlascloud_api_key_fallback(monkeypatch):
    _reset_atlascloud_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("QUERY_LLM_BINDING", "atlascloud")
    monkeypatch.setenv("QUERY_LLM_MODEL", DEFAULT_ATLASCLOUD_MODEL)
    monkeypatch.setenv("QUERY_LLM_BINDING_HOST", DEFAULT_ATLASCLOUD_HOST)
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-secret")

    args = parse_args()

    assert args.query_llm_binding == "atlascloud"
    assert args.query_llm_binding_api_key == "atlas-secret"
