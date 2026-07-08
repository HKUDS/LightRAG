"""Offline tests for LIGHTRAG_KEYWORD_STORAGE env wiring."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import DefaultRAGStorageConfig, parse_args


pytestmark = pytest.mark.offline


def _reset_keyword_storage_env(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_KEYWORD_STORAGE", raising=False)


def test_keyword_storage_defaults_to_bm25(monkeypatch):
    _reset_keyword_storage_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")

    args = parse_args()

    assert args.keyword_storage == "Bm25KeywordStorage"
    assert DefaultRAGStorageConfig.KEYWORD_STORAGE == "Bm25KeywordStorage"


def test_keyword_storage_env_override(monkeypatch):
    _reset_keyword_storage_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LIGHTRAG_KEYWORD_STORAGE", "CustomKeywordStorage")

    args = parse_args()

    assert args.keyword_storage == "CustomKeywordStorage"
