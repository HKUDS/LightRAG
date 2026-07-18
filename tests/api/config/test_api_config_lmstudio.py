import sys

import pytest

from lightrag.api.config import get_default_host, parse_args

pytestmark = pytest.mark.offline


def test_get_default_host_lmstudio(monkeypatch):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)

    assert get_default_host("lmstudio") == "http://localhost:1234/v1"


def test_get_default_host_lmstudio_custom_host(monkeypatch):
    monkeypatch.setenv("LLM_BINDING_HOST", "http://127.0.0.1:8080/v1")

    assert get_default_host("lmstudio") == "http://127.0.0.1:8080/v1"


def test_parse_args_accepts_lmstudio_llm_binding(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.delenv("LLM_BINDING", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING", raising=False)
    monkeypatch.setenv("LLM_BINDING", "lmstudio")
    monkeypatch.setenv("EMBEDDING_BINDING", "lmstudio")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "any-available")
    monkeypatch.setenv("EMBEDDING_MODEL", "any-available")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)

    args = parse_args()

    assert args.llm_binding == "lmstudio"
    assert args.embedding_binding == "lmstudio"
    assert args.llm_binding_host == "http://127.0.0.1:1234/v1"
    assert args.embedding_binding_host == "http://127.0.0.1:1234/v1"
    assert args.llm_model == "any-available"
    assert args.embedding_model == "any-available"
    assert args.embedding_dim is None
