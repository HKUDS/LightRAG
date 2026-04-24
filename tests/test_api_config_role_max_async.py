import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


ROLE_MAX_ASYNC_ENV_KEYS = (
    "MAX_ASYNC",
    "MAX_ASYNC_EXTRACT_LLM",
    "MAX_ASYNC_KEYWORD_LLM",
    "MAX_ASYNC_QUERY_LLM",
    "MAX_ASYNC_VLM_LLM",
)


def _clear_max_async_env(monkeypatch):
    for key in ROLE_MAX_ASYNC_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_role_max_async_defaults_none_when_env_unset(monkeypatch):
    _clear_max_async_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_ASYNC", "10")

    args = parse_args()

    assert args.max_async == 10
    assert args.extract_llm_max_async is None
    assert args.keyword_llm_max_async is None
    assert args.query_llm_max_async is None
    assert args.vlm_llm_max_async is None


def test_role_max_async_env_override_keeps_other_roles_none(monkeypatch):
    _clear_max_async_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_ASYNC", "10")
    monkeypatch.setenv("MAX_ASYNC_EXTRACT_LLM", "7")

    args = parse_args()

    assert args.max_async == 10
    assert args.extract_llm_max_async == 7
    assert args.keyword_llm_max_async is None
    assert args.query_llm_max_async is None
    assert args.vlm_llm_max_async is None


def test_role_max_async_literal_none_string_is_preserved(monkeypatch):
    _clear_max_async_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_ASYNC", "10")
    monkeypatch.setenv("MAX_ASYNC_QUERY_LLM", "None")

    args = parse_args()

    assert args.max_async == 10
    assert args.query_llm_max_async is None
