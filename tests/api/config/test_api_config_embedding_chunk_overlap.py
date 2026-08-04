"""EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE env parsing (API server config layer).

Independent from CHUNK_OVERLAP_SIZE — see LightRAG.embedding_chunk_overlap_token_size.
Validation of the parsed value (negative rejection) happens at LightRAG
construction time, not here; this only covers that the API config layer
reads the env var into args at all.
"""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args
from lightrag.constants import DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE

pytestmark = pytest.mark.offline


def _parse(monkeypatch, value: str | None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv("EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE", value)
    return parse_args()


def test_defaults_when_unset(monkeypatch):
    args = _parse(monkeypatch, None)
    assert (
        args.embedding_chunk_overlap_token_size
        == DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
    )
    assert DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE == 100


def test_env_value_is_honoured(monkeypatch):
    args = _parse(monkeypatch, "42")
    assert args.embedding_chunk_overlap_token_size == 42


def test_zero_is_honoured(monkeypatch):
    """0 is a legal, meaningful value (disables the fallback's overlap) —
    must not be treated as falsy/unset."""
    args = _parse(monkeypatch, "0")
    assert args.embedding_chunk_overlap_token_size == 0


def test_does_not_share_a_value_with_chunk_overlap_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "7")
    monkeypatch.delenv("EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE", raising=False)
    args = parse_args()
    assert args.chunk_overlap_size == 7
    assert (
        args.embedding_chunk_overlap_token_size
        == DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
    )
