"""Ollama embedding dimension resolution (API config layer).

EMBEDDING_DIM has no way to be probed ahead of a real embed call, so the
Ollama binding used a single static default (1024, tuned for bge-m3, its
default model) for every model. That default is silently wrong for any
other model a user points EMBEDDING_MODEL at -- e.g. nomic-embed-text is
actually 768-dim -- and the failure only surfaces later, mid-ingestion, as
a confusing vector-store "dimension mismatch" error with no link back to
the real cause. See issue #3596.

KNOWN_OLLAMA_EMBEDDING_DIMS (lightrag/llm/ollama.py) resolves the real
dimension for recognized models without requiring EMBEDDING_DIM to be set;
unrecognized models keep the previous (bge-m3) default but now log a clear
warning naming the actual model, instead of failing silently.
"""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args

pytestmark = pytest.mark.offline


def _build_embedding_func(monkeypatch, *, model: str, embedding_dim: str | None = None):
    # lightrag.api.lightrag_server transitively imports lightrag.api.auth, whose
    # module-level `AuthHandler()` singleton reads sys.argv (via global_args) on
    # first import. Patch argv before that first import is triggered (below),
    # not just before parse_args() -- otherwise pytest's own argv reaches
    # argparse and the collection-time import fails outright.
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import create_embedding_function_from_args

    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    if embedding_dim is None:
        monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_DIM", embedding_dim)
    args = parse_args()
    return create_embedding_function_from_args(args)


def test_known_model_resolves_its_real_dimension_not_the_bge_m3_default(monkeypatch):
    """The exact regression from #3596: nomic-embed-text is 768-dim, not 1024."""
    embedding_func = _build_embedding_func(monkeypatch, model="nomic-embed-text")

    assert embedding_func.embedding_dim == 768


def test_a_tag_suffix_on_a_known_model_still_resolves(monkeypatch):
    """Model names commonly carry a tag (":latest", ":v1.5", ...); matching
    must not require the user to omit it."""
    embedding_func = _build_embedding_func(monkeypatch, model="nomic-embed-text:latest")

    assert embedding_func.embedding_dim == 768


def test_default_model_is_unaffected(monkeypatch):
    """bge-m3 (the binding's own default model) keeps working exactly as
    before -- this table only adds coverage, it doesn't change existing
    correct behavior."""
    embedding_func = _build_embedding_func(monkeypatch, model="bge-m3")

    assert embedding_func.embedding_dim == 1024


def test_unknown_model_falls_back_to_1024_with_a_clear_warning(monkeypatch):
    """Models this table doesn't recognize keep the historical behavior
    (no regression for anyone currently relying on the 1024 default by
    coincidence) but must no longer fail silently -- the warning is the
    difference between this and the original bug.

    Spies directly on logger.warning rather than using caplog: the
    "lightrag" logger's propagate=False plus its own handler being attached
    on first import (which this test necessarily triggers, since
    lightrag.api.lightrag_server can only be imported after sys.argv is
    patched -- see _build_embedding_func) left caplog's handler seeing no
    records no matter how the import/attach ordering was arranged. A direct
    spy has no such ordering dependency.
    """
    warning_calls: list[str] = []
    from lightrag.utils import logger

    monkeypatch.setattr(
        logger, "warning", lambda msg, *a, **kw: warning_calls.append(msg)
    )

    embedding_func = _build_embedding_func(
        monkeypatch, model="some-future-embedding-model"
    )

    assert embedding_func.embedding_dim == 1024
    assert any("some-future-embedding-model" in msg for msg in warning_calls)
    assert any("EMBEDDING_DIM" in msg for msg in warning_calls)


def test_explicit_embedding_dim_still_wins_over_the_lookup_table(monkeypatch):
    """User-set EMBEDDING_DIM is an explicit override and must take priority
    over table-driven resolution, e.g. for a fine-tuned or quantized variant
    of a known model that changed its output size."""
    embedding_func = _build_embedding_func(
        monkeypatch, model="nomic-embed-text", embedding_dim="512"
    )

    assert embedding_func.embedding_dim == 512
