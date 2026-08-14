"""Startup validation for custom Ollama embedding dimensions."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.offline


def _create_embedding_function(*, model: str | None, embedding_dim: int | None):
    with patch.object(sys, "argv", ["lightrag-server"]):
        from lightrag.api.lightrag_server import create_optimized_embedding_function

    args = SimpleNamespace(
        embedding_dim=embedding_dim,
        embedding_token_limit=None,
        embedding_asymmetric=False,
        embedding_asymmetric_configured=False,
        embedding_query_prefix_configured=False,
        embedding_document_prefix_configured=False,
    )
    return create_optimized_embedding_function(
        config_cache=MagicMock(ollama_embedding_options=None),
        binding="ollama",
        model=model,
        host="http://localhost:11434",
        api_key=None,
        args=args,
    )


def test_custom_ollama_model_requires_explicit_embedding_dimension():
    with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*nomic-embed-text"):
        _create_embedding_function(model="nomic-embed-text", embedding_dim=None)


def test_custom_ollama_model_accepts_explicit_embedding_dimension():
    embedding_func = _create_embedding_function(
        model="nomic-embed-text", embedding_dim=768
    )

    assert embedding_func.embedding_dim == 768


@pytest.mark.parametrize("model", [None, "bge-m3", "bge-m3:latest"])
def test_default_ollama_model_keeps_provider_dimension(model):
    embedding_func = _create_embedding_function(model=model, embedding_dim=None)

    assert embedding_func.embedding_dim == 1024
