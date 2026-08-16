"""Startup validation for custom embedding dimensions across all bindings.

Extends the Ollama-specific guard (test_ollama_embedding_dimension.py) to
every binding that goes through create_optimized_embedding_function.
See: https://github.com/HKUDS/LightRAG/issues/3644
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.offline


def _create_embedding_function(
    *, binding: str, model: str | None, embedding_dim: int | None
):
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
        binding=binding,
        model=model,
        host="http://localhost:11434",
        api_key="test-key",
        args=args,
    )


# --- OpenAI (default: text-embedding-3-small, dim=1536) ---


class TestOpenAIEmbeddingDimGuard:
    def test_custom_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*text-embedding-3-large"):
            _create_embedding_function(
                binding="openai",
                model="text-embedding-3-large",
                embedding_dim=None,
            )

    def test_custom_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="openai",
            model="text-embedding-3-large",
            embedding_dim=3072,
        )
        assert func.embedding_dim == 3072

    def test_default_model_keeps_provider_dimension(self):
        func = _create_embedding_function(
            binding="openai",
            model=None,
            embedding_dim=None,
        )
        assert func.embedding_dim == 1536


# --- Jina (default: jina-embeddings-v4, dim=2048) ---


class TestJinaEmbeddingDimGuard:
    def test_custom_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*jina-embeddings-v3"):
            _create_embedding_function(
                binding="jina",
                model="jina-embeddings-v3",
                embedding_dim=None,
            )

    def test_custom_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="jina",
            model="jina-embeddings-v3",
            embedding_dim=1024,
        )
        assert func.embedding_dim == 1024

    def test_default_model_keeps_provider_dimension(self):
        func = _create_embedding_function(
            binding="jina",
            model=None,
            embedding_dim=None,
        )
        assert func.embedding_dim == 2048


# --- Gemini (default: gemini-embedding-001, dim=1536) ---


class TestGeminiEmbeddingDimGuard:
    def test_custom_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*text-embedding-004"):
            _create_embedding_function(
                binding="gemini",
                model="text-embedding-004",
                embedding_dim=None,
            )

    def test_custom_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="gemini",
            model="text-embedding-004",
            embedding_dim=768,
        )
        assert func.embedding_dim == 768

    def test_default_model_keeps_provider_dimension(self):
        func = _create_embedding_function(
            binding="gemini",
            model=None,
            embedding_dim=None,
        )
        assert func.embedding_dim == 1536


# --- Bedrock (default: amazon.titan-embed-text-v2:0, dim=1024) ---


class TestBedrockEmbeddingDimGuard:
    def test_custom_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*cohere.embed-english-v3"):
            _create_embedding_function(
                binding="bedrock",
                model="cohere.embed-english-v3",
                embedding_dim=None,
            )

    def test_custom_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="bedrock",
            model="cohere.embed-english-v3",
            embedding_dim=1024,
        )
        assert func.embedding_dim == 1024

    def test_default_model_keeps_provider_dimension(self):
        func = _create_embedding_function(
            binding="bedrock",
            model=None,
            embedding_dim=None,
        )
        assert func.embedding_dim == 1024


# --- VoyageAI (default: voyage-3, dim=1024) ---


class TestVoyageAIEmbeddingDimGuard:
    def test_custom_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*voyage-3-lite"):
            _create_embedding_function(
                binding="voyageai",
                model="voyage-3-lite",
                embedding_dim=None,
            )

    def test_custom_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="voyageai",
            model="voyage-3-lite",
            embedding_dim=512,
        )
        assert func.embedding_dim == 512

    def test_default_model_keeps_provider_dimension(self):
        func = _create_embedding_function(
            binding="voyageai",
            model=None,
            embedding_dim=None,
        )
        assert func.embedding_dim == 1024


# --- Azure OpenAI (always requires explicit dim) ---


class TestAzureOpenAIEmbeddingDimGuard:
    def test_any_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*Azure OpenAI"):
            _create_embedding_function(
                binding="azure_openai",
                model="my-custom-embedding-deployment",
                embedding_dim=None,
            )

    def test_configured_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="azure_openai",
            model="my-custom-embedding-deployment",
            embedding_dim=1536,
        )
        assert func.embedding_dim == 1536

    def test_env_deployment_without_model_requires_explicit_dim(self):
        """Regression: AZURE_EMBEDDING_DEPLOYMENT set but EMBEDDING_MODEL unset."""
        with patch.dict(
            "os.environ", {"AZURE_EMBEDDING_DEPLOYMENT": "text-embedding-3-large"}
        ):
            with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*Azure OpenAI"):
                _create_embedding_function(
                    binding="azure_openai",
                    model=None,
                    embedding_dim=None,
                )


# --- lollms (no guard — ignores model parameter) ---


class TestLollmsNoGuard:
    def test_no_guard_applied(self):
        func = _create_embedding_function(
            binding="lollms",
            model="some-model",
            embedding_dim=None,
        )
        assert func is not None
