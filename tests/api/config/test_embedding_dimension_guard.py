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

    def test_message_names_the_deployment_that_wins_at_runtime(self):
        """azure_openai_embed resolves AZURE_EMBEDDING_DEPLOYMENT *before* the
        configured model, so the guard must name the deployment, not the model."""
        with patch.dict("os.environ", {"AZURE_EMBEDDING_DEPLOYMENT": "env-deployment"}):
            with pytest.raises(ValueError) as excinfo:
                _create_embedding_function(
                    binding="azure_openai",
                    model="model-deployment",
                    embedding_dim=None,
                )
        assert "env-deployment" in str(excinfo.value)
        assert "model-deployment" not in str(excinfo.value)


# --- Non-positive EMBEDDING_DIM must never satisfy the guard ---


class TestNonPositiveEmbeddingDim:
    """EMBEDDING_DIM=0 is not "a dimension is configured".

    The effective dimension is resolved with a truthiness test, so 0 silently
    falls back to the provider default. The guard must treat it as unset, and
    the config layer must reject it outright.
    """

    def test_zero_dim_does_not_satisfy_the_custom_model_guard(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*text-embedding-3-large"):
            _create_embedding_function(
                binding="openai",
                model="text-embedding-3-large",
                embedding_dim=0,
            )

    def test_zero_dim_does_not_satisfy_the_azure_guard(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*Azure OpenAI"):
            _create_embedding_function(
                binding="azure_openai",
                model="my-custom-embedding-deployment",
                embedding_dim=0,
            )

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_config_layer_rejects_non_positive_dim(self, monkeypatch, value):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("EMBEDDING_DIM", value)
        from lightrag.api.config import parse_args

        with pytest.raises(SystemExit, match=r"EMBEDDING_DIM must be a positive"):
            parse_args()

    def test_config_layer_accepts_positive_dim(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("EMBEDDING_DIM", "1024")
        from lightrag.api.config import parse_args

        assert parse_args().embedding_dim == 1024


# --- lollms (no guard — ignores model parameter) ---


class TestLollmsNoGuard:
    def test_no_guard_applied(self):
        func = _create_embedding_function(
            binding="lollms",
            model="some-model",
            embedding_dim=None,
        )
        assert func is not None
