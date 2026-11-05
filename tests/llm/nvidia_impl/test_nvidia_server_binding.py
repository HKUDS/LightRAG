"""lightrag-server wiring for the ``nvidia`` embedding binding.

``lightrag.llm.nvidia_openai.nvidia_openai_embed`` already maps LightRAG's
query/document context onto NVIDIA's ``input_type`` (``query`` / ``passage``),
but the server could only reach NVIDIA through the generic ``openai`` binding,
which never sends ``input_type``. These tests pin the dedicated binding:
config parsing, the default host, dispatch to ``nvidia_openai_embed`` with the
configured host / key / model, and query/passage selection in asymmetric mode.

All tests mock ``AsyncOpenAI``; no live API calls.
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lightrag.llm_roles import ROLES

pytestmark = pytest.mark.offline

NVIDIA_DEFAULT_HOST = "https://integrate.api.nvidia.com/v1"
NVIDIA_TEST_MODEL = "nvidia/nemotron-3.5-embed-1b"

# Cleared so a developer-local .env cannot decide the outcome.
_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "VLM_PROCESS_ENABLE",
    "EMBEDDING_BINDING",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_TOKEN_LIMIT",
    "EMBEDDING_SEND_DIM",
    "EMBEDDING_ASYMMETRIC",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    *(
        f"{spec.env_prefix}_{suffix}"
        for spec in ROLES
        for suffix in (
            "LLM_BINDING",
            "LLM_BINDING_HOST",
            "LLM_BINDING_API_KEY",
            "LLM_MODEL",
        )
    ),
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch):
    """Pin sys.argv, import the server module, then clear the environment.

    Importing lightrag.api.lightrag_server runs module-level parse_args() and
    load_dotenv(override=False), so argv is pinned first and the deletions
    happen only after the import.
    """
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    import lightrag.api.lightrag_server  # noqa: F401

    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)


class _FakeAsyncOpenAI:
    """Records constructor kwargs and every embeddings.create() call."""

    instances: list["_FakeAsyncOpenAI"] = []
    dim = 4

    def __init__(self, **kwargs):
        self.client_kwargs = kwargs
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)
        _FakeAsyncOpenAI.instances.append(self)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * self.dim) for _ in kwargs["input"]]
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_openai():
    _FakeAsyncOpenAI.instances = []
    _FakeAsyncOpenAI.dim = 4
    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", _FakeAsyncOpenAI):
        yield _FakeAsyncOpenAI.instances


def _parse_nvidia_args(monkeypatch, **env):
    from lightrag.api.config import parse_args

    monkeypatch.setenv("EMBEDDING_BINDING", "nvidia")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return parse_args()


def _build_embedding_func(args):
    from lightrag.api.lightrag_server import create_embedding_function_from_args

    return create_embedding_function_from_args(args)


def test_nvidia_is_a_provider_task_asymmetric_binding():
    from lightrag.api.config import PROVIDER_ASYMMETRIC_EMBEDDING_BINDINGS

    assert "nvidia" in PROVIDER_ASYMMETRIC_EMBEDDING_BINDINGS


def test_get_default_host_for_nvidia_is_api_catalog(monkeypatch):
    from lightrag.api.config import get_default_host

    # An LLM host configured for another binding must not leak into the
    # embedding-only nvidia default.
    monkeypatch.setenv("LLM_BINDING_HOST", "http://localhost:11434")
    assert get_default_host("nvidia") == NVIDIA_DEFAULT_HOST


def test_parse_args_accepts_nvidia_env_binding(monkeypatch):
    args = _parse_nvidia_args(monkeypatch)

    assert args.embedding_binding == "nvidia"
    assert args.embedding_binding_host == NVIDIA_DEFAULT_HOST


def test_parse_args_accepts_nvidia_cli_binding(monkeypatch):
    from lightrag.api.config import parse_args

    monkeypatch.setattr(
        sys, "argv", ["lightrag-server", "--embedding-binding", "nvidia"]
    )
    args = parse_args()

    assert args.embedding_binding == "nvidia"


def test_embedding_binding_host_overrides_default_for_self_hosted_nim(monkeypatch):
    args = _parse_nvidia_args(
        monkeypatch, EMBEDDING_BINDING_HOST="http://nim.internal:8000/v1"
    )

    assert args.embedding_binding_host == "http://nim.internal:8000/v1"


@pytest.mark.asyncio
async def test_dispatch_passes_host_api_key_and_model(monkeypatch, fake_openai):
    args = _parse_nvidia_args(
        monkeypatch,
        EMBEDDING_BINDING_HOST="http://nim.internal:8000/v1",
        EMBEDDING_BINDING_API_KEY="test-nvidia-key",
        EMBEDDING_MODEL=NVIDIA_TEST_MODEL,
        EMBEDDING_DIM="4",
    )
    func = _build_embedding_func(args)

    result = await func(["hello"])

    assert result.shape == (1, 4)
    [client] = fake_openai
    assert client.client_kwargs == {
        "base_url": "http://nim.internal:8000/v1",
        "api_key": "test-nvidia-key",
    }
    assert client.calls[0]["model"] == NVIDIA_TEST_MODEL
    assert client.calls[0]["input"] == ["hello"]


@pytest.mark.asyncio
async def test_dispatch_uses_default_host_when_unset(monkeypatch, fake_openai):
    args = _parse_nvidia_args(
        monkeypatch,
        EMBEDDING_BINDING_API_KEY="test-nvidia-key",
        EMBEDDING_MODEL=NVIDIA_TEST_MODEL,
        EMBEDDING_DIM="4",
    )
    func = _build_embedding_func(args)

    await func(["hello"])

    assert fake_openai[0].client_kwargs["base_url"] == NVIDIA_DEFAULT_HOST


@pytest.mark.asyncio
async def test_dispatch_without_model_uses_provider_default(monkeypatch, fake_openai):
    from lightrag.llm.nvidia_openai import nvidia_openai_embed

    args = _parse_nvidia_args(monkeypatch, EMBEDDING_BINDING_API_KEY="k")
    func = _build_embedding_func(args)
    _FakeAsyncOpenAI.dim = nvidia_openai_embed.embedding_dim

    await func(["hello"])

    assert func.embedding_dim == nvidia_openai_embed.embedding_dim
    # The signature default of the underlying function is what reaches the API.
    default_model = (
        inspect.signature(nvidia_openai_embed.func).parameters["model"].default
    )
    assert fake_openai[0].calls[0]["model"] == default_model


@pytest.mark.asyncio
async def test_symmetric_mode_always_sends_passage(monkeypatch, fake_openai):
    args = _parse_nvidia_args(
        monkeypatch,
        EMBEDDING_BINDING_API_KEY="k",
        EMBEDDING_MODEL=NVIDIA_TEST_MODEL,
        EMBEDDING_DIM="4",
    )
    func = _build_embedding_func(args)

    assert func.supports_asymmetric is False
    await func(["what is rag?"], context="query")

    assert fake_openai[0].calls[0]["extra_body"]["input_type"] == "passage"


@pytest.mark.asyncio
async def test_asymmetric_mode_maps_query_and_document_contexts(
    monkeypatch, fake_openai
):
    args = _parse_nvidia_args(
        monkeypatch,
        EMBEDDING_BINDING_API_KEY="k",
        EMBEDDING_MODEL=NVIDIA_TEST_MODEL,
        EMBEDDING_DIM="4",
        EMBEDDING_ASYMMETRIC="true",
    )
    func = _build_embedding_func(args)

    assert func.supports_asymmetric is True
    await func(["what is rag?"], context="query")
    await func(["RAG combines retrieval and generation."], context="document")

    input_types = [c.calls[0]["extra_body"]["input_type"] for c in fake_openai]
    assert input_types == ["query", "passage"]


def test_custom_model_requires_explicit_dim():
    from lightrag.api.lightrag_server import create_optimized_embedding_function

    args = SimpleNamespace(
        embedding_dim=None,
        embedding_token_limit=None,
        embedding_asymmetric=False,
        embedding_asymmetric_configured=False,
        embedding_query_prefix_configured=False,
        embedding_document_prefix_configured=False,
    )
    with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*nvidia"):
        create_optimized_embedding_function(
            config_cache=MagicMock(),
            binding="nvidia",
            model=NVIDIA_TEST_MODEL,
            host=NVIDIA_DEFAULT_HOST,
            api_key="k",
            args=args,
        )
