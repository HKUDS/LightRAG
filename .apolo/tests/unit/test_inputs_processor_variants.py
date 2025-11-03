from __future__ import annotations

import pytest

from apolo_app_types.protocols.common import IngressHttp, Preset
from apolo_app_types.protocols.common.hugging_face import HuggingFaceModel
from apolo_app_types.protocols.common.openai_compat import (
    OpenAICompatChatAPI,
    OpenAICompatEmbeddingsAPI,
)
from apolo_app_types.protocols.postgres import CrunchyPostgresUserCredentials

from apolo_apps_lightrag.inputs_processor import LightRAGInputsProcessor
from apolo_apps_lightrag.types import (
    AnthropicLLMProvider,
    GeminiLLMProvider,
    LightRAGAppInputs,
    LightRAGPersistence,
    OllamaEmbeddingProvider,
    OllamaLLMProvider,
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
)


def _make_base_inputs(
    llm_config: object,
    embedding_config: object,
) -> LightRAGAppInputs:
    return LightRAGAppInputs(
        preset=Preset(name="medium"),
        ingress_http=IngressHttp(),
        pgvector_user=CrunchyPostgresUserCredentials(
            user="rag",
            password="secret",
            host="postgres.internal",
            port=5432,
            pgbouncer_host="pgbouncer.internal",
            pgbouncer_port=6432,
            dbname="lightrag",
        ),
        llm_config=llm_config,
        embedding_config=embedding_config,
        persistence=LightRAGPersistence(
            rag_storage_size=20,
            inputs_storage_size=15,
        ),
    )


async def _generate_env(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: object,
    embedding_config: object,
) -> dict[str, object]:
    async def fake_gen_extra_values(**_: object) -> dict[str, object]:
        return {"platform": {"ingress": True}}

    monkeypatch.setattr(
        "apolo_apps_lightrag.inputs_processor.gen_extra_values",
        fake_gen_extra_values,
    )

    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    values = await processor.gen_extra_values(
        _make_base_inputs(llm_config, embedding_config),
        app_name="lightrag-app",
        namespace="apps",
        app_id="instance-123",
        app_secrets_name="lightrag-secrets",
    )
    return values["env"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "llm_config",
        "expected_binding",
        "expected_model",
        "expected_host",
        "expected_api_key",
    ),
    [
        (
            OpenAILLMProvider(model="gpt-4o", api_key="llm-key"),
            "openai",
            "gpt-4o",
            "https://api.openai.com:443/v1",
            "llm-key",
        ),
        (
            AnthropicLLMProvider(model="claude-3", api_key="anthropic-key"),
            "anthropic",
            "claude-3",
            "https://api.anthropic.com:443/v1",
            "anthropic-key",
        ),
        (
            OllamaLLMProvider(host="ollama.internal", model="llama3"),
            "ollama",
            "llama3",
            "http://ollama.internal:11434/api",
            "",
        ),
        (
            GeminiLLMProvider(model="gemini-pro", api_key="gemini-key"),
            "gemini",
            "gemini-pro",
            "https://generativelanguage.googleapis.com:443/v1",
            "gemini-key",
        ),
    ],
)
async def test_inputs_processor_llm_variants(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: object,
    expected_binding: str,
    expected_model: str,
    expected_host: str,
    expected_api_key: str,
) -> None:
    env = await _generate_env(
        monkeypatch,
        llm_config=llm_config,
        embedding_config=OpenAIEmbeddingProvider(api_key="embed-key"),
    )

    assert env["LLM_BINDING"] == expected_binding
    assert env["LLM_MODEL"] == expected_model
    assert env["LLM_BINDING_HOST"] == expected_host
    assert env["LLM_BINDING_API_KEY"] == expected_api_key
    assert env["OPENAI_API_KEY"] == expected_api_key


@pytest.mark.asyncio
async def test_inputs_processor_openai_compat_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_config = OpenAICompatChatAPI(
        host="compat.example.com",
        port=9443,
        protocol="https",
        hf_model=HuggingFaceModel(model_hf_name="hf/awesome-model"),
    )
    object.__setattr__(llm_config, "api_key", "compat-key")

    env = await _generate_env(
        monkeypatch,
        llm_config=llm_config,
        embedding_config=OpenAIEmbeddingProvider(api_key="embed-key"),
    )

    assert env["LLM_BINDING"] == "openai"
    assert env["LLM_MODEL"] == "hf/awesome-model"
    assert env["LLM_BINDING_HOST"] == llm_config.complete_url
    assert env["LLM_BINDING_API_KEY"] == "compat-key"
    assert env["OPENAI_API_KEY"] == "compat-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding_config", "expected_binding", "expected_host", "expected_dim"),
    [
        (
            OpenAIEmbeddingProvider(
                model="text-embedding-3-small", api_key="embed-key"
            ),
            "openai",
            "https://api.openai.com:443/v1",
            1536,
        ),
        (
            OllamaEmbeddingProvider(host="ollama.internal", model="nomic-embed"),
            "ollama",
            "http://ollama.internal:11434/api",
            1024,
        ),
    ],
)
async def test_inputs_processor_embedding_variants(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: object,
    expected_binding: str,
    expected_host: str,
    expected_dim: int,
) -> None:
    env = await _generate_env(
        monkeypatch,
        llm_config=OpenAILLMProvider(api_key="llm-key"),
        embedding_config=embedding_config,
    )

    assert env["EMBEDDING_BINDING"] == expected_binding
    assert env["EMBEDDING_BINDING_HOST"] == expected_host
    assert env["EMBEDDING_DIM"] == expected_dim


@pytest.mark.asyncio
async def test_inputs_processor_openai_compat_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_config = OpenAICompatEmbeddingsAPI(
        host="embeddings.example.com",
        port=8080,
        protocol="https",
        hf_model=HuggingFaceModel(model_hf_name="text-embedding-awesome"),
    )
    object.__setattr__(embedding_config, "api_key", "embed-key")

    env = await _generate_env(
        monkeypatch,
        llm_config=OpenAILLMProvider(api_key="llm-key"),
        embedding_config=embedding_config,
    )

    assert env["EMBEDDING_BINDING"] == "openai"
    assert env["EMBEDDING_MODEL"] == "text-embedding-awesome"
    assert env["EMBEDDING_BINDING_HOST"] == embedding_config.complete_url
    assert env["EMBEDDING_BINDING_API_KEY"] == "embed-key"
    assert env["EMBEDDING_DIM"] == 1536


def test_inputs_processor_raises_for_missing_hf_model() -> None:
    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    inputs = _make_base_inputs(
        llm_config=OpenAICompatChatAPI(
            host="compat.example.com",
            port=9443,
            protocol="https",
        ),
        embedding_config=OpenAIEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="hf_model"):
        processor._extract_llm_config(inputs.llm_config)


def test_inputs_processor_raises_for_missing_embedding_hf_model() -> None:
    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    inputs = _make_base_inputs(
        llm_config=OpenAILLMProvider(api_key="llm-key"),
        embedding_config=OpenAICompatEmbeddingsAPI(
            host="embeddings.example.com",
            port=8080,
            protocol="https",
        ),
    )

    with pytest.raises(ValueError, match="hf_model"):
        processor._extract_embedding_config(inputs.embedding_config)
