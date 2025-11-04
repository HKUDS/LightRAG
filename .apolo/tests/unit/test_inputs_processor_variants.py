from __future__ import annotations

import pytest

from apolo_app_types.protocols.common import IngressHttp, Preset
from apolo_app_types.protocols.common.hugging_face import HuggingFaceModel
from apolo_app_types.protocols.common.openai_compat import OpenAICompatChatAPI
from apolo_app_types.protocols.postgres import CrunchyPostgresUserCredentials

from apolo_apps_lightrag.inputs_processor import LightRAGInputsProcessor
from apolo_apps_lightrag.types import (
    LightRAGAppInputs,
    LightRAGPersistence,
    OpenAIAPICloudProvider,
    OpenAICompatEmbeddingsProvider,
    OpenAICompatibleAPI,
    OpenAIEmbeddingCloudProvider,
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


def _default_llm_provider() -> OpenAIAPICloudProvider:
    return OpenAIAPICloudProvider(
        host="api.openai.com",
        model="gpt-4o",
        api_key="llm-key",
    )


def _default_embedding_provider() -> OpenAIEmbeddingCloudProvider:
    return OpenAIEmbeddingCloudProvider(
        host="api.openai.com",
        model="text-embedding-3-large",
        api_key="embed-key",
        dimensions=3072,
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
        _make_base_inputs(
            llm_config,
            embedding_config,
        ),
        app_name="lightrag-app",
        namespace="apps",
        app_id="instance-123",
        app_secrets_name="lightrag-secrets",
    )
    return values["env"]


def test_light_rag_inputs_select_cloud_provider_when_model_present() -> None:
    inputs = _make_base_inputs(
        llm_config={
            "host": "api.openai.com",
            "model": "gpt-4o",
            "api_key": "llm-key",
        },
        embedding_config=_default_embedding_provider(),
    )

    assert isinstance(inputs.llm_config, OpenAIAPICloudProvider)
    assert inputs.llm_config.model == "gpt-4o"


def test_light_rag_inputs_select_compat_provider_when_hf_model_present() -> None:
    inputs = _make_base_inputs(
        llm_config={
            "host": "router.example.com",
            "protocol": "https",
            "port": 9443,
            "hf_model": {"model_hf_name": "hf/awesome-model"},
        },
        embedding_config=_default_embedding_provider(),
    )

    assert isinstance(inputs.llm_config, OpenAICompatibleAPI)
    assert inputs.llm_config.hf_model is not None
    assert inputs.llm_config.hf_model.model_hf_name == "hf/awesome-model"


def test_light_rag_inputs_select_embedding_cloud_when_model_present() -> None:
    inputs = _make_base_inputs(
        llm_config=_default_llm_provider(),
        embedding_config={
            "host": "api.openai.com",
            "model": "text-embedding-3-large",
            "api_key": "embed-key",
            "dimensions": 3072,
        },
    )

    assert isinstance(inputs.embedding_config, OpenAIEmbeddingCloudProvider)
    assert inputs.embedding_config.model == "text-embedding-3-large"


def test_light_rag_inputs_select_embedding_compat_when_hf_model_present() -> None:
    inputs = _make_base_inputs(
        llm_config=_default_llm_provider(),
        embedding_config={
            "host": "router.example.com",
            "protocol": "https",
            "port": 8443,
            "hf_model": {"model_hf_name": "hf/embed-model"},
            "dimensions": 1024,
        },
    )

    assert isinstance(inputs.embedding_config, OpenAICompatEmbeddingsProvider)
    assert inputs.embedding_config.hf_model is not None
    assert inputs.embedding_config.hf_model.model_hf_name == "hf/embed-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "llm_config",
        "expected_model",
        "expected_host",
    ),
    [
        (
            OpenAIAPICloudProvider(
                host="api.openai.com",
                model="gpt-4o",
                api_key="llm-key",
            ),
            "gpt-4o",
            "https://api.openai.com/v1",
        ),
        (
            OpenAICompatibleAPI(
                host="vllm.internal",
                port=9443,
                protocol="https",
                hf_model=HuggingFaceModel(model_hf_name="something/awesome-model"),
            ),
            "something/awesome-model",
            "https://vllm.internal:9443/v1",
        ),
    ],
)
async def test_inputs_processor_llm_variants(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: object,
    expected_model: str,
    expected_host: str,
) -> None:
    env = await _generate_env(
        monkeypatch,
        llm_config=llm_config,
        embedding_config=_default_embedding_provider(),
    )

    assert env["LLM_BINDING"] == "openai"
    assert env["LLM_MODEL"] == expected_model
    assert env["LLM_BINDING_HOST"] == expected_host
    expected_api_key = getattr(llm_config, "api_key", None) or ""
    assert env["LLM_BINDING_API_KEY"] == expected_api_key
    assert env["OPENAI_API_KEY"] == expected_api_key


@pytest.mark.asyncio
async def test_inputs_processor_openai_compat_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_config = OpenAICompatibleAPI(
        host="compat.example.com",
        port=9443,
        protocol="https",
        hf_model=HuggingFaceModel(model_hf_name="something/awesome-model"),
        api_key="compat-key",
    )

    env = await _generate_env(
        monkeypatch,
        llm_config=llm_config,
        embedding_config=_default_embedding_provider(),
    )

    assert env["LLM_BINDING"] == "openai"
    assert env["LLM_MODEL"] == "something/awesome-model"
    assert env["LLM_BINDING_HOST"] == "https://compat.example.com:9443/v1"
    assert env["LLM_BINDING_API_KEY"] == "compat-key"
    assert env["OPENAI_API_KEY"] == "compat-key"


@pytest.mark.asyncio
async def test_inputs_processor_cloud_provider_base_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_config = OpenAIAPICloudProvider(
        host="openrouter.ai",
        base_path="/api/v1",
        model="openrouter/meta-llama-3-70b",
        api_key="router-key",
    )

    env = await _generate_env(
        monkeypatch,
        llm_config=llm_config,
        embedding_config=_default_embedding_provider(),
    )

    assert env["LLM_BINDING"] == "openai"
    assert env["LLM_MODEL"] == "openrouter/meta-llama-3-70b"
    assert env["LLM_BINDING_HOST"] == "https://openrouter.ai/api/v1"
    assert env["LLM_BINDING_API_KEY"] == "router-key"
    assert env["OPENAI_API_KEY"] == "router-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding_config", "expected_binding", "expected_host", "expected_dim"),
    [
        (
            OpenAIEmbeddingCloudProvider(
                host="api.openai.com",
                model="text-embedding-3-small",
                api_key="embed-key",
                dimensions=1536,
            ),
            "openai",
            "https://api.openai.com/v1",
            1536,
        ),
        (
            OpenAICompatEmbeddingsProvider(
                host="router.example.com",
                port=8443,
                protocol="https",
                hf_model=HuggingFaceModel(model_hf_name="openrouter/embedding-model"),
                dimensions=1024,
            ),
            "openai",
            "https://router.example.com:8443/v1",
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
        llm_config=_default_llm_provider(),
        embedding_config=embedding_config,
    )

    assert env["EMBEDDING_BINDING"] == expected_binding
    assert env["EMBEDDING_BINDING_HOST"] == expected_host
    assert env["EMBEDDING_DIM"] == expected_dim


@pytest.mark.asyncio
async def test_inputs_processor_openai_compat_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_config = OpenAICompatEmbeddingsProvider(
        host="embeddings.example.com",
        port=8080,
        protocol="https",
        hf_model=HuggingFaceModel(model_hf_name="text-embedding-awesome"),
        dimensions=1536,
    )
    object.__setattr__(embedding_config, "api_key", "embed-key")

    env = await _generate_env(
        monkeypatch,
        llm_config=_default_llm_provider(),
        embedding_config=embedding_config,
    )

    assert env["EMBEDDING_BINDING"] == "openai"
    assert env["EMBEDDING_MODEL"] == "text-embedding-awesome"
    assert env["EMBEDDING_BINDING_HOST"] == "https://embeddings.example.com:8080/v1"
    assert env["EMBEDDING_BINDING_API_KEY"] == "embed-key"
    assert env["EMBEDDING_DIM"] == 1536


@pytest.mark.asyncio
async def test_inputs_processor_openai_compat_embedding_requires_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_config = OpenAICompatEmbeddingsProvider(
        host="router.example.com",
        port=8443,
        protocol="https",
        dimensions=1536,
    )

    with pytest.raises(
        ValueError,
        match="requires a Hugging Face model",
    ):
        await _generate_env(
            monkeypatch,
            llm_config=_default_llm_provider(),
            embedding_config=embedding_config,
        )


def test_extract_llm_config_requires_model_for_compatible() -> None:
    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    llm_config = OpenAICompatibleAPI(
        host="router.example.com", protocol="https", port=443
    )

    with pytest.raises(ValueError, match="requires a Hugging Face model"):
        processor._extract_llm_config(llm_config)


def test_extract_llm_config_compatible_with_model() -> None:
    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    llm_config = OpenAIAPICloudProvider(
        host="router.example.com",
        port=443,
        protocol="https",
        model="openrouter/meta-llama",
        api_key="router-key",
    )

    config = processor._extract_llm_config(llm_config)

    assert config["model"] == "openrouter/meta-llama"
    assert config["host"] == "https://router.example.com/v1"
    assert config["api_key"] == "router-key"


def test_extract_llm_config_chat_requires_hf() -> None:
    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    llm_config = OpenAICompatChatAPI(
        host="hf.example.com",
        port=8443,
        protocol="https",
    )

    with pytest.raises(ValueError, match="requires a Hugging Face model"):
        processor._extract_llm_config(llm_config)


def test_extract_llm_config_chat_with_hf_model() -> None:
    processor = LightRAGInputsProcessor(client=object())  # type: ignore[arg-type]
    llm_config = OpenAICompatChatAPI(
        host="hf.example.com",
        port=8443,
        protocol="https",
        hf_model=HuggingFaceModel(model_hf_name="hf/awesome-model"),
    )

    config = processor._extract_llm_config(llm_config)

    assert config["model"] == "hf/awesome-model"
    assert config["host"] == "https://hf.example.com:8443/v1"
