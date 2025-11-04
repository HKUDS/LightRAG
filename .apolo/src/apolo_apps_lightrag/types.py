from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apolo_app_types import AppInputs, AppOutputs
from apolo_app_types.protocols.common import (
    IngressHttp,
    Preset,
    SchemaExtraMetadata,
    SchemaMetaType,
)
from apolo_app_types.protocols.common.networking import HttpApi, RestAPI, ServiceAPI
from apolo_app_types.protocols.common.openai_compat import (
    OpenAICompatChatAPI,
    OpenAICompatEmbeddingsAPI,
)
from apolo_app_types.protocols.postgres import CrunchyPostgresUserCredentials


class OpenAICompatibleAPI(OpenAICompatChatAPI):
    """OpenAI-compatible chat configuration backed by Hugging Face models."""

    base_path: str = "/v1"

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="OpenAI Compatible API",
            description=(
                "Use for self-hosted services (for example vLLM) that expose an "
                "OpenAI-compatible API and are configured via a Hugging Face model."
            ),
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    api_key: str | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="Optional API key used to access the compatible endpoint.",
        ).as_json_schema_extra(),
    )


class LightRAGPersistence(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="LightRAG Persistence",
            description="Configure persistent storage for LightRAG data and inputs.",
        ).as_json_schema_extra(),
    )

    rag_storage_size: int = Field(
        default=10,
        gt=0,
        json_schema_extra=SchemaExtraMetadata(
            title="RAG Storage Size (GB)",
            description="Size of the persistent volume for RAG data storage.",
        ).as_json_schema_extra(),
    )
    inputs_storage_size: int = Field(
        default=5,
        gt=0,
        json_schema_extra=SchemaExtraMetadata(
            title="Inputs Storage Size (GB)",
            description="Size of the persistent volume for input files.",
        ).as_json_schema_extra(),
    )

    @field_validator("rag_storage_size", "inputs_storage_size", mode="before")
    @classmethod
    def validate_storage_size(cls, value: int) -> int:
        if value and isinstance(value, int) and value < 1:
            error_message = "Storage size must be greater than 1GB."
            raise ValueError(error_message)
        return value


class OpenAIAPICloudProvider(RestAPI):
    """Hosted OpenAI-compatible provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="OpenAI API Cloud Provider",
            description="Use for hosted OpenAI-compatible APIs such as OpenAI or OpenRouter.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="HTTPS port for the provider endpoint.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: int | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1"
    model: str = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Model identifier exposed by the provider (for example `gpt-4o`).",
        ).as_json_schema_extra(),
    )
    api_key: str = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="API key used to authenticate with the provider.",
        ).as_json_schema_extra(),
    )


class OpenAICompatEmbeddingsProvider(OpenAICompatEmbeddingsAPI):
    """OpenAI-compatible embeddings configuration for hosted providers."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="OpenAI Compatible API Embeddings",
            description="Use for OpenAI-compatible embeddings APIs (Self-hosted services).",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )
    host: str = Field(
        default="api.openai.com",
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="Hostname of the OpenAI-compatible embeddings endpoint.",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="HTTPS port for the embeddings endpoint.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: float | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1"
    dimensions: int = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Embedding Dimensions",
            description="Embedding vector dimensionality reported by the provider.",
        ).as_json_schema_extra(),
    )


class OpenAIEmbeddingCloudProvider(RestAPI):
    """Official OpenAI embeddings configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="OpenAI Embeddings Cloud Provider",
            description="Use the official OpenAI embeddings endpoint with recommended defaults.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="Hostname for api.openai.com (omit protocol).",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="HTTPS port for the endpoint.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: int | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1"
    provider: Literal["openai"] = "openai"
    model: str = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="OpenAI embedding model identifier.",
        ).as_json_schema_extra(),
    )
    api_key: str = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="OpenAI API key.",
        ).as_json_schema_extra(),
    )
    dimensions: int = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Embedding Dimensions",
            description="Embedding vector dimensionality returned by the model.",
        ).as_json_schema_extra(),
    )


LLMProvider = OpenAICompatibleAPI | OpenAIAPICloudProvider

EmbeddingProvider = OpenAICompatEmbeddingsProvider | OpenAIEmbeddingCloudProvider


LightRAGLLMConfig = LLMProvider
LightRAGEmbeddingConfig = EmbeddingProvider


class LightRAGAppInputs(AppInputs):
    preset: Preset
    ingress_http: IngressHttp
    pgvector_user: CrunchyPostgresUserCredentials
    llm_config: LightRAGLLMConfig = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="LLM Configuration",
            description="LLM provider configuration.",
        ).as_json_schema_extra(),
    )
    embedding_config: LightRAGEmbeddingConfig = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Embedding Configuration",
            description="Embedding provider configuration.",
        ).as_json_schema_extra(),
    )
    persistence: LightRAGPersistence = Field(
        ...,
        json_schema_extra=SchemaExtraMetadata(
            title="Persistence Configuration",
            description="Configure persistent storage for LightRAG data and inputs.",
        ).as_json_schema_extra(),
    )


class LightRAGAppOutputs(AppOutputs):
    """LightRAG outputs."""

    server_url: ServiceAPI[HttpApi] | None = None


__all__ = [
    "LightRAGAppInputs",
    "LightRAGAppOutputs",
    "LightRAGEmbeddingConfig",
    "LightRAGLLMConfig",
    "LightRAGPersistence",
    "OpenAICompatibleAPI",
    "OpenAIAPICloudProvider",
    "OpenAICompatEmbeddingsProvider",
    "OpenAIEmbeddingCloudProvider",
]
