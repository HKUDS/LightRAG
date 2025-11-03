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


class OpenAICompatChatProvider(OpenAICompatChatAPI):
    """LightRAG-specific OpenAI compatible chat configuration."""

    model: str | None = Field(
        default="gpt-4.1",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description=(
                "Model identifier understood by OpenAI-compatible providers "
                "(for example, OpenRouter). Leave empty when using a Hugging "
                "Face model via the fields below."
            ),
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


class OpenAILLMProvider(RestAPI):
    """OpenAI LLM provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="OpenAI LLM Provider",
            description="OpenAI chat completion API configuration.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        default="api.openai.com",
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="OpenAI API host",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="Set the port.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: int | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Set the connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1"
    provider: Literal["openai"] = "openai"
    model: str = Field(
        default="gpt-4.1",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Chat completion model name.",
        ).as_json_schema_extra(),
    )
    api_key: str = Field(
        default="",
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="OpenAI API key.",
        ).as_json_schema_extra(),
    )


class AnthropicLLMProvider(RestAPI):
    """Anthropic LLM provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="Anthropic LLM Provider",
            description="Anthropic Claude API configuration.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        default="api.anthropic.com",
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="Anthropic API host",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="Set the port.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: int | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Set the connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1"
    provider: Literal["anthropic"] = "anthropic"
    model: str = Field(
        default="claude-3-5-sonnet-latest",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Anthropic Claude model name.",
        ).as_json_schema_extra(),
    )
    api_key: str = Field(
        default="",
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="Anthropic API key.",
        ).as_json_schema_extra(),
    )


class OllamaLLMProvider(RestAPI):
    """Ollama LLM provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="Ollama LLM Provider",
            description="Configuration for a self-hosted Ollama server.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="Ollama server host.",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=11434,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="Ollama server port.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["http", "https"] = Field(
        default="http",
        json_schema_extra=SchemaExtraMetadata(
            title="Protocol",
            description="Ollama server protocol.",
        ).as_json_schema_extra(),
    )
    timeout: int | None = Field(
        default=300,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Configure connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/api"
    provider: Literal["ollama"] = "ollama"
    model: str = Field(
        default="llama3.1:8b-instruct-q4_0",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Ollama model name.",
        ).as_json_schema_extra(),
    )


class GeminiLLMProvider(RestAPI):
    """Google Gemini LLM provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="Gemini LLM Provider",
            description="Google Gemini API configuration.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        default="generativelanguage.googleapis.com",
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="Google AI API host",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="Set the port.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: int | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Configure connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1beta"
    provider: Literal["gemini"] = "gemini"
    model: str = Field(
        default="gemini-1.5-pro",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Google Gemini model name.",
        ).as_json_schema_extra(),
    )
    api_key: str = Field(
        default="",
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="Google AI API key.",
        ).as_json_schema_extra(),
    )


LLMProvider = (
    OpenAICompatChatAPI
    | OpenAICompatChatProvider
    | OpenAILLMProvider
    | AnthropicLLMProvider
    | OllamaLLMProvider
    | GeminiLLMProvider
)


class OpenAICompatEmbeddingsProvider(OpenAICompatEmbeddingsAPI):
    """LightRAG-specific OpenAI compatible embeddings configuration."""

    model: str | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description=(
                "Embedding model identifier understood by OpenAI-compatible providers "
                "(for example, OpenRouter or self-hosted vLLM). "
                "Leave empty when using a Hugging Face model via the fields below."
            ),
        ).as_json_schema_extra(),
    )
    dimensions: int = Field(
        default=1536,
        json_schema_extra=SchemaExtraMetadata(
            title="Embedding Dimensions",
            description="Embedding vector dimensionality.",
        ).as_json_schema_extra(),
    )


class OpenAIEmbeddingProvider(RestAPI):
    """OpenAI embedding provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="OpenAI Embedding Provider",
            description="OpenAI embeddings API configuration.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        default="api.openai.com",
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="OpenAI API host",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=443,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="Set the port.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["https"] = "https"
    timeout: int | None = Field(
        default=60,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Set the connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/v1"
    provider: Literal["openai"] = "openai"
    model: str = Field(
        default="text-embedding-3-large",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Embedding model name.",
        ).as_json_schema_extra(),
    )
    api_key: str = Field(
        default="",
        json_schema_extra=SchemaExtraMetadata(
            title="API Key",
            description="OpenAI API key.",
        ).as_json_schema_extra(),
    )
    dimensions: int = Field(
        default=3072,
        json_schema_extra=SchemaExtraMetadata(
            title="Embedding Dimensions",
            description="Embedding vector dimensionality.",
        ).as_json_schema_extra(),
    )


class OllamaEmbeddingProvider(RestAPI):
    """Ollama embedding provider configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="Ollama Embedding Provider",
            description="Ollama local embedding model configuration.",
            meta_type=SchemaMetaType.INLINE,
        ).as_json_schema_extra(),
    )

    host: str = Field(
        json_schema_extra=SchemaExtraMetadata(
            title="Host",
            description="Ollama server host.",
        ).as_json_schema_extra(),
    )
    port: int = Field(
        default=11434,
        json_schema_extra=SchemaExtraMetadata(
            title="Port",
            description="Ollama server port.",
        ).as_json_schema_extra(),
    )
    protocol: Literal["http", "https"] = Field(
        default="http",
        json_schema_extra=SchemaExtraMetadata(
            title="Protocol",
            description="Ollama server protocol.",
        ).as_json_schema_extra(),
    )
    timeout: int | None = Field(
        default=300,
        json_schema_extra=SchemaExtraMetadata(
            title="Timeout",
            description="Configure connection timeout in seconds.",
        ).as_json_schema_extra(),
    )
    base_path: str = "/api"
    provider: Literal["ollama"] = "ollama"
    model: str = Field(
        default="nomic-embed-text",
        json_schema_extra=SchemaExtraMetadata(
            title="Model",
            description="Ollama embedding model name.",
        ).as_json_schema_extra(),
    )


EmbeddingProvider = (
    OpenAICompatEmbeddingsAPI
    | OpenAICompatEmbeddingsProvider
    | OpenAIEmbeddingProvider
    | OllamaEmbeddingProvider
)


LightRAGLLMConfig = LLMProvider
LightRAGEmbeddingConfig = EmbeddingProvider


class LightRAGAppInputs(AppInputs):
    preset: Preset
    ingress_http: IngressHttp
    pgvector_user: CrunchyPostgresUserCredentials
    llm_config: LightRAGLLMConfig = Field(
        default=OpenAICompatChatProvider(host="", port=443, protocol="https"),
        json_schema_extra=SchemaExtraMetadata(
            title="LLM Configuration",
            description="LLM provider configuration.",
        ).as_json_schema_extra(),
    )
    embedding_config: LightRAGEmbeddingConfig = Field(
        default=OpenAICompatEmbeddingsProvider(host="", port=443, protocol="https"),
        json_schema_extra=SchemaExtraMetadata(
            title="Embedding Configuration",
            description="Embedding provider configuration.",
        ).as_json_schema_extra(),
    )
    persistence: LightRAGPersistence = Field(
        default_factory=LightRAGPersistence,
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
    "OpenAICompatChatProvider",
    "OpenAICompatEmbeddingsProvider",
    "OpenAILLMProvider",
    "AnthropicLLMProvider",
    "OllamaLLMProvider",
    "GeminiLLMProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
]
