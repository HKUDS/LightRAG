import logging
import typing as t
from urllib.parse import urlparse

from apolo_app_types.app_types import AppType
from apolo_app_types.helm.apps.base import BaseChartValueProcessor
from apolo_app_types.helm.apps.common import gen_extra_values
from apolo_app_types.helm.utils.deep_merging import merge_list_of_dicts
from apolo_app_types.protocols.common.networking import RestAPI
from apolo_app_types.protocols.common.openai_compat import (
    OpenAICompatChatAPI,
    OpenAICompatEmbeddingsAPI,
)
from apolo_app_types.protocols.common.secrets_ import serialize_optional_secret

from .types import (
    LightRAGAppInputs,
    OpenAIAPICloudProvider,
    OpenAICompatEmbeddingsProvider,
    OpenAICompatibleAPI,
    OpenAIEmbeddingCloudProvider,
)


logger = logging.getLogger(__name__)


def _normalise_complete_url(api: RestAPI) -> str:
    """Return a best-effort fully qualified URL for a RestAPI definition."""
    raw_host = getattr(api, "host", "")
    protocol = getattr(api, "protocol", "https")
    port = getattr(api, "port", 443)
    base_path = getattr(api, "base_path", "/")

    if raw_host.startswith(("http://", "https://")):
        parsed = urlparse(raw_host)
        host = parsed.hostname or ""
        if parsed.scheme:
            protocol = parsed.scheme
        if parsed.port:
            port = parsed.port
        if parsed.path:
            base_path = parsed.path
    else:
        host = raw_host

    if not base_path.startswith("/"):
        base_path = f"/{base_path}"

    if not host:
        # Nothing better to do; fall back to raw host string.
        host = raw_host

    return f"{protocol}://{host}:{port}{base_path}"


class LightRAGInputsProcessor(BaseChartValueProcessor[LightRAGAppInputs]):
    def _extract_llm_config(self, llm_config: t.Any) -> dict[str, t.Any]:
        """Extract LLM configuration from provider-specific config."""
        if isinstance(llm_config, OpenAICompatibleAPI) or isinstance(
            llm_config, OpenAICompatChatAPI
        ):
            if llm_config.hf_model is None:
                msg = (
                    "OpenAI-compatible LLM configuration requires a Hugging Face model"
                )
                raise ValueError(msg)
            model = llm_config.hf_model.model_hf_name
            host = _normalise_complete_url(llm_config)
            return {
                "binding": "openai",
                "model": model,
                "host": host,
                "api_key": getattr(llm_config, "api_key", None),
            }
        if isinstance(llm_config, OpenAIAPICloudProvider):
            host = _normalise_complete_url(llm_config)
            return {
                "binding": "openai",
                "model": llm_config.model,
                "host": host,
                "api_key": llm_config.api_key,
            }
        msg = f"Unsupported LLM configuration type: {type(llm_config)!r}"
        raise ValueError(msg)

    def _extract_embedding_config(self, embedding_config: t.Any) -> dict[str, t.Any]:
        """Extract embedding configuration from provider-specific config."""
        if isinstance(
            embedding_config,
            (OpenAICompatEmbeddingsProvider, OpenAICompatEmbeddingsAPI),
        ):
            if embedding_config.hf_model is None:
                msg = "OpenAI-compatible embedding configuration requires a Hugging Face model"
                raise ValueError(msg)
            model = embedding_config.hf_model.model_hf_name
            host = _normalise_complete_url(embedding_config)
            dimensions = getattr(embedding_config, "dimensions", None)
            if dimensions is None:
                msg = "Embedding configuration must specify dimensions"
                raise ValueError(msg)
            return {
                "binding": "openai",
                "model": model,
                "api_key": getattr(embedding_config, "api_key", None),
                "dimensions": dimensions,
                "host": host,
            }
        if isinstance(embedding_config, OpenAIEmbeddingCloudProvider):
            host = _normalise_complete_url(embedding_config)
            dimensions = embedding_config.dimensions
            return {
                "binding": "openai",
                "model": embedding_config.model,
                "api_key": embedding_config.api_key,
                "dimensions": dimensions,
                "host": host,
            }
        msg = f"Unsupported embedding configuration type: {type(embedding_config)!r}"
        raise ValueError(msg)

    async def _get_environment_values(
        self,
        input_: LightRAGAppInputs,
        app_secrets_name: str,
    ) -> dict[str, t.Any]:
        llm_config = self._extract_llm_config(input_.llm_config)
        embedding_config = self._extract_embedding_config(input_.embedding_config)
        env_config = {
            "HOST": "0.0.0.0",
            "PORT": 9621,
            "WEBUI_TITLE": "Graph RAG Engine",
            "WEBUI_DESCRIPTION": "Simple and Fast Graph Based RAG System",
            "LLM_BINDING": llm_config["binding"],
            "LLM_MODEL": llm_config["model"],
            "LLM_BINDING_HOST": llm_config["host"],
            "LLM_BINDING_API_KEY": serialize_optional_secret(
                llm_config["api_key"], app_secrets_name
            ),
            "OPENAI_API_KEY": serialize_optional_secret(
                llm_config["api_key"], app_secrets_name
            )
            or "",
            "EMBEDDING_BINDING": embedding_config["binding"],
            "EMBEDDING_MODEL": embedding_config["model"],
            "EMBEDDING_DIM": embedding_config["dimensions"],
            "EMBEDDING_BINDING_HOST": embedding_config["host"],
            "EMBEDDING_BINDING_API_KEY": serialize_optional_secret(
                embedding_config["api_key"], app_secrets_name
            )
            or "",
            "LIGHTRAG_KV_STORAGE": "PGKVStorage",
            "LIGHTRAG_VECTOR_STORAGE": "PGVectorStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE": "PGDocStatusStorage",
            "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
            "POSTGRES_HOST": input_.pgvector_user.pgbouncer_host,
            "POSTGRES_PORT": input_.pgvector_user.pgbouncer_port,
            "POSTGRES_USER": input_.pgvector_user.user,
            "POSTGRES_PASSWORD": input_.pgvector_user.password,
            "POSTGRES_DATABASE": input_.pgvector_user.dbname,
            "POSTGRES_WORKSPACE": "default",
        }

        return {"env": env_config}

    async def _get_persistence_values(
        self,
        input_: LightRAGAppInputs,
    ) -> dict[str, t.Any]:
        return {
            "persistence": {
                "enabled": True,
                "ragStorage": {
                    "size": f"{input_.persistence.rag_storage_size}Gi",
                },
                "inputs": {
                    "size": f"{input_.persistence.inputs_storage_size}Gi",
                },
            }
        }

    async def gen_extra_values(
        self,
        input_: LightRAGAppInputs,
        app_name: str,
        namespace: str,
        app_id: str,
        app_secrets_name: str,
        *_: t.Any,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        env_values = await self._get_environment_values(input_, app_secrets_name)
        persistence_values = await self._get_persistence_values(input_)
        platform_values = await gen_extra_values(
            apolo_client=self.client,
            preset_type=input_.preset,
            ingress_http=input_.ingress_http,
            ingress_grpc=None,
            namespace=namespace,
            app_id=app_id,
            app_type=AppType.LightRAG,
        )
        base_values = {
            "replicaCount": 1,
            "image": {
                "repository": "ghcr.io/hkuds/lightrag",
                "tag": "1.3.8",
                "pullPolicy": "IfNotPresent",
            },
            "service": {
                "type": "ClusterIP",
                "port": 9621,
            },
            "nameOverride": "",
            "fullnameOverride": app_name,
        }
        logger.debug("Generated LightRAG values for app %s", app_name)
        return merge_list_of_dicts(
            [
                base_values,
                env_values,
                persistence_values,
                platform_values,
            ]
        )


__all__ = ["LightRAGInputsProcessor"]
