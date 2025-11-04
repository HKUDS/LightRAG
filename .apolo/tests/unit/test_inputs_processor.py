from __future__ import annotations

from typing import Any

import pytest

from apolo_app_types.app_types import AppType
from apolo_app_types.protocols.common import IngressHttp, Preset
from apolo_app_types.protocols.postgres import CrunchyPostgresUserCredentials

from apolo_apps_lightrag.inputs_processor import LightRAGInputsProcessor
from apolo_apps_lightrag.types import (
    LightRAGAppInputs,
    LightRAGPersistence,
    OpenAIAPICloudProvider,
    OpenAIEmbeddingProvider,
)


@pytest.mark.asyncio
async def test_gen_extra_values_merges_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    client_stub = object()
    processor = LightRAGInputsProcessor(client_stub)  # type: ignore[arg-type]

    captured: dict[str, Any] = {}

    async def fake_gen_extra_values(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"platform": {"ingress": True}}

    monkeypatch.setattr(
        "apolo_apps_lightrag.inputs_processor.gen_extra_values",
        fake_gen_extra_values,
    )

    app_inputs = LightRAGAppInputs(
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
        persistence=LightRAGPersistence(
            rag_storage_size=20,
            inputs_storage_size=15,
        ),
        llm_config=OpenAIAPICloudProvider(
            host="api.openai.com",
            model="gpt-4.1",
            api_key="llm-key",
        ),
        embedding_config=OpenAIEmbeddingProvider(
            host="api.openai.com",
            model="text-embedding-3-large",
            api_key="embed-key",
            dimensions=3072,
        ),
    )

    values = await processor.gen_extra_values(
        app_inputs,
        app_name="lightrag-app",
        namespace="apps",
        app_id="instance-123",
        app_secrets_name="lightrag-secrets",
    )

    assert values["replicaCount"] == 1
    assert values["fullnameOverride"] == "lightrag-app"
    assert values["env"]["POSTGRES_HOST"] == "pgbouncer.internal"
    assert values["env"]["LLM_BINDING"] == "openai"
    assert values["persistence"]["ragStorage"]["size"] == "20Gi"
    assert values["persistence"]["inputs"]["size"] == "15Gi"
    assert values["platform"] == {"ingress": True}

    assert captured["apolo_client"] is client_stub
    assert captured["preset_type"] == app_inputs.preset
    assert captured["ingress_http"] == app_inputs.ingress_http
    assert captured["ingress_grpc"] is None
    assert captured["namespace"] == "apps"
    assert captured["app_id"] == "instance-123"
    assert captured["app_type"] == AppType.LightRAG
