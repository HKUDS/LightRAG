from __future__ import annotations

import pytest

from apolo_app_types.outputs.common import INSTANCE_LABEL
from apolo_app_types.protocols.common.networking import WebApp

from apolo_apps_lightrag.outputs_processor import LightRAGOutputsProcessor


@pytest.mark.asyncio
async def test_generate_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = LightRAGOutputsProcessor()
    internal_web = WebApp(
        host="internal.local",
        port=9621,
        base_path="/",
        protocol="http",
    )
    external_web = WebApp(
        host="external.example.com",
        port=443,
        base_path="/",
        protocol="https",
    )

    async def fake_get_internal_external_web_urls(labels: dict[str, str]):
        assert labels == {
            "app.kubernetes.io/name": "lightrag",
            INSTANCE_LABEL: "instance-123",
        }
        return internal_web, external_web

    async def fake_get_service_host_port(match_labels: dict[str, str]):
        assert match_labels == {
            "app.kubernetes.io/name": "lightrag",
            INSTANCE_LABEL: "instance-123",
        }
        return "service.local", 9621

    async def fake_get_ingress_host_port(match_labels: dict[str, str]):
        assert match_labels == {
            "app.kubernetes.io/name": "lightrag",
            INSTANCE_LABEL: "instance-123",
        }
        return "service.example.com", 443

    monkeypatch.setattr(
        "apolo_apps_lightrag.outputs_processor.get_internal_external_web_urls",
        fake_get_internal_external_web_urls,
    )
    monkeypatch.setattr(
        "apolo_apps_lightrag.outputs_processor.get_service_host_port",
        fake_get_service_host_port,
    )
    monkeypatch.setattr(
        "apolo_apps_lightrag.outputs_processor.get_ingress_host_port",
        fake_get_ingress_host_port,
    )

    outputs = await processor.generate_outputs({}, "instance-123")

    assert outputs["app_url"]["internal_url"]["host"] == internal_web.host
    assert outputs["app_url"]["external_url"]["host"] == external_web.host
    assert outputs["server_url"]["internal_url"]["host"] == "service.local"
    assert outputs["server_url"]["internal_url"]["port"] == 9621
    assert outputs["server_url"]["external_url"]["host"] == "service.example.com"
    assert outputs["server_url"]["external_url"]["port"] == 443
