import logging
import typing as t

from apolo_app_types.clients.kube import get_service_host_port
from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types.outputs.common import (
    INSTANCE_LABEL,
    get_internal_external_web_urls,
)
from apolo_app_types.outputs.utils.ingress import get_ingress_host_port
from apolo_app_types.protocols.common.networking import HttpApi, ServiceAPI, WebApp

from .types import LightRAGAppOutputs


logger = logging.getLogger(__name__)


async def _generate_lightrag_outputs(
    helm_values: dict[str, t.Any],
    app_instance_id: str,
) -> LightRAGAppOutputs:
    labels = {"app.kubernetes.io/name": "lightrag", INSTANCE_LABEL: app_instance_id}
    internal_web_app_url, external_web_app_url = await get_internal_external_web_urls(
        labels
    )
    internal_host, internal_port = await get_service_host_port(match_labels=labels)
    internal_server_url = None
    if internal_host:
        internal_server_url = HttpApi(
            host=internal_host,
            port=int(internal_port),
            protocol="http",
        )
    external_server_url = None
    ingress_host_port = await get_ingress_host_port(match_labels=labels)
    if ingress_host_port:
        external_server_url = HttpApi(
            host=ingress_host_port[0],
            port=int(ingress_host_port[1]),
            protocol="https",
        )
    return LightRAGAppOutputs(
        app_url=ServiceAPI[WebApp](
            internal_url=internal_web_app_url,
            external_url=external_web_app_url,
        ),
        server_url=ServiceAPI[HttpApi](
            internal_url=internal_server_url,
            external_url=external_server_url,
        ),
    )


class LightRAGOutputsProcessor(BaseAppOutputsProcessor[LightRAGAppOutputs]):
    async def _generate_outputs(
        self,
        helm_values: dict[str, t.Any],
        app_instance_id: str,
    ) -> LightRAGAppOutputs:
        outputs = await _generate_lightrag_outputs(helm_values, app_instance_id)
        logger.info("Got outputs: %s", outputs)
        return outputs


__all__ = ["LightRAGOutputsProcessor"]
