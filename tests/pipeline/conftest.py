"""Shared helpers for the pipeline test suite."""

from uuid import uuid4

from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import get_pipeline_ingress


async def request_failed_retry(rag) -> str:
    """Publish a sticky manual retry request for ``rag``'s workspace.

    Since the FAILED-retry semantics split, automatic pipeline runs resume
    only PENDING/interrupted documents — a FAILED document re-enters the
    pipeline exclusively through an explicit human request.  This helper is
    the test-suite equivalent of calling ``/documents/reprocess_failed``:
    publish the sticky request, then drive
    ``apipeline_process_enqueue_documents()`` as before.

    Returns the request id (useful for ACK assertions).
    """
    ingress = await get_pipeline_ingress(rag.workspace)
    request_id = uuid4().hex
    ingress.request_manual_retry(
        request_id,
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id=request_id),
    )
    return request_id
