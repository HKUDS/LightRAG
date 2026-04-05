"""
Gemini Batch API provider for LightRAG.

Implements BatchProvider using Google's Gemini Batch Prediction API
(client.aio.batches) for submitting bulk completion requests.
"""

from typing import Any

import pipmaster as pm

if not pm.is_installed("google-genai"):
    pm.install("google-genai")
if not pm.is_installed("google-api-core"):
    pm.install("google-api-core")

from google.genai import types  # noqa: E402

from lightrag.utils import logger  # noqa: E402

from .batch_provider import (  # noqa: E402
    BatchJobState,
    BatchJobStatus,
    BatchProvider,
    BatchRequest,
    BatchResponse,
)
from .gemini import (  # noqa: E402
    _build_generation_config,
    _ensure_api_key,
    _extract_response_text,
    _format_history_messages,
    _get_gemini_client,
)

# Mapping from Gemini JobState strings to our BatchJobState enum
_GEMINI_STATE_MAP: dict[str, BatchJobState] = {
    "JOB_STATE_PENDING": BatchJobState.PENDING,
    "JOB_STATE_QUEUED": BatchJobState.PENDING,
    "JOB_STATE_RUNNING": BatchJobState.RUNNING,
    "JOB_STATE_UPDATING": BatchJobState.RUNNING,
    "JOB_STATE_PAUSED": BatchJobState.RUNNING,
    "JOB_STATE_SUCCEEDED": BatchJobState.SUCCEEDED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": BatchJobState.SUCCEEDED,
    "JOB_STATE_FAILED": BatchJobState.FAILED,
    "JOB_STATE_EXPIRED": BatchJobState.FAILED,
    "JOB_STATE_CANCELLED": BatchJobState.CANCELLED,
    "JOB_STATE_CANCELLING": BatchJobState.RUNNING,
    "JOB_STATE_UNSPECIFIED": BatchJobState.PENDING,
}


class GeminiBatchProvider(BatchProvider):
    """BatchProvider implementation using Google Gemini Batch Prediction API.

    Uses ``client.aio.batches.create()`` with inlined requests for submission
    and ``client.aio.batches.get()`` for polling. Results are returned inline
    in the completed job's ``dest.inlined_responses``.

    Args:
        api_key: Gemini API key. Falls back to environment variables.
        base_url: Optional custom API endpoint.
        timeout: Optional request timeout in seconds for the initial submit call.
        generation_config: Optional base generation config dict (temperature, etc.).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        generation_config: dict[str, Any] | None = None,
    ):
        self._api_key = _ensure_api_key(api_key)
        self._base_url = base_url
        self._timeout = timeout
        self._generation_config = generation_config
        # Convert timeout from seconds to milliseconds for Gemini API
        timeout_ms = timeout * 1000 if timeout else None
        self._client = _get_gemini_client(self._api_key, base_url, timeout_ms)

    def _build_inlined_request(
        self, request: BatchRequest, model: str
    ) -> types.InlinedRequest:
        """Convert a BatchRequest into a Gemini InlinedRequest."""
        # Format history into prompt sections (same as gemini_complete_if_cache)
        history_block = _format_history_messages(request.history_messages)
        prompt_sections = []
        if history_block:
            prompt_sections.append(history_block)
        prompt_sections.append(f"[user] {request.prompt}")
        combined_prompt = "\n".join(prompt_sections)

        config_obj = _build_generation_config(
            self._generation_config,
            system_prompt=request.system_prompt,
            keyword_extraction=False,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "contents": combined_prompt,
            "metadata": {"key": request.key},
        }
        if config_obj is not None:
            kwargs["config"] = config_obj

        return types.InlinedRequest(**kwargs)

    async def submit_completion_batch(
        self, requests: list[BatchRequest], model: str, **kwargs
    ) -> str:
        """Submit a batch of completion requests to Gemini.

        Args:
            requests: List of BatchRequest objects.
            model: Gemini model name (e.g., ``gemini-2.0-flash``).

        Returns:
            Job name string (used as job_id for polling).
        """
        inlined = [self._build_inlined_request(r, model) for r in requests]

        logger.info(
            f"Submitting Gemini batch with {len(inlined)} requests using model {model}"
        )

        job = await self._client.aio.batches.create(
            model=model,
            src=inlined,
        )

        logger.info(f"Gemini batch job created: {job.name}")
        return job.name

    async def get_job_status(self, job_id: str) -> BatchJobStatus:
        """Poll the status of a Gemini batch job."""
        job = await self._client.aio.batches.get(name=job_id)

        state_str = str(job.state) if job.state else "JOB_STATE_UNSPECIFIED"
        state = _GEMINI_STATE_MAP.get(state_str, BatchJobState.PENDING)

        total = 0
        succeeded = 0
        failed = 0
        if job.completion_stats:
            succeeded = job.completion_stats.successful_count or 0
            failed = job.completion_stats.failed_count or 0
            total = succeeded + failed + (job.completion_stats.incomplete_count or 0)

        # Fall back to counting source requests if stats aren't populated yet
        if total == 0 and job.src and job.src.inlined_requests:
            total = len(job.src.inlined_requests)

        return BatchJobStatus(
            job_id=job_id,
            state=state,
            total=total,
            succeeded=succeeded,
            failed=failed,
        )

    async def get_results(self, job_id: str) -> list[BatchResponse]:
        """Retrieve results for a completed Gemini batch job.

        Results are returned from ``job.dest.inlined_responses``, which is
        ordered parallel to the input ``inlined_requests``. Each response's
        key is recovered from the corresponding request's metadata.
        """
        job = await self._client.aio.batches.get(name=job_id)

        responses: list[BatchResponse] = []

        inlined_responses = (
            job.dest.inlined_responses
            if job.dest and job.dest.inlined_responses
            else []
        )
        inlined_requests = (
            job.src.inlined_requests if job.src and job.src.inlined_requests else []
        )

        for i, inlined_resp in enumerate(inlined_responses):
            # Recover the key from the parallel request's metadata
            key = ""
            if i < len(inlined_requests) and inlined_requests[i].metadata:
                key = inlined_requests[i].metadata.get("key", "")

            if inlined_resp.error:
                error_msg = inlined_resp.error.message or "Unknown batch row error"
                responses.append(BatchResponse(key=key, error=error_msg))
                continue

            if inlined_resp.response:
                text, _ = _extract_response_text(inlined_resp.response)
                prompt_tokens = 0
                completion_tokens = 0
                usage = getattr(inlined_resp.response, "usage_metadata", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

                responses.append(
                    BatchResponse(
                        key=key,
                        content=text if text else None,
                        error="Empty response" if not text else None,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
            else:
                responses.append(
                    BatchResponse(key=key, error="No response in batch result")
                )

        return responses

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a Gemini batch job."""
        try:
            await self._client.aio.batches.cancel(name=job_id)
            logger.info(f"Gemini batch job cancelled: {job_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel Gemini batch job {job_id}: {e}")


__all__ = ["GeminiBatchProvider"]
