"""
OpenAI Batch API provider for LightRAG.

Implements BatchProvider using OpenAI's Batch API, which works via
JSONL file upload → batch creation → polling → results file download.
"""

import io
import json
from typing import Any

import pipmaster as pm

if not pm.is_installed("openai"):
    pm.install("openai")

from lightrag.utils import logger  # noqa: E402

from .batch_provider import (  # noqa: E402
    BatchJobState,
    BatchJobStatus,
    BatchProvider,
    BatchRequest,
    BatchResponse,
)
from .openai import create_openai_async_client  # noqa: E402

# Mapping from OpenAI batch status strings to BatchJobState
_OPENAI_STATE_MAP: dict[str, BatchJobState] = {
    "validating": BatchJobState.PENDING,
    "in_progress": BatchJobState.RUNNING,
    "finalizing": BatchJobState.RUNNING,
    "completed": BatchJobState.SUCCEEDED,
    "failed": BatchJobState.FAILED,
    "expired": BatchJobState.FAILED,
    "cancelled": BatchJobState.CANCELLED,
    "cancelling": BatchJobState.RUNNING,
}


class OpenAIBatchProvider(BatchProvider):
    """BatchProvider implementation using OpenAI's Batch API.

    Uses JSONL file upload via the Files API, then ``client.batches.create()``
    for submission, ``client.batches.retrieve()`` for polling, and
    ``client.files.content()`` to download results.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        base_url: Optional custom API endpoint (for OpenAI-compatible providers).
        timeout: Optional request timeout in seconds.
        extra_params: Optional dict of extra parameters to include in each
            request body (e.g., temperature, max_completion_tokens).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        self._client = create_openai_async_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._extra_params = extra_params or {}

    def _build_jsonl_request(self, request: BatchRequest, model: str) -> dict[str, Any]:
        """Convert a BatchRequest into an OpenAI batch JSONL line."""
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        if request.history_messages:
            messages.extend(request.history_messages)
        messages.append({"role": "user", "content": request.prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **self._extra_params,
        }

        return {
            "custom_id": request.key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    async def submit_completion_batch(
        self, requests: list[BatchRequest], model: str, **kwargs
    ) -> str:
        """Submit a batch of completion requests to OpenAI.

        Builds a JSONL file, uploads it via the Files API, then creates
        a batch job.

        Args:
            requests: List of BatchRequest objects.
            model: OpenAI model name (e.g., ``gpt-4o-mini``).

        Returns:
            Batch ID string for tracking.
        """
        # Build JSONL content
        lines = [
            json.dumps(self._build_jsonl_request(r, model), ensure_ascii=False)
            for r in requests
        ]
        jsonl_content = "\n".join(lines)

        logger.info(
            f"Uploading OpenAI batch file with {len(requests)} requests "
            f"using model {model}"
        )
        if lines:
            logger.debug(f"First batch request line: {lines[0][:500]}")

        # Upload JSONL file
        file_obj = io.BytesIO(jsonl_content.encode("utf-8"))
        uploaded = await self._client.files.create(
            file=("batch_requests.jsonl", file_obj, "application/jsonl"),
            purpose="batch",
        )

        logger.info(f"Uploaded batch file: {uploaded.id}")

        # Create batch
        batch = await self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"OpenAI batch job created: {batch.id}")
        return batch.id

    async def get_job_status(self, job_id: str) -> BatchJobStatus:
        """Poll the status of an OpenAI batch job."""
        batch = await self._client.batches.retrieve(job_id)

        state = _OPENAI_STATE_MAP.get(batch.status or "", BatchJobState.PENDING)

        # Log errors and capture error code when batch fails
        error_code = None
        if state == BatchJobState.FAILED:
            errors = getattr(batch, "errors", None)
            if errors and hasattr(errors, "data") and errors.data:
                first_err = errors.data[0]
                error_code = getattr(first_err, "code", None)
                for err in errors.data:
                    logger.error(
                        f"Batch {job_id} error: "
                        f"code={getattr(err, 'code', '?')}, "
                        f"message={getattr(err, 'message', '?')}, "
                        f"line={getattr(err, 'line', '?')}"
                    )
            else:
                logger.error(
                    f"Batch {job_id} failed with status={batch.status}, "
                    f"no error details available"
                )

        total = batch.request_counts.total if batch.request_counts else 0
        succeeded = batch.request_counts.completed if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0

        return BatchJobStatus(
            job_id=job_id,
            state=state,
            total=total,
            succeeded=succeeded,
            failed=failed,
            error_code=error_code,
        )

    async def get_results(self, job_id: str) -> list[BatchResponse]:
        """Retrieve results for a completed OpenAI batch job.

        Downloads the output file and parses each JSONL line. Results
        are keyed by ``custom_id`` which maps back to ``BatchRequest.key``.
        """
        batch = await self._client.batches.retrieve(job_id)

        responses: list[BatchResponse] = []

        if not batch.output_file_id:
            logger.warning(
                f"Batch {job_id} has no output file " f"(status={batch.status})"
            )
            # Check for error file
            if batch.error_file_id:
                error_content = await self._client.files.content(batch.error_file_id)
                for line in error_content.text.strip().split("\n"):
                    if not line:
                        continue
                    row = json.loads(line)
                    responses.append(
                        BatchResponse(
                            key=row.get("custom_id", ""),
                            error=json.dumps(row.get("error", "Unknown error")),
                        )
                    )
            return responses

        # Download and parse results file
        results_content = await self._client.files.content(batch.output_file_id)

        for line in results_content.text.strip().split("\n"):
            if not line:
                continue

            row = json.loads(line)
            custom_id = row.get("custom_id", "")
            result = row.get("response", {})
            status_code = result.get("status_code", 0)
            body = result.get("body", {})

            if status_code == 200 and body:
                # Extract content from choices
                choices = body.get("choices", [])
                content = None
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")

                # Extract usage
                usage = body.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                responses.append(
                    BatchResponse(
                        key=custom_id,
                        content=content if content else None,
                        error="Empty response" if not content else None,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
            else:
                error = row.get("error", {})
                error_msg = (
                    error.get("message", f"HTTP {status_code}")
                    if isinstance(error, dict)
                    else str(error)
                )
                responses.append(BatchResponse(key=custom_id, error=error_msg))

        return responses

    async def cancel_job(self, job_id: str) -> None:
        """Cancel an OpenAI batch job."""
        try:
            await self._client.batches.cancel(job_id)
            logger.info(f"OpenAI batch job cancelled: {job_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel OpenAI batch job {job_id}: {e}")


__all__ = ["OpenAIBatchProvider"]
