"""
Provider-agnostic batch API abstraction for LightRAG.

Defines the BatchProvider interface and supporting data types for submitting
LLM completion requests as batch jobs (e.g., Gemini Batch API, OpenAI Batch API).
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BatchJobState(Enum):
    """Terminal and non-terminal states for a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchRequest:
    """A single completion request within a batch.

    Attributes:
        key: Caller-assigned identifier for result reordering (e.g., chunk_key).
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        history_messages: Optional conversation history in OpenAI message format.
    """

    key: str
    prompt: str
    system_prompt: str | None = None
    history_messages: list[dict[str, Any]] | None = None


@dataclass
class BatchResponse:
    """Result for a single request within a completed batch.

    Attributes:
        key: Matches the BatchRequest.key this response corresponds to.
        content: LLM response text, or None if this row failed.
        error: Error message if this row failed.
        prompt_tokens: Token count for the prompt.
        completion_tokens: Token count for the completion.
    """

    key: str
    content: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class BatchJobStatus:
    """Status of a batch job.

    Attributes:
        job_id: Provider-assigned job identifier.
        state: Current state of the job.
        total: Total number of requests in the batch.
        succeeded: Number of requests that completed successfully.
        failed: Number of requests that failed.
    """

    job_id: str
    state: BatchJobState
    total: int = 0
    succeeded: int = 0
    failed: int = 0


class BatchProvider(ABC):
    """Abstract base class for batch API providers.

    Implementations handle the provider-specific details of submitting,
    polling, and retrieving results from batch completion jobs.
    """

    @abstractmethod
    async def submit_completion_batch(
        self, requests: list[BatchRequest], model: str, **kwargs
    ) -> str:
        """Submit a batch of completion requests.

        Args:
            requests: List of BatchRequest objects to process.
            model: Model name to use for completions.
            **kwargs: Provider-specific options.

        Returns:
            Job ID string for tracking the batch.
        """

    @abstractmethod
    async def get_job_status(self, job_id: str) -> BatchJobStatus:
        """Get the current status of a batch job.

        Args:
            job_id: The job ID returned by submit_completion_batch.

        Returns:
            Current BatchJobStatus.
        """

    @abstractmethod
    async def get_results(self, job_id: str) -> list[BatchResponse]:
        """Retrieve results for a completed batch job.

        Should only be called after the job reaches SUCCEEDED state.
        May include per-row errors for partially failed batches.

        Args:
            job_id: The job ID returned by submit_completion_batch.

        Returns:
            List of BatchResponse objects, one per input request.
        """

    @abstractmethod
    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running or pending batch job.

        Args:
            job_id: The job ID to cancel.
        """

    async def await_completion(
        self,
        job_id: str,
        poll_interval: float = 30.0,
        timeout: float = 3600.0,
    ) -> BatchJobStatus:
        """Poll until the batch job reaches a terminal state.

        Args:
            job_id: The job ID to monitor.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Final BatchJobStatus (SUCCEEDED, FAILED, or CANCELLED).

        Raises:
            TimeoutError: If the job does not complete within the timeout.
        """
        start = time.time()
        while True:
            status = await self.get_job_status(job_id)
            if status.state in (
                BatchJobState.SUCCEEDED,
                BatchJobState.FAILED,
                BatchJobState.CANCELLED,
            ):
                return status

            elapsed = time.time() - start
            if elapsed > timeout:
                await self.cancel_job(job_id)
                raise TimeoutError(f"Batch job {job_id} timed out after {timeout}s")

            await asyncio.sleep(poll_interval)
