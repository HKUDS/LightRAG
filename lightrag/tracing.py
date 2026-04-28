"""Langfuse tracing for LightRAG LLM and embedding calls.

Provides provider-agnostic observability using the Langfuse ``@observe``
decorator and SDK helpers.  The ``@observe`` decorator is applied directly
on LightRAG methods (``ainsert``, ``aquery_llm``, etc.) — this module
provides only thin wrappers and utilities.

Enable by setting environment variables:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com  (optional)
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

logger = logging.getLogger("lightrag")


def is_tracing_enabled() -> bool:
    """Check whether Langfuse tracing is enabled and available.

    Returns True only when both API keys are set AND the ``langfuse``
    package is importable.  Re-evaluated on every call so that env var
    changes take effect without a process restart.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return False

    try:
        import langfuse  # noqa: F401

        return True  # noqa: TRY300
    except ImportError:
        return False


def create_traced_llm_wrapper(
    llm_func: callable, model_name: str = "unknown"
) -> callable:
    """Wrap an LLM function to emit a Langfuse generation on every call.

    Uses ``observe()`` functional form to auto-capture input/output.
    The inner wrapper sets the ``model`` attribute which ``observe()``
    does not support as a direct parameter.
    """
    if not is_tracing_enabled():
        return llm_func

    from langfuse import get_client, observe

    async def traced_llm_call(*args: Any, **kwargs: Any) -> Any:
        client = get_client()
        if client is not None:
            client.update_current_generation(model=model_name)
        return await llm_func(*args, **kwargs)

    return observe(traced_llm_call, name="llm-call", as_type="generation")


def report_token_usage(usage_details: dict[str, int]) -> None:
    """Report token usage to the current Langfuse generation observation.

    Call this from inside an active generation context (e.g. from within
    an LLM provider function) to attach token counts to the generation.
    """
    if not is_tracing_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        if client is not None:
            client.update_current_generation(usage_details=usage_details)
    except Exception as exc:
        logger.warning("Failed to report token usage to Langfuse: %s", exc)


@contextlib.contextmanager
def propagate_trace_attributes(
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    trace_name: str | None = None,
):
    """Set trace-level attributes for all observations created in this context.

    Wraps ``langfuse.propagate_attributes`` with graceful degradation when
    tracing is disabled.
    """
    if not is_tracing_enabled():
        yield
        return

    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            trace_name=trace_name,
        ):
            yield
    except Exception as exc:
        logger.warning("Langfuse propagate_attributes error: %s", exc)
        yield


def flush() -> None:
    """Flush pending Langfuse events."""
    if not is_tracing_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        if client is not None:
            client.flush()
            logger.debug("Langfuse traces flushed")
    except Exception as exc:
        logger.warning("Failed to flush Langfuse traces: %s", exc)


def shutdown() -> None:
    """Gracefully shut down the Langfuse client (flushes + waits for background threads)."""
    if not is_tracing_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        if client is not None:
            client.shutdown()
            logger.debug("Langfuse client shut down")
    except Exception as exc:
        logger.warning("Failed to shut down Langfuse client: %s", exc)
