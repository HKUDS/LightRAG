"""Langfuse tracing for LightRAG LLM and embedding calls.

Provides provider-agnostic observability by wrapping LLM and embedding calls

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


_tracing_available: bool | None = None


def is_tracing_enabled() -> bool:
    """Check whether Langfuse tracing is enabled and available.

    Evaluates once on first call and caches the result.  Returns True only when
    both API keys are set AND the ``langfuse`` package is importable.
    """
    global _tracing_available
    if _tracing_available is not None:
        return _tracing_available

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        _tracing_available = False
        logger.debug("Langfuse tracing disabled: API keys not configured")
        return False

    try:
        import langfuse  # noqa: F401

        _tracing_available = True
        logger.info("Langfuse tracing enabled")
    except ImportError:
        _tracing_available = False
        logger.debug(
            "Langfuse tracing disabled: langfuse package not installed "
            "(install with: pip install lightrag-hku[observability])"
        )
    return _tracing_available


def get_langfuse_client():
    """Return the Langfuse singleton client, or ``None`` when tracing is disabled."""
    if not is_tracing_enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:
        logger.warning("Failed to get Langfuse client: %s", exc)
        return None


_MAX_TRACE_TEXT_LENGTH = 500


def _truncate(text: str | None, max_length: int = _MAX_TRACE_TEXT_LENGTH) -> str | None:
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


@contextlib.asynccontextmanager
async def trace_operation(
    name: str,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a root trace span for a high-level operation (insert / query).

    Yields the observation handle so callers can update output or metadata.
    When tracing is disabled, yields ``None`` with zero overhead.
    """
    if not is_tracing_enabled():
        yield None
        return

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_span(
            name=name,
            input=input_data,
            metadata=metadata,
        ) as span:
            yield span
    except Exception as exc:
        logger.warning("Langfuse trace_operation error: %s", exc)
        yield None


@contextlib.asynccontextmanager
async def trace_generation(
    name: str,
    model: str | None = None,
    input_text: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a generation observation for an LLM call.

    Yields the generation handle.  Callers should call
    ``gen.update(output=..., usage_details=...)`` after the underlying call
    completes.  When tracing is disabled, yields ``None``.
    """
    if not is_tracing_enabled():
        yield None
        return

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_generation(
            name=name,
            model=model,
            input=_truncate(input_text),
            metadata=metadata,
        ) as generation:
            yield generation
    except Exception as exc:
        logger.warning("Langfuse trace_generation error: %s", exc)
        yield None


@contextlib.asynccontextmanager
async def trace_embedding(
    name: str,
    model: str | None = None,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create an embedding observation for an embedding call.

    Uses Langfuse's dedicated ``embedding`` observation type.
    Yields the observation handle.  When tracing is disabled, yields ``None``.
    """
    if not is_tracing_enabled():
        yield None
        return

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_observation(
            as_type="embedding",
            name=name,
            model=model,
            input=input_data,
            metadata=metadata,
        ) as observation:
            yield observation
    except Exception as exc:
        logger.warning("Langfuse trace_embedding error: %s", exc)
        yield None


def create_traced_llm_wrapper(
    llm_func: callable, model_name: str = "unknown"
) -> callable:
    """Wrap an LLM function to emit a Langfuse generation on every call.

    The wrapper extracts the prompt from positional args and ``system_prompt``
    from kwargs, creates a generation observation, calls the original function,
    and updates the observation with the response.

    Args:
        llm_func: The LLM function to wrap (already priority-wrapped).
        model_name: Default model name for the generation observation.

    Returns:
        A wrapped async callable with the same interface as ``llm_func``.
    """
    if not is_tracing_enabled():
        return llm_func

    async def traced_llm_call(*args: Any, **kwargs: Any) -> Any:
        prompt = args[0] if args else kwargs.get("prompt", "")
        system_prompt = kwargs.get("system_prompt")

        try:
            async with trace_generation(
                name="llm-call",
                model=model_name,
                input_text=str(prompt) if prompt else None,
                metadata={
                    "has_system_prompt": system_prompt is not None,
                    "has_history": "history_messages" in kwargs
                    and bool(kwargs["history_messages"]),
                    "stream": kwargs.get("stream", False),
                },
            ) as gen:
                result = await llm_func(*args, **kwargs)
                if gen is not None and isinstance(result, str):
                    gen.update(output=_truncate(result))
                return result
        except Exception:
            # If tracing setup itself fails, fall through to untraced call
            return await llm_func(*args, **kwargs)

    return traced_llm_call


def create_traced_embedding_call(
    func: callable, model_name: str | None, embedding_dim: int
) -> callable:
    """Wrap an embedding function to emit a Langfuse embedding observation on every call.

    Args:
        func: The raw embedding function to wrap.
        model_name: Embedding model name.
        embedding_dim: Expected embedding dimension.

    Returns:
        A wrapped async callable with the same interface as ``func``.
    """
    if not is_tracing_enabled():
        return func

    async def traced_embed_call(*args: Any, **kwargs: Any) -> Any:
        text_count = len(args[0]) if args and isinstance(args[0], (list, tuple)) else 0

        try:
            async with trace_embedding(
                name="embedding",
                model=model_name,
                input_data={"text_count": text_count},
                metadata={
                    "embedding_dim": embedding_dim,
                    "text_count": text_count,
                },
            ) as obs:
                result = await func(*args, **kwargs)
                if obs is not None:
                    obs.update(
                        output=f"{text_count} vectors, dim={embedding_dim}",
                        usage_details={"input_tokens": text_count},
                    )
                return result
        except Exception:
            return await func(*args, **kwargs)

    return traced_embed_call


def flush() -> None:
    """Flush pending Langfuse events."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
            logger.debug("Langfuse traces flushed")
        except Exception as exc:
            logger.warning("Failed to flush Langfuse traces: %s", exc)
