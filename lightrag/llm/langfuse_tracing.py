"""Runtime control for Langfuse tracing.

Credentials and the Langfuse destination remain operator-managed environment
configuration. This module controls only whether configured tracing is active
at runtime.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import MutableMapping
from typing import Any

from lightrag.kg.shared_storage import get_namespace_data
from lightrag.utils import logger


LANGFUSE_TRACING_NAMESPACE = "langfuse_tracing"
LANGFUSE_TRACING_ENABLED_KEY = "enabled"

_langfuse_tracing_state: MutableMapping[str, Any] | None = None
_langfuse_client: Any | None = None


def _configured_default() -> bool:
    """Return the operator-configured tracing default.

    Tracing remains enabled by default when Langfuse credentials are present,
    preserving the behavior that predates LANGFUSE_ENABLE_TRACE.
    """

    value = os.getenv("LANGFUSE_ENABLE_TRACE")
    if value is None:
        return True

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning(
        "Invalid LANGFUSE_ENABLE_TRACE value; expected true/false, "
        "1/0, yes/no, or on/off. Defaulting to enabled."
    )
    return True


def is_langfuse_installed() -> bool:
    """Return whether the optional Langfuse dependency is installed."""

    try:
        return importlib.util.find_spec("langfuse") is not None
    except (ImportError, ValueError):
        return False


def is_langfuse_configured() -> bool:
    """Return whether the operator supplied both required credentials."""

    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def should_export_langfuse_span(_span: Any) -> bool:
    """Evaluate the live shared tracing switch for each span."""

    state = _langfuse_tracing_state
    if state is None:
        return False

    try:
        return bool(state.get(LANGFUSE_TRACING_ENABLED_KEY, False))
    except Exception:
        # Shared storage may already be shutting down. Observability must never
        # break an application request during teardown.
        return False


async def initialize_langfuse_tracing() -> None:
    """Resolve shared state and construct the configured Langfuse client."""

    global _langfuse_tracing_state, _langfuse_client

    state = await get_namespace_data(LANGFUSE_TRACING_NAMESPACE, workspace="")
    state.setdefault(LANGFUSE_TRACING_ENABLED_KEY, _configured_default())
    _langfuse_tracing_state = state

    if not is_langfuse_installed() or not is_langfuse_configured():
        _langfuse_client = None
        return

    from langfuse import Langfuse

    _langfuse_client = Langfuse(
        should_export_span=should_export_langfuse_span,
    )
    logger.info("Langfuse runtime tracing control initialized")


def set_langfuse_tracing_enabled(enabled: bool) -> None:
    """Update the deployment-wide runtime tracing switch."""

    state = _langfuse_tracing_state
    if state is None:
        raise RuntimeError("Langfuse runtime tracing state is unavailable")

    try:
        # A single manager.dict assignment is one atomic IPC operation, so this
        # boolean update does not require a namespace lock.
        state[LANGFUSE_TRACING_ENABLED_KEY] = bool(enabled)
    except Exception as exc:
        raise RuntimeError(
            "Langfuse runtime tracing state is unavailable"
        ) from exc


def get_langfuse_tracing_status() -> dict[str, bool]:
    """Return non-sensitive runtime tracing status."""

    installed = is_langfuse_installed()
    configured = is_langfuse_configured()
    enabled = should_export_langfuse_span(None)

    return {
        "installed": installed,
        "configured": configured,
        "enabled": enabled,
        "active": installed
        and configured
        and enabled
        and _langfuse_client is not None,
    }
