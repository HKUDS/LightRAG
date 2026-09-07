"""Discover import-cheap registration functions in ``lightrag.chunkers``.

Mirrors the parser plugin seam. Each entry point resolves to a zero-argument
function calling register_chunker(ChunkerSpec(...)), never the implementation
itself. SDK users may call discovery before resolving a constructor callback.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from lightrag.chunker.registry import (
    ChunkerBinding,
    _plugin_registration,
    resolve_chunker,
)

ENTRY_POINT_GROUP = "lightrag.chunkers"
logger = logging.getLogger("lightrag")
_loaded = False
_failures: list[tuple[str, str, str]] = []


def _origin(ep) -> str:
    dist = getattr(ep, "dist", None)
    return f"{getattr(dist, 'name', 'unknown distribution')}@{getattr(dist, 'version', '?')} {ep.name}={ep.value}"


def _discover() -> list[str]:
    """Cache registrations and failure diagnostics, never implementations."""
    global _loaded
    if _loaded:
        return []
    _loaded = True
    loaded = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=_origin):
        origin = _origin(ep)
        try:
            with _plugin_registration(origin):
                register = ep.load()
                register()
        except Exception as exc:
            _failures.append((ep.name, origin, str(exc)))
            continue
        loaded.append(ep.name)
    return loaded


def _report_failures(outcome: str) -> None:
    for name, origin, error in _failures:
        logger.error(
            "[chunker-plugins] skipped failed plugin %r (%s): %s; %s",
            name,
            origin,
            error,
            outcome,
        )


def load_third_party_chunkers() -> list[str]:
    """Discover once for SDK registration without selecting a callback.

    Failed providers are logged and skipped, including partial registrations.
    Use load_and_resolve_chunker for selection-aware failure diagnostics.
    No implementation is imported unless the plugin violates the authoring
    contract by eagerly importing it inside its own registration module.
    """
    if _loaded:
        return []
    loaded = _discover()
    _report_failures("discovery only; no chunker selection has been validated")
    return loaded


def load_and_resolve_chunker(name: str | None) -> ChunkerBinding | None:
    """Validate before reporting whether a broken provider affects C startup.

    Gunicorn preload workers inherit the same registration/selection snapshot.
    Log outcomes, not guesses based on a failed entry point's unrelated name.
    """
    _discover()
    try:
        selected = resolve_chunker(name)
    except ValueError:
        _report_failures(
            f"CUSTOM_CHUNKER={name!r} could not be activated; startup aborted"
        )
        raise
    _report_failures(
        f"selected chunker {selected.spec.name!r} validated; C and no-selector chunking remain available"
        if selected
        else "CUSTOM_CHUNKER is unset; built-in callback and existing C admission/fallback are unchanged"
    )
    return selected
