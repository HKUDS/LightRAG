"""Process-wide, name-only entity key plugins.

Installed entry points expose synchronous ``normalize(name: str) -> str``
callables. They must be pure, deterministic, idempotent and merge-only across
versions. See docs/ThirdPartyEntityNameNormalizer.md before changing an index's
selection. Selection is cached until process restart, including when unset.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Callable

from lightrag.constants import (
    DEFAULT_ENTITY_NAME_MAX_BYTES,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)

logger = logging.getLogger("lightrag")
ENTRY_POINT_GROUP = "lightrag.entity_name_normalizers"
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class EntityNameNormalizationError(ValueError):
    """A selected plugin failed; never skip a record or use another key."""


class EntityNameNormalizer:
    """Validate every output without truncating a plugin's canonical key."""

    def __init__(self, name: str, normalize: Callable[[str], str]):
        self.name = name
        self.normalize = normalize

    def _invoke(self, name: str) -> str:
        try:
            result = self.normalize(name)
            if not isinstance(result, str):
                # Close an accidental coroutine without an unawaited warning.
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("normalize(name) must return str")
            if not result:
                logger.warning(
                    "Entity-name normalizer %r returned an empty name; keeping input",
                    self.name,
                )
                result = name
            if (
                len(result) > DEFAULT_ENTITY_NAME_MAX_LENGTH
                or len(result.encode("utf-8")) > DEFAULT_ENTITY_NAME_MAX_BYTES
            ):
                raise ValueError(
                    f"output exceeds {DEFAULT_ENTITY_NAME_MAX_LENGTH} characters "
                    f"or {DEFAULT_ENTITY_NAME_MAX_BYTES} UTF-8 bytes"
                )
            return result
        except Exception as exc:
            raise EntityNameNormalizationError(
                f"Entity-name normalizer {self.name!r} failed: {exc}"
            ) from exc

    def __call__(self, name: str) -> str:
        result = self._invoke(name)
        if self._invoke(result) != result:
            raise EntityNameNormalizationError(
                f"Entity-name normalizer {self.name!r} is not idempotent"
            )
        return result


def _origin(ep) -> str:
    dist = getattr(ep, "dist", None)
    return (
        f"{getattr(dist, 'name', 'unknown distribution')}"
        f"@{getattr(dist, 'version', '?')} {ep.value}"
    )


@lru_cache(maxsize=1)
def get_entity_name_normalizer() -> EntityNameNormalizer | None:
    """Resolve once, failing startup on invalid selection or a failed probe.

    Called by Server startup and SDK construction, before storage work. Module
    callers resolve lazily on first use. Unset does not discover/import plugins.
    Gunicorn preload workers inherit the same selection. The startup probe is
    only a smoke check; input-dependent errors still fail at invocation time.
    """
    name = os.getenv("CUSTOM_ENTITY_NAME_NORMALIZER", "")
    if not name:
        return None
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"CUSTOM_ENTITY_NAME_NORMALIZER={name!r} must be a registered name "
            "matching [a-z0-9][a-z0-9_-]{0,63}, not an import path"
        )
    matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == name]
    if not matches:
        raise ValueError(
            f"CUSTOM_ENTITY_NAME_NORMALIZER: unknown normalizer {name!r}; "
            f"install a plugin exposing a {ENTRY_POINT_GROUP} entry point"
        )
    if len(matches) > 1:
        origins = ", ".join(sorted(_origin(ep) for ep in matches))
        raise ValueError(
            f"CUSTOM_ENTITY_NAME_NORMALIZER selects duplicate name {name!r}: {origins}"
        )
    ep = matches[0]
    try:
        normalize = ep.load()
        if not callable(normalize):
            raise TypeError("entry point must expose a callable normalize(name)")
        if inspect.iscoroutinefunction(normalize) or inspect.iscoroutinefunction(
            getattr(normalize, "__call__", None)
        ):
            raise TypeError("normalize(name) must be synchronous")
        selected = EntityNameNormalizer(name, normalize)
        selected("LightRAG")
    except Exception as exc:
        raise ValueError(
            f"Cannot activate entity-name normalizer {name!r} ({_origin(ep)}): {exc}"
        ) from exc
    logger.info("Entity-name normalizer selected: %r (%s)", name, _origin(ep))
    return selected


def apply_entity_name_normalizer(name: str) -> str:
    """Apply only to final keys: after the extraction cap and space pass."""
    selected = get_entity_name_normalizer()
    return selected(name) if selected is not None and name else name
