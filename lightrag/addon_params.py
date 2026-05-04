"""Addon parameters: observable mapping + normalization helper.

``addon_params`` is a free-form configuration dict on :class:`LightRAG` that
controls things like summary language and entity-type prompt overrides. The
module exposes:

- :class:`ObservableAddonParams` — a ``dict`` subclass that calls a callback
  whenever the contents change so the LightRAG runtime can invalidate cached
  derived state.
- :func:`default_addon_params` — environment-driven defaults.
- :func:`normalize_addon_params` — converts an arbitrary input into a plain
  ``dict`` with the env-driven defaults backfilled.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from lightrag.constants import DEFAULT_SUMMARY_LANGUAGE
from lightrag.utils import get_env_value


def default_addon_params() -> dict[str, Any]:
    return {
        "language": get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE, str),
        "entity_type_prompt_file": get_env_value("ENTITY_TYPE_PROMPT_FILE", "", str),
    }


def normalize_addon_params(addon_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Coerce ``addon_params`` to a plain dict with env defaults backfilled."""
    if addon_params is None:
        normalized = default_addon_params()
    elif isinstance(addon_params, Mapping):
        normalized = dict(addon_params)
    else:
        raise TypeError(
            "addon_params must be a Mapping or None, got "
            f"{type(addon_params).__name__}"
        )

    # When the caller supplies addon_params explicitly, the dataclass
    # default_factory is skipped — fall back to environment variables so
    # ENTITY_TYPE_PROMPT_FILE / SUMMARY_LANGUAGE still apply.
    normalized.setdefault(
        "language", get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE, str)
    )
    normalized.setdefault(
        "entity_type_prompt_file",
        get_env_value("ENTITY_TYPE_PROMPT_FILE", "", str),
    )
    return normalized


class ObservableAddonParams(dict[str, Any]):
    def __init__(
        self,
        *args: Any,
        on_change: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_change = on_change

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._changed()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._changed()

    def clear(self) -> None:
        if self:
            super().clear()
            self._changed()

    def pop(self, key: str, default: Any = ...):
        existed = key in self
        if default is ...:
            value = super().pop(key)
            self._changed()
        else:
            value = super().pop(key, default)
            if existed:
                self._changed()
        return value

    def popitem(self) -> tuple[str, Any]:
        item = super().popitem()
        self._changed()
        return item

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        value = super().setdefault(key, default)
        self._changed()
        return value

    def update(self, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            return
        super().update(*args, **kwargs)
        self._changed()

    def __ior__(self, other: Mapping[str, Any]):
        super().__ior__(other)
        self._changed()
        return self
