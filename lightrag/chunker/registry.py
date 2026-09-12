"""Import-cheap third-party chunker specifications and startup resolution.

Only installed plugins supply implementation references. Operator selection is
a closed-set name, never an import string. Identity is diagnostic, not a fence.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from lightrag.utils import Tokenizer

logger = logging.getLogger("lightrag")
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_RESERVED = frozenset(
    {
        "fixed_token",
        "recursive_character",
        "semantic_vector",
        "paragraph_semantic",
        "f",
        "r",
        "v",
        "p",
        "c",
    }
)


@dataclass(frozen=True)
class ChunkerSpec:
    """Metadata only; ``impl`` is a lazy ``module:attribute`` reference.

    ``version`` is an opaque author-supplied observation. ``executor_safe``
    opts a synchronous, thread-safe implementation into the bounded chunking
    executor; the default preserves the legacy on-event-loop contract.
    Future optional fields can extend this spec without changing entry points.
    """

    name: str
    impl: str
    version: str
    description: str
    executor_safe: bool = False


@dataclass(frozen=True)
class _Registration:
    spec: ChunkerSpec
    origin: str


_REGISTRY: dict[str, _Registration] = {}
_DUPLICATES: set[str] = set()
_registration_origin: str | None = None


@contextmanager
def _plugin_registration(origin: str) -> Iterator[None]:
    """Attribute registrations to their provider; discard partial failures."""
    global _registration_origin
    previous_origin = _registration_origin
    previous = _REGISTRY.copy()
    previous_duplicates = _DUPLICATES.copy()
    _registration_origin = origin
    try:
        yield
    except Exception:
        _REGISTRY.clear()
        _REGISTRY.update(previous)
        _DUPLICATES.clear()
        _DUPLICATES.update(previous_duplicates)
        raise
    finally:
        _registration_origin = previous_origin


def register_chunker(spec: ChunkerSpec, *, origin: str | None = None) -> None:
    """Register without importing an implementation; report every collision."""
    if not isinstance(spec.name, str) or not _NAME.fullmatch(spec.name):
        raise ValueError("chunker name must match [a-z0-9][a-z0-9_-]{0,63}")
    if spec.name in _RESERVED:
        raise ValueError(f"reserved chunker name: {spec.name!r}")
    if not isinstance(spec.impl, str) or not spec.impl.strip():
        raise ValueError(
            f"chunker {spec.name!r} requires a lazy implementation reference"
        )
    if not isinstance(spec.version, str):
        raise ValueError(f"chunker {spec.name!r} version must be a string")
    if (
        not isinstance(spec.description, str)
        or not spec.description.strip()
        or any(c in spec.description for c in "\r\n")
    ):
        raise ValueError(f"chunker {spec.name!r} requires a one-line description")
    if not isinstance(spec.executor_safe, bool):
        raise ValueError(f"chunker {spec.name!r} executor_safe must be a bool")
    resolved_origin = _registration_origin or origin or spec.impl
    previous = _REGISTRY.get(spec.name)
    if previous is not None:
        _DUPLICATES.add(spec.name)
        logger.error(
            "[chunker-plugins] duplicate chunker %r: %s -> %s",
            spec.name,
            previous.origin,
            resolved_origin,
        )
    _REGISTRY[spec.name] = _Registration(spec, resolved_origin)


def registered_chunker_names() -> tuple[str, ...]:
    """Return names without importing any implementation."""
    return tuple(sorted(_REGISTRY))


class ChunkerBinding:
    """Callable injected at construction, with non-authoritative provenance.

    Not a dataclass: LightRAG's ``asdict`` snapshots must retain a callable,
    rather than recursively turning the callback into a metadata dictionary.
    """

    def __init__(self, spec: ChunkerSpec, implementation: Callable[..., Any]):
        self.spec = spec
        self.implementation = implementation

    def __call__(
        self,
        tokenizer: Tokenizer,
        content: str,
        split_by_character: str | None,
        split_by_character_only: bool,
        chunk_overlap_token_size: int,
        chunk_token_size: int,
    ) -> Any:
        args = (
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
        )
        if self.spec.executor_safe:
            from lightrag.utils import run_in_chunking_executor

            return run_in_chunking_executor(self.implementation, *args)
        return self.implementation(*args)


def resolve_chunker(name: str | None) -> ChunkerBinding | None:
    """Fail startup on invalid selection/import/arity; unset returns no override."""
    if name is None or name == "":
        return None
    if not isinstance(name, str):
        raise ValueError("CUSTOM_CHUNKER must be a registered chunker name")
    if any(c in name for c in ":/\\."):
        raise ValueError(
            "CUSTOM_CHUNKER: import paths are not supported; install a plugin using lightrag.chunkers entry points and select its registered name"
        )
    if not _NAME.fullmatch(name):
        raise ValueError("CUSTOM_CHUNKER must match [a-z0-9][a-z0-9_-]{0,63}")
    if name in _DUPLICATES:
        raise ValueError(
            f"CUSTOM_CHUNKER selects duplicate name {name!r}; remove the conflicting registration (see both origins in the error log)"
        )
    registration = _REGISTRY.get(name)
    if registration is None:
        raise ValueError(
            f"CUSTOM_CHUNKER: unknown chunker {name!r}; registered names: {', '.join(registered_chunker_names()) or '(none)'}. Check plugin discovery errors for a failed provider."
        )
    spec = registration.spec
    try:
        module, separator, attribute = spec.impl.partition(":")
        if not module or not separator or not attribute:
            raise ValueError("expected module:attribute")
        implementation = importlib.import_module(module)
        for part in attribute.split("."):
            implementation = getattr(implementation, part)
    except Exception as exc:
        raise ValueError(
            f"cannot load selected chunker {name!r} ({spec.impl}, {registration.origin}): {exc}"
        ) from exc
    if not callable(implementation):
        raise ValueError(f"selected chunker {name!r} ({spec.impl}) is not callable")
    try:
        signature = inspect.signature(implementation)
    except (ValueError, TypeError):
        pass  # Some native callables expose no signature; do not invent a fence.
    else:
        try:
            signature.bind(*([None] * 6))
        except TypeError as exc:
            raise ValueError(
                f"selected chunker {name!r} must accept six positional arguments: {exc}"
            ) from exc
    if spec.executor_safe and (
        inspect.iscoroutinefunction(implementation)
        or inspect.iscoroutinefunction(getattr(implementation, "__call__", None))
    ):
        raise ValueError(
            f"selected async chunker {name!r} cannot request executor offload"
        )
    return ChunkerBinding(spec, implementation)


def chunker_identity(callback: Callable[..., Any]) -> dict[str, Any] | None:
    """Describe the bound callback, never use persisted identity for selection."""
    if isinstance(callback, ChunkerBinding):
        return {
            "name": callback.spec.name,
            "version": callback.spec.version,
            "authoritative": False,
        }
    return None


def log_chunker_selection(selected: ChunkerBinding | None) -> None:
    """One startup summary, including instance-wide legacy-path implications."""
    registered = [
        {
            "name": r.spec.name,
            "version": r.spec.version,
            "origin": r.origin,
            "description": r.spec.description,
        }
        for _, r in sorted(_REGISTRY.items())
    ]
    logger.info(
        "[chunker-plugins] registered=%s; selected=%s%s",
        registered,
        selected.spec.name if selected else "(none)",
        "; serves C and no-selector inserts"
        if selected
        else "; built-in callback unchanged",
    )
