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
_CHUNKING_CONTEXT_MARKER = "__lightrag_accepts_chunking_context__"


@dataclass(frozen=True, kw_only=True)
class ChunkingContext:
    """Document metadata made available to explicitly context-aware chunkers.

    The context is intentionally separate from the chunk text and legacy
    sizing arguments.  Chunkers can use it for source-aware policies without
    changing the text that is embedded or requiring a second callback shape.
    """

    doc_id: str
    file_path: str
    sidecar_location: str | None
    parse_format: str
    parse_engine: str | None
    process_options: str


def accepts_chunking_context(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a constructor-supplied callback as accepting ``context=``.

    This is an explicit opt-in so legacy six-argument callbacks remain
    indistinguishable from their historical invocation.  The marker is read
    at dispatch time and is not inferred from a callback failure.
    """

    if not callable(callback):
        raise TypeError("chunking callback must be callable")
    setattr(callback, _CHUNKING_CONTEXT_MARKER, True)
    return callback


def callback_supports_context(callback: Callable[..., Any]) -> bool:
    """Return whether a callback explicitly opted into ``context=``."""

    return bool(
        getattr(callback, _CHUNKING_CONTEXT_MARKER, False)
        or getattr(callback, "supports_chunking_context", False)
        or getattr(callback, "accepts_chunking_context", False)
    )


def invoke_chunker(
    callback: Callable[..., Any],
    *args: Any,
    context: ChunkingContext,
) -> Any:
    """Invoke a legacy or context-aware callback exactly once.

    Signature probing followed by a retry is deliberately avoided: a
    ``TypeError`` raised inside a callback is a real callback failure and must
    never cause the implementation to run a second time.
    """

    if isinstance(callback, ChunkerBinding):
        return callback(*args, context=context)
    if callback_supports_context(callback):
        return callback(*args, context=context)
    return callback(*args)


@dataclass(frozen=True)
class ChunkerSpec:
    """Metadata only; ``impl`` is a lazy ``module:attribute`` reference.

    ``version`` is an opaque author-supplied observation. ``executor_safe``
    opts a synchronous, thread-safe implementation into the bounded chunking
    executor; ``accepts_context`` opts into the keyword-only
    :class:`ChunkingContext` argument. The defaults preserve the legacy
    six-positional-argument, on-event-loop contract.
    Future optional fields can extend this spec without changing entry points.
    """

    name: str
    impl: str
    version: str
    description: str
    executor_safe: bool = False
    accepts_context: bool = False


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
    if not isinstance(spec.accepts_context, bool):
        raise ValueError(f"chunker {spec.name!r} accepts_context must be a bool")
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
    """Return every registered name without importing any implementation.

    Includes names that collided, so the startup summary can show what a
    provider actually registered. Use :func:`selectable_chunker_names` for
    anything that answers "what may CUSTOM_CHUNKER be set to".
    """
    return tuple(sorted(_REGISTRY))


def selectable_chunker_names() -> tuple[str, ...]:
    """Return the names ``CUSTOM_CHUNKER`` can actually be set to.

    A duplicated name is registered but unselectable -- ``resolve_chunker``
    refuses it -- so offering it as a choice sends the operator to a value
    that is guaranteed to fail startup.
    """
    return tuple(sorted(name for name in _REGISTRY if name not in _DUPLICATES))


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
        *,
        context: ChunkingContext | None = None,
    ) -> Any:
        args = (
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
        )
        kwargs = {"context": context} if self.spec.accepts_context else {}
        if self.spec.executor_safe:
            from lightrag.utils import run_in_chunking_executor

            return run_in_chunking_executor(self.implementation, *args, **kwargs)
        return self.implementation(*args, **kwargs)


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
            f"CUSTOM_CHUNKER: unknown chunker {name!r}; selectable names: {', '.join(selectable_chunker_names()) or '(none)'}. Check plugin discovery errors for a failed provider."
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
        if spec.accepts_context:
            try:
                signature.bind(*([None] * 6), context=None)
            except TypeError as exc:
                raise ValueError(
                    f"selected context-aware chunker {name!r} must accept six positional arguments plus keyword-only context: {exc}"
                ) from exc
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
        "; serves C and no-selector inserts, which therefore forgo source-span"
        " sidecar backfill exactly as a constructor-supplied callback does"
        if selected
        else "; built-in callback unchanged",
    )
