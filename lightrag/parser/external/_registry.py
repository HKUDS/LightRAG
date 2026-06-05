"""Registry for external parser engines.

Maps engine names to their :class:`BaseExternalParser` implementations.
New engines register here instead of adding ``if engine == …`` branches
in the pipeline.

Usage::

    from lightrag.parser.external._registry import get_parser, register_parser

    # At module level (each engine registers itself):
    register_parser("mineru", MinerUParser)
    register_parser("docling", DoclingParser)

    # At runtime:
    parser = get_parser("mineru")
    if parser.is_bundle_valid(raw_dir, source_path):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from lightrag.parser.external._base import BaseExternalParser

_REGISTRY: dict[str, Type[BaseExternalParser]] = {}


def register_parser(engine_name: str, parser_cls: Type[BaseExternalParser]) -> None:
    """Register a parser class for an engine name."""
    _REGISTRY[engine_name] = parser_cls


def get_parser(engine_name: str) -> BaseExternalParser:
    """Get a parser instance for an engine name.

    Raises KeyError if the engine is not registered.
    """
    cls = _REGISTRY.get(engine_name)
    if cls is None:
        raise KeyError(
            f"No parser registered for engine {engine_name!r}. "
            f"Registered engines: {', '.join(sorted(_REGISTRY))}"
        )
    return cls()


def registered_engines() -> list[str]:
    """Return sorted list of registered engine names."""
    return sorted(_REGISTRY)
