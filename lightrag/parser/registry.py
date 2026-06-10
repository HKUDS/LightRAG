"""Central registry for parser engines (mirrors the storage-layer convention).

Like :data:`lightrag.kg.STORAGES` / ``STORAGE_IMPLEMENTATIONS``, this module
holds a module-level literal table of lightweight :class:`ParserSpec`
metadata.  Loading this module imports **no** parser implementation, so
capability queries (suffixes / endpoint / supported engines) never trigger a
heavy ``mineru``/``docling`` facade import (which would pull ``httpx`` etc.).
The implementation class is imported lazily — only when a document is
actually parsed — via :func:`get_parser`.

Capability data lives here (single source of truth); behaviour lives in the
parser classes.  ``constants.PARSER_ENGINE_*`` keeps only the bare name
strings used as identifiers / registry keys.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from lightrag.constants import (
    DEFAULT_MAX_PARALLEL_PARSE_DOCLING,
    DEFAULT_MAX_PARALLEL_PARSE_MINERU,
    DEFAULT_MAX_PARALLEL_PARSE_NATIVE,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)

if TYPE_CHECKING:
    from lightrag.parser.base import BaseParser

# Internal format-handler engine keys (not user-selectable).
PARSER_ENGINE_REUSE = "reuse"
PARSER_ENGINE_PASSTHROUGH = "passthrough"

_VALID_MINERU_API_MODES = {"official", "local"}


# ---------------------------------------------------------------------------
# Endpoint capability closures (env-only; no network).  Canonical home —
# routing.py delegates here.
# ---------------------------------------------------------------------------
def _mineru_endpoint_configured() -> bool:
    mode = os.getenv("MINERU_API_MODE", "local").strip().lower()
    if mode == "official":
        return bool(os.getenv("MINERU_API_TOKEN", "").strip())
    if mode == "local":
        return bool(os.getenv("MINERU_LOCAL_ENDPOINT", "").strip())
    return False


def _mineru_endpoint_requirement() -> str | None:
    mode = os.getenv("MINERU_API_MODE", "local").strip().lower()
    if mode == "official":
        return "MINERU_API_TOKEN"
    if mode == "local":
        return "MINERU_LOCAL_ENDPOINT"
    allowed = ", ".join(sorted(_VALID_MINERU_API_MODES))
    return f"valid MINERU_API_MODE ({allowed})"


def _env_endpoint_configured(env_name: str) -> Callable[[], bool]:
    return lambda: bool(os.getenv(env_name, "").strip())


@dataclass(frozen=True)
class ParserSpec:
    """Lightweight, import-cheap metadata for one parser engine.

    Holds everything the pipeline needs to *route* and *gate* a document
    without importing the parser implementation.  ``impl`` is a
    ``"module:Class"`` string imported lazily by :func:`get_parser`.
    """

    engine_name: str
    impl: str
    suffixes: frozenset[str]
    user_selectable: bool = True
    queue_group: str = PARSER_ENGINE_NATIVE
    concurrency_env: str | None = None
    default_concurrency: int | None = None
    endpoint_configured: Callable[[], bool] = field(default=lambda: True)
    endpoint_requirement: Callable[[], str | None] = field(default=lambda: None)


# ---------------------------------------------------------------------------
# Suffix capabilities (single source of truth; replaces
# constants.PARSER_ENGINE_SUFFIX_CAPABILITIES).
# ---------------------------------------------------------------------------
_LEGACY_SUFFIXES = frozenset(
    {
        "txt",
        "md",
        "mdx",
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "rtf",
        "odt",
        "tex",
        "epub",
        "html",
        "htm",
        "csv",
        "json",
        "xml",
        "yaml",
        "yml",
        "log",
        "conf",
        "ini",
        "properties",
        "sql",
        "bat",
        "sh",
        "c",
        "h",
        "cpp",
        "hpp",
        "py",
        "java",
        "js",
        "ts",
        "swift",
        "go",
        "rb",
        "php",
        "css",
        "scss",
        "less",
    }
)
_MINERU_SUFFIXES = frozenset(
    {
        "pdf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
        "png",
        "jpg",
        "jpeg",
        "jp2",
        "webp",
        "gif",
        "bmp",
    }
)
_DOCLING_SUFFIXES = frozenset(
    {
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "md",
        "html",
        "xhtml",
        "png",
        "jpg",
        "jpeg",
        "tiff",
        "webp",
        "bmp",
    }
)


_REGISTRY: dict[str, ParserSpec] = {
    PARSER_ENGINE_NATIVE: ParserSpec(
        engine_name=PARSER_ENGINE_NATIVE,
        impl="lightrag.parser.docx.parser:NativeDocxParser",
        suffixes=frozenset({"docx"}),
        queue_group=PARSER_ENGINE_NATIVE,
        concurrency_env="MAX_PARALLEL_PARSE_NATIVE",
        default_concurrency=DEFAULT_MAX_PARALLEL_PARSE_NATIVE,
    ),
    PARSER_ENGINE_LEGACY: ParserSpec(
        engine_name=PARSER_ENGINE_LEGACY,
        impl="lightrag.parser.legacy.parser:LegacyParser",
        suffixes=_LEGACY_SUFFIXES,
        queue_group=PARSER_ENGINE_NATIVE,  # shares native pool (local, no network)
    ),
    PARSER_ENGINE_MINERU: ParserSpec(
        engine_name=PARSER_ENGINE_MINERU,
        impl="lightrag.parser.external.mineru.parser:MinerUParser",
        suffixes=_MINERU_SUFFIXES,
        queue_group=PARSER_ENGINE_MINERU,
        concurrency_env="MAX_PARALLEL_PARSE_MINERU",
        default_concurrency=DEFAULT_MAX_PARALLEL_PARSE_MINERU,
        endpoint_configured=_mineru_endpoint_configured,
        endpoint_requirement=_mineru_endpoint_requirement,
    ),
    PARSER_ENGINE_DOCLING: ParserSpec(
        engine_name=PARSER_ENGINE_DOCLING,
        impl="lightrag.parser.external.docling.parser:DoclingParser",
        suffixes=_DOCLING_SUFFIXES,
        queue_group=PARSER_ENGINE_DOCLING,
        concurrency_env="MAX_PARALLEL_PARSE_DOCLING",
        default_concurrency=DEFAULT_MAX_PARALLEL_PARSE_DOCLING,
        endpoint_configured=_env_endpoint_configured("DOCLING_ENDPOINT"),
        endpoint_requirement=lambda: "DOCLING_ENDPOINT",
    ),
    PARSER_ENGINE_REUSE: ParserSpec(
        engine_name=PARSER_ENGINE_REUSE,
        impl="lightrag.parser.noop:ReuseParser",
        suffixes=frozenset(),
        user_selectable=False,
    ),
    PARSER_ENGINE_PASSTHROUGH: ParserSpec(
        engine_name=PARSER_ENGINE_PASSTHROUGH,
        impl="lightrag.parser.noop:PassthroughParser",
        suffixes=frozenset(),
        user_selectable=False,
    ),
}

# (engine_name, impl) -> instance.  Keyed on impl so a re-registration with a
# different implementation is not served a stale cached instance.
_INSTANCE_CACHE: dict[tuple[str, str], "BaseParser"] = {}


def register_parser(spec: ParserSpec) -> None:
    """Register (or override) a parser engine spec.

    A spec that declares ``concurrency_env`` must also provide a
    ``default_concurrency`` so the pipeline never evaluates ``int(None)``
    when the env var is unset.
    """
    if spec.concurrency_env and spec.default_concurrency is None:
        raise ValueError(
            f"ParserSpec {spec.engine_name!r} sets concurrency_env="
            f"{spec.concurrency_env!r} but no default_concurrency"
        )
    _REGISTRY[spec.engine_name] = spec


def parser_specs_snapshot() -> dict[str, ParserSpec]:
    """Return a shallow snapshot of the registry.

    The pipeline takes one snapshot at batch start and threads it through
    queue construction, routing and the parse workers so a concurrent
    ``register_parser`` cannot change the engine set mid-batch.
    """
    return dict(_REGISTRY)


def _table(specs: dict[str, ParserSpec] | None) -> dict[str, ParserSpec]:
    return specs if specs is not None else _REGISTRY


def get_parser(engine: str, *, specs: dict[str, ParserSpec] | None = None):
    """Return a (cached) parser instance for ``engine`` or ``None``.

    Imports the implementation lazily via ``importlib`` — only here does the
    heavy engine package (and e.g. ``httpx``) get pulled in.
    """
    spec = _table(specs).get(engine)
    if spec is None:
        return None
    cache_key = (engine, spec.impl)
    inst = _INSTANCE_CACHE.get(cache_key)
    if inst is None:
        module_path, _, cls_name = spec.impl.partition(":")
        cls = getattr(importlib.import_module(module_path), cls_name)
        inst = cls()
        _INSTANCE_CACHE[cache_key] = inst
    return inst


def supported_parser_engines(
    specs: dict[str, ParserSpec] | None = None,
) -> frozenset[str]:
    """User-selectable engine names (replaces SUPPORTED_PARSER_ENGINES)."""
    return frozenset(
        name for name, spec in _table(specs).items() if spec.user_selectable
    )


def suffix_capabilities(
    engine: str, specs: dict[str, ParserSpec] | None = None
) -> frozenset[str]:
    spec = _table(specs).get(engine)
    return spec.suffixes if spec is not None else frozenset()


def engine_endpoint_configured(
    engine: str, specs: dict[str, ParserSpec] | None = None
) -> bool:
    spec = _table(specs).get(engine)
    return spec.endpoint_configured() if spec is not None else True


def engine_endpoint_requirement(
    engine: str, specs: dict[str, ParserSpec] | None = None
) -> str | None:
    spec = _table(specs).get(engine)
    return spec.endpoint_requirement() if spec is not None else None
