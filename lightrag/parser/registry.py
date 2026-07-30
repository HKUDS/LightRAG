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
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from lightrag.constants import (
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


# ---------------------------------------------------------------------------
# Deployment-configured suffix extensions (``ParserSpec.extra_suffixes_env``)
# ---------------------------------------------------------------------------
_SUFFIX_TOKEN = re.compile(r"^[a-z0-9]+$")


def _parse_env_suffixes(env_name: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """Split ``env_name``'s comma-separated list into (valid, malformed) tokens.

    Normalisation per entry: strip surrounding whitespace, lowercase, drop
    leading dots — so ``" .PPT "`` and ``"ppt"`` are the same suffix and a
    trailing comma is harmless.  Anything else that is not ``[a-z0-9]+`` is
    *malformed*, not a suffix: it can never equal ``Path.suffix``, so silently
    admitting it would leave the operator's actual intent unmet with no
    diagnostic.  Malformed tokens are excluded from the capability set and
    reported by :func:`malformed_env_suffixes` for the startup check.
    """
    valid: set[str] = set()
    malformed: list[str] = []
    for raw in os.getenv(env_name, "").split(","):
        token = raw.strip().lower().lstrip(".")
        if not token:
            continue
        if _SUFFIX_TOKEN.match(token):
            valid.add(token)
        else:
            malformed.append(raw.strip())
    return frozenset(valid), tuple(malformed)


def _env_suffixes(env_name: str) -> frozenset[str]:
    return _parse_env_suffixes(env_name)[0]


class _EnvExtendedSuffixes:
    """Descriptor-typed ``ParserSpec.suffixes``: declared baseline ∪ env extras.

    A spec that names an ``extra_suffixes_env`` reads back its declared
    baseline unioned with that variable's suffixes, resolved **at every
    access** rather than at construction.  ``_REGISTRY`` is a module-level
    literal built at import time, which for the parser debug CLI happens
    inside ``_build_parser()`` — before ``_run()`` imports ``lightrag.utils``
    and triggers ``load_dotenv``.  Baking the env in at construction would
    therefore permanently capture an empty set for anyone configuring the
    variable in ``.env`` rather than the parent shell.

    Reading live keeps every consumer honest without any of them knowing an
    env var is involved: ``spec.suffixes`` *is* the engine's capability, so
    ``suffix_capabilities`` / ``available_engine_suffixes`` stay one-liners.
    The cost is that ``__eq__`` / ``__hash__`` of a spec carrying
    ``extra_suffixes_env`` shift with the environment; specs are values in
    ``_REGISTRY`` and are never used as dict keys or set members.
    """

    def __set_name__(self, owner, name: str) -> None:
        self._attr = f"_{name}"

    def __get__(self, obj, objtype=None) -> frozenset[str]:
        if obj is None:
            # Deliberately not a dataclass default: ``suffixes`` stays a
            # required argument, so a registrant that forgets it still gets a
            # TypeError instead of an engine that silently matches nothing.
            raise AttributeError(self._attr)
        base: frozenset[str] = getattr(obj, self._attr)
        env_name = getattr(obj, "extra_suffixes_env", None)
        return base | _env_suffixes(env_name) if env_name else base

    def __set__(self, obj, value) -> None:
        # frozen dataclass: bypass the instance-assignment guard, and coerce so
        # a registrant passing a plain ``set`` cannot have it mutated in place.
        object.__setattr__(obj, self._attr, frozenset(value))


@dataclass(frozen=True)
class ParserSpec:
    """Lightweight, import-cheap metadata for one parser engine.

    Holds everything the pipeline needs to *route* and *gate* a document
    without importing the parser implementation.  ``impl`` is a
    ``"module:Class"`` string imported lazily by :func:`get_parser`.
    """

    engine_name: str
    impl: str
    suffixes: frozenset[str] = _EnvExtendedSuffixes()
    user_selectable: bool = True
    queue_group: str = PARSER_ENGINE_NATIVE
    # Worker count for this spec's queue_group. The registrant bakes in any
    # env override at registration time (e.g.
    # ``concurrency=int(os.getenv("MAX_PARALLEL_PARSE_MYENGINE", "3"))``),
    # mirroring how the built-in ``max_parallel_parse_*`` LightRAG fields read
    # their env. ``None`` means this spec does not own its group's concurrency
    # (built-in groups are sized by the LightRAG instance field instead).
    concurrency: int | None = None
    endpoint_configured: Callable[[], bool] = field(default=lambda: True)
    endpoint_requirement: Callable[[], str | None] = field(default=lambda: None)
    # Name of a deployment env var whose comma-separated list extends
    # ``suffixes`` (read live — see :class:`_EnvExtendedSuffixes`). For engines
    # whose real format coverage depends on optional packages installed on the
    # endpoint rather than on LightRAG: docling's legacy Office formats need
    # LibreOffice on the docling-serve side, so the baseline declaration stays
    # honest and each deployment opts the rest in. Malformed entries are
    # rejected at startup (``validate_parser_suffix_env_vars``).
    extra_suffixes_env: str | None = None


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
# Formats a MinerU endpoint handles out of the box. Which of the remaining
# MinerU input formats work depends on the endpoint (legacy Office goes through
# LibreOffice on MinerU's side, and the official API's coverage is fixed by the
# service) and on the active MINERU_API_MODE, so they are opted into per
# deployment via MINERU_ADDITIONAL_SUFFIXES (see ``extra_suffixes_env``).
_MINERU_SUFFIXES = frozenset(
    {
        "pdf",
        "docx",
        "pptx",
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


# Formats available in the baseline Docling deployment. Optional converters
# depend on packages installed by the endpoint and are opted into per
# deployment via DOCLING_ADDITIONAL_SUFFIXES (see ``extra_suffixes_env``).
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
        # Single ``native`` engine; the dispatcher picks docx vs markdown by
        # source suffix (see lightrag.parser.native_dispatch).
        impl="lightrag.parser.native_dispatch:NativeParser",
        suffixes=frozenset({"docx", "md", "textpack"}),
        queue_group=PARSER_ENGINE_NATIVE,
        # Built-in groups are sized by the LightRAG ``max_parallel_parse_*``
        # instance field (supports constructor override), so no spec-level
        # ``concurrency`` here.
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
        queue_group=PARSER_ENGINE_MINERU,  # sized by max_parallel_parse_mineru
        endpoint_configured=_mineru_endpoint_configured,
        endpoint_requirement=_mineru_endpoint_requirement,
        extra_suffixes_env="MINERU_ADDITIONAL_SUFFIXES",
    ),
    PARSER_ENGINE_DOCLING: ParserSpec(
        engine_name=PARSER_ENGINE_DOCLING,
        impl="lightrag.parser.external.docling.parser:DoclingParser",
        suffixes=_DOCLING_SUFFIXES,
        queue_group=PARSER_ENGINE_DOCLING,  # sized by max_parallel_parse_docling
        endpoint_configured=_env_endpoint_configured("DOCLING_ENDPOINT"),
        endpoint_requirement=lambda: "DOCLING_ENDPOINT",
        extra_suffixes_env="DOCLING_ADDITIONAL_SUFFIXES",
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
    """Register (or override) a parser engine spec."""
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


def available_engine_suffixes(
    specs: dict[str, ParserSpec] | None = None,
) -> frozenset[str]:
    """Suffixes (lowercase, no dot) parseable by a *currently usable* engine.

    Union over user-selectable engines whose ``endpoint_configured()`` gate
    passes. This is the single source for the API upload allowlist and the
    input-directory scan (``DocumentManager.supported_extensions``): in a
    default deployment (no external endpoints) it equals the local engines'
    suffixes (legacy ∪ native); configuring e.g. ``MINERU_LOCAL_ENDPOINT``
    admits mineru's image/office suffixes; a registered third-party engine's
    suffixes join automatically (subject to its own endpoint gate).
    """
    out: set[str] = set()
    for spec in _table(specs).values():
        if spec.user_selectable and spec.endpoint_configured():
            out |= spec.suffixes
    return frozenset(out)


def suffix_capabilities(
    engine: str, specs: dict[str, ParserSpec] | None = None
) -> frozenset[str]:
    spec = _table(specs).get(engine)
    return spec.suffixes if spec is not None else frozenset()


def malformed_env_suffixes(
    specs: dict[str, ParserSpec] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Env var name -> its malformed entries, over specs declaring one.

    Empty when every ``extra_suffixes_env`` holds a well-formed list (the
    common case, including "unset"). Consumed by the startup check
    ``routing.validate_parser_suffix_env_vars``; kept here so the parsing rule
    and the reporting share one implementation.
    """
    out: dict[str, tuple[str, ...]] = {}
    for spec in _table(specs).values():
        env_name = spec.extra_suffixes_env
        if not env_name:
            continue
        malformed = _parse_env_suffixes(env_name)[1]
        if malformed:
            out[env_name] = malformed
    return out


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
