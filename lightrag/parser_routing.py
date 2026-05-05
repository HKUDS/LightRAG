from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
    PARSER_ENGINE_SUFFIX_CAPABILITIES,
    PROCESS_OPTION_CHUNK_CHARS,
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_HEADING,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_EQUATIONS,
    PROCESS_OPTION_IMAGES,
    PROCESS_OPTION_SKIP_KG,
    PROCESS_OPTION_TABLES,
    SUPPORTED_PARSER_ENGINES,
    SUPPORTED_PROCESS_OPTIONS,
)
from lightrag.utils import logger

_PARSER_RULE_SPLIT_RE = re.compile(r"[;,]")
_PARSER_ENGINE_ENDPOINT_ENV = {
    PARSER_ENGINE_MINERU: "MINERU_ENDPOINT",
    PARSER_ENGINE_DOCLING: "DOCLING_ENDPOINT",
}

# Trailing parser-hint pattern: matches ``.[engine].ext`` at end of basename.
# Group 1 captures the raw engine token (still needs normalize_parser_engine
# and SUPPORTED_PARSER_ENGINES validation); group 2 captures ``.ext`` so it
# can be reattached when stripping the hint.
_PARSER_HINT_RE = re.compile(r"\.\[([^\]]+)\](\.[^.]+)$")


class ParserRoutingConfigError(ValueError):
    """Raised when LIGHTRAG_PARSER contains an invalid routing rule."""


def normalize_parser_engine(engine: Any) -> str:
    """Normalize engine hints such as mineru-iet to mineru."""
    return str(engine or "").strip().split("-", 1)[0].lower()


# ---------------------------------------------------------------------------
# Per-file processing options (i/t/e/!/F/R/S)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessOptions:
    """Decoded view of a ``process_options`` string.

    The ``raw`` string is preserved verbatim (with duplicates and ordering)
    for storage / audit purposes; boolean flags reflect the deduped logical
    state used by the pipeline.
    """

    raw: str = ""
    images: bool = False
    tables: bool = False
    equations: bool = False
    skip_kg: bool = False
    chunking: Literal["F", "R", "S"] = PROCESS_OPTION_CHUNK_FIXED


_PROCESS_OPTION_DEFAULT = ProcessOptions()


def sanitize_process_options(options: Any) -> str:
    """Strip non-supported characters / hyphen / whitespace from an options string.

    Returns the raw token sequence as-is (no dedup, no reorder) so the
    canonical user intent is preserved on disk.  Invalid characters are
    silently dropped — the caller is expected to have already validated.
    """
    if not options:
        return ""
    return "".join(ch for ch in str(options) if ch in SUPPORTED_PROCESS_OPTIONS)


def validate_process_options(
    options: str, *, label: str = "process options"
) -> list[str]:
    """Return a list of error messages for an options string; empty if valid."""
    errors: list[str] = []
    if not options:
        return errors
    seen_chunkers: list[str] = []
    for ch in options:
        if ch in (" ", "-"):
            continue
        if ch not in SUPPORTED_PROCESS_OPTIONS:
            errors.append(f"{label} contains unsupported character {ch!r}")
            continue
        if ch in PROCESS_OPTION_CHUNK_CHARS and ch not in seen_chunkers:
            seen_chunkers.append(ch)
    if len(seen_chunkers) > 1:
        errors.append(
            f"{label} specifies multiple chunking modes "
            f"({'/'.join(seen_chunkers)}); pick one of "
            f"{PROCESS_OPTION_CHUNK_FIXED}/{PROCESS_OPTION_CHUNK_RECURSIVE}/{PROCESS_OPTION_CHUNK_HEADING}"
        )
    return errors


def parse_process_options(options: Any) -> ProcessOptions:
    """Decode a process-options string into a :class:`ProcessOptions` view."""
    raw = sanitize_process_options(options)
    if not raw:
        return _PROCESS_OPTION_DEFAULT
    chars = set(raw)
    chunking: Literal["F", "R", "S"] = PROCESS_OPTION_CHUNK_FIXED
    # Pick the first chunking selector encountered; validate_process_options
    # already filters duplicates upstream.
    for ch in raw:
        if ch in PROCESS_OPTION_CHUNK_CHARS:
            chunking = ch  # type: ignore[assignment]
            break
    return ProcessOptions(
        raw=raw,
        images=PROCESS_OPTION_IMAGES in chars,
        tables=PROCESS_OPTION_TABLES in chars,
        equations=PROCESS_OPTION_EQUATIONS in chars,
        skip_kg=PROCESS_OPTION_SKIP_KG in chars,
        chunking=chunking,
    )


def split_engine_and_options(bracket_inner: str) -> tuple[str | None, str]:
    """Decompose a bracket-hint inner string into ``(engine, options)``.

    Format rules (see docs/FileProcessingConfiguration-zh.md):
        - ``ENGINE-OPTIONS``: first ``-``-separated segment is the engine
          candidate; the remainder is the options string.
        - ``ENGINE``: matches a supported engine name as a whole.
        - ``OPTIONS``: anything else is treated as options-only.
    """
    inner = (bracket_inner or "").strip()
    if not inner:
        return None, ""

    if "-" in inner:
        head, _, tail = inner.partition("-")
        engine_candidate = normalize_parser_engine(head)
        if engine_candidate in SUPPORTED_PARSER_ENGINES:
            return engine_candidate, tail.strip()
        # Unknown engine before "-"; treat the whole thing as opaque options
        # (likely invalid — caller will validate downstream).
        return None, inner

    engine_candidate = normalize_parser_engine(inner)
    if engine_candidate in SUPPORTED_PARSER_ENGINES:
        return engine_candidate, ""
    return None, inner


def parser_suffix(file_path: str | Path) -> str:
    return Path(file_path).suffix.lower().lstrip(".")


def parser_engine_supports_suffix(engine: str, suffix: str) -> bool:
    return suffix.lower().lstrip(".") in PARSER_ENGINE_SUFFIX_CAPABILITIES.get(
        engine, frozenset()
    )


def parser_engine_endpoint_configured(engine: str) -> bool:
    endpoint_env = _PARSER_ENGINE_ENDPOINT_ENV.get(engine)
    if endpoint_env:
        return bool(os.getenv(endpoint_env, "").strip())
    return True


def _engine_is_usable(
    engine: str,
    suffix: str,
    *,
    require_external_endpoint: bool,
) -> bool:
    if engine not in SUPPORTED_PARSER_ENGINES:
        return False
    if not parser_engine_supports_suffix(engine, suffix):
        return False
    if require_external_endpoint and not parser_engine_endpoint_configured(engine):
        return False
    return True


def _filename_hint_match(
    file_path: str | Path,
) -> tuple[re.Match[str], str, str] | None:
    """Locate a supported ``[hint]`` segment in a basename.

    Returns ``(match, engine_or_empty, options)`` when the bracket inner is a
    recognised hint per the spec; otherwise ``None``.  Both branches require
    the options portion to pass :func:`validate_process_options` —
    engine-qualified hints with bad option chars (e.g. ``[native-FR]``,
    ``[native-Q]``) fail the same way options-only hints do, so the
    documented "invalid characters → whole hint fails → defaults apply"
    contract holds for every hint shape.
    """
    basename = Path(file_path).name
    m = _PARSER_HINT_RE.search(basename)
    if not m:
        return None
    engine, options = split_engine_and_options(m.group(1))
    if options:
        option_errors = validate_process_options(options)
        if option_errors:
            logger.warning(
                f"[parser_routing] ignoring filename hint {m.group(0)!r} in "
                f"{basename!r}: {'; '.join(option_errors)}"
            )
            return None
    if engine in SUPPORTED_PARSER_ENGINES:
        return m, engine, options
    if engine is None and options:
        return m, "", options
    return None


def filename_parser_hint(file_path: str | Path) -> str | None:
    """Return the engine inferred from a filename hint, or ``None``."""
    found = _filename_hint_match(file_path)
    if not found:
        return None
    _, engine, _ = found
    return engine or None


def filename_process_options(file_path: str | Path) -> str:
    """Return the raw process-options string from a filename hint."""
    found = _filename_hint_match(file_path)
    if not found:
        return ""
    return found[2]


def filename_parser_directives(file_path: str | Path) -> tuple[str | None, str]:
    """Return ``(engine, options)`` decoded from a filename hint."""
    found = _filename_hint_match(file_path)
    if not found:
        return None, ""
    _, engine, options = found
    return (engine or None), options


def canonicalize_parser_hinted_basename(file_path: str | Path) -> str:
    """Return basename with a supported parser hint removed.

    Only the final ``.[engine].ext`` (or ``.[engine-options].ext`` /
    ``.[options].ext``) segment is stripped, exactly once, and only when the
    bracket content is a recognised hint.  Nested hints such as
    ``name.[native].[mineru].pdf`` therefore become ``name.[native].pdf`` —
    additional outer hints are not unwrapped.
    """
    basename = Path(file_path).name
    found = _filename_hint_match(file_path)
    if not found:
        return basename
    m, _, _ = found
    return f"{basename[: m.start()]}{m.group(2)}"


def parser_rules_from_env() -> str:
    return os.getenv("LIGHTRAG_PARSER", "").strip()


def _iter_parser_rule_items(rules: str) -> list[tuple[int, str]]:
    return [
        (index, item.strip())
        for index, item in enumerate(_PARSER_RULE_SPLIT_RE.split(rules), start=1)
        if item.strip()
    ]


def _rule_pattern_matches_engine_capability(pattern: str, engine: str) -> bool:
    supported_suffixes = PARSER_ENGINE_SUFFIX_CAPABILITIES.get(engine, frozenset())
    return any(fnmatch.fnmatch(suffix, pattern) for suffix in supported_suffixes)


def _rule_engine_and_options(engine_hint: str) -> tuple[str, str]:
    """Split a ``LIGHTRAG_PARSER`` rule's RHS (``engine[-options]``).

    Returns ``(normalized_engine, options_str)``.  Unlike the filename hint
    splitter this always treats the first ``-`` as the engine/options
    boundary, since ``LIGHTRAG_PARSER`` rules cannot be options-only.
    """
    head, _, tail = engine_hint.partition("-")
    return normalize_parser_engine(head), tail.strip()


def validate_parser_routing_config(parser_rules: str | None = None) -> None:
    """Validate LIGHTRAG_PARSER syntax and required external parser endpoints."""
    rules = parser_rules_from_env() if parser_rules is None else parser_rules.strip()
    if not rules:
        return

    errors: list[str] = []
    for index, item in _iter_parser_rule_items(rules):
        label = f"rule {index} ({item!r})"
        if ":" not in item:
            errors.append(f"{label} must use '<suffix-pattern>:<engine>'")
            continue

        pattern, engine_hint = item.split(":", 1)
        pattern = pattern.strip().lower()
        engine_hint = engine_hint.strip()
        engine, options_str = _rule_engine_and_options(engine_hint)

        if not pattern:
            errors.append(f"{label} has an empty suffix pattern")
            continue
        if "." in pattern:
            errors.append(
                f"{label} matches suffixes without dots; use 'pdf', not '*.pdf'"
            )
            continue
        if not engine_hint:
            errors.append(f"{label} has an empty parser engine")
            continue
        if engine not in SUPPORTED_PARSER_ENGINES:
            supported = ", ".join(sorted(SUPPORTED_PARSER_ENGINES))
            errors.append(
                f"{label} uses unsupported parser engine {engine_hint!r}; "
                f"supported engines: {supported}"
            )
            continue
        if not _rule_pattern_matches_engine_capability(pattern, engine):
            supported_suffixes = ", ".join(
                sorted(PARSER_ENGINE_SUFFIX_CAPABILITIES.get(engine, frozenset()))
            )
            errors.append(
                f"{label} does not match any suffix supported by {engine}; "
                f"supported suffixes: {supported_suffixes}"
            )
        endpoint_env = _PARSER_ENGINE_ENDPOINT_ENV.get(engine)
        if endpoint_env and not parser_engine_endpoint_configured(engine):
            errors.append(f"{label} requires {endpoint_env} to be configured")
        if options_str:
            errors.extend(
                f"{label}: {msg}"
                for msg in validate_process_options(
                    options_str, label="process options"
                )
            )

    if errors:
        raise ParserRoutingConfigError(
            "Invalid LIGHTRAG_PARSER configuration: " + "; ".join(errors)
        )


def _matching_rule_directives(
    file_path: str | Path,
    *,
    parser_rules: str | None,
    require_external_endpoint: bool,
) -> tuple[str | None, str]:
    """Find the first matching ``LIGHTRAG_PARSER`` rule for ``file_path``.

    Returns ``(engine, options_str)`` where ``engine`` is ``None`` when no
    usable rule is found.  ``options_str`` is empty when a rule matched but
    has no ``-options`` suffix.
    """
    suffix = parser_suffix(file_path)
    rules = parser_rules_from_env() if parser_rules is None else parser_rules.strip()
    if not rules:
        return None, ""
    for _, item in _iter_parser_rule_items(rules):
        if ":" not in item:
            continue
        pattern, engine_hint = item.split(":", 1)
        pattern = pattern.strip().lower()
        engine, options_str = _rule_engine_and_options(engine_hint.strip())
        if not fnmatch.fnmatch(suffix, pattern):
            continue
        if _engine_is_usable(
            engine,
            suffix,
            require_external_endpoint=require_external_endpoint,
        ):
            return engine, options_str
    return None, ""


def resolve_file_parser_engine(
    file_path: str | Path,
    *,
    parser_rules: str | None = None,
    require_external_endpoint: bool = True,
) -> str:
    """Resolve the extraction engine for a source file before content extraction."""
    engine, _ = resolve_file_parser_directives(
        file_path,
        parser_rules=parser_rules,
        require_external_endpoint=require_external_endpoint,
    )
    return engine


def resolve_file_parser_directives(
    file_path: str | Path,
    *,
    parser_rules: str | None = None,
    require_external_endpoint: bool = True,
) -> tuple[str, str]:
    """Resolve ``(engine, process_options)`` for a source file before extraction.

    Resolution order (mirrors :func:`resolve_file_parser_engine`):
        1. Filename ``[hint]`` — engine and / or options take precedence.
        2. ``LIGHTRAG_PARSER`` rules — first matching rule provides defaults
           for whichever of engine / options the filename hint did not
           specify.
        3. Default engine ``legacy`` with empty options.
    """
    suffix = parser_suffix(file_path)

    hinted_engine, hinted_options = filename_parser_directives(file_path)
    if hinted_engine and not _engine_is_usable(
        hinted_engine, suffix, require_external_endpoint=require_external_endpoint
    ):
        # Hinted engine cannot handle this file (e.g. wrong suffix or missing
        # endpoint); fall back to rule-based resolution but keep the hinted
        # options if any.
        hinted_engine = None

    rule_engine, rule_options = _matching_rule_directives(
        file_path,
        parser_rules=parser_rules,
        require_external_endpoint=require_external_endpoint,
    )

    engine = hinted_engine or rule_engine or PARSER_ENGINE_LEGACY
    options_str = hinted_options or rule_options
    return engine, sanitize_process_options(options_str)


def resolve_stored_document_parser_engine(
    file_path: str | Path,
    content_data: dict[str, Any] | None,
) -> str:
    """Resolve parser engine for a full_docs row during pipeline processing."""
    if content_data:
        doc_format = content_data.get("format", FULL_DOCS_FORMAT_RAW)
        if doc_format == FULL_DOCS_FORMAT_LIGHTRAG and content_data.get(
            "lightrag_document_path"
        ):
            return PARSER_ENGINE_NATIVE
        if doc_format != FULL_DOCS_FORMAT_PENDING_PARSE:
            return PARSER_ENGINE_LEGACY

        suffix = parser_suffix(file_path)
        pending_engine = normalize_parser_engine(content_data.get("parse_engine"))
        if pending_engine in SUPPORTED_PARSER_ENGINES and parser_engine_supports_suffix(
            pending_engine, suffix
        ):
            return pending_engine

    return resolve_file_parser_engine(file_path)
