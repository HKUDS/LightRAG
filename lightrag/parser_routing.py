from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
    PARSER_ENGINE_SUFFIX_CAPABILITIES,
    SUPPORTED_PARSER_ENGINES,
)

_PARSER_RULE_SPLIT_RE = re.compile(r"[;,]")
_PARSER_ENGINE_ENDPOINT_ENV = {
    PARSER_ENGINE_MINERU: "MINERU_ENDPOINT",
    PARSER_ENGINE_DOCLING: "DOCLING_ENDPOINT",
}


class ParserRoutingConfigError(ValueError):
    """Raised when LIGHTRAG_PARSER contains an invalid routing rule."""


def normalize_parser_engine(engine: Any) -> str:
    """Normalize engine hints such as mineru-iet to mineru."""
    return str(engine or "").strip().split("-", 1)[0].lower()


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


def filename_parser_hint(file_path: str | Path) -> str | None:
    m = re.search(r"\.\[([^\]]+)\]\.[^.]+$", Path(file_path).name)
    if not m:
        return None
    engine = normalize_parser_engine(m.group(1))
    return engine if engine in SUPPORTED_PARSER_ENGINES else None


def canonicalize_parser_hinted_basename(file_path: str | Path) -> str:
    """Return basename with a supported parser hint removed.

    Only the final ``.[engine].ext`` segment is stripped, exactly once, and
    only when the bracketed value normalizes to a supported parser engine.
    Nested hints such as ``name.[native].[mineru].pdf`` therefore become
    ``name.[native].pdf`` — additional outer hints are not unwrapped.
    """
    basename = Path(file_path).name
    m = re.search(r"\.\[([^\]]+)\](\.[^.]+)$", basename)
    if not m:
        return basename
    engine = normalize_parser_engine(m.group(1))
    if engine not in SUPPORTED_PARSER_ENGINES:
        return basename
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
        engine = normalize_parser_engine(engine_hint)

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

    if errors:
        raise ParserRoutingConfigError(
            "Invalid LIGHTRAG_PARSER configuration: " + "; ".join(errors)
        )


def resolve_file_parser_engine(
    file_path: str | Path,
    *,
    parser_rules: str | None = None,
    require_external_endpoint: bool = True,
) -> str:
    """Resolve the extraction engine for a source file before content extraction."""
    suffix = parser_suffix(file_path)

    hint = filename_parser_hint(file_path)
    if hint and _engine_is_usable(
        hint, suffix, require_external_endpoint=require_external_endpoint
    ):
        return hint

    rules = parser_rules_from_env() if parser_rules is None else parser_rules.strip()
    if rules:
        for _, item in _iter_parser_rule_items(rules):
            if ":" not in item:
                continue
            pattern, engine_hint = item.split(":", 1)
            pattern = pattern.strip().lower()
            engine = normalize_parser_engine(engine_hint)
            if not fnmatch.fnmatch(suffix, pattern):
                continue
            if _engine_is_usable(
                engine,
                suffix,
                require_external_endpoint=require_external_endpoint,
            ):
                return engine

    return PARSER_ENGINE_LEGACY


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
        pending_engine = normalize_parser_engine(content_data.get("parsed_engine"))
        if pending_engine in SUPPORTED_PARSER_ENGINES and parser_engine_supports_suffix(
            pending_engine, suffix
        ):
            return pending_engine

    return resolve_file_parser_engine(file_path)
