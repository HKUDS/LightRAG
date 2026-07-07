"""Parameter schema + parenthesis-aware scanners for hint parameters.

This module is the single source of truth for the *parameters* that may be
attached to chunk-strategy selectors inside a parser hint or a
``LIGHTRAG_PARSER`` rule, e.g. ``[-R(chunk_ts=800,chunk_ol=80)]`` or
``pdf:legacy-R(chunk_ts=800)``.

Design (see ``docs/FileProcessingPipeline.md``):

* Inside ``(...)`` a comma is **only** a parameter separator; a single
  parameter value never contains ``,`` ``(`` ``)`` or ``]``.
* Parameter names use a readable ``canonical`` form with optional short
  ``alias`` forms; everything is normalised to canonical before use, so the
  internal structures never carry an alias.
* Parameters are declared per *target* (a chunk selector char ``F``/``R``/``V``/
  ``P``) with a type and the set of selectors they apply to.

Chunk-parameter scope: the integer ``chunk_token_size`` (alias ``chunk_ts``)
and ``chunk_overlap_token_size`` (alias ``chunk_ol``), plus the boolean
``drop_references`` (alias ``drop_rf``, paragraph-semantic only), all flowing
through the existing per-document ``chunk_options`` channel.  Float/enum chunk
parameters are still rejected with a friendly error; adding them is a matter of
registering new :class:`ParamSpec` entries and an extra ``kind`` branch in
:func:`parse_chunk_params`.

This module is import-cheap and has no dependency on
:mod:`lightrag.parser.routing` so it can be reused without import cycles.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_PARAGRAH,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_CHUNK_VECTOR,
)

# Characters a parameter value may never contain.  ``]`` can never appear
# anyway (``_PARSER_HINT_RE`` terminates the bracket on ``]``) but is rejected
# explicitly so the error is friendly rather than a silent truncation.
_VALUE_FORBIDDEN = frozenset(",()]")


# ---------------------------------------------------------------------------
# Parenthesis-aware scanners (shared by routing's rule/engine/options splits)
# ---------------------------------------------------------------------------


def split_top_level(text: str, separators: str) -> list[str]:
    """Split ``text`` on any char in ``separators`` at parenthesis depth 0.

    Characters inside ``(...)`` are protected, so a comma used to separate
    parameters never splits the surrounding rule / options string.  Always
    returns at least one element (the whole string when no separator is hit).
    """
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
        elif depth == 0 and ch in separators:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def take_paren_block(text: str, i: int) -> tuple[str | None, int]:
    """Read a balanced ``(...)`` block starting at ``text[i]``.

    Returns ``(inner, index_just_after_closing_paren)`` when ``text[i]`` opens
    a balanced block, otherwise ``(None, i)`` — used for both "no block here"
    (``text[i] != '('``) and "unterminated block".  Callers that need to flag
    an unterminated block compare the returned index / ``None`` accordingly.
    """
    if i >= len(text) or text[i] != "(":
        return None, i
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
    return None, i  # unterminated


# ---------------------------------------------------------------------------
# Parameter schema registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Declares one tunable hint parameter for a set of chunk selectors.

    ``kind`` is ``"int"`` or ``"bool"`` today; the field exists so future
    types (``float``/``enum``/``range_list``) slot in without a structural
    change.  ``targets`` is the set of chunk selector chars the parameter is
    valid for (e.g. overlap is invalid for ``V``, ``drop_references`` only
    applies to ``P``).
    """

    canonical: str
    aliases: frozenset[str]
    kind: str
    targets: frozenset[str]
    is_list: bool = False
    min_value: int | None = None
    names: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", frozenset({self.canonical}) | self.aliases)


_ALL_CHUNK_SELECTORS = frozenset(
    {
        PROCESS_OPTION_CHUNK_FIXED,
        PROCESS_OPTION_CHUNK_RECURSIVE,
        PROCESS_OPTION_CHUNK_VECTOR,
        PROCESS_OPTION_CHUNK_PARAGRAH,
    }
)

# Phase 1 registry — keep this list small; expand it (and only it) to grow
# hint-parameter coverage.
_CHUNK_PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(
        canonical="chunk_token_size",
        aliases=frozenset({"chunk_ts"}),
        kind="int",
        targets=_ALL_CHUNK_SELECTORS,
        min_value=1,
    ),
    ParamSpec(
        canonical="chunk_overlap_token_size",
        aliases=frozenset({"chunk_ol"}),
        kind="int",
        # Semantic-vector (V) chunking has no overlap concept.
        targets=frozenset(
            {
                PROCESS_OPTION_CHUNK_FIXED,
                PROCESS_OPTION_CHUNK_RECURSIVE,
                PROCESS_OPTION_CHUNK_PARAGRAH,
            }
        ),
        min_value=0,
    ),
    # Paragraph-semantic only: drop the trailing reference section before
    # chunking.  Detection-tuning knobs (tail window / heading prefixes) are
    # env-only and read live by the chunker, so only the switch is a hint param.
    ParamSpec(
        canonical="drop_references",
        aliases=frozenset({"drop_rf"}),
        kind="bool",
        targets=frozenset({PROCESS_OPTION_CHUNK_PARAGRAH}),
    ),
)

_CHUNK_PARAM_BY_NAME: dict[str, ParamSpec] = {}
for _spec in _CHUNK_PARAM_SPECS:
    for _name in _spec.names:
        _CHUNK_PARAM_BY_NAME[_name] = _spec


def supported_chunk_param_names() -> str:
    """Comma-joined canonical names, for friendly error messages."""
    return ", ".join(sorted(spec.canonical for spec in _CHUNK_PARAM_SPECS))


def _parse_int(value: str) -> int | None:
    try:
        return int(value, 10)
    except (TypeError, ValueError):
        return None


def parse_chunk_params(
    text: str, *, selector: str, label: str
) -> tuple[dict[str, Any], list[str]]:
    """Parse one chunk-strategy parameter block into a canonical dict.

    ``text`` is the raw text inside ``(...)`` (parameter separators only —
    no surrounding parens).  ``selector`` is the chunk char the block is
    attached to (``F``/``R``/``V``/``P``).  Returns ``(canonical_dict,
    errors)``; ``errors`` is empty iff the block is fully valid.  Aliases are
    normalised to their canonical name in the returned dict.

    A boolean parameter may be written bare as a flag — ``P(drop_rf)`` is
    shorthand for ``P(drop_rf=true)``.  Non-boolean parameters still require
    the explicit ``key=value`` form.
    """
    result: dict[str, Any] = {}
    errors: list[str] = []
    seen: set[str] = set()

    for raw in split_top_level(text, ","):
        segment = raw.strip()
        if not segment:
            errors.append(f"{label}: empty parameter")
            continue
        if "=" in segment:
            key, _, value = segment.partition("=")
            key = key.strip()
            value = value.strip()
            flag_form = False
        else:
            # Bare flag form, e.g. ``drop_rf``.  Only valid for boolean
            # parameters, where it is shorthand for ``drop_rf=true``.
            key = segment
            value = ""
            flag_form = True

        spec = _CHUNK_PARAM_BY_NAME.get(key)
        if spec is None:
            errors.append(
                f"{label}: unknown parameter {key!r}; supported parameters: "
                f"{supported_chunk_param_names()}"
            )
            continue
        if selector not in spec.targets:
            errors.append(
                f"{label}: parameter {spec.canonical!r} is not supported for "
                f"chunk strategy {selector!r}"
            )
            continue
        if flag_form:
            if spec.kind != "bool":
                errors.append(
                    f"{label}: parameter {spec.canonical!r} must be written as "
                    "'key=value'; only boolean flags may be written bare"
                )
                continue
            value = "true"  # bare boolean flag means True
        if any(ch in _VALUE_FORBIDDEN for ch in value):
            errors.append(
                f"{label}: value for {spec.canonical!r} may not contain any of "
                "',' '(' ')' ']'"
            )
            continue
        if spec.canonical in seen and not spec.is_list:
            errors.append(f"{label}: parameter {spec.canonical!r} may not be repeated")
            continue
        seen.add(spec.canonical)

        if spec.kind == "int":
            parsed = _parse_int(value)
            if parsed is None:
                errors.append(
                    f"{label}: value for {spec.canonical!r} must be an integer, "
                    f"got {value!r}"
                )
                continue
            if spec.min_value is not None and parsed < spec.min_value:
                errors.append(
                    f"{label}: value for {spec.canonical!r} must be >= "
                    f"{spec.min_value}, got {parsed}"
                )
                continue
            result[spec.canonical] = parsed
        elif spec.kind == "bool":
            parsed_bool = _parse_bool(value)
            if parsed_bool is None:
                errors.append(
                    f"{label}: value for {spec.canonical!r} must be a boolean "
                    f"(true/false), got {value!r}"
                )
                continue
            result[spec.canonical] = parsed_bool
        else:  # pragma: no cover - only int/bool kinds registered today
            errors.append(
                f"{label}: parameter {spec.canonical!r} has unsupported type "
                f"{spec.kind!r}"
            )

    # Cross-field invariant: reject an explicit overlap >= size pair here so
    # every caller (rule startup validation AND filename-hint validation)
    # rejects it uniformly, instead of only failing later at enqueue.
    overlap_error = chunk_param_overlap_error(result)
    if overlap_error is not None:
        errors.append(f"{label}: {overlap_error}")

    return result, errors


def chunk_param_overlap_error(params: Mapping[str, Any]) -> str | None:
    """Return an error string when a block sets an invalid overlap/size pair.

    Only checks the cross-field invariant when **both** ``chunk_token_size`` and
    ``chunk_overlap_token_size`` are explicitly present in ``params`` (a parsed
    canonical dict).  When only one is given the effective value depends on
    env / ``addon_params`` and cannot be evaluated here — that case is left to
    the upload-time ``_validate_effective_chunk_overlap`` on the resolved
    snapshot.  Returns ``None`` when the pair is valid or not fully specified.
    """
    size = params.get("chunk_token_size")
    overlap = params.get("chunk_overlap_token_size")
    if size is not None and overlap is not None and overlap >= size:
        return (
            f"chunk_overlap_token_size ({overlap}) must be < chunk_token_size ({size})"
        )
    return None


# ---------------------------------------------------------------------------
# Engine parameters (Phase 2) — per-file params attached to the engine token,
# e.g. ``mineru(page_range=1-3,language=en)`` / ``docling(force_ocr=true)``.
# Keyed by engine name (unlike chunk params, which are keyed by F/R/V/P).
# ---------------------------------------------------------------------------

_BOOL_TRUE = frozenset({"1", "true", "yes", "on", "t", "y"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off", "f", "n"})
# A single page-range segment: ``N`` or ``N-M`` (validated for positivity /
# ordering separately).
_PAGE_SEGMENT_RE = re.compile(r"^\d+(?:-\d+)?$")


@dataclass(frozen=True)
class EngineParamSpec:
    """Declares one tunable engine parameter (Phase 2).

    Separate from the chunk :class:`ParamSpec` (which requires a ``targets``
    set of F/R/V/P selectors that is meaningless for engines).  ``kind`` is one
    of ``"str"`` / ``"enum"`` / ``"bool"``.  ``is_list`` marks a repeated-key
    parameter (``page_range``) whose canonical value is a comma-joined string.
    ``enum_values`` constrains an ``"enum"`` parameter.
    """

    canonical: str
    aliases: frozenset[str]
    kind: str
    is_list: bool = False
    enum_values: frozenset[str] | None = None
    names: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", frozenset({self.canonical}) | self.aliases)


# Phase 2 registry — keyed by engine name; expand by adding specs (and, for a
# new engine, a new dict entry).  An engine absent here accepts NO parameters.
_ENGINE_PARAM_SPECS: dict[str, tuple[EngineParamSpec, ...]] = {
    PARSER_ENGINE_MINERU: (
        EngineParamSpec(
            canonical="page_range",
            aliases=frozenset({"pr"}),
            kind="str",
            is_list=True,
        ),
        EngineParamSpec(canonical="language", aliases=frozenset(), kind="str"),
        EngineParamSpec(
            canonical="local_parse_method",
            aliases=frozenset({"local_pm"}),
            kind="enum",
            enum_values=frozenset({"auto", "txt", "ocr"}),
        ),
    ),
    PARSER_ENGINE_DOCLING: (
        EngineParamSpec(canonical="force_ocr", aliases=frozenset({"ocr"}), kind="bool"),
    ),
    PARSER_ENGINE_NATIVE: (
        # Opt-in smart heading discovery for docx (LLM-assisted; the markdown
        # path warns and ignores it — font-size signals don't exist in md).
        EngineParamSpec(canonical="smart_heading", aliases=frozenset(), kind="bool"),
    ),
}

_ENGINE_PARAM_BY_NAME: dict[str, dict[str, EngineParamSpec]] = {
    engine: {name: spec for spec in specs for name in spec.names}
    for engine, specs in _ENGINE_PARAM_SPECS.items()
}


def engine_params_supported(engine: str) -> bool:
    """Whether ``engine`` declares any tunable hint parameters."""
    return engine in _ENGINE_PARAM_SPECS


def supported_engine_param_names(engine: str) -> str:
    """Comma-joined canonical engine-param names, for error messages."""
    specs = _ENGINE_PARAM_SPECS.get(engine, ())
    return ", ".join(sorted(spec.canonical for spec in specs))


def _parse_bool(value: str) -> bool | None:
    low = value.strip().lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    return None


def _mineru_api_mode_is_local() -> bool:
    """True when MinerU runs in local mode (the default when unset).

    Read here (rather than importing ``mineru.cache``) to keep this module a
    dependency-free leaf.  Mirrors ``cache._normalize_api_mode``: anything that
    is not ``official`` is treated as local.
    """
    return (os.getenv("MINERU_API_MODE", "") or "").strip().lower() != "official"


def _validate_page_range_segments(parts: list[str]) -> list[str]:
    """Validate page-range segment shape + the MinerU local single-segment rule.

    ``official`` mode forwards a multi-segment list verbatim; ``local`` accepts
    only a single page / range (mirrors ``cache.local_page_bounds``).  The
    download-time ``local_page_bounds`` remains the final backstop.
    """
    errors: list[str] = []
    for seg in parts:
        if not _PAGE_SEGMENT_RE.match(seg):
            errors.append(
                f"page_range segment {seg!r} must be a page 'N' or range 'N-M'"
            )
            continue
        if "-" in seg:
            left, _, right = seg.partition("-")
            if int(left) < 1 or int(right) < 1:
                errors.append(f"page_range segment {seg!r} must use positive pages")
            elif int(right) < int(left):
                errors.append(f"page_range segment {seg!r} must have end >= start")
        elif int(seg) < 1:
            errors.append(f"page_range segment {seg!r} must be a positive page")
    if not errors and len(parts) > 1 and _mineru_api_mode_is_local():
        errors.append(
            "page_range with MINERU_API_MODE=local supports only a single page "
            "or range such as '1-10'; use MINERU_API_MODE=official for a "
            "multi-segment list"
        )
    return errors


def _coerce_engine_value(
    spec: EngineParamSpec, value: str, *, label: str
) -> tuple[Any, str | None]:
    """Validate + coerce a single engine-param value to its canonical type.

    Returns ``(coerced, error)``; ``error`` is ``None`` when valid.  Shared by
    the text path (:func:`parse_engine_params`) and the resolved-dict path
    (:func:`normalize_engine_params`) so both apply identical rules.  For the
    list-type ``page_range`` the ``value`` is the already-joined comma string.
    """
    # ``local_parse_method`` only feeds the local MinerU request + signature;
    # the official API neither sends it nor folds it into the cache key, so
    # accepting it under official mode would persist a directive that silently
    # does nothing. Reject it here (mirrors the page_range mode rule).
    if spec.canonical == "local_parse_method" and not _mineru_api_mode_is_local():
        return None, (
            f"{label}: 'local_parse_method' only applies to "
            "MINERU_API_MODE=local (the default); the official API ignores it"
        )
    if spec.kind == "bool":
        parsed = _parse_bool(value)
        if parsed is None:
            return None, (
                f"{label}: value for {spec.canonical!r} must be a boolean "
                f"(true/false), got {value!r}"
            )
        return parsed, None
    if spec.kind == "enum":
        if spec.enum_values is not None and value not in spec.enum_values:
            allowed = ", ".join(sorted(spec.enum_values))
            return None, (
                f"{label}: value for {spec.canonical!r} must be one of "
                f"{allowed}, got {value!r}"
            )
        return value, None
    # str (incl. the list-type page_range, whose value is a comma-joined string)
    if not value:
        return None, f"{label}: value for {spec.canonical!r} must be non-empty"
    if spec.is_list and spec.canonical == "page_range":
        segments = [p.strip() for p in value.split(",")]
        seg_errors = _validate_page_range_segments(segments)
        if seg_errors:
            return None, f"{label}: " + "; ".join(seg_errors)
        return ",".join(segments), None
    return value, None


def parse_engine_params(
    text: str, *, engine: str, label: str
) -> tuple[dict[str, Any], list[str]]:
    """Parse one engine parameter block into a canonical, coerced dict.

    ``text`` is the raw text inside ``(...)`` for an engine token; ``engine`` is
    the (bare) engine the block is attached to.  Returns ``(canonical_dict,
    errors)``; aliases are normalised to canonical and values are coerced to
    their declared type (so ``force_ocr`` is a real ``bool``).  A list-type
    ``page_range`` collects repeated keys and joins them with ``,``.
    """
    by_name = _ENGINE_PARAM_BY_NAME.get(engine)
    if by_name is None:
        if text.strip():
            return {}, [f"{label}: parser engine {engine!r} does not accept parameters"]
        return {}, []

    result: dict[str, Any] = {}
    list_values: dict[str, list[str]] = {}
    errors: list[str] = []
    seen: set[str] = set()

    for raw in split_top_level(text, ","):
        segment = raw.strip()
        if not segment:
            errors.append(f"{label}: empty parameter")
            continue
        if "=" not in segment:
            if _PAGE_SEGMENT_RE.match(segment) and (
                "page_range" in by_name or "pr" in by_name
            ):
                errors.append(
                    f"{label}: page lists must repeat the key, e.g. "
                    "'page_range=1-3,page_range=5' (a comma only separates "
                    "parameters)"
                )
            else:
                errors.append(
                    f"{label}: parameter {segment!r} must be written as "
                    "'key=value' (flag parameters are not supported yet)"
                )
            continue
        key, _, value = segment.partition("=")
        key = key.strip()
        value = value.strip()

        spec = by_name.get(key)
        if spec is None:
            errors.append(
                f"{label}: unknown parameter {key!r} for engine {engine!r}; "
                f"supported parameters: {supported_engine_param_names(engine)}"
            )
            continue
        if any(ch in _VALUE_FORBIDDEN for ch in value):
            errors.append(
                f"{label}: value for {spec.canonical!r} may not contain any of "
                "',' '(' ')' ']'"
            )
            continue
        if spec.is_list:
            list_values.setdefault(spec.canonical, []).append(value)
            continue
        if spec.canonical in seen:
            errors.append(f"{label}: parameter {spec.canonical!r} may not be repeated")
            continue
        seen.add(spec.canonical)
        coerced, error = _coerce_engine_value(spec, value, label=label)
        if error is not None:
            errors.append(error)
            continue
        result[spec.canonical] = coerced

    # Join repeated-key list params, then coerce/validate the joined value.
    for canonical, values in list_values.items():
        spec = by_name[canonical]
        coerced, error = _coerce_engine_value(spec, ",".join(values), label=label)
        if error is not None:
            errors.append(error)
            continue
        result[canonical] = coerced

    return result, errors


def normalize_engine_params(
    engine: str, params: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Normalise + validate an already-resolved engine-param dict.

    Used by the pipeline layer where direct (SDK/API) callers bypass routing's
    text parsing.  Returns a **coerced** dict (e.g. ``force_ocr`` becomes a real
    ``bool``, ``page_range`` a validated comma-joined string) so what gets
    persisted is exactly what the engine override seam consumes.  Accepts a
    ``page_range`` value as either a list or a comma-joined string.
    """
    if not params:
        return {}, []
    by_name = _ENGINE_PARAM_BY_NAME.get(engine)
    if by_name is None:
        return {}, [f"parser engine {engine!r} does not accept parameters"]

    result: dict[str, Any] = {}
    errors: list[str] = []
    for key, value in params.items():
        spec = by_name.get(str(key))
        if spec is None:
            errors.append(
                f"unknown parameter {key!r} for engine {engine!r}; supported "
                f"parameters: {supported_engine_param_names(engine)}"
            )
            continue
        if spec.is_list and isinstance(value, (list, tuple)):
            value = ",".join(str(v).strip() for v in value)
        else:
            value = str(value).strip()
        coerced, error = _coerce_engine_value(
            spec, value, label=f"engine parameter {spec.canonical!r}"
        )
        if error is not None:
            errors.append(error)
            continue
        result[spec.canonical] = coerced
    return result, errors


def render_engine_params(
    engine: str, params: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """Render a resolved engine-param dict to the canonical inner text.

    Returns ``(inner_text, errors)`` where ``inner_text`` is the
    ``key=value,...`` string that goes inside the ``parse_engine`` parens (e.g.
    ``page_range=1-3,page_range=5,language=en``).  Normalises first (so the
    output always round-trips through :func:`parse_engine_params`); a list-type
    value is emitted as **repeated keys**, a bool as ``true``/``false``.  Keys
    are sorted for deterministic output.
    """
    normalized, errors = normalize_engine_params(engine, params)
    if errors:
        return "", errors
    by_name = _ENGINE_PARAM_BY_NAME.get(engine, {})
    parts: list[str] = []
    for canonical in sorted(normalized):
        spec = by_name[canonical]
        value = normalized[canonical]
        if spec.is_list:
            parts.extend(f"{canonical}={seg}" for seg in str(value).split(","))
        elif spec.kind == "bool":
            parts.append(f"{canonical}={'true' if value else 'false'}")
        else:
            parts.append(f"{canonical}={value}")
    return ",".join(parts), []
