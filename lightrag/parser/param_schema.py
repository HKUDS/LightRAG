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

Phase 1 scope (intentionally minimal — the foundation supports more):
only ``chunk_token_size`` (alias ``chunk_ts``) and
``chunk_overlap_token_size`` (alias ``chunk_ol``), both integers, flowing
through the existing per-document ``chunk_options`` channel.  Engine-level
parameters and flag/enum/float parameters are **not** accepted yet and are
rejected with a friendly error; adding them later is a matter of registering
new :class:`ParamSpec` entries.

This module is import-cheap and has no dependency on
:mod:`lightrag.parser.routing` so it can be reused without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
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

    ``kind`` is ``"int"`` for every Phase 1 parameter; the field exists so
    future types (``float``/``enum``/``flag``/``range_list``) slot in without
    a structural change.  ``targets`` is the set of chunk selector chars the
    parameter is valid for (e.g. overlap is invalid for ``V``).
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
)

_CHUNK_PARAM_BY_NAME: dict[str, ParamSpec] = {}
for _spec in _CHUNK_PARAM_SPECS:
    for _name in _spec.names:
        _CHUNK_PARAM_BY_NAME[_name] = _spec


def supported_chunk_param_names() -> str:
    """Comma-joined canonical names, for friendly error messages."""
    return ", ".join(sorted(spec.canonical for spec in _CHUNK_PARAM_SPECS))


def engine_params_supported(engine: str) -> bool:  # noqa: ARG001
    """Whether any engine accepts hint parameters yet (Phase 1: none do)."""
    return False


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
    """
    result: dict[str, Any] = {}
    errors: list[str] = []
    seen: set[str] = set()

    for raw in split_top_level(text, ","):
        segment = raw.strip()
        if not segment:
            errors.append(f"{label}: empty parameter")
            continue
        if "=" not in segment:
            errors.append(
                f"{label}: parameter {segment!r} must be written as 'key=value' "
                "(flag parameters are not supported yet)"
            )
            continue
        key, _, value = segment.partition("=")
        key = key.strip()
        value = value.strip()

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
        else:  # pragma: no cover - no non-int kinds registered in Phase 1
            errors.append(
                f"{label}: parameter {spec.canonical!r} has unsupported type "
                f"{spec.kind!r}"
            )

    return result, errors
