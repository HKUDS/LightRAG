from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightrag.constants import (
    DEFAULT_R_SEPARATORS,
    DEFAULT_SENTENCE_SPLIT_REGEX,
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
    PROCESS_OPTION_CHUNK_VECTOR,
    PROCESS_OPTION_CHUNK_PARAGRAH,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_EQUATIONS,
    PROCESS_OPTION_IMAGES,
    PROCESS_OPTION_SKIP_KG,
    PROCESS_OPTION_TABLES,
    ProcessChunkingOption,
    SUPPORTED_PARSER_ENGINES,
    SUPPORTED_PROCESS_OPTIONS,
)
from lightrag.utils import logger, parse_optional_float

import json
from collections.abc import Mapping
from copy import deepcopy

_PARSER_RULE_SPLIT_RE = re.compile(r"[;,]")
_PARSER_ENGINE_ENDPOINT_ENV = {
    PARSER_ENGINE_DOCLING: "DOCLING_ENDPOINT",
}
_VALID_MINERU_API_MODES = {"official", "local"}

# Trailing parser-hint pattern: matches ``.[engine].ext`` at end of basename.
# Group 1 captures the raw engine token (still needs normalize_parser_engine
# and SUPPORTED_PARSER_ENGINES validation); group 2 captures ``.ext`` so it
# can be reattached when stripping the hint.
_PARSER_HINT_RE = re.compile(r"\.\[([^\]]*)\](\.[^.]+)$")


class ParserRoutingConfigError(ValueError):
    """Raised when LIGHTRAG_PARSER contains an invalid routing rule."""


class FilenameParserHintError(ValueError):
    """Raised when a filename parser hint is invalid for ingestion."""


def normalize_parser_engine(engine: Any) -> str:
    """Normalize engine hints such as mineru-iet to mineru."""
    return str(engine or "").strip().split("-", 1)[0].lower()


# ---------------------------------------------------------------------------
# Per-file processing options (i/t/e/!/F/R/V/P)
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
    chunking: ProcessChunkingOption = PROCESS_OPTION_CHUNK_FIXED

    @property
    def chunking_explicit(self) -> bool:
        """True iff ``raw`` actually contains a chunking selector char.

        Distinguishes "user explicitly opted into a chunking strategy"
        from "no chunking selector supplied — pipeline used the default".
        ``chunking`` itself is unreliable for this question because it
        falls back to :data:`PROCESS_OPTION_CHUNK_FIXED` in both cases.
        Used by ``process_single_document`` to decide whether to
        dispatch via the new file-chunker contract or to honor the
        legacy externally-supplied :attr:`LightRAG.chunking_func`.
        """
        return any(c in PROCESS_OPTION_CHUNK_CHARS for c in self.raw)


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
            f"{PROCESS_OPTION_CHUNK_FIXED}/{PROCESS_OPTION_CHUNK_RECURSIVE}/{PROCESS_OPTION_CHUNK_VECTOR}/{PROCESS_OPTION_CHUNK_PARAGRAH}"
        )
    return errors


def parse_process_options(options: Any) -> ProcessOptions:
    """Decode a process-options string into a :class:`ProcessOptions` view."""
    raw = sanitize_process_options(options)
    if not raw:
        return _PROCESS_OPTION_DEFAULT
    chars = set(raw)
    chunking: ProcessChunkingOption = PROCESS_OPTION_CHUNK_FIXED
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


# ---------------------------------------------------------------------------
# Per-chunker parameter snapshot (chunk_options) — counterpart to the
# F/R/V/P selector in ``ProcessOptions``.  ``process_options`` chooses
# the strategy; ``chunk_options`` carries the parameters the chosen
# strategy reads.
#
# Storage shape: the per-document snapshot persisted to
# ``full_docs[doc_id]['chunk_options']`` carries ONLY the sub-dict of
# the chunking strategy selected by ``process_options`` — the other
# strategies' parameters are dropped because they are never consumed
# during processing.  Reparsing a document overwrites both
# ``process_options`` and ``chunk_options`` together.
# ---------------------------------------------------------------------------


# Strategy selector (F/R/V/P) → snapshot sub-dict key.  Single source
# of truth for the slim ``chunk_options`` shape — used by
# :func:`resolve_chunk_options` to pick which strategy block to keep
# and by :func:`slim_chunk_options` to project caller-supplied dicts
# down to the selected strategy.
_CHUNK_STRATEGY_KEYS: dict[str, str] = {
    PROCESS_OPTION_CHUNK_FIXED: "fixed_token",
    PROCESS_OPTION_CHUNK_RECURSIVE: "recursive_character",
    PROCESS_OPTION_CHUNK_VECTOR: "semantic_vector",
    PROCESS_OPTION_CHUNK_PARAGRAH: "paragraph_semantic",
}


def chunk_strategy_key(process_options: Any) -> str:
    """Return the ``chunk_options`` sub-dict key for ``process_options``.

    Accepts a raw options string or a :class:`ProcessOptions` value.
    Falls back to ``"fixed_token"`` when no chunking selector is
    present — F is the default strategy used both by the file-chunker
    dispatcher (when ``chunking_explicit`` is False the legacy
    ``chunking_func`` runs, which defaults to fixed-token chunking
    that reads from the same sub-dict).
    """
    if isinstance(process_options, ProcessOptions):
        strategy = process_options.chunking
    else:
        strategy = parse_process_options(process_options).chunking
    return _CHUNK_STRATEGY_KEYS.get(strategy, "fixed_token")


def slim_chunk_options(
    snapshot: Mapping[str, Any] | None,
    process_options: Any = "",
) -> dict[str, Any]:
    """Project a (possibly full) chunker snapshot down to the active strategy.

    Keeps the top-level ``chunk_token_size`` and the one strategy
    sub-dict picked by :func:`chunk_strategy_key`; everything else is
    discarded.  Idempotent: a slim snapshot whose key already matches
    ``process_options`` passes through unchanged (deep-copied for
    isolation).  When the matching strategy block is absent from the
    input, an empty dict is used so downstream consumers always see a
    dict-shaped slot.
    """
    key = chunk_strategy_key(process_options)
    src: Mapping[str, Any] = snapshot or {}
    result: dict[str, Any] = {}
    if "chunk_token_size" in src:
        result["chunk_token_size"] = deepcopy(src["chunk_token_size"])
    result[key] = deepcopy(dict(src.get(key) or {}))
    return result


def _env_optional_str(key: str) -> str | None:
    """Return the env value as a string, collapsing empty / 'None' to None."""
    raw = os.getenv(key)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.lower() == "none":
        return None
    return raw


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "t", "y")


def default_chunker_config() -> dict[str, Any]:
    """Snapshot the **strategy-specific** env-driven defaults for every shipped chunker.

    Builds a per-strategy sub-dict whose keys mirror each strategy's
    keyword-only signature (so :func:`resolve_chunk_options` can splat
    them straight into the chunker call).

    Provenance / precedence note: this function reads only
    *strategy-specific* env vars (``CHUNK_F_OVERLAP_SIZE``,
    ``CHUNK_R_SIZE``, ``CHUNK_R_OVERLAP_SIZE``, ``CHUNK_R_SEPARATORS``,
    ``CHUNK_V_SIZE``, ``CHUNK_V_*``, ``CHUNK_P_SIZE``,
    ``CHUNK_P_OVERLAP_SIZE``,
    ``CHUNK_F_SPLIT_BY_CHARACTER``…).  It does **not** read the legacy
    top-level envs ``CHUNK_SIZE`` / ``CHUNK_OVERLAP_SIZE``, and it
    deliberately **omits** ``chunk_overlap_token_size`` from a strategy
    sub-dict when its own env var is unset — leaving the slot empty is
    the signal that lets
    :meth:`LightRAG._apply_chunk_size_overlay` apply the legacy
    constructor field (``LightRAG(chunk_overlap_token_size=…)``) and
    finally the legacy ``CHUNK_OVERLAP_SIZE`` env in that order.  Same
    rationale for top-level ``chunk_token_size`` — overlay fills it from
    ``LightRAG(chunk_token_size=…)`` then ``CHUNK_SIZE`` env.  Net
    precedence (high → low): ``addon_params`` explicit > strategy env
    > legacy ctor field > legacy env.

    Read at instance-creation time via
    :func:`lightrag.addon_params.default_addon_params`; users can mutate
    ``addon_params['chunker']`` at runtime to change the defaults applied
    to subsequently enqueued documents (already-enqueued docs hold a
    frozen ``full_docs[doc_id]['chunk_options']`` snapshot).
    """
    config: dict[str, Any] = {
        "fixed_token": {
            "split_by_character": _env_optional_str("CHUNK_F_SPLIT_BY_CHARACTER"),
            "split_by_character_only": _env_bool(
                "CHUNK_F_SPLIT_BY_CHARACTER_ONLY", False
            ),
        },
        "recursive_character": {
            # Default separators include CJK sentence-ending punctuation
            # so Chinese / mixed-language documents split at semantic
            # boundaries instead of falling through to character-level
            # splitting.  See ``constants.DEFAULT_R_SEPARATORS`` for
            # cascade order rationale.
            "separators": json.loads(
                os.getenv("CHUNK_R_SEPARATORS", json.dumps(list(DEFAULT_R_SEPARATORS)))
            ),
        },
        "semantic_vector": {
            "breakpoint_threshold_type": os.getenv(
                "CHUNK_V_BREAKPOINT_THRESHOLD_TYPE", "percentile"
            ),
            "breakpoint_threshold_amount": parse_optional_float(
                os.getenv("CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT")
            ),
            "buffer_size": int(os.getenv("CHUNK_V_BUFFER_SIZE", "1")),
            # Default extends LangChain's English-only sentence splitter
            # with CJK terminators so SemanticChunker can actually find
            # sentence boundaries on Chinese input.  Override per
            # deployment if you need a different language mix.
            "sentence_split_regex": os.getenv(
                "CHUNK_V_SENTENCE_SPLIT_REGEX", DEFAULT_SENTENCE_SPLIT_REGEX
            ),
        },
        "paragraph_semantic": {},
    }

    # Strategy-specific overlap envs only — leave the slot absent when
    # unset so overlay can detect provenance and fill from the legacy
    # tier (constructor field → CHUNK_OVERLAP_SIZE env).
    f_overlap_raw = os.getenv("CHUNK_F_OVERLAP_SIZE")
    if f_overlap_raw is not None:
        config["fixed_token"]["chunk_overlap_token_size"] = int(f_overlap_raw)
    r_overlap_raw = os.getenv("CHUNK_R_OVERLAP_SIZE")
    if r_overlap_raw is not None:
        config["recursive_character"]["chunk_overlap_token_size"] = int(r_overlap_raw)
    p_overlap_raw = os.getenv("CHUNK_P_OVERLAP_SIZE")
    if p_overlap_raw is not None:
        config["paragraph_semantic"]["chunk_overlap_token_size"] = int(p_overlap_raw)

    # P strategy carries its own ``chunk_token_size`` override so the
    # paragraph-semantic merge target can diverge from the global
    # ``CHUNK_SIZE`` (e.g. heading-aligned chunks may want a larger
    # ceiling).  Slot is left absent when unset; the dispatcher falls
    # back to the top-level ``chunk_token_size`` resolved by overlay.
    p_size_raw = os.getenv("CHUNK_P_SIZE")
    if p_size_raw is not None:
        config["paragraph_semantic"]["chunk_token_size"] = int(p_size_raw)

    # R/V strategies likewise carry their own optional ``chunk_token_size``
    # overrides (recursive character splitting may want a smaller target,
    # semantic-vector clustering a larger advisory ceiling).  Same
    # slot-absent convention as P.
    r_size_raw = os.getenv("CHUNK_R_SIZE")
    if r_size_raw is not None:
        config["recursive_character"]["chunk_token_size"] = int(r_size_raw)
    v_size_raw = os.getenv("CHUNK_V_SIZE")
    if v_size_raw is not None:
        config["semantic_vector"]["chunk_token_size"] = int(v_size_raw)

    return config


def resolve_chunk_options(
    addon_params: Mapping[str, Any] | None,
    *,
    process_options: Any = "",
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
) -> dict[str, Any]:
    """Build a per-document slim ``chunk_options`` snapshot.

    Reads the chunker config from ``addon_params['chunker']``, falling
    back to a freshly built :func:`default_chunker_config` when the
    addon-params mapping is missing or hasn't been populated, then
    keeps only the parameters of the strategy selected by
    ``process_options`` (the other strategies' sub-dicts are dropped —
    they would never be consumed during processing).  See
    :func:`slim_chunk_options` for the projection rules and
    :func:`chunk_strategy_key` for the strategy → sub-dict mapping
    (default F → ``fixed_token``).

    The F runtime args from ``LightRAG.ainsert`` overlay the
    ``fixed_token`` sub-dict when (and only when) the active strategy
    is F — for R/V/P these args have no slot to land in and are
    silently dropped:

      - ``split_by_character`` overrides the env when **non-None**.
        ``None`` (signature default) means "use the env / addon_params
        default".
      - ``split_by_character_only`` overrides the env when **True**.
        ``False`` (signature default) means "use the env / addon_params
        default" — there's no clean way to distinguish "unset" from
        "explicit False" with a positional default, so the env wins
        unless the caller actively opts in.

    The returned snapshot is an independent deep copy: mutating it has
    no effect on subsequent resolutions.
    """
    src: Mapping[str, Any] | None = None
    if isinstance(addon_params, Mapping):
        candidate = addon_params.get("chunker")
        if isinstance(candidate, Mapping):
            src = candidate
    if src is None:
        src = default_chunker_config()

    snapshot = slim_chunk_options(src, process_options)
    if chunk_strategy_key(process_options) == "fixed_token":
        fixed = snapshot["fixed_token"]
        if split_by_character is not None:
            fixed["split_by_character"] = split_by_character
        if split_by_character_only:
            fixed["split_by_character_only"] = True
    return snapshot


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
    if engine == PARSER_ENGINE_MINERU:
        mode = os.getenv("MINERU_API_MODE", "local").strip().lower()
        if mode == "official":
            return bool(os.getenv("MINERU_API_TOKEN", "").strip())
        if mode == "local":
            return bool(os.getenv("MINERU_LOCAL_ENDPOINT", "").strip())
        return False
    endpoint_env = _PARSER_ENGINE_ENDPOINT_ENV.get(engine)
    if endpoint_env:
        return bool(os.getenv(endpoint_env, "").strip())
    return True


def parser_engine_endpoint_requirement(engine: str) -> str | None:
    if engine == PARSER_ENGINE_MINERU:
        mode = os.getenv("MINERU_API_MODE", "local").strip().lower()
        if mode == "official":
            return "MINERU_API_TOKEN"
        if mode == "local":
            return "MINERU_LOCAL_ENDPOINT"
        allowed = ", ".join(sorted(_VALID_MINERU_API_MODES))
        return f"valid MINERU_API_MODE ({allowed})"
    return _PARSER_ENGINE_ENDPOINT_ENV.get(engine)


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
    recognised hint per the spec; otherwise ``None``.  This low-level helper
    stays non-throwing because scan grouping and basename canonicalization need
    a best-effort classifier.  Ingestion entrypoints must call
    :func:`resolve_file_parser_directives`, which validates malformed hints and
    raises instead of falling back.
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


def _validate_filename_hint_for_resolution(
    file_path: str | Path,
    *,
    require_external_endpoint: bool,
) -> None:
    """Fail fast for malformed filename hints on ingestion entrypoints."""
    basename = Path(file_path).name
    m = _PARSER_HINT_RE.search(basename)
    if not m:
        return

    inner = m.group(1)
    errors: list[str] = []

    if not inner.strip():
        errors.append(f"filename hint {m.group(0)!r} is empty")
        raise FilenameParserHintError(
            f"Invalid filename parser hint in {basename!r}: " + "; ".join(errors)
        )

    if "-" in inner:
        engine_name, _, options = inner.partition("-")
        engine = normalize_parser_engine(engine_name)
        if engine not in SUPPORTED_PARSER_ENGINES:
            supported = ", ".join(sorted(SUPPORTED_PARSER_ENGINES))
            errors.append(
                f"filename hint {m.group(0)!r} uses unsupported parser engine "
                f"{engine_name.strip()!r}; supported engines: {supported}"
            )
        elif options:
            errors.extend(
                validate_process_options(
                    options,
                    label=f"filename hint {m.group(0)!r} options",
                )
            )
    else:
        engine = normalize_parser_engine(inner)
        if engine in SUPPORTED_PARSER_ENGINES:
            options = ""
        else:
            option_errors = validate_process_options(
                inner,
                label=f"filename hint {m.group(0)!r} options",
            )
            if not option_errors:
                engine = None
                options = inner
            elif all(ch in SUPPORTED_PROCESS_OPTIONS or ch == " " for ch in inner):
                engine = None
                options = inner
                errors.extend(option_errors)
            else:
                supported = ", ".join(sorted(SUPPORTED_PARSER_ENGINES))
                errors.append(
                    f"filename hint {m.group(0)!r} uses unsupported parser engine "
                    f"{inner.strip()!r}; supported engines: {supported}"
                )

    if engine in SUPPORTED_PARSER_ENGINES:
        suffix = parser_suffix(file_path)
        if not parser_engine_supports_suffix(engine, suffix):
            supported_suffixes = ", ".join(
                sorted(PARSER_ENGINE_SUFFIX_CAPABILITIES.get(engine, frozenset()))
            )
            errors.append(
                f"filename hint {m.group(0)!r} uses parser engine {engine!r} "
                f"for unsupported suffix {suffix!r}; supported suffixes: "
                f"{supported_suffixes}"
            )
        endpoint_req = parser_engine_endpoint_requirement(engine)
        if (
            require_external_endpoint
            and endpoint_req
            and not parser_engine_endpoint_configured(engine)
        ):
            errors.append(
                f"filename hint {m.group(0)!r} requires {endpoint_req} "
                "to be configured"
            )

    if errors:
        raise FilenameParserHintError(
            f"Invalid filename parser hint in {basename!r}: " + "; ".join(errors)
        )


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
        endpoint_req = parser_engine_endpoint_requirement(engine)
        if endpoint_req and not parser_engine_endpoint_configured(engine):
            errors.append(f"{label} requires {endpoint_req} to be configured")
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
    _validate_filename_hint_for_resolution(
        file_path,
        require_external_endpoint=require_external_endpoint,
    )

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
        doc_format = content_data.get("parse_format", FULL_DOCS_FORMAT_RAW)
        if doc_format == FULL_DOCS_FORMAT_LIGHTRAG and content_data.get(
            "sidecar_location"
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
