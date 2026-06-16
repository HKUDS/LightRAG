from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightrag.constants import (
    DEFAULT_CHUNK_P_SIZE,
    DEFAULT_R_SEPARATORS,
    DEFAULT_SENTENCE_SPLIT_REGEX,
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_NATIVE,
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
    SUPPORTED_PROCESS_OPTIONS,
)
from lightrag.parser.registry import (
    PARSER_ENGINE_PASSTHROUGH,
    PARSER_ENGINE_REUSE,
    engine_endpoint_configured,
    engine_endpoint_requirement,
    supported_parser_engines,
    suffix_capabilities,
)
from lightrag.parser.param_schema import (
    chunk_param_overlap_error,
    parse_chunk_params,
    split_top_level,
    take_paren_block,
)
from lightrag.utils import logger, parse_optional_float

import json
from collections.abc import Mapping
from copy import deepcopy

# Trailing parser-hint pattern: matches ``.[engine].ext`` at end of basename.
# Group 1 captures the raw engine token (still needs normalize_parser_engine
# and SUPPORTED_PARSER_ENGINES validation); group 2 captures ``.ext`` so it
# can be reattached when stripping the hint.
_PARSER_HINT_RE = re.compile(r"\.\[([^\]]*)\](\.[^.]+)$")

# Per-suffix default engine override, consulted before the global ``legacy``
# fallback. ``.textpack`` is handled only by the native engine, so it routes
# there automatically (no filename hint / LIGHTRAG_PARSER rule needed). ``.md``
# is deliberately absent — it keeps the legacy default and opts into native the
# same way ``.docx`` does (hint or rule).
_DEFAULT_ENGINE_BY_SUFFIX: dict[str, str] = {"textpack": PARSER_ENGINE_NATIVE}


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

    Strategy-specific default backfill: for ``paragraph_semantic`` we
    guarantee a populated ``chunk_token_size`` slot before returning
    (caller-supplied value > ``CHUNK_P_SIZE`` env >
    ``DEFAULT_CHUNK_P_SIZE``).  This is the single chokepoint that
    every enqueue path runs through — both the
    ``resolve_chunk_options`` path (built from addon_params) AND the
    direct ``chunk_options=`` kwarg path (caller supplies the dict)
    flow through here, so the backfill cannot be bypassed by runtime
    addon_params mutation or by passing an explicit ``chunk_options``
    that omits the P slot.  P must NOT inherit the top-level
    ``chunk_token_size`` (global ``CHUNK_SIZE`` / legacy ctor) —
    paragraph-semantic merging needs more headroom than the global
    default.
    """
    key = chunk_strategy_key(process_options)
    src: Mapping[str, Any] = snapshot or {}
    result: dict[str, Any] = {}
    if "chunk_token_size" in src:
        result["chunk_token_size"] = deepcopy(src["chunk_token_size"])
    result[key] = deepcopy(dict(src.get(key) or {}))
    if key == "paragraph_semantic" and "chunk_token_size" not in result[key]:
        p_size_raw = os.getenv("CHUNK_P_SIZE")
        result[key]["chunk_token_size"] = (
            int(p_size_raw) if p_size_raw is not None else DEFAULT_CHUNK_P_SIZE
        )
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
    *strategy-specific* env vars (``CHUNK_F_SIZE``,
    ``CHUNK_F_OVERLAP_SIZE``, ``CHUNK_R_SIZE``, ``CHUNK_R_OVERLAP_SIZE``,
    ``CHUNK_R_SEPARATORS``, ``CHUNK_V_SIZE``, ``CHUNK_V_*``,
    ``CHUNK_P_SIZE``, ``CHUNK_P_OVERLAP_SIZE``,
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
    # ceiling).  Unlike R/V, the slot is ALWAYS populated — when
    # ``CHUNK_P_SIZE`` is unset we use ``DEFAULT_CHUNK_P_SIZE`` (2000)
    # rather than letting the dispatcher fall back to the global
    # ``CHUNK_SIZE`` (1200): paragraph-semantic merging needs more
    # headroom than the global default to keep related paragraphs
    # together, and silently inheriting the smaller global ceiling
    # defeats the strategy's purpose.
    p_size_raw = os.getenv("CHUNK_P_SIZE")
    config["paragraph_semantic"]["chunk_token_size"] = (
        int(p_size_raw) if p_size_raw is not None else DEFAULT_CHUNK_P_SIZE
    )

    # F/R/V strategies likewise carry their own optional ``chunk_token_size``
    # overrides (fixed-token may want a deployment-specific window, recursive
    # character splitting a smaller target, semantic-vector clustering a larger
    # advisory ceiling).  Same slot-absent convention as P: leave the slot
    # absent when the env is unset so the strategy inherits the top-level
    # ``chunk_token_size`` fallback at consumption time.
    f_size_raw = os.getenv("CHUNK_F_SIZE")
    if f_size_raw is not None:
        config["fixed_token"]["chunk_token_size"] = int(f_size_raw)
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
    # P-strategy ``chunk_token_size`` backfill lives in
    # ``slim_chunk_options`` — that's the single chokepoint shared by
    # every enqueue path (this function AND the direct
    # ``chunk_options=`` kwarg path in ``_chunk_options_at``).
    return snapshot


def _extract_param_blocks(
    inner: str,
) -> tuple[str, str | None, dict[str, str], list[str]]:
    """Strip ``(...)`` parameter blocks from a hint / rule inner string.

    Returns ``(stripped, engine_param_text, chunk_param_texts, errors)``:

    * ``stripped`` is ``inner`` with every parameter block removed, so the
      existing engine / selector parsing (:func:`split_engine_and_options`,
      :func:`_rule_engine_and_options`, :func:`validate_process_options`) runs
      on a parameter-free string and legacy behaviour is preserved verbatim.
    * ``engine_param_text`` is the text inside an engine-level ``(...)`` block
      (before the engine/options ``-``) when present, else ``None``.  Engine
      parameters are not accepted in Phase 1; callers reject them.
    * ``chunk_param_texts`` maps each chunk selector char (F/R/V/P) to the raw
      text of the block that immediately follows it.
    * ``errors`` collects structural problems (unbalanced parens, a block not
      following a chunk strategy, duplicate blocks on one char).

    A parameter-free ``inner`` returns ``(inner, None, {}, [])`` unchanged.
    """
    out: list[str] = []
    engine_param: str | None = None
    chunk_params: dict[str, str] = {}
    errors: list[str] = []
    i = 0
    n = len(inner)
    seen_dash = False
    prev_meaningful: str | None = None
    while i < n:
        ch = inner[i]
        if ch == "(":
            block, nxt = take_paren_block(inner, i)
            if block is None:
                errors.append(f"unbalanced '(' in {inner!r}")
                out.append(inner[i:])
                break
            if seen_dash and prev_meaningful in PROCESS_OPTION_CHUNK_CHARS:
                if prev_meaningful in chunk_params:
                    errors.append(
                        f"chunk strategy {prev_meaningful!r} has more than one "
                        "parameter block"
                    )
                else:
                    chunk_params[prev_meaningful] = block
            elif not seen_dash and prev_meaningful is not None:
                # Engine-level block, e.g. ``mineru(page_range=1-3)``.
                if engine_param is not None:
                    errors.append("parser engine has more than one parameter block")
                else:
                    engine_param = block
            else:
                errors.append(
                    f"parameters '({block})' must follow a chunk strategy (F/R/V/P)"
                )
            i = nxt
            prev_meaningful = None
            continue
        out.append(ch)
        if ch == "-":
            seen_dash = True
        if ch != " ":
            prev_meaningful = ch
        i += 1
    return "".join(out), engine_param, chunk_params, errors


def _parse_chunk_param_texts(
    chunk_param_texts: dict[str, str], *, label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Parse raw chunk-param block texts into canonical per-selector dicts.

    Returns ``(chunk_params, errors)``; ``chunk_params`` only contains the
    selectors whose block parsed to a non-empty dict.
    """
    chunk_params: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for selector, text in chunk_param_texts.items():
        parsed, perrors = parse_chunk_params(
            text, selector=selector, label=f"{label} chunk strategy {selector!r}"
        )
        errors.extend(perrors)
        if parsed:
            chunk_params[selector] = parsed
    return chunk_params, errors


def split_engine_and_options(bracket_inner: str) -> tuple[str | None, str]:
    """Decompose a bracket-hint inner string into ``(engine, options)``.

    Format rules (see docs/FileProcessingPipeline-zh.md):
        - ``ENGINE-OPTIONS``: first ``-``-separated segment is the engine
          candidate; the remainder is the options string.
        - ``ENGINE``: matches a supported engine name as a whole.
        - ``-OPTIONS``: leading ``-`` marks an options-only hint.
    """
    inner = (bracket_inner or "").strip()
    if not inner:
        return None, ""

    if inner.startswith("-"):
        return None, inner[1:].strip()

    if "-" in inner:
        head, _, tail = inner.partition("-")
        engine_candidate = normalize_parser_engine(head)
        if engine_candidate in supported_parser_engines():
            return engine_candidate, tail.strip()
        return None, ""

    engine_candidate = normalize_parser_engine(inner)
    if engine_candidate in supported_parser_engines():
        return engine_candidate, ""
    return None, ""


def parser_suffix(file_path: str | Path) -> str:
    return Path(file_path).suffix.lower().lstrip(".")


def parser_engine_supports_suffix(engine: str, suffix: str) -> bool:
    return suffix.lower().lstrip(".") in suffix_capabilities(engine)


def parser_engine_endpoint_configured(engine: str) -> bool:
    # Endpoint capability lives on the registry ParserSpec (single source).
    return engine_endpoint_configured(engine)


def parser_engine_endpoint_requirement(engine: str) -> str | None:
    return engine_endpoint_requirement(engine)


def _engine_is_usable(
    engine: str,
    suffix: str,
    *,
    require_external_endpoint: bool,
) -> bool:
    if engine not in supported_parser_engines():
        return False
    if not parser_engine_supports_suffix(engine, suffix):
        return False
    if require_external_endpoint and not parser_engine_endpoint_configured(engine):
        return False
    return True


def _filename_hint_match(
    file_path: str | Path,
) -> tuple[re.Match[str], str, str, dict[str, dict[str, Any]]] | None:
    """Locate a supported ``[hint]`` segment in a basename.

    Returns ``(match, engine_or_empty, options, chunk_params)`` when the
    bracket inner is a recognised hint per the spec; otherwise ``None``.
    ``chunk_params`` maps a chunk selector char to its canonical parameter
    dict and is empty when the hint carries no parameters.  This low-level
    helper stays non-throwing because scan grouping and basename
    canonicalization need a best-effort classifier.  Ingestion entrypoints
    must call :func:`resolve_parser_directives`, which validates malformed
    hints and raises instead of falling back.
    """
    basename = Path(file_path).name
    m = _PARSER_HINT_RE.search(basename)
    if not m:
        return None
    raw_inner = m.group(1).strip()
    if raw_inner.startswith("-") and not raw_inner[1:].strip():
        return None

    inner, engine_param, chunk_param_texts, struct_errors = _extract_param_blocks(
        raw_inner
    )
    if struct_errors or engine_param is not None:
        # Best-effort classifier: a malformed or not-yet-supported parameter
        # block means "not a usable hint" here — ingestion entrypoints surface
        # the precise error via the strict validator instead.
        reason = "; ".join(struct_errors) or "engine parameters are not supported yet"
        logger.warning(
            f"[parser_routing] ignoring filename hint {m.group(0)!r} in "
            f"{basename!r}: {reason}"
        )
        return None
    inner = inner.strip()

    if (
        "-" in inner
        and not inner.startswith("-")
        and not inner.partition("-")[2].strip()
    ):
        return None
    engine, options = split_engine_and_options(inner)
    if options:
        option_errors = validate_process_options(options)
        if option_errors:
            logger.warning(
                f"[parser_routing] ignoring filename hint {m.group(0)!r} in "
                f"{basename!r}: {'; '.join(option_errors)}"
            )
            return None
    chunk_params, param_errors = _parse_chunk_param_texts(
        chunk_param_texts, label=f"filename hint {m.group(0)!r}"
    )
    if param_errors:
        logger.warning(
            f"[parser_routing] ignoring filename hint {m.group(0)!r} in "
            f"{basename!r}: {'; '.join(param_errors)}"
        )
        return None
    if engine in supported_parser_engines():
        return m, engine, options, chunk_params
    if engine is None and (options or chunk_params):
        return m, "", options, chunk_params
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

    # Strip and validate parameter blocks first; the engine / selector
    # branches below then run on a parameter-free string exactly as before.
    hint_label = f"filename hint {m.group(0)!r}"
    inner, engine_param, chunk_param_texts, struct_errors = _extract_param_blocks(inner)
    errors.extend(f"{hint_label}: {msg}" for msg in struct_errors)
    if engine_param is not None:
        errors.append(
            f"{hint_label} sets parameters on a parser engine "
            f"(got '({engine_param})'); engine parameters are not supported yet — "
            "only chunk strategies (F/R/V/P) accept parameters"
        )
    _, param_errors = _parse_chunk_param_texts(chunk_param_texts, label=hint_label)
    errors.extend(param_errors)

    engine: str | None = None
    options = ""

    if inner.startswith("-"):
        options = inner[1:].strip()
        if not options:
            errors.append(f"filename hint {m.group(0)!r} has empty process options")
        else:
            errors.extend(
                validate_process_options(
                    options,
                    label=f"filename hint {m.group(0)!r} options",
                )
            )
    elif "-" in inner:
        engine_name, _, options = inner.partition("-")
        engine = normalize_parser_engine(engine_name)
        if engine not in supported_parser_engines():
            supported = ", ".join(sorted(supported_parser_engines()))
            errors.append(
                f"filename hint {m.group(0)!r} uses unsupported parser engine "
                f"{engine_name.strip()!r}; supported engines: {supported}"
            )
        elif not options.strip():
            errors.append(f"filename hint {m.group(0)!r} has empty process options")
        else:
            errors.extend(
                validate_process_options(
                    options,
                    label=f"filename hint {m.group(0)!r} options",
                )
            )
    else:
        engine = normalize_parser_engine(inner)
        if engine not in supported_parser_engines():
            supported = ", ".join(sorted(supported_parser_engines()))
            message = (
                f"filename hint {m.group(0)!r} uses unsupported parser engine "
                f"{inner.strip()!r}; supported engines: {supported}"
            )
            if all(ch in SUPPORTED_PROCESS_OPTIONS or ch == " " for ch in inner):
                message += (
                    "; options-only filename hints must start with '-' "
                    f"(use '[-{inner.strip()}]' instead)"
                )
            errors.append(message)

    if engine in supported_parser_engines():
        suffix = parser_suffix(file_path)
        if not parser_engine_supports_suffix(engine, suffix):
            supported_suffixes = ", ".join(sorted(suffix_capabilities(engine)))
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
                f"filename hint {m.group(0)!r} requires {endpoint_req} to be configured"
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
    return found[1] or None


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
    return (found[1] or None), found[2]


def filename_chunk_params(file_path: str | Path) -> dict[str, dict[str, Any]]:
    """Return the per-selector chunk parameters decoded from a filename hint.

    Maps a chunk selector char (F/R/V/P) to its canonical parameter dict;
    empty when the hint carries no parameters or is not a usable hint.
    """
    found = _filename_hint_match(file_path)
    if not found:
        return {}
    return found[3]


def canonicalize_parser_hinted_basename(file_path: str | Path) -> str:
    """Return basename with a supported parser hint removed.

    Only the final ``.[engine].ext`` (or ``.[engine-options].ext`` /
    ``.[-options].ext``) segment is stripped, exactly once, and only when the
    bracket content is a recognised hint.  Nested hints such as
    ``name.[native].[mineru].pdf`` therefore become ``name.[native].pdf`` —
    additional outer hints are not unwrapped.
    """
    basename = Path(file_path).name
    found = _filename_hint_match(file_path)
    if not found:
        return basename
    m = found[0]
    return f"{basename[: m.start()]}{m.group(2)}"


def parser_rules_from_env() -> str:
    return os.getenv("LIGHTRAG_PARSER", "").strip()


def _iter_parser_rule_items(rules: str) -> list[tuple[int, str]]:
    # Parenthesis-aware split: ';' (preferred) or ',' (legacy) separate rules
    # at paren depth 0, so commas inside a chunk-parameter block such as
    # ``R(chunk_ts=800,chunk_ol=80)`` never split the surrounding rule.
    return [
        (index, item.strip())
        for index, item in enumerate(split_top_level(rules, ";,"), start=1)
        if item.strip()
    ]


def _rule_pattern_matches_engine_capability(pattern: str, engine: str) -> bool:
    supported_suffixes = suffix_capabilities(engine)
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
        stripped_hint, engine_param, chunk_param_texts, struct_errors = (
            _extract_param_blocks(engine_hint)
        )
        engine, options_str = _rule_engine_and_options(stripped_hint)

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
        if engine not in supported_parser_engines():
            supported = ", ".join(sorted(supported_parser_engines()))
            errors.append(
                f"{label} uses unsupported parser engine {engine_hint!r}; "
                f"supported engines: {supported}"
            )
            continue
        if not _rule_pattern_matches_engine_capability(pattern, engine):
            supported_suffixes = ", ".join(sorted(suffix_capabilities(engine)))
            errors.append(
                f"{label} does not match any suffix supported by {engine}; "
                f"supported suffixes: {supported_suffixes}"
            )
        endpoint_req = parser_engine_endpoint_requirement(engine)
        if endpoint_req and not parser_engine_endpoint_configured(engine):
            errors.append(f"{label} requires {endpoint_req} to be configured")
        errors.extend(f"{label}: {msg}" for msg in struct_errors)
        if engine_param is not None:
            errors.append(
                f"{label}: parameters on parser engine {engine!r} are not "
                f"supported yet (got '({engine_param})'); only chunk strategies "
                "(F/R/V/P) accept parameters"
            )
        if options_str:
            errors.extend(
                f"{label}: {msg}"
                for msg in validate_process_options(
                    options_str, label="process options"
                )
            )
        parsed_chunk_params, param_errors = _parse_chunk_param_texts(
            chunk_param_texts, label=label
        )
        errors.extend(param_errors)
        for selector, params in parsed_chunk_params.items():
            overlap_msg = chunk_param_overlap_error(params)
            if overlap_msg:
                errors.append(f"{label} chunk strategy {selector!r}: {overlap_msg}")

    if errors:
        raise ParserRoutingConfigError(
            "Invalid LIGHTRAG_PARSER configuration: " + "; ".join(errors)
        )


def _matching_rule_directives(
    file_path: str | Path,
    *,
    parser_rules: str | None,
    require_external_endpoint: bool,
) -> tuple[str | None, str, dict[str, dict[str, Any]]]:
    """Find the first matching ``LIGHTRAG_PARSER`` rule for ``file_path``.

    Returns ``(engine, options_str, chunk_params)`` where ``engine`` is
    ``None`` when no usable rule is found.  ``options_str`` is empty when a
    rule matched but has no ``-options`` suffix; ``chunk_params`` maps a chunk
    selector char to its canonical parameter dict.  Rule syntax is validated
    at startup (:func:`validate_parser_routing_config`), so a malformed
    parameter block here is skipped best-effort rather than raised.
    """
    suffix = parser_suffix(file_path)
    rules = parser_rules_from_env() if parser_rules is None else parser_rules.strip()
    if not rules:
        return None, "", {}
    for _, item in _iter_parser_rule_items(rules):
        if ":" not in item:
            continue
        pattern, engine_hint = item.split(":", 1)
        pattern = pattern.strip().lower()
        stripped_hint, _engine_param, chunk_param_texts, _errs = _extract_param_blocks(
            engine_hint.strip()
        )
        engine, options_str = _rule_engine_and_options(stripped_hint)
        if not fnmatch.fnmatch(suffix, pattern):
            continue
        if _engine_is_usable(
            engine,
            suffix,
            require_external_endpoint=require_external_endpoint,
        ):
            chunk_params, _param_errors = _parse_chunk_param_texts(
                chunk_param_texts, label=f"rule {item!r}"
            )
            return engine, options_str, chunk_params
    return None, "", {}


@dataclass(frozen=True)
class ParserDirectives:
    """Fully resolved per-file parser directives.

    ``process_options`` stays a pure selector string (``i/t/e/!/F/R/V/P``);
    parameters live in separate fields.  ``chunk_params`` maps a chunk
    selector char to its canonical parameter dict and feeds the existing
    ``chunk_options`` channel.  ``engine_params`` is reserved for a future
    per-file engine-parameter channel and is always empty in Phase 1.
    """

    engine: str
    process_options: str
    chunk_params: dict[str, dict[str, Any]]
    engine_params: dict[str, Any]


def resolve_parser_directives(
    file_path: str | Path,
    *,
    parser_rules: str | None = None,
    require_external_endpoint: bool = True,
) -> ParserDirectives:
    """Resolve engine, process options and per-file parameters for a file.

    Resolution order (mirrors :func:`resolve_file_parser_engine`):
        1. Filename ``[hint]`` — engine and / or options take precedence.
        2. ``LIGHTRAG_PARSER`` rules — first matching rule provides defaults
           for whichever of engine / options the filename hint did not
           specify.
        3. Default engine ``legacy`` with empty options.

    Selector (``i/t/e/!/FRVP``) keeps the legacy "filename options wholesale
    override rule options" behaviour.  Chunk parameters overlay per selector
    char: rule parameters first, then filename-hint parameters (filename wins
    on a shared key).
    """
    suffix = parser_suffix(file_path)
    _validate_filename_hint_for_resolution(
        file_path,
        require_external_endpoint=require_external_endpoint,
    )

    hinted_engine, hinted_options = filename_parser_directives(file_path)
    hinted_chunk_params = filename_chunk_params(file_path)
    if hinted_engine and not _engine_is_usable(
        hinted_engine, suffix, require_external_endpoint=require_external_endpoint
    ):
        # Hinted engine cannot handle this file (e.g. wrong suffix or missing
        # endpoint); fall back to rule-based resolution but keep the hinted
        # options if any.
        hinted_engine = None

    rule_engine, rule_options, rule_chunk_params = _matching_rule_directives(
        file_path,
        parser_rules=parser_rules,
        require_external_endpoint=require_external_endpoint,
    )

    default_engine = _DEFAULT_ENGINE_BY_SUFFIX.get(suffix)
    if default_engine and not _engine_is_usable(
        default_engine, suffix, require_external_endpoint=require_external_endpoint
    ):
        default_engine = None

    engine = hinted_engine or rule_engine or default_engine or PARSER_ENGINE_LEGACY
    options_str = hinted_options or rule_options

    # Overlay chunk params per selector char: rule first, filename-hint wins.
    chunk_params: dict[str, dict[str, Any]] = {}
    for selector in set(rule_chunk_params) | set(hinted_chunk_params):
        merged = {
            **rule_chunk_params.get(selector, {}),
            **hinted_chunk_params.get(selector, {}),
        }
        if merged:
            chunk_params[selector] = merged

    return ParserDirectives(
        engine=engine,
        process_options=sanitize_process_options(options_str),
        chunk_params=chunk_params,
        engine_params={},
    )


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

    Backward-compatible thin wrapper over :func:`resolve_parser_directives`;
    callers that also need the per-file chunk / engine parameters should use
    :func:`resolve_parser_directives` directly.
    """
    directives = resolve_parser_directives(
        file_path,
        parser_rules=parser_rules,
        require_external_endpoint=require_external_endpoint,
    )
    return directives.engine, directives.process_options


def resolve_stored_document_parser_engine(
    file_path: str | Path,
    content_data: dict[str, Any] | None,
) -> str:
    """Resolve the parser engine key for a full_docs row during processing.

    Returns a registry key: the internal ``reuse``/``passthrough`` handlers for
    the no-op formats, or a real engine name for ``pending_parse``.  Never
    raises and never filters an unknown stored engine — the parse worker
    decides fallback/validation (so its warning path actually fires).
    """
    if content_data:
        doc_format = content_data.get("parse_format", FULL_DOCS_FORMAT_RAW)
        # All lightrag rows reuse the already-parsed sidecar (sidecar optional;
        # ReuseParser tolerates a missing one). Reached on resume/retry.
        if doc_format == FULL_DOCS_FORMAT_LIGHTRAG:
            return PARSER_ENGINE_REUSE
        # Anything not pending is already-extracted content (raw direct-insert
        # or legacy-extracted RAW) -> pass through verbatim.
        if doc_format != FULL_DOCS_FORMAT_PENDING_PARSE:
            return PARSER_ENGINE_PASSTHROUGH
        # PENDING_PARSE: honour the stored engine verbatim; fall back to
        # filename rules only when it is absent.
        pending_engine = normalize_parser_engine(content_data.get("parse_engine"))
        if pending_engine:
            return pending_engine

    return resolve_file_parser_engine(file_path)
