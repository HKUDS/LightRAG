"""Tree-sitter code-aware chunking — the ``"T"`` strategy.

Splits source code along tree-sitter parse-tree boundaries so a chunk never
cuts a function or class in half. Falls back to the plain fixed-token window
algorithm (the same one the ``"F"`` strategy uses) in three cases the feature
request calls out explicitly:

  - the file's language has no registered grammar (unsupported language);
  - the grammar package for a supported language is not installed;
  - the parse tree reports a syntax error (``root_node.has_error``).

A single function or class body that is itself larger than
``chunk_token_size`` is sub-split with the same token-window algorithm,
rather than emitted as one oversized chunk -- the structural split is a
*preference*, not a hard ceiling override.

Supported languages ship as part of the ``api`` extras group: Python
(``tree-sitter-python``), JavaScript (``tree-sitter-javascript``), and
TypeScript/TSX (``tree-sitter-typescript``) -- each is still lazy-imported and
independently optional, so a deployment missing one degrades gracefully
rather than failing to import this module. Language is resolved from an
explicit ``language=`` override first, then from ``file_path``'s extension.
"""

from __future__ import annotations

from typing import Any

from lightrag.chunker.token_size import chunking_by_fixed_token
from lightrag.utils import Tokenizer, logger

try:
    from tree_sitter import Language, Node, Parser

    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False
    Language = Node = Parser = None  # type: ignore[assignment,misc]


# Suffix -> canonical language name. Checked against ``file_path`` when no
# explicit ``language=`` override is given.
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
}

# Aliases accepted on the explicit ``language=`` override, mapped onto the
# same canonical names ``_EXTENSION_TO_LANGUAGE`` produces.
_LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "python": "python",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
}

# Node types that mark a top-level "semantic unit" (a function or class the
# chunker should try to keep whole) per language. Anything not in this set is
# treated as filler -- imports, top-level statements, comments -- and is
# merged into token-window chunks alongside its neighbouring filler.
_UNIT_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {"function_definition", "class_definition", "decorated_definition"}
    ),
    "javascript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "generator_function_declaration",
        }
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "generator_function_declaration",
            "interface_declaration",
        }
    ),
    "tsx": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "generator_function_declaration",
            "interface_declaration",
        }
    ),
}


def _resolve_language(language: str | None, file_path: str | None) -> str | None:
    """Canonical language name from an explicit override or a file extension."""
    if language:
        return _LANGUAGE_ALIASES.get(language.strip().lower())
    if not file_path:
        return None
    lowered = file_path.strip().lower()
    for suffix, lang in _EXTENSION_TO_LANGUAGE.items():
        if lowered.endswith(suffix):
            return lang
    return None


def _load_grammar(language: str) -> Language | None:
    """Import and build the tree-sitter ``Language`` for one supported name.

    Each grammar package is independently optional (lazy-imported here); a
    missing package degrades to the caller's token-chunking fallback rather
    than raising, exactly like ``chunking_by_recursive_character`` degrades
    when ``langchain-text-splitters`` is absent.
    """
    try:
        if language == "python":
            import tree_sitter_python as grammar

            return Language(grammar.language())
        if language == "javascript":
            import tree_sitter_javascript as grammar

            return Language(grammar.language())
        if language == "typescript":
            import tree_sitter_typescript as grammar

            return Language(grammar.language_typescript())
        if language == "tsx":
            import tree_sitter_typescript as grammar

            return Language(grammar.language_tsx())
    except ImportError:
        return None
    return None


def _byte_to_char_offsets(content: str) -> dict[int, int]:
    """Map every UTF-8 byte offset that starts a character to its char index.

    Tree-sitter reports node boundaries in bytes; ``_source_span`` (like
    every other chunker's) is in characters. Node boundaries always land on
    character boundaries -- tree-sitter never splits a multi-byte codepoint
    -- so a lookup table built once over the whole document answers every
    node's start/end byte in O(1) each, for O(len(content)) total work.
    """
    offsets: dict[int, int] = {0: 0}
    byte_pos = 0
    for char_index, ch in enumerate(content):
        byte_pos += len(ch.encode("utf-8"))
        offsets[byte_pos] = char_index + 1
    return offsets


def _char_span(
    byte_to_char: dict[int, int], start_byte: int, end_byte: int
) -> tuple[int, int] | None:
    start = byte_to_char.get(start_byte)
    end = byte_to_char.get(end_byte)
    if start is None or end is None or start >= end:
        return None
    return start, end


def _fixed_token_chunks(
    tokenizer: Tokenizer,
    content: str,
    char_start: int,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[dict[str, Any]]:
    """Token-window chunk one span of ``content`` and re-anchor its offsets.

    Used both for the whole-file fallback (``char_start=0``) and for
    sub-splitting one oversized unit or filler span, whose text has already
    been sliced out of the parent document.
    """
    sub_chunks = chunking_by_fixed_token(
        tokenizer,
        content,
        chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
        _emit_source_span=True,
    )
    for sub_chunk in sub_chunks:
        span = sub_chunk.get("_source_span")
        if span is not None:
            sub_chunk["_source_span"] = {
                "start": char_start + span["start"],
                "end": char_start + span["end"],
            }
    return sub_chunks


def _collect_units_and_filler(
    root: Node, unit_types: frozenset[str]
) -> list[tuple[bool, int, int]]:
    """Walk the root's direct children into ``(is_unit, start_byte, end_byte)`` spans.

    Only top-level children are inspected -- a method inside a class is part
    of that class's unit span, not chunked separately -- which keeps a class
    together with its methods exactly as the feature request asks
    ("chunks aligned to code structure (function, class, etc.)"). Consecutive
    non-unit children are merged into a single filler span so a run of
    imports or top-level statements becomes one token-windowed chunk instead
    of one per statement.
    """
    spans: list[tuple[bool, int, int]] = []
    filler_start: int | None = None
    filler_end: int | None = None

    def flush_filler() -> None:
        nonlocal filler_start, filler_end
        if filler_start is not None and filler_end is not None:
            spans.append((False, filler_start, filler_end))
        filler_start = filler_end = None

    for child in root.children:
        if child.type in unit_types:
            flush_filler()
            spans.append((True, child.start_byte, child.end_byte))
        else:
            if filler_start is None:
                filler_start = child.start_byte
            filler_end = child.end_byte
    flush_filler()
    return spans


def chunking_by_tree_sitter(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    chunk_overlap_token_size: int = 100,
    language: str | None = None,
    file_path: str | None = None,
) -> list[dict[str, Any]]:
    """Code-structure-aware chunker -- the ``"T"`` chunking strategy.

    Args:
        tokenizer: LightRAG tokenizer, used both for the fallback windows and
            for token-counting each structural chunk.
        content: Source text to split.
        chunk_token_size: Target/ceiling size (tokens) for fallback windows
            and for sub-splitting an oversized unit; not a hard cap on a
            single function/class chunk that fits under it.
        chunk_overlap_token_size: Token overlap used only on the token-window
            fallback paths (whole-file fallback, filler spans, oversized
            units) -- adjacent function/class chunks never overlap, since
            overlapping unrelated code units would not aid retrieval.
        language: Explicit language override (``"python"``, ``"javascript"``,
            ``"typescript"``, ``"tsx"``, or the ``"py"``/``"js"``/``"ts"``
            aliases). Takes precedence over ``file_path`` when given.
        file_path: The document's canonical basename, used to infer the
            language from its extension when ``language`` is not given.

    Returns:
        Ordered list of ``{"tokens", "content", "chunk_order_index",
        "_source_span"}`` dicts, matching the shape every other file-chunker
        strategy returns.
    """
    if not content or not content.strip():
        return []

    resolved_language = _resolve_language(language, file_path)
    if resolved_language is None:
        if language:
            # An explicit override that doesn't match a known alias is
            # almost certainly a typo, not routine degradation -- unlike an
            # unrecognized file extension (the common, expected case),
            # which stays at debug.
            logger.warning(
                "[tree_sitter] language override %r is not recognized; "
                "falling back to fixed-token chunking. Supported: %s.",
                language,
                ", ".join(sorted(set(_LANGUAGE_ALIASES.values()))),
            )
        else:
            logger.debug(
                "[tree_sitter] no language resolved for file_path=%r; "
                "falling back to fixed-token chunking",
                file_path,
            )
        return _fixed_token_chunks(
            tokenizer, content, 0, chunk_token_size, chunk_overlap_token_size
        )

    if not _TREE_SITTER_AVAILABLE:
        logger.warning(
            "[tree_sitter] the 'tree-sitter' package is not installed; "
            "falling back to fixed-token chunking. It ships with the 'api' "
            "extra (`pip install lightrag-hku[api]`); install it directly "
            "(`pip install tree-sitter`) for a bare library install."
        )
        return _fixed_token_chunks(
            tokenizer, content, 0, chunk_token_size, chunk_overlap_token_size
        )

    grammar = _load_grammar(resolved_language)
    if grammar is None:
        pip_package = (
            "tree-sitter-typescript"
            if resolved_language == "tsx"
            else (f"tree-sitter-{resolved_language}")
        )
        logger.warning(
            "[tree_sitter] grammar for language '%s' is not installed; "
            "falling back to fixed-token chunking. It ships with the 'api' "
            "extra (`pip install lightrag-hku[api]`); install it directly "
            "(`pip install %s`) for a bare library install.",
            resolved_language,
            pip_package,
        )
        return _fixed_token_chunks(
            tokenizer, content, 0, chunk_token_size, chunk_overlap_token_size
        )

    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    tree = parser.parse(content_bytes)
    if tree.root_node.has_error:
        logger.warning(
            "[tree_sitter] parse error in a '%s' file; falling back to "
            "fixed-token chunking for the whole document.",
            resolved_language,
        )
        return _fixed_token_chunks(
            tokenizer, content, 0, chunk_token_size, chunk_overlap_token_size
        )

    unit_types = _UNIT_NODE_TYPES.get(resolved_language, frozenset())
    spans = _collect_units_and_filler(tree.root_node, unit_types)
    if not spans:
        return _fixed_token_chunks(
            tokenizer, content, 0, chunk_token_size, chunk_overlap_token_size
        )

    byte_to_char = _byte_to_char_offsets(content)
    results: list[dict[str, Any]] = []
    for is_unit, start_byte, end_byte in spans:
        char_span = _char_span(byte_to_char, start_byte, end_byte)
        if char_span is None:
            continue
        char_start, char_end = char_span
        body = content[char_start:char_end]
        stripped = body.strip()
        if not stripped:
            continue
        # Re-anchor after stripping leading/trailing whitespace so the span
        # matches the trimmed content, the same convention F/R/P follow.
        lstrip_delta = len(body) - len(body.lstrip())
        trimmed_start = char_start + lstrip_delta
        trimmed_end = trimmed_start + len(stripped)

        token_count = len(tokenizer.encode(stripped))
        if token_count <= chunk_token_size:
            results.append(
                {
                    "tokens": token_count,
                    "content": stripped,
                    "chunk_order_index": len(results),
                    "_source_span": {"start": trimmed_start, "end": trimmed_end},
                }
            )
            continue

        # Oversized unit or filler run: sub-split with the token-window
        # fallback rather than emit one chunk that blows past the target.
        for sub_chunk in _fixed_token_chunks(
            tokenizer,
            stripped,
            trimmed_start,
            chunk_token_size,
            chunk_overlap_token_size,
        ):
            sub_chunk["chunk_order_index"] = len(results)
            results.append(sub_chunk)

    if not results:
        return _fixed_token_chunks(
            tokenizer, content, 0, chunk_token_size, chunk_overlap_token_size
        )
    return results
