from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import asyncio
import inspect
import json
import logging
import re
import json_repair
from typing import Any, AsyncIterator, overload, Literal, Callable, Awaitable
from collections import Counter, defaultdict

from lightrag.exceptions import (
    EmptySummaryResponseError,
    IndexFlushError,
    PipelineCancelledException,
)
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    parse_cache_key,
    Tokenizer,
    TokenBudgetError,
    normalize_entity_name,
    remove_spaces_next_to_chinese,
    sanitize_and_normalize_extracted_text,
    sanitize_text_for_encoding,
    repair_vlm_json_escape_damage_nested,
    strip_markdown_code_fence,
    tolerant_load_json_dict,
    _first_structural_opener,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    acount_tokens,
    atruncate_list_by_token_size,
    run_in_tokenizer_executor,
    compute_args_hash,
    resolve_user_prompt,
    handle_cache,
    save_to_cache,
    CacheData,
    is_truncated_response,
    TokenLimitTruncationTally,
    use_llm_func_with_cache,
    get_env_value,
    get_llm_cache_identity,
    serialize_llm_cache_identity,
    remove_think_tags,
    is_reasoning_only_answer,
    reject_reasoning_only_answer,
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    process_chunks_unified,
    safe_vdb_operation_with_exception,
    create_prefixed_exception,
    fix_tuple_delimiter_corruption,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    render_chunks_context_text,
    apply_source_ids_limit,
    merge_source_ids,
    make_relation_chunk_key,
    has_chunk_tracking_row,
    _cooperative_yield,
    cancel_and_drain_tasks,
    wait_tasks_with_drain,
    performance_timing_log,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
    QueryContextResult,
)
from lightrag.chunk_schema import (
    HEADING_BREADCRUMB_SEP,
    format_heading_context,
    format_parent_headings,
    strip_internal_multimodal_markup_for_extraction,
)
from lightrag.prompt import PROMPTS, resolve_entity_extraction_prompt_profile
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
    DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_QUERY_PRIORITY,
    DEFAULT_SUMMARY_PRIORITY,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_SUMMARY_LANGUAGE,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    RELATION_NO_EVIDENCE_SOURCE_IDS,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
    DEFAULT_ENTITY_NAME_MAX_BYTES,
)
from lightrag.kg.shared_storage import (
    PipelineStatusLogger,
    append_pipeline_history,
    get_storage_keyed_lock,
)
import time
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


class QueryProgress:
    """Progress event identifiers emitted during the retrieval pipeline.

    These string constants are sent to the frontend via the progress_callback
    so the UI can show which pipeline step is currently executing.
    """

    EXTRACTING_KEYWORDS = "extracting_keywords"
    RETRIEVING_ENTITIES = "retrieving_entities"
    RETRIEVING_RELATIONS = "retrieving_relations"
    RETRIEVING_CHUNKS = "retrieving_chunks"
    RERANKING = "reranking"
    GENERATING_RESPONSE = "generating_response"


# Type alias for the async progress callback: receives a QueryProgress event string.
ProgressCallback = Callable[[str], Awaitable[None]]


KGRebuildPolicy = Literal["best_effort", "rollback"]


@dataclass
class KGRebuildReport:
    """Non-fatal data-quality gaps observed while rebuilding KG aggregates.

    Missing or unusable extraction material degrades the semantic aggregate,
    but the rebuild can still restore the hard provenance invariant by
    rewriting graph/vector/tracking source IDs to the surviving chunks.
    Operational storage failures are not represented here; the rebuild
    propagates those failures so the caller's journal remains retryable.

    ``reextracted_chunk_ids`` lists the surviving chunks the ``best_effort``
    policy re-extracted with the LLM because their initial extraction was
    missing or unusable; they are not a degradation and stay out of
    ``has_warnings``. ``degraded_entities`` / ``degraded_relationships`` are
    the objects no surviving chunk's records name, kept with their semantic
    fields and rewritten provenance.

    Known degradation semantics: a degraded relationship preserves its stored
    ``weight`` as-is — without per-chunk extraction cache the rolled-back
    chunk's weight contribution cannot be subtracted (precision is
    restored only by a full reprocess). The degradation is recorded in the
    persisted ``kg_recovery_warnings`` so operators can identify affected
    aggregates.
    """

    missing_cache_chunk_ids: set[str] = field(default_factory=set)
    unusable_cache_chunk_ids: set[str] = field(default_factory=set)
    referenced_chunk_ids: set[str] = field(default_factory=set)
    reextracted_chunk_ids: set[str] = field(default_factory=set)
    degraded_entities: dict[str, list[str]] = field(default_factory=dict)
    degraded_relationships: dict[tuple[str, str], list[str]] = field(
        default_factory=dict
    )

    @property
    def has_warnings(self) -> bool:
        return bool(
            self.missing_cache_chunk_ids
            or self.unusable_cache_chunk_ids
            or self.degraded_entities
            or self.degraded_relationships
        )

    @property
    def surviving_chunk_ids(self) -> set[str]:
        """Chunk ids involved in a degraded rebuild (owner-lookup scope).

        Every specific bucket (missing/unusable cache, degraded aggregates)
        is a subset of ``referenced_chunk_ids`` by construction — missing =
        referenced minus cached, unusable/degraded are recorded per rebuild
        target — so the union is simply the full referenced set whenever any
        degradation was recorded, and empty otherwise.
        """
        return set(self.referenced_chunk_ids) if self.has_warnings else set()


def _get_relationship_vdb_timeout_seconds(global_config: dict[str, Any]) -> float:
    """Derive a defensive timeout for relation VDB upserts.

    Rationale:
    - `knowledge_graph_inst.upsert_edge()` for the default NetworkX storage is in-memory and fast.
    - `relationships_vdb.upsert()` performs embedding calls and remote I/O, which is the more likely
      point of silent stalls during relation merge.
    """
    configured = global_config.get("default_embedding_timeout")
    try:
        base_timeout = float(configured)
    except (TypeError, ValueError):
        base_timeout = 30.0
    # Keep a fixed lower bound high enough to avoid false positives on slow providers.
    return max(base_timeout * 3, 120.0)


def _format_relation_edge_label(edge_key: tuple[str, str] | list[str]) -> str:
    if isinstance(edge_key, tuple):
        left, right = edge_key
    else:
        left, right = edge_key[0], edge_key[1]
    return f"{left}->{right}"


def _truncate_entity_identifier(
    identifier: str,
    limit: int,
    chunk_key: str,
    identifier_role: str,
    byte_limit: int = DEFAULT_ENTITY_NAME_MAX_BYTES,
) -> str:
    """Truncate entity identifiers that exceed the configured length limit.

    Enforces both a character limit (``limit``) and a UTF-8 byte limit
    (``byte_limit``). Milvus validates VARCHAR ``max_length`` in BYTES, not
    characters, so a CJK identifier within the character limit can still
    overflow the field (e.g. 256 Chinese chars ~= 694 bytes > 512). The byte
    truncation cuts on a character boundary so the result stays valid UTF-8.
    """

    char_len = len(identifier)
    byte_len = len(identifier.encode("utf-8"))
    if char_len <= limit and byte_len <= byte_limit:
        return identifier

    display_value = identifier[:limit]
    encoded = display_value.encode("utf-8")
    if len(encoded) > byte_limit:
        # Drop the partial trailing multi-byte char left by the byte slice.
        display_value = encoded[:byte_limit].decode("utf-8", errors="ignore")
    preview = identifier[:50]  # Show first 50 characters as preview
    logger.warning(
        "%s: %s len %d chars / %d bytes > %d chars / %d bytes (Name: '%s...')",
        chunk_key,
        identifier_role,
        char_len,
        byte_len,
        limit,
        byte_limit,
        preview,
    )
    return display_value


def _finalize_extracted_identifier(
    identifier: str, chunk_key: str, identifier_role: str
) -> str:
    """Cap an extracted entity identifier, then run the final space pass.

    ``identifier`` must come from ``normalize_entity_name(...,
    final_space_pass=False)``. The pass runs after the cap, never before it:
    the pass shortens a name, so cutting its output can give two spellings
    different keys where the cut of the previous rules gave them one, which
    breaks the merge-only rule of :func:`normalize_extracted_info`. Running it
    on the capped key keeps the new key a function of the stored one.
    """
    capped = _truncate_entity_identifier(
        identifier, DEFAULT_ENTITY_NAME_MAX_LENGTH, chunk_key, identifier_role
    )
    return remove_spaces_next_to_chinese(capped)


def _truncate_section_context(
    heading_path: str,
    tokenizer: "Tokenizer | None",
    max_tokens: int,
) -> str:
    """Token-budget the `---Section Context---` breadcrumb before injection.

    The breadcrumb is metadata layered on top of the (already chunk-sized)
    input text, so an unbounded heading chain could push an otherwise-valid
    chunk past the provider context window. When the path exceeds ``max_tokens``
    we first collapse it to the **first** level (top-level document location)
    and the **last/leaf** level (the chunk's own, most-specific section),
    eliding the middle with ``first → … → leaf``. A token-dense path (emoji /
    byte-level tokenizers) can still exceed the budget even with one or two
    levels, so a hard token cap is always applied as a backstop — the returned
    string is guaranteed to fit ``max_tokens``. ``max_tokens <= 0`` or a missing
    tokenizer disables the cap.
    """
    if not heading_path or tokenizer is None or max_tokens <= 0:
        return heading_path
    if len(tokenizer.encode(heading_path)) <= max_tokens:
        return heading_path
    levels = heading_path.split(HEADING_BREADCRUMB_SEP)
    if len(levels) >= 3:
        heading_path = (
            f"{levels[0]}{HEADING_BREADCRUMB_SEP}…{HEADING_BREADCRUMB_SEP}{levels[-1]}"
        )
    # Backstop: enforce the cap for token-dense short paths (and any collapsed
    # form that is still over budget). Prefer a trailing ellipsis when it fits,
    # but re-encode each candidate because custom/BPE tokenizers may tokenize
    # the suffix differently when it is appended to decoded prefix text.
    tokens = tokenizer.encode(heading_path)
    if len(tokens) > max_tokens:
        ellipsis = "…"
        ellipsis_token_count = len(tokenizer.encode(ellipsis))
        if ellipsis_token_count <= max_tokens:
            for keep in range(max_tokens - ellipsis_token_count, -1, -1):
                candidate = tokenizer.decode(tokens[:keep]).rstrip() + ellipsis
                if len(tokenizer.encode(candidate)) <= max_tokens:
                    return candidate
        for keep in range(max_tokens, -1, -1):
            candidate = tokenizer.decode(tokens[:keep]).rstrip()
            if len(tokenizer.encode(candidate)) <= max_tokens:
                return candidate
        return ""
    return heading_path


def _truncate_vdb_content(content: str, global_config: dict, content_label: str) -> str:
    """Clamp vector-store payload size to stay under embedding limits.

    Uses the safe ``Tokenizer.truncate_by_token_limit`` contract: the result is
    independently re-encoded and verified to actually fit ``threshold``, so
    this no longer needs (or applies) a fixed heuristic safety margin — the
    old ``decode(tokens[:k])`` here had no such verification and was not
    guaranteed to round-trip back to <= ``threshold`` tokens for every
    tokenizer/content combination.
    """
    if not content:
        return content

    embedding_token_limit = global_config.get("embedding_token_limit")
    tokenizer: Tokenizer | None = global_config.get("tokenizer")
    if embedding_token_limit is None or tokenizer is None:
        return content

    threshold = int(embedding_token_limit)
    if threshold <= 0:
        return content

    try:
        span = tokenizer.truncate_by_token_limit(content, threshold)
    except TokenBudgetError as e:
        # Only possible if even the first Unicode code point can't fit the
        # embedding model's own context limit — not a real-world budget.
        logger.error(
            "%s VDB content cannot fit embedding limit %d: %s",
            content_label,
            threshold,
            e,
        )
        raise

    if span.end == len(content):
        return content

    truncated_content = content[span.start : span.end]
    logger.warning(
        "%s VDB content truncated to %d tokens (embedding limit: %d, "
        "original length %d chars)",
        content_label,
        span.token_count,
        threshold,
        len(content),
    )
    return truncated_content


_MM_DISPLAY_NAME_PATTERN = re.compile(
    r"^\[(?:Image|Table|Equation) Name\](.+)$",
    flags=re.MULTILINE,
)


def _parse_mm_display_name(content: str, fallback: str) -> str:
    """Return the friendly name embedded in a multimodal chunk.

    Matches the leading ``[Image Name]…`` / ``[Table Name]…`` /
    ``[Equation Name]…`` segment produced by
    ``LightRAG._build_mm_chunks_from_sidecars`` — the producer-side
    contract is documented in that function's ``_render`` helper. Falls
    back to the sidecar id when the segment is missing or empty so
    callers never end up with a blank label.
    """
    if content:
        match = _MM_DISPLAY_NAME_PATTERN.search(content)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return fallback


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    separator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
    truncation_tally: TokenLimitTruncationTally | None = None,
    truncation_write_ahead: Callable[[TokenLimitTruncationTally], Awaitable[None]]
    | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    # Still sanitize: descriptions read back from existing graph nodes (or
    # injected by non-extraction producers) may carry XML-illegal control
    # characters that would crash the GraphML flush downstream.
    if len(description_list) == 1:
        return sanitize_text_for_encoding(description_list[0]), False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list while periodically yielding so
        # a large merge does not monopolize the event loop in single-worker mode.
        total_tokens = 0
        # Tokenize each description once; the chunk-building pass below
        # reuses these counts instead of re-encoding the same strings.
        desc_token_counts = []
        for i, desc in enumerate(current_list, start=1):
            tokens = len(tokenizer.encode(desc))
            desc_token_counts.append(tokens)
            total_tokens += tokens
            await _cooperative_yield(i, every=32)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                # Sanitize for the same reason as the single-description
                # early return above: inputs merged from pre-existing graph
                # data are not guaranteed XML-safe.
                final_description = sanitize_text_for_encoding(
                    separator.join(current_list)
                )
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                    truncation_tally=truncation_tally,
                    truncation_write_ahead=truncation_write_ahead,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list.
        # Reuse the per-description token counts computed above instead of
        # re-encoding every description a second time.
        for i, (desc, desc_tokens) in enumerate(
            zip(current_list, desc_token_counts), start=1
        ):
            await _cooperative_yield(i, every=32)

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # current_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                    truncation_tally=truncation_tally,
                    truncation_write_ahead=truncation_write_ahead,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
    truncation_tally: TokenLimitTruncationTally | None = None,
    truncation_write_ahead: Callable[[TokenLimitTruncationTally], Awaitable[None]]
    | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        description_type: Type of the descriptions being summarized (entity or relation)
        description_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses
        truncation_tally: Optional accumulator for token-limit truncation. A
            cut-off summary silently becomes the entity's/relation's persisted
            description, so it belongs in the same operator-visible record as
            truncated extraction.
        truncation_write_ahead: Optional hook awaited right after a truncation
            is recorded, BEFORE the summary is returned to the caller that will
            write it into the graph. Lets an additive caller (the custom-chunk
            path, whose resume never purges a failed attempt's mutations) make
            the record durable ahead of the mutation it warns about, so even a
            hard crash cannot leave a truncated summary in the graph with no
            journaled evidence. A hook failure propagates and fails this
            entity/relation — fail-closed into the caller's normal failure
            path.

    Returns:
        Summarized description string, never empty: an empty answer raises
        :class:`~lightrag.exceptions.EmptySummaryResponseError`.
    """
    use_llm_func: callable = global_config["role_llm_funcs"]["extract"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=DEFAULT_SUMMARY_PRIORITY)

    addon_params = global_config.get("addon_params") or {}
    language = global_config.get("_resolved_summary_language")
    if language is None:
        language = addon_params.get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = await atruncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        separator="\n",
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
        llm_cache_identity=get_llm_cache_identity(global_config, "extract"),
    )

    # Check BEFORE sanitizing: sanitize_text_for_encoding rebuilds a plain str
    # and drops the TruncatedResponse marker that remove_think_tags preserved.
    if is_truncated_response(summary):
        subject = f"{description_type}:{description_name}"
        logger.warning(
            f"Token-limit truncation while summarizing descriptions for {subject}"
        )
        if truncation_tally is not None:
            truncation_tally.record("summary", subject)
            if truncation_write_ahead is not None:
                await truncation_write_ahead(truncation_tally)

    # The LLM response is the only description path that bypasses
    # extraction-time sanitization; control chars / surrogates left here
    # would later break GraphML (XML) serialization on write. Strip them
    # at the source, symmetric with how extracted descriptions are cleaned.
    summary = sanitize_text_for_encoding(summary)

    # The summary replaces every fragment it merges, so storing an empty one
    # would erase them. Fail instead: the stored description survives and the
    # document can be retried. An answer that is empty after think-tag
    # removal is never cached (save_to_cache skips falsy content), so the
    # retry asks the LLM again.
    # Accepted residue: an answer of control characters only is cached (it is
    # non-empty until sanitize_text_for_encoding above runs), so a retry
    # replays it and fails again, loudly, until the LLM cache is cleared.
    if not summary.strip():
        message = (
            f"Received empty summary for {description_type}:{description_name} "
            f"({len(description_list)} descriptions): the model returned no "
            "answer text (empty, or reasoning only); if it is a thinking model, "
            "consider disabling thinking mode for the extract role"
        )
        logger.error(message)
        raise EmptySummaryResponseError(message)

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary


# JavaScript Object.prototype member names. When such a value is stored as an
# entity_type it crashes the WebUI knowledge-graph view: the type-color resolver
# indexes a plain object with it and the Legend later calls a string method on
# the resulting non-string map key (see graphColor.ts in the WebUI).
# They are never valid entity types, so reject them at the extraction source.
_RESERVED_ENTITY_TYPES = frozenset({"__proto__", "constructor", "prototype"})


def _normalize_and_validate_entity_type(entity_type: str, context: str) -> str | None:
    """Validate and normalize a sanitized entity_type extracted from an LLM.

    Shared by the text (delimited-record) and JSON extraction paths so both
    reject the same values. Expects ``entity_type`` to have already been passed
    through ``sanitize_and_normalize_extracted_text``.

    Returns the cleaned, space-stripped, lowercased type, or ``None`` when the
    entity must be dropped:
    - empty, or containing a structural delimiter character;
    - all-empty after a comma split;
    - a reserved JavaScript prototype member name (see ``_RESERVED_ENTITY_TYPES``).

    ``context`` is a short caller-supplied descriptor included in warning logs.
    """
    if not entity_type.strip() or any(
        char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
    ):
        logger.warning(f"Entity extraction error: invalid entity type in {context}")
        return None

    # Handle comma-separated entity types by finding the first non-empty token
    if "," in entity_type:
        original = entity_type
        tokens = [t.strip() for t in entity_type.split(",")]
        non_empty = [t for t in tokens if t]
        if not non_empty:
            logger.warning(
                f"Entity extraction error: all tokens empty after comma-split in {context}: '{original}'"
            )
            return None
        entity_type = non_empty[0]
        logger.warning(
            f"Entity type contains comma, taking first non-empty token in {context}: '{original}' -> '{entity_type}'"
        )

    # Remove spaces and convert to lowercase
    entity_type = entity_type.replace(" ", "").lower()

    if entity_type in _RESERVED_ENTITY_TYPES:
        logger.warning(
            f"Entity extraction error: reserved entity type '{entity_type}' rejected in {context}"
        )
        return None

    return entity_type


def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 fields on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = normalize_entity_name(
            record_attributes[1], final_space_pass=False
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )
        entity_type = _normalize_and_validate_entity_type(
            entity_type, f"record {record_attributes}"
        )
        if entity_type is None:
            return None

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(
            record_attributes[3], multiline=True
        )

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if (
        len(record_attributes) != 5 or "relation" not in record_attributes[0]
    ):  # treat "relationship" and "relation" interchangeable
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on RELATION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = normalize_entity_name(record_attributes[1], final_space_pass=False)
        target = normalize_entity_name(record_attributes[2], final_space_pass=False)

        # Validate entity names after all cleaning steps
        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(
            record_attributes[4], multiline=True
        )
        if not edge_description.strip():
            logger.warning(
                f"Relationship extraction error: empty description for relation '{source}'~'{target}' in chunk '{chunk_key}'"
            )
            return None

        edge_source_id = chunk_key
        # Prompt text rows are 5 fields with no weight; keep default 1.0.
        weight = 1.0

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


def _normalize_text_extraction_record_attributes(
    record_attributes: list[str], chunk_key: str
) -> list[str]:
    """Recover the known text-mode failure where relation rows use the entity prefix."""

    if len(record_attributes) != 5:
        return record_attributes

    prefix = record_attributes[0].strip().lower()
    if "entity" not in prefix or "relation" in prefix:
        return record_attributes

    logger.warning(
        "Recovering mis-prefixed relation: `%s` ~ `%s`",
        record_attributes[1],
        record_attributes[2],
    )
    normalized = list(record_attributes)
    normalized[0] = "relation"
    return normalized


def _truncation_cause_suffix(result: Any) -> str:
    """Name token-limit truncation as the cause of a failed extraction parse.

    An undelimited / unparseable extraction response has two very different
    causes — a model that ignored the output contract, and a model that was cut
    off mid-answer — and only the second is fixed by raising the output budget.
    ``TruncatedResponse`` survives ``remove_think_tags``, so the parsers can
    tell them apart instead of leaving the operator to correlate this warning
    with a separate truncation line by chunk id.
    """
    if not is_truncated_response(result):
        return ""
    return " (response was truncated by the model's output token limit)"


def _looks_like_json_extraction_result(result: str) -> bool:
    """Return True for raw or fenced JSON extraction responses."""

    stripped = result.strip()
    if not stripped:
        return False

    if stripped.startswith(("{", "[")):
        return True

    if stripped.startswith("```"):
        return strip_markdown_code_fence(stripped).strip().startswith(("{", "["))

    return False


async def _process_json_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    parsed: dict[str, Any] | None = None,
) -> tuple[dict, dict]:
    """Process a JSON-formatted extraction result from LLM.

    This function parses the LLM response as JSON and extracts entities and relationships.
    It uses :func:`tolerant_load_json_dict` to recover JSON from fenced,
    prose-wrapped, or slightly malformed responses from weaker models.

    Args:
        result: The JSON extraction result from LLM
        chunk_key: The chunk key for source tracking
        timestamp: The timestamp for the extraction
        file_path: The file path for citation
        parsed: Payload already recovered by :func:`tolerant_load_json_dict`,
            for callers that need the parse outcome themselves rather than
            only the extracted entities -- an empty result does not say
            whether the payload was unrecoverable or merely carried nothing,
            and cache rebuild has to tell those apart. ``None`` means "not
            supplied" and is unambiguous: the helper always returns a dict,
            so an unrecoverable payload still arrives here as ``{}``.

    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    # tolerant_load_json_dict strips fences, tolerates leading/trailing prose
    # (including trailing braces), and repairs the malformed-object slips
    # json_repair fixes. It never raises and returns {} for unparseable input or
    # a top-level array (callers require exactly one object). Note: because the
    # helper absorbs the underlying parse exception, only this single "empty or
    # unrecoverable" warning is logged — the low-level decode error is no longer
    # surfaced.
    if parsed is None:
        parsed = tolerant_load_json_dict(result)
    if not parsed:
        logger.warning(
            f"{chunk_key}: JSON extraction result is empty or unrecoverable"
            f"{_truncation_cause_suffix(result)}"
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # Models quoting LaTeX in descriptions routinely under-escape backslashes
    # ("\frac" is valid JSON meaning form feed + "rac"); restore the zero-risk
    # cases before sanitization would otherwise delete the control characters
    # and leave a maimed formula. Covers initial extraction, gleaning, and
    # cache-rebuild — all three flow through this parser.
    parsed = repair_vlm_json_escape_damage_nested(parsed, context=chunk_key)

    # Process entities
    entities_list = parsed.get("entities", [])
    if not isinstance(entities_list, list):
        logger.warning(
            f"{chunk_key}: 'entities' field is not a list in JSON extraction result"
        )
        entities_list = []

    for entity_data in entities_list:
        if not isinstance(entity_data, dict):
            continue

        try:
            entity_name = normalize_entity_name(
                str(entity_data.get("name", "")), final_space_pass=False
            )
            if not entity_name or not entity_name.strip():
                logger.info(
                    f"{chunk_key}: Empty entity name found after sanitization in JSON result"
                )
                continue

            entity_type = sanitize_and_normalize_extracted_text(
                str(entity_data.get("type", "")), remove_inner_quotes=True
            )
            entity_type = _normalize_and_validate_entity_type(
                entity_type, f"{chunk_key}: entity '{entity_name}'"
            )
            if entity_type is None:
                continue

            entity_description = sanitize_and_normalize_extracted_text(
                str(entity_data.get("description", "")), multiline=True
            )
            if not entity_description.strip():
                logger.warning(
                    f"{chunk_key}: Empty description for entity '{entity_name}'"
                )
                continue

            truncated_name = _finalize_extracted_identifier(
                entity_name,
                chunk_key,
                "Entity name",
            )

            node_data = dict(
                entity_name=truncated_name,
                entity_type=entity_type,
                description=entity_description,
                source_id=chunk_key,
                file_path=file_path,
                timestamp=timestamp,
            )
            maybe_nodes[truncated_name].append(node_data)

        except Exception as e:
            logger.warning(
                f"{chunk_key}: Failed to process entity from JSON result: {e}"
            )
            continue

    # Process relationships
    relationships_list = parsed.get("relationships", [])
    if not isinstance(relationships_list, list):
        logger.warning(
            f"{chunk_key}: 'relationships' field is not a list in JSON extraction result"
        )
        relationships_list = []

    for rel_data in relationships_list:
        if not isinstance(rel_data, dict):
            continue

        try:
            source = normalize_entity_name(
                str(rel_data.get("source", "")), final_space_pass=False
            )
            target = normalize_entity_name(
                str(rel_data.get("target", "")), final_space_pass=False
            )

            if not source:
                logger.info(
                    f"{chunk_key}: Empty source entity in JSON relationship result"
                )
                continue
            if not target:
                logger.info(
                    f"{chunk_key}: Empty target entity in JSON relationship result"
                )
                continue
            if source == target:
                logger.debug(f"{chunk_key}: Source and target are the same: '{source}'")
                continue

            edge_keywords = sanitize_and_normalize_extracted_text(
                str(rel_data.get("keywords", "")), remove_inner_quotes=True
            )
            edge_keywords = edge_keywords.replace("，", ",")

            edge_description = sanitize_and_normalize_extracted_text(
                str(rel_data.get("description", "")), multiline=True
            )

            if not edge_description.strip():
                logger.warning(
                    f"{chunk_key}: Empty description for relationship '{source}' ~ '{target}', skipping"
                )
                continue

            truncated_source = _finalize_extracted_identifier(
                source,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _finalize_extracted_identifier(
                target,
                chunk_key,
                "Relation entity",
            )
            # The cap and the final space pass can make the endpoints equal.
            if truncated_source == truncated_target:
                logger.debug(
                    f"{chunk_key}: Source and target are the same: '{truncated_source}'"
                )
                continue

            edge_data = dict(
                src_id=truncated_source,
                tgt_id=truncated_target,
                weight=1.0,
                description=edge_description,
                keywords=edge_keywords,
                source_id=chunk_key,
                file_path=file_path,
                timestamp=timestamp,
            )
            maybe_edges[(truncated_source, truncated_target)].append(edge_data)

        except Exception as e:
            logger.warning(
                f"{chunk_key}: Failed to process relationship from JSON result: {e}"
            )
            continue

    return dict(maybe_nodes), dict(maybe_edges)


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],
    relationships_to_rebuild: dict[tuple[str, str], list[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    rebuild_policy: KGRebuildPolicy = "best_effort",
    doc_status_storage: BaseKVStorage | None = None,
) -> KGRebuildReport:
    """Rebuild entities and relations from their surviving chunks' extractions.

    Each surviving chunk contributes the records of its extraction cache rows,
    attributed through the chunk's ``llm_cache_list``. Relations are rebuilt
    first and entities only after every relation succeeded, so an entity known
    only through its edges reads edges that are already rebuilt. Any per-object
    failure is raised after its siblings have drained; callers treat it as a
    retryable failure of the whole rebuild.

    Policies:

    * ``best_effort`` (document deletion and resume purge): a surviving chunk
      whose INITIAL extraction is missing, is a gleaning row, or does not
      parse is re-extracted through ``extract_entities`` before any object is
      rebuilt, which calls the LLM; a re-extraction failure raises. A missing
      or unparseable gleaning row is accepted as gleaning that did not run.
      Chunks written by ``ainsert_custom_kg`` are never re-extracted.
      Cached records go through ``kg_extraction_validator`` like fresh ones.
    * ``rollback`` (custom-chunk patch rollback): never calls the extraction
      LLM; missing or unusable extraction data is recorded in the report.

    Under both policies, an object no surviving chunk's records name is kept:
    its semantic fields are preserved and its ``source_id`` / ``file_path``
    are rewritten to the surviving chunks. Under ``best_effort`` an entity the
    surviving records name only as a relation endpoint is re-summarized from
    its rebuilt edges instead. Rules, residues and the operator recovery
    procedure: *Deletion rebuild evidence* in
    ``docs/design/PurgeRecoveryContract.md``.

    Args:
        entities_to_rebuild: entity name -> remaining chunk ids.
        relationships_to_rebuild: (src, tgt) -> remaining chunk ids.
        global_config: must carry what ``extract_entities`` needs when the
            ``best_effort`` policy may re-extract (``LightRAG._build_global_config``).
        entity_chunks_storage / relation_chunks_storage: chunk tracking.
        rebuild_policy: see above.
        doc_status_storage: tells ``ainsert_custom_kg`` chunks from pipeline
            chunks on backends that do not keep the chunk row's ``source_id``
            (see ``_find_custom_kg_chunks``). Pass it whenever the
            ``best_effort`` policy may re-extract.
    """
    if rebuild_policy not in ("best_effort", "rollback"):
        raise ValueError(f"Unsupported KG rebuild policy: {rebuild_policy}")

    report = KGRebuildReport()
    if not entities_to_rebuild and not relationships_to_rebuild:
        return report

    # Operation-scoped status logger for per-item FAILURE messages: errors must
    # stay visible in the pipeline history (the WebUI reads it), while per-item
    # success/completion detail goes to the backend log only (volume control).
    status_logger = PipelineStatusLogger(pipeline_status)

    async def _publish(message: str, level: int = logging.INFO) -> None:
        logger.log(level, message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = message
                append_pipeline_history(pipeline_status, message)

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    report.referenced_chunk_ids.update(all_referenced_chunk_ids)

    await _publish(
        f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk "
        "extractions (parallel processing)"
    )

    extract_rows, chunk_data_by_id = await _load_chunk_extract_rows(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    # chunk_id -> {entity_name: [entity_data]} / {(src, tgt): [relationship_data]}
    chunk_entities: dict[str, dict[str, list[dict]]] = {}
    chunk_relationships: dict[str, dict[tuple[str, str], list[dict]]] = {}
    # chunk_id -> unparseable initial row to delete before re-extracting it
    to_reextract: dict[str, str | None] = {}
    custom_kg_chunk_ids = (
        await _find_custom_kg_chunks(chunk_data_by_id, extract_rows, doc_status_storage)
        if rebuild_policy == "best_effort"
        else set()
    )

    kg_extraction_validator = global_config.get("kg_extraction_validator")

    async def _finish_cached_chunk(chunk_id, chunk_data, nodes, edges) -> None:
        # Cached rows hold the raw LLM output, from before the validator ran
        # (it runs after the cache write), so they get the same hook and the
        # same core-generated multimodal records as a fresh extraction. A hook
        # failure fails the rebuild, which is retryable.
        if kg_extraction_validator is not None:
            nodes, edges = await _apply_kg_extraction_validator(
                kg_extraction_validator, chunk_id, chunk_data, nodes, edges
            )
        _inject_multimodal_records(chunk_id, chunk_data, nodes, edges)
        chunk_entities[chunk_id] = nodes
        chunk_relationships[chunk_id] = edges

    for chunk_id in sorted(all_referenced_chunk_ids):
        chunk_data = chunk_data_by_id.get(chunk_id)
        if chunk_data is None:
            # No chunk text: nothing to parse or re-extract from.
            report.missing_cache_chunk_ids.add(chunk_id)
            continue
        entries = extract_rows.get(chunk_id, [])
        nodes: dict[str, list[dict]] = defaultdict(list)
        edges: dict[tuple[str, str], list[dict]] = defaultdict(list)

        if rebuild_policy == "rollback":
            present = [(key, row) for key, row in entries if row is not None]
            if not present:
                report.missing_cache_chunk_ids.add(chunk_id)
                continue
            try:
                for _, row in present:
                    _merge_chunk_records(
                        nodes,
                        edges,
                        *await _parse_extract_row(
                            text_chunks_storage, row, chunk_id, chunk_data
                        ),
                    )
            except Exception as e:
                report.unusable_cache_chunk_ids.add(chunk_id)
                status_message = f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
                logger.warning(status_message)
                status_logger.log(status_message)
            await _finish_cached_chunk(chunk_id, chunk_data, nodes, edges)
            continue

        if chunk_id in custom_kg_chunk_ids:
            # User-supplied content: never sent to the LLM on someone else's
            # deletion. Its objects fall back like any unattested object.
            report.missing_cache_chunk_ids.add(chunk_id)
            continue
        initial_key, initial_row = entries[0] if entries else (None, None)
        if initial_row is None or _is_gleaning_extract_row(initial_row):
            to_reextract[chunk_id] = None
            continue
        try:
            _merge_chunk_records(
                nodes,
                edges,
                *await _parse_extract_row(
                    text_chunks_storage, initial_row, chunk_id, chunk_data
                ),
            )
        except Exception as e:
            logger.warning(
                f"Unparseable initial extraction row {initial_key} for chunk "
                f"{chunk_id}; re-extracting: {e}"
            )
            to_reextract[chunk_id] = initial_key
            continue
        for gleaning_key, gleaning_row in entries[1:]:
            if gleaning_row is None:
                continue  # lost gleaning: accepted, see the contract
            try:
                _merge_chunk_records(
                    nodes,
                    edges,
                    *await _parse_extract_row(
                        text_chunks_storage, gleaning_row, chunk_id, chunk_data
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"Ignoring unparseable gleaning row {gleaning_key} for chunk "
                    f"{chunk_id}: {e}"
                )
        await _finish_cached_chunk(chunk_id, chunk_data, nodes, edges)

    if to_reextract:
        reextracted = await _reextract_chunks(
            to_reextract,
            chunk_data_by_id,
            extract_rows,
            text_chunks_storage=text_chunks_storage,
            llm_response_cache=llm_response_cache,
            global_config=global_config,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            publish=_publish,
        )
        for chunk_id, (nodes, edges) in reextracted.items():
            chunk_entities[chunk_id] = nodes
            chunk_relationships[chunk_id] = edges
        report.reextracted_chunk_ids.update(reextracted)

    if report.missing_cache_chunk_ids:
        await _publish(
            f"Extraction evidence unavailable for "
            f"{len(report.missing_cache_chunk_ids)} of "
            f"{len(all_referenced_chunk_ids)} referenced chunk(s) (no cache row "
            "to read, or a custom KG / missing chunk that is never re-extracted); "
            "rebuild may use structural fallback",
            logging.WARNING,
        )

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0
    structural_fallback = rebuild_policy == "rollback"
    workspace = global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    degraded = await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                        chunk_data_by_id=chunk_data_by_id,
                        structural_fallback=structural_fallback,
                        summarize_from_edges=_named_as_endpoint(
                            entity_name, chunk_ids, chunk_relationships
                        ),
                    )
                    rebuilt_entities_count += 1
                    if degraded:
                        report.degraded_entities[entity_name] = list(chunk_ids)
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.error(status_message)
                    status_logger.log(status_message)
                    raise

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    degraded = await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                        chunk_data_by_id=chunk_data_by_id,
                        structural_fallback=structural_fallback,
                    )
                    rebuilt_relationships_count += 1
                    if degraded:
                        report.degraded_relationships[(src, tgt)] = list(chunk_ids)
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.error(status_message)
                    status_logger.log(status_message)
                    raise

    await _publish(
        f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and "
        f"{len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    )

    def _summary(prefix: str) -> str:
        message = (
            f"{prefix}: {rebuilt_entities_count} entities and "
            f"{rebuilt_relationships_count} relationships rebuilt successfully."
        )
        if failed_entities_count > 0 or failed_relationships_count > 0:
            message += (
                f" Failed: {failed_entities_count} entities, "
                f"{failed_relationships_count} relationships."
            )
        return message

    try:
        # Phase 1: relations. On any failure every sibling is cancelled and
        # drained before the first exception propagates, and no entity is
        # touched.
        await wait_tasks_with_drain(
            [
                asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
                for (src, tgt), chunk_ids in relationships_to_rebuild.items()
            ],
            context="KG rebuild (relations)",
        )
        # Phase 2: entities, which may now read the already-rebuilt edges.
        await wait_tasks_with_drain(
            [
                asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
                for entity_name, chunk_ids in entities_to_rebuild.items()
            ],
            context="KG rebuild (entities)",
        )
    except Exception:
        await _publish(_summary("KG rebuild failed"), logging.ERROR)
        raise

    await _publish(_summary("KG rebuild completed"))

    if rebuild_policy == "best_effort" and (
        report.degraded_entities or report.degraded_relationships
    ):
        kept = [f"`{name}`" for name in report.degraded_entities] + [
            f"`{src}`~`{tgt}`" for src, tgt in report.degraded_relationships
        ]
        sample = ", ".join(kept[:_REBUILD_SUMMARY_SAMPLE])
        if len(kept) > _REBUILD_SUMMARY_SAMPLE:
            sample += f" (+{len(kept) - _REBUILD_SUMMARY_SAMPLE} more)"
        await _publish(
            f"Kept {len(report.degraded_entities)} entity(ies) and "
            f"{len(report.degraded_relationships)} relation(s) that no surviving "
            "chunk's extraction names; their provenance was rewritten to the "
            f"surviving chunks and their description left unchanged: {sample}",
            logging.WARNING,
        )

    if report.has_warnings:
        logger.warning(
            "KG rebuild completed with degraded recovery: "
            "missing_cache_chunks=%d unusable_cache_chunks=%d "
            "entities=%d relationships=%d",
            len(report.missing_cache_chunk_ids),
            len(report.unusable_cache_chunk_ids),
            len(report.degraded_entities),
            len(report.degraded_relationships),
        )

    return report


def _named_as_endpoint(
    entity_name: str,
    chunk_ids: list[str],
    chunk_relationships: dict[str, dict[tuple[str, str], list[dict]]],
) -> bool:
    """True when a surviving chunk's relation records name the entity.

    Without such a record nothing attests the entity under this name (a
    rename, a merge, an older normalization, custom KG content, or name drift
    after re-extraction), and summarizing its stored edges would only replace
    its description with theirs.
    """
    return any(
        entity_name in edge_key
        for chunk_id in chunk_ids
        for edge_key in chunk_relationships.get(chunk_id, ())
    )


# How many object names / document ids a rebuild summary line names before it
# switches to a count: the pipeline history is a bounded ring.
_REBUILD_SUMMARY_SAMPLE = 10

# ``json.dumps`` of the gleaning call's history: a non-empty list of message
# dicts, serialized on one line with the default separators.
_GLEANING_HISTORY_PREFIX = '[{"role": '


def _is_gleaning_extract_row(row: dict[str, Any]) -> bool:
    """True when an extract cache row holds a gleaning response.

    ``use_llm_func_with_cache`` stores the prompt as user prompt, system prompt
    and serialized history joined by newlines, and only the gleaning call
    passes history. ``json.dumps`` escapes the newlines inside the messages, so
    the history is exactly the prompt's last line. A row without a stored
    prompt is treated as an initial extraction.
    """
    prompt = row.get("original_prompt")
    if not isinstance(prompt, str) or "\n" not in prompt:
        return False
    tail = prompt.rsplit("\n", 1)[1]
    if not tail.startswith(_GLEANING_HISTORY_PREFIX):
        return False
    try:
        history = json.loads(tail)
    except ValueError:
        return False
    return (
        isinstance(history, list)
        and bool(history)
        and all(isinstance(msg, dict) and "role" in msg for msg in history)
    )


async def _find_custom_kg_chunks(
    chunk_data_by_id: dict[str, dict[str, Any]],
    extract_rows: dict[str, list[tuple[str, dict[str, Any] | None]]],
    doc_status_storage: BaseKVStorage | None,
) -> set[str]:
    """Chunks written by ``ainsert_custom_kg``, which are never re-extracted.

    Only chunks without any extract key qualify. Such a chunk is custom KG
    when its row carries the caller's ``source_id`` alias, or when its
    ``full_doc_id`` names no ``doc_status`` row: pipeline chunks always belong
    to a document with a status record, while ``ainsert_custom_kg`` creates
    none. The second test is what works on backends that store only the
    pipeline's chunk columns (PostgreSQL drops ``source_id``). A read failure
    raises rather than sending user-supplied text to the LLM.
    """
    custom: set[str] = set()
    undecided: dict[str, str] = {}
    for chunk_id, row in chunk_data_by_id.items():
        if extract_rows.get(chunk_id):
            continue
        doc_id = row.get("full_doc_id")
        if "source_id" in row or not isinstance(doc_id, str) or not doc_id:
            custom.add(chunk_id)
        elif doc_status_storage is not None:
            undecided[chunk_id] = doc_id
    if undecided:
        doc_ids = sorted(set(undecided.values()))
        statuses = await doc_status_storage.get_by_ids(doc_ids)
        known = {doc_id for doc_id, status in zip(doc_ids, statuses) if status}
        custom.update(c for c, d in undecided.items() if d not in known)
    return custom


async def _parse_extract_row(
    text_chunks_storage: BaseKVStorage,
    row: dict[str, Any],
    chunk_id: str,
    chunk_data: dict[str, Any],
) -> tuple[dict, dict]:
    """Parse one extract cache row for ``chunk_id``; raise if it is unusable."""
    result = row.get("return")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("cache row holds no extraction text")
    return await _rebuild_from_extraction_result(
        text_chunks_storage=text_chunks_storage,
        chunk_id=chunk_id,
        extraction_result=result,
        timestamp=row.get("create_time", 0),
        chunk_data=chunk_data,
    )


def _merge_chunk_records(
    nodes: dict[str, list[dict]],
    edges: dict[tuple[str, str], list[dict]],
    new_nodes: dict[str, list[dict]],
    new_edges: dict[tuple[str, str], list[dict]],
) -> None:
    """Fold one row's records into a chunk's records.

    Same rule as ``extract_entities`` uses for gleaning: for a name already
    present, the version with the longer description wins.
    """
    for target, incoming in ((nodes, new_nodes), (edges, new_edges)):
        for key, records in incoming.items():
            if not records:
                continue
            existing = target.get(key)
            if not existing:
                target[key] = list(records)
                continue
            existing_len = len(existing[0].get("description", "") or "")
            new_len = len(records[0].get("description", "") or "")
            if new_len > existing_len:
                target[key] = list(records)


async def _reextract_chunks(
    to_reextract: dict[str, str | None],
    chunk_data_by_id: dict[str, dict[str, Any]],
    extract_rows: dict[str, list[tuple[str, dict[str, Any] | None]]],
    *,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, Any],
    pipeline_status: dict | None,
    pipeline_status_lock,
    publish: Callable[..., Awaitable[None]],
) -> dict[str, tuple[dict, dict]]:
    """Run surviving chunks whose initial extraction is missing through
    ``extract_entities`` and return their records.

    Before the call, an unparseable initial row is deleted (re-extracting would
    otherwise hit it again) and the chunk's extract keys are dropped from its
    ``llm_cache_list``, so the keys the new run attaches, in call order, make
    the new initial extraction the first one again. Raises on any failure.
    """
    doc_ids = sorted(
        {
            str(chunk_data_by_id[chunk_id].get("full_doc_id") or "unknown")
            for chunk_id in to_reextract
        }
    )
    sample = ", ".join(doc_ids[:_REBUILD_SUMMARY_SAMPLE])
    if len(doc_ids) > _REBUILD_SUMMARY_SAMPLE:
        sample += f" (+{len(doc_ids) - _REBUILD_SUMMARY_SAMPLE} more)"
    await publish(
        f"Re-extracting {len(to_reextract)} surviving chunk(s) whose initial "
        f"extraction cache is missing or unusable, from document(s): {sample}. "
        "Reprocess these documents if the knowledge graph lost facts.",
        logging.WARNING,
    )

    stale_rows = [key for key in to_reextract.values() if key]
    if stale_rows:
        await llm_response_cache.delete(stale_rows)

    reset_rows: dict[str, dict[str, Any]] = {}
    for chunk_id in to_reextract:
        row = chunk_data_by_id[chunk_id]
        extract_keys = {key for key, _ in extract_rows.get(chunk_id, [])}
        cache_list = list(row.get("llm_cache_list") or [])
        kept = [key for key in cache_list if key not in extract_keys]
        if kept != cache_list:
            reset_rows[chunk_id] = {**row, "llm_cache_list": kept}
    if reset_rows:
        await text_chunks_storage.upsert(reset_rows)
        chunk_data_by_id.update(reset_rows)

    chunk_ids = list(to_reextract)
    results = await extract_entities(
        {chunk_id: chunk_data_by_id[chunk_id] for chunk_id in chunk_ids},
        global_config,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=llm_response_cache,
        text_chunks_storage=text_chunks_storage,
    )
    return dict(zip(chunk_ids, results))


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, dict[str, Any]]]:
    """Cached extraction results per chunk, for callers that need no row
    identity (the chunk-tracking repair tool).

    Returns ``(cached_results, chunk_data_by_id)``: every chunk with at least
    one usable extract row maps to its ``(extraction_result, create_time)``
    pairs sorted by ``create_time``, and the chunks are ordered by their
    earliest one. Attribution follows :func:`_load_chunk_extract_rows`.
    """
    extract_rows, chunk_data_by_id = await _load_chunk_extract_rows(
        llm_response_cache, chunk_ids, text_chunks_storage
    )
    cached_results: dict[str, list[tuple[str, int]]] = {}
    for chunk_id, rows in extract_rows.items():
        results = sorted(
            (
                (row["return"], row.get("create_time", 0))
                for _, row in rows
                if row is not None
            ),
            key=lambda result: result[1],
        )
        if results:
            cached_results[chunk_id] = results
    ordered = sorted(
        cached_results, key=lambda chunk_id: cached_results[chunk_id][0][1]
    )
    return {
        chunk_id: cached_results[chunk_id] for chunk_id in ordered
    }, chunk_data_by_id


async def _load_chunk_extract_rows(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> tuple[
    dict[str, list[tuple[str, dict[str, Any] | None]]], dict[str, dict[str, Any]]
]:
    """Load each requested chunk's extract cache rows.

    Returns ``(extract_rows, chunk_data_by_id)``. ``extract_rows`` maps every
    chunk whose row exists to its extract keys in ``llm_cache_list`` order --
    call order, so an initial extraction precedes its gleaning -- each with
    its cache row, or ``None`` when the row is gone or is not an extract row.
    A key is an extract key when its flattened form says so; a legacy key that
    does not parse counts only when its row does.

    Rows are attributed through ``llm_cache_list``, never through the row's
    own ``chunk_id``: that field names only the row's first writer, and chunks
    of different documents with identical text share one row.
    """
    chunk_data_by_id: dict[str, dict[str, Any]] = {}
    keys_by_chunk: dict[str, list[str]] = {}

    requested_chunk_ids = list(chunk_ids)
    chunk_data_list = await text_chunks_storage.get_by_ids(requested_chunk_ids)
    for chunk_id, chunk_data in zip(requested_chunk_ids, chunk_data_list):
        if chunk_data and isinstance(chunk_data, dict):
            chunk_data_by_id[chunk_id] = chunk_data
            keys_by_chunk[chunk_id] = [
                key
                for key in dict.fromkeys(chunk_data.get("llm_cache_list") or [])
                if _extract_key_type(key) in ("extract", None)
            ]
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    cache_ids = list(dict.fromkeys(k for keys in keys_by_chunk.values() for k in keys))
    rows_by_key: dict[str, dict[str, Any]] = {}
    if cache_ids:
        cache_data_list = await llm_response_cache.get_by_ids(cache_ids)
        rows_by_key = {
            key: row
            for key, row in zip(cache_ids, cache_data_list)
            if isinstance(row, dict) and row.get("cache_type") == "extract"
        }

    extract_rows: dict[str, list[tuple[str, dict[str, Any] | None]]] = {}
    for chunk_id, keys in keys_by_chunk.items():
        extract_rows[chunk_id] = [
            (key, rows_by_key.get(key))
            for key in keys
            if _extract_key_type(key) == "extract" or key in rows_by_key
        ]

    logger.info(
        f"Found {len(rows_by_key)} extraction cache rows for "
        f"{sum(1 for rows in extract_rows.values() if any(r for _, r in rows))} "
        f"of {len(chunk_ids)} chunks"
    )
    return extract_rows, chunk_data_by_id


def _extract_key_type(cache_key: str) -> str | None:
    """The cache type a flattened key names, or None for a legacy key."""
    parsed = parse_cache_key(cache_key)
    return parsed[1] if parsed else None


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
    warn_on_missing_completion_delimiter: bool = True,
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        completion_delimiter (str): Delimiter for completion
        warn_on_missing_completion_delimiter (bool): Whether a missing
            completion delimiter is worth a WARNING. Callers that already
            reported the failure they are recovering from pass False so the
            same failure is not announced twice; the missing delimiter is
            still logged at DEBUG level for diagnosis.
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if completion_delimiter not in result:
        message = (
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
            f"{_truncation_cause_suffix(result)}"
        )
        if warn_on_missing_completion_delimiter:
            logger.warning(message)
        else:
            logger.debug(message)

    # Split LLL output result to records by "\n"
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to separate record instead of "\n"
    fixed_records = []
    for i, record in enumerate(records, start=1):
        record = record.strip()
        if record is None:
            continue
        entity_marker = f"{tuple_delimiter}entity{tuple_delimiter}"
        entity_records = split_string_by_multi_markers(record, [entity_marker])
        # Every piece after the first, and the first one too when the line
        # opens with the marker (the split drops the empty piece before it),
        # lost its "entity<|#|>" prefix to the split. Put the whole prefix back
        # by position, not by peeking at the text: a bare "entity<|" leaves the
        # record one field short, and a name that itself starts with "entity"
        # or "relation" must not decide what the record is.
        first_piece_split = record.startswith(entity_marker)
        for idx, entity_record in enumerate(entity_records):
            if idx > 0 or first_piece_split:
                entity_record = f"entity{tuple_delimiter}{entity_record}"
            elif not entity_record.startswith(
                "entity"
            ) and not entity_record.startswith("relation"):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            # entity_record now starts with a record keyword, never with a
            # marker, so exactly the pieces after the first lost one.
            for rel_idx, entity_relation_record in enumerate(entity_relation_records):
                if rel_idx > 0 or (
                    not entity_relation_record.startswith("entity")
                    and not entity_relation_record.startswith("relation")
                ):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records.append(entity_relation_record)
        await _cooperative_yield(i, every=8)

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record separators instead new-line"
        )

    delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
    delimiter_core_lower = delimiter_core.lower()
    for i, record in enumerate(fixed_records, start=1):
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core_lower:
            # change delimiter_core to lower case, and fix again
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core_lower, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])
        record_attributes = _normalize_text_extraction_record_attributes(
            record_attributes, chunk_key
        )

        # Try to parse as entity
        entity_data = _handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if entity_data is not None:
            truncated_name = _finalize_extracted_identifier(
                entity_data["entity_name"],
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            await _cooperative_yield(i, every=8)
            continue

        # Try to parse as relationship
        relationship_data = _handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if relationship_data is not None:
            truncated_source = _finalize_extracted_identifier(
                relationship_data["src_id"],
                chunk_key,
                "Relation entity",
            )
            truncated_target = _finalize_extracted_identifier(
                relationship_data["tgt_id"],
                chunk_key,
                "Relation entity",
            )
            # The cap and the final space pass can make the endpoints equal.
            if truncated_source == truncated_target:
                logger.debug(
                    f"Relationship source and target are the same in: {record_attributes}"
                )
            else:
                relationship_data["src_id"] = truncated_source
                relationship_data["tgt_id"] = truncated_target
                maybe_edges[(truncated_source, truncated_target)].append(
                    relationship_data
                )
        await _cooperative_yield(i, every=8)

    return dict(maybe_nodes), dict(maybe_edges)


async def _rebuild_from_extraction_result(
    text_chunks_storage: BaseKVStorage,
    extraction_result: str,
    chunk_id: str,
    timestamp: int,
    chunk_data: dict[str, Any] | None = None,
) -> tuple[dict, dict]:
    """Parse cached extraction result using the same logic as extract_entities.

    Supports both JSON and delimiter-based formats for backward compatibility.
    Attempts JSON parsing first; if the cached result looks like JSON (starts with '{'),
    uses the JSON parser. Otherwise, falls back to the traditional delimiter-based parser.

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path from storage
    if chunk_data is None:
        chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )

    # Auto-detect format: try JSON first if the result looks like JSON
    json_parse_reported_failure = False
    if _looks_like_json_extraction_result(extraction_result):
        # Parse once here and hand the payload down. The parse outcome, not the
        # emptiness of what was extracted from it, is what decides below whether
        # the failure being recovered from was already reported -- the two are
        # different things, and a payload carrying the wrong schema
        # ({"answer": "none found"}) extracts nothing while parsing perfectly.
        parsed = tolerant_load_json_dict(extraction_result)
        # Likely JSON format (from entity_extraction_use_json mode)
        nodes, edges = await _process_json_extraction_result(
            extraction_result,
            chunk_id,
            timestamp,
            file_path,
            parsed=parsed,
        )
        # A successful parse (tolerant_load_json_dict's own non-empty-dict
        # signal) is accepted even when it legitimately yields no nodes/edges.
        if nodes or edges or parsed:
            return nodes, edges
        # Only an unrecoverable payload can reach here, and
        # _process_json_extraction_result has already logged that failure. The
        # delimiter parser below is a speculative rescue for a payload that
        # merely looked like JSON, so its own "no completion delimiter"
        # complaint would report that same single failure a second time; it is
        # demoted to DEBUG for this call.
        json_parse_reported_failure = True

    # Fall back to traditional delimiter-based parsing
    return await _process_extraction_result(
        extraction_result,
        chunk_id,
        timestamp,
        file_path,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        warn_on_missing_completion_delimiter=not json_parse_reported_failure,
    )


def _surviving_chunk_file_paths(
    chunk_ids: list[str],
    chunk_data_by_id: dict[str, dict[str, Any]],
    global_config: dict[str, Any],
) -> list[str]:
    """Return bounded, order-preserving file paths for surviving chunks."""
    file_paths = list(
        dict.fromkeys(
            str(chunk_data_by_id[chunk_id]["file_path"])
            for chunk_id in chunk_ids
            if chunk_id in chunk_data_by_id
            and chunk_data_by_id[chunk_id].get("file_path")
        )
    )
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    if len(file_paths) <= max_file_paths:
        return file_paths

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        limited = file_paths[-max_file_paths:]
    else:
        limited = file_paths[:max_file_paths]
    placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limited.append(
        f"...{placeholder}...({limit_method} {max_file_paths}/{len(file_paths)})"
    )
    return limited


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: list[str],
    chunk_entities: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None,
    chunk_data_by_id: dict[str, dict[str, Any]] | None = None,
    structural_fallback: bool = False,
    summarize_from_edges: bool = True,
) -> bool:
    """Rebuild a single entity from cached extraction results.

    When no surviving record names the entity, it is re-summarized from its
    edges only if ``summarize_from_edges`` (and not ``structural_fallback``);
    otherwise it keeps its semantic fields and gets its provenance rewritten
    to ``chunk_ids``, and the function returns True (degraded).
    """

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return False

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str,
        entity_type: str,
        file_paths: list[str],
        source_chunk_ids: list[str],
        truncation_info: str = "",
    ):
        try:
            updated_entity_data = {
                **current_entity,
                "description": final_description,
                "entity_type": entity_type,
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }

            # Construct and verify the VDB payload BEFORE the first graph
            # mutation: if truncation fails (a deterministic, non-retryable
            # content-shape problem), nothing has been written yet. The
            # actual VDB I/O call still happens after the graph write below.
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_content = _truncate_vdb_content(
                f"{entity_name}\n{final_description}",
                global_config,
                f"entity:{entity_name}",
            )
            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Update entity in graph storage (critical path)
            await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

            # Update entity in vector database (equally critical).  Use safe
            # operation wrapper - VDB failure must throw exception.
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # No merge_source_ids() normalization here: the rebuild caller derives
    # chunk_ids from subtract_source_ids() over either entity_chunks_storage
    # rows or an already-split graph `source_id` (see _purge_kg_contributions),
    # so the ids arrive individual and non-empty. Normalizing again would only
    # re-walk the list on the purge hot path.
    normalized_chunk_ids = chunk_ids

    if entity_chunks_storage is not None and normalized_chunk_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # Collect all entity data from relevant (limited) chunks
    all_entity_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        # An entity the surviving records know only as a relation endpoint is
        # re-summarized from its edges; the caller rebuilds relations first,
        # so those no longer carry the deleted chunks' text.
        edges = (
            []
            if structural_fallback or not summarize_from_edges
            else await knowledge_graph_inst.get_node_edges(entity_name)
        )
        if not edges:
            # Kept, not removed: no surviving chunk's records name it, which
            # also happens to a renamed or merged entity, so absence of
            # evidence is not evidence that only deleted chunks supported it.
            logger.warning(
                "No cached entity data found for `%s`; preserving semantic "
                "fields while repairing provenance",
                entity_name,
            )
            file_paths = _surviving_chunk_file_paths(
                limited_chunk_ids, chunk_data_by_id or {}, global_config
            )
            await _update_entity_storage(
                current_entity.get("description", ""),
                current_entity.get("entity_type", "UNKNOWN"),
                file_paths,
                limited_chunk_ids,
            )
            return True

        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Only the description comes from the edges. Provenance follows the
        # surviving chunks like source_id does: an edge's file_path may name
        # files outside this entity's chunks and carries its own truncation
        # marker, which must not be counted or kept as a path.
        relationship_descriptions = []
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data and edge_data.get("description"):
                relationship_descriptions.append(edge_data["description"])
        file_paths = _surviving_chunk_file_paths(
            limited_chunk_ids, chunk_data_by_id or {}, global_config
        )

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return False

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths_list = []
    seen_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Get most common entity type from raw entity_types list before deduplicating
    if entity_types:
        type_counts = Counter(entity_types)
        entity_type = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[0][
            0
        ]
    else:
        entity_type = current_entity.get("entity_type", "UNKNOWN")

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info. Per-item detail goes to the
    # backend log only; pipeline history keeps the rebuild summary (volume
    # control).
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)
    return False


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: list[str],
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    chunk_data_by_id: dict[str, dict[str, Any]] | None = None,
    structural_fallback: bool = False,
) -> bool:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return False

    # Same as _rebuild_single_entity: chunk_ids reach this function already
    # split and non-empty, so no merge_source_ids() normalization is needed.
    #
    # Drop the historical no-evidence placeholders here, once, for BOTH writes
    # below: this list becomes the authoritative relation_chunks row, and its
    # post-limit form becomes the edge's source_id. Purge carries a placeholder
    # in from a legacy tracking row or edge source_id (it is not one of the
    # deleted chunk IDs, so nothing subtracts it), and a later purge or audit
    # would read it back out of tracking as a real surviving chunk.
    normalized_chunk_ids = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id and chunk_id not in RELATION_NO_EVIDENCE_SOURCE_IDS
    ]

    # Keyed on the INPUT being non-empty, not on the filtered result: a row whose
    # only entry was a placeholder must be rewritten EMPTY -- authoritative "this
    # relation tracks no chunks" under the presence model -- rather than left
    # holding the placeholder because the write was skipped.
    if relation_chunks_storage is not None and chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from relevant chunks. One chunk's records can
    # name this relation more than once (the LLM restated it, or listed both
    # directions); each chunk contributes weight once, keeping its first record's
    # weight -- the same rule _merge_edges_then_upsert applies on insert, so a
    # rebuild does not re-inflate a weight the insert counted once.
    all_relationship_data = []
    weight_by_chunk: dict[str, float] = {}
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    records = chunk_relationships[chunk_id][edge_key]
                    all_relationship_data.extend(records)
                    for rel_data in records:
                        if rel_data.get("weight"):
                            weight_by_chunk.setdefault(chunk_id, rel_data["weight"])

    # Kept, not removed, when no surviving chunk's records name it; see the
    # matching branch of _rebuild_single_entity.
    degraded = not all_relationship_data
    if degraded:
        logger.warning(
            "No cached relation data found for `%s`~`%s`; preserving semantic "
            "fields while repairing provenance",
            src,
            tgt,
        )

    # Merge descriptions and keywords
    descriptions = [current_relationship.get("description", "")] if degraded else []
    raw_keywords = [current_relationship.get("keywords", "")] if degraded else []
    weights = (
        [current_relationship.get("weight", 1.0)]
        if degraded
        else list(weight_by_chunk.values())
    )
    file_paths_list = (
        _surviving_chunk_file_paths(
            limited_chunk_ids, chunk_data_by_id or {}, global_config
        )
        if degraded
        else []
    )
    seen_paths = set()

    # ``all_relationship_data`` is empty by definition when ``degraded``
    # (degraded == not all_relationship_data), so no guard is needed here.
    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            raw_keywords.append(rel_data["keywords"])
        if rel_data.get("file_path"):
            file_path = rel_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    # Collect and normalize keywords by splitting comma-separated tokens
    all_keywords = set()
    for kw_str in raw_keywords:
        if kw_str:
            all_keywords.update(k.strip() for k in kw_str.split(",") if k.strip())

    combined_keywords = (
        ", ".join(sorted(all_keywords))
        if all_keywords
        else current_relationship.get("keywords", "")
    )

    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)
    # A rebuilt relation must not fall below its evidence floor: limited_chunk_ids
    # becomes the edge's source_id, while cached fragments only cover those
    # surviving chunks whose extraction cache is still present (none at all on the
    # degraded path). Their summed weight can therefore under-count the evidence
    # and mint exactly the rows relation edits later refuse. Count the post-limit
    # list, matching _merge_edges_then_upsert and the source_id actually stored on
    # the edge.
    #
    # This is a floor, not boost preservation: a rebuild that found fragments
    # re-derives weight from them, exactly as it re-derives description and
    # keywords, so weight follows evidence DOWN as a purge removes chunks and an
    # edit_relation boost is not carried across. Folding the stored scalar into
    # the max would freeze weight at its pre-purge value forever, since the
    # scalar cannot be decomposed into evidence and boost. The degraded path has
    # nothing to re-derive from, so it seeds ``weights`` with the stored value
    # above and the boost survives there.
    # limited_chunk_ids is derived from the placeholder-filtered list above, so
    # every entry here already counts as real evidence.
    weight = max(float(weight), float(len(set(limited_chunk_ids))))

    # Generate final description from relations or fallback to current
    if degraded:
        final_description = current_relationship.get("description", "")
    elif description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else current_relationship.get("file_path", "unknown_source"),
        "truncate": truncation_info,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else current_relationship.get("description", "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    for node_id in {src, tgt}:
        if not (await knowledge_graph_inst.has_node(node_id)):
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }

            # Construct and verify the VDB payload BEFORE the first graph
            # mutation: if truncation fails (a deterministic, non-retryable
            # content-shape problem), nothing has been written yet. The
            # actual VDB I/O call still happens after the graph write below.
            vdb_data = (
                {
                    compute_mdhash_id(node_id, prefix="ent-"): {
                        "content": _truncate_vdb_content(
                            f"{node_id}\n{node_description}",
                            global_config,
                            f"entity:{node_id}",
                        ),
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                if entities_vdb is not None
                else None
            )

            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            if entities_vdb is not None:
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    # Sort src/tgt for a consistent VDB record identity (smaller string
    # first) — kept separate from the src/tgt used for the graph edge write
    # below, which must use the caller's original direction.
    vdb_src, vdb_tgt = (tgt, src) if src > tgt else (src, tgt)
    rel_vdb_id = compute_mdhash_id(vdb_src + vdb_tgt, prefix="rel-")
    rel_vdb_id_reverse = compute_mdhash_id(vdb_tgt + vdb_src, prefix="rel-")

    # Construct and verify the VDB payload BEFORE the first graph mutation
    # below: if truncation fails, nothing has been written yet. The actual
    # VDB I/O call still happens after the graph write below.
    rel_content = _truncate_vdb_content(
        f"{combined_keywords}\t{vdb_src}\n{vdb_tgt}\n{final_description}",
        global_config,
        f"relation:{vdb_src}-{vdb_tgt}",
    )
    vdb_data = {
        rel_vdb_id: {
            "src_id": vdb_src,
            "tgt_id": vdb_tgt,
            "source_id": updated_relationship_data["source_id"],
            "content": rel_content,
            "keywords": combined_keywords,
            "description": final_description,
            "weight": weight,
            "file_path": updated_relationship_data["file_path"],
        }
    }

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    try:
        # Delete old vector records first (both directions to be safe)
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            if structural_fallback:
                raise
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{vdb_src}-{vdb_tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = (
            f"Failed to rebuild relationship storage for `{vdb_src}-{vdb_tgt}`: {e}"
        )
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info. Per-item detail goes to the
    # backend log only; pipeline history keeps the rebuild summary (volume
    # control).
    status_message = f"Rebuild `{vdb_src}`~`{vdb_tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)
    return degraded


def _combine_descriptions_dedup(
    already_description: list[str], new_descriptions: list[str]
) -> tuple[list[str], int]:
    """Merge stored and newly-extracted descriptions, dropping exact duplicates.

    Stored fragments come first (preserving prior order), then new fragments not
    already present. Deduplicating across stored *and* new (not only within the
    new batch) prevents a re-extracted description from appending a duplicate
    fragment on every reprocess or resume; it also collapses any
    legacy duplicate fragments already stored. Returns the combined list and the
    count of surviving stored fragments, used for accurate merge accounting.

    Fragments are compared and stored *after* ``sanitize_text_for_encoding``:
    stored descriptions were already sanitized before being persisted (the
    single-description return and the join in ``_handle_entity_relation_summary``
    both sanitize), while a freshly re-extracted description is still raw.
    Comparing raw-vs-sanitized would miss the duplicate whenever sanitization
    changed the string (e.g. an XML-illegal control char was stripped), so
    "dirty" descriptions would keep accumulating across reprocess. Sanitizing
    here is idempotent, so the later sanitize calls are unaffected. Fragments
    that sanitize to empty are dropped (they carry no information and would only
    inflate the fragment count / emit ``<sep><sep>`` artifacts on join).

    The two-phase walk is deliberate: ``already_fragment`` is captured after the
    stored pass so it stays an accurate count of surviving stored fragments. Do
    NOT collapse both phases into a single ``dict.fromkeys`` over the
    concatenation — that loses the stored/new boundary the merge accounting log
    depends on.
    """
    combined: list[str] = []
    seen: set[str] = set()

    def _add_sanitized(descriptions: list[str]) -> None:
        for desc in descriptions:
            sanitized = sanitize_text_for_encoding(desc)
            if sanitized and sanitized not in seen:
                seen.add(sanitized)
                combined.append(sanitized)

    _add_sanitized(already_description)
    already_fragment = len(combined)
    _add_sanitized(new_descriptions)
    return combined, already_fragment


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    status_logger: PipelineStatusLogger | None = None,
    truncation_tally: TokenLimitTruncationTally | None = None,
    truncation_write_ahead: Callable[[TokenLimitTruncationTally], Awaitable[None]]
    | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    if status_logger is None:
        # Fallback for direct callers that pass pipeline_status only; a
        # throwaway logger reproduces the one-shot helper's behavior.
        status_logger = PipelineStatusLogger(pipeline_status)
    timing_start = time.perf_counter()
    try:
        already_entity_types = []
        already_source_ids = []
        already_description = []
        already_file_paths = []

        # 1. Get existing node data from knowledge graph
        already_node = await knowledge_graph_inst.get_node(entity_name)
        if already_node:
            existing_entity_type = already_node.get("entity_type")
            # Coerce to str before any string operations: non-string values from
            # API/custom graph paths would otherwise raise TypeError on the comma check.
            if (
                not isinstance(existing_entity_type, str)
                or not existing_entity_type.strip()
            ):
                existing_entity_type = "UNKNOWN"
            # Sanitize entity_type read back from DB to prevent dirty data from propagating
            if "," in existing_entity_type:
                original = existing_entity_type
                tokens = [t.strip() for t in existing_entity_type.split(",")]
                non_empty = [t for t in tokens if t]
                existing_entity_type = non_empty[0] if non_empty else "UNKNOWN"
                logger.warning(
                    f"Entity type read from DB contains comma, taking first non-empty token: '{original}' -> '{existing_entity_type}'"
                )
            already_entity_types.append(existing_entity_type)

            existing_source_id = already_node.get("source_id") or ""
            already_source_ids.extend(existing_source_id.split(GRAPH_FIELD_SEP))

            existing_file_path = already_node.get("file_path") or "unknown_source"
            already_file_paths.extend(existing_file_path.split(GRAPH_FIELD_SEP))

            existing_desc = (already_node.get("description") or "").strip()
            if existing_desc:
                already_description.extend(existing_desc.split(GRAPH_FIELD_SEP))

        new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

        existing_full_source_ids = []
        has_tracking_row = False
        if entity_chunks_storage is not None:
            stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
            # Row presence by schema, never list truthiness (see
            # has_chunk_tracking_row): a present-but-empty row is authoritative
            # and must NOT be reseeded from the graph node's source_id, which is
            # a truncated view that may still name purged chunks.
            has_tracking_row = has_chunk_tracking_row(stored_chunks)
            if has_tracking_row:
                existing_full_source_ids = [
                    chunk_id
                    for chunk_id in stored_chunks.get("chunk_ids", [])
                    if chunk_id
                ]

        if not has_tracking_row:
            existing_full_source_ids = [
                chunk_id for chunk_id in already_source_ids if chunk_id
            ]

        # 2. Merging new source ids with existing ones
        full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

        if entity_chunks_storage is not None and full_source_ids:
            await entity_chunks_storage.upsert(
                {
                    entity_name: {
                        "chunk_ids": full_source_ids,
                        "count": len(full_source_ids),
                    }
                }
            )

        # 3. Finalize source_id by applying source ids limit
        limit_method = global_config.get("source_ids_limit_method")
        max_source_limit = global_config.get("max_source_ids_per_entity")
        source_ids = apply_source_ids_limit(
            full_source_ids,
            max_source_limit,
            limit_method,
            identifier=f"`{entity_name}`",
        )

        # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
        if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
            allowed_source_ids = set(source_ids)
            filtered_nodes = []
            for dp in nodes_data:
                source_id = dp.get("source_id")
                # Skip descriptions sourced from chunks dropped by the limitation cap
                if (
                    source_id
                    and source_id not in allowed_source_ids
                    and source_id not in existing_full_source_ids
                ):
                    continue
                filtered_nodes.append(dp)
            nodes_data = filtered_nodes
        else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
            nodes_data = list(nodes_data)

        # 5. Check if we need to skip summary due to source_ids limit
        if (
            limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
            and len(existing_full_source_ids) >= max_source_limit
            and not nodes_data
        ):
            if already_node:
                logger.info(
                    f"Skipped `{entity_name}`: KEEP old chunks "
                    f"{len(existing_full_source_ids)}/{len(full_source_ids)} (limit {max_source_limit})"
                )
                existing_node_data = dict(already_node)
                return existing_node_data
            else:
                logger.error(
                    f"Internal Error: already_node missing for `{entity_name}`"
                )
                raise ValueError(
                    f"Internal Error: already_node missing for `{entity_name}`"
                )

        # 6.1 Finalize source_id
        source_id = GRAPH_FIELD_SEP.join(source_ids)

        # 6.2 Finalize entity type by highest count. The stored type carries
        # a single vote however many merges established it, so it goes first
        # and wins a tie (sorted() is stable): one conflicting extraction must
        # not flip it. An UNKNOWN placeholder -- written for a relation
        # endpoint never extracted as an entity, or for a missing stored type
        # -- is not evidence, casts no vote and yields to any real extraction;
        # with no other vote it is kept as is.
        type_votes = [t for t in already_entity_types if t.upper() != "UNKNOWN"] + [
            dp["entity_type"] for dp in nodes_data
        ]
        entity_type = sorted(
            Counter(type_votes or already_entity_types).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]

        # 7. Deduplicate nodes by description, keeping first occurrence in the same document
        unique_nodes = {}
        for i, dp in enumerate(nodes_data, start=1):
            desc = dp.get("description")
            if not desc:
                continue
            if desc not in unique_nodes:
                unique_nodes[desc] = dp
            await _cooperative_yield(i, every=32)

        # Sort description by timestamp, then by description length when timestamps are the same
        sorted_nodes = sorted(
            unique_nodes.values(),
            key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
        )
        sorted_descriptions = [dp["description"] for dp in sorted_nodes]

        # Combine stored and new descriptions, deduplicating across both so a
        # re-extracted description does not accumulate on reprocess
        description_list, already_fragment = _combine_descriptions_dedup(
            already_description, sorted_descriptions
        )
        if not description_list:
            fallback_description = f"Entity {entity_name}"
            logger.warning(
                f"Entity `{entity_name}` has no description; fallback to `{fallback_description}`"
            )
            description_list = [fallback_description]

        # Check for cancellation before LLM summary
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    raise PipelineCancelledException(
                        "User cancelled during entity summary"
                    )

        # 8. Get summary description an LLM usage status
        description, llm_was_used = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache,
            truncation_tally=truncation_tally,
            truncation_write_ahead=truncation_write_ahead,
        )

        # 9. Build file_path within MAX_FILE_PATHS
        file_paths_list = []
        seen_paths = set()
        has_placeholder = False  # Indicating file_path has been truncated before

        max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Collect from already_file_paths, excluding placeholder
        for fp in already_file_paths:
            if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
                has_placeholder = True
                continue
            if fp and fp not in seen_paths:
                file_paths_list.append(fp)
                seen_paths.add(fp)

        # Collect from new data
        for i, dp in enumerate(nodes_data, start=1):
            file_path_item = dp.get("file_path")
            if file_path_item and file_path_item not in seen_paths:
                file_paths_list.append(file_path_item)
                seen_paths.add(file_path_item)
            await _cooperative_yield(i, every=32)

        # Apply count limit
        if len(file_paths_list) > max_file_paths:
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            file_path_placeholder = global_config.get(
                "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
            )
            # Add + sign to indicate actual file count is higher
            original_count_str = (
                f"{len(file_paths_list)}+"
                if has_placeholder
                else str(len(file_paths_list))
            )

            if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
                # FIFO: keep tail (newest), discard head
                file_paths_list = file_paths_list[-max_file_paths:]
                file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
            else:
                # KEEP: keep head (earliest), discard tail
                file_paths_list = file_paths_list[:max_file_paths]
                file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

            logger.info(
                f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
            )
        # Finalize file_path
        file_path = GRAPH_FIELD_SEP.join(file_paths_list)

        # 10.Log based on actual LLM usage
        num_fragment = len(description_list)
        if llm_was_used:
            status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
        else:
            status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

        truncation_info = truncation_info_log = ""
        if len(source_ids) < len(full_source_ids):
            # Add truncation info from apply_source_ids_limit if truncation occurred
            truncation_info_log = (
                f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
            )
            if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
                truncation_info = truncation_info_log
            else:
                truncation_info = "KEEP Old"

        deduplicated_num = already_fragment + len(nodes_data) - num_fragment
        dd_message = ""
        if deduplicated_num > 0:
            # Duplicated description detected across multiple chunks for the same entity
            dd_message = f"dd {deduplicated_num}"

        if dd_message or truncation_info_log:
            status_message += (
                f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
            )

        # Add message to pipeline status when merge happens
        if already_fragment > 0 or llm_was_used:
            logger.info(status_message)
            status_logger.log(status_message)
        else:
            logger.debug(status_message)

        # 11. Update both graph and vector db
        node_data = dict(
            entity_id=entity_name,
            entity_type=entity_type,
            description=description,
            source_id=source_id,
            file_path=file_path,
            created_at=int(time.time()),
            truncate=truncation_info,
        )

        # Construct and verify the VDB payload BEFORE the first graph
        # mutation: if truncation fails (a deterministic, non-retryable
        # content-shape problem), nothing has been written yet. The actual
        # VDB I/O call still happens after the graph write below.
        data_for_vdb = None
        if entity_vdb is not None:
            entity_vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
            entity_content = _truncate_vdb_content(
                f"{entity_name}\n{description}",
                global_config,
                f"entity:{entity_name}",
            )
            data_for_vdb = {
                entity_vdb_id: {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "content": entity_content,
                    "source_id": source_id,
                    "file_path": file_path,
                }
            }

        await knowledge_graph_inst.upsert_node(
            entity_name,
            node_data=node_data,
        )
        node_data["entity_name"] = entity_name
        if entity_vdb is not None:
            await safe_vdb_operation_with_exception(
                operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
                operation_name="entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )
        return node_data
    finally:
        performance_timing_log(
            "[_merge_nodes_then_upsert] `%s` completed in %.4fs",
            entity_name,
            time.perf_counter() - timing_start,
        )


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    status_logger: PipelineStatusLogger | None = None,
    truncation_tally: TokenLimitTruncationTally | None = None,
    truncation_write_ahead: Callable[[TokenLimitTruncationTally], Awaitable[None]]
    | None = None,
):
    if status_logger is None:
        # Fallback for direct callers that pass pipeline_status only; a
        # throwaway logger reproduces the one-shot helper's behavior.
        status_logger = PipelineStatusLogger(pipeline_status)
    timing_start = time.perf_counter()
    timing_relation = f"`{src_id}`~`{tgt_id}`"
    try:
        if src_id == tgt_id:
            return None
        relation_key = f"{src_id}->{tgt_id}"

        already_edge = None
        already_weights = []
        already_source_ids = []
        already_description = []
        already_keywords = []
        already_file_paths = []

        # 1. Get existing edge data from graph storage
        if await knowledge_graph_inst.has_edge(src_id, tgt_id):
            already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            # Handle the case where get_edge returns None or missing fields
            if already_edge:
                # Get weight with default 1.0 if missing
                already_weights.append(already_edge.get("weight", 1.0))

                # Get source_id with empty string default if missing or None
                if already_edge.get("source_id") is not None:
                    already_source_ids.extend(
                        already_edge["source_id"].split(GRAPH_FIELD_SEP)
                    )

                # Get file_path with empty string default if missing or None
                if already_edge.get("file_path") is not None:
                    already_file_paths.extend(
                        already_edge["file_path"].split(GRAPH_FIELD_SEP)
                    )

                # Get description with empty string default if missing or None
                if already_edge.get("description") is not None:
                    already_description.extend(
                        already_edge["description"].split(GRAPH_FIELD_SEP)
                    )

                # Get keywords with empty string default if missing or None
                if already_edge.get("keywords") is not None:
                    already_keywords.extend(
                        split_string_by_multi_markers(
                            already_edge["keywords"], [GRAPH_FIELD_SEP]
                        )
                    )

        new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]

        storage_key = make_relation_chunk_key(src_id, tgt_id)
        existing_full_source_ids = []
        has_tracking_row = False
        if relation_chunks_storage is not None:
            stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
            # Row presence by schema, never list truthiness (see
            # has_chunk_tracking_row): a present-but-empty row is authoritative
            # and must NOT be reseeded from the graph edge's source_id, which is
            # a truncated view that may still name purged chunks.
            has_tracking_row = has_chunk_tracking_row(stored_chunks)
            if has_tracking_row:
                # relation_chunks is the authoritative chunk list, so a
                # historical no-evidence placeholder must never enter it: purge
                # and audit would read it back as a real surviving chunk. Filter
                # it out of BOTH baselines -- the stored row (repairing a legacy
                # row on this merge) and, below, the legacy edge's own source_id
                # used when no row exists. Filtering leaves an all-placeholder
                # row empty, which stays authoritative here: the edge cannot own
                # real chunks that tracking never recorded, so there is nothing
                # to reseed.
                existing_full_source_ids = [
                    chunk_id
                    for chunk_id in stored_chunks.get("chunk_ids", [])
                    if chunk_id and chunk_id not in RELATION_NO_EVIDENCE_SOURCE_IDS
                ]

        if not has_tracking_row:
            existing_full_source_ids = [
                chunk_id
                for chunk_id in already_source_ids
                if chunk_id and chunk_id not in RELATION_NO_EVIDENCE_SOURCE_IDS
            ]

        # 2. Merge new source ids with existing ones
        full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

        if relation_chunks_storage is not None and full_source_ids:
            await relation_chunks_storage.upsert(
                {
                    storage_key: {
                        "chunk_ids": full_source_ids,
                        "count": len(full_source_ids),
                    }
                }
            )

        # 3. Finalize source_id by applying source ids limit
        limit_method = global_config.get("source_ids_limit_method")
        max_source_limit = global_config.get("max_source_ids_per_relation")
        source_ids = apply_source_ids_limit(
            full_source_ids,
            max_source_limit,
            limit_method,
            identifier=f"`{src_id}`~`{tgt_id}`",
        )
        limit_method = (
            global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
        )

        # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
        if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
            allowed_source_ids = set(source_ids)
            filtered_edges = []
            for dp in edges_data:
                source_id = dp.get("source_id")
                # Skip relationship fragments sourced from chunks dropped by keep oldest cap
                if (
                    source_id
                    and source_id not in allowed_source_ids
                    and source_id not in existing_full_source_ids
                ):
                    continue
                filtered_edges.append(dp)
            edges_data = filtered_edges
        else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
            edges_data = list(edges_data)

        # 5. Check if we need to skip summary due to source_ids limit
        if (
            limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
            and len(existing_full_source_ids) >= max_source_limit
            and not edges_data
        ):
            if already_edge:
                logger.info(
                    f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks "
                    f"{len(existing_full_source_ids)}/{len(full_source_ids)} (limit {max_source_limit})"
                )
                existing_edge_data = dict(already_edge)
                return existing_edge_data
            else:
                logger.error(
                    f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
                )
                raise ValueError(
                    f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
                )

        # 6.1 Finalize source_id
        source_id = GRAPH_FIELD_SEP.join(source_ids)

        # 6.2 Finalize weight. Each extracted relation contributes weight 1.0
        # (or its parsed strength), so a stored edge's weight tracks the number
        # of distinct contributing sources. On reprocess/resume the same source
        # can be re-fed while it is already reflected in already_weights (the
        # stored scalar), so only sum weights of edges whose source_id is NOT
        # already stored -- otherwise weight double-counts and grows 1 -> 2 -> 3
        # per reprocess (the sibling of description accumulation).
        # Genuinely new sources still add their weight, preserving legitimate
        # multi-document growth.
        #
        # Filter on the EDGE's own source_ids (already_source_ids) -- the same
        # source as already_weights -- NOT existing_full_source_ids, which can
        # come from relation_chunks_storage, a SEPARATE store. relation_chunks
        # can run ahead of the graph edge on a partial write (chunks upserted,
        # upsert_edge not yet), so filtering on it would wrongly skip a genuinely
        # new source (under-count), or, with no stored weight at all, recover the
        # edge with weight 0. Filtering on already_source_ids keeps the exclusion
        # consistent with the stored scalar: when there is no stored edge
        # (already_source_ids empty) nothing is excluded, so a recovered edge
        # keeps its full weight; a re-fed already-stored source is dropped.
        # A falsy source_id is dropped from new_source_ids above (it never
        # persists), so it must not add weight either -- otherwise weight would
        # outgrow the stored source count. Count only sources that actually
        # persist: truthy AND not already reflected in the stored scalar.
        #
        # edges_data can also carry more than one record for the SAME
        # source_id within this one call: a single LLM extraction can state the
        # same relation twice, or once in each direction (both land under the
        # sorted edge key). Each source must contribute weight once, matching
        # the "distinct contributing sources" intent above, not once per
        # record; keep the first record seen per source, the same
        # first-occurrence convention used for description dedup.
        # _rebuild_single_relationship applies the same rule per chunk.
        already_edge_source_set = set(already_source_ids)
        new_weight_by_source: dict[str, float] = {}
        for dp in edges_data:
            dp_source_id = dp.get("source_id")
            if not dp_source_id or dp_source_id in already_edge_source_set:
                continue
            new_weight_by_source.setdefault(dp_source_id, dp["weight"])
        weight = sum(new_weight_by_source.values()) + sum(already_weights)
        # Repair legacy/manual rows that predate the shared weight contract:
        # every distinct real source contributes a baseline of 1, while an
        # existing larger weight remains an optional importance boost.
        evidence_count = len(
            {
                source
                for source in source_ids
                if source and source not in RELATION_NO_EVIDENCE_SOURCE_IDS
            }
        )
        weight = max(float(weight), float(evidence_count))

        # 6.2 Finalize keywords by merging existing and new keywords
        all_keywords = set()
        # Process already_keywords (which are comma-separated)
        for i, keyword_str in enumerate(already_keywords, start=1):
            if keyword_str:  # Skip empty strings
                all_keywords.update(
                    k.strip() for k in keyword_str.split(",") if k.strip()
                )
            await _cooperative_yield(i, every=32)
        # Process new keywords from edges_data
        for i, edge in enumerate(edges_data, start=1):
            if edge.get("keywords"):
                all_keywords.update(
                    k.strip() for k in edge["keywords"].split(",") if k.strip()
                )
            await _cooperative_yield(i, every=32)
        # Join all unique keywords with commas
        keywords = ",".join(sorted(all_keywords))

        # 7. Deduplicate by description, keeping first occurrence in the same document
        unique_edges = {}
        for i, dp in enumerate(edges_data, start=1):
            description_value = dp.get("description")
            if not description_value:
                continue
            if description_value not in unique_edges:
                unique_edges[description_value] = dp
            await _cooperative_yield(i, every=32)

        # Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
        sorted_edges = sorted(
            unique_edges.values(),
            key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
        )
        sorted_descriptions = [dp["description"] for dp in sorted_edges]

        # Combine stored and new descriptions, deduplicating across both so a
        # re-extracted description does not accumulate on reprocess
        description_list, already_fragment = _combine_descriptions_dedup(
            already_description, sorted_descriptions
        )
        if not description_list:
            logger.error(f"Relation {src_id}~{tgt_id} has no description")
            raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

        # Check for cancellation before LLM summary
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    raise PipelineCancelledException(
                        "User cancelled during relation summary"
                    )

        # 8. Get summary description an LLM usage status
        description, llm_was_used = await _handle_entity_relation_summary(
            "Relation",
            f"({src_id}, {tgt_id})",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache,
            truncation_tally=truncation_tally,
            truncation_write_ahead=truncation_write_ahead,
        )

        # 9. Build file_path within MAX_FILE_PATHS limit
        file_paths_list = []
        seen_paths = set()
        has_placeholder = False  # Track if already_file_paths contains placeholder

        max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Collect from already_file_paths, excluding placeholder
        for fp in already_file_paths:
            # Check if this is a placeholder record
            if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
                has_placeholder = True
                continue
            if fp and fp not in seen_paths:
                file_paths_list.append(fp)
                seen_paths.add(fp)

        # Collect from new data
        for i, dp in enumerate(edges_data, start=1):
            file_path_item = dp.get("file_path")
            if file_path_item and file_path_item not in seen_paths:
                file_paths_list.append(file_path_item)
                seen_paths.add(file_path_item)
            await _cooperative_yield(i, every=32)
        # Apply count limit
        if len(file_paths_list) > max_file_paths:
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )

            # Add + sign to indicate actual file count is higher
            original_count_str = (
                f"{len(file_paths_list)}+"
                if has_placeholder
                else str(len(file_paths_list))
            )

            if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
                # FIFO: keep tail (newest), discard head
                file_paths_list = file_paths_list[-max_file_paths:]
                file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
            else:
                # KEEP: keep head (earliest), discard tail
                file_paths_list = file_paths_list[:max_file_paths]
                file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

            logger.info(
                f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
            )
        # Finalize file_path
        file_path = GRAPH_FIELD_SEP.join(file_paths_list)

        # 10. Log based on actual LLM usage
        num_fragment = len(description_list)
        if llm_was_used:
            status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
        else:
            status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

        truncation_info = truncation_info_log = ""
        if len(source_ids) < len(full_source_ids):
            # Add truncation info from apply_source_ids_limit if truncation occurred
            truncation_info_log = (
                f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
            )
            if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
                truncation_info = truncation_info_log
            else:
                truncation_info = "KEEP Old"

        deduplicated_num = already_fragment + len(edges_data) - num_fragment
        dd_message = ""
        if deduplicated_num > 0:
            # Duplicated description detected across multiple chunks for the same entity
            dd_message = f"dd {deduplicated_num}"

        if dd_message or truncation_info_log:
            status_message += (
                f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
            )

        # Add message to pipeline status when merge happens
        if already_fragment > 0 or llm_was_used:
            logger.info(status_message)
            status_logger.log(status_message)
        else:
            logger.debug(status_message)

        # 11. Update both graph and vector db.
        for need_insert_id in [src_id, tgt_id]:
            # Optimization: Use get_node instead of has_node + get_node
            existing_node = await knowledge_graph_inst.get_node(need_insert_id)

            if existing_node is None:
                # Node doesn't exist - create new node
                node_created_at = int(time.time())
                node_data = {
                    "entity_id": need_insert_id,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN",
                    "file_path": file_path,
                    "created_at": node_created_at,
                    "truncate": "",
                }

                # Construct and verify the VDB payload BEFORE the first
                # graph mutation: if truncation fails (a deterministic,
                # non-retryable content-shape problem), nothing has been
                # written yet. The actual VDB I/O call still happens after
                # the graph write below.
                vdb_data = None
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                    entity_content = _truncate_vdb_content(
                        f"{need_insert_id}\n{description}",
                        global_config,
                        f"entity:{need_insert_id}",
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_name": need_insert_id,
                            "source_id": source_id,
                            "entity_type": "UNKNOWN",
                            "file_path": file_path,
                        }
                    }

                await knowledge_graph_inst.upsert_node(
                    need_insert_id, node_data=node_data
                )

                # Update entity_chunks_storage for the newly created entity
                if entity_chunks_storage is not None:
                    chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                    if chunk_ids:
                        await entity_chunks_storage.upsert(
                            {
                                need_insert_id: {
                                    "chunk_ids": chunk_ids,
                                    "count": len(chunk_ids),
                                }
                            }
                        )

                if entity_vdb is not None:
                    await safe_vdb_operation_with_exception(
                        operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                        operation_name="added_entity_upsert",
                        entity_name=f"{need_insert_id} [relation:{relation_key}]",
                        max_retries=3,
                        retry_delay=0.1,
                        timeout_seconds=_get_relationship_vdb_timeout_seconds(
                            global_config
                        ),
                        log_start=False,
                        success_log_threshold_seconds=5.0,
                    )

                # Track entities added during edge processing
                if added_entities is not None:
                    added_entities.append(
                        {
                            "entity_name": need_insert_id,
                            "entity_type": "UNKNOWN",
                            "description": description,
                            "source_id": source_id,
                            "file_path": file_path,
                            "created_at": node_created_at,
                        }
                    )
            else:
                # Node exists - update its source_ids by merging with new source_ids
                updated = False  # Track if any update occurred

                # 1. Get existing full source_ids from entity_chunks_storage
                existing_full_source_ids = []
                has_tracking_row = False
                if entity_chunks_storage is not None:
                    stored_chunks = await entity_chunks_storage.get_by_id(
                        need_insert_id
                    )
                    # Row presence by schema, never list truthiness (see
                    # has_chunk_tracking_row): a present-but-empty row is
                    # authoritative and must NOT be reseeded from the graph
                    # node's possibly-stale source_id.
                    has_tracking_row = has_chunk_tracking_row(stored_chunks)
                    if has_tracking_row:
                        existing_full_source_ids = [
                            chunk_id
                            for chunk_id in stored_chunks.get("chunk_ids", [])
                            if chunk_id
                        ]

                # If no tracking row exists at all, fall back to graph database
                if not has_tracking_row:
                    if existing_node.get("source_id"):
                        existing_full_source_ids = existing_node["source_id"].split(
                            GRAPH_FIELD_SEP
                        )

                # 2. Merge with new source_ids from this relationship
                new_source_ids_from_relation = [
                    chunk_id for chunk_id in source_ids if chunk_id
                ]
                merged_full_source_ids = merge_source_ids(
                    existing_full_source_ids, new_source_ids_from_relation
                )

                # 3. Save merged full list to entity_chunks_storage (conditional)
                if (
                    entity_chunks_storage is not None
                    and merged_full_source_ids != existing_full_source_ids
                ):
                    updated = True
                    await entity_chunks_storage.upsert(
                        {
                            need_insert_id: {
                                "chunk_ids": merged_full_source_ids,
                                "count": len(merged_full_source_ids),
                            }
                        }
                    )

                # 4. Apply source_ids limit for graph and vector db
                limit_method = global_config.get(
                    "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
                )
                max_source_limit = global_config.get("max_source_ids_per_entity")
                limited_source_ids = apply_source_ids_limit(
                    merged_full_source_ids,
                    max_source_limit,
                    limit_method,
                    identifier=f"`{need_insert_id}`",
                )

                # 5. Update graph database and vector database with limited source_ids (conditional)
                limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

                if limited_source_id_str != existing_node.get("source_id", ""):
                    updated = True
                    updated_node_data = {
                        **existing_node,
                        "source_id": limited_source_id_str,
                    }

                    # Construct and verify the VDB payload BEFORE the first
                    # graph mutation: if truncation fails, nothing has been
                    # written yet. The actual VDB I/O call still happens
                    # after the graph write below.
                    vdb_data = None
                    if entity_vdb is not None:
                        entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                        entity_content = _truncate_vdb_content(
                            f"{need_insert_id}\n{existing_node.get('description', '')}",
                            global_config,
                            f"entity:{need_insert_id}",
                        )
                        vdb_data = {
                            entity_vdb_id: {
                                "content": entity_content,
                                "entity_name": need_insert_id,
                                "source_id": limited_source_id_str,
                                "entity_type": existing_node.get(
                                    "entity_type", "UNKNOWN"
                                ),
                                "file_path": existing_node.get(
                                    "file_path", "unknown_source"
                                ),
                            }
                        }

                    await knowledge_graph_inst.upsert_node(
                        need_insert_id, node_data=updated_node_data
                    )

                    # Update vector database
                    if entity_vdb is not None:
                        await safe_vdb_operation_with_exception(
                            operation=lambda payload=vdb_data: entity_vdb.upsert(
                                payload
                            ),
                            operation_name="existing_entity_update",
                            entity_name=f"{need_insert_id} [relation:{relation_key}]",
                            max_retries=3,
                            retry_delay=0.1,
                            timeout_seconds=_get_relationship_vdb_timeout_seconds(
                                global_config
                            ),
                            log_start=False,
                            success_log_threshold_seconds=5.0,
                        )

                # 6. Log once at the end if any update occurred
                if updated:
                    status_message = (
                        f"Chunks appended from relation: `{need_insert_id}`"
                    )
                    logger.info(status_message)
                    status_logger.log(status_message)

        edge_created_at = int(time.time())

        # Sort src_id/tgt_id for a consistent VDB record identity (smaller
        # string first) — kept separate from src_id/tgt_id used for the
        # graph edge write below, which must use the caller's original
        # direction.
        vdb_src_id, vdb_tgt_id = (
            (tgt_id, src_id) if src_id > tgt_id else (src_id, tgt_id)
        )

        # Construct and verify the VDB payload BEFORE the first graph
        # mutation below: if truncation fails (a deterministic,
        # non-retryable content-shape problem), nothing has been written
        # yet. The actual VDB I/O calls still happen after the graph write
        # below.
        vdb_data = None
        if relationships_vdb is not None:
            rel_vdb_id = compute_mdhash_id(vdb_src_id + vdb_tgt_id, prefix="rel-")
            rel_vdb_id_reverse = compute_mdhash_id(
                vdb_tgt_id + vdb_src_id, prefix="rel-"
            )
            rel_content = _truncate_vdb_content(
                f"{keywords}\t{vdb_src_id}\n{vdb_tgt_id}\n{description}",
                global_config,
                f"relationship:{vdb_src_id}-{vdb_tgt_id}",
            )
            vdb_data = {
                rel_vdb_id: {
                    "src_id": vdb_src_id,
                    "tgt_id": vdb_tgt_id,
                    "source_id": source_id,
                    "content": rel_content,
                    "keywords": keywords,
                    "description": description,
                    "weight": weight,
                    "file_path": file_path,
                }
            }

        edge_upsert_started = time.perf_counter()
        await knowledge_graph_inst.upsert_edge(
            src_id,
            tgt_id,
            edge_data=dict(
                weight=weight,
                description=description,
                keywords=keywords,
                source_id=source_id,
                file_path=file_path,
                created_at=edge_created_at,
                truncate=truncation_info,
            ),
        )
        edge_upsert_elapsed = time.perf_counter() - edge_upsert_started
        if edge_upsert_elapsed >= 5.0:
            logger.info(
                "Graph edge upsert slow for `%s` in %.2fs",
                relation_key,
                edge_upsert_elapsed,
            )

        edge_data = dict(
            src_id=src_id,
            tgt_id=tgt_id,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=edge_created_at,
            truncate=truncation_info,
            weight=weight,
        )

        if relationships_vdb is not None:
            try:
                await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
            except Exception as e:
                logger.debug(
                    f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
                )
            relation_status_message = f"Upserting relation VDB: `{relation_key}`"
            logger.info(relation_status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = relation_status_message
            await safe_vdb_operation_with_exception(
                operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
                operation_name="relationship_upsert",
                entity_name=relation_key,
                max_retries=3,
                retry_delay=0.2,
                timeout_seconds=_get_relationship_vdb_timeout_seconds(global_config),
                log_start=False,
                success_log_threshold_seconds=5.0,
            )

        return edge_data
    finally:
        performance_timing_log(
            "[_merge_edges_then_upsert] %s completed in %.4fs",
            timing_relation,
            time.perf_counter() - timing_start,
        )


def collect_kg_merge_candidates(
    chunk_results: list,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Derive the full candidate entity/relation superset a merge may touch.

    Write-ahead recovery anchor: before any graph/vector/tracking
    mutation, the caller must be able to persist a durable candidate set in
    ``full_entities`` / ``full_relations`` so a later purge/retry can discover
    every object the merge might have written. The superset therefore
    includes:

    - every extracted entity name;
    - every endpoint of every extracted relation (relation processing can
      create missing endpoint entities, see ``_merge_edges_then_upsert``);
    - every relation pair, normalized to sorted order (undirected graph).

    The result is a candidate SUPERSET, not proof of existence: purge must
    verify source ownership per candidate and treat absent objects as no-ops.

    Args:
        chunk_results: List of ``(maybe_nodes, maybe_edges)`` tuples as
            produced by ``extract_entities``.

    Returns:
        ``(candidate_entity_names, candidate_relation_pairs)``.
    """
    candidate_entities: set[str] = set()
    candidate_relations: set[tuple[str, str]] = set()
    for maybe_nodes, maybe_edges in chunk_results:
        candidate_entities.update(maybe_nodes.keys())
        for edge_key in maybe_edges:
            pair = tuple(sorted((edge_key[0], edge_key[1])))
            candidate_relations.add(pair)
            candidate_entities.update(pair)
    return candidate_entities, candidate_relations


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
    on_anchors_durable: Callable[[], Awaitable[None]] | None = None,
    truncation_tally: TokenLimitTruncationTally | None = None,
    truncation_write_ahead: Callable[[TokenLimitTruncationTally], Awaitable[None]]
    | None = None,
) -> None:
    """Merge extracted entities/relations into the KG behind write-ahead anchors.

    Phase order (discoverability before mutation):
    0. Phase 0: Persist the full candidate superset to ``full_entities`` /
       ``full_relations`` and flush both BEFORE any graph mutation, so a
       crash mid-merge always leaves a durable recovery anchor that purge /
       retry can discover the contributions from. Both rows are written even
       when empty; a failure here aborts the merge before mutation.
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
        on_anchors_durable: Optional async hook awaited exactly once, after
            both Phase 0 anchors are flushed and before the first graph
            mutation. Lets the caller persist "this document may have touched
            the graph from here on" (``doc_status.metadata.kg_write_state``)
            in the one window where that statement flips, so a fail-closed
            purge can distinguish a document that never reached the graph from
            one whose anchors were lost. A raise propagates and aborts the
            merge before any mutation. Not called when the anchor storages or
            ``doc_id`` are absent (patch-mode merges, which carry their own
            operation journal as the recovery proof).
        truncation_tally: Optional document-scoped accumulator. Description
            summaries are LLM calls too, so a too-small output budget truncates
            them exactly like extraction; the merge folds its own count in so
            the document's ``doc_status.metadata`` summary covers both stages.
        truncation_write_ahead: Optional async hook awaited right after a
            summary truncation is recorded, before the object carrying that
            summary is upserted. For callers whose failure recovery is
            ADDITIVE (the custom-chunk path: a resume keeps the failed
            attempt's graph mutations), the record must be durable before the
            mutation it warns about — a FAILED-transition stamp alone loses
            the event on a hard crash. The pipeline's own merges do not need
            it: a reprocess purges the previous attempt's contributions, so
            nothing truncated survives to warn about. Receives the tally the
            event was recorded into (the merge-local one, folded into
            ``truncation_tally`` only when the merge publishes — after every
            hook call has completed, because failing phases drain their
            sibling tasks first).
    """

    # Check for cancellation at the start of merge
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    # Operation-scoped status logger shared by all entity/relation merge tasks
    # of this document: the first history write fetches the shared list once;
    # every merge log is then a single extend on the cached handle.
    status_logger = PipelineStatusLogger(pipeline_status)

    # Merge-scoped tally: one aggregated status line for the whole merge rather
    # than one per truncated summary, folded into the document-scoped tally so
    # the doc_status record covers extraction and merge together.
    summary_tally = TokenLimitTruncationTally()
    summary_truncation_published = False

    def _publish_summary_truncation() -> None:
        """Publish the merge's truncation aggregate and hand its counts upward.

        Called from a ``finally`` covering both phases, so it runs on EVERY
        merge exit — success, a failing sibling task, and a cancellation
        landing on any await point between the phases (an inter-phase
        ``sleep(0)`` or status-lock acquire). Wrapping only the
        ``wait_tasks_with_drain`` calls was not enough: a cancellation at an
        inter-phase yield skipped the absorb, and the custom-chunk failure
        bookkeeping then recomputed the journal from the unabsorbed operation
        tally — overwriting the summary event its write-ahead hook had
        already persisted. Idempotent, so redundant calls cannot
        double-publish, and await-free, so it is safe to run inside a
        cancellation window.
        """
        nonlocal summary_truncation_published
        if summary_truncation_published:
            return
        summary_truncation_published = True
        if summary_tally:
            noun = "summary" if summary_tally.affected == 1 else "summaries"
            truncation_message = (
                f"Warning: token-limit truncation hit {summary_tally.affected} "
                f"description {noun} during merge "
                f"({summary_tally.events} responses); "
                f"merged descriptions may be incomplete"
            )
            logger.warning(truncation_message)
            status_logger.log(truncation_message)
        if truncation_tally is not None:
            truncation_tally.absorb(summary_tally)

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for i, (maybe_nodes, maybe_edges) in enumerate(chunk_results, start=1):
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)
        await _cooperative_yield(i, every=32)

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        append_pipeline_history(pipeline_status, log_message)

    # ===== Phase 0: write-ahead recovery indexes =====
    # Persist the candidate superset BEFORE any graph/vector/tracking
    # mutation. Candidates are a superset, not proof of existence: purge
    # verifies ownership per candidate and skips absent objects. Empty rows
    # are written too — a reprocess that now yields fewer (or zero)
    # candidates must overwrite the stale anchors of the previous attempt.
    if full_entities_storage and full_relations_storage and doc_id:
        candidate_entities, candidate_relations = collect_kg_merge_candidates(
            chunk_results
        )
        log_message = (
            f"Phase 0: Writing recovery indexes for {doc_id}: "
            f"{len(candidate_entities)} candidate entities, "
            f"{len(candidate_relations)} candidate relations"
        )
        logger.info(log_message)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            append_pipeline_history(pipeline_status, log_message)

        await full_entities_storage.upsert(
            {
                doc_id: {
                    "entity_names": sorted(candidate_entities),
                    "count": len(candidate_entities),
                }
            }
        )
        await full_relations_storage.upsert(
            {
                doc_id: {
                    "relation_pairs": [
                        list(pair) for pair in sorted(candidate_relations)
                    ],
                    "count": len(candidate_relations),
                }
            }
        )
        # Narrow flush barrier: both anchors must be durable before the first
        # graph mutation. A failure propagates — merging without a recovery
        # anchor would make a later partial failure undiscoverable.
        for storage_inst in (full_entities_storage, full_relations_storage):
            try:
                await storage_inst.index_done_callback()
            except Exception as e:
                namespace = getattr(storage_inst, "final_namespace", None) or getattr(
                    storage_inst, "namespace", ""
                )
                raise IndexFlushError(type(storage_inst).__name__, namespace, e) from e

        # Anchors are durable: this is the last instant at which "no graph
        # mutation has happened" is still true. A raise here aborts the merge
        # with the anchors already in place — the safe direction.
        if on_anchors_durable is not None:
            await on_anchors_durable()

    entity_tasks: list[asyncio.Task] = []
    edge_tasks: list[asyncio.Task] = []
    try:
        # Get max async tasks limit from global_config for semaphore control
        graph_max_async = global_config.get("llm_model_max_async", 4) * 2
        semaphore = asyncio.Semaphore(graph_max_async)

        # ===== Phase 1: Process all entities concurrently =====
        log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
        logger.info(log_message)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            append_pipeline_history(pipeline_status, log_message)

        async def _locked_process_entity_name(entity_name, entities):
            async with semaphore:
                # Check for cancellation before processing entity
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during entity merge"
                            )

                workspace = global_config.get("workspace", "")
                namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
                async with get_storage_keyed_lock(
                    [entity_name], namespace=namespace, enable_logging=False
                ):
                    try:
                        logger.debug(f"Processing entity {entity_name}")
                        entity_data = await _merge_nodes_then_upsert(
                            entity_name,
                            entities,
                            knowledge_graph_inst,
                            entity_vdb,
                            global_config,
                            pipeline_status,
                            pipeline_status_lock,
                            llm_response_cache,
                            entity_chunks_storage,
                            status_logger=status_logger,
                            truncation_tally=summary_tally,
                            truncation_write_ahead=truncation_write_ahead,
                        )

                        return entity_data

                    except Exception as e:
                        error_msg = f"Error processing entity `{entity_name}`: {e}"
                        logger.error(error_msg)

                        # Try to update pipeline status, but don't let status update failure affect main exception
                        try:
                            if (
                                pipeline_status is not None
                                and pipeline_status_lock is not None
                            ):
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    append_pipeline_history(pipeline_status, error_msg)
                        except Exception as status_error:
                            logger.error(
                                f"Failed to update pipeline status: {status_error}"
                            )

                        # Re-raise the original exception with a prefix
                        prefixed_exception = create_prefixed_exception(
                            e, f"`{entity_name}`"
                        )
                        raise prefixed_exception from e

        # Create entity processing tasks
        for i, (entity_name, entities) in enumerate(all_nodes.items(), start=1):
            task = asyncio.create_task(
                _locked_process_entity_name(entity_name, entities)
            )
            entity_tasks.append(task)
            await _cooperative_yield(i, every=16)

        # Execute entity tasks; on any failure every sibling is cancelled and
        # drained before the first exception propagates (no background writes
        # survive failure handling).
        processed_entities = []
        if entity_tasks:
            processed_entities = await wait_tasks_with_drain(
                entity_tasks, context=f"entity merge {doc_id}"
            )
            await asyncio.sleep(0)

        # ===== Phase 2: Process all relationships concurrently =====
        log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
        logger.info(log_message)
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            append_pipeline_history(pipeline_status, log_message)

        async def _locked_process_edges(edge_key, edges):
            async with semaphore:
                # Check for cancellation before processing edges
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during relation merge"
                            )

                workspace = global_config.get("workspace", "")
                namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
                sorted_edge_key = sorted([edge_key[0], edge_key[1]])
                edge_label = _format_relation_edge_label(edge_key)

                async with get_storage_keyed_lock(
                    sorted_edge_key,
                    namespace=namespace,
                    enable_logging=False,
                ):
                    try:
                        added_entities = []  # Track entities added during edge processing

                        edge_data = await _merge_edges_then_upsert(
                            edge_key[0],
                            edge_key[1],
                            edges,
                            knowledge_graph_inst,
                            relationships_vdb,
                            entity_vdb,
                            global_config,
                            pipeline_status,
                            pipeline_status_lock,
                            llm_response_cache,
                            added_entities,  # Pass list to collect added entities
                            relation_chunks_storage,
                            entity_chunks_storage,  # Add entity_chunks_storage parameter
                            status_logger=status_logger,
                            truncation_tally=summary_tally,
                            truncation_write_ahead=truncation_write_ahead,
                        )

                        if edge_data is None:
                            return None, []

                        return edge_data, added_entities

                    except Exception as e:
                        error_msg = f"Error processing relation `{edge_label}`: {e}"
                        logger.error(error_msg)

                        # Try to update pipeline status, but don't let status update failure affect main exception
                        try:
                            if (
                                pipeline_status is not None
                                and pipeline_status_lock is not None
                            ):
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    append_pipeline_history(pipeline_status, error_msg)
                        except Exception as status_error:
                            logger.error(
                                f"Failed to update pipeline status: {status_error}"
                            )

                        # Re-raise the original exception with a prefix
                        prefixed_exception = create_prefixed_exception(
                            e, f"{edge_label}"
                        )
                        raise prefixed_exception from e

        # Create relationship processing tasks
        edge_task_labels: dict[asyncio.Task, str] = {}
        for i, (edge_key, edges) in enumerate(all_edges.items(), start=1):
            task = asyncio.create_task(_locked_process_edges(edge_key, edges))
            edge_tasks.append(task)
            edge_task_labels[task] = _format_relation_edge_label(edge_key)
            await _cooperative_yield(i, every=32)

        # Execute relationship tasks; failures cancel + drain every sibling
        # before propagating (see wait_tasks_with_drain).
        processed_edges = []
        all_added_entities = []

        if edge_tasks:
            edge_results = await wait_tasks_with_drain(
                edge_tasks,
                context=f"relation merge {doc_id}",
                task_labels=edge_task_labels,
            )
            for edge_data, added_entities in edge_results:
                if edge_data is not None:
                    processed_edges.append(edge_data)
                all_added_entities.extend(added_entities)

            logger.info(
                "Phase 2 relation processing completed for %s: edges=%d added_entities=%d",
                doc_id,
                len(processed_edges),
                len(all_added_entities),
            )
            await asyncio.sleep(0)

        # NOTE: the recovery indexes are written (and flushed) in Phase 0, BEFORE
        # graph mutation. The historical post-merge "Phase 3" write — which
        # derived the rows from in-memory merge results and swallowed its own
        # exceptions — is gone: a merge whose anchors cannot be persisted no
        # longer mutates the graph at all.

    except (Exception, asyncio.CancelledError):
        # wait_tasks_with_drain only covers cancellation while it is waiting.
        # The creation loops yield (_cooperative_yield) before it starts, so a
        # cancellation landing there would leave the writers created so far
        # merging into the graph after this coroutine has exited and recovery
        # has begun. Same exit rule as extract_entities: stop and collect
        # every scheduled writer first. Tasks already drained by
        # wait_tasks_with_drain are done, so this is then a no-op.
        await cancel_and_drain_tasks(entity_tasks + edge_tasks)
        raise
    finally:
        # On EVERY exit — the inter-phase await points (sleep(0) yields,
        # status-lock acquires) sit outside any narrower handler, and a
        # cancellation landing there must still absorb the summary tally
        # before the caller's failure bookkeeping recomputes from it.
        _publish_summary_truncation()

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        append_pipeline_history(pipeline_status, log_message)


async def _apply_kg_extraction_validator(
    validator: Callable[..., Any],
    chunk_key: str,
    chunk_dp: dict[str, Any],
    maybe_nodes: dict[str, list[dict]],
    maybe_edges: dict[tuple[str, str], list[dict]],
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """Run ``kg_extraction_validator`` on one chunk's extracted records.

    Shared by ingestion (``extract_entities``) and the deletion rebuild's
    cache path, so a cached row is held to the same rules as a fresh
    extraction. Caller-facing contract: "Extraction Quality Hook" in
    docs/ProgramingWithCore.md. Call it before ``_inject_multimodal_records``.
    """
    validated = validator(
        chunk_key,
        # The EXTRACTION-VISIBLE text: parser-internal markup stripped, then
        # the sanitize_text_for_encoding that use_llm_func_with_cache applies
        # to the whole prompt. The invariant is that the hook sees
        # byte-for-byte what sat between the ---Input Text--- fences of the
        # prompt the provider received. strip=False follows from it: the chunk
        # is a FRAGMENT sitting inside those fences, so its own boundary
        # whitespace is interior to the prompt and stays model-visible, while
        # the wrapper's strip only trims the template's ends.
        sanitize_text_for_encoding(
            strip_internal_multimodal_markup_for_extraction(chunk_dp["content"]),
            strip=False,
        ),
        maybe_nodes,
        maybe_edges,
    )
    if inspect.isawaitable(validated):
        validated = await validated
    # Validate the shape before unpacking: a bare unpack accepts a two-key
    # dict and silently binds its KEYS as maybe_nodes / maybe_edges, surfacing
    # much later inside the merge as an unrelated-looking failure.
    if (
        not isinstance(validated, (tuple, list))
        or len(validated) != 2
        or not all(isinstance(part, dict) for part in validated)
    ):
        raise TypeError(
            "kg_extraction_validator must return a (maybe_nodes, "
            "maybe_edges) pair of dicts; got "
            f"{type(validated).__name__} for chunk {chunk_key}"
        )
    return validated[0], validated[1]


def _inject_multimodal_records(
    chunk_key: str,
    chunk_dp: dict[str, Any],
    maybe_nodes: dict[str, list[dict]],
    maybe_edges: dict[tuple[str, str], list[dict]],
) -> None:
    """Add the core-generated multimodal entity and its edges for one chunk.

    Drawing, table and equation chunks carry a ``sidecar`` block; their entity
    and its association edges are built from that metadata, not from LLM
    output, so a cached extraction row never contains them. Both ingestion
    (``extract_entities``) and the deletion rebuild call this on the chunk's
    parsed records, so the rebuild sees the same evidence ingestion merged.
    Mutates ``maybe_nodes`` / ``maybe_edges`` in place.
    """
    file_path = chunk_dp.get("file_path", "unknown_source")
    sidecar_block = chunk_dp.get("sidecar")
    if isinstance(sidecar_block, dict):
        sidecar_type = sidecar_block.get("type")
        sidecar_id = sidecar_block.get("id")
        if (
            sidecar_type in {"drawing", "table", "equation"}
            and isinstance(sidecar_id, str)
            and sidecar_id
        ):
            mm_entity_name = sidecar_id
            now_ts = int(time.time())
            mm_nodes_list = maybe_nodes.setdefault(mm_entity_name, [])
            mm_nodes_list.append(
                {
                    "entity_name": mm_entity_name,
                    "entity_type": sidecar_type,
                    # description == the full multimodal chunk content so
                    # the extracted entity carries the same grounding
                    # surface the prompt produced; analyze_multimodal's
                    # description/name field is already inlined there.
                    "description": chunk_dp.get("content", "") or "",
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": now_ts,
                }
            )
            heading_block = chunk_dp.get("heading")
            heading_label = ""
            if isinstance(heading_block, dict):
                heading_label = str(heading_block.get("heading") or "").strip()
            # Omit the "in section ..." clause entirely when the chunk has no
            # real heading — a literal "unknown" filler would otherwise leak
            # into the relation description, embedding, and retrieval as noise.
            location = (
                f"in section {heading_label} of document"
                if heading_label
                else "of document"
            )
            mm_display_name = _parse_mm_display_name(
                chunk_dp.get("content", "") or "", sidecar_id
            )
            for tgt in list(maybe_nodes.keys()):
                if tgt == mm_entity_name:
                    continue
                edge_key = (mm_entity_name, tgt)
                edge_list = maybe_edges.setdefault(edge_key, [])
                edge_list.append(
                    {
                        "src_id": mm_entity_name,
                        "tgt_id": tgt,
                        "weight": 1.0,
                        "description": (
                            f"{tgt} is associated with {sidecar_type} "
                            f"{mm_display_name} {location} "
                            f'"{file_path}"'
                        ),
                        "keywords": "associated with, contained in",
                        "source_id": chunk_key,
                        "file_path": file_path,
                        "timestamp": now_ts,
                    }
                )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
    truncation_tally: TokenLimitTruncationTally | None = None,
) -> list:
    """Extract entities and relations from ``chunks``.

    ``truncation_tally``: optional document-scoped accumulator. Extraction owns
    a stage-scoped tally regardless (it publishes the first truncation
    immediately and one aggregate at the end); passing this one additionally
    hands the counts to the pipeline, which stamps them into
    ``doc_status.metadata`` so the condition outlives the bounded status ring.
    """
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    # Operation-scoped status logger shared by all chunk tasks: the first
    # history write fetches the shared list once; every per-chunk log is then
    # a single extend on the cached handle.
    status_logger = PipelineStatusLogger(pipeline_status)

    # Extraction-scoped truncation tally; see _publish_truncation_summary below.
    stage_tally = TokenLimitTruncationTally()

    # Chunks whose LLM cache write was skipped because the chunk could not
    # carry the reference to it. A set, not a tally: there is no
    # per-stage breakdown to report and nothing is persisted — see
    # _publish_cache_skip_summary below.
    cache_skip_chunks: set[str] = set()

    # Optional per-chunk extraction-quality hook; None leaves the pipeline
    # unchanged. See the call site in _process_single_content below.
    kg_extraction_validator = global_config.get("kg_extraction_validator")

    use_llm_func: callable = global_config["role_llm_funcs"]["extract"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    # Cap on the gleaning LLM call's combined input (system + history user
    # prompt + history assistant response + continue prompt).  Pulled from
    # the same env knob that gates ``analyze_multimodal``'s sidecar trimming
    # so both EXTRACT-role consumers share one source of truth.  ``0``
    # disables the gleaning guard (gleaning always runs regardless of size).
    max_extract_input_tokens = get_env_value(
        "MAX_EXTRACT_INPUT_TOKENS",
        DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
        int,
    )
    extract_tokenizer: Tokenizer | None = global_config.get("tokenizer")

    # Check if JSON structured output mode is enabled
    use_json_extraction = global_config.get("entity_extraction_use_json", False)

    ordered_chunks = list(chunks.items())
    if not ordered_chunks:
        return []
    # add language and example number params to prompt
    addon_params = global_config.get("addon_params") or {}
    language = global_config.get("_resolved_summary_language")
    if language is None:
        language = addon_params.get("language", DEFAULT_SUMMARY_LANGUAGE)
    prompt_profile = global_config.get("_entity_extraction_prompt_profile")
    if prompt_profile is None:
        # Fallback for callers that construct global_config directly (e.g. tests
        # or custom wiring). Re-run the resolver so behavior matches the cached
        # path that LightRAG.__post_init__ populates, instead of duplicating
        # guidance/override logic here.
        prompt_profile = resolve_entity_extraction_prompt_profile(
            addon_params, use_json_extraction
        )
    entity_types_guidance = prompt_profile["entity_types_guidance"]

    max_total_records = global_config["entity_extract_max_records"]
    max_entity_records = global_config["entity_extract_max_entities"]

    if use_json_extraction:
        # JSON mode: use JSON-specific prompts without delimiters
        examples = "\n".join(prompt_profile["entity_extraction_json_examples"])
        context_base = dict(
            entity_types_guidance=entity_types_guidance,
            examples=examples,
            language=language,
            max_total_records=max_total_records,
            max_entity_records=max_entity_records,
        )
    else:
        # Text mode: use traditional delimiter-based prompts
        examples = "\n".join(prompt_profile["entity_extraction_examples"])
        example_context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types_guidance=entity_types_guidance,
            language=language,
        )
        # add example's format
        examples = examples.format(**example_context_base)

        context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types_guidance=entity_types_guidance,
            examples=examples,
            language=language,
            max_total_records=max_total_records,
            max_entity_records=max_entity_records,
        )

    # The system prompt (and, in JSON mode, the continue prompt) is built
    # solely from context_base, which is fixed for the whole extraction run.
    # Format them once here instead of re-formatting per chunk, and reuse a
    # precomputed token count for the same strings in the gleaning guard —
    # the system prompt embeds the multi-thousand-token examples block. The
    # encodes stay lazy behind the same condition that gates the gleaning
    # guard, so runs with gleaning off never pay them.
    gleaning_precheck = (
        entity_extract_max_gleaning > 0
        and extract_tokenizer is not None
        and max_extract_input_tokens > 0
    )
    if use_json_extraction:
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_json_system_prompt"
        ].format(**context_base)
        json_continue_extraction_user_prompt = PROMPTS[
            "entity_continue_extraction_json_user_prompt"
        ].format(**context_base)
        gleaning_invariant_tokens = (
            len(extract_tokenizer.encode(entity_extraction_system_prompt))
            + len(extract_tokenizer.encode(json_continue_extraction_user_prompt))
            if gleaning_precheck
            else 0
        )
    else:
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_system_prompt"
        ].format(**context_base)
        gleaning_invariant_tokens = (
            len(extract_tokenizer.encode(entity_extraction_system_prompt))
            if gleaning_precheck
            else 0
        )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        # Strip parser-internal markup (<cite refid>, <drawing id/path/src>,
        # <equation id>) before building the extraction prompt. The stored
        # chunk content is left intact so query-time citations still resolve.
        content = strip_internal_multimodal_markup_for_extraction(chunk_dp["content"])
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Build the optional `---Section Context---` block from the chunk's
        # heading breadcrumb. The marker/wrapping lives entirely in the prompt
        # template; here we only produce the data and decide whether to inject
        # it. Each level is char-capped inside format_heading_context, and the
        # joined path is token-budgeted here so heading metadata can never push
        # an otherwise-valid chunk past the provider context window. When the
        # chunk carries no heading, the block is an empty string so the user
        # prompt stays byte-identical to the no-context form.
        heading_path = _truncate_section_context(
            format_heading_context(chunk_dp),
            extract_tokenizer,
            DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
        )
        heading_context_block = (
            PROMPTS["entity_extraction_section_context"].format(
                heading_path=heading_path
            )
            if heading_path
            else ""
        )

        def _report_truncation(result: str, stage: str) -> None:
            if not is_truncated_response(result):
                return
            location = f"chunk {chunk_key} in {file_path}"
            # The server log keeps every occurrence with full identity; the
            # bounded pipeline-status ring gets the first one plus the
            # end-of-stage aggregate (see _publish_truncation_summary).
            logger.warning(
                f"Token-limit truncation during {stage} entity extraction "
                f"for {location}"
            )
            if stage_tally.record(stage, chunk_key):
                status_logger.log(
                    f"Warning: token-limit truncation during {stage} entity "
                    f"extraction for {location}; further occurrences are "
                    f"reported as one summary at the end of extraction"
                )

        def _report_cache_skip(cache_type: str) -> None:
            """The extraction cache is off for this chunk; say so out loud.

            ``use_llm_func_with_cache`` skips the cache write when the chunk
            could not carry the reference to it. That is the safe direction,
            but it means the extraction cache silently stopped working for this
            document — the next run re-calls the LLM for these chunks. Reported on the
            same discipline as truncation above: every occurrence to the server
            log, the first one plus an end-of-stage aggregate to the bounded
            pipeline-status ring. Synchronous and non-raising, per the
            ``on_cache_skipped`` contract.
            """
            location = f"chunk {chunk_key} in {file_path}"
            logger.warning(
                f"LLM {cache_type} cache write skipped for {location}: its cache "
                f"reference could not be recorded on the chunk"
            )
            first = not cache_skip_chunks
            cache_skip_chunks.add(chunk_key)
            if first:
                status_logger.log(
                    f"Warning: LLM cache write skipped for {location} because its "
                    f"cache reference could not be recorded; further occurrences "
                    f"are reported as one summary at the end of extraction"
                )

        if use_json_extraction:
            # JSON mode: use JSON prompts and pass entity_extraction flag to LLM provider.
            # The local must be bound explicitly: the text-mode branch below
            # assigns this name, which makes it local to the whole closure
            # (Python scoping is static), so the gleaning LLM call a few
            # dozen lines down would otherwise read an unbound local here.
            entity_continue_extraction_user_prompt = (
                json_continue_extraction_user_prompt
            )
            entity_extraction_user_prompt = PROMPTS[
                "entity_extraction_json_user_prompt"
            ].format(
                **{
                    **context_base,
                    "input_text": content,
                    "heading_context_block": heading_context_block,
                }
            )
        else:
            # Text mode: use traditional delimiter-based prompts. The
            # chunk-invariant system prompt is built once in the enclosing
            # scope. The continue prompt stays per chunk because a custom
            # prompt file may reference {input_text}; the default template's
            # placeholders do not include it, but the call site keeps passing
            # the kwarg for that override case.
            entity_extraction_user_prompt = PROMPTS[
                "entity_extraction_user_prompt"
            ].format(
                **{
                    **context_base,
                    "input_text": content,
                    "heading_context_block": heading_context_block,
                }
            )
            entity_continue_extraction_user_prompt = PROMPTS[
                "entity_continue_extraction_user_prompt"
            ].format(**{**context_base, "input_text": content})

        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            text_chunks_storage=text_chunks_storage,
            on_cache_skipped=_report_cache_skip,
            response_format=({"type": "json_object"} if use_json_extraction else None),
            llm_cache_identity=get_llm_cache_identity(global_config, "extract"),
        )

        _report_truncation(final_result, "initial")

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction with appropriate parser
        if use_json_extraction:
            maybe_nodes, maybe_edges = await _process_json_extraction_result(
                final_result,
                chunk_key,
                timestamp,
                file_path,
            )
        else:
            maybe_nodes, maybe_edges = await _process_extraction_result(
                final_result,
                chunk_key,
                timestamp,
                file_path,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        run_gleaning = entity_extract_max_gleaning > 0
        if (
            run_gleaning
            and extract_tokenizer is not None
            and max_extract_input_tokens > 0
        ):
            # Gleaning replays the initial extraction's user/assistant pair
            # via ``history_messages`` and appends a "continue" instruction.
            # When the initial response was large (many entities/edges) or
            # the chunk content is itself near the budget, that combined
            # payload can blow past MAX_EXTRACT_INPUT_TOKENS and yield a
            # provider ``context_length_exceeded`` error.  Pre-check here
            # and skip rather than fail.
            gleaning_token_count = (
                gleaning_invariant_tokens
                + sum(
                    len(extract_tokenizer.encode(msg.get("content", "") or ""))
                    for msg in history
                )
                + (
                    # Text mode only: the continue prompt embeds the chunk
                    # content, so its token count is chunk-specific.
                    len(
                        extract_tokenizer.encode(entity_continue_extraction_user_prompt)
                    )
                    if not use_json_extraction
                    else 0
                )
            )
            if gleaning_token_count > max_extract_input_tokens:
                logger.warning(
                    f"Gleaning stopped for chunk {chunk_key}: "
                    f"Input tokens ({gleaning_token_count}) exceeded limit "
                    f"({max_extract_input_tokens})."
                )
                run_gleaning = False

        if run_gleaning:
            glean_result, timestamp = await use_llm_func_with_cache(
                entity_continue_extraction_user_prompt,
                use_llm_func,
                system_prompt=entity_extraction_system_prompt,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                text_chunks_storage=text_chunks_storage,
                on_cache_skipped=_report_cache_skip,
                response_format=(
                    {"type": "json_object"} if use_json_extraction else None
                ),
                llm_cache_identity=get_llm_cache_identity(global_config, "extract"),
            )

            _report_truncation(glean_result, "gleaning")

            # Process gleaning result with appropriate parser
            if use_json_extraction:
                glean_nodes, glean_edges = await _process_json_extraction_result(
                    glean_result,
                    chunk_key,
                    timestamp,
                    file_path,
                )
            else:
                glean_nodes, glean_edges = await _process_extraction_result(
                    glean_result,
                    chunk_key,
                    timestamp,
                    file_path,
                    tuple_delimiter=context_base["tuple_delimiter"],
                    completion_delimiter=context_base["completion_delimiter"],
                )

            # Merge results - compare description lengths to choose better version
            for i, (entity_name, glean_entities) in enumerate(
                glean_nodes.items(), start=1
            ):
                if entity_name in maybe_nodes:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_nodes[entity_name][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_entities[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_nodes[entity_name] = list(glean_entities)
                    # Otherwise keep original version
                else:
                    # New entity from gleaning stage
                    maybe_nodes[entity_name] = list(glean_entities)
                await _cooperative_yield(i, every=8)

            for i, (edge_key, glean_edge_list) in enumerate(
                glean_edges.items(), start=1
            ):
                if edge_key in maybe_edges:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_edges[edge_key][0].get("description", "") or ""
                    )
                    glean_desc_len = len(
                        glean_edge_list[0].get("description", "") or ""
                    )

                    if glean_desc_len > original_desc_len:
                        maybe_edges[edge_key] = list(glean_edge_list)
                    # Otherwise keep original version
                else:
                    # New edge from gleaning stage
                    maybe_edges[edge_key] = list(glean_edge_list)
                await _cooperative_yield(i, every=8)

        # No end-of-chunk cache-key attach here, by design. Each extract cache
        # row is attached to this chunk BEFORE it is written, inside
        # use_llm_func_with_cache; collecting the keys in memory and attaching
        # them once at the end is precisely what orphaned them. See *LLM
        # extraction cache reachability* in the contract doc,
        # docs/design/PurgeRecoveryContract.md. Nothing here needs to re-attach:
        # the extraction and gleaning calls above have each recorded their own
        # key, and the multimodal injection below builds records from sidecar
        # metadata without calling the LLM.

        # Optional extraction-quality hook: the last word on what the LLM
        # extracted from this chunk. Caller-facing contract, including why core
        # never prunes on the hook's behalf, is documented under "Extraction
        # Quality Hook" in docs/ProgramingWithCore.md.
        #
        # Deliberately placed BEFORE the multimodal injection below. That entity
        # is core-generated, not LLM output, and it is linked to every surviving
        # entity of the chunk: filtering afterwards would both expose it to
        # rules written for LLM output (a "name must appear in chunk_text"
        # grounding check deletes it) and leave the injected
        # (multimodal_entity, rejected_entity) edges dangling, so the merge's
        # endpoint upsert would re-create the very entities the hook rejected.
        if kg_extraction_validator is not None:
            maybe_nodes, maybe_edges = await _apply_kg_extraction_validator(
                kg_extraction_validator, chunk_key, chunk_dp, maybe_nodes, maybe_edges
            )

        # Inject multimodal entity + associations for drawing/table/equation
        # chunks. Placed inside the chunk's concurrency slot (rather than a
        # centralized post-pass) so each multimodal chunk benefits from the
        # chunk-level concurrency already enforced here, and after the
        # validator (see the comment above it).
        _inject_multimodal_records(chunk_key, chunk_dp, maybe_nodes, maybe_edges)

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)
        status_logger.log(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    def _publish_truncation_summary() -> None:
        """Publish one aggregated truncation line and hand the counts upward.

        Called exactly once, from a ``finally`` covering the wait/drain on
        the chunk tasks, so it runs on EVERY exit — success, a failing chunk,
        and an external cancellation landing on the wait itself (which used
        to bypass the two explicit call sites and lose already-recorded
        truncations from the caller's document tally). Await-free, so it is
        safe inside a cancellation unwind. A no-op when nothing truncated.
        """
        if not stage_tally:
            return
        truncation_message = (
            f"Warning: token-limit truncation hit {stage_tally.affected} of "
            f"{total_chunks} chunks during entity extraction "
            f"({stage_tally.stage_breakdown()}); "
            f"extracted knowledge may be incomplete"
        )
        logger.warning(truncation_message)
        status_logger.log(truncation_message)
        if truncation_tally is not None:
            truncation_tally.absorb(stage_tally)

    def _publish_cache_skip_summary() -> None:
        """Publish one aggregated line for chunks whose cache write was skipped.

        Called from the same ``finally`` as _publish_truncation_summary and
        under the same rules: exactly once, on every exit, await-free so it is
        safe inside a cancellation unwind, a no-op when nothing was skipped.
        Nothing is handed upward — unlike truncation this is not recorded on
        the document, because the trigger is an unwritable text_chunks storage
        rather than a property of the document's content.
        """
        if not cache_skip_chunks:
            return
        cache_skip_message = (
            f"Warning: LLM cache writes were skipped for {len(cache_skip_chunks)} "
            f"of {total_chunks} chunks during entity extraction because their "
            f"cache references could not be recorded; those extraction results "
            f"were not cached and will be recomputed on the next run"
        )
        logger.warning(cache_skip_message)
        status_logger.log(cache_skip_message)

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                result = await _process_single_content(chunk)
                # Yield once between chunk completions so API coroutines can resume
                # even when many chunk tasks are hitting cache and finishing quickly.
                await asyncio.sleep(0)
                return result
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    try:
        # Wait for tasks to complete or for the first exception to occur
        # This allows us to cancel remaining tasks if any task fails
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        # Check if any task raised an exception and ensure all exceptions are retrieved.
        #
        # ORDER: `done` is a SET whose iteration order derives from object identity
        # (Task inherits object.__hash__, i.e. its memory address), so it is neither
        # completion order nor creation order, and it is unstable across processes.
        # Collecting results from it made `chunk_results` a different permutation on
        # every run of the same document. That permutation reaches persisted state
        # through merge_nodes_and_edges: `source_id` and `file_path` are built by
        # order-preserving dedup with no sort to fall back on, and
        # apply_source_ids_limit then truncates a DIFFERENT subset of chunks. (The
        # description lists are sorted by (timestamp, -len) downstream, so those are
        # only exposed on ties -- the persisted id/path fields are the unguarded
        # ones.) Results are therefore materialised from `tasks`, a list built in
        # `ordered_chunks` order, so chunk_results[i] always maps to ordered_chunks[i].
        #
        # The two passes cannot be folded into one loop over `tasks`: on the
        # exception path `tasks` still holds pending entries, and calling
        # .exception() on those raises InvalidStateError, which the `except Exception`
        # below would latch as `first_exception`, masking the real failure.
        first_exception = None
        chunk_results = []

        for task in done:
            try:
                exception = task.exception()
                if exception is not None and first_exception is None:
                    first_exception = exception
            except Exception as e:
                if first_exception is None:
                    first_exception = e

        # No exception means FIRST_EXCEPTION behaved as ALL_COMPLETED: `pending` is
        # empty and every task in `tasks` holds a result. On the exception path we
        # are about to unwind, so materialising results there would be wasted work.
        if first_exception is None:
            chunk_results = [task.result() for task in tasks]

        # If any task failed, cancel all pending tasks and raise the first exception
        if first_exception is not None:
            # Cancel all pending tasks
            for pending_task in pending:
                pending_task.cancel()

            # Wait for cancellation to complete
            if pending:
                await asyncio.wait(pending)

            # Add progress prefix to the exception message
            progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

            # Re-raise the original exception with a prefix
            prefixed_exception = create_prefixed_exception(
                first_exception, progress_prefix
            )
            raise prefixed_exception from first_exception
    except (Exception, asyncio.CancelledError):
        # asyncio.wait never cancels the tasks it waits on. Without this, an
        # external cancellation of this coroutine (pipeline cancel, server
        # shutdown) leaves the chunk tasks running: still calling the LLM for
        # a document already abandoned, and failing later with nobody left to
        # retrieve the exception ("Task exception was never retrieved").
        # Same rule as wait_tasks_with_drain: cancel what still runs and
        # collect every outcome before unwinding. On the failure path above
        # the tasks are already done; this then only retrieves whatever the
        # cancelled siblings raised.
        await cancel_and_drain_tasks(tasks)
        raise
    finally:
        # On EVERY exit — mirrors merge_nodes_and_edges. The success and
        # failure paths used to publish explicitly, but an external
        # cancellation landing on the asyncio.wait above bypassed both,
        # and the caller's failure bookkeeping (custom chunks) then
        # persisted FAILED with no llm_truncation for responses that had
        # already been recorded. Await-free, called exactly once.
        _publish_truncation_summary()
        _publish_cache_skip_summary()

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results


# Policy version of the query-answer cache (cache_type="query"). Bump it when the
# meaning of an entry changes in a way the other key fields cannot express, so
# entries written by older versions become unreachable instead of being served
# under a key whose semantics have moved. v2 retires every entry written before
# the _answer_cache_kv bypass below: such an entry may hold history-conditioned
# text filed under a history-blind key, and entries record no history, so a
# tainted entry cannot be told apart from a clean one. Only the answer cache is
# versioned; keyword/extract/summary entries never see conversation_history.
_ANSWER_CACHE_POLICY_VERSION = "query-answer-cache-v2"


def _answer_cache_kv(
    query_param: QueryParam, hashing_kv: BaseKVStorage | None
) -> BaseKVStorage | None:
    """Return the storage backing the query-answer cache, or None to bypass it.

    The answer cache key deliberately excludes ``conversation_history``: every
    turn of a conversation carries a different history, so keying on it would
    make each entry unique and turn a cache that is meant to be shared across
    callers into a per-session one. But the history *is* handed to the model as
    ``history_messages``, so an answer generated under it is not interchangeable
    with the history-blind key it would be filed under. Requests carrying a
    history therefore use neither side of the cache:

    - they never write, so caller-supplied history cannot decide the answer that
      other callers will be served for the same question;
    - they never read, so a multi-turn caller is not served an answer that was
      generated while ignoring its history.

    Keyword extraction is unaffected and keeps using ``hashing_kv`` directly: it
    derives keywords from the query text alone and never receives the history.
    """
    if query_param.conversation_history:
        logger.debug(
            " == LLM cache == Query answer cache bypassed: conversation_history "
            f"is set ({len(query_param.conversation_history)} message(s), "
            f"mode:{query_param.mode})"
        )
        return None
    return hashing_kv


def _drop_reasoning_only_cache_hit(
    cached_result: tuple[str, int] | None,
) -> tuple[str, int] | None:
    """Treat a cached reasoning-only answer as a cache miss.

    Entries written before ``reject_reasoning_only_answer`` guarded the cache
    write can hold a ``<think>`` trace with no answer. Serving one would
    replay it forever; a miss regenerates the answer, and the regenerated
    one overwrites the entry (or the generation raises, as for a fresh miss).
    """
    if cached_result is not None and is_reasoning_only_answer(cached_result[0]):
        logger.warning(
            " == LLM cache == Ignoring a cached query answer that carries only "
            "a reasoning trace; regenerating it"
        )
        return None
    return cached_result


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
    progress_callback: ProgressCallback | None = None,
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Based on different query_param settings, different fields will be populated:
        - only_need_context=True: content contains context string
        - only_need_prompt=True: content contains complete prompt
        - stream=True: response_iterator contains streaming response, raw_data contains complete data
        - default: content contains LLM response text, raw_data contains complete data

        Returns None when no relevant context could be constructed for the query.
    """
    if not query:
        return QueryResult(content=PROMPTS["fail_response"], llm_generated=False)

    # Apply higher priority (5) to query relation LLM function
    use_model_func = partial(
        global_config["role_llm_funcs"]["query"], _priority=DEFAULT_QUERY_PRIORITY
    )
    llm_cache_identity = get_llm_cache_identity(global_config, "query")

    if progress_callback:
        await progress_callback(QueryProgress.EXTRACTING_KEYWORDS)
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords. An empty list that the search covers with a
    # fallback is reported there (_resolve_kg_search_queries), with the
    # substitution it makes; warn here only when nothing stands in for it.
    if (
        ll_keywords == []
        and query_param.mode in ["local", "hybrid", "mix"]
        and not hl_keywords
    ):
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and (
        query_param.mode in ["hybrid", "mix"]
        or (query_param.mode == "global" and not ll_keywords)
    ):
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return QueryResult(content=PROMPTS["fail_response"], llm_generated=False)

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build query context (unified interface)
    context_result = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
        progress_callback=progress_callback,
        # The token budget must be computed against the template this function
        # will actually render below, not the default one.
        system_prompt=system_prompt,
        user_prompt_prefix=global_config.get("user_prompt_prefix", ""),
    )

    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    # Report the keyword lists themselves: splitting the joined search strings
    # on ", " again would break a keyword that contains one ("Washington, D.C.").
    context_result.raw_data.setdefault("metadata", {})["keywords"] = {
        "high_level": list(hl_keywords) if hl_keywords else [],
        "low_level": list(ll_keywords) if ll_keywords else [],
    }

    # Return different content based on query parameters
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_result.context,
            raw_data=context_result.raw_data,
            llm_generated=False,
        )

    effective_user_prompt = resolve_user_prompt(
        query_param.user_prompt,
        global_config.get("user_prompt_prefix", ""),
        query_param.disable_user_prompt_prefix,
    )
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=effective_user_prompt.slot,
        context_data=context_result.context,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(
            content=prompt_content,
            raw_data=context_result.raw_data,
            llm_generated=False,
        )

    # Call LLM
    tokenizer: Tokenizer = global_config["tokenizer"]
    # Guarded rather than unconditional: this whole block exists for a debug log,
    # and an f-string is evaluated eagerly, so the three encodes below used to run
    # at every log level over the largest string on the query path.
    if logger.isEnabledFor(logging.DEBUG):
        query_prompt_tokens = await acount_tokens(tokenizer, query)
        sys_prompt_tokens = await acount_tokens(tokenizer, sys_prompt)
        len_of_prompts = await acount_tokens(tokenizer, query + sys_prompt)
        logger.debug(
            f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens "
            f"(Query: {query_prompt_tokens}, System: {sys_prompt_tokens})"
        )

    # Handle cache
    answer_cache_kv = _answer_cache_kv(query_param, hashing_kv)
    # The chunk-selection settings must be read from the STORAGE snapshot, not
    # from the `global_config` parameter, even though the two usually agree.
    # A storage captures `asdict(self)` once at construction, while `aquery`
    # rebuilds `global_config` on every call, so the two diverge as soon as the
    # attribute is mutated on a live LightRAG. The code that actually consumes
    # these settings (`_find_most_related_text_unit_from_entities` /
    # `..._from_relationships`) reads the snapshot, so keying on the same dict
    # the consumer reads is what keeps the key and the retrieved context in
    # sync. Do NOT "unify" this with the `global_config` reads above.
    retrieval_config = (
        text_chunks_db.global_config if text_chunks_db is not None else global_config
    )
    related_chunk_number = retrieval_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )
    kg_chunk_pick_method = retrieval_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    keyword_fallback = _kg_keyword_fallback(
        query_param.mode, ll_keywords_str, hl_keywords_str
    )
    args_hash = compute_args_hash(
        _ANSWER_CACHE_POLICY_VERSION,
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        hl_keywords_str,
        ll_keywords_str,
        # The COMPOSED instructions, so changing the server-side prefix
        # invalidates entries generated under the old one. With no prefix
        # configured this is byte-identical to the previous
        # `query_param.user_prompt or ""`, so existing entries keep hitting --
        # which is why _ANSWER_CACHE_POLICY_VERSION does not need a bump.
        # `disable_user_prompt_prefix` is deliberately NOT a separate key
        # component: it only ever acts through this value, and adding it would
        # split the cache between two requests that build identical prompts.
        effective_user_prompt.text,
        query_param.enable_rerank,
        global_config.get("enable_content_headings", False),
        # Unconditional, and read from `retrieval_config` -- see the comment on
        # its assignment above for why the source differs from the line above.
        "\n<kg_chunk_selection>\n",
        related_chunk_number,
        kg_chunk_pick_method,
        # Unconditional: max_total_tokens now caps the whole rendered KG
        # context, so the same request can keep fewer graph records or chunks
        # (with regenerated reference IDs) than the code that wrote an older
        # entry. Serving that entry pairs its citations with a reference list
        # that no longer has them, so every KG answer written before must
        # miss. Only the KG key carries it: the naive context is unchanged.
        "\n<kg_total_budget>\n",
        *(("\n<system_prompt>\n", system_prompt) if system_prompt else ()),
        # Present only when the search swaps keyword lists between stores. An
        # answer cached before that routing existed was written from another
        # store's context, so it must miss; every other entry keeps hitting,
        # which is why _ANSWER_CACHE_POLICY_VERSION does not need a bump.
        *(("\n<kg_keyword_fallback>\n", keyword_fallback) if keyword_fallback else ()),
        "\n<llm_identity>\n",
        serialize_llm_cache_identity(llm_cache_identity),
    )

    cached_result = _drop_reasoning_only_cache_hit(
        await handle_cache(
            answer_cache_kv, args_hash, user_query, query_param.mode, cache_type="query"
        )
    )

    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        if progress_callback:
            await progress_callback(QueryProgress.GENERATING_RESPONSE)
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )
        # A reasoning trace with no answer is a failed generation, not an
        # answer: never return it as one, never cache it.
        reject_reasoning_only_answer(response)

        if (
            answer_cache_kv
            and answer_cache_kv.global_config.get("enable_llm_cache")
            and not is_truncated_response(response)
        ):
            queryparam_dict = {
                "answer_cache_version": _ANSWER_CACHE_POLICY_VERSION,
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "hl_keywords": hl_keywords_str,
                "ll_keywords": ll_keywords_str,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
                "enable_content_headings": global_config.get(
                    "enable_content_headings", False
                ),
                "related_chunk_number": related_chunk_number,
                "kg_chunk_pick_method": kg_chunk_pick_method,
            }
            await save_to_cache(
                answer_cache_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response,
            raw_data=context_result.raw_data,
            is_streaming=True,
        )


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords directly from the current query text.
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


# The placeholder tokens of PROMPTS["keywords_extraction_examples"], with an
# optional numeric suffix so the template may list several per array. Matched
# by name, never by shape: "<div>" is a legitimate keyword for an HTML query.
_KEYWORD_TEMPLATE_PLACEHOLDER = re.compile(r"<(?:high|low)_level_keyword(?:_\d+)?>")


def _normalize_keyword_list(raw_values: Any, field_name: str) -> list[str]:
    """Normalize keyword payloads into a clean list of strings.

    When the field is a plain string (e.g. LLM returned CSV), split on
    newlines/commas/semicolons. List-shaped payloads are preserved per-item so
    multi-word phrases that legitimately contain commas are not broken apart.
    """

    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        raw_values = [
            part.strip()
            for part in re.split(r"[\n,;]+", raw_values)
            if part and part.strip()
        ]

    if not isinstance(raw_values, list):
        logger.warning(
            "Keyword extraction field '%s' is not a list: %r",
            field_name,
            raw_values,
        )
        return []

    normalized: list[str] = []
    for idx, value in enumerate(raw_values):
        if isinstance(value, str):
            cleaned = value.strip()
            # An echoed template placeholder is not a keyword: kept, it would
            # be embedded for retrieval and cached for every later run.
            if cleaned and not _KEYWORD_TEMPLATE_PLACEHOLDER.fullmatch(cleaned):
                normalized.append(cleaned)
            continue

        logger.warning(
            "Keyword extraction field '%s' contains non-string element at index %d: %r",
            field_name,
            idx,
            value,
        )

    return normalized


def _select_keywords_object(text: str, candidates: list[Any]) -> Any:
    """Pick the keyword object out of a reply that holds several JSON objects.

    ``json_repair`` returns a list when a reply repeats itself, e.g. a fenced
    object, a "let me fix my format" line, then a second object. Returns the
    first object that carries a real keyword, so an echoed format template, an
    echoed empty-list example or an unrelated leading object does not shadow
    the answer; failing that the first object, so an all-empty reply is still
    a valid "no keywords" answer.

    Returns ``candidates`` unchanged, which the caller rejects, when the reply
    is a top-level array or holds no object at all: an array element is never
    taken for the whole answer.
    """
    opener, _ = _first_structural_opener(text)
    if opener != "{":
        return candidates

    objects = [candidate for candidate in candidates if isinstance(candidate, dict)]
    for candidate in objects:
        if any(
            _normalize_keyword_list(candidate.get(field_name), field_name)
            for field_name in ("high_level_keywords", "low_level_keywords")
        ):
            return candidate
    return objects[0] if objects else candidates


def _parse_keywords_payload(result: Any) -> tuple[bool, list[str], list[str]]:
    """Parse keyword extraction responses from heterogeneous provider outputs."""

    payload: Any

    if result is None:
        return False, [], []

    if hasattr(result, "model_dump") and callable(result.model_dump):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    elif isinstance(result, str):
        cleaned_result = remove_think_tags(result)
        unfenced_result = strip_markdown_code_fence(cleaned_result)
        if unfenced_result is not cleaned_result:
            logger.debug(
                "Stripped markdown code fence from keyword extraction response"
            )
            cleaned_result = unfenced_result
        try:
            payload = json.loads(cleaned_result)
        except json.JSONDecodeError as strict_error:
            try:
                payload = json_repair.loads(cleaned_result)
                logger.warning(
                    "Keyword extraction response required JSON repair: %s; response: %r",
                    strict_error,
                    cleaned_result[:500],
                )
                if isinstance(payload, list):
                    payload = _select_keywords_object(cleaned_result, payload)
                elif not isinstance(payload, dict):
                    # json_repair reads '#' as a comment to the end of the line,
                    # so a "Result #1: {...}" lead-in repairs to ''. Recover the
                    # first object instead; keep the scalar (and so the
                    # rejection below) when there is none.
                    payload = tolerant_load_json_dict(cleaned_result) or payload
            except Exception as repair_error:
                logger.error(
                    "JSON parsing error: %s; repair failed: %s; response: %r",
                    strict_error,
                    repair_error,
                    cleaned_result[:500],
                )
                return False, [], []
    else:
        logger.error(
            "Unsupported keyword extraction response type: %s",
            type(result).__name__,
        )
        return False, [], []

    if not isinstance(payload, dict):
        logger.error(
            "Keyword extraction payload is not a JSON object: %s",
            type(payload).__name__,
        )
        return False, [], []

    hl_keywords = _normalize_keyword_list(
        payload.get("high_level_keywords"), "high_level_keywords"
    )
    ll_keywords = _normalize_keyword_list(
        payload.get("low_level_keywords"), "low_level_keywords"
    )
    return True, hl_keywords, ll_keywords


KEYWORD_EXTRACTION_JSON_MODES = ("auto", "json_object", "none")


def normalize_keyword_extraction_json_mode(value: Any) -> str:
    """Return the canonical ``keyword_extraction_json_mode``, or raise ``ValueError``."""
    mode = str(value).strip().lower()
    if mode not in KEYWORD_EXTRACTION_JSON_MODES:
        raise ValueError(
            "keyword_extraction_json_mode (KEYWORD_EXTRACTION_JSON_MODE) must be "
            f"'auto', 'json_object' or 'none', got {value!r}"
        )
    return mode


KG_CHUNK_PICK_METHODS = ("VECTOR", "WEIGHT")


def normalize_kg_chunk_pick_method(value: Any) -> str:
    """Return the canonical ``kg_chunk_pick_method``, or raise ``ValueError``.

    Applied once in ``LightRAG.__post_init__``, before the storages snapshot
    the configuration, so the query path reads only canonical values.
    """
    method = value.strip().upper() if isinstance(value, str) else None
    if method not in KG_CHUNK_PICK_METHODS:
        raise ValueError(
            "kg_chunk_pick_method (KG_CHUNK_PICK_METHOD) must be "
            f"'VECTOR' or 'WEIGHT', got {value!r}"
        )
    return method


def _is_unsupported_response_format_error(error: BaseException) -> bool:
    """Whether the provider rejected the ``response_format=json_object`` constraint itself.

    The fallback exists for OpenAI-compatible servers that accept only
    ``json_schema``/``text`` (LM Studio answers ``'response_format.type' must be
    'json_schema' or 'text'``). The message must name the field (``response_format``
    or ``json_object``) AND carry a rejection marker, so a generic 400, a timeout, or
    a transport error never qualifies: only this compatibility failure may be retried
    without the constraint, everything else must propagate.
    """
    # The OpenAI SDK prints the error ``type`` of every 400 as
    # ``invalid_request_error``; it says nothing about which field was rejected.
    message = str(error).lower().replace("invalid_request_error", "")
    if "response_format" not in message and "json_object" not in message:
        return False
    rejection_markers = (
        "unsupported",
        "not supported",
        "does not support",
        "not allowed",
        "not accept",
        "invalid",
        "must be",
        "only accept",
        "only support",
    )
    return any(marker in message for marker in rejection_markers)


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """
    json_mode = normalize_keyword_extraction_json_mode(
        global_config.get("keyword_extraction_json_mode", "auto")
    )

    # 1. Build the examples
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    addon_params = global_config.get("addon_params") or {}
    language = global_config.get("_resolved_summary_language")
    if language is None:
        language = addon_params.get("language", DEFAULT_SUMMARY_LANGUAGE)

    # 2. Handle cache if needed - add cache type for keywords
    llm_cache_identity = get_llm_cache_identity(global_config, "keyword")
    args_hash = compute_args_hash(
        param.mode,
        text,
        language,
        "\n<llm_identity>\n",
        serialize_llm_cache_identity(llm_cache_identity),
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        is_valid_payload, hl_keywords, ll_keywords = _parse_keywords_payload(
            cached_response
        )
        if is_valid_payload:
            return hl_keywords, ll_keywords
        else:
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 3. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    # Debug-only, so pay for the encode only when it will actually be printed.
    if logger.isEnabledFor(logging.DEBUG):
        len_of_prompts = await acount_tokens(tokenizer, kw_prompt)
        logger.debug(
            f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens "
            f"(Prompt: {len_of_prompts})"
        )

    # 4. Call the LLM for keyword extraction
    # Apply higher priority (5) to query relation LLM function
    use_model_func = partial(
        global_config["role_llm_funcs"]["keyword"], _priority=DEFAULT_QUERY_PRIORITY
    )

    # JSON mode policy (see keyword_extraction_json_mode): `auto` keeps the
    # API-enforced JSON constraint on the first attempt — supported providers must not
    # be downgraded to prompt-only parsing, which silently degrades to empty keywords —
    # and drops it only after the provider rejects the field.
    llm_kwargs: dict[str, Any] = (
        {"response_format": {"type": "json_object"}}
        if json_mode in ("auto", "json_object")
        else {}
    )

    try:
        result = await use_model_func(kw_prompt, **llm_kwargs)
    except Exception as error:
        if not (
            json_mode == "auto"
            and llm_kwargs
            and _is_unsupported_response_format_error(error)
        ):
            raise
        logger.warning(
            "Keyword extraction: provider rejected response_format=json_object (%s); "
            "retrying once without it",
            error,
        )
        result = await use_model_func(kw_prompt)

    # 5. Parse out JSON from the LLM response with tolerant provider normalization
    _, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    # 6. Cache only the processed keywords with cache type
    #    Skip caching when the LLM response was truncated by the token limit:
    #    the parsed keyword set may be incomplete, and caching it would replay
    #    the partial result on every later run.
    if (hl_keywords or ll_keywords) and not is_truncated_response(result):
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                # Metadata only. Keyword extraction has no {user_prompt} slot,
                # so neither this nor the server-side prefix influences it, and
                # neither belongs in the keyword cache key above.
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    # No broad try/except here -- mirrors _get_node_data/_get_edge_data, whose
    # entities_vdb/relationships_vdb queries are also left to propagate. A
    # backend query failure is not "zero relevant chunks": swallowing it here
    # let a transient vector-store error surface as a confident "no results"
    # answer instead of a retrieval failure (and, in mix mode, silently
    # dropped the vector-search branch while KG results kept flowing).
    search_top_k = query_param.chunk_top_k or query_param.top_k
    cosine_threshold = chunks_vdb.cosine_better_than_threshold

    results = await chunks_vdb.query(
        query, top_k=search_top_k, query_embedding=query_embedding
    )
    if not results:
        logger.info(
            f"Naive query: 0 chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
        )
        return []

    valid_chunks = []
    for result in results:
        if "content" in result:
            chunk_with_metadata = {
                "content": result["content"],
                "created_at": result.get("created_at", None),
                "file_path": result.get("file_path", "unknown_source"),
                "source_type": "vector",  # Mark the source type
                "chunk_id": result.get("id"),  # Add chunk_id for deduplication
            }
            valid_chunks.append(chunk_with_metadata)

    logger.info(
        f"Naive query: {len(valid_chunks)} chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
    )
    return valid_chunks


def _kg_keyword_fallback(mode: str, ll_keywords: str, hl_keywords: str) -> str:
    """Name the keyword fallback a KG search applies; "" when it applies none.

    The name is part of the answer-cache key (see ``kg_query``), so renaming a
    value retires the cached answers of every request that took that fallback.
    """
    if not ll_keywords and hl_keywords and mode in ("local", "hybrid", "mix"):
        return "entities_by_high_level_keywords"
    if not hl_keywords and ll_keywords and mode == "global":
        return "relationships_by_low_level_keywords"
    return ""


def _resolve_kg_search_queries(
    mode: str, ll_keywords: str, hl_keywords: str
) -> tuple[str, str]:
    """Return ``(entity_query, relation_query)`` for a KG search; "" skips a store.

    ``local`` owns the entity store, ``global`` the relationship store,
    ``hybrid`` / ``mix`` both. Low-level keywords normally drive the entity
    search and high-level keywords the relationship search.

    When keyword extraction leaves a mode's own list empty, the other list
    stands in for it on the SAME store. Weaker models routinely put the named
    entities under ``high_level_keywords`` and return no low-level ones, which
    would otherwise leave the entity store unqueried. The fallbacks are:

    * low-level empty, ``local`` / ``hybrid`` / ``mix``: high-level keywords
      query the entity store.
    * high-level empty, ``global``: low-level keywords query the relationship
      store.

    A mode never falls through to the store it does not own: ``local`` with no
    low-level keywords stays an entity search, it does not become ``global``.
    """
    entity_query = ll_keywords if mode in ("local", "hybrid", "mix") else ""
    relation_query = hl_keywords if mode in ("global", "hybrid", "mix") else ""

    fallback = _kg_keyword_fallback(mode, ll_keywords, hl_keywords)
    if fallback == "entities_by_high_level_keywords":
        entity_query = hl_keywords
        logger.warning(
            "low_level_keywords is empty: querying entities with "
            "high_level_keywords instead (mode=%s)",
            mode,
        )
    elif fallback == "relationships_by_low_level_keywords":
        relation_query = ll_keywords
        logger.warning(
            "high_level_keywords is empty: querying relationships with "
            "low_level_keywords instead (mode=global)"
        )
    return entity_query, relation_query


async def _perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Pure search logic that retrieves raw entities, relations, and vector chunks.
    No token truncation or formatting - just raw search results.
    """

    # Initialize result containers
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []
    vector_chunks = []
    chunk_tracking = {}

    # Handle different query modes

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute embeddings needed by the selected mode in a single batch call.
    # Only embed texts that the active retrieval branches will actually use:
    #   - query        → used by _get_vector_context (chunks VDB)
    #   - ll_keywords  → used by _get_node_data (entities VDB) in local/hybrid/mix
    #   - hl_keywords  → used by _get_edge_data (relationships VDB) in global/hybrid/mix
    # Batching avoids 2-3 sequential API round-trips.
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )

    actual_embedding_func = text_chunks_db.embedding_func
    query_embedding = None
    ll_embedding = None
    hl_embedding = None

    mode = query_param.mode
    entity_query, relation_query = _resolve_kg_search_queries(
        mode, ll_keywords, hl_keywords
    )
    need_ll = bool(entity_query)
    need_hl = bool(relation_query)

    if actual_embedding_func:
        texts_to_embed: list[str] = []
        text_purposes: list[str] = []

        if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
            texts_to_embed.append(query)
            text_purposes.append("query")

        if need_ll:
            texts_to_embed.append(entity_query)
            text_purposes.append("ll")

        # A fallback can route one keyword string to both stores; embed it once.
        share_keyword_embedding = need_ll and relation_query == entity_query
        if need_hl and not share_keyword_embedding:
            texts_to_embed.append(relation_query)
            text_purposes.append("hl")

        if texts_to_embed:
            try:
                all_embeddings = await actual_embedding_func(
                    texts_to_embed, context="query", _priority=DEFAULT_QUERY_PRIORITY
                )
                for i, purpose in enumerate(text_purposes):
                    if purpose == "query":
                        query_embedding = all_embeddings[i]
                    elif purpose == "ll":
                        ll_embedding = all_embeddings[i]
                    elif purpose == "hl":
                        hl_embedding = all_embeddings[i]
                if share_keyword_embedding:
                    hl_embedding = ll_embedding
                logger.debug(
                    "Pre-computed %d embeddings in single batch (purposes: %s)",
                    len(texts_to_embed),
                    ", ".join(text_purposes),
                )
            except Exception as e:
                logger.warning(f"Failed to batch pre-compute embeddings: {e}")

    # Each store is searched with the query string resolved above, so a mode
    # never reaches a store it does not own (see _resolve_kg_search_queries).
    if entity_query:
        if progress_callback:
            await progress_callback(QueryProgress.RETRIEVING_ENTITIES)
        local_entities, local_relations = await _get_node_data(
            entity_query,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
            query_embedding=ll_embedding,
        )

    if relation_query:
        if progress_callback:
            await progress_callback(QueryProgress.RETRIEVING_RELATIONS)
        global_relations, global_entities = await _get_edge_data(
            relation_query,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
            query_embedding=hl_embedding,
        )

    # Get vector chunks for mix mode
    if query_param.mode == "mix" and chunks_vdb:
        if progress_callback:
            await progress_callback(QueryProgress.RETRIEVING_CHUNKS)
        vector_chunks = await _get_vector_context(
            query,
            chunks_vdb,
            query_param,
            query_embedding,
        )
        # Track vector chunks with source metadata
        for i, chunk in enumerate(vector_chunks):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                chunk_tracking[chunk_id] = {
                    "source": "C",
                    "frequency": 1,  # Vector chunks always have frequency 1
                    "order": i + 1,  # 1-based order in vector search results
                }
            else:
                logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Round-robin merge entities
    final_entities = []
    seen_entities = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    # Round-robin merge relations
    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, {len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, str],
    query: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.

    Returns ``entities_context`` / ``relations_context`` (the records handed to
    the prompt) plus ``filtered_entities`` / ``filtered_relations`` (the matching
    original records handed to chunk selection).

    Ordering rule for any selector added here: the ``*_context`` lists are this
    stage's output and the only importance ranking downstream sees. The
    ``filtered_*`` lists MUST follow that same order -- stage 3 attributes a
    shared chunk to the earlier-positioned record and allocates chunk quota by
    list position -- so a selector that reorders must not leave the
    ``filtered_*`` lists in stage-1 retrieval order.

    A selector may reorder and drop, but must not invent: a ``*_context``
    record naming an entity or relation that was not in this stage's input
    cannot be resolved back to an original, so it reaches the prompt but is
    dropped from ``filtered_*`` with a warning.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data.
        # First occurrence wins: the filtered rebuild below resolves records
        # through this map, and duplicate names must keep the record that was
        # retrieved first. Accepted divergence if a name ever repeats: the
        # prompt shows the context row a reordering selector picked, while
        # filtered_* carries the first-retrieved original. Unreachable today --
        # stage 1 already dedups entities by name and relations by sorted pair.
        entity_id_to_original.setdefault(entity_name, entity)

        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data.
        # First occurrence wins, with the same rule and the same accepted
        # divergence as entities above.
        relation_key = (entity1, entity2)
        relation_id_to_original.setdefault(relation_key, relation)

        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = await atruncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: json.dumps(x, ensure_ascii=False),
            separator="\n",
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = await atruncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: json.dumps(x, ensure_ascii=False),
            separator="\n",
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    budget_exhausted = False
    if query is not None:
        fitted = await run_in_tokenizer_executor(
            _fit_kg_context_to_total_budget,
            entities_context,
            relations_context,
            [],
            query,
            query_param,
            global_config,
            system_prompt,
        )
        budget_exhausted = fitted is None
        entities_context, relations_context = fitted[:2] if fitted else ([], [])

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context.
    #
    # Walk the *_context lists (this stage's own output order) and resolve each
    # record through the pre-truncation maps. Stage 2's order is the single
    # source of truth for downstream importance: stage 3 deduplicates chunk
    # attribution by first occurrence and hands the list to
    # pick_by_weighted_polling, whose quota decreases with list position. Do NOT
    # rebuild these by filtering final_entities / final_relations -- that
    # reimposes stage-1 retrieval order and silently discards any reordering a
    # stage-2 selector (e.g. a reranker) performed.
    #
    # A context record that resolves to nothing was invented or renamed by a
    # stage-2 selector: it reaches the prompt but cannot reach chunk selection,
    # so it is dropped here and reported rather than lost silently.
    filtered_entities = []
    filtered_entity_id_to_original = {}
    unresolved_entities = []
    for entity_context in entities_context:
        name = entity_context.get("entity")
        if name in filtered_entity_id_to_original:
            continue
        original = entity_id_to_original.get(name)
        if original is None:
            unresolved_entities.append(name)
            continue
        filtered_entities.append(original)
        filtered_entity_id_to_original[name] = original

    if unresolved_entities:
        logger.warning(
            f"Dropping {len(unresolved_entities)} entity records absent from the "
            f"pre-truncation map: {unresolved_entities[:5]}"
        )

    filtered_relations = []
    filtered_relation_id_to_original = {}
    unresolved_relations = []
    for relation_context in relations_context:
        pair = (relation_context.get("entity1"), relation_context.get("entity2"))
        if pair in filtered_relation_id_to_original:
            continue
        original = relation_id_to_original.get(pair)
        if original is None:
            unresolved_relations.append(pair)
            continue
        filtered_relations.append(original)
        filtered_relation_id_to_original[pair] = original

    if unresolved_relations:
        logger.warning(
            f"Dropping {len(unresolved_relations)} relation records absent from the "
            f"pre-truncation map: {unresolved_relations[:5]}"
        )

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
        "token_budget_exceeded": budget_exhausted,
    }


def _fit_kg_context_to_total_budget(
    entities: list[dict],
    relations: list[dict],
    chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict,
    system_prompt: str | None,
) -> tuple[list[dict], list[dict], list[dict], str, list[dict]] | None:
    """Keep whole ranked records under the rendered system-prompt/query budget.

    Run in the tokenizer executor. Graph records are interleaved by rank, keeping
    each source's prefix order; chunks use the remaining capacity. Return None
    when the empty context and required prompt alone cannot fit. Conversation
    history and provider message framing are outside this query-context budget.
    """
    tokenizer = global_config["tokenizer"]
    limit = query_param.max_total_tokens
    template = system_prompt or PROMPTS["rag_response"]
    user_prompt = resolve_user_prompt(
        query_param.user_prompt,
        global_config.get("user_prompt_prefix", ""),
        query_param.disable_user_prompt_prefix,
    ).slot
    query_tokens = len(tokenizer.encode(query))

    def render(entity_records, relation_records, chunk_records):
        references, rendered_chunks = generate_reference_list_from_chunks(chunk_records)
        context = PROMPTS["kg_query_context"].format(
            entities_str="\n".join(
                json.dumps(record, ensure_ascii=False) for record in entity_records
            ),
            relations_str="\n".join(
                json.dumps(record, ensure_ascii=False) for record in relation_records
            ),
            text_chunks_str=render_chunks_context_text(rendered_chunks),
            reference_list_str="\n".join(
                f"[{ref['reference_id']}] {ref['file_path']}" for ref in references
            ),
        )
        prompt = template.format(
            context_data=context,
            response_type=query_param.response_type or "Multiple Paragraphs",
            user_prompt=user_prompt,
        )
        fits = len(tokenizer.encode(prompt)) + query_tokens <= limit
        return fits, (
            entity_records,
            relation_records,
            rendered_chunks,
            context,
            references,
        )

    if not render([], [], [])[0]:
        logger.warning("max_total_tokens cannot fit the required query prompt")
        return None

    ranked = [
        (kind, records[index])
        for index in range(max(len(entities), len(relations)))
        for kind, records in (("entity", entities), ("relation", relations))
        if index < len(records)
    ]

    def graph_prefix(count):
        records = ranked[:count]
        return (
            [record for kind, record in records if kind == "entity"],
            [record for kind, record in records if kind == "relation"],
        )

    # Search prefixes, then verify the exact survivor rendering. Tokenization
    # need not be strictly monotonic at record or template boundaries.
    def fit_prefix(count, candidate):
        low, high = 0, count
        while low < high:
            mid = (low + high + 1) // 2
            if candidate(mid)[0]:
                low = mid
            else:
                high = mid - 1
        fits, result = candidate(low)
        while not fits and low:
            low -= 1
            fits, result = candidate(low)
        return result

    if not render(entities, relations, [])[0]:
        all_entities, all_relations = entities, relations
        entities, relations, _, _, _ = fit_prefix(
            len(ranked), lambda count: render(*graph_prefix(count), [])
        )
        # An oversized row in one source must not block the other source's
        # smaller records. Extend each remaining source prefix independently.
        entities, relations, _, _, _ = fit_prefix(
            len(all_entities),
            lambda count: render(all_entities[:count], relations, []),
        )
        entities, relations, _, _, _ = fit_prefix(
            len(all_relations),
            lambda count: render(entities, all_relations[:count], []),
        )
    fits, result = render(entities, relations, chunks)
    if fits:
        return result
    return fit_prefix(
        len(chunks), lambda count: render(entities, relations, chunks[:count])
    )


async def _attach_content_headings(
    chunks: list[dict], text_chunks_db: BaseKVStorage | None
) -> None:
    """Backfill the ``content_headings`` field onto chunks in place.

    Vector chunks never carry a ``heading`` field (chunks_vdb does not store it),
    and the round-robin merge drops the entity/relation headings too. So we look
    the chunks up by ``chunk_id`` in text_chunks storage and attach the parent
    heading path. Only chunks that actually have parent headings get the field,
    so empty paths are omitted from the JSON sent to the LLM.

    Each level is char-capped inside ``format_parent_headings`` and the joined
    path is token-budgeted against ``DEFAULT_MAX_SECTION_CONTEXT_TOKENS`` here —
    mirroring the extraction breadcrumb (see ``_truncate_section_context``). The
    query-context token truncation only keeps/drops whole chunks; it never trims
    this metadata inside a retained chunk, so a deep heading chain (the per-level
    cap bounds each level's length, not the number of levels) is bounded here.
    """
    if not text_chunks_db or not chunks:
        return
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    chunk_ids = [c.get("chunk_id") for c in chunks]
    chunk_data_list = await text_chunks_db.get_by_ids(chunk_ids)

    def _backfill() -> None:
        for chunk, data in zip(chunks, chunk_data_list):
            if not isinstance(data, dict):
                continue
            headings = _truncate_section_context(
                format_parent_headings(data),
                tokenizer,
                DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
            )
            if headings:
                chunk["content_headings"] = headings

    # The whole loop is one submission, keeping submissions O(1) per request.
    # It has to leave the event loop at all because this tokenizer is the
    # long-lived one the storages captured at init, i.e. the SAME object every
    # concurrent query uses — and two of its other consumers
    # (_apply_token_truncation, process_chunks_unified) now run in the tokenizer
    # executor. Left here, two concurrent queries would contend: one holding the
    # tokenizer in a worker thread while the other waits for it on the loop.
    await run_in_tokenizer_executor(_backfill)


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage = None,
    text_chunks_db: BaseKVStorage = None,
    query_param: QueryParam = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # Get chunks from entities
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Get chunks from relations
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Round-robin merge chunks from different sources with deduplication
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from entity chunks (Local mode)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (deduplicated {origin_len - len(merged_chunks)})"
    )

    # Backfill heading path before token truncation so it counts toward the budget
    if text_chunks_db and text_chunks_db.global_config.get(
        "enable_content_headings", False
    ):
        await _attach_content_headings(merged_chunks, text_chunks_db)

    return merged_chunks


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunk_tracking: dict = None,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
    progress_callback: ProgressCallback | None = None,
    system_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Budget against the template that will ACTUALLY be rendered. `kg_query`
    # picks its template after this function returns, so without the forwarded
    # `system_prompt` the estimate silently used the default one -- charging a
    # caller's custom template at the wrong size, and charging the prefix even
    # when that template has no {user_prompt} placeholder to render it into.
    # `system_prompt_template` is kept as a lower-priority fallback: nothing in
    # this repo writes it, but a downstream user may set it on their own config.
    sys_prompt_template = (
        system_prompt
        or global_config.get("system_prompt_template")
        or PROMPTS["rag_response"]
    )

    def budget_failure():
        raw_data = convert_to_user_format([], [], [], [], query_param.mode)
        raw_data["status"] = "failure"
        raw_data["message"] = "max_total_tokens cannot fit the query context."
        raw_data.setdefault("metadata", {})["token_budget_exceeded"] = True
        return "", raw_data

    had_graph_context = bool(entities_context or relations_context)
    fitted = await run_in_tokenizer_executor(
        _fit_kg_context_to_total_budget,
        entities_context,
        relations_context,
        [],
        query,
        query_param,
        global_config,
        system_prompt,
    )
    if fitted is None:
        return budget_failure()
    entities_context, relations_context = fitted[:2]
    if (
        had_graph_context
        and not entities_context
        and not relations_context
        and not merged_chunks
    ):
        return budget_failure()

    kg_context_template = PROMPTS["kg_query_context"]
    # This preliminary estimate retains the legacy empty-prefix behavior;
    # the final rendered-budget check below accounts for the actual slot.
    effective_user_prompt = resolve_user_prompt(
        query_param.user_prompt,
        global_config.get("user_prompt_prefix", ""),
        query_param.disable_user_prompt_prefix,
    )
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = await acount_tokens(tokenizer, pre_kg_context)

    # Calculate preliminary system prompt tokens
    # Charged even on the only_need_context path, which returns before this
    # prompt is ever rendered. That is deliberate, not waste: only_need_context
    # and only_need_prompt are debug switches whose job is to PREVIEW the real
    # request. Skipping the charge here would make them report more chunks than
    # a real query retrieves, so an operator would size their context against a
    # number the live path never delivers.
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=effective_user_prompt.text,
    )
    sys_prompt_tokens = await acount_tokens(tokenizer, pre_sys_prompt)

    # Calculate available tokens for text chunks
    query_tokens = await acount_tokens(tokenizer, query)
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )
    if (
        not entities_context
        and not relations_context
        and merged_chunks
        and available_chunk_tokens <= 0
    ):
        logger.warning(
            f"Token budget left no room for any of {len(merged_chunks)} retrieved "
            f"chunks (max_total_tokens: {max_total_tokens}, available for chunks: "
            f"{available_chunk_tokens}); the query returns no result. "
            "Raise max_total_tokens."
        )
        return budget_failure()

    logger.debug(
        f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Apply token truncation to chunks using the dynamic limit
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
        source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
        progress_callback=progress_callback,
    )

    # The reference paths and repeated template slots are part of the total
    # budget too. Re-render after dropping chunks so their IDs stay consistent.
    fitted = await run_in_tokenizer_executor(
        _fit_kg_context_to_total_budget,
        entities_context,
        relations_context,
        truncated_chunks,
        query,
        query_param,
        global_config,
        system_prompt,
    )
    if fitted is None:
        return budget_failure()
    entities_context, relations_context, retained_chunks, result, reference_list = (
        fitted
    )
    if truncated_chunks and not retained_chunks:
        outcome = (
            "the answer uses graph context only"
            if entities_context or relations_context
            else "the query returns no result"
        )
        logger.warning(
            f"Token budget left no room for any of {len(truncated_chunks)} "
            f"retrieved chunks once their reference list was rendered "
            f"(max_total_tokens: {max_total_tokens}); {outcome}. "
            "Raise max_total_tokens or lower max_entity_tokens / max_relation_tokens."
        )
    if (
        truncated_chunks
        and not entities_context
        and not relations_context
        and not retained_chunks
    ):
        return budget_failure()
    truncated_chunks = retained_chunks

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not truncated_chunks:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
    )
    final_data_payload = final_data.get("data", {})
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data_payload.get('entities', []))} entities, {len(final_data_payload.get('relationships', []))} relationships, {len(final_data_payload.get('chunks', []))} chunks"
    )
    return result, final_data


# Now let's update the old _build_query_context to use the new architecture
async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
    progress_callback: ProgressCallback | None = None,
    system_prompt: str | None = None,
    user_prompt_prefix: str | None = None,
) -> QueryContextResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Returns unified QueryContextResult containing both context and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Stage 1: Pure search
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
        progress_callback=progress_callback,
    )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["chunk_tracking"]:
                return None

    # Prompt rendering uses the live query config, while chunk selection must
    # keep the storage snapshot. Override only the prefix for token budgeting.
    budget_config = text_chunks_db.global_config
    if user_prompt_prefix is not None:
        budget_config = {**budget_config, "user_prompt_prefix": user_prompt_prefix}

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        budget_config,
        query=query,
        system_prompt=system_prompt,
    )

    if truncation_result.get("token_budget_exceeded"):
        return None

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
    )

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=budget_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
        progress_callback=progress_callback,
        system_prompt=system_prompt,
    )

    # A budget failure also returns an empty context, so this covers both.
    if not context:
        return None

    # Add processing info to raw_data (preserve existing metadata including
    # query_mode). kg_query fills in metadata["keywords"]: it holds the lists.
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding=None,
):
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    results = await entities_vdb.query(
        query, top_k=query_param.top_k, query_embedding=query_embedding
    )

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entities, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


def _vector_chunk_quota(max_related_chunks: int, group_count: int) -> int:
    """How many chunks VECTOR-mode selection may draw, scaled by group count.

    Do not remove the floor of 1: it restores parity with the WEIGHT path,
    which guarantees at least one chunk per group via
    ``pick_by_weighted_polling(..., min_related_chunks=1)``. This quota is the
    same linear-gradient budget as WEIGHT's ``n * (max + 1) / 2`` minus the
    ``n / 2`` term, so ``(max_related_chunks=1, group_count=1)`` is the single
    input where VECTOR would otherwise truncate to 0 and drop below that floor.
    A 0 quota makes ``pick_by_vector_similarity`` return ``[]``, which the call
    sites read as "vector selection failed" and silently downgrade to WEIGHT.

    The ``max_related_chunks <= 0`` guard is what keeps ``related_chunk_number=0``
    a genuine kill switch rather than a silent 1; the floor applies only once
    both inputs are positive.

    The ``group_count <= 0`` branch is defensive only: both call sites already
    guard on their group list being non-empty (entities_with_chunks /
    relations_with_chunks) before reaching here, so group_count is always
    >= 1 in practice.

    Args:
        max_related_chunks: Configured ``related_chunk_number``; 0 disables
            KG-related chunk selection entirely.
        group_count: Number of entity/relation groups that carry chunks.

    Returns:
        Number of chunks VECTOR selection may draw, 0 when disabled.
    """
    if max_related_chunks <= 0 or group_count <= 0:
        return 0
    return max(1, int(max_related_chunks * group_count / 2))


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Drop entities emptied by deduplication so they do not inflate the
    # per-group chunk budget used by the selection strategies.
    entities_with_chunks = [
        entity_info for entity_info in entities_with_chunks if entity_info["chunks"]
    ]

    # Defensive only: unreachable on this path. Deduplication drops a chunk
    # solely because an earlier-positioned entity already claimed it, so the
    # first entity keeps every chunk it carries and at least one group always
    # survives. The relation path's equivalent guard IS reachable, because it
    # additionally excludes the chunks already delivered by the entity path,
    # which can empty every relation group.
    if not entities_with_chunks:
        logger.info(
            f"Find no entity-related chunks from {len(node_datas)} entities after deduplication"
        )
        return []

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and not (query and chunks_vdb):
        # Without this switch neither branch below runs and every chunk is lost.
        logger.warning(
            "Vector similarity chunk selection needs both a query and chunks_vdb, "
            "falling back to WEIGHT method"
        )
        kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "VECTOR":
        num_of_chunks = _vector_chunk_quota(
            max_related_chunks, len(entities_with_chunks)
        )

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding=None,
):
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, query_embedding=query_embedding
    )

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entities, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and not (query and chunks_vdb):
        # Without this switch neither branch below runs and every chunk is lost.
        logger.warning(
            "Vector similarity chunk selection needs both a query and chunks_vdb, "
            "falling back to WEIGHT method"
        )
        kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "VECTOR":
        num_of_chunks = _vector_chunk_quota(
            max_related_chunks, len(relations_with_chunks)
        )

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entities, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    text_chunks_db: BaseKVStorage | None = None,
    return_raw_data: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    text_chunks_db: BaseKVStorage | None = None,
    return_raw_data: Literal[False] = False,
) -> str | AsyncIterator[str]: ...


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    text_chunks_db: BaseKVStorage | None = None,
    progress_callback: ProgressCallback | None = None,
) -> QueryResult | None:
    """
    Execute naive query and return unified QueryResult object.

    Args:
        query: Query string
        chunks_vdb: Document chunks vector database
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Returns None when no relevant chunks are retrieved.
    """

    if not query:
        return QueryResult(content=PROMPTS["fail_response"], llm_generated=False)

    # Apply higher priority (5) to query relation LLM function
    use_model_func = partial(
        global_config["role_llm_funcs"]["query"], _priority=DEFAULT_QUERY_PRIORITY
    )
    llm_cache_identity = get_llm_cache_identity(global_config, "query")

    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"], llm_generated=False)

    if progress_callback:
        await progress_callback(QueryProgress.RETRIEVING_CHUNKS)
    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        logger.info(
            "[naive_query] No relevant document chunks found; returning no-result."
        )
        return None

    # Backfill heading path before token truncation so it counts toward the budget
    if global_config.get("enable_content_headings", False):
        await _attach_content_headings(chunks, text_chunks_db)

    # Calculate dynamic token limit for chunks
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate system prompt template tokens (excluding content_data)
    effective_user_prompt = resolve_user_prompt(
        query_param.user_prompt,
        global_config.get("user_prompt_prefix", ""),
        query_param.disable_user_prompt_prefix,
    )
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a preliminary system prompt with empty content_data to calculate overhead.
    # As in _build_context_str, the user prompt is charged even when
    # only_need_context will return before this prompt is sent: those switches
    # preview the real request, so their chunk count must match it.
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=effective_user_prompt.slot,
        content_data="",  # Empty for overhead calculation
    )

    # Calculate available tokens for chunks
    sys_prompt_tokens = await acount_tokens(tokenizer, pre_sys_prompt)
    query_tokens = await acount_tokens(tokenizer, query)
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
        progress_callback=progress_callback,
    )

    if not processed_chunks:
        logger.info(
            "[naive_query] No chunks survived final filtering; returning no-result."
        )
        return None

    # Generate reference list from processed chunks using the new common function
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )

    logger.info(f"Final context: {len(processed_chunks_with_ref_ids)} chunks")

    # Build raw data structure for naive mode using processed chunks with reference IDs
    raw_data = convert_to_user_format(
        [],  # naive mode has no entities
        [],  # naive mode has no relationships
        processed_chunks_with_ref_ids,
        reference_list,
        "naive",
    )

    # Add complete metadata for naive mode
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": [],  # naive mode has no keyword extraction
        "low_level": [],  # naive mode has no keyword extraction
    }
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }

    # Build chunks_context from processed chunks with reference IDs
    text_units_str = render_chunks_context_text(processed_chunks_with_ref_ids)
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    naive_context_template = PROMPTS["naive_query_context"]
    context_content = naive_context_template.format(
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_content, raw_data=raw_data, llm_generated=False
        )

    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=effective_user_prompt.slot,
        content_data=context_content,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(
            content=prompt_content, raw_data=raw_data, llm_generated=False
        )

    # Handle cache
    answer_cache_kv = _answer_cache_kv(query_param, hashing_kv)
    args_hash = compute_args_hash(
        _ANSWER_CACHE_POLICY_VERSION,
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        # The COMPOSED instructions, so changing the server-side prefix
        # invalidates entries generated under the old one. With no prefix
        # configured this is byte-identical to the previous
        # `query_param.user_prompt or ""`, so existing entries keep hitting --
        # which is why _ANSWER_CACHE_POLICY_VERSION does not need a bump.
        # `disable_user_prompt_prefix` is deliberately NOT a separate key
        # component: it only ever acts through this value, and adding it would
        # split the cache between two requests that build identical prompts.
        effective_user_prompt.text,
        query_param.enable_rerank,
        global_config.get("enable_content_headings", False),
        *(("\n<system_prompt>\n", system_prompt) if system_prompt else ()),
        "\n<llm_identity>\n",
        serialize_llm_cache_identity(llm_cache_identity),
    )
    cached_result = _drop_reasoning_only_cache_hit(
        await handle_cache(
            answer_cache_kv, args_hash, user_query, query_param.mode, cache_type="query"
        )
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )
        # A reasoning trace with no answer is a failed generation, not an
        # answer: never return it as one, never cache it.
        reject_reasoning_only_answer(response)

        if (
            answer_cache_kv
            and answer_cache_kv.global_config.get("enable_llm_cache")
            and not is_truncated_response(response)
        ):
            queryparam_dict = {
                "answer_cache_version": _ANSWER_CACHE_POLICY_VERSION,
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
                "enable_content_headings": global_config.get(
                    "enable_content_headings", False
                ),
            }
            await save_to_cache(
                answer_cache_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        return QueryResult(content=response, raw_data=raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response, raw_data=raw_data, is_streaming=True
        )
