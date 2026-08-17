"""Pipeline-specific helpers for document status, identity, and content.

These helpers are shared by the LightRAG pipeline mixin (lightrag/pipeline.py)
and by other LightRAG methods that touch the document ingestion paths
(custom-chunks ingest, deletion, etc.). They are kept out of utils.py because
they are tied to the doc_status / full_docs domain rather than to general
text/token utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, urlsplit

from lightrag.base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    SourceAbsent,
    SourceResolution,
)
from lightrag.constants import (
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    DUPLICATE_DEMOTION_METADATA_KEYS,
    FILE_EXTRACTION_SUMMARY_PREFIX,
    FULL_DOCS_FORMAT_LIGHTRAG,
    KG_PURGE_METADATA_KEY,
    KG_WRITE_STATE_METADATA_KEY,
    LIGHTRAG_DOC_CONTENT_PREFIX,
    PARSED_DIR_NAME,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_NATIVE,
)
from lightrag import pipeline_metrics
from lightrag.exceptions import StorageCapabilityError, StorageRecordNotFoundError
from lightrag.parser.routing import (
    canonicalize_parser_hinted_basename,
    default_chunker_config,
)
from lightrag.utils import (
    compute_mdhash_id,
    get_content_summary,
    LLM_TRUNCATION_METADATA_KEY,
    logger,
    move_file_to_parsed_dir,
)


PLACEHOLDER_DOCUMENT_SOURCES = {"", "no-file-path", "unknown_source"}
SIDECAR_LOCATION_UNKNOWN = "unknown_source"


def apply_trusted_sentence_split_regex(
    v_opts: dict[str, Any],
    addon_params: Any,
    *,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Force ``sentence_split_regex`` to the live, operator-controlled value.

    ``v_opts`` comes from the per-doc ``chunk_options`` snapshot persisted in
    ``full_docs``. Every other chunker parameter deliberately wins from that
    snapshot so a document re-processes reproducibly across env changes; this
    one key is the exception, and is re-read live from
    ``addon_params['chunker']['semantic_vector']`` (seeded by
    ``CHUNK_V_SENTENCE_SPLIT_REGEX``) on every run.

    The reason is that the value is splatted into ``re.split`` against the
    document body. A backtracking pattern such as ``(a+)+$`` runs for
    exponential time, and CPython's regex engine holds the GIL throughout, so
    the semantic-vector chunker's ``asyncio.to_thread`` hop does not keep the
    event loop alive — the worker process stops serving every endpoint until
    it is restarted (GHSA-32jh-39m7-8x84, CWE-1333).

    ``chunk_options`` is snapshotted at ENQUEUE time, before chunking runs, so
    a build that still accepted the field on ``/documents/text`` persisted the
    attacker's pattern to storage *and* left the document in ``PROCESSING``
    when the worker froze. ``PROCESSING`` is an auto-resume status, so without
    this scrub an upgraded server would reload the stored pattern and freeze
    again on every restart — a boot loop that restarting cannot clear.
    Rejecting the field at the request model only protects new requests; this
    is what disarms snapshots already on disk.

    The replacement is unconditional, so provenance never has to be
    reconstructed — which also means a *legitimate* per-document pattern
    handed to ``apipeline_enqueue_documents(chunk_options=…)`` by an SDK caller
    is discarded the same way. That is intended: the only supported way to
    change the splitter is ``addon_params['chunker']['semantic_vector']`` /
    ``CHUNK_V_SENTENCE_SPLIT_REGEX``. Because the loss is silent from the
    caller's side, every discard of a *differing* snapshot value is logged at
    WARNING with the doc id.

    This is the single point where the guarantee is enforced: the ``"V"``
    dispatch in :meth:`_PipelineMixin.process_single_document` is currently the
    only consumer of ``chunk_options['semantic_vector']``. Any new consumer
    MUST route through this helper — the poisoned value is left in place in
    ``full_docs`` (inert, not rewritten), so an unscrubbed read would revive it.

    Returns a new dict; ``v_opts`` is not mutated.
    """
    sanitized = {k: v for k, v in v_opts.items() if k != "sentence_split_regex"}
    chunker_cfg = (addon_params or {}).get("chunker") or default_chunker_config()
    live_regex = (chunker_cfg.get("semantic_vector") or {}).get("sentence_split_regex")
    if live_regex:
        sanitized["sentence_split_regex"] = live_regex

    snapshot_regex = v_opts.get("sentence_split_regex")
    if snapshot_regex is not None and snapshot_regex != live_regex:
        # Audit line for a decision with consequences: either a pre-fix build
        # persisted an attacker pattern (the case this helper exists for), or
        # an SDK caller's per-doc value is being dropped. The pattern is
        # untrusted text, so log a bounded repr rather than the raw string.
        shown = repr(snapshot_regex)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        logger.warning(
            "Discarded persisted sentence_split_regex %s for doc %s; using the "
            "operator-configured pattern instead (GHSA-32jh-39m7-8x84). Set "
            "CHUNK_V_SENTENCE_SPLIT_REGEX or addon_params['chunker']"
            "['semantic_vector'] to change the V sentence splitter.",
            shown,
            doc_id or "<unknown>",
        )
    return sanitized


def build_chunks_dict_from_chunking_result(
    chunking_result: list[dict[str, Any]],
    *,
    doc_id: str,
    file_path: str,
) -> dict[str, dict[str, Any]]:
    """Assemble the per-doc chunks dict written into chunks_vdb / text_chunks.

    Resolves a stable ``chunk_key`` for each entry — preferring an explicit
    ``chunk_id`` (auto-prefixed with ``doc_id-`` if not already), falling back
    to a positional ``chunk-NNN`` derived from ``chunk_order_index``, and
    finally hashing on collision so two entries inside one document never
    overwrite each other.
    """
    chunks: dict[str, dict[str, Any]] = {}
    for dp in chunking_result:
        chunk_content = dp.get("content", "")
        if not chunk_content:
            continue
        raw_chunk_id = dp.get("chunk_id", "")
        order = dp.get("chunk_order_index")
        if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
            chunk_key = (
                raw_chunk_id
                if raw_chunk_id.startswith(f"{doc_id}-")
                else f"{doc_id}-{raw_chunk_id}"
            )
        elif isinstance(order, int):
            chunk_key = f"{doc_id}-chunk-{order:03d}"
        else:
            chunk_key = compute_mdhash_id(f"{doc_id}:{chunk_content}", prefix="chunk-")

        # Hard collision guard (same chunk_id inside one document).
        if chunk_key in chunks:
            chunk_key = compute_mdhash_id(
                f"{doc_id}:{order}:{chunk_content}",
                prefix="chunk-",
            )
        # Preserve any pre-populated cache ids on dp (multimodal chunks
        # arrive with analysis cache ids already attached so document
        # deletion can find them via the per-chunk llm_cache_list).
        existing_cache_list = dp.get("llm_cache_list")
        seed_cache_list: list[str] = []
        if isinstance(existing_cache_list, list):
            seen: set[str] = set()
            for entry in existing_cache_list:
                key = str(entry or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    seed_cache_list.append(key)
        stored_chunk = {k: v for k, v in dp.items() if k != "_source_span"}
        chunks[chunk_key] = {
            **stored_chunk,
            "full_doc_id": doc_id,
            "file_path": file_path,
            "llm_cache_list": seed_cache_list,
        }
    return chunks


def chunk_fields_from_status_doc(
    status_doc: DocProcessingStatus,
) -> tuple[list[str], int]:
    """Return (chunks_list, chunks_count) preserved from a status document.

    Filters out any non-string or empty chunk IDs.  When chunks_count is
    absent or invalid, it is inferred from the length of chunks_list.
    """
    chunks_list: list[str] = []
    if isinstance(status_doc.chunks_list, list):
        chunks_list = [
            chunk_id
            for chunk_id in status_doc.chunks_list
            if isinstance(chunk_id, str) and chunk_id
        ]

    if isinstance(status_doc.chunks_count, int) and status_doc.chunks_count >= 0:
        return chunks_list, status_doc.chunks_count

    return chunks_list, len(chunks_list)


def resolve_doc_file_path(
    status_doc: DocProcessingStatus | None = None,
    content_data: dict[str, Any] | None = None,
) -> str:
    """Resolve the best available document file path.

    Returns the first non-placeholder ``file_path`` from doc_status, then
    full_docs. Both are already canonicalized at write time, so this only
    has to skip placeholder sentinels.
    """
    for source in (
        getattr(status_doc, "file_path", None),
        content_data.get("file_path") if content_data else None,
    ):
        if not isinstance(source, str):
            continue
        candidate = source.strip()
        if candidate and candidate not in PLACEHOLDER_DOCUMENT_SOURCES:
            return candidate
    return "unknown_source"


def normalize_document_file_path(file_path: Any) -> str:
    """Return the canonical basename stored as ``file_path``.

    Strips any supported ``[hint]`` segment so ``abc.docx`` and
    ``abc.[native-iet].docx`` map to the same key. Collapses placeholders to
    ``"unknown_source"``. Idempotent.
    """
    source = str(file_path or "").strip()
    if source in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    canonical = canonicalize_parser_hinted_basename(source).strip()
    if canonical in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    return canonical or "unknown_source"


# Back-compat alias retained until call sites that import the old name are
# all switched over (the public surface is ``normalize_document_file_path``).
document_canonical_key = normalize_document_file_path


def has_known_document_source(source_key: str) -> bool:
    return source_key not in PLACEHOLDER_DOCUMENT_SOURCES


def doc_status_field(doc: Any, field: str, default: Any = "") -> Any:
    if isinstance(doc, dict):
        return doc.get(field, default)
    return getattr(doc, field, default)


def read_source_file_basename(data: Any) -> str | None:
    """Read the source-file basename with backward compatibility.

    The ``source_file_name`` → ``source_file`` rename means documents enqueued
    before the change persisted the old key in full_docs content_data /
    doc_status.metadata.  Read through here so resumed/legacy docs still resolve
    their original source basename; write sites always use the new
    ``source_file`` key.  Lives here (not in pipeline.py) so utils_pipeline
    helpers can reuse it without importing back into the pipeline module.
    """
    if not isinstance(data, dict):
        return None
    return data.get("source_file") or data.get("source_file_name")


# Long-lived per-document metadata fields that must survive every
# doc_status state transition.  ``process_options`` records the user's
# per-file processing strategy at enqueue time and is read by analyze /
# chunk / KG-skip stages and by admin/list APIs throughout the document's
# lifetime, so we cannot let an intermediate transition (PARSING /
# ANALYZING / PROCESSING / PROCESSED / FAILED upsert) clobber it.
# ``parse_warnings`` records non-fatal parser warnings (e.g. legacy docx
# tables missing ``w14:paraId``) that admins should be able to surface
# alongside the document record after PROCESSED.
# ``chunk_opts`` is written when entering PROCESSING (via ``extraction_meta``)
# and records the actual chunker params used for that document in the same
# format as the ``Chunking <strategy>: ...`` log line (params portion only).
# Carrying it forward keeps the value visible after PROCESSING -> FAILED,
# whose ``metadata_extra`` only carries timing fields.
# ``parse_start_time`` / ``analyzing_start_time`` are Unix epoch seconds
# stamped at the entry of ``_parse_worker`` / ``_analyze_worker`` (mirrors
# the existing ``process_start_time`` set when entering PROCESSING) so
# per-stage durations can be derived from doc_status post-mortem.
# ``parse_end_time`` is the paired Unix epoch seconds stamped by
# ``_parse_worker`` when the parse stage actually runs (cache-miss branch,
# covering ``parse_native`` too which has no cache concept). Absent on
# cache-hit attempts (``parse_stage_skipped`` is set instead).
# ``analyzing_end_time`` is the paired Unix epoch seconds stamped by
# ``_analyze_worker`` only when ``analyze_multimodal`` returns with
# ``multimodal_processed=True`` (the explicit "fully completed" sentinel).
# It is intentionally NOT stamped on soft-swallowed exception paths or on
# malformed/empty sidecar early returns inside ``analyze_multimodal``, so
# operators can distinguish "analyze actually completed" from "analyze
# attempted but bailed".
# ``parse_stage_skipped`` is written by ``parse_mineru`` / ``parse_docling``
# when the raw bundle cache is valid and the parse stage round trip is
# skipped; absence == not skipped (e.g. native parser, or cache miss).
# ``analyzing_stage_skipped`` is its analyze-stage counterpart, written by
# ``analyze_multimodal``'s three user/config early-return branches (no
# blocks_path, blocks file missing, or user opted out of every i/t/e
# modality). Soft-swallowed exception paths are intentionally NOT considered
# "skipped" — they write neither end_time nor skipped (failure is its own
# state, captured via the FAILED transition's ``error_msg``).
# Within each stage, the ``*_end_time`` and ``*_stage_skipped`` fields are
# mutually exclusive (at most one is written per attempt; both may be
# absent if analyze soft-failed).
# ``source_file`` records the original pending-parse source basename used
# by parser workers; it is intentionally separate from canonical ``file_path``.
#
# The order of this tuple is the rendering order of metadata fields in
# the WebUI ``DocumentStatusDetailsDialog`` (carry-over builds the new
# metadata dict by iterating this tuple, and dict / JSON / JSX preserve
# insertion order all the way to the rendered output). Keep fields
# grouped by stage: parse-stage fields together, analyze-stage fields
# together, etc., so the dialog reads top-to-bottom along the pipeline.
def resolve_doc_status_parse_engine(
    parse_format: str | None,
    explicit_engine: str | None,
) -> str:
    """Resolve the ``parse_engine`` value recorded in doc_status metadata.

    Single source of truth shared by the parse stage (``_parse_worker``) and
    the process stage (``process_single_document``) so both stamp the *same*
    value — preventing a value "jump" when the field is written early at
    PARSING and re-written later at PROCESSING.  ``explicit_engine`` wins when
    a parser reported the engine it actually ran (or full_docs carries an
    enqueue-time directive); otherwise fall back to ``native`` for the
    structured ``lightrag`` format and ``legacy`` for everything else
    (``raw`` passthrough), mirroring how a pre-engine corpus was processed.
    """
    if explicit_engine:
        return explicit_engine
    return (
        PARSER_ENGINE_NATIVE
        if parse_format == FULL_DOCS_FORMAT_LIGHTRAG
        else PARSER_ENGINE_LEGACY
    )


# Public, persistent audit trail for structurally successful but semantically
# degraded KG recovery. The WebUI uses this key to distinguish a processed
# document with recovery warnings from an ordinary informational metadata row.
KG_RECOVERY_WARNINGS_METADATA_KEY = "kg_recovery_warnings"


_DOC_STATUS_METADATA_CARRY_OVER_KEYS: tuple[str, ...] = (
    "process_options",
    "source_file",
    "parse_warnings",
    "chunk_opts",
    "parse_start_time",
    "parse_end_time",
    "parse_stage_skipped",
    "parse_format",
    "parse_engine",
    # Parse-stage LLM cache keys (docx smart_heading); must survive
    # PARSING → ANALYZING → PROCESSING → PROCESSED/FAILED so document
    # deletion can purge them (delete_llm_cache=True).
    "smartheading_llm_cache_ids",
    "analyzing_start_time",
    "analyzing_end_time",
    "analyzing_stage_skipped",
    # Custom-chunk patch journal: the durable recovery anchor for an
    # in-flight/failed ainsert_custom_chunks operation. Must survive every
    # status transition until the operation commits or is rolled back.
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    # KG write-progress marker and whole-document purge journal (issue #3400
    # fail-closed purge). Both are load-bearing recovery state, not display
    # fields: ``kg_write_state`` is the proof that lets a purge clean up an
    # anchor-less document that never reached the graph, and ``kg_purge`` is
    # what distinguishes "anchors already deleted by a purge that got that far"
    # from "anchors were never there". Dropping either on a transition turns a
    # resumable purge into a permanent fail-closed refusal.
    KG_WRITE_STATE_METADATA_KEY,
    KG_PURGE_METADATA_KEY,
    # Duplicate demotion. ``metadata.is_duplicate`` is not a display field: it is
    # the ONLY thing that makes a row ineligible as a primary candidate for its
    # canonical source (``_basename_of`` returns None for it), so an operator's
    # source-conflict repair lives or dies with it. Dropping it on a transition
    # silently re-promoted the demoted row and put the key back in conflict —
    # and since a repair deliberately keeps the row's content and status, that
    # row is a normal document the pipeline will happily transition.
    *DUPLICATE_DEMOTION_METADATA_KEYS,
)


def make_custom_chunk_id(doc_key: str, chunk_text: str) -> str:
    """Deterministic, document-scoped custom-chunk id (issue #3400 Phase 3).

    The components are length-prefixed so the ``(doc, text)`` pair is
    unambiguous: plain concatenation would give ``doc_id="a", chunk="bc"``
    and ``doc_id="ab", chunk="c"`` the same hash input, letting two different
    documents share one ``text_chunks``/``chunks_vdb`` row — the later upsert
    would clobber ``full_doc_id`` and rollback/deletion of either document
    could remove the other's chunk.
    """
    return compute_mdhash_id(f"{len(doc_key)}:{doc_key}:{chunk_text}", prefix="chunk-")


def make_custom_chunk_operation_id(doc_key: str, chunk_ids: list[str]) -> str:
    """Deterministic operation id for one custom-chunk invocation.

    Hashes the document key plus the ordered chunk-id set (length-prefixed
    like :func:`make_custom_chunk_id`, so an arbitrary caller-supplied
    ``doc_id`` cannot collide with the chunk-id list of another document).
    The same logical input therefore maps to the same operation across
    retries.
    """
    joined = "|".join(chunk_ids)
    return compute_mdhash_id(f"{len(doc_key)}:{doc_key}:{joined}", prefix="op-")


def doc_status_custom_chunk_patch(status_doc: Any) -> dict[str, Any] | None:
    """Return the custom-chunk patch journal from a doc-status record, if any.

    Accepts either a ``DocProcessingStatus`` object or a raw storage dict.
    Returns ``None`` when the document carries no journal (the normal case).
    """
    if status_doc is None:
        return None
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return None
    journal = raw_metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY)
    return journal if isinstance(journal, dict) else None


def make_kg_purge_operation_id(doc_key: str, chunk_ids: list[str]) -> str:
    """Deterministic operation id for one whole-document purge (issue #3400).

    Identifies "the same logical purge" across retries so a resumed run can
    trust the journaled phase. Derived from the document key plus its chunk-id
    SET (sorted and de-duplicated, unlike
    :func:`make_custom_chunk_operation_id` which pins an ordered list): a purge
    assembles ``chunk_ids`` from ``chunks_list`` unioned with a custom-chunk
    journal, so the order is an assembly artifact while the set is the thing
    that determines what gets deleted. Length-prefixed like
    :func:`make_custom_chunk_id` so a caller-supplied ``doc_id`` cannot collide
    with another document's chunk-id list.
    """
    joined = "|".join(sorted(set(chunk_ids)))
    return compute_mdhash_id(f"{len(doc_key)}:{doc_key}:{joined}", prefix="purge-")


def doc_status_kg_write_state(status_doc: Any) -> str | None:
    """Return how far this document's KG write progressed, if recorded.

    ``None`` means UNKNOWN — a pre-#3416 document. The marker is monotonic
    and never cleared: a PROCESSED document keeps ``graph_mutation_started``
    (its anchors serve as the proof from then on). A fail-closed purge
    treats UNKNOWN as "may have touched the graph".
    """
    if status_doc is None:
        return None
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return None
    state = raw_metadata.get(KG_WRITE_STATE_METADATA_KEY)
    return state if isinstance(state, str) and state else None


def doc_status_kg_purge_journal(status_doc: Any) -> dict[str, Any] | None:
    """Return the whole-document purge journal from a doc-status record, if any.

    Accepts either a ``DocProcessingStatus`` object or a raw storage dict.
    Returns ``None`` when no purge is in flight (the normal case).
    """
    if status_doc is None:
        return None
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return None
    journal = raw_metadata.get(KG_PURGE_METADATA_KEY)
    return journal if isinstance(journal, dict) else None


async def require_doc_status_record(
    doc_status: Any, doc_id: str, *, purpose: str
) -> dict[str, Any]:
    """Point-read a doc_status row that a load-bearing metadata write needs.

    For writers of KG recovery state (``kg_write_state``, ``kg_purge``): they
    read the row only to rebuild the opaque ``metadata`` blob around the key
    they change, and every one of them runs while the row is guaranteed to
    exist (mid-merge under the pipeline reservation, mid-purge before the
    caller's finalization). ``None`` is therefore never a state to proceed
    past — it is the read failing to prove the state the write depends on.

    Strict where the backend supports it, so a transport/index failure raises
    with its real cause instead of masquerading as an absent row (a plain
    ``get_by_id`` may treat backend trouble as a best-effort miss). Whatever
    the read path, ``None`` raises ``StorageRecordNotFoundError``: proceeding
    silently is how the marker stays ``pre_graph`` while the graph is written,
    or how a destructive purge runs unjournaled — both of which manufacture
    exactly the unprovable states issue #3400's fail-closed contract exists to
    prevent. Callers abort instead: an aborted merge (anchors already durable)
    or an aborted purge (nothing deleted yet) is always the safe direction.
    """
    strict = getattr(doc_status, "supports_strict_point_reads", False)
    record = await (
        doc_status.get_by_id_strict(doc_id) if strict else doc_status.get_by_id(doc_id)
    )
    if record is None:
        raise StorageRecordNotFoundError(
            f"doc_status record for {doc_id} is "
            f"{'confirmed absent' if strict else 'absent or unreadable'} "
            f"while trying to {purpose}; refusing to proceed without it"
        )
    return record


def doc_status_metadata_carry_over(status_doc: Any) -> dict[str, Any]:
    """Return the subset of ``status_doc.metadata`` to preserve across upserts.

    ``doc_status`` storage backends generally treat the ``metadata`` field
    as an opaque blob and **replace** it on every upsert, so callers must
    explicitly carry forward fields they want to keep.  This helper centralises
    the list of fields we always carry: today only ``process_options``, but
    new long-lived metadata can be added by extending
    ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``.
    """
    if status_doc is None:
        return {}
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return {}
    carry: dict[str, Any] = {}
    for key in _DOC_STATUS_METADATA_CARRY_OVER_KEYS:
        if key in raw_metadata and raw_metadata[key] not in (None, ""):
            carry[key] = raw_metadata[key]
    return carry


def doc_status_transition_metadata(
    status_doc: Any,
    *,
    extra: dict[str, Any] | None = None,
    drop: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a doc_status ``metadata`` payload that preserves carry-over fields.

    Use at every state-transition upsert site so the user's
    ``process_options`` (and any future long-lived metadata fields) survive
    PENDING → PARSING → ANALYZING → PROCESSING → PROCESSED / FAILED.

    ``drop`` removes keys AFTER carry-over and ``extra`` are applied, which is
    the only way to retire a carried-over key: passing it in ``extra`` with a
    ``None``/empty value would persist that value rather than remove the key,
    and omitting it just lets carry-over put it back. Used to clear the
    ``kg_write_state`` / ``kg_purge`` recovery markers at PROCESSED, where the
    document's recovery anchors take over as the proof.
    """
    payload = doc_status_metadata_carry_over(status_doc)
    if extra:
        payload.update(extra)
    for key in drop:
        payload.pop(key, None)
    return payload


# Long-lived per-document *directives* that record how the document should be
# processed (written at enqueue time by ``_initial_doc_status``).  Unlike the
# full carry-over list above, this is the subset that must survive a *reset*
# back to PENDING: it deliberately EXCLUDES the per-attempt timing / result
# fields (``parse_*`` / ``analyzing_*`` / ``parse_warnings`` / ``chunk_opts``),
# which the next attempt regenerates and which would otherwise show stale
# values while the document waits in PENDING.
_DOC_STATUS_METADATA_DIRECTIVE_KEYS: tuple[str, ...] = (
    "process_options",
    "source_file",
    # Defense in depth: journaled custom-chunk patch rows are excluded from
    # pipeline processing/reset entirely, but if one ever reaches a reset the
    # journal must survive — stripping it would orphan the operation's staged
    # data with no recovery anchor (issue #3400).
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    # A FAILED→PENDING reset is exactly the case that must not strip these: the
    # retry re-runs extraction, and if a previous attempt's purge is half-done
    # (anchors deleted, journal at ``anchors_pending``) the journal is the only
    # thing keeping that purge resumable rather than permanently refused. The
    # write-state marker rides along for the same reason — the reset leaves the
    # document PENDING, i.e. still pre-graph.
    KG_WRITE_STATE_METADATA_KEY,
    KG_PURGE_METADATA_KEY,
    # A demotion is an operator decision, not a per-attempt result: the manual
    # FAILED→PENDING retry must not undo it. A demoted row keeps its content and
    # its FAILED status, so it is reset like any other failed document — and
    # without these keys that reset handed the canonical source back to it,
    # putting the key straight back into conflict.
    *DUPLICATE_DEMOTION_METADATA_KEYS,
)


# Per-attempt metadata produced by a parse/analyze/process attempt.  Used as
# the *precise* trigger for normalising an already-PENDING document's metadata
# (see ``apipeline_process_enqueue_documents``' consistency check): only a
# document carrying one of these stale keys is rebuilt to directives-only, so
# unrelated (future / custom) metadata is never dropped just for being
# non-directive.  Covers the timing/skip pairs, parser warnings, and the
# ``extraction_meta`` fields stamped when entering PROCESSING.
_DOC_STATUS_METADATA_ATTEMPT_KEYS: frozenset[str] = frozenset(
    {
        "parse_start_time",
        "parse_end_time",
        "parse_stage_skipped",
        "analyzing_start_time",
        "analyzing_end_time",
        "analyzing_stage_skipped",
        "parse_warnings",
        "chunk_opts",
        "process_start_time",
        "process_end_time",
        "parse_format",
        "parse_engine",
        "chunk_method",
        "skip_kg",
        "mm_chunks",
        "hard_fallback_split",
        "error_type",
        "error_stage",
        # Token-limit truncation summary for THIS attempt. Deliberately absent
        # from the carry-over and directive whitelists: a re-run with a larger
        # output budget must start from a clean slate, and the terminal
        # transition rewrites the key only when the new attempt truncated too.
        LLM_TRUNCATION_METADATA_KEY,
    }
)


def doc_status_reset_metadata(status_doc: Any) -> dict[str, Any]:
    """Build the ``metadata`` payload for a reset back to PENDING.

    Keeps only the long-lived processing directives
    (``_DOC_STATUS_METADATA_DIRECTIVE_KEYS``) and drops every per-attempt
    timing/result field, so a document that is interrupted and re-queued does
    not surface stale parse/analyze timings (or warnings / chunk opts) while it
    waits in PENDING.  ``source_file`` is read with legacy ``source_file_name``
    tolerance and normalised onto the new key, mirroring the parse worker's own
    normalisation so a resumed legacy pending_parse doc keeps its source hint.
    """
    payload: dict[str, Any] = {}
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return payload
    # Iterate the directive whitelist so a future addition to
    # _DOC_STATUS_METADATA_DIRECTIVE_KEYS is automatically carried across a
    # reset.  ``source_file`` is read with legacy ``source_file_name``
    # tolerance and normalised onto the new key; all other directives carry
    # verbatim when present and non-blank.
    for key in _DOC_STATUS_METADATA_DIRECTIVE_KEYS:
        if key == "source_file":
            value = read_source_file_basename(raw_metadata)
        else:
            value = raw_metadata.get(key)
        if value not in (None, ""):
            payload[key] = value
    return payload


def doc_status_parse_failure_fields(
    error: BaseException,
    *,
    status_doc: Any,
    engine_hint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the FAILED-upsert extras for a parse-stage failure.

    Returns ``(extra_fields, metadata_extra)`` for
    ``_upsert_doc_status_transition``.  Restores the UI fields that the
    pre-deferral enqueue-time error documents carried: a human-readable
    ``content_summary`` (written only when the document has none —
    ``pending_parse`` docs enqueue with an empty summary, while raw
    passthrough docs keep their real one — or when the existing summary
    is itself a ``[File Extraction]``-generated one from a previous
    failed attempt, which the retry reset preserves and would otherwise
    go stale next to the fresh ``error_msg``) plus ``metadata.error_type`` /
    ``metadata.error_stage`` for error classification.  ``error_type``
    keeps the legacy ``file_extraction_error`` value consumers may match
    on; ``error_stage`` distinguishes the parse-worker failure from an
    enqueue-time error document.  Both metadata keys ride
    ``metadata_extra`` and are deliberately NOT in the carry-over /
    directive whitelists, so the next transition (retry reset, PARSING)
    drops them automatically — mirroring how ``error_msg`` is cleared.

    ``engine_hint`` (the parse worker's resolved engine key, falling back
    to its queue-group id) is stamped as ``parse_engine`` only when the
    failure happened before the post-parse stamp put one on
    ``status_doc.metadata``; an existing value always wins via carry-over.
    """
    error_text = str(error)
    if not error_text.strip():
        # Some exceptions (e.g. certain httpx transport errors) stringify to
        # an empty message; fall back to the class name so the WebUI never
        # drops the Error Message row for a FAILED document.
        error_text = type(error).__name__
    extra_fields: dict[str, Any] = {"error_msg": error_text}
    current_summary = str(
        doc_status_field(status_doc, "content_summary", "") or ""
    ).strip()
    if not current_summary or current_summary.startswith(
        FILE_EXTRACTION_SUMMARY_PREFIX
    ):
        extra_fields["content_summary"] = (
            FILE_EXTRACTION_SUMMARY_PREFIX + get_content_summary(error_text)
        )
    metadata_extra: dict[str, Any] = {
        "error_type": "file_extraction_error",
        "error_stage": "parse",
    }
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    has_engine = isinstance(raw_metadata, dict) and raw_metadata.get("parse_engine")
    if engine_hint and not has_engine:
        metadata_extra["parse_engine"] = engine_hint
    return extra_fields, metadata_extra


def doc_status_metadata_has_attempt_fields(status_doc: Any) -> bool:
    """True when ``status_doc.metadata`` carries any per-attempt field.

    Used to decide whether an already-PENDING document needs its metadata
    normalised to directives-only — avoids a redundant upsert for documents
    whose metadata is already clean (or only holds non-attempt custom fields).
    """
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return False
    return not _DOC_STATUS_METADATA_ATTEMPT_KEYS.isdisjoint(raw_metadata)


def doc_status_value(doc: Any) -> str:
    status = doc_status_field(doc, "status", "")
    if isinstance(status, DocStatus):
        return status.value
    return str(status or "")


# Statuses that count against the admission capacity (LR2 §9.1): work the
# pipeline still owes. PROCESSED and FAILED are finished — a FAILED document
# only becomes active again through an explicit manual retry, which is exempt
# from the cap anyway.
ADMISSION_ACTIVE_STATUSES: tuple[DocStatus, ...] = (
    DocStatus.PENDING,
    DocStatus.PARSING,
    DocStatus.ANALYZING,
    DocStatus.PROCESSING,
)


def _overrides_base(doc_status: Any, method_name: str) -> bool:
    """Whether this backend replaced ``DocStatusStorage``'s fail-closed default."""
    base = getattr(DocStatusStorage, method_name, None)
    implementation = getattr(type(doc_status), method_name, None)
    return implementation is not None and implementation is not base


def describe_doc_status_capabilities(doc_status: Any) -> dict[str, bool]:
    """Report which strict capabilities this doc_status backend actually has.

    The scheduling contract fails CLOSED on a missing capability: admission
    refuses with 503, a stale FAILED stub is kept instead of deleted, source
    conflicts cannot be listed. Each of those is a silent, confusing degradation
    from the outside — an operator staring at 503s needs to be told that the
    configured backend simply cannot count, not to go hunting for a database
    problem (LR2 Phase 6 item 2).

    Probes the class rather than calling anything: /health must stay cheap and
    side-effect free.
    """
    return {
        # Bounded keyset paging + typed source resolution are @abstractmethod on
        # DocStatusStorage, so a first-party backend always has them; a
        # third-party subclass that stubs them out shows up here as False.
        "scheduling_pages": _overrides_base(doc_status, "get_docs_by_statuses_page"),
        "typed_source_resolution": _overrides_base(
            doc_status, "resolve_doc_source_strict"
        ),
        # Base defaults raise StorageCapabilityError — these three are the ones a
        # deployment can genuinely lack.
        "strict_active_count": _overrides_base(doc_status, "count_docs_by_statuses"),
        "source_conflict_listing": _overrides_base(
            doc_status, "list_source_conflicts_page"
        ),
        "source_conflict_repair": _overrides_base(doc_status, "repair_source_conflict"),
        # Class-level opt-in: gates the STALE_STUB deletion during a scan.
        "strict_point_reads": bool(
            getattr(type(doc_status), "supports_strict_point_reads", False)
        ),
    }


# What each strict capability costs when it is absent, in the terms an operator
# experiences. Only the capabilities a deployment can genuinely lack appear here;
# the two @abstractmethod ones cannot be missing on an instantiable backend.
_CAPABILITY_CONSEQUENCES: dict[str, str] = {
    "strict_active_count": (
        "MAX_PENDING_DOCUMENTS admission cannot count active documents, so every "
        "upload and text insert is refused with HTTP 503 while admission is enabled"
    ),
    "source_conflict_listing": (
        "GET /documents/source_conflicts answers 501, so a scan that stops on a "
        "duplicate file source cannot be diagnosed through the API"
    ),
    "source_conflict_repair": (
        "POST /documents/source_conflicts/repair answers 501, so a duplicate file "
        "source has to be resolved by deleting documents"
    ),
    "strict_point_reads": (
        "a scan cannot confirm that a stale FAILED stub is really gone, so it keeps "
        "the stub and re-examines that file on every scan"
    ),
}


def enforce_strict_storage_capabilities(
    doc_status: Any, *, require: bool = False
) -> dict[str, bool]:
    """Report (and optionally refuse to start on) missing strict capabilities.

    The scheduling contract fails closed on every one of these, which from the
    outside looks like a broken database rather than a backend that was never
    able to do the thing. ``/health`` publishes the same probe, but a status page
    only helps someone who already suspects a problem — the startup log is where
    an operator finds out before the first 503 (LR2 §11, Phase 1 acceptance).

    Never raises for its own reasons: a probe failure is logged and treated as
    "nothing to report", because refusing to start over a broken *diagnostic*
    would be worse than the degradation it was looking for. ``require=True``
    raises :class:`StorageCapabilityError` only for genuinely missing
    capabilities.
    """
    try:
        capabilities = describe_doc_status_capabilities(doc_status)
    except Exception as probe_error:  # pragma: no cover - defensive
        logger.warning(f"Could not probe doc_status capabilities: {probe_error}")
        return {}

    missing = [
        name
        for name, consequence in _CAPABILITY_CONSEQUENCES.items()
        if not capabilities.get(name, False)
    ]
    if not missing:
        return capabilities

    backend = type(doc_status).__name__
    details = "; ".join(
        f"{name} — {_CAPABILITY_CONSEQUENCES[name]}" for name in missing
    )
    if require:
        raise StorageCapabilityError(
            f"{backend} is missing strict capabilities required by "
            f"PIPELINE_REQUIRE_STRICT_STORAGE_READS=true: {details}"
        )
    logger.warning(
        f"{backend} is missing strict doc_status capabilities: {details}. "
        "Set PIPELINE_REQUIRE_STRICT_STORAGE_READS=true to refuse to start "
        "instead; /health reports the same gaps under 'capabilities'."
    )
    return capabilities


async def count_active_documents(doc_status: DocStatusStorage) -> int:
    """Strictly count the documents that occupy admission capacity.

    Deliberately re-queried per admission decision instead of tracked in a
    maintained counter: a counter is a second source of truth that drifts on
    crash, backend swap or external write, and admission is the one place where
    a drifting count either wedges ingestion or lets it grow unbounded.

    Fail-closed: ``count_docs_by_statuses`` raises rather than returning zeros
    on failure, and this helper does not soften that — a caller that cannot
    learn the count must refuse the request (503), never assume there is room.
    """
    started = time.monotonic()
    try:
        return await doc_status.count_docs_by_statuses(
            list(ADMISSION_ACTIVE_STATUSES), strict=True
        )
    finally:
        # Recorded even on failure: a count that times out is exactly the case an
        # operator needs to see when uploads start answering 503.
        pipeline_metrics.observe(
            pipeline_metrics.ACTIVE_COUNT_SECONDS, time.monotonic() - started
        )


# Sidecar item ids embed ``doc_hash`` (= doc_id without the ``doc-`` prefix),
# and for pending_parse uploads doc_id derives from the filename — so the
# same content under two filenames renders with different ids in
# ``merged_text``. Strip those surfaces before hashing so cross-filename
# content_hash dedup actually fires.
_SIDECAR_ID_PATTERN = re.compile(r"\b(tb|im|eq)-[0-9a-f]{32}-(\d{4})\b")
_ASSET_PATH_PATTERN = re.compile(r'(?<=path=")[^"]*\.blocks\.assets/')


def normalize_merged_text_for_hash(content: str) -> str:
    """Strip filename-derived prefixes from sidecar ids and asset paths.

    Idempotent and safe on plain text (matches the doc_hash literal only —
    32 lowercase hex digits between the modality prefix and a 4-digit
    sequence). RAW text bodies without sidecar markup pass through
    unchanged.
    """
    if not content:
        return content
    content = _SIDECAR_ID_PATTERN.sub(r"\1-<DOC>-\2", content)
    content = _ASSET_PATH_PATTERN.sub("<ASSETS>/", content)
    return content


def compute_text_content_hash(content: str) -> str:
    """MD5 hex digest of text content used for cross-filename dedup.

    Input is normalized via :func:`normalize_merged_text_for_hash` first so
    sidecar-rendered bodies dedupe across filenames despite carrying
    filename-derived item ids and asset paths.
    """
    return compute_mdhash_id(normalize_merged_text_for_hash(content), prefix="")


def compute_file_content_hash(path_str: str) -> str | None:
    """Stream-compute MD5 of a file's bytes; returns None if unreadable.

    Resolves the LightRAG ``*.blocks.jsonl`` conventions used by
    ``_load_lightrag_document_content`` so the hash matches the actual
    document body regardless of whether ``path_str`` points at the blocks
    file directly or its parent directory/base name.
    """
    if not path_str:
        return None
    try:
        path = Path(path_str)
        if path.is_dir():
            candidates = sorted(path.glob("*.blocks.jsonl"))
            if not candidates:
                return None
            path = candidates[0]
        elif not (path.exists() and path.is_file()):
            blocks_path = Path(path_str + ".blocks.jsonl")
            if blocks_path.exists() and blocks_path.is_file():
                path = blocks_path
            else:
                return None
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute file content hash for {path_str}: {e}")
        return None


def configured_input_dir() -> Path:
    input_dir = os.getenv("INPUT_DIR", "").strip()
    return Path(input_dir) if input_dir else Path.cwd() / "inputs"


async def resolve_existing_doc_source(
    doc_status: DocStatusStorage, file_path: Any
) -> SourceResolution:
    """Resolve a file path to a typed, fail-closed source resolution.

    Inputs are normalized via :func:`normalize_document_file_path` so callers
    may pass either the bare canonical name (``abc.docx``) or a hint-bearing
    variant (``abc.[native-iet].docx``); both resolve to the same logical
    document. A name with no usable identity (``unknown_source``) collides with
    nothing and is reported :class:`SourceAbsent` without querying.

    Replaces ``get_doc_by_file_basename`` for the enqueue duplicate check. That
    method is best-effort by contract — several backends swallow transport
    errors and return ``None`` — and enqueue reads "no match" as "not a
    duplicate", so a storage blip admitted a second row for a filename that
    already existed. It also returns SOME primary on a historical basename
    collision, hiding the collision instead of surfacing it. The strict
    resolver raises instead of guessing, and distinguishes Absent / Unique /
    Conflict so the caller can refuse the ambiguous case explicitly.
    """
    basename = normalize_document_file_path(file_path)
    if basename == "unknown_source":
        return SourceAbsent()
    return await doc_status.resolve_doc_source_strict(basename)


@asynccontextmanager
async def source_candidate_set_lock(workspace: str, canonical_source_key: str):
    """Serialize writers that change WHICH documents claim one canonical source.

    LR2 §5.5 requires a source-conflict commit to be mutually exclusive with
    every writer that can change that key's candidate set — enqueue, delete,
    manual reset, scan stale-stub deletion AND the processing stage's duplicate
    marking. This is the keyed lock both sides take, defined once so the two
    cannot compute a different namespace or key and silently stop excluding each
    other (they are in different modules and only the string makes them the same
    lock).

    Lock ORDER, when a caller needs more than this one: the workspace's
    enqueue-serialize lock OUTSIDE, this keyed lock inside, any
    ``pipeline_status`` lock innermost. That is the order the enqueue path would
    naturally take (it already holds enqueue-serialize and would reach for a
    keyed source lock inside it), so every holder of both agrees and no cycle
    exists. ``source_conflict_repair_lock`` acquires them in exactly that order.

    Process-LOCAL machinery, so the scope is one deployment (its worker
    processes included — they share the Manager the master created). A standalone
    CLI or a second deployment writing the same database is outside it; the CAS
    and the post-commit re-resolve are what keep those honest.

    A document with no usable source identity (``unknown_source`` and friends) is
    never a primary candidate for any key — every backend's candidate query
    excludes it — so there is nothing to serialize and no lock is taken. That is
    not a shortcut: locking on the sentinel would funnel EVERY unsourced document
    in the workspace through one keyed lock.
    """
    from lightrag.constants import SOURCE_CONFLICT_LOCK_NAMESPACE
    from lightrag.kg.shared_storage import get_storage_keyed_lock

    canonical = normalize_document_file_path(canonical_source_key)
    if not canonical or canonical in PLACEHOLDER_DOCUMENT_SOURCES:
        yield
        return
    namespace = (
        f"{workspace}:{SOURCE_CONFLICT_LOCK_NAMESPACE}"
        if workspace
        else SOURCE_CONFLICT_LOCK_NAMESPACE
    )
    async with get_storage_keyed_lock(
        [canonical], namespace=namespace, enable_logging=False
    ):
        yield


async def get_existing_doc_by_content_hash(
    doc_status: DocStatusStorage,
    content_hash: str,
    *,
    candidate_doc_id: str = "",
) -> tuple[str, Any] | None:
    """Find an existing doc_status record by content hash.

    ``candidate_doc_id`` is the id the caller is about to create — the enqueue
    dedup check runs before that row exists, so excluding it removes nothing by
    id; what it buys is the other half of ``exclude_doc_id`` (see the base
    contract): a row that merely POINTS at that id is not a holder of the
    content, it is a record that the content belongs to the document being
    created. Without it, a document whose primary row was deleted can never be
    re-ingested — its doc id is derived from the file name, so a re-upload asks
    about the very id the leftover pointer names, and the pointer keeps
    answering for it.
    """
    if not content_hash:
        return None
    return await doc_status.get_doc_by_content_hash(
        content_hash, exclude_doc_id=candidate_doc_id or None
    )


async def get_duplicate_doc_by_content_hash(
    doc_status: DocStatusStorage, content_hash: str, current_doc_id: str
) -> tuple[str, Any] | None:
    """Find ANOTHER doc_status record with the same content hash.

    Excludes ``current_doc_id`` in-query (LR2 Phase 2.5): by the time the
    post-parse duplicate check runs, the current doc's own content_hash is
    already persisted to its doc_status row, so an ``exclude_doc_id``-less
    lookup would return the doc itself. Passing ``exclude_doc_id`` gets the
    earliest OTHER holder of the hash in one bounded/indexed query — replacing
    the previous ``get_docs_by_statuses(list(DocStatus))`` full-store scan
    fallback, which materialized the entire doc_status store (every status,
    including the whole PROCESSED corpus with chunks_list) once per document.

    The same exclusion also drops a row that points at ``current_doc_id`` as its
    own original while the search continues past it, so this can neither close an
    ``is_duplicate`` cycle nor hide a genuine third holder behind one (see the
    base contract for both halves).
    """
    if not content_hash:
        return None
    return await doc_status.get_doc_by_content_hash(
        content_hash, exclude_doc_id=current_doc_id
    )


def make_lightrag_doc_content(merged_text: str) -> str:
    """Build the ``full_docs.content`` value for ``format=lightrag`` records.

    The result has shape ``"{{LRdoc}}<merged_text>"`` — the marker prefix
    distinguishes lightrag-format full_docs from raw-format ones, and the
    body is the complete merged text from the ``.blocks.jsonl`` content
    lines so F-chunking can run identically on raw and lightrag inputs
    (the prefix is stripped at chunking time via
    ``strip_lightrag_doc_prefix``).
    """
    return f"{LIGHTRAG_DOC_CONTENT_PREFIX}{merged_text or ''}"


def strip_lightrag_doc_prefix(content: str | None, parse_format: str | None) -> str:
    """Return the bare body for a stored ``full_docs.content`` value.

    The ``{{LRdoc}}`` marker is stripped **only** when ``parse_format``
    indicates the record is in lightrag format.  Any other ``parse_format``
    (``raw``, ``pending_parse``, ``None`` ...) returns the content
    unchanged so a raw document whose literal body happens to start with
    ``{{LRdoc}}`` is never silently truncated.

    Centralizing the format check here turns "must check format before
    stripping" from a caller-side discipline into a structural property of
    the function: any future call site that forgets to gate is protected
    automatically.
    """
    if (
        parse_format == FULL_DOCS_FORMAT_LIGHTRAG
        and isinstance(content, str)
        and content.startswith(LIGHTRAG_DOC_CONTENT_PREFIX)
    ):
        return content[len(LIGHTRAG_DOC_CONTENT_PREFIX) :]
    return content or ""


# ---------------------------------------------------------------------------
# Document path / artifact helpers (moved from _PipelineMixin)
# ---------------------------------------------------------------------------


def input_dir_path() -> Path:
    return configured_input_dir()


def parsed_dir() -> Path:
    """Return the project-wide parsed-artifact root: ``<input_dir>/__parsed__``."""
    return input_dir_path() / PARSED_DIR_NAME


def parsed_artifact_dir_for(
    file_path: str, *, parent_hint: Path | str | None = None
) -> Path:
    """Return the per-document sidecar directory for ``file_path``.

    ``file_path`` must already be canonical (run ``normalize_document_file_path``
    first if unsure). When ``parent_hint`` is supplied (e.g. the live source
    file's parent), the sidecar is placed next to it under ``__parsed__/``
    rather than under the global ``input_dir``; this keeps test isolation
    intact when the source lives outside ``INPUT_DIR``. On collision with an
    existing non-directory entry, the helper appends ``_001``..``_999`` and
    finally a unix timestamp suffix.
    """
    if parent_hint is not None:
        hint = Path(parent_hint)
        # ``hint`` may already point at a ``__parsed__/`` dir (e.g. when the
        # caller re-archived a source); reuse it in place rather than nesting.
        root = hint if hint.name == PARSED_DIR_NAME else hint / PARSED_DIR_NAME
    else:
        root = parsed_dir()
    source_name = (
        canonicalize_parser_hinted_basename(file_path or "document") or "document"
    )
    artifact_name = f"{source_name}.parsed"
    artifact_dir = root / artifact_name
    if not artifact_dir.exists() or artifact_dir.is_dir():
        return artifact_dir

    for i in range(1, 1000):
        candidate = root / f"{artifact_name}_{i:03d}"
        if not candidate.exists() or candidate.is_dir():
            return candidate

    return root / f"{artifact_name}_{int(time.time())}"


# ---------------------------------------------------------------------------
# Sidecar URI helpers (``full_docs.sidecar_location``)
# ---------------------------------------------------------------------------
#
# Sidecar URI scheme conventions:
#   - Local:  ``file:///abs/path/to/abc.parsed/``   (trailing slash required)
#   - Remote: ``s3://bucket/workspace/abc.parsed/`` (future; resolver returns
#             None today so local readers gracefully skip)
#   - Unknown sentinel: literal string ``"unknown_source"``


def sidecar_uri_for(parsed_artifact_dir: Path | str) -> str:
    """Build the canonical sidecar URI for a local artifact directory.

    The result always ends with ``/`` so a reader can distinguish a directory
    from a file at the URI level. Non-ASCII characters are percent-encoded.
    """
    p = Path(parsed_artifact_dir).resolve()
    encoded = quote(str(p), safe="/")
    return f"file://{encoded}/"


def resolve_sidecar_uri(uri: str | None) -> Path | None:
    """Decode a sidecar URI into a local filesystem Path.

    Returns None for the unknown sentinel, empty input, or any non-``file://``
    scheme (remote schemes will get their own resolvers).
    """
    if not uri or uri == SIDECAR_LOCATION_UNKNOWN:
        return None
    parts = urlsplit(uri)
    if parts.scheme != "file":
        return None
    path_str = unquote(parts.path)
    if path_str.endswith("/") and len(path_str) > 1:
        path_str = path_str[:-1]
    return Path(path_str)


def sidecar_blocks_path(uri: str | None) -> str | None:
    """Locate the first ``*.blocks.jsonl`` file inside a sidecar URI.

    Returns the absolute path as a string, or None when the URI cannot be
    resolved locally or the directory holds no blocks file.
    """
    d = resolve_sidecar_uri(uri)
    if d is None or not d.is_dir():
        return None
    candidates = sorted(d.glob("*.blocks.jsonl"))
    return str(candidates[0]) if candidates else None


def sidecar_modality_path(uri: str | None, modality: str) -> str | None:
    """Return the path for a sidecar modality JSON (drawings/tables/equations).

    Does not require the file to exist — callers check. Returns None when the
    sidecar URI cannot be resolved or has no blocks file to anchor the name.
    """
    blocks = sidecar_blocks_path(uri)
    if not blocks:
        return None
    return f"{blocks[: -len('.blocks.jsonl')]}.{modality}.json"


def sidecar_assets_dir_for_uri(uri: str | None) -> Path | None:
    """Return the ``*.blocks.assets/`` directory Path for a sidecar URI.

    The directory may not exist; callers create it on first asset write.
    """
    blocks = sidecar_blocks_path(uri)
    if not blocks:
        return None
    return Path(f"{blocks[: -len('.blocks.jsonl')]}.blocks.assets")


# ---------------------------------------------------------------------------
# Source archive helpers
# ---------------------------------------------------------------------------


async def archive_docx_source_after_full_docs_sync(source_path: str) -> str | None:
    source = Path(source_path)
    try:
        target = await move_file_to_parsed_dir(source, skip_if_already_parsed=True)
    except Exception as e:
        logger.warning(
            f"[parse] Source archive skipped after full_docs sync: {source_path}: {e}"
        )
        return None
    if target is None:
        return None
    if target != source:
        logger.debug(
            f"[parse] Archived DOCX source after full_docs sync: {source} -> {target}"
        )
    return str(target)


async def archive_source_after_full_docs_sync(source_path: str) -> str | None:
    return await archive_docx_source_after_full_docs_sync(source_path)


# ---------------------------------------------------------------------------
# LightRAG Document blocks loader
# ---------------------------------------------------------------------------


async def load_lightrag_document_content(sidecar_uri: str) -> tuple[str, str]:
    """Load LightRAG Document blocks and return ``(merged_text, blocks_path)``.

    ``sidecar_uri`` is a sidecar location URI (see ``sidecar_uri_for``); this
    locates the ``*.blocks.jsonl`` file inside it, reads the content lines
    (skipping the meta header at index 0 and any non-content entries), and
    returns the merged body plus the absolute blocks path.
    """
    resolved = sidecar_blocks_path(sidecar_uri)
    if resolved is None:
        raise FileNotFoundError(
            f"LightRAG blocks file not found from sidecar uri: {sidecar_uri}"
        )
    blocks_path = Path(resolved)

    merged_parts: list[str] = []
    with blocks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if i == 0:
                continue
            if obj.get("type") != "content":
                continue
            content = obj.get("content", "")
            if isinstance(content, str) and content.strip():
                merged_parts.append(content)

    return "\n\n".join(merged_parts), str(blocks_path)


# ---------------------------------------------------------------------------
# Payload introspection helpers (parser response normalization)
# ---------------------------------------------------------------------------


def get_by_path(payload: Any, path: str) -> Any:
    if not path:
        return None
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def extract_content_list_from_payload(
    payload: Any,
) -> list[dict[str, Any]] | None:
    """Try to find a MinerU/Docling-like content list from arbitrary JSON payload."""
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            first = payload[0]
            if "type" in first or "label" in first or "text" in first:
                return cast(list[dict[str, Any]], payload)
        return None
    if not isinstance(payload, dict):
        return None

    # Common direct keys first
    for key in ("content_list", "content", "items", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            extracted = extract_content_list_from_payload(value)
            if extracted is not None:
                return extracted
        elif isinstance(value, dict):
            extracted = extract_content_list_from_payload(value)
            if extracted is not None:
                return extracted

    # Deep search as fallback
    for value in payload.values():
        extracted = extract_content_list_from_payload(value)
        if extracted is not None:
            return extracted
    return None


def normalize_parser_result_to_content_list(
    parser_result: str | list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalize parser result to structured content list if possible."""
    if parser_result is None:
        return None
    if isinstance(parser_result, list):
        return extract_content_list_from_payload(parser_result)
    if isinstance(parser_result, dict):
        return extract_content_list_from_payload(parser_result)
    text = str(parser_result).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return extract_content_list_from_payload(payload)
    except Exception:
        return None


# Multimodal entity injection used to live here as a centralized post-pass
# over all chunk_results. It has been moved into
# :func:`lightrag.operate.extract_entities._process_single_content` so each
# multimodal chunk injects its own entity/relation records while still under
# its concurrency slot.  The chunk's ``sidecar.type`` (drawing/table/equation)
# is the dispatch key; see operate.py for the new logic.
