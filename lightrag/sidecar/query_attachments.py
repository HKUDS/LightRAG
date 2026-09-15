"""Query-time sidecar join: index figures on retrieval chunks and resolve
``data.attachments`` from ``drawings.json``.

PR-1 pipeline (no present-mode / no id-selection hook):

  Collect — ``collect_im_ids_on_truncated_chunks``: figure bucket, then text bucket;
     merge ``figure_ids + text_ids`` (cross-bucket figure wins; **not** truncated
     list linear order). Memory only; no drawings.json / metadata / disk I/O.

  Index — ``index_figures_on_chunks`` → ``apply_index_figures_to_raw_data`` writes
     ``metadata.drawing_candidate_whitelist`` in Collect order (drawings.json JSON
     filter: entry + non-empty path; **no disk verify**; skip only, no reorder).

  Enrich — ``enrich_raw_data_attachments`` reads whitelist → ``data.attachments``
     (same order; ``verify_file`` on disk; skip only; operational failures degrade
     to ``attachments: []``; unexpected errors propagate).

File sections (top to bottom): ID helpers → Collect → Index → Enrich → Resolve.
Pipeline stages first; Resolve is shared toolbox at the bottom (drawings.json,
prefetch, asset paths). Index uses JSON filter only; Enrich adds disk verify.


"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from lightrag.chunk_schema import normalize_chunk_sidecar
from lightrag.utils import logger, validate_file_path_security
from lightrag.utils_pipeline import (
    resolve_sidecar_uri,
    sidecar_assets_dir_for_uri,
    sidecar_modality_path,
)

if TYPE_CHECKING:
    from lightrag.base import BaseKVStorage

# ---------------------------------------------------------------------------
# Constants / module state
# ---------------------------------------------------------------------------

IM_ID_RE = re.compile(r"^im-(?P<doc_hash>[a-f0-9]+)-(?P<seq>\d+)$")
DRAWING_TAG_RE = re.compile(
    r'<drawing\b[^>]*\bid=(["\'])(?P<im_id>im-[^"\']+)\1',
    flags=re.IGNORECASE,
)

AttachmentDict = dict[str, Any]

DRAWING_CANDIDATE_WHITELIST_KEY = "drawing_candidate_whitelist"

# ``(im_id, chunk_doc_id, chunk_id)`` — optional Collect audit hook (Index only).
OnImIdFoundCallback = Callable[[str, str, str | None], None]

# Process-global drawings index cache: (mtime, index) per sidecar URI.
# - mtime invalidates entries when drawings.json appears or is rewritten
#   (avoids pinning a negative ``{}`` while a doc is still parsing).
# - LRU cap bounds memory in long-lived servers.
# - Lock protects cache access from concurrent ``asyncio.to_thread`` callers.
_DRAWINGS_INDEX_CACHE_MAX = 128
_drawings_index_cache: OrderedDict[str, tuple[float | None, dict[str, Any]]] = (
    OrderedDict()
)
_drawings_index_cache_lock = threading.Lock()

# Storage / I/O failures that may degrade attachments or Index prefetch — not bugs.
_SIDECAR_DEGRADE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    json.JSONDecodeError,
    ConnectionError,
    TimeoutError,
)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def is_im_id(value: str | None) -> bool:
    """Return True if ``value`` matches the canonical ``im-<doc_hash>-<seq>`` form."""
    if not value or not isinstance(value, str):
        return False
    return IM_ID_RE.match(value.strip()) is not None


def doc_id_from_im_id(im_id: str) -> str | None:
    """Derive ``doc-<doc_hash>`` from an im-id, or None if the im-id is invalid."""
    match = IM_ID_RE.match(im_id.strip())
    if not match:
        return None
    return f"doc-{match.group('doc_hash')}"


def doc_id_from_chunk_id(chunk_id: str | None) -> str | None:
    """Derive ``doc-<hash>`` from a ``doc-<hash>-chunk-…`` chunk_id when possible."""
    if not chunk_id or not isinstance(chunk_id, str):
        return None
    if chunk_id.startswith("doc-"):
        rest = chunk_id[4:]
        dash = rest.find("-")
        if dash == -1:
            return chunk_id
        return f"doc-{rest[:dash]}"
    return None


# ---------------------------------------------------------------------------
# Collect (ordered im-ids from truncated chunks)
#
# Memory only: route each chunk to figure or text bucket; merge
# ``figure_ids + text_ids`` (cross-bucket figure wins). Optional
# ``on_im_id_found`` audit hook (Index passes mismatch warn). No drawings.json,
# no metadata write, no disk I/O.
# ---------------------------------------------------------------------------


def _append_unique_im_id(bucket: list[str], seen: set[str], im_id: str | None) -> None:
    """Append ``im_id`` to ``bucket`` if valid and not yet in bucket ``seen``."""
    if not im_id or not is_im_id(im_id):
        return
    im_id = im_id.strip()
    if im_id in seen:
        return
    seen.add(im_id)
    bucket.append(im_id)


def collect_im_ids_from_text_chunk(chunk: dict[str, Any]) -> list[str]:
    """Collect im-ids from inline ``<drawing>`` tags in chunk content."""
    ordered: list[str] = []
    seen: set[str] = set()
    for match in DRAWING_TAG_RE.finditer(chunk.get("content") or ""):
        _append_unique_im_id(ordered, seen, match.group("im_id"))
    return ordered


def _chunk_doc_id(record: dict[str, Any] | None, chunk_id: str) -> str | None:
    """Doc id attributed to a chunk row (``full_doc_id`` or parsed from ``chunk_id``)."""
    if record:
        doc_id = record.get("full_doc_id") or doc_id_from_chunk_id(chunk_id)
    else:
        doc_id = doc_id_from_chunk_id(chunk_id)
    return str(doc_id).strip() if doc_id else None


def _warn_im_id_doc_hash_mismatch(
    im_id: str,
    chunk_doc_id: str,
    chunk_id: str | None = None,
) -> None:
    """Log when chunk attribution disagrees with the im-id embedded doc hash.

    Matches ``OnImIdFoundCallback`` for Index:
    ``collect_im_ids_on_truncated_chunks(..., on_im_id_found=_warn_im_id_doc_hash_mismatch)``.
    Observability only — resolve uses ``doc_id_from_im_id(im_id)`` (im-id authoritative).
    """
    im_doc_id = doc_id_from_im_id(im_id)
    if not im_doc_id or not chunk_doc_id or im_doc_id == chunk_doc_id:
        return
    chunk_hint = f" {chunk_id}" if chunk_id else ""
    logger.warning(
        "[query_attachments] im-id doc hash mismatch: %s embedded %s but chunk%s "
        "suggests %s; resolve uses %s (im-id authoritative)",
        im_id,
        im_doc_id,
        chunk_hint,
        chunk_doc_id,
        im_doc_id,
    )


def _notify_im_id_sighting(
    im_id: str,
    chunk_doc_id: str | None,
    chunk_id: str | None,
    on_im_id_found: OnImIdFoundCallback | None,
) -> None:
    """Optional Index audit hook for one valid im-id sighting on a chunk."""
    if on_im_id_found and chunk_doc_id:
        on_im_id_found(im_id, chunk_doc_id, chunk_id)


def _collect_im_id_sighting(
    im_id: str | None,
    bucket: list[str],
    seen: set[str],
    chunk_doc_id: str | None,
    chunk_id: str | None,
    on_im_id_found: OnImIdFoundCallback | None,
) -> None:
    """Notify optional audit hook, then append one im-id to a Collect bucket."""
    if im_id and is_im_id(im_id):
        _notify_im_id_sighting(im_id.strip(), chunk_doc_id, chunk_id, on_im_id_found)
    _append_unique_im_id(bucket, seen, im_id)


def _collect_figure_chunk_im_ids(
    sidecar: dict[str, Any] | None,
    figure_ids: list[str],
    seen_figure: set[str],
    chunk_doc_id: str | None,
    chunk_id: str | None,
    on_im_id_found: OnImIdFoundCallback | None,
) -> None:
    """Collect the single figure sidecar primary ``id`` into the figure bucket."""
    _collect_im_id_sighting(
        sidecar.get("id") if sidecar else None,
        figure_ids,
        seen_figure,
        chunk_doc_id,
        chunk_id,
        on_im_id_found,
    )


def _collect_text_chunk_im_ids(
    chunk: dict[str, Any],
    text_ids: list[str],
    seen_text: set[str],
    chunk_doc_id: str | None,
    chunk_id: str | None,
    on_im_id_found: OnImIdFoundCallback | None,
) -> None:
    """Collect inline ``<drawing>`` im-ids from a text chunk into the text bucket."""
    for im_id in collect_im_ids_from_text_chunk(chunk):
        _collect_im_id_sighting(
            im_id, text_ids, seen_text, chunk_doc_id, chunk_id, on_im_id_found
        )


def _merge_figure_text_im_ids(figure_ids: list[str], text_ids: list[str]) -> list[str]:
    """Merge Collect buckets: ``figure_ids + text_ids`` (cross-bucket figure wins)."""
    figure_set = set(figure_ids)
    text_ids = [im_id for im_id in text_ids if im_id not in figure_set]
    return figure_ids + text_ids


def collect_im_ids_on_truncated_chunks(
    truncated_chunks: list[dict[str, Any]],
    chunk_records_by_id: dict[str, dict[str, Any] | None],
    *,
    on_im_id_found: OnImIdFoundCallback | None = None,
) -> list[str]:
    """Collect im-ids across truncated chunks (figure bucket, then text bucket).

    Input:
        truncated_chunks: Retrieval chunks after token truncation.
        chunk_records_by_id: ``chunk_id ->`` KV record (sidecar), from
            ``_fetch_chunk_records``.
        on_im_id_found: Optional ``(im_id, chunk_doc_id, chunk_id)`` callback.
            Index passes ``_warn_im_id_doc_hash_mismatch``; omit for collect-only
            (unit tests, no chunk scan side effects).

    Output:
        Ordered im-id list: ``figure_ids + text_ids``. Figure im-ids always
        precede text im-ids regardless of ``truncated_chunks`` order.

    Steps:
        1. One pass over ``truncated_chunks``; route each chunk to figure or text bucket.
        2. ``_collect_figure_chunk_im_ids`` / ``_collect_text_chunk_im_ids`` — sighting,
           optional ``on_im_id_found``, append (local dedupe per bucket).
        3. ``_merge_figure_text_im_ids`` — figure segment first; drop text duplicates
           already in figure bucket.
    """
    figure_ids: list[str] = []
    text_ids: list[str] = []
    seen_figure: set[str] = set()
    seen_text: set[str] = set()

    for chunk in truncated_chunks:
        chunk_id = chunk.get("chunk_id") or ""
        record = chunk_records_by_id.get(chunk_id)
        sidecar = normalize_chunk_sidecar(record) if record else None
        chunk_doc_id = _chunk_doc_id(record, chunk_id)
        chunk_id_hint = chunk_id or None

        if sidecar and sidecar.get("type") == "drawing":
            _collect_figure_chunk_im_ids(
                sidecar,
                figure_ids,
                seen_figure,
                chunk_doc_id,
                chunk_id_hint,
                on_im_id_found,
            )
        else:
            _collect_text_chunk_im_ids(
                chunk,
                text_ids,
                seen_text,
                chunk_doc_id,
                chunk_id_hint,
                on_im_id_found,
            )

    return _merge_figure_text_im_ids(figure_ids, text_ids)


# ---------------------------------------------------------------------------
# Index (Collect → metadata.drawing_candidate_whitelist)
#
# Operate hook after truncation: fetch chunk KV → collect → ``im_ids_to_doc_map``
# → ``_whitelist_indexable_drawings`` (Resolve prefetch below; JSON filter: entry +
# non-empty path; **no disk verify**). ``apply_index_figures_to_raw_data`` writes
# whitelist in Collect order (skip only, no reorder).
# ---------------------------------------------------------------------------


@dataclass
class IndexFiguresResult:
    """Result of indexing figures on truncated retrieval chunks.

    PR-1 output is ``whitelist`` only (written to metadata via ``apply``).
    Chunks are not returned or mutated here — callers keep their input list for
    LLM context / ``data.chunks`` (no ``figure_ids`` on chunks in PR-1).
    Letter labels / Figure List / References lines are deferred to PR-5.
    """

    whitelist: list[str] = field(default_factory=list)


def _is_indexable_in_drawings_index(im_id: str, drawings_index: dict[str, Any]) -> bool:
    """Return True if im-id exists in a prefetched drawings index with a non-empty path.

    Pure in-memory check after ``_prefetch_drawings_indexes_by_doc`` (no KV / disk I/O).
    """
    entry = drawings_index.get(im_id)
    if not isinstance(entry, dict):
        return False
    return bool(str(entry.get("path") or "").strip())


async def _whitelist_indexable_drawings(
    ordered_im_ids: list[str],
    im_to_doc: dict[str, str],
    full_docs_db: BaseKVStorage | None,
) -> list[str]:
    """Index JSON filter: prefetch drawings.json, keep Collect-order ids with non-empty path (no disk verify)."""
    if not ordered_im_ids or full_docs_db is None:
        return []

    _, drawings_index_by_doc = await _prefetch_drawings_indexes_by_doc(
        ordered_im_ids, im_to_doc, full_docs_db
    )
    whitelist: list[str] = []
    for im_id in ordered_im_ids:
        doc_id = im_to_doc.get(im_id)
        if not doc_id:
            continue
        drawings_index = drawings_index_by_doc.get(doc_id)
        if not drawings_index:
            continue
        if _is_indexable_in_drawings_index(im_id, drawings_index):
            whitelist.append(im_id)
    return whitelist


async def index_figures_on_chunks(
    truncated_chunks: list[dict[str, Any]],
    text_chunks_db: BaseKVStorage | None,
    full_docs_db: BaseKVStorage | None,
) -> IndexFiguresResult:
    """Collect-once Index: build resolvable im-id whitelist for metadata.

    PR-1 does not attach ``figure_ids`` to chunks and does not write letter
    labels / Figure List / References lines (those are PR-5).

    Input:
        truncated_chunks: Retrieval chunks after token truncation (same set for LLM).
        text_chunks_db: Chunk KV for sidecar records (optional).
        full_docs_db: Full-doc KV for sidecar / drawings.json (required to index).

    Output:
        ``IndexFiguresResult`` with ordered ``whitelist`` only (no chunk copies).

    Steps:
        1. Fetch chunk records by ``chunk_id``.
        2. Collect im-ids via ``collect_im_ids_on_truncated_chunks`` (figure bucket,
           then text bucket; ``on_im_id_found`` mismatch audit in same pass).
        3. ``im_ids_to_doc_map`` — doc routing from im-id embedded hash only.
        4. ``_whitelist_indexable_drawings`` — drawings.json JSON filter (skip only).
        5. Return ``IndexFiguresResult(whitelist=...)``.
    """
    if not truncated_chunks:
        return IndexFiguresResult()

    chunk_ids = [c.get("chunk_id") for c in truncated_chunks if c.get("chunk_id")]
    chunk_records_by_id = await _fetch_chunk_records(text_chunks_db, chunk_ids)

    collected_im_ids = collect_im_ids_on_truncated_chunks(
        truncated_chunks,
        chunk_records_by_id,
        on_im_id_found=_warn_im_id_doc_hash_mismatch,
    )
    im_to_doc = im_ids_to_doc_map(collected_im_ids)
    whitelist = await _whitelist_indexable_drawings(
        collected_im_ids, im_to_doc, full_docs_db
    )
    return IndexFiguresResult(whitelist=whitelist)


def apply_index_figures_to_raw_data(
    raw_data: dict[str, Any],
    index_result: IndexFiguresResult,
) -> None:
    """Write ``metadata.drawing_candidate_whitelist`` in Collect order (in-place; no chunk mutation)."""
    metadata = raw_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        raw_data["metadata"] = metadata
    # Preserve Index collect order for enrich → attachments (PR-1 Related figures UX).
    metadata[DRAWING_CANDIDATE_WHITELIST_KEY] = list(index_result.whitelist)


# ---------------------------------------------------------------------------
# Enrich (whitelist → data.attachments)
#
# LightRAG return hook (PR-1 stage 3): read ``metadata.drawing_candidate_whitelist``
# only (no chunk rescan). Delegates to Resolve helpers below
# (``drawing_candidate_whitelist_from_raw_data``, ``resolve_drawing_attachments``)
# with ``verify_file``. ``_SIDECAR_DEGRADE_ERRORS`` → ``attachments: []``;
# other exceptions propagate. Order preserved (skip only).
# ---------------------------------------------------------------------------


async def enrich_raw_data_attachments(
    raw_data: dict[str, Any],
    full_docs_db: BaseKVStorage | None,
    *,
    verify_file: bool = True,
) -> dict[str, Any]:
    """Enrich ``data.attachments`` from metadata whitelist (resolve-only).

    Operational failures (``_SIDECAR_DEGRADE_ERRORS``) degrade to
    ``attachments: []`` without aborting the query. Programming errors and
    unexpected ``Exception`` subclasses propagate so bugs are not masked.
    Does not scan chunk content for im-ids.
    PR-1 enrich input is ``metadata.drawing_candidate_whitelist`` only
    (no present-mode, no ``include_im_ids`` override; subset selection is PR-4).

    Input:
        raw_data: Query result envelope (mutated and returned).
        full_docs_db: Full-doc KV for ``sidecar_location`` and drawings resolve.
        verify_file: Require on-disk assets when resolving.

    Output:
        The same ``raw_data`` with ``data.attachments`` set to a list (possibly empty).

    Steps:
        1. Read im-ids via ``drawing_candidate_whitelist_from_raw_data``.
        2. ``im_ids_to_doc_map(im_ids)`` — doc routing from im-id embedded hash
           only (no chunk fetch; mismatch logging runs at Index Collect).
        3. ``resolve_drawing_attachments`` — prefetch drawings + verify on disk.
        4. On operational failure during resolve, log warning and set
           ``attachments`` to [] (same ``_SIDECAR_DEGRADE_ERRORS`` tuple as Index).
    """
    if not raw_data:
        return raw_data
    data = raw_data.get("data")
    if not isinstance(data, dict):
        data = {}
        raw_data["data"] = data

    im_ids = drawing_candidate_whitelist_from_raw_data(raw_data)
    if not im_ids or full_docs_db is None:
        data["attachments"] = []
        return raw_data

    im_to_doc = im_ids_to_doc_map(im_ids)

    try:
        # attachments[] order matches whitelist (collect first-seen order).
        data["attachments"] = await resolve_drawing_attachments(
            im_ids, im_to_doc, full_docs_db, verify_file=verify_file
        )
    except _SIDECAR_DEGRADE_ERRORS as exc:
        logger.warning("[query_attachments] enrich failed, degrading to []: %s", exc)
        data["attachments"] = []

    return raw_data


# ---------------------------------------------------------------------------
# Resolve (shared toolbox: im→doc, drawings.json, asset path)
#
# After pipeline stages above — not a stage on its own. Shared by Index and Enrich:
#   Index (above) — ``_prefetch_drawings_indexes_by_doc``, ``load_drawings_index``;
#           called from ``_whitelist_indexable_drawings`` / ``_is_indexable_in_drawings_index``
#           in Index block. JSON layer only (no disk verify).
#   Enrich (above) — ``drawing_candidate_whitelist_from_raw_data``, ``im_ids_to_doc_map``,
#           ``resolve_drawing_attachments`` / ``resolve_single_drawing``;
#           same drawings.json + optional ``verify_file`` on disk.
# ---------------------------------------------------------------------------


def drawing_candidate_whitelist_from_raw_data(raw_data: dict[str, Any]) -> list[str]:
    """Read valid im-ids from ``metadata.drawing_candidate_whitelist`` (order preserved; defensive dedupe)."""
    metadata = raw_data.get("metadata")
    if not isinstance(metadata, dict):
        return []
    raw_list = metadata.get(DRAWING_CANDIDATE_WHITELIST_KEY) or []
    if not isinstance(raw_list, list):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for im_id in raw_list:
        im_id_str = str(im_id)
        if not is_im_id(im_id_str) or im_id_str in seen:
            continue
        seen.add(im_id_str)
        ordered.append(im_id_str)
    return ordered


def im_ids_to_doc_map(im_ids: list[str]) -> dict[str, str]:
    """Pure ``im_id -> doc-…`` map via embedded hash (authoritative for prefetch / resolve)."""
    mapping: dict[str, str] = {}
    for im_id in im_ids:
        doc_id = doc_id_from_im_id(im_id)
        if doc_id:
            mapping[im_id] = doc_id
    return mapping


def _drawings_index_mtime(drawings_path: str | None) -> float | None:
    """Return ``drawings.json`` mtime for cache validation, or None if absent."""
    if not drawings_path:
        return None
    path = Path(drawings_path)
    if not path.is_file():
        return None
    return path.stat().st_mtime


def _drawings_index_cache_lookup(
    sidecar_uri: str, mtime: float | None
) -> dict[str, Any] | None:
    """Return cached index on mtime hit, else ``None``.

    Caller must hold ``_drawings_index_cache_lock``.
    """
    cached = _drawings_index_cache.get(sidecar_uri)
    if cached is None:
        return None
    cached_mtime, cached_index = cached
    if cached_mtime != mtime:
        return None
    _drawings_index_cache.move_to_end(sidecar_uri)
    return cached_index


def _store_drawings_index_cache(
    sidecar_uri: str, mtime: float | None, index: dict[str, Any]
) -> None:
    """Store ``(mtime, index)`` and evict oldest entries when over the LRU cap.

    Caller must hold ``_drawings_index_cache_lock``.
    """
    _drawings_index_cache[sidecar_uri] = (mtime, index)
    _drawings_index_cache.move_to_end(sidecar_uri)
    while len(_drawings_index_cache) > _DRAWINGS_INDEX_CACHE_MAX:
        _drawings_index_cache.popitem(last=False)


def _read_drawings_index_from_disk(
    drawings_path: str | None,
    mtime: float | None,
) -> tuple[float | None, dict[str, Any]]:
    """Load the ``drawings`` map from disk without touching the process cache."""
    if not drawings_path:
        return None, {}

    path = Path(drawings_path)
    if not path.is_file():
        logger.debug("[query_attachments] drawings.json missing: %s", drawings_path)
        return None, {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[query_attachments] Failed to read drawings index %s: %s",
            drawings_path,
            exc,
        )
        return mtime, {}

    drawings = payload.get("drawings") if isinstance(payload, dict) else None
    index = drawings if isinstance(drawings, dict) else {}
    return mtime, index


def load_drawings_index(sidecar_uri: str | None) -> dict[str, Any]:
    """Load and LRU-cache ``drawings.json`` ``drawings`` map (thread-safe; ``{}`` on failure)."""
    if not sidecar_uri:
        return {}

    drawings_path = sidecar_modality_path(sidecar_uri, "drawings")
    mtime = _drawings_index_mtime(drawings_path)

    with _drawings_index_cache_lock:
        hit = _drawings_index_cache_lookup(sidecar_uri, mtime)
        if hit is not None:
            return hit

    store_mtime, index = _read_drawings_index_from_disk(drawings_path, mtime)

    with _drawings_index_cache_lock:
        hit = _drawings_index_cache_lookup(sidecar_uri, mtime)
        if hit is not None:
            return hit
        _store_drawings_index_cache(sidecar_uri, store_mtime, index)
        return index


async def aload_drawings_index(sidecar_uri: str | None) -> dict[str, Any]:
    """Async wrapper: load drawings index without blocking the event loop.

    ``load_drawings_index`` uses blocking ``read_text()``; call this from
    ``index_figures_on_chunks`` / ``resolve_drawing_attachments``.
    """
    return await asyncio.to_thread(load_drawings_index, sidecar_uri)


def clear_drawings_index_cache() -> None:
    """Clear the process-global drawings index cache (mainly for tests)."""
    with _drawings_index_cache_lock:
        _drawings_index_cache.clear()


def resolve_asset_path(sidecar_uri: str | None, relative_path: str) -> Path | None:
    """Resolve a drawings.json relative path under the sidecar root (path-traversal safe).

    Delegates to :func:`validate_file_path_security` so embedded NUL / other
    malformed paths become ``None`` instead of raising ``ValueError`` from
    ``Path.resolve()`` (which Enrich would fail-loud on). Callers treat
    ``None`` as skip — same as a missing jpg.
    """
    root = resolve_sidecar_uri(sidecar_uri)
    if root is None or not relative_path:
        return None

    candidate = validate_file_path_security(relative_path, root)
    if candidate is not None and candidate.is_file():
        return candidate

    # Basename fallback under ``*.blocks.assets`` (same layout Index/Enrich use).
    assets_dir = sidecar_assets_dir_for_uri(sidecar_uri)
    if assets_dir is None:
        return None
    nested = validate_file_path_security(Path(relative_path).name, assets_dir)
    if nested is not None and nested.is_file():
        return nested
    return None


async def aresolve_asset_path(
    sidecar_uri: str | None, relative_path: str
) -> Path | None:
    """Async wrapper: verify asset path without blocking the event loop.

    ``resolve_asset_path`` uses blocking ``stat`` / ``is_file`` / ``glob``; call
    this from ``resolve_single_drawing`` on the enrich path.
    """
    return await asyncio.to_thread(resolve_asset_path, sidecar_uri, relative_path)


async def resolve_single_drawing(
    *,
    im_id: str,
    doc_id: str,
    sidecar_uri: str | None,
    drawings_index: dict[str, Any],
    verify_file: bool = True,
) -> AttachmentDict | None:
    """Build one attachment dict from a drawings entry; None if missing or ``verify_file`` fails."""
    entry = drawings_index.get(im_id)
    if not isinstance(entry, dict):
        return None

    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        return None

    if verify_file and await aresolve_asset_path(sidecar_uri, relative_path) is None:
        return None

    attachment: AttachmentDict = {
        "im_id": im_id,
        "doc_id": doc_id,
        "type": "drawing",
        "format": str(entry.get("format") or "jpg"),
        "path": relative_path,
    }
    heading = entry.get("heading")
    if heading:
        attachment["heading"] = str(heading)
    analysis = entry.get("llm_analyze_result")
    if isinstance(analysis, dict):
        title = analysis.get("name")
        if title:
            attachment["title"] = str(title)
    return attachment


async def _fetch_chunk_records(
    text_chunks_db: BaseKVStorage | None,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any] | None]:
    """Batch-load chunk KV records by ``chunk_id`` (empty dict on missing storage / errors)."""
    if not text_chunks_db or not chunk_ids:
        return {}
    unique_ids = list(dict.fromkeys(chunk_ids))
    try:
        records = await text_chunks_db.get_by_ids(unique_ids)
    except _SIDECAR_DEGRADE_ERRORS as exc:
        logger.warning("[query_attachments] chunk record fetch failed: %s", exc)
        return {}
    by_id: dict[str, dict[str, Any] | None] = {cid: None for cid in unique_ids}
    for record in records:
        if not record:
            continue
        cid = record.get("chunk_id") or record.get("_id") or record.get("id")
        if cid:
            by_id[str(cid)] = record
    return by_id


async def _prefetch_drawings_indexes_by_doc(
    ordered_im_ids: list[str],
    im_to_doc: dict[str, str],
    full_docs_db: BaseKVStorage,
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    """Prefetch ``full_doc`` + drawings index once per unique doc (first-seen im-id order)."""
    doc_ids_in_order: list[str] = []
    seen_doc: set[str] = set()
    for im_id in ordered_im_ids:
        doc_id = im_to_doc.get(im_id)
        if not doc_id or doc_id in seen_doc:
            continue
        seen_doc.add(doc_id)
        doc_ids_in_order.append(doc_id)

    sidecar_uri_by_doc: dict[str, str | None] = {}
    drawings_index_by_doc: dict[str, dict[str, Any]] = {}
    for doc_id in doc_ids_in_order:
        try:
            doc_record = await full_docs_db.get_by_id(doc_id)
        except _SIDECAR_DEGRADE_ERRORS as exc:
            logger.warning(
                "[query_attachments] full_doc fetch failed for %s: %s", doc_id, exc
            )
            continue
        if not doc_record:
            continue
        sidecar_uri = doc_record.get("sidecar_location")
        try:
            drawings_index = await aload_drawings_index(sidecar_uri)
        except _SIDECAR_DEGRADE_ERRORS as exc:
            logger.warning(
                "[query_attachments] drawings index load failed for %s: %s",
                doc_id,
                exc,
            )
            continue
        if not drawings_index:
            continue
        sidecar_uri_by_doc[doc_id] = sidecar_uri
        drawings_index_by_doc[doc_id] = drawings_index

    return sidecar_uri_by_doc, drawings_index_by_doc


async def resolve_drawing_attachments(
    im_ids: list[str],
    im_to_doc: dict[str, str],
    full_docs_db: BaseKVStorage,
    *,
    verify_file: bool = True,
) -> list[AttachmentDict]:
    """Resolve im-ids into attachment dicts via full_docs + drawings.json.

    Input:
        im_ids: Drawing ids in collect order (whitelist order).
        im_to_doc: im_id -> doc_id map.
        full_docs_db: Full-doc KV for ``sidecar_location``.
        verify_file: Pass-through to ``resolve_single_drawing``.

    Output:
        List of attachment dicts in the same order as ``im_ids`` (skips
        unresolvable items without reordering survivors).

    Steps:
        1. Prefetch full_doc + drawings index per document (I/O batching).
        2. Walk ``im_ids`` in input order; resolve each with ``resolve_single_drawing``.
    """
    if not im_ids or not im_to_doc:
        return []

    sidecar_uri_by_doc, drawings_index_by_doc = await _prefetch_drawings_indexes_by_doc(
        im_ids, im_to_doc, full_docs_db
    )

    # Output follows whitelist / collect order, not doc_id or im_id sort order.
    attachments: list[AttachmentDict] = []
    for im_id in im_ids:
        doc_id = im_to_doc.get(im_id)
        if not doc_id:
            continue
        drawings_index = drawings_index_by_doc.get(doc_id)
        if not drawings_index:
            continue
        item = await resolve_single_drawing(
            im_id=im_id,
            doc_id=doc_id,
            sidecar_uri=sidecar_uri_by_doc.get(doc_id),
            drawings_index=drawings_index,
            verify_file=verify_file,
        )
        if item:
            attachments.append(item)

    return attachments
