"""Query-time sidecar join: collect drawing ``im-id`` values from retrieval
``raw_data`` and resolve them to on-disk asset paths via ``drawings.json``.

PR-1~3: scan retrieval ``raw_data`` and enrich ``data.attachments`` (full set).

PR-4 adds candidate menus for Channel A (``format_drawing_menu_for_prompt``),
``<present_drawings>`` parsing, and subset resolution via ``include_im_ids``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.chunk_schema import normalize_chunk_sidecar
from lightrag.utils import get_env_value, logger
from lightrag.utils_pipeline import (
    resolve_sidecar_uri,
    sidecar_assets_dir_for_uri,
    sidecar_modality_path,
)

if TYPE_CHECKING:
    from lightrag.base import BaseKVStorage

# im-{doc_hash}-{seq}, e.g. im-116da71bde652ed78cf28934d0712aa6-0002
IM_ID_RE = re.compile(r"^im-(?P<doc_hash>[a-f0-9]+)-(?P<seq>\d+)$")
DRAWING_TAG_RE = re.compile(
    r'<drawing\b[^>]*\bid="(im-[^"]+)"',
    flags=re.IGNORECASE,
)
PRESENT_DRAWINGS_TAG_RE = re.compile(
    r"<present_drawings>\s*([^<]*?)\s*</present_drawings>",
    flags=re.IGNORECASE | re.DOTALL,
)

AttachmentDict = dict[str, Any]
DrawingCandidateDict = dict[str, Any]

DEFAULT_ATTACHMENT_CANDIDATE_TOP_K = 8

PRESENT_DRAWINGS_METADATA_KEY = "present_drawings"
DRAWING_CANDIDATE_WHITELIST_KEY = "drawing_candidate_whitelist"
DEFERRED_ATTACHMENT_ENRICH_KEY = "deferred_attachment_enrich"

_PRESENT_DRAWINGS_OPEN = "<present_drawings>"
_STREAM_HOLDBACK_CHARS = len(_PRESENT_DRAWINGS_OPEN) + 40

# Per-sidecar-uri cache for drawings.json payloads (process-local, read-only).
_drawings_index_cache: dict[str, dict[str, Any]] = {}


def is_im_id(value: str | None) -> bool:
    """Return True when *value* matches the canonical drawing id form."""
    if not value or not isinstance(value, str):
        return False
    return IM_ID_RE.match(value.strip()) is not None


def doc_id_from_im_id(im_id: str) -> str | None:
    """Derive ``doc-{hash}`` from ``im-{hash}-{seq}``."""
    match = IM_ID_RE.match(im_id.strip())
    if not match:
        return None
    return f"doc-{match.group('doc_hash')}"


def doc_id_from_chunk_id(chunk_id: str | None) -> str | None:
    """Parse ``doc-{hash}-…`` prefix from a chunk id."""
    if not chunk_id or not isinstance(chunk_id, str):
        return None
    if chunk_id.startswith("doc-"):
        rest = chunk_id[4:]
        dash = rest.find("-")
        if dash == -1:
            return chunk_id
        return f"doc-{rest[:dash]}"
    return None


def collect_im_ids_from_entities(entities: list[dict[str, Any]]) -> set[str]:
    """Collect drawing ids from KG entity names."""
    found: set[str] = set()
    for entity in entities:
        name = entity.get("entity_name") or entity.get("entity") or ""
        if is_im_id(name):
            found.add(name.strip())
    return found


def collect_im_ids_from_chunk_content(chunks: list[dict[str, Any]]) -> set[str]:
    """Collect drawing ids embedded in chunk ``content`` as ``<drawing …/>`` tags."""
    found: set[str] = set()
    for chunk in chunks:
        content = chunk.get("content") or ""
        if not content:
            continue
        for match in DRAWING_TAG_RE.finditer(content):
            im_id = match.group(1).strip()
            if is_im_id(im_id):
                found.add(im_id)
    return found


def collect_im_ids_from_sidecar_records(
    chunk_records: list[dict[str, Any] | None],
) -> set[str]:
    """Collect drawing ids from persisted chunk ``sidecar`` metadata."""
    found: set[str] = set()
    for record in chunk_records:
        if not record:
            continue
        sidecar = normalize_chunk_sidecar(record)
        if not sidecar:
            continue
        for ref in sidecar.get("refs") or []:
            if ref.get("type") == "drawing" and is_im_id(ref.get("id")):
                found.add(ref["id"])
        if sidecar.get("type") == "drawing" and is_im_id(sidecar.get("id")):
            found.add(sidecar["id"])
    return found


def collect_im_ids(
    raw_data: dict[str, Any],
    chunk_records_by_id: dict[str, dict[str, Any] | None] | None = None,
) -> set[str]:
    """Gather all drawing im-ids referenced by the current retrieval ``raw_data``."""
    data = raw_data.get("data") or {}
    entities = data.get("entities") or []
    chunks = data.get("chunks") or []

    found = collect_im_ids_from_entities(entities)
    found.update(collect_im_ids_from_chunk_content(chunks))

    if chunk_records_by_id:
        records = [
            chunk_records_by_id.get(chunk.get("chunk_id") or "")
            for chunk in chunks
            if chunk.get("chunk_id")
        ]
        found.update(collect_im_ids_from_sidecar_records(records))

    return found


def collect_im_ids_ordered(
    raw_data: dict[str, Any],
    chunk_records_by_id: dict[str, dict[str, Any] | None] | None = None,
) -> list[str]:
    """Gather drawing im-ids in chunk appearance order (deduped).

    Chunk ``<drawing>`` tags come first (retrieval order), then entity names,
    then sidecar refs on persisted chunk records.
    """
    data = raw_data.get("data") or {}
    chunks = data.get("chunks") or []
    ordered: list[str] = []
    seen: set[str] = set()

    def _append(im_id: str) -> None:
        im_id = im_id.strip()
        if is_im_id(im_id) and im_id not in seen:
            seen.add(im_id)
            ordered.append(im_id)

    for chunk in chunks:
        content = chunk.get("content") or ""
        for match in DRAWING_TAG_RE.finditer(content):
            _append(match.group(1))

    for entity in data.get("entities") or []:
        name = entity.get("entity_name") or entity.get("entity") or ""
        if is_im_id(name):
            _append(name)

    if chunk_records_by_id:
        records = [
            chunk_records_by_id.get(chunk.get("chunk_id") or "")
            for chunk in chunks
            if chunk.get("chunk_id")
        ]
        for record in records:
            if not record:
                continue
            sidecar = normalize_chunk_sidecar(record)
            if not sidecar:
                continue
            for ref in sidecar.get("refs") or []:
                if ref.get("type") == "drawing" and is_im_id(ref.get("id")):
                    _append(ref["id"])
            if sidecar.get("type") == "drawing" and is_im_id(sidecar.get("id")):
                _append(sidecar["id"])

    return ordered


def parse_present_drawings(text: str) -> tuple[str, list[str]]:
    """Parse ``<present_drawings>…</present_drawings>`` from LLM output.

    Returns ``(clean_markdown, selected_im_ids)``. When the tag is missing or
    empty, ``selected_im_ids`` is ``[]``. Invalid token shapes are dropped.
    """
    if not text:
        return "", []

    match = PRESENT_DRAWINGS_TAG_RE.search(text)
    if not match:
        return text.strip(), []

    inner = match.group(1).strip()
    selected: list[str] = []
    if inner:
        for token in inner.split(","):
            im_id = token.strip()
            if is_im_id(im_id):
                selected.append(im_id)

    clean = PRESENT_DRAWINGS_TAG_RE.sub("", text).strip()
    return clean, selected


def filter_im_ids_by_whitelist(
    selected: list[str],
    allowed: set[str],
) -> list[str]:
    """Keep *selected* im-ids that appear in *allowed*, preserving order."""
    if not selected or not allowed:
        return []
    seen: set[str] = set()
    filtered: list[str] = []
    for im_id in selected:
        if im_id in allowed and im_id not in seen:
            seen.add(im_id)
            filtered.append(im_id)
    return filtered


def format_drawing_menu_for_prompt(
    candidates: list[DrawingCandidateDict],
) -> str:
    """Format a lean drawing menu appendix for the query LLM prompt."""
    lines = [
        "## Optional figures (select only from the ids below; do not invent ids)",
        "",
    ]
    if not candidates:
        lines.extend(
            [
                "(No drawing candidates in this retrieval result.)",
                "",
                "Output at the end of your answer:",
                "<present_drawings></present_drawings>",
            ]
        )
        return "\n".join(lines)

    for item in candidates:
        im_id = item.get("im_id") or ""
        title = str(item.get("title") or im_id).strip()
        lines.append(f"- {im_id} | {title}")

    lines.extend(
        [
            "",
            "If the user needs original figures (form/dimensions/inspection diagrams), "
            "end your answer with:",
            "<present_drawings>im-…,im-…</present_drawings>",
            "If no figures should be shown (e.g. bibliography / overview only), end with:",
            "<present_drawings></present_drawings>",
        ]
    )
    return "\n".join(lines)


def _candidate_from_drawing_entry(
    im_id: str,
    doc_id: str,
    entry: dict[str, Any],
) -> DrawingCandidateDict:
    candidate: DrawingCandidateDict = {"im_id": im_id, "doc_id": doc_id}
    heading = entry.get("heading")
    if heading:
        candidate["heading"] = str(heading)
    analysis = entry.get("llm_analyze_result")
    if isinstance(analysis, dict):
        title = analysis.get("name")
        if title:
            candidate["title"] = str(title)
    if "title" not in candidate:
        candidate["title"] = im_id
    return candidate


async def build_drawing_candidates(
    raw_data: dict[str, Any],
    text_chunks_db: BaseKVStorage | None,
    full_docs_db: BaseKVStorage | None,
    *,
    top_k: int = DEFAULT_ATTACHMENT_CANDIDATE_TOP_K,
) -> tuple[list[DrawingCandidateDict], set[str]]:
    """Build ordered drawing candidates (metadata only) for PR-4 prompt menus.

    Returns ``(candidates, whitelist)`` where *whitelist* is the set of im-ids
    that may be selected for presentation. Uses ``verify_file=False`` so
    missing assets still appear in the menu (Channel B skips them later).
    """
    if not raw_data or full_docs_db is None:
        return [], set()

    if "data" not in raw_data or not isinstance(raw_data["data"], dict):
        return [], set()

    data = raw_data["data"]
    chunks = data.get("chunks") or []
    chunk_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    chunk_records_by_id = await _fetch_chunk_records(text_chunks_db, chunk_ids)

    ordered_ids = collect_im_ids_ordered(raw_data, chunk_records_by_id)
    if not ordered_ids:
        return [], set()

    if top_k > 0:
        ordered_ids = ordered_ids[:top_k]

    im_to_doc = build_im_to_doc_map(set(ordered_ids), chunk_records_by_id, chunks)

    by_doc: dict[str, list[str]] = {}
    for im_id in ordered_ids:
        doc_id = im_to_doc.get(im_id) or doc_id_from_im_id(im_id)
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(im_id)

    candidates: list[DrawingCandidateDict] = []
    whitelist: set[str] = set()

    for doc_id in sorted(by_doc.keys()):
        doc_record = await full_docs_db.get_by_id(doc_id)
        if not doc_record:
            continue
        sidecar_uri = doc_record.get("sidecar_location")
        drawings_index = load_drawings_index(sidecar_uri)
        if not drawings_index:
            continue
        for im_id in by_doc[doc_id]:
            entry = drawings_index.get(im_id)
            if not isinstance(entry, dict):
                continue
            # Metadata-only: do not require jpg on disk for the menu.
            if not str(entry.get("path") or "").strip():
                continue
            candidate = _candidate_from_drawing_entry(im_id, doc_id, entry)
            candidates.append(candidate)
            whitelist.add(im_id)

    # Preserve chunk order among resolved candidates.
    order_index = {im_id: idx for idx, im_id in enumerate(ordered_ids)}
    candidates.sort(key=lambda c: order_index.get(c["im_id"], len(order_index)))
    return candidates, whitelist


def get_attachment_present_mode(global_config: dict[str, Any] | None = None) -> str:
    """Return ``llm_select`` or ``off`` (PR-4 presentation mode)."""
    if global_config is not None:
        mode = global_config.get("attachment_present_mode")
        if mode is not None:
            return str(mode).lower()
    return get_env_value("ATTACHMENT_PRESENT_MODE", "off", str).lower()


def get_attachment_candidate_top_k(global_config: dict[str, Any] | None = None) -> int:
    if global_config is not None:
        top_k = global_config.get("attachment_candidate_top_k")
        if top_k is not None:
            return int(top_k)
    return get_env_value(
        "ATTACHMENT_CANDIDATE_TOP_K",
        DEFAULT_ATTACHMENT_CANDIDATE_TOP_K,
        int,
    )


def is_llm_select_present_mode(global_config: dict[str, Any] | None = None) -> bool:
    return get_attachment_present_mode(global_config) == "llm_select"


def drawing_candidate_whitelist_from_raw_data(raw_data: dict[str, Any]) -> set[str]:
    metadata = raw_data.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    raw_list = metadata.get(DRAWING_CANDIDATE_WHITELIST_KEY) or []
    if not isinstance(raw_list, list):
        return set()
    return {im_id for im_id in raw_list if is_im_id(str(im_id))}


async def append_drawing_menu_to_sys_prompt(
    sys_prompt: str,
    raw_data: dict[str, Any],
    text_chunks_db: BaseKVStorage | None,
    full_docs_db: BaseKVStorage | None,
    global_config: dict[str, Any] | None,
) -> str:
    """Append the PR-4 lean drawing menu when ``llm_select`` mode is active."""
    if not is_llm_select_present_mode(global_config):
        return sys_prompt

    top_k = get_attachment_candidate_top_k(global_config)
    candidates, whitelist = await build_drawing_candidates(
        raw_data,
        text_chunks_db,
        full_docs_db,
        top_k=top_k,
    )
    metadata = raw_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        raw_data["metadata"] = metadata
    metadata[DRAWING_CANDIDATE_WHITELIST_KEY] = sorted(whitelist)
    menu = format_drawing_menu_for_prompt(candidates)
    return f"{sys_prompt}\n\n{menu}"


def apply_present_drawings_to_raw_data(
    raw_data: dict[str, Any],
    llm_text: str,
    *,
    whitelist: set[str] | None = None,
) -> str:
    """Parse ``<present_drawings>``, store selection in metadata, return clean text."""
    allowed = whitelist if whitelist is not None else drawing_candidate_whitelist_from_raw_data(
        raw_data
    )
    clean, selected = parse_present_drawings(llm_text)
    filtered = filter_im_ids_by_whitelist(selected, allowed)
    metadata = raw_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        raw_data["metadata"] = metadata
    metadata[PRESENT_DRAWINGS_METADATA_KEY] = filtered

    # Tags are stripped above; return the fully cleaned assistant text for callers.
    return clean

    


def resolve_include_im_ids_for_enrich(
    raw_data: dict[str, Any],
    global_config: dict[str, Any] | None,
) -> set[str] | None:
    """Return ``None`` for v1.0 full enrich, or a (possibly empty) subset for PR-4."""
    if not is_llm_select_present_mode(global_config):
        return None

    metadata = raw_data.get("metadata")
    if not isinstance(metadata, dict):
        return set()

    if PRESENT_DRAWINGS_METADATA_KEY not in metadata:
        # Streaming: LLM output not finalized yet.
        return set()

    selected = metadata.get(PRESENT_DRAWINGS_METADATA_KEY) or []
    if not isinstance(selected, list):
        return set()

    whitelist = drawing_candidate_whitelist_from_raw_data(raw_data)
    return set(filter_im_ids_by_whitelist([str(x) for x in selected], whitelist))


class PresentDrawingsStreamWrapper:
    """Wrap a streaming LLM iterator and strip trailing ``<present_drawings>`` tags.

    After the stream completes, call :meth:`finalize_attachments` to parse the
    full text, update ``raw_data`` metadata, and resolve sidecar attachments.
    """

    def __init__(
        self,
        source: AsyncIterator[str],
        *,
        raw_data: dict[str, Any],
        whitelist: set[str],
    ) -> None:
        self._source = source
        self._raw_data = raw_data
        self._whitelist = whitelist
        self._full_text = ""

    def __aiter__(self) -> AsyncIterator[str]:
        return self._stream_chunks()

    async def _stream_chunks(self) -> AsyncIterator[str]:
        pending = ""
        open_tag = _PRESENT_DRAWINGS_OPEN.lower()
        async for chunk in self._source:
            if not chunk:
                continue
            self._full_text += chunk
            pending += chunk
            tag_idx = pending.lower().find(open_tag)
            if tag_idx >= 0:
                emit = pending[:tag_idx]
                pending = pending[tag_idx:]
                if emit:
                    yield emit
            elif len(pending) > _STREAM_HOLDBACK_CHARS:
                emit = pending[:-_STREAM_HOLDBACK_CHARS]
                pending = pending[-_STREAM_HOLDBACK_CHARS:]
                if emit:
                    yield emit
        clean_tail, _ = parse_present_drawings(pending)
        if clean_tail:
            # Flush the final holdback buffer after stripping any trailing present_drawings tag.
            yield clean_tail

    async def finalize_attachments(
        self,
        text_chunks_db: BaseKVStorage | None,
        full_docs_db: BaseKVStorage | None,
        global_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Parse the accumulated stream and enrich ``raw_data`` attachments."""
        apply_present_drawings_to_raw_data(
            self._raw_data,
            self._full_text,
            whitelist=self._whitelist,
        )
        metadata = self._raw_data.get("metadata")
        if isinstance(metadata, dict):
            metadata[DEFERRED_ATTACHMENT_ENRICH_KEY] = False

        include_im_ids = resolve_include_im_ids_for_enrich(
            self._raw_data, global_config
        )
        return await enrich_raw_data_attachments(
            self._raw_data,
            text_chunks_db,
            full_docs_db,
            include_im_ids=include_im_ids,
        )


def wrap_present_drawings_stream(
    source: AsyncIterator[str],
    *,
    raw_data: dict[str, Any],
    whitelist: set[str],
) -> PresentDrawingsStreamWrapper:
    """Return a stream wrapper for PR-4 deferred attachment resolution."""
    return PresentDrawingsStreamWrapper(
        source,
        raw_data=raw_data,
        whitelist=whitelist,
    )


def is_present_drawings_stream_wrapper(
    value: object,
) -> bool:
    return isinstance(value, PresentDrawingsStreamWrapper)


def build_im_to_doc_map(
    im_ids: set[str],
    chunk_records_by_id: dict[str, dict[str, Any] | None] | None = None,
    raw_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Map each im-id to a ``doc_id`` for sidecar lookup."""
    mapping: dict[str, str] = {}

    for im_id in im_ids:
        doc_id = doc_id_from_im_id(im_id)
        if doc_id:
            mapping.setdefault(im_id, doc_id)

    if chunk_records_by_id and raw_chunks:
        for chunk in raw_chunks:
            chunk_id = chunk.get("chunk_id") or ""
            record = chunk_records_by_id.get(chunk_id)
            doc_id = None
            if record:
                doc_id = record.get("full_doc_id") or doc_id_from_chunk_id(chunk_id)
            else:
                doc_id = doc_id_from_chunk_id(chunk_id)
            if not doc_id:
                continue
            content = chunk.get("content") or ""
            for match in DRAWING_TAG_RE.finditer(content):
                im_id = match.group(1).strip()
                if is_im_id(im_id):
                    mapping.setdefault(im_id, doc_id)
            sidecar = normalize_chunk_sidecar(record) if record else None
            if sidecar and sidecar.get("type") == "drawing" and is_im_id(sidecar.get("id")):
                mapping.setdefault(sidecar["id"], doc_id)

    return mapping


def load_drawings_index(sidecar_uri: str | None) -> dict[str, Any]:
    """Load ``drawings.json`` for a sidecar URI (cached per URI string)."""
    if not sidecar_uri:
        return {}
    cached = _drawings_index_cache.get(sidecar_uri)
    if cached is not None:
        return cached

    drawings_path = sidecar_modality_path(sidecar_uri, "drawings")
    if not drawings_path:
        _drawings_index_cache[sidecar_uri] = {}
        return {}

    path = Path(drawings_path)
    if not path.is_file():
        logger.debug("[query_attachments] drawings.json missing: %s", drawings_path)
        _drawings_index_cache[sidecar_uri] = {}
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[query_attachments] Failed to read drawings index %s: %s",
            drawings_path,
            exc,
        )
        _drawings_index_cache[sidecar_uri] = {}
        return {}

    drawings = payload.get("drawings") if isinstance(payload, dict) else None
    index = drawings if isinstance(drawings, dict) else {}
    _drawings_index_cache[sidecar_uri] = index
    return index


def clear_drawings_index_cache() -> None:
    """Clear the process-local drawings index cache (for tests)."""
    _drawings_index_cache.clear()


def _path_within_root(candidate: Path, root: Path) -> bool:
    """Return True when *candidate* resolves under *root* (no path escape)."""
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_asset_path(sidecar_uri: str | None, relative_path: str) -> Path | None:
    """Resolve a drawings.json ``path`` field to an on-disk file."""
    root = resolve_sidecar_uri(sidecar_uri)
    if root is None or not relative_path:
        return None
    root_resolved = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate.is_file() and _path_within_root(candidate, root_resolved):
        return candidate
    assets_dir = sidecar_assets_dir_for_uri(sidecar_uri)
    if assets_dir is not None:
        nested = (assets_dir / Path(relative_path).name).resolve()
        if nested.is_file() and _path_within_root(nested, root_resolved):
            return nested
    return None


_DRAWING_MEDIA_TYPES: dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
}


def media_type_for_drawing_format(fmt: str | None) -> str:
    """Map a sidecar drawing ``format`` field to an HTTP media type."""
    if not fmt:
        return "image/jpeg"
    return _DRAWING_MEDIA_TYPES.get(str(fmt).lower(), "application/octet-stream")


async def resolve_drawing_asset_file(
    doc_id: str,
    im_id: str,
    full_docs_db: BaseKVStorage,
) -> tuple[Path, str] | None:
    """Resolve ``(doc_id, im_id)`` to an on-disk asset path and media type."""
    if not is_im_id(im_id) or not doc_id:
        return None

    doc_record = await full_docs_db.get_by_id(doc_id)
    if not doc_record:
        return None

    sidecar_uri = doc_record.get("sidecar_location")
    drawings_index = load_drawings_index(sidecar_uri)
    if not drawings_index:
        return None

    attachment = resolve_single_drawing(
        im_id=im_id,
        doc_id=doc_id,
        sidecar_uri=sidecar_uri,
        drawings_index=drawings_index,
        verify_file=True,
    )
    if not attachment:
        return None

    asset_path = resolve_asset_path(sidecar_uri, str(attachment.get("path") or ""))
    if asset_path is None:
        return None

    return asset_path, media_type_for_drawing_format(attachment.get("format"))


def resolve_single_drawing(
    *,
    im_id: str,
    doc_id: str,
    sidecar_uri: str | None,
    drawings_index: dict[str, Any],
    verify_file: bool = True,
) -> AttachmentDict | None:
    """Resolve one ``(doc_id, im_id)`` pair to an attachment dict."""
    entry = drawings_index.get(im_id)
    if not isinstance(entry, dict):
        logger.debug(
            "[query_attachments] im-id %s not found in drawings index for %s",
            im_id,
            doc_id,
        )
        return None

    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        return None

    if verify_file:
        asset_path = resolve_asset_path(sidecar_uri, relative_path)
        if asset_path is None:
            logger.debug(
                "[query_attachments] asset missing for %s: %s",
                im_id,
                relative_path,
            )
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
    """Batch-load text chunk records keyed by chunk id."""
    if not text_chunks_db or not chunk_ids:
        return {}
    unique_ids = list(dict.fromkeys(chunk_ids))
    records = await text_chunks_db.get_by_ids(unique_ids)
    by_id: dict[str, dict[str, Any] | None] = {cid: None for cid in unique_ids}
    for record in records:
        if not record:
            continue
        cid = record.get("chunk_id") or record.get("_id") or record.get("id")
        if cid:
            by_id[str(cid)] = record
    return by_id


async def resolve_drawing_attachments(
    im_ids: set[str],
    im_to_doc: dict[str, str],
    full_docs_db: BaseKVStorage,
    *,
    verify_file: bool = True,
) -> list[AttachmentDict]:
    """Resolve im-ids to attachment metadata via ``full_docs`` sidecar locations."""
    if not im_ids or not im_to_doc:
        return []

    # Group im-ids by doc_id so we load each sidecar once.
    by_doc: dict[str, list[str]] = {}
    for im_id in sorted(im_ids):
        doc_id = im_to_doc.get(im_id) or doc_id_from_im_id(im_id)
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(im_id)

    attachments: list[AttachmentDict] = []
    for doc_id in sorted(by_doc.keys()):
        doc_record = await full_docs_db.get_by_id(doc_id)
        if not doc_record:
            logger.debug("[query_attachments] full_doc missing: %s", doc_id)
            continue
        sidecar_uri = doc_record.get("sidecar_location")
        drawings_index = load_drawings_index(sidecar_uri)
        if not drawings_index:
            continue
        for im_id in by_doc[doc_id]:
            item = resolve_single_drawing(
                im_id=im_id,
                doc_id=doc_id,
                sidecar_uri=sidecar_uri,
                drawings_index=drawings_index,
                verify_file=verify_file,
            )
            if item:
                attachments.append(item)

    return attachments


async def enrich_raw_data_attachments(
    raw_data: dict[str, Any],
    text_chunks_db: BaseKVStorage | None,
    full_docs_db: BaseKVStorage | None,
    *,
    verify_file: bool = True,
    include_im_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Enrich ``raw_data['data']['attachments']`` from the current retrieval result.

    When *include_im_ids* is ``None`` (default), resolve every im-id found in the
    retrieval result (PR-1~3 behaviour). When set, only those ids are resolved
    (PR-4: after ``<present_drawings>`` parsing). An empty set yields no
    attachments.

    Never raises — failures degrade to an empty attachment list so query paths
    remain unaffected.
    """
    if not raw_data:
        return raw_data
    if "data" not in raw_data or not isinstance(raw_data["data"], dict):
        raw_data["data"] = {}

    data = raw_data["data"]
    chunks = data.get("chunks") or []
    chunk_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]

    chunk_records_by_id = await _fetch_chunk_records(text_chunks_db, chunk_ids)
    im_ids = collect_im_ids(raw_data, chunk_records_by_id)
    if include_im_ids is not None:
        im_ids = {im_id for im_id in include_im_ids if is_im_id(im_id)}

    if not im_ids:
        data["attachments"] = []
        return raw_data

    if full_docs_db is None:
        data["attachments"] = []
        return raw_data

    im_to_doc = build_im_to_doc_map(im_ids, chunk_records_by_id, chunks)
    data["attachments"] = await resolve_drawing_attachments(
        im_ids,
        im_to_doc,
        full_docs_db,
        verify_file=verify_file,
    )
    return raw_data
