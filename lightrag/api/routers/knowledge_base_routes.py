


"""Knowledge-base (multi-tenant workspace) routes.

Each knowledge base (KB) is a distinct LightRAG ``workspace`` namespace over the
same storage backends.  A KB is materialized the first time it is touched: the
workspace's storage is initialized lazily and cached for the process lifetime.

All KB document operations (upload, paginated list, delete) are routed through
the *same* document module entry points used by the global ``/documents``
endpoints (``pipeline_index_file`` / ``background_delete_documents``), so the
pipeline contract — admission, dedup, parser routing, status transitions —
behaves identically for KB-scoped and global traffic.

Endpoints
---------
* ``GET    /knowledge_bases``                              list all KBs
* ``POST   /knowledge_bases``                              create a KB
* ``GET    /knowledge_bases/{kb_id}``                      KB detail
* ``DELETE /knowledge_bases/{kb_id}``                      delete a KB (drops its data)
* ``GET    /knowledge_bases/{kb_id}/documents``            paginated document list
* ``POST   /knowledge_bases/{kb_id}/documents/upload``     upload + ingest a file
* ``POST   /knowledge_bases/{kb_id}/documents/delete``     delete documents in this KB
* ``GET    /knowledge_bases/{kb_id}/task_history``         recent ingestion history
"""

import asyncio
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from lightrag import LightRAG
from lightrag.api.routers.document_routes import (
    _acquire_destructive_busy,
    _release_destructive_busy,
    background_delete_documents,
    DocumentManager,
    FilenameParserHintError,
    find_existing_file_by_file_path,
    get_existing_doc_by_file_path_candidates,
    get_doc_status_value,
    get_managed_background_tasks,
    InsertResponse,
    pipeline_index_file,
    sanitize_filename,
    UNKNOWN_FILE_SOURCE,
)
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    internal_server_error,
)
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    start_reserved_background_task,
)
from lightrag.parser.routing import resolve_parser_directives
from lightrag.utils import generate_track_id, logger
from ..config import global_args

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge_bases", tags=["Knowledge Bases"])

# ---------------------------------------------------------------------------
# Knowledge-base display metadata (names).
#
# A KB's workspace id is its storage isolation key and cannot be renamed
# without migrating every backend, so the user-visible name is stored as
# lightweight metadata next to the input directory. ``name`` defaults to the
# workspace id when no metadata exists.
# ---------------------------------------------------------------------------
_KB_META_LOCK = asyncio.Lock()
_KB_META_FILENAME = ".kb_meta.json"


def _kb_meta_path() -> Path:
    return _input_base() / _KB_META_FILENAME


async def _read_kb_meta() -> Dict[str, Dict[str, str]]:
    path = _kb_meta_path()
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {
                    str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)
                }
    except Exception as exc:  # noqa: BLE001 - metadata must never break the API
        logger.warning("Failed to read KB metadata: %s", exc)
    return {}


async def _write_kb_meta(meta: Dict[str, Dict[str, str]]) -> None:
    path = _kb_meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def _kb_display_name(kb_id: str, meta: Dict[str, Dict[str, str]]) -> str:
    entry = meta.get(kb_id)
    name = (entry or {}).get("name", "") or ""
    return name.strip() or kb_id


# ---------------------------------------------------------------------------
# In-process cache of workspace-isolated LightRAG instances.
# Keyed by workspace name.  Lazily populated by ``_get_rag``.
# ---------------------------------------------------------------------------
_KB_RAG_CACHE: Dict[str, LightRAG] = {}
_KB_RAG_LOCK = asyncio.Lock()


def _safe_kb_id(kb_id: str) -> str:
    """Sanitize a KB id into a safe workspace token (keeps Unicode)."""
    # Strip leading/trailing whitespace and replace dangerous filesystem chars
    kb = re.sub(r'[/\\:*?"<>|]', "_", kb_id.strip())
    if not kb or not kb.strip("_"):
        raise ValueError("knowledge base id must not be empty")
    return kb


async def _get_rag(kb_id: str, build_rag_instance) -> LightRAG:
    """Return a cached, initialized LightRAG instance for ``kb_id``.

    Uses double-checked locking to avoid redundant instance creation under
    concurrent access.
    """
    ws = _safe_kb_id(kb_id)
    cached = _KB_RAG_CACHE.get(ws)
    if cached is not None:
        return cached
    async with _KB_RAG_LOCK:
        cached = _KB_RAG_CACHE.get(ws)
        if cached is not None:
            return cached
        rag = build_rag_instance(ws)
        await rag.initialize_storages()
        _KB_RAG_CACHE[ws] = rag
    return rag


# Directories under input_dir that are NOT knowledge bases.
_SKIP_DIRS = frozenset({"__parsed__"})


def _input_base() -> Path:
    return Path(global_args.input_dir)


def _discover_kbs() -> List[str]:
    """List workspace names that already exist on disk."""
    base = _input_base()
    if not base.exists():
        return []
    found = []
    for child in base.iterdir():
        if child.is_dir() and not child.name.startswith(".") and child.name not in _SKIP_DIRS:
            found.append(child.name)
    return sorted(found)


def _kb_doc_manager(kb_id: str) -> DocumentManager:
    return DocumentManager(global_args.input_dir, workspace=_safe_kb_id(kb_id))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class KnowledgeBaseCreate(BaseModel):
    id: str = Field(..., description="Unique knowledge base id / workspace name")
    description: Optional[str] = Field(None, description="Optional description")


class KnowledgeBaseRename(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=128, description="New display name"
    )


class KnowledgeBaseSummary(BaseModel):
    id: str
    name: str = Field("", description="Display name (defaults to the workspace id)")
    description: Optional[str] = None
    document_count: int = 0
    created_at: Optional[str] = None
    is_default: bool = False


class KnowledgeBaseDetail(KnowledgeBaseSummary):
    status_distribution: Dict[str, int] = Field(default_factory=dict)
    storage: Dict[str, str] = Field(default_factory=dict)


class DocumentListItem(BaseModel):
    id: str
    file_name: str
    status: str
    chunk_count: int = 0
    size: int = 0
    updated_at: Optional[str] = None
    multimodal: bool = False


class DocumentListResponse(BaseModel):
    items: List[DocumentListItem]
    total: int
    page: int
    page_size: int
    status_counts: Dict[str, int] = Field(default_factory=dict)


class TaskHistoryItem(BaseModel):
    track_id: str
    doc_id: str = ""
    file_name: str
    status: str
    updated_at: Optional[str] = None


class DeleteKbDocumentRequest(BaseModel):
    document_ids: List[str] = Field(..., description="Document ids to delete")
    force: bool = Field(False, description="Bypass pipeline-busy admission")
    delete_file: bool = Field(
        False, description="Also remove the uploaded source file"
    )
    delete_llm_cache: bool = Field(
        False, description="Also purge matching LLM cache entries"
    )


class DeleteKbDocumentResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_doc_status(raw: Any) -> DocProcessingStatus:
    if isinstance(raw, DocProcessingStatus):
        return raw
    return DocProcessingStatus(raw)


def _doc_to_item(
    doc_id: str, status: Any, input_dir: Path | None = None
) -> DocumentListItem:
    """Map a doc_status row to the KB document-list item.

    ``file_path`` is the canonical basename the pipeline stores — the original
    uploaded file name — while ``chunks_count`` carries the chunk total.
    ``size`` is not part of doc_status, so it is read from the workspace input
    directory (or its ``__parsed__`` archive for older uploads) on a
    best-effort basis (0 when the file is gone).
    """
    st = _parse_doc_status(status)
    content = getattr(st, "content", None) or {}
    raw_name = getattr(st, "file_path", None) or getattr(st, "file_name", None)
    if not raw_name or str(raw_name).strip() in (
        "",
        UNKNOWN_FILE_SOURCE,
        "no-file-path",
    ):
        raw_name = (
            content.get("file_name") if isinstance(content, dict) else None
        ) or doc_id
    file_name = Path(str(raw_name)).name or doc_id
    size = 0
    if input_dir is not None and file_name != doc_id:
        # Current uploads keep the file in the workspace input dir; older
        # ones were archived under ``__parsed__`` after ingestion.
        for candidate in (
            Path(input_dir) / file_name,
            Path(input_dir) / "__parsed__" / file_name,
        ):
            try:
                size = candidate.stat().st_size
                break
            except OSError:
                continue
    return DocumentListItem(
        id=doc_id,
        file_name=file_name,
        status=get_doc_status_value(st),
        chunk_count=int(getattr(st, "chunks_count", None) or 0),
        size=size,
        updated_at=getattr(st, "updated_at", None),
        multimodal=bool(getattr(st, "multimodal", False)),
    )


# Document status keys surfaced by the KB catalogue/detail endpoints. Some
# backends (or the document routes) add aggregate keys like ``all`` on top of
# the per-status counts; they are metadata, not a status, and would inflate
# every ``sum(counts.values())`` document total if kept.
_KB_STATUS_KEYS = (
    "pending",
    "processing",
    "parsing",
    "analyzing",
    "processed",
    "failed",
)


async def _counts_for(rag: LightRAG) -> Dict[str, int]:
    """Status counts for a workspace, restricted to the known status keys.

    Backends may key by ``DocStatus`` enum members or plain strings, so each
    key is coerced through ``.value`` when available before lowercasing;
    anything outside the known statuses (e.g. an ``all`` summary) is dropped.
    """
    try:
        raw = await rag.doc_status.get_all_status_counts()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("get_all_status_counts failed: %s", e)
        return {}
    if not isinstance(raw, dict):
        return {}
    counts: Dict[str, int] = {}
    for key, value in raw.items():
        name = getattr(key, "value", key)
        name = str(name).lower()
        if name not in _KB_STATUS_KEYS:
            continue
        try:
            counts[name] = counts.get(name, 0) + int(value)
        except (TypeError, ValueError):
            continue
    return counts


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def create_knowledge_base_routes(
    rag: LightRAG,
    api_key: str,
    args,
    build_rag_instance,
):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("", response_model=List[KnowledgeBaseSummary])
    async def list_knowledge_bases(
        request: Request, authenticated: bool = Depends(combined_auth)
    ):
        """List every knowledge base discovered on disk.

        The server's default workspace is an internal singleton, not a
        user-facing knowledge base: it is hidden from the catalogue so only
        explicitly created knowledge bases (workspace subdirectories) are
        listed. Its documents remain countable through the dashboard totals
        and the global ``/documents`` endpoints.
        """
        try:
            discovered = set(_discover_kbs())
            meta = await _read_kb_meta()
            result: List[KnowledgeBaseSummary] = []
            for kb_id in sorted(discovered):
                if kb_id == "default":
                    # Internal default workspace — never surfaced as a KB.
                    continue
                try:
                    # Reuse the primary instance for the default workspace so
                    # its stats match the global /documents view exactly.
                    kb_rag = await _get_rag(kb_id, build_rag_instance)
                    counts = await _counts_for(kb_rag)
                    doc_count = sum(counts.values())
                except Exception as e:
                    logger.warning("KB %s stats failed: %s", kb_id, e)
                    doc_count = 0
                    counts = {}
                result.append(
                    KnowledgeBaseSummary(
                        id=kb_id,
                        name=_kb_display_name(kb_id, meta),
                        document_count=doc_count,
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {e}")
            raise internal_server_error(e)

    @router.post("", response_model=KnowledgeBaseSummary)
    async def create_knowledge_base(
        payload: KnowledgeBaseCreate,
        request: Request,
        authenticated: bool = Depends(combined_auth),
    ):
        """Create a new knowledge base (initializes its workspace storage)."""
        try:
            kb_id = _safe_kb_id(payload.id)
            kb_rag = await _get_rag(kb_id, build_rag_instance)
            # Materialize the workspace's input directory so the KB shows up in
            # the directory-based discovery used by list/stats endpoints even
            # when no documents have been uploaded yet.
            doc_manager = _kb_doc_manager(kb_id)
            doc_manager.input_dir.mkdir(parents=True, exist_ok=True)
            counts = await _counts_for(kb_rag)
            return KnowledgeBaseSummary(
                id=kb_id,
                name=kb_id,
                description=payload.description,
                document_count=sum(counts.values()),
            )
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            raise internal_server_error(e)

    @router.get("/{kb_id}", response_model=KnowledgeBaseDetail)
    async def get_knowledge_base(
        kb_id: str,
        request: Request,
        authenticated: bool = Depends(combined_auth),
    ):
        """Get detail for a single knowledge base."""
        try:
            ws = _safe_kb_id(kb_id)
            kb_rag = await _get_rag(ws, build_rag_instance)
            counts = await _counts_for(kb_rag)
            meta = await _read_kb_meta()
            return KnowledgeBaseDetail(
                id=ws,
                name=_kb_display_name(ws, meta),
                document_count=sum(counts.values()),
                status_distribution=counts,
                storage={
                    "kv_storage": kb_rag.kv_storage,
                    "vector_storage": kb_rag.vector_storage,
                    "graph_storage": kb_rag.graph_storage,
                    "doc_status_storage": kb_rag.doc_status_storage,
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting knowledge base {kb_id}: {e}")
            raise internal_server_error(e)

    @router.patch("/{kb_id}", response_model=KnowledgeBaseSummary)
    async def rename_knowledge_base(
        kb_id: str,
        payload: KnowledgeBaseRename,
        request: Request,
        authenticated: bool = Depends(combined_auth),
    ):
        """Rename a knowledge base (display name only; storage id is fixed)."""
        try:
            ws = _safe_kb_id(kb_id)
            name = payload.name.strip()
            if not name:
                raise HTTPException(
                    status_code=422, detail="Knowledge base name must not be empty"
                )
            async with _KB_META_LOCK:
                meta = await _read_kb_meta()
                meta[ws] = {"name": name}
                await _write_kb_meta(meta)
            kb_rag = await _get_rag(ws, build_rag_instance)
            counts = await _counts_for(kb_rag)
            return KnowledgeBaseSummary(
                id=ws,
                name=name,
                document_count=sum(counts.values()),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error renaming knowledge base {kb_id}: {e}")
            raise internal_server_error(e)

    @router.delete("/{kb_id}")
    async def delete_knowledge_base(
        kb_id: str,
        request: Request,
        authenticated: bool = Depends(combined_auth),
    ):
        """Delete a knowledge base and drop all of its data."""
        try:
            ws = _safe_kb_id(kb_id)
            async with _KB_META_LOCK:
                meta = await _read_kb_meta()
                if ws in meta:
                    del meta[ws]
                    await _write_kb_meta(meta)
            # Remove cached instance so a later re-create starts fresh.
            async with _KB_RAG_LOCK:
                _KB_RAG_CACHE.pop(ws, None)
            # Best-effort removal of the input directory.
            input_dir = _kb_doc_manager(ws).input_dir
            if input_dir.exists():
                shutil.rmtree(input_dir, ignore_errors=True)
            return {"status": "success", "message": f"Knowledge base '{ws}' deleted."}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting knowledge base {kb_id}: {e}")
            raise internal_server_error(e)

    @router.get("/{kb_id}/documents", response_model=DocumentListResponse)
    async def list_documents(
        kb_id: str,
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=200),
        status: Optional[str] = Query(None, description="Filter by status"),
        search: Optional[str] = Query(None, description="Search by file name"),
        sort_field: str = Query("updated_at", description="Sort field"),
        sort_direction: str = Query("desc", description="Sort direction"),
        authenticated: bool = Depends(combined_auth),
    ):
        """Paginated, searchable, filterable document list for a KB."""
        try:
            ws = _safe_kb_id(kb_id)
            kb_rag = await _get_rag(ws, build_rag_instance)
            doc_manager = _kb_doc_manager(ws)
            status_filters = None
            if status:
                status_filters = [DocStatus(s) for s in status.split(",") if s]
            if sort_field not in ("created_at", "updated_at", "id", "file_path"):
                sort_field = "updated_at"
            if sort_direction not in ("asc", "desc"):
                sort_direction = "desc"
            docs, total = await kb_rag.doc_status.get_docs_paginated(
                status_filters=status_filters,
                page=page,
                page_size=page_size,
                sort_field=sort_field,
                sort_direction=sort_direction,
            )
            items = [
                _doc_to_item(doc_id, st, doc_manager.input_dir)
                for doc_id, st in docs
            ]
            if search:
                needle = search.lower()
                items = [
                    it for it in items if needle in (it.file_name or "").lower()
                ]
                total = len(items)
            counts = await _counts_for(kb_rag)
            return DocumentListResponse(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
                status_counts=counts,
            )
        except Exception as e:
            logger.error(f"Error listing documents for {kb_id}: {e}")
            raise internal_server_error(e)

    @router.post(
        "/{kb_id}/documents/upload",
        response_model=InsertResponse,
    )
    async def upload_document(
        kb_id: str,
        request: Request,
        file: UploadFile = File(...),
        parser: str = Form("auto"),
        image_processing: bool = Form(False),
        table_processing: bool = Form(False),
        formula_processing: bool = Form(False),
        enable_vlm: bool = Form(False),
        authenticated: bool = Depends(combined_auth),
    ):
        """Upload a file into a specific knowledge base and ingest it.

        The ingestion runs through the shared document module
        (``pipeline_index_file``), honouring the caller's parser picker and
        multimodal toggles: ``parser`` overrides the parser engine selection,
        while ``image_processing`` / ``table_processing`` / ``formula_processing``
        map to the ``i`` / ``t`` / ``e`` process options consumed by the
        pipeline's multimodal analysis stage.
        """
        try:
            ws = _safe_kb_id(kb_id)
            kb_rag = await _get_rag(ws, build_rag_instance)
            doc_manager = _kb_doc_manager(ws)

            safe_filename = sanitize_filename(file.filename, doc_manager.input_dir)
            try:
                filename_supported = doc_manager.is_supported_file(safe_filename)
            except FilenameParserHintError as hint_error:
                from fastapi import HTTPException

                raise HTTPException(status_code=400, detail=str(hint_error))
            if not filename_supported:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            # Resolve the caller's parser picker synchronously so a bad choice
            # fails fast (400) instead of surfacing later as a FAILED document.
            forced_engine = None
            if parser and parser.strip().lower() not in ("", "auto"):
                forced_engine = parser.strip()
                try:
                    resolve_parser_directives(
                        file_path=Path(safe_filename),
                        forced_engine=forced_engine,
                    )
                except FilenameParserHintError as engine_error:
                    raise HTTPException(status_code=400, detail=str(engine_error))

            # Map the multimodal toggles onto the pipeline's i/t/e selectors.
            forced_process_options = "".join(
                selector
                for selector, enabled in (
                    ("i", image_processing),
                    ("t", table_processing),
                    ("e", formula_processing),
                )
                if enabled
            )
            if enable_vlm and not getattr(global_args, "vlm_process_enable", False):
                logger.warning(
                    "KB %s upload %s: VLM requested but VLM_PROCESS_ENABLE "
                    "is off on the server; image analysis will be skipped "
                    "(tables/formulas still process).",
                    ws,
                    safe_filename,
                )

            file_path = doc_manager.input_dir / safe_filename
            existing = await get_existing_doc_by_file_path_candidates(
                kb_rag.doc_status, file_path
            )
            if existing:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=409,
                    detail=f"Document '{safe_filename}' already exists in this knowledge base.",
                )
            if file_path.exists() or find_existing_file_by_file_path(
                doc_manager.input_dir, safe_filename
            ):
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=409,
                    detail=f"Input directory already contains '{safe_filename}'.",
                )

            # Persist file content.
            import aiofiles

            async with aiofiles.open(file_path, "wb") as out_file:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    await out_file.write(chunk)

            track_id = generate_track_id("kb_upload")

            async def _indexing_work():
                try:
                    await pipeline_index_file(
                        kb_rag,
                        file_path,
                        track_id,
                        forced_engine=forced_engine,
                        forced_process_options=forced_process_options,
                    )
                except Exception as idx_err:
                    logger.error(
                        "KB %s ingest failed for %s: %s",
                        ws,
                        safe_filename,
                        idx_err,
                    )

            asyncio.create_task(_indexing_work())

            return InsertResponse(
                status="success",
                message=f"File '{safe_filename}' uploaded to '{ws}'. Processing in background.",
                track_id=track_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading to KB {kb_id}: {e}")
            raise internal_server_error(e)

    @router.post(
        "/{kb_id}/documents/delete",
        response_model=DeleteKbDocumentResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_documents(
        kb_id: str,
        payload: DeleteKbDocumentRequest,
        request: Request,
        managed_tasks: set = Depends(get_managed_background_tasks),
    ):
        """Delete documents from a KB via the shared document module.

        Mirrors the global ``/documents/delete_document`` contract: it atomically
        reserves the destructive pipeline slot for this KB's workspace, then hands
        the reservation to a managed background task that calls
        ``background_delete_documents`` — the same storage-teardown path used for
        global deletes.
        """
        try:
            ws = _safe_kb_id(kb_id)
            kb_rag = await _get_rag(ws, build_rag_instance)
            doc_manager = _kb_doc_manager(ws)
            doc_ids = [d for d in payload.document_ids if d]
            if not doc_ids:
                raise HTTPException(status_code=400, detail="No document ids provided.")
            existing_records = await kb_rag.doc_status.get_by_ids(doc_ids)
            missing = [
                did for did, rec in zip(doc_ids, existing_records) if rec is None
            ]
            if missing:
                raise HTTPException(
                    status_code=404,
                    detail=f"Documents not found: {', '.join(missing)}",
                )

            destructive_token = uuid4().hex

            async def _delete_work(started):
                started.set()
                await background_delete_documents(
                    kb_rag,
                    doc_manager,
                    doc_ids,
                    payload.delete_file,
                    payload.delete_llm_cache,
                    destructive_token,
                )

            async def _delete_backstop():
                await _release_destructive_busy(kb_rag, destructive_token)

            handed_off = False
            try:
                acquired, reason = await _acquire_destructive_busy(
                    kb_rag,
                    destructive_token,
                    kind="delete",
                    operation_record={"kind": "delete", "doc_ids": doc_ids},
                )
                if not acquired:
                    return DeleteKbDocumentResponse(
                        status="busy",
                        message=reason
                        or "Cannot delete documents while pipeline is busy",
                    )
                await start_reserved_background_task(
                    managed_tasks,
                    work=_delete_work,
                    backstop_release=_delete_backstop,
                )
                handed_off = True
                return DeleteKbDocumentResponse(
                    status="deletion_started",
                    message=(
                        f"Deletion of {len(doc_ids)} document(s) from '{ws}' "
                        "has been initiated."
                    ),
                )
            finally:
                if not handed_off:
                    await _release_destructive_busy(kb_rag, destructive_token)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting documents from KB {kb_id}: {e}")
            raise internal_server_error(e)

    @router.get("/{kb_id}/task_history", response_model=List[TaskHistoryItem])
    async def task_history(
        kb_id: str,
        request: Request,
        limit: int = Query(20, ge=1, le=100),
        authenticated: bool = Depends(combined_auth),
    ):
        """Recent ingestion history (most recently updated documents)."""
        try:
            ws = _safe_kb_id(kb_id)
            kb_rag = await _get_rag(ws, build_rag_instance)
            docs, _ = await kb_rag.doc_status.get_docs_paginated(
                page=1,
                page_size=limit,
                sort_field="updated_at",
                sort_direction="desc",
            )
            items = []
            for doc_id, st in docs:
                st_obj = _parse_doc_status(st)
                content = getattr(st_obj, "content", None) or {}
                file_name = (
                    getattr(st_obj, "file_name", None)
                    or (content.get("file_name") if isinstance(content, dict) else None)
                    or doc_id
                )
                items.append(
                    TaskHistoryItem(
                        track_id=getattr(st_obj, "track_id", "") or doc_id,
                        doc_id=doc_id,
                        file_name=file_name,
                        status=get_doc_status_value(st_obj),
                        updated_at=getattr(st_obj, "updated_at", None),
                    )
                )
            return items
        except Exception as e:
            logger.error(f"Error getting task history for {kb_id}: {e}")
            raise internal_server_error(e)

    return router
