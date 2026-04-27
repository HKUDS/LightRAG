from __future__ import annotations

from pathlib import Path
from typing import Any

import aiofiles
from fastapi import BackgroundTasks, HTTPException, UploadFile, status

from lightrag.base import QueryParam
from lightrag.utils import generate_track_id

from lightrag_enterprise.system import (
    ACTIVITY_ACTIVITY_READ,
    ACTIVITY_AREA_READ,
    ACTIVITY_ASSISTANTS_READ,
    ACTIVITY_DOCUMENT_DELETE,
    ACTIVITY_DOCUMENT_READ,
    ACTIVITY_DOCUMENT_UPLOAD,
    ACTIVITY_QUERY,
    AccessControlService,
    ApprovalService,
    AuditService,
    Principal,
)
from lightrag_enterprise.system.runtime import approvals_enforced, private_strict_enabled

from .models import (
    LittleBullActivityItem,
    LittleBullArea,
    LittleBullAssistant,
    LittleBullDocument,
    LittleBullDocumentsResponse,
    LittleBullQueryRequest,
    LittleBullQueryResponse,
    LittleBullUploadResponse,
)


AREA_PRESENTATION = {
    "default": ("Default", "📁", "#2563EB"),
    "casa": ("Casa", "🏠", "#FACC15"),
    "familia": ("Família", "👨‍👩‍👧", "#F97316"),
    "financas": ("Finanças", "💳", "#22C55E"),
    "trabalho": ("Trabalho", "💼", "#2563EB"),
    "estudos": ("Estudos", "📚", "#7C3AED"),
    "negocio": ("Pequeno negócio", "🧾", "#0F766E"),
}

WORKSPACE_PRIVATE_POLICY = "little_bull.workspace_contains_private_data"


def sanitize_upload_filename(filename: str, input_dir: Path) -> str:
    clean_name = filename.replace("/", "").replace("\\", "").replace("..", "")
    clean_name = "".join(char for char in clean_name if ord(char) >= 32 and char != "\x7f")
    clean_name = clean_name.strip().strip(".")
    if not clean_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    try:
        final_path = (input_dir / clean_name).resolve()
        if not final_path.is_relative_to(input_dir.resolve()):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsafe filename")
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename") from exc
    return clean_name


class LittleBullService:
    def __init__(
        self,
        *,
        rag: Any,
        doc_manager: Any,
        repository: Any,
        access: AccessControlService,
        audit: AuditService,
        approvals: ApprovalService,
    ) -> None:
        self.rag = rag
        self.doc_manager = doc_manager
        self.repository = repository
        self.access = access
        self.audit = audit
        self.approvals = approvals

    def _require(self, principal: Principal, activity: str, workspace_id: str | None = None) -> None:
        decision = self.access.require(principal, activity=activity, workspace_id=workspace_id)
        if not decision.allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason)

    def _require_backed_workspace(self, workspace_id: str) -> None:
        current_workspace = getattr(self.rag, "workspace", None) or "default"
        if workspace_id != current_workspace:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Workspace '{workspace_id}' is authorized but is not attached to the current "
                    f"LightRAG data plane '{current_workspace}'."
                ),
            )

    async def _workspace_tenant(self, workspace_id: str | None) -> str | None:
        if workspace_id is None:
            return None
        workspace = await self.repository.get_workspace(workspace_id)
        return workspace.tenant_id if workspace else None

    async def list_areas(self, principal: Principal) -> list[LittleBullArea]:
        self._require(principal, ACTIVITY_AREA_READ)
        workspaces = await self.repository.list_workspaces(
            None if principal.is_master_global else principal.tenant_id
        )
        if not principal.is_master_global:
            workspaces = [
                workspace for workspace in workspaces if workspace.workspace_id in principal.workspace_ids
            ]
        counts = await self._status_counts_safe()
        areas: list[LittleBullArea] = []
        for workspace in workspaces:
            label, emoji, accent = AREA_PRESENTATION.get(
                workspace.slug,
                (workspace.name, "📁", "#2563EB"),
            )
            areas.append(
                LittleBullArea(
                    id=workspace.workspace_id,
                    label=label or workspace.name,
                    slug=workspace.slug,
                    description=workspace.description,
                    privacy=workspace.privacy,
                    document_count=sum(counts.values()),
                    ready_count=counts.get("processed", 0),
                    processing_count=counts.get("processing", 0) + counts.get("pending", 0),
                    accent=accent,
                    emoji=emoji,
                )
            )
        return areas

    async def list_documents(
        self, principal: Principal, *, workspace_id: str, page: int = 1, page_size: int = 50
    ) -> LittleBullDocumentsResponse:
        self._require(principal, ACTIVITY_DOCUMENT_READ, workspace_id)
        self._require_backed_workspace(workspace_id)
        try:
            (documents_with_ids, total_count), status_counts = await self._documents_paginated(
                page=page,
                page_size=page_size,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        documents: list[LittleBullDocument] = []
        for doc_id, doc in documents_with_ids:
            file_path = str(getattr(doc, "file_path", "") or "")
            title = Path(file_path).name or doc_id
            documents.append(
                LittleBullDocument(
                    id=doc_id,
                    file_path=file_path,
                    title=title,
                    status=str(getattr(doc, "status", "unknown")),
                    content_summary=str(getattr(doc, "content_summary", "") or ""),
                    content_length=int(getattr(doc, "content_length", 0) or 0),
                    updated_at=str(getattr(doc, "updated_at", "") or "") or None,
                    created_at=str(getattr(doc, "created_at", "") or "") or None,
                    track_id=getattr(doc, "track_id", None),
                    chunks_count=getattr(doc, "chunks_count", None),
                    metadata=getattr(doc, "metadata", {}) or {},
                )
            )
        await self.audit.record(
            principal=principal,
            action=ACTIVITY_DOCUMENT_READ,
            tenant_id=await self._workspace_tenant(workspace_id),
            workspace_id=workspace_id,
            result="success",
            metadata={"total_count": total_count},
        )
        return LittleBullDocumentsResponse(
            documents=documents,
            total_count=total_count,
            status_counts=status_counts,
        )

    async def upload_document(
        self,
        principal: Principal,
        *,
        workspace_id: str,
        file: UploadFile,
        background_tasks: BackgroundTasks,
        confidentiality: str = "normal",
    ) -> LittleBullUploadResponse:
        self._require(principal, ACTIVITY_DOCUMENT_UPLOAD, workspace_id)
        self._require_backed_workspace(workspace_id)
        input_dir = Path(self.doc_manager.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = sanitize_upload_filename(file.filename or "document", input_dir)
        if hasattr(self.doc_manager, "is_supported_file") and not self.doc_manager.is_supported_file(safe_filename):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")
        file_path = input_dir / safe_filename
        if file_path.exists():
            return LittleBullUploadResponse(
                status="duplicated",
                message=f"File '{safe_filename}' already exists.",
                track_id=None,
                workspace_id=workspace_id,
            )

        async with aiofiles.open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        tenant_id = await self._workspace_tenant(workspace_id)
        if confidentiality in {"sensivel", "privado"}:
            await self.repository.set_policy(
                WORKSPACE_PRIVATE_POLICY,
                True,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )

        track_id = generate_track_id("little_bull_upload")
        from lightrag.api.routers.document_routes import pipeline_index_file

        background_tasks.add_task(pipeline_index_file, self.rag, file_path, track_id)
        await self.audit.record(
            principal=principal,
            action=ACTIVITY_DOCUMENT_UPLOAD,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            result="queued",
            metadata={"file_name": safe_filename, "track_id": track_id, "confidentiality": confidentiality},
        )
        return LittleBullUploadResponse(
            status="success",
            message=f"File '{safe_filename}' uploaded and queued for indexing.",
            track_id=track_id,
            workspace_id=workspace_id,
        )

    async def query(self, principal: Principal, request: LittleBullQueryRequest) -> LittleBullQueryResponse:
        self._require(principal, ACTIVITY_QUERY, request.workspace_id)
        self._require_backed_workspace(request.workspace_id)
        tenant_id = await self._workspace_tenant(request.workspace_id)
        workspace_contains_private_data = bool(
            await self.repository.get_policy(
                WORKSPACE_PRIVATE_POLICY,
                tenant_id=tenant_id,
                workspace_id=request.workspace_id,
            )
        )
        requires_private_profile = request.confidentiality in {"sensivel", "privado"} or workspace_contains_private_data
        if requires_private_profile and private_strict_enabled():
            if request.model_profile != "privado":
                await self.audit.record(
                    principal=principal,
                    action=ACTIVITY_QUERY,
                    tenant_id=tenant_id,
                    workspace_id=request.workspace_id,
                    result="blocked",
                    model=request.model_profile,
                    metadata={
                        "reason": "private_local_required",
                        "confidentiality": request.confidentiality,
                        "workspace_contains_private_data": workspace_contains_private_data,
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Private/local profile is required for sensitive or private documents.",
                )

        param = QueryParam(
            mode=request.mode,
            response_type=request.response_type,
            stream=False,
            conversation_history=request.conversation_history,
        )
        if request.top_k is not None:
            param.top_k = request.top_k
        result = await self.rag.aquery_llm(request.query, param=param)
        llm_response = result.get("llm_response", {}) if isinstance(result, dict) else {}
        data = result.get("data", {}) if isinstance(result, dict) else {}
        response_content = llm_response.get("content") or "No relevant context found for the query."
        references = data.get("references", []) if request.include_references else []
        if request.include_references and request.include_chunk_content:
            references = self._enrich_references(references, data.get("chunks", []))
        await self.audit.record(
            principal=principal,
            action=ACTIVITY_QUERY,
            tenant_id=tenant_id,
            workspace_id=request.workspace_id,
            result="success",
            model=request.model_profile,
            metadata={"mode": request.mode, "reference_count": len(references)},
        )
        return LittleBullQueryResponse(
            response=response_content,
            references=references,
            workspace_id=request.workspace_id,
            model_profile=request.model_profile,
        )

    async def delete_document(self, principal: Principal, *, workspace_id: str, document_id: str) -> dict[str, Any]:
        self._require(principal, ACTIVITY_DOCUMENT_DELETE, workspace_id)
        self._require_backed_workspace(workspace_id)
        if approvals_enforced():
            approval = await self.approvals.request(
                principal=principal,
                action=ACTIVITY_DOCUMENT_DELETE,
                tenant_id=await self._workspace_tenant(workspace_id),
                workspace_id=workspace_id,
                reason="Document deletion requires human approval.",
                payload={"document_id": document_id},
            )
            await self.audit.record(
                principal=principal,
                action=ACTIVITY_DOCUMENT_DELETE,
                tenant_id=approval.tenant_id,
                workspace_id=workspace_id,
                result="pending_approval",
                approval_id=approval.approval_id,
                metadata={"document_id": document_id},
            )
            return {
                "status": "pending_approval",
                "message": "Document deletion is waiting for human approval.",
                "approval": approval.to_dict(),
            }
        if hasattr(self.rag, "adelete_by_doc_id"):
            await self.rag.adelete_by_doc_id(document_id)
        await self.audit.record(
            principal=principal,
            action=ACTIVITY_DOCUMENT_DELETE,
            tenant_id=await self._workspace_tenant(workspace_id),
            workspace_id=workspace_id,
            result="success",
            metadata={"document_id": document_id},
        )
        return {"status": "success", "message": "Document deleted.", "doc_id": document_id}

    async def list_activity(self, principal: Principal, *, workspace_id: str, limit: int = 50) -> list[LittleBullActivityItem]:
        self._require(principal, ACTIVITY_ACTIVITY_READ, workspace_id)
        events = await self.audit.list(
            tenant_id=await self._workspace_tenant(workspace_id),
            workspace_id=workspace_id,
            limit=limit,
        )
        return [
            LittleBullActivityItem(
                id=event.event_id,
                action=event.action,
                result=event.result,
                created_at=event.created_at.isoformat(),
                actor_user_id=event.actor_user_id,
                workspace_id=event.workspace_id,
                metadata=event.metadata,
            )
            for event in events
        ]

    async def list_assistants(self, principal: Principal, *, workspace_id: str) -> list[LittleBullAssistant]:
        self._require(principal, ACTIVITY_ASSISTANTS_READ, workspace_id)
        return [
            LittleBullAssistant(
                id="simple_answer",
                name="Resposta simples",
                description="Responde em linguagem direta usando o workspace ativo.",
                response_rules=["Usar fontes quando disponiveis", "Evitar resposta sem contexto"],
            ),
            LittleBullAssistant(
                id="checklist",
                name="Checklist",
                description="Transforma documentos em listas de providencias.",
                response_rules=["Mostrar itens acionaveis", "Preservar fontes"],
            ),
            LittleBullAssistant(
                id="private_local",
                name="Privado/local",
                description="Perfil exigido para dados sensiveis ou privados.",
                response_rules=["Nao usar modelo hospedado por padrao", "Registrar auditoria"],
            ),
        ]

    async def _documents_paginated(self, *, page: int, page_size: int) -> tuple[tuple[list[tuple[str, Any]], int], dict[str, int]]:
        docs_task = self.rag.doc_status.get_docs_paginated(
            status_filter=None,
            page=page,
            page_size=page_size,
            sort_field="updated_at",
            sort_direction="desc",
        )
        counts_task = self.rag.doc_status.get_all_status_counts()
        documents = await docs_task
        counts = await counts_task
        return documents, dict(counts or {})

    async def _status_counts_safe(self) -> dict[str, int]:
        try:
            return dict(await self.rag.doc_status.get_all_status_counts())
        except Exception:
            return {}

    @staticmethod
    def _enrich_references(references: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ref_id_to_content: dict[str, list[str]] = {}
        for chunk in chunks:
            ref_id = chunk.get("reference_id")
            content = chunk.get("content")
            if ref_id and content:
                ref_id_to_content.setdefault(ref_id, []).append(content)
        enriched: list[dict[str, Any]] = []
        for reference in references:
            copy = dict(reference)
            if copy.get("reference_id") in ref_id_to_content:
                copy["content"] = ref_id_to_content[copy["reference_id"]]
            enriched.append(copy)
        return enriched
