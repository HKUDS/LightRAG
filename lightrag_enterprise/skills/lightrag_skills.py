from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightrag import LightRAG, QueryParam

from lightrag_enterprise.audit.audit_log import AuditSink


@dataclass
class LightRAGSkillService:
    """Thin skill wrapper around a LightRAG instance.

    The wrapper is intentionally small: enterprise policy should live outside
    LightRAG while the core remains responsible for retrieval and KG lifecycle.
    """

    rag: LightRAG
    audit_sink: AuditSink | None = None

    async def query_lightrag(
        self, query: str, *, mode: str = "mix", include_references: bool = True
    ) -> dict[str, Any]:
        param = QueryParam(mode=mode, include_references=include_references)
        result = await self.rag.aquery_llm(query, param)
        await self._audit("query_lightrag", {"mode": mode})
        return {"status": "success", "data": result, "error": None}

    async def query_lightrag_context_only(
        self, query: str, *, mode: str = "mix"
    ) -> dict[str, Any]:
        param = QueryParam(mode=mode, only_need_context=True)
        result = await self.rag.aquery_data(query, param)
        await self._audit("query_lightrag_context_only", {"mode": mode})
        return {"status": "success", "data": result, "error": None}

    async def ingest_document(
        self,
        content: str,
        *,
        document_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        track_id = await self.rag.ainsert(
            content, ids=document_id, file_paths=file_path or "unknown_source"
        )
        await self._audit("ingest_document", {"document_id": document_id})
        return {"status": "success", "data": {"track_id": track_id}, "error": None}

    async def delete_document_by_id(self, document_id: str) -> dict[str, Any]:
        result = await self.rag.adelete_by_doc_id(document_id)
        await self._audit("delete_document_by_id", {"document_id": document_id})
        return {"status": "success", "data": result.__dict__, "error": None}

    async def _audit(self, action: str, metadata: dict[str, Any]) -> None:
        if self.audit_sink is not None:
            await self.audit_sink.write(
                actor="skill-service",
                action=action,
                tenant_id="unknown",
                workspace=self.rag.workspace,
                metadata=metadata,
            )
