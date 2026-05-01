from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any

from .approvals import ApprovalService
from .models import ApprovalRequest, ApprovalStatus, Principal
from .permissions import ACTIVITY_DOCUMENT_DELETE


@dataclass(frozen=True)
class ApprovalExecutionOutcome:
    approval: ApprovalRequest
    audit_result: str
    action_executed: bool
    metadata: dict[str, Any]


class ApprovalExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        approval: ApprovalRequest,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.approval = approval
        self.metadata = metadata or {}


class ApprovalActionExecutor:
    """Executes the small allowlisted set of actions that can run after approval."""

    def __init__(
        self,
        rag: Any,
        *,
        workspace_rag_resolver: Callable[[str], Awaitable[Any]] | None = None,
        action_handlers: dict[
            str,
            Callable[..., Awaitable[ApprovalExecutionOutcome]],
        ]
        | None = None,
    ) -> None:
        self.rag = rag
        self.workspace_rag_resolver = workspace_rag_resolver
        self.action_handlers = action_handlers or {}

    def supports(self, approval: ApprovalRequest) -> bool:
        return (
            approval.action == ACTIVITY_DOCUMENT_DELETE
            or approval.action in self.action_handlers
        )

    async def execute_if_supported(
        self,
        *,
        approval: ApprovalRequest,
        approvals: ApprovalService,
        principal: Principal,
    ) -> ApprovalExecutionOutcome:
        if not self.supports(approval):
            return ApprovalExecutionOutcome(
                approval=approval,
                audit_result="approved",
                action_executed=False,
                metadata={"executor": "unsupported_action"},
            )
        if approval.status == ApprovalStatus.EXECUTED:
            return ApprovalExecutionOutcome(
                approval=approval,
                audit_result="already_executed",
                action_executed=False,
                metadata={"idempotent": True},
            )
        if approval.status in {ApprovalStatus.EXECUTING, ApprovalStatus.FAILED}:
            return ApprovalExecutionOutcome(
                approval=approval,
                audit_result=approval.status.value,
                action_executed=False,
                metadata={"idempotent": True},
            )
        if approval.status != ApprovalStatus.APPROVED:
            return ApprovalExecutionOutcome(
                approval=approval,
                audit_result=approval.status.value,
                action_executed=False,
                metadata={"executor": "not_approved"},
            )

        action_handler = self.action_handlers.get(approval.action)
        if action_handler is not None:
            try:
                return await action_handler(
                    approval=approval,
                    approvals=approvals,
                    principal=principal,
                )
            except ApprovalExecutionError:
                raise
            except Exception as exc:
                failed = await approvals.mark_failed(approval.approval_id, principal)
                raise ApprovalExecutionError(
                    str(exc),
                    approval=failed,
                    metadata={
                        "action": approval.action,
                        "executor": "registered_handler",
                    },
                ) from exc

        executing = await approvals.begin_execution(approval.approval_id, principal)
        if executing is None:
            current = await approvals.get(approval.approval_id)
            current = current or approval
            return ApprovalExecutionOutcome(
                approval=current,
                audit_result="already_taken",
                action_executed=False,
                metadata={"idempotent": True, "status": current.status.value},
            )

        metadata: dict[str, Any] = {}
        try:
            metadata = self._document_delete_metadata(executing)
            await self._delete_document(metadata["document_id"], executing.workspace_id)
        except Exception as exc:
            failed = await approvals.mark_failed(executing.approval_id, principal)
            raise ApprovalExecutionError(
                str(exc),
                approval=failed,
                metadata=getattr(exc, "metadata", None) or metadata,
            ) from exc

        executed = await approvals.mark_executed(executing.approval_id, principal)
        return ApprovalExecutionOutcome(
            approval=executed,
            audit_result="executed",
            action_executed=True,
            metadata=metadata,
        )

    def _document_delete_metadata(self, approval: ApprovalRequest) -> dict[str, Any]:
        document_id = approval.metadata.get("document_id") or approval.metadata.get(
            "doc_id"
        )
        if not document_id:
            raise ApprovalExecutionError(
                "Document deletion approval is missing document_id.",
                approval=approval,
                metadata={"reason": "missing_document_id"},
            )
        return {"document_id": str(document_id)}

    async def _delete_document(
        self, document_id: str, workspace_id: str | None
    ) -> None:
        rag = self.rag
        if workspace_id and self.workspace_rag_resolver is not None:
            rag = await self.workspace_rag_resolver(workspace_id)
        if not hasattr(rag, "adelete_by_doc_id"):
            raise RuntimeError("Current LightRAG instance cannot delete documents.")
        await rag.adelete_by_doc_id(document_id)
