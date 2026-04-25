from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    name: str
    mission: str
    allowed_skills: tuple[str, ...]
    safety_notes: tuple[str, ...] = field(default_factory=tuple)


AGENT_SPECS: dict[str, AgentSpec] = {
    "orchestrator_agent": AgentSpec(
        name="orchestrator_agent",
        mission="Coordinate enterprise workflows around LightRAG without bypassing policy gates.",
        allowed_skills=("query_lightrag", "route_model_by_policy", "audit_action"),
        safety_notes=("Never expose private planning traces.",),
    ),
    "planner_agent": AgentSpec(
        name="planner_agent",
        mission="Compare bounded execution plans and emit only decisions, trade-offs, and validations.",
        allowed_skills=("route_model_by_policy", "validate_json_output", "audit_action"),
    ),
    "retrieval_agent": AgentSpec(
        name="retrieval_agent",
        mission="Use LightRAG as knowledge layer, retrieval core, and memory engine.",
        allowed_skills=(
            "query_lightrag",
            "query_lightrag_context_only",
            "ingest_document",
            "ingest_batch",
        ),
    ),
    "crm_agent": AgentSpec(
        name="crm_agent",
        mission="Operate CRM domain records using governed skills and auditable actions.",
        allowed_skills=("create_crm_contact", "update_crm_contact", "create_ticket"),
    ),
    "internal_chat_agent": AgentSpec(
        name="internal_chat_agent",
        mission="Answer internal chat questions with citations and conversation memory boundaries.",
        allowed_skills=("search_conversations", "summarize_thread", "query_lightrag"),
    ),
    "workflow_agent": AgentSpec(
        name="workflow_agent",
        mission="Execute reusable enterprise workflows after guardrail checks.",
        allowed_skills=("create_ticket", "update_ticket", "audit_action"),
    ),
    "integration_agent": AgentSpec(
        name="integration_agent",
        mission="Coordinate connectors through allowlisted operations only.",
        allowed_skills=("audit_action", "validate_json_output"),
    ),
    "model_router_agent": AgentSpec(
        name="model_router_agent",
        mission="Select runtime-visible and policy-permitted models.",
        allowed_skills=("sync_model_catalog", "get_model_catalog", "route_model_by_policy"),
    ),
    "compliance_audit_agent": AgentSpec(
        name="compliance_audit_agent",
        mission="Review actions for RBAC, ACL, retention, privacy, and audit completeness.",
        allowed_skills=("audit_action", "check_cost_policy"),
    ),
    "critic_evaluator_agent": AgentSpec(
        name="critic_evaluator_agent",
        mission="Challenge plans and outputs for unsupported claims, security gaps, and regressions.",
        allowed_skills=("validate_json_output", "audit_action"),
    ),
}


def get_agent_spec(name: str) -> AgentSpec:
    return AGENT_SPECS[name]
