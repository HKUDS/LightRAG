from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonSchema = dict[str, Any]


def object_schema(
    properties: JsonSchema, required: list[str] | None = None
) -> JsonSchema:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class SkillContract:
    name: str
    description: str
    input_schema: JsonSchema
    output_schema: JsonSchema
    security_policy: str
    audit_event: str
    error_strategy: str = "Return structured error and write audit event."


@dataclass
class SkillRegistry:
    contracts: dict[str, SkillContract] = field(default_factory=dict)

    def register(self, contract: SkillContract) -> None:
        self.contracts[contract.name] = contract

    def get(self, name: str) -> SkillContract:
        return self.contracts[name]

    def list_names(self) -> list[str]:
        return sorted(self.contracts)


def _contract(name: str, description: str, input_schema: JsonSchema) -> SkillContract:
    return SkillContract(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=object_schema(
            {
                "status": {"type": "string", "enum": ["success", "error"]},
                "data": {"type": "object"},
                "error": {"type": ["string", "null"]},
            },
            ["status", "data"],
        ),
        security_policy=(
            "Require tenant/workspace scope, RBAC permission, prompt-injection "
            "screening for user text, and audit logging before side effects."
        ),
        audit_event=f"skill.{name}",
    )


def build_default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    query_schema = object_schema(
        {
            "tenant_id": {"type": "string"},
            "workspace": {"type": "string"},
            "query": {"type": "string"},
            "mode": {"type": "string"},
            "include_references": {"type": "boolean"},
        },
        ["tenant_id", "workspace", "query"],
    )
    doc_schema = object_schema(
        {
            "tenant_id": {"type": "string"},
            "workspace": {"type": "string"},
            "content": {"type": "string"},
            "document_id": {"type": "string"},
            "file_path": {"type": "string"},
            "metadata": {"type": "object"},
        },
        ["tenant_id", "workspace", "content"],
    )
    simple_id_schema = object_schema(
        {
            "tenant_id": {"type": "string"},
            "workspace": {"type": "string"},
            "id": {"type": "string"},
        },
        ["tenant_id", "workspace", "id"],
    )
    crm_schema = object_schema(
        {
            "tenant_id": {"type": "string"},
            "workspace": {"type": "string"},
            "payload": {"type": "object"},
        },
        ["tenant_id", "workspace", "payload"],
    )

    for name, description, schema in [
        (
            "query_lightrag",
            "Query LightRAG with governed model and ACL policy.",
            query_schema,
        ),
        (
            "query_lightrag_context_only",
            "Return retrieved context and citations without LLM generation.",
            query_schema,
        ),
        (
            "ingest_document",
            "Ingest one document through LightRAG pipeline.",
            doc_schema,
        ),
        ("ingest_batch", "Ingest a batch of documents.", doc_schema),
        (
            "reindex_workspace",
            "Rebuild workspace knowledge from chunks.",
            simple_id_schema,
        ),
        (
            "delete_document_by_id",
            "Delete a document and derived KG/vector data.",
            simple_id_schema,
        ),
        (
            "delete_entity",
            "Delete an entity from KG and vector storage.",
            simple_id_schema,
        ),
        ("delete_relation", "Delete a relation between two entities.", crm_schema),
        (
            "merge_entities",
            "Merge duplicate entities with explicit strategy.",
            crm_schema,
        ),
        ("sync_model_catalog", "Sync hosted/local model catalog.", crm_schema),
        ("get_model_catalog", "Read visible and permitted model catalog.", crm_schema),
        ("route_model_by_policy", "Select model profile by policy.", crm_schema),
        (
            "check_cost_policy",
            "Validate request estimate against tenant caps.",
            crm_schema,
        ),
        ("create_crm_contact", "Create CRM contact record.", crm_schema),
        ("update_crm_contact", "Update CRM contact record.", crm_schema),
        ("create_ticket", "Create help desk ticket.", crm_schema),
        ("update_ticket", "Update help desk ticket.", crm_schema),
        ("search_conversations", "Search internal chat conversations.", query_schema),
        ("summarize_thread", "Summarize a chat thread with citations.", crm_schema),
        ("generate_report", "Generate auditable business report.", crm_schema),
        ("audit_action", "Append structured audit event.", crm_schema),
        (
            "validate_json_output",
            "Validate model output against a JSON schema.",
            crm_schema,
        ),
    ]:
        registry.register(_contract(name, description, schema))
    return registry


DEFAULT_SKILL_REGISTRY = build_default_skill_registry()
