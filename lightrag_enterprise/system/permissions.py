from __future__ import annotations

MASTER_ROLE = "master"
OPERATOR_ROLE = "operador"
MANAGER_ROLE = "gerente"

ANY_PERMISSION = "*"

ACTIVITY_AREA_READ = "little_bull.areas.read"
ACTIVITY_WORKSPACE_MANAGE = "little_bull.workspaces.manage"
ACTIVITY_DOCUMENT_READ = "little_bull.documents.read"
ACTIVITY_DOCUMENT_UPLOAD = "little_bull.documents.upload"
ACTIVITY_DOCUMENT_DELETE = "little_bull.documents.delete"
ACTIVITY_DOCUMENT_REINDEX = "little_bull.documents.reindex"
ACTIVITY_QUERY = "little_bull.query"
ACTIVITY_ASSISTANTS_READ = "little_bull.assistants.read"
ACTIVITY_ACTIVITY_READ = "little_bull.activity.read"
ACTIVITY_APPROVAL_READ = "little_bull.approvals.read"
ACTIVITY_APPROVAL_DECIDE = "little_bull.approvals.decide"
ACTIVITY_AUDIT_READ = "little_bull.audit.read"
ACTIVITY_ADMIN = "little_bull.admin"
ACTIVITY_MODEL_MANAGE = "little_bull.models.manage"
ACTIVITY_AGENT_MANAGE = "little_bull.agents.manage"
ACTIVITY_CONVERSATION_READ = "little_bull.conversations.read"
ACTIVITY_CONVERSATION_SAVE = "little_bull.conversations.save"
ACTIVITY_CONVERSATION_EXPORT = "little_bull.conversations.export"
ACTIVITY_CORRELATION_SUGGEST = "little_bull.correlations.suggest"
ACTIVITY_CORRELATION_DECIDE = "little_bull.correlations.decide"
ACTIVITY_POLICY_MANAGE = "little_bull.policies.manage"
ACTIVITY_CORE_CACHE_CLEAR = "little_bull.core.cache.clear"
ACTIVITY_CORE_GRAPH_CREATE = "little_bull.core.graph.create"
ACTIVITY_CORE_GRAPH_MUTATE = "little_bull.core.graph.mutate"
ACTIVITY_CORE_OLLAMA_USE = "little_bull.core.ollama.use"
ACTIVITY_CORE_PIPELINE_MANAGE = "little_bull.core.pipeline.manage"
ACTIVITY_CORE_QUERY_DATA = "little_bull.core.query.data"

DESTRUCTIVE_ACTIVITIES = {
    ACTIVITY_DOCUMENT_DELETE,
    ACTIVITY_DOCUMENT_REINDEX,
    ACTIVITY_CORE_CACHE_CLEAR,
    ACTIVITY_CORE_GRAPH_MUTATE,
    "little_bull.graph.merge",
}

OPERATOR_PERMISSIONS = frozenset(
    {
        ACTIVITY_AREA_READ,
        ACTIVITY_DOCUMENT_READ,
        ACTIVITY_DOCUMENT_UPLOAD,
        ACTIVITY_QUERY,
        ACTIVITY_ASSISTANTS_READ,
        ACTIVITY_ACTIVITY_READ,
        ACTIVITY_CONVERSATION_READ,
        ACTIVITY_CONVERSATION_SAVE,
        ACTIVITY_CONVERSATION_EXPORT,
        ACTIVITY_CORRELATION_SUGGEST,
    }
)

MANAGER_PERMISSIONS = OPERATOR_PERMISSIONS | frozenset(
    {
        ACTIVITY_WORKSPACE_MANAGE,
        ACTIVITY_DOCUMENT_DELETE,
        ACTIVITY_DOCUMENT_REINDEX,
        ACTIVITY_CORE_CACHE_CLEAR,
        ACTIVITY_CORE_GRAPH_CREATE,
        ACTIVITY_CORE_GRAPH_MUTATE,
        ACTIVITY_CORE_OLLAMA_USE,
        ACTIVITY_CORE_PIPELINE_MANAGE,
        ACTIVITY_CORE_QUERY_DATA,
        ACTIVITY_APPROVAL_READ,
        ACTIVITY_APPROVAL_DECIDE,
        ACTIVITY_AUDIT_READ,
        ACTIVITY_CORRELATION_DECIDE,
    }
)

MASTER_PERMISSIONS = frozenset({ANY_PERMISSION})

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    MASTER_ROLE: MASTER_PERMISSIONS,
    OPERATOR_ROLE: OPERATOR_PERMISSIONS,
    MANAGER_ROLE: MANAGER_PERMISSIONS,
}


def permissions_for_roles(
    roles: tuple[str, ...] | list[str] | set[str],
) -> frozenset[str]:
    permissions: set[str] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(str(role), frozenset()))
    return frozenset(permissions)


def permission_allowed(granted: frozenset[str], required: str) -> bool:
    return ANY_PERMISSION in granted or required in granted
