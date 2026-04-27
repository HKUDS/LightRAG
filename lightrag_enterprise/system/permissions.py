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
ACTIVITY_QUERY = "little_bull.query"
ACTIVITY_ASSISTANTS_READ = "little_bull.assistants.read"
ACTIVITY_ACTIVITY_READ = "little_bull.activity.read"
ACTIVITY_APPROVAL_READ = "little_bull.approvals.read"
ACTIVITY_APPROVAL_DECIDE = "little_bull.approvals.decide"
ACTIVITY_AUDIT_READ = "little_bull.audit.read"
ACTIVITY_ADMIN = "little_bull.admin"
ACTIVITY_POLICY_MANAGE = "little_bull.policies.manage"

DESTRUCTIVE_ACTIVITIES = {
    ACTIVITY_DOCUMENT_DELETE,
    "little_bull.documents.reindex",
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
    }
)

MANAGER_PERMISSIONS = OPERATOR_PERMISSIONS | frozenset(
    {
        ACTIVITY_WORKSPACE_MANAGE,
        ACTIVITY_DOCUMENT_DELETE,
        ACTIVITY_APPROVAL_READ,
        ACTIVITY_APPROVAL_DECIDE,
        ACTIVITY_AUDIT_READ,
    }
)

MASTER_PERMISSIONS = frozenset({ANY_PERMISSION})

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    MASTER_ROLE: MASTER_PERMISSIONS,
    OPERATOR_ROLE: OPERATOR_PERMISSIONS,
    MANAGER_ROLE: MANAGER_PERMISSIONS,
}


def permissions_for_roles(roles: tuple[str, ...] | list[str] | set[str]) -> frozenset[str]:
    permissions: set[str] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(str(role), frozenset()))
    return frozenset(permissions)


def permission_allowed(granted: frozenset[str], required: str) -> bool:
    return ANY_PERMISSION in granted or required in granted

