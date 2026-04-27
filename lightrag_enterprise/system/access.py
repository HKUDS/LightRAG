from __future__ import annotations

from .models import AccessDecision, Principal
from .permissions import (
    DESTRUCTIVE_ACTIVITIES,
    permission_allowed,
)


class AccessControlService:
    def require(
        self,
        principal: Principal,
        *,
        activity: str,
        workspace_id: str | None = None,
        require_approval: bool = False,
    ) -> AccessDecision:
        if workspace_id and not principal.can_access_workspace(workspace_id):
            return AccessDecision(False, "Workspace is outside principal scope.")
        if not permission_allowed(principal.permissions, activity):
            return AccessDecision(False, "Role lacks required activity permission.")
        if require_approval or activity in DESTRUCTIVE_ACTIVITIES:
            return AccessDecision(True, "Allowed after approval gate.", requires_approval=True)
        return AccessDecision(True, "Allowed by RBAC policy.")

