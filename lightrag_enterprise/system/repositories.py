from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol

from .models import (
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    Membership,
    SystemUser,
    Tenant,
    Workspace,
    new_id,
    utc_now,
)


class SystemRepository(Protocol):
    async def has_users(self) -> bool: ...
    async def create_user(self, user: SystemUser) -> SystemUser: ...
    async def get_user_by_username(self, username: str) -> SystemUser | None: ...
    async def get_user(self, user_id: str) -> SystemUser | None: ...
    async def list_users(self) -> list[SystemUser]: ...
    async def create_tenant(self, tenant: Tenant) -> Tenant: ...
    async def get_tenant(self, tenant_id: str) -> Tenant | None: ...
    async def list_tenants(self) -> list[Tenant]: ...
    async def create_workspace(self, workspace: Workspace) -> Workspace: ...
    async def get_workspace(self, workspace_id: str) -> Workspace | None: ...
    async def list_workspaces(self, tenant_id: str | None = None) -> list[Workspace]: ...
    async def create_membership(self, membership: Membership) -> Membership: ...
    async def list_memberships_for_user(self, user_id: str) -> list[Membership]: ...
    async def write_audit_event(self, event: AuditEvent) -> AuditEvent: ...
    async def list_audit_events(
        self, tenant_id: str | None = None, workspace_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]: ...
    async def create_approval_request(self, approval: ApprovalRequest) -> ApprovalRequest: ...
    async def get_approval_request(self, approval_id: str) -> ApprovalRequest | None: ...
    async def update_approval_status(
        self, approval_id: str, status: ApprovalStatus, decided_by: str
    ) -> ApprovalRequest: ...
    async def list_approval_requests(
        self, tenant_id: str | None = None, workspace_id: str | None = None, status: ApprovalStatus | None = None
    ) -> list[ApprovalRequest]: ...
    async def get_policy(self, key: str, tenant_id: str | None = None, workspace_id: str | None = None) -> Any: ...
    async def set_policy(self, key: str, value: Any, tenant_id: str | None = None, workspace_id: str | None = None) -> None: ...


class InMemorySystemRepository:
    def __init__(self) -> None:
        self.users: dict[str, SystemUser] = {}
        self.tenants: dict[str, Tenant] = {}
        self.workspaces: dict[str, Workspace] = {}
        self.memberships: dict[str, Membership] = {}
        self.audit_events: list[AuditEvent] = []
        self.approvals: dict[str, ApprovalRequest] = {}
        self.policies: dict[tuple[str | None, str | None, str], Any] = {}
        self._seed_default_scope()

    def _seed_default_scope(self) -> None:
        tenant = Tenant(tenant_id="default", name="Default")
        workspace = Workspace(
            workspace_id="default",
            tenant_id=tenant.tenant_id,
            name="Default",
            slug="default",
            description="Local-first default workspace",
        )
        self.tenants[tenant.tenant_id] = tenant
        self.workspaces[workspace.workspace_id] = workspace

    async def has_users(self) -> bool:
        return bool(self.users)

    async def create_user(self, user: SystemUser) -> SystemUser:
        if any(existing.username == user.username for existing in self.users.values()):
            raise ValueError(f"User already exists: {user.username}")
        self.users[user.user_id] = user
        return user

    async def get_user_by_username(self, username: str) -> SystemUser | None:
        return next((user for user in self.users.values() if user.username == username), None)

    async def get_user(self, user_id: str) -> SystemUser | None:
        return self.users.get(user_id)

    async def list_users(self) -> list[SystemUser]:
        return list(self.users.values())

    async def create_tenant(self, tenant: Tenant) -> Tenant:
        self.tenants[tenant.tenant_id] = tenant
        return tenant

    async def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self.tenants.get(tenant_id)

    async def list_tenants(self) -> list[Tenant]:
        return list(self.tenants.values())

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        self.workspaces[workspace.workspace_id] = workspace
        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        return self.workspaces.get(workspace_id)

    async def list_workspaces(self, tenant_id: str | None = None) -> list[Workspace]:
        workspaces = list(self.workspaces.values())
        if tenant_id:
            workspaces = [workspace for workspace in workspaces if workspace.tenant_id == tenant_id]
        return workspaces

    async def create_membership(self, membership: Membership) -> Membership:
        self.memberships[membership.membership_id] = membership
        return membership

    async def list_memberships_for_user(self, user_id: str) -> list[Membership]:
        return [membership for membership in self.memberships.values() if membership.user_id == user_id]

    async def write_audit_event(self, event: AuditEvent) -> AuditEvent:
        self.audit_events.append(event)
        return event

    async def list_audit_events(
        self, tenant_id: str | None = None, workspace_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        events = self.audit_events
        if tenant_id:
            events = [event for event in events if event.tenant_id == tenant_id]
        if workspace_id:
            events = [event for event in events if event.workspace_id == workspace_id]
        return list(reversed(events))[:limit]

    async def create_approval_request(self, approval: ApprovalRequest) -> ApprovalRequest:
        self.approvals[approval.approval_id] = approval
        return approval

    async def get_approval_request(self, approval_id: str) -> ApprovalRequest | None:
        return self.approvals.get(approval_id)

    async def update_approval_status(
        self, approval_id: str, status: ApprovalStatus, decided_by: str
    ) -> ApprovalRequest:
        approval = self.approvals.get(approval_id)
        if approval is None:
            raise KeyError(approval_id)
        updated = replace(approval, status=status, decided_by=decided_by, decided_at=utc_now())
        self.approvals[approval_id] = updated
        return updated

    async def list_approval_requests(
        self, tenant_id: str | None = None, workspace_id: str | None = None, status: ApprovalStatus | None = None
    ) -> list[ApprovalRequest]:
        approvals = list(self.approvals.values())
        if tenant_id:
            approvals = [approval for approval in approvals if approval.tenant_id == tenant_id]
        if workspace_id:
            approvals = [approval for approval in approvals if approval.workspace_id == workspace_id]
        if status:
            approvals = [approval for approval in approvals if approval.status == status]
        return list(reversed(approvals))

    async def get_policy(self, key: str, tenant_id: str | None = None, workspace_id: str | None = None) -> Any:
        return self.policies.get((tenant_id, workspace_id, key))

    async def set_policy(self, key: str, value: Any, tenant_id: str | None = None, workspace_id: str | None = None) -> None:
        self.policies[(tenant_id, workspace_id, key)] = value


class PostgresSystemRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        return self._pool

    @staticmethod
    def _parse_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    def _user_from_row(self, row: Any) -> SystemUser:
        return SystemUser(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            is_master_global=row["is_master_global"],
            is_active=row["is_active"],
            permission_version=row["permission_version"],
            created_at=self._parse_dt(row["created_at"]),
        )

    def _tenant_from_row(self, row: Any) -> Tenant:
        return Tenant(row["tenant_id"], row["name"], self._parse_dt(row["created_at"]))

    def _workspace_from_row(self, row: Any) -> Workspace:
        return Workspace(
            workspace_id=row["workspace_id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            slug=row["slug"],
            description=row["description"] or "",
            privacy=row["privacy"] or "team",
            created_at=self._parse_dt(row["created_at"]),
        )

    def _membership_from_row(self, row: Any) -> Membership:
        return Membership(
            membership_id=row["membership_id"],
            user_id=row["user_id"],
            tenant_id=row["tenant_id"],
            workspace_id=row["workspace_id"],
            roles=tuple(row["roles"] or []),
            created_at=self._parse_dt(row["created_at"]),
        )

    def _approval_from_row(self, row: Any) -> ApprovalRequest:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return ApprovalRequest(
            approval_id=row["approval_id"],
            action=row["action"],
            actor_user_id=row["actor_user_id"],
            tenant_id=row["tenant_id"],
            workspace_id=row["workspace_id"],
            payload_hash=row["payload_hash"],
            reason=row["reason"],
            status=ApprovalStatus(row["status"]),
            requested_at=self._parse_dt(row["requested_at"]),
            decided_at=self._parse_dt(row["decided_at"]) if row["decided_at"] else None,
            decided_by=row["decided_by"],
            metadata=metadata or {},
        )

    def _audit_from_row(self, row: Any) -> AuditEvent:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return AuditEvent(
            event_id=row["event_id"],
            actor_user_id=row["actor_user_id"],
            action=row["action"],
            tenant_id=row["tenant_id"],
            workspace_id=row["workspace_id"],
            result=row["result"],
            approval_id=row["approval_id"],
            model=row["model"],
            metadata=metadata or {},
            created_at=self._parse_dt(row["created_at"]),
        )

    async def has_users(self) -> bool:
        pool = await self._get_pool()
        return bool(await pool.fetchval("SELECT EXISTS (SELECT 1 FROM system_users)"))

    async def create_user(self, user: SystemUser) -> SystemUser:
        pool = await self._get_pool()
        await pool.execute(
            """
            INSERT INTO system_users
                (user_id, username, password_hash, display_name, is_master_global, is_active, permission_version, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            ON CONFLICT (username) DO NOTHING
            """,
            user.user_id,
            user.username,
            user.password_hash,
            user.display_name,
            user.is_master_global,
            user.is_active,
            user.permission_version,
            user.created_at,
        )
        created = await self.get_user(user.user_id)
        if created is None:
            raise ValueError(f"User already exists: {user.username}")
        return created

    async def get_user_by_username(self, username: str) -> SystemUser | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM system_users WHERE username=$1", username)
        return self._user_from_row(row) if row else None

    async def get_user(self, user_id: str) -> SystemUser | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM system_users WHERE user_id=$1", user_id)
        return self._user_from_row(row) if row else None

    async def list_users(self) -> list[SystemUser]:
        pool = await self._get_pool()
        return [self._user_from_row(row) for row in await pool.fetch("SELECT * FROM system_users ORDER BY created_at DESC")]

    async def create_tenant(self, tenant: Tenant) -> Tenant:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO system_tenants (tenant_id, name, created_at)
            VALUES ($1,$2,$3)
            ON CONFLICT (tenant_id) DO UPDATE SET name=EXCLUDED.name
            RETURNING *
            """,
            tenant.tenant_id,
            tenant.name,
            tenant.created_at,
        )
        return self._tenant_from_row(row)

    async def get_tenant(self, tenant_id: str) -> Tenant | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM system_tenants WHERE tenant_id=$1", tenant_id)
        return self._tenant_from_row(row) if row else None

    async def list_tenants(self) -> list[Tenant]:
        pool = await self._get_pool()
        return [self._tenant_from_row(row) for row in await pool.fetch("SELECT * FROM system_tenants ORDER BY name")]

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO system_workspaces (workspace_id, tenant_id, name, slug, description, privacy, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (workspace_id) DO UPDATE SET
                tenant_id=EXCLUDED.tenant_id, name=EXCLUDED.name, slug=EXCLUDED.slug,
                description=EXCLUDED.description, privacy=EXCLUDED.privacy
            RETURNING *
            """,
            workspace.workspace_id,
            workspace.tenant_id,
            workspace.name,
            workspace.slug,
            workspace.description,
            workspace.privacy,
            workspace.created_at,
        )
        return self._workspace_from_row(row)

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM system_workspaces WHERE workspace_id=$1", workspace_id)
        return self._workspace_from_row(row) if row else None

    async def list_workspaces(self, tenant_id: str | None = None) -> list[Workspace]:
        pool = await self._get_pool()
        if tenant_id:
            rows = await pool.fetch("SELECT * FROM system_workspaces WHERE tenant_id=$1 ORDER BY name", tenant_id)
        else:
            rows = await pool.fetch("SELECT * FROM system_workspaces ORDER BY name")
        return [self._workspace_from_row(row) for row in rows]

    async def create_membership(self, membership: Membership) -> Membership:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO system_memberships (membership_id, user_id, tenant_id, workspace_id, roles, created_at)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (user_id, workspace_id) DO UPDATE SET roles=EXCLUDED.roles
            RETURNING *
            """,
            membership.membership_id,
            membership.user_id,
            membership.tenant_id,
            membership.workspace_id,
            list(membership.roles),
            membership.created_at,
        )
        return self._membership_from_row(row)

    async def list_memberships_for_user(self, user_id: str) -> list[Membership]:
        pool = await self._get_pool()
        rows = await pool.fetch("SELECT * FROM system_memberships WHERE user_id=$1", user_id)
        return [self._membership_from_row(row) for row in rows]

    async def write_audit_event(self, event: AuditEvent) -> AuditEvent:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO system_audit_events
                (event_id, actor_user_id, action, tenant_id, workspace_id, result, approval_id, model, metadata, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10)
            RETURNING *
            """,
            event.event_id,
            event.actor_user_id,
            event.action,
            event.tenant_id,
            event.workspace_id,
            event.result,
            event.approval_id,
            event.model,
            json.dumps(event.metadata),
            event.created_at,
        )
        return self._audit_from_row(row)

    async def list_audit_events(
        self, tenant_id: str | None = None, workspace_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        pool = await self._get_pool()
        clauses: list[str] = []
        args: list[Any] = []
        if tenant_id:
            args.append(tenant_id)
            clauses.append(f"tenant_id=${len(args)}")
        if workspace_id:
            args.append(workspace_id)
            clauses.append(f"workspace_id=${len(args)}")
        args.append(limit)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await pool.fetch(
            f"SELECT * FROM system_audit_events {where} ORDER BY created_at DESC LIMIT ${len(args)}",
            *args,
        )
        return [self._audit_from_row(row) for row in rows]

    async def create_approval_request(self, approval: ApprovalRequest) -> ApprovalRequest:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO system_approval_requests
                (approval_id, action, actor_user_id, tenant_id, workspace_id, payload_hash, reason, status, requested_at, decided_at, decided_by, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb)
            RETURNING *
            """,
            approval.approval_id,
            approval.action,
            approval.actor_user_id,
            approval.tenant_id,
            approval.workspace_id,
            approval.payload_hash,
            approval.reason,
            approval.status.value,
            approval.requested_at,
            approval.decided_at,
            approval.decided_by,
            json.dumps(approval.metadata),
        )
        return self._approval_from_row(row)

    async def get_approval_request(self, approval_id: str) -> ApprovalRequest | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM system_approval_requests WHERE approval_id=$1", approval_id)
        return self._approval_from_row(row) if row else None

    async def update_approval_status(
        self, approval_id: str, status: ApprovalStatus, decided_by: str
    ) -> ApprovalRequest:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            UPDATE system_approval_requests
            SET status=$2, decided_by=$3, decided_at=$4
            WHERE approval_id=$1
            RETURNING *
            """,
            approval_id,
            status.value,
            decided_by,
            utc_now(),
        )
        if row is None:
            raise KeyError(approval_id)
        return self._approval_from_row(row)

    async def list_approval_requests(
        self, tenant_id: str | None = None, workspace_id: str | None = None, status: ApprovalStatus | None = None
    ) -> list[ApprovalRequest]:
        pool = await self._get_pool()
        clauses: list[str] = []
        args: list[Any] = []
        if tenant_id:
            args.append(tenant_id)
            clauses.append(f"tenant_id=${len(args)}")
        if workspace_id:
            args.append(workspace_id)
            clauses.append(f"workspace_id=${len(args)}")
        if status:
            args.append(status.value)
            clauses.append(f"status=${len(args)}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await pool.fetch(
            f"SELECT * FROM system_approval_requests {where} ORDER BY requested_at DESC",
            *args,
        )
        return [self._approval_from_row(row) for row in rows]

    async def get_policy(self, key: str, tenant_id: str | None = None, workspace_id: str | None = None) -> Any:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT value FROM system_policies
            WHERE key=$1 AND tenant_id IS NOT DISTINCT FROM $2 AND workspace_id IS NOT DISTINCT FROM $3
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            key,
            tenant_id,
            workspace_id,
        )
        if row is None:
            return None
        value = row["value"]
        return json.loads(value) if isinstance(value, str) else value

    async def set_policy(self, key: str, value: Any, tenant_id: str | None = None, workspace_id: str | None = None) -> None:
        pool = await self._get_pool()
        await pool.execute(
            """
            INSERT INTO system_policies (policy_id, tenant_id, workspace_id, key, value, updated_at)
            VALUES ($1,$2,$3,$4,$5::jsonb,$6)
            """,
            new_id("pol"),
            tenant_id,
            workspace_id,
            key,
            json.dumps(value),
            utc_now(),
        )


def default_tenant_and_workspace() -> tuple[Tenant, Workspace]:
    tenant = Tenant(tenant_id="default", name="Default")
    workspace = Workspace(
        workspace_id="default",
        tenant_id=tenant.tenant_id,
        name="Default",
        slug="default",
        description="Local-first default workspace",
    )
    return tenant, workspace


def membership_for_master(user_id: str, tenant_id: str = "default", workspace_id: str = "default") -> Membership:
    return Membership(
        membership_id=new_id("mbr"),
        user_id=user_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        roles=("master",),
    )
