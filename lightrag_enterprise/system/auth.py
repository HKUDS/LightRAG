from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any

import bcrypt
import jwt

from .models import Principal, SystemUser, new_id
from .permissions import MASTER_ROLE, permissions_for_roles
from .repositories import SystemRepository

BCRYPT_PASSWORD_PREFIX = "{bcrypt}"
DEFAULT_SYSTEM_TOKEN_SECRET = "lightrag-little-bull-local-dev-secret"


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return f"{BCRYPT_PASSWORD_PREFIX}{hashed}"


def verify_password(plain_password: str, stored_password: str) -> bool:
    if not stored_password.startswith(BCRYPT_PASSWORD_PREFIX):
        return stored_password == plain_password
    hashed = stored_password[len(BCRYPT_PASSWORD_PREFIX) :]
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


class SystemAuthService:
    def __init__(
        self,
        repository: SystemRepository,
        *,
        secret: str | None = None,
        algorithm: str = "HS256",
        expire_hours: int = 24,
    ) -> None:
        self.repository = repository
        self.secret = secret or os.getenv("LIGHTRAG_SYSTEM_TOKEN_SECRET") or os.getenv("TOKEN_SECRET")
        if self.secret is None and os.getenv("LIGHTRAG_SYSTEM_ALLOW_INSECURE_DEV_SECRET", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self.secret = DEFAULT_SYSTEM_TOKEN_SECRET
        self.algorithm = algorithm
        self.expire_hours = expire_hours

    def require_token_secret(self) -> None:
        if not self.secret:
            raise RuntimeError(
                "LIGHTRAG_SYSTEM_TOKEN_SECRET or TOKEN_SECRET must be set before enterprise tokens can be issued."
            )

    async def has_users(self) -> bool:
        return await self.repository.has_users()

    async def bootstrap_master(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
    ) -> SystemUser:
        if await self.repository.has_users():
            raise ValueError("System already has users; bootstrap is closed.")
        user = SystemUser(
            user_id=new_id("usr"),
            username=username,
            password_hash=hash_password(password),
            display_name=display_name or username,
            is_master_global=True,
        )
        return await self.repository.create_user(user)

    async def authenticate(self, username: str, password: str) -> tuple[SystemUser, Principal]:
        user = await self.repository.get_user_by_username(username)
        if user is None or not user.is_active or not verify_password(password, user.password_hash):
            raise ValueError("Incorrect credentials")
        return user, await self.principal_for_user(user)

    async def principal_for_user(self, user: SystemUser) -> Principal:
        memberships = await self.repository.list_memberships_for_user(user.user_id)
        roles: set[str] = {role for membership in memberships for role in membership.roles}
        workspace_ids = tuple(sorted({membership.workspace_id for membership in memberships}))
        tenant_id = memberships[0].tenant_id if memberships else None
        if user.is_master_global:
            roles.add(MASTER_ROLE)
            workspaces = await self.repository.list_workspaces()
            workspace_ids = tuple(sorted({workspace.workspace_id for workspace in workspaces}))
            if tenant_id is None and workspaces:
                tenant_id = workspaces[0].tenant_id
        permissions = permissions_for_roles(roles)
        return Principal(
            user_id=user.user_id,
            subject=user.username,
            tenant_id=tenant_id,
            is_master_global=user.is_master_global,
            roles=tuple(sorted(roles)),
            workspace_ids=workspace_ids,
            permission_version=user.permission_version,
            permissions=permissions,
        )

    def create_token(self, principal: Principal) -> str:
        self.require_token_secret()
        expire = datetime.now(timezone.utc) + timedelta(hours=self.expire_hours)
        payload: dict[str, Any] = principal.to_token_payload()
        payload["exp"] = expire
        payload["role"] = "master" if principal.is_master_global else (principal.roles[0] if principal.roles else "user")
        payload["metadata"] = {"auth_mode": "enterprise"}
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict[str, Any]:
        self.require_token_secret()
        return jwt.decode(token, self.secret, algorithms=[self.algorithm])

    async def principal_from_token(self, token: str) -> Principal:
        payload = self.decode_token(token)
        user_id = payload["user_id"]
        user = await self.repository.get_user(user_id)
        if user is None or not user.is_active:
            raise ValueError("Invalid token principal")
        if int(payload.get("permission_version", 0)) != user.permission_version:
            raise ValueError("Token permission version is stale")
        return await self.principal_for_user(user)
