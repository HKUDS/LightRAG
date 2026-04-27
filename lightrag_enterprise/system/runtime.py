from __future__ import annotations

import os
from functools import lru_cache

from fastapi import HTTPException, Request, status

from .access import AccessControlService
from .approvals import ApprovalService
from .audit import AuditService
from .auth import SystemAuthService
from .db import get_database_url
from .repositories import InMemorySystemRepository, PostgresSystemRepository, SystemRepository


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def little_bull_functional_enabled() -> bool:
    return env_flag("LITTLE_BULL_FUNCTIONAL_ENABLED", True)


def private_strict_enabled() -> bool:
    return env_flag("LITTLE_BULL_PRIVATE_STRICT", True)


def approvals_enforced() -> bool:
    return env_flag("LITTLE_BULL_APPROVALS_ENFORCED", True)


@lru_cache(maxsize=1)
def get_system_repository() -> SystemRepository:
    database_url = get_database_url()
    if database_url:
        return PostgresSystemRepository(database_url)
    return InMemorySystemRepository()


@lru_cache(maxsize=1)
def get_system_auth_service() -> SystemAuthService:
    return SystemAuthService(get_system_repository())


@lru_cache(maxsize=1)
def get_access_service() -> AccessControlService:
    return AccessControlService()


@lru_cache(maxsize=1)
def get_audit_service() -> AuditService:
    return AuditService(get_system_repository())


@lru_cache(maxsize=1)
def get_approval_service() -> ApprovalService:
    return ApprovalService(get_system_repository())


async def require_principal(request: Request):
    if not little_bull_functional_enabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Little Bull functional API is disabled")

    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    try:
        return await get_system_auth_service().principal_from_token(token)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Little Bull token") from exc

