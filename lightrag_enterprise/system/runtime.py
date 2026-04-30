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


def little_bull_graph_v2_enabled() -> bool:
    return env_flag("LITTLE_BULL_GRAPH_V2_ENABLED", False)


def little_bull_qdrant_data_plane_enabled() -> bool:
    return env_flag("LITTLE_BULL_QDRANT_DATA_PLANE_ENABLED", False)


def little_bull_postgres_control_plane_required() -> bool:
    return env_flag("LITTLE_BULL_POSTGRES_CONTROL_PLANE_REQUIRED", True)


def little_bull_obsidian_workspace_enabled() -> bool:
    return env_flag("LITTLE_BULL_OBSIDIAN_WORKSPACE_ENABLED", False)


def little_bull_clean_knowledge_base_allowed() -> bool:
    return env_flag("LITTLE_BULL_CLEAN_KNOWLEDGE_BASE_ALLOWED", False)


def in_memory_system_repository_allowed() -> bool:
    return env_flag("LIGHTRAG_SYSTEM_ALLOW_IN_MEMORY_REPOSITORY", False)


def private_strict_enabled() -> bool:
    return env_flag("LITTLE_BULL_PRIVATE_STRICT", True)


def approvals_enforced() -> bool:
    return env_flag("LITTLE_BULL_APPROVALS_ENFORCED", True)


@lru_cache(maxsize=1)
def get_system_repository() -> SystemRepository:
    database_url = get_database_url()
    if database_url:
        return PostgresSystemRepository(database_url)
    if (
        little_bull_functional_enabled()
        and little_bull_postgres_control_plane_required()
        and not in_memory_system_repository_allowed()
    ):
        raise RuntimeError(
            "LITTLE_BULL_FUNCTIONAL_ENABLED=true requires LIGHTRAG_SYSTEM_DATABASE_URL "
            "or DATABASE_URL. Set LIGHTRAG_SYSTEM_ALLOW_IN_MEMORY_REPOSITORY=true only "
            "for local tests or throwaway development."
        )
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
