from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    MANAGER = "manager"
    AGENT = "agent"
    VIEWER = "viewer"
    SERVICE = "service"


ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {"*"},
    Role.MANAGER: {"read", "write", "query", "ingest", "audit"},
    Role.AGENT: {"read", "query", "ticket:write"},
    Role.VIEWER: {"read", "query"},
    Role.SERVICE: {"read", "write", "query", "ingest", "audit", "job:run"},
}


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: set[Role]
    workspaces: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ResourceScope:
    tenant_id: str
    workspace: str
    action: str
    document_acl: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


def evaluate_access(principal: Principal, resource: ResourceScope) -> AccessDecision:
    if principal.tenant_id != resource.tenant_id:
        return AccessDecision(False, "Tenant mismatch.")
    if principal.workspaces and resource.workspace not in principal.workspaces:
        return AccessDecision(False, "Workspace is outside principal scope.")
    if resource.document_acl and principal.subject not in resource.document_acl:
        return AccessDecision(False, "Document ACL denied.")
    permissions = set().union(*(ROLE_PERMISSIONS[role] for role in principal.roles))
    if "*" in permissions or resource.action in permissions:
        return AccessDecision(True, "Allowed by RBAC policy.")
    return AccessDecision(False, "Role lacks required permission.")


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
    "cpf_like": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
}


PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|system) instructions", re.I),
    re.compile(r"reveal (the )?(system|developer|hidden) prompt", re.I),
    re.compile(r"disable (safety|policy|guardrails)", re.I),
    re.compile(r"you are now (developer|system|root)", re.I),
    re.compile(r"call tool .* without (approval|permission)", re.I),
]


def detect_pii(text: str) -> dict[str, list[str]]:
    return {
        name: pattern.findall(text)
        for name, pattern in PII_PATTERNS.items()
        if pattern.findall(text)
    }


def mask_pii(text: str) -> str:
    masked = text
    for name, pattern in PII_PATTERNS.items():
        masked = pattern.sub(f"[MASKED_{name.upper()}]", masked)
    return masked


def detect_prompt_injection(text: str) -> list[str]:
    return [
        pattern.pattern
        for pattern in PROMPT_INJECTION_PATTERNS
        if pattern.search(text)
    ]
