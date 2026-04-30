from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag_enterprise.system import AuditService, MANAGER_ROLE, MASTER_ROLE, Principal
from lightrag_enterprise.system.permissions import permissions_for_roles
from lightrag_enterprise.system.policy_keys import PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY
from lightrag_enterprise.system.repositories import InMemorySystemRepository
from lightrag_enterprise.system.router import create_system_router


def _principal(role: str, *, master: bool = False) -> Principal:
    roles = (MASTER_ROLE,) if master else (role,)
    return Principal(
        user_id=f"usr_{role}",
        subject=role,
        tenant_id="default",
        is_master_global=master,
        roles=roles,
        workspace_ids=("default",),
        permission_version=1,
        permissions=permissions_for_roles(roles),
    )


def _client(repo: InMemorySystemRepository, principal: Principal, monkeypatch) -> TestClient:
    import lightrag_enterprise.system.router as system_router

    monkeypatch.setattr(system_router, "get_system_repository", lambda: repo)
    monkeypatch.setattr(system_router, "get_audit_service", lambda: AuditService(repo))
    app = FastAPI()
    app.include_router(create_system_router())
    app.dependency_overrides[system_router.require_principal] = lambda: principal
    return TestClient(app)


def _payload() -> dict:
    return {
        "enabled": True,
        "tenant_id": "default",
        "workspace_id": "default",
        "provider": "openrouter",
        "binding": "openai",
        "binding_host": "https://openrouter.ai/api/v1",
        "allowed_model_ids": ["openai/gpt-4o-mini"],
        "allowed_confidentiality": ["sensivel", "privado"],
        "expires_at": "2099-01-01T00:00:00Z",
        "approval_id": "apr_policy",
        "reason": "MASTER approved OpenRouter for private workspace data.",
        "ticket_id": "LB-42",
    }


def test_master_can_set_hosted_private_llm_exception_policy(monkeypatch):
    repo = InMemorySystemRepository()
    master = _principal(MASTER_ROLE, master=True)
    client = _client(repo, master, monkeypatch)

    response = client.post("/system/policies/private-data/hosted-llm-exception", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["policy_hash"]
    policy = repo.policies[("default", "default", PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY)]
    assert policy["provider"] == "openrouter"
    assert policy["binding_host"] == "https://openrouter.ai/api/v1"
    assert policy["allowed_model_ids"] == ["openai/gpt-4o-mini"]
    assert policy["approved_by"] == master.user_id
    events = repo.audit_events
    assert events[-1].action == "little_bull.policies.private_data.hosted_llm_exception.update"
    assert events[-1].approval_id == "apr_policy"
    assert events[-1].metadata["policy_key"] == PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY
    assert events[-1].metadata["policy_hash"] == body["policy_hash"]


def test_non_master_cannot_set_hosted_private_llm_exception_policy(monkeypatch):
    repo = InMemorySystemRepository()
    manager = _principal(MANAGER_ROLE)
    client = _client(repo, manager, monkeypatch)

    response = client.post("/system/policies/private-data/hosted-llm-exception", json=_payload())

    assert response.status_code == 403
    assert repo.policies == {}


def test_policy_endpoint_rejects_expired_policy(monkeypatch):
    repo = InMemorySystemRepository()
    master = _principal(MASTER_ROLE, master=True)
    client = _client(repo, master, monkeypatch)
    payload = _payload()
    payload["expires_at"] = "2020-01-01T00:00:00Z"

    response = client.post("/system/policies/private-data/hosted-llm-exception", json=payload)

    assert response.status_code == 400
    assert repo.policies == {}
