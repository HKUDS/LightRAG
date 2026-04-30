from __future__ import annotations

import os
import time
from uuid import uuid4
from typing import Any

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.requires_db,
]

TRUE_VALUES = {"1", "true", "yes", "on"}
NO_CONTEXT_FALLBACK = "No relevant context found for the query."


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is required for the Little Bull real API smoke")
    return value


def _database_url() -> str:
    return os.getenv("LIGHTRAG_SYSTEM_DATABASE_URL") or os.getenv("DATABASE_URL") or ""


def _assert_local_test_database(database_url: str) -> None:
    lowered = database_url.lower()
    allow_non_test = _truthy(os.getenv("LITTLE_BULL_E2E_ALLOW_NON_TEST_DB"))
    local_markers = ("localhost", "127.0.0.1", "[::1]", "host.docker.internal")
    if not any(marker in lowered for marker in local_markers) and not allow_non_test:
        pytest.skip(
            "Little Bull E2E requires a local PostgreSQL URL, or "
            "LITTLE_BULL_E2E_ALLOW_NON_TEST_DB=1 for an explicit override"
        )

    database_name = lowered.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0]
    test_markers = ("test", "e2e", "smoke")
    if not any(marker in database_name for marker in test_markers) and not allow_non_test:
        pytest.skip(
            "Little Bull E2E requires a dedicated test/e2e/smoke database name, or "
            "LITTLE_BULL_E2E_ALLOW_NON_TEST_DB=1 for an explicit override"
        )


def _assert_llm_api_configured() -> None:
    llm_signals = (
        "LLM_BINDING_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_HOST",
        "LLM_BINDING_HOST",
        "LITTLE_BULL_PRIVATE_LOCAL_HOST",
        "LITTLE_BULL_PRIVATE_LOCAL_MODEL",
    )
    if not any(os.getenv(name) for name in llm_signals):
        pytest.skip(
            "Little Bull E2E requires an LLM API/local API signal such as "
            "OLLAMA_HOST, LLM_BINDING_HOST, or a hosted provider API key"
        )


def _require_smoke_config(*, require_llm: bool = True) -> tuple[str, str, str]:
    if not _truthy(os.getenv("LITTLE_BULL_E2E")):
        pytest.skip("Set LITTLE_BULL_E2E=1 to run the real Little Bull API smoke")
    database_url = _database_url()
    if not database_url:
        pytest.skip("LIGHTRAG_SYSTEM_DATABASE_URL or DATABASE_URL is required")
    _assert_local_test_database(database_url)
    if require_llm:
        _assert_llm_api_configured()
    return (
        os.getenv("LIGHTRAG_API_BASE_URL", "http://127.0.0.1:9621").rstrip("/"),
        _require_env("LITTLE_BULL_E2E_MASTER_USERNAME"),
        _require_env("LITTLE_BULL_E2E_MASTER_PASSWORD"),
    )


def _fail_response(response: Any, label: str) -> None:
    detail = response.text[:800] if hasattr(response, "text") else str(response)
    pytest.fail(f"{label} failed with HTTP {response.status_code}: {detail}")


def _bootstrap_if_requested(client: Any, username: str, password: str) -> None:
    if not _truthy(os.getenv("LITTLE_BULL_E2E_BOOTSTRAP")):
        return
    token = _require_env("LITTLE_BULL_BOOTSTRAP_TOKEN")
    response = client.post(
        "/system/bootstrap-master",
        headers={"X-Little-Bull-Bootstrap-Token": token},
        json={
            "username": username,
            "password": password,
            "display_name": "Little Bull E2E MASTER",
            "tenant_name": "Little Bull E2E",
            "workspace_name": "Default",
        },
    )
    if response.status_code not in {200, 409}:
        _fail_response(response, "bootstrap-master")


def _login(client: Any, username: str, password: str) -> str:
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if response.status_code != 200:
        _fail_response(response, "auth/login")
    payload = response.json()
    assert payload["token_type"] == "bearer"
    principal = payload["principal"]
    assert principal["is_master_global"] is True
    assert "default" in principal["workspace_ids"]
    return payload["access_token"]


def _workspace_id(client: Any, headers: dict[str, str]) -> str:
    configured = os.getenv("LITTLE_BULL_E2E_WORKSPACE_ID")
    if configured:
        return configured
    response = client.get("/system/workspaces", headers=headers)
    if response.status_code != 200:
        _fail_response(response, "system/workspaces")
    workspaces = response.json()["workspaces"]
    assert any(workspace["workspace_id"] == "default" for workspace in workspaces)
    return "default"


def _phase3_workspace_id(client: Any, headers: dict[str, str]) -> str:
    configured = os.getenv("LITTLE_BULL_E2E_PHASE3_WORKSPACE_ID")
    if configured:
        return configured
    if not _truthy(os.getenv("LITTLE_BULL_E2E_PHASE3_CREATE_WORKSPACE")):
        return _workspace_id(client, headers)

    suffix = uuid4().hex[:12]
    workspace_id = f"phase3-{suffix}"
    response = client.post(
        "/little-bull/admin/knowledge-bases",
        headers=headers,
        json={
            "workspace_id": workspace_id,
            "name": f"Little Bull Phase 3 {suffix}",
            "slug": workspace_id,
            "description": "Non-destructive Phase 3 data-plane pilot workspace.",
            "privacy": "team",
        },
    )
    if response.status_code != 200:
        _fail_response(response, "little-bull/admin/knowledge-bases")
    payload = response.json()
    assert payload["workspace_id"] == workspace_id
    return workspace_id


def _attach_data_plane(client: Any, headers: dict[str, str], workspace_id: str) -> None:
    response = client.post(
        f"/little-bull/admin/knowledge-bases/{workspace_id}/attach-data-plane",
        headers=headers,
    )
    if response.status_code != 200:
        _fail_response(response, "little-bull/admin/knowledge-bases/attach-data-plane")
    payload = response.json()
    assert payload["workspace_id"] == workspace_id
    assert payload["data_plane_attached"] is True


def _ensure_phase4_classification(
    client: Any,
    headers: dict[str, str],
    workspace_id: str,
) -> tuple[str, str]:
    configured_group = os.getenv("LITTLE_BULL_E2E_GROUP_ID")
    configured_subgroup = os.getenv("LITTLE_BULL_E2E_SUBGROUP_ID")
    if configured_group and configured_subgroup:
        return configured_group, configured_subgroup

    suffix = uuid4().hex[:12]
    group_response = client.post(
        "/little-bull/knowledge-groups",
        headers=headers,
        params={"workspace_id": workspace_id},
        json={
            "name": f"Smoke Group {suffix}",
            "slug": f"smoke-group-{suffix}",
            "privacy": "team",
        },
    )
    if group_response.status_code != 200:
        _fail_response(group_response, "little-bull/knowledge-groups")
    group_id = group_response.json()["group_id"]

    subgroup_response = client.post(
        "/little-bull/knowledge-subgroups",
        headers=headers,
        params={"workspace_id": workspace_id},
        json={
            "group_id": group_id,
            "name": f"Smoke Subgroup {suffix}",
            "slug": f"smoke-subgroup-{suffix}",
            "privacy": "team",
        },
    )
    if subgroup_response.status_code != 200:
        _fail_response(subgroup_response, "little-bull/knowledge-subgroups")
    subgroup_id = subgroup_response.json()["subgroup_id"]
    return group_id, subgroup_id


def _query_payload(workspace_id: str) -> dict[str, Any]:
    confidentiality = os.getenv("LITTLE_BULL_E2E_CONFIDENTIALITY", "normal").strip().lower()
    model_profile = os.getenv("LITTLE_BULL_E2E_MODEL_PROFILE", "equilibrado").strip().lower()
    return {
        "workspace_id": workspace_id,
        "query": "Reply with one short sentence confirming Little Bull API smoke is working.",
        "mode": os.getenv("LITTLE_BULL_E2E_QUERY_MODE", "bypass"),
        "response_type": "Single Sentence",
        "include_references": False,
        "confidentiality": confidentiality,
        "model_profile": model_profile,
    }


def _expects_private_runtime(payload: dict[str, Any]) -> bool:
    if _truthy(os.getenv("LITTLE_BULL_E2E_HOSTED_PRIVATE_EXCEPTION")):
        return payload["model_profile"] == "privado"
    return payload["confidentiality"] in {"sensivel", "privado"} or payload["model_profile"] == "privado"


def _expects_hosted_private_exception(payload: dict[str, Any]) -> bool:
    return (
        _truthy(os.getenv("LITTLE_BULL_E2E_HOSTED_PRIVATE_EXCEPTION"))
        and payload["model_profile"] != "privado"
        and payload["confidentiality"] in {"sensivel", "privado"}
    )


def _assert_smoke_response_proves_llm_call(response: str) -> None:
    normalized = response.strip()
    assert normalized
    assert normalized != NO_CONTEXT_FALLBACK, (
        "Little Bull real API smoke received the no-context fallback instead of "
        "evidence that the configured LLM answered the smoke prompt."
    )


def _upload_smoke_document(client: Any, headers: dict[str, str], workspace_id: str) -> tuple[str, str]:
    group_id, subgroup_id = _ensure_phase4_classification(client, headers, workspace_id)
    phrase = f"little bull indexed smoke phrase {uuid4().hex}"
    filename = f"little-bull-smoke-{uuid4().hex}.txt"
    response = client.post(
        "/little-bull/documents/upload",
        headers=headers,
        params={
            "workspace_id": workspace_id,
            "group_id": group_id,
            "subgroup_id": subgroup_id,
            "confidentiality": "normal",
        },
        files={"file": (filename, f"{phrase}\n", "text/plain")},
    )
    if response.status_code != 200:
        _fail_response(response, "little-bull/documents/upload")
    payload = response.json()
    assert payload["workspace_id"] == workspace_id
    assert payload["status"] in {"success", "duplicated"}
    return filename, phrase


def _wait_for_processed_document(
    client: Any,
    headers: dict[str, str],
    workspace_id: str,
    filename: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + int(os.getenv("LITTLE_BULL_E2E_UPLOAD_TIMEOUT_SECONDS", "180"))
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(
            "/little-bull/documents",
            headers=headers,
            params={"workspace_id": workspace_id, "page": 1, "page_size": 100},
        )
        if response.status_code != 200:
            _fail_response(response, "little-bull/documents")
        last_payload = response.json()
        for document in last_payload["documents"]:
            if filename in {document.get("title"), document.get("file_path")}:
                status = str(document.get("status", "")).lower()
                if status == "processed" or status.endswith(".processed"):
                    return document
                if status == "failed" or status.endswith(".failed"):
                    pytest.fail(f"Uploaded smoke document failed indexing: {document}")
        time.sleep(2)
    pytest.fail(f"Uploaded smoke document was not processed in time. Last documents payload: {last_payload}")


def test_little_bull_real_api_data_plane_attach_smoke():
    if not _truthy(os.getenv("LITTLE_BULL_E2E_DATA_PLANE_ATTACH")):
        pytest.skip(
            "Set LITTLE_BULL_E2E_DATA_PLANE_ATTACH=1 to run the no-LLM data-plane attach smoke"
        )
    httpx = pytest.importorskip("httpx")
    base_url, username, password = _require_smoke_config(require_llm=False)

    with httpx.Client(base_url=base_url, timeout=90, follow_redirects=True) as client:
        try:
            auth_status = client.get("/auth-status")
        except httpx.ConnectError as exc:
            pytest.skip(f"LightRAG API server is not reachable at {base_url}: {exc}")
        if auth_status.status_code >= 500:
            _fail_response(auth_status, "auth-status")

        _bootstrap_if_requested(client, username, password)
        token = _login(client, username, password)
        headers = {"Authorization": f"Bearer {token}"}
        workspace_id = _phase3_workspace_id(client, headers)
        _attach_data_plane(client, headers, workspace_id)

        bases_response = client.get("/little-bull/admin/knowledge-bases", headers=headers)
        if bases_response.status_code != 200:
            _fail_response(bases_response, "little-bull/admin/knowledge-bases")
        bases = bases_response.json()["knowledge_bases"]
        assert any(
            base["workspace_id"] == workspace_id and base["data_plane_attached"] is True
            for base in bases
        )

        documents_response = client.get(
            "/little-bull/documents",
            headers=headers,
            params={"workspace_id": workspace_id, "page": 1, "page_size": 20},
        )
        if documents_response.status_code != 200:
            _fail_response(documents_response, "little-bull/documents")
        documents_payload = documents_response.json()
        assert "documents" in documents_payload
        assert "status_counts" in documents_payload

        activity_response = client.get(
            "/little-bull/activity",
            headers=headers,
            params={"workspace_id": workspace_id, "limit": 20},
        )
        if activity_response.status_code != 200:
            _fail_response(activity_response, "little-bull/activity")
        activity = activity_response.json()["activity"]
        assert any(
            event["action"] == "little_bull.workspaces.manage"
            and event["result"] == "data_plane_attached"
            for event in activity
        )

        audit_response = client.get("/audit/events", headers=headers, params={"limit": 100})
        if audit_response.status_code != 200:
            _fail_response(audit_response, "audit/events")
        events = audit_response.json()["events"]
        assert any(
            event["action"] == "little_bull.workspaces.manage"
            and event["workspace_id"] == workspace_id
            and event["result"] == "data_plane_attached"
            for event in events
        )


def test_little_bull_real_api_upload_queue_audit_smoke():
    if not _truthy(os.getenv("LITTLE_BULL_E2E_UPLOAD_QUEUE")):
        pytest.skip("Set LITTLE_BULL_E2E_UPLOAD_QUEUE=1 to run upload queue/audit smoke")
    httpx = pytest.importorskip("httpx")
    base_url, username, password = _require_smoke_config(require_llm=False)

    with httpx.Client(base_url=base_url, timeout=90, follow_redirects=True) as client:
        _bootstrap_if_requested(client, username, password)
        token = _login(client, username, password)
        headers = {"Authorization": f"Bearer {token}"}
        workspace_id = _phase3_workspace_id(client, headers)
        _attach_data_plane(client, headers, workspace_id)
        filename, _phrase = _upload_smoke_document(client, headers, workspace_id)

        documents_response = client.get(
            "/little-bull/documents",
            headers=headers,
            params={"workspace_id": workspace_id, "page": 1, "page_size": 100},
        )
        if documents_response.status_code != 200:
            _fail_response(documents_response, "little-bull/documents")
        documents = documents_response.json()["documents"]
        assert any(filename in {document.get("title"), document.get("file_path")} for document in documents)

        audit_response = client.get("/audit/events", headers=headers, params={"limit": 100})
        if audit_response.status_code != 200:
            _fail_response(audit_response, "audit/events")
        events = audit_response.json()["events"]
        upload_events = [
            event
            for event in events
            if event["action"] == "little_bull.documents.upload"
            and event["workspace_id"] == workspace_id
            and event["result"] == "queued"
        ]
        assert upload_events
        assert upload_events[0]["metadata"]["track_id"]
        assert upload_events[0]["metadata"]["file_name"] == filename


def test_little_bull_real_api_query_smoke():
    httpx = pytest.importorskip("httpx")
    base_url, username, password = _require_smoke_config()

    with httpx.Client(base_url=base_url, timeout=90, follow_redirects=True) as client:
        try:
            auth_status = client.get("/auth-status")
        except httpx.ConnectError as exc:
            pytest.skip(f"LightRAG API server is not reachable at {base_url}: {exc}")
        if auth_status.status_code >= 500:
            _fail_response(auth_status, "auth-status")

        _bootstrap_if_requested(client, username, password)
        token = _login(client, username, password)
        headers = {"Authorization": f"Bearer {token}"}
        workspace_id = _workspace_id(client, headers)
        _attach_data_plane(client, headers, workspace_id)
        payload = _query_payload(workspace_id)

        anonymous_response = client.post("/little-bull/query", json=payload)
        assert anonymous_response.status_code in {401, 403}

        query_response = client.post("/little-bull/query", headers=headers, json=payload)
        if query_response.status_code != 200:
            _fail_response(query_response, "little-bull/query")
        query_payload = query_response.json()
        assert query_payload["workspace_id"] == workspace_id
        assert query_payload["model_profile"] == payload["model_profile"]
        _assert_smoke_response_proves_llm_call(query_payload["response"])

        activity_response = client.get(
            "/little-bull/activity",
            headers=headers,
            params={"workspace_id": workspace_id, "limit": 20},
        )
        if activity_response.status_code != 200:
            _fail_response(activity_response, "little-bull/activity")
        activity = activity_response.json()["activity"]
        assert any(
            event["action"] == "little_bull.query" and event["result"] == "success"
            for event in activity
        )

        audit_response = client.get("/audit/events", headers=headers, params={"limit": 100})
        if audit_response.status_code != 200:
            _fail_response(audit_response, "audit/events")
        events = audit_response.json()["events"]
        query_events = [
            event
            for event in events
            if event["action"] == "little_bull.query"
            and event["workspace_id"] == workspace_id
        ]
        success_events = [event for event in query_events if event["result"] == "success"]
        assert success_events
        private_gateway = success_events[0]["metadata"]["private_gateway"]
        if _expects_hosted_private_exception(payload):
            assert any(event["result"] == "allowed" for event in query_events)
            assert private_gateway["hosted_private_exception"] is True
            assert private_gateway["requires_private_runtime"] is False
        elif _expects_private_runtime(payload):
            assert any(event["result"] == "allowed" for event in query_events)
            assert private_gateway["requires_private_runtime"] is True
        else:
            assert private_gateway["requires_private_runtime"] is False


def test_little_bull_real_api_upload_index_query_smoke():
    if not _truthy(os.getenv("LITTLE_BULL_E2E_UPLOAD")):
        pytest.skip("Set LITTLE_BULL_E2E_UPLOAD=1 to run upload/index/query smoke")
    httpx = pytest.importorskip("httpx")
    base_url, username, password = _require_smoke_config()

    with httpx.Client(base_url=base_url, timeout=180, follow_redirects=True) as client:
        _bootstrap_if_requested(client, username, password)
        token = _login(client, username, password)
        headers = {"Authorization": f"Bearer {token}"}
        workspace_id = _workspace_id(client, headers)
        _attach_data_plane(client, headers, workspace_id)
        filename, phrase = _upload_smoke_document(client, headers, workspace_id)
        document = _wait_for_processed_document(client, headers, workspace_id, filename)

        payload = {
            "workspace_id": workspace_id,
            "query": f"What exact smoke phrase appears in {filename}?",
            "mode": "naive",
            "response_type": "Single Sentence",
            "include_references": True,
            "include_chunk_content": True,
            "confidentiality": "normal",
            "model_profile": os.getenv("LITTLE_BULL_E2E_MODEL_PROFILE", "equilibrado").strip().lower(),
        }
        response = client.post("/little-bull/query", headers=headers, json=payload)
        if response.status_code != 200:
            _fail_response(response, "little-bull/query indexed")
        query_payload = response.json()
        _assert_smoke_response_proves_llm_call(query_payload["response"])
        assert query_payload["references"], (
            "Indexed query returned no references, so it did not prove retrieval against the uploaded document."
        )
        assert any(
            phrase in str(reference) or document["id"] in str(reference)
            for reference in query_payload["references"]
        )
