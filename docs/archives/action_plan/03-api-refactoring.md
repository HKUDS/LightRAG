# Action Plan: API Refactoring & WebUI Cleanup

## Objective

Refactor the Tenant API to follow multi-tenant best practices by removing explicit `tenant_id` from path parameters and relying on context injection. Update the WebUI to consume these new endpoints.

## Phase 1: Backend API Refactoring (`lightrag/api/routers/tenant_routes.py`)

### 1.1 Introduce Context-Aware Endpoints

- Create `/tenants/me` endpoint to retrieve current tenant info based on context.
- Create `/knowledge-bases` endpoints (GET, POST) that infer `tenant_id` from `TenantContext`.
- Create `/knowledge-bases/{kb_id}` endpoints (GET, PUT, DELETE) that infer `tenant_id` from `TenantContext`.

### 1.2 Deprecate/Secure Explicit Path Endpoints

- Mark `/tenants/{tenant_id}` endpoints as deprecated or restrict them to "super-admin" roles only.
- Ensure `list_tenants` (`/tenants`) is protected or strictly rate-limited.

### 1.3 Update Dependency Injection

- Ensure `get_tenant_context` is used consistently across all new endpoints.

## Phase 2: Frontend WebUI Refactoring (`lightrag_webui/src/api/tenant.ts`)

### 2.1 Update API Client

- Implement `fetchCurrentTenant` using `/tenants/me`.
- Update `fetchKnowledgeBasesPaginated` to use `/knowledge-bases`.
- Update `createKnowledgeBase`, `updateKnowledgeBase`, `deleteKnowledgeBase` to use `/knowledge-bases` endpoints.

### 2.2 Context Management

- Ensure the frontend sets the `X-Tenant-ID` header (or relies on subdomain) for all requests.
- Remove `tenantId` arguments from API functions where it's no longer needed (or make them optional for backward compatibility).

## Phase 3: Cleanup & Verification

### 3.1 Remove Dead Code

- Remove unused API functions in frontend.
- Remove unused routes in backend (after verification).

### 3.2 Testing

- Verify that a user logged in as Tenant A cannot access Tenant B's data via the new endpoints.
- Verify that the WebUI still functions correctly with the new API structure.
