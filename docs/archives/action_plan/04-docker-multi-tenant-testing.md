# Action Plan: Multi-Tenant Docker Testing & UI Updates

## Objective

Enable a seamless multi-tenant testing environment using Docker and update the WebUI to fully support the new implicit tenant context architecture.

## Phase 1: Docker Environment (`starter/`)

### 1.1 Review & Fix `docker-compose.yml`

- **Current Status**: `starter/docker-compose.yml` defines a full stack (Postgres, Redis, API, WebUI).
- **Action**:
  - Verify build contexts are correct relative to `starter/` directory.
  - Ensure `lightrag-api` and `lightrag-webui` services are building from the latest code.
  - Check network configuration to ensure containers can communicate.

### 1.2 Create Testing Script (`starter/test_multi_tenant.sh`)

- **Goal**: Automate the spin-up and verification process.
- **Steps**:
  1. Teardown existing containers and volumes (`docker-compose down -v`).
  2. Build and start services (`docker-compose up -d --build`).
  3. Wait for services to be healthy.
  4. Execute `scripts/init_demo_tenants.py` to populate data.
  5. Run `curl` integration tests to verify tenant isolation (e.g., Tenant A cannot see Tenant B's data).

## Phase 2: WebUI Analysis & Updates (`lightrag_webui/`)

### 2.1 Analyze Existing UI

- **Current State**: The UI likely uses the old API client or hardcoded paths.
- **Needs Removal**:
  - Direct usage of `tenantId` in URL paths in components (e.g., `axios.get('/tenants/${id}/...')`).
- **Needs Update**:
  - **API Client**: Already updated in `src/api/tenant.ts`.
  - **State Management**: Ensure a global `useTenant` hook or store exists to manage the currently selected `tenantId`.
  - **Interceptors**: Add an Axios interceptor to automatically inject `X-Tenant-ID` header from the global state into all requests.
  - **Components**: Update `KnowledgeBaseList`, `DocumentList`, etc., to remove `tenantId` props if they are just passing it down for API calls.

### 2.2 Implementation Steps

- [ ] **Step 1**: Create/Update `TenantContext` or Store to manage `currentTenant`.
- [ ] **Step 2**: Configure Axios interceptor in `src/api/client.ts` (or similar) to attach `X-Tenant-ID`.
- [ ] **Step 3**: Update `TenantSelector` component to set the global `currentTenant`.
- [ ] **Step 4**: Refactor main views to rely on the global tenant context instead of URL params for tenant ID.

## Phase 3: Execution & Verification

### 3.1 Run the Test Suite

- Execute `starter/test_multi_tenant.sh`.
- Verify logs show successful data population.
- Verify UI allows switching tenants and shows correct data for each.

### 3.2 Manual Verification

- Open `http://localhost:3000`.
- Login/Select "Acme Corp".
- Verify "Production KB" is visible.
- Switch to "TechStart".
- Verify "Main KB" is visible and "Production KB" is GONE.
