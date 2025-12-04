# Action Plan: Secure Tenant API

## Problem
The endpoint `GET /api/v1/tenants` exposes a list of all tenants, which violates multi-tenant isolation principles. This endpoint allows any user (or unauthenticated attacker) to enumerate all customers, posing a significant security risk (Information Disclosure).

## Goal
Remove public access to tenant listing. Ensure tenant management (listing) is restricted to administrators via a secure endpoint.

## Steps

### 1. Audit & Verification
- [ ] Verify the current implementation of `GET /api/v1/tenants`.
- [ ] Confirm that `lightrag/api/routers/tenant_routes.py` does NOT contain the list endpoint.
- [ ] Confirm that `lightrag/api/routers/admin_routes.py` contains the list endpoint under `/api/v1/admin/tenants`.
- [ ] Investigate why `GET /api/v1/tenants` is currently accessible (if it is).

### 2. Remediation
- [ ] Ensure `GET /api/v1/tenants` is removed or returns 404/403.
- [ ] Ensure `GET /api/v1/admin/tenants` is accessible only to admins.
- [ ] Verify `lightrag/api/lightrag_server.py` router mounting.

### 3. Testing
- [ ] Update `test_multitenant.sh` to:
    - [ ] Verify `GET /api/v1/tenants` returns 404 or 403.
    - [ ] Verify `GET /api/v1/admin/tenants` works (with admin auth).
    - [ ] Verify `GET /api/v1/tenants/me` works for tenant users.

## Success Criteria
- `GET /api/v1/tenants` is no longer accessible.
- Tenant enumeration is only possible via the admin API.
- Multi-tenant isolation is preserved.
