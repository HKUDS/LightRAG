# Task Log - Secure Multi-Tenant API

## Actions

- Removed public `list_tenants` endpoint from `lightrag/api/routers/tenant_routes.py`.
- Updated `lightrag/api/config.py` to support `AUTH_USER` and `AUTH_PASS` environment variables as a fallback for `AUTH_ACCOUNTS`.
- Updated `lightrag/api/lightrag_server.py` to properly initialize `tenant_storage` in the application lifespan.
- Updated `test_multitenant.sh` to verify security and functionality.

## Decisions

- Decided to modify `config.py` to support legacy/docker env vars instead of forcing `AUTH_ACCOUNTS` in `docker-compose.yml`, preserving backward compatibility and ease of use.
- Decided to initialize `tenant_storage` explicitly in `lifespan` because it is created outside of the main `LightRAG` instance management.

## Results

- Public API `/api/v1/tenants` now returns 405 Method Not Allowed.
- Admin API `/api/v1/admin/tenants` is accessible with valid credentials.
- Authentication works correctly with `AUTH_USER` and `AUTH_PASS`.
- Tenant creation and listing works without `JsonKVStorage` errors.
- All tests in `test_multitenant.sh` pass.
