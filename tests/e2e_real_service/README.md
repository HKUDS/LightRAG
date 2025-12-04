# E2E Real Service Tests

This directory contains end-to-end tests that run against a live LightRAG API instance.
These tests verify the actual behavior of the system, including database interactions, authentication, and multi-tenant isolation.

## Prerequisites

1. A running LightRAG stack (API, Postgres, Redis, Ollama).
   You can start it using:
   ```bash
   ./scripts/start-dev-stack.sh
   ```

2. Python dependencies:
   ```bash
   pip install requests
   ```

## Running the Tests

Run the isolation test script:

```bash
python tests/e2e_real_service/test_api_isolation.py
```

## What it Tests

1. **Authentication**: Logs in as admin to get a JWT token.
2. **Tenant Creation**: Creates two distinct tenants (Tenant A and Tenant B).
3. **KB Creation**: Creates a Knowledge Base for each tenant.
4. **Data Ingestion**: Uploads unique "secret" documents to each tenant's KB.
5. **Indexing**: Waits for the documents to be processed and indexed.
6. **Isolation Verification**:
   - Verifies Tenant A can retrieve its own secret.
   - Verifies Tenant B can retrieve its own secret.
   - **CRITICAL**: Verifies Tenant A CANNOT retrieve Tenant B's secret.
   - **CRITICAL**: Verifies Tenant B CANNOT retrieve Tenant A's secret.

## Configuration

You can override defaults using environment variables:

- `LIGHTRAG_API_URL`: URL of the API (default: `http://localhost:9621`)
- `AUTH_USER`: Admin username (default: `admin`)
- `AUTH_PASS`: Admin password (default: `admin123`)
