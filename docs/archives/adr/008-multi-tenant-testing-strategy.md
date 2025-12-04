# ADR 008: Multi-Tenant Testing Strategy for ./starter Environment

## Status: Proposed

## Context

The `./starter` directory provides a local development and testing environment for the LightRAG multi-tenant implementation. This environment must support two distinct operational modes:

1. **No Multi-Tenant Mode (Compatibility Mode)**: Behaves like the original single-workspace LightRAG, maintaining backward compatibility with the `main` branch
2. **Multi-Tenant Mode (Production Mode)**: Demonstrates full multi-tenant isolation with multiple tenants and knowledge bases

The testing strategy must ensure both modes work correctly and that switching between them is seamless and reproducible.

## Problem Statement

Current state (as of November 2025):
- The multi-tenant architecture is implemented in `feat/multi-tenant` branch
- The `./starter` directory uses Docker Compose with PostgreSQL, Redis, LightRAG API, and WebUI
- The environment supports automatic tenant/KB resolution with "default" tenant and KB names
- No clear testing protocol exists for validating both single-tenant and multi-tenant scenarios
- Documentation doesn't reflect the actual implementation details

### Key Challenges

1. **Backward Compatibility**: Must verify that old single-tenant code paths still work
2. **Switching Modes**: Need clear procedures to enable/disable multi-tenancy for testing
3. **Data Isolation**: Must verify tenant/KB isolation at both API and database levels
4. **Reproducibility**: Tests must be idempotent and produce consistent results
5. **Environment Configuration**: Starter must clearly document all configuration options for each mode

## Decision

### Multi-Mode Testing Architecture

We implement a **configurable testing environment** in `./starter` that can operate in two modes via environment variables:

```
Mode Selection:
├─ MULTITENANT_MODE=off    → Single-tenant compatibility mode (like main branch)
├─ MULTITENANT_MODE=on     → Full multi-tenant mode (default)
└─ MULTITENANT_MODE=demo   → Multi-tenant demo mode (2 pre-configured tenants)
```

### Scenario 1: No Multi-Tenant Mode (Compatibility Mode)

**Goal**: Verify that LightRAG works exactly as it did before multi-tenancy was added.

**Configuration**:
```env
MULTITENANT_MODE=off
DEFAULT_TENANT=default
DEFAULT_KB=default
WORKSPACE_ISOLATION_TYPE=legacy
```

**Behavior**:
- API endpoints work WITHOUT requiring X-Tenant-ID or X-KB-ID headers
- All operations use "default" tenant and "default" KB internally
- Storage uses legacy workspace namespace: `tenant_id_kb_id` → `default_default`
- No tenant context validation errors
- Fully backward compatible with main branch code

**Testing Scenarios**:

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| **T1.1** | Upload document without tenant headers | ✓ Document stored in default workspace |
| **T1.2** | Query without tenant headers | ✓ Results returned from default workspace |
| **T1.3** | Create knowledge graph without tenant headers | ✓ Graph entities stored in default workspace |
| **T1.4** | Access WebUI without specifying tenant | ✓ UI works with implicit default tenant |
| **T1.5** | Database contains no tenant_id/kb_id fields | ✓ Tables use legacy workspace field only |
| **T1.6** | Mix of old client and new client requests | ✓ Both work without conflicts |
| **T1.7** | Verify authorization doesn't require tenant access | ✓ Auth tokens work with role only (no tenant scope) |
| **T1.8** | All stored data is in `default_default` namespace | ✓ Data isolation via workspace namespace only |

**Backward Compatibility Verification**:
```python
# Should work identically to main branch
rag = LightRAG(working_dir="./rag_data")  # No tenant_id/kb_id
await rag.insert("document text")
results = await rag.query("query text")

# Should NOT require X-Tenant-ID header
curl -X POST http://localhost:8000/api/v1/documents/insert \
  -H "Authorization: Bearer token"  \
  -H "Content-Type: application/json" \
  -d '{"document": "text"}'
```

**Database Schema**:
- Uses legacy tables: `lightrag_doc_full`, `lightrag_doc_chunks`, etc.
- `workspace` field acts as tenant/KB namespace
- No `tenant_id`, `kb_id` columns added
- Composite indexes: `(workspace, id)` not `(tenant_id, kb_id, id)`

### Scenario 2: Single Tenant with Multiple KBs (Controlled Multi-Tenant)

**Goal**: Test multi-tenant architecture with a single tenant serving multiple knowledge bases.

**Configuration**:
```env
MULTITENANT_MODE=on
DEFAULT_TENANT=tenant-1
DEFAULT_KB=kb-default
CREATE_DEFAULT_KB=kb-default,kb-secondary,kb-experimental
```

**Behavior**:
- API requires X-Tenant-ID (resolves to tenant-1 if not provided)
- API requires X-KB-ID (can be any KB in the tenant)
- Different KBs can coexist for the same tenant
- Complete data isolation: `tenant-1:kb-default:entity1` vs `tenant-1:kb-secondary:entity1`
- Same tenant has access to all its KBs

**Testing Scenarios**:

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| **T2.1** | Insert into kb-default | ✓ Document stored in tenant-1_kb-default namespace |
| **T2.2** | Insert into kb-secondary | ✓ Document stored in tenant-1_kb-secondary namespace |
| **T2.3** | Query from kb-default returns kb-default data only | ✓ No cross-KB data leakage |
| **T2.4** | Query from kb-secondary returns kb-secondary data only | ✓ No cross-KB data leakage |
| **T2.5** | Switch KB in same request sequence | ✓ Isolation maintained at database level |
| **T2.6** | Create duplicate entity names in different KBs | ✓ Database allows duplicates (namespace isolated) |
| **T2.7** | Generate graph for kb-default | ✓ Graph contains only kb-default entities |
| **T2.8** | Database contains tenant_id and kb_id columns | ✓ Composite keys prevent collisions |

**Query Examples**:
```bash
# Query KB-1
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer token" \
  -H "X-Tenant-ID: tenant-1" \
  -H "X-KB-ID: kb-default" \
  -d '{"query": "text"}'

# Query KB-2 (same tenant)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer token" \
  -H "X-Tenant-ID: tenant-1" \
  -H "X-KB-ID: kb-secondary" \
  -d '{"query": "text"}'
```

**Database Schema**:
- Tables include both `workspace` (legacy) and `tenant_id`, `kb_id` (new)
- Workspace auto-generated as `tenant_id_kb_id` for backward compatibility
- Composite indexes: `(tenant_id, kb_id, id)`
- Unique constraints prevent accidental cross-KB entity conflicts

### Scenario 3: Multiple Tenants (Full Multi-Tenant)

**Goal**: Demonstrate complete multi-tenant isolation with multiple independent organizations.

**Configuration**:
```env
MULTITENANT_MODE=demo
# Pre-configured demo tenants:
# - Tenant 1: "acme-corp"
#   ├─ kb-prod (Production knowledge base)
#   └─ kb-dev (Development knowledge base)
# - Tenant 2: "techstart"
#   ├─ kb-main (Main knowledge base)
#   └─ kb-backup (Backup knowledge base)
```

**Behavior**:
- API requires X-Tenant-ID and X-KB-ID on every request
- No "default" fallback - must be explicit
- Complete isolation: acme-corp cannot access techstart data
- Separate resource quotas per tenant
- Independent configurations (LLM models, embedding models)

**Testing Scenarios**:

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| **T3.1** | acme-corp inserts into kb-prod | ✓ Data isolated in acme-corp_kb-prod |
| **T3.2** | techstart inserts into kb-main | ✓ Data isolated in techstart_kb-main |
| **T3.3** | acme-corp queries kb-prod | ✓ Returns only acme-corp_kb-prod data |
| **T3.4** | techstart queries kb-main | ✓ Returns only techstart_kb-main data |
| **T3.5** | acme-corp attempts to query techstart KB | ✗ 403 Forbidden (access denied) |
| **T3.6** | User with acme-corp JWT tries techstart KB | ✗ Permission denied at API layer |
| **T3.7** | Different entity names in different tenants | ✓ Database allows identical names in different tenant_id values |
| **T3.8** | Database NEVER returns cross-tenant data | ✓ Query filters enforce `tenant_id` constraint |
| **T3.9** | Even with DB admin creds, workspace isolation works | ✓ Cannot accidentally query wrong tenant at SQL level |
| **T3.10** | Delete acme-corp entity doesn't affect techstart | ✓ DELETE uses composite key (tenant_id, kb_id, id) |
| **T3.11** | Verify JWT tokens scope to specific tenant | ✓ Token contains tenant_id, prevents cross-tenant access |
| **T3.12** | Resource quotas enforced per tenant | ✓ acme-corp quota limits don't affect techstart |

**Request Examples**:
```bash
# acme-corp accessing kb-prod
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer acme-corp-token" \
  -H "X-Tenant-ID: acme-corp" \
  -H "X-KB-ID: kb-prod" \
  -d '{"query": "revenue"}'

# techstart accessing kb-main
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer techstart-token" \
  -H "X-Tenant-ID: techstart" \
  -H "X-KB-ID: kb-main" \
  -d '{"query": "funding"}'

# acme-corp trying to access techstart data (should fail)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer acme-corp-token" \
  -H "X-Tenant-ID: techstart" \
  -H "X-KB-ID: kb-main" \
  -d '{"query": "data"}' \
# Response: 403 Forbidden - "User does not have access to tenant"
```

**Database Verification**:
```sql
-- Verify tenant isolation at SQL level
SELECT COUNT(*) FROM lightrag_doc_full 
WHERE tenant_id = 'acme-corp' AND kb_id = 'kb-prod';  -- Count acme-corp documents

SELECT COUNT(*) FROM lightrag_doc_full 
WHERE tenant_id = 'techstart' AND kb_id = 'kb-main';  -- Count techstart documents

-- Verify no cross-tenant data in single query
SELECT DISTINCT tenant_id, kb_id FROM lightrag_doc_full;  -- Should see: acme-corp, techstart only

-- Verify composite indexes exist
\di lightrag_doc_full*  -- Should show idx on (tenant_id, kb_id, id)
```

## Implementation Details

### Environment Variable Configuration

**File: `./starter/env.example`** (updated with new options):

```env
# ============================================================================
# TESTING MODE CONFIGURATION
# ============================================================================

# Choose testing mode:
#   off   = Single-tenant compatibility mode (like main branch)
#   on    = Multi-tenant mode with single default tenant
#   demo  = Multi-tenant mode with 2 pre-configured tenants
MULTITENANT_MODE=demo

# For MULTITENANT_MODE=on, create additional KBs
# Format: kb_name,kb_name,kb_name
CREATE_DEFAULT_KB=kb-default,kb-secondary,kb-experimental

# ============================================================================
# TENANT CONFIGURATION (Used when MULTITENANT_MODE != off)
# ============================================================================

DEFAULT_TENANT=default
DEFAULT_KB=default

# Pre-configured demo tenants (for MULTITENANT_MODE=demo)
DEMO_TENANT_1_NAME=acme-corp
DEMO_TENANT_1_KBS=kb-prod,kb-dev

DEMO_TENANT_2_NAME=techstart
DEMO_TENANT_2_KBS=kb-main,kb-backup

# ============================================================================
```

### Docker Compose Modifications

**File: `./starter/docker-compose.yml`** (add initialization script):

```yaml
services:
  postgres:
    environment:
      MULTITENANT_MODE: ${MULTITENANT_MODE:-demo}
    volumes:
      - ./init-postgres.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
      - ./init-demo-tenants.sql:/docker-entrypoint-initdb.d/02-demo-tenants.sql:ro  # New

  lightrag-api:
    environment:
      MULTITENANT_MODE: ${MULTITENANT_MODE:-demo}
      DEFAULT_TENANT: ${DEFAULT_TENANT:-default}
      DEFAULT_KB: ${DEFAULT_KB:-default}
      CREATE_DEFAULT_KB: ${CREATE_DEFAULT_KB:-kb-default}
```

### Initialization SQL Scripts

**New File: `./starter/init-demo-tenants.sql`**

Creates the demo tenants and KBs when `MULTITENANT_MODE=demo`:

```sql
-- Only run if MULTITENANT_MODE=demo
-- This is handled via environment variable in docker-entrypoint

INSERT INTO tenants (tenant_id, tenant_name, description, is_active)
VALUES 
  ('acme-corp', 'ACME Corporation', 'Demo tenant 1: Large enterprise', true),
  ('techstart', 'TechStart Inc', 'Demo tenant 2: Startup', true);

INSERT INTO knowledge_bases (kb_id, tenant_id, kb_name, description, is_active)
VALUES 
  ('kb-prod', 'acme-corp', 'Production KB', 'Live production data', true),
  ('kb-dev', 'acme-corp', 'Development KB', 'Dev/staging data', true),
  ('kb-main', 'techstart', 'Main KB', 'Primary knowledge base', true),
  ('kb-backup', 'techstart', 'Backup KB', 'Backup and archival', true);
```

### Testing Procedures

#### Procedure 1: Run Compatibility Mode Tests

```bash
cd ./starter

# 1. Configure for compatibility mode
cp env.example .env
echo "MULTITENANT_MODE=off" >> .env
echo "WORKSPACE_ISOLATION_TYPE=legacy" >> .env

# 2. Start services
make setup
make up
make init-db

# 3. Run compatibility tests
pytest ../tests/test_backward_compatibility.py -v

# 4. Run manual tests
python3 reproduce_issue.py  # Should work without tenant headers
```

**Expected Results**:
- All T1.x test cases pass
- No authorization failures due to missing tenant context
- Database uses workspace namespace only
- Queries return data regardless of tenant headers (or missing headers)

#### Procedure 2: Run Single-Tenant Multi-KB Tests

```bash
cd ./starter

# 1. Configure for single tenant with multiple KBs
cp env.example .env
echo "MULTITENANT_MODE=on" >> .env
echo "DEFAULT_TENANT=tenant-1" >> .env
echo "CREATE_DEFAULT_KB=kb-default,kb-secondary,kb-experimental" >> .env

# 2. Start services
make setup
make up
make init-db

# 3. Run isolation tests
pytest ../tests/test_multi_tenant_backends.py::TestTenantIsolation -v

# 4. Manual test: Verify KB isolation
# Create document in kb-default
curl -X POST http://localhost:8000/api/v1/documents/insert \
  -H "X-Tenant-ID: tenant-1" \
  -H "X-KB-ID: kb-default" \
  -d '{"document": "document in kb-default"}'

# Query should return document
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-Tenant-ID: tenant-1" \
  -H "X-KB-ID: kb-default"

# Query in different KB should return NOTHING
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-Tenant-ID: tenant-1" \
  -H "X-KB-ID: kb-secondary"
```

**Expected Results**:
- All T2.x test cases pass
- Data isolated by KB within same tenant
- No cross-KB data leakage at API or database level
- Composite keys work correctly

#### Procedure 3: Run Full Multi-Tenant Tests

```bash
cd ./starter

# 1. Configure for full multi-tenant demo mode (default)
cp env.example .env
echo "MULTITENANT_MODE=demo" >> .env

# 2. Start services
make setup
make up
make init-db

# 3. Run full multi-tenant tests
pytest ../tests/test_multi_tenant_backends.py -v
pytest ../tests/test_tenant_security.py -v

# 4. Manual test: Verify cross-tenant isolation
# Insert as acme-corp
curl -X POST http://localhost:8000/api/v1/documents/insert \
  -H "Authorization: Bearer acme-token" \
  -H "X-Tenant-ID: acme-corp" \
  -H "X-KB-ID: kb-prod" \
  -d '{"document": "acme document"}'

# Try to query as techstart (should fail or return empty)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer acme-token" \
  -H "X-Tenant-ID: techstart" \
  -H "X-KB-ID: kb-main"
# Expected: 403 Forbidden

# Query as acme should succeed
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer acme-token" \
  -H "X-Tenant-ID: acme-corp" \
  -H "X-KB-ID: kb-prod"
# Expected: 200 OK with results

# 5. Database verification
make db-shell
# SELECT COUNT(*) FROM lightrag_doc_full WHERE tenant_id='acme-corp';
# SELECT DISTINCT tenant_id FROM lightrag_doc_full;
```

**Expected Results**:
- All T3.x test cases pass
- Cross-tenant access denied (403)
- Complete data isolation at database level
- No authorization bypasses

## Testing Matrix

| Mode | Tenant Headers Required | Default Tenant | Multiple KBs | Multiple Tenants | Test File |
|------|-------------------------|------------------|--------------|------------------|-----------|
| **off** | No | Always default | ❌ Single workspace | ❌ Single tenant | `test_backward_compatibility.py` |
| **on** | Yes | Provided/resolved | ✓ Multiple per tenant | ❌ Single tenant only | `test_multi_tenant_backends.py` |
| **demo** | Yes | None/explicit | ✓ Multiple per tenant | ✓ 2 pre-configured | `test_tenant_security.py` |

## Test Coverage

### Unit Tests

- **test_backward_compatibility.py**: Validates old code paths still work
- **test_multi_tenant_backends.py**: Validates storage layer isolation
- **test_tenant_models.py**: Validates data models
- **test_tenant_security.py**: Validates permission/authorization

### Integration Tests

- **API Layer**: test_tenant_api_routes.py
- **Database**: test_tenant_storage_phase3.py
- **Graph Operations**: test_graph_storage.py

### Manual Verification

- Database schema validation (composite keys, indexes)
- Cross-tenant access attempts (should fail)
- KB isolation verification
- Authorization enforcement

## Consequences

### Positive

1. **Flexible Testing**: Can test backward compatibility and new multi-tenant features
2. **Clear Procedures**: Step-by-step procedures for each testing scenario
3. **Reproducibility**: Environment variables make tests repeatable
4. **Safety**: Explicit mode selection prevents accidental data mixing
5. **Documentation**: Clear understanding of what each mode does
6. **Validation**: Comprehensive test matrix covers all scenarios

### Negative/Tradeoffs

1. **Configuration Complexity**: Three modes add configuration overhead
2. **Initialization Scripts**: Must maintain both legacy and multi-tenant initialization
3. **Testing Duration**: Running all three modes sequentially takes time
4. **Documentation Maintenance**: Must keep mode-specific docs up to date
5. **Docker Image Size**: Includes both legacy and new code paths

## Rollback/Migration

### From Compatibility Mode to Multi-Tenant

```bash
# 1. Back up existing data
make db-backup

# 2. Switch mode
sed -i 's/MULTITENANT_MODE=off/MULTITENANT_MODE=on/' .env

# 3. Migrate database schema
# This requires: DROP old tables, CREATE new tables with tenant columns
# Migration script: scripts/migrate_to_multitenant.sql

# 4. Restart services
make restart
```

### From Multi-Tenant back to Compatibility Mode

```bash
# 1. Back up multi-tenant data
make db-backup

# 2. Switch mode
sed -i 's/MULTITENANT_MODE=on/MULTITENANT_MODE=off/' .env

# 3. Extract single tenant data
# SELECT * FROM lightrag_doc_full WHERE tenant_id='default' 
# INTO workspace-based tables

# 4. Restart services
make restart
```

## Verification Checklist

Before considering the ADR complete:

- [ ] `MULTITENANT_MODE=off` works identically to main branch
- [ ] `MULTITENANT_MODE=on` prevents cross-KB data access
- [ ] `MULTITENANT_MODE=demo` prevents cross-tenant data access
- [ ] Environment variable switching is seamless
- [ ] All test cases (T1-T3) pass in their respective modes
- [ ] Database schema matches mode requirements
- [ ] Documentation reflects actual implementation
- [ ] Integration tests run successfully
- [ ] Manual verification procedures validate isolation
- [ ] Authorization failures work correctly (403, 401, etc.)

## References

### Related ADRs
- ADR 001: Multi-Tenant Architecture Overview
- ADR 002: Implementation Strategy
- ADR 003: Data Models and Storage
- ADR 004: API Design
- ADR 005: Security Analysis

### Implementation Files
- `lightrag/models/tenant.py`: TenantContext, Tenant, KnowledgeBase models
- `lightrag/tenant_rag_manager.py`: Per-tenant instance management
- `lightrag/api/dependencies.py`: Tenant context extraction
- `tests/test_backward_compatibility.py`: Legacy compatibility tests
- `tests/test_multi_tenant_backends.py`: Multi-tenant backend tests
- `tests/test_tenant_security.py`: Security validation

### Starter Files
- `starter/docker-compose.yml`: Service orchestration
- `starter/env.example`: Configuration template
- `starter/Makefile`: Testing procedures
- `starter/init-postgres.sql`: Database initialization

## Next Steps

1. **Implement environment variable handling** in docker-entrypoint-initdb.d scripts
2. **Create demo tenant initialization** SQL script (init-demo-tenants.sql)
3. **Update Makefile** with mode-specific test targets
4. **Create test runner script** that runs appropriate tests for each mode
5. **Document mode selection** in README.md
6. **Create CI/CD workflow** to test all three modes automatically
7. **Add health checks** that validate mode-specific expectations
8. **Create migration scripts** for switching between modes
9. **Update all existing ADRs** to reference this testing strategy
10. **Add mode detection** to API startup (warn if wrong mode configuration)

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-22  
**Author**: Architecture Design Process  
**Status**: Proposed - Ready for Implementation Review

**Implementation Notes**:
- Based on actual code examination of feat/multi-tenant branch
- Verified against: tenant.py, tenant_rag_manager.py, dependencies.py, docker-compose.yml
- Tested scenarios aligned with actual test files in tests/ directory
- Configuration options match env.example and existing environment setup
