# Multi-Tenancy Implementation Plan

**Goal**: Upgrade LightRAG to a battle-tested, production-grade multi-tenant architecture.

## Phase 1: Tenant Identification & Middleware

- [ ] **Step 1.1**: Create `lightrag/api/middleware/tenant.py`.
  - Implement `TenantMiddleware` to extract tenant from subdomain (optional) and JWT.
  - Use Redis to cache `subdomain -> tenant_id` resolution.
  - Set `request.state.tenant_id`.
- [ ] **Step 1.2**: Update `lightrag/api/dependencies.py`.
  - Update `get_tenant_context` to read from `request.state`.
  - Remove reliance on `X-Tenant-ID` header when subdomain/JWT is present (enforce source of truth).

## Phase 2: PostgreSQL Row-Level Security (RLS)

- [ ] **Step 2.1**: Update `lightrag/kg/postgres_tenant_support.py`.
  - Add SQL to enable RLS on tables: `ALTER TABLE ... ENABLE ROW LEVEL SECURITY`.
  - Add SQL to create policies: `CREATE POLICY ... USING (tenant_id = current_setting('app.tenant_id'))`.
- [ ] **Step 2.2**: Update Database Connection Logic.
  - In `lightrag/kg/postgres_impl.py` (or equivalent), ensure `app.tenant_id` is set for each session/connection.
  - Use `SET LOCAL app.tenant_id = ...` at the start of transactions.

## Phase 3: MongoDB Strict Scoping

- [ ] **Step 3.1**: Create `lightrag/kg/mongo_repo.py`.
  - Implement `MongoTenantRepo` class.
  - It should take `tenant_id` in `__init__`.
  - Override `find`, `find_one`, `insert_one`, etc., to automatically inject `tenant_id`.
- [ ] **Step 3.2**: Refactor `lightrag/kg/mongo_impl.py`.
  - Use `MongoTenantRepo` instead of raw `motor` collection.

## Phase 4: Graph Database Session Wrapper (Neo4j, Memgraph)

- [ ] **Step 4.1**: Create `lightrag/kg/graph_session.py`.
  - Implement `GraphTenantSession` abstract base class.
  - Implement `Neo4jTenantSession` and `MemgraphTenantSession`.
  - Wrap `run` method to inject `tenant_id` parameter and append `WHERE n.tenant_id = $tenant_id` if missing (or rely on strict parameterized queries).
- [ ] **Step 4.2**: Refactor `lightrag/kg/neo4j_impl.py` and `memgraph_impl.py`.
  - Use `GraphTenantSession`.

## Phase 5: Vector Database Strict Scoping

- [ ] **Step 5.1**: Create `lightrag/kg/vector_repo.py`.
  - Implement `VectorTenantRepo` abstract base class.
  - Implement specific repositories for Qdrant, Milvus, FAISS, Nano.
  - **Qdrant**: Automatically add `must` filter for `tenant_id` and `kb_id` to all searches.
  - **Milvus**: Automatically append `tenant_id == "..."` to expressions.
  - **FAISS**: Manage tenant-specific indices (e.g., `index_tenant_kb`) to avoid scanning all vectors.
  - **Nano**: Enforce metadata filtering.
- [ ] **Step 5.2**: Refactor Vector Implementations.
  - Update `qdrant_impl.py`, `milvus_impl.py`, `faiss_impl.py`, `nano_vector_db_impl.py` to use the new repositories.

## Phase 6: Redis Strict Prefixing

- [ ] **Step 6.1**: Enforce `RedisTenantNamespace`.
  - Ensure all Redis interactions in `lightrag/kg/redis_impl.py` use the namespace wrapper.

## Phase 7: Verification

- [ ] **Step 7.1**: Create tests in `tests/test_multi_tenant_security.py`.
  - Test RLS: Try to access another tenant's data via raw SQL.
  - Test Middleware: Verify subdomain resolution.
  - Test Isolation: Verify data separation across all backends (SQL, NoSQL, Graph, Vector).
