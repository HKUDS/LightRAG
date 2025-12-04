# Multi-Tenancy Audit Report

**Date:** November 21, 2025
**Project:** LightRAG
**Auditor:** GitHub Copilot

## Executive Summary

The current multi-tenancy implementation in LightRAG relies on **application-level isolation**. While it provides helper classes (`TenantSQLBuilder`, `MongoTenantHelper`, etc.) to filter data by `tenant_id` and `kb_id`, it lacks **enforcement at the database or framework level**. This design is susceptible to data leaks if developers fail to use the helpers correctly.

The "battle-tested" approach requires **Row-Level Security (RLS)** for PostgreSQL, **strict repository wrappers** for NoSQL stores, and **middleware-enforced tenant identification** (subdomains + JWT).

## Gap Analysis

| Feature | Current Implementation | Battle-Tested Standard | Gap Severity |
| :--- | :--- | :--- | :--- |
| **Tenant Identification** | Headers (`X-Tenant-ID`) or JWT metadata. No subdomain support. | Subdomains (`tenant.app.com`) + JWT `tenant_id` claim. | **High** |
| **PostgreSQL Isolation** | `WHERE` clause filtering via `TenantSQLBuilder`. | **Row-Level Security (RLS)** + Tenant UUID PK. | **Critical** |
| **MongoDB Isolation** | Manual field filtering via `MongoTenantHelper`. | **Tenant-scoped Repository** or ODM Middleware (Beanie). | **High** |
| **Neo4j/Memgraph Isolation** | Cypher query modification via helper. | **Tenant Session Wrapper** or Label Prefixing. | **High** |
| **Vector DB Isolation** | Metadata filtering via helper. | **Tenant-scoped Repository** or Collection Separation. | **High** |
| **Redis Isolation** | Key prefixing via `RedisTenantNamespace` (manual usage). | **Key Prefixing** enforced by wrapper/dependency. | **Medium** |
| **Framework Enforcement** | Optional dependencies in routers. | **Global Middleware** + Dependency Injection. | **High** |

## Detailed Findings

### 1. Tenant Identification

*   **Current**: `lightrag/api/dependencies.py` extracts `tenant_id` from headers or JWT.
*   **Risk**: Clients can potentially spoof `X-Tenant-ID` if not strictly validated against the JWT. Subdomains are not used, making it harder to isolate tenants at the DNS/networking level (e.g., for CORS or cookies).

### 2. PostgreSQL

*   **Current**: `lightrag/kg/postgres_tenant_support.py` modifies SQL strings.
*   **Risk**: "Trusting the application code". A raw SQL query without the builder will leak data. RLS is the only way to prevent this at the database engine level.

### 3. MongoDB

*   **Current**: `lightrag/kg/mongo_tenant_support.py` provides helper methods.
*   **Risk**: Developers must remember to call `add_tenant_fields` and `get_tenant_filter`.

### 4. Neo4j

*   **Current**: `lightrag/kg/graph_tenant_support.py` injects `WHERE` clauses.
*   **Risk**: Complex Cypher queries might be difficult to parse and modify correctly. A session wrapper that enforces parameters is safer.

### 5. Redis

*   **Current**: `lightrag/kg/redis_tenant_support.py` provides `RedisTenantNamespace`.
*   **Risk**: Manual usage of the namespace wrapper is required.

### 6. Vector Databases (Qdrant, Milvus, FAISS, Nano)

*   **Current**: `lightrag/kg/vector_tenant_support.py` provides helper methods for metadata filtering and ID prefixing.
*   **Risk**: Similar to other NoSQL stores, developers must manually apply filters and metadata.
    *   **Qdrant**: Relies on `must` conditions in filters.
    *   **Milvus**: Relies on `expr` strings.
    *   **FAISS**: Relies on index naming or metadata filtering (which can be slow if not optimized).
    *   **Nano**: Relies on metadata filtering.

### 7. Other Graph Databases (Memgraph, NetworkX)

*   **Current**: `lightrag/kg/graph_tenant_support.py` covers these.
*   **Risk**:
    *   **Memgraph**: Similar to Neo4j, relies on Cypher query modification.
    *   **NetworkX**: In-memory graph. Isolation relies on creating subgraphs or filtering edges manually. If the graph is persisted, it needs careful handling.

## Recommendations

1.  **Implement Subdomain Middleware**: Add middleware to resolve `tenant_id` from subdomains and validate it against Redis/DB.
2.  **Enable PostgreSQL RLS**:
    *   Add `tenant_id` to `current_setting`.
    *   Enable RLS on all tables.
    *   Create policies to enforce isolation.
3.  **Refactor MongoDB Access**: Create a `MongoTenantRepo` class that wraps the collection and automatically applies filters.
4.  **Refactor Neo4j/Memgraph Access**: Create a `GraphTenantSession` class that wraps the driver session.
5.  **Refactor Vector DB Access**: Create a `VectorTenantRepo` class (or specific implementations) that wraps the client and enforces metadata/filtering.
6.  **Global Dependency**: Ensure `get_tenant_context` is used globally or at the router level for all tenant-specific endpoints.

## Action Plan

See `docs/action_plan/02-implementation-plan.md` for the detailed steps.
